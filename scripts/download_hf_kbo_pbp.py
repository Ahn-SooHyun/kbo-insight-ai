from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download, hf_hub_url


LOGGER = logging.getLogger(__name__)

REPO_ID = "slothman3878/kbo_playbyplay"
REPO_TYPE = "dataset"
SOURCE = "Hugging Face Dataset"

SUPPORTED_SEASONS = (2023, 2024, 2025, 2026)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "hf_kbo_pbp"
MANIFEST_PATH = OUTPUT_DIR / "download_manifest.csv"

SHA256_CHUNK_SIZE = 1024 * 1024
FULL_COMMIT_SHA_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")

MANIFEST_COLUMNS = [
    "source",
    "source_url",
    "repo_id",
    "remote_filename",
    "season",
    "requested_revision",
    "resolved_revision",
    "downloaded_at",
    "row_count",
    "column_count",
    "file_size_bytes",
    "sha256",
    "license",
]


class DownloadError(RuntimeError):
    """Raw Dataset 확보를 중단해야 하는 오류를 나타낸다."""


@dataclass(frozen=True)
class Target:
    """원격 원본 파일과 로컬 저장 파일의 대응 정보를 나타낸다."""

    remote_filename: str
    local_filename: str
    season: int | None
    is_parquet: bool


@dataclass(frozen=True)
class StagedFile:
    """staging 영역에 확보하고 검증한 원본 파일 정보를 나타낸다."""

    target: Target
    path: Path
    sha256: str
    file_size_bytes: int
    row_count: int | None
    column_count: int | None


def parse_args() -> argparse.Namespace:
    """다운로드에 필요한 최소 CLI 옵션을 파싱한다."""
    parser = argparse.ArgumentParser(
        description="Hugging Face KBO Play-by-Play 원본 다운로드"
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="조회할 Hugging Face revision입니다. 기본값: main",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=list(SUPPORTED_SEASONS),
        help="다운로드할 시즌입니다. 기본값: 2023 2024 2025 2026",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="원격 구조와 파일 매핑만 확인하고 다운로드하지 않습니다.",
    )
    return parser.parse_args()


def normalize_seasons(values: list[int]) -> tuple[int, ...]:
    """시즌 중복을 제거하고 Issue 범위를 벗어난 입력을 거부한다."""
    seasons = tuple(dict.fromkeys(values))

    unsupported = sorted(set(seasons) - set(SUPPORTED_SEASONS))
    if unsupported:
        raise DownloadError(
            f"지원하지 않는 시즌입니다: {unsupported}. "
            f"지원 시즌: {list(SUPPORTED_SEASONS)}"
        )

    if not seasons:
        raise DownloadError("최소 1개 이상의 시즌을 지정해야 합니다.")

    return seasons


def calculate_sha256(path: Path) -> str:
    """파일을 chunk 단위로 읽어 SHA256 checksum을 계산한다."""
    digest = hashlib.sha256()

    with path.open("rb") as file_obj:
        while chunk := file_obj.read(SHA256_CHUNK_SIZE):
            digest.update(chunk)

    return digest.hexdigest()


def resolve_repository(
    api: HfApi,
    requested_revision: str,
) -> tuple[str, str, list[str]]:
    """revision을 full commit SHA로 고정하고 repository metadata를 조회한다."""
    try:
        dataset_info = api.dataset_info(
            repo_id=REPO_ID,
            revision=requested_revision,
        )

        resolved_revision = getattr(dataset_info, "sha", None)

        if (
            not isinstance(resolved_revision, str)
            or not FULL_COMMIT_SHA_PATTERN.fullmatch(resolved_revision)
        ):
            raise DownloadError(
                "유효한 full commit SHA를 확인하지 못했습니다. "
                f"조회값: {resolved_revision!r}"
            )

        repo_files = sorted(
            api.list_repo_files(
                repo_id=REPO_ID,
                revision=resolved_revision,
                repo_type=REPO_TYPE,
            )
        )
    except DownloadError:
        raise
    except Exception as exc:
        raise DownloadError(
            f"Dataset repository 조회에 실패했습니다: "
            f"{REPO_ID}@{requested_revision}"
        ) from exc

    if not repo_files:
        raise DownloadError(
            "Dataset repository에서 파일을 찾지 못했습니다."
        )

    card_data = getattr(dataset_info, "card_data", None)

    if isinstance(card_data, dict):
        license_value = card_data.get("license")
    else:
        license_value = getattr(card_data, "license", None)

    if isinstance(license_value, (list, tuple, set)):
        license_text = ";".join(
            str(value)
            for value in license_value
            if value
        )
    elif license_value:
        license_text = str(license_value)
    else:
        license_text = "unknown"
        LOGGER.warning(
            "Dataset metadata에서 license를 확인하지 못했습니다."
        )

    return resolved_revision, license_text, repo_files


def select_targets(
    repo_files: list[str],
    seasons: tuple[int, ...],
) -> list[Target]:
    """실제 repository 구조에서 시즌 Parquet와 schema.yaml을 선택한다."""
    season_files: dict[int, str] = {}

    for season in seasons:
        season_pattern = re.compile(
            rf"(?<!\d){season}(?!\d)"
        )

        candidates = [
            filename
            for filename in repo_files
            if filename.lower().endswith(".parquet")
            and season_pattern.search(filename)
        ]

        if len(candidates) != 1:
            candidate_text = (
                "\n".join(
                    f"  - {filename}"
                    for filename in candidates
                )
                or "  - 없음"
            )

            raise DownloadError(
                f"{season} 시즌 Parquet 후보를 "
                "유일하게 결정할 수 없습니다.\n"
                f"{candidate_text}"
            )

        season_files[season] = candidates[0]

    schema_candidates = [
        filename
        for filename in repo_files
        if PurePosixPath(filename).name.lower()
        == "schema.yaml"
    ]

    if len(schema_candidates) > 1:
        parquet_parents = {
            PurePosixPath(filename).parent
            for filename in season_files.values()
        }

        schema_candidates = [
            filename
            for filename in schema_candidates
            if PurePosixPath(filename).parent
            in parquet_parents
        ]

    if len(schema_candidates) != 1:
        candidate_text = (
            "\n".join(
                f"  - {filename}"
                for filename in schema_candidates
            )
            or "  - 없음"
        )

        raise DownloadError(
            "schema.yaml 후보를 "
            "유일하게 결정할 수 없습니다.\n"
            f"{candidate_text}"
        )

    targets = [
        Target(
            remote_filename=season_files[season],
            local_filename=f"{season}.parquet",
            season=season,
            is_parquet=True,
        )
        for season in seasons
    ]

    targets.append(
        Target(
            remote_filename=schema_candidates[0],
            local_filename="schema.yaml",
            season=None,
            is_parquet=False,
        )
    )

    return targets


def read_parquet_shape(path: Path) -> tuple[int, int]:
    """전체 데이터를 로드하지 않고 Parquet footer에서 행/열 수를 읽는다."""
    try:
        metadata = pq.ParquetFile(path).metadata
    except Exception as exc:
        raise DownloadError(
            f"Parquet metadata 읽기에 실패했습니다: {path}"
        ) from exc

    return metadata.num_rows, metadata.num_columns


def stage_file(
    target: Target,
    resolved_revision: str,
    stage_dir: Path,
) -> StagedFile:
    """원격 원본을 staging 영역에 확보하고 최소 metadata를 계산한다."""
    LOGGER.info(
        "원본 확보: %s",
        target.remote_filename,
    )

    try:
        cached_path = Path(
            hf_hub_download(
                repo_id=REPO_ID,
                filename=target.remote_filename,
                repo_type=REPO_TYPE,
                revision=resolved_revision,
            )
        )

        staged_path = stage_dir / target.local_filename
        shutil.copyfile(
            cached_path,
            staged_path,
        )
    except Exception as exc:
        raise DownloadError(
            "다운로드 또는 staging 파일 생성에 실패했습니다: "
            f"{target.remote_filename}"
        ) from exc

    file_sha256 = calculate_sha256(staged_path)
    file_size_bytes = staged_path.stat().st_size

    if target.is_parquet:
        row_count, column_count = read_parquet_shape(
            staged_path
        )
    else:
        row_count = None
        column_count = None

    return StagedFile(
        target=target,
        path=staged_path,
        sha256=file_sha256,
        file_size_bytes=file_size_bytes,
        row_count=row_count,
        column_count=column_count,
    )


def validate_existing_raw(
    staged_files: list[StagedFile],
) -> None:
    """기존 Raw checksum이 다르면 덮어쓰기 전에 전체 작업을 중단한다."""
    for staged in staged_files:
        destination = (
            OUTPUT_DIR
            / staged.target.local_filename
        )

        if not destination.exists():
            continue

        if not destination.is_file():
            raise DownloadError(
                f"대상 경로가 파일이 아닙니다: {destination}"
            )

        existing_sha256 = calculate_sha256(
            destination
        )

        if existing_sha256 != staged.sha256:
            raise DownloadError(
                "기존 Raw와 현재 revision의 "
                "checksum이 다릅니다. "
                "자동 덮어쓰기를 중단합니다.\n"
                f"  파일: {destination}\n"
                f"  기존 SHA256: {existing_sha256}\n"
                f"  원격 SHA256: {staged.sha256}"
            )


def load_manifest() -> list[dict[str, str]]:
    """기존 manifest가 있으면 필수 컬럼을 검증하여 읽는다."""
    if not MANIFEST_PATH.exists():
        return []

    try:
        with MANIFEST_PATH.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file_obj:
            reader = csv.DictReader(file_obj)

            fieldnames = reader.fieldnames or []

            missing_columns = [
                column
                for column in MANIFEST_COLUMNS
                if column not in fieldnames
            ]

            if missing_columns:
                raise DownloadError(
                    "기존 manifest에 필수 컬럼이 없습니다: "
                    f"{missing_columns}"
                )

            return [
                {
                    column: row.get(column, "")
                    for column in MANIFEST_COLUMNS
                }
                for row in reader
            ]
    except OSError as exc:
        raise DownloadError(
            "기존 manifest 읽기에 실패했습니다."
        ) from exc


def make_manifest_rows(
    staged_files: list[StagedFile],
    requested_revision: str,
    resolved_revision: str,
    license_text: str,
) -> list[dict[str, str]]:
    """다운로드된 원본 파일 1개당 1행의 provenance 정보를 만든다."""
    downloaded_at = (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )

    rows: list[dict[str, str]] = []

    for staged in staged_files:
        target = staged.target

        source_url = hf_hub_url(
            repo_id=REPO_ID,
            filename=target.remote_filename,
            repo_type=REPO_TYPE,
            revision=resolved_revision,
        )

        rows.append(
            {
                "source": SOURCE,
                "source_url": source_url,
                "repo_id": REPO_ID,
                "remote_filename": target.remote_filename,
                "season": (
                    ""
                    if target.season is None
                    else str(target.season)
                ),
                "requested_revision": requested_revision,
                "resolved_revision": resolved_revision,
                "downloaded_at": downloaded_at,
                "row_count": (
                    ""
                    if staged.row_count is None
                    else str(staged.row_count)
                ),
                "column_count": (
                    ""
                    if staged.column_count is None
                    else str(staged.column_count)
                ),
                "file_size_bytes": str(
                    staged.file_size_bytes
                ),
                "sha256": staged.sha256,
                "license": license_text,
            }
        )

    return rows


def merge_manifest(
    existing_rows: list[dict[str, str]],
    new_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    """동일 revision과 checksum의 manifest 행이 반복 추가되지 않게 병합한다."""
    merged_rows: list[dict[str, str]] = []

    seen_keys: set[
        tuple[str, str, str, str]
    ] = set()

    for row in [
        *existing_rows,
        *new_rows,
    ]:
        key = (
            row["repo_id"],
            row["remote_filename"],
            row["resolved_revision"],
            row["sha256"],
        )

        if key in seen_keys:
            continue

        seen_keys.add(key)
        merged_rows.append(row)

    return merged_rows


def write_staged_manifest(
    rows: list[dict[str, str]],
    stage_dir: Path,
) -> Path:
    """manifest를 UTF-8 without BOM으로 staging 영역에 먼저 작성한다."""
    staged_manifest = (
        stage_dir
        / "download_manifest.csv"
    )

    try:
        with staged_manifest.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as file_obj:
            writer = csv.DictWriter(
                file_obj,
                fieldnames=MANIFEST_COLUMNS,
            )
            writer.writeheader()
            writer.writerows(rows)
    except OSError as exc:
        raise DownloadError(
            "staging manifest 작성에 실패했습니다."
        ) from exc

    return staged_manifest


def commit_staged_files(
    staged_files: list[StagedFile],
) -> None:
    """검증된 신규 원본만 최종 Raw 위치로 이동한다."""
    for staged in staged_files:
        destination = (
            OUTPUT_DIR
            / staged.target.local_filename
        )

        if destination.exists():
            LOGGER.info(
                "기존 Raw 유지: %s",
                destination.relative_to(
                    PROJECT_ROOT
                ),
            )
            continue

        try:
            os.replace(
                staged.path,
                destination,
            )
        except OSError as exc:
            raise DownloadError(
                "Raw 파일 최종 이동에 실패했습니다: "
                f"{destination}"
            ) from exc

        LOGGER.info(
            "Raw 저장 완료: %s",
            destination.relative_to(
                PROJECT_ROOT
            ),
        )


def commit_manifest(
    staged_manifest: Path,
) -> None:
    """manifest 내용이 변경된 경우에만 최종 파일을 교체한다."""
    if MANIFEST_PATH.exists():
        current_sha256 = calculate_sha256(
            MANIFEST_PATH
        )
        staged_sha256 = calculate_sha256(
            staged_manifest
        )

        if current_sha256 == staged_sha256:
            LOGGER.info(
                "manifest 변경 없음"
            )
            return

    try:
        os.replace(
            staged_manifest,
            MANIFEST_PATH,
        )
    except OSError as exc:
        raise DownloadError(
            "manifest 저장에 실패했습니다."
        ) from exc

    LOGGER.info(
        "manifest 저장 완료: %s",
        MANIFEST_PATH.relative_to(
            PROJECT_ROOT
        ),
    )


def log_repository_info(
    repo_files: list[str],
    targets: list[Target],
    requested_revision: str,
    resolved_revision: str,
    license_text: str,
) -> None:
    """실행 검증에 필요한 revision과 원격 파일 매핑을 출력한다."""
    LOGGER.info(
        "repo_id=%s",
        REPO_ID,
    )
    LOGGER.info(
        "requested_revision=%s",
        requested_revision,
    )
    LOGGER.info(
        "resolved_revision=%s",
        resolved_revision,
    )
    LOGGER.info(
        "license=%s",
        license_text,
    )

    LOGGER.info(
        "원격 파일 목록 (%d개):",
        len(repo_files),
    )

    for filename in repo_files:
        LOGGER.info(
            "  - %s",
            filename,
        )

    LOGGER.info(
        "선택된 원본 파일:"
    )

    for target in targets:
        label = (
            "schema"
            if target.season is None
            else str(target.season)
        )

        LOGGER.info(
            "  - %s: %s -> %s",
            label,
            target.remote_filename,
            target.local_filename,
        )


def log_file_metadata(
    staged_files: list[StagedFile],
) -> None:
    """Issue 검증에 필요한 파일 metadata를 요약 출력한다."""
    LOGGER.info(
        "파일 metadata 요약:"
    )

    for staged in staged_files:
        LOGGER.info(
            "  - %s | rows=%s | columns=%s "
            "| bytes=%s | sha256=%s",
            staged.target.local_filename,
            (
                staged.row_count
                if staged.row_count is not None
                else ""
            ),
            (
                staged.column_count
                if staged.column_count is not None
                else ""
            ),
            staged.file_size_bytes,
            staged.sha256,
        )


def run(
    requested_revision: str,
    seasons: tuple[int, ...],
    inspect_only: bool,
) -> None:
    """원격 구조 확인부터 Raw 저장과 manifest 기록까지 수행한다."""
    api = HfApi()

    (
        resolved_revision,
        license_text,
        repo_files,
    ) = resolve_repository(
        api,
        requested_revision,
    )

    targets = select_targets(
        repo_files,
        seasons,
    )

    log_repository_info(
        repo_files,
        targets,
        requested_revision,
        resolved_revision,
        license_text,
    )

    if inspect_only:
        LOGGER.info(
            "inspect-only 모드: "
            "다운로드를 수행하지 않습니다."
        )
        return

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.TemporaryDirectory(
        prefix=".hf_kbo_pbp_",
        dir=OUTPUT_DIR,
    ) as temporary_directory:
        stage_dir = Path(
            temporary_directory
        )

        staged_files = [
            stage_file(
                target,
                resolved_revision,
                stage_dir,
            )
            for target in targets
        ]

        # 모든 충돌을 확인한 뒤에만 기존 Raw 영역을 변경한다.
        validate_existing_raw(
            staged_files
        )

        manifest_rows = merge_manifest(
            load_manifest(),
            make_manifest_rows(
                staged_files,
                requested_revision,
                resolved_revision,
                license_text,
            ),
        )

        staged_manifest = (
            write_staged_manifest(
                manifest_rows,
                stage_dir,
            )
        )

        commit_staged_files(
            staged_files
        )

        commit_manifest(
            staged_manifest
        )

        log_file_metadata(
            staged_files
        )


def main() -> int:
    """CLI 입력 검증과 최상위 오류 처리를 수행한다."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    args = parse_args()

    try:
        seasons = normalize_seasons(
            args.seasons
        )

        run(
            requested_revision=args.revision,
            seasons=seasons,
            inspect_only=args.inspect_only,
        )
    except (DownloadError, OSError) as exc:
        LOGGER.error(
            "%s",
            exc,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())