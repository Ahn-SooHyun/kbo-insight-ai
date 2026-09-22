from __future__ import annotations

import argparse
import csv
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Sequence

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype


LOGGER = logging.getLogger(__name__)

DEFAULT_SEASONS = (2023, 2024, 2025, 2026)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "hf_kbo_pbp"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "hf_kbo_pbp"
)

MANIFEST_FILENAME = "download_manifest.csv"
SCHEMA_FILENAME = "schema.yaml"
VALIDATION_REPORT_FILENAME = "validation_report.csv"
SCHEMA_REPORT_FILENAME = "schema_report.csv"

CRITICAL_COLUMNS = (
    "game_pk",
    "game_date",
    "batter",
    "pitcher",
    "at_bat_number",
    "pitch_number",
)

PITCH_KEY_COLUMNS = (
    "game_pk",
    "at_bat_number",
    "pitch_number",
)

ANALYSIS_COLUMNS = (
    "events",
    "pitch_type",
    "release_speed_kmh",
    "plate_x",
    "plate_z",
)

MANIFEST_REQUIRED_COLUMNS = (
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
)

VALIDATION_REPORT_COLUMNS = (
    "season",
    "file_name",
    "file_exists",
    "load_success",
    "manifest_match",
    "resolved_revision",
    "raw_sha256",
    "row_count",
    "column_count",
    "game_date_parse_failures",
    "season_mismatch_rows",
    "duplicate_rows",
    "pitch_key_duplicate_rows",
    "game_pk_null_count",
    "game_date_null_count",
    "batter_null_count",
    "pitcher_null_count",
    "at_bat_number_null_count",
    "pitch_number_null_count",
    "events_null_rate",
    "pitch_type_null_rate",
    "release_speed_kmh_null_rate",
    "plate_x_null_rate",
    "plate_z_null_rate",
    "unique_games",
    "unique_batters",
    "unique_pitchers",
    "missing_critical_columns",
    "missing_analysis_columns",
    "status",
    "messages",
)

SCHEMA_REPORT_COLUMNS = (
    "season",
    "column",
    "dtype",
    "non_null_count",
    "null_count",
    "null_rate",
    "n_unique",
    "sample_value",
)


class ValidationError(RuntimeError):
    """검증 전체를 중단해야 하는 전제 조건 오류를 나타낸다."""


@dataclass
class DataFrameInspection:
    """한 시즌 DataFrame 검증에서 계산한 결과를 보관한다."""

    metrics: dict[str, object]
    schema_rows: list[dict[str, object]]
    fail_messages: list[str]
    warn_messages: list[str]


def parse_args() -> argparse.Namespace:
    """Raw 검증 스크립트의 CLI 인자를 파싱한다."""
    parser = argparse.ArgumentParser(
        description=(
            "로컬 KBO Play-by-Play Raw Dataset을 검증하고 "
            "validation/schema report를 생성합니다."
        )
    )

    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help=(
            "Raw 데이터 디렉터리입니다. "
            f"기본값: {DEFAULT_RAW_DIR}"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "검증 결과 저장 디렉터리입니다. "
            f"기본값: {DEFAULT_OUTPUT_DIR}"
        ),
    )

    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEASONS),
        help=(
            "검증할 시즌입니다. "
            "기본값: 2023 2024 2025 2026"
        ),
    )

    return parser.parse_args()


def normalize_seasons(
    values: Sequence[int],
) -> tuple[int, ...]:
    """중복 시즌을 제거하고 Issue 범위 밖 시즌 입력을 거부한다."""
    seasons = tuple(dict.fromkeys(values))

    if not seasons:
        raise ValidationError(
            "최소 1개 이상의 시즌을 지정해야 합니다."
        )

    unsupported = sorted(
        set(seasons) - set(DEFAULT_SEASONS)
    )

    if unsupported:
        raise ValidationError(
            f"지원하지 않는 시즌입니다: {unsupported}. "
            f"지원 시즌: {list(DEFAULT_SEASONS)}"
        )

    return seasons


def calculate_sha256(
    path: Path,
) -> str:
    """파일 전체를 읽어 SHA256 checksum을 계산한다."""
    try:
        with path.open("rb") as file_obj:
            return hashlib.file_digest(
                file_obj,
                "sha256",
            ).hexdigest()
    except OSError as exc:
        raise ValidationError(
            f"SHA256 계산에 실패했습니다: {path}"
        ) from exc


def load_manifest(
    path: Path,
) -> list[dict[str, str]]:
    """download manifest의 필수 컬럼을 확인하고 전체 행을 읽는다."""
    if not path.is_file():
        raise ValidationError(
            f"download manifest가 없습니다: {path}"
        )

    try:
        with path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file_obj:
            reader = csv.DictReader(file_obj)
            fieldnames = reader.fieldnames or []

            missing_columns = [
                column
                for column in MANIFEST_REQUIRED_COLUMNS
                if column not in fieldnames
            ]

            if missing_columns:
                raise ValidationError(
                    "download manifest에 "
                    "필수 컬럼이 없습니다: "
                    f"{missing_columns}"
                )

            rows = [
                {
                    column: (
                        row.get(column) or ""
                    ).strip()
                    for column in fieldnames
                }
                for row in reader
            ]

    except ValidationError:
        raise
    except (OSError, csv.Error) as exc:
        raise ValidationError(
            "download manifest 읽기에 "
            f"실패했습니다: {path}"
        ) from exc

    if not rows:
        raise ValidationError(
            f"download manifest가 비어 있습니다: {path}"
        )

    return rows


def resolve_snapshot_revision(
    raw_dir: Path,
    manifest_rows: Sequence[dict[str, str]],
) -> str:
    """
    현재 schema.yaml checksum과 일치하는
    Raw snapshot revision을 결정한다.
    """
    schema_path = raw_dir / SCHEMA_FILENAME

    if not schema_path.is_file():
        raise ValidationError(
            f"{SCHEMA_FILENAME} 파일이 없습니다: "
            f"{schema_path}"
        )

    schema_sha256 = calculate_sha256(
        schema_path
    )

    matching_rows = [
        row
        for row in manifest_rows
        if (
            PurePosixPath(
                row["remote_filename"]
            ).name.lower()
            == SCHEMA_FILENAME
            and row["sha256"].lower()
            == schema_sha256
        )
    ]

    revisions = {
        row["resolved_revision"]
        for row in matching_rows
        if row["resolved_revision"]
    }

    if not matching_rows:
        raise ValidationError(
            "현재 schema.yaml checksum과 "
            "일치하는 manifest provenance를 "
            "찾지 못했습니다."
        )

    if len(revisions) != 1:
        raise ValidationError(
            "현재 schema.yaml의 "
            "resolved_revision을 유일하게 "
            "결정할 수 없습니다: "
            f"{sorted(revisions)}"
        )

    return next(iter(revisions))


def find_season_manifest_row(
    manifest_rows: Sequence[dict[str, str]],
    season: int,
    resolved_revision: str,
) -> tuple[dict[str, str] | None, str | None]:
    """
    현재 snapshot에서 시즌 Parquet provenance 행을
    유일하게 찾는다.
    """
    candidates = [
        row
        for row in manifest_rows
        if (
            row["season"] == str(season)
            and row["resolved_revision"]
            == resolved_revision
        )
    ]

    if not candidates:
        return (
            None,
            (
                f"{season} 시즌 manifest provenance가 "
                f"snapshot {resolved_revision}에 없습니다."
            ),
        )

    if len(candidates) > 1:
        return (
            None,
            (
                f"{season} 시즌 manifest provenance가 "
                f"{len(candidates)}개라 "
                "유일하게 결정할 수 없습니다."
            ),
        )

    if not candidates[0]["sha256"]:
        return (
            None,
            (
                f"{season} 시즌 manifest "
                "SHA256이 비어 있습니다."
            ),
        )

    return candidates[0], None


def capture_raw_snapshot(
    raw_dir: Path,
    seasons: Sequence[int],
) -> dict[str, str | None]:
    """
    기대 Raw 파일의 존재 상태와 checksum을 기록한다.

    검증 전후 값을 비교하여 Raw 파일이
    변경되지 않았음을 확인한다.
    """
    paths = [
        *(
            raw_dir / f"{season}.parquet"
            for season in seasons
        ),
        raw_dir / SCHEMA_FILENAME,
        raw_dir / MANIFEST_FILENAME,
    ]

    snapshot: dict[str, str | None] = {}

    for path in paths:
        if path.is_file():
            snapshot[path.name] = (
                calculate_sha256(path)
            )
        elif path.exists():
            raise ValidationError(
                "Raw 기대 경로가 일반 파일이 "
                f"아닙니다: {path}"
            )
        else:
            snapshot[path.name] = None

    return snapshot


def parse_game_dates(
    series: pd.Series,
) -> pd.Series:
    """
    game_date 원본을 수정하지 않고
    datetime으로 안전하게 변환한다.
    """
    if is_datetime64_any_dtype(
        series.dtype
    ):
        source = series

    elif is_numeric_dtype(
        series.dtype
    ):
        source = (
            series
            .astype("string")
            .str.replace(
                r"\.0$",
                "",
                regex=True,
            )
        )

    else:
        source = (
            series
            .astype("string")
            .str.strip()
        )

    return pd.to_datetime(
        source,
        errors="coerce",
        utc=True,
    )


def get_null_rate(
    series: pd.Series,
    row_count: int,
) -> float | None:
    """
    빈 데이터셋을 제외하고
    0.0~1.0 범위의 결측률을 계산한다.
    """
    if row_count == 0:
        return None

    return float(
        series.isna().mean()
    )


def get_sample_value(
    series: pd.Series,
) -> str:
    """
    첫 non-null 값을
    CSV 한 행에 안전한 문자열로 변환한다.
    """
    non_null_values = (
        series[series.notna()]
    )

    if non_null_values.empty:
        return ""

    value = non_null_values.iloc[0]

    if isinstance(value, bytes):
        text = value.decode(
            "utf-8",
            errors="replace",
        )
    else:
        text = str(value)

    return (
        text
        .replace("\r", "\\r")
        .replace("\n", "\\n")
    )


def build_schema_rows(
    df: pd.DataFrame,
    season: int,
) -> list[dict[str, object]]:
    """
    실제 DataFrame의 모든 컬럼을
    시즌별 schema report 행으로 변환한다.
    """
    row_count = len(df)

    rows: list[
        dict[str, object]
    ] = []

    for column in df.columns:
        series = df[column]

        null_count = int(
            series.isna().sum()
        )

        rows.append(
            {
                "season": season,
                "column": str(column),
                "dtype": str(
                    series.dtype
                ),
                "non_null_count": (
                    row_count
                    - null_count
                ),
                "null_count": (
                    null_count
                ),
                "null_rate": (
                    get_null_rate(
                        series,
                        row_count,
                    )
                ),
                "n_unique": int(
                    series.nunique(
                        dropna=True
                    )
                ),
                "sample_value": (
                    get_sample_value(
                        series
                    )
                ),
            }
        )

    return rows


def inspect_dataframe(
    df: pd.DataFrame,
    season: int,
) -> DataFrameInspection:
    """
    한 시즌 DataFrame의 구조, 결측, 중복,
    시즌 일치 여부를 검증한다.
    """
    row_count, column_count = (
        df.shape
    )

    missing_critical_columns = [
        column
        for column in CRITICAL_COLUMNS
        if column not in df.columns
    ]

    missing_analysis_columns = [
        column
        for column in ANALYSIS_COLUMNS
        if column not in df.columns
    ]

    fail_messages: list[str] = []
    warn_messages: list[str] = []

    if missing_critical_columns:
        fail_messages.append(
            "Critical column 누락: "
            + ", ".join(
                missing_critical_columns
            )
        )

    if missing_analysis_columns:
        warn_messages.append(
            "분석 컬럼 누락: "
            + ", ".join(
                missing_analysis_columns
            )
        )

    if row_count == 0:
        warn_messages.append(
            "데이터 행이 0개입니다."
        )

    duplicate_rows = int(
        df.duplicated().sum()
    )

    metrics: dict[
        str,
        object,
    ] = {
        "row_count": int(
            row_count
        ),
        "column_count": int(
            column_count
        ),
        "game_date_parse_failures": None,
        "season_mismatch_rows": None,
        "duplicate_rows": (
            duplicate_rows
        ),
        "pitch_key_duplicate_rows": None,
        "missing_critical_columns": (
            ";".join(
                missing_critical_columns
            )
        ),
        "missing_analysis_columns": (
            ";".join(
                missing_analysis_columns
            )
        ),
    }

    if duplicate_rows > 0:
        warn_messages.append(
            "완전 중복 행 "
            f"{duplicate_rows}개 발견"
        )

    for column in CRITICAL_COLUMNS:
        metric_name = (
            f"{column}_null_count"
        )

        if column not in df.columns:
            metrics[metric_name] = None
            continue

        null_count = int(
            df[column]
            .isna()
            .sum()
        )

        metrics[metric_name] = (
            null_count
        )

        if null_count > 0:
            fail_messages.append(
                f"{column} 결측 "
                f"{null_count}개 발견"
            )

    if "game_date" in df.columns:
        parsed_dates = (
            parse_game_dates(
                df["game_date"]
            )
        )

        original_non_null = (
            df["game_date"]
            .notna()
        )

        parse_failures = int(
            (
                original_non_null
                & parsed_dates.isna()
            ).sum()
        )

        season_mismatch_rows = int(
            (
                parsed_dates.notna()
                & (
                    parsed_dates.dt.year
                    != season
                )
            ).sum()
        )

        metrics[
            "game_date_parse_failures"
        ] = parse_failures

        metrics[
            "season_mismatch_rows"
        ] = season_mismatch_rows

        if parse_failures > 0:
            fail_messages.append(
                "game_date 변환 실패 "
                f"{parse_failures}개 발견"
            )

        if season_mismatch_rows > 0:
            fail_messages.append(
                "game_date 기준 시즌 불일치 "
                f"{season_mismatch_rows}개 발견"
            )

    if all(
        column in df.columns
        for column in PITCH_KEY_COLUMNS
    ):
        complete_key_mask = (
            df[
                list(
                    PITCH_KEY_COLUMNS
                )
            ]
            .notna()
            .all(axis=1)
        )

        duplicate_key_rows = int(
            df.loc[
                complete_key_mask
            ]
            .duplicated(
                subset=list(
                    PITCH_KEY_COLUMNS
                ),
                keep="first",
            )
            .sum()
        )

        metrics[
            "pitch_key_duplicate_rows"
        ] = duplicate_key_rows

        if duplicate_key_rows > 0:
            fail_messages.append(
                "Pitch key 중복 "
                f"{duplicate_key_rows}개 발견"
            )

    for column in ANALYSIS_COLUMNS:
        metric_name = (
            f"{column}_null_rate"
        )

        if column not in df.columns:
            metrics[metric_name] = None
            continue

        metrics[metric_name] = (
            get_null_rate(
                df[column],
                row_count,
            )
        )

        if (
            row_count > 0
            and int(
                df[column]
                .isna()
                .sum()
            )
            == row_count
        ):
            warn_messages.append(
                f"{column} 컬럼이 "
                "전체 결측입니다."
            )

    unique_targets = {
        "unique_games": "game_pk",
        "unique_batters": "batter",
        "unique_pitchers": "pitcher",
    }

    for (
        metric_name,
        column,
    ) in unique_targets.items():
        if column in df.columns:
            metrics[metric_name] = int(
                df[column].nunique(
                    dropna=True
                )
            )
        else:
            metrics[metric_name] = None

    schema_rows = (
        build_schema_rows(
            df=df,
            season=season,
        )
    )

    return DataFrameInspection(
        metrics=metrics,
        schema_rows=schema_rows,
        fail_messages=fail_messages,
        warn_messages=warn_messages,
    )


def decide_status(
    fail_messages: Sequence[str],
    warn_messages: Sequence[str],
) -> str:
    """
    구조적 실패를 우선하여
    PASS/WARN/FAIL을 결정한다.
    """
    if fail_messages:
        return "FAIL"

    if warn_messages:
        return "WARN"

    return "PASS"


def format_messages(
    fail_messages: Sequence[str],
    warn_messages: Sequence[str],
) -> str:
    """
    검증 상태 근거를 사람이 읽을 수 있는
    단일 문자열로 만든다.
    """
    messages = [
        *(
            f"FAIL: {message}"
            for message
            in fail_messages
        ),
        *(
            f"WARN: {message}"
            for message
            in warn_messages
        ),
    ]

    return " | ".join(
        messages
    )


def make_empty_validation_row(
    season: int,
    resolved_revision: str,
) -> dict[str, object]:
    """
    실패 시에도 안정적인
    validation report schema를 유지한다.
    """
    row = {
        column: None
        for column
        in VALIDATION_REPORT_COLUMNS
    }

    row.update(
        {
            "season": season,
            "file_name": (
                f"{season}.parquet"
            ),
            "file_exists": False,
            "load_success": False,
            "manifest_match": False,
            "resolved_revision": (
                resolved_revision
            ),
            "raw_sha256": None,
            "missing_critical_columns": "",
            "missing_analysis_columns": "",
            "status": "FAIL",
            "messages": "",
        }
    )

    return row


def validate_season(
    raw_dir: Path,
    season: int,
    resolved_revision: str,
    manifest_rows: Sequence[
        dict[str, str]
    ],
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
]:
    """
    한 시즌 파일을 독립적으로 검증하여
    다른 시즌 실패와 격리한다.
    """
    row = (
        make_empty_validation_row(
            season=season,
            resolved_revision=(
                resolved_revision
            ),
        )
    )

    schema_rows: list[
        dict[str, object]
    ] = []

    fail_messages: list[str] = []
    warn_messages: list[str] = []

    file_path = (
        raw_dir
        / f"{season}.parquet"
    )

    (
        manifest_row,
        manifest_error,
    ) = find_season_manifest_row(
        manifest_rows=manifest_rows,
        season=season,
        resolved_revision=(
            resolved_revision
        ),
    )

    if manifest_error:
        fail_messages.append(
            manifest_error
        )

    if not file_path.is_file():
        if file_path.exists():
            fail_messages.append(
                "Raw 경로가 일반 파일이 "
                f"아닙니다: {file_path}"
            )
        else:
            fail_messages.append(
                "Raw 파일이 없습니다: "
                f"{file_path}"
            )

        row["status"] = (
            decide_status(
                fail_messages,
                warn_messages,
            )
        )

        row["messages"] = (
            format_messages(
                fail_messages,
                warn_messages,
            )
        )

        return (
            row,
            schema_rows,
        )

    row["file_exists"] = True

    raw_sha256 = (
        calculate_sha256(
            file_path
        )
    )

    row["raw_sha256"] = (
        raw_sha256
    )

    if manifest_row is not None:
        expected_sha256 = (
            manifest_row["sha256"]
            .lower()
        )

        manifest_match = (
            expected_sha256
            == raw_sha256
        )

        row["manifest_match"] = (
            manifest_match
        )

        if not manifest_match:
            fail_messages.append(
                "Raw SHA256과 manifest "
                "SHA256이 일치하지 않습니다."
            )

    try:
        df = pd.read_parquet(
            file_path,
            engine="pyarrow",
        )

        row["load_success"] = True

    except Exception as exc:
        fail_messages.append(
            "Parquet 로드 실패: "
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        row["status"] = (
            decide_status(
                fail_messages,
                warn_messages,
            )
        )

        row["messages"] = (
            format_messages(
                fail_messages,
                warn_messages,
            )
        )

        return (
            row,
            schema_rows,
        )

    inspection = (
        inspect_dataframe(
            df=df,
            season=season,
        )
    )

    row.update(
        inspection.metrics
    )

    schema_rows = (
        inspection.schema_rows
    )

    fail_messages.extend(
        inspection.fail_messages
    )

    warn_messages.extend(
        inspection.warn_messages
    )

    row["status"] = (
        decide_status(
            fail_messages,
            warn_messages,
        )
    )

    row["messages"] = (
        format_messages(
            fail_messages,
            warn_messages,
        )
    )

    del df

    return (
        row,
        schema_rows,
    )


def write_csv_report(
    rows: Sequence[
        dict[str, object]
    ],
    columns: Sequence[str],
    path: Path,
) -> None:
    """
    CSV를 UTF-8 without BOM으로 임시 파일에
    쓴 뒤 최종 파일로 교체한다.
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = (
        path.with_suffix(
            path.suffix + ".tmp"
        )
    )

    try:
        dataframe = pd.DataFrame(
            rows,
            columns=list(columns),
        )

        dataframe.to_csv(
            temp_path,
            index=False,
            encoding="utf-8",
        )

        temp_path.replace(
            path
        )

    except Exception as exc:
        if temp_path.exists():
            temp_path.unlink()

        raise ValidationError(
            "CSV report 저장에 "
            f"실패했습니다: {path}"
        ) from exc


def run_validation(
    raw_dir: Path,
    output_dir: Path,
    seasons: Sequence[int],
) -> int:
    """
    전체 검증을 실행하고 FAIL 존재 여부를
    프로세스 종료 코드로 반환한다.
    """
    normalized_seasons = (
        normalize_seasons(
            seasons
        )
    )

    if not raw_dir.is_dir():
        raise ValidationError(
            "Raw 디렉터리가 없습니다: "
            f"{raw_dir}"
        )

    manifest_path = (
        raw_dir
        / MANIFEST_FILENAME
    )

    manifest_rows = (
        load_manifest(
            manifest_path
        )
    )

    resolved_revision = (
        resolve_snapshot_revision(
            raw_dir=raw_dir,
            manifest_rows=(
                manifest_rows
            ),
        )
    )

    before_snapshot = (
        capture_raw_snapshot(
            raw_dir=raw_dir,
            seasons=(
                normalized_seasons
            ),
        )
    )

    LOGGER.info(
        "Raw 검증 시작: "
        "revision=%s, seasons=%s",
        resolved_revision,
        list(normalized_seasons),
    )

    validation_rows: list[
        dict[str, object]
    ] = []

    schema_rows: list[
        dict[str, object]
    ] = []

    for season in normalized_seasons:
        (
            validation_row,
            season_schema_rows,
        ) = validate_season(
            raw_dir=raw_dir,
            season=season,
            resolved_revision=(
                resolved_revision
            ),
            manifest_rows=(
                manifest_rows
            ),
        )

        validation_rows.append(
            validation_row
        )

        schema_rows.extend(
            season_schema_rows
        )

        LOGGER.info(
            "%s 시즌 검증 결과: %s",
            season,
            validation_row["status"],
        )

    after_snapshot = (
        capture_raw_snapshot(
            raw_dir=raw_dir,
            seasons=(
                normalized_seasons
            ),
        )
    )

    if (
        before_snapshot
        != after_snapshot
    ):
        changed_files = [
            name
            for name
            in before_snapshot
            if (
                before_snapshot[name]
                != after_snapshot[name]
            )
        ]

        raise ValidationError(
            "검증 실행 중 Raw 변경이 "
            "감지되었습니다: "
            + ", ".join(
                changed_files
            )
        )

    validation_report_path = (
        output_dir
        / VALIDATION_REPORT_FILENAME
    )

    schema_report_path = (
        output_dir
        / SCHEMA_REPORT_FILENAME
    )

    write_csv_report(
        rows=validation_rows,
        columns=(
            VALIDATION_REPORT_COLUMNS
        ),
        path=(
            validation_report_path
        ),
    )

    write_csv_report(
        rows=schema_rows,
        columns=(
            SCHEMA_REPORT_COLUMNS
        ),
        path=(
            schema_report_path
        ),
    )

    LOGGER.info(
        "validation report 저장: %s",
        validation_report_path,
    )

    LOGGER.info(
        "schema report 저장: %s",
        schema_report_path,
    )

    LOGGER.info(
        "Raw 불변성 확인 완료"
    )

    has_fail = any(
        row["status"] == "FAIL"
        for row
        in validation_rows
    )

    return (
        1
        if has_fail
        else 0
    )


def main() -> int:
    """CLI entry point를 실행한다."""
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
    )

    args = parse_args()

    try:
        return run_validation(
            raw_dir=args.raw_dir,
            output_dir=(
                args.output_dir
            ),
            seasons=args.seasons,
        )

    except ValidationError as exc:
        LOGGER.error(
            "%s",
            exc,
        )

        return 1

    except Exception:
        LOGGER.exception(
            "예상하지 못한 오류로 "
            "검증을 완료하지 못했습니다."
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )