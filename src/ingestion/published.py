"""승인된 공개 배포 파일의 수집과 불변 원본 버전 관리를 담당한다."""

import csv
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field, replace
from datetime import date, datetime
from io import BytesIO, StringIO
import json
import os
from pathlib import Path
import re
import shutil
import time
from urllib.parse import urlsplit
from uuid import uuid4

from ..settings import Settings
from .pilot import request_once, sha256, utc_now


SCHEMA_VERSION = 1
RUN_ID_PATTERN = re.compile(r"[0-9a-f]{32}")
REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")
IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}")
PATH_PART_PATTERN = re.compile(r"[A-Za-z0-9_.-]+")
SUPPORTED_FORMATS = frozenset({".csv", ".json", ".jsonl", ".parquet"})
DATA_KINDS = frozenset({
    "schedule_results",
    "play_by_play",
    "player_game",
    "starter_history",
    "postseason",
    "rules",
})
PERMITTED_USES = frozenset({
    "download", "store", "process", "train", "publish",
})
GAME_STATES = frozenset({
    "scheduled",
    "postponed",
    "cancelled",
    "suspended",
    "completed",
    "unknown",
})
SUCCESS_STATES = frozenset({"included", "no_data"})


class ManifestError(ValueError):
    """수집 명세 또는 실행 식별자가 올바르지 않을 때 발생한다."""


class ArtifactError(ValueError):
    """응답 파일의 최소 구조를 검사할 수 없을 때 발생한다."""


def _text(value: object) -> bool:
    """비어 있지 않은 문자열인지 확인한다."""
    return isinstance(value, str) and bool(value.strip())


def _string_list(value: object) -> bool:
    """중복 없는 문자열 목록인지 확인한다."""
    return (
        isinstance(value, list)
        and all(_text(item) for item in value)
        and len(set(value)) == len(value)
    )


def _public_url(value: str) -> bool:
    """인증 정보가 없는 공개 HTTPS 근거 주소인지 확인한다."""
    try:
        parts = urlsplit(value)
        return (
            parts.scheme == "https"
            and bool(parts.hostname)
            and parts.username is None
            and parts.password is None
            and parts.port in (None, 443)
        )
    except ValueError:
        return False


@dataclass(frozen=True)
class Target:
    """출처별 고정 배포 파일과 승인 범위 및 최소 검사 조건."""

    target_id: str
    source: str
    repository: str
    revision: str
    filename: str
    kind: str
    season: int
    competition: str
    start_date: str
    end_date: str
    publisher: str
    upstream: str
    evidence_url: str
    approval_ref: str | None
    permitted_uses: list[str]
    required_fields: list[str]
    max_bytes: int
    retries: int = 0
    interval_seconds: int = 1
    expected_bytes: int | None = None
    expected_sha256: str | None = None
    records_path: list[str] = field(default_factory=list)
    published_at: str | None = None
    publication_evidence: str | None = None
    season_complete: bool | None = None
    season_evidence: str | None = None
    empty_evidence: str | None = None
    game_id_field: str | None = None
    expected_game_ids: list[str] | None = None
    coverage_evidence: str | None = None
    game_status_field: str | None = None
    status_map: dict[str, str] = field(default_factory=dict)

    def validate(self, maximum: int) -> None:
        """파일 경로, 범위, 고정 판본, 이용 조건을 검증한다."""
        required_text = (
            self.target_id,
            self.source,
            self.repository,
            self.revision,
            self.filename,
            self.kind,
            self.competition,
            self.start_date,
            self.end_date,
            self.publisher,
            self.upstream,
            self.evidence_url,
        )
        if not all(_text(value) for value in required_text):
            raise ManifestError("대상 필수 문자열이 올바르지 않습니다.")

        if not IDENTIFIER_PATTERN.fullmatch(self.target_id):
            raise ManifestError("대상 식별자가 올바르지 않습니다.")

        if self.source not in {"github", "hf"}:
            raise ManifestError("지원하지 않는 공개 배포 출처입니다.")

        repository_parts = self.repository.split("/")
        if len(repository_parts) != 2:
            raise ManifestError("저장소는 owner/repository 형식이어야 합니다.")

        for part in repository_parts + self.filename.split("/"):
            if (
                part in {"", ".", ".."}
                or not PATH_PART_PATTERN.fullmatch(part)
            ):
                raise ManifestError("저장소 또는 파일 경로가 올바르지 않습니다.")

        if not REVISION_PATTERN.fullmatch(self.revision):
            raise ManifestError("고정된 40자리 판본이 필요합니다.")

        if Path(self.filename).suffix.lower() not in SUPPORTED_FORMATS:
            raise ManifestError("지원하지 않는 배포 파일 형식입니다.")

        if self.kind not in DATA_KINDS:
            raise ManifestError("지원하지 않는 자료 종류입니다.")

        if self.competition not in {"regular", "postseason"}:
            raise ManifestError("정규시즌과 포스트시즌을 구분해야 합니다.")

        if type(self.season) is not int or not 1982 <= self.season <= 9999:
            raise ManifestError("시즌 값이 올바르지 않습니다.")

        try:
            start = date.fromisoformat(self.start_date)
            end = date.fromisoformat(self.end_date)
        except ValueError:
            raise ManifestError("자료 기간은 ISO 날짜 형식이어야 합니다.") from None

        if start > end:
            raise ManifestError("자료 시작일이 종료일보다 늦습니다.")

        if not _public_url(self.evidence_url):
            raise ManifestError("공개 출처 및 이용 근거 주소가 필요합니다.")

        if not _string_list(self.permitted_uses):
            raise ManifestError("허용 용도 목록이 올바르지 않습니다.")

        if not set(self.permitted_uses) <= PERMITTED_USES:
            raise ManifestError("지원하지 않는 허용 용도입니다.")

        if self.permitted_uses and not _text(self.approval_ref):
            raise ManifestError("허용 용도에 대응하는 승인 근거가 필요합니다.")

        if not _string_list(self.required_fields) or not self.required_fields:
            raise ManifestError("최소 검사 필드가 필요합니다.")

        if type(self.max_bytes) is not int or not 1 <= self.max_bytes <= maximum:
            raise ManifestError("파일 크기 제한을 확인하세요.")

        if type(self.retries) is not int or not 0 <= self.retries <= 2:
            raise ManifestError("추가 시도 횟수는 0~2회여야 합니다.")

        if (
            type(self.interval_seconds) is not int
            or not 1 <= self.interval_seconds <= 60
        ):
            raise ManifestError("요청 간격은 1~60초여야 합니다.")

        if self.expected_bytes is not None and (
            type(self.expected_bytes) is not int
            or not 1 <= self.expected_bytes <= self.max_bytes
        ):
            raise ManifestError("기대 파일 크기가 올바르지 않습니다.")

        if self.expected_sha256 is not None and (
            not isinstance(self.expected_sha256, str)
            or not DIGEST_PATTERN.fullmatch(self.expected_sha256)
        ):
            raise ManifestError("기대 SHA-256이 올바르지 않습니다.")

        if not isinstance(self.records_path, list) or not all(
            _text(item) for item in self.records_path
        ):
            raise ManifestError("JSON 레코드 경로가 올바르지 않습니다.")

        if self.published_at is not None:
            try:
                published = datetime.fromisoformat(self.published_at)
            except (TypeError, ValueError):
                raise ManifestError("공개 시각이 올바르지 않습니다.") from None
            if published.tzinfo is None or not _text(self.publication_evidence):
                raise ManifestError("공개 시각의 시간대와 근거가 필요합니다.")

        if self.season_complete is not None and (
            type(self.season_complete) is not bool
            or not _text(self.season_evidence)
        ):
            raise ManifestError("시즌 완료 판정에는 근거가 필요합니다.")

        if self.empty_evidence is not None and not _text(self.empty_evidence):
            raise ManifestError("정상 무자료의 근거가 올바르지 않습니다.")

        if self.game_id_field is not None and (
            self.game_id_field not in self.required_fields
        ):
            raise ManifestError("경기 ID는 최소 검사 필드에 포함해야 합니다.")

        if self.expected_game_ids is not None and (
            not _string_list(self.expected_game_ids)
            or self.game_id_field is None
            or not _text(self.coverage_evidence)
        ):
            raise ManifestError("기대 경기 목록과 자동 확보 근거가 필요합니다.")

        if self.game_status_field is not None and (
            self.game_status_field not in self.required_fields
            or self.game_id_field is None
        ):
            raise ManifestError("경기 상태 검사에 필요한 필드가 없습니다.")

        if not isinstance(self.status_map, dict) or any(
            not _text(key) or value not in GAME_STATES
            for key, value in self.status_map.items()
        ):
            raise ManifestError("경기 상태 대응표가 올바르지 않습니다.")

    @property
    def approved(self) -> bool:
        """명세에서 다운로드와 저장이 함께 승인됐는지 반환한다."""
        return bool(self.approval_ref) and {
            "download", "store",
        } <= set(self.permitted_uses)

    @property
    def url(self) -> str:
        """인증 정보 없는 고정 판본 배포 주소만 구성한다."""
        if self.source == "hf":
            return (
                f"https://huggingface.co/datasets/{self.repository}/"
                f"resolve/{self.revision}/{self.filename}"
            )
        return (
            f"https://raw.githubusercontent.com/{self.repository}/"
            f"{self.revision}/{self.filename}"
        )

    @property
    def identity(self) -> list[str | int]:
        """내용 변경 비교용 논리 파일 식별자를 반환한다."""
        return [
            self.source,
            self.repository,
            self.filename,
            self.kind,
            self.season,
            self.competition,
        ]


def _read_json(path: Path) -> object:
    """UTF-8 JSON을 읽으며 입력 오류의 원문은 노출하지 않는다."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeError, ValueError, RecursionError):
        raise ManifestError("JSON 명세 또는 이력 형식이 올바르지 않습니다.") from None


def _targets(payload: object, settings: Settings) -> list[Target]:
    """고정 명세 전체를 검증하고 실행 대상 목록을 구성한다."""
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != SCHEMA_VERSION
        or not isinstance(payload.get("targets"), list)
        or not payload["targets"]
    ):
        raise ManifestError("유효한 자료 대상 명세가 필요합니다.")

    targets: list[Target] = []
    identifiers: set[str] = set()

    for item in payload["targets"]:
        if not isinstance(item, dict):
            raise ManifestError("자료 대상은 JSON 객체여야 합니다.")
        try:
            target = Target(**item)
            target.validate(settings.hf_collection_max_bytes)
        except TypeError:
            raise ManifestError("대상 명세 필드 또는 타입을 확인하세요.") from None

        identifier = target.target_id.casefold()
        if identifier in identifiers:
            raise ManifestError("자료 대상 식별자가 중복됩니다.")
        identifiers.add(identifier)
        targets.append(target)

    return targets


def _reject_constant(value: str) -> None:
    """JSON 규격에 없는 수치 상수를 거절한다."""
    raise ArtifactError("invalid_json_constant")


def _records(
    target: Target,
    body: bytes,
) -> Iterator[dict[str, object]]:
    """파일의 레코드를 읽되 표준 데이터로 변환하지 않는다."""
    suffix = Path(target.filename).suffix.lower()
    fields = set(target.required_fields)

    if suffix == ".csv":
        reader = csv.DictReader(
            StringIO(body.decode("utf-8-sig")),
            strict=True,
        )
        headings = reader.fieldnames
        if (
            headings is None
            or len(headings) != len(set(headings))
            or not fields <= set(headings)
        ):
            raise ArtifactError("missing_or_invalid_fields")

        for row in reader:
            if None in row or None in row.values():
                raise ArtifactError("invalid_csv_row")
            yield row
        return

    if suffix == ".parquet":
        try:
            import pyarrow.parquet as parquet
        except ImportError:
            raise ArtifactError("inspection_dependency_missing") from None

        table = parquet.ParquetFile(BytesIO(body))
        if not fields <= set(table.schema_arrow.names):
            raise ArtifactError("missing_or_invalid_fields")

        for batch in table.iter_batches(batch_size=4096):
            yield from batch.to_pylist()
        return

    if suffix == ".jsonl":
        for line in body.decode("utf-8-sig").splitlines():
            row = json.loads(line, parse_constant=_reject_constant)
            if not isinstance(row, dict) or not fields <= set(row):
                raise ArtifactError("invalid_record")
            yield row
        return

    payload = json.loads(body, parse_constant=_reject_constant)
    for key in target.records_path:
        if not isinstance(payload, dict) or key not in payload:
            raise ArtifactError("missing_records")
        payload = payload[key]

    if not isinstance(payload, list):
        raise ArtifactError("invalid_records")

    for row in payload:
        if not isinstance(row, dict) or not fields <= set(row):
            raise ArtifactError("invalid_record")
        yield row


def inspect_body(target: Target, body: bytes) -> dict[str, object]:
    """크기·해시·파싱 및 선택적인 원본 경기 범위를 검사한다."""
    if not body:
        raise ArtifactError("empty_body")

    if target.expected_bytes is not None and len(body) != target.expected_bytes:
        raise ArtifactError("artifact_size_mismatch")

    if (
        target.expected_sha256 is not None
        and sha256(body) != target.expected_sha256
    ):
        raise ArtifactError("artifact_hash_mismatch")

    count = 0
    invalid_ids = 0
    games: dict[str, set[str]] = {}

    try:
        for row in _records(target, body):
            count += 1
            if target.game_id_field is None:
                continue

            identity = row.get(target.game_id_field)
            if (
                not isinstance(identity, (str, int))
                or isinstance(identity, bool)
                or not str(identity).strip()
            ):
                invalid_ids += 1
                continue

            states = games.setdefault(str(identity), set())
            if target.game_status_field is not None:
                raw_state = row.get(target.game_status_field)
                states.add(target.status_map.get(str(raw_state), "unknown"))

    except ArtifactError:
        raise
    except (
        ValueError,
        OSError,
        UnicodeError,
        csv.Error,
        RecursionError,
        NotImplementedError,
    ):
        raise ArtifactError("artifact_parse_failed") from None

    if count == 0 and target.empty_evidence is None:
        raise ArtifactError("empty_records")

    state_counts: Counter[str] = Counter()
    conflicts: list[str] = []
    for game_id, states in games.items():
        if len(states) > 1:
            conflicts.append(game_id)
            state_counts["unknown"] += 1
        elif states:
            state_counts[next(iter(states))] += 1

    expected = target.expected_game_ids
    missing = None if expected is None else sorted(set(expected) - games.keys())
    additional = None if expected is None else sorted(games.keys() - set(expected))

    coverage_status = "not_assessed"
    if expected is not None:
        complete = not (
            missing
            or additional
            or invalid_ids
            or conflicts
            or state_counts.get("unknown", 0)
        )
        coverage_status = "matched" if complete else "incomplete"

    return {
        "rows": count,
        "game_ids": len(games),
        "invalid_game_id_rows": invalid_ids,
        "missing_game_ids": missing,
        "additional_game_ids": additional,
        "game_state_counts": dict(state_counts),
        "conflicting_game_ids": sorted(conflicts),
        "coverage_status": coverage_status,
        "season_complete": target.season_complete,
        "season_evidence": target.season_evidence,
        "semantic_validation": "not_performed",
    }


def _publish(path: Path, body: bytes, *, reuse: bool = False) -> bool:
    """원본을 배타 게시하고 같은 내용만 재사용한다.

    중단된 임시 파일을 성공 원본으로 사용하지 않는다.
    기존 파일의 내용을 수정하거나 덮어쓰지 않는다.
    """
    temporary = path.parent / f".{uuid4().hex}.part"
    try:
        with temporary.open("xb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())

        try:
            os.link(temporary, path)
            created = True
        except FileExistsError:
            if not reuse:
                raise
            created = False

        if sha256(path.read_bytes()) != sha256(body):
            raise OSError("저장된 원본의 해시가 일치하지 않습니다.")

        return created
    finally:
        temporary.unlink(missing_ok=True)


def _save_json(path: Path, payload: object) -> None:
    """완전한 JSON만 덮어쓰기 없이 게시한다."""
    body = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ) + "\n"
    ).encode("utf-8")
    _publish(path, body)


def _root(settings: Settings) -> Path:
    """기존 표본 이력과 분리된 전체 수집 저장 경로."""
    return settings.raw_dir / "published"


def _run_dir(settings: Settings, run_id: str) -> Path:
    """유효한 실행 식별자로 기존 실행 경로를 구성한다."""
    if not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id):
        raise ManifestError("실행 ID가 올바르지 않습니다.")
    return _root(settings) / "runs" / run_id


def _record(path: Path) -> dict[str, object]:
    """완전하게 기록된 JSON 객체를 반환한다."""
    result = _read_json(path)
    if not isinstance(result, dict):
        raise ManifestError("결과 이력 형식이 올바르지 않습니다.")
    return result


def _verify_object(root: Path, record: dict[str, object]) -> bool:
    """성공 기록과 연결된 불변 원본의 현재 해시를 대조한다."""
    digest = record.get("sha256")
    if not isinstance(digest, str) or not DIGEST_PATTERN.fullmatch(digest):
        return False
    path = root / "objects" / f"{digest}.raw"
    return path.is_file() and sha256(path.read_bytes()) == digest


def _previous_version(
    root: Path,
    target: Target,
) -> dict[str, object] | None:
    """동일한 논리 파일의 가장 최근 성공 관측을 찾는다."""
    candidates: list[dict[str, object]] = []
    for path in (root / "runs").glob("*/*.result.json"):
        record = _record(path)
        if (
            record.get("identity") == target.identity
            and record.get("status") in SUCCESS_STATES
        ):
            candidates.append(record)

    if not candidates:
        return None

    return max(candidates, key=lambda item: str(item.get("observed_at", "")))


def _collect_target(
    settings: Settings,
    target: Target,
    run_dir: Path,
) -> dict[str, object]:
    """파일 하나를 제한적으로 수집하고 시도별 이력을 기록한다."""
    root = _root(settings)
    previous = _previous_version(root, target)

    # 기존 표본 전송기의 요청 제한을 파일별로 적용한다.
    limits = replace(
        settings,
        collection_max_bytes=target.max_bytes,
        hf_collection_max_bytes=target.max_bytes,
    )

    if shutil.disk_usage(root).free < target.max_bytes:
        raise OSError("원본 저장 공간이 부족합니다.")

    base: dict[str, object] = {
        "target_id": target.target_id,
        "run_id": run_dir.name,
        "identity": target.identity,
        "revision": target.revision,
        "source": target.source,
        "source_published_at": target.published_at,
        "source_url": target.url,
        "approval_ref": target.approval_ref,
        "permitted_uses": target.permitted_uses,
    }

    for number in range(1, target.retries + 2):
        prefix = f"{target.target_id}.attempt-{number:02d}"
        time.sleep(target.interval_seconds * (2 ** (number - 1)))

        started_at = utc_now()
        _save_json(
            run_dir / f"{prefix}.started.json",
            {
                "run_id": run_dir.name,
                "target_id": target.target_id,
                "attempt": number,
                "request_started_at": started_at,
            },
        )

        attempt = request_once(
            target.url,
            limits,
            hf=target.source == "hf",
        )
        observed_at = utc_now()
        failure = attempt.failure
        inspection: dict[str, object] | None = None

        if failure is None and not attempt.body_complete:
            failure = "incomplete_response"

        if failure is None:
            try:
                inspection = inspect_body(target, attempt.body)
            except ArtifactError as exc:
                failure = str(exc)

        digest = sha256(attempt.body) if attempt.body else None

        result: dict[str, object] = {
            **base,
            "attempt": number,
            "request_started_at": started_at,
            "observed_at": observed_at,
            "http_status": attempt.http_status,
            "http_requests": attempt.http_requests,
            "redirects_followed": attempt.redirects,
            "bytes": len(attempt.body),
            "body_complete": attempt.body_complete,
            "sha256": digest,
            "failure": failure,
            "status": "failed",
            "object_path": None,
        }

        if failure is None and inspection is not None:
            object_path = root / "objects" / f"{digest}.raw"
            created = _publish(object_path, attempt.body, reuse=True)

            relationship = "first_observation"
            if previous is not None:
                relationship = (
                    "same_content"
                    if previous.get("sha256") == digest
                    else "correction_candidate"
                )

            result.update({
                "status": "no_data" if inspection["rows"] == 0 else "included",
                "object_path": object_path.relative_to(root).as_posix(),
                "object_created": created,
                "stored_at": utc_now(),
                "version_relationship": relationship,
                "previous_version": None if previous is None else {
                    "run_id": previous.get("run_id"),
                    "revision": previous.get("revision"),
                    "sha256": previous.get("sha256"),
                },
                "inspection": inspection,
            })

        _save_json(run_dir / f"{prefix}.json", result)

        if failure is None:
            return result

        if not attempt.retryable or number > target.retries:
            return result

    raise RuntimeError("파일 수집이 종료 상태에 도달하지 못했습니다.")


def report_files(settings: Settings, run_id: str) -> dict[str, object]:
    """실행 명세와 실제 원본을 읽기 전용으로 대조한다."""
    run_dir = _run_dir(settings, run_id)
    snapshot = _record(run_dir / "manifest.json")
    targets = _targets(snapshot, settings)
    results: list[dict[str, object]] = []
    counts: Counter[str] = Counter()

    for target in targets:
        result_path = run_dir / f"{target.target_id}.result.json"
        if result_path.is_file():
            result = _record(result_path)
            if (
                result.get("status") in SUCCESS_STATES
                and not _verify_object(_root(settings), result)
            ):
                result = {
                    **result,
                    "status": "damaged_or_missing",
                    "failure": "stored_object_verification_failed",
                }
        else:
            attempted = any(
                run_dir.glob(f"{target.target_id}.attempt-*.started.json")
            )
            result = {
                "target_id": target.target_id,
                "status": "interrupted" if attempted else "not_attempted",
            }

        results.append(result)
        counts[str(result["status"])] += 1

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "mode": snapshot.get("mode"),
        "resume_of": snapshot.get("resume_of"),
        "expected_targets": len(targets),
        "status_counts": dict(counts),
        "file_collection_complete": all(
            result.get("status") in SUCCESS_STATES
            for result in results
        ),
        "results": results,
    }


def collect_files(
    settings: Settings,
    *,
    manifest_path: Path | None = None,
    mode: str = "collect",
    run_id: str | None = None,
) -> dict[str, object]:
    """고정 명세 수집·갱신 또는 이전 실행의 파일 단위 재개를 수행한다."""
    if mode not in {"collect", "update", "resume"}:
        raise ManifestError("지원하지 않는 수집 실행 방식입니다.")

    previous_run: Path | None = None

    if mode == "resume":
        if run_id is None:
            raise ManifestError("재개할 실행 ID가 필요합니다.")
        previous_run = _run_dir(settings, run_id)
        payload = _read_json(previous_run / "manifest.json")
    else:
        if manifest_path is None:
            raise ManifestError("수집 대상 명세 경로가 필요합니다.")
        payload = _read_json(manifest_path)

    targets = _targets(payload, settings)
    root = _root(settings)
    (root / "objects").mkdir(parents=True, exist_ok=True)

    current = root / "runs" / uuid4().hex
    current.mkdir(parents=True, exist_ok=False)

    package_dir = Path(__file__).resolve().parents[1]
    code_hashes = {
        name: sha256((package_dir / name).read_bytes())
        for name in (
            "ingestion/published.py",
            "ingestion/pilot.py",
            "settings.py",
            "__main__.py",
        )
    }

    _save_json(
        current / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": current.name,
            "created_at": utc_now(),
            "mode": mode,
            "resume_of": run_id,
            "code_sha256": code_hashes,
            "timeout_seconds": settings.collection_timeout_seconds,
            "max_redirects": settings.hf_collection_max_redirects,
            "targets": [asdict(target) for target in targets],
        },
    )

    limited_sources: set[tuple[str, str]] = set()

    for target in targets:
        source_key = (target.source, target.repository)
        result_path = current / f"{target.target_id}.result.json"

        result: dict[str, object] = {
            "run_id": current.name,
            "target_id": target.target_id,
            "identity": target.identity,
            "revision": target.revision,
            "source_url": target.url,
            "observed_at": utc_now(),
        }

        if not target.approved:
            result.update({
                "status": "blocked",
                "failure": "collection_not_approved",
            })
            _save_json(result_path, result)
            continue

        if previous_run is not None:
            previous_path = (
                previous_run / f"{target.target_id}.result.json"
            )
            if previous_path.is_file():
                previous = _record(previous_path)
                if (
                    previous.get("status") in SUCCESS_STATES
                    and _verify_object(root, previous)
                ):
                    result = {
                        **previous,
                        "run_id": current.name,
                        "reused_from_run": previous_run.name,
                        "reused_at": utc_now(),
                        "object_created": False,
                    }
                    _save_json(result_path, result)
                    continue

        if source_key in limited_sources:
            result.update({
                "status": "access_limited",
                "failure": "source_access_limit",
            })
            _save_json(result_path, result)
            continue

        result = _collect_target(settings, target, current)
        _save_json(result_path, result)

        if result.get("http_status") in {403, 429}:
            limited_sources.add(source_key)

    return report_files(settings, current.name)