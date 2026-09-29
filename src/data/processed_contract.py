from __future__ import annotations

import hashlib
import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd


LOGGER = logging.getLogger(__name__)

PREDICTION_CONTRACT_VERSION = "prediction_dataset_contract_v1"
FEATURE_CATALOG_VERSION = "feature_catalog_v1"
SPLIT_VERSION = "season_split_v1"


class ProcessedContractError(RuntimeError):
    """Processed Dataset 공통 계약을 위반했을 때 발생하는 오류다."""


@dataclass(frozen=True)
class TemporalSplitRange:
    """Half-open 날짜 구간으로 정의한 고정 시간 Split 한 구간이다."""

    name: str
    start: pd.Timestamp
    end_exclusive: pd.Timestamp


def _normalize_timestamp(value: object, *, label: str) -> pd.Timestamp:
    """날짜 계약 비교에 사용할 timezone-naive 자정 Timestamp로 정규화한다."""
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ProcessedContractError(f"{label} 날짜를 해석할 수 없습니다: {value!r}") from exc

    if pd.isna(parsed):
        raise ProcessedContractError(f"{label} 날짜가 null일 수 없습니다.")
    if parsed.tzinfo is not None:
        parsed = parsed.tz_convert(None)
    return parsed.normalize()


def parse_split_ranges(
    raw_ranges: Sequence[Mapping[str, object]],
) -> tuple[TemporalSplitRange, ...]:
    """설정의 Split 구간을 검증 가능한 immutable 구조로 변환한다."""
    if not raw_ranges:
        raise ProcessedContractError("split_ranges 설정이 비어 있습니다.")

    ranges: list[TemporalSplitRange] = []
    seen_names: set[str] = set()
    for index, item in enumerate(raw_ranges):
        name = str(item.get("name", "")).strip()
        if not name:
            raise ProcessedContractError(f"split_ranges[{index}].name이 비어 있습니다.")
        if name in seen_names:
            raise ProcessedContractError(f"Split 이름이 중복되었습니다: {name}")
        seen_names.add(name)

        start = _normalize_timestamp(
            item.get("start"),
            label=f"split_ranges[{index}].start",
        )
        end_exclusive = _normalize_timestamp(
            item.get("end_exclusive"),
            label=f"split_ranges[{index}].end_exclusive",
        )
        if start >= end_exclusive:
            raise ProcessedContractError(
                f"Split 날짜 구간이 역전되었거나 비어 있습니다: {name} "
                f"[{start.date()}, {end_exclusive.date()})"
            )
        ranges.append(
            TemporalSplitRange(
                name=name,
                start=start,
                end_exclusive=end_exclusive,
            )
        )

    ordered = tuple(sorted(ranges, key=lambda item: (item.start, item.end_exclusive, item.name)))
    for previous, current in zip(ordered, ordered[1:]):
        if previous.end_exclusive > current.start:
            raise ProcessedContractError(
                "Split 날짜 구간이 중복됩니다: "
                f"{previous.name}와 {current.name}"
            )
        if previous.end_exclusive < current.start:
            raise ProcessedContractError(
                "Split 날짜 구간 사이에 누락 구간이 있습니다: "
                f"{previous.name}와 {current.name}"
            )
    return ordered


def assign_temporal_split(
    prediction_date: object,
    ranges: Sequence[TemporalSplitRange],
) -> str:
    """Prediction Date를 정확히 하나의 고정 시간 Split에 할당한다."""
    date = _normalize_timestamp(prediction_date, label="prediction_date")
    matches = [
        item.name
        for item in ranges
        if item.start <= date < item.end_exclusive
    ]
    if len(matches) != 1:
        raise ProcessedContractError(
            "prediction_date가 정확히 하나의 Split에 속하지 않습니다: "
            f"{date.date()} -> {matches}"
        )
    return matches[0]


def assign_temporal_split_series(
    prediction_dates: pd.Series,
    ranges: Sequence[TemporalSplitRange],
) -> pd.Series:
    """Series의 모든 날짜에 결정적으로 Split을 할당한다."""
    values = [assign_temporal_split(value, ranges) for value in prediction_dates.tolist()]
    return pd.Series(values, index=prediction_dates.index, dtype="string")


def validate_label_interval(
    *,
    label_start: object,
    label_end_exclusive: object,
    label_available_at: object | None = None,
) -> None:
    """#26/#27이 재사용할 최소 half-open Label Interval 불변식을 검증한다."""
    start = _normalize_timestamp(label_start, label="label_start")
    end = _normalize_timestamp(label_end_exclusive, label="label_end_exclusive")
    if start >= end:
        raise ProcessedContractError(
            "label interval은 비어 있거나 역전될 수 없습니다: "
            f"[{start.date()}, {end.date()})"
        )
    if label_available_at is not None:
        available = _normalize_timestamp(label_available_at, label="label_available_at")
        if available < end:
            raise ProcessedContractError(
                "label_available_at은 label_end_exclusive보다 빠를 수 없습니다."
            )


def validate_x_allowlist(
    *,
    available_columns: Iterable[str],
    x_columns: Sequence[str],
    forbidden_columns: Iterable[str],
    forbidden_prefixes: Sequence[str] = (),
) -> None:
    """명시적 X allowlist가 결과/품질/분할/제어 정보를 포함하지 않는지 검증한다."""
    available = set(available_columns)
    forbidden = set(forbidden_columns)
    x_list = list(x_columns)

    if not x_list:
        raise ProcessedContractError("x_columns allowlist가 비어 있습니다.")
    if len(x_list) != len(set(x_list)):
        raise ProcessedContractError("x_columns allowlist에 중복 컬럼이 있습니다.")

    missing = [column for column in x_list if column not in available]
    if missing:
        raise ProcessedContractError(f"x_columns에 Dataset에 없는 컬럼이 있습니다: {missing}")

    direct_violations = [column for column in x_list if column in forbidden]
    prefix_violations = [
        column
        for column in x_list
        if any(column.startswith(prefix) for prefix in forbidden_prefixes)
    ]
    violations = sorted(set(direct_violations + prefix_violations))
    if violations:
        raise ProcessedContractError(
            "x_columns에 금지된 Target/Audit/Quality/Control 컬럼이 포함되었습니다: "
            f"{violations}"
        )


def calculate_sha256(path: Path) -> str:
    """파일 내용을 변경하지 않고 SHA256을 계산한다."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ProcessedContractError(f"SHA256 계산에 실패했습니다: {path}") from exc
    return digest.hexdigest()


def calculate_input_hashes(paths: Mapping[str, Path]) -> dict[str, str]:
    """Named Input Artifact들의 SHA256을 결정적인 key 순서로 계산한다."""
    return {name: calculate_sha256(paths[name]) for name in sorted(paths)}


def assert_input_hashes_unchanged(
    before: Mapping[str, str],
    paths: Mapping[str, Path],
) -> dict[str, str]:
    """Builder 전후 Input Artifact가 byte-level로 바뀌지 않았는지 검증한다."""
    after = calculate_input_hashes(paths)
    missing = sorted(set(before) - set(after))
    extra = sorted(set(after) - set(before))
    changed = sorted(
        name
        for name in set(before) & set(after)
        if before[name] != after[name]
    )
    if missing or extra or changed:
        raise ProcessedContractError(
            "Builder 실행 중 입력 Artifact 집합 또는 SHA256이 변경되었습니다: "
            f"missing={missing}, extra={extra}, changed={changed}"
        )
    return after


def _json_scalar(value: object) -> object:
    """Fingerprint JSON에서 pandas/numpy scalar를 결정적인 JSON 값으로 변환한다."""
    if value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def stable_json_fingerprint(payload: Mapping[str, object]) -> str:
    """변동 metadata를 제외한 Mapping의 결정적 SHA256을 계산한다."""
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def content_fingerprint(
    frame: pd.DataFrame,
    *,
    sort_columns: Sequence[str] | None = None,
) -> str:
    """Column 순서, dtype, 결정적 Row 순서를 포함한 Dataset SHA256을 계산한다."""
    source = frame
    if sort_columns:
        missing = [column for column in sort_columns if column not in frame.columns]
        if missing:
            raise ProcessedContractError(
                f"Fingerprint sort column이 Dataset에 없습니다: {missing}"
            )
        source = (
            frame.sort_values(list(sort_columns), kind="mergesort")
            .reset_index(drop=True)
        )

    payload = {
        "schema": [
            {"name": column, "dtype": str(source[column].dtype)}
            for column in source.columns
        ],
        "rows": [
            [_json_scalar(value) for value in row]
            for row in source.itertuples(index=False, name=None)
        ],
    }
    return stable_json_fingerprint(payload)


def schema_manifest(frame: pd.DataFrame) -> list[dict[str, str]]:
    """Output Column 순서와 dtype을 직렬화한다."""
    return [
        {"name": column, "dtype": str(frame[column].dtype)}
        for column in frame.columns
    ]


def path_is_within(path: Path, parent: Path) -> bool:
    """path가 parent 자신 또는 하위 경로인지 확인한다."""
    resolved_path = path.resolve(strict=False)
    resolved_parent = parent.resolve(strict=False)
    return resolved_path == resolved_parent or resolved_parent in resolved_path.parents


def validate_output_paths(
    *,
    input_paths: Sequence[Path],
    output_paths: Sequence[Path],
    project_root: Path,
) -> None:
    """입력 덮어쓰기와 Raw/Interim 영역으로의 Processed Output 생성을 차단한다."""
    resolved_inputs = [path.resolve(strict=False) for path in input_paths]
    resolved_outputs = [path.resolve(strict=False) for path in output_paths]

    if len(resolved_outputs) != len(set(resolved_outputs)):
        raise ProcessedContractError("Processed Output 경로끼리 충돌합니다.")
    collisions = sorted(set(resolved_inputs) & set(resolved_outputs))
    if collisions:
        raise ProcessedContractError(
            "Processed Output 경로가 입력 Artifact와 충돌합니다: "
            f"{[str(path) for path in collisions]}"
        )

    protected_roots = (
        project_root / "data" / "raw",
        project_root / "data" / "interim",
    )
    for output in resolved_outputs:
        for protected_root in protected_roots:
            if path_is_within(output, protected_root):
                raise ProcessedContractError(
                    "Processed Output을 Raw/Interim 영역에 쓸 수 없습니다: "
                    f"{output}"
                )


def invalidate_completion_marker(path: Path) -> None:
    """재실행 시작 전에 이전 완료 Manifest를 제거해 실패 Run의 오인 소비를 막는다."""
    if not path.exists():
        return
    if path.is_dir():
        raise ProcessedContractError(f"완료 Manifest 경로가 디렉터리입니다: {path}")
    try:
        path.unlink()
    except OSError as exc:
        raise ProcessedContractError(
            f"이전 완료 Manifest를 무효화하지 못했습니다: {path}"
        ) from exc


def read_json(path: Path, *, label: str) -> dict[str, object]:
    """UTF-8 JSON Artifact를 Mapping으로 읽는다."""
    if not path.is_file():
        raise ProcessedContractError(f"{label} JSON이 없습니다: {path}")
    try:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ProcessedContractError(f"{label} JSON을 읽지 못했습니다: {path}") from exc
    if not isinstance(payload, dict):
        raise ProcessedContractError(f"{label} JSON 최상위 값은 object여야 합니다.")
    return payload


def read_parquet(path: Path, *, label: str) -> pd.DataFrame:
    """명시적 pyarrow 엔진으로 Parquet을 읽는다."""
    if not path.is_file():
        raise ProcessedContractError(f"{label} Parquet이 없습니다: {path}")
    try:
        return pd.read_parquet(path, engine="pyarrow")
    except Exception as exc:
        raise ProcessedContractError(f"{label} Parquet을 읽지 못했습니다: {path}") from exc


def write_parquet_atomic(frame: pd.DataFrame, output_path: Path) -> None:
    """Parquet을 같은 디렉터리 임시 파일에 쓴 뒤 원자적으로 교체한다."""
    if output_path.exists() and output_path.is_dir():
        raise ProcessedContractError(f"Parquet Output 경로가 디렉터리입니다: {output_path}")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ProcessedContractError(
            f"Parquet Output 디렉터리를 만들 수 없습니다: {output_path.parent}"
        ) from exc

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{output_path.stem}.",
            suffix=".parquet",
            dir=output_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        frame.to_parquet(temporary_path, engine="pyarrow", index=False)
        temporary_path.replace(output_path)
    except Exception as exc:
        raise ProcessedContractError(f"Parquet 저장에 실패했습니다: {output_path}") from exc
    finally:
        if temporary_path is not None and temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError:
                LOGGER.warning("임시 Parquet 파일을 삭제하지 못했습니다: %s", temporary_path)


def write_json_atomic(payload: Mapping[str, object], output_path: Path) -> None:
    """JSON을 같은 디렉터리 임시 파일에 쓴 뒤 원자적으로 교체한다."""
    if output_path.exists() and output_path.is_dir():
        raise ProcessedContractError(f"JSON Output 경로가 디렉터리입니다: {output_path}")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ProcessedContractError(
            f"JSON Output 디렉터리를 만들 수 없습니다: {output_path.parent}"
        ) from exc

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{output_path.stem}.",
            suffix=".json",
            dir=output_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(
                payload,
                temporary_file,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                default=str,
            )
            temporary_file.write("\n")
        temporary_path.replace(output_path)
    except Exception as exc:
        raise ProcessedContractError(f"JSON 저장에 실패했습니다: {output_path}") from exc
    finally:
        if temporary_path is not None and temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError:
                LOGGER.warning("임시 JSON 파일을 삭제하지 못했습니다: %s", temporary_path)
