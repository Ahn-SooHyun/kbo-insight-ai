"""공개 배포 원본의 인수, 판본 선택 및 중간 자료 생성을 담당한다."""

from collections import Counter
from collections.abc import Iterator
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re


Row = dict[str, object]

RUN_PATTERN = re.compile(r"[0-9a-f]{32}")
HASH_PATTERN = re.compile(r"[0-9a-f]{64}")

POSTSEASON_STAGES = (
    "wildCard",
    "semiPlayoff",
    "playoff",
    "koreanSeries",
)

EXPECTED_TARGETS = frozenset({
    *(f"dent-results-{year}" for year in range(2001, 2027)),
    *(f"hf-pbp-{year}" for year in range(2023, 2027)),
    "window-starters-2023",
    "window-pitching-2023",
    "window-daily-results-2026",
    "dent-postseason-2026-wildcard-probe",
    "lopes-postseason-2010-2018",
    *(
        f"dialektike-postseason-{month}"
        for month in (
            "2019-10",
            "2020-11",
            "2021-11",
            "2022-10",
            "2022-11",
            "2023-10",
            "2023-11",
            "2024-10",
            "2025-10",
        )
    ),
})

USE_APPROVAL = {
    "approved_on": "2026-11-25",
    "scope": "비영리 대학 과제용 로컬 표준화·학습·내부 평가",
    "approval_reference": (
        "https://chatgpt.com/library?"
        "file_id=libfile_6490a996b59881918e281ff5b107e97b"
        "&account_id=personal"
    ),
    "redistribution": False,
    "historical_permissions_modified": False,
    "source_conditions": {
        "hf": "CC BY 4.0 배포본과 별도 원출처 조건을 구분",
        "dent": "자료별 이용허락 미확인, 승인된 로컬 비영리 용도",
        "window": "실제 경기 기록의 출처 조건과 시즌 집계 조건을 구분",
        "lopes": "교육 분석 용도 및 파생 원출처 조건",
        "dialektike": "코드 GPL 표기와 상류 일정 자료 권리를 구분",
    },
}


class SourceError(ValueError):
    """수집 이력, 원본 무결성 또는 스키마 오류."""


@dataclass(frozen=True)
class SourceVersion:
    """수집 명세와 실제 저장 결과가 연결된 불변 원본 판본."""

    target: Row
    result: Row
    path: Path

    @property
    def target_id(self) -> str:
        return str(self.target["target_id"])

    @property
    def sha256(self) -> str:
        return str(self.result["sha256"])

    @property
    def revision(self) -> str:
        return str(self.target["revision"])

    @property
    def filename(self) -> str:
        return str(self.target["filename"])

    @property
    def version_key(self) -> str:
        return f"{self.target_id}:{self.sha256}"


@dataclass(frozen=True)
class Intake:
    """전체 판본, 채택 판본 및 보존된 실행 이력."""

    versions: tuple[SourceVersion, ...]
    selected: dict[str, SourceVersion]
    history: tuple[Row, ...]
    raw_bytes: int
    object_count: int


def sha256_file(path: Path) -> str:
    """대용량 파일을 분할하여 SHA-256을 계산한다."""
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def read_json(path: Path) -> object:
    """기존 JSON을 변경하지 않고 읽는다."""
    try:
        return json.loads(path.read_bytes())
    except (ValueError, UnicodeError) as exc:
        raise SourceError(f"JSON 형식 오류: {path}") from exc


def as_row(value: object) -> Row:
    """JSON 객체 구조를 확인한다."""
    if not isinstance(value, dict):
        raise SourceError("JSON 객체가 필요합니다.")

    if not all(isinstance(key, str) for key in value):
        raise SourceError("문자열 필드명이 필요합니다.")

    return value


def write_json(path: Path, value: object) -> None:
    """NaN을 허용하지 않는 UTF-8 JSON 파일을 신규 저장한다."""
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(
            value,
            stream,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
            sort_keys=True,
        )
        stream.write("\n")


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise SourceError(f"필수 문자열 오류: {field}")
    return value


def _rows_at_path(value: object, fields: object) -> list[Row]:
    """명세에 저장된 records_path를 사용해 레코드를 읽는다."""
    if not isinstance(fields, list):
        raise SourceError("records_path 형식 오류")

    for field in fields:
        value = as_row(value)[str(field)]

    if not isinstance(value, list):
        raise SourceError("레코드 배열이 필요합니다.")

    return [as_row(row) for row in value]


def intake_sources(
    published_dir: Path,
    *,
    as_of_run: str | None = None,
    expected_targets: frozenset[str] = EXPECTED_TARGETS,
) -> Intake:
    """선정 범위의 전체 원본을 검증하고 시점별 채택 판본을 결정한다.

    같은 해시를 참조하는 재실행은 새 원본으로 세지 않는다.
    실패한 수집 결과와 원본의 이전 판본은 제거하지 않는다.
    """
    published_dir = published_dir.resolve()

    runs_dir = published_dir / "runs"
    objects_dir = published_dir / "objects"

    manifests: list[tuple[str, str, Path, Row]] = []

    for path in runs_dir.glob("*/manifest.json"):
        manifest = as_row(read_json(path))
        run_id = _text(manifest.get("run_id"), "run_id")

        if (
            not RUN_PATTERN.fullmatch(run_id)
            or run_id != path.parent.name
        ):
            raise SourceError(f"실행 ID 불일치: {path}")

        created_at = _text(
            manifest.get("created_at"),
            "created_at",
        )

        manifests.append((
            created_at,
            run_id,
            path,
            manifest,
        ))

    if not manifests:
        raise SourceError("수집 실행 이력이 없습니다.")

    manifests.sort(key=lambda item: (item[0], item[1]))

    if as_of_run is not None and not any(
        item[1] == as_of_run for item in manifests
    ):
        raise SourceError("지정한 수집 실행 ID가 없습니다.")

    versions: dict[str, SourceVersion] = {}
    selected: dict[str, SourceVersion] = {}
    verified_objects: dict[str, int] = {}
    history: list[Row] = []

    selection_open = True

    for _, run_id, manifest_path, manifest in manifests:
        targets = manifest.get("targets")

        if not isinstance(targets, list):
            raise SourceError(f"대상 명세 오류: {run_id}")

        for item in targets:
            target = as_row(item)
            target_id = _text(
                target.get("target_id"),
                "target_id",
            )

            if target_id not in expected_targets:
                raise SourceError(
                    f"승인된 선정 범위 외 대상: {target_id}"
                )

            result_path = (
                manifest_path.parent / f"{target_id}.result.json"
            )
            result = as_row(read_json(result_path))

            if (
                result.get("run_id") != run_id
                or result.get("target_id") != target_id
            ):
                raise SourceError(
                    f"대상과 결과 이력 불일치: {target_id}"
                )

            identity = [
                target.get("source"),
                target.get("repository"),
                target.get("filename"),
                target.get("kind"),
                target.get("season"),
                target.get("competition"),
            ]

            if result.get("identity") != identity:
                raise SourceError(
                    f"논리적 수집 대상 불일치: {target_id}"
                )

            history.append({
                "run_id": run_id,
                "target": target,
                "result": result,
                "manifest_path": (
                    manifest_path.relative_to(published_dir).as_posix()
                ),
                "result_path": (
                    result_path.relative_to(published_dir).as_posix()
                ),
                "eligible_at_selected_run": selection_open,
            })

            if result.get("status") != "included":
                continue

            if (
                result.get("body_complete") is not True
                or result.get("revision") != target.get("revision")
            ):
                raise SourceError(
                    f"불완전한 성공 이력: {target_id}"
                )

            digest = _text(result.get("sha256"), "sha256")

            if not HASH_PATTERN.fullmatch(digest):
                raise SourceError("잘못된 원본 해시")

            expected_path = f"objects/{digest}.raw"

            if result.get("object_path") != expected_path:
                raise SourceError(
                    f"원본 저장 경로 불일치: {target_id}"
                )

            path = (published_dir / expected_path).resolve()

            if path.parent != objects_dir.resolve():
                raise SourceError("원본 저장 경계를 벗어난 경로")

            if not path.is_file():
                raise SourceError(f"원본 누락: {target_id}")

            size = path.stat().st_size

            if size != result.get("bytes"):
                raise SourceError(f"원본 크기 불일치: {target_id}")

            expected_hash = target.get("expected_sha256")

            if expected_hash is not None and expected_hash != digest:
                raise SourceError(
                    f"명세 해시 불일치: {target_id}"
                )

            expected_bytes = target.get("expected_bytes")

            if expected_bytes is not None and expected_bytes != size:
                raise SourceError(
                    f"명세 크기 불일치: {target_id}"
                )

            if digest not in verified_objects:
                if sha256_file(path) != digest:
                    raise SourceError(
                        f"원본 SHA-256 불일치: {target_id}"
                    )

                verified_objects[digest] = size

            source = SourceVersion(target, result, path)

            canonical = versions.setdefault(
                source.version_key,
                source,
            )

            if selection_open:
                selected[target_id] = canonical

        if run_id == as_of_run:
            selection_open = False

    missing_targets = expected_targets - selected.keys()

    if missing_targets:
        raise SourceError(
            "채택할 성공 판본이 없는 대상: "
            + ", ".join(sorted(missing_targets))
        )

    actual_objects = {
        path.stem
        for path in objects_dir.glob("*.raw")
        if path.is_file()
    }

    if actual_objects != verified_objects.keys():
        raise SourceError(
            "성공 수집 이력과 실제 원본 목록이 일치하지 않습니다."
        )

    return Intake(
        versions=tuple(
            sorted(
                versions.values(),
                key=lambda source: source.version_key,
            )
        ),
        selected=selected,
        history=tuple(history),
        raw_bytes=sum(verified_objects.values()),
        object_count=len(verified_objects),
    )


def _json_records(source: SourceVersion) -> list[Row]:
    """Dent 포스트시즌은 지정된 네 단계 전체를 펼쳐 읽는다."""
    value = read_json(source.path)

    if source.target_id != "dent-postseason-2026-wildcard-probe":
        return _rows_at_path(
            value,
            source.target.get("records_path"),
        )

    postseason = as_row(as_row(value)["2026"])
    records: list[Row] = []

    for stage in POSTSEASON_STAGES:
        series = as_row(postseason[stage])
        games = _rows_at_path(series, ["games"])

        for game in games:
            records.append({
                **game,
                "_stage": stage,
                "_series_complete": series.get("isComplete"),
                "_season_complete": postseason.get("isComplete"),
            })

    return records


def read_source_rows(
    source: SourceVersion,
) -> Iterator[Row]:
    """원본 확장자가 아닌 명세의 실제 파일 형식에 따라 읽는다."""
    suffix = Path(source.filename).suffix.lower()

    if suffix == ".json":
        yield from _json_records(source)
        return

    if suffix == ".jsonl":
        with source.path.open(
            encoding="utf-8-sig",
        ) as stream:
            for line in stream:
                if line.strip():
                    yield as_row(json.loads(line))
        return

    if suffix == ".csv":
        with source.path.open(
            encoding="utf-8-sig",
            newline="",
        ) as stream:
            reader = csv.DictReader(stream)

            if not reader.fieldnames:
                raise SourceError("CSV 헤더가 없습니다.")

            if len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise SourceError("CSV 중복 필드명")

            for row in reader:
                if None in row or None in row.values():
                    raise SourceError("CSV 필드 개수 불일치")
                yield as_row(row)
        return

    raise SourceError(
        f"지원하지 않는 원본 형식: {source.filename}"
    )


def materialize_source(
    source: SourceVersion,
    directory: Path,
) -> Row:
    """원본 대응 번호를 추가한 중간 자료와 스키마 증거를 저장한다."""
    suffix = Path(source.filename).suffix.lower()
    prefix = f"{source.target_id}-{source.sha256}"
    required = set(source.target.get("required_fields", []))

    if suffix == ".parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pq.read_table(source.path)

        if not required.issubset(table.column_names):
            raise SourceError(
                f"필수 Parquet 필드 누락: {source.target_id}"
            )

        if "_source_row" in table.column_names:
            raise SourceError("원본 대응 필드명 충돌")

        output = directory / f"{prefix}.parquet"

        table = table.append_column(
            "_source_row",
            pa.array(range(table.num_rows), type=pa.int64()),
        )

        pq.write_table(table, output)

        rows = table.num_rows
        schema = {
            field.name: {
                "type": str(field.type),
                "null_count": table[field.name].null_count,
            }
            for field in table.schema
        }

    else:
        output = directory / f"{prefix}.jsonl"
        field_types: dict[str, set[str]] = {}
        null_counts: Counter[str] = Counter()
        rows = 0

        with output.open(
            "x",
            encoding="utf-8",
            newline="\n",
        ) as stream:
            for index, row in enumerate(read_source_rows(source)):
                if not required.issubset(row):
                    raise SourceError(
                        f"필수 필드 누락: {source.target_id}, {index}"
                    )

                if "_source_row" in row:
                    raise SourceError("원본 대응 필드명 충돌")

                row = {**row, "_source_row": index}

                for field, value in row.items():
                    field_types.setdefault(field, set()).add(
                        type(value).__name__
                    )

                    if value is None or value == "":
                        null_counts[field] += 1

                stream.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n"
                )

                rows += 1

        schema = {
            field: {
                "types": sorted(types),
                "null_or_empty_count": null_counts[field],
            }
            for field, types in field_types.items()
        }

    inspection = as_row(source.result.get("inspection", {}))
    expected_rows = inspection.get("rows")

    # 포스트시즌 기존 수집 검증은 와일드카드 배열만 검사했다.
    if (
        source.target_id != "dent-postseason-2026-wildcard-probe"
        and expected_rows is not None
        and rows != expected_rows
    ):
        raise SourceError(
            f"기존 검사 행 수와 불일치: {source.target_id}"
        )

    return {
        "target_id": source.target_id,
        "source_sha": source.sha256,
        "revision": source.revision,
        "raw_path": f"objects/{source.sha256}.raw",
        "interim_file": output.name,
        "interim_sha256": sha256_file(output),
        "rows": rows,
        "schema": schema,
        "target": source.target,
        "collection_result": source.result,
    }


def read_interim_rows(path: Path) -> Iterator[Row]:
    """중간 자료를 읽어 processed 변환에 전달한다."""
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)

        for batch in parquet.iter_batches(batch_size=8192):
            yield from batch.to_pylist()

        return

    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                yield as_row(json.loads(line))

        return

    raise SourceError("지원하지 않는 중간 자료 형식")