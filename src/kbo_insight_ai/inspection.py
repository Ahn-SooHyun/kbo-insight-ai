"""검증된 HF 원본을 읽기 전용으로 검사하고 행 위치가 연결된 근거를 반환한다."""

from importlib.metadata import version
import json
from pathlib import Path
import re

from kbo_insight_ai.collection import (
    HF_FILENAME,
    HF_REVISION,
    HF_SAMPLE_BYTES,
    HF_SAMPLE_SHA256,
    sha256,
)
from kbo_insight_ai.settings import Settings


INSPECTION_VERSION = "ct2-01-inspection-1"
RUN_ID_PATTERN = re.compile(r"[0-9a-f]{32}")
SAMPLE_COLUMNS = (
    "game_pk", "inning", "inning_topbot", "at_bat_number", "pitch_number",
    "batter", "pitcher", "balls", "strikes", "outs_when_up",
    "on_1b", "on_2b", "on_3b", "events", "pitch_result", "type",
    "home_score", "away_score", "post_home_score", "post_away_score",
    "post_outs", "post_on_1b", "post_on_2b", "post_on_3b",
    "release_speed_kmh", "plate_x", "plate_z",
)


class InspectionError(ValueError):
    """원본·이력·필드 또는 검사 환경이 검사의 전제를 충족하지 않는다."""


def _read_record(path: Path) -> dict[str, object]:
    """작은 이력 JSON만 읽고 손상된 값을 원문 노출 없이 거절한다."""
    try:
        if path.stat().st_size > 65536:
            raise InspectionError("수집 이력이 허용 크기를 넘었습니다.")
        record = json.loads(path.read_bytes())
    except (OSError, ValueError, RecursionError):
        raise InspectionError("수집 이력을 읽을 수 없습니다.") from None
    if not isinstance(record, dict):
        raise InspectionError("수집 이력의 구조가 올바르지 않습니다.")
    return record


def inspect_sample(
    settings: Settings,
    run_id: str,
    game_id: str = "20230401HHWO02023",
) -> dict[str, object]:
    """고정 원본을 재검증한 뒤 필드·표본 사례를 반환한다. 쓰기·통신은 없다."""
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise InspectionError("run_id: 저장된 32자리 실행 ID가 필요합니다.")

    try:
        base = (settings.raw_dir / "collection").resolve()
        run_dir = (base / run_id).resolve()
        if not run_dir.is_relative_to(base):
            raise InspectionError("원본 경로가 수집 경계 밖에 있습니다.")
        request = _read_record(run_dir / "request.json")
        result = _read_record(run_dir / "result.json")
        target = request.get("request")
        if (
            request.get("run_id") != run_id
            or result.get("run_id") != run_id
            or request.get("source") != "huggingface_parquet"
            or request.get("source_version") != HF_REVISION
            or not isinstance(target, dict)
            or target.get("filename") != HF_FILENAME
            or target.get("expected_sha256") != HF_SAMPLE_SHA256
            or target.get("expected_bytes") != HF_SAMPLE_BYTES
            or result.get("status") != "artifact_verified"
            or result.get("body_complete") is not True
            or result.get("sha256") != HF_SAMPLE_SHA256
            or result.get("bytes") != HF_SAMPLE_BYTES
            or result.get("body_path") != "attempt-01.parquet"
        ):
            raise InspectionError("검증된 고정 HF 원본의 이력이 필요합니다.")

        path = (run_dir / "attempt-01.parquet").resolve()
        if not path.is_relative_to(run_dir):
            raise InspectionError("원본 파일이 실행 경계 밖에 있습니다.")
        if path.stat().st_size != HF_SAMPLE_BYTES:
            raise InspectionError("원본 크기가 수집 근거와 다릅니다.")
        raw_bytes = path.read_bytes()
        if len(raw_bytes) != HF_SAMPLE_BYTES:
            raise InspectionError("원본 크기가 읽는 동안 달라졌습니다.")
        digest = sha256(raw_bytes)
        if digest != HF_SAMPLE_SHA256:
            raise InspectionError("원본 해시가 수집 근거와 다릅니다.")
    except OSError:
        raise InspectionError("원본 경로나 파일을 읽을 수 없습니다.") from None

    try:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except ImportError:
        raise InspectionError(
            "inspection 선택 설치의 pyarrow가 필요합니다."
        ) from None

    try:
        parquet = pq.ParquetFile(pa.BufferReader(raw_bytes))
        table = parquet.read()
        required = {*SAMPLE_COLUMNS, "game_date"}
        if not required.issubset(table.column_names):
            raise InspectionError("표본 검사에 필요한 필드가 없습니다.")
        if "_source_row_index" in table.column_names:
            raise InspectionError("원본 행 위치 필드가 충돌합니다.")
        table = table.append_column(
            "_source_row_index",
            pa.array(range(table.num_rows), type=pa.int64()),
        )
        columns = ["_source_row_index", *SAMPLE_COLUMNS]
        game = table.filter(pc.equal(table["game_pk"], game_id)).select(columns)
        if not game.num_rows:
            raise InspectionError("선택 경기의 원본 행이 없습니다.")

        def examples(mask: object) -> list[dict[str, object]]:
            """원본 물리 순서에서 세 행만 추출한다."""
            return table.filter(mask).select(columns).slice(0, 3).to_pylist()

        cases = {
            "first_pa": game.filter(
                pc.equal(game["at_bat_number"], game["at_bat_number"][0].as_py())
            ).to_pylist(),
            "pitch_zero": examples(pc.equal(table["pitch_number"], 0)),
            "null_event": examples(pc.is_null(table["events"])),
            "bunt_foul_W": examples(pc.equal(table["pitch_result"], "W")),
            "bunt_swing_V": examples(pc.equal(table["pitch_result"], "V")),
            "sac_bunt": examples(pc.equal(table["events"], "sac_bunt")),
        }
        groups = table.group_by(["game_pk", "at_bat_number"]).aggregate(
            [("batter", "count_distinct"), ("pitcher", "count_distinct")]
        )
        groups = groups.sort_by(
            [("game_pk", "ascending"), ("at_bat_number", "ascending")]
        )
        participant_changes: dict[str, object] = {}
        for column in ("batter_count_distinct", "pitcher_count_distinct"):
            found = groups.filter(pc.greater(groups[column], 1))
            participant_changes[column] = {
                "pa_key_count": found.num_rows,
                "first_keys": found.slice(0, 3).to_pylist(),
            }
            if found.num_rows:
                key = found.slice(0, 1).to_pylist()[0]
                mask = pc.and_(
                    pc.equal(table["game_pk"], key["game_pk"]),
                    pc.equal(table["at_bat_number"], key["at_bat_number"]),
                )
                cases[column] = table.filter(mask).select(columns).to_pylist()

        rows = game.to_pylist()
        appearances: list[list[dict[str, object]]] = []
        for row in rows:
            if not appearances or appearances[-1][0]["at_bat_number"] != row["at_bat_number"]:
                appearances.append([])
            appearances[-1].append(row)
        transitions = []
        for previous, following in zip(appearances, appearances[1:]):
            before, after = previous[-1], following[0]
            if (before["inning"], before["inning_topbot"]) != (
                after["inning"], after["inning_topbot"]
            ):
                transitions.append(
                    {"previous_pa_last": before, "next_pa_first": after}
                )
        cases["selected_game_half_transitions"] = transitions[:2]
        cases["selected_game_last_pa"] = appearances[-1]

        def labeled_pa_count(mask: object) -> int:
            """관측 라벨이 있는 타석 키 수만 세며 정답 건수로 확정하지 않는다."""
            selected = table.filter(mask).select(["game_pk", "at_bat_number"])
            return len({
                (row["game_pk"], row["at_bat_number"])
                for row in selected.to_pylist()
            })

        observed_counts = {
            "zero_pitch_rows": table.filter(
                pc.equal(table["pitch_number"], 0)
            ).num_rows,
            "positive_pitch_rows": table.filter(
                pc.greater(table["pitch_number"], 0)
            ).num_rows,
            "W_rows": table.filter(pc.equal(table["pitch_result"], "W")).num_rows,
            "V_rows": table.filter(pc.equal(table["pitch_result"], "V")).num_rows,
            "sac_bunt_labeled_pa_keys": labeled_pa_count(
                pc.equal(table["events"], "sac_bunt")
            ),
            "null_events_pa_keys": labeled_pa_count(pc.is_null(table["events"])),
            "positive_pitch_null_plate_z_rows": table.filter(
                pc.and_(
                    pc.greater(table["pitch_number"], 0),
                    pc.is_null(table["plate_z"]),
                )
            ).num_rows,
        }
        date_bounds = pc.min_max(table["game_date"]).as_py()
        result_data = {
            "inspection_version": INSPECTION_VERSION,
            "inspection_code_sha256": sha256(Path(__file__).read_bytes()),
            "run_id": run_id,
            "source_version": HF_REVISION,
            "raw_path": str(path),
            "raw_sha256": digest,
            "raw_bytes": HF_SAMPLE_BYTES,
            "pyarrow_version": version("pyarrow"),
            "physical_file_rows": parquet.metadata.num_rows,
            "physical_file_columns": len(parquet.schema_arrow),
            "physical_file_row_groups": parquet.num_row_groups,
            "file_date_bounds": {key: str(value) for key, value in date_bounds.items()},
            "file_distinct_games": len(pc.unique(table["game_pk"])),
            "file_distinct_pa_keys": groups.num_rows,
            "selected_game": game_id,
            "selected_game_rows": game.num_rows,
            "schema": [
                {
                    "name": field.name,
                    "type": str(field.type),
                    "nullable": field.nullable,
                    "null_count": table[field.name].null_count,
                }
                for field in parquet.schema_arrow
            ],
            "observed_counts": observed_counts,
            "participant_changes": participant_changes,
            "case_rows": cases,
            "quality_status": "not_assessed",
            "event_role_confirmation": "requires_independent_source",
        }
        # JSON 출력에 유한하지 않은 값이 섞이지 않는지도 쓰기 없이 확인한다.
        json.dumps(result_data, allow_nan=False)
        return result_data
    except InspectionError:
        raise
    except (pa.ArrowException, OSError, ValueError, KeyError, TypeError, IndexError):
        raise InspectionError("원본 필드 또는 Parquet 구조를 검사할 수 없습니다.") from None
