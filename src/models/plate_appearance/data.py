"""M2 학습 자료의 계약, 시간 분할과 타석 품질 근거를 검증한다."""

from collections import Counter
from datetime import date
from pathlib import Path
import sqlite3

from . import model as m2
from ...datasets.model_data import fingerprint, inside
from ...datasets.published_sources import (
    RUN_PATTERN,
    as_row,
    read_json,
    sha256_file,
)
from ...settings import Settings
from ...validation.standard_data import PA_EVENTS


SPLITS = ("train", "validation")
BOUNDARIES = (
    "2025-01-01",
    "2026-01-01",
    "2026-11-26",
)

INTAKE_RUN = "259b38decaac4ed68dd5a596cfbe4849"
INTAKE_SHA256 = (
    "c8383a9d3e8930830a137d1b6791cbb26"
    "e2316dbbbdf4f0f436ceda9a43890d0"
)
STANDARD_RUN = "e020dafedc8b4fd9ac6696d0e4e4a92e"
STANDARD_VERIFICATION = "37b56c39caf5435e8fb2a17f993ec4df"

EXPECTED = {
    "train": {
        "eligible": 113272,
        "games": 1440,
        "excluded": 538,
        "excluded:participant_changed": 122,
        "excluded:unlabeled_half_end_candidate": 311,
        "eligible:no_actual_pitch": 188,
    },
    "validation": {
        "eligible": 55691,
        "games": 720,
        "excluded": 449,
        "excluded:participant_changed": 56,
        "excluded:unlabeled_half_end_candidate": 143,
        "eligible:no_actual_pitch": 122,
    },
}


def source(
    settings: Settings,
    run_id: str,
) -> tuple[Path, dict[str, object]]:
    """확정 인수판의 매니페스트·DB·허용 목록을 확인한다."""

    if not RUN_PATTERN.fullmatch(run_id) or run_id != INTAKE_RUN:
        raise ValueError("현재 M2 개발용 인수 실행 ID 불일치")

    if set(m2.CLASSES) != PA_EVENTS:
        raise ValueError("M2 클래스 사전과 표준 결과 코드 불일치")

    directory = settings.processed_dir / "model_data" / run_id

    manifest_path = inside(directory, "manifest.json")
    contract_path = inside(directory, "feature_target_contract.json")
    eda_path = inside(directory, "train_eda.json")
    supplement_path = inside(directory, "supplement-impact.json")
    database = inside(directory, "model_data.sqlite3")

    manifest = as_row(read_json(manifest_path))
    definitions = as_row(read_json(contract_path))
    contract = as_row(definitions["m2"])
    allowed = as_row(contract["inputs"])
    config = as_row(manifest["config"])
    standard = as_row(manifest["source"])

    limits = tuple(
        config[key]
        for key in (
            "validation_start",
            "test_start",
            "observation_end",
        )
    )

    contract_hash = fingerprint(definitions)
    eda_hash = fingerprint(read_json(eda_path))

    if (
        manifest["status"] != "completed"
        or manifest["run_id"] != run_id
        or config["timezone"] != "Asia/Seoul"
        or limits != BOUNDARIES
        or standard["run_id"] != STANDARD_RUN
        or standard["verification_id"] != STANDARD_VERIFICATION
        or manifest["contract_sha256"] != contract_hash
        or manifest["eda_sha256"] != eda_hash
        or contract["index"] != "m2_index"
        or contract["target_columns"] != [
            "sample_id", "target_event"
        ]
        or not set(m2.INPUT_COLUMNS) <= allowed.keys()
    ):
        raise ValueError("M2 선행 자료 계약 불일치")

    if manifest["database_sha256"] != INTAKE_SHA256:
        raise ValueError("인수 매니페스트 DB 해시 불일치")

    database_hash = sha256_file(database)

    if database_hash != INTAKE_SHA256:
        raise ValueError("실제 모델 DB SHA-256 불일치")

    # 보완 영향 자료는 학습 변수로 사용하지 않고 출처만 보존한다.
    as_row(read_json(supplement_path))

    provenance = {
        "model_data_run_id": run_id,
        "model_data_version": config["version"],
        "database": str(database),
        "database_sha256": database_hash,
        "contract_sha256": contract_hash,
        "eda_sha256": eda_hash,
        "manifest_sha256": sha256_file(manifest_path),
        "supplement_impact_sha256": sha256_file(supplement_path),
        "standard_run_id": STANDARD_RUN,
        "standard_verification_id": STANDARD_VERIFICATION,
        "timezone": config["timezone"],
        "validation_start": limits[0],
        "test_start": limits[1],
        "observation_end_exclusive": limits[2],
        "allowed_input_columns": sorted(allowed),
        "selected_input_columns": list(m2.INPUT_COLUMNS),
    }

    return database, provenance


def expected_split(day: str) -> str:
    """KST 날짜의 Train·Validation·Test 소속을 결정한다."""

    if not isinstance(day, str):
        raise ValueError("경기 날짜가 문자열이 아닙니다.")

    if date.fromisoformat(day).isoformat() != day:
        raise ValueError("경기 날짜 형식 오류")

    if day >= BOUNDARIES[2]:
        raise ValueError("관측 종료일 이후 표본 존재")

    if day < BOUNDARIES[0]:
        return "train"

    if day < BOUNDARIES[1]:
        return "validation"

    return "test"


def verify_keys(
    db: sqlite3.Connection,
) -> None:
    """정답 값이 아닌 인덱스와 키만으로 분할·연결 구조를 검사한다."""

    for game, day, split in db.execute(
        """
        SELECT game_id, sample_date, split
        FROM m2_index
        GROUP BY game_id, sample_date, split
        """
    ):
        m2.identifier(game, "game_id")

        if split != expected_split(day):
            raise ValueError("M2 인덱스 날짜와 분할 불일치")

    mixed = db.execute(
        """
        SELECT game_id
        FROM m2_index
        GROUP BY game_id
        HAVING COUNT(DISTINCT split) <> 1
            OR COUNT(DISTINCT sample_date) <> 1
        LIMIT 1
        """
    ).fetchone()

    if mixed is not None:
        raise ValueError("동일 경기의 날짜 또는 분할이 섞였습니다.")

    for table in ("m2_index", "m2_inputs", "m2_targets"):
        duplicated = db.execute(
            f"""
            SELECT sample_id
            FROM {table}
            GROUP BY sample_id
            HAVING sample_id IS NULL OR COUNT(*) <> 1
            LIMIT 1
            """
        ).fetchone()

        if duplicated is not None:
            raise ValueError(f"{table}: NULL 또는 중복 표본 키")

    for table in ("m2_inputs", "m2_targets"):
        orphan = db.execute(
            f"""
            SELECT t.sample_id
            FROM {table} t
            LEFT JOIN m2_index x USING(sample_id)
            WHERE x.sample_id IS NULL
                OR x.status IS NOT 'eligible'
            LIMIT 1
            """
        ).fetchone()

        if orphan is not None:
            raise ValueError(f"{table}: 고아 또는 제외 표본 연결")

    actual = {
        row[1]
        for row in db.execute("PRAGMA table_info(m2_inputs)")
    }

    if not {"sample_id", *m2.INPUT_COLUMNS} <= actual:
        raise ValueError("M2 입력 테이블의 필수 컬럼 누락")

    # 계약에서 허용한 사전 입력 외의 직접적인 사후 열을 차단한다.
    if any(
        column.startswith(("target_", "post_", "end_", "credited_"))
        or column in {
            "event", "winner", "participant_changed",
            "no_actual_pitch", "physical_pitch_rows",
            "source_sha", "exclusion_reason", "label_kind",
            "pitch_type", "pitch_result", "release_speed_kmh",
        }
        for column in actual
    ):
        raise ValueError("M2 입력 테이블에 금지된 사후 정보 존재")


def exclusion_reason(row: sqlite3.Row) -> str | None:
    """기존 모델 자료 생성 코드의 판정 우선순위를 재현한다."""

    if row["label_kind"] is None:
        return "missing_audit"

    if row["participant_changed"] != 0:
        return "participant_changed"

    if row["start_complete"] != 1:
        return "incomplete_start"

    if row["order_complete"] != 1:
        return "incomplete_order"

    if (
        row["state_valid"] != 1
        or row["state_consistent"] != 1
        or row["runs_consistent"] != 1
    ):
        return "invalid_state"

    if row["label_kind"] != "observed_result":
        return str(row["label_kind"])

    if row["result_valid"] != 1:
        return "invalid_result"

    return None


def verify_participants(
    row: sqlite3.Row,
    inputs: dict[str, object],
) -> None:
    """시작 참가자와 종료·귀속 후보를 구분해 확인한다."""

    for role in ("batter", "pitcher"):
        start = row[f"start_{role}_id"]
        end = row[f"end_{role}_id"]
        credited = row[f"credited_{role}_candidate"]

        if inputs[f"{role}_id"] != start:
            raise ValueError("M2 시작 참가자와 입력 ID 불일치")

        if start != end:
            raise ValueError("유효 타석에서 참가자 변경 발견")

        if credited is not None and credited != start:
            raise ValueError("M2 결과 귀속 후보 불일치")


def read_samples(
    db: sqlite3.Connection,
) -> tuple[
    dict[str, list[m2.Sample]],
    dict[str, Counter[str]],
]:
    """개발 표본만 정답에 연결하고 감사·제외 열은 모델에 전달하지 않는다."""

    db.row_factory = sqlite3.Row
    verify_keys(db)

    samples: dict[str, list[m2.Sample]] = {
        split: [] for split in SPLITS
    }
    ledger: dict[str, Counter[str]] = {
        split: Counter() for split in SPLITS
    }
    invalid_targets: Counter[str] = Counter()

    projection = ", ".join(
        f'i."{column}" AS "{column}"'
        for column in m2.INPUT_COLUMNS
    )

    # Test는 키와 분할 구조만 검사하며 정답 내용은 조회하지 않는다.
    rows = db.execute(
        f"""
        WITH development AS (
            SELECT *
            FROM m2_index
            WHERE split IN ('train', 'validation')
                AND sample_date < ?
        )
        SELECT
            x.*,
            i.sample_id AS input_id,
            t.sample_id AS target_id,
            t.target_event,
            {projection}
        FROM development x
        LEFT JOIN m2_inputs i
            ON x.sample_id = i.sample_id
        LEFT JOIN m2_targets t
            ON x.sample_id = t.sample_id
        ORDER BY x.sample_date, x.game_id, x.pa_number
        """,
        (BOUNDARIES[1],),
    )

    seen: set[str] = set()

    for row in rows:
        split = row["split"]
        sample_id = m2.identifier(row["sample_id"], "sample_id")
        game_id = m2.identifier(row["game_id"], "game_id")
        number = m2.count(
            row["pa_number"], "pa_number", minimum=1
        )

        if sample_id != f"{game_id}:{number}" or sample_id in seen:
            raise ValueError("M2 타석 키 불일치 또는 중복")

        seen.add(sample_id)

        if expected_split(row["sample_date"]) != split:
            raise ValueError("M2 개발 날짜 분할 오류")

        audit = ledger[split]
        audit["index_rows"] += 1

        physical = m2.count(
            row["physical_pitch_rows"],
            "physical_pitch_rows",
        )

        no_pitch = int(physical == 0)

        if row["no_actual_pitch"] != no_pitch:
            raise ValueError("무투구 감사 플래그 불일치")

        status = row["status"]
        reason = exclusion_reason(row)

        if status == "excluded":
            if (
                not row["exclusion_reason"]
                or row["exclusion_reason"] != (reason or "unresolved")
                or row["input_id"] is not None
                or row["target_id"] is not None
            ):
                raise ValueError("제외 타석의 입력·정답 또는 사유 오류")

            audit["excluded"] += 1
            audit[f"excluded:{row['exclusion_reason']}"] += 1

            if no_pitch:
                audit["excluded:no_actual_pitch"] += 1

            continue

        if status != "eligible":
            raise ValueError("지원하지 않는 M2 인덱스 상태")

        audit["eligible"] += 1

        if (
            reason is not None
            or row["exclusion_reason"] is not None
            or row["input_id"] != sample_id
            or row["target_id"] != sample_id
        ):
            raise ValueError("유효 타석의 품질 또는 연결 오류")

        inputs = {
            column: row[column]
            for column in m2.INPUT_COLUMNS
        }

        m2.validate_inputs(inputs)
        verify_participants(row, inputs)

        target = row["target_event"]

        if target != row["event"]:
            raise ValueError("인덱스 사건과 확정 정답 불일치")

        if target is None:
            invalid_targets[f"{split}:missing"] += 1
            continue

        if target not in m2.CLASS_INDEX:
            invalid_targets[f"{split}:unknown:{target}"] += 1
            continue

        if no_pitch:
            if target != "walk":
                raise ValueError("인수판의 유효 무투구 결과 불일치")

            audit["eligible:no_actual_pitch"] += 1
            audit["eligible:no_actual_pitch:walk"] += 1

        for role in ("batter", "pitcher"):
            if row[f"credited_{role}_candidate"] is None:
                audit[f"eligible:missing_credited_{role}"] += 1

        samples[split].append(
            m2.Sample(
                sample_id=sample_id,
                game_id=game_id,
                day=row["sample_date"],
                inputs=inputs,
                target=m2.target_class(target),
            )
        )

    if invalid_targets:
        raise ValueError(
            "결측·미등록 정답으로 평가 중단: "
            f"{dict(invalid_targets)}"
        )

    for split in SPLITS:
        if not samples[split]:
            raise ValueError(f"{split}: 유효 타석이 없습니다.")

        if len(samples[split]) != ledger[split]["eligible"]:
            raise ValueError("유효 표본 수와 감사 집계 불일치")

    return samples, ledger


def split_summary(
    samples: dict[str, list[m2.Sample]],
    ledger: dict[str, Counter[str]],
) -> dict[str, object]:
    """실제 사용 분모와 범주·결측·제외 건수를 기록한다."""

    result: dict[str, object] = {}

    for split in SPLITS:
        rows = samples[split]
        frequencies = Counter(row.target for row in rows)

        result[split] = {
            "samples": len(rows),
            "games": len({row.game_id for row in rows}),
            "first_date": min(row.day for row in rows),
            "last_date": max(row.day for row in rows),
            "index_and_exclusions": dict(ledger[split]),
            "target_frequency": {
                name: frequencies[name]
                for name in m2.CLASSES
            },
            "mapped_target_frequency": {
                name: frequencies[name]
                for name in m2.CLASSES
            },
            "missing_target_count": 0,
            "unknown_target_count": 0,
            "missing_inputs": {
                column: sum(
                    row.inputs[column] is None
                    for row in rows
                )
                for column in m2.INPUT_COLUMNS
            },
        }

    return result


def verify_intake(
    samples: dict[str, list[m2.Sample]],
    ledger: dict[str, Counter[str]],
) -> None:
    """고정 인수판의 수량과 대표 예외 건수를 직접 집계해 대조한다."""

    for split in SPLITS:
        actual = dict(ledger[split])
        actual["games"] = len(
            {sample.game_id for sample in samples[split]}
        )

        for key, expected in EXPECTED[split].items():
            if actual.get(key, 0) != expected:
                raise ValueError(
                    f"{split} {key}: "
                    f"인수 {expected}, 실제 {actual.get(key, 0)}"
                )

    train_frequency = Counter(
        sample.target for sample in samples["train"]
    )

    for event, expected in (
        ("catcher_interference", 3),
        ("triple_play", 1),
    ):
        if train_frequency[event] != expected:
            raise ValueError(
                f"Train 희소 결과 인수 불일치: {event}"
            )