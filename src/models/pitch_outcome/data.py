"""M5 자료 인수, 실제 투구 경계 및 표준 자료를 검증한다."""

from collections import Counter
from collections.abc import Mapping
from contextlib import closing
from dataclasses import dataclass
from datetime import date
import logging
from pathlib import Path
import sqlite3

from . import model as m5
from ...datasets.model_data import fingerprint, inside
from ...datasets.published_sources import (
    as_row,
    read_json,
    sha256_file,
)
from ...settings import Settings
from ...validation.standard_data import PITCH_RESULT_TYPES


LOGGER = logging.getLogger(__name__)

INTAKE_RUN = "259b38decaac4ed68dd5a596cfbe4849"
INTAKE_SHA256 = (
    "c8383a9d3e8930830a137d1b6791cbb26"
    "e2316dbbbdf4f0f436ceda9a43890d0"
)

STANDARD_RUN = "e020dafedc8b4fd9ac6696d0e4e4a92e"
STANDARD_VERIFICATION = "37b56c39caf5435e8fb2a17f993ec4df"
DATA_VERSION = "1.2.0"

BOUNDARIES = (
    "2025-01-01",
    "2026-01-01",
    "2026-11-26",
)

SPLITS = ("train", "validation")

EXPECTED = {
    "train": (442636, 1440),
    "validation": (216869, 720),
}

INDEX_COLUMNS = (
    "sample_id",
    "game_id",
    "sample_date",
    "split",
    "status",
    "pa_number",
    "pitch_number",
    "source_sha",
    "source_row",
)

TARGET_COLUMNS = (
    "sample_id",
    "target_pitch_type",
    "target_pitch_result",
    "target_joint",
)

# 학습 특징 중 실제 공 직전 상태와 참가자를 표준 DB에 대조한다.
SOURCE_INPUTS = {
    "season": "g.season",
    "competition": "g.competition",
    "inning": "a.inning",
    "half": "a.half",
    "batter_id": "p.batter_id",
    "pitcher_id": "p.pitcher_id",
    "pitch_number": "p.pitch_number",
    "balls": "p.balls",
    "strikes": "p.strikes",
    "outs": "p.outs_when_up",
    "home_score": "p.home_score",
    "away_score": "p.away_score",
    "on_first": "(p.on_1b IS NOT NULL)",
    "on_second": "(p.on_2b IS NOT NULL)",
    "on_third": "(p.on_3b IS NOT NULL)",
}


@dataclass(slots=True)
class Dataset:
    """개발용 표본과 별도 감사·재현 정보를 반환한다."""

    rows: dict[str, list[m5.Sample]]
    provenance: dict[str, object]
    examples: dict[str, dict[str, object]]


def expected_split(day: str) -> str:
    """KST 날짜 경계에 따라 기존 분할을 검사한다."""
    m5.require(
        isinstance(day, str)
        and date.fromisoformat(day).isoformat() == day,
        "경기 날짜 형식 오류",
    )

    m5.require(day < BOUNDARIES[2], "관측 종료일 이후 자료")

    if day < BOUNDARIES[0]:
        return "train"

    if day < BOUNDARIES[1]:
        return "validation"

    return "test"


def source(
    settings: Settings,
    run_id: str,
) -> tuple[Path, Path, dict[str, object]]:
    """확정 인수판의 매니페스트, 자료 계약 및 지문을 검사한다."""
    m5.require(run_id == INTAKE_RUN, "M5 인수 실행 ID 오류")

    directory = settings.processed_dir / "model_data" / run_id

    database = inside(directory, "model_data.sqlite3")
    manifest_path = inside(directory, "manifest.json")
    contract_path = inside(directory, "feature_target_contract.json")
    eda_path = inside(directory, "train_eda.json")
    supplement_path = inside(directory, "supplement-impact.json")

    manifest = as_row(read_json(manifest_path))
    definitions = as_row(read_json(contract_path))
    contract = as_row(definitions["m5"])
    config = as_row(manifest["config"])
    standard_metadata = as_row(manifest["source"])
    allowed = as_row(contract["inputs"])

    limits = tuple(
        config[name]
        for name in (
            "validation_start",
            "test_start",
            "observation_end",
        )
    )

    m5.require(
        manifest["status"] == "completed"
        and manifest["run_id"] == run_id
        and config["version"] == DATA_VERSION
        and config["timezone"] == "Asia/Seoul"
        and config["seed"] == settings.seed
        and limits == BOUNDARIES
        and standard_metadata["run_id"] == STANDARD_RUN
        and standard_metadata["verification_id"]
        == STANDARD_VERIFICATION,
        "M5 인수 매니페스트 계약 불일치",
    )

    m5.require(
        contract["index"] == "m5_index"
        and contract["target_columns"] == list(TARGET_COLUMNS)
        and set(allowed) == set(m5.ALL_COLUMNS)
        and set(m5.INPUT_COLUMNS) <= set(allowed),
        "M5 입력·정답 허용 목록 불일치",
    )

    contract_hash = fingerprint(definitions)
    eda_hash = fingerprint(read_json(eda_path))

    m5.require(
        contract_hash == manifest["contract_sha256"]
        and eda_hash == manifest["eda_sha256"],
        "자료 계약 또는 Train EDA 지문 불일치",
    )

    m5.require(
        manifest["database_sha256"] == INTAKE_SHA256
        and sha256_file(database) == INTAKE_SHA256,
        "인수 모델 DB SHA-256 불일치",
    )

    standard = inside(
        settings.processed_dir / "standard" / STANDARD_RUN,
        f"verified-{STANDARD_VERIFICATION}.sqlite3",
    )

    standard_hash = sha256_file(standard)

    m5.require(
        standard_hash == standard_metadata["database_sha256"],
        "표준 검증 DB SHA-256 불일치",
    )

    as_row(read_json(supplement_path))

    provenance = {
        "model_data_run_id": run_id,
        "model_data_version": DATA_VERSION,
        "database": str(database),
        "database_sha256": INTAKE_SHA256,
        "standard_database": str(standard),
        "standard_database_sha256": standard_hash,
        "standard_run_id": STANDARD_RUN,
        "standard_verification_id": STANDARD_VERIFICATION,
        "contract_sha256": contract_hash,
        "eda_sha256": eda_hash,
        "manifest_sha256": sha256_file(manifest_path),
        "supplement_impact_sha256": sha256_file(supplement_path),
        "timezone": "Asia/Seoul",
        "boundaries": list(BOUNDARIES),
        "allowed_input_columns": sorted(allowed),
        "selected_input_columns": list(m5.INPUT_COLUMNS),
        "competition": "regular",
    }

    return database, standard, provenance


def verify_keys(db: sqlite3.Connection) -> None:
    """전체 인덱스와 키만으로 일대일 연결 및 분할을 검증한다."""
    schemas = {
        "m5_index": set(INDEX_COLUMNS),
        "m5_inputs": {"sample_id", *m5.ALL_COLUMNS},
        "m5_targets": set(TARGET_COLUMNS),
    }

    for table, expected in schemas.items():
        actual = {
            row[1]
            for row in db.execute(f"PRAGMA table_info({table})")
        }

        m5.require(actual == expected, f"{table}: 스키마 오류")

        duplicate = db.execute(
            f"""
            SELECT sample_id
            FROM {table}
            GROUP BY sample_id
            HAVING sample_id IS NULL OR COUNT(*) <> 1
            LIMIT 1
            """
        ).fetchone()

        m5.require(
            duplicate is None,
            f"{table}: 중복 또는 NULL 표본 키",
        )

    for table in ("m5_inputs", "m5_targets"):
        orphan = db.execute(
            f"""
            SELECT t.sample_id
            FROM {table} t
            LEFT JOIN m5_index x USING(sample_id)
            WHERE x.sample_id IS NULL
            LIMIT 1
            """
        ).fetchone()

        missing = db.execute(
            f"""
            SELECT x.sample_id
            FROM m5_index x
            LEFT JOIN {table} t USING(sample_id)
            WHERE t.sample_id IS NULL
            LIMIT 1
            """
        ).fetchone()

        m5.require(
            orphan is None and missing is None,
            f"{table}: 고아 또는 누락 표본",
        )

    invalid = db.execute(
        """
        SELECT sample_id
        FROM m5_index
        WHERE game_id IS NULL
          OR sample_date IS NULL
          OR split IS NULL
          OR status IS NOT 'eligible'
          OR typeof(pa_number) <> 'integer'
          OR typeof(pitch_number) <> 'integer'
          OR pa_number < 1
          OR pitch_number < 1
          OR source_sha IS NULL
          OR typeof(source_row) <> 'integer'
          OR source_row < 0
          OR sample_id IS NOT
              game_id || ':' || pa_number || ':' || pitch_number
        LIMIT 1
        """
    ).fetchone()

    m5.require(invalid is None, "M5 실제 투구 인덱스 오류")

    duplicate_pitch = db.execute(
        """
        SELECT game_id, pa_number, pitch_number
        FROM m5_index
        GROUP BY game_id, pa_number, pitch_number
        HAVING COUNT(*) <> 1
        LIMIT 1
        """
    ).fetchone()

    m5.require(
        duplicate_pitch is None,
        "동일 경기·타석·실제 투구 번호 중복",
    )

    mixed = db.execute(
        """
        SELECT game_id
        FROM m5_index
        GROUP BY game_id
        HAVING COUNT(DISTINCT sample_date) <> 1
            OR COUNT(DISTINCT split) <> 1
        LIMIT 1
        """
    ).fetchone()

    m5.require(mixed is None, "동일 경기의 분할 또는 날짜 혼합")

    for game_id, day, split in db.execute(
        """
        SELECT game_id, sample_date, split
        FROM m5_index
        GROUP BY game_id, sample_date, split
        """
    ):
        m5.identifier(game_id, "game_id")
        m5.require(
            split == expected_split(day),
            "경기 날짜·시간 분할 불일치",
        )

    # 이 함수는 최종 Test의 정답 열 값을 조회하지 않는다.


def audit_previous(
    row: Mapping[str, object],
) -> tuple[object, object, str, bool]:
    """원천 직전 순번의 코드와 결측·유효 표본 간격을 구분한다."""
    number = m5.integer(row["audit_pitch_number"], "pitch_number", 1)

    if number == 1:
        m5.require(
            row["previous_number"] is None,
            "첫 공에 이전 실제 투구 행이 존재합니다.",
        )
        return None, None, "first_pitch", False

    if row["previous_number"] is None:
        return None, None, "missing_previous_number", True

    m5.require(
        row["previous_number"] == number - 1,
        "이전 실제 투구 순번 불일치",
    )

    if row["previous_order_valid"] != 1:
        return None, None, "previous_order_invalid", True

    if (
        row["previous_sha"] != row["source_sha"]
        or row["previous_source_row"] is None
        or m5.integer(
            row["previous_source_row"], "previous_source_row"
        ) >= m5.integer(row["source_row"], "source_row")
    ):
        return None, None, "previous_source_order_invalid", True

    if row["previous_target_valid"] != 1:
        return None, None, "previous_target_invalid", True

    pitch_type = row["previous_actual_type"]
    result = row["previous_actual_result"]

    m5.target_index(
        pitch_type, result, f"{pitch_type}:{result}"
    )

    # 생성 계약은 이전 공의 유효 코드와 원본 순서를 요구한다.
    # 직전 공 자체가 상태 품질 때문에 모델에서 제외됐을 수도 있다.
    if row["previous_state_valid"] != 1:
        return pitch_type, result, "previous_model_excluded", True

    return pitch_type, result, "previous_available", False


def audit_current(row: Mapping[str, object]) -> None:
    """모델 인덱스·입력·정답과 동일 원천 실제 공을 대조한다."""
    m5.require(
        row["game_result_valid"] == 1
        and row["game_status"] == "Final"
        and row["actual_date"] == row["sample_date"]
        and row["state_game_id"] == row["game_id"]
        and row["actual_pitch_number"] == row["audit_pitch_number"]
        and row["actual_source_sha"] == row["source_sha"]
        and row["actual_source_row"] == row["source_row"]
        and row["actual_target_valid"] == 1
        and row["actual_order_valid"] == 1
        and row["actual_state_valid"] == 1,
        "실제 투구의 표준 품질 또는 원천 연결 오류",
    )

    for name in SOURCE_INPUTS:
        m5.require(
            row[name] == row[f"actual_{name}"],
            f"공 직전 원천 입력 불일치: {name}",
        )

    m5.require(
        row["target_pitch_type"] == row["actual_pitch_type"]
        and row["target_pitch_result"] == row["actual_pitch_result"]
        and PITCH_RESULT_TYPES.get(row["actual_pitch_result"])
        == row["actual_result_type"],
        "표준 구종·결과 코드와 모델 정답 불일치",
    )


def source_exclusions(
    db: sqlite3.Connection,
) -> dict[str, dict[str, object]]:
    """표준 투구와 무투구를 분리하고 우선순위별 제외 건수를 기록한다."""
    result: dict[str, dict[str, object]] = {}

    for split, lower, upper in (
        ("train", "0001-01-01", BOUNDARIES[0]),
        ("validation", BOUNDARIES[0], BOUNDARIES[1]),
    ):
        reasons = dict(db.execute(
            """
            SELECT
                CASE
                    WHEN g.result_valid <> 1
                         OR g.status <> 'Final' THEN 'game_excluded'
                    WHEN a.game_id IS NULL THEN 'pa_state_excluded'
                    WHEN p.pitch_number < 1 THEN 'invalid_pitch_number'
                    WHEN p.order_valid <> 1 THEN 'invalid_order'
                    WHEN p.state_valid <> 1 THEN 'invalid_state'
                    WHEN p.target_valid <> 1 THEN 'invalid_target'
                    ELSE 'eligible'
                END reason,
                COUNT(*)
            FROM verified.pitches p
            JOIN verified.games g USING(game_id)
            LEFT JOIN verified.eligible_states a
                USING(game_id, pa_number)
            WHERE g.competition = 'regular'
              AND g.date >= ? AND g.date < ?
            GROUP BY reason
            """,
            (lower, upper),
        ))

        non_pitch = db.execute(
            """
            SELECT COUNT(*)
            FROM verified.non_pitch p
            JOIN verified.games g USING(game_id)
            WHERE g.competition = 'regular'
              AND g.date >= ? AND g.date < ?
            """,
            (lower, upper),
        ).fetchone()[0]

        checked_games, positive_rows, non_pitch_rows = db.execute(
            """
            SELECT
                COUNT(*),
                COALESCE(SUM(c.positive_pitch_rows), 0),
                COALESCE(SUM(c.non_pitch_rows), 0)
            FROM verified.game_checks c
            JOIN verified.games g USING(game_id)
            WHERE g.competition = 'regular'
              AND g.date >= ? AND g.date < ?
            """,
            (lower, upper),
        ).fetchone()

        checked_stored = db.execute(
            """
            SELECT COUNT(*)
            FROM verified.pitches p
            JOIN verified.game_checks c USING(game_id)
            JOIN verified.games g USING(game_id)
            WHERE g.competition = 'regular'
              AND g.date >= ? AND g.date < ?
            """,
            (lower, upper),
        ).fetchone()[0]

        checked_non_pitch = db.execute(
            """
            SELECT COUNT(*)
            FROM verified.non_pitch p
            JOIN verified.game_checks c USING(game_id)
            JOIN verified.games g USING(game_id)
            WHERE g.competition = 'regular'
              AND g.date >= ? AND g.date < ?
            """,
            (lower, upper),
        ).fetchone()[0]

        result[split] = {
            "standard_pitch_rows": sum(reasons.values()),
            "eligible": reasons.get("eligible", 0),
            "excluded": (
                sum(reasons.values()) - reasons.get("eligible", 0)
            ),
            "exclusive_reasons": reasons,
            "non_pitch_rows": non_pitch,
            "raw_checked_games": checked_games,
            "raw_positive_pitch_rows": positive_rows,
            "raw_non_pitch_rows": non_pitch_rows,
            "checked_standard_pitch_rows": checked_stored,
            "checked_standard_non_pitch_rows": checked_non_pitch,
            "raw_minus_standard_pitch_rows": (
                positive_rows - checked_stored
            ),
            "raw_minus_standard_non_pitch_rows": (
                non_pitch_rows - checked_non_pitch
            ),
            "reason_policy": (
                "game > pa_state > pitch_number > "
                "order > state > target"
            ),
        }

    return result


def row_query() -> str:
    """감사 열과 허용 입력을 명시적으로 선택한다."""
    selected_inputs = ",\n".join(
        f'i."{name}" AS "{name}"'
        for name in m5.INPUT_COLUMNS
    )

    source_inputs = ",\n".join(
        f'{expression} AS "actual_{name}"'
        for name, expression in SOURCE_INPUTS.items()
    )

    return f"""
        SELECT
            x.sample_id,
            x.game_id,
            x.sample_date,
            x.pa_number,
            x.pitch_number AS audit_pitch_number,
            x.source_sha,
            x.source_row,
            {selected_inputs},
            {source_inputs},
            t.target_pitch_type,
            t.target_pitch_result,
            t.target_joint,
            g.result_valid AS game_result_valid,
            g.status AS game_status,
            g.date AS actual_date,
            a.game_id AS state_game_id,
            p.source_sha AS actual_source_sha,
            p.source_row AS actual_source_row,
            p.target_valid AS actual_target_valid,
            p.order_valid AS actual_order_valid,
            p.state_valid AS actual_state_valid,
            p.pitch_type AS actual_pitch_type,
            p.pitch_result AS actual_pitch_result,
            p.pitch_result_type AS actual_result_type,
            prev.pitch_number AS previous_number,
            prev.source_sha AS previous_sha,
            prev.source_row AS previous_source_row,
            prev.order_valid AS previous_order_valid,
            prev.state_valid AS previous_state_valid,
            prev.target_valid AS previous_target_valid,
            prev.pitch_type AS previous_actual_type,
            prev.pitch_result AS previous_actual_result
        FROM m5_index x
        JOIN m5_inputs i USING(sample_id)
        JOIN m5_targets t USING(sample_id)
        JOIN verified.games g ON g.game_id = x.game_id
        LEFT JOIN verified.eligible_states a
            ON a.game_id = x.game_id
            AND a.pa_number = x.pa_number
        LEFT JOIN verified.pitches p
            ON p.game_id = x.game_id
            AND p.pa_number = x.pa_number
            AND p.pitch_number = x.pitch_number
        LEFT JOIN verified.pitches prev
            ON prev.game_id = x.game_id
            AND prev.pa_number = x.pa_number
            AND prev.pitch_number = x.pitch_number - 1
        WHERE x.split = ?
          AND g.competition = 'regular'
        ORDER BY x.game_id, x.pa_number, x.pitch_number
    """


def read_split(
    db: sqlite3.Connection,
    split: str,
) -> tuple[
    list[m5.Sample],
    dict[str, object],
    dict[str, dict[str, object]],
]:
    """개발 분할의 실제 투구만 감사 후 희소 학습 자료로 변환한다."""
    m5.require(split in SPLITS, "최종 Test 정답 접근 금지")

    rows: list[m5.Sample] = []
    games: set[str] = set()
    days: set[str] = set()
    reasons: Counter[str] = Counter()
    examples: dict[str, dict[str, object]] = {}
    joint_counts: Counter[str] = Counter()

    first_pitch_count = 0
    previous_missing_count = 0
    eligible_sequence_gaps = 0

    last_pa: tuple[str, int] | None = None
    last_number = 0
    last_source_sha: str | None = None
    last_source_row = -1

    for raw in db.execute(row_query(), (split,)):
        row = dict(raw)

        audit_current(row)

        game_id = m5.identifier(row["game_id"], "game_id")
        day = row["sample_date"]
        pa = m5.integer(row["pa_number"], "pa_number", 1)
        number = m5.integer(
            row["audit_pitch_number"], "pitch_number", 1
        )
        sha = m5.identifier(row["source_sha"], "source_sha")
        source_row = m5.integer(row["source_row"], "source_row")

        m5.require(
            expected_split(day) == split
            and int(day[:4]) == row["season"],
            "개발 자료 날짜 또는 시즌 불일치",
        )

        pa_key = (game_id, pa)

        if pa_key == last_pa:
            m5.require(
                number > last_number
                and sha == last_source_sha
                and source_row > last_source_row,
                "동일 타석의 실제 원천 투구 순서 오류",
            )

        last_pa = pa_key
        last_number = number
        last_source_sha = sha
        last_source_row = source_row

        previous_type, previous_result, reason, gap = (
            audit_previous(row)
        )

        m5.require(
            row["previous_pitch_type"] == previous_type
            and row["previous_pitch_result"] == previous_result,
            "모델 자료와 원천 직전 실제 투구 불일치",
        )

        inputs = {
            name: row[name]
            for name in m5.INPUT_COLUMNS
        }

        keys = m5.encode(inputs)

        target = m5.target_index(
            row["target_pitch_type"],
            row["target_pitch_result"],
            row["target_joint"],
        )

        rows.append(m5.Sample(keys, target))
        games.add(game_id)
        days.add(day)
        reasons[reason] += 1
        joint_counts[m5.CLASSES[target]] += 1

        first_pitch_count += int(number == 1)
        previous_missing_count += int(previous_type is None)
        eligible_sequence_gaps += int(gap)

        example_key = f"{split}:{reason}"

        if example_key not in examples:
            examples[example_key] = {
                "sample_id": row["sample_id"],
                "split": split,
                "reason": reason,
                "input": inputs,
            }

    m5.require(bool(rows), "개발 분할 표본 누락")

    m5.require(
        (len(rows), len(games)) == EXPECTED[split],
        f"{split}: 인수 표본 또는 경기 수 불일치",
    )

    observed_types = {
        key.split(":")[0] for key in joint_counts
    }
    observed_results = {
        key.split(":")[1] for key in joint_counts
    }

    summary: dict[str, object] = {
        "samples": len(rows),
        "games": len(games),
        "first_date": min(days),
        "last_date": max(days),
        "first_pitch_count": first_pitch_count,
        "previous_missing_count": previous_missing_count,
        "middle_previous_missing_count": (
            previous_missing_count - first_pitch_count
        ),
        "eligible_sequence_gap_count": eligible_sequence_gaps,
        "previous_reasons": dict(sorted(reasons.items())),
        "observed_pitch_types": sorted(observed_types),
        "observed_pitch_results": sorted(observed_results),
        "observed_joint_count": len(joint_counts),
        "joint_counts": dict(sorted(joint_counts.items())),
    }

    if split == "train":
        m5.require(
            first_pitch_count == 113508
            and previous_missing_count == 113516
            and len(observed_types) == 8
            and len(observed_results) == 7
            and len(joint_counts) == 53,
            "Train 인수 클래스 또는 직전 투구 건수 불일치",
        )

    LOGGER.info(
        "M5 %s: pitches=%d games=%d dates=%s~%s",
        split,
        len(rows),
        len(games),
        min(days),
        max(days),
    )

    return rows, summary, examples


def load(settings: Settings, run_id: str) -> Dataset:
    """인수 자료를 변경하지 않고 Train·Validation만 읽는다."""
    database, standard, provenance = source(settings, run_id)

    rows: dict[str, list[m5.Sample]] = {}
    counts: dict[str, dict[str, object]] = {}
    examples: dict[str, dict[str, object]] = {}

    with closing(sqlite3.connect(
        database.resolve().as_uri() + "?mode=ro",
        uri=True,
    )) as db:
        db.row_factory = sqlite3.Row

        db.execute(
            "ATTACH DATABASE ? AS verified",
            (standard.resolve().as_uri() + "?mode=ro",),
        )
        db.execute("PRAGMA query_only=ON")

        verify_keys(db)

        unmatched_game = db.execute(
            """
            SELECT x.game_id
            FROM m5_index x
            LEFT JOIN verified.eligible_games g
                ON g.game_id = x.game_id
            WHERE x.split IN ('train', 'validation')
              AND (
                  g.game_id IS NULL
                  OR g.date IS NOT x.sample_date
              )
            LIMIT 1
            """
        ).fetchone()

        m5.require(
            unmatched_game is None,
            "개발 인덱스의 표준 경기 연결 오류",
        )

        exclusions = source_exclusions(db)

        for split in SPLITS:
            rows[split], counts[split], sample_examples = read_split(
                db, split
            )

            m5.require(
                exclusions[split]["eligible"]
                == len(rows[split]),
                f"{split}: 표준 유효 투구와 모델 자료 수 불일치",
            )

            examples.update(sample_examples)

        postseason = {
            split: count
            for split, count in db.execute(
                """
                SELECT x.split, COUNT(*)
                FROM m5_index x
                JOIN verified.games g ON g.game_id = x.game_id
                WHERE x.split IN ('train', 'validation')
                  AND g.competition <> 'regular'
                GROUP BY x.split
                """
            )
        }

    provenance.update({
        "counts": counts,
        "source_exclusions": exclusions,
        "non_regular_excluded": {
            split: postseason.get(split, 0)
            for split in SPLITS
        },
        "test_target_values_read": False,
        "test_index_keys_checked": True,
        "source_access": "read_only",
    })

    return Dataset(rows, provenance, examples)