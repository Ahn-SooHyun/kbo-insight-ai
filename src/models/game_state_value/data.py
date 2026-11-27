"""경기 상황 자료의 인수, 시간 분할 및 원본 경계를 검증한다."""

from collections import Counter
from contextlib import closing
from dataclasses import dataclass
from datetime import date
from itertools import groupby
from pathlib import Path
import sqlite3

from . import model as m4
from ...datasets.model_data import fingerprint, inside
from ...datasets.published_sources import (
    as_row,
    read_json,
    sha256_file,
)
from ...settings import Settings


INTAKE_RUN = "259b38decaac4ed68dd5a596cfbe4849"

INTAKE_SHA256 = (
    "c8383a9d3e8930830a137d1b6791cbb26"
    "e2316dbbbdf4f0f436ceda9a43890d0"
)

STANDARD_RUN = "e020dafedc8b4fd9ac6696d0e4e4a92e"
STANDARD_VERIFICATION = "37b56c39caf5435e8fb2a17f993ec4df"
DATA_VERSION = "1.2.0"

SPLITS = ("train", "validation")

BOUNDARIES = (
    "2025-01-01",
    "2026-01-01",
    "2026-11-26",
)

# 상태 수, 타석 쌍 수, 해당 상태가 존재하는 경기 수.
EXPECTED = {
    ("train", "eligible"): (220986, 110493, 1398),
    ("train", "excluded_half"): (6424, 3212, 62),
    ("validation", "eligible"): (82848, 41424, 533),
    ("validation", "excluded_half"): (28932, 14466, 188),
}


@dataclass(frozen=True, slots=True)
class Pair:
    """동일 경기·타석에서 관측한 전후 상태와 감사 근거."""

    game_id: str
    pair_id: str
    sample_date: str
    split: str
    pre: m4.Observation
    post: m4.Observation
    final: m4.FinalEvidence | None
    actual_runs: int


def expected_split(day: str) -> str:
    """기존 KST 날짜 경계와 일치하는 분할을 반환한다."""
    m4.require(isinstance(day, str), "경기 날짜 타입 오류")

    m4.require(
        date.fromisoformat(day).isoformat() == day,
        "경기 날짜 형식 오류",
    )

    m4.require(
        day < BOUNDARIES[2],
        "관측 종료일 이후 자료",
    )

    if day < BOUNDARIES[0]:
        return "train"
    if day < BOUNDARIES[1]:
        return "validation"
    return "test"


def source(
    settings: Settings,
    run_id: str,
) -> tuple[Path, Path, dict[str, object]]:
    """확정 인수판의 자료 계약과 두 데이터베이스 지문을 검사한다."""
    m4.require(run_id == INTAKE_RUN, "M4 인수 실행 ID 불일치")

    directory = settings.processed_dir / "model_data" / run_id

    database = inside(directory, "model_data.sqlite3")
    manifest_path = inside(directory, "manifest.json")
    contract_path = inside(directory, "feature_target_contract.json")
    eda_path = inside(directory, "train_eda.json")
    supplement_path = inside(directory, "supplement-impact.json")

    manifest = as_row(read_json(manifest_path))
    definitions = as_row(read_json(contract_path))
    contract = as_row(definitions["m4"])
    config = as_row(manifest["config"])
    provenance = as_row(manifest["source"])
    allowed = as_row(contract["inputs"])

    limits = tuple(
        config[name]
        for name in (
            "validation_start",
            "test_start",
            "observation_end",
        )
    )

    m4.require(
        manifest["status"] == "completed"
        and manifest["run_id"] == run_id
        and config["version"] == DATA_VERSION
        and config["timezone"] == "Asia/Seoul"
        and limits == BOUNDARIES
        and config["seed"] == settings.seed
        and provenance["run_id"] == STANDARD_RUN
        and provenance["verification_id"] == STANDARD_VERIFICATION,
        "M4 인수 매니페스트 계약 불일치",
    )

    m4.require(
        contract["index"] == "m4_index"
        and contract["target_columns"] == [
            "sample_id",
            "target_result",
            "target_remaining_runs",
        ]
        and set(allowed) == set(m4.ALL_COLUMNS)
        and set(m4.INPUT_COLUMNS) <= set(allowed),
        "M4 입력·정답 허용 목록 불일치",
    )

    contract_hash = fingerprint(definitions)
    eda_hash = fingerprint(read_json(eda_path))

    m4.require(
        contract_hash == manifest["contract_sha256"]
        and eda_hash == manifest["eda_sha256"],
        "자료 계약 또는 Train EDA 지문 불일치",
    )

    m4.require(
        manifest["database_sha256"] == INTAKE_SHA256
        and sha256_file(database) == INTAKE_SHA256,
        "모델 DB SHA-256 불일치",
    )

    standard_directory = (
        settings.processed_dir / "standard" / STANDARD_RUN
    )

    standard = inside(
        standard_directory,
        f"verified-{STANDARD_VERIFICATION}.sqlite3",
    )

    standard_hash = sha256_file(standard)

    m4.require(
        standard_hash == provenance["database_sha256"],
        "표준 검증 DB SHA-256 불일치",
    )

    as_row(read_json(supplement_path))

    metadata = {
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
        "validation_start": BOUNDARIES[0],
        "test_start": BOUNDARIES[1],
        "observation_end_exclusive": BOUNDARIES[2],
        "allowed_input_columns": sorted(allowed),
        "selected_input_columns": list(m4.INPUT_COLUMNS),
    }

    return database, standard, metadata


def verify_keys(db: sqlite3.Connection) -> None:
    """정답 값 조회 없이 전체 분할의 키·연결 구조를 확인한다."""
    expected_columns = {
        "m4_inputs": {"sample_id", *m4.ALL_COLUMNS},
        "m4_targets": {
            "sample_id",
            "target_result",
            "target_remaining_runs",
        },
        "m4_index": {
            "sample_id", "game_id", "sample_date",
            "split", "status", "pair_id", "pa_number",
            "phase", "terminal_evidence", "source_sha",
            "start_source_row", "end_source_row",
        },
    }

    for table, expected in expected_columns.items():
        columns = {
            row[1]
            for row in db.execute(f"PRAGMA main.table_info({table})")
        }

        m4.require(columns == expected, f"자료 스키마 불일치: {table}")

        invalid = db.execute(f"""
            SELECT 1
            FROM {table}
            GROUP BY sample_id
            HAVING sample_id IS NULL OR COUNT(*) <> 1
            LIMIT 1
        """).fetchone()

        m4.require(invalid is None, f"표본 키 중복 또는 NULL: {table}")

    for table in ("m4_inputs", "m4_targets"):
        orphan = db.execute(f"""
            SELECT 1
            FROM {table} d
            LEFT JOIN m4_index i USING (sample_id)
            WHERE i.sample_id IS NULL OR i.status <> 'eligible'
            LIMIT 1
        """).fetchone()

        missing = db.execute(f"""
            SELECT 1
            FROM m4_index i
            LEFT JOIN {table} d USING (sample_id)
            WHERE i.status = 'eligible' AND d.sample_id IS NULL
            LIMIT 1
        """).fetchone()

        m4.require(
            orphan is None and missing is None,
            f"입력·정답의 일대일 연결 오류: {table}",
        )

    invalid_index = db.execute("""
        SELECT 1
        FROM m4_index
        WHERE game_id IS NULL
           OR sample_date IS NULL
           OR split IS NULL
           OR pair_id IS NULL
           OR pa_number IS NULL
           OR phase IS NULL
           OR status IS NULL
           OR phase NOT IN ('pre', 'post')
           OR status NOT IN ('eligible', 'excluded_half')
           OR pair_id <> game_id || ':' || pa_number
           OR sample_id <> pair_id || ':' || phase
        LIMIT 1
    """).fetchone()

    m4.require(invalid_index is None, "M4 인덱스 계약 오류")

    for game_id, day, split in db.execute("""
        SELECT game_id, sample_date, split
        FROM m4_index
        GROUP BY game_id, sample_date, split
    """):
        m4.require(bool(game_id), "경기 식별자 누락")
        m4.require(split == expected_split(day), "경기 시간 분할 오류")

    invalid_game = db.execute("""
        SELECT 1
        FROM m4_index
        GROUP BY game_id
        HAVING COUNT(DISTINCT split) <> 1
            OR COUNT(DISTINCT sample_date) <> 1
        LIMIT 1
    """).fetchone()

    m4.require(invalid_game is None, "동일 경기의 분할 교차")

    invalid_pair = db.execute("""
        SELECT 1
        FROM m4_index
        GROUP BY pair_id
        HAVING COUNT(*) <> 2
            OR SUM(phase = 'pre') <> 1
            OR SUM(phase = 'post') <> 1
            OR COUNT(DISTINCT game_id) <> 1
            OR COUNT(DISTINCT pa_number) <> 1
            OR COUNT(DISTINCT split) <> 1
            OR COUNT(DISTINCT status) <> 1
        LIMIT 1
    """).fetchone()

    m4.require(invalid_pair is None, "타석 쌍 연결 오류")


def split_counts(
    db: sqlite3.Connection,
) -> dict[str, dict[str, dict[str, int]]]:
    """개발 분할의 유효·제외 상태 및 경기 수를 확인한다."""
    actual = {
        (split, status): (states, pairs, games)
        for split, status, states, pairs, games in db.execute("""
            SELECT split, status, COUNT(*),
                   COUNT(DISTINCT pair_id),
                   COUNT(DISTINCT game_id)
            FROM m4_index
            WHERE split IN ('train', 'validation')
            GROUP BY split, status
        """)
    }

    m4.require(actual == EXPECTED, "인수 개발 표본 건수 불일치")

    result: dict[str, dict[str, dict[str, int]]] = {
        split: {} for split in SPLITS
    }

    for (split, status), (states, pairs, games) in actual.items():
        result[split][status] = {
            "states": states,
            "pairs": pairs,
            "games": games,
        }

    return result


def development_query() -> str:
    """허용 입력과 개발 정답만 조회하며 표준 상태를 독립 대조한다."""
    selected = ", ".join(
        f"x.{name} AS {name}"
        for name in m4.INPUT_COLUMNS
    )

    raw_state = ", ".join(
        f"""
        CASE WHEN i.phase = 'pre'
             THEN p.pre_{name}
             ELSE p.post_{name}
        END AS raw_{name}
        """
        for name in ("home_score", "away_score", "outs")
    )

    raw_bases = ", ".join(
        f"""
        CASE WHEN i.phase = 'pre'
             THEN p.pre_on_{base} IS NOT NULL
             ELSE p.post_on_{base} IS NOT NULL
        END AS raw_{name}
        """
        for base, name in (
            ("1b", "on_first"),
            ("2b", "on_second"),
            ("3b", "on_third"),
        )
    )

    return f"""
        WITH development_games AS (
            SELECT DISTINCT game_id
            FROM m4_index
            WHERE split IN ('train', 'validation')
              AND status = 'eligible'
        ),
        last_game AS (
            SELECT p.game_id, MAX(p.pa_number) AS last_pa
            FROM verified.plate_appearances p
            JOIN development_games d USING (game_id)
            GROUP BY p.game_id
        ),
        last_half AS (
            SELECT p.game_id, p.inning, p.half,
                   MAX(p.pa_number) AS last_half_pa
            FROM verified.plate_appearances p
            JOIN development_games d USING (game_id)
            GROUP BY p.game_id, p.inning, p.half
        )
        SELECT
            i.sample_id AS index_sample_id,
            i.game_id,
            i.pair_id,
            i.pa_number,
            i.sample_date,
            i.split,
            i.phase AS index_phase,
            i.terminal_evidence,
            i.source_sha AS index_source_sha,
            i.start_source_row AS index_start_row,
            i.end_source_row AS index_end_row,
            {selected},
            t.target_result,
            t.target_remaining_runs,
            p.inning AS raw_inning,
            p.half AS raw_half,
            p.source_sha AS raw_source_sha,
            p.start_source_row AS raw_start_row,
            p.end_source_row AS raw_end_row,
            {raw_state},
            {raw_bases},
            g.season AS official_season,
            g.competition AS official_competition,
            g.date AS official_date,
            g.home_team_id AS official_home_team,
            g.away_team_id AS official_away_team,
            g.status AS official_status,
            g.home_score AS official_home_score,
            g.away_score AS official_away_score,
            g.winner,
            g.result_valid,
            c.final_score_match,
            c.order_valid,
            c.boundary_valid,
            c.terminal_state,
            lg.last_pa,
            lh.last_half_pa,
            h.post_outs AS half_end_outs,
            h.post_home_score AS half_end_home_score,
            h.post_away_score AS half_end_away_score
        FROM m4_index i
        JOIN m4_inputs x USING (sample_id)
        JOIN m4_targets t USING (sample_id)
        JOIN verified.eligible_states p
          ON p.game_id = i.game_id
         AND p.pa_number = i.pa_number
        JOIN verified.games g
          ON g.game_id = i.game_id
        JOIN verified.game_checks c
          ON c.game_id = i.game_id
        JOIN last_game lg
          ON lg.game_id = i.game_id
        JOIN last_half lh
          ON lh.game_id = p.game_id
         AND lh.inning = p.inning
         AND lh.half = p.half
        JOIN verified.plate_appearances h
          ON h.game_id = lh.game_id
         AND h.pa_number = lh.last_half_pa
        WHERE i.split IN ('train', 'validation')
          AND i.status = 'eligible'
        ORDER BY i.game_id, i.pa_number, i.phase
    """


def read_observation(
    row: sqlite3.Row,
) -> tuple[m4.Observation, m4.FinalEvidence | None]:
    """현재 상태와 원본을 대조하고 마지막 타석만 종료값을 생성한다."""
    state = m4.parse_state({
        name: row[name]
        for name in m4.INPUT_COLUMNS
    })

    m4.require(
        state.phase == row["index_phase"]
        and state.inning == row["raw_inning"]
        and state.half == row["raw_half"]
        and state.season == row["official_season"]
        and state.competition == row["official_competition"]
        and state.home_team_id == row["official_home_team"]
        and state.away_team_id == row["official_away_team"]
        and row["sample_date"] == row["official_date"],
        "모델 자료와 표준 경기 상태 불일치",
    )

    m4.require(
        row["index_source_sha"] == row["raw_source_sha"]
        and row["index_start_row"] == row["raw_start_row"]
        and row["index_end_row"] == row["raw_end_row"],
        "관측 타석 출처 추적 불일치",
    )

    for name in (
        "home_score", "away_score", "outs",
        "on_first", "on_second", "on_third",
    ):
        m4.require(
            getattr(state, name) == row[f"raw_{name}"],
            f"관측 상태 원본 불일치: {name}",
        )

    m4.require(
        row["official_status"] == "Final"
        and row["result_valid"] == 1
        and row["final_score_match"] == 1
        and row["order_valid"] == 1
        and row["boundary_valid"] == 1,
        "공식 경기 종료 또는 품질 근거 부족",
    )

    home_final = m4.count(
        row["official_home_score"],
        "official_home_score",
    )

    away_final = m4.count(
        row["official_away_score"],
        "official_away_score",
    )

    winner = (
        "home" if home_final > away_final
        else "away" if away_final > home_final
        else "tie"
    )

    m4.require(winner == row["winner"], "공식 종료 정답 불일치")

    expected_result = (
        "tie" if winner == "tie"
        else "win" if (
            (state.attacking_home and winner == "home")
            or (not state.attacking_home and winner == "away")
        )
        else "loss"
    )

    m4.require(
        row["target_result"] == expected_result,
        "공격팀 관점 승리 정답 불일치",
    )

    m4.require(
        state.home_score <= home_final
        and state.away_score <= away_final,
        "현재 득점이 공식 종료 득점을 초과",
    )

    half_outs = row["half_end_outs"]

    if half_outs == 3:
        m4.require(
            row["terminal_evidence"] == "three_outs",
            "반이닝 3아웃 근거 불일치",
        )

    else:
        m4.require(
            row["terminal_evidence"]
            == "home_leads_partial_bottom_candidate"
            and row["terminal_state"]
            == "home_leads_partial_bottom_candidate"
            and row["last_half_pa"] == row["last_pa"]
            and state.half == "bot"
            and winner == "home"
            and row["half_end_home_score"] == home_final
            and row["half_end_away_score"] == away_final,
            "부분 반이닝 종료 근거 부족",
        )

    # 반이닝 전체에 반복되는 terminal_evidence를
    # 개별 타석의 종료 표시로 직접 사용하지 않는다.
    evidence = None

    if (
        state.phase == "post"
        and row["pa_number"] == row["last_pa"]
    ):
        evidence = m4.parse_final({
            "game_id": row["game_id"],
            "pa_number": row["pa_number"],
            "last_pa": row["last_pa"],
            "source_sha": row["raw_source_sha"],
            "end_source_row": row["raw_end_row"],
            "official_status": row["official_status"],
            "official_home_score": home_final,
            "official_away_score": away_final,
            "winner": winner,
            "result_valid": row["result_valid"],
            "final_score_match": row["final_score_match"],
            "order_valid": row["order_valid"],
            "boundary_valid": row["boundary_valid"],
            "terminal_state": row["terminal_state"],
        })

    final = m4.final_result(state, evidence)

    half_end_score = (
        row["half_end_home_score"]
        if state.attacking_home
        else row["half_end_away_score"]
    )

    remaining = m4.count(
        row["target_remaining_runs"],
        "target_remaining_runs",
    )

    m4.require(
        half_end_score - state.score == remaining,
        "반이닝 잔여 실제 득점 정답 불일치",
    )

    if state.phase == "post":
        if state.outs == 3:
            m4.require(
                row["pa_number"] == row["last_half_pa"],
                "마지막 타석 이전의 3아웃 상태",
            )

        if state.outs == 3 or final is not None:
            m4.require(remaining == 0, "종료 상태 잔여 정답 오류")

        if row["pa_number"] == row["last_half_pa"]:
            m4.require(
                state.outs == 3 or final is not None,
                "반이닝 종료 상태 확인 불가",
            )

    win = int(expected_result == "win")

    m4.require(
        final is None or final == win,
        "관측 종료 결과와 승리 정답 불일치",
    )

    return m4.Observation(state, win, remaining, final), evidence


def read_pairs(db: sqlite3.Connection) -> dict[str, list[Pair]]:
    """동일 타석의 두 경계를 연결하고 실제 득점 일관성을 검사한다."""
    db.row_factory = sqlite3.Row

    result: dict[str, list[Pair]] = {
        split: [] for split in SPLITS
    }

    cursor = db.execute(development_query())

    for _, grouped in groupby(
        cursor,
        key=lambda row: (row["game_id"], row["pa_number"]),
    ):
        rows = list(grouped)

        m4.require(len(rows) == 2, "원본 대조 후 타석 상태 수 오류")

        by_phase = {
            row["index_phase"]: row
            for row in rows
        }

        m4.require(
            set(by_phase) == {"pre", "post"},
            "타석 경계 누락",
        )

        before = by_phase["pre"]
        after = by_phase["post"]

        m4.require(
            before["game_id"] == after["game_id"]
            and before["pair_id"] == after["pair_id"]
            and before["sample_date"] == after["sample_date"]
            and before["split"] == after["split"]
            and before["target_result"] == after["target_result"],
            "전후 타석 식별자 또는 경기 정답 불일치",
        )

        pre, pre_evidence = read_observation(before)
        post, post_evidence = read_observation(after)

        m4.require(pre_evidence is None, "타석 직전 종료 근거 유입")

        actual_runs = m4.validate_pair(pre.state, post.state)

        m4.require(
            pre.win == post.win
            and pre.remaining - post.remaining == actual_runs,
            "전후 정답 및 실제 득점 관계 불일치",
        )

        split = before["split"]

        m4.require(
            split == expected_split(before["sample_date"])
            and split in SPLITS,
            "학습·검증 기간 외 타석",
        )

        result[split].append(Pair(
            game_id=before["game_id"],
            pair_id=before["pair_id"],
            sample_date=before["sample_date"],
            split=split,
            pre=pre,
            post=post,
            final=post_evidence,
            actual_runs=actual_runs,
        ))

    return result


def ending_counts(pairs: list[Pair]) -> dict[str, int]:
    """개별 타석의 검증된 종료 형태를 별도로 집계한다."""
    counts: Counter[str] = Counter()

    for pair in pairs:
        pre = pair.pre.state
        post = pair.post.state

        if pair.final is not None:
            counts["confirmed_final"] += 1

            if pair.final.winner == "tie":
                counts["confirmed_final_tie"] += 1

            elif pair.post.final == 1:
                counts["confirmed_final_win"] += 1

            else:
                counts["confirmed_final_loss"] += 1

            if post.outs == 3:
                counts["confirmed_final_three_outs"] += 1
            else:
                counts["confirmed_partial_final"] += 1

                if (
                    post.half == "bot"
                    and post.inning >= 9
                    and pre.difference <= 0
                    and post.difference > 0
                ):
                    counts["observed_walkoff_pattern"] += 1

        elif post.outs == 3:
            counts["three_outs_game_continues"] += 1

        else:
            counts["nonterminal_post"] += 1

    return dict(counts)


def observations(pairs: list[Pair]) -> list[m4.Observation]:
    """타석 쌍의 양쪽 상태를 하나의 동일 표본 목록으로 구성한다."""
    return [
        observation
        for pair in pairs
        for observation in (pair.pre, pair.post)
    ]


def load(
    settings: Settings,
    run_id: str,
) -> tuple[dict[str, list[Pair]], dict[str, object]]:
    """확정 인수판에서 Train·Validation만 읽기 전용으로 가져온다."""
    database, standard, metadata = source(settings, run_id)

    with closing(sqlite3.connect(
        database.resolve().as_uri() + "?mode=ro",
        uri=True,
    )) as db:
        db.execute(
            "ATTACH DATABASE ? AS verified",
            (standard.resolve().as_uri() + "?mode=ro",),
        )
        db.execute("PRAGMA query_only=ON")

        for schema in ("main", "verified"):
            m4.require(
                db.execute(
                    f"PRAGMA {schema}.quick_check"
                ).fetchone()[0] == "ok",
                f"SQLite 무결성 오류: {schema}",
            )

        verify_keys(db)
        counts = split_counts(db)
        pairs = read_pairs(db)

    for split in SPLITS:
        expected = EXPECTED[(split, "eligible")]
        actual = (
            len(pairs[split]) * 2,
            len(pairs[split]),
            len({pair.game_id for pair in pairs[split]}),
        )

        m4.require(actual == expected, f"실제 사용 표본 불일치: {split}")

    train_games = {
        pair.game_id for pair in pairs["train"]
    }

    validation_games = {
        pair.game_id for pair in pairs["validation"]
    }

    m4.require(
        train_games.isdisjoint(validation_games),
        "Train·Validation 동일 경기 중복",
    )

    metadata["counts"] = counts

    metadata["ending_counts"] = {
        split: ending_counts(pairs[split])
        for split in SPLITS
    }

    metadata["excluded_reason"] = {
        "excluded_half": "검증된 전체 반이닝 종료 근거 부족",
    }

    metadata["test_target_values_read"] = False

    return pairs, metadata