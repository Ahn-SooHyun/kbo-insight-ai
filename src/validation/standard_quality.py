"""완료된 표준 자료의 추적성·집계·후속 사용 범위를 검증한다."""

from collections import Counter
from contextlib import closing
import hashlib
from itertools import groupby, zip_longest
import json
import logging
from pathlib import Path
import sqlite3
from uuid import uuid4

from ..datasets.published_sources import (
    EXPECTED_TARGETS,
    RUN_PATTERN,
    SourceVersion,
    as_row,
    intake_sources,
    read_interim_rows,
    read_json,
    read_source_rows,
    sha256_file,
    write_json,
)
from ..settings import Settings
from .standard_data import (
    HIT_EVENTS,
    NON_AB_EVENTS,
    PA_EVENTS,
    TABLE_NAMES,
    StandardDataError,
    after_state,
    before_state,
    pa_state_valid,
    pitch_order_valid,
    regular_game_key,
)


LOGGER = logging.getLogger(__name__)
QUALITY_VERSION = "1.0.0"
Row = dict[str, object]

AUDIT_SCHEMA = """
CREATE TABLE pa_audit (
    game_id TEXT NOT NULL,
    pa_number INTEGER NOT NULL,
    start_complete INTEGER NOT NULL,
    order_complete INTEGER NOT NULL,
    state_consistent INTEGER NOT NULL,
    pitcher_changed INTEGER NOT NULL,
    runs_consistent INTEGER NOT NULL,
    label_kind TEXT NOT NULL,
    PRIMARY KEY (game_id, pa_number)
) STRICT;

CREATE TABLE aggregate_checks (
    game_id TEXT NOT NULL,
    team_id INTEGER NOT NULL,
    role TEXT NOT NULL,
    player_id TEXT NOT NULL,
    metric TEXT NOT NULL,
    direct_value INTEGER,
    observed_value INTEGER,
    status TEXT NOT NULL,
    PRIMARY KEY (game_id, team_id, role, player_id, metric)
) STRICT;

CREATE TABLE player_admission (
    game_id TEXT NOT NULL,
    team_id INTEGER NOT NULL,
    player_id TEXT NOT NULL,
    role TEXT NOT NULL,
    preliminary_candidate INTEGER NOT NULL,
    direct_mismatches INTEGER NOT NULL,
    team_mismatches INTEGER NOT NULL,
    identity_ambiguous INTEGER NOT NULL,
    game_state_uncertain INTEGER NOT NULL,
    admitted INTEGER NOT NULL,
    PRIMARY KEY (game_id, team_id, player_id, role)
) STRICT;

CREATE TABLE postseason_series (
    source_sha TEXT NOT NULL,
    season INTEGER NOT NULL,
    stage TEXT NOT NULL,
    series_complete INTEGER,
    season_complete INTEGER,
    metadata_json TEXT NOT NULL,
    PRIMARY KEY (source_sha, season, stage)
) STRICT;

CREATE TABLE postseason_schedule (
    source_sha TEXT NOT NULL,
    source_row INTEGER NOT NULL,
    stage TEXT NOT NULL,
    series_game_number INTEGER NOT NULL,
    date TEXT NOT NULL,
    away_name TEXT NOT NULL,
    home_name TEXT NOT NULL,
    final_observed INTEGER NOT NULL,
    linked_game_id TEXT,
    PRIMARY KEY (source_sha, source_row)
) STRICT;

CREATE VIEW eligible_games AS
SELECT * FROM games WHERE result_valid=1 AND status='Final';

CREATE VIEW eligible_pa AS
SELECT p.*
FROM plate_appearances p JOIN pa_audit a USING (game_id, pa_number)
WHERE p.result_valid=1 AND p.state_valid=1 AND p.participant_changed=0
  AND a.start_complete=1 AND a.order_complete=1
  AND a.state_consistent=1 AND a.runs_consistent=1
  AND a.label_kind='observed_result';

CREATE VIEW eligible_states AS
SELECT p.*
FROM plate_appearances p JOIN pa_audit a USING (game_id, pa_number)
WHERE p.state_valid=1 AND a.start_complete=1
  AND a.order_complete=1 AND a.state_consistent=1 AND a.runs_consistent=1
  AND a.label_kind<>'unresolved';

CREATE VIEW eligible_pitches AS
SELECT p.*
FROM pitches p JOIN eligible_states s USING (game_id, pa_number)
WHERE p.target_valid=1 AND p.state_valid=1 AND p.order_valid=1;

CREATE VIEW location_candidates AS
SELECT * FROM eligible_pitches WHERE plate_x IS NOT NULL AND plate_z IS NOT NULL;

CREATE VIEW eligible_player_games AS
SELECT p.*
FROM player_games p
JOIN player_admission a USING (game_id, team_id, player_id, role)
WHERE a.admitted=1;

CREATE VIEW direct_pitching_candidates AS
SELECT d.*, CASE WHEN hf_candidate_id IS NULL THEN 'unlinked'
                 ELSE 'contextual_id_match' END AS identity_status
FROM pitching_records d WHERE d.valid=1;

CREATE VIEW starter_candidates AS
SELECT s.*
FROM starters s
WHERE s.valid=1 AND NOT EXISTS (
    SELECT 1 FROM starters t
    WHERE t.game_id=s.game_id AND t.side=s.side AND t.valid=1
      AND t.name<>s.name
);
"""

ELIGIBLE_NAMES = (
    "eligible_games", "eligible_pa", "eligible_states", "eligible_pitches",
    "eligible_player_games", "direct_pitching_candidates", "starter_candidates",
    "location_candidates",
)

AUDIT_TABLES = (
    "pa_audit", "aggregate_checks", "player_admission",
    "postseason_series", "postseason_schedule",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StandardDataError(message)


def _json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    )


def _inside(directory: Path, name: str) -> Path:
    path = (directory / name).resolve()

    _require(
        path.is_relative_to(directory.resolve()),
        "자료 경로 경계 오류",
    )
    _require(path.is_file(), f"필수 파일 누락: {name}")

    return path


def _digest_tables(
    db: sqlite3.Connection,
    names: tuple[str, ...],
) -> Row:
    """기존 표준 자료와 동일한 행 순서·JSON 방식의 논리 해시."""
    result: Row = {}

    for name in names:
        digest = hashlib.sha256()
        count = 0

        for row in db.execute(f"SELECT * FROM {name} ORDER BY rowid"):
            digest.update(_json(list(row)).encode("utf-8"))
            digest.update(b"\n")
            count += 1

        result[name] = {
            "rows": count,
            "sha256": digest.hexdigest(),
        }

    return result


def _verify_database(
    db: sqlite3.Connection,
    previous: Row,
) -> Row:
    """부모 표준 DB가 이전 완료 보고서와 동일한지 확인한다."""
    _require(
        db.execute("PRAGMA integrity_check").fetchone()[0] == "ok",
        "표준 DB 무결성 오류",
    )
    _require(
        not db.execute("PRAGMA foreign_key_check").fetchone(),
        "표준 DB 외래 키 오류",
    )

    actual = _digest_tables(db, TABLE_NAMES)
    recorded = as_row(previous["tables"])

    for name in TABLE_NAMES:
        expected = as_row(recorded[name])

        _require(
            actual[name] == {
                "rows": expected["rows"],
                "sha256": expected["sha256"],
            },
            f"표준 DB 내용과 품질 보고 불일치: {name}",
        )

    return actual


def _verify_interim(
    source: SourceVersion,
    path: Path,
    rows: int,
) -> None:
    """해시뿐 아니라 각 원본의 실제 내용과 중간 행을 대조한다."""
    if path.suffix == ".parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq

        raw = pq.read_table(source.path)
        intermediate = pq.read_table(path)

        _require(
            raw.num_rows == rows and intermediate.num_rows == rows,
            f"Parquet 행 수 불일치: {source.target_id}",
        )
        _require(
            intermediate["_source_row"].combine_chunks().equals(
                pa.array(range(rows), type=pa.int64())
            )
            and raw.equals(intermediate.drop(["_source_row"])),
            f"원본·중간 Parquet 내용 불일치: {source.target_id}",
        )
        return

    missing = object()
    count = 0

    for index, (raw, staged) in enumerate(
        zip_longest(
            read_source_rows(source),
            read_interim_rows(path),
            fillvalue=missing,
        )
    ):
        _require(
            raw is not missing and staged is not missing,
            f"원본·중간 행 수 불일치: {source.target_id}",
        )
        _require(
            staged == {**raw, "_source_row": index},
            f"원본·중간 행 불일치: {source.target_id}:{index}",
        )
        count += 1

    _require(
        count == rows,
        f"중간 행 수 불일치: {source.target_id}",
    )


def _load_inputs(
    settings: Settings,
    run_id: str,
) -> tuple[Path, Row, Row, list[tuple[SourceVersion, Path, int]]]:
    """원본, 변환 이력, 중간 자료 및 채택 판본을 재검증한다."""
    _require(
        bool(RUN_PATTERN.fullmatch(run_id)),
        "표준화 실행 ID 형식 오류",
    )

    processed = settings.processed_dir / "standard" / run_id
    interim = settings.interim_dir / "standard" / run_id

    report = as_row(read_json(_inside(processed, "quality.json")))
    manifest_path = _inside(interim, "manifest.json")
    manifest = as_row(read_json(manifest_path))

    _require(
        report["status"] == "completed"
        and report["run_id"] == manifest["run_id"] == run_id
        and report["input_fingerprint"] == manifest["input_fingerprint"]
        and report["interim_manifest_sha256"] == sha256_file(manifest_path),
        "완료 실행과 중간 명세 연결 오류",
    )

    inputs = as_row(manifest["inputs"])

    _require(
        hashlib.sha256(_json(inputs).encode("utf-8")).hexdigest()
        == manifest["input_fingerprint"],
        "변환 입력 지문 불일치",
    )

    for relative, digest in as_row(
        inputs["code_and_settings_sha256"]
    ).items():
        _require(
            sha256_file(_inside(settings.project_root, relative)) == digest,
            f"변환 당시 코드·설정과 불일치: {relative}",
        )

    intake = intake_sources(
        settings.raw_dir / "published",
        as_of_run=manifest["as_of_run"],
    )

    _require(
        list(intake.history) == manifest["collection_history"],
        "수집 실행 이력과 중간 명세 불일치",
    )
    _require(
        intake.object_count == manifest["object_count"] == report["source_objects"]
        and intake.raw_bytes == manifest["raw_bytes"] == report["source_bytes"],
        "원본 수량·크기 불일치",
    )

    versions = {
        (version.target_id, version.sha256): version
        for version in intake.versions
    }

    selected = []
    seen = set()

    for record in manifest["versions"]:
        key = (record["target_id"], record["source_sha"])

        _require(
            key in versions and key not in seen,
            "원본 판본 목록 불일치",
        )

        seen.add(key)
        source = versions[key]

        _require(
            record["target"] == source.target
            and record["collection_result"] == source.result,
            f"판본 메타데이터 불일치: {source.target_id}",
        )

        path = _inside(interim, record["interim_file"])

        _require(
            sha256_file(path) == record["interim_sha256"],
            f"중간 파일 해시 오류: {source.target_id}",
        )

        _verify_interim(source, path, record["rows"])

        chosen = (
            intake.selected[source.target_id].sha256
            == source.sha256
        )

        _require(
            record["selected"] is chosen,
            "채택 판본 불일치",
        )

        if chosen:
            selected.append((source, path, record["rows"]))

    _require(
        seen == versions.keys()
        and {source.target_id for source, _, _ in selected}
        == EXPECTED_TARGETS,
        "선정 전체 원본을 검증하지 못했습니다.",
    )

    return processed, report, manifest, selected


def _account_rows(
    db: sqlite3.Connection,
    source: SourceVersion,
    total: int,
) -> Row:
    """선정 원본 물리 행마다 변환 또는 제외·격리 근거를 확인한다."""
    key = source.target_id

    if key.startswith("hf-pbp-"):
        tables = ("pitches", "non_pitch")
        entities = ("pbp", "pitch")

    elif key.startswith("dialektike-"):
        tables = ("schedule_reference",)
        entities = ("schedule",)

    elif key.startswith(("dent-", "lopes-")):
        tables = ("games",)
        entities = ("game", "postseason", "source_row")

    elif key == "window-pitching-2023":
        tables = ("pitching_records",)
        entities = ("player_game",)

    else:
        tables = ("starters",)
        entities = ("starter",)

    mapped: set[int] = set()

    for table in tables:
        values = [
            row[0]
            for row in db.execute(
                f"SELECT source_row FROM {table} WHERE source_sha=?",
                (source.sha256,),
            )
        ]

        # 선발 원본의 한 경기 행은 홈·원정 두 관측으로 변환한다.
        if table != "starters":
            _require(
                len(values) == len(set(values))
                and not mapped.intersection(values),
                f"중복된 원본 행 변환: {key}",
            )

        mapped.update(values)

    placeholders = ",".join("?" for _ in entities)

    excluded = {
        row[0]
        for row in db.execute(
            f"""
            SELECT source_row
            FROM issues
            WHERE source_sha=?
              AND entity IN ({placeholders})
              AND source_row IS NOT NULL
              AND severity IN ('quarantine','excluded')
            """,
            (source.sha256, *entities),
        )
    }

    _require(
        mapped | excluded == set(range(total)),
        f"미추적 또는 범위 외 원본 행: {key}",
    )

    return {
        "target_id": key,
        "source_sha": source.sha256,
        "physical_rows": total,
        "mapped_rows": len(mapped),
        "excluded_only": len(excluded - mapped),
        "mapped_with_restriction": len(mapped & excluded),
    }


def inspect_pa(
    rows: list[Row],
) -> tuple[int, int, int, int, str]:
    """타석 그룹의 시작·순서·득점·상태·결과를 서로 구분한다."""
    first, last = rows[0], rows[-1]

    start_complete = (
        first["pitch_number"] in (0, 1)
        and first["balls"] == first["strikes"] == 0
    )

    order = pitch_order_valid(rows)
    state = pa_state_valid(rows)

    if state:
        for row in rows:
            pre = before_state(row)
            post = after_state(row)

            non_batting_score = (
                0 if row["inning_topbot"] == "top" else 1
            )

            if post[non_batting_score] != pre[non_batting_score]:
                state = False

            if any(post[index] < pre[index] for index in range(3)):
                state = False

            for bases in (pre[3:], post[3:]):
                runners = [
                    value for value in bases
                    if value is not None
                ]

                if len(runners) != len(set(runners)):
                    state = False

    pre = before_state(first)
    post = after_state(last)

    runs = post[0] + post[1] - pre[0] - pre[1]

    runs_consistent = all(
        row["runs_scored"] == runs
        for row in rows
    )

    events = {row["events"] for row in rows}

    if len(events) == 1 and first["events"] in PA_EVENTS:
        kind = "observed_result"

    elif events == {None} and last["post_outs"] == 3:
        kind = "unlabeled_half_end_candidate"

    else:
        kind = "unresolved"

    return (
        int(start_complete),
        int(order),
        int(state),
        int(runs_consistent),
        kind,
    )


def _audit_pbp(
    db: sqlite3.Connection,
    source: SourceVersion,
    path: Path,
) -> None:
    """중간 원본의 물리 순서와 표준 타석·실제 투구 대응을 재검사한다."""
    for _, game_rows in groupby(
        read_interim_rows(path),
        key=lambda row: row["game_pk"],
    ):
        rows = list(game_rows)
        first = rows[0]

        game = regular_game_key(
            first["game_pk"],
            first["game_date"],
            first["away_team"],
            first["home_team"],
            hf=True,
        )

        stored = {
            row["pa_number"]: row
            for row in db.execute(
                "SELECT * FROM plate_appearances WHERE game_id=?",
                (game,),
            )
        }

        pitch_rows = {
            row["source_row"]: row
            for row in db.execute(
                "SELECT * FROM pitches WHERE game_id=?",
                (game,),
            )
        }

        non_pitch = {
            row["source_row"]: row
            for row in db.execute(
                "SELECT * FROM non_pitch WHERE game_id=?",
                (game,),
            )
        }

        quarantined = {
            row[0]
            for row in db.execute(
                """
                SELECT source_row
                FROM issues
                WHERE source_sha=?
                  AND source_row BETWEEN ? AND ?
                  AND entity IN ('pbp','pitch')
                  AND severity='quarantine'
                """,
                (
                    source.sha256,
                    rows[0]["_source_row"],
                    rows[-1]["_source_row"],
                ),
            )
        }

        for number, group in groupby(
            rows,
            key=lambda row: row["at_bat_number"],
        ):
            raw = list(group)
            pa = stored.get(number)

            if pa is None:
                _require(
                    all(
                        row["_source_row"] in quarantined
                        for row in raw
                    ),
                    f"타석 누락 근거 없음: {game}:{number}",
                )
                continue

            _require(
                pa["source_sha"] == source.sha256
                and pa["start_source_row"] == raw[0]["_source_row"]
                and pa["end_source_row"] == raw[-1]["_source_row"]
                and pa["start_batter_id"] == "hf:" + raw[0]["batter"]
                and pa["end_batter_id"] == "hf:" + raw[-1]["batter"]
                and pa["start_pitcher_id"] == "hf:" + raw[0]["pitcher"]
                and pa["end_pitcher_id"] == "hf:" + raw[-1]["pitcher"],
                f"타석 원본 연결 오류: {game}:{number}",
            )

            batters = {"hf:" + row["batter"] for row in raw}
            pitchers = {"hf:" + row["pitcher"] for row in raw}
            events = {row["events"] for row in raw}

            event = raw[0]["events"] if len(events) == 1 else None

            _require(
                pa["event"] == event
                and bool(pa["participant_changed"])
                == (len(batters) > 1 or len(pitchers) > 1)
                and pa["physical_pitch_rows"]
                == sum(row["pitch_number"] > 0 for row in raw)
                and (
                    pa["credited_batter_candidate"] is None
                    or (
                        pa["result_valid"] == 1
                        and batters == {pa["credited_batter_candidate"]}
                    )
                )
                and (
                    pa["credited_pitcher_candidate"] is None
                    or (
                        pa["result_valid"] == 1
                        and pitchers == {pa["credited_pitcher_candidate"]}
                    )
                ),
                f"타석 결과·귀속 후보 변환 오류: {game}:{number}",
            )

            state_fields = (
                "home_score",
                "away_score",
                "outs",
                "on_1b",
                "on_2b",
                "on_3b",
            )

            for prefix, state in (
                ("pre", before_state(raw[0])),
                ("post", after_state(raw[-1])),
            ):
                expected = (
                    *state[:3],
                    *(
                        None if value is None else "hf:" + value
                        for value in state[3:]
                    ),
                )

                _require(
                    tuple(
                        pa[f"{prefix}_{field}"]
                        for field in state_fields
                    ) == expected,
                    f"타석 상태 변환 오류: {game}:{number}",
                )

            for row in raw:
                index = row["_source_row"]

                output = (
                    non_pitch
                    if row["pitch_number"] == 0
                    else pitch_rows
                ).get(index)

                if output is None:
                    _require(
                        index in quarantined,
                        f"투구 누락 근거 없음: {game}:{index}",
                    )
                    continue

                _require(
                    output["source_sha"] == source.sha256
                    and output["pa_number"] == number,
                    "투구 원본 행 연결 오류",
                )

                if row["pitch_number"] > 0:
                    _require(
                        all(
                            output[field] == row[field]
                            for field in (
                                "pitch_number",
                                "pitch_type",
                                "pitch_result",
                                "balls",
                                "strikes",
                                "outs_when_up",
                                "home_score",
                                "away_score",
                            )
                        )
                        and output["batter_id"] == "hf:" + row["batter"]
                        and output["pitcher_id"] == "hf:" + row["pitcher"]
                        and output["pitch_result_type"] == row["type"]
                        and all(
                            output[field] == (
                                None
                                if row[field] is None
                                else "hf:" + row[field]
                            )
                            for field in ("on_1b", "on_2b", "on_3b")
                        ),
                        f"실제 투구 필드 변환 오류: {game}:{index}",
                    )

            start, order, state, runs, kind = inspect_pa(raw)

            db.execute(
                "INSERT INTO pa_audit VALUES (?,?,?,?,?,?,?,?)",
                (
                    game,
                    number,
                    start,
                    order,
                    state,
                    int(len(pitchers) > 1),
                    runs,
                    kind,
                ),
            )

    count = db.execute(
        "SELECT COUNT(*) FROM pa_audit"
    ).fetchone()[0]

    expected = db.execute(
        """
        SELECT COUNT(*)
        FROM plate_appearances
        WHERE source_sha IN (
            SELECT DISTINCT p.source_sha
            FROM plate_appearances p
            JOIN pa_audit a USING(game_id,pa_number)
        )
        """
    ).fetchone()[0]

    _require(count == expected, "표준 타석 감사 누락")


def audit_observed_counts(db: sqlite3.Connection) -> None:
    """관측된 선수 경기 집계를 타석·투구 단위에서 다시 계산한다.

    이미 관측된 참가자 행에만 0을 사용하며 미관측 선수 행은 생성하지 않는다.
    """
    hits = ",".join(
        "'" + value + "'"
        for value in sorted(HIT_EVENTS)
    )

    non_ab = ",".join(
        "'" + value + "'"
        for value in sorted(NON_AB_EVENTS)
    )

    query = f"""
    WITH credited AS (
      SELECT a.game_id,
        CASE a.half WHEN 'top' THEN g.away_team_id ELSE g.home_team_id END AS team_id,
        a.credited_batter_candidate AS player_id,'hitter' AS role,a.event
      FROM plate_appearances a JOIN games g USING(game_id)
      WHERE a.credited_batter_candidate IS NOT NULL
      UNION ALL
      SELECT a.game_id,
        CASE a.half WHEN 'top' THEN g.home_team_id ELSE g.away_team_id END AS team_id,
        a.credited_pitcher_candidate,'pitcher',a.event
      FROM plate_appearances a JOIN games g USING(game_id)
      WHERE a.credited_pitcher_candidate IS NOT NULL
    ), counts AS (
      SELECT game_id,team_id,player_id,role,COUNT(*) AS n,
        SUM(event NOT IN ({non_ab})) AS ab,SUM(event IN ({hits})) AS hits,
        SUM(event='home_run') AS hr,SUM(event='walk') AS bb,
        SUM(event='hit_by_pitch') AS hbp,SUM(event='strikeout') AS so
      FROM credited GROUP BY 1,2,3,4
    ), thrown AS (
      SELECT p.game_id,
        CASE a.half WHEN 'top' THEN g.home_team_id ELSE g.away_team_id END AS team_id,
        p.pitcher_id AS player_id,COUNT(*) AS n
      FROM pitches p JOIN plate_appearances a USING(game_id,pa_number)
      JOIN games g USING(game_id) GROUP BY 1,2,3
    ), outs AS (
      SELECT a.game_id,
        CASE a.half WHEN 'top' THEN g.home_team_id ELSE g.away_team_id END AS team_id,
        a.end_pitcher_id AS player_id,SUM(a.post_outs-a.pre_outs) AS n
      FROM plate_appearances a JOIN pa_audit q USING(game_id,pa_number)
      JOIN games g USING(game_id)
      WHERE a.state_valid=1 AND q.pitcher_changed=0 GROUP BY 1,2,3
    )
    SELECT p.game_id,p.player_id,p.role FROM player_games p
    LEFT JOIN counts c USING(game_id,team_id,player_id,role)
    LEFT JOIN thrown t USING(game_id,team_id,player_id)
    LEFT JOIN outs o USING(game_id,team_id,player_id)
    WHERE p.observed_ab IS NOT COALESCE(c.ab,0)
       OR p.observed_hits IS NOT COALESCE(c.hits,0)
       OR p.observed_home_runs IS NOT COALESCE(c.hr,0)
       OR p.observed_walks IS NOT COALESCE(c.bb,0)
       OR p.observed_hbp IS NOT COALESCE(c.hbp,0)
       OR p.observed_strikeouts IS NOT COALESCE(c.so,0)
       OR (p.role='hitter' AND p.observed_pa IS NOT COALESCE(c.n,0))
       OR (p.role='pitcher' AND (p.observed_bf IS NOT COALESCE(c.n,0)
          OR p.observed_pitches IS NOT COALESCE(t.n,0)
          OR p.observed_outs IS NOT COALESCE(o.n,0)))
    LIMIT 1
    """

    _require(
        db.execute(query).fetchone() is None,
        "PBP 선수 관측 집계 재계산 불일치",
    )


METRICS = {
    "at_bats": ("at_bats", "observed_ab"),
    "hits": ("hits", "observed_hits"),
    "home_runs": ("home_runs", "observed_home_runs"),
    "strikeouts": ("strikeouts", "observed_strikeouts"),
    "walks_hbp": ("walks_hbp", "observed_walks+observed_hbp"),
    "batters_faced": ("batters_faced", "observed_pa"),
}


def _record_comparison(
    db: sqlite3.Connection,
    game: str,
    team: int,
    role: str,
    player: str,
    metric: str,
    direct: int | None,
    observed: int | None,
) -> None:
    status = (
        "unavailable"
        if direct is None or observed is None
        else "match"
        if direct == observed
        else "mismatch"
    )

    db.execute(
        "INSERT INTO aggregate_checks VALUES (?,?,?,?,?,?,?,?)",
        (
            game,
            team,
            role,
            player,
            metric,
            direct,
            observed,
            status,
        ),
    )


def _audit_aggregates(db: sqlite3.Connection) -> None:
    """실제 투수 기록에서 상대 타자 집계 및 BB+HBP 교차 검사를 확대한다."""
    for row in db.execute(
        """
        SELECT d.game_id,d.team_id,d.hf_candidate_id,d.walks_hbp,
               p.observed_walks+p.observed_hbp AS observed
        FROM pitching_records d
        JOIN player_games p
          ON p.game_id=d.game_id
         AND p.team_id=d.team_id
         AND p.player_id=d.hf_candidate_id
         AND p.role='pitcher'
        WHERE d.valid=1
        ORDER BY d.game_id,d.team_id,d.hf_candidate_id
        """
    ):
        _record_comparison(
            db,
            row["game_id"],
            row["team_id"],
            "pitcher",
            row["hf_candidate_id"],
            "walks_hbp",
            row["walks_hbp"],
            row["observed"],
        )

    for group in db.execute(
        """
        SELECT d.game_id,d.team_id,g.away_team_id,g.home_team_id,
               SUM(d.valid=0) AS invalid_rows
        FROM pitching_records d
        JOIN games g USING(game_id)
        GROUP BY d.game_id,d.team_id
        ORDER BY d.game_id,d.team_id
        """
    ):
        game = group["game_id"]
        pitching_team = group["team_id"]

        batting_team = (
            group["away_team_id"]
            if pitching_team == group["home_team_id"]
            else group["home_team_id"]
        )

        for metric, (direct, observed) in METRICS.items():
            direct_value = db.execute(
                f"""
                SELECT SUM({direct})
                FROM pitching_records
                WHERE game_id=? AND team_id=?
                """,
                (game, pitching_team),
            ).fetchone()[0]

            observed_value = db.execute(
                f"""
                SELECT SUM({observed})
                FROM player_games
                WHERE game_id=? AND team_id=?
                  AND role='hitter'
                """,
                (game, batting_team),
            ).fetchone()[0]

            _record_comparison(
                db,
                game,
                batting_team,
                "hitter",
                "",
                metric,
                None if group["invalid_rows"] else direct_value,
                observed_value,
            )

        direct = db.execute(
            """
            SELECT SUM(walks_hbp)
            FROM pitching_records
            WHERE game_id=? AND team_id=?
            """,
            (game, pitching_team),
        ).fetchone()[0]

        observed = db.execute(
            """
            SELECT SUM(observed_walks+observed_hbp)
            FROM player_games
            WHERE game_id=? AND team_id=? AND role='pitcher'
            """,
            (game, pitching_team),
        ).fetchone()[0]

        _record_comparison(
            db,
            game,
            pitching_team,
            "pitcher",
            "",
            "walks_hbp",
            None if group["invalid_rows"] else direct,
            observed,
        )


def apply_player_gate(db: sqlite3.Connection) -> None:
    """직접·팀 집계 불일치를 반영한 최종 관측 집계 후보를 결정한다."""
    personal: Counter[tuple[str, int, str, str]] = Counter()
    team: Counter[tuple[str, int, str]] = Counter()

    for row in db.execute(
        """
        SELECT d.game_id,d.team_id,d.hf_candidate_id,c.metric
        FROM comparisons c
        JOIN pitching_records d USING(source_sha,source_row)
        WHERE c.status='mismatch'
          AND (c.metric LIKE 'player:%' OR c.metric LIKE 'team:%')
        """
    ):
        if row["metric"].startswith("player:"):
            personal[
                (
                    row["game_id"],
                    row["team_id"],
                    row["hf_candidate_id"],
                    "pitcher",
                )
            ] += 1

        else:
            team[
                (
                    row["game_id"],
                    row["team_id"],
                    "pitcher",
                )
            ] += 1

    for row in db.execute(
        "SELECT * FROM aggregate_checks WHERE status='mismatch'"
    ):
        if row["player_id"]:
            personal[
                (
                    row["game_id"],
                    row["team_id"],
                    row["player_id"],
                    row["role"],
                )
            ] += 1

        else:
            team[
                (
                    row["game_id"],
                    row["team_id"],
                    row["role"],
                )
            ] += 1

    uncertain_games = {
        row[0]
        for row in db.execute(
            """
            SELECT DISTINCT game_id
            FROM pa_audit
            WHERE start_complete=0
               OR order_complete=0
               OR state_consistent=0
               OR runs_consistent=0
               OR label_kind='unresolved'
            """
        )
    }

    identities = {
        tuple(row)
        for row in db.execute(
            """
            SELECT player_id,season,team_id
            FROM player_observations
            GROUP BY player_id,season,team_id
            HAVING COUNT(DISTINCT name)>1
            """
        )
    }

    records = []

    for row in db.execute(
        """
        SELECT p.*,g.season
        FROM player_games p
        JOIN games g USING(game_id)
        ORDER BY p.game_id,p.team_id,p.player_id,p.role
        """
    ):
        key = (
            row["game_id"],
            row["team_id"],
            row["player_id"],
            row["role"],
        )

        direct = personal[key]
        group = team[(key[0], key[1], key[3])]

        ambiguous = int(
            (key[2], row["season"], key[1]) in identities
        )
        uncertain = int(key[0] in uncertain_games)

        admitted = int(
            row["coverage_candidate"] == 1
            and not (direct or group or ambiguous or uncertain)
        )

        records.append((
            *key,
            row["coverage_candidate"],
            direct,
            group,
            ambiguous,
            uncertain,
            admitted,
        ))

    db.executemany(
        "INSERT INTO player_admission VALUES (?,?,?,?,?,?,?,?,?,?)",
        records,
    )

    # 원래 후보는 player_admission에 보존하고 최종 DB의 후보는 좁힌다.
    db.execute(
        """
        UPDATE player_games
        SET coverage_candidate=0
        WHERE EXISTS (
            SELECT 1
            FROM player_admission a
            WHERE a.game_id=player_games.game_id
              AND a.team_id=player_games.team_id
              AND a.player_id=player_games.player_id
              AND a.role=player_games.role
              AND a.admitted=0
        )
        """
    )


def _audit_postseason(
    db: sqlite3.Connection,
    source: SourceVersion,
    path: Path,
) -> None:
    """미완료 판본의 상대 미정 일정을 포함해 시리즈 원문 근거를 보존한다."""
    raw = as_row(as_row(read_json(source.path))["2026"])

    for stage in (
        "wildCard",
        "semiPlayoff",
        "playoff",
        "koreanSeries",
    ):
        series = as_row(raw[stage])

        metadata = {
            key: value
            for key, value in series.items()
            if key != "games"
        }

        metadata["seeds"] = raw.get("seeds")
        metadata["championId"] = raw.get("championId")

        db.execute(
            "INSERT INTO postseason_series VALUES (?,?,?,?,?,?)",
            (
                source.sha256,
                2026,
                stage,
                series.get("isComplete"),
                raw.get("isComplete"),
                _json(metadata),
            ),
        )

    for row in read_interim_rows(path):
        linked = db.execute(
            """
            SELECT game_id
            FROM games
            WHERE source_sha=? AND source_row=?
            """,
            (
                source.sha256,
                row["_source_row"],
            ),
        ).fetchone()

        db.execute(
            "INSERT INTO postseason_schedule VALUES (?,?,?,?,?,?,?,?,?)",
            (
                source.sha256,
                row["_source_row"],
                row["_stage"],
                row["game"],
                row["date"],
                row["away"],
                row["home"],
                int(row.get("isFinal") is True),
                linked[0] if linked else None,
            ),
        )


def _handoff(db: sqlite3.Connection) -> Row:
    """후속 작업이 사용할 데이터의 키·분모·후보·제한을 기록한다."""
    counts = {
        name: db.execute(
            f"SELECT COUNT(*) FROM {name}"
        ).fetchone()[0]
        for name in ELIGIBLE_NAMES
    }

    base_tables = {
        "eligible_games": "games",
        "eligible_pa": "plate_appearances",
        "eligible_states": "plate_appearances",
        "eligible_pitches": "pitches",
        "location_candidates": "pitches",
        "eligible_player_games": "player_games",
        "direct_pitching_candidates": "pitching_records",
        "starter_candidates": "starters",
    }

    coverage = {}

    for view, base in base_tables.items():
        eligible = {
            (row[0], row[1]): row[2]
            for row in db.execute(
                f"""
                SELECT g.season,g.competition,COUNT(*)
                FROM {view} v
                JOIN games g ON v.game_id=g.game_id
                GROUP BY 1,2
                """
            )
        }

        coverage[view] = [
            {
                "season": row[0],
                "competition": row[1],
                "total": row[2],
                "candidates": eligible.get((row[0], row[1]), 0),
                "not_admitted": (
                    row[2] - eligible.get((row[0], row[1]), 0)
                ),
            }
            for row in db.execute(
                f"""
                SELECT g.season,g.competition,COUNT(*)
                FROM {base} v
                JOIN games g ON v.game_id=g.game_id
                GROUP BY 1,2
                ORDER BY 1,2
                """
            )
        ]

    seasons = [
        dict(row)
        for row in db.execute(
            """
            SELECT season,competition,
                   COUNT(*) AS observed_rows,
                   SUM(result_valid) AS final_results,
                   MAX(season_complete) AS completion_evidence
            FROM games
            GROUP BY season,competition
            ORDER BY season,competition
            """
        )
    ]

    # 실제 공개 시간은 자료별 명세에 없으므로 정답 후보와 공개 입력을 구분한다.
    return {
        "candidate_counts": counts,
        "coverage_by_season": coverage,
        "season_evidence": seasons,
        "pa_audit": [
            dict(row)
            for row in db.execute(
                """
                SELECT label_kind,start_complete,order_complete,
                       state_consistent,runs_consistent,
                       COUNT(*) AS count
                FROM pa_audit
                GROUP BY 1,2,3,4,5
                ORDER BY 1,2,3,4,5
                """
            )
        ],
        "player_admission": [
            dict(row)
            for row in db.execute(
                """
                SELECT role,
                       COUNT(*) AS total,
                       SUM(preliminary_candidate) AS preliminary,
                       SUM(admitted) AS admitted,
                       SUM(direct_mismatches>0) AS direct_conflicts,
                       SUM(team_mismatches>0) AS team_conflicts,
                       SUM(identity_ambiguous) AS identity_conflicts,
                       SUM(game_state_uncertain) AS uncertain_game
                FROM player_admission
                GROUP BY role
                ORDER BY role
                """
            )
        ],
        "aggregate_checks": [
            dict(row)
            for row in db.execute(
                """
                SELECT role,metric,status,COUNT(*) AS count
                FROM aggregate_checks
                GROUP BY 1,2,3
                ORDER BY 1,2,3
                """
            )
        ],
        "model_contract": {
            "M1": {
                "view": "eligible_games",
                "key": ["game_id"],
                "period": (
                    "2001-2026 regular; "
                    "2010-2018,2026 postseason observed"
                ),
                "limit": (
                    "사후 결과; 과거 일정/선발 공개시점 "
                    "및 공식 규칙 미확인"
                ),
            },
            "M2": {
                "view": "eligible_pa",
                "key": ["game_id", "pa_number"],
                "period": "2023-2026",
                "limit": (
                    "타석 시작·순서·결과·참가자·상태 통과 후보"
                ),
            },
            "M3": {
                "view": "eligible_player_games",
                "key": [
                    "game_id",
                    "team_id",
                    "player_id",
                    "role",
                ],
                "period": (
                    "2023-2026 PBP; direct pitching 2026 subset"
                ),
                "limit": (
                    "관측 집계 후보이지 공식 기록이나 월요일별 "
                    "최종 정답이 아님; NULL/미관측과 0 구분; "
                    "독립 자료 없는 지표는 확정 금지"
                ),
            },
            "M4": {
                "view": "eligible_states",
                "key": ["game_id", "pa_number"],
                "period": "2023-2026",
                "limit": "타석 전후 상태; 미래 입력 금지",
            },
            "M5": {
                "view": "eligible_pitches",
                "key": [
                    "game_id",
                    "pa_number",
                    "pitch_number",
                ],
                "period": "2023-2026",
                "limit": (
                    "실제 투구 구종·결과 후보; "
                    "좌표 결측·원단위 미확인 별도"
                ),
            },
            "starters": {
                "view": "starter_candidates",
                "limit": (
                    "사후 직접/최초 관측; 경기 전 공개 증거 아님"
                ),
            },
            "postseason": {
                "tables": [
                    "postseason_series",
                    "postseason_schedule",
                    "schedule_reference",
                ],
                "limit": (
                    "원본 시리즈 상태와 일정 참조; "
                    "2019-2025 점수/승자 및 공식 시즌 규칙 확정 금지"
                ),
            },
        },
    }


def validate_standard(
    settings: Settings,
    run_id: str,
    *,
    compare_with: str | None = None,
) -> Row:
    """검증된 부모 DB의 별도 최종 사본을 생성하고 비교 근거를 저장한다."""
    LOGGER.info("표준 자료 및 원본 대응 검사 시작: %s", run_id)

    directory, quality, manifest, selected = _load_inputs(
        settings,
        run_id,
    )

    db_path = _inside(directory, "standard.sqlite3")

    audit_id = uuid4().hex

    output_path = directory / f"verified-{audit_id}.sqlite3"
    report_path = directory / f"verified-{audit_id}.json"

    previous = None

    if compare_with is not None:
        parts = compare_with.split(":")

        _require(
            len(parts) == 2
            and all(RUN_PATTERN.fullmatch(part) for part in parts),
            "비교 키는 표준화 실행 ID:검증 ID 형식입니다.",
        )

        previous = as_row(
            read_json(
                _inside(
                    settings.processed_dir / "standard" / parts[0],
                    f"verified-{parts[1]}.json",
                )
            )
        )

        _require(
            previous["status"] == "completed",
            "비교 검증이 완료되지 않았습니다.",
        )

    _require(
        not output_path.exists(),
        "기존 검증 산출물 충돌",
    )

    with closing(
        sqlite3.connect(
            db_path.as_uri() + "?mode=ro",
            uri=True,
        )
    ) as parent:
        parent.row_factory = sqlite3.Row

        parent_hashes = _verify_database(parent, quality)

        row_accounting = [
            _account_rows(parent, source, count)
            for source, _, count in selected
        ]

        with closing(sqlite3.connect(output_path)) as db:
            db.row_factory = sqlite3.Row

            parent.backup(db)
            db.executescript(AUDIT_SCHEMA)

            try:
                for source, path, _ in selected:
                    if source.target_id.startswith("hf-pbp-"):
                        LOGGER.info(
                            "타석·투구 원본 대응 검사: %s",
                            source.target_id,
                        )

                        _audit_pbp(db, source, path)

                    elif (
                        source.target_id
                        == "dent-postseason-2026-wildcard-probe"
                    ):
                        _audit_postseason(db, source, path)

                _require(
                    db.execute(
                        "SELECT COUNT(*) FROM pa_audit"
                    ).fetchone()[0]
                    == db.execute(
                        "SELECT COUNT(*) FROM plate_appearances"
                    ).fetchone()[0],
                    "전체 논리적 타석 감사 누락",
                )

                LOGGER.info("선수 집계 재계산 및 원천 간 교차 검사")

                audit_observed_counts(db)
                _audit_aggregates(db)
                apply_player_gate(db)

                _require(
                    not db.execute(
                        """
                        SELECT 1
                        FROM player_admission
                        WHERE admitted=1
                          AND (
                              direct_mismatches>0
                              OR team_mismatches>0
                              OR identity_ambiguous>0
                              OR game_state_uncertain>0
                          )
                        LIMIT 1
                        """
                    ).fetchone(),
                    "불일치 기록의 사용 차단 실패",
                )

                _require(
                    db.execute(
                        "PRAGMA integrity_check"
                    ).fetchone()[0] == "ok"
                    and not db.execute(
                        "PRAGMA foreign_key_check"
                    ).fetchone(),
                    "최종 검증 DB 무결성 오류",
                )

                hashes = _digest_tables(
                    db,
                    TABLE_NAMES + AUDIT_TABLES,
                )

                handoff = _handoff(db)

                db.commit()

            except BaseException:
                db.rollback()
                raise

    inputs = {
        "parent_input_fingerprint": quality["input_fingerprint"],
        "parent_tables": parent_hashes,
        "quality_version": QUALITY_VERSION,
        "quality_code_sha256": sha256_file(Path(__file__)),
    }

    fingerprint = hashlib.sha256(
        _json(inputs).encode("utf-8")
    ).hexdigest()

    comparison = None

    if previous is not None:
        same = previous["input_fingerprint"] == fingerprint
        old = as_row(previous["tables"])

        changed = [
            name
            for name in hashes
            if hashes[name] != old[name]
        ]

        comparison = {
            "same_input": same,
            "changed_tables": changed,
            "reproducible": not changed if same else None,
        }

    completed = (
        comparison is None
        or comparison["reproducible"] is not False
    )

    result = {
        "status": (
            "completed" if completed else "reproduction_failed"
        ),
        "run_id": run_id,
        "verification_id": audit_id,
        "verification_key": f"{run_id}:{audit_id}",
        "input_fingerprint": fingerprint,
        "inputs": inputs,
        "source_objects": manifest["object_count"],
        "logical_targets": len(selected),
        "parent_run": quality["run_id"],
        "parent_coverage": quality["coverage"],
        "source_specs": {
            path.name: sha256_file(path)
            for path in sorted(
                (settings.raw_dir / "sources").glob("*.json")
            )
        },
        "row_accounting": row_accounting,
        "tables": hashes,
        "handoff": handoff,
        "comparison": comparison,
        "database_path": str(output_path),
        "report_path": str(report_path),
    }

    write_json(report_path, result)

    _require(
        completed,
        "동일 입력의 최종 검증 결과 재현 실패",
    )

    LOGGER.info("표준 자료 최종 품질 검사 완료: %s", audit_id)

    return {
        "status": "completed",
        "verification_key": result["verification_key"],
        "database_path": str(output_path),
        "report_path": str(report_path),
        "candidate_counts": handoff["candidate_counts"],
        "comparison": comparison,
    }