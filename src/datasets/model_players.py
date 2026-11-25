"""월요일 기준 선수 모집단·과거 집계·기간별 관측 정답을 만든다."""

from collections import defaultdict
from datetime import date, timedelta
import sqlite3


ROLES = {"hitter": "m3_hitter", "pitcher": "m3_pitcher"}
COMMON_COUNTS = ("ab", "hits", "home_runs", "walks", "hbp", "strikeouts")
ROLE_COUNTS = {
    "hitter": ("pa", *COMMON_COUNTS),
    "pitcher": ("bf", "outs", "pitches", *COMMON_COUNTS),
}
HORIZONS = ("7d", "28d", "remaining")
DAY = timedelta(days=1)


def monday_on_or_after(value: date) -> date:
    """KST 날짜 기준 다음 월요일(당일 월요일 포함)."""
    return value + timedelta(days=(-value.weekday()) % 7)


def prepare_player_history(db: sqlite3.Connection) -> None:
    """이름 관측 시점을 원본 행에 연결하고 시점별 이력 채택 근거를 준비한다.

    최종 선수 admission은 정답용이다. 과거 입력은 경기별 품질 판정과
    기준 날짜 전에 이미 관측된 이름 충돌만 사용한다.
    """
    if db.execute("SELECT 1 FROM sqlite_temp_master WHERE name='player_history_base'").fetchone():
        return
    db.executescript("""
    CREATE TEMP TABLE identity_locations AS
    SELECT source_sha,source_row,game_id FROM source.pitches
    UNION ALL SELECT source_sha,source_row,game_id FROM source.non_pitch;
    CREATE INDEX temp.identity_locations_key ON identity_locations(source_sha,source_row);
    CREATE TABLE player_identity_evidence AS
    SELECT o.*,l.game_id,g.date first_seen_date,g.season observed_season
    FROM source.player_observations o LEFT JOIN identity_locations l USING(source_sha,source_row)
    LEFT JOIN source.games g USING(game_id)
    ORDER BY o.player_id,o.season,o.team_id,o.name;
    """)
    expected = db.execute("SELECT COUNT(*) FROM source.player_observations").fetchone()[0]
    actual = db.execute("SELECT COUNT(*),SUM(first_seen_date IS NULL OR season IS NOT observed_season) FROM player_identity_evidence").fetchone()
    if actual[0] != expected or actual[1]:
        raise ValueError("선수 이름 관측 원본 행과 경기 날짜 연결 오류")
    # 원본 행의 첫 관측 기록을 최초 관측일로 쓸 수 있는 날짜 순서를 확인한다.
    if db.execute("""
      WITH ranges AS (
        SELECT l.source_sha,l.game_id,g.date,MIN(l.source_row) first_row,MAX(l.source_row) last_row
        FROM identity_locations l JOIN source.games g USING(game_id)
        WHERE l.source_sha IN (SELECT DISTINCT source_sha FROM player_identity_evidence)
        GROUP BY l.source_sha,l.game_id,g.date
      ), ordered AS (
        SELECT *,LAG(date) OVER(PARTITION BY source_sha ORDER BY first_row) previous_date,
          LAG(last_row) OVER(PARTITION BY source_sha ORDER BY first_row) previous_end FROM ranges
      ) SELECT 1 FROM ordered WHERE previous_date>date OR previous_end>=first_row LIMIT 1
    """).fetchone():
        raise ValueError("이름 최초 관측을 위한 원본 경기 날짜 순서 불확실")
    # 여러 시즌/팀을 이름만으로 병합하지 않는다. 두 번째 이름이 보인 날짜부터 충돌이다.
    db.executescript("""
    CREATE TEMP TABLE player_identity_conflicts AS
    WITH ordered AS (
      SELECT player_id,season,team_id,first_seen_date,
        ROW_NUMBER() OVER(PARTITION BY player_id,season,team_id ORDER BY first_seen_date,name) n
      FROM player_identity_evidence
    ) SELECT player_id,season,team_id,first_seen_date conflict_date FROM ordered WHERE n=2;
    CREATE UNIQUE INDEX temp.player_identity_conflicts_key ON player_identity_conflicts(player_id,season,team_id);
    CREATE TEMP TABLE player_history_base AS
    SELECT p.*,g.season,g.date,a.admitted,a.identity_ambiguous,c.conflict_date,
      (a.preliminary_candidate=1 AND a.direct_mismatches=0 AND a.team_mismatches=0
        AND a.game_state_uncertain=0) history_candidate
    FROM source.player_games p JOIN source.player_admission a USING(game_id,team_id,player_id,role)
    JOIN source.eligible_games g USING(game_id)
    LEFT JOIN player_identity_conflicts c ON c.player_id=p.player_id AND c.season=g.season AND c.team_id=p.team_id
    WHERE g.competition='regular';
    CREATE INDEX temp.player_history_base_key ON player_history_base(player_id,role,date);
    DROP TABLE identity_locations;
    """)
    if db.execute("""SELECT 1 FROM player_history_base
      WHERE identity_ambiguous<>(conflict_date IS NOT NULL)
         OR admitted<>(history_candidate=1 AND conflict_date IS NULL) LIMIT 1""").fetchone():
        raise ValueError("최종 선수 품질 판정과 시점별 식별 근거 불일치")


def _summarize(rows: list[sqlite3.Row], counts: tuple[str, ...], *, historical: bool = False) -> tuple[int, ...]:
    """관측 집계를 합산한다. 입력은 시점별 이력, 정답은 최종 admission을 사용한다."""
    flag = "history_candidate" if historical else "admitted"
    admitted = [row for row in rows if row[flag] == 1]
    return (len({row["game_id"] for row in admitted}),) + tuple(
        sum(int(row[f"observed_{field}"]) for row in admitted) for field in counts
    )


def _status(
    *, ready: bool, admitted: int, rejected: int, missing_games: int,
    risky_games: int, league_risky_games: int, league_identity_conflicts: int,
    league_rejected_rows: int = 0,
) -> str:
    """누락·부분·미완료와 관측 0을 분리한다. 공식 출전 여부를 추정하지 않는다."""
    if not ready:
        return "partial_unfinished" if admitted else "unobserved"
    if rejected or missing_games or risky_games:
        return "partial" if admitted else "missing"
    if admitted:
        return "valid_observed"
    if league_risky_games or league_identity_conflicts or league_rejected_rows:
        return "unobserved"
    return "zero_observed"


def _build_role_quality(db: sqlite3.Connection) -> None:
    """완료 경기의 양 팀·역할별 구조/선수 집계 위험을 별도 보존한다."""
    db.executescript("""
    CREATE TABLE player_game_role_quality AS
    WITH games AS (
      SELECT g.* FROM source.eligible_games g WHERE g.competition='regular' AND g.season IN
        (SELECT DISTINCT x.season FROM source.eligible_games x JOIN source.game_checks c USING(game_id))
    ), teams AS (
      SELECT game_id,season,date,home_team_id team_id FROM games
      UNION ALL SELECT game_id,season,date,away_team_id FROM games
    ), roles(role) AS (VALUES ('hitter'),('pitcher')),
    audit AS (
      SELECT game_id,COUNT(*) n,MAX(start_complete=0 OR order_complete=0 OR state_consistent=0
        OR runs_consistent=0 OR label_kind='unresolved') uncertain
      FROM source.pa_audit GROUP BY game_id
    ), admission AS (
      SELECT game_id,team_id,role,COUNT(*) n,SUM(admitted=0) rejected_rows,
        MAX(team_mismatches) team_mismatches,MAX(game_state_uncertain) uncertain,
        SUM(identity_ambiguous) identity_conflicts
      FROM source.player_admission GROUP BY game_id,team_id,role
    )
    SELECT t.game_id,t.season,t.date,t.team_id,r.role,
      (c.game_id IS NULL OR c.final_score_match<>1) missing_game,
      (c.game_id IS NULL OR c.order_valid<>1 OR c.boundary_valid<>1
        OR c.terminal_state='terminal_reason_unresolved' OR COALESCE(q.n,0)<>c.pa_count
        OR COALESCE(q.uncertain,0)>0 OR COALESCE(a.uncertain,0)>0) state_uncertain,
      COALESCE(a.n,0) role_rows,COALESCE(a.rejected_rows,0) rejected_rows,
      COALESCE(a.team_mismatches,0) team_mismatches,COALESCE(a.identity_conflicts,0) identity_conflicts,
      (c.game_id IS NULL OR c.final_score_match<>1 OR c.order_valid<>1 OR c.boundary_valid<>1
        OR c.terminal_state='terminal_reason_unresolved' OR COALESCE(q.n,0)<>c.pa_count
        OR COALESCE(q.uncertain,0)>0 OR COALESCE(a.uncertain,0)>0 OR a.n IS NULL
        OR COALESCE(a.team_mismatches,0)>0 OR COALESCE(a.identity_conflicts,0)>0) risky
    FROM teams t CROSS JOIN roles r LEFT JOIN source.game_checks c USING(game_id)
    LEFT JOIN audit q USING(game_id) LEFT JOIN admission a USING(game_id,team_id,role)
    ORDER BY t.season,t.date,t.game_id,t.team_id,r.role;
    CREATE UNIQUE INDEX player_game_role_quality_key ON player_game_role_quality(game_id,team_id,role);
    """)


def build_players(db: sqlite3.Connection, observation_end: str) -> None:
    """시즌 내 과거 관측 선수의 주별 입력과 3개 기간 정답·관측 상태를 생성한다."""
    db.row_factory = sqlite3.Row
    config = db.execute("SELECT * FROM config").fetchone()
    validation_start = date.fromisoformat(config["validation_start"])
    test_start = date.fromisoformat(config["test_start"])
    cutoff = date.fromisoformat(observation_end)
    prepare_player_history(db)
    _build_role_quality(db)
    db.execute("""CREATE TABLE m3_window_games(season INTEGER,anchor_date TEXT,horizon TEXT,game_id TEXT,
                PRIMARY KEY(season,anchor_date,horizon,game_id))""")
    db.execute("""CREATE TABLE m3_window_teams(sample_id TEXT,role TEXT,horizon TEXT,team_id INTEGER,
                PRIMARY KEY(sample_id,role,horizon,team_id))""")
    for role, model in ROLES.items():
        counts = ROLE_COUNTS[role]
        metrics = ("games", *counts)
        features = ",".join(f"{prefix}_{field} INTEGER" for prefix in ("season", "recent28") for field in metrics)
        observed = ",".join(f"observed_{field} INTEGER" for field in metrics)
        targets = ",".join(f"target_{field} INTEGER" for field in metrics)
        db.executescript(f"""
        CREATE TABLE {model}_index(sample_id TEXT PRIMARY KEY,player_id TEXT,role TEXT,season INTEGER,
          sample_date TEXT,split TEXT,status TEXT,last_observed_date TEXT);
        CREATE TABLE {model}_inputs(sample_id TEXT PRIMARY KEY,player_id TEXT,season INTEGER,known_team_id INTEGER,
          days_since_observed INTEGER,prior_rejected INTEGER,{features});
        CREATE TABLE {model}_targets(sample_id TEXT,horizon TEXT,window_start TEXT,window_end TEXT,
          observed_through_exclusive TEXT,ready_date TEXT,status TEXT,usable INTEGER,observation_scope TEXT,
          league_games INTEGER,missing_games INTEGER,league_risky_games INTEGER,team_risky_games INTEGER,
          league_identity_conflicts INTEGER,league_rejected_rows INTEGER,team_rejected_rows INTEGER,
          player_rejected INTEGER,ratio_status TEXT,
          {observed},{targets},target_hit_rate REAL,target_strikeout_rate REAL,
          PRIMARY KEY(sample_id,horizon));
        """)

    seasons = db.execute("""
      SELECT g.season,MIN(g.date) first_date,MAX(g.date) last_date,MIN(COALESCE(g.season_complete,0)) complete
      FROM source.eligible_games g WHERE g.competition='regular' AND g.season IN
        (SELECT DISTINCT x.season FROM source.eligible_games x JOIN source.game_checks c USING(game_id))
      GROUP BY g.season ORDER BY g.season
    """).fetchall()
    for season_row in seasons:
        season = season_row["season"]
        first = date.fromisoformat(season_row["first_date"])
        last = date.fromisoformat(season_row["last_date"])
        season_end = last + DAY if season_row["complete"] == 1 else None
        # 경기 단위 누락과 팀·역할별 품질 제한을 정답 창에 연결한다.
        games = db.execute("""
          SELECT g.game_id,g.date
          FROM source.eligible_games g
          WHERE g.competition='regular' AND g.season=? ORDER BY g.date,g.game_id
        """, (season,)).fetchall()
        quality: dict[tuple[str, str], list[sqlite3.Row]] = defaultdict(list)
        for q in db.execute("SELECT * FROM player_game_role_quality WHERE season=? ORDER BY game_id,team_id,role", (season,)):
            quality[(q["game_id"], q["role"])].append(q)
        histories: dict[tuple[str, str], list[sqlite3.Row]] = defaultdict(list)
        for row in db.execute("""
          SELECT * FROM player_history_base WHERE season=?
          ORDER BY date,game_id,team_id,player_id,role
        """, (season,)):
            histories[(row["role"], row["player_id"])].append(row)

        anchor = monday_on_or_after(first)
        while anchor <= last and anchor < cutoff:
            anchor_text = anchor.isoformat()
            split = "train" if anchor < validation_start else "validation" if anchor < test_start else "test"
            label_cutoff = validation_start if split == "train" else test_start if split == "validation" else cutoff
            windows = []
            for horizon in HORIZONS:
                end = (anchor + timedelta(days=7 if horizon == "7d" else 28)
                       if horizon != "remaining" else season_end)
                # 종료 근거가 없는 시즌은 잠정 관측 범위만 보존한다.
                observed_end = min(end or last + DAY, last + DAY, cutoff)
                ready = end is not None and end <= cutoff and (end <= last + DAY or season_end is not None)
                selected = [game for game in games if anchor_text <= game["date"] < observed_end.isoformat()]
                db.executemany("INSERT INTO m3_window_games VALUES (?,?,?,?)", (
                    (season, anchor_text, horizon, game["game_id"]) for game in selected
                ))
                windows.append((horizon, end, observed_end, ready, selected))

            for (role, player_id), rows in sorted(histories.items()):
                # 미래 참가 여부나 미래 팀은 모집단·입력에 쓰지 않는다.
                past = [row for row in rows if row["date"] < anchor_text and
                        (row["conflict_date"] is None or row["conflict_date"] >= anchor_text)]
                if not past:
                    continue
                model = ROLES[role]
                counts = ROLE_COUNTS[role]
                metrics = ("games", *counts)
                sample_id = f"{season}:{anchor_text}:{player_id}"
                last_date = past[-1]["date"]
                latest_teams = {row["team_id"] for row in past if row["date"] == last_date}
                known_team = next(iter(latest_teams)) if len(latest_teams) == 1 else None
                recent = [row for row in past if row["date"] >= (anchor - timedelta(days=28)).isoformat()]
                past_counts = _summarize(past, counts, historical=True)
                recent_counts = _summarize(recent, counts, historical=True)
                rejected_count = sum(row["history_candidate"] == 0 for row in past)
                db.execute(f"INSERT INTO {model}_index VALUES (?,?,?,?,?,?,?,?)", (
                    sample_id, player_id, role, season, anchor_text, split, "candidate", last_date,
                ))
                inputs = (sample_id, player_id, season, known_team,
                          (anchor - date.fromisoformat(last_date)).days, rejected_count,
                          *(past_counts if past_counts[0] else (0, *([None] * len(counts)))),
                          *(recent_counts if recent_counts[0] else (0, *([None] * len(counts)))))
                db.execute(f"INSERT INTO {model}_inputs VALUES ({','.join('?' for _ in inputs)})", inputs)

                for horizon, end, observed_end, ready, selected in windows:
                    future = [row for row in rows if anchor_text <= row["date"] < observed_end.isoformat()]
                    totals = _summarize(future, counts)
                    rejected = sum(row["admitted"] == 0 for row in future)
                    # 정답의 자료 범위를 검사할 때만 실제 미래 관측 팀을 사용한다.
                    relevant_teams = latest_teams | {row["team_id"] for row in future}
                    db.executemany("INSERT INTO m3_window_teams VALUES (?,?,?,?)", (
                        (sample_id, role, horizon, team_id) for team_id in sorted(relevant_teams)
                    ))
                    coverage = [q for game in selected for q in quality[(game["game_id"], role)]]
                    missing = len({q["game_id"] for q in coverage if q["missing_game"]})
                    risky = len({q["game_id"] for q in coverage if q["risky"]})
                    team_risky = len({q["game_id"] for q in coverage if q["risky"] and q["team_id"] in relevant_teams})
                    conflicts = sum(q["identity_conflicts"] for q in coverage)
                    league_rejected = sum(q["rejected_rows"] for q in coverage)
                    team_rejected = sum(q["rejected_rows"] for q in coverage if q["team_id"] in relevant_teams)
                    status = _status(ready=ready, admitted=totals[0], rejected=rejected,
                                     missing_games=missing, risky_games=team_risky,
                                     league_risky_games=risky, league_identity_conflicts=conflicts,
                                     league_rejected_rows=league_rejected)
                    usable = status in {"valid_observed", "zero_observed"} and end is not None and end <= label_cutoff
                    observed_values = totals if totals[0] or status == "zero_observed" else (None,) * len(metrics)
                    target_values = totals if usable else (None,) * len(metrics)
                    values = dict(zip(metrics, totals))
                    denominator = values["pa" if role == "hitter" else "bf"]
                    ratio_status = "not_requested" if horizon == "7d" else (
                        "unavailable" if not usable else "zero_denominator" if not values["ab"] or not denominator else "valid"
                    )
                    hit_rate = values["hits"] / values["ab"] if usable and horizon != "7d" and values["ab"] else None
                    so_rate = values["strikeouts"] / denominator if usable and horizon != "7d" and denominator else None
                    target = (sample_id, horizon, anchor_text, end.isoformat() if end else None,
                              observed_end.isoformat(), end.isoformat() if ready and end else None, status, int(usable),
                              "observed_pbp_known_or_observed_teams_not_official_roster", len(selected), missing, risky,
                              team_risky, conflicts, league_rejected, team_rejected, rejected, ratio_status,
                              *observed_values, *target_values, hit_rate, so_rate)
                    db.execute(f"INSERT INTO {model}_targets VALUES ({','.join('?' for _ in target)})", target)
            anchor += timedelta(days=7)