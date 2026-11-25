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


def _summarize(rows: list[sqlite3.Row], counts: tuple[str, ...]) -> tuple[int, ...]:
    """채택된 관측 집계만 합산한다. 이닝은 아웃 정수로 유지한다."""
    admitted = [row for row in rows if row["admitted"] == 1]
    return (len({row["game_id"] for row in admitted}),) + tuple(
        sum(int(row[f"observed_{field}"]) for row in admitted) for field in counts
    )


def _status(
    *, ready: bool, admitted: int, rejected: int, missing_games: int,
    risky_games: int, league_risky_games: int, league_identity_conflicts: int,
) -> str:
    """누락·부분·미완료와 관측 0을 분리한다. 공식 출전 여부를 추정하지 않는다."""
    if not ready:
        return "partial_unfinished" if admitted else "unobserved"
    if rejected or missing_games or risky_games:
        return "partial" if admitted else "missing"
    if admitted:
        return "valid_observed"
    if league_risky_games or league_identity_conflicts:
        return "unobserved"
    return "zero_observed"


def build_players(db: sqlite3.Connection, observation_end: str) -> None:
    """시즌 내 과거 관측 선수의 주별 입력과 3개 기간 정답·관측 상태를 생성한다."""
    db.row_factory = sqlite3.Row
    config = db.execute("SELECT * FROM config").fetchone()
    validation_start = date.fromisoformat(config["validation_start"])
    test_start = date.fromisoformat(config["test_start"])
    cutoff = date.fromisoformat(observation_end)
    db.execute("""CREATE TABLE m3_window_games(season INTEGER,anchor_date TEXT,horizon TEXT,game_id TEXT,
                PRIMARY KEY(season,anchor_date,horizon,game_id))""")
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
          league_identity_conflicts INTEGER,player_rejected INTEGER,ratio_status TEXT,
          {observed},{targets},target_hit_rate REAL,target_strikeout_rate REAL,
          PRIMARY KEY(sample_id,horizon));
        """)

    seasons = db.execute("""
      SELECT g.season,MIN(g.date) first_date,MAX(g.date) last_date,MAX(g.season_complete) complete
      FROM source.eligible_games g WHERE g.competition='regular' AND g.season IN
        (SELECT DISTINCT x.season FROM source.eligible_games x JOIN source.game_checks c USING(game_id))
      GROUP BY g.season ORDER BY g.season
    """).fetchall()
    for season_row in seasons:
        season = season_row["season"]
        first = date.fromisoformat(season_row["first_date"])
        last = date.fromisoformat(season_row["last_date"])
        season_end = last + DAY if season_row["complete"] == 1 else None
        # 경기 단위 누락과 참가자 식별 불확실성을 별도로 수집한다.
        games = db.execute("""
          SELECT g.game_id,g.date,g.home_team_id,g.away_team_id,
           CASE WHEN c.game_id IS NULL OR c.final_score_match<>1 THEN 1 ELSE 0 END missing,
           CASE WHEN c.game_id IS NULL OR c.order_valid<>1 OR c.boundary_valid<>1
             OR c.terminal_state='terminal_reason_unresolved' THEN 1 ELSE 0 END risky,
           COALESCE(a.identity_conflicts,0) identity_conflicts
          FROM source.eligible_games g LEFT JOIN source.game_checks c USING(game_id)
          LEFT JOIN (SELECT game_id,SUM(identity_ambiguous) identity_conflicts
                     FROM source.player_admission GROUP BY game_id) a USING(game_id)
          WHERE g.competition='regular' AND g.season=? ORDER BY g.date,g.game_id
        """, (season,)).fetchall()
        histories: dict[tuple[str, str], list[sqlite3.Row]] = defaultdict(list)
        for row in db.execute("""
          SELECT p.*,g.date,a.admitted,a.identity_ambiguous FROM source.player_games p
          JOIN source.player_admission a USING(game_id,team_id,player_id,role)
          JOIN source.eligible_games g USING(game_id) WHERE g.competition='regular' AND g.season=?
          ORDER BY g.date,p.game_id,p.team_id,p.player_id,p.role
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
                past = [row for row in rows if row["date"] < anchor_text and not row["identity_ambiguous"]]
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
                past_counts = _summarize(past, counts)
                recent_counts = _summarize(recent, counts)
                rejected_count = sum(row["admitted"] == 0 for row in past)
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
                    missing = sum(game["missing"] for game in selected)
                    risky = sum(game["risky"] for game in selected)
                    team_risky = sum(game["risky"] and (game["home_team_id"] in relevant_teams
                                    or game["away_team_id"] in relevant_teams) for game in selected)
                    conflicts = sum(game["identity_conflicts"] for game in selected)
                    status = _status(ready=ready, admitted=totals[0], rejected=rejected,
                                     missing_games=missing, risky_games=team_risky,
                                     league_risky_games=risky, league_identity_conflicts=conflicts)
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
                              "observed_pbp_participants_not_official_roster", len(selected), missing, risky,
                              team_risky, conflicts, rejected, ratio_status,
                              *observed_values, *target_values, hit_rate, so_rate)
                    db.execute(f"INSERT INTO {model}_targets VALUES ({','.join('?' for _ in target)})", target)
            anchor += timedelta(days=7)