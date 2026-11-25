"""경기·타석·상황·실제 투구의 시점 제한 입력과 정답을 구성한다."""

import sqlite3


# 날짜별 집계로 당일 경기 결과(더블헤더 포함)를 모두 제외한다.
CONTEXT_SQL = """
CREATE TEMP TABLE team_daily AS
WITH sides AS (
 SELECT season,date,home_team_id team,competition,home_score rf,away_score ra FROM source.eligible_games
 UNION ALL
 SELECT season,date,away_team_id,competition,away_score,home_score FROM source.eligible_games
)
SELECT team,season,date,SUM(competition='regular') n,
 SUM(CASE WHEN competition='regular' THEN rf ELSE 0 END) rf,
 SUM(CASE WHEN competition='regular' THEN ra ELSE 0 END) ra,
 SUM(competition='regular' AND rf>ra) wins,
 SUM(competition='regular' AND rf=ra) ties
FROM sides GROUP BY team,season,date;
CREATE TEMP TABLE team_prior AS
SELECT team,season,date,
 COALESCE(SUM(n) OVER season_history,0) season_games,
 SUM(rf) OVER season_history season_rf,SUM(ra) OVER season_history season_ra,
 SUM(wins) OVER season_history season_wins,SUM(ties) OVER season_history season_ties,
 COALESCE(SUM(n) OVER recent,0) recent_games,
 SUM(rf) OVER recent recent_rf,SUM(ra) OVER recent recent_ra,
 SUM(wins) OVER recent recent_wins,
 MAX(CASE WHEN n>0 THEN date END) OVER season_history last_date
FROM team_daily
WINDOW season_history AS (PARTITION BY team,season ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
 recent AS (PARTITION BY team,season ORDER BY julianday(date) RANGE BETWEEN 28 PRECEDING AND 1 PRECEDING);
CREATE UNIQUE INDEX temp.team_prior_key ON team_prior(team,season,date);

CREATE TEMP TABLE game_context AS
SELECT g.game_id,g.date,g.season,g.competition,g.home_team_id,g.away_team_id,
 CASE WHEN g.date<c.validation_start THEN 'train' WHEN g.date<c.test_start THEN 'validation' ELSE 'test' END split,
 h.season_games home_season_games,h.season_rf home_season_rf,h.season_ra home_season_ra,
 h.season_wins home_season_wins,h.season_ties home_season_ties,
 h.recent_games home_recent_games,h.recent_rf home_recent_rf,h.recent_ra home_recent_ra,
 h.recent_wins home_recent_wins,CAST(julianday(g.date)-julianday(h.last_date) AS INTEGER) home_days_since_game,
 a.season_games away_season_games,a.season_rf away_season_rf,a.season_ra away_season_ra,
 a.season_wins away_season_wins,a.season_ties away_season_ties,
 a.recent_games away_recent_games,a.recent_rf away_recent_rf,a.recent_ra away_recent_ra,
 a.recent_wins away_recent_wins,CAST(julianday(g.date)-julianday(a.last_date) AS INTEGER) away_days_since_game
FROM source.eligible_games g CROSS JOIN config c
JOIN team_prior h ON h.team=g.home_team_id AND h.season=g.season AND h.date=g.date
JOIN team_prior a ON a.team=g.away_team_id AND a.season=g.season AND a.date=g.date;
CREATE UNIQUE INDEX temp.game_context_key ON game_context(game_id);

CREATE TEMP TABLE player_prior AS
WITH daily AS (
 SELECT p.player_id,p.role,g.date,
 SUM(a.admitted=1) n,SUM(a.admitted=0) rejected,
 SUM(CASE WHEN a.admitted=1 THEN COALESCE(p.observed_pa,p.observed_bf) END) opportunities,
 SUM(CASE WHEN a.admitted=1 THEN p.observed_hits END) hits,
 SUM(CASE WHEN a.admitted=1 THEN p.observed_ab END) ab,
 SUM(CASE WHEN a.admitted=1 THEN p.observed_strikeouts END) so
 FROM source.player_games p JOIN source.player_admission a USING(game_id,team_id,player_id,role)
 JOIN source.eligible_games g USING(game_id) WHERE g.competition='regular'
 GROUP BY p.player_id,p.role,g.date
)
SELECT player_id,role,date,
 COALESCE(SUM(n) OVER history,0) prior_games,
 COALESCE(SUM(rejected) OVER history,0) prior_rejected,
 SUM(opportunities) OVER history prior_opportunities,SUM(hits) OVER history prior_hits,
 SUM(ab) OVER history prior_ab,SUM(so) OVER history prior_so,
 COALESCE(SUM(n) OVER recent,0) recent_games,SUM(opportunities) OVER recent recent_opportunities,
 SUM(hits) OVER recent recent_hits
FROM daily
WINDOW history AS (PARTITION BY player_id,role ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
 recent AS (PARTITION BY player_id,role ORDER BY julianday(date) RANGE BETWEEN 28 PRECEDING AND 1 PRECEDING);
CREATE UNIQUE INDEX temp.player_prior_key ON player_prior(player_id,role,date);
"""

TEAM_FEATURES = (
    "season_games", "season_rf", "season_ra", "season_wins", "season_ties",
    "recent_games", "recent_rf", "recent_ra", "recent_wins", "days_since_game",
)
PLAYER_FEATURES = (
    "prior_games", "prior_rejected", "prior_opportunities", "prior_hits", "prior_ab",
    "prior_so", "recent_games", "recent_opportunities", "recent_hits",
)
GAME_FEATURES = ("season", "competition", "home_team_id", "away_team_id") + tuple(
    f"{side}_{field}" for side in ("home", "away") for field in TEAM_FEATURES
)
HISTORY_FEATURES = tuple(
    f"{role}_{field}" for role in ("batter", "pitcher") for field in PLAYER_FEATURES
)


def _select(alias: str, names: tuple[str, ...]) -> str:
    return ",".join(f"{alias}.{name}" for name in names)


def _player_select() -> str:
    return ",".join(
        f"{alias}.{field} AS {role}_{field}"
        for alias, role in (("b", "batter"), ("p", "pitcher"))
        for field in PLAYER_FEATURES
    )


def build_events(db: sqlite3.Connection) -> None:
    """검증 후보를 입력·정답·인덱스로 분리하고 현재 사건의 사후 값을 차단한다."""
    db.executescript(CONTEXT_SQL)
    game = _select("g", GAME_FEATURES)
    history = _player_select()
    db.executescript(f"""
    CREATE TABLE m1_index AS SELECT g.game_id sample_id,g.game_id,g.date sample_date,g.split,
        'eligible' status,s.source_sha,s.source_row
      FROM game_context g JOIN source.eligible_games s USING(game_id) ORDER BY g.game_id;
    CREATE TABLE m1_inputs AS SELECT g.game_id sample_id,{game} FROM game_context g ORDER BY g.game_id;
    CREATE TABLE m1_targets AS SELECT game_id sample_id,home_score target_home_runs,
       away_score target_away_runs,winner target_result FROM source.eligible_games ORDER BY game_id;

    CREATE TABLE m2_index AS SELECT a.game_id||':'||a.pa_number sample_id,a.game_id,g.date sample_date,g.split,
      'eligible' status,a.pa_number,a.source_sha,a.start_source_row,a.end_source_row
      FROM source.eligible_pa a JOIN game_context g USING(game_id) ORDER BY a.game_id,a.pa_number;
    CREATE TABLE m2_inputs AS SELECT a.game_id||':'||a.pa_number sample_id,{game},a.inning,a.half,
      a.start_batter_id batter_id,a.start_pitcher_id pitcher_id,a.pre_home_score home_score,
      a.pre_away_score away_score,a.pre_outs outs,(a.pre_on_1b IS NOT NULL) on_first,
      (a.pre_on_2b IS NOT NULL) on_second,(a.pre_on_3b IS NOT NULL) on_third,{history}
      FROM source.eligible_pa a JOIN game_context g USING(game_id)
      LEFT JOIN player_prior b ON b.player_id=a.start_batter_id AND b.role='hitter' AND b.date=g.date
      LEFT JOIN player_prior p ON p.player_id=a.start_pitcher_id AND p.role='pitcher' AND p.date=g.date
      ORDER BY a.game_id,a.pa_number;
    CREATE TABLE m2_targets AS SELECT game_id||':'||pa_number sample_id,event target_event
      FROM source.eligible_pa ORDER BY game_id,pa_number;

    -- 같은 반이닝 전체 경계가 검증되어야 잔여 득점이 확정되는 후보가 된다.
    CREATE TEMP TABLE half_ends AS
    WITH ranked AS (
      SELECT a.*,ROW_NUMBER() OVER(PARTITION BY a.game_id,a.inning,a.half ORDER BY a.pa_number DESC) last_rank,
       MIN(CASE WHEN e.game_id IS NOT NULL THEN 1 ELSE 0 END)
         OVER(PARTITION BY a.game_id,a.inning,a.half) full_half,
       MAX(a.pa_number) OVER(PARTITION BY a.game_id) last_game_pa
      FROM source.plate_appearances a LEFT JOIN source.eligible_states e USING(game_id,pa_number)
    )
    SELECT r.game_id,r.inning,r.half,r.pa_number last_pa,
      CASE WHEN r.post_outs=3 THEN 'three_outs' ELSE c.terminal_state END terminal_evidence,
      CASE WHEN r.half='top' THEN r.post_away_score ELSE r.post_home_score END end_score
    FROM ranked r JOIN source.game_checks c USING(game_id) JOIN source.eligible_games f USING(game_id)
    WHERE r.last_rank=1 AND r.full_half=1 AND c.final_score_match=1 AND c.order_valid=1
      AND c.boundary_valid=1 AND c.terminal_state<>'terminal_reason_unresolved'
      AND (r.post_outs=3 OR (r.pa_number=r.last_game_pa AND r.half='bot'
        AND c.terminal_state='home_leads_partial_bottom_candidate' AND f.winner='home'
        AND r.post_home_score=f.home_score AND r.post_away_score=f.away_score));
    CREATE UNIQUE INDEX temp.half_key ON half_ends(game_id,inning,half);
    CREATE TEMP TABLE boundaries(phase TEXT PRIMARY KEY);
    INSERT INTO boundaries VALUES ('pre'),('post');
    CREATE TABLE m4_index AS SELECT a.game_id||':'||a.pa_number||':'||b.phase sample_id,
      a.game_id,g.date sample_date,g.split,CASE WHEN h.game_id IS NULL THEN 'excluded_half' ELSE 'eligible' END status,
      a.game_id||':'||a.pa_number pair_id,a.pa_number,b.phase,h.terminal_evidence,a.source_sha,a.start_source_row,a.end_source_row
      FROM source.eligible_states a JOIN game_context g USING(game_id) CROSS JOIN boundaries b
      LEFT JOIN half_ends h USING(game_id,inning,half) ORDER BY a.game_id,a.pa_number,b.phase;
    CREATE TABLE m4_inputs AS SELECT i.sample_id,{game},a.inning,a.half,i.phase,
      CASE WHEN a.half='top' THEN g.away_team_id ELSE g.home_team_id END perspective_team_id,
      CASE WHEN i.phase='pre' THEN a.pre_home_score ELSE a.post_home_score END home_score,
      CASE WHEN i.phase='pre' THEN a.pre_away_score ELSE a.post_away_score END away_score,
      CASE WHEN i.phase='pre' THEN a.pre_outs ELSE a.post_outs END outs,
      CASE WHEN i.phase='pre' THEN a.pre_on_1b IS NOT NULL ELSE a.post_on_1b IS NOT NULL END on_first,
      CASE WHEN i.phase='pre' THEN a.pre_on_2b IS NOT NULL ELSE a.post_on_2b IS NOT NULL END on_second,
      CASE WHEN i.phase='pre' THEN a.pre_on_3b IS NOT NULL ELSE a.post_on_3b IS NOT NULL END on_third
      FROM m4_index i JOIN source.eligible_states a USING(game_id,pa_number)
      JOIN game_context g USING(game_id) WHERE i.status='eligible' ORDER BY i.sample_id;
    CREATE TABLE m4_targets AS SELECT i.sample_id,
      CASE WHEN f.winner='tie' THEN 'tie' WHEN (a.half='top' AND f.winner='away')
       OR (a.half='bot' AND f.winner='home') THEN 'win' ELSE 'loss' END target_result,
      h.end_score-CASE WHEN a.half='top' THEN
        CASE WHEN i.phase='pre' THEN a.pre_away_score ELSE a.post_away_score END
        ELSE CASE WHEN i.phase='pre' THEN a.pre_home_score ELSE a.post_home_score END END target_remaining_runs
      FROM m4_index i JOIN source.eligible_states a USING(game_id,pa_number)
      JOIN source.eligible_games f USING(game_id) JOIN half_ends h USING(game_id,inning,half)
      WHERE i.status='eligible' ORDER BY i.sample_id;

    CREATE TABLE m5_index AS SELECT q.game_id||':'||q.pa_number||':'||q.pitch_number sample_id,q.game_id,
      g.date sample_date,g.split,'eligible' status,q.pa_number,q.pitch_number,q.source_sha,q.source_row
      FROM source.eligible_pitches q JOIN game_context g USING(game_id) ORDER BY q.game_id,q.pa_number,q.pitch_number;
    CREATE TABLE m5_inputs AS SELECT q.game_id||':'||q.pa_number||':'||q.pitch_number sample_id,{game},a.inning,a.half,
      q.batter_id,q.pitcher_id,q.pitch_number,q.balls,q.strikes,q.outs_when_up outs,q.home_score,q.away_score,
      (q.on_1b IS NOT NULL) on_first,(q.on_2b IS NOT NULL) on_second,(q.on_3b IS NOT NULL) on_third,
      CASE WHEN prev.target_valid=1 THEN prev.pitch_type END previous_pitch_type,
      CASE WHEN prev.target_valid=1 THEN prev.pitch_result END previous_pitch_result,{history}
      FROM source.eligible_pitches q JOIN source.eligible_states a USING(game_id,pa_number)
      JOIN game_context g USING(game_id)
      LEFT JOIN source.pitches prev ON prev.game_id=q.game_id AND prev.pa_number=q.pa_number
        AND prev.pitch_number=q.pitch_number-1 AND prev.order_valid=1 AND prev.source_sha=q.source_sha AND prev.source_row<q.source_row
      LEFT JOIN player_prior b ON b.player_id=q.batter_id AND b.role='hitter' AND b.date=g.date
      LEFT JOIN player_prior p ON p.player_id=q.pitcher_id AND p.role='pitcher' AND p.date=g.date
      ORDER BY q.game_id,q.pa_number,q.pitch_number;
    CREATE TABLE m5_targets AS SELECT game_id||':'||pa_number||':'||pitch_number sample_id,
      pitch_type target_pitch_type,pitch_result target_pitch_result,
      pitch_type||':'||pitch_result target_joint FROM source.eligible_pitches ORDER BY game_id,pa_number,pitch_number;
    """)


def build_support(db: sqlite3.Connection) -> None:
    """시즌 전망·예상 선발의 사후 정답 근거와 과거 이력 자료를 보존한다."""
    db.executescript("""
    CREATE TABLE season_coverage AS SELECT season,competition,MIN(date) first_date,MAX(date) last_date,
      COUNT(*) final_games,MAX(season_complete) completion_evidence FROM source.eligible_games
      GROUP BY season,competition ORDER BY season,competition;
    CREATE TABLE season_series_evidence AS SELECT * FROM source.postseason_series ORDER BY season,stage,source_sha;
    CREATE TABLE season_schedule_evidence AS SELECT * FROM source.postseason_schedule ORDER BY date,source_sha,source_row;
    CREATE TABLE schedule_reference_evidence AS SELECT * FROM source.schedule_reference ORDER BY date,source_sha,source_row;
    CREATE TABLE starter_evidence AS SELECT s.*,g.date FROM source.starter_candidates s
      JOIN source.eligible_games g USING(game_id) ORDER BY s.game_id,s.side,s.source_sha,s.source_row,s.method;
    CREATE TABLE starter_actual AS SELECT game_id,side,team_id,date,MIN(name) name,
      CASE WHEN COUNT(DISTINCT player_id)=1 THEN MAX(player_id) END player_id,COUNT(*) evidence_count
      FROM starter_evidence GROUP BY game_id,side,team_id,date HAVING COUNT(DISTINCT name)=1
      ORDER BY game_id,side;
    CREATE TABLE starter_history AS SELECT game_id,side,team_id,date,player_id,name,evidence_count
      FROM starter_actual WHERE player_id IS NOT NULL ORDER BY date,game_id,side;
    CREATE TABLE pitching_actual_evidence AS SELECT p.*,g.date FROM source.direct_pitching_candidates p
      JOIN source.eligible_games g USING(game_id) ORDER BY p.game_id,p.side,p.source_sha,p.source_row;
    CREATE TEMP TABLE pitcher_history AS SELECT p.player_id,g.date,p.observed_pitches
      FROM source.eligible_player_games p JOIN source.eligible_games g USING(game_id)
      WHERE p.role='pitcher' AND g.competition='regular';
    CREATE INDEX temp.pitcher_history_key ON pitcher_history(player_id,date);
    CREATE TABLE starter_pre_game AS
    WITH teams AS (
      SELECT game_id,date,home_team_id team_id FROM game_context
      UNION ALL SELECT game_id,date,away_team_id FROM game_context
    ), candidates AS (
    SELECT t.game_id,t.date,t.team_id,h.player_id,COUNT(*) starts_last_28d,
      MAX(h.date) previous_start_date,CAST(julianday(t.date)-julianday(MAX(h.date)) AS INTEGER) days_since_start
    FROM teams t JOIN starter_history h ON h.team_id=t.team_id AND h.date<t.date
      AND h.date>=date(t.date,'-28 days') GROUP BY t.game_id,t.date,t.team_id,h.player_id
    )
    SELECT c.*,MAX(p.date) previous_appearance_date,CAST(julianday(c.date)-julianday(MAX(p.date)) AS INTEGER) days_since_appearance,
      SUM(CASE WHEN p.date>=date(c.date,'-7 days') THEN p.observed_pitches END) observed_pitches_last_7d
    FROM candidates c LEFT JOIN pitcher_history p ON p.player_id=c.player_id AND p.date<c.date
    GROUP BY c.game_id,c.date,c.team_id,c.player_id ORDER BY c.game_id,c.team_id,c.player_id;
    """)