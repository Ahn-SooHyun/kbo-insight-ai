"""모델 자료의 날짜·타석/투구 순서·선수 기간·학습 EDA 격리를 검증한다."""

from datetime import date
import sqlite3
import unittest

from src.datasets.model_data import train_eda, validate
from src.datasets.model_events import build_events, build_support
from src.datasets.model_players import _status, build_players, monday_on_or_after
from src.validation.standard_data import SCHEMA
from src.validation.standard_quality import AUDIT_SCHEMA


class ModelDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db = sqlite3.connect(":memory:")
        self.addCleanup(self.db.close)
        self.db.row_factory = sqlite3.Row
        self.db.execute("ATTACH DATABASE ':memory:' AS source")
        # 실제 표준 스키마 계약과 같은 열·뷰로 작은 사례를 만든다.
        self.db.executescript(SCHEMA.replace("CREATE TABLE ", "CREATE TABLE source.").replace("CREATE INDEX ", "CREATE INDEX source."))
        self.db.executescript(AUDIT_SCHEMA.replace("CREATE TABLE ", "CREATE TABLE source.").replace("CREATE VIEW ", "CREATE VIEW source."))
        self.db.execute("CREATE TEMP TABLE config(validation_start TEXT,test_start TEXT,observation_end TEXT)")
        self.db.execute("INSERT INTO config VALUES ('2024-04-15','2024-04-29','2024-05-21')")

    def game(self, key: str, day: str, home: int, away: int, *, complete: int | None = 1, bad: bool = False) -> None:
        self.db.execute("INSERT INTO source.games VALUES (?,?,2024,'regular',NULL,?,NULL,0,2,1,?,?,'Final',?,?,1,?,'game',0)", (
            key, key, day, away, home, 'home' if home > away else 'away' if away > home else 'tie', int(home == away), complete,
        ))
        self.db.execute("INSERT INTO source.game_checks VALUES (?,'pbp',2,1,2,0,1,?,?,?)", (
            key, int(not bad), int(not bad), 'half_ended_three_outs',
        ))

    def player(self, key: str, player_id: str, team: int = 2, *, admitted: int = 1, role: str = 'hitter') -> None:
        self.db.execute("INSERT INTO source.player_games VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,0,0,?,NULL,'pbp')", (
            key, team, player_id, role, 4 if role == 'hitter' else None, 4, 1, 0, 0, 0, 1,
            4 if role == 'pitcher' else None, 3 if role == 'pitcher' else None, 10 if role == 'pitcher' else None, admitted,
        ))
        self.db.execute("INSERT INTO source.player_admission VALUES (?,?,?,?,?,0,0,0,0,?)", (key, team, player_id, role, admitted, admitted))

    def pa(self, key: str, *, event: str | None = 'single', changed: int = 0, label: str = 'observed_result') -> None:
        self.db.execute("""INSERT INTO source.plate_appearances VALUES (?,1,1,'top','b','b','p','p','b','p',?,?,?,1,2,1,
          0,0,0,NULL,NULL,NULL,0,1,3,NULL,NULL,NULL,'pbp',0,1)""", (key, changed, event, int(event is not None)))
        self.db.execute("INSERT INTO source.pa_audit VALUES (?,1,1,1,1,0,1,?)", (key, label))
        for number, pitch, result in ((1, 'FF', 'S'), (2, 'SL', 'H')):
            self.db.execute("""INSERT INTO source.pitches VALUES (?,1,?,'b','p',?,?,'S',150.0,0.1,2.0,
              0,?,0,0,0,NULL,NULL,NULL,1,1,1,'pbp',?)""", (key, number, pitch, result, number - 1, number - 1))

    def test_same_day_results_and_current_pitch_are_not_inputs(self) -> None:
        self.game('a','2024-04-01',3,1)
        self.game('b','2024-04-08',0,1)
        self.game('c','2024-04-08',10,1)
        self.game('d','2024-04-09',1,1)
        self.pa('b')
        self.player('a','b')
        self.player('b','b')
        self.player('b','p',team=1,role='pitcher')
        build_events(self.db)
        self.assertEqual([tuple(r) for r in self.db.execute("SELECT sample_id,home_season_games,home_season_rf FROM m1_inputs WHERE sample_id IN ('b','c','d') ORDER BY sample_id")], [('b',1,3),('c',1,3),('d',3,13)])
        self.assertEqual(tuple(self.db.execute("SELECT previous_pitch_type,previous_pitch_result,batter_prior_games FROM m5_inputs WHERE sample_id='b:1:2'").fetchone()), ('FF','S',1))
        self.assertIsNone(self.db.execute("SELECT previous_pitch_type FROM m5_inputs WHERE sample_id='b:1:1'").fetchone()[0])
        self.assertEqual(self.db.execute("SELECT target_joint FROM m5_targets WHERE sample_id='b:1:2'").fetchone()[0], 'SL:H')
        self.assertNotIn('release_speed_kmh', [r[1] for r in self.db.execute('PRAGMA table_info(m5_inputs)')])

    def test_m4_pair_residual_and_unlabeled_pa(self) -> None:
        self.game('a','2024-04-01',0,1)
        self.pa('a',event=None,label='unlabeled_half_end_candidate')
        build_events(self.db)
        self.assertEqual(self.db.execute('SELECT COUNT(*) FROM m2_targets').fetchone()[0],0)
        self.assertEqual([tuple(r) for r in self.db.execute('SELECT sample_id,target_result,target_remaining_runs FROM m4_targets ORDER BY sample_id')], [('a:1:post','win',0),('a:1:pre','win',1)])
        self.assertEqual(self.db.execute("SELECT perspective_team_id FROM m4_inputs WHERE sample_id='a:1:pre'").fetchone()[0],2)

    def test_m3_monday_population_zero_partial_and_cutoff(self) -> None:
        for key, day in [('a','2024-04-01'),('b','2024-04-08'),('c','2024-04-14'),('d','2024-04-15'),('e','2024-05-05')]:
            self.game(key,day,99 if key == 'e' else 77 if key == 'd' else 1,0)
        self.player('a','b')
        self.player('a','absent')
        self.player('a','rejected')
        self.player('b','b',team=1)
        self.player('c','b',team=1)
        self.player('d','b',team=1)
        self.player('b','future')
        self.player('b','rejected',admitted=0)
        build_events(self.db)
        build_players(self.db,'2024-05-21')
        build_support(self.db)
        counts = validate(self.db)
        self.assertIn('m3_hitter_target_status',counts)
        self.assertEqual(monday_on_or_after(date(2024,4,7)),date(2024,4,8))
        key = '2024:2024-04-08:b'
        self.assertEqual(tuple(self.db.execute("SELECT known_team_id,season_pa FROM m3_hitter_inputs WHERE sample_id=?", (key,)).fetchone()),(2,4))
        self.assertEqual(tuple(self.db.execute("SELECT target_pa,window_end,usable FROM m3_hitter_targets WHERE sample_id=? AND horizon='7d'",(key,)).fetchone()),(8,'2024-04-15',1))
        self.assertEqual(tuple(self.db.execute("SELECT target_pa,usable FROM m3_hitter_targets WHERE sample_id=? AND horizon='28d'",(key,)).fetchone()),(None,0))
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM m3_hitter_index WHERE player_id='future' AND sample_date='2024-04-08'").fetchone()[0],0)
        self.assertEqual(tuple(self.db.execute("SELECT status,target_pa FROM m3_hitter_targets WHERE sample_id='2024:2024-04-08:absent' AND horizon='7d'").fetchone()),('zero_observed',0))
        self.assertEqual(tuple(self.db.execute("SELECT status,target_pa FROM m3_hitter_targets WHERE sample_id='2024:2024-04-08:rejected' AND horizon='7d'").fetchone()),('missing',None))
        self.assertEqual(train_eda(self.db)['m1']['targets']['target_home_runs']['max'],1)

    def test_uncertain_season_and_identity_are_not_zero(self) -> None:
        self.game('a','2024-04-01',1,0,complete=None)
        self.game('b','2024-04-08',1,0,complete=None,bad=True)
        self.game('c','2024-05-05',1,0,complete=None)
        self.player('a','b')
        build_events(self.db)
        build_players(self.db,'2024-05-21')
        self.assertEqual(tuple(self.db.execute("SELECT status,window_end,usable FROM m3_hitter_targets WHERE sample_id='2024:2024-04-08:b' AND horizon='remaining'").fetchone()),('unobserved',None,0))
        self.assertEqual(_status(ready=True,admitted=0,rejected=0,missing_games=0,risky_games=0,league_risky_games=1,league_identity_conflicts=0),'unobserved')

    def test_m4_partial_bottom_tie_and_bad_half(self) -> None:
        self.game('a','2024-04-01',1,0)
        self.game('b','2024-04-02',0,0)
        self.game('c','2024-04-03',0,1,bad=True)
        for key in ('a','b','c'):
            self.pa(key)
        self.db.execute("UPDATE source.game_checks SET terminal_state='home_leads_partial_bottom_candidate' WHERE game_id='a'")
        self.db.execute("UPDATE source.plate_appearances SET half='bot',post_home_score=1,post_away_score=0,post_outs=1 WHERE game_id='a'")
        self.db.execute("UPDATE source.plate_appearances SET post_away_score=0 WHERE game_id='b'")
        build_events(self.db)
        self.assertEqual(tuple(self.db.execute("SELECT target_result,target_remaining_runs FROM m4_targets WHERE sample_id='a:1:pre'").fetchone()),('win',1))
        self.assertEqual(self.db.execute("SELECT target_result FROM m4_targets WHERE sample_id='b:1:post'").fetchone()[0],'tie')
        self.assertEqual(self.db.execute("SELECT status FROM m4_index WHERE sample_id='c:1:pre'").fetchone()[0],'excluded_half')

    def test_m3_zero_denominator_and_missing_source_game(self) -> None:
        self.db.execute("UPDATE config SET validation_start='2024-05-15',test_start='2024-05-19'")
        for key, day in [('a','2024-04-01'),('b','2024-04-08'),('c','2024-05-05')]:
            self.game(key,day,1,0)
        self.player('a','b')
        self.player('b','b')
        self.db.execute("UPDATE source.player_games SET observed_ab=0,observed_hits=0,observed_pa=0,observed_strikeouts=0 WHERE game_id='b'")
        build_events(self.db)
        build_players(self.db,'2024-05-21')
        self.assertEqual(tuple(self.db.execute("SELECT usable,ratio_status,target_hit_rate,target_strikeout_rate FROM m3_hitter_targets WHERE sample_id='2024:2024-04-08:b' AND horizon='28d'").fetchone()),(1,'zero_denominator',None,None))
        self.assertEqual(_status(ready=True,admitted=1,rejected=0,missing_games=1,risky_games=0,league_risky_games=0,league_identity_conflicts=0),'partial')


if __name__ == '__main__':
    unittest.main()