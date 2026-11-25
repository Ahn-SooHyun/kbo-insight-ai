"""최종 품질 판정과 실제 변환 함수의 회귀 테스트."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.datasets.published_sources import SourceVersion
from src.datasets.standardize import _convert_pbp_game
from src.validation.standard_data import StandardDataError, StandardStore
from src.validation.standard_quality import (
    AUDIT_SCHEMA,
    _account_rows,
    _audit_postseason,
    _verify_database,
    _verify_interim,
    apply_player_gate,
    audit_observed_counts,
    inspect_pa,
)
from .test_standard_data import make_pbp_row


class QualityTests(unittest.TestCase):

    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)

        self.store = StandardStore(
            self.root / "standard.sqlite3"
        )
        self.db = self.store.connection

        self.source = SourceVersion(
            {
                "target_id": "hf-pbp-2023",
                "season": 2023,
            },
            {
                "sha256": "a" * 64,
            },
            self.root / "raw",
        )

        self.store.put("games", {
            "game_id": "20230401HHWO0",
            "source_key": "20230401HHWO0",
            "season": 2023,
            "competition": "regular",
            "date": "2023-04-01",
            "away_team_id": 2010,
            "home_team_id": 2009,
            "away_score": 0,
            "home_score": 0,
            "status": "Final",
            "winner": "tie",
            "is_tie": 1,
            "result_valid": 1,
            "source_sha": "b" * 64,
            "source_row": 0,
        })

        self.db.executescript(AUDIT_SCHEMA)

    def tearDown(self) -> None:
        self.store.close()
        self.temp.cleanup()

    def _convert(
        self,
        rows: list[dict[str, object]],
    ) -> None:
        _convert_pbp_game(
            self.store,
            self.source,
            rows,
        )

        # 합성 원본의 각 논리적 타석을 감사 테이블에 연결한다.
        for number in sorted({
            row["at_bat_number"]
            for row in rows
        }):
            pa = [
                row
                for row in rows
                if row["at_bat_number"] == number
            ]

            start, order, state, runs, kind = inspect_pa(pa)

            self.db.execute(
                "INSERT INTO pa_audit VALUES (?,?,?,?,?,?,?,?)",
                (
                    "20230401HHWO0",
                    number,
                    start,
                    order,
                    state,
                    int(
                        len({
                            row["pitcher"]
                            for row in pa
                        }) > 1
                    ),
                    runs,
                    kind,
                ),
            )

    def test_repeated_events_and_non_pitch_count(self) -> None:
        rows = [
            make_pbp_row(
                _source_row=0,
                pitch_number=1,
            ),
            make_pbp_row(
                _source_row=1,
                pitch_number=2,
                strikes=1,
            ),
            make_pbp_row(
                _source_row=2,
                at_bat_number=2,
                pitch_number=0,
                outs_when_up=1,
                post_outs=1,
                events="walk",
                pitch_type=None,
                pitch_name=None,
                pitch_result=None,
                type=None,
                post_on_1b="63704",
            ),
        ]

        self._convert(rows)

        audit_observed_counts(self.db)

        self.assertEqual(
            self.db.execute(
                "SELECT COUNT(*) FROM pitches"
            ).fetchone()[0],
            2,
        )
        self.assertEqual(
            self.db.execute(
                "SELECT COUNT(*) FROM non_pitch"
            ).fetchone()[0],
            1,
        )
        self.assertEqual(
            self.db.execute(
                """
                SELECT observed_pa,observed_ab
                FROM player_games
                WHERE role='hitter'
                """
            ).fetchone()[:],
            (2, 1),
        )

    def test_participant_change_not_credited_automatically(self) -> None:
        self._convert([
            make_pbp_row(
                _source_row=0,
                pitch_number=1,
            ),
            make_pbp_row(
                _source_row=1,
                pitch_number=2,
                pitcher="12345",
            ),
        ])

        row = self.db.execute(
            "SELECT * FROM plate_appearances"
        ).fetchone()

        self.assertEqual(
            row["participant_changed"],
            1,
        )
        self.assertIsNone(
            row["credited_pitcher_candidate"]
        )
        self.assertEqual(
            self.db.execute(
                "SELECT COUNT(*) FROM eligible_pa"
            ).fetchone()[0],
            0,
        )

        audit_observed_counts(self.db)

    def test_first_pitch_missing_and_unlabeled_half(self) -> None:
        self.assertEqual(
            inspect_pa([
                make_pbp_row(
                    pitch_number=2,
                    balls=1,
                )
            ])[:2],
            (0, 0),
        )

        self.assertEqual(
            inspect_pa([
                make_pbp_row(
                    events=None,
                    outs_when_up=2,
                    post_outs=3,
                )
            ])[-1],
            "unlabeled_half_end_candidate",
        )

    def test_runs_discrepancy_not_validated(self) -> None:
        row = make_pbp_row(runs_scored=1)

        self.assertEqual(
            inspect_pa([row])[3],
            0,
        )

    def test_score_change_for_non_batting_team(self) -> None:
        row = make_pbp_row(
            post_home_score=1,
            runs_scored=1,
        )

        self.assertEqual(
            inspect_pa([row])[2],
            0,
        )

    def test_missing_raw_row_is_not_silent(self) -> None:
        self._convert([
            make_pbp_row(_source_row=0)
        ])

        self.assertEqual(
            _account_rows(
                self.db,
                self.source,
                1,
            )["mapped_rows"],
            1,
        )

        with self.assertRaises(StandardDataError):
            _account_rows(
                self.db,
                self.source,
                2,
            )

    def test_recomputed_player_totals_detect_error(self) -> None:
        self._convert([
            make_pbp_row(_source_row=0)
        ])

        audit_observed_counts(self.db)

        self.db.execute(
            "UPDATE player_games SET observed_hits=999"
        )

        with self.assertRaises(StandardDataError):
            audit_observed_counts(self.db)

    def test_direct_and_team_disagreement_blocks_candidate(self) -> None:
        self._convert([
            make_pbp_row(_source_row=0)
        ])

        self.assertEqual(
            self.db.execute(
                """
                SELECT coverage_candidate
                FROM player_games
                WHERE role='pitcher'
                """
            ).fetchone()[0],
            1,
        )

        self.store.put("pitching_records", {
            "source_sha": "c" * 64,
            "source_row": 0,
            "game_id": "20230401HHWO0",
            "side": "home",
            "team_id": 2009,
            "name": "투수",
            "raw_player_id": "68341",
            "hf_candidate_id": "hf:68341",
            "is_starter": 1,
            "valid": 1,
        })

        self.store.comparison(
            "c" * 64,
            0,
            "player:hits",
            "20230401HHWO0",
            1,
            0,
        )

        self.store.comparison(
            "c" * 64,
            0,
            "team:home:hits",
            "20230401HHWO0",
            1,
            0,
        )

        apply_player_gate(self.db)

        gate = self.db.execute(
            """
            SELECT *
            FROM player_admission
            WHERE role='pitcher'
            """
        ).fetchone()

        self.assertEqual(
            gate["preliminary_candidate"],
            1,
        )
        self.assertEqual(
            gate["direct_mismatches"],
            1,
        )
        self.assertEqual(
            gate["team_mismatches"],
            1,
        )
        self.assertEqual(
            gate["admitted"],
            0,
        )

        self.assertEqual(
            self.db.execute(
                """
                SELECT coverage_candidate
                FROM player_games
                WHERE role='pitcher'
                """
            ).fetchone()[0],
            0,
        )

    def test_completed_teams_preserve_prior_result(self) -> None:
        previous = self.store.summary()

        _verify_database(
            self.db,
            previous,
        )

        self.db.execute(
            "UPDATE games SET home_score=1"
        )

        with self.assertRaises(StandardDataError):
            _verify_database(
                self.db,
                previous,
            )

    def test_interim_must_equal_raw_content(self) -> None:
        raw = self.root / "raw.json"
        intermediate = self.root / "intermediate.jsonl"

        raw.write_text(
            json.dumps([{"id": "a"}]),
            encoding="utf-8",
        )

        intermediate.write_text(
            '{"id":"b","_source_row":0}\n',
            encoding="utf-8",
        )

        source = SourceVersion(
            {
                "target_id": "test",
                "filename": "test.json",
                "records_path": [],
            },
            {
                "sha256": "d" * 64,
            },
            raw,
        )

        with self.assertRaises(StandardDataError):
            _verify_interim(
                source,
                intermediate,
                1,
            )

    def test_postseason_unassigned_team_remains_schedule(self) -> None:
        raw = self.root / "postseason.json"

        stages = (
            "wildCard",
            "semiPlayoff",
            "playoff",
            "koreanSeries",
        )

        record = {
            "game": 1,
            "date": "2026-11-01",
            "away": "Playoff Winner",
            "home": "KT Wiz",
        }

        value = {
            "2026": {
                "isComplete": False,
                **{
                    stage: {
                        "isComplete": False,
                        "games": (
                            [record]
                            if stage == "koreanSeries"
                            else []
                        ),
                    }
                    for stage in stages
                },
            }
        }

        raw.write_text(
            json.dumps(value),
            encoding="utf-8",
        )

        intermediate = self.root / "postseason.jsonl"

        intermediate.write_text(
            json.dumps({
                **record,
                "_source_row": 0,
                "_stage": "koreanSeries",
            }) + "\n",
            encoding="utf-8",
        )

        source = SourceVersion(
            {
                "target_id": "dent-postseason-2026-wildcard-probe",
            },
            {
                "sha256": "e" * 64,
            },
            raw,
        )

        _audit_postseason(
            self.db,
            source,
            intermediate,
        )

        self.assertEqual(
            self.db.execute(
                "SELECT COUNT(*) FROM postseason_series"
            ).fetchone()[0],
            4,
        )

        row = self.db.execute(
            "SELECT * FROM postseason_schedule"
        ).fetchone()

        self.assertEqual(
            row["final_observed"],
            0,
        )
        self.assertIsNone(
            row["linked_game_id"]
        )


if __name__ == "__main__":
    unittest.main()