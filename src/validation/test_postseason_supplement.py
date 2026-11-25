"""포스트시즌 판본·종료 근거·원본 중복 참조를 검증한다."""

from copy import deepcopy
import sqlite3
import unittest

from src.datasets.model_data import verify_version_inventory
from src.datasets.postseason_supplement import (
    flatten_postseason,
    regular_completion_evidence,
)


def series_fixture() -> dict[str, object]:
    """실제 승수와 경기별 종료 근거를 대조하기 위한 최소 자료."""
    return {
        "season": 2024,
        "games": [
            {"gamePk": "regular-game-not-for-postseason"}
        ],
        "postseason": {
            "wildCard": {
                "isComplete": True,
                "winnerId": 2002,
            },
            "koreanSeries": {
                "name": "Korean Series",
                "isComplete": True,
                "championId": 2001,
                "team1": {
                    "id": 2001,
                    "wins": 1,
                    "isWinner": True,
                },
                "adv": {
                    "id": 2008,
                    "wins": 0,
                    "isWinner": False,
                },
                "games": [
                    {
                        "game": 1,
                        "date": "2024-10-21",
                        "away": "Samsung Lions",
                        "home": "KIA Tigers",
                        "awayScore": 1,
                        "homeScore": 5,
                        "winnerId": 2001,
                    }
                ],
            },
        },
    }


class PostseasonSupplementTests(unittest.TestCase):
    def test_only_existing_postseason_arrays_are_expanded(self) -> None:
        raw = series_fixture()

        rows, series = flatten_postseason(raw, 2024)

        self.assertEqual(len(rows), 1)
        self.assertEqual(len(series), 2)

        self.assertEqual(
            rows[0]["_stage"],
            "koreanSeries",
        )

        self.assertEqual(
            rows[0]["_completion_basis"],
            "completed_series_score_winner",
        )

        self.assertNotIn("isFinal", rows[0])
        self.assertIsNone(rows[0]["_season_complete"])

        self.assertFalse(
            series[0]["metadata"]["_games_array_present"]
        )

        self.assertEqual(
            series[0]["score_rows"],
            0,
        )

        self.assertFalse(
            series[1]["all_rounds_score_observed"]
        )

        self.assertNotIn(
            "_source_row",
            raw["postseason"]["koreanSeries"]["games"][0],
        )

    def test_explicit_final_marker(self) -> None:
        raw = series_fixture()

        raw["postseason"]["koreanSeries"]["games"][0][
            "isFinal"
        ] = True

        rows, _ = flatten_postseason(raw, 2024)

        self.assertEqual(
            rows[0]["_completion_basis"],
            "explicit_is_final",
        )

    def test_absent_final_marker_in_incomplete_series(self) -> None:
        raw = series_fixture()

        raw["postseason"]["koreanSeries"][
            "isComplete"
        ] = False

        rows, _ = flatten_postseason(raw, 2024)

        self.assertEqual(
            rows[0]["_completion_basis"],
            "not_final",
        )

    def test_explicit_nonfinal_is_not_inferred(self) -> None:
        raw = series_fixture()

        series = raw["postseason"]["koreanSeries"]
        series["isComplete"] = False
        series["games"][0]["isFinal"] = False

        rows, _ = flatten_postseason(raw, 2024)

        self.assertEqual(
            rows[0]["_completion_basis"],
            "not_final",
        )

    def test_invalid_score_winner_is_rejected(self) -> None:
        raw = series_fixture()

        raw["postseason"]["koreanSeries"]["games"][0][
            "winnerId"
        ] = 2008

        with self.assertRaises(ValueError):
            flatten_postseason(raw, 2024)

    def test_duplicate_series_game_is_rejected(self) -> None:
        raw = series_fixture()

        games = raw["postseason"]["koreanSeries"]["games"]
        games.append(deepcopy(games[0]))

        with self.assertRaises(ValueError):
            flatten_postseason(raw, 2024)

    def test_series_score_mismatch_is_rejected(self) -> None:
        raw = series_fixture()

        raw["postseason"]["koreanSeries"]["team1"][
            "wins"
        ] = 2

        with self.assertRaises(ValueError):
            flatten_postseason(raw, 2024)

    def test_wrong_season_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            flatten_postseason(series_fixture(), 2025)

    def test_shared_raw_hash_keeps_distinct_logical_targets(self) -> None:
        records = [
            {
                "target_id": "dent-results-2024",
                "source_sha": "a" * 64,
                "selected": True,
            },
            {
                "target_id": "dent-postseason-2024",
                "source_sha": "a" * 64,
                "selected": True,
            },
        ]

        verify_version_inventory(
            records,
            object_count=1,
            logical_targets=2,
        )

        with self.assertRaises(ValueError):
            verify_version_inventory(
                records,
                object_count=2,
                logical_targets=2,
            )

    def test_duplicate_version_is_rejected(self) -> None:
        record = {
            "target_id": "dent-results-2024",
            "source_sha": "a" * 64,
            "selected": True,
        }

        with self.assertRaises(ValueError):
            verify_version_inventory(
                [record, record],
                object_count=1,
                logical_targets=1,
            )

    def test_missing_selected_version_is_rejected(self) -> None:
        records = [
            {
                "target_id": "old-target",
                "source_sha": "a" * 64,
                "selected": False,
            },
            {
                "target_id": "new-target",
                "source_sha": "b" * 64,
                "selected": True,
            },
        ]

        with self.assertRaises(ValueError):
            verify_version_inventory(
                records,
                object_count=2,
                logical_targets=1,
            )

    def test_missing_pbp_games_rejects_completion_evidence(self) -> None:
        with sqlite3.connect(":memory:") as db:
            db.executescript(
                """
                CREATE TABLE games(
                    game_id TEXT,
                    date TEXT,
                    season INTEGER,
                    competition TEXT,
                    home_team_id INTEGER,
                    away_team_id INTEGER,
                    source_sha TEXT,
                    result_valid INTEGER,
                    status TEXT
                );

                CREATE TABLE game_checks(
                    game_id TEXT,
                    final_score_match INTEGER,
                    source_sha TEXT
                );

                INSERT INTO games VALUES (
                    'game-1',
                    '2023-10-17',
                    2023,
                    'regular',
                    2001,
                    2002,
                    'regular-hash',
                    1,
                    'Final'
                );
                """
            )

            with self.assertRaises(ValueError):
                regular_completion_evidence(db, 2023)


if __name__ == "__main__":
    unittest.main()