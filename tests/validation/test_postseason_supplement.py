"""포스트시즌 판본·종료 근거·원본 중복 참조를 검증한다."""

from copy import deepcopy
from contextlib import closing
import sqlite3
import unittest

from src.datasets.model_data import verify_version_inventory
from src.datasets.postseason_supplement import (
    flatten_postseason,
    normalize_season,
    regular_completion_evidence,
)


def series_fixture() -> dict[str, object]:
    """실제 Dent 원본과 동일한 문자열 시즌 구조의 최소 자료."""
    return {
        "season": "2024",
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
    def test_string_and_integer_season(self) -> None:
        """실제 문자열 시즌과 기존 정수 시즌을 동일하게 처리한다."""
        for season in ("2019", "2020", "2024", "2025"):
            with self.subTest(season=season):
                self.assertEqual(
                    normalize_season(season),
                    int(season),
                )

        raw = series_fixture()
        string_rows, string_series = flatten_postseason(raw, 2024)

        raw["season"] = 2024

        integer_rows, integer_series = flatten_postseason(raw, 2024)

        self.assertEqual(string_rows, integer_rows)
        self.assertEqual(string_series, integer_series)

    def test_invalid_season_types_are_rejected(self) -> None:
        """잘못된 입력 타입과 형식을 시즌으로 해석하지 않는다."""
        invalid = (
            None,
            True,
            False,
            2024.0,
            "2024.0",
            " 2024",
            "2024 ",
            "024",
            "abcd",
            "２０２４",
            [],
            {},
            0,
        )

        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    normalize_season(value)

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

        self.assertEqual(series[0]["score_rows"], 0)

        self.assertFalse(
            series[1]["all_rounds_score_observed"]
        )

        self.assertNotIn(
            "_source_row",
            raw["postseason"]["koreanSeries"]["games"][0],
        )

        self.assertEqual(raw["season"], "2024")

    def test_invalid_games_array_is_rejected(self) -> None:
        raw = series_fixture()

        raw["postseason"]["koreanSeries"]["games"] = None

        with self.assertRaises(ValueError):
            flatten_postseason(raw, 2024)

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

    def test_invalid_winner_type_is_rejected(self) -> None:
        raw = series_fixture()

        raw["postseason"]["koreanSeries"]["games"][0][
            "winnerId"
        ] = 2001.0

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
        raw = series_fixture()

        with self.assertRaises(ValueError):
            flatten_postseason(raw, 2025)

        raw["season"] = 2024

        with self.assertRaises(ValueError):
            flatten_postseason(raw, 2025)

    def test_wrong_game_date_season_is_rejected(self) -> None:
        raw = series_fixture()

        raw["postseason"]["koreanSeries"]["games"][0][
            "date"
        ] = "2025-10-21"

        with self.assertRaises(ValueError):
            flatten_postseason(raw, 2024)

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
        with closing(sqlite3.connect(":memory:")) as db:
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

    def test_complete_regular_season_evidence(self) -> None:
        """720경기 및 팀별 144경기와 PBP 집합 대조를 확인한다."""
        with closing(sqlite3.connect(":memory:")) as db:
            db.executescript(
                """
                CREATE TABLE games(
                    game_id TEXT PRIMARY KEY,
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
                    game_id TEXT PRIMARY KEY,
                    final_score_match INTEGER,
                    source_sha TEXT
                );
                """
            )

            games = []

            for home in range(2001, 2011):
                for away in range(home + 1, 2011):
                    for index in range(16):
                        games.append((
                            f"{home}:{away}:{index}",
                            "2023-10-17",
                            2023,
                            "regular",
                            home,
                            away,
                            "regular-hash",
                            1,
                            "Final",
                        ))

            db.executemany(
                "INSERT INTO games VALUES (?,?,?,?,?,?,?,?,?)",
                games,
            )

            db.executemany(
                "INSERT INTO game_checks VALUES (?,?,?)",
                (
                    (game[0], 1, "pbp-hash")
                    for game in games
                ),
            )

            evidence = regular_completion_evidence(db, 2023)

            self.assertEqual(evidence["completed_games"], 720)
            self.assertEqual(evidence["pbp_games"], 720)
            self.assertTrue(evidence["game_sets_match"])

            self.assertEqual(
                evidence["label_ready_date"],
                "2023-10-18",
            )

            self.assertTrue(
                all(
                    count == 144
                    for count in evidence["team_games"].values()
                )
            )

            db.execute(
                "UPDATE game_checks SET final_score_match=0 "
                "WHERE game_id=?",
                (games[0][0],),
            )

            with self.assertRaises(ValueError):
                regular_completion_evidence(db, 2023)


if __name__ == "__main__":
    unittest.main()