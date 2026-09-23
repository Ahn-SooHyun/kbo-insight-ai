from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from scripts.build_game_tables import (
    GAMES_COLUMNS,
    TEAM_GAMES_COLUMNS,
    GameTableBuildError,
    build_game_table_files,
    build_game_tables,
    build_games,
    build_team_games,
)


def make_pa(
    *,
    game_pk: str = "20230401HHOB02023",
    game_date: str = "2023-04-01",
    season: int = 2023,
    at_bat_number: int = 1,
    inning: int = 1,
    is_home_batting: bool = False,
    home_team: str = "OB",
    away_team: str = "HH",
    home_score_before: int = 0,
    away_score_before: int = 0,
    runs_scored: int = 0,
    post_home_score: int = 0,
    post_away_score: int = 0,
    pitch_rows: int = 1,
    pitch_count: int = 1,
    **overrides: object,
) -> dict[str, object]:
    """테스트용 Canonical Plate Appearance Row를 만든다."""
    if is_home_batting:
        batting_team = home_team
        fielding_team = away_team
    else:
        batting_team = away_team
        fielding_team = home_team

    row: dict[str, object] = {
        "game_pk": game_pk,
        "game_date": game_date,
        "season": season,
        "at_bat_number": at_bat_number,
        "inning": inning,
        "batting_team": batting_team,
        "fielding_team": fielding_team,
        "is_home_batting": is_home_batting,
        "home_score_before": home_score_before,
        "away_score_before": away_score_before,
        "runs_scored": runs_scored,
        "post_home_score": post_home_score,
        "post_away_score": post_away_score,
        "pitch_rows": pitch_rows,
        "pitch_count": pitch_count,
    }

    row.update(
        overrides
    )

    return row


def make_home_win_game(
    *,
    game_pk: str = "20230401HHOB02023",
    game_date: str = "2023-04-01",
    season: int = 2023,
    final_inning: int = 9,
) -> pd.DataFrame:
    """
    Home Team이 2:0으로 앞선 상태에서
    Top 9 종료 후 Bottom 9 없이 끝나는 경기 PA를 만든다.
    """
    rows = [
        make_pa(
            game_pk=game_pk,
            game_date=game_date,
            season=season,
            at_bat_number=1,
            inning=1,
            is_home_batting=False,
            home_score_before=0,
            away_score_before=0,
            runs_scored=0,
            post_home_score=0,
            post_away_score=0,
            pitch_rows=4,
            pitch_count=3,
        ),
        make_pa(
            game_pk=game_pk,
            game_date=game_date,
            season=season,
            at_bat_number=2,
            inning=1,
            is_home_batting=True,
            home_score_before=0,
            away_score_before=0,
            runs_scored=2,
            post_home_score=2,
            post_away_score=0,
            pitch_rows=2,
            pitch_count=2,
        ),
        make_pa(
            game_pk=game_pk,
            game_date=game_date,
            season=season,
            at_bat_number=3,
            inning=final_inning,
            is_home_batting=False,
            home_score_before=2,
            away_score_before=0,
            runs_scored=0,
            post_home_score=2,
            post_away_score=0,
            pitch_rows=5,
            pitch_count=4,
        ),
    ]

    return pd.DataFrame(
        rows
    )


def make_away_win_game() -> pd.DataFrame:
    """Away Team이 3:1로 승리하는 경기 PA를 만든다."""
    rows = [
        make_pa(
            game_pk="20230402HHOB02023",
            game_date="2023-04-02",
            at_bat_number=1,
            inning=1,
            is_home_batting=False,
            home_score_before=0,
            away_score_before=0,
            runs_scored=2,
            post_home_score=0,
            post_away_score=2,
        ),
        make_pa(
            game_pk="20230402HHOB02023",
            game_date="2023-04-02",
            at_bat_number=2,
            inning=1,
            is_home_batting=True,
            home_score_before=0,
            away_score_before=2,
            runs_scored=1,
            post_home_score=1,
            post_away_score=2,
        ),
        make_pa(
            game_pk="20230402HHOB02023",
            game_date="2023-04-02",
            at_bat_number=3,
            inning=9,
            is_home_batting=False,
            home_score_before=1,
            away_score_before=2,
            runs_scored=1,
            post_home_score=1,
            post_away_score=3,
        ),
        make_pa(
            game_pk="20230402HHOB02023",
            game_date="2023-04-02",
            at_bat_number=4,
            inning=9,
            is_home_batting=True,
            home_score_before=1,
            away_score_before=3,
            runs_scored=0,
            post_home_score=1,
            post_away_score=3,
        ),
    ]

    return pd.DataFrame(
        rows
    )


def make_tie_game(
    *,
    game_pk: str = "20230403HHOB02023",
    game_date: str = "2023-04-03",
    season: int = 2023,
) -> pd.DataFrame:
    """1:1 Tie 경기 PA를 만든다."""
    rows = [
        make_pa(
            game_pk=game_pk,
            game_date=game_date,
            season=season,
            at_bat_number=1,
            inning=1,
            is_home_batting=False,
            home_score_before=0,
            away_score_before=0,
            runs_scored=1,
            post_home_score=0,
            post_away_score=1,
        ),
        make_pa(
            game_pk=game_pk,
            game_date=game_date,
            season=season,
            at_bat_number=2,
            inning=1,
            is_home_batting=True,
            home_score_before=0,
            away_score_before=1,
            runs_scored=1,
            post_home_score=1,
            post_away_score=1,
        ),
        make_pa(
            game_pk=game_pk,
            game_date=game_date,
            season=season,
            at_bat_number=3,
            inning=9,
            is_home_batting=False,
            home_score_before=1,
            away_score_before=1,
            runs_scored=0,
            post_home_score=1,
            post_away_score=1,
        ),
        make_pa(
            game_pk=game_pk,
            game_date=game_date,
            season=season,
            at_bat_number=4,
            inning=9,
            is_home_batting=True,
            home_score_before=1,
            away_score_before=1,
            runs_scored=0,
            post_home_score=1,
            post_away_score=1,
        ),
    ]

    return pd.DataFrame(
        rows
    )


class GameTransformationTest(
    unittest.TestCase
):
    """games 핵심 변환 규칙을 검증한다."""

    def test_build_normal_game_as_one_row(
        self,
    ) -> None:
        games = build_games(
            make_home_win_game()
        )

        self.assertEqual(
            len(games),
            1,
        )

        self.assertEqual(
            list(games.columns),
            list(GAMES_COLUMNS),
        )

        self.assertFalse(
            games["game_pk"]
            .duplicated()
            .any()
        )

    def test_restore_home_and_away_team(
        self,
    ) -> None:
        games = build_games(
            make_home_win_game()
        )

        game = games.iloc[0]

        self.assertEqual(
            game["home_team"],
            "OB",
        )

        self.assertEqual(
            game["away_team"],
            "HH",
        )

    def test_home_win(
        self,
    ) -> None:
        games = build_games(
            make_home_win_game()
        )

        game = games.iloc[0]

        self.assertEqual(
            game["final_home_score"],
            2,
        )

        self.assertEqual(
            game["final_away_score"],
            0,
        )

        self.assertTrue(
            bool(
                game["home_win"]
            )
        )

        self.assertFalse(
            bool(
                game["away_win"]
            )
        )

        self.assertFalse(
            bool(
                game["is_tie"]
            )
        )

        self.assertEqual(
            game["winner_team"],
            "OB",
        )

        self.assertEqual(
            game["loser_team"],
            "HH",
        )

    def test_away_win(
        self,
    ) -> None:
        games = build_games(
            make_away_win_game()
        )

        game = games.iloc[0]

        self.assertFalse(
            bool(
                game["home_win"]
            )
        )

        self.assertTrue(
            bool(
                game["away_win"]
            )
        )

        self.assertFalse(
            bool(
                game["is_tie"]
            )
        )

        self.assertEqual(
            game["winner_team"],
            "HH",
        )

        self.assertEqual(
            game["loser_team"],
            "OB",
        )

    def test_tie_keeps_winner_and_loser_null(
        self,
    ) -> None:
        games = build_games(
            make_tie_game()
        )

        game = games.iloc[0]

        self.assertFalse(
            bool(
                game["home_win"]
            )
        )

        self.assertFalse(
            bool(
                game["away_win"]
            )
        )

        self.assertTrue(
            bool(
                game["is_tie"]
            )
        )

        self.assertTrue(
            pd.isna(
                game["winner_team"]
            )
        )

        self.assertTrue(
            pd.isna(
                game["loser_team"]
            )
        )

    def test_home_win_without_bottom_nine_has_nine_innings(
        self,
    ) -> None:
        games = build_games(
            make_home_win_game()
        )

        self.assertEqual(
            games.iloc[0][
                "innings_played"
            ],
            9,
        )

    def test_extra_inning_uses_maximum_observed_inning(
        self,
    ) -> None:
        source = (
            make_home_win_game(
                final_inning=10
            )
        )

        games = build_games(
            source
        )

        self.assertEqual(
            games.iloc[0][
                "innings_played"
            ],
            10,
        )

    def test_incomplete_last_pa_still_supplies_final_post_score(
        self,
    ) -> None:
        source = (
            make_home_win_game()
        )

        source["pa_completed"] = True

        source.loc[
            source[
                "at_bat_number"
            ].eq(3),
            "pa_completed",
        ] = False

        games = build_games(
            source
        )

        game = games.iloc[0]

        self.assertEqual(
            game["final_home_score"],
            2,
        )

        self.assertEqual(
            game["final_away_score"],
            0,
        )

    def test_runs_scored_reconciliation_matches_final_score(
        self,
    ) -> None:
        source = (
            make_home_win_game()
        )

        games = build_games(
            source
        )

        game = games.iloc[0]

        first = (
            source
            .sort_values(
                "at_bat_number"
            )
            .iloc[0]
        )

        expected_home = (
            first["home_score_before"]
            + source.loc[
                source[
                    "is_home_batting"
                ],
                "runs_scored",
            ].sum()
        )

        expected_away = (
            first["away_score_before"]
            + source.loc[
                ~source[
                    "is_home_batting"
                ],
                "runs_scored",
            ].sum()
        )

        self.assertEqual(
            game["final_home_score"],
            expected_home,
        )

        self.assertEqual(
            game["final_away_score"],
            expected_away,
        )

    def test_reject_score_reconciliation_mismatch(
        self,
    ) -> None:
        source = (
            make_home_win_game()
        )

        source.loc[
            source[
                "at_bat_number"
            ].eq(3),
            "post_home_score",
        ] = 3

        with self.assertRaises(
            GameTableBuildError
        ):
            build_games(
                source
            )

    def test_reject_game_context_conflict(
        self,
    ) -> None:
        source = (
            make_home_win_game()
        )

        source.loc[
            source[
                "at_bat_number"
            ].eq(2),
            "fielding_team",
        ] = "LG"

        with self.assertRaises(
            GameTableBuildError
        ):
            build_games(
                source
            )

    def test_reject_duplicate_pa_key(
        self,
    ) -> None:
        source = (
            make_home_win_game()
        )

        duplicate = (
            source.iloc[
                [0]
            ]
            .copy()
        )

        source = pd.concat(
            [
                source,
                duplicate,
            ],
            ignore_index=True,
        )

        with self.assertRaises(
            GameTableBuildError
        ):
            build_games(
                source
            )

    def test_pa_count_equals_home_plus_away(
        self,
    ) -> None:
        games = build_games(
            make_home_win_game()
        )

        game = games.iloc[0]

        self.assertEqual(
            game["plate_appearances"],
            3,
        )

        self.assertEqual(
            game[
                "home_plate_appearances"
            ],
            1,
        )

        self.assertEqual(
            game[
                "away_plate_appearances"
            ],
            2,
        )

        self.assertEqual(
            game["plate_appearances"],
            (
                game[
                    "home_plate_appearances"
                ]
                + game[
                    "away_plate_appearances"
                ]
            ),
        )

    def test_pitch_rows_and_pitch_count_are_summed(
        self,
    ) -> None:
        games = build_games(
            make_home_win_game()
        )

        game = games.iloc[0]

        self.assertEqual(
            game["pitch_rows"],
            11,
        )

        self.assertEqual(
            game["pitch_count"],
            9,
        )


class TeamGameTransformationTest(
    unittest.TestCase
):
    """team_games Long Format 변환 규칙을 검증한다."""

    def test_team_games_has_exactly_two_rows_per_game(
        self,
    ) -> None:
        games, team_games = (
            build_game_tables(
                make_home_win_game()
            )
        )

        self.assertEqual(
            len(team_games),
            2 * len(games),
        )

        counts = (
            team_games.groupby(
                "game_pk"
            )
            .size()
        )

        self.assertTrue(
            counts.eq(2).all()
        )

    def test_home_and_away_team_game_rows_are_mirrored(
        self,
    ) -> None:
        games, team_games = (
            build_game_tables(
                make_home_win_game()
            )
        )

        self.assertEqual(
            list(team_games.columns),
            list(TEAM_GAMES_COLUMNS),
        )

        game = games.iloc[0]

        home = (
            team_games.loc[
                team_games[
                    "is_home"
                ]
            ]
            .iloc[0]
        )

        away = (
            team_games.loc[
                ~team_games[
                    "is_home"
                ]
            ]
            .iloc[0]
        )

        self.assertEqual(
            home["team"],
            game["home_team"],
        )

        self.assertEqual(
            home["opponent"],
            game["away_team"],
        )

        self.assertEqual(
            away["team"],
            game["away_team"],
        )

        self.assertEqual(
            away["opponent"],
            game["home_team"],
        )

        self.assertEqual(
            home["runs_for"],
            away["runs_against"],
        )

        self.assertEqual(
            home["runs_against"],
            away["runs_for"],
        )

        self.assertEqual(
            home["run_diff"],
            -away["run_diff"],
        )

        self.assertEqual(
            home["plate_appearances"],
            away[
                "opponent_plate_appearances"
            ],
        )

        self.assertEqual(
            away["plate_appearances"],
            home[
                "opponent_plate_appearances"
            ],
        )

        self.assertTrue(
            bool(
                home["win"]
            )
        )

        self.assertTrue(
            bool(
                away["loss"]
            )
        )

    def test_team_game_key_is_unique(
        self,
    ) -> None:
        games = build_games(
            make_home_win_game()
        )

        team_games = (
            build_team_games(
                games
            )
        )

        self.assertFalse(
            team_games.duplicated(
                subset=[
                    "game_pk",
                    "team",
                ]
            ).any()
        )


class DeterminismTest(
    unittest.TestCase
):
    """입력 순서와 반복 실행에 대한 결정성을 검증한다."""

    def test_input_pa_row_order_does_not_change_outputs(
        self,
    ) -> None:
        source = pd.concat(
            [
                make_home_win_game(),
                make_away_win_game(),
            ],
            ignore_index=True,
        )

        shuffled = (
            source.sample(
                frac=1.0,
                random_state=17,
            )
            .reset_index(
                drop=True
            )
        )

        expected_games, expected_team_games = (
            build_game_tables(
                source
            )
        )

        actual_games, actual_team_games = (
            build_game_tables(
                shuffled
            )
        )

        pd.testing.assert_frame_equal(
            actual_games,
            expected_games,
        )

        pd.testing.assert_frame_equal(
            actual_team_games,
            expected_team_games,
        )

    def test_multiple_seasons_and_games_have_deterministic_sort(
        self,
    ) -> None:
        game_2024 = (
            make_home_win_game(
                game_pk="20240402HHOB02024",
                game_date="2024-04-02",
                season=2024,
            )
        )

        game_2023_late = (
            make_tie_game(
                game_pk="20230403HHOB02023",
                game_date="2023-04-03",
                season=2023,
            )
        )

        game_2023_early = (
            make_home_win_game(
                game_pk="20230401HHOB02023",
                game_date="2023-04-01",
                season=2023,
            )
        )

        source = pd.concat(
            [
                game_2024,
                game_2023_late,
                game_2023_early,
            ],
            ignore_index=True,
        )

        source = (
            source.sample(
                frac=1.0,
                random_state=9,
            )
            .reset_index(
                drop=True
            )
        )

        games, team_games = (
            build_game_tables(
                source
            )
        )

        self.assertEqual(
            games[
                "game_pk"
            ].tolist(),
            [
                "20230401HHOB02023",
                "20230403HHOB02023",
                "20240402HHOB02024",
            ],
        )

        expected_team_sort = (
            team_games
            .sort_values(
                by=[
                    "season",
                    "game_date",
                    "game_pk",
                    "is_home",
                ],
                kind="mergesort",
            )
            .reset_index(
                drop=True
            )
        )

        pd.testing.assert_frame_equal(
            team_games,
            expected_team_sort,
        )

    def test_rerun_produces_same_dataframes(
        self,
    ) -> None:
        source = pd.concat(
            [
                make_home_win_game(),
                make_tie_game(),
            ],
            ignore_index=True,
        )

        first_games, first_team_games = (
            build_game_tables(
                source
            )
        )

        second_games, second_team_games = (
            build_game_tables(
                source
            )
        )

        pd.testing.assert_frame_equal(
            first_games,
            second_games,
        )

        pd.testing.assert_frame_equal(
            first_team_games,
            second_team_games,
        )


class ParquetIOTest(
    unittest.TestCase
):
    """Parquet round-trip dtype과 파일 재실행을 검증한다."""

    def test_parquet_round_trip_preserves_output_dtypes(
        self,
    ) -> None:
        source = pd.concat(
            [
                make_home_win_game(),
                make_tie_game(),
            ],
            ignore_index=True,
        )

        with TemporaryDirectory() as temp_dir:
            root = Path(
                temp_dir
            )

            input_path = (
                root
                / "plate_appearances.parquet"
            )

            output_dir = (
                root
                / "derived"
            )

            source.to_parquet(
                input_path,
                engine="pyarrow",
                index=False,
            )

            build_game_table_files(
                input_path=input_path,
                output_dir=output_dir,
            )

            saved_games = (
                pd.read_parquet(
                    output_dir
                    / "games.parquet",
                    engine="pyarrow",
                )
            )

            saved_team_games = (
                pd.read_parquet(
                    output_dir
                    / "team_games.parquet",
                    engine="pyarrow",
                )
            )

            self.assertEqual(
                str(
                    saved_games[
                        "game_date"
                    ].dtype
                ),
                "datetime64[us]",
            )

            self.assertEqual(
                str(
                    saved_games[
                        "game_pk"
                    ].dtype
                ),
                "string",
            )

            self.assertEqual(
                str(
                    saved_games[
                        "winner_team"
                    ].dtype
                ),
                "string",
            )

            self.assertEqual(
                str(
                    saved_games[
                        "season"
                    ].dtype
                ),
                "Int64",
            )

            self.assertEqual(
                str(
                    saved_games[
                        "final_home_score"
                    ].dtype
                ),
                "Int64",
            )

            self.assertEqual(
                str(
                    saved_games[
                        "home_win"
                    ].dtype
                ),
                "boolean",
            )

            self.assertEqual(
                str(
                    saved_team_games[
                        "game_date"
                    ].dtype
                ),
                "datetime64[us]",
            )

            self.assertEqual(
                str(
                    saved_team_games[
                        "team"
                    ].dtype
                ),
                "string",
            )

            self.assertEqual(
                str(
                    saved_team_games[
                        "runs_for"
                    ].dtype
                ),
                "Int64",
            )

            self.assertEqual(
                str(
                    saved_team_games[
                        "is_home"
                    ].dtype
                ),
                "boolean",
            )

    def test_file_rebuild_is_reproducible_and_preserves_input(
        self,
    ) -> None:
        source = pd.concat(
            [
                make_home_win_game(),
                make_away_win_game(),
            ],
            ignore_index=True,
        )

        with TemporaryDirectory() as temp_dir:
            root = Path(
                temp_dir
            )

            input_path = (
                root
                / "plate_appearances.parquet"
            )

            output_dir = (
                root
                / "derived"
            )

            source.to_parquet(
                input_path,
                engine="pyarrow",
                index=False,
            )

            input_before = (
                input_path.read_bytes()
            )

            first_games, first_team_games = (
                build_game_table_files(
                    input_path=input_path,
                    output_dir=output_dir,
                )
            )

            first_saved_games = (
                pd.read_parquet(
                    output_dir
                    / "games.parquet",
                    engine="pyarrow",
                )
            )

            first_saved_team_games = (
                pd.read_parquet(
                    output_dir
                    / "team_games.parquet",
                    engine="pyarrow",
                )
            )

            second_games, second_team_games = (
                build_game_table_files(
                    input_path=input_path,
                    output_dir=output_dir,
                )
            )

            second_saved_games = (
                pd.read_parquet(
                    output_dir
                    / "games.parquet",
                    engine="pyarrow",
                )
            )

            second_saved_team_games = (
                pd.read_parquet(
                    output_dir
                    / "team_games.parquet",
                    engine="pyarrow",
                )
            )

            self.assertEqual(
                input_path.read_bytes(),
                input_before,
            )

            pd.testing.assert_frame_equal(
                first_games,
                second_games,
            )

            pd.testing.assert_frame_equal(
                first_team_games,
                second_team_games,
            )

            pd.testing.assert_frame_equal(
                first_saved_games,
                second_saved_games,
            )

            pd.testing.assert_frame_equal(
                first_saved_team_games,
                second_saved_team_games,
            )


class PartialSeasonTest(
    unittest.TestCase
):
    """2026 Partial Season을 완료 시즌으로 특수 처리하지 않는지 검증한다."""

    def test_2026_is_only_treated_as_observed_game_fact(
        self,
    ) -> None:
        source = (
            make_home_win_game(
                game_pk="20260401HHOB02026",
                game_date="2026-04-01",
                season=2026,
            )
        )

        games, team_games = (
            build_game_tables(
                source
            )
        )

        self.assertEqual(
            games["season"].tolist(),
            [2026],
        )

        self.assertEqual(
            len(games),
            1,
        )

        self.assertEqual(
            len(team_games),
            2,
        )

        self.assertNotIn(
            "season_complete",
            games.columns,
        )

        self.assertNotIn(
            "final_rank",
            games.columns,
        )


if __name__ == "__main__":
    unittest.main()