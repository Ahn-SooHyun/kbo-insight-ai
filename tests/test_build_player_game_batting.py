from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from scripts.build_player_game_batting import (
    EVENT_TO_COUNT_COLUMN,
    OUTPUT_COLUMNS,
    RAW_DATA_DIR,
    PlayerGameBattingBuildError,
    build_player_game_batting,
    build_player_game_batting_file,
)


def make_pa(
    *,
    game_pk: str = "20230401HHOB02023",
    game_date: str = "2023-04-01",
    season: int = 2023,
    at_bat_number: int = 1,
    batting_team: str = "HH",
    fielding_team: str = "OB",
    is_home_batting: bool = False,
    batter: str = "1001",
    batter_name: str = "타자1",
    event: object = "single",
    **overrides: object,
) -> dict[str, object]:
    """테스트용 Canonical Plate Appearance Row를 만든다."""
    row: dict[str, object] = {
        "game_pk": game_pk,
        "game_date": game_date,
        "season": season,
        "at_bat_number": at_bat_number,
        "batting_team": batting_team,
        "fielding_team": fielding_team,
        "is_home_batting": is_home_batting,
        "batter": batter,
        "batter_name": batter_name,
        "event": event,
    }

    row.update(
        overrides
    )

    return row


def make_event_frame(
    events: list[object],
    *,
    game_pk: str = "20230401HHOB02023",
    game_date: str = "2023-04-01",
    season: int = 2023,
    batter: str = "1001",
    batter_name: str = "타자1",
    batting_team: str = "HH",
    fielding_team: str = "OB",
    is_home_batting: bool = False,
) -> pd.DataFrame:
    """하나의 Player Game에 여러 Event를 가진 Synthetic PA를 만든다."""
    rows = [
        make_pa(
            game_pk=game_pk,
            game_date=game_date,
            season=season,
            at_bat_number=index,
            batting_team=batting_team,
            fielding_team=fielding_team,
            is_home_batting=is_home_batting,
            batter=batter,
            batter_name=batter_name,
            event=event,
        )
        for index, event in enumerate(
            events,
            start=1,
        )
    ]

    return pd.DataFrame(
        rows
    )


class PlayerGameBattingAggregationTest(
    unittest.TestCase
):
    """Player Game Batting 핵심 집계 규칙을 검증한다."""

    def test_builds_one_player_game_row(
        self,
    ) -> None:
        result = build_player_game_batting(
            pd.DataFrame(
                [
                    make_pa()
                ]
            )
        )

        self.assertEqual(
            len(result),
            1,
        )

        self.assertEqual(
            list(result.columns),
            list(OUTPUT_COLUMNS),
        )

    def test_aggregates_multiple_completed_pas_for_same_batter(
        self,
    ) -> None:
        source = make_event_frame(
            [
                "single",
                "walk",
                "strikeout",
            ]
        )

        result = build_player_game_batting(
            source
        )

        row = result.iloc[0]

        self.assertEqual(
            row["pa"],
            3,
        )

        self.assertEqual(
            row["single"],
            1,
        )

        self.assertEqual(
            row["bb"],
            1,
        )

        self.assertEqual(
            row["so"],
            1,
        )

    def test_separates_multiple_batters(
        self,
    ) -> None:
        source = pd.DataFrame(
            [
                make_pa(
                    at_bat_number=1,
                    batter="1001",
                    batter_name="타자1",
                ),
                make_pa(
                    at_bat_number=2,
                    batter="1002",
                    batter_name="타자2",
                ),
            ]
        )

        result = build_player_game_batting(
            source
        )

        self.assertEqual(
            len(result),
            2,
        )

        self.assertEqual(
            set(
                result["batter"]
                .tolist()
            ),
            {
                "1001",
                "1002",
            },
        )

    def test_team_opponent_and_is_home_context(
        self,
    ) -> None:
        result = build_player_game_batting(
            pd.DataFrame(
                [
                    make_pa(
                        batting_team="HH",
                        fielding_team="OB",
                        is_home_batting=False,
                    )
                ]
            )
        )

        row = result.iloc[0]

        self.assertEqual(
            row["team"],
            "HH",
        )

        self.assertEqual(
            row["opponent"],
            "OB",
        )

        self.assertFalse(
            bool(
                row["is_home"]
            )
        )

    def test_same_batter_can_have_different_team_in_different_games(
        self,
    ) -> None:
        source = pd.DataFrame(
            [
                make_pa(
                    game_pk="20230401HHOB02023",
                    game_date="2023-04-01",
                    batting_team="HH",
                    fielding_team="OB",
                    batter="1001",
                    batter_name="타자1",
                ),
                make_pa(
                    game_pk="20230701LTKT02023",
                    game_date="2023-07-01",
                    batting_team="LT",
                    fielding_team="KT",
                    batter="1001",
                    batter_name="타자1",
                ),
            ]
        )

        result = build_player_game_batting(
            source
        )

        self.assertEqual(
            result["team"].tolist(),
            [
                "HH",
                "LT",
            ],
        )

    def test_incomplete_pa_is_excluded(
        self,
    ) -> None:
        source = make_event_frame(
            [
                "single",
                None,
            ]
        )

        result = build_player_game_batting(
            source
        )

        self.assertEqual(
            result.iloc[0]["pa"],
            1,
        )

        self.assertEqual(
            result.iloc[0]["single"],
            1,
        )

    def test_player_game_without_completed_pa_is_not_created(
        self,
    ) -> None:
        source = pd.DataFrame(
            [
                make_pa(
                    at_bat_number=1,
                    batter="1001",
                    batter_name="완료타자",
                    event="single",
                ),
                make_pa(
                    at_bat_number=2,
                    batter="1002",
                    batter_name="미완료타자",
                    event=None,
                ),
            ]
        )

        result = build_player_game_batting(
            source
        )

        self.assertEqual(
            result["batter"].tolist(),
            [
                "1001",
            ],
        )

    def test_counts_single_double_triple_and_home_run(
        self,
    ) -> None:
        result = build_player_game_batting(
            make_event_frame(
                [
                    "single",
                    "double",
                    "triple",
                    "home_run",
                ]
            )
        )

        row = result.iloc[0]

        self.assertEqual(
            row["single"],
            1,
        )
        self.assertEqual(
            row["double"],
            1,
        )
        self.assertEqual(
            row["triple"],
            1,
        )
        self.assertEqual(
            row["hr"],
            1,
        )
        self.assertEqual(
            row["h"],
            4,
        )
        self.assertEqual(
            row["tb"],
            10,
        )

    def test_counts_walk_hbp_strikeout_sac_fly_and_sac_bunt(
        self,
    ) -> None:
        result = build_player_game_batting(
            make_event_frame(
                [
                    "walk",
                    "hit_by_pitch",
                    "strikeout",
                    "sac_fly",
                    "sac_bunt",
                ]
            )
        )

        row = result.iloc[0]

        self.assertEqual(
            row["bb"],
            1,
        )
        self.assertEqual(
            row["hbp"],
            1,
        )
        self.assertEqual(
            row["so"],
            1,
        )
        self.assertEqual(
            row["sf"],
            1,
        )
        self.assertEqual(
            row["sh"],
            1,
        )

    def test_counts_special_non_hit_events(
        self,
    ) -> None:
        result = build_player_game_batting(
            make_event_frame(
                [
                    "double_play",
                    "triple_play",
                    "field_error",
                    "fielders_choice",
                    "catcher_interference",
                ]
            )
        )

        row = result.iloc[0]

        self.assertEqual(
            row["double_play"],
            1,
        )
        self.assertEqual(
            row["triple_play"],
            1,
        )
        self.assertEqual(
            row["field_error"],
            1,
        )
        self.assertEqual(
            row["fielders_choice"],
            1,
        )
        self.assertEqual(
            row["catcher_interference"],
            1,
        )

    def test_field_out_is_included_in_pa_and_ab(
        self,
    ) -> None:
        result = build_player_game_batting(
            make_event_frame(
                [
                    "field_out",
                ]
            )
        )

        row = result.iloc[0]

        self.assertEqual(
            row["pa"],
            1,
        )

        self.assertEqual(
            row["ab"],
            1,
        )

        self.assertEqual(
            row["h"],
            0,
        )

    def test_h_formula(
        self,
    ) -> None:
        result = build_player_game_batting(
            make_event_frame(
                [
                    "single",
                    "single",
                    "double",
                    "triple",
                    "home_run",
                    "field_out",
                ]
            )
        )

        row = result.iloc[0]

        self.assertEqual(
            row["h"],
            (
                row["single"]
                + row["double"]
                + row["triple"]
                + row["hr"]
            ),
        )

    def test_ab_formula(
        self,
    ) -> None:
        result = build_player_game_batting(
            make_event_frame(
                [
                    "single",
                    "walk",
                    "hit_by_pitch",
                    "sac_bunt",
                    "sac_fly",
                    "catcher_interference",
                    "field_out",
                ]
            )
        )

        row = result.iloc[0]

        expected = (
            row["pa"]
            - row["bb"]
            - row["hbp"]
            - row["sh"]
            - row["sf"]
            - row[
                "catcher_interference"
            ]
        )

        self.assertEqual(
            row["ab"],
            expected,
        )

    def test_tb_formula(
        self,
    ) -> None:
        result = build_player_game_batting(
            make_event_frame(
                [
                    "single",
                    "double",
                    "triple",
                    "home_run",
                ]
            )
        )

        row = result.iloc[0]

        expected = (
            row["single"]
            + 2 * row["double"]
            + 3 * row["triple"]
            + 4 * row["hr"]
        )

        self.assertEqual(
            row["tb"],
            expected,
        )

    def test_avg_obp_slg_and_ops(
        self,
    ) -> None:
        result = build_player_game_batting(
            make_event_frame(
                [
                    "single",
                    "double",
                    "walk",
                    "hit_by_pitch",
                    "sac_fly",
                    "field_out",
                ]
            )
        )

        row = result.iloc[0]

        self.assertEqual(
            row["ab"],
            3,
        )

        self.assertAlmostEqual(
            float(
                row["avg"]
            ),
            0.667,
            places=3,
        )

        self.assertAlmostEqual(
            float(
                row["obp"]
            ),
            0.667,
            places=3,
        )

        self.assertAlmostEqual(
            float(
                row["slg"]
            ),
            1.000,
            places=3,
        )

        self.assertAlmostEqual(
            float(
                row["ops"]
            ),
            1.667,
            places=3,
        )

    def test_rate_stats_are_rounded_to_three_decimals(
        self,
    ) -> None:
        result = build_player_game_batting(
            make_event_frame(
                [
                    "single",
                    "single",
                    "field_out",
                ]
            )
        )

        row = result.iloc[0]

        self.assertEqual(
            float(
                row["avg"]
            ),
            0.667,
        )

        self.assertEqual(
            float(
                row["obp"]
            ),
            0.667,
        )

        self.assertEqual(
            float(
                row["slg"]
            ),
            0.667,
        )

        self.assertEqual(
            float(
                row["ops"]
            ),
            1.334,
        )

    def test_zero_denominator_keeps_undefined_rates_null(
        self,
    ) -> None:
        walk_result = build_player_game_batting(
            make_event_frame(
                [
                    "walk",
                ]
            )
        )

        walk_row = (
            walk_result.iloc[0]
        )

        self.assertEqual(
            walk_row["ab"],
            0,
        )

        self.assertTrue(
            pd.isna(
                walk_row["avg"]
            )
        )

        self.assertEqual(
            float(
                walk_row["obp"]
            ),
            1.0,
        )

        self.assertTrue(
            pd.isna(
                walk_row["slg"]
            )
        )

        self.assertTrue(
            pd.isna(
                walk_row["ops"]
            )
        )

        interference_result = (
            build_player_game_batting(
                make_event_frame(
                    [
                        "catcher_interference",
                    ]
                )
            )
        )

        interference_row = (
            interference_result.iloc[0]
        )

        self.assertTrue(
            pd.isna(
                interference_row["avg"]
            )
        )

        self.assertTrue(
            pd.isna(
                interference_row["obp"]
            )
        )

        self.assertTrue(
            pd.isna(
                interference_row["slg"]
            )
        )

        self.assertTrue(
            pd.isna(
                interference_row["ops"]
            )
        )

    def test_unknown_non_null_event_is_rejected(
        self,
    ) -> None:
        source = pd.DataFrame(
            [
                make_pa(
                    event="other",
                )
            ]
        )

        with self.assertRaises(
            PlayerGameBattingBuildError
        ):
            build_player_game_batting(
                source
            )

    def test_duplicate_pa_key_is_rejected(
        self,
    ) -> None:
        source = pd.DataFrame(
            [
                make_pa(
                    at_bat_number=1,
                    batter="1001",
                ),
                make_pa(
                    at_bat_number=1,
                    batter="1002",
                ),
            ]
        )

        with self.assertRaises(
            PlayerGameBattingBuildError
        ):
            build_player_game_batting(
                source
            )

    def test_player_game_context_conflict_is_rejected(
        self,
    ) -> None:
        conflict_cases = {
            "game_date": {
                "game_date": "2023-04-02",
            },
            "season": {
                "season": 2024,
            },
            "batter_name": {
                "batter_name": "다른타자명",
            },
            "team": {
                "batting_team": "KT",
            },
            "opponent": {
                "fielding_team": "KT",
            },
            "is_home": {
                "is_home_batting": True,
            },
        }

        for name, overrides in conflict_cases.items():
            with self.subTest(
                context=name
            ):
                first = make_pa(
                    at_bat_number=1,
                )

                second = make_pa(
                    at_bat_number=2,
                    **overrides,
                )

                source = pd.DataFrame(
                    [
                        first,
                        second,
                    ]
                )

                with self.assertRaises(
                    PlayerGameBattingBuildError
                ):
                    build_player_game_batting(
                        source
                    )

    def test_output_grain_is_unique(
        self,
    ) -> None:
        source = pd.DataFrame(
            [
                make_pa(
                    at_bat_number=1,
                    batter="1001",
                ),
                make_pa(
                    at_bat_number=2,
                    batter="1001",
                ),
                make_pa(
                    at_bat_number=3,
                    batter="1002",
                    batter_name="타자2",
                ),
            ]
        )

        result = build_player_game_batting(
            source
        )

        self.assertFalse(
            result.duplicated(
                subset=[
                    "game_pk",
                    "batter",
                ]
            ).any()
        )

    def test_completed_pa_count_reconciliation(
        self,
    ) -> None:
        source = pd.DataFrame(
            [
                make_pa(
                    at_bat_number=1,
                    batter="1001",
                    event="single",
                ),
                make_pa(
                    at_bat_number=2,
                    batter="1001",
                    event=None,
                ),
                make_pa(
                    at_bat_number=3,
                    batter="1002",
                    batter_name="타자2",
                    event="walk",
                ),
            ]
        )

        result = build_player_game_batting(
            source
        )

        completed_count = int(
            source["event"]
            .notna()
            .sum()
        )

        self.assertEqual(
            int(
                result["pa"]
                .sum()
            ),
            completed_count,
        )

    def test_event_cross_tab_reconciliation(
        self,
    ) -> None:
        events = [
            "single",
            "double",
            "triple",
            "home_run",
            "walk",
            "hit_by_pitch",
            "strikeout",
            "field_out",
            "double_play",
            "triple_play",
            "sac_bunt",
            "sac_fly",
            "field_error",
            "fielders_choice",
            "catcher_interference",
        ]

        source = make_event_frame(
            events
        )

        result = build_player_game_batting(
            source
        )

        for event, column in EVENT_TO_COUNT_COLUMN.items():
            with self.subTest(
                event=event
            ):
                self.assertEqual(
                    int(
                        result[
                            column
                        ]
                        .sum()
                    ),
                    1,
                )

        exposed_sum = int(
            result[
                list(
                    EVENT_TO_COUNT_COLUMN.values()
                )
            ]
            .sum(
                axis=1
            )
            .sum()
        )

        field_out_count = (
            int(
                result["pa"]
                .sum()
            )
            - exposed_sum
        )

        self.assertEqual(
            field_out_count,
            1,
        )

    def test_input_row_shuffle_is_deterministic(
        self,
    ) -> None:
        source = pd.DataFrame(
            [
                make_pa(
                    at_bat_number=1,
                    batter="1001",
                    event="single",
                ),
                make_pa(
                    at_bat_number=2,
                    batter="1002",
                    batter_name="타자2",
                    event="walk",
                ),
                make_pa(
                    at_bat_number=3,
                    batter="1001",
                    event="double",
                ),
                make_pa(
                    game_pk="20230402HHOB02023",
                    game_date="2023-04-02",
                    at_bat_number=1,
                    batter="1001",
                    event="strikeout",
                ),
            ]
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

        expected = build_player_game_batting(
            source
        )

        actual = build_player_game_batting(
            shuffled
        )

        pd.testing.assert_frame_equal(
            expected,
            actual,
        )

    def test_multi_season_and_game_sorting(
        self,
    ) -> None:
        source = pd.DataFrame(
            [
                make_pa(
                    game_pk="20240402HHOB02024",
                    game_date="2024-04-02",
                    season=2024,
                    at_bat_number=1,
                ),
                make_pa(
                    game_pk="20230403HHOB02023",
                    game_date="2023-04-03",
                    season=2023,
                    at_bat_number=1,
                ),
                make_pa(
                    game_pk="20230401HHOB02023",
                    game_date="2023-04-01",
                    season=2023,
                    at_bat_number=1,
                ),
            ]
        )

        result = build_player_game_batting(
            source
        )

        self.assertEqual(
            result["game_pk"].tolist(),
            [
                "20230401HHOB02023",
                "20230403HHOB02023",
                "20240402HHOB02024",
            ],
        )

    def test_player_id_preserves_string_meaning(
        self,
    ) -> None:
        source = pd.DataFrame(
            [
                make_pa(
                    batter="00123",
                )
            ]
        )

        result = build_player_game_batting(
            source
        )

        self.assertEqual(
            str(
                result["batter"].dtype
            ),
            "string",
        )

        self.assertEqual(
            result.iloc[0]["batter"],
            "00123",
        )

    def test_rerun_produces_same_dataframe(
        self,
    ) -> None:
        source = pd.DataFrame(
            [
                make_pa(
                    at_bat_number=1,
                    batter="1001",
                    event="single",
                ),
                make_pa(
                    at_bat_number=2,
                    batter="1001",
                    event="walk",
                ),
                make_pa(
                    at_bat_number=3,
                    batter="1002",
                    batter_name="타자2",
                    event="field_out",
                ),
            ]
        )

        first = build_player_game_batting(
            source
        )

        second = build_player_game_batting(
            source
        )

        pd.testing.assert_frame_equal(
            first,
            second,
        )

    def test_rejects_invalid_identity_or_team_context(
        self,
    ) -> None:
        invalid_cases = {
            "null_batter": {
                "batter": None,
            },
            "empty_batter": {
                "batter": "",
            },
            "null_team": {
                "batting_team": None,
            },
            "null_opponent": {
                "fielding_team": None,
            },
            "same_team": {
                "batting_team": "OB",
                "fielding_team": "OB",
            },
        }

        for name, overrides in invalid_cases.items():
            with self.subTest(
                case=name
            ):
                row = make_pa()

                row.update(
                    overrides
                )

                with self.assertRaises(
                    PlayerGameBattingBuildError
                ):
                    build_player_game_batting(
                        pd.DataFrame(
                            [
                                row
                            ]
                        )
                    )

    def test_prohibited_run_and_rbi_columns_are_absent(
        self,
    ) -> None:
        result = build_player_game_batting(
            pd.DataFrame(
                [
                    make_pa()
                ]
            )
        )

        self.assertNotIn(
            "rbi",
            result.columns,
        )

        self.assertNotIn(
            "runs",
            result.columns,
        )

        self.assertNotIn(
            "batter_runs",
            result.columns,
        )


class PlayerGameBattingParquetIOTest(
    unittest.TestCase
):
    """Parquet round-trip, 경로 안전성 및 입력 불변성을 검증한다."""

    def test_parquet_round_trip_preserves_output_dtypes(
        self,
    ) -> None:
        source = pd.concat(
            [
                make_event_frame(
                    [
                        "single",
                        "walk",
                    ]
                ),
                make_event_frame(
                    [
                        "catcher_interference",
                    ],
                    game_pk="20230402HHOB02023",
                    game_date="2023-04-02",
                    batter="1002",
                    batter_name="타자2",
                ),
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

            output_path = (
                root
                / "derived"
                / "player_game_batting.parquet"
            )

            source.to_parquet(
                input_path,
                engine="pyarrow",
                index=False,
            )

            build_player_game_batting_file(
                input_path=input_path,
                output_path=output_path,
            )

            saved = pd.read_parquet(
                output_path,
                engine="pyarrow",
            )

            expected_dtypes = {
                "game_pk": "string",
                "game_date": "datetime64[us]",
                "season": "Int64",
                "batter": "string",
                "batter_name": "string",
                "team": "string",
                "opponent": "string",
                "is_home": "boolean",
                "pa": "Int64",
                "ab": "Int64",
                "h": "Int64",
                "single": "Int64",
                "double": "Int64",
                "triple": "Int64",
                "hr": "Int64",
                "bb": "Int64",
                "hbp": "Int64",
                "so": "Int64",
                "sf": "Int64",
                "sh": "Int64",
                "tb": "Int64",
                "double_play": "Int64",
                "triple_play": "Int64",
                "field_error": "Int64",
                "fielders_choice": "Int64",
                "catcher_interference": "Int64",
                "avg": "Float64",
                "obp": "Float64",
                "slg": "Float64",
                "ops": "Float64",
            }

            for column, expected_dtype in expected_dtypes.items():
                with self.subTest(
                    column=column
                ):
                    self.assertEqual(
                        str(
                            saved[
                                column
                            ].dtype
                        ),
                        expected_dtype,
                    )

    def test_file_build_preserves_input_pa_file(
        self,
    ) -> None:
        source = make_event_frame(
            [
                "single",
                "double",
                "walk",
            ]
        )

        with TemporaryDirectory() as temp_dir:
            root = Path(
                temp_dir
            )

            input_path = (
                root
                / "plate_appearances.parquet"
            )

            output_path = (
                root
                / "derived"
                / "player_game_batting.parquet"
            )

            source.to_parquet(
                input_path,
                engine="pyarrow",
                index=False,
            )

            input_before = (
                input_path.read_bytes()
            )

            build_player_game_batting_file(
                input_path=input_path,
                output_path=output_path,
            )

            self.assertEqual(
                input_path.read_bytes(),
                input_before,
            )

    def test_file_rebuild_is_deterministic(
        self,
    ) -> None:
        source = pd.DataFrame(
            [
                make_pa(
                    at_bat_number=1,
                    batter="1001",
                    event="single",
                ),
                make_pa(
                    at_bat_number=2,
                    batter="1001",
                    event="double",
                ),
                make_pa(
                    at_bat_number=3,
                    batter="1002",
                    batter_name="타자2",
                    event="walk",
                ),
            ]
        )

        with TemporaryDirectory() as temp_dir:
            root = Path(
                temp_dir
            )

            input_path = (
                root
                / "plate_appearances.parquet"
            )

            output_path = (
                root
                / "player_game_batting.parquet"
            )

            source.to_parquet(
                input_path,
                engine="pyarrow",
                index=False,
            )

            first = build_player_game_batting_file(
                input_path=input_path,
                output_path=output_path,
            )

            first_saved = pd.read_parquet(
                output_path,
                engine="pyarrow",
            )

            second = build_player_game_batting_file(
                input_path=input_path,
                output_path=output_path,
            )

            second_saved = pd.read_parquet(
                output_path,
                engine="pyarrow",
            )

            pd.testing.assert_frame_equal(
                first,
                second,
            )

            pd.testing.assert_frame_equal(
                first_saved,
                second_saved,
            )

    def test_raw_directory_output_is_rejected(
        self,
    ) -> None:
        source = pd.DataFrame(
            [
                make_pa()
            ]
        )

        with TemporaryDirectory() as temp_dir:
            input_path = (
                Path(
                    temp_dir
                )
                / "plate_appearances.parquet"
            )

            source.to_parquet(
                input_path,
                engine="pyarrow",
                index=False,
            )

            output_path = (
                RAW_DATA_DIR
                / "__player_game_batting_test__.parquet"
            )

            with self.assertRaises(
                PlayerGameBattingBuildError
            ):
                build_player_game_batting_file(
                    input_path=input_path,
                    output_path=output_path,
                )

            self.assertFalse(
                output_path.exists()
            )

    def test_input_pa_overwrite_output_is_rejected(
        self,
    ) -> None:
        source = pd.DataFrame(
            [
                make_pa()
            ]
        )

        with TemporaryDirectory() as temp_dir:
            input_path = (
                Path(
                    temp_dir
                )
                / "plate_appearances.parquet"
            )

            source.to_parquet(
                input_path,
                engine="pyarrow",
                index=False,
            )

            input_before = (
                input_path.read_bytes()
            )

            with self.assertRaises(
                PlayerGameBattingBuildError
            ):
                build_player_game_batting_file(
                    input_path=input_path,
                    output_path=input_path,
                )

            self.assertEqual(
                input_path.read_bytes(),
                input_before,
            )


if __name__ == "__main__":
    unittest.main()