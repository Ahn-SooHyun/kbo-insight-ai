from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from scripts.build_plate_appearances import (
    OUTPUT_COLUMNS,
    PA_KEY,
    PlateAppearanceBuildError,
    build_plate_appearance_file,
    build_plate_appearances,
    normalize_seasons,
)


def make_row(
    **overrides: object,
) -> dict[str, object]:
    """테스트용 최소 Raw Pitch Row를 만든다."""
    row: dict[str, object] = {
        "game_pk": "20230401HHOB02023",
        "game_date": "2023-04-01",
        "home_team": "OB",
        "away_team": "HH",
        "inning": 1,
        "inning_topbot": "top",
        "at_bat_number": 1,
        "pitch_number": 1,
        "batter": "B1",
        "pitcher": "P1",
        "batter_name": "타자1",
        "pitcher_name": "투수1",
        "strikes": 0,
        "outs_when_up": 0,
        "on_1b": None,
        "on_2b": None,
        "on_3b": None,
        "home_score": 0,
        "away_score": 0,
        "type": "B",
        "stand": "R",
        "events": "field_out",
        "post_home_score": 0,
        "post_away_score": 0,
        "post_outs": 1,
        "runs_scored": 0,
        "post_on_1b": None,
        "post_on_2b": None,
        "post_on_3b": None,
    }

    row.update(
        overrides
    )

    return row


def make_mixed_dataframe() -> pd.DataFrame:
    """
    Pitch Summary, 미완료 PA, Pitch-less PA,
    Batter 교체를 함께 검증할 Raw DataFrame을 만든다.
    """
    rows = [
        make_row(
            at_bat_number=1,
            pitch_number=1,
            batter="B1",
            batter_name="타자1",
            strikes=0,
            type="B",
            events="single",
            home_score=0,
            away_score=1,
            post_home_score=0,
            post_away_score=1,
            post_outs=0,
            post_on_1b="B1",
        ),
        make_row(
            at_bat_number=1,
            pitch_number=2,
            batter="B1",
            batter_name="타자1",
            strikes=0,
            type="S",
            events="single",
            home_score=0,
            away_score=1,
            post_home_score=0,
            post_away_score=1,
            post_outs=0,
            post_on_1b="B1",
        ),
        make_row(
            at_bat_number=1,
            pitch_number=3,
            batter="B1",
            batter_name="타자1",
            strikes=1,
            type="X",
            events="single",
            home_score=0,
            away_score=1,
            post_home_score=0,
            post_away_score=1,
            post_outs=0,
            post_on_1b="B1",
        ),
        make_row(
            inning_topbot="bot",
            at_bat_number=2,
            pitch_number=1,
            batter="B2",
            batter_name="타자2",
            strikes=0,
            type="S",
            events=None,
            outs_when_up=1,
            on_1b="R1",
            home_score=2,
            away_score=1,
            post_home_score=2,
            post_away_score=1,
            post_outs=3,
        ),
        make_row(
            inning_topbot="bot",
            at_bat_number=2,
            pitch_number=2,
            batter="B2",
            batter_name="타자2",
            strikes=1,
            type="B",
            events=None,
            outs_when_up=1,
            on_1b="R1",
            home_score=2,
            away_score=1,
            post_home_score=2,
            post_away_score=1,
            post_outs=3,
        ),
        make_row(
            inning=2,
            inning_topbot="top",
            at_bat_number=3,
            pitch_number=0,
            batter="B3",
            batter_name="타자3",
            strikes=0,
            type=None,
            events="walk",
            home_score=2,
            away_score=1,
            post_home_score=2,
            post_away_score=1,
            post_outs=0,
            post_on_1b="B3",
        ),
        make_row(
            inning=2,
            inning_topbot="bot",
            at_bat_number=4,
            pitch_number=1,
            batter="OLD",
            batter_name="선발타자",
            stand="R",
            strikes=0,
            type="B",
            events="single",
            home_score=2,
            away_score=1,
            post_home_score=2,
            post_away_score=1,
            post_outs=0,
            post_on_1b="NEW",
        ),
        make_row(
            inning=2,
            inning_topbot="bot",
            at_bat_number=4,
            pitch_number=2,
            batter="NEW",
            batter_name="대타",
            stand="L",
            strikes=0,
            type="X",
            events="single",
            home_score=2,
            away_score=1,
            post_home_score=2,
            post_away_score=1,
            post_outs=0,
            post_on_1b="NEW",
        ),
        make_row(
            inning=3,
            inning_topbot="top",
            at_bat_number=5,
            pitch_number=1,
            batter="OLD_SO",
            batter_name="선발타자2",
            stand="R",
            strikes=0,
            type="S",
            events="strikeout",
            home_score=2,
            away_score=1,
            post_home_score=2,
            post_away_score=1,
            post_outs=1,
        ),
        make_row(
            inning=3,
            inning_topbot="top",
            at_bat_number=5,
            pitch_number=2,
            batter="OLD_SO",
            batter_name="선발타자2",
            stand="R",
            strikes=1,
            type="S",
            events="strikeout",
            home_score=2,
            away_score=1,
            post_home_score=2,
            post_away_score=1,
            post_outs=1,
        ),
        make_row(
            inning=3,
            inning_topbot="top",
            at_bat_number=5,
            pitch_number=3,
            batter="NEW_SO",
            batter_name="대타2",
            stand="L",
            strikes=2,
            type="S",
            events="strikeout",
            home_score=2,
            away_score=1,
            post_home_score=2,
            post_away_score=1,
            post_outs=1,
        ),
    ]

    return pd.DataFrame(
        rows
    )


class NormalizeSeasonsTest(
    unittest.TestCase
):
    """시즌 입력 정규화 규칙을 검증한다."""

    def test_remove_duplicate_and_sort_seasons(
        self,
    ) -> None:
        self.assertEqual(
            normalize_seasons(
                [
                    2025,
                    2023,
                    2025,
                    2024,
                ]
            ),
            (
                2023,
                2024,
                2025,
            ),
        )

    def test_reject_unsupported_season(
        self,
    ) -> None:
        with self.assertRaises(
            PlateAppearanceBuildError
        ):
            normalize_seasons(
                [2027]
            )


class PlateAppearanceTransformationTest(
    unittest.TestCase
):
    """Plate Appearance 핵심 변환 규칙을 검증한다."""

    def get_pa(
        self,
        df: pd.DataFrame,
        at_bat_number: int,
    ) -> pd.Series:
        """at_bat_number로 단일 PA Row를 선택한다."""
        matched = df.loc[
            df[
                "at_bat_number"
            ].eq(
                at_bat_number
            )
        ]

        self.assertEqual(
            len(matched),
            1,
        )

        return matched.iloc[0]

    def test_preserve_pa_grain_and_team_state(
        self,
    ) -> None:
        source = (
            make_mixed_dataframe()
        )

        result = (
            build_plate_appearances(
                source,
                2023,
            )
        )

        expected_pa_count = (
            source.loc[
                :,
                PA_KEY,
            ]
            .drop_duplicates()
            .shape[0]
        )

        self.assertEqual(
            len(result),
            expected_pa_count,
        )

        self.assertFalse(
            result.duplicated(
                subset=list(PA_KEY)
            ).any()
        )

        self.assertEqual(
            list(result.columns),
            list(OUTPUT_COLUMNS),
        )

        top_pa = self.get_pa(
            result,
            1,
        )

        self.assertEqual(
            top_pa["batting_team"],
            "HH",
        )
        self.assertEqual(
            top_pa["fielding_team"],
            "OB",
        )
        self.assertFalse(
            bool(
                top_pa[
                    "is_home_batting"
                ]
            )
        )
        self.assertEqual(
            top_pa[
                "batting_score_before"
            ],
            1,
        )
        self.assertEqual(
            top_pa[
                "fielding_score_before"
            ],
            0,
        )
        self.assertEqual(
            top_pa[
                "score_diff_before"
            ],
            1,
        )

        bottom_pa = self.get_pa(
            result,
            2,
        )

        self.assertEqual(
            bottom_pa[
                "batting_team"
            ],
            "OB",
        )
        self.assertEqual(
            bottom_pa[
                "fielding_team"
            ],
            "HH",
        )
        self.assertTrue(
            bool(
                bottom_pa[
                    "is_home_batting"
                ]
            )
        )
        self.assertEqual(
            bottom_pa[
                "batting_score_before"
            ],
            2,
        )
        self.assertEqual(
            bottom_pa[
                "fielding_score_before"
            ],
            1,
        )

    def test_count_actual_pitches_by_type(
        self,
    ) -> None:
        result = (
            build_plate_appearances(
                make_mixed_dataframe(),
                2023,
            )
        )

        pa = self.get_pa(
            result,
            1,
        )

        self.assertEqual(
            pa["pitch_rows"],
            3,
        )
        self.assertEqual(
            pa["pitch_count"],
            3,
        )
        self.assertEqual(
            pa["ball_pitch_count"],
            1,
        )
        self.assertEqual(
            pa["strike_pitch_count"],
            1,
        )
        self.assertEqual(
            pa["in_play_pitch_count"],
            1,
        )

    def test_pitchless_pa_keeps_row_with_zero_pitch_count(
        self,
    ) -> None:
        result = (
            build_plate_appearances(
                make_mixed_dataframe(),
                2023,
            )
        )

        pa = self.get_pa(
            result,
            3,
        )

        self.assertEqual(
            pa["event"],
            "walk",
        )
        self.assertTrue(
            bool(
                pa["pa_completed"]
            )
        )
        self.assertEqual(
            pa["pitch_rows"],
            1,
        )
        self.assertEqual(
            pa["pitch_count"],
            0,
        )
        self.assertEqual(
            pa["ball_pitch_count"],
            0,
        )
        self.assertEqual(
            pa["strike_pitch_count"],
            0,
        )
        self.assertEqual(
            pa["in_play_pitch_count"],
            0,
        )

    def test_incomplete_pa_is_preserved(
        self,
    ) -> None:
        result = (
            build_plate_appearances(
                make_mixed_dataframe(),
                2023,
            )
        )

        pa = self.get_pa(
            result,
            2,
        )

        self.assertTrue(
            pd.isna(
                pa["event"]
            )
        )
        self.assertFalse(
            bool(
                pa["pa_completed"]
            )
        )

    def test_general_batter_replacement_credits_finishing_batter(
        self,
    ) -> None:
        result = (
            build_plate_appearances(
                make_mixed_dataframe(),
                2023,
            )
        )

        pa = self.get_pa(
            result,
            4,
        )

        self.assertEqual(
            pa["batter"],
            "NEW",
        )
        self.assertEqual(
            pa["batter_name"],
            "대타",
        )
        self.assertEqual(
            pa["stand"],
            "L",
        )

    def test_two_strike_substitution_strikeout_credits_starting_batter(
        self,
    ) -> None:
        result = (
            build_plate_appearances(
                make_mixed_dataframe(),
                2023,
            )
        )

        pa = self.get_pa(
            result,
            5,
        )

        self.assertEqual(
            pa["event"],
            "strikeout",
        )
        self.assertEqual(
            pa["batter"],
            "OLD_SO",
        )
        self.assertEqual(
            pa["batter_name"],
            "선발타자2",
        )
        self.assertEqual(
            pa["stand"],
            "R",
        )

    def test_use_actual_first_and_last_rows_with_null_values(
        self,
    ) -> None:
        source = pd.DataFrame(
            [
                make_row(
                    at_bat_number=10,
                    pitch_number=1,
                    batter="B10",
                    batter_name="첫이름",
                    on_1b=None,
                    post_on_1b="EARLY_VALUE",
                    type="B",
                    events="single",
                ),
                make_row(
                    at_bat_number=10,
                    pitch_number=2,
                    batter="B10",
                    batter_name=None,
                    on_1b="LATE_VALUE",
                    post_on_1b=None,
                    type="X",
                    events="single",
                ),
            ]
        )

        result = (
            build_plate_appearances(
                source,
                2023,
            )
        )

        pa = self.get_pa(
            result,
            10,
        )

        self.assertTrue(
            pd.isna(
                pa[
                    "on_1b_before"
                ]
            )
        )

        self.assertTrue(
            pd.isna(
                pa[
                    "post_on_1b"
                ]
            )
        )

        self.assertTrue(
            pd.isna(
                pa[
                    "batter_name"
                ]
            )
        )

    def test_output_is_deterministic_after_raw_row_reordering(
        self,
    ) -> None:
        source = (
            make_mixed_dataframe()
        )

        shuffled = (
            source.sample(
                frac=1.0,
                random_state=7,
            )
            .reset_index(
                drop=True
            )
        )

        expected = (
            build_plate_appearances(
                source,
                2023,
            )
        )

        actual = (
            build_plate_appearances(
                shuffled,
                2023,
            )
        )

        pd.testing.assert_frame_equal(
            actual,
            expected,
        )

    def test_reject_unclassified_actual_pitch_type(
        self,
    ) -> None:
        source = pd.DataFrame(
            [
                make_row(
                    pitch_number=1,
                    type=None,
                )
            ]
        )

        with self.assertRaises(
            PlateAppearanceBuildError
        ):
            build_plate_appearances(
                source,
                2023,
            )


class PlateAppearanceFileBuildTest(
    unittest.TestCase
):
    """Parquet I/O와 재실행 시 결과 일관성을 검증한다."""

    def test_build_file_preserves_raw_and_is_reproducible(
        self,
    ) -> None:
        source = (
            make_mixed_dataframe()
        )

        with TemporaryDirectory() as temp_dir:
            root = Path(
                temp_dir
            )

            raw_dir = (
                root
                / "data"
                / "raw"
                / "hf_kbo_pbp"
            )

            output_path = (
                root
                / "data"
                / "interim"
                / "hf_kbo_pbp"
                / "derived"
                / "plate_appearances.parquet"
            )

            raw_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            raw_path = (
                raw_dir
                / "2023.parquet"
            )

            source.to_parquet(
                raw_path,
                engine="pyarrow",
                index=False,
            )

            raw_before = (
                raw_path.read_bytes()
            )

            first_result = (
                build_plate_appearance_file(
                    raw_dir=raw_dir,
                    output_path=output_path,
                    seasons=[2023],
                )
            )

            self.assertTrue(
                output_path.is_file()
            )

            self.assertEqual(
                raw_path.read_bytes(),
                raw_before,
            )

            first_saved = (
                pd.read_parquet(
                    output_path,
                    engine="pyarrow",
                )
            )

            second_result = (
                build_plate_appearance_file(
                    raw_dir=raw_dir,
                    output_path=output_path,
                    seasons=[2023],
                )
            )

            second_saved = (
                pd.read_parquet(
                    output_path,
                    engine="pyarrow",
                )
            )

            self.assertEqual(
                raw_path.read_bytes(),
                raw_before,
            )

            pd.testing.assert_frame_equal(
                first_result,
                second_result,
            )

            pd.testing.assert_frame_equal(
                first_saved,
                second_saved,
            )

    def test_reject_output_path_inside_raw_directory(
        self,
    ) -> None:
        source = (
            make_mixed_dataframe()
        )

        with TemporaryDirectory() as temp_dir:
            raw_dir = (
                Path(temp_dir)
                / "raw"
            )

            raw_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            source.to_parquet(
                raw_dir
                / "2023.parquet",
                engine="pyarrow",
                index=False,
            )

            with self.assertRaises(
                PlateAppearanceBuildError
            ):
                build_plate_appearance_file(
                    raw_dir=raw_dir,
                    output_path=(
                        raw_dir
                        / "plate_appearances.parquet"
                    ),
                    seasons=[2023],
                )


if __name__ == "__main__":
    unittest.main()