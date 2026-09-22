from __future__ import annotations

import unittest

import pandas as pd

from scripts.validate_hf_kbo_pbp import (
    ValidationError,
    build_schema_rows,
    decide_status,
    get_sample_value,
    inspect_dataframe,
    normalize_seasons,
)


def make_valid_dataframe() -> pd.DataFrame:
    """정상 검증 케이스에 사용할 최소 PBP 형태 DataFrame을 만든다."""
    return pd.DataFrame(
        {
            "game_pk": [
                1001,
                1001,
            ],
            "game_date": [
                "2023-04-01",
                "2023-04-01",
            ],
            "batter": [
                10,
                11,
            ],
            "pitcher": [
                20,
                20,
            ],
            "at_bat_number": [
                1,
                2,
            ],
            "pitch_number": [
                1,
                1,
            ],
            "events": [
                None,
                "single",
            ],
            "pitch_type": [
                "FF",
                "SL",
            ],
            "release_speed_kmh": [
                145.1,
                132.4,
            ],
            "plate_x": [
                0.1,
                -0.2,
            ],
            "plate_z": [
                2.5,
                2.8,
            ],
        }
    )


class NormalizeSeasonsTest(
    unittest.TestCase
):
    """시즌 입력 정규화 규칙을 검증한다."""

    def test_remove_duplicate_seasons(
        self,
    ) -> None:
        self.assertEqual(
            normalize_seasons(
                [
                    2023,
                    2023,
                    2024,
                ]
            ),
            (
                2023,
                2024,
            ),
        )

    def test_reject_unsupported_season(
        self,
    ) -> None:
        with self.assertRaises(
            ValidationError
        ):
            normalize_seasons(
                [2027]
            )


class DataFrameInspectionTest(
    unittest.TestCase
):
    """DataFrame 품질 검증 핵심 규칙을 검증한다."""

    def test_valid_dataframe_has_no_fail_or_warn(
        self,
    ) -> None:
        result = inspect_dataframe(
            make_valid_dataframe(),
            2023,
        )

        self.assertEqual(
            result.fail_messages,
            [],
        )

        self.assertEqual(
            result.warn_messages,
            [],
        )

        self.assertEqual(
            result.metrics[
                "pitch_key_duplicate_rows"
            ],
            0,
        )

        self.assertEqual(
            result.metrics[
                "game_date_parse_failures"
            ],
            0,
        )

        self.assertEqual(
            result.metrics[
                "season_mismatch_rows"
            ],
            0,
        )

        self.assertEqual(
            result.metrics[
                "events_null_rate"
            ],
            0.5,
        )

    def test_missing_critical_column_is_fail(
        self,
    ) -> None:
        df = (
            make_valid_dataframe()
            .drop(
                columns=[
                    "game_pk"
                ]
            )
        )

        result = inspect_dataframe(
            df,
            2023,
        )

        self.assertIn(
            "game_pk",
            result.metrics[
                "missing_critical_columns"
            ],
        )

        self.assertTrue(
            any(
                "Critical column 누락"
                in message
                for message
                in result.fail_messages
            )
        )

    def test_invalid_date_and_season_mismatch_are_fail(
        self,
    ) -> None:
        df = (
            make_valid_dataframe()
        )

        df.loc[
            0,
            "game_date",
        ] = "invalid-date"

        df.loc[
            1,
            "game_date",
        ] = "2024-04-01"

        result = inspect_dataframe(
            df,
            2023,
        )

        self.assertEqual(
            result.metrics[
                "game_date_parse_failures"
            ],
            1,
        )

        self.assertEqual(
            result.metrics[
                "season_mismatch_rows"
            ],
            1,
        )

        self.assertTrue(
            any(
                "game_date 변환 실패"
                in message
                for message
                in result.fail_messages
            )
        )

        self.assertTrue(
            any(
                "시즌 불일치"
                in message
                for message
                in result.fail_messages
            )
        )

    def test_pitch_key_duplicate_is_fail(
        self,
    ) -> None:
        df = (
            make_valid_dataframe()
        )

        df.loc[
            1,
            [
                "game_pk",
                "at_bat_number",
                "pitch_number",
            ],
        ] = [
            1001,
            1,
            1,
        ]

        result = inspect_dataframe(
            df,
            2023,
        )

        self.assertEqual(
            result.metrics[
                "pitch_key_duplicate_rows"
            ],
            1,
        )

        self.assertTrue(
            any(
                "Pitch key 중복"
                in message
                for message
                in result.fail_messages
            )
        )

    def test_critical_null_is_fail(
        self,
    ) -> None:
        df = (
            make_valid_dataframe()
        )

        df.loc[
            0,
            "pitcher",
        ] = None

        result = inspect_dataframe(
            df,
            2023,
        )

        self.assertEqual(
            result.metrics[
                "pitcher_null_count"
            ],
            1,
        )

        self.assertTrue(
            any(
                "pitcher 결측"
                in message
                for message
                in result.fail_messages
            )
        )

    def test_all_null_analysis_column_is_warn(
        self,
    ) -> None:
        df = (
            make_valid_dataframe()
        )

        df["plate_x"] = None

        result = inspect_dataframe(
            df,
            2023,
        )

        self.assertEqual(
            result.metrics[
                "plate_x_null_rate"
            ],
            1.0,
        )

        self.assertTrue(
            any(
                "plate_x 컬럼이 전체 결측"
                in message
                for message
                in result.warn_messages
            )
        )

    def test_partial_analysis_null_does_not_warn(
        self,
    ) -> None:
        result = inspect_dataframe(
            make_valid_dataframe(),
            2023,
        )

        self.assertEqual(
            result.metrics[
                "events_null_rate"
            ],
            0.5,
        )

        self.assertFalse(
            any(
                "events 컬럼이 전체 결측"
                in message
                for message
                in result.warn_messages
            )
        )

    def test_schema_report_contains_all_columns(
        self,
    ) -> None:
        df = (
            make_valid_dataframe()
        )

        rows = build_schema_rows(
            df,
            2023,
        )

        self.assertEqual(
            len(rows),
            len(df.columns),
        )

        self.assertEqual(
            {
                row["column"]
                for row
                in rows
            },
            set(df.columns),
        )

    def test_sample_value_escapes_newline(
        self,
    ) -> None:
        series = pd.Series(
            [
                None,
                "first\nvalue",
                "second",
            ]
        )

        self.assertEqual(
            get_sample_value(
                series
            ),
            r"first\nvalue",
        )


class StatusDecisionTest(
    unittest.TestCase
):
    """PASS/WARN/FAIL 우선순위를 검증한다."""

    def test_fail_has_highest_priority(
        self,
    ) -> None:
        self.assertEqual(
            decide_status(
                ["fail"],
                ["warn"],
            ),
            "FAIL",
        )

    def test_warn_when_only_warning_exists(
        self,
    ) -> None:
        self.assertEqual(
            decide_status(
                [],
                ["warn"],
            ),
            "WARN",
        )

    def test_pass_when_no_messages_exist(
        self,
    ) -> None:
        self.assertEqual(
            decide_status(
                [],
                [],
            ),
            "PASS",
        )


if __name__ == "__main__":
    unittest.main()