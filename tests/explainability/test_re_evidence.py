"""잔여 기대득점의 실제 통계 선택 및 종료 근거 검증."""

from dataclasses import asdict
import unittest

from src.explainability import contributions as c
from src.models.game_state_value import model as m4
from src.models.game_state_value import value as v4


def make_state(**changes: object) -> m4.State:
    """기존 M4 입력 계약을 만족하는 합성 경기 상태를 생성한다."""
    raw = {
        "season": 2024,
        "competition": "regular",
        "home_team_id": 2001,
        "away_team_id": 2002,
        "perspective_team_id": 2001,
        "inning": 9,
        "half": "bot",
        "phase": "pre",
        "home_score": 3,
        "away_score": 3,
        "outs": 0,
        "on_first": 0,
        "on_second": 0,
        "on_third": 0,
        "home_season_games": 100,
        "home_season_rf": 500,
        "home_season_ra": 450,
        "home_season_wins": 55,
        "away_season_games": 100,
        "away_season_rf": 450,
        "away_season_ra": 500,
        "away_season_wins": 45,
    }

    raw.update(changes)

    return m4.parse_state(raw)


class RemainingRunsEvidenceTests(unittest.TestCase):
    """RE 통계의 실제 선택 키와 예측값 일치 여부를 확인한다."""

    def setUp(self) -> None:
        self.state = make_state()

        (
            self.exact,
            self.base_out,
            self.outs,
            self.global_key,
        ) = v4.re_keys(self.state)

    def test_exact_statistics_selected(self) -> None:
        table = {
            self.exact: [30, 45],
            self.base_out: [100, 100],
            self.outs: [200, 200],
            self.global_key: [1000, 800],
        }

        expected = v4.predict_re(
            table,
            self.state,
        )

        result = c.re_evidence(
            table,
            self.state,
            None,
            expected,
        )

        self.assertEqual(
            result["selected_key"],
            self.exact,
        )

        self.assertEqual(
            result["selected_entry"],
            [30, 45],
        )

        self.assertEqual(
            result["selected_support"],
            30,
        )

        self.assertEqual(
            result["selected_runs_sum"],
            45,
        )

        self.assertEqual(
            result["reconstructed_re"],
            1.5,
        )

        self.assertEqual(
            result["max_absolute_error"],
            0.0,
        )

        self.assertEqual(
            result["lookup_trace"][0]["status"],
            "selected",
        )

    def test_sparse_exact_falls_back_to_base_out(self) -> None:
        table = {
            self.exact: [29, 60],
            self.base_out: [100, 75],
            self.outs: [200, 200],
            self.global_key: [1000, 800],
        }

        result = c.re_evidence(
            table,
            self.state,
            None,
            v4.predict_re(
                table,
                self.state,
            ),
        )

        self.assertEqual(
            result["selected_key"],
            self.base_out,
        )

        self.assertEqual(
            result["reconstructed_re"],
            0.75,
        )

        self.assertEqual(
            [
                item["status"]
                for item in result["lookup_trace"]
            ],
            [
                "insufficient_support",
                "selected",
            ],
        )

    def test_missing_groups_fall_back_to_outs(self) -> None:
        table = {
            self.outs: [200, 100],
            self.global_key: [1000, 800],
        }

        result = c.re_evidence(
            table,
            self.state,
            None,
            v4.predict_re(
                table,
                self.state,
            ),
        )

        self.assertEqual(
            result["selected_key"],
            self.outs,
        )

        self.assertEqual(
            result["reconstructed_re"],
            0.5,
        )

        self.assertEqual(
            [
                item["status"]
                for item in result["lookup_trace"]
            ],
            [
                "missing",
                "missing",
                "selected",
            ],
        )

    def test_global_fallback_below_minimum_support(self) -> None:
        table = {
            self.exact: [5, 10],
            self.base_out: [10, 15],
            self.outs: [15, 20],
            self.global_key: [20, 10],
        }

        result = c.re_evidence(
            table,
            self.state,
            None,
            v4.predict_re(
                table,
                self.state,
            ),
        )

        self.assertEqual(
            result["selected_key"],
            "global",
        )

        self.assertEqual(
            result["selected_support"],
            20,
        )

        self.assertEqual(
            result["reconstructed_re"],
            0.5,
        )

        self.assertEqual(
            [
                item["status"]
                for item in result["lookup_trace"]
            ],
            [
                "insufficient_support",
                "insufficient_support",
                "insufficient_support",
                "selected",
            ],
        )

    def test_three_outs_is_deterministic(self) -> None:
        state = make_state(
            phase="post",
            outs=3,
        )

        table = {
            "global": [1000, 800],
        }

        result = c.re_evidence(
            table,
            state,
            None,
            v4.predict_re(
                table,
                state,
            ),
        )

        self.assertEqual(
            result["status"],
            "deterministic",
        )

        self.assertEqual(
            result["reason"],
            "three_outs_half_ended",
        )

        self.assertIsNone(
            result["selected_key"]
        )

        self.assertIsNone(
            result["selected_entry"]
        )

        self.assertEqual(
            result["reconstructed_re"],
            0.0,
        )

        self.assertEqual(
            result["lookup_trace"],
            [],
        )

    def test_confirmed_partial_final_is_deterministic(self) -> None:
        state = make_state(
            phase="post",
            home_score=4,
            outs=1,
        )

        table = {
            "global": [1000, 800],
        }

        result = c.re_evidence(
            table,
            state,
            1,
            v4.predict_re(
                table,
                state,
                1,
            ),
        )

        self.assertEqual(
            result["reason"],
            "confirmed_game_final",
        )

        self.assertIsNone(
            result["selected_entry"]
        )

        self.assertEqual(
            result["reconstructed_re"],
            0.0,
        )

    def test_confirmed_nonwin_is_not_missing(self) -> None:
        state = make_state(
            phase="post",
            outs=3,
        )

        result = c.re_evidence(
            {},
            state,
            0,
            (0.0, "half_ended", None),
        )

        self.assertEqual(
            result["reason"],
            "confirmed_game_final",
        )

    def test_wrong_expected_runs_rejected(self) -> None:
        table = {
            self.exact: [30, 45],
            "global": [1000, 800],
        }

        with self.assertRaises(ValueError):
            c.re_evidence(
                table,
                self.state,
                None,
                (0.5, "exact", 30),
            )

    def test_wrong_expected_support_rejected(self) -> None:
        table = {
            self.exact: [30, 45],
            "global": [1000, 800],
        }

        with self.assertRaises(ValueError):
            c.re_evidence(
                table,
                self.state,
                None,
                (1.5, "exact", 31),
            )

    def test_wrong_expected_level_rejected(self) -> None:
        table = {
            self.exact: [30, 45],
            "global": [1000, 800],
        }

        with self.assertRaises(ValueError):
            c.re_evidence(
                table,
                self.state,
                None,
                (1.5, "global", 30),
            )

    def test_zero_support_rejected(self) -> None:
        table = {
            "global": [0, 0],
        }

        with self.assertRaises(ValueError):
            c.re_evidence(
                table,
                self.state,
                None,
                (0.0, "global", 0),
            )

    def test_missing_statistics_rejected(self) -> None:
        with self.assertRaises(ValueError):
            c.re_evidence(
                {},
                self.state,
                None,
                (0.0, "global", 1),
            )

    def test_terminal_prediction_mismatch_rejected(self) -> None:
        state = make_state(
            phase="post",
            outs=3,
        )

        with self.assertRaises(ValueError):
            c.re_evidence(
                {},
                state,
                None,
                (0.5, "half_ended", None),
            )

    def test_local_explanation_contains_selected_statistics(self) -> None:
        width = len(c.M4_FEATURE_NAMES)

        predictor = {
            "name": "logistic_core",
            "weights": [0.0] * (width + 1),
            "means": [0.0] * width,
            "scales": [1.0] * width,
        }

        table = {
            self.exact: [30, 45],
            "global": [1000, 800],
        }

        operating = {
            "predictor": predictor,
            "re_table": table,
        }

        row = {
            "input": asdict(self.state),
            "operating": {
                "win": m4.predict_win(
                    predictor,
                    self.state,
                ),
                "re": v4.predict_re(
                    table,
                    self.state,
                ),
            },
        }

        truth = {
            "final_evidence": None,
            "confirmed_final": False,
        }

        linear = c.build_linear(
            "m4",
            operating,
            {
                "means": dict.fromkeys(
                    c.M4_FEATURE_NAMES,
                    0.0,
                ),
            },
        )

        result = c.explain_m4(
            operating,
            row,
            truth,
            linear,
            aggregate=True,
            pair=None,
        )

        self.assertEqual(
            result["re_selected_key"],
            self.exact,
        )

        self.assertEqual(
            result["re_selected_entry"],
            [30, 45],
        )

        self.assertEqual(
            result["evaluated_re_evidence"][
                "reconstructed_re"
            ],
            result["evaluated"][
                "remaining_expected_runs"
            ],
        )

        self.assertEqual(
            result["raw_re_evidence"][
                "selected_key"
            ],
            self.exact,
        )

        self.assertEqual(
            result["max_absolute_error"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
