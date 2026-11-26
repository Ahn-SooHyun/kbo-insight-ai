"""M1 백테스트의 시간 순서, 비교 표본과 회귀 후보를 검증한다."""

from dataclasses import replace
from math import fsum
import unittest

from src.models.game_outcome import backtest, poisson
from src.models.game_outcome import model as m1

from .test_game_outcome import inputs


def games() -> list[m1.Game]:
    """세 검증 시즌과 그보다 앞선 학습 경기를 만든다."""
    rows = (
        ("2022-04-02", 5, 4),
        ("2022-04-03", 3, 3),
        ("2023-04-01", 6, 2),
        ("2023-04-02", 4, 5),
        ("2024-04-01", 2, 6),
        ("2024-04-02", 3, 3),
        ("2025-04-01", 8, 5),
        ("2025-04-02", 4, 4),
    )
    return [
        m1.Game(day, day, inputs(), (home, away), m1.score_outcome(home, away))
        for day, home, away in rows
    ]


class GameOutcomeBacktestTests(unittest.TestCase):
    def test_expanding_windows_and_reserved_test(self) -> None:
        development = games()
        folds = backtest.season_folds(development)
        self.assertEqual([(year, len(train), len(validation)) for year, train, validation in folds],
                         [(2023, 2, 2), (2024, 4, 2), (2025, 6, 2)])
        for _, train, validation in folds:
            self.assertLess(max(game.day for game in train), min(game.day for game in validation))
        future = replace(development[-1], sample_id="future", day="2026-04-01")
        with self.assertRaisesRegex(ValueError, "최종 Test"):
            backtest.season_folds(development + [future])

    def test_fit_sees_only_earlier_labels(self) -> None:
        development = games()
        changed = development[:]
        changed[2] = replace(changed[2], runs=(4, 4), outcome="tie")
        first = backtest.benchmark(development)
        second = backtest.benchmark(changed)
        self.assertEqual(first["seasons"][0]["fit"], second["seasons"][0]["fit"])
        self.assertNotEqual(first["seasons"][0]["metrics"], second["seasons"][0]["metrics"])
        self.assertNotEqual(first["seasons"][1]["fit"], second["seasons"][1]["fit"])

    def test_venue_baseline_uses_same_tie_estimation(self) -> None:
        train = games()[:2]
        forecast, weight = backtest.venue_forecast(m1.fit(train), train)
        expected = (1 + 0.5) / (2 + 1)
        self.assertAlmostEqual(forecast["result_probabilities"]["tie"], expected, places=12)
        self.assertEqual(forecast["joint"]["equal_score_weight"], weight)
        self.assertAlmostEqual(fsum(forecast["result_probabilities"].values()), 1.0, places=12)

    def test_optional_poisson_uses_same_validation_games(self) -> None:
        development = games()
        default = backtest.benchmark(development)
        expanded = backtest.benchmark(development, include_poisson=True)
        self.assertEqual(default["pooled_games"], 6)
        for name in (backtest.BASELINE, backtest.PROFILE):
            self.assertEqual(default["pooled_metrics"][name], expanded["pooled_metrics"][name])
        candidate = expanded["pooled_metrics"][backtest.POISSON]
        self.assertEqual(candidate["outcomes"]["games"], 6)
        self.assertEqual(candidate["scores"]["pooled"]["team_rows"], 12)
        self.assertIn("poisson_regression_minus_profile_histogram", expanded["pooled_deltas"])

    def test_poisson_learns_home_effect_and_bounds_tail(self) -> None:
        train = [
            replace(game, runs=(9, 2), outcome="home")
            for game in games()[:4]
        ]
        candidate = poisson.fit(train, mean=5.5, histogram_cap=m1.MIN_CAP)
        forecast = poisson.predict(candidate, inputs())
        self.assertGreater(candidate.coefficients[1], 0)
        self.assertGreater(forecast["home_expected_runs"], forecast["away_expected_runs"])
        self.assertAlmostEqual(fsum(forecast["home_pmf"]), 1.0)
        self.assertAlmostEqual(fsum(forecast["result_probabilities"].values()), 1.0)
        self.assertLess(forecast["tail"]["omitted_joint_mass_bound"], 1e-12)
        self.assertLess(forecast["tail"]["expected_runs_error_bound_per_side"], 1e-10)