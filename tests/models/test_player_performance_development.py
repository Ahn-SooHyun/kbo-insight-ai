"""M3 개선 후보·공정한 선정·공통 비율·시즌 경계 검증."""

from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from uuid import uuid4

from src.datasets.model_data import fingerprint
from src.datasets.published_sources import write_json
from src.models.player_performance import (
    SCHEMA,
    VERSION,
    data,
    development,
    evaluation,
    model,
)
from src.models.player_performance.__main__ import (
    compare,
    load_asset,
    method_settings,
    predict,
    runtime,
    source_hashes,
    validate_intervals,
)
from tests.models.test_player_performance import (
    counts,
    request,
    samples,
)


def metrics_result(
    mae: float,
    rmse: float,
) -> dict[str, float]:
    """선정 규칙만 검사하는 합성 평가 결과."""
    return {
        "normalized_macro_mae": mae,
        "normalized_macro_rmse": rmse,
    }


class CandidateTests(unittest.TestCase):
    """후보 사전·공동 구조·마스크·기존 예측 호환 검증."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.training = samples(
            "hitter",
            date(2023, 6, 5),
        )
        cls.family = development.fit_family(
            "hitter",
            cls.training,
            data.CALIBRATION_START,
        )

    def test_candidate_inventory(self) -> None:
        self.assertEqual(
            len(development.CANDIDATES), 11
        )
        self.assertEqual(
            len(self.family), 11
        )

        modes = [
            candidate.specification.mode
            for candidate in self.family.values()
        ]

        self.assertEqual(modes.count("history"), 3)
        self.assertEqual(modes.count("residual"), 6)
        self.assertEqual(modes.count("direct"), 2)

    def test_original_baseline_unchanged(self) -> None:
        candidate = self.family[
            development.BASELINE_NAME
        ]
        player = self.training[0].player

        self.assertEqual(
            candidate.predict(player),
            candidate.baseline.predict(player),
        )

    def test_original_full_residual_tree_unchanged(self) -> None:
        player = self.training[0].player

        for depth in model.CANDIDATE_DEPTHS:
            with self.subTest(depth=depth):
                name = (
                    f"joint_residual_tree_depth_{depth}"
                )

                original = model.fit(
                    "hitter",
                    self.training,
                    data.CALIBRATION_START,
                    depth,
                )

                self.assertEqual(
                    original.predict(player),
                    self.family[name].predict(player),
                )

    def test_shrink_candidates_share_tree(self) -> None:
        original = self.family[
            "joint_residual_tree_depth_2"
        ]
        shrink = self.family[
            "joint_residual_tree_depth_2_shrink_25"
        ]

        self.assertIs(
            original.tree,
            shrink.tree,
        )
        self.assertIs(
            original.baseline,
            shrink.baseline,
        )

    def test_shrink_applies_before_projection(self) -> None:
        candidate = self.family[
            "joint_residual_tree_depth_2_shrink_25"
        ]
        player = self.training[0].player
        baseline = candidate.baseline.predict(
            player
        )

        residuals = candidate.tree.predict(
            model.features(player)
        )

        raw = tuple(
            base + 0.25 * residual * scale
            for base, residual, scale in zip(
                baseline,
                residuals,
                candidate.baseline.scales,
            )
        )

        self.assertEqual(
            candidate.predict(player),
            model.project(player, raw),
        )

    def test_direct_training_uses_observation_mask(self) -> None:
        sample = self.training[0]
        cutoff = (
            sample.player.anchor
            + timedelta(days=7)
        )

        baseline = self.family[
            development.BASELINE_NAME
        ].baseline

        rows = development.direct_training_rows(
            [sample],
            baseline,
            cutoff,
        )

        self.assertEqual(len(rows), 1)

        width = len(data.metrics("hitter"))
        observed = dict(rows[0].residuals)

        self.assertEqual(
            set(observed),
            set(range(width)),
        )

        truth = data.labels_for(
            sample, cutoff
        )

        for index in range(width):
            self.assertAlmostEqual(
                observed[index],
                truth[index] / baseline.scales[index],
            )

    def test_direct_tree_joint_support(self) -> None:
        candidate = self.family[
            "joint_direct_tree_depth_2"
        ]

        self.assertEqual(
            len(candidate.tree.values), 24
        )
        self.assertEqual(
            len(candidate.tree.support), 24
        )
        self.assertTrue(
            all(
                count == len(self.training)
                for count in candidate.tree.support
            )
        )

    def test_pitcher_joint_dimensions_for_all_candidates(
        self,
    ) -> None:
        training = samples(
            "pitcher",
            date(2023, 6, 5),
        )
        family = development.fit_family(
            "pitcher",
            training,
            data.CALIBRATION_START,
        )
        player = training[0].player

        for name, candidate in family.items():
            with self.subTest(candidate=name):
                self.assertEqual(
                    len(candidate.predict(player)),
                    30,
                )

    def test_validation_cannot_be_used_for_fit(self) -> None:
        validation = samples(
            "hitter",
            date(2025, 6, 2),
            30,
        )

        with self.assertRaises(ValueError):
            development.fit_family(
                "hitter",
                validation,
                data.TEST_START,
            )

    def test_rejected_history_backoff(self) -> None:
        row = request()
        row["prior_rejected"] = 1

        for prefix in ("season", "recent28"):
            for name in data.metrics("hitter"):
                row[f"{prefix}_{name}"] = (
                    0 if name == "games" else None
                )

        player = data.parse_input(row)

        original = self.family[
            development.BASELINE_NAME
        ]
        improved = self.family[
            "history_balanced_50"
        ]

        self.assertEqual(
            original.predict(player)[0], 0
        )
        self.assertGreater(
            improved.predict(player)[0], 0
        )
        self.assertIsNone(
            player.current_counts
        )


class SelectionTests(unittest.TestCase):
    """기준 모델 대비 채택 조건과 후퇴 규칙 검증."""

    def test_worse_candidates_retain_baseline(self) -> None:
        results = {
            development.BASELINE_NAME:
                metrics_result(0.4, 0.5),
            "joint_residual_tree_depth_2":
                metrics_result(0.45, 0.6),
            "joint_residual_tree_depth_4":
                metrics_result(0.42, 0.52),
        }

        selected, decisions = (
            development.choose_candidate(results)
        )

        self.assertEqual(
            selected,
            development.BASELINE_NAME,
        )
        self.assertFalse(
            decisions[
                "joint_residual_tree_depth_2"
            ]["eligible"]
        )

    def test_mae_improvement_with_worse_rmse_rejected(
        self,
    ) -> None:
        results = {
            development.BASELINE_NAME:
                metrics_result(0.4, 0.5),
            "joint_residual_tree_depth_2":
                metrics_result(0.3, 0.6),
        }

        selected, decisions = (
            development.choose_candidate(results)
        )

        self.assertEqual(
            selected,
            development.BASELINE_NAME,
        )
        self.assertTrue(
            decisions[
                "joint_residual_tree_depth_2"
            ]["strict_mae_improvement"]
        )
        self.assertFalse(
            decisions[
                "joint_residual_tree_depth_2"
            ]["nonworse_rmse"]
        )

    def test_improved_candidate_selected(self) -> None:
        results = {
            development.BASELINE_NAME:
                metrics_result(0.4, 0.5),
            "joint_residual_tree_depth_2":
                metrics_result(0.39, 0.48),
            "joint_residual_tree_depth_4":
                metrics_result(0.38, 0.47),
        }

        selected, decisions = (
            development.choose_candidate(results)
        )

        self.assertEqual(
            selected,
            "joint_residual_tree_depth_4",
        )
        self.assertTrue(
            decisions[selected][
                "passes_improvement_gate"
            ]
        )

    def test_equal_metrics_retain_baseline(self) -> None:
        results = {
            development.BASELINE_NAME:
                metrics_result(0.4, 0.5),
            "history_balanced_50":
                metrics_result(0.4, 0.5),
        }

        selected, _ = (
            development.choose_candidate(results)
        )

        self.assertEqual(
            selected,
            development.BASELINE_NAME,
        )


class CommonRatioTests(unittest.TestCase):
    """분모가 서로 다른 후보의 동일 표본 비율 평가."""

    def test_zero_denominator_excluded_from_both_models(
        self,
    ) -> None:
        player = data.parse_input(request())

        comparison = development.RatioComparison(
            ("baseline", "candidate")
        )

        comparison.add(
            0.3,
            10.0,
            {
                "baseline": (3.0, 10.0),
                "candidate": (0.0, 0.0),
            },
            player,
        )

        comparison.add(
            0.4,
            10.0,
            {
                "baseline": (3.0, 10.0),
                "candidate": (4.0, 10.0),
            },
            player,
        )

        result = comparison.result(2)

        self.assertEqual(
            result["truth_valid_n"], 2
        )
        self.assertEqual(
            result["common_n"], 1
        )
        self.assertEqual(
            result[
                "excluded_by_prediction_denominator_n"
            ],
            1,
        )

        baseline = result["models"]["baseline"]
        candidate = result["models"]["candidate"]

        self.assertEqual(
            baseline["n"], candidate["n"]
        )
        self.assertEqual(
            candidate[
                "zero_prediction_denominator_n"
            ],
            1,
        )
        self.assertAlmostEqual(
            baseline["mae"], 0.1
        )
        self.assertAlmostEqual(
            candidate["mae"], 0.0
        )

    def test_no_common_rows_not_zero_error(self) -> None:
        player = data.parse_input(request())

        comparison = development.RatioComparison(
            ("baseline", "candidate")
        )

        comparison.add(
            0.3,
            10.0,
            {
                "baseline": (3.0, 10.0),
                "candidate": (0.0, 0.0),
            },
            player,
        )

        result = comparison.result(1)

        self.assertEqual(
            result["common_n"], 0
        )
        self.assertIsNone(
            result["models"]["baseline"]["mae"]
        )
        self.assertIsNone(
            result["models"]["candidate"]["mae"]
        )

    def test_all_and_pairwise_comparison_inventory(
        self,
    ) -> None:
        training = samples(
            "hitter",
            date(2023, 6, 5),
        )
        validation = samples(
            "hitter",
            date(2025, 6, 2),
            30,
        )

        family = development.fit_family(
            "hitter",
            training,
            data.CALIBRATION_START,
        )

        result = development.compare_ratios(
            family,
            validation,
            "hitter",
            data.TEST_START,
        )

        self.assertEqual(
            len(result["all_candidates_common"]),
            6,
        )
        self.assertEqual(
            len(result["baseline_pairwise_common"]),
            10,
        )

        self.assertEqual(
            result["truth_policy"]["7d"],
            "count_derived_diagnostic",
        )

        for row in result[
            "all_candidates_common"
        ].values():
            self.assertEqual(
                len(row["models"]), 11
            )
            self.assertTrue(
                all(
                    metric["n"] == row["common_n"]
                    for metric in row["models"].values()
                )
            )


class OutputAuditTests(unittest.TestCase):
    """시즌 전망·사전 종료 시나리오·출력 계약 검증."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.training = samples(
            "hitter",
            date(2023, 6, 5),
        )
        cls.family = development.fit_family(
            "hitter",
            cls.training,
            data.CALIBRATION_START,
        )
        cls.intervals = {
            key: {
                "radius": 2.0,
                "reason": None,
            }
            for key in data.output_keys("hitter")
        }

    def test_all_candidates_known_end_contract(
        self,
    ) -> None:
        for name, fitted in self.family.items():
            for days in development.SCENARIO_DAYS:
                with self.subTest(
                    candidate=name,
                    days=days,
                ):
                    row = request()
                    anchor = date.fromisoformat(
                        row["sample_date"]
                    )

                    row.update({
                        "known_end_exclusive": (
                            anchor + timedelta(days=days)
                        ).isoformat(),
                        "known_end_as_of": (
                            anchor - timedelta(days=1)
                        ).isoformat(),
                        "known_end_source":
                            "synthetic_test",
                    })

                    player = data.parse_input(row)

                    result = model.format_prediction(
                        fitted,
                        player,
                        self.intervals,
                    )

                    periods, _ = (
                        development.verify_prediction(
                            player, result
                        )
                    )

                    if days <= 7:
                        self.assertEqual(
                            periods["7d"],
                            periods["28d"],
                        )
                    if days <= 28:
                        self.assertEqual(
                            periods["28d"],
                            periods["remaining"],
                        )

    def test_caller_verified_current_cumulative(
        self,
    ) -> None:
        row = request()
        row.update({
            "current_counts":
                counts("hitter", 20),
            "current_scope":
                "caller_verified",
            "current_as_of":
                row["sample_date"],
        })

        player = data.parse_input(row)

        fitted = self.family[
            "joint_direct_tree_depth_2"
        ]

        result = model.format_prediction(
            fitted,
            player,
            self.intervals,
        )

        periods, final = (
            development.verify_prediction(
                player, result
            )
        )

        for name in data.metrics("hitter"):
            self.assertAlmostEqual(
                final[name],
                row["current_counts"][name]
                + periods["remaining"][name],
            )

        self.assertEqual(
            result["season_projection"]["status"],
            "caller_verified_cumulative_projection",
        )

    def test_missing_current_cumulative_not_official_total(
        self,
    ) -> None:
        row = request()
        row["prior_rejected"] = 1

        for prefix in ("season", "recent28"):
            for name in data.metrics("hitter"):
                row[f"{prefix}_{name}"] = (
                    0 if name == "games" else None
                )

        player = data.parse_input(row)

        fitted = self.family[
            "history_balanced_50"
        ]
        result = model.format_prediction(
            fitted,
            player,
            self.intervals,
        )

        _, final = (
            development.verify_prediction(
                player, result
            )
        )

        self.assertIsNone(final)
        self.assertEqual(
            result["season_projection"]["status"],
            "unavailable",
        )

    def test_tampered_ratio_rejected(self) -> None:
        player = self.training[0].player
        fitted = self.family[
            development.BASELINE_NAME
        ]

        result = model.format_prediction(
            fitted,
            player,
            self.intervals,
        )
        result["periods"]["7d"]["derived"][
            "hit_rate"
        ]["numerator"] = -1

        with self.assertRaises(ValueError):
            development.verify_prediction(
                player, result
            )

    def test_tampered_interval_rejected(self) -> None:
        player = self.training[0].player
        fitted = self.family[
            development.BASELINE_NAME
        ]

        result = model.format_prediction(
            fitted,
            player,
            self.intervals,
        )
        result["periods"]["7d"]["intervals"][
            "games"
        ]["upper"] = -1

        with self.assertRaises(ValueError):
            development.verify_prediction(
                player, result
            )

    def test_tampered_season_sum_rejected(self) -> None:
        player = self.training[0].player
        fitted = self.family[
            development.BASELINE_NAME
        ]

        result = model.format_prediction(
            fitted,
            player,
            self.intervals,
        )
        result["season_projection"][
            "counts"
        ]["games"] += 10

        with self.assertRaises(ValueError):
            development.verify_prediction(
                player, result
            )


class DevelopmentTests(unittest.TestCase):
    """전체 시간순 개발 및 신규 자산 재현 검증."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.training = samples(
            "hitter",
            date(2023, 6, 5),
        )
        cls.calibration = samples(
            "hitter",
            date(2024, 6, 3),
            30,
        )
        cls.validation = samples(
            "hitter",
            date(2025, 6, 2),
            30,
        )

        cls.family = development.fit_family(
            "hitter",
            cls.training,
            data.CALIBRATION_START,
        )

    def test_complete_development_both_roles(self) -> None:
        for role in ("hitter", "pitcher"):
            with self.subTest(role=role):
                training = samples(
                    role,
                    date(2023, 6, 5),
                )
                calibration = samples(
                    role,
                    date(2024, 6, 3),
                    30,
                )
                validation = samples(
                    role,
                    date(2025, 6, 2),
                    30,
                )

                dataset = data.Dataset(
                    train=training + calibration,
                    validation=validation,
                    provenance={},
                    audit={},
                )

                result = development.develop(
                    dataset, role
                )
                report = result.report

                self.assertEqual(
                    report["candidate_count"], 11
                )
                self.assertEqual(
                    report["operating_predictors"], 1
                )
                self.assertFalse(
                    report["final_test_used"]
                )
                self.assertEqual(
                    report["selected_output_audit"][
                        "verified_validation_predictions"
                    ],
                    30,
                )

                candidates = report["candidates"]
                baseline = candidates[
                    development.BASELINE_NAME
                ]["validation"]
                chosen = candidates[
                    report["selected_model"]
                ]["validation"]

                self.assertLessEqual(
                    chosen["normalized_macro_mae"],
                    baseline["normalized_macro_mae"]
                    + development.SELECTION_TOLERANCE,
                )
                self.assertLessEqual(
                    chosen["normalized_macro_rmse"],
                    baseline["normalized_macro_rmse"]
                    + development.SELECTION_TOLERANCE,
                )

    def test_asset_roundtrip_all_predictor_types(
        self,
    ) -> None:
        root = Path(
            __file__
        ).resolve().parents[2]

        hashes = source_hashes(root)
        player = self.training[0].player

        variants = (
            development.BASELINE_NAME,
            "history_balanced_50",
            "joint_residual_tree_depth_2_shrink_25",
            "joint_direct_tree_depth_2",
        )

        with TemporaryDirectory() as directory:
            folder = Path(directory)

            for index, name in enumerate(variants):
                with self.subTest(candidate=name):
                    fitted = self.family[name]

                    intervals = evaluation.calibrate(
                        fitted,
                        self.calibration,
                        data.VALIDATION_START,
                    )

                    payload = {
                        "schema": SCHEMA,
                        "version": VERSION,
                        "run_id": uuid4().hex,
                        "role": "hitter",
                        "runtime": runtime(),
                        "method": method_settings(),
                        "source_hashes": hashes,
                        "features": list(
                            model.feature_names("hitter")
                        ),
                        "outputs": list(
                            data.output_keys("hitter")
                        ),
                        "selected_model":
                            fitted.name,
                        "selected_is_tree":
                            fitted.tree is not None,
                        "model":
                            fitted.to_payload(),
                        "intervals": intervals,
                        "final_test_used": False,
                    }

                    asset = folder / f"model-{index}.json"
                    input_path = (
                        folder / f"input-{index}.json"
                    )
                    expected_path = (
                        folder / f"expected-{index}.json"
                    )
                    actual_path = (
                        folder / f"actual-{index}.json"
                    )

                    write_json(
                        asset,
                        {
                            "payload": payload,
                            "fingerprint":
                                fingerprint(payload),
                        },
                    )

                    restored, loaded_intervals, info = (
                        load_asset(asset)
                    )

                    self.assertEqual(
                        fitted.name, restored.name
                    )
                    self.assertEqual(
                        fitted.predict(player),
                        restored.predict(player),
                    )
                    self.assertEqual(
                        intervals,
                        loaded_intervals,
                    )
                    self.assertEqual(
                        info["operating_predictors"],
                        1,
                    )

                    expected = model.format_prediction(
                        fitted,
                        player,
                        intervals,
                    )

                    write_json(
                        input_path,
                        self.training[0].request,
                    )
                    write_json(
                        expected_path,
                        expected,
                    )

                    predict(
                        asset,
                        input_path,
                        actual_path,
                    )

                    self.assertTrue(
                        compare(
                            expected_path,
                            actual_path,
                        )["equal"]
                    )

    def test_wrong_candidate_depth_rejected(
        self,
    ) -> None:
        payload = deepcopy(
            self.family[
                "joint_direct_tree_depth_2"
            ].to_payload()
        )

        payload["depth"] = 4

        with self.assertRaises(ValueError):
            development.restore(payload)

    def test_history_asset_cannot_contain_tree(
        self,
    ) -> None:
        payload = deepcopy(
            self.family[
                development.BASELINE_NAME
            ].to_payload()
        )

        payload["tree"] = (
            self.family[
                "joint_direct_tree_depth_2"
            ].to_payload()["tree"]
        )

        with self.assertRaises(ValueError):
            development.restore(payload)

    def test_invalid_interval_quantile_rank_rejected(
        self,
    ) -> None:
        fitted = self.family[
            development.BASELINE_NAME
        ]
        intervals = evaluation.calibrate(
            fitted,
            self.calibration,
            data.VALIDATION_START,
        )

        intervals["7d.games"][
            "quantile_rank"
        ] = 1

        with self.assertRaises(ValueError):
            validate_intervals(
                "hitter",
                intervals,
            )


if __name__ == "__main__":
    unittest.main()