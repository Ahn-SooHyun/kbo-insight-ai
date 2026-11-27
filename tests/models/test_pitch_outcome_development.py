"""M5 시간순 보정·규제·후보 선정·독립 재현 추가 검증."""

from collections import Counter
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from math import log, sqrt
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import cast
import unittest
from unittest.mock import patch

from src.datasets.model_data import fingerprint
from src.datasets.published_sources import sha256_file
from src.evaluation import pitch_outcome as evaluation
from src.models.pitch_outcome import SCHEMA, VERSION
from src.models.pitch_outcome import data, development
from src.models.pitch_outcome import model as m5
from src.models.pitch_outcome.__main__ import (
    compare_predictions,
    load_payload,
    main,
    method_settings,
    save_json,
    train,
    verify_run,
)
from src.settings import Settings

from .test_pitch_outcome import (
    example_input,
    example_sample,
)


def training_samples() -> list[m5.Sample]:
    """구종·결과와 활성 특징이 다른 작은 합성 Train 자료."""
    second = {
        **example_input(),
        "batter_id": "batter-2",
        "pitcher_id": "pitcher-2",
        "pitch_number": 2,
        "balls": 1,
        "previous_pitch_type": "FF",
        "previous_pitch_result": "B",
    }

    third = {
        **example_input(),
        "strikes": 1,
        "pitch_number": 2,
        "previous_pitch_type": "SL",
        "previous_pitch_result": "S",
    }

    return [
        example_sample("FF:B"),
        example_sample("SL:S", second),
        example_sample("CU:H", third),
        example_sample("FF:T"),
    ]


def dense_regularized_reference(
    samples: list[m5.Sample],
    seed: int,
) -> dict[str, list[float]]:
    """작은 자료에서 모든 계수를 매번 갱신하는 L2 기준 계산."""
    counts = m5.training_counts(samples)

    occurrences = Counter(
        key
        for sample in samples
        for key in sample.keys
    )

    vocabulary = {
        key
        for key, count in occurrences.items()
        if not key.startswith("id:")
        or count >= m5.MIN_ID_COUNT
    }

    vocabulary.update((
        m5.BATTER_OTHER,
        m5.PITCHER_OTHER,
    ))

    weights = {
        key: [0.0] * m5.N_CLASSES
        for key in sorted(vocabulary)
    }

    weights["bias"] = [
        log(value)
        for value in m5.global_prior(counts)
    ]

    averages = {
        key: [0.0] * m5.N_CLASSES
        for key in weights
    }

    rng = Random(seed)
    order = list(range(len(samples)))

    for epoch in range(m5.EPOCHS):
        rng.shuffle(order)

        rate = m5.LEARNING_RATE / sqrt(epoch + 1)
        decay = 1.0 - rate * development.L2

        for index in order:
            sample = samples[index]

            names = set(
                development.active_names(
                    weights, sample.keys
                )
            )

            probabilities = m5.softmax(
                m5.logits_for(
                    m5.active_weights(weights, sample.keys)
                )
            )

            gradient = [
                -rate * value
                for value in probabilities
            ]

            gradient[sample.target] += rate

            # 실제 구현의 지연 감쇠와 달리 모든 특징에 즉시 적용한다.
            for name, coefficients in weights.items():
                factor = (
                    1.0 if name == "bias" else decay
                )

                coefficients[:] = [
                    factor * weight
                    + (update if name in names else 0.0)
                    for weight, update in zip(
                        coefficients, gradient
                    )
                ]

        if epoch >= m5.EPOCHS - development.AVERAGE_EPOCHS:
            for name, coefficients in weights.items():
                averages[name] = [
                    total + value
                    for total, value in zip(
                        averages[name], coefficients
                    )
                ]

    return {
        name: [
            value / development.AVERAGE_EPOCHS
            for value in coefficients
        ]
        for name, coefficients in averages.items()
    }


def mock_metric(
    loss: float,
    brier: float,
    top_ece: float,
    macro_ece: float,
    samples: int = 10,
) -> dict[str, object]:
    """후보 선정 로직만 독립 검사하는 합성 평가 결과."""
    return {
        "joint": {
            "samples": samples,
            "log_loss": loss,
            "brier": brier,
            "top_label_ece": top_ece,
            "classes": {
                "FF:B": {
                    "one_vs_rest_ece": macro_ece,
                },
            },
        },
    }


def mock_selection_results() -> dict[str, dict[str, object]]:
    """보정 개선 및 보정 악화 후보를 함께 구성한다."""
    return {
        "frequency": mock_metric(
            3.0, 0.95, 0.02, 0.01
        ),
        "conditional_frequency": mock_metric(
            2.9, 0.94, 0.02, 0.01
        ),
        "logistic": mock_metric(
            2.7, 0.92, 0.05, 0.03
        ),
        "logistic_temperature": mock_metric(
            2.65, 0.91, 0.03, 0.02
        ),
        "regularized": mock_metric(
            2.68, 0.915, 0.04, 0.02
        ),
        "regularized_temperature": mock_metric(
            2.64, 0.91, 0.06, 0.03
        ),
    }


class PitchDevelopmentPartitionTests(unittest.TestCase):
    def test_period_boundaries(self) -> None:
        self.assertEqual(
            development.period_for("2025-06-30"),
            "calibration",
        )

        self.assertEqual(
            development.period_for("2025-07-01"),
            "selection",
        )

        self.assertEqual(
            development.period_for("2025-08-31"),
            "selection",
        )

        self.assertEqual(
            development.period_for("2025-09-01"),
            "reporting",
        )

        for day in ("2024-12-31", "2026-01-01"):
            with self.subTest(day=day):
                with self.assertRaises(ValueError):
                    development.period_for(day)

    def test_complete_partition(self) -> None:
        samples = training_samples()

        metadata = [
            ("G1", 1, 1, "2025-06-30"),
            ("G1", 1, 2, "2025-06-30"),
            ("G2", 1, 1, "2025-07-01"),
            ("G3", 1, 1, "2025-09-01"),
        ]

        partition = development.partition_rows(
            samples, metadata
        )

        self.assertEqual(
            partition.rows["calibration"],
            samples[:2],
        )

        self.assertEqual(
            partition.rows["selection"],
            samples[2:3],
        )

        self.assertEqual(
            partition.rows["reporting"],
            samples[3:],
        )

        self.assertEqual(
            partition.summary["calibration"]["games"],
            1,
        )

    def test_same_game_cannot_cross_periods(self) -> None:
        metadata = [
            ("G1", 1, 1, "2025-06-30"),
            ("G1", 1, 2, "2025-07-01"),
            ("G2", 1, 1, "2025-09-01"),
            ("G3", 1, 1, "2025-09-02"),
        ]

        with self.assertRaises(ValueError):
            development.partition_rows(
                training_samples(), metadata
            )

    def test_partition_rejects_missing_or_duplicate_rows(self) -> None:
        samples = training_samples()

        with self.assertRaises(ValueError):
            development.partition_rows(
                samples,
                [("G1", 1, 1, "2025-06-30")],
            )

        metadata = [
            ("G1", 1, 1, "2025-06-30"),
            ("G1", 1, 1, "2025-06-30"),
            ("G2", 1, 1, "2025-07-01"),
            ("G3", 1, 1, "2025-09-01"),
        ]

        with self.assertRaises(ValueError):
            development.partition_rows(samples, metadata)


class PitchRegularizationTests(unittest.TestCase):
    def test_lazy_decay_matches_dense_l2(self) -> None:
        samples = training_samples()

        actual = development.fit_regularized(
            samples, 42
        )

        expected = dense_regularized_reference(
            samples, 42
        )

        self.assertEqual(
            set(actual["weights"]),
            set(expected),
        )

        for name, coefficients in expected.items():
            for actual_value, expected_value in zip(
                actual["weights"][name],
                coefficients,
            ):
                self.assertAlmostEqual(
                    actual_value,
                    expected_value,
                    places=12,
                )

    def test_regularized_seed_reproduction(self) -> None:
        samples = training_samples()

        first = development.fit_regularized(
            samples, 42
        )

        second = development.fit_regularized(
            samples, 42
        )

        self.assertEqual(first, second)
        self.assertEqual(first["kind"], "logistic")

        self.assertEqual(
            first["counts"],
            m5.training_counts(samples),
        )

        prediction = m5.predict(
            m5.Runtime(first),
            example_input(),
        )

        self.assertAlmostEqual(
            sum(prediction["joint"].values()),
            1.0,
        )

    def test_temperature_is_folded_into_coefficients(self) -> None:
        base = m5.fit_logistic(
            training_samples(), 42
        )

        original = deepcopy(base)

        identity = development.scaled_model(
            base, 1.0
        )

        self.assertEqual(base, identity)

        scaled = development.scaled_model(
            base, 1.5
        )

        for name, coefficients in base["weights"].items():
            self.assertEqual(
                scaled["weights"][name],
                [value / 1.5 for value in coefficients],
            )

        self.assertEqual(base, original)
        self.assertEqual(scaled["counts"], base["counts"])
        self.assertEqual(scaled["kind"], "logistic")

        with self.assertRaises(ValueError):
            development.scaled_model(base, 0.0)

    def test_temperature_loss_matches_runtime(self) -> None:
        model = m5.fit_logistic(
            training_samples(), 42
        )

        calibration_rows = [
            example_sample("FF:B"),
            example_sample("SV:V"),
            example_sample("SL:S"),
        ]

        fitted = development.fit_temperature(
            model,
            calibration_rows,
        )

        self.assertIn(
            fitted["temperature"],
            development.TEMPERATURES,
        )

        for record in fitted["losses"]:
            scaled = development.scaled_model(
                model,
                record["temperature"],
            )

            runtime = m5.Runtime(scaled)

            expected = -sum(
                log(
                    runtime.probabilities(sample.keys)[
                        sample.target
                    ]
                )
                for sample in calibration_rows
            ) / len(calibration_rows)

            self.assertAlmostEqual(
                record["log_loss"],
                expected,
                places=12,
            )

        self.assertEqual(
            fitted,
            development.fit_temperature(
                model, calibration_rows
            ),
        )

    def test_temperature_fit_metadata_rejects_wrong_selection(self) -> None:
        model = m5.fit_logistic(
            training_samples(), 42
        )

        fitted = development.fit_temperature(
            model,
            [example_sample()],
        )

        wrong = deepcopy(fitted)

        wrong["temperature"] = next(
            temperature
            for temperature in development.TEMPERATURES
            if temperature != fitted["temperature"]
        )

        with self.assertRaises(ValueError):
            development.validate_temperature_fit(
                wrong, 1
            )


class PitchSelectionTests(unittest.TestCase):
    def test_reject_calibration_with_worse_ece(self) -> None:
        result = development.select_candidates(
            mock_selection_results()
        )

        self.assertEqual(
            result["selected_candidate"],
            "logistic_temperature",
        )

        decisions = result["decisions"]

        self.assertTrue(
            decisions["logistic_temperature"]["eligible"]
        )

        self.assertFalse(
            decisions["regularized_temperature"]["eligible"]
        )

        self.assertFalse(
            decisions["regularized_temperature"][
                "calibration_gate"
            ]
        )

    def test_identical_calibration_is_not_improvement(self) -> None:
        results = mock_selection_results()

        results["logistic_temperature"] = deepcopy(
            results["logistic"]
        )

        selected = development.select_candidates(results)

        self.assertFalse(
            selected["decisions"]["logistic_temperature"][
                "eligible"
            ]
        )

    def test_fallback_to_best_frequency(self) -> None:
        results = mock_selection_results()

        for name in development.CANDIDATES:
            if name not in development.FREQUENCIES:
                results[name] = mock_metric(
                    3.5, 0.98, 0.08, 0.04
                )

        selected = development.select_candidates(results)

        self.assertEqual(
            selected["selected_candidate"],
            "conditional_frequency",
        )

    def test_candidate_denominator_mismatch(self) -> None:
        results = mock_selection_results()

        results["regularized"]["joint"]["samples"] = 9

        with self.assertRaises(ValueError):
            development.select_candidates(results)


class PitchDetailedEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = m5.fit_frequency(
            training_samples()
        )

        samples = [
            example_sample("FF:B"),
            example_sample("ST:H"),
            example_sample("SV:V"),
        ]

        actual = evaluation.evaluate(
            self.model,
            samples,
        )

        self.results = {
            name: deepcopy(actual)
            for name in development.CANDIDATES
        }

    def test_support_and_all_denominators(self) -> None:
        development.validate_evaluations(
            self.results,
            self.model["counts"],
            3,
        )

        support = development.support_summary(
            self.results["frequency"]
        )

        joint = support["joint"]

        self.assertEqual(
            joint["train_support_groups"]["unseen"]["count"],
            2,
        )

        self.assertEqual(
            joint["unsupported_output_samples"],
            0,
        )

        self.assertEqual(
            set(
                joint[
                    "train_unseen_classes_with_validation_support"
                ]
            ),
            {"ST:H", "SV:V"},
        )

    def test_corrupted_calibration_denominator(self) -> None:
        metrics = self.results["logistic"]["joint"]

        metrics["classes"]["FF:B"]["calibration"][0][
            "count"
        ] += 1

        with self.assertRaises(ValueError):
            development.validate_evaluations(
                self.results,
                self.model["counts"],
                3,
            )

    def test_corrupted_support_group(self) -> None:
        self.results["frequency"]["joint"][
            "train_support_groups"
        ]["unseen"]["count"] = 0

        with self.assertRaises(ValueError):
            development.validate_evaluations(
                self.results,
                self.model["counts"],
                3,
            )


class PitchIndependentComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.prediction = m5.predict(
            m5.Runtime(
                m5.fit_frequency(training_samples())
            ),
            example_input(),
        )

    def test_equal_predictions(self) -> None:
        restored = json.loads(
            json.dumps(self.prediction)
        )

        result = compare_predictions(
            restored,
            self.prediction,
        )

        self.assertTrue(result["reproduce_equal"])
        self.assertEqual(result["joint_classes"], 70)
        self.assertEqual(result["pitch_type_classes"], 10)
        self.assertEqual(result["pitch_result_classes"], 7)

    def test_reject_wrong_marginals(self) -> None:
        modified = deepcopy(self.prediction)

        modified["pitch_type"]["FF"] += 0.1

        with self.assertRaises(ValueError):
            compare_predictions(
                modified,
                self.prediction,
            )

    def test_reject_different_valid_predictions(self) -> None:
        other = m5.predict(
            m5.Runtime(
                m5.fit_frequency(
                    [example_sample("SL:S")]
                )
            ),
            example_input(),
        )

        with self.assertRaises(ValueError):
            compare_predictions(
                other,
                self.prediction,
            )

    def test_compare_cli_success_and_failure(self) -> None:
        with TemporaryDirectory() as directory:
            expected = Path(directory) / "prediction.json"

            save_json(
                expected,
                self.prediction,
            )

            output = io.StringIO()

            with (
                patch(
                    "sys.stdin",
                    io.StringIO(
                        json.dumps(self.prediction)
                    ),
                ),
                redirect_stdout(output),
            ):
                exit_code = main([
                    "compare",
                    "--expected",
                    str(expected),
                ])

            self.assertEqual(exit_code, 0)

            self.assertTrue(
                json.loads(output.getvalue())[
                    "reproduce_equal"
                ]
            )

            with (
                patch("sys.stdin", io.StringIO("{}")),
                self.assertLogs(
                    "src.models.pitch_outcome.__main__",
                    level="ERROR",
                ),
            ):
                exit_code = main([
                    "compare",
                    "--expected",
                    str(expected),
                ])

            self.assertEqual(exit_code, 1)

    def test_cli_asset_requires_training_metadata(self) -> None:
        model = m5.fit_frequency(
            training_samples()
        )

        payload = {
            "schema": SCHEMA,
            "version": VERSION,
            "training_run_id": "a" * 32,
            "configured_seed": 42,
            "selected_model": model["kind"],
            "operational_predictor_count": 1,
            "input_columns": list(m5.INPUT_COLUMNS),
            "allowed_input_columns": list(m5.ALL_COLUMNS),
            "method_settings": method_settings(),
            "model": model,
        }

        with TemporaryDirectory() as directory:
            asset = Path(directory) / "model.json"

            save_json(
                asset,
                {
                    "payload": payload,
                    "sha256": fingerprint(payload),
                },
            )

            # 저수준 합성 자산 검사는 허용한다.
            self.assertEqual(
                load_payload(asset)["selected_model"],
                "frequency",
            )

            # 실제 사용자 CLI는 학습 실행 메타데이터를 요구한다.
            with self.assertRaises(ValueError):
                load_payload(
                    asset,
                    training_asset=True,
                )


class PitchDevelopmentWorkflowTests(unittest.TestCase):
    def test_synthetic_train_save_verify_workflow(self) -> None:
        """합성 자료로 후보 적합부터 저장·검증까지 일괄 확인한다."""
        train_rows = training_samples()

        validation_rows = [
            example_sample("FF:B"),
            example_sample("ST:H"),
            example_sample("SL:S"),
        ]

        metadata = [
            ("G1", 1, 1, "2025-06-30"),
            ("G2", 1, 1, "2025-07-01"),
            ("G3", 1, 1, "2025-09-01"),
        ]

        partition = development.partition_rows(
            validation_rows,
            metadata,
        )

        first = example_input()

        missing_previous = {
            **example_input(),
            "pitch_number": 2,
        }

        available_previous = {
            **example_input(),
            "pitch_number": 2,
            "previous_pitch_type": "FF",
            "previous_pitch_result": "B",
        }

        examples = {
            "train:first_pitch": {
                "input": first,
            },
            "train:previous_target_invalid": {
                "input": missing_previous,
            },
            "validation:first_pitch": {
                "input": first,
            },
            "validation:previous_available": {
                "input": available_previous,
            },
        }

        with TemporaryDirectory() as directory:
            root = Path(directory)

            database = root / "synthetic_model.sqlite3"
            standard = root / "synthetic_standard.sqlite3"

            database.write_bytes(b"synthetic-model")
            standard.write_bytes(b"synthetic-standard")

            provenance = {
                "database": str(database),
                "database_sha256": sha256_file(database),
                "standard_database": str(standard),
                "standard_database_sha256": sha256_file(
                    standard
                ),
                "counts": {
                    "train": {
                        "samples": len(train_rows),
                    },
                    "validation": {
                        "samples": len(validation_rows),
                        "games": 3,
                        "first_date": "2025-06-30",
                        "last_date": "2025-09-01",
                    },
                },
                "source_exclusions": {},
                "non_regular_excluded": {},
                "test_target_values_read": False,
            }

            dataset = data.Dataset(
                rows={
                    "train": train_rows,
                    "validation": validation_rows,
                },
                provenance=provenance,
                examples=examples,
            )

            settings = cast(
                Settings,
                SimpleNamespace(
                    seed=42,
                    models_dir=root / "models",
                    results_dir=root / "results",
                ),
            )

            with (
                patch.object(
                    data,
                    "load",
                    return_value=dataset,
                ),
                patch.object(
                    development,
                    "partition_from_source",
                    return_value=partition,
                ),
            ):
                result = train(
                    settings,
                    data.INTAKE_RUN,
                )

            self.assertEqual(
                result["operational_predictor_count"],
                1,
            )

            self.assertFalse(
                result["final_test_evaluated"]
            )

            self.assertFalse(
                result["reporting_used_for_selection"]
            )

            paths = result["paths"]

            verified = verify_run(
                Path(paths["model"]),
                Path(paths["evaluation"]),
            )

            self.assertTrue(
                verified["reproduce_equal"]
            )

            self.assertEqual(
                verified["selected_candidate"],
                result["selected_candidate"],
            )

            self.assertEqual(
                verified["selected_temperature"],
                result["selected_temperature"],
            )


if __name__ == "__main__":
    unittest.main()