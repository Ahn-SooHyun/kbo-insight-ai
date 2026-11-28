"""예측 설명의 가산성·분모·표본 선정·무결성 회귀 테스트."""

from datetime import date
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.evaluation import game_outcome as e1
from src.evaluation import plate_appearance as e2
from src.explainability import (
    GLOBAL_SAMPLE_LIMIT,
    SEED,
)
from src.explainability import contributions as c
from src.explainability import diagnostics as d
from src.explainability import __main__ as cli
from src.models.plate_appearance import model as m2
from src.models.pitch_outcome import model as m5
from src.models.player_performance import data as d3
from src.models.player_performance import model as m3
from src.models.player_performance import development as dev3


def sample(key: str) -> dict[str, object]:
    """표본 선정 테스트용 최소 식별자."""
    return {
        "prediction": {
            "sample_id": key,
        },
        "target": {
            "sample_id": key,
        },
    }


class ComparisonTests(unittest.TestCase):
    """기여 및 저장 예측 비교의 수치·구조 검증."""

    def test_equal_values(self) -> None:
        self.assertEqual(
            c.compare(
                {"a": [1, 2.0]},
                {"a": (1, 2.0)},
                "test",
            ),
            0.0,
        )

    def test_float_tolerance(self) -> None:
        error = c.compare(
            0.5 + 1e-12,
            0.5,
            "test",
        )

        self.assertLess(error, 1e-9)

    def test_numeric_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            c.compare(0.4, 0.5, "test")

    def test_missing_key(self) -> None:
        with self.assertRaises(ValueError):
            c.compare(
                {"a": 1},
                {"b": 1},
                "test",
            )

    def test_boolean_is_not_integer(self) -> None:
        with self.assertRaises(ValueError):
            c.compare(True, 1, "test")

    def test_null_is_not_zero(self) -> None:
        with self.assertRaises(ValueError):
            c.compare(None, 0, "test")

    def test_nonfinite_value(self) -> None:
        with self.assertRaises(ValueError):
            c.compare(
                float("nan"),
                0.0,
                "test",
            )

    def test_vector_dimension_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            c.add([1.0], [2.0, 3.0])


class LinearExplanationTests(unittest.TestCase):
    """정확한 선형 SHAP 산식 및 희소 특징 설명."""

    def test_single_output_contributions(self) -> None:
        linear = c.LinearExplanation(
            labels=("output",),
            bias=[1.0],
            weights={
                "x": [2.0],
                "y": [-3.0],
            },
            means={
                "x": 1.0,
                "y": 2.0,
            },
        )

        result = linear.explain(
            {"x": 3.0, "y": 4.0},
            [-5.0],
            aggregate=True,
        )

        self.assertEqual(
            result["base_values"],
            [-3.0],
        )

        self.assertEqual(
            result["active_shap_values"],
            {
                "x": [4.0],
                "y": [-6.0],
            },
        )

        self.assertEqual(
            result["logits"],
            [-5.0],
        )

        self.assertEqual(
            result["inactive_shap_sum"],
            [0.0],
        )

    def test_inactive_feature_contribution(self) -> None:
        linear = c.LinearExplanation(
            labels=("output",),
            bias=[1.0],
            weights={
                "x": [2.0],
                "y": [-3.0],
            },
            means={
                "x": 1.0,
                "y": 2.0,
            },
        )

        result = linear.explain(
            {"x": 3.0},
            [7.0],
            aggregate=True,
        )

        self.assertEqual(
            result["inactive_shap_sum"],
            [6.0],
        )

        self.assertEqual(
            result["logits"],
            [7.0],
        )

    def test_multioutput_contributions(self) -> None:
        linear = c.LinearExplanation(
            labels=("a", "b"),
            bias=[1.0, 2.0],
            weights={
                "x": [2.0, -1.0],
            },
            means={
                "x": 0.5,
            },
        )

        result = linear.explain(
            {"x": 2.0},
            [5.0, 0.0],
            aggregate=True,
        )

        self.assertEqual(
            result["logits"],
            [5.0, 0.0],
        )

    def test_global_absolute_shap(self) -> None:
        linear = c.LinearExplanation(
            labels=("output",),
            bias=[0.0],
            weights={
                "x": [2.0],
            },
            means={
                "x": 1.0,
            },
        )

        linear.explain(
            {},
            [0.0],
            aggregate=True,
        )

        linear.explain(
            {"x": 3.0},
            [6.0],
            aggregate=True,
        )

        result = linear.report()

        self.assertEqual(
            result["samples"],
            2,
        )

        self.assertAlmostEqual(
            result["features"]["x"][
                "mean_across_outputs"
            ],
            3.0,
        )

    def test_failure_sample_not_added_to_global(self) -> None:
        linear = c.LinearExplanation(
            labels=("output",),
            bias=[0.0],
            weights={
                "x": [2.0],
            },
            means={
                "x": 1.0,
            },
        )

        linear.explain(
            {"x": 1.0},
            [2.0],
            aggregate=True,
        )

        linear.explain(
            {"x": 100.0},
            [200.0],
            aggregate=False,
        )

        report = linear.report()

        self.assertEqual(
            report["samples"],
            1,
        )

        self.assertEqual(
            report["features"]["x"][
                "mean_across_outputs"
            ],
            0.0,
        )

    def test_no_global_samples(self) -> None:
        linear = c.LinearExplanation(
            labels=("output",),
            bias=[0.0],
            weights={"x": [1.0]},
            means={"x": 0.0},
        )

        self.assertEqual(
            linear.report()["status"],
            "unsupported",
        )

    def test_background_feature_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            c.LinearExplanation(
                labels=("output",),
                bias=[0.0],
                weights={"x": [1.0]},
                means={"y": 0.0},
            )

    def test_unknown_feature_rejected(self) -> None:
        linear = c.LinearExplanation(
            labels=("output",),
            bias=[0.0],
            weights={"x": [1.0]},
            means={"x": 0.0},
        )

        with self.assertRaises(ValueError):
            linear.explain(
                {"unknown": 1.0},
                [1.0],
                aggregate=False,
            )

    def test_wrong_expected_logit_rejected(self) -> None:
        linear = c.LinearExplanation(
            labels=("output",),
            bias=[0.0],
            weights={"x": [2.0]},
            means={"x": 0.0},
        )

        with self.assertRaises(ValueError):
            linear.explain(
                {"x": 1.0},
                [3.0],
                aggregate=False,
            )


class SelectionTests(unittest.TestCase):
    """고정 전체 표본과 오류 사례 선정의 결정성."""

    def test_global_selection_order_independent(self) -> None:
        first = d.SampleSelection(limit=3)
        second = d.SampleSelection(limit=3)

        keys = [
            "sample-1",
            "sample-2",
            "sample-3",
            "sample-4",
            "sample-5",
        ]

        for key in keys:
            first.add(sample(key))

        for key in reversed(keys):
            second.add(sample(key))

        self.assertEqual(
            set(first.selected()),
            set(second.selected()),
        )

    def test_global_selection_matches_hash_order(self) -> None:
        selection = d.SampleSelection(limit=2)

        keys = ["a", "b", "c", "d"]

        for key in keys:
            selection.add(sample(key))

        expected = sorted(
            keys,
            key=lambda key: (
                sha256(
                    f"{SEED}:{key}".encode("utf-8")
                ).hexdigest(),
                key,
            ),
        )[:2]

        self.assertEqual(
            set(selection.selected()),
            set(expected),
        )

    def test_global_sample_limit(self) -> None:
        selection = d.SampleSelection()

        for index in range(
            GLOBAL_SAMPLE_LIMIT + 10
        ):
            selection.add(
                sample(f"id-{index}")
            )

        self.assertEqual(
            len(selection.selected()),
            GLOBAL_SAMPLE_LIMIT,
        )

    def test_failure_rank_and_tie_break(self) -> None:
        failures = d.FailureCases(limit=2)

        failures.observe(
            "error",
            sample("b"),
            score=4.0,
        )

        failures.observe(
            "error",
            sample("a"),
            score=4.0,
        )

        failures.observe(
            "error",
            sample("c"),
            score=2.0,
        )

        result = failures.report()["error"]

        self.assertEqual(
            result["count"],
            3,
        )

        self.assertEqual(
            [
                item["sample_id"]
                for item in result["cases"]
            ],
            ["a", "b"],
        )

    def test_failure_population(self) -> None:
        failures = d.FailureCases()

        failures.observe(
            "error",
            sample("a"),
            score=1.0,
        )

        failures.observe(
            "error",
            sample("b"),
        )

        failures.observe(
            "error",
            sample("c"),
            eligible=False,
        )

        result = failures.report()["error"]

        self.assertEqual(
            result["eligible_samples"],
            2,
        )

        self.assertEqual(
            result["count"],
            1,
        )

        self.assertEqual(
            result["fraction"],
            0.5,
        )

    def test_multiple_reasons_deduplicate_sample(self) -> None:
        failures = d.FailureCases()

        failures.observe(
            "first",
            sample("a"),
            score=1.0,
        )

        failures.observe(
            "second",
            sample("a"),
            score=2.0,
        )

        samples, reasons = failures.selected()

        self.assertEqual(
            set(samples),
            {"a"},
        )

        self.assertEqual(
            reasons["a"],
            ["first", "second"],
        )

    def test_ineligible_failure_rejected(self) -> None:
        failures = d.FailureCases()

        with self.assertRaises(ValueError):
            failures.observe(
                "error",
                sample("a"),
                eligible=False,
                score=1.0,
            )


class CalibrationTests(unittest.TestCase):
    """확률 신뢰도와 기존 M2 평가기 호환 검사."""

    def test_m2_streaming_matches_existing_metrics(self) -> None:
        rows = [
            m2.Sample(
                sample_id="game:1",
                game_id="game",
                day="2026-04-01",
                inputs={},
                target=m2.CLASSES[0],
            ),
            m2.Sample(
                sample_id="game:2",
                game_id="game",
                day="2026-04-01",
                inputs={},
                target=m2.CLASSES[1],
            ),
        ]

        uniform = [
            1.0 / m2.N_CLASSES
        ] * m2.N_CLASSES

        second = [
            0.1 / (m2.N_CLASSES - 1)
        ] * m2.N_CLASSES

        second[1] = 0.9

        predictions = [
            uniform,
            second,
        ]

        expected = e2.metrics(
            rows,
            predictions,
        )

        metric = d.M2StreamingScore()

        for row, probability in zip(
            rows,
            predictions,
            strict=True,
        ):
            metric.add(
                m2.CLASS_INDEX[row.target],
                probability,
                [True] * m2.N_CLASSES,
            )

        actual = metric.report()

        c.compare(
            {
                key: actual[key]
                for key in expected
            },
            expected,
            "M2 streaming",
        )

    def test_m2_probability_one_last_bin(self) -> None:
        metric = d.M2StreamingScore()

        values = [0.0] * m2.N_CLASSES
        values[0] = 1.0

        metric.add(
            0,
            values,
            [True] * m2.N_CLASSES,
        )

        result = metric.report()

        intervals = result["calibration"][
            m2.CLASSES[0]
        ]

        self.assertEqual(
            intervals[-1]["count"],
            1,
        )

        self.assertTrue(
            intervals[-1]["upper_inclusive"]
        )

        self.assertEqual(
            result["log_loss"],
            0.0,
        )

    def test_m2_rule_conflict_count(self) -> None:
        metric = d.M2StreamingScore()

        values = [0.0] * m2.N_CLASSES
        values[1] = 1.0

        mask = [True] * m2.N_CLASSES
        mask[0] = False

        metric.add(
            0,
            values,
            mask,
        )

        report = metric.report()

        self.assertEqual(
            report["forbidden_target_count"],
            1,
        )

        self.assertEqual(
            report["forbidden_nonzero_samples"],
            0,
        )

    def test_m2_forbidden_positive_mass(self) -> None:
        metric = d.M2StreamingScore()

        values = [0.0] * m2.N_CLASSES

        values[0] = 0.8
        values[1] = 0.2

        mask = [True] * m2.N_CLASSES
        mask[1] = False

        metric.add(
            0,
            values,
            mask,
        )

        report = metric.report()

        self.assertEqual(
            report["forbidden_nonzero_samples"],
            1,
        )

        self.assertAlmostEqual(
            report["forbidden_probability_mean"],
            0.2,
        )

    def test_empty_bin_preserves_null(self) -> None:
        result = d.calibration_support({
            "lower": 0.0,
            "upper": 0.1,
            "count": 0,
            "mean_probability": None,
            "observed_rate": None,
        })

        self.assertEqual(
            result["support_status"],
            "empty",
        )

        self.assertIsNone(
            result["mean_probability"]
        )

    def test_sparse_bin_status(self) -> None:
        result = d.calibration_support({
            "lower": 0.9,
            "upper": 1.0,
            "count": 3,
            "mean_probability": 0.95,
        })

        self.assertEqual(
            result["support_status"],
            "sparse",
        )

        self.assertTrue(
            result["upper_inclusive"]
        )

    def test_supported_bin_status(self) -> None:
        result = d.calibration_support({
            "low": 0.2,
            "high": 0.3,
            "games": 30,
            "mean_probability": 0.25,
        })

        self.assertEqual(
            result["support_status"],
            "observed",
        )

    def test_empty_bin_numeric_mean_rejected(self) -> None:
        with self.assertRaises(ValueError):
            d.calibration_support({
                "lower": 0.0,
                "upper": 0.1,
                "count": 0,
                "mean_probability": 0.0,
            })

    def test_m1_existing_bins_preserved(self) -> None:
        outcome = {
            "games": 2,
            "calibration": {
                "home": {
                    "bins": [
                        {
                            "games": 2,
                            "mean_probability": 0.6,
                            "observed_frequency": 0.5,
                        }
                    ],
                },
            },
        }

        result = d.m1_ece({
            "results": {
                "operating": {
                    "outcomes": outcome,
                },
                "baseline": {
                    "outcomes": outcome,
                },
            },
        })

        self.assertEqual(
            result["results"]["operating"]["bin_edges"],
            e1.BINS,
        )

        self.assertAlmostEqual(
            result["results"]["operating"][
                "macro_classwise_ece"
            ],
            0.1,
        )

    def test_zero_denominator_is_unavailable(self) -> None:
        self.assertIsNone(
            d.ratio(0.0, 0)
        )


class ModelStructureTests(unittest.TestCase):
    """실제 모델의 특징·잔차·계산 경로 검사."""

    def test_m5_unknown_player_fallback(self) -> None:
        operating = SimpleNamespace(
            model={
                "weights": {
                    m5.BATTER_OTHER: [1.0],
                    "count:0:0": [2.0],
                },
            },
        )

        with patch.object(
            m5,
            "encode",
            return_value=(
                "id:batter:value:unknown",
                "count:0:0",
                "unknown_general_feature",
            ),
        ):
            vector = c.m5_vector(
                operating,
                {},
            )

        self.assertEqual(
            vector,
            {
                m5.BATTER_OTHER: 1.0,
                "count:0:0": 1.0,
            },
        )

    def test_tree_path_left_branch(self) -> None:
        root = m3.Node(
            values=(1.0,),
            support=(10,),
            feature=0,
            threshold=2.0,
            left=m3.Node(
                values=(3.0,),
                support=(5,),
            ),
            right=m3.Node(
                values=(-1.0,),
                support=(5,),
            ),
        )

        path, leaf = c.tree_path(
            root,
            (2.0,),
            ("feature",),
        )

        self.assertEqual(
            path[0]["branch"],
            "left",
        )

        self.assertEqual(
            path[0]["residual_change"],
            [2.0],
        )

        self.assertEqual(
            leaf.values,
            (3.0,),
        )

    def test_tree_path_right_branch(self) -> None:
        root = m3.Node(
            values=(1.0,),
            support=(10,),
            feature=0,
            threshold=2.0,
            left=m3.Node(
                values=(3.0,),
                support=(5,),
            ),
            right=m3.Node(
                values=(-1.0,),
                support=(5,),
            ),
        )

        path, leaf = c.tree_path(
            root,
            (3.0,),
            ("feature",),
        )

        self.assertEqual(
            path[0]["branch"],
            "right",
        )

        self.assertEqual(
            leaf.values,
            (-1.0,),
        )

    def test_m3_history_reconstruction(self) -> None:
        role = "hitter"

        width = len(d3.metrics(role))

        baseline = m3.Baseline(
            role=role,
            daily_prior=(0.0,) * width,
            remaining_end_offset=190.0,
            scales=(1.0,) * len(
                d3.output_keys(role)
            ),
        )

        fitted = dev3.Predictor(
            role=role,
            depth=0,
            baseline=baseline,
            tree=None,
            variant="history_balanced_50",
        )

        player = d3.PlayerInput(
            role=role,
            sample_id="player:2026-06-01",
            player_id="player",
            season=2026,
            anchor=date(2026, 6, 1),
            last_observed=date(2026, 5, 31),
            known_team_id=None,
            days_since_observed=1,
            prior_rejected=0,
            season_counts=(0,) * width,
            recent_counts=(0,) * width,
            current_counts=None,
            current_scope="unavailable",
            known_end=None,
            end_evidence=None,
        )

        result = c.player_history(
            fitted,
            player,
        )

        c.compare(
            result["history_projected"],
            fitted.predict(player),
            "M3 history",
        )

    def test_global_summary_excludes_causal_claim(self) -> None:
        summary = c.ExplanationSummary()

        summary.add({
            "method": "example",
            "stages": {
                "change": [1.0, -2.0],
            },
        })

        result = summary.report()

        self.assertEqual(
            result["mean_absolute_stage_changes"][
                "change"
            ]["values"],
            [1.0, 2.0],
        )

        self.assertEqual(
            result["interpretation"],
            "calculation_stage_changes_not_causal_effects",
        )


class IntegrityTests(unittest.TestCase):
    """파일 무결성 및 재현 실행 계약의 오류 처리."""

    def test_digest_match(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "input.json"

            path.write_bytes(b"test")

            digest = sha256(b"test").hexdigest()

            self.assertEqual(
                cli.verify_digest(
                    root,
                    path,
                    digest,
                ),
                path.resolve(),
            )

    def test_digest_tampering_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "input.json"

            path.write_bytes(b"original")

            digest = sha256(
                b"original"
            ).hexdigest()

            path.write_bytes(b"modified")

            with self.assertRaises(ValueError):
                cli.verify_digest(
                    root,
                    path,
                    digest,
                )

    def test_external_path_rejected(self) -> None:
        with (
            TemporaryDirectory() as first,
            TemporaryDirectory() as second,
        ):
            root = Path(first)

            path = Path(second) / "input.json"
            path.write_bytes(b"test")

            with self.assertRaises(ValueError):
                cli.verify_digest(
                    root,
                    path,
                    sha256(b"test").hexdigest(),
                )

    def test_conflicting_reference_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)

            path = root / "input.json"
            path.write_bytes(b"test")

            inventory = {}

            cli.remember_reference(
                root,
                inventory,
                path,
                sha256(b"test").hexdigest(),
            )

            with self.assertRaises(ValueError):
                cli.remember_reference(
                    root,
                    inventory,
                    path,
                    sha256(b"other").hexdigest(),
                )

    def test_json_output_cannot_overwrite(self) -> None:
        with TemporaryDirectory() as directory:
            path = (
                Path(directory)
                / "result.json"
            )

            cli.write_json(
                path,
                {"status": "completed"},
            )

            with self.assertRaises(FileExistsError):
                cli.write_json(
                    path,
                    {"status": "changed"},
                )

            result = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(
                result["status"],
                "completed",
            )

    def test_m4_pair_key_alignment(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)

            predictions = root / "predictions.jsonl"
            targets = root / "targets.jsonl"

            prediction_rows = [
                {
                    "sample_id": f"game:1:{phase}",
                    "sample_date": "2026-04-01",
                    "input": {
                        "phase": phase,
                    },
                }
                for phase in ("post", "pre")
            ]

            target_rows = [
                {
                    "sample_id": row["sample_id"],
                    "final_evidence": None,
                }
                for row in prediction_rows
            ]

            for path, rows in (
                (predictions, prediction_rows),
                (targets, target_rows),
            ):
                path.write_text(
                    "".join(
                        json.dumps(row) + "\n"
                        for row in rows
                    ),
                    encoding="utf-8",
                )

            result = list(
                d.sample_rows(
                    "m4",
                    predictions,
                    targets,
                )
            )

            self.assertEqual(
                len(result),
                2,
            )

            self.assertEqual(
                result[0]["pair"]["pre"]["sample_id"],
                "game:1:pre",
            )

    def test_m4_missing_pre_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)

            predictions = root / "predictions.jsonl"
            targets = root / "targets.jsonl"

            predictions.write_text(
                json.dumps({
                    "sample_id": "game:1:post",
                    "sample_date": "2026-04-01",
                    "input": {
                        "phase": "post",
                    },
                }) + "\n",
                encoding="utf-8",
            )

            targets.write_text(
                json.dumps({
                    "sample_id": "game:1:post",
                }) + "\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                list(
                    d.sample_rows(
                        "m4",
                        predictions,
                        targets,
                    )
                )

    def test_policy_preserves_six_models(self) -> None:
        policy = cli.policy()

        self.assertEqual(
            len(policy["models"]),
            6,
        )

        self.assertEqual(
            policy["new_operating_models"],
            0,
        )

        self.assertFalse(
            policy["test_used_for_training"]
        )

        self.assertFalse(
            policy["test_used_for_selection"]
        )

        self.assertFalse(
            policy["test_used_for_calibration"]
        )


if __name__ == "__main__":
    unittest.main()
