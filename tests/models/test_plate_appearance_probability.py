
"""M2 운영 확률 제한, 보정, 모델 선택과 저장 재현을 검증한다."""

from contextlib import closing
from dataclasses import replace
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.datasets.model_data import fingerprint
from src.datasets.published_sources import (
    read_json,
    write_json,
)
from src.models.plate_appearance import data
from src.models.plate_appearance import model as m2
from src.models.plate_appearance import probability as operational
from src.models.plate_appearance import __main__ as workflow
from src.settings import load_settings

from tests.models.test_plate_appearance import (
    add_pa,
    fixture,
    inputs,
    sample,
)


def dated_sample(
    target: str,
    day: str,
    game_id: str,
    number: int = 1,
) -> m2.Sample:
    """시간 분할 테스트에서 사용할 합성 타석을 구성한다."""

    return replace(
        sample(target),
        sample_id=f"{game_id}:{number}",
        game_id=game_id,
        day=day,
    )


class RuleProbabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.train = [
            sample(
                "single" if index % 3 == 0 else "field_out",
                sample_id=f"train:{index + 1}",
            )
            for index in range(36)
        ]

        global_base, conditional_base = m2.fit_frequencies(
            cls.train
        )

        logistic_base = m2.fit_logistic(
            cls.train, 42
        )

        cls.global_model = operational.make_model(
            global_base, "global"
        )

        cls.conditional_model = operational.make_model(
            conditional_base, "conditional"
        )

        cls.logistic_model = operational.make_model(
            logistic_base, "logistic"
        )

    def test_all_24_starting_states(self) -> None:
        """모든 시작 아웃·주자 조합에서 규칙을 확인한다."""

        for outs in range(3):
            for runners in range(8):
                state = inputs()

                state.update(
                    {
                        "outs": outs,
                        "on_first": int(bool(runners & 1)),
                        "on_second": int(bool(runners & 2)),
                        "on_third": int(bool(runners & 4)),
                    }
                )

                count = runners.bit_count()

                with self.subTest(
                    outs=outs, runners=runners
                ):
                    mask = operational.allowed_mask(state)

                    common = outs < 2 and count >= 1

                    self.assertEqual(
                        mask[m2.CLASS_INDEX["double_play"]],
                        common,
                    )

                    self.assertEqual(
                        mask[m2.CLASS_INDEX["triple_play"]],
                        outs == 0 and count >= 2,
                    )

                    self.assertEqual(
                        mask[m2.CLASS_INDEX["sac_bunt"]],
                        common,
                    )

                    self.assertEqual(
                        mask[m2.CLASS_INDEX["sac_fly"]],
                        common,
                    )

                    for model in (
                        self.global_model,
                        self.conditional_model,
                        self.logistic_model,
                    ):
                        probabilities = (
                            operational.predict_probabilities(
                                model, state
                            )
                        )

                        self.assertAlmostEqual(
                            sum(probabilities),
                            1.0,
                        )

                        for probability, allowed in zip(
                            probabilities,
                            mask,
                            strict=True,
                        ):
                            if allowed:
                                self.assertGreater(
                                    probability, 0.0
                                )
                            else:
                                self.assertEqual(
                                    probability, 0.0
                                )

    def test_invalid_starting_state(self) -> None:
        """비정상 아웃과 주자 점유 값을 거부한다."""

        for key, value in (
            ("outs", 3),
            ("outs", -1),
            ("on_first", 2),
            ("on_second", True),
            ("on_third", None),
        ):
            state = inputs()
            state[key] = value

            with self.subTest(key=key, value=value):
                with self.assertRaises(ValueError):
                    operational.predict_probabilities(
                        self.global_model,
                        state,
                    )

    def test_rule_conflicts_fail_closed(self) -> None:
        """실제 정답이 규칙과 충돌해도 임의 제거하지 않는다."""

        samples = [
            sample("double_play", sample_id="conflict:1"),
            sample("triple_play", sample_id="conflict:2"),
            sample("single", sample_id="valid:1"),
        ]

        report = operational.audit_rules(samples)

        self.assertEqual(report["samples"], 3)
        self.assertEqual(report["conflict_count"], 2)

        self.assertEqual(
            report["conflicts_by_class"],
            {
                "double_play": 1,
                "triple_play": 1,
            },
        )

        with self.assertLogs(
            operational.LOGGER,
            level="ERROR",
        ):
            with self.assertRaises(ValueError):
                operational.require_rule_consistency(
                    samples, "train"
                )

        with self.assertRaises(ValueError):
            operational.fit_regularized_logistic(
                samples, 42
            )

    def test_masked_softmax_with_extreme_logits(self) -> None:
        """금지 클래스의 큰 로짓이 확률 계산을 방해하지 않는다."""

        mask = operational.allowed_mask(inputs())

        logits = [0.0] * m2.N_CLASSES

        for index, allowed in enumerate(mask):
            if not allowed:
                logits[index] = 1000.0

        probabilities = operational.masked_softmax(
            logits, mask
        )

        self.assertAlmostEqual(
            sum(probabilities), 1.0
        )

        for probability, allowed in zip(
            probabilities,
            mask,
            strict=True,
        ):
            if not allowed:
                self.assertEqual(probability, 0.0)

    def test_possible_unseen_classes_remain_positive(self) -> None:
        """Train에서 미출현한 결과도 가능하면 양의 확률을 유지한다."""

        state = inputs()

        state.update(
            {
                "on_first": 1,
                "on_second": 1,
            }
        )

        probabilities = operational.predict_probabilities(
            self.global_model,
            state,
        )

        for name in (
            "triple_play",
            "catcher_interference",
            "home_run",
        ):
            self.assertGreater(
                probabilities[m2.CLASS_INDEX[name]],
                0.0,
            )

    def test_invalid_conditional_group_is_rejected(self) -> None:
        """조건부 모델 내부에 불가능한 정답 빈도가 있으면 실패한다."""

        _, conditional = m2.fit_frequencies(
            [sample("double_play")]
        )

        with self.assertRaises(ValueError):
            operational.make_model(
                conditional, "conditional"
            )

    def test_regularized_seed_and_unknown_player(self) -> None:
        """개선 학습의 결정성과 신규 ID 후퇴를 확인한다."""

        first = operational.fit_regularized_logistic(
            self.train, 42
        )

        second = operational.fit_regularized_logistic(
            self.train, 42
        )

        self.assertEqual(first, second)

        self.assertIn(
            "id:batter:hf:batter",
            first["weights"],
        )

        model = operational.make_model(
            first, "regularized"
        )

        state_a = inputs()
        state_b = inputs()

        state_a["batter_id"] = "hf:new-a"
        state_b["batter_id"] = "hf:new-b"

        self.assertEqual(
            operational.predict_probabilities(
                model, state_a
            ),
            operational.predict_probabilities(
                model, state_b
            ),
        )

    def test_calibration_preserves_structural_zeros(self) -> None:
        """Temperature와 절편 보정 이후에도 금지 확률은 0이다."""

        offsets = [0.1] * m2.N_CLASSES

        model = operational.make_model(
            self.logistic_model["base"],
            "logistic",
            "temperature_bias",
            1.50,
            offsets,
        )

        state = inputs()
        state["outs"] = 2

        probabilities = operational.predict_probabilities(
            model, state
        )

        self.assertAlmostEqual(
            sum(probabilities), 1.0
        )

        for name in operational.RULE_EVENTS:
            self.assertEqual(
                probabilities[m2.CLASS_INDEX[name]],
                0.0,
            )


class CalibrationSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.train = [
            sample(
                "single" if index % 2 == 0 else "field_out",
                sample_id=f"train:{index + 1}",
            )
            for index in range(36)
        ]

        global_base, conditional_base = m2.fit_frequencies(
            cls.train
        )

        logistic_base = m2.fit_logistic(
            cls.train, 42
        )

        cls.candidates = {
            "global": operational.make_model(
                global_base, "global"
            ),
            "conditional": operational.make_model(
                conditional_base, "conditional"
            ),
            "logistic": operational.make_model(
                logistic_base, "logistic"
            ),
            "logistic_temperature": operational.make_model(
                logistic_base,
                "logistic",
                "temperature",
                1.15,
            ),
            "logistic_temperature_bias": (
                operational.make_model(
                    logistic_base,
                    "logistic",
                    "temperature_bias",
                    1.15,
                    [0.1] * m2.N_CLASSES,
                )
            ),
        }

    def metrics_with(
        self,
        log_loss: float,
        brier: float,
        ece: float,
    ) -> operational.DetailedMetrics:
        """선정 규칙만 분리해서 검사할 비교 지표를 구성한다."""

        rows = [sample("single")]

        predictions = [
            operational.predict_probabilities(
                self.candidates["global"],
                rows[0].inputs,
            )
        ]

        metrics = operational.score_predictions(
            rows,
            predictions,
            require_rules=True,
        )

        metrics["log_loss"] = log_loss
        metrics["brier"] = brier
        metrics["macro_classwise_ece"] = ece

        return metrics

    def test_calibration_fit_and_rare_class_support(self) -> None:
        """보정 적합은 전반기 자료만 사용하고 희소 절편을 고정한다."""

        calibration = [
            dated_sample(
                "single" if index % 2 == 0 else "field_out",
                "2025-04-01",
                f"early-{index}",
            )
            for index in range(64)
        ]

        first = operational.fit_calibrations(
            self.candidates["logistic"],
            calibration,
        )

        second = operational.fit_calibrations(
            self.candidates["logistic"],
            calibration,
        )

        self.assertEqual(first, second)

        temperature, bias, report = first

        self.assertEqual(
            temperature["calibration"],
            "temperature",
        )

        self.assertEqual(
            bias["calibration"],
            "temperature_bias",
        )

        self.assertIn(
            temperature["temperature"],
            operational.TEMPERATURES,
        )

        self.assertEqual(
            set(report["supported_bias_classes"]),
            {"single", "field_out"},
        )

        for name in m2.CLASSES:
            if name not in ("single", "field_out"):
                self.assertEqual(
                    bias["class_offsets"][
                        m2.CLASS_INDEX[name]
                    ],
                    0.0,
                )

    def test_calibration_rejects_selection_period(self) -> None:
        """선정 및 Test 자료가 보정 학습으로 유입되지 않는다."""

        for day in (
            "2024-04-01",
            "2025-07-01",
            "2026-04-01",
        ):
            with self.subTest(day=day):
                with self.assertRaises(ValueError):
                    operational.fit_calibrations(
                        self.candidates["logistic"],
                        [
                            dated_sample(
                                "single",
                                day,
                                "wrong-period",
                            )
                        ],
                    )

    def test_validation_split_boundaries(self) -> None:
        """경계 날짜를 포함하여 보정·선정 구간을 분리한다."""

        rows = [
            dated_sample(
                "single", "2025-06-30", "early"
            ),
            dated_sample(
                "walk", "2025-07-01", "late"
            ),
        ]

        early, late = operational.split_validation(
            rows
        )

        self.assertEqual(len(early), 1)
        self.assertEqual(len(late), 1)

        self.assertEqual(
            early[0].day, "2025-06-30"
        )

        self.assertEqual(
            late[0].day, "2025-07-01"
        )

    def test_mixed_game_and_test_rows_are_rejected(self) -> None:
        """동일 경기 분리와 2026년 자료 유입을 차단한다."""

        mixed = [
            dated_sample(
                "single", "2025-06-30", "same-game"
            ),
            dated_sample(
                "walk", "2025-07-01", "same-game", 2
            ),
        ]

        with self.assertRaises(ValueError):
            operational.split_validation(mixed)

        test_rows = [
            dated_sample(
                "single", "2025-06-30", "early"
            ),
            dated_sample(
                "walk", "2026-04-01", "test"
            ),
        ]

        with self.assertRaises(ValueError):
            operational.split_validation(test_rows)

        with self.assertRaises(ValueError):
            operational.split_validation(
                [test_rows[0]]
            )

    def test_classwise_ece_formula(self) -> None:
        """ECE가 구간별 표본 가중 절대 오차인지 검사한다."""

        rows = [
            sample("single", sample_id="fixture:1"),
            sample("double", sample_id="fixture:2"),
        ]

        vector = [0.0] * m2.N_CLASSES
        vector[m2.CLASS_INDEX["single"]] = 1.0

        result = operational.score_predictions(
            rows,
            [vector, vector],
            require_rules=True,
        )

        self.assertAlmostEqual(
            result["classwise_ece"]["single"],
            0.5,
        )

        self.assertAlmostEqual(
            result["classwise_ece"]["double"],
            0.5,
        )

        self.assertAlmostEqual(
            result["macro_classwise_ece"],
            1.0 / m2.N_CLASSES,
        )

    def test_rule_violation_metrics(self) -> None:
        """기존 무제한 후보의 불가능 확률을 측정한다."""

        rows = [sample("single")]

        raw = m2.predict_probabilities(
            self.candidates["global"]["base"],
            rows[0].inputs,
        )

        report = operational.score_predictions(
            rows,
            [raw],
            require_rules=False,
        )

        self.assertEqual(
            report["forbidden_nonzero_samples"], 1
        )

        self.assertGreater(
            report["forbidden_probability_mean"], 0.0
        )

        with self.assertRaises(ValueError):
            operational.score_predictions(
                rows,
                [raw],
                require_rules=True,
            )

    def test_calibration_selection_gate(self) -> None:
        """확률 손실과 보정 지표의 공동 개선 조건을 확인한다."""

        results = {
            "global": self.metrics_with(
                2.0, 0.8, 0.05
            ),
            "conditional": self.metrics_with(
                1.8, 0.7, 0.04
            ),
            "logistic": self.metrics_with(
                1.7, 0.65, 0.06
            ),
            "logistic_temperature": self.metrics_with(
                1.69, 0.64, 0.03
            ),
            "logistic_temperature_bias": self.metrics_with(
                1.68, 0.63, 0.07
            ),
        }

        selected, report = operational.select_model(
            results,
            self.candidates,
        )

        self.assertEqual(
            selected,
            "logistic_temperature",
        )

        decisions = report["candidate_decisions"]

        self.assertFalse(
            decisions["logistic_temperature_bias"][
                "calibration_gate_passed"
            ]
        )

    def test_selection_falls_back_to_frequency(self) -> None:
        """개선 조건을 충족하지 못하면 빈도 기준을 선택한다."""

        results = {
            "global": self.metrics_with(
                2.0, 0.8, 0.05
            ),
            "conditional": self.metrics_with(
                1.8, 0.7, 0.04
            ),
            "logistic": self.metrics_with(
                2.1, 0.9, 0.06
            ),
            "logistic_temperature": self.metrics_with(
                2.0, 0.85, 0.05
            ),
            "logistic_temperature_bias": self.metrics_with(
                1.9, 0.8, 0.04
            ),
        }

        selected, report = operational.select_model(
            results,
            self.candidates,
        )

        self.assertEqual(selected, "conditional")

        self.assertEqual(
            report["eligible_logistic_candidates"],
            [],
        )

    def test_invalid_calibration_parameters(self) -> None:
        """지원하지 않는 보정 파라미터를 거부한다."""

        base = self.candidates["logistic"]["base"]

        for temperature in (
            0.0,
            -1.0,
            100.0,
            float("nan"),
        ):
            with self.subTest(temperature=temperature):
                with self.assertRaises(ValueError):
                    operational.make_model(
                        base,
                        "logistic",
                        "temperature",
                        temperature,
                    )

        with self.assertRaises(ValueError):
            operational.make_model(
                base,
                "logistic",
                "temperature_bias",
                1.0,
                [3.0] * m2.N_CLASSES,
            )

        with self.assertRaises(ValueError):
            operational.make_model(
                base,
                "logistic",
                "temperature",
                1.15,
                [0.1] * m2.N_CLASSES,
            )


class OperationalAssetTests(unittest.TestCase):
    def test_legacy_schema_is_not_operational(self) -> None:
        """기존 무제한 v1 자산을 운영 v2로 읽지 않는다."""

        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"

            payload = {
                "schema": m2.SCHEMA,
            }

            write_json(
                path,
                {
                    "payload": payload,
                    "sha256": fingerprint(payload),
                },
            )

            with self.assertRaises(ValueError):
                workflow.load_operational_model(path)

    def test_synthetic_training_storage_and_verification(self) -> None:
        """자료 조회부터 후보 선정·저장·재현까지 통합 검증한다."""

        with TemporaryDirectory() as directory:
            root = Path(directory)

            database = root / "fixture.sqlite3"

            with closing(fixture()) as source:
                for index in range(36):
                    add_pa(
                        source,
                        f"train-{index}",
                        "2024-04-01",
                        "train",
                        event=(
                            "single"
                            if index % 2 == 0
                            else "field_out"
                        ),
                    )

                for index in range(6):
                    add_pa(
                        source,
                        f"early-{index}",
                        "2025-04-01",
                        "validation",
                        event=(
                            "single"
                            if index % 2 == 0
                            else "field_out"
                        ),
                    )

                    add_pa(
                        source,
                        f"late-{index}",
                        "2025-08-01",
                        "validation",
                        event=(
                            "walk"
                            if index % 2 == 0
                            else "field_out"
                        ),
                    )

                source.commit()

                with closing(
                    sqlite3.connect(database)
                ) as destination:
                    source.backup(destination)

            project_root = (
                Path(__file__).resolve().parents[2]
            )

            settings = replace(
                load_settings(
                    project_root,
                    environ={},
                ),
                models_dir=root / "models",
                results_dir=root / "results",
                seed=42,
            )

            # 합성 자료에서만 실제 인수 DB의 고정 해시와
            # 고정 분모 검사 경계를 대체한다.
            # 실제 자료 조회와 모델 학습·평가·저장은 그대로 실행한다.
            with patch.object(
                data,
                "source",
                return_value=(
                    database,
                    {"model_data_run_id": "fixture"},
                ),
            ), patch.object(
                data,
                "verify_intake",
                return_value=None,
            ):
                report = workflow.train_run(
                    settings,
                    "fixture",
                )

            self.assertEqual(
                report["model_schema"],
                operational.SCHEMA,
            )

            self.assertEqual(
                report["operational_predictor_count"],
                1,
            )

            self.assertEqual(
                report["experimental_candidate_count"],
                8,
            )

            self.assertEqual(
                report["development_periods"][
                    "calibration"
                ]["samples"],
                6,
            )

            self.assertEqual(
                report["development_periods"][
                    "selection"
                ]["samples"],
                6,
            )

            self.assertEqual(
                report["final_test"],
                "held_out_targets_not_evaluated",
            )

            paths = report["paths"]

            asset = Path(paths["model"])
            evaluation_path = Path(paths["evaluation"])

            model = workflow.load_operational_model(
                asset
            )

            self.assertEqual(
                operational.candidate_name(model),
                report["selected_model"],
            )

            verification = workflow.verify_run(
                asset,
                evaluation_path,
            )

            self.assertTrue(
                verification["source_hash_match"]
            )

            self.assertTrue(
                verification["evaluation_hash_match"]
            )

            self.assertTrue(
                verification["example_prediction_match"]
            )

            self.assertTrue(
                verification[
                    "forbidden_probability_verified"
                ]
            )

            self.assertEqual(
                len(
                    list(
                        settings.models_dir.rglob(
                            "model.json"
                        )
                    )
                ),
                1,
            )

            self.assertFalse(
                asset.with_name(
                    "model.pending.json"
                ).exists()
            )

            # 현재 코드의 소스 지문이 바뀌면 검증을 거부한다.
            with patch.object(
                workflow,
                "source_hashes",
                return_value={"changed": "0" * 64},
            ):
                with self.assertRaises(ValueError):
                    workflow.verify_run(
                        asset,
                        evaluation_path,
                    )

            # 평가 파일 변조도 저장된 평가 지문으로 감지한다.
            altered = read_json(evaluation_path)
            altered["operational_predictor_count"] = 2

            altered_path = root / "altered.json"
            write_json(altered_path, altered)

            with self.assertRaises(ValueError):
                workflow.verify_run(
                    asset,
                    altered_path,
                )


if __name__ == "__main__":
    unittest.main()
