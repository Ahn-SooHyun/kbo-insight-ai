"""M2 타석 자료 경계·확률·평가·저장 재현의 합성 테스트."""

from collections import Counter
from dataclasses import replace
from math import log
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from src.datasets.model_data import fingerprint
from src.datasets.published_sources import write_json
from src.evaluation import plate_appearance as evaluation
from src.models.plate_appearance import data
from src.models.plate_appearance import model as m2
from src.models.plate_appearance import __main__ as workflow
from src.validation.standard_data import PA_EVENTS


def inputs() -> dict[str, object]:
    """이력 없는 선수와 시즌 첫 경기의 유효 사전 입력."""

    result: dict[str, object] = {
        name: 0 if name.endswith("_games") else None
        for name in m2.INPUT_COLUMNS
    }

    result.update(
        {
            "competition": "regular",
            "home_team_id": 2001,
            "away_team_id": 2002,
            "batter_id": "hf:batter",
            "pitcher_id": "hf:pitcher",
            "inning": 1,
            "half": "top",
            "outs": 0,
            "on_first": 0,
            "on_second": 0,
            "on_third": 0,
            "home_score": 0,
            "away_score": 0,
        }
    )

    return result


def fixture() -> sqlite3.Connection:
    """기존 모델 자료에서 사용하는 타석 키와 감사 열을 구성한다."""

    db = sqlite3.connect(":memory:")

    db.executescript(
        """
        CREATE TABLE m2_index (
            sample_id, game_id, sample_date, split, status,
            exclusion_reason, pa_number, source_sha,
            start_source_row, end_source_row,
            start_batter_id, end_batter_id,
            start_pitcher_id, end_pitcher_id,
            credited_batter_candidate, credited_pitcher_candidate,
            participant_changed, physical_pitch_rows, no_actual_pitch,
            event, result_valid, state_valid,
            start_complete, order_complete,
            state_consistent, runs_consistent, label_kind
        );

        CREATE TABLE m2_targets (
            sample_id,
            target_event
        );
        """
    )

    columns = ", ".join(
        f'"{name}"'
        for name in m2.INPUT_COLUMNS
    )

    db.execute(
        f"CREATE TABLE m2_inputs (sample_id, {columns})"
    )

    return db


def add_pa(
    db: sqlite3.Connection,
    game_id: str,
    day: str,
    split: str,
    *,
    number: int = 1,
    event: str | None = "single",
    status: str = "eligible",
    changed: bool = False,
    no_pitch: bool = False,
    label_kind: str = "observed_result",
) -> None:
    """확정·교체·미확정·무투구 타석을 필요한 형태로 삽입한다."""

    sample_id = f"{game_id}:{number}"
    state = inputs()

    reason = None

    if status == "excluded":
        reason = (
            "participant_changed"
            if changed else label_kind
        )

    index = {
        "sample_id": sample_id,
        "game_id": game_id,
        "sample_date": day,
        "split": split,
        "status": status,
        "exclusion_reason": reason,
        "pa_number": number,
        "source_sha": "fixture",
        "start_source_row": 0,
        "end_source_row": 1,
        "start_batter_id": state["batter_id"],
        "end_batter_id": (
            "hf:replacement"
            if changed else state["batter_id"]
        ),
        "start_pitcher_id": state["pitcher_id"],
        "end_pitcher_id": state["pitcher_id"],
        "credited_batter_candidate": (
            None if changed else state["batter_id"]
        ),
        "credited_pitcher_candidate": state["pitcher_id"],
        "participant_changed": int(changed),
        "physical_pitch_rows": 0 if no_pitch else 4,
        "no_actual_pitch": int(no_pitch),
        "event": event,
        "result_valid": int(event is not None),
        "state_valid": 1,
        "start_complete": 1,
        "order_complete": 1,
        "state_consistent": 1,
        "runs_consistent": 1,
        "label_kind": label_kind,
    }

    placeholders = ", ".join("?" for _ in index)

    db.execute(
        f"""
        INSERT INTO m2_index ({", ".join(index)})
        VALUES ({placeholders})
        """,
        tuple(index.values()),
    )

    if status != "eligible":
        return

    values = [
        sample_id,
        *(state[column] for column in m2.INPUT_COLUMNS),
    ]

    placeholders = ", ".join("?" for _ in values)

    db.execute(
        f"INSERT INTO m2_inputs VALUES ({placeholders})",
        values,
    )

    db.execute(
        "INSERT INTO m2_targets VALUES (?, ?)",
        (sample_id, event),
    )


def sample(
    target: str,
    *,
    sample_id: str = "fixture:1",
) -> m2.Sample:
    """순수 모델 평가에서 사용할 유효 타석 한 건."""

    return m2.Sample(
        sample_id=sample_id,
        game_id="fixture",
        day="2024-04-01",
        inputs=inputs(),
        target=target,
    )


class PlateAppearanceDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db = fixture()
        self.addCleanup(self.db.close)

        add_pa(
            self.db, "train-game", "2024-04-01", "train"
        )
        add_pa(
            self.db,
            "validation-game",
            "2025-04-01",
            "validation",
            event="walk",
            no_pitch=True,
        )

    def test_no_pitch_walk_and_exclusion_reasons(self) -> None:
        add_pa(
            self.db,
            "train-game",
            "2024-04-01",
            "train",
            number=2,
            event="walk",
            no_pitch=True,
        )

        add_pa(
            self.db,
            "train-game",
            "2024-04-01",
            "train",
            number=3,
            status="excluded",
            changed=True,
        )

        add_pa(
            self.db,
            "train-game",
            "2024-04-01",
            "train",
            number=4,
            status="excluded",
            event=None,
            label_kind="unlabeled_half_end_candidate",
        )

        samples, ledger = data.read_samples(self.db)

        self.assertEqual(len(samples["train"]), 2)
        self.assertEqual(
            ledger["train"]["eligible:no_actual_pitch"], 1
        )
        self.assertEqual(
            ledger["train"]["excluded:participant_changed"], 1
        )
        self.assertEqual(
            ledger["train"]["excluded:unlabeled_half_end_candidate"], 1
        )

        for item in samples["train"]:
            self.assertNotIn("no_actual_pitch", item.inputs)
            self.assertNotIn("event", item.inputs)
            self.assertNotIn("game_id", item.inputs)

    def test_final_test_target_is_not_evaluated(self) -> None:
        add_pa(
            self.db,
            "test-game",
            "2026-04-01",
            "test",
            event=None,
        )

        samples, _ = data.read_samples(self.db)

        self.assertEqual(set(samples), {"train", "validation"})
        self.assertEqual(len(samples["train"]), 1)
        self.assertEqual(len(samples["validation"]), 1)

    def test_wrong_date_split(self) -> None:
        self.db.execute(
            """
            UPDATE m2_index
            SET split='validation'
            WHERE game_id='train-game'
            """
        )

        with self.assertRaises(ValueError):
            data.read_samples(self.db)

    def test_same_game_cannot_cross_splits(self) -> None:
        add_pa(
            self.db,
            "train-game",
            "2025-04-01",
            "validation",
            number=2,
        )

        with self.assertRaises(ValueError):
            data.read_samples(self.db)

    def test_duplicate_input_key(self) -> None:
        self.db.execute(
            "INSERT INTO m2_inputs SELECT * FROM m2_inputs"
        )

        with self.assertRaises(ValueError):
            data.read_samples(self.db)

    def test_missing_and_orphan_target(self) -> None:
        with self.db:
            self.db.execute(
                """
                DELETE FROM m2_targets
                WHERE sample_id='train-game:1'
                """
            )

        with self.assertRaises(ValueError):
            data.read_samples(self.db)

        self.db.execute(
            """
            INSERT INTO m2_targets
            VALUES ('orphan:1', 'single')
            """
        )

        with self.assertRaises(ValueError):
            data.read_samples(self.db)

    def test_start_and_end_participant_mismatch(self) -> None:
        self.db.execute(
            """
            UPDATE m2_index
            SET end_pitcher_id='hf:replacement'
            WHERE sample_id='train-game:1'
            """
        )

        with self.assertRaises(ValueError):
            data.read_samples(self.db)

    def test_wrong_credited_participant(self) -> None:
        self.db.execute(
            """
            UPDATE m2_index
            SET credited_batter_candidate='hf:other'
            WHERE sample_id='train-game:1'
            """
        )

        with self.assertRaises(ValueError):
            data.read_samples(self.db)

    def test_wrong_start_input(self) -> None:
        self.db.execute(
            """
            UPDATE m2_inputs
            SET batter_id='hf:replacement'
            WHERE sample_id='train-game:1'
            """
        )

        with self.assertRaises(ValueError):
            data.read_samples(self.db)

    def test_unknown_and_missing_targets_fail_closed(self) -> None:
        for event in ("unregistered_event", None):
            with self.subTest(event=event):
                self.db.execute(
                    """
                    UPDATE m2_targets
                    SET target_event=?
                    WHERE sample_id='train-game:1'
                    """,
                    (event,),
                )

                self.db.execute(
                    """
                    UPDATE m2_index
                    SET event=?
                    WHERE sample_id='train-game:1'
                    """,
                    (event,),
                )

                with self.assertRaises(ValueError):
                    data.read_samples(self.db)

    def test_forbidden_input_column(self) -> None:
        self.db.execute(
            "ALTER TABLE m2_inputs ADD COLUMN target_event"
        )

        with self.assertRaises(ValueError):
            data.read_samples(self.db)

    def test_wrong_no_pitch_result(self) -> None:
        self.db.execute(
            """
            UPDATE m2_index
            SET physical_pitch_rows=0, no_actual_pitch=1
            WHERE sample_id='train-game:1'
            """
        )

        with self.assertRaises(ValueError):
            data.read_samples(self.db)

    def test_intake_counts_are_not_assumed(self) -> None:
        samples, ledger = data.read_samples(self.db)

        with self.assertRaises(ValueError):
            data.verify_intake(samples, ledger)


class PlateAppearanceModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.train = [
            sample(
                "single" if index % 3 == 0 else "field_out",
                sample_id=f"fixture:{index + 1}",
            )
            for index in range(36)
        ]

    def test_fixed_class_dictionary(self) -> None:
        self.assertEqual(set(m2.CLASSES), PA_EVENTS)
        self.assertEqual(len(m2.CLASSES), 15)

    def test_invalid_inputs_and_leakage(self) -> None:
        for key, value in (
            ("outs", 3),
            ("inning", 0),
            ("on_first", 2),
            ("on_second", True),
            ("half", "unknown"),
            ("home_score", -1),
            ("competition", "postseason"),
            ("batter_id", ""),
            ("pitcher_id", None),
            ("home_team_id", 2002),
            ("batter_prior_games", -1),
        ):
            with self.subTest(key=key):
                state = inputs()
                state[key] = value

                with self.assertRaises(ValueError):
                    m2.validate_inputs(state)

        for key in (
            "target_event",
            "end_batter_id",
            "no_actual_pitch",
            "prior_rejected",
            "game_id",
            "sample_id",
        ):
            with self.subTest(forbidden=key):
                state = inputs()
                state[key] = 0

                with self.assertRaises(ValueError):
                    m2.validate_inputs(state)

    def test_missing_history_is_not_observed_zero(self) -> None:
        missing = inputs()

        observed = inputs()
        observed.update(
            {
                "batter_prior_games": 1,
                "batter_prior_opportunities": 4,
                "batter_prior_ab": 4,
                "batter_prior_hits": 0,
                "batter_prior_so": 0,
            }
        )

        missing_features = m2.features(missing)
        observed_features = m2.features(observed)

        self.assertIn(
            "batter:prior_hit_rate:missing",
            missing_features,
        )
        self.assertNotIn(
            "batter:prior_hit_rate:missing",
            observed_features,
        )

        self.assertNotEqual(
            missing_features,
            observed_features,
        )

    def test_zero_denominator_and_impossible_counts(self) -> None:
        state = inputs()

        state.update(
            {
                "batter_prior_ab": 0,
                "batter_prior_hits": 0,
            }
        )

        encoded = m2.features(state)

        self.assertIn(
            "batter:prior_hit_rate:zero_denominator",
            encoded,
        )

        state["batter_prior_hits"] = 1

        with self.assertRaises(ValueError):
            m2.features(state)

    def test_frequency_fallback_and_unseen_classes(self) -> None:
        global_model, conditional = m2.fit_frequencies(self.train)

        base = m2.predict_probabilities(global_model, inputs())

        self.assertTrue(all(value > 0 for value in base))
        self.assertAlmostEqual(sum(base), 1.0)

        state = inputs()
        state["outs"] = 2
        state["on_first"] = 1

        self.assertEqual(
            m2.predict_probabilities(conditional, state),
            base,
        )

        small = self.train[:3]
        global_small, conditional_small = m2.fit_frequencies(small)

        self.assertEqual(
            m2.predict_probabilities(conditional_small, inputs()),
            m2.predict_probabilities(global_small, inputs()),
        )

    def test_supported_conditional_group(self) -> None:
        other_rows = []

        for index in range(36):
            state = inputs()
            state["outs"] = 2

            other_rows.append(
                replace(
                    sample(
                        "strikeout",
                        sample_id=f"other:{index + 1}",
                    ),
                    inputs=state,
                )
            )

        global_model, conditional = m2.fit_frequencies(
            self.train + other_rows
        )

        m2.validate_model(conditional)

        baseline = m2.predict_probabilities(
            global_model, inputs()
        )
        conditioned = m2.predict_probabilities(
            conditional, inputs()
        )

        self.assertGreater(
            conditioned[m2.CLASS_INDEX["field_out"]],
            baseline[m2.CLASS_INDEX["field_out"]],
        )

    def test_logistic_seed_and_unknown_player(self) -> None:
        first = m2.fit_logistic(self.train, 42)
        second = m2.fit_logistic(self.train, 42)

        self.assertEqual(first, second)

        self.assertIn(
            "id:batter:hf:batter",
            first["weights"],
        )

        unknown_a = inputs()
        unknown_b = inputs()

        unknown_a["batter_id"] = "hf:new-a"
        unknown_b["batter_id"] = "hf:new-b"

        unknown_prediction = m2.predict_probabilities(
            first, unknown_a
        )

        self.assertEqual(
            unknown_prediction,
            m2.predict_probabilities(first, unknown_b),
        )

        self.assertTrue(
            all(value > 0 for value in unknown_prediction)
        )

    def test_prediction_keys_and_probability_validation(self) -> None:
        model, _ = m2.fit_frequencies(self.train)
        forecast = m2.predict(model, inputs())

        self.assertEqual(
            forecast["class_order"], list(m2.CLASSES)
        )

        probabilities = forecast["probabilities"]

        self.assertIsInstance(probabilities, dict)
        self.assertEqual(set(probabilities), set(m2.CLASSES))
        self.assertAlmostEqual(sum(probabilities.values()), 1.0)

        for invalid in (
            [],
            [0.0] * m2.N_CLASSES,
            [float("nan")] * m2.N_CLASSES,
            [-1.0] * m2.N_CLASSES,
            [2.0] * m2.N_CLASSES,
        ):
            with self.assertRaises(ValueError):
                m2.check_probabilities(invalid)

    def test_asset_roundtrip_and_hash_mismatch(self) -> None:
        model = m2.fit_logistic(self.train, 42)

        payload = {
            "schema": m2.SCHEMA,
            "class_order": list(m2.CLASSES),
            "class_names": m2.CLASS_NAMES,
            "input_columns": list(m2.INPUT_COLUMNS),
            "method": m2.METHOD,
            "training_run_id": "a" * 32,
            "configured_seed": 42,
            "model": model,
        }

        with TemporaryDirectory() as directory:
            asset = Path(directory) / "model.json"

            write_json(
                asset,
                {
                    "payload": payload,
                    "sha256": fingerprint(payload),
                },
            )

            restored = workflow.load_model(asset)

            self.assertEqual(
                m2.predict(model, inputs()),
                m2.predict(restored, inputs()),
            )

            broken = Path(directory) / "broken.json"

            write_json(
                broken,
                {
                    "payload": payload,
                    "sha256": "0" * 64,
                },
            )

            with self.assertRaises(ValueError):
                workflow.load_model(broken)

    def test_corrupt_model_parameters(self) -> None:
        model, _ = m2.fit_frequencies(self.train)
        model["counts"] = [0] * m2.N_CLASSES

        with self.assertRaises(ValueError):
            m2.validate_model(model)


class PlateAppearanceEvaluationTests(unittest.TestCase):
    def test_multiclass_metrics_and_calibration(self) -> None:
        samples = [
            sample("single"),
            sample("double", sample_id="fixture:2"),
        ]

        probabilities = [0.0] * m2.N_CLASSES
        probabilities[m2.CLASS_INDEX["single"]] = 1.0

        result = evaluation.metrics(
            samples,
            [probabilities, probabilities],
        )

        self.assertAlmostEqual(
            result["log_loss"],
            -log(evaluation.LOG_EPSILON) / 2,
        )

        self.assertAlmostEqual(result["brier"], 1.0)
        self.assertAlmostEqual(result["accuracy"], 0.5)

        self.assertAlmostEqual(
            result["macro_f1"],
            (2.0 / 3.0) / m2.N_CLASSES,
        )

        self.assertEqual(
            result["confusion_matrix"]["double"]["single"],
            1,
        )

        for bins in result["calibration"].values():
            self.assertEqual(
                sum(int(str(item["count"])) for item in bins),
                len(samples),
            )

        final_bin = result["calibration"]["single"][-1]

        self.assertEqual(final_bin["count"], 2)
        self.assertEqual(final_bin["mean_probability"], 1.0)
        self.assertEqual(final_bin["observed_rate"], 0.5)

    def test_prediction_length_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            evaluation.metrics([sample("single")], [])

    def test_selection_rule(self) -> None:
        model, _ = m2.fit_frequencies([sample("single")])
        base = evaluation.evaluate(model, [sample("single")])

        results = {
            "global": {
                **base, "log_loss": 2.0, "brier": 0.8
            },
            "conditional": {
                **base, "log_loss": 1.8, "brier": 0.7
            },
            "logistic": {
                **base, "log_loss": 1.7, "brier": 0.6
            },
        }

        self.assertEqual(
            evaluation.select(results)[0],
            "logistic",
        )

        results["logistic"]["brier"] = 0.9

        self.assertEqual(
            evaluation.select(results)[0],
            "conditional",
        )

    def test_selected_class_frequencies(self) -> None:
        samples = [
            sample("single"),
            sample("walk"),
            sample("walk"),
        ]

        model, _ = m2.fit_frequencies(samples)
        expected = Counter(item.target for item in samples)

        self.assertEqual(
            dict(zip(m2.CLASSES, model["counts"], strict=True)),
            {name: expected[name] for name in m2.CLASSES},
        )


if __name__ == "__main__":
    unittest.main()