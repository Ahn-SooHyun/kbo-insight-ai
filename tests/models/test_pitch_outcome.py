"""M5 입력·공동 확률·원천 경계·평가·자산 검증 테스트."""

import json
from math import log
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.datasets.model_data import fingerprint
from src.evaluation import pitch_outcome as evaluation
from src.models.pitch_outcome import SCHEMA, VERSION
from src.models.pitch_outcome import data
from src.models.pitch_outcome import model as m5
from src.models.pitch_outcome.__main__ import (
    load_runtime,
    method_settings,
    save_json,
)


SOURCE_SHA = "a" * 64


def example_input() -> dict[str, object]:
    """현재 실제 공의 정답을 포함하지 않는 유효한 첫 공 입력."""
    return {
        "season": 2025,
        "competition": "regular",
        "inning": 1,
        "half": "top",
        "batter_id": "batter-1",
        "pitcher_id": "pitcher-1",
        "pitch_number": 1,
        "balls": 0,
        "strikes": 0,
        "outs": 0,
        "home_score": 0,
        "away_score": 0,
        "on_first": 0,
        "on_second": 0,
        "on_third": 0,
        "previous_pitch_type": None,
        "previous_pitch_result": None,
        "batter_prior_opportunities": 100,
        "batter_prior_so": 20,
        "pitcher_prior_opportunities": 100,
        "pitcher_prior_so": 25,
    }


def example_sample(
    joint: str = "FF:B",
    inputs: dict[str, object] | None = None,
) -> m5.Sample:
    """합성 특징과 공동 정답을 구성한다."""
    return m5.Sample(
        m5.encode(example_input() if inputs is None else inputs),
        m5.CLASS_INDEX[joint],
    )


def previous_row() -> dict[str, object]:
    """원본 순서가 검증된 두 번째 공의 감사 자료."""
    return {
        "audit_pitch_number": 2,
        "source_sha": SOURCE_SHA,
        "source_row": 2,
        "previous_number": 1,
        "previous_order_valid": 1,
        "previous_sha": SOURCE_SHA,
        "previous_source_row": 1,
        "previous_target_valid": 1,
        "previous_state_valid": 1,
        "previous_actual_type": "FF",
        "previous_actual_result": "B",
    }


def make_model_tables(db: sqlite3.Connection) -> None:
    """실제 계약과 같은 열을 갖는 합성 모델 DB를 구성한다."""
    db.executescript(
        """
        CREATE TABLE m5_index (
            sample_id TEXT,
            game_id TEXT,
            sample_date TEXT,
            split TEXT,
            status TEXT,
            pa_number INTEGER,
            pitch_number INTEGER,
            source_sha TEXT,
            source_row INTEGER
        );

        CREATE TABLE m5_targets (
            sample_id TEXT,
            target_pitch_type TEXT,
            target_pitch_result TEXT,
            target_joint TEXT
        );
        """
    )

    columns = ", ".join(
        f'"{name}"' for name in m5.ALL_COLUMNS
    )

    db.execute(
        f"CREATE TABLE m5_inputs (sample_id TEXT, {columns})"
    )


def insert_sample(
    db: sqlite3.Connection,
    day: str,
    split: str,
    game_id: str,
    number: int,
    inputs: dict[str, object] | None = None,
    joint: str = "FF:B",
) -> None:
    """실제 자료와 동일한 표본 키 규칙을 적용한다."""
    sample_id = f"{game_id}:1:{number}"

    db.execute(
        """
        INSERT INTO m5_index
        VALUES (?, ?, ?, ?, 'eligible', 1, ?, ?, ?)
        """,
        (
            sample_id,
            game_id,
            day,
            split,
            number,
            SOURCE_SHA,
            number,
        ),
    )

    raw = example_input() if inputs is None else inputs
    values = [sample_id] + [
        raw.get(name) for name in m5.ALL_COLUMNS
    ]

    placeholders = ", ".join("?" for _ in values)

    db.execute(
        f"INSERT INTO m5_inputs VALUES ({placeholders})",
        values,
    )

    pitch_type, result = joint.split(":")

    db.execute(
        "INSERT INTO m5_targets VALUES (?, ?, ?, ?)",
        (sample_id, pitch_type, result, joint),
    )


class PitchInputTests(unittest.TestCase):
    def test_dictionary(self) -> None:
        self.assertEqual(len(m5.TYPES), 10)
        self.assertEqual(len(m5.RESULTS), 7)
        self.assertEqual(len(m5.CLASSES), 70)
        self.assertEqual(len(set(m5.CLASSES)), 70)

        for index, label in enumerate(m5.CLASSES):
            pitch_type, result = label.split(":")
            self.assertEqual(
                m5.target_index(pitch_type, result, label),
                index,
            )

    def test_first_pitch(self) -> None:
        keys = m5.encode(example_input())

        self.assertIn("previous:first", keys)
        self.assertIn("count:0:0", keys)

    def test_first_pitch_cannot_have_previous(self) -> None:
        inputs = {
            **example_input(),
            "previous_pitch_type": "FF",
            "previous_pitch_result": "B",
        }

        with self.assertRaises(ValueError):
            m5.encode(inputs)

    def test_middle_missing_is_not_first(self) -> None:
        inputs = {
            **example_input(),
            "pitch_number": 3,
        }

        keys = m5.encode(inputs)

        self.assertIn("previous:missing", keys)
        self.assertNotIn("previous:first", keys)

    def test_previous_pitch(self) -> None:
        inputs = {
            **example_input(),
            "pitch_number": 2,
            "previous_pitch_type": "FF",
            "previous_pitch_result": "B",
        }

        self.assertIn(
            "previous:FF:B", m5.encode(inputs)
        )

    def test_previous_partial_missing(self) -> None:
        inputs = {
            **example_input(),
            "pitch_number": 2,
            "previous_pitch_type": "FF",
        }

        with self.assertRaises(ValueError):
            m5.encode(inputs)

    def test_current_pitch_leakage_rejected(self) -> None:
        for field in (
            "target_joint",
            "target_pitch_type",
            "target_pitch_result",
            "pitch_type",
            "pitch_result",
            "release_speed_kmh",
            "plate_x",
            "plate_z",
            "post_outs",
            "end_batter_id",
            "event",
            "sample_id",
            "source_row",
        ):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    m5.encode({
                        **example_input(),
                        field: "forbidden",
                    })

    def test_invalid_inputs(self) -> None:
        for name, value in (
            ("balls", 4),
            ("strikes", 3),
            ("outs", 3),
            ("pitch_number", 0),
            ("pitch_number", True),
            ("inning", 0),
            ("half", "unknown"),
            ("competition", "postseason"),
            ("on_first", 2),
            ("home_score", -1),
            ("pitcher_id", ""),
            ("batter_prior_so", 101),
        ):
            with self.subTest(name=name, value=value):
                with self.assertRaises(ValueError):
                    m5.encode({
                        **example_input(),
                        name: value,
                    })

    def test_missing_input(self) -> None:
        inputs = example_input()
        del inputs["balls"]

        with self.assertRaises(ValueError):
            m5.encode(inputs)

    def test_target_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            m5.target_index("FF", "B", "SL:B")

    def test_unknown_target(self) -> None:
        with self.assertRaises(ValueError):
            m5.target_index("unknown", "B", "unknown:B")

    def test_history_missing_preserved(self) -> None:
        inputs = {
            **example_input(),
            "batter_prior_opportunities": None,
            "batter_prior_so": None,
        }

        self.assertIn(
            "history:batter:missing", m5.encode(inputs)
        )


class PitchBoundaryTests(unittest.TestCase):
    def test_first_pitch(self) -> None:
        row = {
            **previous_row(),
            "audit_pitch_number": 1,
            "previous_number": None,
        }

        self.assertEqual(
            data.audit_previous(row),
            (None, None, "first_pitch", False),
        )

    def test_valid_previous(self) -> None:
        self.assertEqual(
            data.audit_previous(previous_row()),
            ("FF", "B", "previous_available", False),
        )

    def test_missing_previous_number(self) -> None:
        row = {
            **previous_row(),
            "previous_number": None,
        }

        self.assertEqual(
            data.audit_previous(row),
            (None, None, "missing_previous_number", True),
        )

    def test_wrong_previous_number(self) -> None:
        row = {
            **previous_row(),
            "audit_pitch_number": 3,
        }

        with self.assertRaises(ValueError):
            data.audit_previous(row)

    def test_invalid_previous_order(self) -> None:
        row = {
            **previous_row(),
            "previous_order_valid": 0,
        }

        self.assertEqual(
            data.audit_previous(row),
            (None, None, "previous_order_invalid", True),
        )

    def test_wrong_source(self) -> None:
        row = {
            **previous_row(),
            "previous_sha": "b" * 64,
        }

        self.assertEqual(
            data.audit_previous(row),
            (
                None, None,
                "previous_source_order_invalid", True,
            ),
        )

    def test_future_source_row(self) -> None:
        row = {
            **previous_row(),
            "previous_source_row": 3,
        }

        self.assertEqual(
            data.audit_previous(row),
            (
                None, None,
                "previous_source_order_invalid", True,
            ),
        )

    def test_invalid_previous_target(self) -> None:
        row = {
            **previous_row(),
            "previous_target_valid": 0,
        }

        self.assertEqual(
            data.audit_previous(row),
            (None, None, "previous_target_invalid", True),
        )

    def test_previous_model_exclusion(self) -> None:
        row = {
            **previous_row(),
            "previous_state_valid": 0,
        }

        self.assertEqual(
            data.audit_previous(row),
            ("FF", "B", "previous_model_excluded", True),
        )

    def test_date_boundaries(self) -> None:
        self.assertEqual(
            data.expected_split("2024-12-31"), "train"
        )
        self.assertEqual(
            data.expected_split("2025-01-01"), "validation"
        )
        self.assertEqual(
            data.expected_split("2026-01-01"), "test"
        )

        with self.assertRaises(ValueError):
            data.expected_split("2026-11-26")


class PitchProbabilityTests(unittest.TestCase):
    def test_frequency_positive_unseen(self) -> None:
        model = m5.fit_frequency([example_sample()])
        runtime = m5.Runtime(model)
        joint = runtime.probabilities(
            m5.encode(example_input())
        )

        self.assertEqual(len(joint), 70)
        self.assertTrue(all(value > 0 for value in joint))
        self.assertAlmostEqual(sum(joint), 1.0)

        self.assertGreater(
            joint[m5.CLASS_INDEX["SV:V"]], 0.0
        )

    def test_conditional_frequency_and_backoff(self) -> None:
        first = example_sample("FF:B")
        second_inputs = {
            **example_input(),
            "balls": 1,
            "pitch_number": 2,
        }
        second = example_sample("SL:S", second_inputs)

        model = m5.fit_frequency(
            [first] * 60 + [second] * 10,
            conditional=True,
        )

        runtime = m5.Runtime(model)

        self.assertGreater(
            runtime.probabilities(first.keys)[first.target],
            runtime.prior[first.target],
        )

        self.assertEqual(
            runtime.probabilities(second.keys),
            runtime.prior,
        )

    def test_marginals_match_axes(self) -> None:
        runtime = m5.Runtime(
            m5.fit_frequency([example_sample()])
        )

        prediction = m5.predict(runtime, example_input())

        joint = prediction["joint"]
        pitch_types = prediction["pitch_type"]
        results = prediction["pitch_result"]

        self.assertAlmostEqual(sum(joint.values()), 1.0)
        self.assertAlmostEqual(sum(pitch_types.values()), 1.0)
        self.assertAlmostEqual(sum(results.values()), 1.0)

        for pitch_type in m5.TYPES:
            self.assertAlmostEqual(
                pitch_types[pitch_type],
                sum(
                    joint[f"{pitch_type}:{result}"]
                    for result in m5.RESULTS
                ),
            )

        for result in m5.RESULTS:
            self.assertAlmostEqual(
                results[result],
                sum(
                    joint[f"{pitch_type}:{result}"]
                    for pitch_type in m5.TYPES
                ),
            )

    def test_invalid_probabilities(self) -> None:
        for probabilities in (
            [],
            [float("nan")],
            [-0.1, 1.1],
            [0.1, 0.2],
        ):
            with self.subTest(probabilities=probabilities):
                with self.assertRaises(ValueError):
                    m5.check_probabilities(probabilities)

    def test_logistic_seed_reproduction(self) -> None:
        samples = [
            example_sample("FF:B"),
            example_sample("SL:S"),
            example_sample("CU:H"),
        ]

        first = m5.fit_logistic(samples, 42)
        second = m5.fit_logistic(samples, 42)

        self.assertEqual(first, second)
        self.assertEqual(first["kind"], "logistic")

    def test_unknown_player_backoff(self) -> None:
        model = m5.fit_logistic(
            [example_sample("FF:B")], 42
        )

        runtime = m5.Runtime(model)

        unknown = {
            **example_input(),
            "batter_id": "unknown-batter",
            "pitcher_id": "unknown-pitcher",
        }

        self.assertNotIn(
            "id:batter:value:unknown-batter",
            model["weights"],
        )

        self.assertEqual(
            m5.predict(runtime, example_input()),
            m5.predict(runtime, unknown),
        )

    def test_bad_model_shape(self) -> None:
        model = m5.fit_frequency([example_sample()])
        model["counts"].pop()

        with self.assertRaises(ValueError):
            m5.validate_model(model)


class PitchEvaluationTests(unittest.TestCase):
    def test_uniform_log_loss_and_brier(self) -> None:
        score = evaluation.Score(
            m5.CLASSES, [1] * m5.N_CLASSES
        )

        score.add(
            0, [1.0 / m5.N_CLASSES] * m5.N_CLASSES
        )

        report = score.report()

        self.assertAlmostEqual(
            report["log_loss"], log(m5.N_CLASSES)
        )

        self.assertAlmostEqual(
            report["brier"], 1.0 - 1.0 / m5.N_CLASSES
        )

    def test_unseen_validation_retained(self) -> None:
        model = m5.fit_frequency(
            [example_sample("FF:B")]
        )

        metrics = evaluation.evaluate(
            model,
            [
                example_sample("FF:B"),
                example_sample("SV:V"),
            ],
        )

        for axis in evaluation.AXES:
            result = metrics[axis]

            self.assertEqual(result["samples"], 2)
            self.assertEqual(
                result["train_support_groups"]["unseen"]["count"],
                1,
            )
            self.assertEqual(
                result["unsupported_output_samples"], 0
            )

            self.assertEqual(
                sum(
                    item["count"]
                    for item in result["top_label_calibration"]
                ),
                2,
            )

    def test_selection_and_tie_break(self) -> None:
        metrics = {
            name: {
                "joint": {
                    "samples": 10,
                    "log_loss": 1.0,
                    "brier": 0.8,
                }
            }
            for name in m5.METHODS
        }

        self.assertEqual(
            evaluation.select(metrics), "frequency"
        )

        metrics["logistic"]["joint"]["log_loss"] = 0.9

        self.assertEqual(
            evaluation.select(metrics), "logistic"
        )


class PitchDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db = sqlite3.connect(":memory:")
        self.db.row_factory = sqlite3.Row

        make_model_tables(self.db)

    def tearDown(self) -> None:
        self.db.close()

    def test_key_integrity_and_test_isolation(self) -> None:
        insert_sample(
            self.db, "2024-05-01", "train", "G1", 1
        )

        insert_sample(
            self.db, "2025-05-01", "validation", "G2", 1
        )

        insert_sample(
            self.db, "2026-05-01", "test", "G3", 1
        )

        # 키 검사에서 어떠한 정답 값도 직접 읽지 못하게 한다.
        def authorizer(
            action: int,
            table: str | None,
            column: str | None,
            database: str | None,
            trigger: str | None,
        ) -> int:
            if (
                action == sqlite3.SQLITE_READ
                and table == "m5_targets"
                and column in data.TARGET_COLUMNS[1:]
            ):
                return sqlite3.SQLITE_DENY

            return sqlite3.SQLITE_OK

        self.db.set_authorizer(authorizer)

        data.verify_keys(self.db)

    def test_duplicate_target(self) -> None:
        insert_sample(
            self.db, "2025-05-01", "validation", "G1", 1
        )

        self.db.execute(
            "INSERT INTO m5_targets SELECT * FROM m5_targets"
        )

        with self.assertRaises(ValueError):
            data.verify_keys(self.db)

    def test_missing_input(self) -> None:
        insert_sample(
            self.db, "2025-05-01", "validation", "G1", 1
        )
        self.db.execute("DELETE FROM m5_inputs")

        with self.assertRaises(ValueError):
            data.verify_keys(self.db)

    def test_same_game_mixed_split(self) -> None:
        insert_sample(
            self.db, "2024-05-01", "train", "G1", 1
        )
        insert_sample(
            self.db, "2025-05-01", "validation", "G1", 2
        )

        with self.assertRaises(ValueError):
            data.verify_keys(self.db)

    def test_non_pitch_index_rejected(self) -> None:
        insert_sample(
            self.db, "2025-05-01", "validation", "G1", 0
        )

        with self.assertRaises(ValueError):
            data.verify_keys(self.db)

    def test_final_test_read_rejected(self) -> None:
        with self.assertRaises(ValueError):
            data.read_split(self.db, "test")

    def test_standard_source_integration(self) -> None:
        """표준 실제 공, 제외 공, 무투구 및 직전 공을 대조한다."""
        self.db.execute(
            "ATTACH DATABASE ':memory:' AS verified"
        )

        self.db.executescript(
            """
            CREATE TABLE verified.games (
                game_id TEXT,
                date TEXT,
                season INTEGER,
                competition TEXT,
                result_valid INTEGER,
                status TEXT
            );

            CREATE TABLE verified.eligible_states (
                game_id TEXT,
                pa_number INTEGER,
                inning INTEGER,
                half TEXT
            );

            CREATE TABLE verified.pitches (
                game_id TEXT,
                pa_number INTEGER,
                pitch_number INTEGER,
                batter_id TEXT,
                pitcher_id TEXT,
                balls INTEGER,
                strikes INTEGER,
                outs_when_up INTEGER,
                home_score INTEGER,
                away_score INTEGER,
                on_1b TEXT,
                on_2b TEXT,
                on_3b TEXT,
                source_sha TEXT,
                source_row INTEGER,
                target_valid INTEGER,
                order_valid INTEGER,
                state_valid INTEGER,
                pitch_type TEXT,
                pitch_result TEXT,
                pitch_result_type TEXT
            );

            CREATE TABLE verified.non_pitch (
                game_id TEXT
            );

            CREATE TABLE verified.game_checks (
                game_id TEXT,
                positive_pitch_rows INTEGER,
                non_pitch_rows INTEGER
            );

            INSERT INTO verified.games
            VALUES ('G1', '2025-05-01', 2025, 'regular', 1, 'Final');

            INSERT INTO verified.eligible_states
            VALUES ('G1', 1, 1, 'top');

            INSERT INTO verified.non_pitch VALUES ('G1');

            INSERT INTO verified.game_checks VALUES ('G1', 3, 1);
            """
        )

        for number, pitch_type, result, result_type, valid in (
            (1, "FF", "B", "B", 1),
            (2, "SL", "S", "S", 1),
            (3, None, None, None, 0),
        ):
            values = (
                "G1", 1, number,
                "batter-1", "pitcher-1",
                number - 1, 0, 0, 0, 0,
                None, None, None,
                SOURCE_SHA, number,
                valid, 1, 1,
                pitch_type, result, result_type,
            )

            placeholders = ", ".join("?" for _ in values)

            self.db.execute(
                f"INSERT INTO verified.pitches VALUES ({placeholders})",
                values,
            )

        insert_sample(
            self.db,
            "2025-05-01",
            "validation",
            "G1",
            1,
        )

        insert_sample(
            self.db,
            "2025-05-01",
            "validation",
            "G1",
            2,
            {
                **example_input(),
                "pitch_number": 2,
                "balls": 1,
                "previous_pitch_type": "FF",
                "previous_pitch_result": "B",
            },
            "SL:S",
        )

        data.verify_keys(self.db)

        with patch.dict(
            data.EXPECTED, {"validation": (2, 1)}
        ):
            rows, summary, examples = data.read_split(
                self.db, "validation"
            )

        self.assertEqual(len(rows), 2)
        self.assertEqual(summary["first_pitch_count"], 1)
        self.assertEqual(summary["previous_missing_count"], 1)

        self.assertIn(
            "validation:previous_available", examples
        )

        exclusions = data.source_exclusions(self.db)["validation"]

        self.assertEqual(exclusions["eligible"], 2)
        self.assertEqual(exclusions["excluded"], 1)
        self.assertEqual(exclusions["non_pitch_rows"], 1)

        self.assertEqual(
            exclusions["exclusive_reasons"]["invalid_target"], 1
        )

        # 실제 원천과 다른 이전 공을 삽입하면 학습을 거부한다.
        self.db.execute(
            """
            UPDATE m5_inputs
            SET previous_pitch_type = 'CU'
            WHERE sample_id = 'G1:1:2'
            """
        )

        with patch.dict(
            data.EXPECTED, {"validation": (2, 1)}
        ):
            with self.assertRaises(ValueError):
                data.read_split(self.db, "validation")


class PitchAssetTests(unittest.TestCase):
    def test_saved_model_reproduction_and_tamper(self) -> None:
        model = m5.fit_logistic(
            [
                example_sample("FF:B"),
                example_sample("SL:S"),
            ],
            42,
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
            path = Path(directory) / "model.json"

            save_json(
                path,
                {
                    "payload": payload,
                    "sha256": fingerprint(payload),
                },
            )

            self.assertEqual(
                m5.predict(m5.Runtime(model), example_input()),
                m5.predict(load_runtime(path), example_input()),
            )

            modified = json.loads(
                path.read_text(encoding="utf-8")
            )
            modified["payload"]["configured_seed"] = 43

            path.write_text(
                json.dumps(modified), encoding="utf-8"
            )

            with self.assertRaises(ValueError):
                load_runtime(path)


if __name__ == "__main__":
    unittest.main()