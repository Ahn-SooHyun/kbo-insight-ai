"""M1의 시간 경계, 결합 분포, 입력과 자산 계약을 합성 자료로 검증한다."""

from contextlib import closing
import json
from math import fsum
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from src.datasets.model_data import fingerprint
from src.evaluation import game_outcome as evaluation
from src.models.game_outcome import __main__ as workflow
from src.models.game_outcome import model as m1


def inputs() -> dict[str, object]:
    """시즌 첫 경기의 유효한 사전 입력을 만든다."""
    return {"competition": "regular", **{
        name: 0 if name.endswith("_games") else None for name in m1.INPUT_COLUMNS[1:]
    }}


def fixture() -> sqlite3.Connection:
    """최종 Test에 유효 정답이 없는 합성 DB를 만든다."""
    db = sqlite3.connect(":memory:")
    db.executescript("""
        CREATE TABLE m1_index (sample_id, game_id, sample_date, split, status);
        CREATE TABLE m1_targets (sample_id, target_home_runs, target_away_runs, target_result);
        INSERT INTO m1_index VALUES
          ('old','old','2024-09-01','train','eligible'),
          ('tie','tie','2024-09-02','train','eligible'),
          ('val','val','2025-04-01','validation','eligible'),
          ('post','post','2025-10-10','validation','eligible'),
          ('future','future','2026-04-01','test','eligible');
        INSERT INTO m1_targets VALUES ('old',7,5,'home'), ('tie',4,4,'tie'), ('val',2,6,'away');
    """)
    db.execute(f"CREATE TABLE m1_inputs (sample_id, {', '.join(m1.INPUT_COLUMNS)})")
    placeholders = ", ".join("?" for _ in range(len(m1.INPUT_COLUMNS) + 1))
    for sample in ("old", "tie", "val", "post"):
        row = inputs()
        if sample == "post":
            row["competition"] = "postseason"
        db.execute(f"INSERT INTO m1_inputs VALUES ({placeholders})", (sample, *row.values()))
    return db


class GameOutcomeTests(unittest.TestCase):
    def setUp(self) -> None:
        with closing(fixture()) as db:
            self.games, self.ledger = workflow.read_games(db)
        self.model = m1.fit(self.games["train"])

    def test_split_and_double_rows(self) -> None:
        self.assertEqual([g.sample_id for g in self.games["train"]], ["old", "tie"])
        self.assertEqual([g.runs for g in self.games["validation"]], [(2, 6)])
        self.assertEqual(self.ledger["validation"]["postseason"], 1)
        with closing(fixture()) as db:
            db.execute("UPDATE m1_index SET split='validation' WHERE sample_id='future'")
            with self.assertRaisesRegex(ValueError, "분할"):
                workflow.read_games(db)

    def test_rejects_duplicate_and_inconsistent_official_final(self) -> None:
        with closing(fixture()) as db:
            db.execute("INSERT INTO m1_targets SELECT * FROM m1_targets WHERE sample_id='val'")
            with self.assertRaisesRegex(ValueError, "중복"):
                workflow.read_games(db)
        with closing(fixture()) as db:
            db.execute("UPDATE m1_targets SET target_result='home' WHERE sample_id='tie'")
            with self.assertRaisesRegex(ValueError, "불일치"):
                workflow.read_games(db)

    def test_joint_marginals_outcomes_score_and_tail(self) -> None:
        prediction = m1.predict(self.model, inputs())
        joint = prediction["joint"]
        grid = [
            [h * a * (joint["equal_score_weight"] if x == y else joint["different_score_weight"])
             / joint["normalizer"] for y, a in enumerate(joint["away_base_pmf"])]
            for x, h in enumerate(joint["home_base_pmf"])
        ]
        for label in m1.OUTCOMES:
            expected = fsum(mass for x, row in enumerate(grid) for y, mass in enumerate(row)
                            if m1.score_outcome(x, y) == label)
            self.assertAlmostEqual(expected, prediction["result_probabilities"][label], places=12)
        for side, rows in (("home", grid), ("away", tuple(zip(*grid, strict=True)))):
            marginal = tuple(fsum(row) for row in rows)
            for actual, expected in zip(prediction[f"{side}_pmf"], marginal, strict=True):
                self.assertAlmostEqual(actual, expected, places=12)
            expected = m1.mean(marginal)
            self.assertAlmostEqual(prediction[f"{side}_expected_runs"], expected, places=12)
            self.assertEqual(prediction["representative_score"][side], int(expected + 0.5))
        self.assertLess(prediction["tail"]["omitted_joint_mass_bound"], 1e-12)
        self.assertAlmostEqual(evaluation.crps((0.0, 1.0), 3), 2.0)
        self.assertEqual(m1.score_outcome(11, 10), "home")

    def test_input_uses_allowlist_and_checks_missing_values(self) -> None:
        m1.predict(self.model, inputs())
        for changes in (
            {"target_home_runs": 2}, {"competition": "postseason"},
            {"home_recent_games": 1}, {"home_season_games": True},
            {"away_season_games": 1, "away_season_rf": None},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                m1.predict(self.model, inputs() | changes)
        missing = inputs()
        del missing["away_recent_ra"]
        with self.assertRaises(ValueError):
            m1.predict(self.model, missing)

    def test_asset_roundtrip_and_checksum(self) -> None:
        payload = {
            "schema": m1.SCHEMA, "method": workflow.METHOD,
            "input_columns": list(m1.INPUT_COLUMNS), "model": self.model,
        }
        wrapper = {"payload": payload, "sha256": fingerprint(payload)}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            path.write_text(json.dumps(wrapper), encoding="utf-8")
            self.assertEqual(m1.predict(workflow.load_model(path), inputs()), m1.predict(self.model, inputs()))
            path.write_text(json.dumps(wrapper | {"sha256": "damaged"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "무결성"):
                workflow.load_model(path)
            with self.assertRaises(FileNotFoundError):
                workflow.load_model(path.with_name("missing.json"))

    def test_metrics_use_same_games(self) -> None:
        report = evaluation.evaluate(self.model, self.games["train"], self.games["validation"])
        for key in ("baseline", "candidate"):
            self.assertEqual(report[key]["scores"]["pooled"]["team_rows"], 2)
            self.assertEqual(report[key]["outcomes"]["games"], 1)
            for outcome in m1.OUTCOMES:
                bins = report[key]["outcomes"]["calibration"][outcome]["bins"]
                self.assertEqual(sum(entry["games"] for entry in bins), 1)