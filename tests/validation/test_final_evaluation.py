"""고정 자산·최종 평가·누수 차단·저장 지표 재계산 검증."""

from contextlib import closing
from dataclasses import asdict
from datetime import date
import hashlib
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from models.test_game_state_value import state
from models.test_plate_appearance import inputs as pa_inputs
from models.test_player_performance import (
    counts,
    request,
    target_row,
)

from src.datasets.model_data import fingerprint
from src.evaluation import final_evaluation as workflow
from src.evaluation import final_scoring as scoring
from src.evaluation import fixed_assets as assets
from src.models.plate_appearance import model as m2
from src.models.plate_appearance import probability as p2
from src.models.pitch_outcome import model as m5
from src.models.pitch_outcome import data as d5
from src.models.player_performance import data as d3


def store_rows(
    directory: Path,
    predictions: list[dict[str, object]],
    targets: list[dict[str, object]],
) -> tuple[Path, Path]:
    """합성 추론과 정답을 실제 저장 계약으로 작성한다."""
    prediction_path = directory / "predictions.jsonl"
    target_path = directory / "targets.jsonl"

    workflow.write_rows(
        prediction_path,
        predictions,
    )

    workflow.write_rows(
        target_path,
        targets,
    )

    return prediction_path, target_path


class HistoricalAssetTests(unittest.TestCase):
    """기존 저장 형식의 실제 바이트 지문 검증."""

    def test_exact_original_bytes(self) -> None:
        """원래 바이트가 일치하면 직접 지문을 사용한다."""
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"

            original = assets.original_json_bytes(
                {"payload": {"value": 1}}
            )

            path.write_bytes(original)

            expected = hashlib.sha256(
                original
            ).hexdigest()

            report = assets.verify_historical_asset(
                path,
                expected,
            )

            self.assertEqual(
                report["mode"],
                "exact_bytes",
            )

            self.assertEqual(
                path.read_bytes(),
                original,
            )

    def test_reconstructed_original_bytes(self) -> None:
        """표현 형식이 달라도 기존 바이트를 증명하면 허용한다."""
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"

            value = {
                "payload": {"b": 2, "a": 1},
                "fingerprint": "example",
            }

            original = assets.original_json_bytes(
                value
            )

            expected = hashlib.sha256(
                original
            ).hexdigest()

            current = json.dumps(
                value,
                separators=(",", ":"),
            ).encode("utf-8")

            path.write_bytes(current)

            report = assets.verify_historical_asset(
                path,
                expected,
            )

            self.assertEqual(
                report["mode"],
                "original_serialization_reconstructed",
            )

            self.assertNotEqual(
                report["actual_sha256"],
                expected,
            )

            self.assertEqual(
                report["reconstructed_sha256"],
                expected,
            )

            self.assertEqual(
                path.read_bytes(),
                current,
            )

    def test_modified_payload_rejected(self) -> None:
        """내용이 달라진 모델은 원래 저장 형식으로도 통과할 수 없다."""
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"

            original = assets.original_json_bytes(
                {"payload": {"value": 1}}
            )

            expected = hashlib.sha256(
                original
            ).hexdigest()

            path.write_bytes(
                assets.original_json_bytes(
                    {"payload": {"value": 2}}
                )
            )

            with self.assertRaises(ValueError):
                assets.verify_historical_asset(
                    path,
                    expected,
                )

    def test_duplicate_json_keys_rejected(self) -> None:
        """바이트 지문 일치와 별개로 중복 JSON 키를 거부한다."""
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"

            raw = b'{"value":1,"value":2}'

            path.write_bytes(raw)

            with self.assertRaises(ValueError):
                assets.verify_historical_asset(
                    path,
                    hashlib.sha256(raw).hexdigest(),
                )


class FinalBoundaryTests(unittest.TestCase):
    """시간 분할·원본 보호·정답 접근 차단."""

    def test_fixed_split_boundaries(self) -> None:
        """세 분할의 포함 및 제외 경계를 확인한다."""
        cases = (
            ("2024-12-31", "train"),
            ("2025-01-01", "validation"),
            ("2025-12-31", "validation"),
            ("2026-01-01", "test"),
            ("2026-11-25", "test"),
        )

        for day, split in cases:
            with self.subTest(day=day):
                workflow.validate_day(day, split)

        with self.assertRaises(ValueError):
            workflow.validate_day(
                "2026-11-26",
                "test",
            )

        with self.assertRaises(ValueError):
            workflow.validate_day(
                "2026-01-01",
                "validation",
            )

    def test_prediction_authorizer(self) -> None:
        """추론 연결은 허용 입력만 읽고 정답·사후 열을 거부한다."""
        with closing(
            sqlite3.connect(":memory:")
        ) as db:
            db.executescript(
                """
                CREATE TABLE m1_index(
                    sample_id TEXT,
                    game_id TEXT,
                    event TEXT
                );

                CREATE TABLE m1_inputs(
                    sample_id TEXT,
                    competition TEXT,
                    target_home_runs INTEGER
                );

                CREATE TABLE m1_targets(
                    sample_id TEXT,
                    target_home_runs INTEGER
                );

                INSERT INTO m1_index
                    VALUES ('g1', 'g1', 'final');

                INSERT INTO m1_inputs
                    VALUES ('g1', 'regular', 3);

                INSERT INTO m1_targets
                    VALUES ('g1', 3);
                """
            )

            db.set_authorizer(
                workflow.prediction_authorizer(
                    "m1",
                    ("competition",),
                )
            )

            result = db.execute(
                """
                SELECT x.sample_id, i.competition
                FROM m1_index x
                JOIN m1_inputs i USING(sample_id)
                """
            ).fetchone()

            self.assertEqual(
                result,
                ("g1", "regular"),
            )

            for query in (
                "SELECT event FROM m1_index",
                "SELECT target_home_runs FROM m1_inputs",
                "SELECT target_home_runs FROM m1_targets",
                "DELETE FROM m1_inputs",
            ):
                with self.subTest(query=query):
                    with self.assertRaises(
                        sqlite3.DatabaseError
                    ):
                        db.execute(query)

    def test_read_only_database(self) -> None:
        """원본 DB 연결에서 실제 쓰기를 차단한다."""
        with TemporaryDirectory() as directory:
            path = Path(directory) / "data.sqlite3"

            with closing(sqlite3.connect(path)) as db:
                db.execute(
                    "CREATE TABLE sample(value INTEGER)"
                )

                db.execute(
                    "INSERT INTO sample VALUES (1)"
                )

                db.commit()

            before = assets.describe_file(path)

            with workflow.open_database(path) as db:
                self.assertEqual(
                    db.execute(
                        "SELECT value FROM sample"
                    ).fetchone()[0],
                    1,
                )

                with self.assertRaises(
                    sqlite3.DatabaseError
                ):
                    db.execute(
                        "UPDATE sample SET value=2"
                    )

            self.assertEqual(
                assets.describe_file(path),
                before,
            )

    def test_development_pitch_loader_blocks_test(self) -> None:
        """기존 M5 학습 경로의 Test 차단을 유지한다."""
        with closing(
            sqlite3.connect(":memory:")
        ) as db:
            with self.assertRaises(ValueError):
                d5.read_split(db, "test")

    def test_final_state_query_is_separate(self) -> None:
        """M4 최종 SQL은 개발 분할을 조회하지 않는다."""
        query = workflow.m4_final_query()

        self.assertIn(
            "i.split='test'",
            query,
        )

        self.assertNotIn(
            "'train'",
            query,
        )

        self.assertNotIn(
            "'validation'",
            query,
        )


class SavedArtifactTests(unittest.TestCase):
    """명세·저장 결과·표본 키 재현 계약."""

    def test_manifest_hash_mismatch_rejected(self) -> None:
        """고정 명세 지문 불일치 시 성능 평가를 차단한다."""
        with TemporaryDirectory() as directory:
            root = Path(directory)

            output = (
                root / "final_evaluation" / "run"
            )

            output.mkdir(parents=True)

            manifest = output / "manifest.json"

            manifest.write_text(
                "{}",
                encoding="utf-8",
            )

            settings = SimpleNamespace(
                results_dir=root
            )

            with self.assertRaises(ValueError):
                workflow.load_manifest(
                    settings,
                    manifest,
                    "0" * 64,
                    verify_assets=False,
                )

    def test_saved_file_tampering_rejected(self) -> None:
        """평가 이후 산출물 수정이 있으면 재계산을 차단한다."""
        with TemporaryDirectory() as directory:
            root = Path(directory)

            path = root / "predictions.jsonl"

            path.write_text(
                '{"sample_id":"a"}\n',
                encoding="utf-8",
            )

            original = assets.describe_file(path)

            self.assertEqual(
                workflow.checked_file(
                    original,
                    root,
                ),
                path.resolve(),
            )

            path.write_text(
                '{"sample_id":"b"}\n',
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                workflow.checked_file(
                    original,
                    root,
                )

    def test_duplicate_prediction_key_rejected(self) -> None:
        """동일 예측 키의 중복 저장을 거부한다."""
        with TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"

            with self.assertRaises(ValueError):
                workflow.write_rows(
                    path,
                    [
                        {"sample_id": "a"},
                        {"sample_id": "a"},
                    ],
                )

    def test_prediction_target_mismatch_rejected(self) -> None:
        """예측과 실제 정답 키가 다르면 계산하지 않는다."""
        with TemporaryDirectory() as directory:
            prediction, target = store_rows(
                Path(directory),
                [{"sample_id": "a"}],
                [{"sample_id": "b"}],
            )

            with self.assertRaises(ValueError):
                list(
                    scoring.paired_rows(
                        prediction,
                        target,
                    )
                )

    def test_missing_target_rejected(self) -> None:
        """예측과 정답 행 수 차이를 거부한다."""
        with TemporaryDirectory() as directory:
            prediction, target = store_rows(
                Path(directory),
                [
                    {"sample_id": "a"},
                    {"sample_id": "b"},
                ],
                [{"sample_id": "a"}],
            )

            with self.assertRaises(ValueError):
                list(
                    scoring.paired_rows(
                        prediction,
                        target,
                    )
                )

    def test_source_rebuild_detects_difference(self) -> None:
        """동일 수량이라도 원천 재생성 값이 다르면 실패한다."""
        with closing(
            sqlite3.connect(":memory:")
        ) as db:
            db.executescript(
                """
                CREATE TABLE rebuilt(
                    sample_id TEXT,
                    value INTEGER
                );

                CREATE TABLE locked(
                    sample_id TEXT,
                    value INTEGER
                );

                INSERT INTO rebuilt VALUES ('a', 1);
                INSERT INTO locked VALUES ('a', 2);
                """
            )

            report = workflow.compare_sql_rows(
                db,
                "SELECT * FROM rebuilt",
                "SELECT * FROM locked",
            )

            self.assertFalse(
                report["equal"]
            )

            self.assertEqual(
                report["rebuilt_only"],
                1,
            )

            self.assertEqual(
                report["locked_only"],
                1,
            )

    def test_common_denominator_validation(self) -> None:
        """공통 분모는 양쪽 유효 분모를 초과할 수 없다."""
        self.assertEqual(
            scoring.population(
                10, 8, 7, 6
            )["excluded_from_common"],
            4,
        )

        with self.assertRaises(ValueError):
            scoring.population(
                10, 8, 7, 9
            )

    def test_improvement_directions(self) -> None:
        """오차·분류·구간 지표의 방향을 구분한다."""
        self.assertEqual(
            scoring.difference(
                1.0, 2.0
            )["judgment"],
            "improved",
        )

        self.assertEqual(
            scoring.difference(
                0.8,
                0.7,
                direction="higher",
            )["judgment"],
            "improved",
        )

        self.assertEqual(
            scoring.difference(
                0.91,
                0.99,
                direction="target",
                target=0.90,
            )["judgment"],
            "improved",
        )

        self.assertEqual(
            scoring.difference(
                None,
                1.0,
            )["judgment"],
            "unavailable",
        )


class PlayerFinalTargetTests(unittest.TestCase):
    """M3 최종 시점·기간·Count 및 분모 검사."""

    def test_final_target_allowed_only_in_final_loader(self) -> None:
        """2026 정답은 최종 평가 전용 검사에서만 허용한다."""
        player = d3.parse_input(
            request(
                "hitter",
                date(2026, 6, 1),
            )
        )

        row = target_row(
            player,
            "7d",
            1,
        )

        result = scoring.parse_final_target(
            row,
            player,
        )

        self.assertTrue(result.usable)

        with self.assertRaises(ValueError):
            d3.parse_target(
                row,
                player,
                "test",
            )

    def test_observed_zero_is_not_missing(self) -> None:
        """관측 0 Count는 유효 정답이며 비율 분모는 별도 처리한다."""
        player = d3.parse_input(
            request(
                "pitcher",
                date(2026, 6, 1),
            )
        )

        row = target_row(
            player,
            "28d",
            0,
        )

        result = scoring.parse_final_target(
            row,
            player,
        )

        self.assertTrue(result.usable)

        self.assertTrue(
            all(value == 0 for value in result.counts)
        )

        self.assertIsNone(
            result.hit_rate
        )

        self.assertIsNone(
            result.strikeout_rate
        )

    def test_unfinished_remaining_is_masked(self) -> None:
        """완료 근거 없는 잔여 정답을 관측 0으로 처리하지 않는다."""
        player = d3.parse_input(
            request(
                "hitter",
                date(2026, 6, 1),
            )
        )

        row = target_row(
            player,
            "remaining",
            1,
        )

        row.update({
            "window_end": None,
            "ready_date": None,
            "status": "partial_unfinished",
            "usable": 0,
            "ratio_status": "unavailable",
            "target_hit_rate": None,
            "target_strikeout_rate": None,
        })

        for name in d3.metrics(player.role):
            row[f"target_{name}"] = None

        result = scoring.parse_final_target(
            row,
            player,
        )

        self.assertFalse(
            result.usable
        )

        self.assertTrue(
            all(value is None for value in result.counts)
        )

    def test_target_risk_rejected(self) -> None:
        """품질 위험이 있는 Count 정답 사용을 차단한다."""
        player = d3.parse_input(
            request(
                "hitter",
                date(2026, 6, 1),
            )
        )

        row = target_row(
            player,
            "7d",
            1,
        )

        row["missing_games"] = 1

        with self.assertRaises(ValueError):
            scoring.parse_final_target(
                row,
                player,
            )


class FinalScoringTests(unittest.TestCase):
    """실제 저장 방식과 동일한 합성 지표 재계산."""

    def test_m1_saved_scores(self) -> None:
        """득점·승무패·꼬리 진단을 저장 자료에서 재계산한다."""
        forecast = {
            "home_pmf": [0.5, 0.5],
            "away_pmf": [0.5, 0.5],
            "result_probabilities": {
                "home": 0.25,
                "tie": 0.50,
                "away": 0.25,
            },
            "tail": {
                "omitted_joint_mass_bound": 0.0,
                "expected_runs_error_bound_per_side": 0.0,
            },
        }

        rows = [{
            "sample_id": "game1",
            "sample_date": "2026-04-01",
            "input": {},
            "operating": forecast,
            "baseline": forecast,
        }]

        truth = [{
            "sample_id": "game1",
            "target": {
                "home": 1,
                "away": 0,
                "result": "home",
            },
        }]

        with TemporaryDirectory() as directory:
            prediction, target = store_rows(
                Path(directory),
                rows,
                truth,
            )

            result = scoring.score_m1(
                prediction,
                target,
            )

            self.assertEqual(
                result["population"]["common_valid"],
                1,
            )

            self.assertEqual(
                result["comparison"]["scores"][
                    "pooled"
                ]["mae"]["judgment"],
                "equal",
            )

            self.assertEqual(
                fingerprint(result),
                fingerprint(
                    scoring.score_m1(
                        prediction,
                        target,
                    )
                ),
            )

    def test_m2_saved_probabilities(self) -> None:
        """M2 고정 범주와 허용 확률의 평가를 재계산한다."""
        inputs = pa_inputs()

        mask = p2.allowed_mask(inputs)

        count = sum(mask)

        probabilities = [
            1.0 / count if allowed else 0.0
            for allowed in mask
        ]

        target_class = next(
            label
            for label, allowed in zip(
                m2.CLASSES,
                mask,
            )
            if allowed
        )

        rows = [{
            "sample_id": "game1:1",
            "game_id": "game1",
            "sample_date": "2026-04-01",
            "input": inputs,
            "operating": probabilities,
            "baseline": probabilities,
        }]

        truth = [{
            "sample_id": "game1:1",
            "target": target_class,
        }]

        baseline = {
            "name": "global",
            "train_counts": [1] * m2.N_CLASSES,
        }

        with TemporaryDirectory() as directory:
            prediction, target = store_rows(
                Path(directory),
                rows,
                truth,
            )

            result = scoring.score_m2(
                prediction,
                target,
                baseline,
            )

            self.assertTrue(
                result["contract_valid"]
            )

            self.assertEqual(
                result["population"]["common_valid"],
                1,
            )

            self.assertEqual(
                fingerprint(result),
                fingerprint(
                    scoring.score_m2(
                        prediction,
                        target,
                        baseline,
                    )
                ),
            )

    def test_m4_terminal_and_nonterminal_scores(self) -> None:
        """확정 종료와 미종료 상태의 성능을 구분한다."""
        post = asdict(
            state(
                season=2026,
                phase="post",
                home_score=4,
            )
        )

        pre = asdict(
            state(season=2026)
        )

        forecast = {
            "win": [0.5, "logistic_core", None],
            "re": [0.5, "global", 100],
        }

        rows = [
            {
                "sample_id": "game1:1:post",
                "input": post,
                "operating": forecast,
                "baseline": forecast,
            },
            {
                "sample_id": "game1:1:pre",
                "input": pre,
                "operating": forecast,
                "baseline": forecast,
            },
        ]

        truth = [
            {
                "sample_id": "game1:1:post",
                "win": 1,
                "remaining": 0,
                "confirmed_final": True,
            },
            {
                "sample_id": "game1:1:pre",
                "win": 1,
                "remaining": 1,
                "confirmed_final": False,
            },
        ]

        with TemporaryDirectory() as directory:
            prediction, target = store_rows(
                Path(directory),
                rows,
                truth,
            )

            result = scoring.score_m4(
                prediction,
                target,
            )

            op = result["results"]["operating"]

            self.assertEqual(
                op["win"]["confirmed_final"]["log_loss"],
                0.0,
            )

            self.assertEqual(
                op["win"]["nonterminal"]["count"],
                1,
            )

            self.assertGreater(
                op["win"]["nonterminal"]["log_loss"],
                0.0,
            )

            self.assertEqual(
                fingerprint(result),
                fingerprint(
                    scoring.score_m4(
                        prediction,
                        target,
                    )
                ),
            )

    def test_m5_joint_marginals_and_unseen_support(self) -> None:
        """공동·주변 축, 미출현 범주 및 ECE를 재계산한다."""
        probabilities = [
            1.0 / m5.N_CLASSES
        ] * m5.N_CLASSES

        rows = [{
            "sample_id": "game1:1:1",
            "operating": probabilities,
            "frequency": probabilities,
            "conditional_frequency": probabilities,
        }]

        truth = [{
            "sample_id": "game1:1:1",
            "target": 0,
        }]

        baseline = {
            "train_counts": (
                [0] + [1] * (m5.N_CLASSES - 1)
            ),
        }

        with TemporaryDirectory() as directory:
            prediction, target = store_rows(
                Path(directory),
                rows,
                truth,
            )

            result = scoring.score_m5(
                prediction,
                target,
                baseline,
            )

            op = result["results"]["operating"]

            for axis in (
                "joint",
                "pitch_type",
                "pitch_result",
            ):
                self.assertEqual(
                    op[axis]["samples"],
                    1,
                )

                self.assertIn(
                    "macro_classwise_ece",
                    op[axis],
                )

            self.assertEqual(
                op["joint"]["train_support_groups"][
                    "unseen"
                ]["count"],
                1,
            )

            self.assertEqual(
                fingerprint(result),
                fingerprint(
                    scoring.score_m5(
                        prediction,
                        target,
                        baseline,
                    )
                ),
            )

    def test_m3_hitter_and_pitcher_common_ratios(self) -> None:
        """양 역할의 Count, 구간, 비율 공통 분모를 재계산한다."""
        for role in ("hitter", "pitcher"):
            with self.subTest(role=role):
                inputs = request(
                    role,
                    date(2026, 6, 1),
                )

                player = d3.parse_input(inputs)

                operational = {}
                baseline_predictions = {}
                targets = {}

                for horizon, factor in zip(
                    d3.HORIZONS,
                    (1, 3, 8),
                ):
                    actual = counts(role, factor)

                    targets[horizon] = target_row(
                        player,
                        horizon,
                        factor,
                    )

                    operational[horizon] = {
                        "counts": actual,
                        "intervals": {
                            name: {
                                "lower": 0.0,
                                "upper": value + 1.0,
                                "reason": None,
                            }
                            for name, value in actual.items()
                        },
                    }

                    base_counts = counts(
                        role,
                        0 if horizon == "7d" else factor,
                    )

                    baseline_predictions[horizon] = {
                        "counts": base_counts,
                        "intervals": {
                            name: {
                                "lower": 0.0,
                                "upper": value + 1.0,
                                "reason": None,
                            }
                            for name, value in base_counts.items()
                        },
                    }

                rows = [{
                    "sample_id": player.sample_id,
                    "input": inputs,
                    "operating": operational,
                    "baseline": baseline_predictions,
                }]

                truth = [{
                    "sample_id": player.sample_id,
                    "season_end_exclusive": "2026-10-15",
                    "targets": targets,
                }]

                baseline_asset = {
                    "scales": {
                        key: 1.0
                        for key in d3.output_keys(role)
                    },
                }

                with TemporaryDirectory() as directory:
                    prediction, target = store_rows(
                        Path(directory),
                        rows,
                        truth,
                    )

                    result = scoring.score_m3(
                        prediction,
                        target,
                        role,
                        baseline_asset,
                    )

                    ratio = result["ratios"][
                        "7d.hit_rate"
                    ]

                    self.assertEqual(
                        ratio["population"][
                            "operating_valid"
                        ],
                        1,
                    )

                    self.assertEqual(
                        ratio["population"][
                            "baseline_valid"
                        ],
                        0,
                    )

                    self.assertEqual(
                        ratio["population"][
                            "common_valid"
                        ],
                        0,
                    )

                    self.assertEqual(
                        ratio["comparison"][
                            "mae"
                        ]["judgment"],
                        "unavailable",
                    )

                    self.assertEqual(
                        result["records"][
                            "7d.games"
                        ]["results"][
                            "operating"
                        ]["mae"],
                        0.0,
                    )

                    self.assertEqual(
                        fingerprint(result),
                        fingerprint(
                            scoring.score_m3(
                                prediction,
                                target,
                                role,
                                baseline_asset,
                            )
                        ),
                    )


if __name__ == "__main__":
    unittest.main()
