"""M3 사전 입력·정답 마스크·공동 회귀·구간·재현 테스트."""

from copy import deepcopy
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from uuid import uuid4

from src.datasets.model_data import fingerprint
from src.datasets.published_sources import write_json
from src.models.player_performance import SCHEMA, VERSION
from src.models.player_performance import data, evaluation, model
from src.models.player_performance.__main__ import (
    compare,
    load_asset,
    method_settings,
    runtime,
    source_hashes,
)


def counts(role: str, factor: int) -> dict[str, int]:
    """구성 관계를 만족하는 합성 Count."""
    result = {
        "games": factor,
        "ab": 3 * factor,
        "hits": factor,
        "home_runs": 0,
        "walks": factor,
        "hbp": 0,
        "strikeouts": factor,
    }

    if role == "hitter":
        result["pa"] = 4 * factor
    else:
        result.update({
            "bf": 4 * factor,
            "outs": 2 * factor,
            "pitches": 15 * factor,
        })

    return result


def request(
    role: str = "hitter",
    anchor: date = date(2023, 6, 5),
    index: int = 1,
) -> dict[str, object]:
    """월요일 이전에 확인한 합성 선수 입력."""
    player_id = f"hf:{index}"
    row = {
        "role": role,
        "sample_id": (
            f"{anchor.year}:{anchor.isoformat()}:{player_id}"
        ),
        "player_id": player_id,
        "season": anchor.year,
        "sample_date": anchor.isoformat(),
        "last_observed_date": (
            anchor - timedelta(days=1)
        ).isoformat(),
        "known_team_id": 1,
        "days_since_observed": 1,
        "prior_rejected": 0,
        "season_active": True,
    }

    for prefix, factor in (
        ("season", 5 + index % 5),
        ("recent28", 1 + index % 5),
    ):
        row.update({
            f"{prefix}_{name}": value
            for name, value in counts(role, factor).items()
        })

    return row


def target_row(
    player: data.PlayerInput,
    horizon: str,
    factor: int,
) -> dict[str, object]:
    """데이터 생성기의 기간별 정답 스키마에 맞춘 합성 행."""
    end = (
        date(player.season, 10, 15)
        if horizon == "remaining"
        else player.anchor + timedelta(
            days=7 if horizon == "7d" else 28
        )
    )
    season_end = date(player.season, 10, 15)
    values = counts(player.role, factor)

    row = {
        "sample_id": player.sample_id,
        "horizon": horizon,
        "window_start": player.anchor.isoformat(),
        "window_end": end.isoformat(),
        "observed_through_exclusive":
            min(end, season_end).isoformat(),
        "ready_date": end.isoformat(),
        "status": (
            "valid_observed" if factor else "zero_observed"
        ),
        "usable": 1,
        "ratio_status": (
            "not_requested"
            if horizon == "7d"
            else "valid" if factor else "zero_denominator"
        ),
        "target_hit_rate": (
            None if horizon == "7d" or not factor else 1 / 3
        ),
        "target_strikeout_rate": (
            None if horizon == "7d" or not factor else 1 / 4
        ),
        **dict.fromkeys(data.RISK_FIELDS, 0),
    }

    for name, value in values.items():
        row[f"target_{name}"] = value
        row[f"observed_{name}"] = value

    return row


def sample(
    role: str,
    anchor: date,
    index: int,
) -> data.Sample:
    """단일 선수의 관측 완료 공동 정답."""
    row = request(role, anchor, index)
    player = data.parse_input(row)
    split = data.split_for(anchor)
    base = index % 6 + 1

    targets = tuple(
        data.parse_target(
            target_row(player, horizon, base * multiplier),
            player,
            split,
        )
        for horizon, multiplier in zip(
            data.HORIZONS, (1, 3, 8)
        )
    )

    return data.Sample(row, player, split, targets)


def samples(
    role: str,
    anchor: date,
    n: int = 100,
) -> list[data.Sample]:
    """여러 선수·기준 주가 있는 합성 시간순 자료."""
    return [
        sample(
            role,
            anchor + timedelta(weeks=index % 3),
            index,
        )
        for index in range(n)
    ]


class InputTests(unittest.TestCase):
    """모집단·역할·사전 시점·누적 입력 계약."""

    def test_role_dimensions(self) -> None:
        self.assertEqual(len(data.output_keys("hitter")), 24)
        self.assertEqual(len(data.output_keys("pitcher")), 30)

    def test_monday_and_identity(self) -> None:
        player = data.parse_input(request())
        self.assertEqual(player.anchor.weekday(), 0)
        self.assertLess(player.last_observed, player.anchor)

    def test_non_monday_rejected(self) -> None:
        row = request(anchor=date(2023, 6, 6))
        with self.assertRaises(ValueError):
            data.parse_input(row)

    def test_same_day_history_rejected(self) -> None:
        row = request()
        row["last_observed_date"] = row["sample_date"]
        row["days_since_observed"] = 0

        with self.assertRaises(ValueError):
            data.parse_input(row)

    def test_identity_mismatch_rejected(self) -> None:
        row = request()
        row["player_id"] = "hf:different"

        with self.assertRaises(ValueError):
            data.parse_input(row)

    def test_future_feature_rejected(self) -> None:
        row = request()
        row["target_hits"] = 10

        with self.assertRaises(ValueError):
            data.parse_input(row)

    def test_recent_count_cannot_exceed_season(self) -> None:
        row = request()
        row["recent28_games"] = 1000

        with self.assertRaises(ValueError):
            data.parse_input(row)

    def test_missing_history_not_zero_count(self) -> None:
        row = request()

        for name in data.metrics("hitter"):
            row[f"recent28_{name}"] = (
                0 if name == "games" else None
            )

        player = data.parse_input(row)
        features = model.features(player)

        self.assertIsNone(player.recent_counts[1])
        self.assertIn(-1.0, features)

    def test_count_composition(self) -> None:
        row = counts("hitter", 1)
        row["hits"] = 100

        with self.assertRaises(ValueError):
            data.check_counts("hitter", row)

    def test_current_cumulative_scope(self) -> None:
        row = request()
        row["current_counts"] = counts("hitter", 20)
        row["current_scope"] = "caller_verified"
        row["current_as_of"] = row["sample_date"]

        player = data.parse_input(row)

        self.assertEqual(
            player.current_scope, "caller_verified"
        )
        self.assertEqual(player.current_counts[0], 20)

    def test_wrong_cumulative_boundary(self) -> None:
        row = request()
        row["current_counts"] = counts("hitter", 20)
        row["current_scope"] = "caller_verified"
        row["current_as_of"] = "2023-06-12"

        with self.assertRaises(ValueError):
            data.parse_input(row)

    def test_finished_season_rejected(self) -> None:
        row = request()
        row["season_active"] = False

        with self.assertRaises(ValueError):
            data.parse_input(row)

    def test_end_known_after_anchor_rejected(self) -> None:
        row = request()
        row["known_end_exclusive"] = "2023-06-10"
        row["known_end_as_of"] = "2023-06-06"
        row["known_end_source"] = "synthetic evidence"

        with self.assertRaises(ValueError):
            data.parse_input(row)


class TargetTests(unittest.TestCase):
    """관측 완료·실제 0·부분 정답·분할 검사."""

    def setUp(self) -> None:
        self.player = data.parse_input(request())

    def test_fixed_window(self) -> None:
        row = target_row(self.player, "7d", 2)
        target = data.parse_target(row, self.player, "train")

        self.assertEqual(
            target.window_end,
            self.player.anchor + timedelta(days=7),
        )
        self.assertEqual(target.ratio_status, "not_requested")

    def test_wrong_window_rejected(self) -> None:
        row = target_row(self.player, "28d", 2)
        row["window_end"] = "2023-06-20"

        with self.assertRaises(ValueError):
            data.parse_target(row, self.player, "train")

    def test_clean_zero(self) -> None:
        row = target_row(self.player, "28d", 0)
        target = data.parse_target(row, self.player, "train")

        self.assertTrue(target.usable)
        self.assertTrue(all(value == 0 for value in target.counts))
        self.assertEqual(target.ratio_status, "zero_denominator")
        self.assertIsNone(target.hit_rate)

    def test_risky_zero_rejected(self) -> None:
        row = target_row(self.player, "28d", 0)
        row["league_risky_games"] = 1

        with self.assertRaises(ValueError):
            data.parse_target(row, self.player, "train")

    def test_partial_observed_not_target(self) -> None:
        row = target_row(self.player, "28d", 2)
        row["status"] = "partial"
        row["usable"] = 0
        row["ratio_status"] = "unavailable"
        row["target_hit_rate"] = None
        row["target_strikeout_rate"] = None

        for name in data.metrics("hitter"):
            row[f"target_{name}"] = None

        target = data.parse_target(row, self.player, "train")

        self.assertFalse(target.usable)
        self.assertTrue(
            all(value is None for value in target.counts)
        )
        self.assertGreater(row["observed_games"], 0)

    def test_unusable_target_value_rejected(self) -> None:
        row = target_row(self.player, "28d", 2)
        row["status"] = "partial"
        row["usable"] = 0

        with self.assertRaises(ValueError):
            data.parse_target(row, self.player, "train")

    def test_logical_cutoff_masks_future_target(self) -> None:
        row = sample("hitter", date(2023, 6, 5), 1)
        cutoff = row.player.anchor + timedelta(days=7)
        labels = data.labels_for(row, cutoff)
        width = len(data.metrics("hitter"))

        self.assertTrue(
            all(value is not None for value in labels[:width])
        )
        self.assertTrue(
            all(value is None for value in labels[width:])
        )

    def test_final_test_boundary(self) -> None:
        self.assertEqual(
            data.split_for(date(2024, 12, 30)), "train"
        )
        self.assertEqual(
            data.split_for(date(2025, 1, 6)), "validation"
        )
        self.assertEqual(
            data.split_for(date(2026, 1, 5)), "test"
        )

    def test_valid_ratio_requires_matching_count(self) -> None:
        row = target_row(self.player, "28d", 2)
        row["target_hit_rate"] = 0.9

        with self.assertRaises(ValueError):
            data.parse_target(row, self.player, "train")


class JointModelTests(unittest.TestCase):
    """공동 출력·공유 분기·마스크·시즌 Count 합산 검사."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.training = samples(
            "hitter", date(2023, 6, 5)
        )
        cls.fitted = model.fit(
            "hitter",
            cls.training,
            data.CALIBRATION_START,
            2,
        )

    def test_shared_tree_dimensions(self) -> None:
        self.assertIsNotNone(self.fitted.tree)

        predicted = self.fitted.predict(
            self.training[0].player
        )

        self.assertEqual(len(predicted), 24)
        self.assertEqual(
            len(self.fitted.tree.values), 24
        )

    def test_pitcher_joint_dimensions(self) -> None:
        rows = samples(
            "pitcher", date(2023, 6, 5)
        )
        fitted = model.fit(
            "pitcher",
            rows,
            data.CALIBRATION_START,
            2,
        )

        self.assertEqual(
            len(fitted.predict(rows[0].player)), 30
        )

    def test_shared_split_and_mask_support(self) -> None:
        rows = [
            model.TrainingRow(
                (float(i),),
                (
                    (0, -1.0 if i < 50 else 1.0),
                    *(
                        ((1, 2.0 if i < 50 else -2.0),)
                        if i % 2 == 0 else ()
                    ),
                ),
            )
            for i in range(100)
        ]

        tree = model.fit_tree(rows, 2, 2)

        self.assertEqual(tree.support, (100, 50))
        self.assertIsNotNone(tree.feature)
        self.assertNotEqual(
            tree.predict((0.0,)),
            tree.predict((99.0,)),
        )

    def test_mask_excluded_from_tree_support(self) -> None:
        rows = self.training.copy()
        original = rows[0]
        target = replace(
            original.targets[1],
            usable=False,
            status="partial",
            counts=tuple(
                None for _ in original.targets[1].counts
            ),
            ratio_status="unavailable",
            hit_rate=None,
            strikeout_rate=None,
        )
        rows[0] = replace(
            original,
            targets=(
                original.targets[0],
                target,
                original.targets[2],
            ),
        )

        fitted = model.fit(
            "hitter", rows, data.CALIBRATION_START, 2
        )

        width = len(data.metrics("hitter"))

        self.assertEqual(
            fitted.tree.support[0], len(rows)
        )
        self.assertEqual(
            fitted.tree.support[width], len(rows) - 1
        )

    def test_nonnegative_monotonic_projection(self) -> None:
        player = self.training[0].player
        names = data.metrics(player.role)
        raw = (
            *([8.0] * len(names)),
            *([-1.0] * len(names)),
            *([20.0] * len(names)),
        )
        projected = model.project(player, raw)

        for i, _ in enumerate(names):
            self.assertLessEqual(
                projected[i],
                projected[len(names) + i],
            )
            self.assertLessEqual(
                projected[len(names) + i],
                projected[2 * len(names) + i],
            )

        for start in range(0, len(projected), len(names)):
            data.check_counts(
                player.role,
                dict(zip(
                    names,
                    projected[start:start + len(names)],
                )),
            )

    def test_known_season_end_equalizes_periods(self) -> None:
        row = request()
        row["known_end_exclusive"] = "2023-06-10"
        row["known_end_as_of"] = "2023-06-04"
        row["known_end_source"] = "synthetic pre-anchor evidence"

        player = data.parse_input(row)
        points = self.fitted.predict(player)
        width = len(data.metrics("hitter"))

        self.assertEqual(points[:width], points[width:2 * width])
        self.assertEqual(points[:width], points[2 * width:])

    def test_zero_denominator(self) -> None:
        result = model.ratio(0.0, 0.0)

        self.assertIsNone(result["value"])
        self.assertEqual(result["reason"], "zero_denominator")

    def test_outs_to_innings(self) -> None:
        row = counts("pitcher", 2)
        row["outs"] = 5

        result = model.derived("pitcher", row)

        self.assertAlmostEqual(result["innings"], 5 / 3)

    def test_season_projection_uses_counts(self) -> None:
        player = self.training[0].player
        intervals = {
            key: {"radius": 2.0, "reason": None}
            for key in data.output_keys("hitter")
        }
        result = model.format_prediction(
            self.fitted,
            player,
            intervals,
        )

        remaining = result["periods"]["remaining"]["counts"]
        season = result["season_projection"]
        current = season["current_counts"]

        for name in data.metrics("hitter"):
            self.assertAlmostEqual(
                season["counts"][name],
                current[name] + remaining[name],
            )

        final = season["counts"]

        if final["ab"]:
            self.assertAlmostEqual(
                season["derived"]["hit_rate"]["value"],
                final["hits"] / final["ab"],
            )

        self.assertEqual(
            season["status"],
            "observed_pbp_provisional_projection",
        )

    def test_restore_matches_prediction(self) -> None:
        restored = model.restore(
            self.fitted.to_payload()
        )
        player = self.training[0].player

        self.assertEqual(
            restored.predict(player),
            self.fitted.predict(player),
        )

    def test_corrupt_tree_dimension_rejected(self) -> None:
        payload = deepcopy(self.fitted.to_payload())
        payload["tree"]["values"] = [0.0]

        with self.assertRaises(ValueError):
            model.restore(payload)


class EvaluationTests(unittest.TestCase):
    """시간순 구간 적합과 동일 평가 표본 검증."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.training = samples(
            "hitter", date(2023, 6, 5)
        )
        cls.calibration = samples(
            "hitter", date(2024, 6, 3), 30
        )
        cls.validation = samples(
            "hitter", date(2025, 6, 2), 30
        )
        cls.fitted = model.fit(
            "hitter",
            cls.training,
            data.CALIBRATION_START,
            2,
        )

    def test_calibration_support(self) -> None:
        intervals = evaluation.calibrate(
            self.fitted,
            self.calibration,
            data.VALIDATION_START,
        )

        row = intervals["7d.games"]

        self.assertEqual(row["n"], 30)
        self.assertEqual(row["anchor_weeks"], 3)
        self.assertIsNotNone(row["radius"])

    def test_sparse_calibration(self) -> None:
        intervals = evaluation.calibrate(
            self.fitted,
            self.calibration[:2],
            data.VALIDATION_START,
        )

        self.assertIsNone(
            intervals["7d.games"]["radius"]
        )

    def test_interval_contains_point_and_nonnegative(self) -> None:
        interval = model.prediction_interval(
            1.0,
            {"radius": 3.0},
        )

        self.assertEqual(interval["lower"], 0.0)
        self.assertEqual(interval["upper"], 4.0)

    def test_known_end_interval_not_calibrated(self) -> None:
        interval = model.prediction_interval(
            1.0,
            {"radius": 3.0},
            known_end=True,
        )

        self.assertIsNone(interval["lower"])
        self.assertIsNone(interval["upper"])

    def test_count_and_ratio_denominators(self) -> None:
        intervals = evaluation.calibrate(
            self.fitted,
            self.calibration,
            data.VALIDATION_START,
        )
        report = evaluation.evaluate(
            self.fitted,
            self.validation,
            data.TEST_START,
            intervals,
        )

        self.assertEqual(
            report["record_metrics"]["7d.games"]["n"],
            len(self.validation),
        )
        self.assertEqual(
            report["ratio_metrics"]["7d"]["truth_source"],
            "count_derived_diagnostic",
        )
        self.assertEqual(
            report["ratio_metrics"]["28d"]["hit_rate"][
                "truth_valid_n"
            ],
            len(self.validation),
        )

    def test_development_selects_single_joint_tree(self) -> None:
        dataset = data.Dataset(
            train=self.training + self.calibration,
            validation=self.validation,
            provenance={},
            audit={},
        )
        developed = evaluation.develop(dataset, "hitter")

        self.assertIsNotNone(developed.selected.tree)
        self.assertIn(
            developed.selected.depth,
            model.CANDIDATE_DEPTHS,
        )
        self.assertEqual(
            len(developed.report["candidates"]), 3
        )
        self.assertFalse(
            developed.report["final_test_used"]
        )


class AssetTests(unittest.TestCase):
    """저장 자산·지문·DB 없는 재로딩을 검증한다."""

    def test_asset_round_trip_and_corrupt_fingerprint(self) -> None:
        training = samples(
            "hitter", date(2023, 6, 5)
        )
        calibration = samples(
            "hitter", date(2024, 6, 3), 30
        )
        fitted = model.fit(
            "hitter", training, data.CALIBRATION_START, 2
        )
        intervals = evaluation.calibrate(
            fitted, calibration, data.VALIDATION_START
        )
        root = Path(__file__).resolve().parents[2]

        payload = {
            "schema": SCHEMA,
            "version": VERSION,
            "run_id": uuid4().hex,
            "role": "hitter",
            "runtime": runtime(),
            "method": method_settings(),
            "source_hashes": source_hashes(root),
            "features": list(
                model.feature_names("hitter")
            ),
            "outputs": list(
                data.output_keys("hitter")
            ),
            "selected_model": fitted.name,
            "model": fitted.to_payload(),
            "intervals": intervals,
            "final_test_used": False,
        }

        player = training[0].player
        expected = model.format_prediction(
            fitted, player, intervals
        )

        with TemporaryDirectory() as directory:
            folder = Path(directory)
            asset = folder / "model.json"
            corrupted = folder / "corrupted.json"
            expected_path = folder / "expected.json"
            actual_path = folder / "actual.json"

            write_json(
                asset,
                {
                    "payload": payload,
                    "fingerprint": fingerprint(payload),
                },
            )
            restored, loaded_intervals, metadata = load_asset(asset)
            actual = model.format_prediction(
                restored,
                player,
                loaded_intervals,
            )

            self.assertEqual(expected, actual)
            self.assertEqual(metadata["operating_predictors"], 1)

            write_json(expected_path, expected)
            write_json(actual_path, actual)

            self.assertTrue(
                compare(expected_path, actual_path)["equal"]
            )

            write_json(
                corrupted,
                {
                    "payload": payload,
                    "fingerprint": "invalid",
                },
            )

            with self.assertRaises(ValueError):
                load_asset(corrupted)


if __name__ == "__main__":
    unittest.main()