"""시즌 전망·보조 분석 독립 검산의 정상 및 경계 테스트."""

from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.simulation.postseason.rules import get_rules as postseason_rules
from src.simulation.regular_season.rules import get_rules as regular_rules
from src.simulation.support_analysis.data import (
    Appearance,
    Identity,
    Start,
    StarterData,
)
from src.validation import season_forecast_checks as checks
from src.validation.season_forecast import (
    build_jobs,
    write_json,
)


class FileContractTest(unittest.TestCase):
    """검증 명세 및 파일 보존 조건을 검사한다."""

    def test_json_roundtrip(self) -> None:
        """신규 JSON 저장 및 읽기 결과가 일치해야 한다."""
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "result.json"
            expected = {"status": "completed", "value": None}

            write_json(path, expected)

            self.assertEqual(checks.read_json(path), expected)
            self.assertEqual(len(checks.file_hash(path)), 64)

    def test_existing_file_is_not_overwritten(self) -> None:
        """기존 검증 결과의 덮어쓰기를 거부해야 한다."""
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "result.json"

            write_json(path, {"original": True})

            with self.assertRaises(FileExistsError):
                write_json(path, {"original": False})

            self.assertEqual(
                checks.read_json(path),
                {"original": True},
            )

    def test_file_change_changes_digest(self) -> None:
        """파일 내용 변경을 SHA-256으로 탐지해야 한다."""
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "file.txt"

            path.write_text("before", encoding="utf-8")
            before = checks.file_hash(path)

            path.write_text("after", encoding="utf-8")

            self.assertNotEqual(before, checks.file_hash(path))

    def test_bool_is_not_integer(self) -> None:
        """실제 이진 사건과 정수 자료의 타입을 구별한다."""
        with self.assertRaises(ValueError):
            checks.equal(1, True, "bool")

    def test_none_is_not_zero(self) -> None:
        """미관측 상태를 실제 0으로 인정하지 않는다."""
        with self.assertRaises(ValueError):
            checks.equal(0, None, "missing")

    def test_float_tolerance(self) -> None:
        """부동소수점의 표현 오차만 허용한다."""
        checks.equal(0.1 + 0.2, 0.3, "float")

        with self.assertRaises(ValueError):
            checks.equal(0.31, 0.3, "float")


class DistributionTest(unittest.TestCase):
    """유효 분모 및 미결정 사건의 독립 집계를 검증한다."""

    def test_partial_event_probability(self) -> None:
        """확정 및 미결정 결과를 서로 다른 분모로 집계한다."""
        result = checks.probability([True, False, None, True])

        self.assertEqual(result["total"], 4)
        self.assertEqual(result["valid"], 3)
        self.assertEqual(result["unresolved"], 1)
        self.assertEqual(result["positive"], 2)
        self.assertEqual(result["probability"], 2 / 3)
        self.assertEqual(result["lower_bound"], 0.5)
        self.assertEqual(result["upper_bound"], 0.75)

    def test_all_unresolved(self) -> None:
        """유효 분모가 없는 확률을 임의 확정하지 않는다."""
        result = checks.probability([None, None])

        self.assertIsNone(result["probability"])
        self.assertEqual(result["lower_bound"], 0.0)
        self.assertEqual(result["upper_bound"], 1.0)
        self.assertEqual(result["status"], "unsupported")

    def test_empty_conditional_group(self) -> None:
        """성립하지 않은 대진은 조건부 확률을 제공하지 않는다."""
        result = checks.probability([])

        self.assertIsNone(result["probability"])
        self.assertIsNone(result["lower_bound"])
        self.assertIsNone(result["upper_bound"])

    def test_event_integer_rejected(self) -> None:
        """정수 1을 이진 사건의 True로 암묵 변환하지 않는다."""
        with self.assertRaises(ValueError):
            checks.probability([1])

    def test_distribution_reasons(self) -> None:
        """미결정 원인을 범주별로 보존한다."""
        result = checks.distribution(
            [1, None, 0, None],
            [None, "tie", None, "tie"],
            (0, 1),
        )

        self.assertEqual(result["valid"], 2)
        self.assertEqual(result["unresolved_reasons"], {"tie": 2})
        self.assertEqual(result["counts"], {"0": 1, "1": 1})
        self.assertEqual(result["expected"], 0.5)

    def test_invalid_distribution_category(self) -> None:
        """허용되지 않은 순위 범주를 차단한다."""
        with self.assertRaises(ValueError):
            checks.distribution([11], [None], range(1, 11))

    def test_distribution_without_observations(self) -> None:
        """모든 값이 미결정일 때 확률과 평균은 null이어야 한다."""
        result = checks.distribution(
            [None],
            ["missing"],
            (0, 1),
        )

        self.assertIsNone(result["expected"])
        self.assertEqual(
            result["probabilities"],
            {"0": None, "1": None},
        )


class StarterTest(unittest.TestCase):
    """선발 평가 분모와 시점 절단을 검증한다."""

    def test_starter_denominators(self) -> None:
        """후보가 없는 정답도 전체 식별 정답에 포함한다."""
        samples = [
            {
                "actual_player_id": "A",
                "baseline": ["A", "B"],
            },
            {
                "actual_player_id": "B",
                "baseline": [],
            },
            {
                "actual_player_id": None,
                "baseline": ["C"],
            },
            {
                "actual_player_id": "D",
                "baseline": ["E", "D"],
            },
        ]

        result = checks.starter_metrics(samples, "baseline")

        self.assertEqual(result["team_game_samples"], 4)
        self.assertEqual(result["truth_available"], 3)
        self.assertEqual(result["candidate_provided"], 3)
        self.assertEqual(result["evaluable_samples"], 2)
        self.assertEqual(result["top1_hits"], 1)
        self.assertEqual(result["top_k_hits"], 2)
        self.assertEqual(result["top1_accuracy"], 0.5)
        self.assertEqual(result["top_k_accuracy"], 1.0)
        self.assertEqual(
            result["top1_all_identified_truth"],
            1 / 3,
        )

    def test_empty_candidates(self) -> None:
        """후보가 없으면 유효 적중률은 정의되지 않는다."""
        result = checks.starter_metrics(
            [{
                "actual_player_id": "A",
                "baseline": [],
            }],
            "baseline",
        )

        self.assertIsNone(result["top1_accuracy"])
        self.assertEqual(result["top1_all_identified_truth"], 0.0)

    def test_common_sample_comparison(self) -> None:
        """두 규칙은 같은 유효 측의 결과를 비교해야 한다."""
        samples = [
            {
                "actual_player_id": "A",
                "truth_reason": None,
                "baseline": ["A"],
                "proposed": ["B", "A"],
            },
            {
                "actual_player_id": "B",
                "truth_reason": None,
                "baseline": ["B"],
                "proposed": [],
            },
        ]

        result = checks.starter_comparison(samples)

        self.assertEqual(result["common_samples"], 1)
        self.assertEqual(
            result["common_baseline"]["top1_accuracy"],
            1.0,
        )
        self.assertEqual(
            result["common_proposed"]["top1_accuracy"],
            0.0,
        )

    def test_future_rows_removed(self) -> None:
        """당일·미래의 선발·식별·등판 기록을 모두 제거한다."""
        past = date(2025, 8, 31)
        cutoff = date(2025, 9, 1)
        future = date(2025, 9, 2)

        data = StarterData(
            starts={
                "A": (
                    Start("g1", past, 2003, "A", "A", 1),
                    Start("g2", cutoff, 2003, "A", "A", 1),
                    Start("g3", future, 2003, "A", "A", 1),
                ),
            },
            identities={
                "A": (
                    Identity(past, 2003, "A"),
                    Identity(future, 2010, "A"),
                ),
            },
            appearances={
                "A": (
                    Appearance("g1", past, 2003, 90),
                    Appearance("g2", cutoff, 2003, 100),
                ),
            },
        )

        result = checks.before_cutoff(data, cutoff)

        self.assertEqual(len(result.starts["A"]), 1)
        self.assertEqual(len(result.identities["A"]), 1)
        self.assertEqual(len(result.appearances["A"]), 1)
        self.assertEqual(len(data.starts["A"]), 3)


class RecordAndDifficultyTest(unittest.TestCase):
    """팀 성적 및 난이도 계산의 경계 조건을 확인한다."""

    def test_team_record(self) -> None:
        """승률과 경기당 득실점은 서로 다른 분모를 사용한다."""
        result = checks.record_stats([(5, 2), (1, 4), (3, 3)])

        self.assertEqual(result["wins"], 1)
        self.assertEqual(result["losses"], 1)
        self.assertEqual(result["ties"], 1)
        self.assertEqual(result["win_rate"], 0.5)
        self.assertEqual(result["runs_for_per_game"], 3.0)
        self.assertEqual(result["runs_against_per_game"], 3.0)
        self.assertEqual(result["win_rate_denominator"], 2)
        self.assertEqual(result["run_rate_denominator"], 3)

    def test_all_ties(self) -> None:
        """모든 경기가 무승부이면 승률은 정의되지 않는다."""
        result = checks.record_stats([(3, 3), (2, 2)])

        self.assertIsNone(result["win_rate"])
        self.assertEqual(result["win_rate_denominator"], 0)
        self.assertEqual(result["runs_for_per_game"], 2.5)

    def test_empty_team_record(self) -> None:
        """관측 경기가 없는 상황을 실제 0득점과 구분한다."""
        result = checks.record_stats([])

        self.assertEqual(result["games"], 0)
        self.assertIsNone(result["runs_for_per_game"])
        self.assertEqual(result["status"], "no_observations")

    def test_difficulty_formula(self) -> None:
        """팀 관점 기대 손실을 직접 계산한다."""
        probabilities = {
            "win": 0.6,
            "tie": 0.1,
            "loss": 0.3,
        }

        row = {
            "probabilities": probabilities,
            "difficulty": 0.35,
            "status": "supported",
            "reason": None,
            "weight": 1,
        }

        self.assertAlmostEqual(
            checks.check_difficulty(row, probabilities),
            0.35,
        )

    def test_missing_forecast(self) -> None:
        """전망 누락을 0의 난이도로 취급하지 않는다."""
        row = {
            "probabilities": None,
            "difficulty": None,
            "status": "unsupported",
            "reason": "forecast_missing",
            "weight": 1,
        }

        self.assertIsNone(checks.check_difficulty(row, None))

    def test_invalid_probability_sum(self) -> None:
        """승무패 확률 합이 1이 아닌 결과를 거부한다."""
        with self.assertRaises(ValueError):
            checks.check_difficulty(
                {},
                {"win": 0.7, "tie": 0.2, "loss": 0.4},
            )

    def test_incorrect_difficulty_rejected(self) -> None:
        """정상 확률에 잘못 계산한 난이도를 붙이면 실패한다."""
        probabilities = {
            "win": 0.6,
            "tie": 0.1,
            "loss": 0.3,
        }

        row = {
            "probabilities": probabilities,
            "difficulty": 0.3,
            "status": "supported",
            "reason": None,
            "weight": 1,
        }

        with self.assertRaises(ValueError):
            checks.check_difficulty(row, probabilities)


class StabilityAndPlanTest(unittest.TestCase):
    """반복 안정성 및 사전 고정 실행 조건을 확인한다."""

    def test_stability_warning(self) -> None:
        """큰 확률 변화는 진단 경고로 별도 기록한다."""
        reference = {
            "event": {"valid": 10000, "probability": 0.2},
        }
        candidate = {
            "event": {"valid": 10000, "probability": 0.8},
        }

        result = checks.stability(reference, candidate)

        self.assertEqual(result["warning_count"], 1)
        self.assertTrue(result["diagnostic_only"])

    def test_small_denominator_not_certified(self) -> None:
        """작은 유효 분모에는 동일한 변동 경고 기준을 적용하지 않는다."""
        reference = {
            "event": {"valid": 5, "probability": 0.0},
        }
        candidate = {
            "event": {"valid": 5, "probability": 1.0},
        }

        result = checks.stability(reference, candidate)
        comparison = result["comparisons"][0]

        self.assertIsNone(comparison["warning_threshold"])

    def test_parent_job_order(self) -> None:
        """자식 전망은 대응하는 부모 실행이 완료된 뒤 수행한다."""
        jobs = build_jobs(Path("model.json"))
        names = [job["name"] for job in jobs]

        for label in ("repeat", "long", "seed"):
            parent = f"regular_{label}"
            child = f"post_{label}"

            self.assertLess(names.index(parent), names.index(child))

            job = next(item for item in jobs if item["name"] == child)

            self.assertIn(f"@parent:{parent}", job["args"])
            self.assertNotIn("--iterations", job["args"])

    def test_2026_regular_rules_rejected(self) -> None:
        """미지원 시즌에 기존 정규시즌 규칙을 적용하지 않는다."""
        with self.assertRaises(ValueError):
            regular_rules(2026)

    def test_2026_postseason_rules_rejected(self) -> None:
        """미지원 시즌에 기존 포스트시즌 규칙을 적용하지 않는다."""
        with self.assertRaises(ValueError):
            postseason_rules(2026)


if __name__ == "__main__":
    unittest.main()
