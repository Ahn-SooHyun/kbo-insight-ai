"""시즌 전망 검증 실행 경로 및 이전 증거 인수 테스트."""

from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from src.validation import season_forecast as validation


class RegularSourceTest(unittest.TestCase):
    """실자료 검산 진입점의 표준·M1 교차 검증을 검사한다."""

    @patch("src.validation.season_forecast.read_completed")
    @patch("src.validation.season_forecast.checks.regular_games")
    def test_completed_loader_is_called(
        self,
        regular_games: Mock,
        read_completed: Mock,
    ) -> None:
        """기존 완료 경기 로더가 실제 호출 경로에 연결되어야 한다."""
        standard = Mock()
        model = Mock()

        games = [
            {"game_id": str(index)}
            for index in range(validation.EXPECTED_GAMES)
        ]

        completed = tuple(
            range(validation.EXPECTED_GAMES)
        )

        regular_games.return_value = games
        read_completed.return_value = completed

        actual_games, actual_completed = validation.read_regular_inputs(
            standard,
            model,
        )

        regular_games.assert_called_once_with(standard)

        read_completed.assert_called_once_with(
            standard,
            model,
            2025,
            date(2026, 1, 1),
        )

        self.assertEqual(actual_games, games)
        self.assertEqual(actual_completed, completed)

    @patch("src.validation.season_forecast.read_completed")
    @patch("src.validation.season_forecast.checks.regular_games")
    def test_standard_and_model_count_mismatch(
        self,
        regular_games: Mock,
        read_completed: Mock,
    ) -> None:
        """표준 자료와 M1 경기 수가 다르면 실패해야 한다."""
        regular_games.return_value = [
            {"game_id": "g1"},
        ]

        read_completed.return_value = ()

        with self.assertRaises(ValueError):
            validation.read_regular_inputs(Mock(), Mock())

    @patch("src.validation.season_forecast.read_completed")
    @patch("src.validation.season_forecast.checks.regular_games")
    def test_incomplete_regular_season_rejected(
        self,
        regular_games: Mock,
        read_completed: Mock,
    ) -> None:
        """완료 시즌에 필요한 경기 수가 부족하면 실패해야 한다."""
        regular_games.return_value = [
            {"game_id": "g1"},
        ]

        read_completed.return_value = ("g1",)

        with self.assertRaises(ValueError):
            validation.read_regular_inputs(Mock(), Mock())


class ReuseInputTest(unittest.TestCase):
    """이전 실행의 입력 지문 인수 범위를 검증한다."""

    def make_snapshots(
        self,
        root: Path,
    ) -> tuple[dict[str, str], dict[str, str]]:
        """이전 계획과 수정 이후 입력 지문의 합성 표본."""
        runner = str(
            (root / validation.RUNNER_PATH).resolve()
        )

        document = str(
            (root / validation.DOC_PATH).resolve()
        )

        operational = str(
            (
                root / "src/simulation/regular_season/engine.py"
            ).resolve()
        )

        new_test = str(
            (root / validation.RESUME_TEST_PATH).resolve()
        )

        old = {
            runner: "old_runner",
            document: "old_document",
            operational: "unchanged",
        }

        new = {
            runner: "new_runner",
            document: "new_document",
            operational: "unchanged",
            new_test: "new_test",
        }

        return old, new

    @patch("src.validation.season_forecast.historical_hashes")
    def test_approved_revision_is_reusable(
        self,
        historical: Mock,
    ) -> None:
        """이전 검증 코드와 운영 코드의 계보가 확인되면 인수한다."""
        historical.side_effect = (
            lambda root, relative: (
                {"old_runner"}
                if relative == validation.RUNNER_PATH
                else {"old_document"}
            )
        )

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            old, new = self.make_snapshots(root)

            validation.verify_old_inputs(root, old, new)

        self.assertEqual(historical.call_count, 2)

    @patch("src.validation.season_forecast.historical_hashes")
    def test_operational_code_change_rejected(
        self,
        historical: Mock,
    ) -> None:
        """운영 코드의 지문이 변경되면 이전 계산을 인수하지 않는다."""
        historical.return_value = {
            "old_runner",
            "old_document",
        }

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            old, new = self.make_snapshots(root)

            operational = str(
                (
                    root / "src/simulation/regular_season/engine.py"
                ).resolve()
            )

            new[operational] = "changed"

            with self.assertRaises(ValueError):
                validation.verify_old_inputs(root, old, new)

    @patch("src.validation.season_forecast.historical_hashes")
    def test_unverified_old_revision_rejected(
        self,
        historical: Mock,
    ) -> None:
        """중간 커밋과 이전 검증 코드 지문이 다르면 거부한다."""
        historical.return_value = {
            "unexpected_hash",
        }

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            old, new = self.make_snapshots(root)

            with self.assertRaises(ValueError):
                validation.verify_old_inputs(root, old, new)

    @patch("src.validation.season_forecast.historical_hashes")
    def test_unrelated_new_file_rejected(
        self,
        historical: Mock,
    ) -> None:
        """검증 범위 밖 신규 코드가 혼입되면 인수를 거부한다."""
        historical.return_value = {
            "old_runner",
            "old_document",
        }

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            old, new = self.make_snapshots(root)

            unrelated = str(
                (root / "src/unrelated.py").resolve()
            )

            new[unrelated] = "unexpected"

            with self.assertRaises(ValueError):
                validation.verify_old_inputs(root, old, new)


class CommandContractTest(unittest.TestCase):
    """실행 명령 및 최신 테스트 결과의 계약을 검증한다."""

    def test_parent_run_argument(self) -> None:
        """인수된 실제 부모 실행 ID가 자식 명령에 전달되어야 한다."""
        job = {
            "args": [
                "-m",
                "src.simulation.postseason",
                "--parent-run",
                "@parent:regular_repeat",
            ],
        }

        previous = {
            "regular_repeat": {
                "result": {
                    "run_id": "actual_parent_run",
                },
            },
        }

        command = validation.expected_command(
            job,
            previous,
        )

        self.assertEqual(
            command[-1],
            "actual_parent_run",
        )

        self.assertNotIn(
            "@parent:regular_repeat",
            command,
        )

    def test_missing_parent_rejected(self) -> None:
        """부모 성공 기록 없이 자식 명령을 구성하지 않는다."""
        job = {
            "args": [
                "--parent-run",
                "@parent:missing",
            ],
        }

        with self.assertRaises(ValueError):
            validation.expected_command(job, {})

    def test_support_output_argument(self) -> None:
        """보조 분석 출력 경로가 실제 실행 명령에 포함되어야 한다."""
        job = {
            "args": [
                "--output",
                "@output",
            ],
        }

        output = Path("results/support/new.json")

        command = validation.expected_command(
            job,
            {},
            output,
        )

        self.assertEqual(command[-1], str(output))

    def test_successful_test_result(self) -> None:
        """종료 결과에서 실제 실행된 테스트 수를 기록한다."""
        result = validation.validate_test_result(
            "regression",
            "",
            "Ran 670 tests in 12.000s\n\nOK\n",
        )

        self.assertEqual(
            result,
            {"tests": 670},
        )

    def test_skipped_tests_not_certified(self) -> None:
        """건너뛴 테스트를 전체 정상 통과로 인정하지 않는다."""
        with self.assertRaises(ValueError):
            validation.validate_test_result(
                "regression",
                "",
                "Ran 670 tests in 12.000s\n\nOK (skipped=1)\n",
            )

    def test_regression_count_decrease_rejected(self) -> None:
        """기존에 통과한 회귀 테스트 수보다 감소하면 실패한다."""
        with self.assertRaises(ValueError):
            validation.validate_test_result(
                "regression",
                "",
                "Ran 10 tests in 0.001s\n\nOK\n",
            )

    def test_all_real_data_jobs_preserved(self) -> None:
        """기존 실자료 실행 12건과 최신 테스트 3건을 유지한다."""
        jobs = validation.build_jobs(
            Path("model.json")
        )

        real = [
            job
            for job in jobs
            if job["kind"] != "test"
        ]

        tests = [
            job
            for job in jobs
            if job["kind"] == "test"
        ]

        self.assertEqual(len(real), 12)
        self.assertEqual(len(tests), 3)

        validation_job = next(
            job
            for job in tests
            if job["name"] == "validation_tests"
        )

        self.assertIn(
            "tests.validation.test_season_forecast",
            validation_job["args"],
        )

        self.assertIn(
            "tests.validation.test_season_forecast_resume",
            validation_job["args"],
        )


class DevelopmentDiagnosticTest(unittest.TestCase):
    """실제 시즌 오차의 독립 계산을 검증한다."""

    def test_development_metric_recalculation(self) -> None:
        """승수·순위·이진 사건 오차를 각각 계산한다."""
        summary = {
            "teams": {
                "2003": {
                    "final_wins": {"expected": 80.0},
                    "rank": {"expected": 2.0},
                    "regular_season_first": {
                        "expected": 0.5,
                    },
                    "postseason_qualification": {
                        "expected": 0.75,
                    },
                },
            },
        }

        actual = {
            "2003": {
                "final_wins": 85,
                "rank": 1,
                "first": True,
                "qualifies": True,
            },
        }

        result = validation.recalculate_development_metrics(
            summary,
            actual,
        )

        self.assertEqual(
            result["wins_mae"],
            {
                "team_rows": 1,
                "value": 5.0,
            },
        )

        self.assertEqual(
            result["rank_mae"]["value"],
            1.0,
        )

        self.assertAlmostEqual(
            result["first_brier"]["value"],
            0.25,
        )

        self.assertAlmostEqual(
            result["qualification_brier"]["value"],
            0.0625,
        )

    def test_missing_event_denominator(self) -> None:
        """미결정 사건을 실제 0으로 계산하지 않는다."""
        summary = {
            "teams": {
                "2003": {
                    "final_wins": {"expected": None},
                    "rank": {"expected": None},
                    "regular_season_first": {
                        "expected": None,
                    },
                    "postseason_qualification": {
                        "expected": None,
                    },
                },
            },
        }

        actual = {
            "2003": {
                "final_wins": 85,
                "rank": 1,
                "first": True,
                "qualifies": True,
            },
        }

        result = validation.recalculate_development_metrics(
            summary,
            actual,
        )

        for metric in result.values():
            self.assertEqual(
                metric,
                {
                    "team_rows": 0,
                    "value": None,
                },
            )


if __name__ == "__main__":
    unittest.main()
