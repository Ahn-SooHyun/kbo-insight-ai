
"""정규시즌 CLI 반환 계약, 산출물 저장 및 실패 처리 통합 검증."""

from contextlib import redirect_stdout
from dataclasses import replace
from datetime import date
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
import unittest
from unittest.mock import patch

from src.datasets.published_sources import as_row, sha256_file
from src.models.game_outcome import model as m1
from src.settings import load_settings
from src.simulation.regular_season import (
    ITERATION_SCHEMA,
    SCENARIO_SCHEMA,
    SUMMARY_SCHEMA,
)
from src.simulation.regular_season import __main__ as cli
from src.simulation.regular_season.data import Scenario
from src.simulation.regular_season.engine import Iteration, summarize
from src.simulation.regular_season.rules import (
    Fixture,
    PlayedGame,
    SeasonRules,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CUTOFF = date(2025, 9, 1)
ITERATIONS = 8
SEED = 42
TEST_RUN_ID = "0" * 32


def read_object(path: Path) -> dict[str, object]:
    """저장된 JSON의 객체 타입을 확인하여 반환한다."""

    return as_row(
        json.loads(path.read_text(encoding="utf-8"))
    )


def read_iterations(path: Path) -> list[Iteration]:
    """반복별 JSON Lines 파일을 읽는다."""

    with path.open(encoding="utf-8") as stream:
        return [
            cast(Iteration, as_row(json.loads(line)))
            for line in stream
            if line.strip()
        ]


class TestRegularSeasonCLI(unittest.TestCase):
    """실제 CLI와 저장 로직을 연결하여 반환 계약을 검증한다."""

    def setUp(self) -> None:
        """기존 자료를 변경하지 않는 임시 실행 환경을 만든다."""

        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)

        self.root = Path(directory.name)

        settings = load_settings(PROJECT_ROOT)

        # 기존 설정 구조를 유지하되 테스트 산출물은
        # 실제 프로젝트의 결과 폴더와 분리한다.
        self.settings = replace(
            settings,
            results_dir=self.root / "results",
        )

        self.asset = self.root / "model.json"

        self.rules = SeasonRules(
            season=2025,
            teams=(2002, 2003),
            games_per_pair=1,
            playoff_slots=1,
        )

        self.fixture = Fixture(
            game_id="20250901KTLG0",
            day=CUTOFF,
            home=2003,
            away=2002,
            number=0,
        )

        # 실제 운영 자산에 의존하지 않는 테스트 전용 분포다.
        # 모델 추론과 결합 득점 표본 추출은 실제 구현을 사용한다.
        model: m1.Model = {
            "cap": 39,
            "league_mean": 4.0,
            "site_counts": {
                "home": [1] * 40,
                "away": [1] * 40,
            },
            "profile_counts": {},
            "tie_weight": 0.5,
        }

        self.scenario = Scenario(
            rules=self.rules,
            cutoff=CUTOFF,
            completed=(),
            remaining=(self.fixture,),
            model=model,
            standard_db=self.root / "standard.sqlite3",
            model_db=self.root / "model_data.sqlite3",
            metadata={
                "season": 2025,
                "as_of_kst": "2025-09-01T00:00:00+09:00",
                "missing_games": 0,
                "final_test_used": False,
                "formal_backtest_supported": False,
            },
        )

    def arguments(
        self,
        *,
        iterations: int = ITERATIONS,
        diagnostic: bool = False,
    ) -> list[str]:
        """실제 CLI 파서에 전달할 테스트 인자를 구성한다."""

        arguments = [
            "--project-root",
            str(PROJECT_ROOT),
            "simulate",
            "--run-id",
            TEST_RUN_ID,
            "--asset",
            str(self.asset),
            "--season",
            "2025",
            "--as-of",
            CUTOFF.isoformat(),
            "--iterations",
            str(iterations),
            "--seed",
            str(SEED),
        ]

        if diagnostic:
            arguments.append("--diagnostic")

        return arguments

    def verify_outputs(
        self,
        result: dict[str, object],
        *,
        diagnostic: bool,
    ) -> None:
        """CLI 반환값과 저장 산출물의 계약을 대조한다."""

        self.assertEqual(result["status"], "completed")

        # 반복 횟수 필드가 파일 경로로 덮어써지는
        # 기존 문제가 재발하면 반드시 실패해야 한다.
        self.assertIs(type(result["iterations"]), int)
        self.assertEqual(result["iterations"], ITERATIONS)
        self.assertIsInstance(result["iterations_path"], str)

        self.assertEqual(result["seed"], SEED)
        self.assertEqual(result["season"], 2025)

        self.assertEqual(
            result["as_of_kst"],
            self.scenario.metadata["as_of_kst"],
        )

        self.assertEqual(result["completed_games"], 0)
        self.assertEqual(result["remaining_games"], 1)
        self.assertEqual(result["missing_games"], 0)

        result_dir = Path(str(result["result_dir"]))

        self.assertTrue(result_dir.is_dir())

        self.assertEqual(
            result_dir.parent,
            self.settings.results_dir / cli.RESULT_DIRECTORY,
        )

        manifest_path = Path(str(result["manifest"]))
        summary_path = Path(str(result["summary"]))
        scenario_path = Path(str(result["scenario"]))
        iterations_path = Path(str(result["iterations_path"]))

        self.assertEqual(
            manifest_path,
            result_dir / "manifest.json",
        )

        self.assertEqual(
            summary_path,
            result_dir / "summary.json",
        )

        self.assertEqual(
            scenario_path,
            result_dir / "scenario.json",
        )

        self.assertEqual(
            iterations_path,
            result_dir / "iterations.jsonl",
        )

        manifest = read_object(manifest_path)
        summary = read_object(summary_path)
        scenario = read_object(scenario_path)
        rows = read_iterations(iterations_path)

        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["run_id"], result["run_id"])

        self.assertIs(type(manifest["iterations"]), int)

        self.assertEqual(
            manifest["iterations"],
            result["iterations"],
        )

        self.assertEqual(manifest["seed"], result["seed"])

        self.assertIs(manifest["final_test_used"], False)

        self.assertIs(
            manifest["formal_backtest_supported"],
            False,
        )

        self.assertIs(
            manifest["development_diagnostic"],
            diagnostic,
        )

        self.assertEqual(summary["schema"], SUMMARY_SCHEMA)
        self.assertEqual(scenario["schema"], SCENARIO_SCHEMA)

        self.assertEqual(
            manifest["schemas"],
            {
                "scenario": SCENARIO_SCHEMA,
                "summary": SUMMARY_SCHEMA,
                "iterations": ITERATION_SCHEMA,
            },
        )

        self.assertEqual(len(rows), ITERATIONS)

        self.assertEqual(
            summary["requested_iterations"],
            len(rows),
        )

        self.assertEqual(
            [row["iteration_id"] for row in rows],
            list(range(1, ITERATIONS + 1)),
        )

        self.assertTrue(
            all(
                row["schema"] == ITERATION_SCHEMA
                for row in rows
            )
        )

        self.assertEqual(
            summary,
            summarize(self.rules, rows),
        )

        # CLI 반환 키의 변경으로 매니페스트의 기존 파일
        # 계약이나 후속 포스트시즌 인계가 깨지면 안 된다.
        files = as_row(manifest["files"])

        expected_files = {
            "scenario",
            "summary",
            "iterations",
        }

        if diagnostic:
            expected_files.update({
                "diagnostic",
                "baseline_iterations",
            })

        self.assertEqual(set(files), expected_files)

        for name, raw in files.items():
            item = as_row(raw)
            path = Path(str(item["path"]))

            self.assertTrue(path.is_file(), name)
            self.assertEqual(path.parent, result_dir)

            self.assertEqual(
                item["sha256"],
                sha256_file(path),
                name,
            )

            cli_key = (
                "iterations_path"
                if name == "iterations"
                else name
            )

            self.assertEqual(
                result[cli_key],
                item["path"],
                name,
            )

        if diagnostic:
            diagnostic_path = Path(
                str(result["diagnostic"])
            )

            baseline_path = Path(
                str(result["baseline_iterations"])
            )

            metrics = read_object(diagnostic_path)
            baseline_rows = read_iterations(baseline_path)

            self.assertEqual(
                len(baseline_rows),
                ITERATIONS,
            )

            self.assertEqual(
                metrics["baseline_summary"],
                summarize(self.rules, baseline_rows),
            )

            self.assertEqual(
                metrics["iterations"],
                ITERATIONS,
            )

            self.assertIs(metrics["formal_backtest"], False)
            self.assertIs(metrics["final_test_used"], False)

        else:
            self.assertNotIn("diagnostic", result)
            self.assertNotIn("baseline_iterations", result)

            self.assertFalse(
                (result_dir / "diagnostic.json").exists()
            )

            self.assertFalse(
                (
                    result_dir / "baseline_iterations.jsonl"
                ).exists()
            )

    def test_cli_returns_iteration_count_and_path(self) -> None:
        """일반 실행에서 반복 횟수와 경로가 분리되는지 검증한다."""

        stdout = StringIO()

        # 외부 데이터 연결만 대체한다.
        # CLI 파싱, 모델 추론, 반복 계산, 집계 및 저장은
        # 실제 구현을 사용한다.
        with (
            patch.object(
                cli,
                "load_settings",
                return_value=self.settings,
            ),
            patch.object(
                cli,
                "load_scenario",
                return_value=self.scenario,
            ) as scenario_loader,
            patch.object(
                cli,
                "read_diagnostic_truth",
                side_effect=AssertionError(
                    "일반 실행에서는 미래 정답을 읽으면 안 됩니다."
                ),
            ) as truth_loader,
            redirect_stdout(stdout),
        ):
            exit_code = cli.main(self.arguments())

        self.assertEqual(exit_code, 0)

        scenario_loader.assert_called_once_with(
            self.settings,
            TEST_RUN_ID,
            self.asset,
            2025,
            CUTOFF,
            None,
        )

        truth_loader.assert_not_called()

        result = as_row(json.loads(stdout.getvalue()))

        self.verify_outputs(
            result,
            diagnostic=False,
        )

    def test_cli_diagnostic_preserves_output_contract(self) -> None:
        """개발 진단을 추가해도 기존 출력 계약이 유지되는지 검증한다."""

        stdout = StringIO()

        actual = (
            PlayedGame(
                self.fixture,
                home_runs=3,
                away_runs=1,
            ),
        )

        with (
            patch.object(
                cli,
                "load_settings",
                return_value=self.settings,
            ),
            patch.object(
                cli,
                "load_scenario",
                return_value=self.scenario,
            ),
            patch.object(
                cli,
                "read_diagnostic_truth",
                return_value=actual,
            ) as truth_loader,
            redirect_stdout(stdout),
        ):
            exit_code = cli.main(
                self.arguments(diagnostic=True)
            )

        self.assertEqual(exit_code, 0)

        truth_loader.assert_called_once_with(
            self.scenario
        )

        result = as_row(json.loads(stdout.getvalue()))

        self.verify_outputs(
            result,
            diagnostic=True,
        )

    def test_cli_rejects_invalid_iteration_count(self) -> None:
        """잘못된 반복 횟수는 자료 로딩과 저장 전에 거부한다."""

        for count in (0, -1):
            with self.subTest(iterations=count):
                stdout = StringIO()

                with (
                    patch.object(
                        cli,
                        "load_settings",
                        return_value=self.settings,
                    ),
                    patch.object(
                        cli,
                        "load_scenario",
                    ) as scenario_loader,
                    self.assertLogs(
                        cli.LOGGER,
                        level="ERROR",
                    ),
                    redirect_stdout(stdout),
                ):
                    exit_code = cli.main(
                        self.arguments(iterations=count)
                    )

                self.assertEqual(exit_code, 2)

                scenario_loader.assert_not_called()

                self.assertEqual(stdout.getvalue(), "")

                self.assertFalse(
                    self.settings.results_dir.exists()
                )

    def test_cli_input_failure_creates_no_completion(self) -> None:
        """자료 로딩 실패 시 완료 결과가 생성되지 않는지 검증한다."""

        stdout = StringIO()

        with (
            patch.object(
                cli,
                "load_settings",
                return_value=self.settings,
            ),
            patch.object(
                cli,
                "load_scenario",
                side_effect=FileNotFoundError(
                    "테스트용 입력 자산 누락"
                ),
            ),
            self.assertLogs(
                cli.LOGGER,
                level="ERROR",
            ) as logs,
            redirect_stdout(stdout),
        ):
            exit_code = cli.main(self.arguments())

        self.assertEqual(exit_code, 2)
        self.assertEqual(stdout.getvalue(), "")

        self.assertTrue(
            any(
                "정규시즌 전망 실행 실패" in message
                for message in logs.output
            )
        )

        self.assertFalse(
            self.settings.results_dir.exists()
        )


if __name__ == "__main__":
    unittest.main()
