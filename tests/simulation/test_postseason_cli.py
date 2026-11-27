"""포스트시즌 자료 인수 및 CLI 통합 테스트."""

from contextlib import redirect_stderr, redirect_stdout
from datetime import date
from io import StringIO
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.datasets.published_sources import (
    sha256_file,
    write_json,
)
from src.simulation.postseason import (
    ITERATION_SCHEMA,
    SCENARIO_SCHEMA,
    SUMMARY_SCHEMA,
)
from src.simulation.postseason.__main__ import main, run
from src.simulation.postseason.data import (
    Inputs,
    read_archive,
    read_parent,
    seed_input,
)
from src.simulation.postseason.engine import (
    ArchiveState,
    SeedInput,
)
from src.simulation.regular_season import (
    ITERATION_SCHEMA as PARENT_ITERATION_SCHEMA,
    SCENARIO_SCHEMA as PARENT_SCENARIO_SCHEMA,
    SUMMARY_SCHEMA as PARENT_SUMMARY_SCHEMA,
    VERSION as PARENT_VERSION,
)
from src.simulation.regular_season.engine import (
    make_iteration,
    summarize as summarize_regular,
)
from src.simulation.regular_season.rules import (
    RULE_VERSION as RANK_RULE_VERSION,
    SeasonState,
    get_rules as get_regular_rules,
    resolve_standings,
    validate_schedule,
)


SEASON = 2025

SEEDS = (
    2003,
    2010,
    2005,
    2008,
    2004,
)

SOURCE = "a" * 64
PARENT_RUN = "b" * 32

REGULAR_RULES = get_regular_rules(SEASON)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCHEDULE_COLUMNS = """
    source_sha TEXT,
    source_row INTEGER,
    stage TEXT,
    series_game_number INTEGER,
    date TEXT,
    away_name TEXT,
    home_name TEXT,
    final_observed INTEGER,
    linked_game_id TEXT
"""

SERIES_COLUMNS = """
    source_sha TEXT,
    season INTEGER,
    stage TEXT,
    series_complete INTEGER,
    season_complete INTEGER,
    metadata_json TEXT
"""

GAME_COLUMNS = """
    game_id TEXT PRIMARY KEY,
    source_key TEXT,
    stage TEXT,
    date TEXT,
    game_number INTEGER,
    home_team_id INTEGER,
    away_team_id INTEGER,
    home_score INTEGER,
    away_score INTEGER,
    winner TEXT,
    source_sha TEXT,
    source_row INTEGER,
    season INTEGER,
    competition TEXT
"""

NAMES = {
    2003: "LG Twins",
    2010: "Hanwha Eagles",
    2005: "SSG Landers",
    2008: "Samsung Lions",
    2004: "NC Dinos",
}


def read_json(path: Path) -> dict:
    """테스트 산출물의 JSON 객체를 읽는다."""
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


class PostseasonArchiveTests(unittest.TestCase):
    """시점 경계와 원천 판본 검증을 검사한다."""

    def setUp(self) -> None:
        self.standard = sqlite3.connect(":memory:")
        self.model = sqlite3.connect(":memory:")

        self.addCleanup(self.standard.close)
        self.addCleanup(self.model.close)

        self.standard.execute(
            f"CREATE TABLE eligible_games ({GAME_COLUMNS})"
        )

        self.standard.execute(
            f"CREATE TABLE postseason_schedule ({SCHEDULE_COLUMNS})"
        )

        self.standard.execute(
            f"CREATE TABLE postseason_series ({SERIES_COLUMNS})"
        )

        self.model.execute(
            f"CREATE TABLE season_schedule_evidence ({SCHEDULE_COLUMNS})"
        )

        self.model.execute(
            f"CREATE TABLE season_series_evidence ({SERIES_COLUMNS})"
        )

    def add_game(
        self,
        stage: str,
        number: int,
        day: str,
        home: int,
        away: int,
        home_score: int,
        away_score: int,
    ) -> str:
        """양쪽 자료판에 연결된 합성 완료 경기와 일정을 추가한다."""
        source_key = f"{SEASON}:{stage}:{number}"
        game_id = f"postseason:dent:{source_key}"

        source_row = self.standard.execute(
            "SELECT COUNT(*) FROM postseason_schedule"
        ).fetchone()[0]

        winner = (
            "home"
            if home_score > away_score
            else "away"
            if away_score > home_score
            else "tie"
        )

        self.standard.execute(
            """
            INSERT INTO eligible_games
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                game_id,
                source_key,
                stage,
                day,
                number,
                home,
                away,
                home_score,
                away_score,
                winner,
                SOURCE,
                source_row,
                SEASON,
                "postseason",
            ),
        )

        schedule = (
            SOURCE,
            source_row,
            stage,
            number,
            day,
            NAMES[away],
            NAMES[home],
            0,
            game_id,
        )

        self.standard.execute(
            """
            INSERT INTO postseason_schedule
            VALUES (?,?,?,?,?,?,?,?,?)
            """,
            schedule,
        )

        self.model.execute(
            """
            INSERT INTO season_schedule_evidence
            VALUES (?,?,?,?,?,?,?,?,?)
            """,
            schedule,
        )

        return game_id

    def add_wildcard_completion(
        self,
        higher_wins: int,
        lower_wins: int,
        score_rows: int,
    ) -> None:
        """원천 완료 증거의 합성 레코드를 추가한다."""
        metadata = {
            "isComplete": True,
            "_score_rows": score_rows,
            "winnerId": 2008,
            "seeds": {
                str(rank): {"id": team}
                for rank, team in enumerate(
                    SEEDS,
                    start=1,
                )
            },
            "team4": {
                "id": 2008,
                "wins": higher_wins,
                "isWinner": True,
            },
            "team5": {
                "id": 2004,
                "wins": lower_wins,
                "isWinner": False,
            },
        }

        row = (
            SOURCE,
            SEASON,
            "wildCard",
            1,
            None,
            json.dumps(metadata),
        )

        self.standard.execute(
            """
            INSERT INTO postseason_series
            VALUES (?,?,?,?,?,?)
            """,
            row,
        )

        self.model.execute(
            """
            INSERT INTO season_series_evidence
            VALUES (?,?,?,?,?,?)
            """,
            row,
        )

    def test_completed_series_without_final_observed_flag(self) -> None:
        game_id = self.add_game(
            "wildCard",
            1,
            "2025-10-06",
            2008,
            2004,
            1,
            0,
        )

        self.add_wildcard_completion(1, 0, 1)

        archive, metadata = read_archive(
            self.standard,
            self.model,
            SEASON,
            date(2025, 10, 7),
            SEEDS,
        )

        self.assertEqual(
            archive.games["wildCard"][0].game_id,
            game_id,
        )

        self.assertEqual(
            archive.completion_evidence["wildCard"]["winner"],
            2008,
        )

        self.assertEqual(archive.blocked, {})
        self.assertFalse(metadata["official_record_verified"])

    def test_cutoff_excludes_same_day_and_future(self) -> None:
        self.add_game(
            "wildCard",
            1,
            "2025-10-06",
            2008,
            2004,
            0,
            1,
        )

        self.add_game(
            "wildCard",
            2,
            "2025-10-07",
            2008,
            2004,
            1,
            0,
        )

        self.add_wildcard_completion(1, 1, 2)

        archive, metadata = read_archive(
            self.standard,
            self.model,
            SEASON,
            date(2025, 10, 7),
            SEEDS,
        )

        self.assertEqual(
            len(archive.games["wildCard"]),
            1,
        )

        self.assertEqual(
            archive.games["wildCard"][0].day,
            "2025-10-06",
        )

        self.assertEqual(
            archive.completion_evidence,
            {},
        )

        self.assertEqual(archive.blocked, {})

        self.assertEqual(
            metadata["eligible_postseason_games"],
            1,
        )

    def test_future_completion_metadata_not_used(self) -> None:
        self.add_wildcard_completion(1, 1, 2)

        archive, _ = read_archive(
            self.standard,
            self.model,
            SEASON,
            date(2025, 10, 6),
            SEEDS,
        )

        self.assertEqual(archive.games, {})
        self.assertEqual(archive.completion_evidence, {})
        self.assertEqual(archive.blocked, {})

    def test_schedule_evidence_mismatch(self) -> None:
        self.add_game(
            "wildCard",
            1,
            "2025-10-06",
            2008,
            2004,
            1,
            0,
        )

        self.model.execute(
            "DELETE FROM season_schedule_evidence"
        )

        with self.assertRaisesRegex(
            ValueError,
            "일정 근거 불일치",
        ):
            read_archive(
                self.standard,
                self.model,
                SEASON,
                date(2025, 10, 7),
                SEEDS,
            )

    def test_schedule_only_is_not_result(self) -> None:
        game_id = self.add_game(
            "wildCard",
            1,
            "2025-10-06",
            2008,
            2004,
            1,
            0,
        )

        self.standard.execute(
            "DELETE FROM eligible_games WHERE game_id = ?",
            (game_id,),
        )

        archive, _ = read_archive(
            self.standard,
            self.model,
            SEASON,
            date(2025, 10, 7),
            SEEDS,
        )

        self.assertIn("wildCard", archive.blocked)
        self.assertEqual(archive.games, {})

    def test_wrong_home_is_blocked(self) -> None:
        self.add_game(
            "wildCard",
            1,
            "2025-10-06",
            2004,
            2008,
            1,
            0,
        )

        archive, _ = read_archive(
            self.standard,
            self.model,
            SEASON,
            date(2025, 10, 7),
            SEEDS,
        )

        self.assertIn(
            "홈 배정",
            archive.blocked["wildCard"],
        )

    def test_missing_completion_evidence(self) -> None:
        self.add_game(
            "wildCard",
            1,
            "2025-10-06",
            2008,
            2004,
            1,
            0,
        )

        archive, _ = read_archive(
            self.standard,
            self.model,
            SEASON,
            date(2025, 10, 7),
            SEEDS,
        )

        self.assertIn(
            "완료 근거",
            archive.blocked["wildCard"],
        )

    def test_later_stage_cannot_replace_unresolved_stage(self) -> None:
        self.add_game(
            "semiPlayoff",
            1,
            "2025-10-09",
            2005,
            2008,
            1,
            0,
        )

        archive, _ = read_archive(
            self.standard,
            self.model,
            SEASON,
            date(2025, 10, 10),
            SEEDS,
        )

        self.assertIn(
            "후속 라운드",
            archive.blocked["wildCard"],
        )


class PostseasonParentTests(unittest.TestCase):
    """정규시즌 선행 실행의 해시 및 반복 인수를 검사한다."""

    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)

        self.root = Path(self.temporary.name)

        self.settings = SimpleNamespace(
            project_root=PROJECT_ROOT,
            results_dir=self.root,
        )

        self.parent = (
            self.root
            / "regular_season"
            / PARENT_RUN
        )

        self.parent.mkdir(parents=True)

        self.cutoff = date(2025, 9, 1)

        self.provenance = {
            "model_data_sha256": "1" * 64,
            "standard_run_id": "2" * 32,
            "standard_verification_id": "3" * 32,
            "standard_database_sha256": "4" * 64,
            "model_run_id": "5" * 32,
            "model_schema": "m1-final-score-v1",
            "model_asset_sha256": "6" * 64,
            "model_data": {
                "model_data_run_id": "7" * 32,
                "contract_sha256": "8" * 64,
                "manifest_sha256": "9" * 64,
            },
        }

        state = SeasonState(REGULAR_RULES)

        self.row = make_iteration(
            1,
            state,
            resolve_standings(state),
        )

        missing = validate_schedule(
            REGULAR_RULES,
            [],
        )

        metadata = {
            "season": SEASON,
            "as_of_kst": "2025-09-01T00:00:00+09:00",
            "final_test_used": False,
            "rules": {
                "version": RANK_RULE_VERSION,
            },
            "provenance": self.provenance,
            "completed_games": 0,
            "remaining_games": 0,
            "missing_games": REGULAR_RULES.total_games,
            "missing_pairs": missing,
        }

        scenario = {
            "schema": PARENT_SCENARIO_SCHEMA,
            "metadata": metadata,
            "completed": [],
            "remaining": [],
        }

        write_json(
            self.parent / "scenario.json",
            scenario,
        )

        write_json(
            self.parent / "summary.json",
            summarize_regular(
                REGULAR_RULES,
                [self.row],
            ),
        )

        (self.parent / "iterations.jsonl").write_text(
            json.dumps(self.row) + "\n",
            encoding="utf-8",
        )

        source_root = (
            PROJECT_ROOT
            / "src"
            / "simulation"
            / "regular_season"
        )

        files = {
            key: {
                "sha256": sha256_file(
                    self.parent / filename
                ),
            }
            for key, filename in (
                ("scenario", "scenario.json"),
                ("summary", "summary.json"),
                ("iterations", "iterations.jsonl"),
            )
        }

        manifest = {
            "status": "completed",
            "run_id": PARENT_RUN,
            "version": PARENT_VERSION,
            "season": SEASON,
            "as_of_kst": metadata["as_of_kst"],
            "seed": 42,
            "iterations": 1,
            "final_test_used": False,
            "schemas": {
                "scenario": PARENT_SCENARIO_SCHEMA,
                "summary": PARENT_SUMMARY_SCHEMA,
                "iterations": PARENT_ITERATION_SCHEMA,
            },
            "source_sha256": {
                name: sha256_file(
                    source_root / name
                )
                for name in (
                    "__init__.py",
                    "__main__.py",
                    "data.py",
                    "engine.py",
                    "rules.py",
                )
            },
            "metadata": metadata,
            "files": files,
        }

        write_json(
            self.parent / "manifest.json",
            manifest,
        )

    def read(self):
        """합성 선행 실행을 실제 인수 함수로 읽는다."""
        return read_parent(
            self.settings,
            PARENT_RUN,
            SEASON,
            self.cutoff,
            (),
            self.provenance,
        )

    def test_parent_preserves_unresolved_seed(self) -> None:
        draws, metadata = self.read()

        self.assertEqual(len(draws), 1)
        self.assertEqual(draws[0].iteration_id, 1)
        self.assertIsNone(draws[0].seeds)
        self.assertEqual(metadata["seed_resolved"], 0)

    def test_parent_hash_mismatch(self) -> None:
        with (
            self.parent / "iterations.jsonl"
        ).open("a", encoding="utf-8") as stream:
            stream.write("\n")

        with self.assertRaisesRegex(
            ValueError,
            "파일 해시 불일치",
        ):
            self.read()

    def test_parent_time_mismatch(self) -> None:
        self.cutoff = date(2025, 9, 2)

        with self.assertRaisesRegex(
            ValueError,
            "기준 시각 불일치",
        ):
            self.read()

    def test_invalid_seed_resolution(self) -> None:
        self.row["postseason_seeds"] = list(SEEDS)

        with self.assertRaisesRegex(
            ValueError,
            "시드와 확정 순위 불일치",
        ):
            seed_input(
                self.row,
                REGULAR_RULES.teams,
            )


class PostseasonCliTests(unittest.TestCase):
    """CLI의 결과 파일, 종료 코드 및 재현성을 검사한다."""

    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)

        self.root = Path(self.temporary.name)

        self.settings = SimpleNamespace(
            project_root=PROJECT_ROOT,
            results_dir=self.root,
            seed=42,
            log_level="CRITICAL",
        )

        self.inputs = Inputs(
            season=SEASON,
            cutoff=date(2025, 9, 1),
            model={},
            state=SeasonState(REGULAR_RULES),
            draws=tuple(
                SeedInput(
                    iteration_id=index,
                    seeds=SEEDS,
                    qualifiers=tuple(sorted(SEEDS)),
                    qualification={
                        team: team in SEEDS
                        for team in REGULAR_RULES.teams
                    },
                    regular_first=SEEDS[0],
                )
                for index in range(1, 11)
            ),
            archive=ArchiveState(),
            metadata={
                "as_of_kst": "2025-09-01T00:00:00+09:00",
                "mode": "synthetic_test",
                "formal_backtest_supported": False,
                "final_test_used": False,
            },
        )

        self.args = [
            "simulate",
            "--run-id", "c" * 32,
            "--asset", str(self.root / "model.json"),
            "--season", str(SEASON),
            "--as-of", "2025-09-01",
        ]

    def run_simulation(self) -> dict:
        """실제 계산과 파일 저장을 실행하되 모델 자료만 대체한다."""
        with (
            patch(
                "src.simulation.postseason.__main__.load_inputs",
                return_value=self.inputs,
            ),
            patch(
                "src.simulation.postseason.engine.m1.predict",
                return_value={
                    "result_probabilities": {
                        "home": 0.55,
                        "tie": 0.02,
                        "away": 0.43,
                    },
                },
            ),
        ):
            return run(
                self.settings,
                "c" * 32,
                self.root / "model.json",
                SEASON,
                "2025-09-01",
                PARENT_RUN,
                None,
                42,
                False,
            )

    def test_cli_output_contract(self) -> None:
        result = self.run_simulation()

        manifest = read_json(
            Path(result["manifest"])
        )

        scenario = read_json(
            Path(result["scenario"])
        )

        summary = read_json(
            Path(result["summary"])
        )

        self.assertEqual(
            manifest["status"],
            "completed",
        )

        self.assertEqual(
            scenario["schema"],
            SCENARIO_SCHEMA,
        )

        self.assertEqual(
            summary["schema"],
            SUMMARY_SCHEMA,
        )

        self.assertEqual(
            result["iterations"],
            10,
        )

        for file in manifest["files"].values():
            self.assertEqual(
                sha256_file(Path(file["path"])),
                file["sha256"],
            )

        with Path(
            result["iterations_path"]
        ).open(encoding="utf-8") as stream:
            rows = [
                json.loads(line)
                for line in stream
            ]

        self.assertEqual(len(rows), 10)

        for index, row in enumerate(rows, start=1):
            self.assertEqual(
                row["schema"],
                ITERATION_SCHEMA,
            )

            self.assertEqual(
                row["iteration_id"],
                index,
            )

            self.assertEqual(
                row["postseason_seeds"],
                list(SEEDS),
            )

    def test_reproducible_core_output_files(self) -> None:
        first = self.run_simulation()
        second = self.run_simulation()

        self.assertNotEqual(
            first["run_id"],
            second["run_id"],
        )

        for name in (
            "scenario",
            "summary",
            "iterations_path",
        ):
            self.assertEqual(
                sha256_file(Path(first[name])),
                sha256_file(Path(second[name])),
            )

    def test_main_success(self) -> None:
        result = {
            "status": "completed",
            "iterations": 10,
        }

        output = StringIO()

        with (
            patch(
                "src.simulation.postseason.__main__.load_settings",
                return_value=self.settings,
            ),
            patch(
                "src.simulation.postseason.__main__.run",
                return_value=result,
            ) as mocked,
            redirect_stdout(output),
        ):
            exit_code = main([
                *self.args,
                "--parent-run",
                PARENT_RUN,
            ])

        self.assertEqual(exit_code, 0)

        self.assertEqual(
            json.loads(output.getvalue()),
            result,
        )

        self.assertIsNone(
            mocked.call_args.kwargs["iterations"],
        )

    def test_archive_requires_explicit_assumption(self) -> None:
        with (
            patch(
                "src.simulation.postseason.__main__.load_settings",
                return_value=self.settings,
            ),
            redirect_stderr(StringIO()),
        ):
            exit_code = main(self.args)

        self.assertEqual(exit_code, 2)

    def test_parent_iterations_cannot_be_overridden(self) -> None:
        with (
            patch(
                "src.simulation.postseason.__main__.load_settings",
                return_value=self.settings,
            ),
            redirect_stderr(StringIO()),
        ):
            exit_code = main([
                *self.args,
                "--parent-run",
                PARENT_RUN,
                "--iterations",
                "50",
            ])

        self.assertEqual(exit_code, 2)

    def test_parent_and_archive_modes_cannot_be_combined(self) -> None:
        with (
            patch(
                "src.simulation.postseason.__main__.load_settings",
                return_value=self.settings,
            ),
            redirect_stderr(StringIO()),
        ):
            exit_code = main([
                *self.args,
                "--parent-run",
                PARENT_RUN,
                "--archive-assumption",
            ])

        self.assertEqual(exit_code, 2)


if __name__ == "__main__":
    unittest.main()