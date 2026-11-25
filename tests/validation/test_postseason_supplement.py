"""보완 원본 선택, 기존 CLI 통합 및 선발 후보 추가 범위를 검증한다."""

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.__main__ import main
from src.datasets.postseason_supplement import (
    collection_run_for_supplement,
    standardize_selected,
    unchanged_tables,
    verify_starter_extension,
)
from src.datasets.published_sources import (
    EXPECTED_TARGETS,
    SUPPLEMENT_TARGETS,
    SourceError,
    intake_sources,
)


BASE_RUN = "1" * 32
SUPPLEMENT_RUN = "2" * 32
UNKNOWN_RUN = "3" * 32

OLD_GAME = "20240927200120080"
NEW_GAME = "postseason:dent:2024:koreanSeries:1"


def create_collection_run(
    published: Path,
    run_id: str,
    created_at: str,
    target_ids: frozenset[str],
    body: bytes,
) -> Path:
    """하나의 물리 파일을 여러 논리 대상으로 참조하는 합성 이력을 생성한다."""
    objects = published / "objects"
    objects.mkdir(parents=True, exist_ok=True)

    digest = hashlib.sha256(body).hexdigest()
    object_path = objects / f"{digest}.raw"

    if not object_path.exists():
        object_path.write_bytes(body)

    directory = published / "runs" / run_id
    directory.mkdir(parents=True, exist_ok=False)

    targets = []

    for target_id in sorted(target_ids):
        target = {
            "target_id": target_id,
            "source": "github",
            "repository": "example/dataset",
            "filename": "games.json",
            "kind": "results",
            "season": 2024,
            "competition": "regular",
            "revision": "a" * 40,
        }

        targets.append(target)

        result = {
            "run_id": run_id,
            "target_id": target_id,
            "identity": [
                target["source"],
                target["repository"],
                target["filename"],
                target["kind"],
                target["season"],
                target["competition"],
            ],
            "status": "included",
            "body_complete": True,
            "revision": target["revision"],
            "sha256": digest,
            "bytes": len(body),
            "object_path": f"objects/{digest}.raw",
        }

        (directory / f"{target_id}.result.json").write_text(
            json.dumps(result),
            encoding="utf-8",
        )

    (directory / "manifest.json").write_text(
        json.dumps({
            "run_id": run_id,
            "created_at": created_at,
            "targets": targets,
        }),
        encoding="utf-8",
    )

    return object_path


class SourceScopeTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)

        self.root = Path(temporary.name)
        self.raw = self.root / "raw"
        self.published = self.raw / "published"

        self.base_object = create_collection_run(
            self.published,
            BASE_RUN,
            "2026-11-25T00:00:00+00:00",
            EXPECTED_TARGETS,
            b'{"scope":"base"}',
        )

        self.supplement_object = create_collection_run(
            self.published,
            SUPPLEMENT_RUN,
            "2026-11-26T00:00:00+00:00",
            SUPPLEMENT_TARGETS,
            b'{"scope":"supplement"}',
        )

    def test_legacy_scope_keeps_base_targets_only(self) -> None:
        """과거 표준화 범위에서 승인된 새 수집 이력을 구분한다."""
        intake = intake_sources(
            self.published,
            as_of_run=BASE_RUN,
        )

        self.assertEqual(set(intake.selected), EXPECTED_TARGETS)
        self.assertEqual(len(intake.versions), len(EXPECTED_TARGETS))
        self.assertEqual(intake.object_count, 1)

        self.assertEqual(
            intake.raw_bytes,
            self.base_object.stat().st_size,
        )

        self.assertFalse(
            any(
                item["target"]["target_id"] in SUPPLEMENT_TARGETS
                for item in intake.history
            )
        )

    def test_supplement_scope_includes_both_target_groups(self) -> None:
        """보완 경로는 기존 및 추가 논리 대상을 모두 선택한다."""
        intake = intake_sources(
            self.published,
            as_of_run=SUPPLEMENT_RUN,
            expected_targets=(
                EXPECTED_TARGETS | SUPPLEMENT_TARGETS
            ),
        )

        self.assertEqual(
            set(intake.selected),
            EXPECTED_TARGETS | SUPPLEMENT_TARGETS,
        )

        self.assertEqual(intake.object_count, 2)

        self.assertEqual(
            intake.raw_bytes,
            self.base_object.stat().st_size
            + self.supplement_object.stat().st_size,
        )

    def test_outside_scope_corruption_is_not_ignored(self) -> None:
        """기존 논리 선정 범위 밖의 승인 원본도 해시를 검사한다."""
        original = self.supplement_object.read_bytes()

        self.supplement_object.write_bytes(
            b"x" * len(original)
        )

        with self.assertRaises(SourceError):
            intake_sources(
                self.published,
                as_of_run=BASE_RUN,
            )

    def test_unknown_target_is_still_rejected(self) -> None:
        """등록되지 않은 대상의 기존 차단 기준을 보존한다."""
        create_collection_run(
            self.published,
            UNKNOWN_RUN,
            "2026-11-27T00:00:00+00:00",
            frozenset({"unknown-target"}),
            b"unknown",
        )

        with self.assertRaises(SourceError):
            intake_sources(self.published)

    def test_collection_cutoff_selects_correct_workflow(self) -> None:
        settings = SimpleNamespace(raw_dir=self.raw)

        self.assertIsNone(
            collection_run_for_supplement(
                settings,
                BASE_RUN,
            )
        )

        self.assertEqual(
            collection_run_for_supplement(
                settings,
                SUPPLEMENT_RUN,
            ),
            SUPPLEMENT_RUN,
        )

        self.assertEqual(
            collection_run_for_supplement(
                settings,
                None,
            ),
            SUPPLEMENT_RUN,
        )

    def test_previous_cutoff_routes_to_original_standardization(self) -> None:
        """추가 수집 이전 실행은 기존 표준화에 전달한다."""
        settings = SimpleNamespace(raw_dir=self.raw)

        with patch(
            "src.datasets.postseason_supplement.standardize",
            return_value={"status": "completed"},
        ) as execute:
            result = standardize_selected(
                settings,
                as_of_run=BASE_RUN,
            )

        self.assertEqual(result["status"], "completed")

        execute.assert_called_once_with(
            settings,
            as_of_run=BASE_RUN,
            compare_run=None,
        )

    def test_supplement_cutoff_routes_to_supplement_standardization(self) -> None:
        """보완 수집 실행은 기존 표준판을 기준으로 증분 생성한다."""
        settings = SimpleNamespace(raw_dir=self.raw)
        verification = "4" * 32

        with patch(
            "src.datasets.postseason_supplement.build_standard_supplement",
            return_value={"status": "completed"},
        ) as execute:
            result = standardize_selected(
                settings,
                as_of_run=SUPPLEMENT_RUN,
                base_run=BASE_RUN,
                base_verification=verification,
            )

        self.assertEqual(result["status"], "completed")

        execute.assert_called_once_with(
            settings,
            SUPPLEMENT_RUN,
            BASE_RUN,
            verification,
            compare_with=None,
        )


class StarterExtensionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db = sqlite3.connect(":memory:")
        self.addCleanup(self.db.close)

        self.db.execute(
            "ATTACH DATABASE ':memory:' AS previous"
        )

        for schema in ("main", "previous"):
            self.db.executescript(
                f"""
                CREATE TABLE {schema}.starter_pre_game(
                    game_id TEXT,
                    date TEXT,
                    team_id INTEGER,
                    player_id TEXT,
                    starts_last_28d INTEGER,
                    previous_start_date TEXT,
                    days_since_start INTEGER,
                    previous_appearance_date TEXT,
                    days_since_appearance INTEGER,
                    observed_pitches_last_7d INTEGER
                );

                CREATE TABLE {schema}.m1_index(
                    sample_id TEXT,
                    game_id TEXT,
                    sample_date TEXT
                );

                CREATE TABLE {schema}.m1_inputs(
                    sample_id TEXT,
                    competition TEXT,
                    home_team_id INTEGER,
                    away_team_id INTEGER
                );

                CREATE TABLE {schema}.starter_history(
                    game_id TEXT,
                    date TEXT,
                    team_id INTEGER,
                    player_id TEXT
                );
                """
            )

            self.db.execute(
                f"""
                INSERT INTO {schema}.starter_pre_game
                VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    OLD_GAME,
                    "2024-09-27",
                    2001,
                    "hf:100",
                    1,
                    "2024-09-20",
                    7,
                    "2024-09-20",
                    7,
                    80,
                ),
            )

            self.db.execute(
                f"INSERT INTO {schema}.m1_index VALUES (?,?,?)",
                (
                    OLD_GAME,
                    OLD_GAME,
                    "2024-09-27",
                ),
            )

            self.db.execute(
                f"INSERT INTO {schema}.m1_inputs VALUES (?,?,?,?)",
                (
                    OLD_GAME,
                    "regular",
                    2001,
                    2008,
                ),
            )

            self.db.execute(
                f"INSERT INTO {schema}.starter_history VALUES (?,?,?,?)",
                (
                    OLD_GAME,
                    "2024-09-27",
                    2001,
                    "hf:100",
                ),
            )

        self.db.execute(
            "INSERT INTO main.m1_index VALUES (?,?,?)",
            (
                NEW_GAME,
                NEW_GAME,
                "2024-10-21",
            ),
        )

        self.db.execute(
            "INSERT INTO main.m1_inputs VALUES (?,?,?,?)",
            (
                NEW_GAME,
                "postseason",
                2001,
                2008,
            ),
        )

        self.db.execute(
            """
            INSERT INTO main.starter_pre_game
            VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            (
                NEW_GAME,
                "2024-10-21",
                2001,
                "hf:100",
                1,
                "2024-09-27",
                24,
                "2024-09-27",
                24,
                None,
            ),
        )

    def test_valid_postseason_extension(self) -> None:
        result = verify_starter_extension(self.db)

        self.assertEqual(result["previous_rows"], 1)
        self.assertEqual(result["current_rows"], 2)
        self.assertEqual(result["added_rows"], 1)
        self.assertEqual(result["added_games"], 1)
        self.assertEqual(result["removed_or_changed_rows"], 0)
        self.assertEqual(result["invalid_added_rows"], 0)

    def test_existing_row_change_is_rejected(self) -> None:
        self.db.execute(
            """
            UPDATE main.starter_pre_game
            SET starts_last_28d=2 WHERE game_id=?
            """,
            (OLD_GAME,),
        )

        with self.assertRaises(ValueError):
            verify_starter_extension(self.db)

    def test_existing_row_deletion_is_rejected(self) -> None:
        self.db.execute(
            "DELETE FROM main.starter_pre_game WHERE game_id=?",
            (OLD_GAME,),
        )

        with self.assertRaises(ValueError):
            verify_starter_extension(self.db)

    def test_duplicate_row_is_rejected(self) -> None:
        self.db.execute(
            """
            INSERT INTO main.starter_pre_game
            SELECT * FROM main.starter_pre_game WHERE game_id=?
            """,
            (NEW_GAME,),
        )

        with self.assertRaises(ValueError):
            verify_starter_extension(self.db)

    def test_future_start_is_rejected(self) -> None:
        self.db.execute(
            """
            UPDATE main.starter_pre_game
            SET previous_start_date='2024-10-21',
                days_since_start=0
            WHERE game_id=?
            """,
            (NEW_GAME,),
        )

        with self.assertRaises(ValueError):
            verify_starter_extension(self.db)

    def test_future_appearance_is_rejected(self) -> None:
        self.db.execute(
            """
            UPDATE main.starter_pre_game
            SET previous_appearance_date='2024-10-22',
                days_since_appearance=-1
            WHERE game_id=?
            """,
            (NEW_GAME,),
        )

        with self.assertRaises(ValueError):
            verify_starter_extension(self.db)

    def test_non_postseason_extension_is_rejected(self) -> None:
        self.db.execute(
            """
            UPDATE main.m1_inputs
            SET competition='regular' WHERE sample_id=?
            """,
            (NEW_GAME,),
        )

        with self.assertRaises(ValueError):
            verify_starter_extension(self.db)

    def test_unrelated_team_is_rejected(self) -> None:
        self.db.execute(
            """
            UPDATE main.starter_pre_game
            SET team_id=2005 WHERE game_id=?
            """,
            (NEW_GAME,),
        )

        with self.assertRaises(ValueError):
            verify_starter_extension(self.db)

    def test_missing_past_start_evidence_is_rejected(self) -> None:
        self.db.execute("DELETE FROM main.starter_history")

        with self.assertRaises(ValueError):
            verify_starter_extension(self.db)

    def test_existing_game_cannot_receive_new_starter_rows(self) -> None:
        """기존 경기의 선발 후보 추가를 정상 변경으로 간주하지 않는다."""
        self.db.execute(
            """
            INSERT INTO previous.m1_index VALUES (?,?,?)
            """,
            (
                NEW_GAME,
                NEW_GAME,
                "2024-10-21",
            ),
        )

        with self.assertRaises(ValueError):
            verify_starter_extension(self.db)


class WorkflowCliTests(unittest.TestCase):
    def test_standardize_cli_forwards_supplement_arguments(self) -> None:
        settings = SimpleNamespace(log_level="ERROR")

        with (
            patch(
                "src.__main__.load_settings",
                return_value=settings,
            ),
            patch(
                "src.datasets.postseason_supplement.standardize_selected",
                return_value={"status": "completed"},
            ) as execute,
            redirect_stdout(io.StringIO()),
        ):
            status = main([
                "standardize",
                "--as-of-run", SUPPLEMENT_RUN,
                "--base-run", BASE_RUN,
                "--base-verification", "4" * 32,
            ])

        self.assertEqual(status, 0)

        execute.assert_called_once_with(
            settings,
            as_of_run=SUPPLEMENT_RUN,
            compare_run=None,
            base_run=BASE_RUN,
            base_verification="4" * 32,
        )

    def test_validation_cli_forwards_comparison(self) -> None:
        settings = SimpleNamespace(log_level="ERROR")
        comparison = f"{BASE_RUN}:{'4' * 32}"

        with (
            patch(
                "src.__main__.load_settings",
                return_value=settings,
            ),
            patch(
                "src.datasets.postseason_supplement.validate_selected",
                return_value={"status": "completed"},
            ) as execute,
            redirect_stdout(io.StringIO()),
        ):
            status = main([
                "validate-standard",
                "--run-id", SUPPLEMENT_RUN,
                "--compare-with", comparison,
            ])

        self.assertEqual(status, 0)

        execute.assert_called_once_with(
            settings,
            SUPPLEMENT_RUN,
            compare_with=comparison,
        )

    def test_unexpected_table_changes_remain_blocked(self) -> None:
        """허용 대상 외의 모든 모델 테이블 변경을 차단한다."""
        previous = {
            "starter_pre_game": {"rows": 1, "sha256": "a"},
            "m2_targets": {"rows": 2, "sha256": "b"},
        }

        current = {
            "starter_pre_game": {"rows": 2, "sha256": "c"},
            "m2_targets": {"rows": 2, "sha256": "b"},
        }

        self.assertEqual(
            unchanged_tables(
                previous,
                current,
                {"starter_pre_game"},
            ),
            ["starter_pre_game"],
        )

        current["m2_targets"]["sha256"] = "changed"

        with self.assertRaises(ValueError):
            unchanged_tables(
                previous,
                current,
                {"starter_pre_game"},
            )


if __name__ == "__main__":
    unittest.main()