"""포스트시즌 원본·완료 근거·판본·CLI 및 선발 후보 추가 범위를 검증한다."""

from copy import deepcopy
from contextlib import closing, redirect_stdout
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
from src.datasets.model_data import verify_version_inventory
from src.datasets.postseason_supplement import (
    collection_run_for_supplement,
    flatten_postseason,
    normalize_season,
    regular_completion_evidence,
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


def series_fixture() -> dict[str, object]:
    """실제 Dent 원본과 동일한 문자열 시즌 구조의 최소 자료."""
    return {
        "season": "2024",
        "games": [
            {"gamePk": "regular-game-not-for-postseason"}
        ],
        "postseason": {
            "wildCard": {
                "isComplete": True,
                "winnerId": 2002,
            },
            "koreanSeries": {
                "name": "Korean Series",
                "isComplete": True,
                "championId": 2001,
                "team1": {
                    "id": 2001,
                    "wins": 1,
                    "isWinner": True,
                },
                "adv": {
                    "id": 2008,
                    "wins": 0,
                    "isWinner": False,
                },
                "games": [
                    {
                        "game": 1,
                        "date": "2024-10-21",
                        "away": "Samsung Lions",
                        "home": "KIA Tigers",
                        "awayScore": 1,
                        "homeScore": 5,
                        "winnerId": 2001,
                    }
                ],
            },
        },
    }


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


class PostseasonSupplementTests(unittest.TestCase):
    """원본 형식, 실제 경기 결과 및 정규시즌 완료 근거를 검사한다."""

    def test_string_and_integer_season(self) -> None:
        """문자열 시즌과 정수 시즌을 동일하게 처리한다."""
        for season in ("2019", "2020", "2024", "2025"):
            with self.subTest(season=season):
                self.assertEqual(
                    normalize_season(season),
                    int(season),
                )

        raw = series_fixture()

        string_rows, string_series = flatten_postseason(
            raw,
            2024,
        )

        raw["season"] = 2024

        integer_rows, integer_series = flatten_postseason(
            raw,
            2024,
        )

        self.assertEqual(string_rows, integer_rows)
        self.assertEqual(string_series, integer_series)

    def test_invalid_season_types_are_rejected(self) -> None:
        """잘못된 입력 타입과 형식을 시즌으로 해석하지 않는다."""
        invalid = (
            None,
            True,
            False,
            2024.0,
            "2024.0",
            " 2024",
            "2024 ",
            "024",
            "abcd",
            "２０２４",
            [],
            {},
            0,
        )

        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    normalize_season(value)

    def test_only_existing_postseason_arrays_are_expanded(self) -> None:
        """존재하는 경기 배열만 변환하고 미확보 라운드를 보존한다."""
        raw = series_fixture()

        rows, series = flatten_postseason(raw, 2024)

        self.assertEqual(len(rows), 1)
        self.assertEqual(len(series), 2)

        self.assertEqual(
            rows[0]["_stage"],
            "koreanSeries",
        )

        self.assertEqual(
            rows[0]["_completion_basis"],
            "completed_series_score_winner",
        )

        self.assertNotIn("isFinal", rows[0])
        self.assertIsNone(rows[0]["_season_complete"])

        self.assertFalse(
            series[0]["metadata"]["_games_array_present"]
        )

        self.assertEqual(series[0]["score_rows"], 0)

        self.assertFalse(
            series[1]["all_rounds_score_observed"]
        )

        self.assertNotIn(
            "_source_row",
            raw["postseason"]["koreanSeries"]["games"][0],
        )

        self.assertEqual(raw["season"], "2024")

    def test_invalid_games_array_is_rejected(self) -> None:
        """존재하지만 배열이 아닌 경기 필드를 거절한다."""
        raw = series_fixture()
        raw["postseason"]["koreanSeries"]["games"] = None

        with self.assertRaises(ValueError):
            flatten_postseason(raw, 2024)

    def test_explicit_final_marker(self) -> None:
        """원본의 명시적 경기 종료 표시를 구분한다."""
        raw = series_fixture()

        raw["postseason"]["koreanSeries"]["games"][0][
            "isFinal"
        ] = True

        rows, _ = flatten_postseason(raw, 2024)

        self.assertEqual(
            rows[0]["_completion_basis"],
            "explicit_is_final",
        )

    def test_absent_final_marker_in_incomplete_series(self) -> None:
        """미완료 시리즈의 점수를 종료 결과로 추정하지 않는다."""
        raw = series_fixture()

        raw["postseason"]["koreanSeries"][
            "isComplete"
        ] = False

        rows, _ = flatten_postseason(raw, 2024)

        self.assertEqual(
            rows[0]["_completion_basis"],
            "not_final",
        )

    def test_explicit_nonfinal_is_not_inferred(self) -> None:
        """명시된 경기 미종료 상태를 임의로 변경하지 않는다."""
        raw = series_fixture()

        series = raw["postseason"]["koreanSeries"]
        series["isComplete"] = False
        series["games"][0]["isFinal"] = False

        rows, _ = flatten_postseason(raw, 2024)

        self.assertEqual(
            rows[0]["_completion_basis"],
            "not_final",
        )

    def test_invalid_score_winner_is_rejected(self) -> None:
        """실제 점수와 승자가 불일치하면 실패한다."""
        raw = series_fixture()

        raw["postseason"]["koreanSeries"]["games"][0][
            "winnerId"
        ] = 2008

        with self.assertRaises(ValueError):
            flatten_postseason(raw, 2024)

    def test_invalid_winner_type_is_rejected(self) -> None:
        """실수 형태의 승자 ID를 정수 ID로 임의 처리하지 않는다."""
        raw = series_fixture()

        raw["postseason"]["koreanSeries"]["games"][0][
            "winnerId"
        ] = 2001.0

        with self.assertRaises(ValueError):
            flatten_postseason(raw, 2024)

    def test_duplicate_series_game_is_rejected(self) -> None:
        """같은 시리즈 내 중복 경기 차수를 차단한다."""
        raw = series_fixture()

        games = raw["postseason"]["koreanSeries"]["games"]
        games.append(deepcopy(games[0]))

        with self.assertRaises(ValueError):
            flatten_postseason(raw, 2024)

    def test_series_score_mismatch_is_rejected(self) -> None:
        """실제 경기 승수와 시리즈 승수가 다르면 실패한다."""
        raw = series_fixture()

        raw["postseason"]["koreanSeries"]["team1"][
            "wins"
        ] = 2

        with self.assertRaises(ValueError):
            flatten_postseason(raw, 2024)

    def test_wrong_season_is_rejected(self) -> None:
        """정수 및 문자열 시즌 모두 기대 시즌과 일치해야 한다."""
        raw = series_fixture()

        with self.assertRaises(ValueError):
            flatten_postseason(raw, 2025)

        raw["season"] = 2024

        with self.assertRaises(ValueError):
            flatten_postseason(raw, 2025)

    def test_wrong_game_date_season_is_rejected(self) -> None:
        """경기 날짜가 원본 시즌과 다르면 거절한다."""
        raw = series_fixture()

        raw["postseason"]["koreanSeries"]["games"][0][
            "date"
        ] = "2025-10-21"

        with self.assertRaises(ValueError):
            flatten_postseason(raw, 2024)

    def test_shared_raw_hash_keeps_distinct_logical_targets(self) -> None:
        """같은 물리 해시를 참조하는 서로 다른 논리 대상을 유지한다."""
        records = [
            {
                "target_id": "dent-results-2024",
                "source_sha": "a" * 64,
                "selected": True,
            },
            {
                "target_id": "dent-postseason-2024",
                "source_sha": "a" * 64,
                "selected": True,
            },
        ]

        verify_version_inventory(
            records,
            object_count=1,
            logical_targets=2,
        )

        with self.assertRaises(ValueError):
            verify_version_inventory(
                records,
                object_count=2,
                logical_targets=2,
            )

    def test_duplicate_version_is_rejected(self) -> None:
        """동일 논리 대상의 중복 판본은 거절한다."""
        record = {
            "target_id": "dent-results-2024",
            "source_sha": "a" * 64,
            "selected": True,
        }

        with self.assertRaises(ValueError):
            verify_version_inventory(
                [record, record],
                object_count=1,
                logical_targets=1,
            )

    def test_missing_selected_version_is_rejected(self) -> None:
        """성공 판본이 있어도 채택 대상이 누락되면 실패한다."""
        records = [
            {
                "target_id": "old-target",
                "source_sha": "a" * 64,
                "selected": False,
            },
            {
                "target_id": "new-target",
                "source_sha": "b" * 64,
                "selected": True,
            },
        ]

        with self.assertRaises(ValueError):
            verify_version_inventory(
                records,
                object_count=2,
                logical_targets=1,
            )

    def test_missing_pbp_games_rejects_completion_evidence(self) -> None:
        """완료 경기와 PBP 대응 집합이 없으면 시즌 완료를 확정하지 않는다."""
        with closing(sqlite3.connect(":memory:")) as db:
            db.executescript(
                """
                CREATE TABLE games(
                    game_id TEXT,
                    date TEXT,
                    season INTEGER,
                    competition TEXT,
                    home_team_id INTEGER,
                    away_team_id INTEGER,
                    source_sha TEXT,
                    result_valid INTEGER,
                    status TEXT
                );

                CREATE TABLE game_checks(
                    game_id TEXT,
                    final_score_match INTEGER,
                    source_sha TEXT
                );

                INSERT INTO games VALUES (
                    'game-1',
                    '2023-10-17',
                    2023,
                    'regular',
                    2001,
                    2002,
                    'regular-hash',
                    1,
                    'Final'
                );
                """
            )

            with self.assertRaises(ValueError):
                regular_completion_evidence(db, 2023)

    def test_complete_regular_season_evidence(self) -> None:
        """720경기, 팀별 144경기, PBP 집합 및 실제 점수를 대조한다."""
        with closing(sqlite3.connect(":memory:")) as db:
            db.executescript(
                """
                CREATE TABLE games(
                    game_id TEXT PRIMARY KEY,
                    date TEXT,
                    season INTEGER,
                    competition TEXT,
                    home_team_id INTEGER,
                    away_team_id INTEGER,
                    source_sha TEXT,
                    result_valid INTEGER,
                    status TEXT
                );

                CREATE TABLE game_checks(
                    game_id TEXT PRIMARY KEY,
                    final_score_match INTEGER,
                    source_sha TEXT
                );
                """
            )

            games = []

            for home in range(2001, 2011):
                for away in range(home + 1, 2011):
                    for index in range(16):
                        games.append((
                            f"{home}:{away}:{index}",
                            "2023-10-17",
                            2023,
                            "regular",
                            home,
                            away,
                            "regular-hash",
                            1,
                            "Final",
                        ))

            db.executemany(
                "INSERT INTO games VALUES (?,?,?,?,?,?,?,?,?)",
                games,
            )

            db.executemany(
                "INSERT INTO game_checks VALUES (?,?,?)",
                (
                    (game[0], 1, "pbp-hash")
                    for game in games
                ),
            )

            evidence = regular_completion_evidence(db, 2023)

            self.assertEqual(evidence["completed_games"], 720)
            self.assertEqual(evidence["pbp_games"], 720)
            self.assertTrue(evidence["game_sets_match"])

            self.assertEqual(
                evidence["label_ready_date"],
                "2023-10-18",
            )

            self.assertTrue(
                all(
                    count == 144
                    for count in evidence["team_games"].values()
                )
            )

            db.execute(
                "UPDATE game_checks SET final_score_match=0 "
                "WHERE game_id=?",
                (games[0][0],),
            )

            with self.assertRaises(ValueError):
                regular_completion_evidence(db, 2023)


class SourceScopeTests(unittest.TestCase):
    """기존 및 보완 원본의 선정 범위와 실행 경계를 검사한다."""

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
        """기존 표준화 범위에서 보완 원본의 논리 대상을 분리한다."""
        intake = intake_sources(
            self.published,
            as_of_run=BASE_RUN,
        )

        self.assertEqual(set(intake.selected), EXPECTED_TARGETS)

        self.assertEqual(
            len(intake.versions),
            len(EXPECTED_TARGETS),
        )

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
        """보완 경로가 기존 및 추가 논리 대상을 모두 선택하는지 검사한다."""
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
        """기존 범위 밖의 승인된 보완 원본도 해시 오류를 거절한다."""
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
        """등록되지 않은 논리 대상의 기존 차단 기준을 유지한다."""
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
        """수집 실행 경계에 따라 기존 및 보완 경로를 분리한다."""
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
        """보완 수집 이전의 실행은 기존 표준화 경로를 사용한다."""
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
        """보완 수집 실행은 기준 표준판에서 증분 생성한다."""
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
    """기존 선발 후보 불변성과 신규 포스트시즌 후보의 정보 시점을 검사한다."""

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
                f"""
                INSERT INTO {schema}.m1_index
                VALUES (?,?,?)
                """,
                (
                    OLD_GAME,
                    OLD_GAME,
                    "2024-09-27",
                ),
            )

            self.db.execute(
                f"""
                INSERT INTO {schema}.m1_inputs
                VALUES (?,?,?,?)
                """,
                (
                    OLD_GAME,
                    "regular",
                    2001,
                    2008,
                ),
            )

            self.db.execute(
                f"""
                INSERT INTO {schema}.starter_history
                VALUES (?,?,?,?)
                """,
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
        """기존 행이 유지되고 신규 포스트시즌 사전 기록만 추가된다."""
        result = verify_starter_extension(self.db)

        self.assertEqual(result["previous_rows"], 1)
        self.assertEqual(result["current_rows"], 2)
        self.assertEqual(result["added_rows"], 1)
        self.assertEqual(result["added_games"], 1)
        self.assertEqual(result["removed_or_changed_rows"], 0)
        self.assertEqual(result["invalid_added_rows"], 0)

    def test_existing_row_change_is_rejected(self) -> None:
        """기존 선발 후보 행의 값 변경을 거절한다."""
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
        """기존 선발 후보 행의 삭제를 거절한다."""
        self.db.execute(
            """
            DELETE FROM main.starter_pre_game
            WHERE game_id=?
            """,
            (OLD_GAME,),
        )

        with self.assertRaises(ValueError):
            verify_starter_extension(self.db)

    def test_duplicate_row_is_rejected(self) -> None:
        """같은 경기·팀·선수 행의 중복을 거절한다."""
        self.db.execute(
            """
            INSERT INTO main.starter_pre_game
            SELECT * FROM main.starter_pre_game
            WHERE game_id=?
            """,
            (NEW_GAME,),
        )

        with self.assertRaises(ValueError):
            verify_starter_extension(self.db)

    def test_future_start_is_rejected(self) -> None:
        """기준 경기 당일 이후의 선발 이력 혼입을 차단한다."""
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
        """기준 경기 이후의 등판 이력 혼입을 차단한다."""
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
        """보완 범위 밖 정규시즌 경기의 신규 행을 거절한다."""
        self.db.execute(
            """
            UPDATE main.m1_inputs
            SET competition='regular'
            WHERE sample_id=?
            """,
            (NEW_GAME,),
        )

        with self.assertRaises(ValueError):
            verify_starter_extension(self.db)

    def test_unrelated_team_is_rejected(self) -> None:
        """해당 경기의 참가 팀이 아닌 선수 기록을 거절한다."""
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
        """실제 과거 선발 근거가 없는 신규 후보를 거절한다."""
        self.db.execute("DELETE FROM main.starter_history")

        with self.assertRaises(ValueError):
            verify_starter_extension(self.db)

    def test_existing_game_cannot_receive_new_starter_rows(self) -> None:
        """기존 경기의 신규 선발 후보는 정상 보완으로 인정하지 않는다."""
        self.db.execute(
            """
            INSERT INTO previous.m1_index
            VALUES (?,?,?)
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
    """기존 CLI 호환 및 모델 변경 범위 제한을 검사한다."""

    def test_standardize_cli_forwards_supplement_arguments(self) -> None:
        """기존 표준화 CLI가 보완 기준 및 비교 설정을 전달한다."""
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
        """기존 검증 CLI가 검증 대상과 비교 실행 ID를 전달한다."""
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
        """허용 대상 이외의 모델 테이블 변경을 지속 차단한다."""
        previous = {
            "starter_pre_game": {
                "rows": 1,
                "sha256": "a",
            },
            "m2_targets": {
                "rows": 2,
                "sha256": "b",
            },
        }

        current = {
            "starter_pre_game": {
                "rows": 2,
                "sha256": "c",
            },
            "m2_targets": {
                "rows": 2,
                "sha256": "b",
            },
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