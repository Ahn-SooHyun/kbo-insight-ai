"""보조 분석 운영 규칙 선정 및 실제 입력 경계 회귀 테스트."""

from argparse import Namespace
from datetime import date, timedelta
from pathlib import Path
import sqlite3
from types import SimpleNamespace
from typing import cast
import unittest
from unittest.mock import patch

from src.models.game_outcome import model as m1
from src.settings import Settings
from src.simulation.regular_season.data import Scenario
from src.simulation.regular_season.rules import (
    Fixture,
    PlayedGame,
    Record,
    get_rules,
)
from src.simulation.support_analysis import (
    STARTER_RULE_VERSION,
    STARTER_SELECTED_METHOD,
    TOP_K,
    VERSION,
    starter_selection,
)
from src.simulation.support_analysis.__main__ import execute
from src.simulation.support_analysis.analysis import (
    forecast_remaining,
    rank_starters,
    record_summary,
    starter_report,
)
from src.simulation.support_analysis.data import (
    Appearance,
    Identity,
    Start,
    StarterData,
    read_starter_data,
)
from src.simulation.support_analysis.evaluation import read_truth


CUTOFF = date(2025, 9, 1)
TEAM = 1
OPPONENT = 2


def prior_day(days: int) -> date:
    """기준일 이전 합성 경기 날짜."""

    return CUTOFF - timedelta(days=days)


def make_fixture(
    game_id: str,
    day: date,
    home: int = TEAM,
    away: int = OPPONENT,
) -> Fixture:
    """합성 자료 계산에 사용하는 경기 식별 정보."""

    return Fixture(
        game_id=game_id,
        day=day,
        home=home,
        away=away,
        number=0,
    )


def make_data() -> StarterData:
    """기준 규칙과 제안 규칙이 서로 다른 순위를 생성하는 자료."""

    return StarterData(
        starts={
            "A": (
                Start(
                    "a",
                    prior_day(5),
                    TEAM,
                    "A",
                    "A",
                    1,
                ),
            ),
            "B": (
                Start(
                    "b",
                    prior_day(10),
                    TEAM,
                    "B",
                    "B",
                    1,
                ),
            ),
        },
        identities={
            "A": (
                Identity(
                    prior_day(5),
                    TEAM,
                    "A",
                ),
            ),
            "B": (
                Identity(
                    prior_day(10),
                    TEAM,
                    "B",
                ),
            ),
        },
        appearances={
            "A": (
                Appearance(
                    "a",
                    prior_day(5),
                    TEAM,
                    80,
                ),
            ),
        },
    )


def make_completed() -> list[PlayedGame]:
    """후보 생성 자료의 과거 완료 경기."""

    return [
        PlayedGame(
            make_fixture(
                "b",
                prior_day(10),
            ),
            3,
            2,
        ),
        PlayedGame(
            make_fixture(
                "a",
                prior_day(5),
            ),
            2,
            3,
        ),
    ]


def make_model_database() -> sqlite3.Connection:
    """실제 모델 자료에서 조회하는 최소 스키마를 구성한다."""

    db = sqlite3.connect(":memory:")

    db.executescript(
        """
        CREATE TABLE starter_history (
            game_id TEXT,
            side TEXT,
            team_id INTEGER,
            date TEXT,
            player_id TEXT,
            name TEXT,
            evidence_count INTEGER
        );

        CREATE TABLE starter_actual (
            game_id TEXT,
            side TEXT,
            team_id INTEGER,
            date TEXT,
            player_id TEXT,
            name TEXT,
            evidence_count INTEGER
        );

        CREATE TABLE player_identity_evidence (
            player_id TEXT,
            team_id INTEGER,
            name TEXT,
            first_seen_date TEXT,
            game_id TEXT,
            season INTEGER,
            observed_season INTEGER
        );
        """
    )

    db.executemany(
        """
        INSERT INTO starter_history
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "past",
                "home",
                TEAM,
                prior_day(5).isoformat(),
                "A",
                "A",
                1,
            ),
            (
                "target",
                "home",
                TEAM,
                CUTOFF.isoformat(),
                "A",
                "A",
                1,
            ),
            (
                "future",
                "home",
                TEAM,
                (CUTOFF + timedelta(days=1)).isoformat(),
                "B",
                "B",
                1,
            ),
        ],
    )

    db.execute(
        """
        INSERT INTO starter_actual
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "target",
            "home",
            TEAM,
            CUTOFF.isoformat(),
            "A",
            "A",
            1,
        ),
    )

    db.executemany(
        """
        INSERT INTO player_identity_evidence
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "A",
                TEAM,
                "A",
                prior_day(5).isoformat(),
                "past",
                2025,
                2025,
            ),
            (
                "A",
                TEAM,
                "A",
                CUTOFF.isoformat(),
                "target",
                2025,
                2025,
            ),
        ],
    )

    return db


def make_standard_database() -> sqlite3.Connection:
    """투수 이력 조회에 필요한 합성 표준 자료를 구성한다."""

    db = sqlite3.connect(":memory:")

    db.executescript(
        """
        CREATE TABLE eligible_games (
            game_id TEXT,
            date TEXT,
            season INTEGER,
            competition TEXT
        );

        CREATE TABLE player_games (
            game_id TEXT,
            team_id INTEGER,
            player_id TEXT,
            role TEXT,
            observed_pitches INTEGER
        );

        CREATE TABLE player_admission (
            game_id TEXT,
            team_id INTEGER,
            player_id TEXT,
            role TEXT,
            preliminary_candidate INTEGER,
            direct_mismatches INTEGER,
            team_mismatches INTEGER,
            game_state_uncertain INTEGER
        );
        """
    )

    for game_id, day, pitches in (
        ("past", prior_day(5), 80),
        ("target", CUTOFF, 100),
        (
            "future",
            CUTOFF + timedelta(days=1),
            90,
        ),
    ):
        db.execute(
            """
            INSERT INTO eligible_games
            VALUES (?, ?, ?, ?)
            """,
            (
                game_id,
                day.isoformat(),
                2025,
                "regular",
            ),
        )

        db.execute(
            """
            INSERT INTO player_games
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                game_id,
                TEAM,
                "A",
                "pitcher",
                pitches,
            ),
        )

        db.execute(
            """
            INSERT INTO player_admission
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                game_id,
                TEAM,
                "A",
                "pitcher",
                1,
                0,
                0,
                0,
            ),
        )

    return db


class TestOperationalSelection(unittest.TestCase):
    """검증에서 선정한 운영 규칙과 비교 규칙을 분리한다."""

    def test_operational_rule_is_baseline(self) -> None:
        ranking = rank_starters(
            make_data(),
            TEAM,
            CUTOFF,
            CUTOFF,
        )

        self.assertEqual(
            STARTER_SELECTED_METHOD,
            "baseline",
        )

        self.assertEqual(
            ranking.operational,
            ranking.baseline,
        )

        self.assertEqual(
            ranking.operational[0].player_id,
            "B",
        )

        self.assertEqual(
            ranking.proposed[0].player_id,
            "A",
        )

    def test_starter_report_uses_selected_order(self) -> None:
        result = starter_report(
            make_data(),
            TEAM,
            make_fixture("target", CUTOFF),
            CUTOFF,
            make_completed(),
            CUTOFF,
        )

        candidates = cast(
            list[dict[str, object]],
            result["candidates"],
        )

        self.assertEqual(
            result["selected_method"],
            "baseline",
        )

        self.assertEqual(
            result["rule_version"],
            STARTER_RULE_VERSION,
        )

        self.assertEqual(
            [row["player_id"] for row in candidates],
            ["B", "A"],
        )

        self.assertLessEqual(
            len(candidates),
            TOP_K,
        )

        self.assertIsNone(
            result["official_announcement"]
        )

    def test_rule_selection_metadata(self) -> None:
        result = starter_selection()

        self.assertEqual(
            result["selected_method"],
            "baseline",
        )

        self.assertEqual(
            result["selection_split"],
            "validation",
        )

        self.assertEqual(
            result["selection_season"],
            2025,
        )

        self.assertFalse(result["final_test_used"])

        self.assertFalse(
            result["independent_final_evaluation_completed"]
        )

    def test_cli_evaluation_records_selection(self) -> None:
        settings = cast(
            Settings,
            SimpleNamespace(),
        )

        args = Namespace(
            command="evaluate",
        )

        with patch(
            "src.simulation.support_analysis.__main__.run_evaluation",
            return_value={
                "scope": "starter_validation",
                "comparison": {
                    "baseline": 1,
                    "proposed": 0,
                },
            },
        ):
            result = execute(
                settings,
                args,
            )

        self.assertEqual(
            result["starter_selection"],
            starter_selection(),
        )

        self.assertEqual(
            result["version"],
            VERSION,
        )

        self.assertEqual(
            result["status"],
            "completed",
        )

        self.assertIn(
            "source_sha256",
            result,
        )


class TestWorkloadEvidence(unittest.TestCase):
    """관측 투구 합계와 자료의 완전성을 구분한다."""

    def test_known_start_without_appearance_is_marked(self) -> None:
        data = make_data()

        data.starts["A"] += (
            Start(
                "missing",
                prior_day(3),
                TEAM,
                "A",
                "A",
                1,
            ),
        )

        result = rank_starters(
            data,
            TEAM,
            CUTOFF,
            CUTOFF,
        )

        candidate = next(
            row
            for row in result.operational
            if row.player_id == "A"
        )

        self.assertEqual(
            candidate.observed_pitches_last_7d,
            80,
        )

        self.assertEqual(
            candidate.workload_observed_games,
            1,
        )

        self.assertEqual(
            candidate.workload_missing_known_starts,
            1,
        )

        self.assertEqual(
            candidate.workload_status,
            "partial_observed",
        )

    def test_no_observed_workload_remains_unknown(self) -> None:
        data = make_data()
        data.appearances.clear()

        result = rank_starters(
            data,
            TEAM,
            CUTOFF,
            CUTOFF,
        )

        candidate = next(
            row
            for row in result.operational
            if row.player_id == "A"
        )

        self.assertIsNone(
            candidate.observed_pitches_last_7d
        )

        self.assertEqual(
            candidate.workload_missing_known_starts,
            1,
        )

        self.assertEqual(
            candidate.workload_status,
            "unobserved",
        )

    def test_observed_pitch_count_missing_is_not_zero(self) -> None:
        data = make_data()

        data.appearances["A"] = (
            Appearance(
                "a",
                prior_day(5),
                TEAM,
                None,
            ),
        )

        result = rank_starters(
            data,
            TEAM,
            CUTOFF,
            CUTOFF,
        )

        candidate = next(
            row
            for row in result.operational
            if row.player_id == "A"
        )

        self.assertIsNone(
            candidate.observed_pitches_last_7d
        )

        self.assertEqual(
            candidate.workload_missing_pitch_counts,
            1,
        )

        self.assertEqual(
            candidate.workload_missing_known_starts,
            0,
        )

        self.assertEqual(
            candidate.workload_status,
            "partial_observed",
        )


class TestDatabaseTimeBoundary(unittest.TestCase):
    """SQL 조회부터 후보 생성까지 날짜 경계를 검사한다."""

    def test_target_truth_and_future_mutations(self) -> None:
        model = make_model_database()
        standard = make_standard_database()

        self.addCleanup(model.close)
        self.addCleanup(standard.close)

        completed = [
            PlayedGame(
                make_fixture(
                    "past",
                    prior_day(5),
                ),
                4,
                2,
            ),
        ]

        target = PlayedGame(
            make_fixture(
                "target",
                CUTOFF,
            ),
            3,
            2,
        )

        first = read_starter_data(
            standard,
            model,
            completed,
            2025,
            CUTOFF,
        )

        first_rank = rank_starters(
            first,
            TEAM,
            CUTOFF,
            CUTOFF,
        )

        self.assertEqual(
            first_rank.operational[0].player_id,
            "A",
        )

        self.assertEqual(
            first_rank.operational[0].observed_pitches_last_7d,
            80,
        )

        truth_before = read_truth(
            model,
            [*completed, target],
            CUTOFF + timedelta(days=1),
        )

        model.execute(
            """
            UPDATE starter_actual
            SET player_id = 'B', name = 'B'
            WHERE game_id = 'target'
            """
        )

        model.execute(
            """
            UPDATE starter_history
            SET player_id = 'B', name = 'B'
            WHERE date >= ?
            """,
            (CUTOFF.isoformat(),),
        )

        model.execute(
            """
            UPDATE player_identity_evidence
            SET name = 'future_conflict'
            WHERE first_seen_date >= ?
            """,
            (CUTOFF.isoformat(),),
        )

        standard.execute(
            """
            UPDATE player_games
            SET observed_pitches = 999
            WHERE game_id IN ('target', 'future')
            """
        )

        standard.execute(
            """
            UPDATE player_admission
            SET preliminary_candidate = 0
            WHERE game_id IN ('target', 'future')
            """
        )

        second = read_starter_data(
            standard,
            model,
            completed,
            2025,
            CUTOFF,
        )

        second_rank = rank_starters(
            second,
            TEAM,
            CUTOFF,
            CUTOFF,
        )

        truth_after = read_truth(
            model,
            [*completed, target],
            CUTOFF + timedelta(days=1),
        )

        self.assertEqual(first, second)
        self.assertEqual(first_rank, second_rank)

        self.assertEqual(
            truth_before["target", TEAM].player_id,
            "A",
        )

        self.assertEqual(
            truth_after["target", TEAM].player_id,
            "B",
        )

        self.assertNotEqual(
            truth_before,
            truth_after,
        )


class TestFinalOutputContracts(unittest.TestCase):
    """최종 출력에서 분모 및 무결성 실패를 검사한다."""

    def test_all_ties_have_explicit_reason(self) -> None:
        record = Record()
        record.update(3, 3)

        result = record_summary(record)

        self.assertIsNone(
            result["win_rate"]
        )

        self.assertEqual(
            result["win_rate_denominator"],
            0,
        )

        self.assertEqual(
            result["win_rate_reason"],
            "no_decided_games",
        )

        self.assertEqual(
            result["runs_for_per_game"],
            3.0,
        )

        self.assertIsNone(
            result["run_rate_reason"]
        )

    def test_empty_record_has_run_rate_reason(self) -> None:
        result = record_summary(Record())

        self.assertIsNone(
            result["runs_for_per_game"]
        )

        self.assertEqual(
            result["run_rate_reason"],
            "no_observations",
        )

    def test_starter_report_rejects_unrelated_team(self) -> None:
        with self.assertRaises(ValueError):
            starter_report(
                make_data(),
                999,
                make_fixture("target", CUTOFF),
                CUTOFF,
                make_completed(),
                CUTOFF,
            )

    def test_all_remaining_forecasts_are_validated(self) -> None:
        rules = get_rules(2025)

        home, away = rules.teams[:2]

        scenario = Scenario(
            rules=rules,
            cutoff=CUTOFF,
            completed=(),
            remaining=(
                make_fixture(
                    "first",
                    CUTOFF,
                    home,
                    away,
                ),
                make_fixture(
                    "second",
                    CUTOFF + timedelta(days=1),
                    home,
                    away,
                ),
            ),
            model=cast(m1.Model, {}),
            standard_db=Path("standard.sqlite3"),
            model_db=Path("model.sqlite3"),
            metadata={},
        )

        valid = cast(
            m1.Forecast,
            {
                "result_probabilities": {
                    "home": 0.6,
                    "tie": 0.1,
                    "away": 0.3,
                },
            },
        )

        invalid = cast(
            m1.Forecast,
            {
                "result_probabilities": {
                    "home": 0.8,
                    "tie": 0.3,
                    "away": 0.3,
                },
            },
        )

        with patch(
            "src.simulation.support_analysis.analysis.m1.predict",
            side_effect=[valid, invalid],
        ):
            with self.assertRaises(ValueError):
                forecast_remaining(scenario)


if __name__ == "__main__":
    unittest.main()