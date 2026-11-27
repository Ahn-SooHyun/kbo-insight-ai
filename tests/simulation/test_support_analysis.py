"""예상 선발·팀 전력·일정 난이도의 계산 및 시점 경계 테스트."""

from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
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

from src.simulation.support_analysis.__main__ import (
    build_parser,
    save_result,
)
from src.simulation.support_analysis.analysis import (
    FixtureForecast,
    analyze,
    difficulty_value,
    game_difficulty,
    rank_starters,
    ratio,
    record_summary,
    schedule_difficulty,
    starter_report,
    team_probabilities,
    team_window,
)
from src.simulation.support_analysis.data import (
    Appearance,
    Identity,
    Start,
    StarterData,
    check_side,
    completed_index,
    read_starts,
    require_count,
)
from src.simulation.support_analysis.evaluation import (
    EvaluationSample,
    Truth,
    compare_samples,
    method_metrics,
    read_truth,
)


CUTOFF = date(2025, 9, 1)
TEAM = 1
OPPONENT = 2


def past_day(days: int) -> date:
    """기준일 이전의 합성 자료 날짜."""

    return CUTOFF - timedelta(days=days)


def start(
    player: str,
    days: int,
    team: int = TEAM,
) -> Start:
    """식별자가 확인된 합성 선발 이력."""

    return Start(
        f"{player}-{days}",
        past_day(days),
        team,
        player,
        player,
        1,
    )


def identity(
    player: str,
    days: int,
    team: int = TEAM,
) -> Identity:
    """합성 선수 식별 근거."""

    return Identity(
        past_day(days),
        team,
        player,
    )


def sample_data() -> StarterData:
    """순위 규칙의 차이를 확인하는 합성 자료."""

    return StarterData(
        starts={
            "A": (start("A", 5),),
            "B": (start("B", 10),),
        },
        identities={
            "A": (identity("A", 5),),
            "B": (identity("B", 10),),
        },
        appearances={
            "A": (
                Appearance(
                    "A-5",
                    past_day(5),
                    TEAM,
                    90,
                ),
            ),
        },
    )


def fixture(
    game_id: str = "game",
    day: date = CUTOFF,
    home: int = TEAM,
    away: int = OPPONENT,
    number: int = 0,
) -> Fixture:
    """순수 계산 테스트용 경기."""

    return Fixture(
        game_id,
        day,
        home,
        away,
        number,
    )


def forecast(
    home: float = 0.6,
    tie: float = 0.1,
    away: float = 0.3,
) -> m1.Forecast:
    """승무패 계산에 필요한 테스트용 M1 출력."""

    return cast(
        m1.Forecast,
        {
            "result_probabilities": {
                "home": home,
                "tie": tie,
                "away": away,
            },
        },
    )


class TestDataContract(unittest.TestCase):
    """선발 이력·경기 키 및 관측값 검증."""

    def test_missing_pitch_count_is_not_zero(self) -> None:
        self.assertIsNone(
            require_count(
                None,
                "pitches",
                nullable=True,
            )
        )

    def test_boolean_count_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            require_count(True, "pitches")

    def test_negative_count_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            require_count(-1, "pitches")

    def test_zero_count_is_observed_value(self) -> None:
        self.assertEqual(
            require_count(0, "pitches"),
            0,
        )

    def test_home_away_mismatch_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            check_side(
                fixture(),
                "away",
                TEAM,
                CUTOFF,
            )

    def test_date_mismatch_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            check_side(
                fixture(),
                "home",
                TEAM,
                past_day(1),
            )

    def test_future_completed_game_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            completed_index(
                [
                    PlayedGame(
                        fixture(),
                        3,
                        2,
                    )
                ],
                CUTOFF,
            )

    def test_duplicate_completed_game_is_rejected(self) -> None:
        game = PlayedGame(
            fixture(day=past_day(1)),
            3,
            2,
        )

        with self.assertRaises(ValueError):
            completed_index(
                [game, game],
                CUTOFF,
            )

    def test_sql_history_excludes_same_day(self) -> None:
        db = sqlite3.connect(":memory:")
        self.addCleanup(db.close)

        db.execute(
            """
            CREATE TABLE starter_history (
                game_id TEXT,
                side TEXT,
                team_id INTEGER,
                date TEXT,
                player_id TEXT,
                name TEXT,
                evidence_count INTEGER
            )
            """
        )

        past = fixture(
            "past",
            past_day(5),
        )

        current = fixture("current")

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
                    past.day.isoformat(),
                    "A",
                    "A",
                    1,
                ),
                (
                    "current",
                    "home",
                    TEAM,
                    CUTOFF.isoformat(),
                    "B",
                    "B",
                    1,
                ),
            ],
        )

        result = read_starts(
            db,
            {
                "past": past,
                "current": current,
            },
            2025,
            CUTOFF,
        )

        self.assertEqual(set(result), {"A"})


class TestStarterRanking(unittest.TestCase):
    """후보 순위와 미래 자료 차단."""

    def test_baseline_and_proposed_order(self) -> None:
        result = rank_starters(
            sample_data(),
            TEAM,
            CUTOFF,
            CUTOFF,
        )

        self.assertEqual(
            result.baseline[0].player_id,
            "B",
        )

        self.assertEqual(
            result.proposed[0].player_id,
            "A",
        )

    def test_missing_workload_is_preserved(self) -> None:
        result = rank_starters(
            sample_data(),
            TEAM,
            CUTOFF,
            CUTOFF,
        )

        values = {
            candidate.player_id:
            candidate.observed_pitches_last_7d
            for candidate in result.proposed
        }

        self.assertEqual(values["A"], 90)
        self.assertIsNone(values["B"])

    def test_future_rows_do_not_change_ranking(self) -> None:
        data = sample_data()

        original = rank_starters(
            data,
            TEAM,
            CUTOFF,
            CUTOFF,
        )

        data.starts["A"] += (
            start("A", 0, OPPONENT),
            start("A", -1, OPPONENT),
        )

        data.identities["A"] += (
            Identity(
                CUTOFF,
                OPPONENT,
                "Future name",
            ),
        )

        data.appearances["A"] += (
            Appearance(
                "future",
                CUTOFF,
                OPPONENT,
                200,
            ),
        )

        actual = rank_starters(
            data,
            TEAM,
            CUTOFF,
            CUTOFF,
        )

        self.assertEqual(original, actual)

    def test_identity_conflict_excludes_candidate(self) -> None:
        data = sample_data()

        data.identities["A"] += (
            Identity(
                past_day(1),
                TEAM,
                "Other name",
            ),
        )

        result = rank_starters(
            data,
            TEAM,
            CUTOFF,
            CUTOFF,
        )

        self.assertNotIn(
            "A",
            [row.player_id for row in result.proposed],
        )

        self.assertEqual(
            result.exclusions["identity_name_conflict"],
            1,
        )

    def test_latest_other_team_excludes_candidate(self) -> None:
        data = sample_data()

        data.identities["A"] += (
            identity("A", 1, OPPONENT),
        )

        result = rank_starters(
            data,
            TEAM,
            CUTOFF,
            CUTOFF,
        )

        self.assertEqual(
            result.exclusions["latest_team_mismatch"],
            1,
        )

    def test_same_day_team_conflict_is_excluded(self) -> None:
        data = sample_data()

        data.identities["A"] += (
            identity("A", 1, TEAM),
            identity("A", 1, OPPONENT),
        )

        result = rank_starters(
            data,
            TEAM,
            CUTOFF,
            CUTOFF,
        )

        self.assertEqual(
            result.exclusions["latest_team_conflict"],
            1,
        )

    def test_missing_identity_is_excluded(self) -> None:
        data = sample_data()
        del data.identities["A"]

        result = rank_starters(
            data,
            TEAM,
            CUTOFF,
            CUTOFF,
        )

        self.assertEqual(
            result.exclusions["team_identity_not_observed"],
            1,
        )

    def test_28_day_boundary_is_inclusive(self) -> None:
        data = StarterData(
            starts={
                "A": (start("A", 28),),
                "B": (start("B", 29),),
            },
            identities={
                "A": (identity("A", 28),),
                "B": (identity("B", 29),),
            },
            appearances={},
        )

        result = rank_starters(
            data,
            TEAM,
            CUTOFF,
            CUTOFF,
        )

        self.assertEqual(
            [row.player_id for row in result.proposed],
            ["A"],
        )

    def test_equal_candidates_use_player_id(self) -> None:
        data = StarterData(
            starts={
                "B": (start("B", 5),),
                "A": (start("A", 5),),
            },
            identities={
                "B": (identity("B", 5),),
                "A": (identity("A", 5),),
            },
            appearances={},
        )

        result = rank_starters(
            data,
            TEAM,
            CUTOFF,
            CUTOFF,
        )

        self.assertEqual(
            [row.player_id for row in result.proposed],
            ["A", "B"],
        )

    def test_empty_history_returns_no_candidates(self) -> None:
        result = rank_starters(
            StarterData({}, {}, {}),
            TEAM,
            CUTOFF,
            CUTOFF,
        )

        self.assertEqual(result.proposed, ())
        self.assertEqual(result.observed_starts, 0)

    def test_unknown_announcement_is_not_official(self) -> None:
        completed = [
            PlayedGame(
                fixture(
                    "past",
                    past_day(5),
                ),
                3,
                2,
            ),
        ]

        data = StarterData(
            starts={"A": (start("A", 5),)},
            identities={"A": (identity("A", 5),)},
            appearances={},
        )

        result = starter_report(
            data,
            TEAM,
            fixture(),
            CUTOFF,
            completed,
            CUTOFF,
        )

        self.assertIsNone(
            result["official_announcement"]
        )

        self.assertEqual(
            result["official_announcement_status"],
            "publication_evidence_unavailable",
        )

    def test_non_next_fixture_is_unsupported(self) -> None:
        result = starter_report(
            sample_data(),
            TEAM,
            fixture(
                day=CUTOFF + timedelta(days=1)
            ),
            CUTOFF,
            [],
            CUTOFF,
        )

        self.assertEqual(
            result["reason"],
            "not_next_team_game_date",
        )

    def test_long_horizon_is_unsupported(self) -> None:
        target = CUTOFF + timedelta(days=8)

        result = starter_report(
            sample_data(),
            TEAM,
            fixture(day=target),
            CUTOFF,
            [],
            target,
        )

        self.assertEqual(
            result["reason"],
            "starter_forecast_horizon_exceeded",
        )


class TestTeamProfile(unittest.TestCase):
    """팀 기록, 홈·원정 및 무승부 분모 검증."""

    def test_zero_denominator(self) -> None:
        self.assertIsNone(ratio(0, 0))

    def test_empty_record(self) -> None:
        row = record_summary(Record())

        self.assertEqual(row["games"], 0)
        self.assertIsNone(row["win_rate"])
        self.assertIsNone(row["runs_for_per_game"])

    def test_win_rate_excludes_ties(self) -> None:
        record = Record()

        record.update(5, 3)
        record.update(1, 4)
        record.update(2, 2)

        row = record_summary(record)

        self.assertEqual(row["games"], 3)
        self.assertEqual(row["decisions"], 2)
        self.assertEqual(row["win_rate"], 0.5)

    def test_home_away_and_opponent_totals(self) -> None:
        games = [
            PlayedGame(
                fixture("a", past_day(4)),
                5,
                2,
            ),
            PlayedGame(
                fixture(
                    "b",
                    past_day(3),
                    OPPONENT,
                    TEAM,
                ),
                4,
                1,
            ),
            PlayedGame(
                fixture("c", past_day(2)),
                3,
                3,
            ),
        ]

        result = team_window(
            TEAM,
            (TEAM, OPPONENT),
            games,
            past_day(28),
            CUTOFF,
        )

        overall = cast(
            dict[str, object],
            result["overall"],
        )

        self.assertEqual(overall["games"], 3)
        self.assertEqual(overall["runs_for"], 9)
        self.assertEqual(overall["runs_against"], 9)
        self.assertEqual(overall["wins"], 1)
        self.assertEqual(overall["losses"], 1)
        self.assertEqual(overall["ties"], 1)

    def test_same_day_result_excluded(self) -> None:
        games = [
            PlayedGame(
                fixture("same-day"),
                30,
                0,
            ),
        ]

        result = team_window(
            TEAM,
            (TEAM, OPPONENT),
            games,
            past_day(28),
            CUTOFF,
        )

        overall = cast(
            dict[str, object],
            result["overall"],
        )

        self.assertEqual(overall["games"], 0)

    def test_duplicate_result_is_rejected(self) -> None:
        game = PlayedGame(
            fixture("duplicate", past_day(1)),
            3,
            2,
        )

        with self.assertRaises(ValueError):
            team_window(
                TEAM,
                (TEAM, OPPONENT),
                [game, game],
                past_day(28),
                CUTOFF,
            )


class TestScheduleDifficulty(unittest.TestCase):
    """팀 관점 확률과 잔여 일정 분모 검증."""

    def test_home_and_away_probability_conversion(self) -> None:
        game = fixture()
        prediction = forecast()

        home = team_probabilities(
            prediction,
            game,
            TEAM,
        )

        away = team_probabilities(
            prediction,
            game,
            OPPONENT,
        )

        self.assertAlmostEqual(home["win"], 0.6)
        self.assertAlmostEqual(away["win"], 0.3)
        self.assertAlmostEqual(home["loss"], 0.3)
        self.assertAlmostEqual(away["loss"], 0.6)

    def test_difficulty_complement(self) -> None:
        row = FixtureForecast(
            fixture(),
            forecast(),
        )

        home = game_difficulty(row, TEAM)
        away = game_difficulty(row, OPPONENT)

        self.assertAlmostEqual(
            cast(float, home["difficulty"]),
            0.35,
        )

        self.assertAlmostEqual(
            cast(float, away["difficulty"]),
            0.65,
        )

        self.assertAlmostEqual(
            cast(float, home["difficulty"])
            + cast(float, away["difficulty"]),
            1.0,
        )

    def test_draw_only_difficulty(self) -> None:
        self.assertEqual(
            difficulty_value({
                "win": 0.0,
                "tie": 1.0,
                "loss": 0.0,
            }),
            0.5,
        )

    def test_invalid_probability_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            team_probabilities(
                forecast(0.8, 0.2, 0.3),
                fixture(),
                TEAM,
            )

    def test_no_remaining_games(self) -> None:
        row = schedule_difficulty(
            TEAM,
            [],
            0,
            publication_verified=False,
        )

        self.assertEqual(
            row["status"],
            "no_remaining_games_under_assumption",
        )

        self.assertIsNone(row["difficulty"])

    def test_unassigned_schedule(self) -> None:
        row = schedule_difficulty(
            TEAM,
            [],
            4,
            publication_verified=False,
        )

        self.assertEqual(row["status"], "schedule_missing")
        self.assertEqual(row["unassigned_games"], 4)

    def test_missing_forecasts(self) -> None:
        row = schedule_difficulty(
            TEAM,
            [FixtureForecast(fixture(), None)],
            0,
            publication_verified=False,
        )

        self.assertEqual(row["status"], "no_valid_forecast")
        self.assertIsNone(row["difficulty"])

    def test_partial_forecast_denominator(self) -> None:
        forecasts = [
            FixtureForecast(
                fixture("a"),
                forecast(),
            ),
            FixtureForecast(
                fixture(
                    "b",
                    CUTOFF + timedelta(days=1),
                ),
                None,
            ),
        ]

        row = schedule_difficulty(
            TEAM,
            forecasts,
            2,
            publication_verified=False,
        )

        self.assertEqual(row["status"], "partial")
        self.assertEqual(row["denominator"], 1)
        self.assertEqual(row["expected_remaining_games"], 4)
        self.assertEqual(row["missing_forecasts"], 1)
        self.assertEqual(row["unassigned_games"], 2)

        self.assertAlmostEqual(
            cast(float, row["difficulty"]),
            0.35,
        )

    def test_equal_weight_average(self) -> None:
        rows = [
            FixtureForecast(
                fixture("a"),
                forecast(),
            ),
            FixtureForecast(
                fixture(
                    "b",
                    CUTOFF + timedelta(days=1),
                    OPPONENT,
                    TEAM,
                ),
                forecast(),
            ),
        ]

        result = schedule_difficulty(
            TEAM,
            rows,
            0,
            publication_verified=False,
        )

        self.assertAlmostEqual(
            cast(float, result["difficulty"]),
            0.5,
        )

        self.assertEqual(result["denominator"], 2)

    def test_duplicate_schedule_rejected(self) -> None:
        row = FixtureForecast(
            fixture(),
            forecast(),
        )

        with self.assertRaises(ValueError):
            schedule_difficulty(
                TEAM,
                [row, row],
                0,
                publication_verified=False,
            )

    def test_doubleheader_keeps_separate_games(self) -> None:
        rows = [
            FixtureForecast(
                fixture("dh1", number=1),
                forecast(),
            ),
            FixtureForecast(
                fixture("dh2", number=2),
                forecast(),
            ),
        ]

        result = schedule_difficulty(
            TEAM,
            rows,
            0,
            publication_verified=False,
        )

        self.assertEqual(result["denominator"], 2)
        self.assertEqual(result["valid_scheduled_games"], 2)


class TestEvaluation(unittest.TestCase):
    """선발 적중률·제공 범위 및 공통 분모 검증."""

    def test_coverage_and_accuracy_are_distinct(self) -> None:
        rows = [
            EvaluationSample(
                "a",
                TEAM,
                CUTOFF,
                Truth("A", None),
                ("A",),
                ("A",),
            ),
            EvaluationSample(
                "b",
                TEAM,
                CUTOFF,
                Truth("B", None),
                (),
                (),
            ),
        ]

        result = method_metrics(rows, "proposed")

        self.assertEqual(result["candidate_coverage"], 0.5)
        self.assertEqual(result["top1_accuracy"], 1.0)
        self.assertEqual(
            result["top1_all_identified_truth"],
            0.5,
        )

    def test_no_truth_has_null_accuracy(self) -> None:
        rows = [
            EvaluationSample(
                "a",
                TEAM,
                CUTOFF,
                Truth(None, "missing"),
                ("A",),
                ("A",),
            ),
        ]

        result = method_metrics(rows, "proposed")

        self.assertEqual(result["evaluable_samples"], 0)
        self.assertIsNone(result["top1_accuracy"])

    def test_top_k_differs_from_top1(self) -> None:
        rows = [
            EvaluationSample(
                "a",
                TEAM,
                CUTOFF,
                Truth("B", None),
                ("A", "B"),
                ("A", "B"),
            ),
        ]

        result = method_metrics(rows, "proposed")

        self.assertEqual(result["top1_accuracy"], 0.0)
        self.assertEqual(result["top_k_accuracy"], 1.0)

    def test_common_comparison_denominator(self) -> None:
        rows = [
            EvaluationSample(
                "a",
                TEAM,
                CUTOFF,
                Truth("A", None),
                ("B", "A"),
                ("A", "B"),
            ),
            EvaluationSample(
                "b",
                TEAM,
                CUTOFF,
                Truth("B", None),
                (),
                ("B",),
            ),
        ]

        result = compare_samples(rows)

        self.assertEqual(result["common_samples"], 1)

        baseline = cast(
            dict[str, object],
            result["common_baseline"],
        )

        proposed = cast(
            dict[str, object],
            result["common_proposed"],
        )

        self.assertEqual(baseline["top1_accuracy"], 0.0)
        self.assertEqual(proposed["top1_accuracy"], 1.0)

    def test_actual_truth_change_does_not_change_rank(self) -> None:
        ranking = rank_starters(
            sample_data(),
            TEAM,
            CUTOFF,
            CUTOFF,
        )

        players = tuple(
            candidate.player_id
            for candidate in ranking.proposed
        )

        first = EvaluationSample(
            "a",
            TEAM,
            CUTOFF,
            Truth("A", None),
            players,
            players,
        )

        second = replace(
            first,
            truth=Truth("B", None),
        )

        self.assertEqual(
            first.proposed,
            second.proposed,
        )

        self.assertNotEqual(
            method_metrics([first], "proposed")["top1_hits"],
            method_metrics([second], "proposed")["top1_hits"],
        )

    def test_truth_with_missing_identifier(self) -> None:
        db = sqlite3.connect(":memory:")
        self.addCleanup(db.close)

        db.execute(
            """
            CREATE TABLE starter_actual (
                game_id TEXT,
                side TEXT,
                team_id INTEGER,
                date TEXT,
                player_id TEXT,
                name TEXT,
                evidence_count INTEGER
            )
            """
        )

        game = PlayedGame(
            fixture("a", past_day(1)),
            2,
            1,
        )

        db.execute(
            """
            INSERT INTO starter_actual
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "a",
                "home",
                TEAM,
                past_day(1).isoformat(),
                None,
                "A",
                1,
            ),
        )

        result = read_truth(
            db,
            [game],
            CUTOFF,
        )

        self.assertIsNone(
            result["a", TEAM].player_id
        )

        self.assertEqual(
            result["a", TEAM].reason,
            "truth_player_id_unresolved",
        )


class TestIntegration(unittest.TestCase):
    """새 진입점의 기존 M1 연결 및 출력 보호."""

    def make_scenario(self) -> Scenario:
        """실제 시즌 규칙을 사용하는 최소 합성 시나리오."""

        rules = get_rules(2025)

        home, away = rules.teams[:2]

        completed = (
            PlayedGame(
                fixture(
                    "past",
                    past_day(1),
                    home,
                    away,
                ),
                4,
                2,
            ),
        )

        remaining = (
            fixture(
                "future",
                CUTOFF,
                home,
                away,
            ),
        )

        return Scenario(
            rules=rules,
            cutoff=CUTOFF,
            completed=completed,
            remaining=remaining,
            model=cast(m1.Model, {}),
            standard_db=Path("standard.sqlite3"),
            model_db=Path("model.sqlite3"),
            metadata={},
        )

    def test_game_analysis_reuses_existing_m1(self) -> None:
        scenario = self.make_scenario()

        with patch(
            "src.simulation.support_analysis.analysis.m1.predict",
            return_value=forecast(),
        ) as prediction:
            result = analyze(
                scenario,
                StarterData({}, {}, {}),
                "future",
            )

        self.assertEqual(result["scope"], "game")
        self.assertEqual(result["new_primary_models"], 0)
        self.assertFalse(result["final_test_used"])

        prediction.assert_called_once()

        inputs = prediction.call_args.args[1]

        self.assertEqual(
            set(inputs),
            set(m1.INPUT_COLUMNS),
        )

        self.assertEqual(
            inputs["home_season_games"],
            1,
        )

        self.assertEqual(
            inputs["home_season_rf"],
            4,
        )

    def test_unknown_game_is_rejected(self) -> None:
        scenario = self.make_scenario()

        with patch(
            "src.simulation.support_analysis.analysis.m1.predict",
            return_value=forecast(),
        ):
            with self.assertRaises(ValueError):
                analyze(
                    scenario,
                    StarterData({}, {}, {}),
                    "unknown",
                )

    def test_cli_evaluate_arguments(self) -> None:
        args = build_parser().parse_args([
            "--run-id",
            "a" * 32,
            "--asset",
            "model.json",
            "evaluate",
        ])

        self.assertEqual(args.command, "evaluate")

    def test_cli_game_requires_season_and_game_id(self) -> None:
        args = build_parser().parse_args([
            "--run-id",
            "a" * 32,
            "--asset",
            "model.json",
            "game",
            "--season",
            "2025",
            "--as-of",
            "2025-09-01",
            "--game-id",
            "sample",
        ])

        self.assertEqual(args.season, 2025)
        self.assertEqual(args.game_id, "sample")

    def test_result_cannot_overwrite_existing_file(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)

            settings = cast(
                Settings,
                SimpleNamespace(results_dir=root),
            )

            target = root / "result.json"

            save_result(
                settings,
                target,
                {"value": 1},
            )

            with self.assertRaises(FileExistsError):
                save_result(
                    settings,
                    target,
                    {"value": 2},
                )

            self.assertIn(
                '"value": 1',
                target.read_text(encoding="utf-8"),
            )

    def test_output_outside_results_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)

            settings = cast(
                Settings,
                SimpleNamespace(
                    results_dir=root / "results"
                ),
            )

            with self.assertRaises(ValueError):
                save_result(
                    settings,
                    root / "outside.json",
                    {"value": 1},
                )


if __name__ == "__main__":
    unittest.main()
