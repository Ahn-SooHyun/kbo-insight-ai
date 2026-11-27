"""정규시즌 순위 규칙, 자료 시점, 반복 계산 및 출력 계약 테스트."""

from datetime import date, timedelta
from fractions import Fraction
from itertools import combinations
import json
from pathlib import Path
from random import Random
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from src.models.game_outcome import model as m1
from src.simulation.regular_season import SNAPSHOT_SCHEMA
from src.simulation.regular_season.data import (
    fixture_from_values,
    load_schedule_snapshot,
    parse_cutoff,
    read_archive_schedule,
    read_completed,
)
from src.simulation.regular_season.engine import (
    JointSampler,
    M1Sampler,
    build_state,
    diagnostic_metrics,
    distribution,
    inputs_for,
    simulate,
    summarize,
)
from src.simulation.regular_season.rules import (
    KBO_2025,
    Fixture,
    PlayedGame,
    Record,
    SeasonRules,
    get_rules,
    rate_bounds,
    resolve_standings,
    validate_schedule,
)


START = date(2025, 8, 1)


def game(
    index: int,
    home: int,
    away: int,
    home_runs: int,
    away_runs: int,
) -> PlayedGame:
    """손으로 검산 가능한 작은 가상 경기."""

    return PlayedGame(
        Fixture(
            f"synthetic-{index}",
            START + timedelta(days=index),
            home,
            away,
            0,
        ),
        home_runs,
        away_runs,
    )


def synthetic_rules(
    teams: tuple[int, ...] = (1, 2, 3),
    slots: int = 2,
    games_per_pair: int = 1,
    previous_rank: dict[int, int] | None = None,
) -> SeasonRules:

    return SeasonRules(
        season=2025,
        teams=teams,
        games_per_pair=games_per_pair,
        playoff_slots=slots,
        previous_rank=(
            previous_rank
            if previous_rank is not None
            else {}
        ),
    )


class TestSeasonRules(unittest.TestCase):

    def test_official_2025_rules(self) -> None:
        self.assertEqual(
            len(KBO_2025.teams),
            10,
        )

        self.assertEqual(
            KBO_2025.games_per_team,
            144,
        )

        self.assertEqual(
            KBO_2025.total_games,
            720,
        )

        self.assertEqual(
            KBO_2025.playoff_slots,
            5,
        )

        with self.assertRaises(ValueError):
            get_rules(2026)

    def test_win_rate_excludes_ties(self) -> None:
        record = Record(
            wins=3,
            losses=2,
            ties=4,
        )

        self.assertEqual(
            record.win_rate,
            Fraction(3, 5),
        )

        self.assertEqual(record.games, 9)

        self.assertIsNone(
            Record(ties=3).win_rate
        )

    def test_rate_bounds(self) -> None:
        record = Record(
            wins=3,
            losses=2,
            ties=1,
        )

        self.assertEqual(
            rate_bounds(record, 4),
            (
                Fraction(3, 9),
                Fraction(7, 9),
            ),
        )

    def test_unique_final_ranking(self) -> None:
        rules = synthetic_rules()

        state = build_state(
            rules,
            [
                game(1, 1, 2, 3, 1),
                game(2, 1, 3, 2, 0),
                game(3, 2, 3, 4, 2),
            ],
        )

        result = resolve_standings(state)

        self.assertEqual(
            [result[team].rank for team in rules.teams],
            [1, 2, 3],
        )

        self.assertTrue(result[1].first)
        self.assertTrue(result[2].qualifies)
        self.assertFalse(result[3].qualifies)

    def test_first_place_tiebreak_required(self) -> None:
        rules = synthetic_rules()

        state = build_state(
            rules,
            [
                game(1, 1, 2, 1, 1),
                game(2, 1, 3, 2, 0),
                game(3, 2, 3, 3, 0),
            ],
        )

        result = resolve_standings(state)

        for team in (1, 2):
            self.assertIsNone(result[team].rank)
            self.assertIsNone(result[team].first)
            self.assertTrue(result[team].qualifies)

            self.assertEqual(
                result[team].reason,
                "separate_tiebreak_game_required",
            )

        self.assertFalse(result[3].first)
        self.assertFalse(result[3].qualifies)

    def test_qualification_boundary_tiebreak(self) -> None:
        rules = synthetic_rules()

        state = build_state(
            rules,
            [
                game(1, 1, 2, 2, 0),
                game(2, 1, 3, 2, 0),
                game(3, 2, 3, 1, 1),
            ],
        )

        result = resolve_standings(state)

        self.assertEqual(result[1].rank, 1)
        self.assertTrue(result[1].qualifies)

        for team in (2, 3):
            self.assertIsNone(result[team].rank)
            self.assertIsNone(result[team].qualifies)
            self.assertFalse(result[team].first)

    def test_multi_team_tie_resolved_by_runs(self) -> None:
        rules = synthetic_rules()

        state = build_state(
            rules,
            [
                game(1, 1, 2, 1, 0),
                game(2, 2, 3, 2, 0),
                game(3, 3, 1, 3, 0),
            ],
        )

        result = resolve_standings(state)

        self.assertEqual(
            [result[team].rank for team in rules.teams],
            [3, 2, 1],
        )

    def test_previous_rank_breaks_remaining_tie(self) -> None:
        rules = synthetic_rules(
            previous_rank={
                1: 2,
                2: 3,
                3: 1,
            },
        )

        state = build_state(
            rules,
            [
                game(1, 1, 2, 1, 0),
                game(2, 2, 3, 1, 0),
                game(3, 3, 1, 1, 0),
            ],
        )

        result = resolve_standings(state)

        self.assertEqual(
            [result[team].rank for team in rules.teams],
            [2, 3, 1],
        )

    def test_missing_previous_rank_not_invented(self) -> None:
        rules = synthetic_rules()

        state = build_state(
            rules,
            [
                game(1, 1, 2, 1, 0),
                game(2, 2, 3, 1, 0),
                game(3, 3, 1, 1, 0),
            ],
        )

        result = resolve_standings(state)

        for standing in result.values():
            self.assertIsNone(standing.rank)

            self.assertEqual(
                standing.reason,
                "previous_rank_unavailable",
            )

    def test_lower_ranks_can_be_shared(self) -> None:
        rules = synthetic_rules(
            teams=(1, 2, 3, 4),
        )

        state = build_state(
            rules,
            [
                game(1, 1, 2, 1, 0),
                game(2, 1, 3, 1, 0),
                game(3, 1, 4, 1, 0),
                game(4, 2, 3, 1, 0),
                game(5, 2, 4, 1, 0),
                game(6, 3, 4, 0, 0),
            ],
        )

        result = resolve_standings(state)

        self.assertEqual(
            [result[team].rank for team in rules.teams],
            [1, 2, 3, 3],
        )

        self.assertFalse(result[3].qualifies)
        self.assertFalse(result[4].qualifies)

    def test_all_ties_have_undefined_win_rate(self) -> None:
        rules = synthetic_rules()

        state = build_state(
            rules,
            [
                game(1, 1, 2, 0, 0),
                game(2, 1, 3, 0, 0),
                game(3, 2, 3, 0, 0),
            ],
        )

        result = resolve_standings(state)

        for standing in result.values():
            self.assertIsNone(standing.rank)

    def test_incomplete_season_preserves_clinched_first(self) -> None:
        rules = synthetic_rules(
            teams=(1, 2, 3, 4),
        )

        completed = [
            game(1, 1, 2, 1, 0),
            game(2, 1, 3, 1, 0),
            game(3, 1, 4, 1, 0),
        ]

        state = build_state(rules, completed)

        result = resolve_standings(state)

        self.assertEqual(result[1].rank, 1)
        self.assertTrue(result[1].qualifies)

        self.assertFalse(result[2].first)
        self.assertFalse(result[3].first)
        self.assertFalse(result[4].first)


class TestScheduleAndInputs(unittest.TestCase):

    def test_duplicate_schedule_rejected(self) -> None:
        rules = synthetic_rules()

        fixture = game(1, 1, 2, 1, 0).fixture

        with self.assertRaises(ValueError):
            validate_schedule(
                rules,
                [fixture, fixture],
            )

    def test_same_day_conflict_rejected(self) -> None:
        rules = synthetic_rules()

        fixtures = [
            Fixture("a", START, 1, 2, 0),
            Fixture("b", START, 1, 3, 0),
        ]

        with self.assertRaises(ValueError):
            validate_schedule(
                rules,
                fixtures,
            )

    def test_doubleheader_and_missing_schedule(self) -> None:
        rules = synthetic_rules(
            games_per_pair=2,
        )

        fixtures = [
            Fixture("a", START, 1, 2, 1),
            Fixture("b", START, 1, 2, 2),
        ]

        missing = validate_schedule(
            rules,
            fixtures,
        )

        self.assertEqual(
            sum(
                item["missing_games"]
                for item in missing
            ),
            4,
        )

    def test_pair_game_limit(self) -> None:
        rules = synthetic_rules()

        fixtures = [
            game(1, 1, 2, 1, 0).fixture,
            game(2, 1, 2, 1, 0).fixture,
        ]

        with self.assertRaises(ValueError):
            validate_schedule(
                rules,
                fixtures,
            )

    def test_real_game_key_checks(self) -> None:
        fixture = fixture_from_values(
            "20250901LGLT0",
            "2025-09-01",
            0,
            2007,
            2003,
        )

        self.assertEqual(fixture.home, 2007)
        self.assertEqual(fixture.away, 2003)

        with self.assertRaises(ValueError):
            fixture_from_values(
                "20250901LGLT0",
                "2025-09-01",
                0,
                2003,
                2007,
            )

        with self.assertRaises(ValueError):
            fixture_from_values(
                "20250901LGLT1",
                "2025-09-01",
                2,
                2007,
                2003,
            )

    def test_cutoff_date_validation(self) -> None:
        self.assertEqual(
            parse_cutoff("2025-09-01", 2025),
            date(2025, 9, 1),
        )

        for invalid in (
            "2025-9-1",
            "2025-09-01T12:00:00",
            "2026-09-01",
        ):
            with self.assertRaises(ValueError):
                parse_cutoff(invalid, 2025)

    def test_recent_window_is_28_days(self) -> None:
        rules = synthetic_rules(
            games_per_pair=4,
        )

        cutoff = date(2025, 9, 1)

        completed = [
            PlayedGame(
                Fixture(
                    "old",
                    cutoff - timedelta(days=29),
                    1,
                    2,
                    0,
                ),
                5,
                1,
            ),
            PlayedGame(
                Fixture(
                    "boundary",
                    cutoff - timedelta(days=28),
                    1,
                    2,
                    0,
                ),
                2,
                3,
            ),
            PlayedGame(
                Fixture(
                    "recent",
                    cutoff - timedelta(days=1),
                    1,
                    2,
                    0,
                ),
                4,
                2,
            ),
        ]

        state = build_state(
            rules,
            completed,
        )

        inputs = inputs_for(
            state,
            Fixture(
                "next",
                cutoff,
                1,
                2,
                0,
            ),
        )

        self.assertEqual(
            inputs["home_season_games"],
            3,
        )

        self.assertEqual(
            inputs["home_recent_games"],
            2,
        )

        self.assertEqual(
            inputs["home_recent_rf"],
            6,
        )

        self.assertEqual(
            inputs["home_recent_ra"],
            5,
        )

    def test_same_day_history_is_rejected(self) -> None:
        rules = synthetic_rules()

        completed = game(1, 1, 2, 2, 0)
        state = build_state(rules, [completed])

        with self.assertRaises(ValueError):
            inputs_for(
                state,
                Fixture(
                    "same-day",
                    completed.fixture.day,
                    1,
                    2,
                    2,
                ),
            )

    def test_sql_future_scores_never_loaded(self) -> None:
        standard = sqlite3.connect(":memory:")
        model_data = sqlite3.connect(":memory:")

        self.addCleanup(standard.close)
        self.addCleanup(model_data.close)

        standard.executescript(
            """
            CREATE TABLE eligible_games (
                game_id TEXT,
                source_key TEXT,
                date TEXT,
                game_number INTEGER,
                home_team_id INTEGER,
                away_team_id INTEGER,
                home_score,
                away_score,
                winner TEXT,
                season INTEGER,
                competition TEXT
            );

            INSERT INTO eligible_games VALUES (
                '20250831LGLT0',
                '20250831LGLT0',
                '2025-08-31',
                0, 2007, 2003, 3, 1,
                'home', 2025, 'regular'
            );

            INSERT INTO eligible_games VALUES (
                '20250902LGLT0',
                '20250902LGLT0',
                '2025-09-02',
                0, 2007, 2003,
                'future_score_must_not_be_read',
                'future_score_must_not_be_read',
                'future_result',
                2025, 'regular'
            );
            """
        )

        model_data.executescript(
            """
            CREATE TABLE m1_index (
                game_id TEXT,
                sample_id TEXT,
                sample_date TEXT,
                status TEXT
            );

            CREATE TABLE m1_inputs (
                sample_id TEXT,
                home_team_id INTEGER,
                away_team_id INTEGER,
                season INTEGER,
                competition TEXT
            );

            CREATE TABLE m1_targets (
                sample_id TEXT,
                target_home_runs INTEGER,
                target_away_runs INTEGER,
                target_result TEXT
            );

            INSERT INTO m1_index VALUES (
                '20250831LGLT0',
                '20250831LGLT0',
                '2025-08-31',
                'eligible'
            );

            INSERT INTO m1_inputs VALUES (
                '20250831LGLT0',
                2007, 2003, 2025, 'regular'
            );

            INSERT INTO m1_targets VALUES (
                '20250831LGLT0',
                3, 1, 'home'
            );
            """
        )

        cutoff = date(2025, 9, 1)

        completed = read_completed(
            standard,
            model_data,
            2025,
            cutoff,
        )

        remaining = read_archive_schedule(
            standard,
            2025,
            cutoff,
        )

        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0].home_runs, 3)
        self.assertEqual(len(remaining), 1)

        self.assertEqual(
            remaining[0].game_id,
            "20250902LGLT0",
        )

        self.assertFalse(
            hasattr(remaining[0], "home_runs")
        )

    def test_snapshot_rescheduled_fixture(self) -> None:
        payload = {
            "schema": SNAPSHOT_SCHEMA,
            "season": 2025,
            "published_at": "2025-08-31T20:00:00+09:00",
            "source_url": "https://example.invalid/schedule",
            "version": "test-only",
            "fixtures": [
                {
                    "game_id": "20250831LGLT0",
                    "date": "2025-08-31",
                    "home_team_id": 2007,
                    "away_team_id": 2003,
                    "game_number": 0,
                    "status": "postponed",
                },
                {
                    "game_id": "20250902LGLT0",
                    "date": "2025-09-02",
                    "home_team_id": 2007,
                    "away_team_id": 2003,
                    "game_number": 0,
                    "status": "scheduled",
                    "rescheduled_from": "20250831LGLT0",
                },
            ],
        }

        with TemporaryDirectory() as directory:
            path = Path(directory) / "schedule.json"

            path.write_text(
                json.dumps(payload),
                encoding="utf-8",
            )

            fixtures, metadata = load_schedule_snapshot(
                path,
                2025,
                date(2025, 9, 1),
            )

            self.assertEqual(len(fixtures), 1)

            self.assertEqual(
                fixtures[0].game_id,
                "20250902LGLT0",
            )

            self.assertFalse(
                metadata["publication_independently_verified"]
            )

            payload["published_at"] = (
                "2025-09-02T00:00:00+09:00"
            )

            path.write_text(
                json.dumps(payload),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                load_schedule_snapshot(
                    path,
                    2025,
                    date(2025, 9, 1),
                )

    def test_snapshot_rejects_future_result_fields(self) -> None:
        payload = {
            "schema": SNAPSHOT_SCHEMA,
            "season": 2025,
            "published_at": "2025-08-31T20:00:00+09:00",
            "source_url": "https://example.invalid/schedule",
            "version": "test-only",
            "fixtures": [
                {
                    "game_id": "20250902LGLT0",
                    "date": "2025-09-02",
                    "home_team_id": 2007,
                    "away_team_id": 2003,
                    "game_number": 0,
                    "status": "scheduled",
                    "home_runs": 5,
                },
            ],
        }

        with TemporaryDirectory() as directory:
            path = Path(directory) / "schedule.json"

            path.write_text(
                json.dumps(payload),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                load_schedule_snapshot(
                    path,
                    2025,
                    date(2025, 9, 1),
                )


class TestSimulation(unittest.TestCase):

    def test_joint_sampling_uses_score_pairs(self) -> None:
        forecast = m1.combine(
            (0.5, 0.5),
            (0.5, 0.5),
            0.5,
            0.0,
            0.0,
        )

        sampler = JointSampler(forecast)
        random = Random(42)

        scores = {
            sampler.sample(random)
            for _ in range(200)
        }

        self.assertEqual(
            scores,
            {
                (0, 0),
                (0, 1),
                (1, 0),
                (1, 1),
            },
        )

    def test_invalid_joint_probability_rejected(self) -> None:
        forecast = m1.combine(
            (0.5, 0.5),
            (0.5, 0.5),
            0.5,
            0.0,
            0.0,
        )

        forecast["result_probabilities"]["home"] += 0.1
        forecast["result_probabilities"]["away"] -= 0.1

        with self.assertRaises(ValueError):
            JointSampler(forecast)

    def test_m1_sampler_cache(self) -> None:
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

        rules = synthetic_rules()
        state = build_state(rules, [])

        inputs = inputs_for(
            state,
            Fixture(
                "first",
                START,
                1,
                2,
                0,
            ),
        )

        sampler = M1Sampler(model)
        random = Random(42)

        sampler(inputs, random)
        sampler(inputs, random)

        self.assertEqual(
            len(sampler.cache),
            1,
        )

    def test_future_actual_result_is_rejected(self) -> None:
        rules = synthetic_rules()

        with self.assertRaises(ValueError):
            simulate(
                rules,
                START,
                [game(1, 1, 2, 1, 0)],
                [],
                lambda inputs, random: (1, 0),
                1,
                42,
            )

    def test_daily_updates_exclude_same_day_doubleheader(self) -> None:
        rules = synthetic_rules(
            teams=(1, 2),
            slots=1,
            games_per_pair=4,
        )

        cutoff = date(2025, 9, 1)

        completed = [
            PlayedGame(
                Fixture(
                    "past",
                    cutoff - timedelta(days=1),
                    1,
                    2,
                    0,
                ),
                2,
                0,
            )
        ]

        remaining = [
            Fixture("dh1", cutoff, 1, 2, 1),
            Fixture("dh2", cutoff, 1, 2, 2),
            Fixture(
                "tomorrow",
                cutoff + timedelta(days=1),
                1,
                2,
                0,
            ),
        ]

        observed_counts = []

        def sampler(inputs, random):
            observed_counts.append(
                (
                    inputs["home_season_games"],
                    inputs["home_season_rf"],
                )
            )

            return 1, 0

        rows = simulate(
            rules,
            cutoff,
            completed,
            remaining,
            sampler,
            1,
            42,
        )

        self.assertEqual(
            observed_counts,
            [
                (1, 2),
                (1, 2),
                (3, 4),
            ],
        )

        self.assertEqual(
            rows[0]["teams"]["1"]["final_wins"],
            4,
        )

    def test_same_seed_and_iteration_prefix(self) -> None:
        rules = synthetic_rules(
            teams=(1, 2, 3, 4),
        )

        fixtures = [
            Fixture(
                f"game-{index}",
                START + timedelta(days=index),
                home,
                away,
                0,
            )
            for index, (home, away) in enumerate(
                combinations(rules.teams, 2)
            )
        ]

        def sampler(inputs, random):
            return (
                random.randrange(5),
                random.randrange(5),
            )

        first = simulate(
            rules, START, [], fixtures,
            sampler, 12, 42,
        )

        second = simulate(
            rules, START, [], fixtures,
            sampler, 12, 42,
        )

        larger = simulate(
            rules, START, [], fixtures,
            sampler, 20, 42,
        )

        different_seed = simulate(
            rules, START, [], fixtures,
            sampler, 12, 43,
        )

        self.assertEqual(first, second)
        self.assertEqual(first, larger[:12])
        self.assertNotEqual(first, different_seed)

        self.assertEqual(
            summarize(rules, first),
            summarize(rules, second),
        )

    def test_ended_season_never_calls_sampler(self) -> None:
        rules = synthetic_rules()

        completed = [
            game(1, 1, 2, 1, 0),
            game(2, 1, 3, 1, 0),
            game(3, 2, 3, 1, 0),
        ]

        def unexpected_sampler(inputs, random):
            raise AssertionError(
                "완료 시즌에서 모델을 호출하면 안 됩니다."
            )

        rows = simulate(
            rules,
            START + timedelta(days=10),
            completed,
            [],
            unexpected_sampler,
            3,
            42,
        )

        for row in rows:
            self.assertEqual(
                row["postseason_seeds"],
                [1, 2],
            )

        result = summarize(rules, rows)

        self.assertEqual(
            result["league"]["first_joint_probabilities"]["1"],
            1.0,
        )

    def test_unresolved_first_retains_qualification(self) -> None:
        rules = synthetic_rules()

        completed = [
            game(1, 1, 2, 1, 1),
            game(2, 1, 3, 1, 0),
            game(3, 2, 3, 1, 0),
        ]

        rows = simulate(
            rules,
            START + timedelta(days=10),
            completed,
            [],
            lambda inputs, random: (0, 0),
            2,
            42,
        )

        summary = summarize(rules, rows)

        self.assertEqual(
            rows[0]["qualifiers"],
            [1, 2],
        )

        self.assertIsNone(
            rows[0]["postseason_seeds"]
        )

        first = summary["teams"]["1"]["regular_season_first"]

        self.assertEqual(first["valid"], 0)
        self.assertEqual(first["unresolved"], 2)

        self.assertIsNone(first["expected"])

        qualification = (
            summary["teams"]["1"]["postseason_qualification"]
        )

        self.assertEqual(
            qualification["expected"],
            1.0,
        )

    def test_incomplete_schedule_has_no_false_final_wins(self) -> None:
        rules = synthetic_rules(
            teams=(1, 2, 3, 4),
        )

        completed = [
            game(1, 1, 2, 1, 0),
            game(2, 1, 3, 1, 0),
            game(3, 1, 4, 1, 0),
        ]

        rows = simulate(
            rules,
            START + timedelta(days=10),
            completed,
            [],
            lambda inputs, random: (0, 0),
            2,
            42,
        )

        result = summarize(rules, rows)

        self.assertEqual(
            result["teams"]["1"]["final_wins"]["expected"],
            3.0,
        )

        self.assertIsNone(
            result["teams"]["2"]["final_wins"]["expected"]
        )

        self.assertEqual(
            result["teams"]["1"]["regular_season_first"]["expected"],
            1.0,
        )

    def test_distribution_valid_denominator(self) -> None:
        result = distribution(
            [1, None, 0, 1],
            [None, "tiebreak", None, None],
            (0, 1),
        )

        self.assertEqual(result["requested"], 4)
        self.assertEqual(result["valid"], 3)
        self.assertEqual(result["unresolved"], 1)

        self.assertAlmostEqual(
            result["probabilities"]["1"],
            2 / 3,
        )

        self.assertEqual(
            result["unresolved_reasons"],
            {"tiebreak": 1},
        )

    def test_zero_valid_denominator(self) -> None:
        result = distribution(
            [None, None],
            ["missing", "missing"],
            (0, 1),
        )

        self.assertEqual(
            result["status"],
            "unsupported",
        )

        self.assertEqual(result["valid"], 0)
        self.assertIsNone(result["expected"])

        self.assertEqual(
            result["probabilities"],
            {
                "0": None,
                "1": None,
            },
        )

    def test_diagnostic_metrics_for_identical_truth(self) -> None:
        rules = synthetic_rules()

        completed = [
            game(1, 1, 2, 1, 0),
            game(2, 1, 3, 1, 0),
            game(3, 2, 3, 1, 0),
        ]

        rows = simulate(
            rules,
            START + timedelta(days=10),
            completed,
            [],
            lambda inputs, random: (0, 0),
            2,
            42,
        )

        summary = summarize(
            rules,
            rows,
        )

        metrics = diagnostic_metrics(
            rules,
            summary,
            completed,
        )

        self.assertEqual(
            metrics["wins_mae"]["value"],
            0.0,
        )

        self.assertEqual(
            metrics["rank_mae"]["value"],
            0.0,
        )

        self.assertEqual(
            metrics["first_brier"]["value"],
            0.0,
        )

        self.assertEqual(
            metrics["qualification_brier"]["value"],
            0.0,
        )

        self.assertFalse(
            metrics["formal_backtest"]
        )


if __name__ == "__main__":
    unittest.main()