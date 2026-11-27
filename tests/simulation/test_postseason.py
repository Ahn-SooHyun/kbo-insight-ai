"""포스트시즌 규칙 및 공동 확률 계산 테스트."""

from datetime import date, timedelta
from math import fsum
from random import Random
import unittest
from unittest.mock import Mock

from src.simulation.postseason import MAX_SERIES_GAMES
from src.simulation.postseason.engine import (
    ArchiveState,
    SeedInput,
    probability_summary,
    sample_outcome,
    simulate,
    simulate_round,
    summarize,
    validate_probabilities,
)
from src.simulation.postseason.rules import (
    Game,
    Outcome,
    SeriesState,
    StageRule,
    get_rules,
    replay_series,
)


TEAMS = tuple(range(1, 11))
WC, SEMI, PO, KS = get_rules(2025)


def observed(
    number: int,
    home: int,
    away: int,
    outcome: Outcome,
) -> Game:
    """합성 완료 경기 레코드를 생성한다."""
    scores = {
        "home": (1, 0),
        "away": (0, 1),
        "tie": (1, 1),
    }

    home_score, away_score = scores[outcome]

    return Game(
        number=number,
        home=home,
        away=away,
        outcome=outcome,
        game_id=f"test-game-{number}",
        day=(
            date(2025, 10, 1)
            + timedelta(days=number)
        ).isoformat(),
        home_score=home_score,
        away_score=away_score,
    )


def play(
    state: SeriesState,
    outcome: Outcome,
) -> None:
    """다음 공식 홈 배정에 맞는 합성 경기를 적용한다."""
    home = state.next_home()

    if home is None:
        raise AssertionError("다음 경기 미확정")

    away = (
        state.lower
        if home == state.higher
        else state.higher
    )

    state.apply(
        Game(
            len(state.games) + 1,
            home,
            away,
            outcome,
        )
    )


def draw(
    iteration_id: int,
    seeds: tuple[int, ...] | None = (
        1, 2, 3, 4, 5
    ),
) -> SeedInput:
    """합성 정규시즌 반복 입력을 생성한다."""
    qualification = {
        team: (
            team in seeds
            if seeds is not None
            else False
            if team == 10
            else None
        )
        for team in TEAMS
    }

    return SeedInput(
        iteration_id=iteration_id,
        seeds=seeds,
        qualifiers=(
            tuple(sorted(seeds))
            if seeds is not None
            else None
        ),
        qualification=qualification,
        regular_first=(
            seeds[0] if seeds is not None else 1
        ),
        reasons=(
            ("separate_tiebreak_game_required",)
            if seeds is None
            else ()
        ),
    )


def home_sampler(
    home: int,
    away: int,
    rng: Random,
) -> Outcome:
    """홈 승리만 반환하는 결정론적 테스트 표본기."""
    return "home"


def random_sampler(
    home: int,
    away: int,
    rng: Random,
) -> Outcome:
    """재현성 검증을 위한 합성 범주형 표본기."""
    return sample_outcome(
        {
            "home": 0.55,
            "tie": 0.02,
            "away": 0.43,
        },
        rng,
    )


class PostseasonRuleTests(unittest.TestCase):
    """공식 시즌별 규칙과 예외 경계를 검사한다."""

    def test_unsupported_season(self) -> None:
        with self.assertRaises(ValueError):
            get_rules(2024)

    def test_wildcard_higher_win(self) -> None:
        state = SeriesState(WC, 4, 5)

        play(state, "home")

        self.assertEqual(state.winner, 4)
        self.assertEqual(state.higher_wins, 1)
        self.assertIsNone(state.next_home())

    def test_wildcard_tie_advantage(self) -> None:
        state = SeriesState(WC, 4, 5)

        play(state, "tie")

        self.assertEqual(state.winner, 4)
        self.assertEqual(state.higher_wins, 0)
        self.assertEqual(state.lower_wins, 0)
        self.assertEqual(state.ties, 1)

    def test_wildcard_lower_two_wins(self) -> None:
        state = SeriesState(WC, 4, 5)

        play(state, "away")

        self.assertIsNone(state.winner)
        self.assertEqual(state.next_home(), 4)

        play(state, "away")

        self.assertEqual(state.winner, 5)

    def test_wildcard_loss_then_tie(self) -> None:
        state = SeriesState(WC, 4, 5)

        play(state, "away")
        play(state, "tie")

        self.assertEqual(state.winner, 4)
        self.assertEqual(state.lower_wins, 1)

    def test_semi_and_playoff_home_order(self) -> None:
        for rule in (SEMI, PO):
            with self.subTest(stage=rule.stage):
                state = SeriesState(rule, 1, 2)

                homes = []

                for _ in range(5):
                    homes.append(state.next_home())
                    play(state, "home")

                self.assertEqual(homes, [1, 1, 2, 2, 1])
                self.assertEqual(state.winner, 1)

    def test_korean_series_home_order(self) -> None:
        state = SeriesState(KS, 1, 2)

        homes = []

        for _ in range(7):
            homes.append(state.next_home())
            play(state, "home")

        self.assertEqual(
            homes,
            [1, 1, 2, 2, 2, 1, 1],
        )

        self.assertEqual(state.winner, 1)

    def test_early_series_completion(self) -> None:
        state = SeriesState(SEMI, 1, 2)

        play(state, "home")
        play(state, "home")
        play(state, "away")

        self.assertEqual(len(state.games), 3)
        self.assertEqual(state.winner, 1)
        self.assertIsNone(state.next_home())

    def test_tie_does_not_add_win(self) -> None:
        state = SeriesState(PO, 1, 2)

        play(state, "tie")

        self.assertEqual(state.higher_wins, 0)
        self.assertEqual(state.lower_wins, 0)
        self.assertEqual(state.next_home(), 1)

    def test_extra_game_uses_tied_home(self) -> None:
        state = SeriesState(SEMI, 1, 2)

        for outcome in (
            "tie", "home", "home", "away", "away"
        ):
            play(state, outcome)

        self.assertIsNone(state.winner)
        self.assertEqual(state.higher_wins, 2)
        self.assertEqual(state.lower_wins, 2)
        self.assertEqual(state.next_home(), 1)

        play(state, "home")

        self.assertEqual(state.winner, 1)
        self.assertEqual(len(state.games), 6)

    def test_multiple_tie_venues_remain_unresolved(self) -> None:
        state = SeriesState(SEMI, 1, 2)

        for outcome in (
            "tie", "home", "tie", "home", "home"
        ):
            play(state, outcome)

        self.assertIsNone(state.winner)
        self.assertIsNone(state.next_home())

        self.assertEqual(
            state.unresolved_reason,
            "multiple_tie_venues_order_unverified",
        )

    def test_repeated_extra_tie(self) -> None:
        state = SeriesState(SEMI, 1, 2)

        for outcome in (
            "tie", "home", "home", "away", "away"
        ):
            play(state, outcome)

        play(state, "tie")

        self.assertEqual(state.next_home(), 1)

        play(state, "home")

        self.assertEqual(state.winner, 1)
        self.assertEqual(len(state.games), 7)

    def test_simulation_safety_limit(self) -> None:
        rule = StageRule(
            "semiPlayoff",
            2,
            1,
            (0,),
            (),
        )

        state = SeriesState(rule, 1, 2)

        for _ in range(MAX_SERIES_GAMES):
            play(state, "tie")

        self.assertIsNone(state.winner)
        self.assertIsNone(state.next_home())

        self.assertEqual(
            state.unresolved_reason,
            "simulation_game_limit",
        )

    def test_reject_wrong_home(self) -> None:
        state = SeriesState(KS, 1, 2)

        with self.assertRaises(ValueError):
            state.apply(Game(1, 2, 1, "home"))

    def test_reject_missing_game_number(self) -> None:
        state = SeriesState(WC, 4, 5)

        with self.assertRaises(ValueError):
            state.apply(Game(2, 4, 5, "away"))

    def test_reject_extra_game_after_completion(self) -> None:
        state = SeriesState(WC, 4, 5)

        play(state, "home")

        with self.assertRaises(ValueError):
            state.apply(Game(2, 4, 5, "away"))

    def test_reject_score_outcome_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            Game(
                1,
                4,
                5,
                "away",
                "test-game",
                "2025-10-06",
                10,
                0,
            )

    def test_replay_observed_wildcard(self) -> None:
        state = replay_series(
            WC,
            4,
            5,
            (
                observed(1, 4, 5, "away"),
                observed(2, 4, 5, "home"),
            ),
        )

        self.assertEqual(state.winner, 4)
        self.assertEqual(state.higher_wins, 1)
        self.assertEqual(state.lower_wins, 1)

    def test_reject_observed_game_after_simulation(self) -> None:
        state = SeriesState(WC, 4, 5)

        play(state, "away")

        with self.assertRaises(ValueError):
            state.apply(observed(2, 4, 5, "home"))


class PostseasonProbabilityTests(unittest.TestCase):
    """M1 범주형 표본 추출과 사건별 분모를 검사한다."""

    def test_probability_validation(self) -> None:
        actual = validate_probabilities({
            "home": 0.5,
            "tie": 0.1,
            "away": 0.4,
        })

        self.assertEqual(actual, (0.5, 0.1, 0.4))

    def test_invalid_probability_values(self) -> None:
        examples = (
            {"home": 0.5, "away": 0.5},
            {"home": 0.8, "tie": 0.2, "away": 0.2},
            {"home": -0.1, "tie": 0.5, "away": 0.6},
            {"home": float("nan"), "tie": 0.0, "away": 1.0},
            {"home": True, "tie": 0.0, "away": 0.0},
        )

        for values in examples:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    validate_probabilities(values)

    def test_zero_probability_not_sampled(self) -> None:
        rng = Random(42)

        for _ in range(100):
            result = sample_outcome(
                {
                    "home": 0.0,
                    "tie": 0.0,
                    "away": 1.0,
                },
                rng,
            )

            self.assertEqual(result, "away")

    def test_zero_denominator(self) -> None:
        result = probability_summary([None, None])

        self.assertEqual(result["valid"], 0)
        self.assertIsNone(result["probability"])
        self.assertEqual(result["status"], "unsupported")

    def test_empty_conditional_group(self) -> None:
        result = probability_summary([])

        self.assertIsNone(result["probability"])
        self.assertIsNone(result["lower_bound"])
        self.assertIsNone(result["upper_bound"])

    def test_probability_bounds(self) -> None:
        result = probability_summary(
            [True, False, None, None]
        )

        self.assertEqual(result["valid"], 2)
        self.assertEqual(result["unresolved"], 2)
        self.assertEqual(result["probability"], 0.5)
        self.assertEqual(result["lower_bound"], 0.25)
        self.assertEqual(result["upper_bound"], 0.75)


class PostseasonSimulationTests(unittest.TestCase):
    """동일 반복 대진 연결 및 상태 고정을 검사한다."""

    def test_bracket_connection(self) -> None:
        row = simulate(
            2025,
            [draw(1)],
            home_sampler,
            42,
        )[0]

        self.assertEqual(
            row.rounds["wildCard"].participants,
            (4, 5),
        )

        self.assertEqual(
            row.rounds["semiPlayoff"].participants,
            (3, 4),
        )

        self.assertEqual(
            row.rounds["playoff"].participants,
            (2, 3),
        )

        self.assertEqual(
            row.rounds["koreanSeries"].participants,
            (1, 2),
        )

        self.assertEqual(row.champion, 1)

    def test_preserve_unresolved_seed(self) -> None:
        sampler = Mock(return_value="home")

        row = simulate(
            2025,
            [draw(1, None)],
            sampler,
            42,
        )[0]

        sampler.assert_not_called()

        self.assertEqual(row.draw.iteration_id, 1)
        self.assertIsNone(row.champion)

        self.assertEqual(
            row.rounds["wildCard"].reason,
            "seed_unresolved",
        )

    def test_iteration_id_validation(self) -> None:
        with self.assertRaises(ValueError):
            simulate(
                2025,
                [draw(2)],
                home_sampler,
                42,
            )

    def test_seed_and_qualifier_consistency(self) -> None:
        original = draw(1)

        invalid = SeedInput(
            1,
            original.seeds,
            (1, 2, 3, 4, 6),
            original.qualification,
            1,
        )

        with self.assertRaises(ValueError):
            simulate(
                2025,
                [invalid],
                home_sampler,
                42,
            )

    def test_same_seed_reproducibility(self) -> None:
        draws = [
            draw(index)
            for index in range(1, 101)
        ]

        first = simulate(
            2025,
            draws,
            random_sampler,
            42,
        )

        second = simulate(
            2025,
            draws,
            random_sampler,
            42,
        )

        self.assertEqual(first, second)

    def test_prefix_reproducibility(self) -> None:
        draws = [
            draw(index)
            for index in range(1, 201)
        ]

        first = simulate(
            2025,
            draws[:100],
            random_sampler,
            42,
        )

        second = simulate(
            2025,
            draws,
            random_sampler,
            42,
        )

        self.assertEqual(first, second[:100])

    def test_fixed_completed_series(self) -> None:
        sampler = Mock(return_value="away")

        archive = ArchiveState(
            games={
                "wildCard": (
                    observed(1, 4, 5, "home"),
                ),
            },
            completion_evidence={
                "wildCard": {"winner": 4},
            },
        )

        result = simulate_round(
            WC,
            4,
            5,
            sampler,
            Random(42),
            archive,
        )

        sampler.assert_not_called()

        self.assertTrue(result.fixed)
        self.assertEqual(result.winner, 4)

    def test_fixed_prefix_and_remaining_game(self) -> None:
        sampler = Mock(return_value="tie")

        first = observed(1, 4, 5, "away")

        archive = ArchiveState(
            games={
                "wildCard": (first,),
            },
        )

        result = simulate_round(
            WC,
            4,
            5,
            sampler,
            Random(42),
            archive,
        )

        self.assertEqual(sampler.call_count, 1)
        self.assertEqual(result.games[0], first)
        self.assertEqual(result.winner, 4)
        self.assertFalse(result.fixed)

    def test_missing_completion_evidence(self) -> None:
        sampler = Mock(return_value="home")

        archive = ArchiveState(
            games={
                "wildCard": (
                    observed(1, 4, 5, "home"),
                ),
            },
        )

        result = simulate_round(
            WC,
            4,
            5,
            sampler,
            Random(42),
            archive,
        )

        self.assertIsNone(result.winner)

        self.assertEqual(
            result.reason,
            "archive_completion_evidence_missing",
        )

        sampler.assert_not_called()

    def test_archive_home_conflict(self) -> None:
        sampler = Mock(return_value="home")

        archive = ArchiveState(
            games={
                "koreanSeries": (
                    observed(1, 1, 2, "home"),
                    observed(2, 1, 2, "home"),
                    observed(3, 2, 1, "home"),
                    observed(4, 2, 1, "home"),
                    observed(5, 1, 2, "home"),
                ),
            },
        )

        result = simulate_round(
            KS,
            1,
            2,
            sampler,
            Random(42),
            archive,
        )

        self.assertIsNone(result.winner)
        self.assertEqual(len(result.games), 4)

        self.assertIn(
            "archive_rule_conflict",
            result.reason or "",
        )

        sampler.assert_not_called()

    def test_upstream_conflict_propagates(self) -> None:
        sampler = Mock(return_value="home")

        archive = ArchiveState(
            blocked={
                "wildCard": "missing_game",
            },
        )

        row = simulate(
            2025,
            [draw(1)],
            sampler,
            42,
            archive,
        )[0]

        self.assertIsNone(row.champion)

        self.assertEqual(
            row.rounds["semiPlayoff"].reason,
            "previous_round_unresolved",
        )

        sampler.assert_not_called()

    def test_joint_and_team_denominators(self) -> None:
        rows = simulate(
            2025,
            [draw(1), draw(2, None)],
            home_sampler,
            42,
        )

        result = summarize(2025, TEAMS, rows)

        self.assertEqual(result["seed_resolved"], 1)
        self.assertEqual(result["seed_unresolved"], 1)

        league = result["league"]["koreanSeries"]

        self.assertEqual(league["valid"], 1)
        self.assertEqual(league["unresolved"], 1)

        self.assertAlmostEqual(
            fsum(
                league["joint_win_probabilities"].values()
            ),
            1.0,
        )

        eliminated = result["teams"]["10"][
            "korean_series_champion"
        ]

        self.assertEqual(eliminated["valid"], 2)
        self.assertEqual(eliminated["probability"], 0.0)

    def test_conditional_matchup_denominator(self) -> None:
        rows = simulate(
            2025,
            [draw(1), draw(2, None)],
            home_sampler,
            42,
        )

        result = summarize(2025, TEAMS, rows)

        groups = result["conditional_matchups"]["wildCard"]

        formed = groups["4:5"]
        absent = groups["5:4"]

        self.assertEqual(formed["formed"], 1)
        self.assertEqual(formed["series_valid"], 1)
        self.assertEqual(formed["formation_unresolved"], 1)

        self.assertEqual(
            formed["teams"]["4"]["series_win"]["probability"],
            1.0,
        )

        self.assertEqual(absent["formed"], 0)

        self.assertIsNone(
            absent["teams"]["5"]["series_win"]["probability"]
        )

    def test_champion_is_korean_series_participant(self) -> None:
        rows = simulate(
            2025,
            [
                draw(index)
                for index in range(1, 101)
            ],
            random_sampler,
            42,
        )

        for row in rows:
            if row.champion is None:
                continue

            self.assertIn(
                row.champion,
                row.rounds["koreanSeries"].participants,
            )


if __name__ == "__main__":
    unittest.main()