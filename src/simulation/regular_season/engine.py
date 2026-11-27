"""M1 결합 득점 분포 기반 정규시즌 반복 계산과 확률 집계."""

from bisect import bisect_right
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import date, timedelta
from itertools import accumulate, groupby
from math import fsum, isclose, isfinite, log
from random import Random
from typing import TypedDict, cast

from ...models.game_outcome import model as m1

from . import ITERATION_SCHEMA, SUMMARY_SCHEMA
from .rules import (
    Fixture,
    PlayedGame,
    SeasonRules,
    SeasonState,
    Standing,
    fixture_order,
    resolve_standings,
    validate_schedule,
)


RECENT_DAYS = 28
LOG_LOSS_FLOOR = 1e-15

Score = tuple[int, int]
ScoreFunction = Callable[
    [Mapping[str, object], Random],
    Score,
]


class TeamResult(TypedDict):
    """한 반복에서 계산한 구단별 결과."""

    wins: int
    losses: int
    ties: int
    games: int
    final_wins: int | None
    observed_win_rate: float | None
    rank: int | None
    first: bool | None
    qualifies: bool | None
    best_rank: int
    worst_rank: int
    reason: str | None


class Iteration(TypedDict):
    """후속 포스트시즌 계산에 전달할 한 반복의 공동 결과."""

    schema: str
    iteration_id: int
    teams: dict[str, TeamResult]
    qualifiers: list[int] | None
    postseason_seeds: list[int] | None
    rank_resolved: bool
    qualification_resolved: bool


def cumulative(weights: Sequence[float]) -> tuple[float, ...]:
    """양의 확률 질량을 누적 분포로 변환한다."""

    if not weights:
        raise ValueError("빈 확률 분포")

    if any(not isfinite(value) or value <= 0 for value in weights):
        raise ValueError("표본 추출 가중치는 유한한 양수여야 합니다.")

    total = fsum(weights)

    if not isfinite(total) or total <= 0:
        raise ValueError("확률 분포의 합이 올바르지 않습니다.")

    result = [
        min(1.0, value / total)
        for value in accumulate(weights)
    ]

    result[-1] = 1.0

    return tuple(result)


class JointSampler:
    """M1의 승무패 확률과 조건부 득점 분포에서 점수 쌍을 추출한다."""

    def __init__(self, forecast: m1.Forecast) -> None:
        if (
            forecast["schema"] != m1.SCHEMA
            or forecast["score_scope"] != "official_final"
        ):
            raise ValueError("지원하지 않는 M1 예측 출력")

        probabilities = forecast["result_probabilities"]

        if set(probabilities) != set(m1.OUTCOMES):
            raise ValueError("M1 승무패 범주 불일치")

        masses = tuple(
            probabilities[outcome]
            for outcome in m1.OUTCOMES
        )

        m1.check_pmf(masses)

        joint = forecast["joint"]

        home_pmf = joint["home_base_pmf"]
        away_pmf = joint["away_base_pmf"]

        m1.check_pmf(home_pmf)
        m1.check_pmf(away_pmf)

        equal = joint["equal_score_weight"]
        different = joint["different_score_weight"]
        normalizer = joint["normalizer"]

        if (
            not all(
                isfinite(value) and value >= 0
                for value in (equal, different)
            )
            or not isfinite(normalizer)
            or normalizer <= 0
        ):
            raise ValueError("M1 결합 분포의 가중치 오류")

        scores: list[list[Score]] = [[], [], []]
        weights: list[list[float]] = [[], [], []]

        for home_runs, home_probability in enumerate(home_pmf):
            for away_runs, away_probability in enumerate(away_pmf):

                if home_runs > away_runs:
                    category = 0
                elif home_runs == away_runs:
                    category = 1
                else:
                    category = 2

                weight = (
                    equal if category == 1 else different
                )

                probability = (
                    home_probability
                    * away_probability
                    * weight
                    / normalizer
                )

                if not isfinite(probability) or probability < 0:
                    raise ValueError("결합 득점 셀 확률 오류")

                if probability == 0:
                    continue

                scores[category].append(
                    (home_runs, away_runs)
                )

                weights[category].append(probability)

        # result_probabilities와 joint에서 재계산한 확률이
        # 일치해야 동일한 예측의 조건부 표본 추출이 가능하다.
        for category, expected in enumerate(masses):
            actual = fsum(weights[category])

            if not isclose(
                expected,
                actual,
                rel_tol=0.0,
                abs_tol=m1.TOLERANCE,
            ):
                raise ValueError("승무패 확률과 결합 득점 분포 불일치")

            if expected > 0 and not weights[category]:
                raise ValueError("조건부 득점 분포가 비어 있습니다.")

        self.categories = tuple(
            category
            for category, probability in enumerate(masses)
            if probability > 0
        )

        self.outcome_cdf = cumulative([
            masses[category]
            for category in self.categories
        ])

        self.scores = tuple(
            tuple(group)
            for group in scores
        )

        self.score_cdfs = tuple(
            cumulative(group) if group else ()
            for group in weights
        )

    def sample(self, random: Random) -> Score:
        """승무패를 먼저 표본 추출한 뒤 해당 조건의 득점을 추출한다."""

        category_index = bisect_right(
            self.outcome_cdf,
            random.random(),
        )

        category = self.categories[category_index]

        score_index = bisect_right(
            self.score_cdfs[category],
            random.random(),
        )

        return self.scores[category][score_index]


class M1Sampler:
    """저장 모델의 기존 추론을 재사용하고 동일 프로필 결과를 캐시한다."""

    def __init__(
        self,
        model: m1.Model,
        *,
        venue_only: bool = False,
    ) -> None:
        self.model = model
        self.venue_only = venue_only
        self.cache: dict[tuple[str, str], JointSampler] = {}

    def __call__(
        self,
        inputs: Mapping[str, object],
        random: Random,
    ) -> Score:

        history = m1.history(inputs)

        if self.venue_only:
            # 개발 진단용 비교 계산이다.
            # 별도 모델 학습이나 신규 운영 예측기를 추가하지 않는다.
            key = ("venue", "venue")

            if key not in self.cache:
                home = m1.pmf(
                    self.model["site_counts"]["home"]
                )

                away = m1.pmf(
                    self.model["site_counts"]["away"]
                )

                cap = self.model["cap"]

                tail = m1.TAIL_SHARE * 0.5 ** (cap + 1)

                forecast = m1.combine(
                    home,
                    away,
                    self.model["tie_weight"],
                    2 * tail,
                    4 * (cap + 2) * tail,
                )

                self.cache[key] = JointSampler(forecast)

        else:
            mean = self.model["league_mean"]

            key = (
                m1.profile(history, "home", mean),
                m1.profile(history, "away", mean),
            )

            if key not in self.cache:
                forecast = m1.predict(
                    self.model,
                    inputs,
                )

                self.cache[key] = JointSampler(forecast)

        return self.cache[key].sample(random)


def build_state(
    rules: SeasonRules,
    completed: Sequence[PlayedGame],
) -> SeasonState:
    """완료 경기로부터 기준 시즌 성적을 생성한다."""

    state = SeasonState(rules)

    for game in sorted(
        completed,
        key=lambda result: fixture_order(result.fixture),
    ):
        state.add(game)

    return state


def inputs_for(
    state: SeasonState,
    fixture: Fixture,
) -> dict[str, object]:
    """해당 경기일 이전에 존재하는 이력만으로 M1 입력을 재구성한다."""

    values: dict[str, object] = {
        "competition": "regular",
    }

    recent_start = fixture.day - timedelta(days=RECENT_DAYS)

    for side, team in (
        ("home", fixture.home),
        ("away", fixture.away),
    ):
        record = state.records[team]
        history = state.history[team]

        # 같은 날 먼저 끝난 더블헤더도 입력에서 제외해야 한다.
        if history and history[-1][0] >= fixture.day:
            raise ValueError(
                f"경기일 당일 또는 이후 이력이 입력에 포함됩니다: "
                f"{fixture.game_id}"
            )

        recent_games = 0
        recent_rf = 0
        recent_ra = 0

        for day, scored, allowed in reversed(history):
            if day < recent_start:
                break

            recent_games += 1
            recent_rf += scored
            recent_ra += allowed

        values.update({
            f"{side}_season_games": record.games,
            f"{side}_season_rf": record.runs_for,
            f"{side}_season_ra": record.runs_against,
            f"{side}_recent_games": recent_games,
            f"{side}_recent_rf": recent_rf,
            f"{side}_recent_ra": recent_ra,
        })

    if set(values) != set(m1.INPUT_COLUMNS):
        raise ValueError("M1 입력 허용 목록 불일치")

    m1.history(values)

    return values


def team_result(
    state: SeasonState,
    team: int,
    standing: Standing,
) -> TeamResult:
    """성적과 공식 순위 판정을 하나의 구단 결과로 변환한다."""

    record = state.records[team]
    rate = record.win_rate

    return {
        "wins": record.wins,
        "losses": record.losses,
        "ties": record.ties,
        "games": record.games,
        "final_wins": (
            record.wins
            if record.games == state.rules.games_per_team
            else None
        ),
        "observed_win_rate": float(rate) if rate is not None else None,
        "rank": standing.rank,
        "first": standing.first,
        "qualifies": standing.qualifies,
        "best_rank": standing.best_rank,
        "worst_rank": standing.worst_rank,
        "reason": standing.reason,
    }


def make_iteration(
    iteration_id: int,
    state: SeasonState,
    standings: Mapping[int, Standing],
) -> Iteration:
    """후속 시리즈 계산에서 그대로 읽을 수 있는 공동 결과를 만든다."""

    rules = state.rules

    teams = {
        str(team): team_result(
            state,
            team,
            standings[team],
        )
        for team in rules.teams
    }

    qualification_resolved = all(
        value.qualifies is not None
        for value in standings.values()
    )

    qualifiers: list[int] | None = None
    postseason_seeds: list[int] | None = None

    if qualification_resolved:
        qualifiers = sorted(
            team
            for team in rules.teams
            if standings[team].qualifies is True
        )

        if len(qualifiers) != rules.playoff_slots:
            raise ValueError("확정 진출 구단 수 불일치")

        ranked = {
            standings[team].rank: team
            for team in qualifiers
            if standings[team].rank is not None
        }

        expected = set(range(1, rules.playoff_slots + 1))

        if set(ranked) == expected:
            postseason_seeds = [
                ranked[rank]
                for rank in range(1, rules.playoff_slots + 1)
            ]

    return {
        "schema": ITERATION_SCHEMA,
        "iteration_id": iteration_id,
        "teams": teams,
        "qualifiers": qualifiers,
        "postseason_seeds": postseason_seeds,
        "rank_resolved": all(
            standing.rank is not None
            for standing in standings.values()
        ),
        "qualification_resolved": qualification_resolved,
    }


def verify_fixed(
    fixed: Mapping[int, Standing],
    final: Mapping[int, Standing],
) -> None:
    """기준 시점에서 이미 확정한 상태가 변경되지 않았는지 검사한다."""

    for team, before in fixed.items():
        after = final[team]

        if before.rank is not None and after.rank != before.rank:
            raise ValueError(f"확정 순위 변경: {team}")

        if before.first is not None and after.first != before.first:
            raise ValueError(f"확정 1위 상태 변경: {team}")

        if (
            before.qualifies is not None
            and after.qualifies != before.qualifies
        ):
            raise ValueError(f"확정 진출 상태 변경: {team}")


def simulate(
    rules: SeasonRules,
    cutoff: date,
    completed: Sequence[PlayedGame],
    remaining: Sequence[Fixture],
    sampler: ScoreFunction,
    iterations: int,
    seed: int,
) -> list[Iteration]:
    """기준 성적을 고정하고 잔여 경기만 독립 반복 계산한다."""

    if type(iterations) is not int or iterations <= 0:
        raise ValueError("반복 횟수는 양의 정수여야 합니다.")

    if type(seed) is not int:
        raise ValueError("Seed는 정수여야 합니다.")

    if any(
        game.fixture.day >= cutoff
        for game in completed
    ):
        raise ValueError("기준 이후 실제 완료 경기가 포함됩니다.")

    if any(game.day < cutoff for game in remaining):
        raise ValueError("기준 이전 경기의 반복 계산은 허용하지 않습니다.")

    validate_schedule(
        rules,
        [
            *(game.fixture for game in completed),
            *remaining,
        ],
    )

    initial = build_state(rules, completed)
    fixed = resolve_standings(initial)

    ordered = sorted(remaining, key=fixture_order)

    # 당일 경기 전체의 예측을 먼저 생성한 후
    # 그날의 모의 결과를 일괄 반영한다.
    batches = [
        tuple(games)
        for _, games in groupby(
            ordered,
            key=lambda game: game.day,
        )
    ]

    random = Random(seed)
    results: list[Iteration] = []

    for iteration_id in range(1, iterations + 1):
        state = initial.copy()

        for batch in batches:
            generated = []

            for fixture in batch:
                inputs = inputs_for(state, fixture)

                home_runs, away_runs = sampler(
                    inputs,
                    random,
                )

                generated.append(
                    PlayedGame(
                        fixture,
                        home_runs,
                        away_runs,
                    )
                )

            for result in generated:
                state.add(result)

        standings = resolve_standings(state)

        verify_fixed(fixed, standings)

        results.append(
            make_iteration(
                iteration_id,
                state,
                standings,
            )
        )

    return results


def distribution(
    values: Sequence[int | None],
    reasons: Sequence[str | None],
    categories: Sequence[int],
) -> dict[str, object]:
    """전체 반복과 유효 반복을 구분하여 확률 분포를 집계한다."""

    if len(values) != len(reasons):
        raise ValueError("집계값과 미결정 사유의 길이 불일치")

    valid_values = [
        value for value in values if value is not None
    ]

    counts = Counter(valid_values)
    valid = len(valid_values)

    if not set(counts).issubset(categories):
        raise ValueError("허용 범위 밖의 집계값")

    probabilities = {
        str(category): (
            counts[category] / valid if valid else None
        )
        for category in categories
    }

    if valid and not isclose(
        fsum(
            value
            for value in probabilities.values()
            if value is not None
        ),
        1.0,
        rel_tol=0.0,
        abs_tol=m1.TOLERANCE,
    ):
        raise ValueError("집계 확률 합이 1이 아닙니다.")

    unresolved_reasons = Counter(
        reason or "unresolved"
        for value, reason in zip(
            values,
            reasons,
            strict=True,
        )
        if value is None
    )

    return {
        "status": "supported" if valid else "unsupported",
        "requested": len(values),
        "valid": valid,
        "unresolved": len(values) - valid,
        "unresolved_reasons": dict(unresolved_reasons),
        "counts": {
            str(category): counts[category]
            for category in categories
        },
        "probabilities": probabilities,
        "expected": (
            sum(valid_values) / valid
            if valid
            else None
        ),
    }


def binary_values(
    values: Sequence[bool | None],
) -> list[int | None]:
    """미결정 상태를 0으로 바꾸지 않고 이진 사건을 집계한다."""

    return [
        None if value is None else int(value)
        for value in values
    ]


def summarize(
    rules: SeasonRules,
    rows: Sequence[Iteration],
) -> dict[str, object]:
    """반복별 기록만으로 팀별 최종 전망을 재현한다."""

    if not rows:
        raise ValueError("집계할 반복 결과가 없습니다.")

    team_keys = {str(team) for team in rules.teams}

    for index, row in enumerate(rows, start=1):
        if (
            row["schema"] != ITERATION_SCHEMA
            or row["iteration_id"] != index
            or set(row["teams"]) != team_keys
        ):
            raise ValueError("반복 식별자 또는 스키마 불일치")

        team_rows = row["teams"]

        if sum(
            item["wins"]
            for item in team_rows.values()
        ) != sum(
            item["losses"]
            for item in team_rows.values()
        ):
            raise ValueError("리그 전체 승·패 합계 불일치")

        for item in team_rows.values():
            if (
                item["wins"] + item["losses"] + item["ties"]
                != item["games"]
            ):
                raise ValueError("구단별 경기 수 집계 오류")

            if item["rank"] is not None:
                if item["first"] != (item["rank"] == 1):
                    raise ValueError("순위와 1위 판정 불일치")

                if item["qualifies"] != (
                    item["rank"] <= rules.playoff_slots
                ):
                    raise ValueError("순위와 진출 판정 불일치")

        if row["qualification_resolved"]:
            expected = sorted(
                int(team)
                for team, item in team_rows.items()
                if item["qualifies"] is True
            )

            if (
                row["qualifiers"] != expected
                or len(expected) != rules.playoff_slots
                or any(
                    item["qualifies"] is None
                    for item in team_rows.values()
                )
            ):
                raise ValueError("반복별 진출 구단 불일치")

        seeds = row["postseason_seeds"]

        if seeds is not None:
            if (
                len(seeds) != rules.playoff_slots
                or len(set(seeds)) != len(seeds)
                or sorted(seeds) != row["qualifiers"]
            ):
                raise ValueError("포스트시즌 시드 구성 불일치")

            for rank, team in enumerate(seeds, start=1):
                if team_rows[str(team)]["rank"] != rank:
                    raise ValueError("포스트시즌 시드와 순위 불일치")

    teams: dict[str, object] = {}

    for team in rules.teams:
        results = [
            row["teams"][str(team)]
            for row in rows
        ]

        reasons = [
            result["reason"]
            for result in results
        ]

        wins = distribution(
            [result["final_wins"] for result in results],
            ["incomplete_schedule"] * len(results),
            tuple(range(rules.games_per_team + 1)),
        )

        rank = distribution(
            [result["rank"] for result in results],
            reasons,
            tuple(range(1, len(rules.teams) + 1)),
        )

        first = distribution(
            binary_values([
                result["first"]
                for result in results
            ]),
            reasons,
            (0, 1),
        )

        qualification = distribution(
            binary_values([
                result["qualifies"]
                for result in results
            ]),
            reasons,
            (0, 1),
        )

        teams[str(team)] = {
            "team_id": team,
            "final_wins": wins,
            "rank": rank,
            "regular_season_first": first,
            "postseason_qualification": qualification,
        }

    # 구단별 유효 분모가 다를 수 있으므로
    # 리그 전체에서 동시에 확정된 반복의 확률도 따로 제공한다.
    first_rows = [
        row for row in rows
        if all(
            item["first"] is not None
            for item in row["teams"].values()
        )
    ]

    qualification_rows = [
        row for row in rows
        if row["qualification_resolved"]
    ]

    for row in first_rows:
        if sum(
            item["first"] is True
            for item in row["teams"].values()
        ) != 1:
            raise ValueError("한 반복의 정규시즌 1위 구단 수 오류")

    first_joint = {
        str(team): (
            sum(
                row["teams"][str(team)]["first"] is True
                for row in first_rows
            ) / len(first_rows)
            if first_rows else None
        )
        for team in rules.teams
    }

    qualification_joint = {
        str(team): (
            sum(
                row["teams"][str(team)]["qualifies"] is True
                for row in qualification_rows
            ) / len(qualification_rows)
            if qualification_rows else None
        )
        for team in rules.teams
    }

    if first_rows and not isclose(
        fsum(
            value
            for value in first_joint.values()
            if value is not None
        ),
        1.0,
        rel_tol=0.0,
        abs_tol=m1.TOLERANCE,
    ):
        raise ValueError("리그 공동 분모의 1위 확률 합 오류")

    if qualification_rows and not isclose(
        fsum(
            value
            for value in qualification_joint.values()
            if value is not None
        ),
        rules.playoff_slots,
        rel_tol=0.0,
        abs_tol=m1.TOLERANCE,
    ):
        raise ValueError("리그 공동 분모의 진출 확률 합 오류")

    return {
        "schema": SUMMARY_SCHEMA,
        "season": rules.season,
        "requested_iterations": len(rows),
        "teams": teams,
        "league": {
            "rank_resolved_iterations": sum(
                row["rank_resolved"]
                for row in rows
            ),
            "postseason_seed_resolved_iterations": sum(
                row["postseason_seeds"] is not None
                for row in rows
            ),
            "first_joint_valid": len(first_rows),
            "first_joint_unresolved": len(rows) - len(first_rows),
            "first_joint_probabilities": first_joint,
            "qualification_joint_valid": len(qualification_rows),
            "qualification_joint_unresolved": (
                len(rows) - len(qualification_rows)
            ),
            "qualification_joint_probabilities": qualification_joint,
        },
    }


def diagnostic_metrics(
    rules: SeasonRules,
    summary: Mapping[str, object],
    actual: Sequence[PlayedGame],
) -> dict[str, object]:
    """개발 시나리오의 실제 완료 시즌 대비 오차를 별도로 계산한다."""

    state = build_state(rules, actual)
    truth = resolve_standings(state)

    forecasts = cast(
        dict[str, dict[str, object]],
        summary["teams"],
    )

    win_errors: list[float] = []
    rank_errors: list[float] = []

    first_brier: list[float] = []
    first_log: list[float] = []

    qualification_brier: list[float] = []
    qualification_log: list[float] = []

    actual_teams = {}

    for team in rules.teams:
        forecast = forecasts[str(team)]

        actual_result = team_result(
            state,
            team,
            truth[team],
        )

        actual_teams[str(team)] = actual_result

        wins = cast(
            dict[str, object],
            forecast["final_wins"],
        )["expected"]

        rank = cast(
            dict[str, object],
            forecast["rank"],
        )["expected"]

        if wins is not None and actual_result["final_wins"] is not None:
            win_errors.append(
                abs(float(cast(float, wins)) - actual_result["final_wins"])
            )

        if rank is not None and actual_result["rank"] is not None:
            rank_errors.append(
                abs(float(cast(float, rank)) - actual_result["rank"])
            )

        binary = (
            (
                "regular_season_first",
                actual_result["first"],
                first_brier,
                first_log,
            ),
            (
                "postseason_qualification",
                actual_result["qualifies"],
                qualification_brier,
                qualification_log,
            ),
        )

        for metric, observed, brier, logarithmic in binary:
            probability = cast(
                dict[str, object],
                forecast[metric],
            )["expected"]

            if probability is None or observed is None:
                continue

            predicted = float(cast(float, probability))
            target = int(observed)

            brier.append(
                (predicted - target) ** 2
            )

            observed_probability = (
                predicted if observed else 1 - predicted
            )

            logarithmic.append(
                -log(
                    max(
                        observed_probability,
                        LOG_LOSS_FLOOR,
                    )
                )
            )

    def metric(values: Sequence[float]) -> dict[str, object]:
        return {
            "team_rows": len(values),
            "value": (
                fsum(values) / len(values)
                if values else None
            ),
        }

    return {
        "mode": "retrospective_development_diagnostic",
        "independent_seasons": 1,
        "formal_backtest": False,
        "final_test_used": False,
        "actual_teams": actual_teams,
        "wins_mae": metric(win_errors),
        "rank_mae": metric(rank_errors),
        "first_brier": metric(first_brier),
        "first_binary_log_loss": metric(first_log),
        "qualification_brier": metric(qualification_brier),
        "qualification_binary_log_loss": metric(qualification_log),
        "probability_denominator": (
            "구단별 유효 반복을 분모로 계산한 조건부 확률"
        ),
    }