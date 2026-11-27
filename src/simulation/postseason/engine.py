"""포스트시즌 대진 연결, M1 표본 추출 및 확률 집계."""

from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date
from math import fsum, isclose, isfinite
from random import Random
from typing import Literal

from ...models.game_outcome import model as m1
from ...settings import MAX_SEED
from ..regular_season.engine import inputs_for
from ..regular_season.rules import Fixture, SeasonState

from . import ITERATION_SCHEMA, SUMMARY_SCHEMA
from .rules import (
    OUTCOMES,
    STAGES,
    Game,
    Outcome,
    SeriesState,
    StageRule,
    get_rules,
)


EventKind = Literal["reach", "win"]

Sampler = Callable[[int, int, Random], Outcome]


@dataclass(frozen=True)
class SeedInput:
    """정규시즌 한 반복에서 인수한 공동 순위 결과."""

    iteration_id: int
    seeds: tuple[int, ...] | None
    qualifiers: tuple[int, ...] | None
    qualification: dict[int, bool | None]
    regular_first: int | None
    reasons: tuple[str, ...] = ()

    # 전체 대진이 미결정이어도 구단별 확정 상태를 보존한다.
    ranks: dict[int, int | None] = field(
        default_factory=dict
    )

    first_status: dict[int, bool | None] = field(
        default_factory=dict
    )


@dataclass
class ArchiveState:
    """기준 시점 이전의 원천 판본 기록과 검증 상태."""

    games: dict[str, tuple[Game, ...]] = field(
        default_factory=dict
    )
    blocked: dict[str, str] = field(default_factory=dict)
    completion_evidence: dict[
        str, dict[str, object]
    ] = field(default_factory=dict)


@dataclass(frozen=True)
class RoundResult:
    """한 반복의 라운드 진행 결과."""

    participants: tuple[int, int] | None
    winner: int | None
    games: tuple[Game, ...]
    reason: str | None
    fixed: bool


@dataclass(frozen=True)
class IterationResult:
    """부모 반복 식별자를 유지한 공동 시리즈 결과."""

    draw: SeedInput
    rounds: dict[str, RoundResult]

    @property
    def champion(self) -> int | None:
        """한국시리즈 우승 구단."""
        return self.rounds["koreanSeries"].winner

    def to_dict(self) -> dict[str, object]:
        """JSON Lines 출력 계약으로 변환한다."""
        return {
            "schema": ITERATION_SCHEMA,
            "iteration_id": self.draw.iteration_id,
            "postseason_seeds": self.draw.seeds,
            "qualifiers": self.draw.qualifiers,
            "qualification": self.draw.qualification,
            "regular_first": self.draw.regular_first,
            "regular_ranks": self.draw.ranks,
            "regular_first_status": self.draw.first_status,
            "seed_reasons": self.draw.reasons,
            "rounds": {
                stage: asdict(result)
                for stage, result in self.rounds.items()
            },
            "korean_series_champion": self.champion,
        }


def validate_probabilities(
    probabilities: dict[str, float],
) -> tuple[float, float, float]:
    """M1 승·무·패 분포의 범주와 확률 질량을 검사한다."""
    if set(probabilities) != set(OUTCOMES):
        raise ValueError("M1 경기 결과 범주 불일치")

    values = tuple(
        probabilities[outcome]
        for outcome in OUTCOMES
    )

    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or not 0 <= value <= 1
        for value in values
    ):
        raise ValueError("M1 경기 결과 확률 범위 오류")

    if not isclose(
        fsum(values),
        1.0,
        rel_tol=0.0,
        abs_tol=m1.TOLERANCE,
    ):
        raise ValueError("M1 경기 결과 확률 합 불일치")

    return values[0], values[1], values[2]


def sample_outcome(
    probabilities: dict[str, float],
    rng: Random,
) -> Outcome:
    """검증된 범주형 분포에서 경기 결과 하나를 표본 추출한다."""
    values = validate_probabilities(probabilities)
    threshold = rng.random() * fsum(values)

    cumulative = 0.0
    last_positive: Outcome = "home"

    outcomes: tuple[Outcome, ...] = ("home", "tie", "away")

    for outcome, probability in zip(
        outcomes,
        values,
        strict=True,
    ):
        if probability == 0:
            continue

        last_positive = outcome
        cumulative += probability

        if threshold < cumulative:
            return outcome

    # 부동소수점 누적 오차만 보정하며 확률 0인 결과는 뽑지 않는다.
    return last_positive


class FixedProfileSampler:
    """기준일 이전 정규시즌 프로필을 이용하는 M1 전이 시나리오."""

    def __init__(
        self,
        model: m1.Model,
        state: SeasonState,
        cutoff: date,
    ) -> None:
        self.model = model
        self.state = state
        self.cutoff = cutoff
        self.cache: dict[
            tuple[int, int],
            dict[str, float],
        ] = {}

    def probabilities(
        self,
        home: int,
        away: int,
    ) -> dict[str, float]:
        """홈·원정 순서별 M1 확률을 계산하고 재사용한다."""
        key = (home, away)

        if key not in self.cache:
            fixture = Fixture(
                game_id=f"postseason-profile:{home}:{away}",
                day=self.cutoff,
                home=home,
                away=away,
                number=0,
            )

            # 실제 포스트시즌을 정규시즌 학습 표본으로 추가하지 않는다.
            # 기존 M1의 정규시즌 입력 계약을 가정 시나리오에서 재사용한다.
            inputs = inputs_for(self.state, fixture)

            forecast = m1.predict(self.model, inputs)

            probabilities = dict(
                forecast["result_probabilities"]
            )

            validate_probabilities(probabilities)
            self.cache[key] = probabilities

        return self.cache[key]

    def __call__(
        self,
        home: int,
        away: int,
        rng: Random,
    ) -> Outcome:
        """동일 M1 확률에서 승·무·패를 추출한다."""
        return sample_outcome(
            self.probabilities(home, away),
            rng,
        )


def simulate_round(
    rule: StageRule,
    higher: int,
    lower: int,
    sampler: Sampler,
    rng: Random,
    archive: ArchiveState,
) -> RoundResult:
    """확정 경기 인수 후 미완료 차전만 모의한다."""
    state = SeriesState(rule, higher, lower)

    reason = archive.blocked.get(rule.stage)

    for game in archive.games.get(rule.stage, ()):
        try:
            state.apply(game)

        except ValueError as exc:
            if reason is None:
                reason = f"archive_rule_conflict: {exc}"

            break

    if reason is not None:
        return RoundResult(
            state.participants,
            None,
            tuple(state.games),
            reason,
            False,
        )

    if state.winner is not None:
        if rule.stage not in archive.completion_evidence:
            return RoundResult(
                state.participants,
                None,
                tuple(state.games),
                "archive_completion_evidence_missing",
                False,
            )

        return RoundResult(
            state.participants,
            state.winner,
            tuple(state.games),
            None,
            True,
        )

    while state.winner is None:
        home = state.next_home()

        if home is None:
            return RoundResult(
                state.participants,
                None,
                tuple(state.games),
                state.unresolved_reason,
                False,
            )

        away = lower if home == higher else higher

        outcome = sampler(home, away, rng)

        state.apply(
            Game(
                number=len(state.games) + 1,
                home=home,
                away=away,
                outcome=outcome,
            )
        )

    return RoundResult(
        state.participants,
        state.winner,
        tuple(state.games),
        None,
        False,
    )


def simulate_bracket(
    season: int,
    draw: SeedInput,
    sampler: Sampler,
    seed: int,
    archive: ArchiveState,
) -> IterationResult:
    """같은 정규시즌 반복에서 발생한 시드를 끝까지 연결한다."""
    rules = get_rules(season)

    if draw.seeds is None:
        return IterationResult(
            draw,
            {
                rule.stage: RoundResult(
                    None,
                    None,
                    (),
                    "seed_unresolved",
                    False,
                )
                for rule in rules
            },
        )

    seeds = draw.seeds

    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("포스트시즌 시드 중복 또는 수량 오류")

    if (
        draw.qualifiers is None
        or tuple(sorted(seeds)) != draw.qualifiers
        or draw.regular_first != seeds[0]
    ):
        raise ValueError("포스트시즌 시드와 인수 순위 불일치")

    # 반복별 난수 스트림을 분리하여 미결정 행의 존재 여부나
    # 다른 반복의 경기 수가 해당 반복 난수열을 바꾸지 않게 한다.
    rng = Random(f"{seed}:{draw.iteration_id}")

    rounds: dict[str, RoundResult] = {}
    advancing = seeds[4]

    for index, rule in enumerate(rules):
        if index > 0 and advancing is None:
            rounds[rule.stage] = RoundResult(
                None,
                None,
                (),
                "previous_round_unresolved",
                False,
            )
            continue

        if advancing is None:
            raise ValueError("초기 와일드카드 참가 구단 누락")

        result = simulate_round(
            rule,
            seeds[rule.seed_index],
            advancing,
            sampler,
            rng,
            archive,
        )

        rounds[rule.stage] = result
        advancing = result.winner

    return IterationResult(draw, rounds)


def simulate(
    season: int,
    draws: Sequence[SeedInput],
    sampler: Sampler,
    seed: int,
    archive: ArchiveState | None = None,
) -> list[IterationResult]:
    """인수받은 모든 반복을 누락 없이 계산한다."""
    get_rules(season)

    if type(seed) is not int or not 0 <= seed <= MAX_SEED:
        raise ValueError("Seed 범위 오류")

    if not draws:
        raise ValueError("시뮬레이션 반복이 없습니다.")

    archive = archive if archive is not None else ArchiveState()

    results = []

    for index, draw in enumerate(draws, start=1):
        if (
            type(draw.iteration_id) is not int
            or draw.iteration_id != index
        ):
            raise ValueError("반복 ID의 중복 또는 누락")

        results.append(
            simulate_bracket(
                season,
                draw,
                sampler,
                seed,
                archive,
            )
        )

    return results


def regular_first_event(
    draw: SeedInput,
    team: int,
) -> bool | None:
    """부모 반복의 개별 정규시즌 1위 사건을 보존한다."""
    if draw.regular_first is not None:
        return draw.regular_first == team

    if draw.qualification.get(team) is False:
        return False

    return draw.first_status.get(team)


def team_event(
    row: IterationResult,
    team: int,
    rule: StageRule,
    kind: EventKind,
) -> bool | None:
    """개별 사건의 참·거짓·미결정 상태를 구별한다."""
    result = row.rounds[rule.stage]

    if result.participants is not None:
        if kind == "reach":
            return team in result.participants

        if team not in result.participants:
            return False

        if result.winner is not None:
            return result.winner == team

        return None

    if row.draw.qualification.get(team) is False:
        return False

    seeds = row.draw.seeds

    if seeds is not None:
        # 해당 단계보다 높은 시드 또는 탈락 구단은
        # 이 라운드의 참가·승리 사건에 해당하지 않는다.
        if team not in seeds[rule.seed_index:]:
            return False

        if (
            kind == "reach"
            and team == seeds[rule.seed_index]
        ):
            return True

    else:
        # 전체 시드가 미확정이어도 개별 확정 순위로
        # 직접 진출이나 해당 라운드 미참가 여부를 판단할 수 있다.
        rank = row.draw.ranks.get(team)

        if rank is not None:
            direct_rank = rule.seed_index + 1

            if rank < direct_rank or rank > 5:
                return False

            if kind == "reach":
                if rank == direct_rank:
                    return True

                if rule.is_wildcard and rank == 5:
                    return True

    if kind == "reach" and rule.stage == "koreanSeries":
        if regular_first_event(row.draw, team) is True:
            return True

    for stage in STAGES:
        if stage == rule.stage:
            break

        previous = row.rounds[stage]

        if (
            previous.participants is not None
            and team in previous.participants
            and previous.winner is not None
            and previous.winner != team
        ):
            return False

    return None


def probability_summary(
    values: Sequence[bool | None],
) -> dict[str, object]:
    """사건별 유효 분모, 미결정 수 및 논리적 범위를 집계한다."""
    total = len(values)
    positive = sum(value is True for value in values)
    negative = sum(value is False for value in values)
    valid = positive + negative
    unresolved = total - valid

    status = (
        "unsupported"
        if valid == 0
        else "partial"
        if unresolved > 0
        else "estimated"
    )

    return {
        "total": total,
        "valid": valid,
        "unresolved": unresolved,
        "positive": positive,
        "negative": negative,
        "probability": positive / valid if valid else None,
        "lower_bound": positive / total if total else None,
        "upper_bound": (
            (positive + unresolved) / total
            if total
            else None
        ),
        "all_iterations_resolved": (
            total > 0 and unresolved == 0
        ),
        "status": status,
    }


def summarize(
    season: int,
    teams: Sequence[int],
    rows: Sequence[IterationResult],
) -> dict[str, object]:
    """반복별 공동 결과에서 전체 및 대진 조건부 확률을 계산한다."""
    rules = get_rules(season)

    if not rows:
        raise ValueError("집계할 포스트시즌 결과가 없습니다.")

    if len(set(teams)) != len(teams):
        raise ValueError("집계 구단 ID 중복")

    total = len(rows)

    team_results: dict[str, object] = {}
    league: dict[str, object] = {}
    conditionals: dict[str, object] = {}

    for team in teams:
        team_results[str(team)] = {
            "regular_first": probability_summary([
                regular_first_event(row.draw, team)
                for row in rows
            ]),
            "rounds": {
                rule.stage: {
                    "reach": probability_summary([
                        team_event(row, team, rule, "reach")
                        for row in rows
                    ]),
                    "win": probability_summary([
                        team_event(row, team, rule, "win")
                        for row in rows
                    ]),
                }
                for rule in rules
            },
            "korean_series_entry": probability_summary([
                team_event(row, team, rules[-1], "reach")
                for row in rows
            ]),
            "korean_series_champion": probability_summary([
                team_event(row, team, rules[-1], "win")
                for row in rows
            ]),
        }

    for rule in rules:
        stage = rule.stage

        valid = [
            row
            for row in rows
            if row.rounds[stage].winner is not None
        ]

        wins = Counter(
            row.rounds[stage].winner
            for row in valid
        )

        if sum(wins.values()) != len(valid):
            raise ValueError("라운드 공동 우승 분모 오류")

        if not set(wins).issubset(teams):
            raise ValueError("지원하지 않는 시리즈 승리 구단")

        joint = {
            str(team): (
                wins[team] / len(valid)
                if valid
                else None
            )
            for team in teams
        }

        if valid and not isclose(
            fsum(
                value
                for value in joint.values()
                if value is not None
            ),
            1.0,
            rel_tol=0.0,
            abs_tol=m1.TOLERANCE,
        ):
            raise ValueError("라운드 공동 확률 합 오류")

        reasons = Counter(
            row.rounds[stage].reason
            or "unknown"
            for row in rows
            if row.rounds[stage].winner is None
        )

        groups: dict[
            tuple[int, int],
            list[IterationResult],
        ] = defaultdict(list)

        for row in rows:
            result = row.rounds[stage]

            if result.participants is not None:
                groups[result.participants].append(row)

        formed = sum(
            len(group)
            for group in groups.values()
        )

        league[stage] = {
            "total": total,
            "valid": len(valid),
            "unresolved": total - len(valid),
            "matchup_formed": formed,
            "matchup_unresolved": total - formed,
            "fixed": sum(
                row.rounds[stage].fixed
                for row in rows
            ),
            "joint_win_probabilities": joint,
            "reasons": dict(reasons),
        }

        matchup_results = {}

        for higher in teams:
            for lower in teams:
                if higher == lower:
                    continue

                group = groups.get((higher, lower), [])

                series_valid = sum(
                    row.rounds[stage].winner is not None
                    for row in group
                )

                key = f"{higher}:{lower}"

                matchup_results[key] = {
                    "higher_seed_team": higher,
                    "lower_seed_team": lower,
                    "formed": len(group),
                    "formation_unresolved": total - formed,
                    "series_valid": series_valid,
                    "series_unresolved": (
                        len(group) - series_valid
                    ),
                    "teams": {
                        str(team): {
                            "series_win": probability_summary([
                                team_event(
                                    row,
                                    team,
                                    rule,
                                    "win",
                                )
                                for row in group
                            ]),
                            "champion": probability_summary([
                                team_event(
                                    row,
                                    team,
                                    rules[-1],
                                    "win",
                                )
                                for row in group
                            ]),
                        }
                        for team in (higher, lower)
                    },
                }

        conditionals[stage] = matchup_results

    resolved_seeds = sum(
        row.draw.seeds is not None
        for row in rows
    )

    seed_reasons = Counter(
        reason
        for row in rows
        if row.draw.seeds is None
        for reason in (
            row.draw.reasons
            or ("seed_unresolved",)
        )
    )

    return {
        "schema": SUMMARY_SCHEMA,
        "season": season,
        "requested_iterations": total,
        "seed_resolved": resolved_seeds,
        "seed_unresolved": total - resolved_seeds,
        "seed_reasons": dict(seed_reasons),
        "teams": team_results,
        "league": league,
        "conditional_matchups": conditionals,
    }