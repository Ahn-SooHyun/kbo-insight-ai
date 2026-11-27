"""시즌별 경기 구성, 승률, 동률 및 포스트시즌 진출 규칙."""

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import date
from fractions import Fraction
from itertools import combinations, groupby


RULE_SOURCE = (
    "https://6ptotvmi5753.edge.naverncp.com/KBO_FILE/ebook/pdf/"
    "2025_%EB%A6%AC%EA%B7%B8%EA%B7%9C%EC%A0%95.pdf"
)

PREVIOUS_RANK_SOURCE = (
    "https://www.koreabaseball.com/MediaNews/Notice/View.aspx?bdSe=11396"
)

RULE_VERSION = "kbo-2025-ranking-v1"

TEAM_NAMES = {
    2001: "KIA",
    2002: "KT",
    2003: "LG",
    2004: "NC",
    2005: "SSG",
    2006: "두산",
    2007: "롯데",
    2008: "삼성",
    2009: "키움",
    2010: "한화",
}

# KBO가 2025-03-07 발표한 전년도 최종 구단 순위.
# 개별 경기의 수동 입력이 아니라 시즌 규칙 적용에 필요한
# 공식 순위 근거를 보존한다.
PREVIOUS_RANK_2025 = {
    2001: 1,
    2008: 2,
    2003: 3,
    2006: 4,
    2002: 5,
    2005: 6,
    2007: 7,
    2010: 8,
    2004: 9,
    2009: 10,
}


@dataclass(frozen=True)
class Fixture:
    """정규시즌 경기 식별 정보."""

    game_id: str
    day: date
    home: int
    away: int
    number: int

    def __post_init__(self) -> None:
        if not self.game_id:
            raise ValueError("경기 ID가 비어 있습니다.")

        if self.home == self.away:
            raise ValueError("홈·원정 구단이 같습니다.")

        if type(self.number) is not int or self.number not in (0, 1, 2):
            raise ValueError("경기 차수는 0, 1, 2만 허용합니다.")


@dataclass(frozen=True)
class PlayedGame:
    """실제 또는 모의 경기의 공식 종료 득점."""

    fixture: Fixture
    home_runs: int
    away_runs: int

    def __post_init__(self) -> None:
        for score in (self.home_runs, self.away_runs):
            if type(score) is not int or score < 0:
                raise ValueError("득점은 0 이상의 정수여야 합니다.")


@dataclass(frozen=True)
class SeasonRules:
    """지원 시즌의 구단 구성과 순위 결정 계약."""

    season: int
    teams: tuple[int, ...]
    games_per_pair: int
    playoff_slots: int
    previous_rank: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.teams) < 2:
            raise ValueError("최소 두 구단이 필요합니다.")

        if len(set(self.teams)) != len(self.teams):
            raise ValueError("구단 ID가 중복됩니다.")

        if self.games_per_pair < 1:
            raise ValueError("구단 간 경기 수는 양수여야 합니다.")

        if not 1 <= self.playoff_slots <= len(self.teams):
            raise ValueError("포스트시즌 진출 구단 수 오류")

        if not set(self.previous_rank).issubset(self.teams):
            raise ValueError("전년도 순위에 지원하지 않는 구단이 있습니다.")

        if any(
            type(rank) is not int or not 1 <= rank <= len(self.teams)
            for rank in self.previous_rank.values()
        ):
            raise ValueError("전년도 순위 범위 오류")

        if len(set(self.previous_rank.values())) != len(self.previous_rank):
            raise ValueError("전년도 확정 순위가 중복됩니다.")

    @property
    def games_per_team(self) -> int:
        """구단별 정규시즌 경기 수."""

        return (len(self.teams) - 1) * self.games_per_pair

    @property
    def total_games(self) -> int:
        """리그 전체 정규시즌 경기 수."""

        return len(self.teams) * self.games_per_team // 2


KBO_2025 = SeasonRules(
    season=2025,
    teams=tuple(sorted(TEAM_NAMES)),
    games_per_pair=16,
    playoff_slots=5,
    previous_rank=PREVIOUS_RANK_2025,
)


def get_rules(season: int) -> SeasonRules:
    """공식 적용 근거가 확인된 시즌 규칙을 반환한다."""

    if season == KBO_2025.season:
        return KBO_2025

    raise ValueError(f"지원하지 않는 시즌 규칙: {season}")


@dataclass
class Record:
    """구단별 승·패·무승부 및 누적 득실점."""

    wins: int = 0
    losses: int = 0
    ties: int = 0
    runs_for: int = 0
    runs_against: int = 0

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.ties

    @property
    def win_rate(self) -> Fraction | None:
        """무승부를 제외한 공식 승률을 정확한 분수로 반환한다."""

        decisions = self.wins + self.losses

        if decisions == 0:
            return None

        return Fraction(self.wins, decisions)

    def update(self, runs_for: int, runs_against: int) -> None:
        """경기 한 건의 성적을 누적한다."""

        self.runs_for += runs_for
        self.runs_against += runs_against

        if runs_for > runs_against:
            self.wins += 1
        elif runs_for < runs_against:
            self.losses += 1
        else:
            self.ties += 1


class SeasonState:
    """반복마다 독립적으로 관리되는 시즌 성적과 상대 전적."""

    def __init__(self, rules: SeasonRules) -> None:
        self.rules = rules

        self.records = {
            team: Record()
            for team in rules.teams
        }

        self.head_wins: Counter[tuple[int, int]] = Counter()
        self.head_runs: Counter[tuple[int, int]] = Counter()

        self.history: dict[int, list[tuple[date, int, int]]] = {
            team: []
            for team in rules.teams
        }

    def copy(self) -> "SeasonState":
        """기준 성적을 복제하여 반복 간 상태 공유를 차단한다."""

        result = SeasonState(self.rules)

        result.records = {
            team: replace(record)
            for team, record in self.records.items()
        }

        result.head_wins = self.head_wins.copy()
        result.head_runs = self.head_runs.copy()

        result.history = {
            team: games.copy()
            for team, games in self.history.items()
        }

        return result

    def add(self, game: PlayedGame) -> None:
        """완료된 실제 또는 모의 경기 결과를 반영한다."""

        fixture = game.fixture

        sides = (
            (
                fixture.home,
                fixture.away,
                game.home_runs,
                game.away_runs,
            ),
            (
                fixture.away,
                fixture.home,
                game.away_runs,
                game.home_runs,
            ),
        )

        for team, opponent, scored, allowed in sides:
            if team not in self.records:
                raise ValueError(f"지원하지 않는 구단: {team}")

            if (
                self.history[team]
                and self.history[team][-1][0] > fixture.day
            ):
                raise ValueError("경기 결과의 날짜 순서가 잘못되었습니다.")

            self.records[team].update(scored, allowed)

            self.head_wins[team, opponent] += int(scored > allowed)
            self.head_runs[team, opponent] += scored

            self.history[team].append(
                (fixture.day, scored, allowed)
            )


@dataclass(frozen=True)
class Standing:
    """순위·1위·진출의 개별 확정 여부."""

    rank: int | None
    first: bool | None
    qualifies: bool | None
    best_rank: int
    worst_rank: int
    reason: str | None = None


def fixture_order(fixture: Fixture) -> tuple[date, int, str]:
    """결과와 무관한 결정적 처리 순서."""

    return fixture.day, fixture.number, fixture.game_id


def validate_schedule(
    rules: SeasonRules,
    fixtures: Sequence[Fixture],
) -> list[dict[str, int]]:
    """경기 중복·충돌·편성 한도를 검사하고 미편성 수를 반환한다."""

    seen: set[str] = set()

    pairs: Counter[tuple[int, int]] = Counter()

    daily: dict[
        tuple[date, int],
        list[Fixture],
    ] = defaultdict(list)

    for game in fixtures:
        if game.game_id in seen:
            raise ValueError(f"중복 경기 ID: {game.game_id}")

        seen.add(game.game_id)

        if game.day.year != rules.season:
            raise ValueError(f"지원 시즌 밖의 경기: {game.game_id}")

        if game.home not in rules.teams or game.away not in rules.teams:
            raise ValueError(f"지원하지 않는 구단: {game.game_id}")

        pair = tuple(sorted((game.home, game.away)))
        pairs[pair] += 1

        if pairs[pair] > rules.games_per_pair:
            raise ValueError(f"구단 간 경기 수 초과: {pair}")

        daily[game.day, game.home].append(game)
        daily[game.day, game.away].append(game)

    for key, games in daily.items():
        if len(games) == 1:
            continue

        if len(games) != 2:
            raise ValueError(f"동일 구단의 일일 경기 충돌: {key}")

        first, second = games

        if (
            {first.number, second.number} != {1, 2}
            or first.home != second.home
            or first.away != second.away
        ):
            raise ValueError(f"더블헤더 편성 충돌: {key}")

    missing = []

    for first, second in combinations(sorted(rules.teams), 2):
        count = rules.games_per_pair - pairs[first, second]

        if count:
            missing.append({
                "team_a": first,
                "team_b": second,
                "missing_games": count,
            })

    return missing


def rate_bounds(
    record: Record,
    remaining: int,
) -> tuple[Fraction, Fraction]:
    """잔여 경기에서 가능한 승률의 보수적 하한·상한."""

    if remaining < 0:
        raise ValueError("잔여 경기 수가 음수입니다.")

    denominator = record.wins + record.losses + remaining

    if denominator == 0:
        return Fraction(0), Fraction(1)

    return (
        Fraction(record.wins, denominator),
        Fraction(record.wins + remaining, denominator),
    )


def bounded_standings(
    state: SeasonState,
) -> dict[int, Standing]:
    """불완전 일정에서도 수학적으로 확정 가능한 상태를 보존한다.

    다른 팀의 경기 결과 간 의존성을 모두 열거하지 않으므로
    계산된 범위는 보수적인 외부 경계다.
    """

    rules = state.rules

    bounds = {
        team: rate_bounds(
            record,
            rules.games_per_team - record.games,
        )
        for team, record in state.records.items()
    }

    standings = {}

    for team in rules.teams:
        lower, upper = bounds[team]

        guaranteed_above = sum(
            other_lower > upper
            for other, (other_lower, _) in bounds.items()
            if other != team
        )

        guaranteed_below = sum(
            other_upper < lower
            for other, (_, other_upper) in bounds.items()
            if other != team
        )

        best = guaranteed_above + 1
        worst = len(rules.teams) - guaranteed_below

        rank = best if best == worst else None

        first = (
            True if worst == 1
            else False if best > 1
            else None
        )

        qualifies = (
            True if worst <= rules.playoff_slots
            else False if best > rules.playoff_slots
            else None
        )

        standings[team] = Standing(
            rank=rank,
            first=first,
            qualifies=qualifies,
            best_rank=best,
            worst_rank=worst,
            reason=None if rank is not None
            else "remaining_games_or_undefined_win_rate",
        )

    return standings


def range_standing(
    first_rank: int,
    last_rank: int,
    slots: int,
    reason: str | None = None,
) -> Standing:
    """순위 가능 범위에서 1위 및 진출 여부를 독립 판정한다."""

    return Standing(
        rank=first_rank if first_rank == last_rank else None,
        first=(
            True if last_rank == 1
            else False if first_rank > 1
            else None
        ),
        qualifies=(
            True if last_rank <= slots
            else False if first_rank > slots
            else None
        ),
        best_rank=first_rank,
        worst_rank=last_rank,
        reason=reason,
    )


def resolve_tied_group(
    state: SeasonState,
    group: list[int],
    start: int,
) -> dict[int, Standing]:
    """공식 동률 규칙으로 한 승률 동률 집단을 처리한다."""

    rules = state.rules
    slots = rules.playoff_slots

    end = start + len(group) - 1

    if len(group) == 1:
        return {
            group[0]: range_standing(start, start, slots)
        }

    # 1위 또는 진출 경계의 정확히 두 팀 동률은
    # 결정전 결과 없이 최종 순서를 확정하지 않는다.
    if len(group) == 2 and start in (1, slots):
        return {
            team: range_standing(
                start,
                end,
                slots,
                "separate_tiebreak_game_required",
            )
            for team in group
        }

    # 진출권 밖에서 시작하는 동률 집단은 공동 순위다.
    if start > slots:
        return {
            team: range_standing(start, start, slots)
            for team in group
        }

    # 동률 집단 내부의 전체 상대 전적 다승·다득점.
    mini_table = {
        team: (
            sum(
                state.head_wins[team, other]
                for other in group
                if other != team
            ),
            sum(
                state.head_runs[team, other]
                for other in group
                if other != team
            ),
        )
        for team in group
    }

    ordered = sorted(
        group,
        key=mini_table.__getitem__,
        reverse=True,
    )

    standings: dict[int, Standing] = {}
    position = start

    for _, members in groupby(ordered, key=mini_table.__getitem__):
        subgroup = list(members)

        if len(subgroup) == 1:
            standings[subgroup[0]] = range_standing(
                position,
                position,
                slots,
            )

        elif all(team in rules.previous_rank for team in subgroup):
            previous_order = sorted(
                subgroup,
                key=rules.previous_rank.__getitem__,
            )

            for offset, team in enumerate(previous_order):
                rank = position + offset

                standings[team] = range_standing(
                    rank,
                    rank,
                    slots,
                )

        else:
            last = position + len(subgroup) - 1

            for team in subgroup:
                standings[team] = range_standing(
                    position,
                    last,
                    slots,
                    "previous_rank_unavailable",
                )

        position += len(subgroup)

    return standings


def resolve_standings(
    state: SeasonState,
) -> dict[int, Standing]:
    """완결된 시즌 순위 또는 미완결 상태의 확정 범위를 반환한다."""

    rules = state.rules

    if any(
        record.games > rules.games_per_team
        for record in state.records.values()
    ):
        raise ValueError("구단별 정규시즌 경기 수 초과")

    complete = all(
        record.games == rules.games_per_team
        for record in state.records.values()
    )

    rates = {
        team: record.win_rate
        for team, record in state.records.items()
    }

    if not complete or any(rate is None for rate in rates.values()):
        return bounded_standings(state)

    valid_rates = {
        team: rate
        for team, rate in rates.items()
        if rate is not None
    }

    ordered = sorted(
        rules.teams,
        key=valid_rates.__getitem__,
        reverse=True,
    )

    standings: dict[int, Standing] = {}
    position = 1

    for _, members in groupby(
        ordered,
        key=valid_rates.__getitem__,
    ):
        group = list(members)

        standings.update(
            resolve_tied_group(state, group, position)
        )

        position += len(group)

    return standings