"""2025년 KBO 포스트시즌 경기 및 시리즈 규칙."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from . import MAX_SERIES_GAMES, SUPPORTED_SEASON


Outcome = Literal["home", "tie", "away"]

OUTCOMES = ("home", "tie", "away")

RULE_VERSION = "kbo-2025-postseason-v1"

RULE_SOURCE = (
    "https://6ptotvmi5753.edge.naverncp.com/KBO_FILE/ebook/pdf/"
    "2025_%EB%A6%AC%EA%B7%B8%EA%B7%9C%EC%A0%95.pdf"
)

OPERATION_SOURCE = (
    "https://www.koreabaseball.com/MediaNews/Notice/"
    "View.aspx?bdSe=11694"
)

OPERATION_PUBLICATION = "2025-09-28"

EXTRA_INNING_LIMIT = 15


@dataclass(frozen=True)
class StageRule:
    """한 라운드의 진입 시드와 경기 편성 규칙."""

    stage: str
    seed_index: int
    wins_needed: int
    home_order: tuple[int, ...]
    articles: tuple[int, ...]

    @property
    def is_wildcard(self) -> bool:
        """와일드카드의 별도 진출 조건 적용 여부."""
        return self.stage == "wildCard"


RULES = (
    StageRule(
        "wildCard",
        3,
        2,
        (0, 0),
        (30, 31, 32, 33),
    ),
    StageRule(
        "semiPlayoff",
        2,
        3,
        (0, 0, 1, 1, 0),
        (34, 35, 36, 37),
    ),
    StageRule(
        "playoff",
        1,
        3,
        (0, 0, 1, 1, 0),
        (38, 39, 40, 41),
    ),
    StageRule(
        "koreanSeries",
        0,
        4,
        (0, 0, 1, 1, 1, 0, 0),
        (42, 43, 44, 46),
    ),
)

STAGES = tuple(rule.stage for rule in RULES)


def get_rules(season: int) -> tuple[StageRule, ...]:
    """공식 적용 근거가 확인된 시즌의 규칙을 반환한다."""
    if type(season) is not int or season != SUPPORTED_SEASON:
        raise ValueError(f"지원하지 않는 포스트시즌: {season}")

    return RULES


@dataclass(frozen=True)
class Game:
    """시리즈의 실제 기록 또는 모의 경기 결과."""

    number: int
    home: int
    away: int
    outcome: Outcome
    game_id: str | None = None
    day: str | None = None
    home_score: int | None = None
    away_score: int | None = None

    def __post_init__(self) -> None:
        if type(self.number) is not int or self.number < 1:
            raise ValueError("시리즈 차전은 양의 정수여야 합니다.")

        if (
            type(self.home) is not int
            or type(self.away) is not int
            or self.home == self.away
        ):
            raise ValueError("홈·원정 구단 식별자 오류")

        if self.outcome not in OUTCOMES:
            raise ValueError("지원하지 않는 경기 결과")

        if self.game_id is None:
            if any(
                value is not None
                for value in (
                    self.day,
                    self.home_score,
                    self.away_score,
                )
            ):
                raise ValueError("모의 경기의 실제 기록 혼입")

            return

        if not self.game_id or not self.day:
            raise ValueError("실제 경기 식별자 또는 날짜 누락")

        home = self.home_score
        away = self.away_score

        if (
            type(home) is not int
            or type(away) is not int
            or home < 0
            or away < 0
        ):
            raise ValueError("실제 경기 점수 오류")

        expected = (
            "home"
            if home > away
            else "away"
            if away > home
            else "tie"
        )

        if self.outcome != expected:
            raise ValueError("득점과 경기 결과 불일치")

    @property
    def winner(self) -> int | None:
        """승리 구단 ID를 반환하며 무승부는 None이다."""
        if self.outcome == "home":
            return self.home

        if self.outcome == "away":
            return self.away

        return None

    @property
    def observed(self) -> bool:
        """실제 자료에서 인수한 경기인지 확인한다."""
        return self.game_id is not None


@dataclass
class SeriesState:
    """한 대진의 경기 진행 및 종료 상태."""

    rule: StageRule
    higher: int
    lower: int
    games: list[Game] = field(default_factory=list, init=False)
    higher_wins: int = field(default=0, init=False)
    lower_wins: int = field(default=0, init=False)
    ties: int = field(default=0, init=False)
    winner: int | None = field(default=None, init=False)
    pending_homes: list[int] = field(
        default_factory=list,
        init=False,
    )

    def __post_init__(self) -> None:
        if self.higher == self.lower:
            raise ValueError("동일 구단으로 시리즈를 구성할 수 없습니다.")

    @property
    def participants(self) -> tuple[int, int]:
        """상위·하위 시드 순서의 참가 구단."""
        return self.higher, self.lower

    @property
    def unresolved_reason(self) -> str | None:
        """공식 순서가 불명확하거나 안전 한도에 도달한 사유."""
        if self.winner is not None:
            return None

        if len(self.games) >= MAX_SERIES_GAMES:
            return "simulation_game_limit"

        if len(self.games) < len(self.rule.home_order):
            return None

        if not self.pending_homes:
            return "series_without_winner_or_remaining_game"

        # 복수 무승부가 서로 다른 구장에서 발생했을 때
        # 추가 경기의 구장별 진행 우선순위를 임의로 확정하지 않는다.
        if len(set(self.pending_homes)) > 1:
            return "multiple_tie_venues_order_unverified"

        return None

    def next_home(self) -> int | None:
        """다음 차전 홈 구단을 반환한다."""
        if self.winner is not None:
            return None

        if self.unresolved_reason is not None:
            return None

        index = len(self.games)

        if index < len(self.rule.home_order):
            side = self.rule.home_order[index]
            return self.participants[side]

        return self.pending_homes[0]

    def apply(self, game: Game) -> None:
        """한 경기를 적용하고 시리즈 승수를 갱신한다."""
        if self.winner is not None:
            raise ValueError("종료된 시리즈에 경기를 추가할 수 없습니다.")

        expected_home = self.next_home()

        if expected_home is None:
            raise ValueError(
                self.unresolved_reason or "다음 경기 미확정"
            )

        if game.number != len(self.games) + 1:
            raise ValueError("시리즈 차전 누락 또는 중복")

        if {game.home, game.away} != set(self.participants):
            raise ValueError("시리즈 참가 구단 불일치")

        if game.home != expected_home:
            raise ValueError("시리즈 공식 홈 배정 불일치")

        if game.observed:
            if any(
                existing.game_id == game.game_id
                for existing in self.games
            ):
                raise ValueError("완료 경기 ID 중복")

            if (
                self.games
                and self.games[-1].day is not None
                and game.day <= self.games[-1].day
            ):
                raise ValueError("실제 시리즈 경기 날짜 순서 오류")

            if any(not item.observed for item in self.games):
                raise ValueError("모의 경기 뒤 실제 경기 삽입 금지")

        extra = len(self.games) >= len(self.rule.home_order)

        if extra:
            self.pending_homes.pop(0)

        self.games.append(game)

        if game.winner == self.higher:
            self.higher_wins += 1

        elif game.winner == self.lower:
            self.lower_wins += 1

        else:
            self.ties += 1

            if not self.rule.is_wildcard:
                self.pending_homes.append(game.home)

        if self.rule.is_wildcard:
            self._resolve_wildcard()
            return

        if self.higher_wins >= self.rule.wins_needed:
            self.winner = self.higher

        elif self.lower_wins >= self.rule.wins_needed:
            self.winner = self.lower

    def _resolve_wildcard(self) -> None:
        """4위 어드밴티지를 실제 승수와 구별하여 적용한다."""
        if self.higher_wins >= 1 or self.ties >= 1:
            self.winner = self.higher

        elif self.lower_wins >= 2:
            self.winner = self.lower


def replay_series(
    rule: StageRule,
    higher: int,
    lower: int,
    games: Sequence[Game],
) -> SeriesState:
    """이미 완료된 경기를 시리즈 규칙에 따라 재생한다."""
    state = SeriesState(rule, higher, lower)

    for game in games:
        if not game.observed:
            raise ValueError("완료 경기 재생에 모의 결과 혼입")

        state.apply(game)

    return state