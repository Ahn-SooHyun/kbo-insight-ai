"""예상 선발 계산에 필요한 시점 이전 자료를 읽기 전용으로 인수한다."""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
import sqlite3

from ..regular_season.data import parse_day
from ..regular_season.rules import Fixture, PlayedGame


@dataclass(frozen=True)
class Start:
    """식별자가 확인된 과거 선발 기록."""

    game_id: str
    day: date
    team_id: int
    player_id: str
    name: str
    evidence_count: int


@dataclass(frozen=True)
class Identity:
    """경기 기록에서 관측한 선수 이름과 소속 근거."""

    day: date
    team_id: int
    name: str


@dataclass(frozen=True)
class Appearance:
    """품질 조건을 충족한 과거 투수 경기 기록."""

    game_id: str
    day: date
    team_id: int
    observed_pitches: int | None


@dataclass
class StarterData:
    """선발·식별·등판 이력을 선수별로 보존한 자료."""

    starts: dict[str, tuple[Start, ...]]
    identities: dict[str, tuple[Identity, ...]]
    appearances: dict[str, tuple[Appearance, ...]]


def valid_id(value: object) -> bool:
    """빈 문자열이나 누락된 식별자를 후보로 사용하지 않는다."""

    return isinstance(value, str) and bool(value.strip())


def require_count(
    value: object,
    field: str,
    *,
    nullable: bool = False,
) -> int | None:
    """관측 횟수의 타입과 범위를 검사하며 결측을 보존한다."""

    if value is None and nullable:
        return None

    if type(value) is not int or value < 0:
        raise ValueError(f"잘못된 관측 횟수: {field}")

    return value


def check_side(
    fixture: Fixture,
    side: str,
    team_id: int,
    day: date,
) -> None:
    """선발 기록의 경기 날짜 및 홈·원정 소속을 검증한다."""

    if side not in ("home", "away"):
        raise ValueError(f"선발 홈·원정 구분 오류: {fixture.game_id}")

    expected = fixture.home if side == "home" else fixture.away

    if day != fixture.day or team_id != expected:
        raise ValueError(f"선발 경기 정보 불일치: {fixture.game_id}")


def completed_index(
    completed: Sequence[PlayedGame],
    cutoff: date,
) -> dict[str, Fixture]:
    """중복 및 기준일 이후 경기 혼입을 차단한다."""

    result: dict[str, Fixture] = {}

    for game in completed:
        fixture = game.fixture

        if fixture.day >= cutoff:
            raise ValueError("기준일 이후 완료 경기 혼입")

        if fixture.game_id in result:
            raise ValueError(f"완료 경기 중복: {fixture.game_id}")

        result[fixture.game_id] = fixture

    return result


def read_starts(
    db: sqlite3.Connection,
    fixtures: dict[str, Fixture],
    season: int,
    cutoff: date,
) -> dict[str, tuple[Start, ...]]:
    """식별자가 존재하는 이전 선발 이력을 읽는다."""

    rows = db.execute(
        """
        SELECT game_id, side, team_id, date,
               player_id, name, evidence_count
        FROM starter_history
        WHERE date >= ? AND date < ?
        ORDER BY date, game_id, side, player_id
        """,
        (
            date(season, 1, 1).isoformat(),
            cutoff.isoformat(),
        ),
    )

    grouped: dict[str, list[Start]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()

    for game_id, side, team, raw_day, player, name, count in rows:
        fixture = fixtures.get(game_id)

        # 인수 자료에는 포스트시즌 등 분석 범위 밖 기록도 존재한다.
        if fixture is None:
            continue

        day = parse_day(raw_day)
        check_side(fixture, side, team, day)

        key = (game_id, side)

        if key in seen:
            raise ValueError(f"선발 이력 중복: {key}")

        seen.add(key)

        if not valid_id(player) or not valid_id(name):
            raise ValueError(f"선발 이력의 식별 근거 누락: {key}")

        evidence_count = require_count(count, "evidence_count")

        if evidence_count is None or evidence_count == 0:
            raise ValueError(f"선발 원본 근거 누락: {key}")

        grouped[player].append(
            Start(
                game_id,
                day,
                team,
                player,
                name,
                evidence_count,
            )
        )

    return {
        player: tuple(records)
        for player, records in grouped.items()
    }


def read_identities(
    db: sqlite3.Connection,
    fixtures: dict[str, Fixture],
    season: int,
    cutoff: date,
) -> dict[str, tuple[Identity, ...]]:
    """기준일 이전의 이름 및 소속 관측 자료만 읽는다."""

    rows = db.execute(
        """
        SELECT player_id, team_id, name,
               first_seen_date, game_id, observed_season
        FROM player_identity_evidence
        WHERE season = ?
          AND first_seen_date < ?
        ORDER BY first_seen_date, player_id, team_id, name
        """,
        (season, cutoff.isoformat()),
    )

    grouped: dict[str, list[Identity]] = defaultdict(list)
    seen: set[tuple[str, int, str, date]] = set()

    for player, team, name, raw_day, game_id, observed_season in rows:
        fixture = fixtures.get(game_id)

        if fixture is None:
            continue

        day = parse_day(raw_day)

        if (
            observed_season != season
            or fixture.day != day
            or team not in (fixture.home, fixture.away)
        ):
            raise ValueError(f"선수 식별 근거 불일치: {game_id}")

        if not valid_id(player) or not valid_id(name):
            continue

        key = (player, team, name, day)

        # 같은 이름의 반복 관측은 충돌로 취급하지 않는다.
        if key in seen:
            continue

        seen.add(key)

        grouped[player].append(
            Identity(day, team, name)
        )

    return {
        player: tuple(records)
        for player, records in grouped.items()
    }


def read_appearances(
    db: sqlite3.Connection,
    fixtures: dict[str, Fixture],
    season: int,
    cutoff: date,
) -> dict[str, tuple[Appearance, ...]]:
    """시점별로 사용할 수 있는 투수 관측 자료를 인수한다.

    시즌 전체를 기준으로 확정된 최종 admitted 및
    identity_ambiguous 값은 사용하지 않는다.
    """

    rows = db.execute(
        """
        SELECT p.game_id, g.date, p.team_id,
               p.player_id, p.observed_pitches
        FROM player_games p
        JOIN player_admission a
          USING (game_id, team_id, player_id, role)
        JOIN eligible_games g USING (game_id)
        WHERE g.season = ?
          AND g.competition = 'regular'
          AND g.date < ?
          AND p.role = 'pitcher'
          AND a.preliminary_candidate = 1
          AND a.direct_mismatches = 0
          AND a.team_mismatches = 0
          AND a.game_state_uncertain = 0
        ORDER BY g.date, p.game_id, p.player_id
        """,
        (season, cutoff.isoformat()),
    )

    grouped: dict[str, list[Appearance]] = defaultdict(list)
    seen: set[tuple[str, str, int]] = set()

    for game_id, raw_day, team, player, pitches in rows:
        fixture = fixtures.get(game_id)

        if fixture is None:
            raise ValueError(f"완료 경기 집합 불일치: {game_id}")

        day = parse_day(raw_day)

        if day != fixture.day or team not in (
            fixture.home,
            fixture.away,
        ):
            raise ValueError(f"투수 기록 경기 정보 불일치: {game_id}")

        if not valid_id(player):
            continue

        key = (game_id, player, team)

        if key in seen:
            raise ValueError(f"투수 경기 기록 중복: {key}")

        seen.add(key)

        observed = require_count(
            pitches,
            "observed_pitches",
            nullable=True,
        )

        grouped[player].append(
            Appearance(game_id, day, team, observed)
        )

    return {
        player: tuple(records)
        for player, records in grouped.items()
    }


def read_starter_data(
    standard: sqlite3.Connection,
    model: sqlite3.Connection,
    completed: Sequence[PlayedGame],
    season: int,
    cutoff: date,
) -> StarterData:
    """같은 완료 경기 집합과 기준일에서 후보 입력을 구성한다."""

    fixtures = completed_index(completed, cutoff)

    return StarterData(
        starts=read_starts(
            model,
            fixtures,
            season,
            cutoff,
        ),
        identities=read_identities(
            model,
            fixtures,
            season,
            cutoff,
        ),
        appearances=read_appearances(
            standard,
            fixtures,
            season,
            cutoff,
        ),
    )
