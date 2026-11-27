"""기존 경기 예측을 재사용하는 보조 분석 계산 로직."""

from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from math import fsum, isclose, isfinite

from ...models.game_outcome import model as m1
from ..regular_season.data import (
    Scenario,
    cutoff_timestamp,
    fixture_to_dict,
)
from ..regular_season.engine import build_state, inputs_for
from ..regular_season.rules import (
    Fixture,
    PlayedGame,
    Record,
    validate_schedule,
)

from . import (
    DRAW_DIFFICULTY_WEIGHT,
    MAX_STARTER_FORECAST_DAYS,
    MIN_TEAM_SAMPLE,
    PROBABILITY_TOLERANCE,
    RECENT_DAYS,
    ROTATION_INTERVAL_DAYS,
    SCHEDULE_RULE_VERSION,
    SCHEMA,
    STARTER_HISTORY_DAYS,
    STARTER_RULE_VERSION,
    TEAM_RULE_VERSION,
    TOP_K,
    WORKLOAD_DAYS,
)
from .data import StarterData


@dataclass(frozen=True)
class Candidate:
    """선발 확률이 아닌 관측 근거를 보존하는 순위 후보."""

    player_id: str
    name: str
    team_id: int
    starts_last_28d: int
    previous_start_date: str
    days_since_start: int
    projected_start_interval: int
    previous_observed_appearance_date: str
    days_since_observed_appearance: int
    projected_appearance_interval: int
    observed_pitches_last_7d: int | None
    workload_observed_games: int
    workload_missing_pitch_counts: int
    last_team_evidence_date: str


@dataclass
class StarterRanking:
    """공통 후보 집합에서 비교하는 두 가지 순위."""

    baseline: tuple[Candidate, ...]
    proposed: tuple[Candidate, ...]
    exclusions: dict[str, int]
    observed_starts: int


@dataclass(frozen=True)
class FixtureForecast:
    """검증된 잔여 경기와 해당 경기의 M1 출력."""

    fixture: Fixture
    forecast: m1.Forecast | None


def ratio(
    numerator: int | float,
    denominator: int,
) -> float | None:
    """분모가 없는 값은 실제 0과 구분한다."""

    return numerator / denominator if denominator else None


def baseline_key(
    candidate: Candidate,
) -> tuple[int, str, str]:
    """이전 선발 이후 가장 오래 경과한 선수를 우선한다."""

    return (
        -candidate.projected_start_interval,
        candidate.previous_start_date,
        candidate.player_id,
    )


def proposed_key(
    candidate: Candidate,
) -> tuple[int, int, bool, int, int, str]:
    """짧은 등판 간격과 로테이션 간격을 우선 고려한다.

    최근 투구 수는 관측값일 때만 사용한다.
    결측 자료를 투구 수 0으로 해석하지 않는다.
    """

    short_rest_penalty = max(
        0,
        2 - candidate.projected_appearance_interval,
    )

    rotation_distance = abs(
        candidate.projected_start_interval
        - ROTATION_INTERVAL_DAYS
    )

    pitches = candidate.observed_pitches_last_7d

    return (
        short_rest_penalty,
        rotation_distance,
        pitches is None,
        pitches if pitches is not None else 0,
        -candidate.starts_last_28d,
        candidate.player_id,
    )


def rank_starters(
    data: StarterData,
    team_id: int,
    cutoff: date,
    target_day: date,
) -> StarterRanking:
    """기준일 이전 기록으로 예상 선발 후보를 생성한다.

    대상 경기일까지 추가 등판이 없다는 조건부 간격을
    순위 계산에 사용한다. 실제 미래 휴식을 확정하지 않는다.
    """

    if target_day < cutoff:
        raise ValueError("예상 선발 대상일이 기준일보다 빠릅니다.")

    history_start = cutoff - timedelta(
        days=STARTER_HISTORY_DAYS
    )

    workload_start = cutoff - timedelta(
        days=WORKLOAD_DAYS
    )

    exclusions: Counter[str] = Counter()
    candidates: list[Candidate] = []
    observed_starts = 0

    for player_id, all_starts in data.starts.items():
        starts = [
            row
            for row in all_starts
            if row.day < cutoff
        ]

        recent = [
            row
            for row in starts
            if row.team_id == team_id
            and row.day >= history_start
        ]

        if not recent:
            continue

        observed_starts += len(recent)

        identities = [
            row
            for row in data.identities.get(player_id, ())
            if row.day < cutoff
        ]

        appearances = [
            row
            for row in data.appearances.get(player_id, ())
            if row.day < cutoff
        ]

        names = {
            row.name for row in starts
        } | {
            row.name for row in identities
        }

        if len(names) != 1:
            exclusions["identity_name_conflict"] += 1
            continue

        team_identities = [
            row
            for row in identities
            if row.team_id == team_id
        ]

        if not team_identities:
            exclusions["team_identity_not_observed"] += 1
            continue

        affiliations = [
            (row.day, row.team_id)
            for row in starts
        ] + [
            (row.day, row.team_id)
            for row in identities
        ] + [
            (row.day, row.team_id)
            for row in appearances
        ]

        last_seen = max(day for day, _ in affiliations)

        latest_teams = {
            team
            for day, team in affiliations
            if day == last_seen
        }

        if len(latest_teams) != 1:
            exclusions["latest_team_conflict"] += 1
            continue

        if team_id not in latest_teams:
            exclusions["latest_team_mismatch"] += 1
            continue

        if last_seen < history_start:
            exclusions["team_evidence_stale"] += 1
            continue

        previous_start = max(
            row.day for row in recent
        )

        # 확인된 선발 역시 실제 등판 관측에 해당한다.
        appearance_days = [
            row.day for row in starts
        ] + [
            row.day for row in appearances
        ]

        previous_appearance = max(appearance_days)

        workload_rows = [
            row
            for row in appearances
            if row.day >= workload_start
        ]

        missing_pitches = sum(
            row.observed_pitches is None
            for row in workload_rows
        )

        observed_pitches = None

        if workload_rows and missing_pitches == 0:
            observed_pitches = sum(
                row.observed_pitches
                for row in workload_rows
                if row.observed_pitches is not None
            )

        candidates.append(
            Candidate(
                player_id=player_id,
                name=next(iter(names)),
                team_id=team_id,
                starts_last_28d=len(recent),
                previous_start_date=previous_start.isoformat(),
                days_since_start=(
                    cutoff - previous_start
                ).days,
                projected_start_interval=(
                    target_day - previous_start
                ).days,
                previous_observed_appearance_date=(
                    previous_appearance.isoformat()
                ),
                days_since_observed_appearance=(
                    cutoff - previous_appearance
                ).days,
                projected_appearance_interval=(
                    target_day - previous_appearance
                ).days,
                observed_pitches_last_7d=observed_pitches,
                workload_observed_games=len(workload_rows),
                workload_missing_pitch_counts=missing_pitches,
                last_team_evidence_date=last_seen.isoformat(),
            )
        )

    return StarterRanking(
        baseline=tuple(
            sorted(candidates, key=baseline_key)
        ),
        proposed=tuple(
            sorted(candidates, key=proposed_key)
        ),
        exclusions=dict(sorted(exclusions.items())),
        observed_starts=observed_starts,
    )


def starter_report(
    data: StarterData,
    team_id: int,
    fixture: Fixture,
    cutoff: date,
    completed: Sequence[PlayedGame],
    next_day: date,
) -> dict[str, object]:
    """예상 후보와 자료 지원 범위를 경기별로 표현한다."""

    history_start = cutoff - timedelta(
        days=STARTER_HISTORY_DAYS
    )

    team_games = sum(
        history_start <= game.fixture.day < cutoff
        and team_id in (
            game.fixture.home,
            game.fixture.away,
        )
        for game in completed
    )

    result: dict[str, object] = {
        "team_id": team_id,
        "rule_version": STARTER_RULE_VERSION,
        "as_of_kst": cutoff_timestamp(cutoff),
        "target_date": fixture.day.isoformat(),
        "history_start_inclusive": history_start.isoformat(),
        "history_end_exclusive": cutoff.isoformat(),
        "history_games": team_games,
        "official_announcement": None,
        "official_announcement_status": (
            "publication_evidence_unavailable"
        ),
        "candidate_limit": TOP_K,
        "status": "unsupported",
        "reason": None,
        "candidates": [],
        "candidate_count": 0,
        "eligible_candidate_count": 0,
        "observed_starts": None,
        "starter_history_coverage": None,
        "exclusions": {},
        "ranking_method": (
            "short_rest_rotation_workload_starts_player_id"
        ),
        "limitations": [
            "observed_affiliation_is_not_current_roster",
            "observed_pitches_are_not_total_workload",
            "future_appearances_are_not_known",
            "rank_is_not_probability",
            "same_day_starters_not_jointly_assigned",
        ],
    }

    if fixture.day != next_day:
        result["reason"] = "not_next_team_game_date"
        return result

    if (
        fixture.day - cutoff
    ).days > MAX_STARTER_FORECAST_DAYS:
        result["reason"] = "starter_forecast_horizon_exceeded"
        return result

    ranking = rank_starters(
        data,
        team_id,
        cutoff,
        fixture.day,
    )

    if ranking.observed_starts > team_games:
        raise ValueError("선발 이력과 팀 경기 수 불일치")

    result.update({
        "observed_starts": ranking.observed_starts,
        "starter_history_coverage": ratio(
            ranking.observed_starts,
            team_games,
        ),
        "exclusions": ranking.exclusions,
        "eligible_candidate_count": len(ranking.proposed),
        "candidate_count": min(
            len(ranking.proposed),
            TOP_K,
        ),
    })

    if not ranking.proposed:
        result["reason"] = "insufficient_eligible_starter_history"
        return result

    result["status"] = "estimated"

    if (
        ranking.observed_starts < team_games
        or ranking.exclusions
        or len(ranking.proposed) < TOP_K
    ):
        result["status"] = "estimated_limited"
        result["reason"] = "partial_starter_evidence"

    result["candidates"] = [
        {
            "rank": rank,
            **asdict(candidate),
        }
        for rank, candidate in enumerate(
            ranking.proposed[:TOP_K],
            start=1,
        )
    ]

    return result


def record_summary(
    record: Record,
) -> dict[str, object]:
    """기존 Record의 집계와 지표별 분모를 함께 반환한다."""

    games = record.games
    decisions = record.wins + record.losses

    status = "supported"

    if games == 0:
        status = "no_observations"
    elif games < MIN_TEAM_SAMPLE:
        status = "limited_sample"

    return {
        "status": status,
        "reason": (
            None
            if status == "supported"
            else status
        ),
        "games": games,
        "wins": record.wins,
        "losses": record.losses,
        "ties": record.ties,
        "runs_for": record.runs_for,
        "runs_against": record.runs_against,
        "run_difference": (
            record.runs_for - record.runs_against
        ),
        "decisions": decisions,
        "win_rate": ratio(
            record.wins,
            decisions,
        ),
        "runs_for_per_game": ratio(
            record.runs_for,
            games,
        ),
        "runs_against_per_game": ratio(
            record.runs_against,
            games,
        ),
        "run_difference_per_game": ratio(
            record.runs_for - record.runs_against,
            games,
        ),
        "win_rate_denominator": decisions,
        "run_rate_denominator": games,
    }


def team_window(
    team_id: int,
    opponents: Sequence[int],
    completed: Sequence[PlayedGame],
    start: date,
    end: date,
) -> dict[str, object]:
    """기간·장소·상대별 완료 경기 기록을 집계한다."""

    total = Record()
    home = Record()
    away = Record()

    versus = {
        opponent: Record()
        for opponent in opponents
        if opponent != team_id
    }

    seen: set[str] = set()

    for game in completed:
        fixture = game.fixture

        if fixture.game_id in seen:
            raise ValueError(f"완료 경기 중복: {fixture.game_id}")

        seen.add(fixture.game_id)

        if not start <= fixture.day < end:
            continue

        if team_id == fixture.home:
            scored = game.home_runs
            allowed = game.away_runs
            opponent = fixture.away
            venue_record = home

        elif team_id == fixture.away:
            scored = game.away_runs
            allowed = game.home_runs
            opponent = fixture.home
            venue_record = away

        else:
            continue

        if opponent not in versus:
            raise ValueError(f"등록되지 않은 상대 구단: {opponent}")

        total.update(scored, allowed)
        venue_record.update(scored, allowed)
        versus[opponent].update(scored, allowed)

    if total.games != home.games + away.games:
        raise ValueError("홈·원정 경기 수 검산 실패")

    if total.games != sum(
        row.games for row in versus.values()
    ):
        raise ValueError("상대별 경기 수 검산 실패")

    return {
        "start_inclusive": start.isoformat(),
        "end_exclusive": end.isoformat(),
        "overall": record_summary(total),
        "home": record_summary(home),
        "away": record_summary(away),
        "opponents": {
            str(opponent): record_summary(record)
            for opponent, record in sorted(
                versus.items()
            )
        },
    }


def team_profile(
    team_id: int,
    teams: Sequence[int],
    completed: Sequence[PlayedGame],
    cutoff: date,
) -> dict[str, object]:
    """동일 시점의 시즌 누적 및 최근 경기력을 반환한다."""

    season_start = date(cutoff.year, 1, 1)

    recent_start = max(
        season_start,
        cutoff - timedelta(days=RECENT_DAYS),
    )

    return {
        "team_id": team_id,
        "rule_version": TEAM_RULE_VERSION,
        "as_of_kst": cutoff_timestamp(cutoff),
        "season": team_window(
            team_id,
            teams,
            completed,
            season_start,
            cutoff,
        ),
        "recent": team_window(
            team_id,
            teams,
            completed,
            recent_start,
            cutoff,
        ),
        "minimum_sample_reference": MIN_TEAM_SAMPLE,
        "venue_method": "separate_observed_records",
        "opponent_method": "separate_observed_records",
    }


def team_probabilities(
    forecast: m1.Forecast,
    fixture: Fixture,
    team_id: int,
) -> dict[str, float]:
    """기존 M1의 홈·무·원정 확률을 팀 관점으로 변환한다."""

    probabilities = forecast["result_probabilities"]

    if set(probabilities) != {"home", "tie", "away"}:
        raise ValueError("M1 승무패 확률 키 오류")

    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or not 0 <= value <= 1
        for value in probabilities.values()
    ):
        raise ValueError("M1 확률 범위 오류")

    if not isclose(
        fsum(probabilities.values()),
        1.0,
        rel_tol=0.0,
        abs_tol=PROBABILITY_TOLERANCE,
    ):
        raise ValueError("M1 승무패 확률 합 오류")

    if team_id == fixture.home:
        win = probabilities["home"]
        loss = probabilities["away"]

    elif team_id == fixture.away:
        win = probabilities["away"]
        loss = probabilities["home"]

    else:
        raise ValueError("경기 참가 팀이 아닙니다.")

    return {
        "win": win,
        "tie": probabilities["tie"],
        "loss": loss,
    }


def difficulty_value(
    probabilities: dict[str, float],
) -> float:
    """패배 1, 무승부 0.5, 승리 0의 기대 손실을 계산한다."""

    return (
        probabilities["loss"]
        + DRAW_DIFFICULTY_WEIGHT * probabilities["tie"]
    )


def game_difficulty(
    row: FixtureForecast,
    team_id: int,
) -> dict[str, object]:
    """경기별 난이도 및 상대·장소를 보존한다."""

    fixture = row.fixture

    if team_id == fixture.home:
        venue = "home"
        opponent = fixture.away

    elif team_id == fixture.away:
        venue = "away"
        opponent = fixture.home

    else:
        raise ValueError("난이도 대상 팀과 경기 불일치")

    result: dict[str, object] = {
        "game_id": fixture.game_id,
        "date": fixture.day.isoformat(),
        "team_id": team_id,
        "opponent_team_id": opponent,
        "venue": venue,
        "probabilities": None,
        "difficulty": None,
        "status": "unsupported",
        "reason": "forecast_missing",
        "weight": 1,
    }

    if row.forecast is None:
        return result

    probabilities = team_probabilities(
        row.forecast,
        fixture,
        team_id,
    )

    result.update({
        "probabilities": probabilities,
        "difficulty": difficulty_value(probabilities),
        "status": "supported",
        "reason": None,
    })

    return result


def schedule_difficulty(
    team_id: int,
    forecasts: Sequence[FixtureForecast],
    missing_games: int,
    *,
    publication_verified: bool,
) -> dict[str, object]:
    """잔여 경기의 난이도를 같은 가중치로 평균한다."""

    if type(missing_games) is not int or missing_games < 0:
        raise ValueError("미편성 경기 수 오류")

    values: list[float] = []
    rows: list[dict[str, object]] = []
    seen: set[str] = set()

    for forecast in forecasts:
        fixture = forecast.fixture

        if team_id not in (fixture.home, fixture.away):
            continue

        if fixture.game_id in seen:
            raise ValueError("잔여 일정 경기 중복")

        seen.add(fixture.game_id)

        row = game_difficulty(forecast, team_id)
        rows.append(row)

        value = row["difficulty"]

        if isinstance(value, (int, float)):
            values.append(float(value))

    scheduled = len(rows)
    valid = len(values)
    missing_forecasts = scheduled - valid
    expected = scheduled + missing_games

    if expected == 0:
        status = "no_remaining_games_under_assumption"

    elif scheduled == 0:
        status = "schedule_missing"

    elif valid == 0:
        status = "no_valid_forecast"

    elif missing_forecasts or missing_games:
        status = "partial"

    else:
        status = "supported_under_assumption"

    return {
        "team_id": team_id,
        "rule_version": SCHEDULE_RULE_VERSION,
        "status": status,
        "reason": (
            None
            if status == "supported_under_assumption"
            else status
        ),
        "expected_remaining_games": expected,
        "valid_scheduled_games": scheduled,
        "independently_confirmed_schedule_games": (
            scheduled if publication_verified else 0
        ),
        "unverified_schedule_games": (
            0 if publication_verified else scheduled
        ),
        "unassigned_games": missing_games,
        "forecast_games": valid,
        "missing_forecasts": missing_forecasts,
        "forecast_coverage": ratio(valid, scheduled),
        "expected_schedule_coverage": ratio(
            valid,
            expected,
        ),
        "difficulty_sum": fsum(values) if valid else None,
        "difficulty": ratio(fsum(values), valid),
        "denominator": valid,
        "draw_weight": DRAW_DIFFICULTY_WEIGHT,
        "weighting": "equal_per_valid_fixture",
        "games": rows,
    }


def forecast_remaining(
    scenario: Scenario,
) -> tuple[FixtureForecast, ...]:
    """기준일 관측 상태를 고정한 M1 잔여 경기 전망을 계산한다.

    기존 시즌 시뮬레이션의 동적 반복 계산을 변경하지 않는다.
    미래 경기 결과를 임의 생성하여 여기에 입력하지 않는다.
    """

    state = build_state(
        scenario.rules,
        scenario.completed,
    )

    rows: list[FixtureForecast] = []

    for fixture in scenario.remaining:
        if fixture.day < scenario.cutoff:
            raise ValueError("기준일 이전 잔여 일정 혼입")

        inputs = inputs_for(state, fixture)

        forecast = m1.predict(
            scenario.model,
            inputs,
        )

        rows.append(
            FixtureForecast(fixture, forecast)
        )

    return tuple(rows)


def analyze(
    scenario: Scenario,
    data: StarterData,
    game_id: str | None = None,
) -> dict[str, object]:
    """경기 또는 시즌 기준의 통합 보조 분석을 생성한다."""

    cutoff = scenario.cutoff
    teams = scenario.rules.teams

    if any(
        game.fixture.day >= cutoff
        for game in scenario.completed
    ):
        raise ValueError("기준일 이후 결과 혼입")

    missing_pairs = validate_schedule(
        scenario.rules,
        [
            *(game.fixture for game in scenario.completed),
            *scenario.remaining,
        ],
    )

    missing_by_team: Counter[int] = Counter()

    for row in missing_pairs:
        missing_by_team[row["team_a"]] += row["missing_games"]
        missing_by_team[row["team_b"]] += row["missing_games"]

    forecasts = forecast_remaining(scenario)

    if game_id is not None and not any(
        row.fixture.game_id == game_id
        for row in forecasts
    ):
        raise ValueError("대상 경기가 유효 잔여 일정에 없습니다.")

    next_days: dict[int, date] = {}

    for row in forecasts:
        fixture = row.fixture

        for team in (fixture.home, fixture.away):
            next_days[team] = min(
                fixture.day,
                next_days.get(team, fixture.day),
            )

    selected = [
        row
        for row in forecasts
        if game_id is None
        or row.fixture.game_id == game_id
    ]

    selected_teams = (
        set(teams)
        if game_id is None
        else {
            team
            for row in selected
            for team in (
                row.fixture.home,
                row.fixture.away,
            )
        }
    )

    game_reports: list[dict[str, object]] = []

    for row in selected:
        fixture = row.fixture

        game_reports.append({
            "fixture": fixture_to_dict(fixture),
            "m1_forecast": row.forecast,
            "starters": {
                str(team): starter_report(
                    data,
                    team,
                    fixture,
                    cutoff,
                    scenario.completed,
                    next_days[team],
                )
                for team in (fixture.home, fixture.away)
            },
            "difficulty": {
                str(team): game_difficulty(row, team)
                for team in (fixture.home, fixture.away)
            },
        })

    team_reports = {
        str(team): {
            "profile": team_profile(
                team,
                teams,
                scenario.completed,
                cutoff,
            ),
            "schedule": schedule_difficulty(
                team,
                forecasts,
                missing_by_team[team],
                publication_verified=False,
            ),
        }
        for team in sorted(selected_teams)
    }

    return {
        "schema": SCHEMA,
        "scope": "game" if game_id is not None else "season",
        "season": scenario.rules.season,
        "as_of_kst": cutoff_timestamp(cutoff),
        "metadata": scenario.metadata,
        "prediction_method": (
            "existing_m1_fixed_observed_history"
        ),
        "scenario_status": "development_assumption",
        "formal_backtest_supported": False,
        "final_test_used": False,
        "games": game_reports,
        "teams": team_reports,
        "missing_pairs": missing_pairs,
        "rule_versions": {
            "starter": STARTER_RULE_VERSION,
            "team": TEAM_RULE_VERSION,
            "schedule": SCHEDULE_RULE_VERSION,
        },
        "new_primary_models": 0,
    }