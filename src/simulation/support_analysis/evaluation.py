"""2025년 검증 자료의 선발 규칙 비교 및 자료 범위 감사."""

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
import sqlite3

from ..regular_season.data import parse_day
from ..regular_season.rules import PlayedGame

from . import (
    SCHEMA,
    STARTER_RULE_VERSION,
    SUPPORTED_SEASON,
    TOP_K,
)
from .analysis import (
    StarterRanking,
    rank_starters,
    ratio,
)
from .data import (
    StarterData,
    check_side,
    require_count,
    valid_id,
)


@dataclass(frozen=True)
class Truth:
    """평가에만 사용하는 대상 경기 실제 선발 정답."""

    player_id: str | None
    reason: str | None


@dataclass(frozen=True)
class EvaluationSample:
    """두 규칙의 공통 평가 표본."""

    game_id: str
    team_id: int
    day: date
    truth: Truth
    baseline: tuple[str, ...]
    proposed: tuple[str, ...]


def read_truth(
    db: sqlite3.Connection,
    completed: Sequence[PlayedGame],
    end: date,
) -> dict[tuple[str, int], Truth]:
    """실제 선발 정답을 읽되 예상 후보 입력에는 전달하지 않는다."""

    fixtures = {
        game.fixture.game_id: game.fixture
        for game in completed
    }

    rows = db.execute(
        """
        SELECT game_id, side, team_id, date,
               player_id, name, evidence_count
        FROM starter_actual
        WHERE date >= ? AND date < ?
        ORDER BY date, game_id, side
        """,
        (
            date(SUPPORTED_SEASON, 1, 1).isoformat(),
            end.isoformat(),
        ),
    )

    result: dict[tuple[str, int], Truth] = {}

    for game_id, side, team, raw_day, player, name, count in rows:
        fixture = fixtures.get(game_id)

        if fixture is None:
            continue

        check_side(
            fixture,
            side,
            team,
            parse_day(raw_day),
        )

        key = (game_id, team)

        if key in result:
            raise ValueError(f"실제 선발 정답 중복: {key}")

        evidence = require_count(count, "evidence_count")

        if evidence is None or evidence == 0:
            result[key] = Truth(
                None,
                "truth_evidence_missing",
            )

        elif not valid_id(player):
            result[key] = Truth(
                None,
                "truth_player_id_unresolved",
            )

        elif not valid_id(name):
            result[key] = Truth(
                None,
                "truth_name_missing",
            )

        else:
            result[key] = Truth(player, None)

    return result


def read_candidate_reference(
    db: sqlite3.Connection,
    completed: Sequence[PlayedGame],
    end: date,
) -> dict[tuple[str, int], dict[str, str]]:
    """과거 후보표와 이전 선발일을 검증용으로 인수한다.

    이 자료를 예정 경기의 후보 입력에 재사용하지 않는다.
    """

    fixtures = {
        game.fixture.game_id: game.fixture
        for game in completed
    }

    rows = db.execute(
        """
        SELECT game_id, date, team_id, player_id,
               starts_last_28d, previous_start_date,
               days_since_start, previous_appearance_date,
               days_since_appearance, observed_pitches_last_7d
        FROM starter_pre_game
        WHERE date >= ? AND date < ?
        ORDER BY date, game_id, team_id, player_id
        """,
        (
            date(SUPPORTED_SEASON, 1, 1).isoformat(),
            end.isoformat(),
        ),
    )

    grouped: dict[
        tuple[str, int],
        dict[str, str],
    ] = defaultdict(dict)

    for (
        game_id,
        raw_day,
        team,
        player,
        starts,
        raw_previous,
        days,
        raw_appearance,
        appearance_days,
        pitches,
    ) in rows:
        fixture = fixtures.get(game_id)

        if fixture is None:
            continue

        day = parse_day(raw_day)
        previous = parse_day(raw_previous)

        if (
            fixture.day != day
            or team not in (fixture.home, fixture.away)
        ):
            raise ValueError("과거 선발 후보 경기 키 불일치")

        if (
            not valid_id(player)
            or previous >= day
            or days != (day - previous).days
            or not 1 <= days <= 28
        ):
            raise ValueError("과거 선발 후보 날짜 또는 ID 오류")

        start_count = require_count(
            starts,
            "starts_last_28d",
        )

        if start_count is None or start_count == 0:
            raise ValueError("선발 후보의 선발 표본 누락")

        if raw_appearance is None:
            if appearance_days is not None:
                raise ValueError("등판일 누락 및 간격 불일치")

        else:
            appearance = parse_day(raw_appearance)

            if (
                appearance >= day
                or appearance_days != (day - appearance).days
            ):
                raise ValueError("과거 후보 등판 정보 누수")

        require_count(
            pitches,
            "observed_pitches_last_7d",
            nullable=True,
        )

        key = (game_id, team)

        if player in grouped[key]:
            raise ValueError("과거 선발 후보 중복")

        grouped[key][player] = previous.isoformat()

    return dict(grouped)


def make_sample(
    game_id: str,
    team_id: int,
    day: date,
    truth: Truth,
    ranking: StarterRanking,
) -> EvaluationSample:
    """같은 대상·정답에서 두 규칙의 순위 후보를 기록한다."""

    return EvaluationSample(
        game_id=game_id,
        team_id=team_id,
        day=day,
        truth=truth,
        baseline=tuple(
            row.player_id
            for row in ranking.baseline[:TOP_K]
        ),
        proposed=tuple(
            row.player_id
            for row in ranking.proposed[:TOP_K]
        ),
    )


def method_metrics(
    samples: Sequence[EvaluationSample],
    method: str,
) -> dict[str, object]:
    """후보 제공 범위와 실제 정답 적중률을 별도로 계산한다."""

    if method not in ("baseline", "proposed"):
        raise ValueError("지원하지 않는 평가 방법")

    truth_count = 0
    provided = 0
    evaluable = 0
    top1_hits = 0
    topk_hits = 0
    candidate_total = 0
    candidate_sizes: Counter[int] = Counter()

    for sample in samples:
        candidates = (
            sample.baseline
            if method == "baseline"
            else sample.proposed
        )

        candidate_total += len(candidates)
        candidate_sizes[len(candidates)] += 1

        if candidates:
            provided += 1

        truth = sample.truth.player_id

        if truth is None:
            continue

        truth_count += 1

        if not candidates:
            continue

        evaluable += 1
        top1_hits += int(candidates[0] == truth)
        topk_hits += int(truth in candidates)

    return {
        "method": method,
        "team_game_samples": len(samples),
        "truth_available": truth_count,
        "candidate_provided": provided,
        "candidate_coverage": ratio(
            provided,
            len(samples),
        ),
        "evaluable_samples": evaluable,
        "evaluable_truth_coverage": ratio(
            evaluable,
            truth_count,
        ),
        "top1_hits": top1_hits,
        "top1_accuracy": ratio(
            top1_hits,
            evaluable,
        ),
        "top_k": TOP_K,
        "top_k_hits": topk_hits,
        "top_k_accuracy": ratio(
            topk_hits,
            evaluable,
        ),
        "top1_all_identified_truth": ratio(
            top1_hits,
            truth_count,
        ),
        "top_k_all_identified_truth": ratio(
            topk_hits,
            truth_count,
        ),
        "candidate_count_total": candidate_total,
        "mean_candidates_when_provided": ratio(
            candidate_total,
            provided,
        ),
        "candidate_size_distribution": {
            str(size): count
            for size, count in sorted(
                candidate_sizes.items()
            )
        },
    }


def compare_samples(
    samples: Sequence[EvaluationSample],
) -> dict[str, object]:
    """두 방법의 전체 및 공통 표본 성능을 계산한다."""

    common = [
        sample
        for sample in samples
        if sample.truth.player_id is not None
        and sample.baseline
        and sample.proposed
    ]

    exclusions: Counter[str] = Counter()

    for sample in samples:
        if sample.truth.player_id is None:
            exclusions[
                sample.truth.reason
                or "truth_unavailable"
            ] += 1

        elif not sample.baseline and not sample.proposed:
            exclusions["neither_method_has_candidates"] += 1

        elif not sample.baseline:
            exclusions["baseline_candidates_missing"] += 1

        elif not sample.proposed:
            exclusions["proposed_candidates_missing"] += 1

    return {
        "baseline": method_metrics(samples, "baseline"),
        "proposed": method_metrics(samples, "proposed"),
        "common_samples": len(common),
        "common_baseline": method_metrics(
            common,
            "baseline",
        ),
        "common_proposed": method_metrics(
            common,
            "proposed",
        ),
        "common_exclusions": dict(
            sorted(exclusions.items())
        ),
    }


def audit_starter_tables(
    db: sqlite3.Connection,
    end: date,
) -> dict[str, object]:
    """개발 기간의 시즌·팀별 선발 자료 확보 범위를 집계한다."""

    result: dict[str, object] = {}

    keys = {
        "starter_actual": (
            "game_id || ':' || side || ':' || team_id"
        ),
        "starter_history": (
            "game_id || ':' || side || ':' || team_id"
        ),
        "starter_pre_game": (
            "game_id || ':' || team_id || ':' || player_id"
        ),
    }

    # 테이블 이름과 집계 키는 내부 고정 목록에서만 사용한다.
    for table, key in keys.items():
        rows = db.execute(
            f"""
            SELECT substr(date, 1, 4), team_id,
                   COUNT(*),
                   COUNT(player_id),
                   COUNT(*) - COUNT(DISTINCT {key})
            FROM {table}
            WHERE date < ?
            GROUP BY substr(date, 1, 4), team_id
            ORDER BY substr(date, 1, 4), team_id
            """,
            (end.isoformat(),),
        )

        result[table] = [
            {
                "season": season,
                "team_id": team,
                "rows": count,
                "identified_rows": identified,
                "unidentified_rows": count - identified,
                "duplicate_or_null_keys": duplicate,
            }
            for season, team, count, identified, duplicate in rows
        ]

    conflicts = db.execute(
        """
        WITH grouped AS (
            SELECT substr(date, 1, 4) season,
                   team_id, game_id, side,
                   COUNT(DISTINCT name) names,
                   COUNT(DISTINCT player_id) identifiers
            FROM starter_evidence
            WHERE date < ?
            GROUP BY season, team_id, game_id, side
        )
        SELECT season, team_id,
               SUM(names > 1),
               SUM(identifiers > 1)
        FROM grouped
        GROUP BY season, team_id
        ORDER BY season, team_id
        """,
        (end.isoformat(),),
    )

    result["starter_evidence_conflicts"] = [
        {
            "season": season,
            "team_id": team,
            "name_conflicts": names,
            "identifier_conflicts": identifiers,
        }
        for season, team, names, identifiers in conflicts
    ]

    identity_conflicts = db.execute(
        """
        SELECT season, team_id, COUNT(*)
        FROM (
            SELECT season, team_id, player_id
            FROM player_identity_evidence
            WHERE first_seen_date < ?
            GROUP BY season, team_id, player_id
            HAVING COUNT(DISTINCT name) > 1
        )
        GROUP BY season, team_id
        ORDER BY season, team_id
        """,
        (end.isoformat(),),
    )

    result["identity_conflicts"] = [
        {
            "season": season,
            "team_id": team,
            "conflicting_player_ids": count,
        }
        for season, team, count in identity_conflicts
    ]

    return {
        "end_exclusive": end.isoformat(),
        "competition_scope": "all_available_competitions",
        "tables": result,
    }


def evaluate_starters(
    db: sqlite3.Connection,
    completed: Sequence[PlayedGame],
    data: StarterData,
) -> dict[str, object]:
    """2025년 정규시즌 검증 경기의 양 팀을 평가한다."""

    start = date(SUPPORTED_SEASON, 1, 1)
    end = date(SUPPORTED_SEASON + 1, 1, 1)

    if any(
        not start <= game.fixture.day < end
        for game in completed
    ):
        raise ValueError("2025년 검증 기간 밖 경기 혼입")

    validation_ids = [
        row[0]
        for row in db.execute(
            """
            SELECT x.game_id
            FROM m1_index x
            JOIN m1_inputs i USING (sample_id)
            WHERE x.split = 'validation'
              AND x.status = 'eligible'
              AND i.season = ?
              AND i.competition = 'regular'
              AND x.sample_date >= ?
              AND x.sample_date < ?
            """,
            (
                SUPPORTED_SEASON,
                start.isoformat(),
                end.isoformat(),
            ),
        )
    ]

    completed_ids = {
        game.fixture.game_id
        for game in completed
    }

    if (
        len(validation_ids) != len(set(validation_ids))
        or set(validation_ids) != completed_ids
        or len(completed_ids) != len(completed)
    ):
        raise ValueError("검증 경기 집합 또는 중복 불일치")

    truth = read_truth(db, completed, end)

    reference = read_candidate_reference(
        db,
        completed,
        end,
    )

    samples: list[EvaluationSample] = []
    exclusions: Counter[str] = Counter()
    candidate_audit: Counter[str] = Counter()

    for game in sorted(
        completed,
        key=lambda item: (
            item.fixture.day,
            item.fixture.number,
            item.fixture.game_id,
        ),
    ):
        fixture = game.fixture

        for team in (fixture.home, fixture.away):
            key = (fixture.game_id, team)

            ranking = rank_starters(
                data,
                team,
                fixture.day,
                fixture.day,
            )

            exclusions.update(ranking.exclusions)

            previous = {
                row.player_id: row.previous_start_date
                for row in ranking.proposed
            }

            original = reference.get(key, {})

            candidate_audit["team_game_samples"] += 1
            candidate_audit["reference_candidate_rows"] += len(
                original
            )

            candidate_audit[
                "reconstructed_candidate_rows"
            ] += len(previous)

            candidate_audit["identical_candidate_sets"] += int(
                set(original) == set(previous)
            )

            candidate_audit["reference_only_candidates"] += len(
                set(original) - set(previous)
            )

            candidate_audit[
                "reconstructed_only_candidates"
            ] += len(
                set(previous) - set(original)
            )

            for player in set(original) & set(previous):
                candidate_audit["shared_candidates"] += 1

                candidate_audit[
                    "shared_previous_start_matches"
                ] += int(
                    original[player] == previous[player]
                )

            samples.append(
                make_sample(
                    fixture.game_id,
                    team,
                    fixture.day,
                    truth.get(
                        key,
                        Truth(
                            None,
                            "truth_missing_or_conflicted",
                        ),
                    ),
                    ranking,
                )
            )

    by_team: dict[int, list[EvaluationSample]] = defaultdict(list)

    for sample in samples:
        by_team[sample.team_id].append(sample)

    return {
        "schema": SCHEMA,
        "scope": "starter_validation",
        "season": SUPPORTED_SEASON,
        "rule_version": STARTER_RULE_VERSION,
        "start_inclusive": start.isoformat(),
        "end_exclusive": end.isoformat(),
        "split": "validation",
        "games": len(completed),
        "team_game_samples": len(samples),
        "comparison": compare_samples(samples),
        "by_team": {
            str(team): compare_samples(rows)
            for team, rows in sorted(by_team.items())
        },
        "candidate_exclusions": dict(
            sorted(exclusions.items())
        ),
        "candidate_reference_audit": dict(
            sorted(candidate_audit.items())
        ),
        "data_audit": audit_starter_tables(db, end),
        "samples": [
            {
                "game_id": sample.game_id,
                "team_id": sample.team_id,
                "date": sample.day.isoformat(),
                "actual_player_id": sample.truth.player_id,
                "truth_reason": sample.truth.reason,
                "baseline": list(sample.baseline),
                "proposed": list(sample.proposed),
            }
            for sample in samples
        ],
        "formal_backtest_supported": False,
        "final_test_used": False,
        "evaluation_scope": (
            "fixed_verified_archive_development_comparison"
        ),
    }
