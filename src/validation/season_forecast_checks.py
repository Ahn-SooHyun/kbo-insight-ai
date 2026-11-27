"""시즌 전망 및 보조 분석 산출물의 독립 검산."""

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from hashlib import sha256
from itertools import permutations
import json
from math import fsum, isclose, isfinite, sqrt
from pathlib import Path
import sqlite3
from typing import Any

from src.simulation.postseason.rules import Game, SeriesState, get_rules
from src.simulation.regular_season.data import read_completed
from src.simulation.regular_season.rules import KBO_2025
from src.simulation.support_analysis.analysis import rank_starters
from src.simulation.support_analysis.data import (
    StarterData,
    read_starter_data,
)


# 서로 다른 기존 JSON 스키마를 읽는 I/O 경계에서만 사용한다.
JsonObject = dict[str, Any]

SEASON = 2025
TEAMS = KBO_2025.teams
STAGES = tuple(rule.stage for rule in get_rules(SEASON))
SEED_INDEX = {
    rule.stage: rule.seed_index
    for rule in get_rules(SEASON)
}
GAMES_PER_TEAM = KBO_2025.games_per_team
PLAYOFF_SLOTS = KBO_2025.playoff_slots
TOP_K = 3
RECENT_DAYS = 28
DRAW_WEIGHT = 0.5
TOLERANCE = 1e-12
STABILITY_MIN_VALID = 30
STABILITY_MULTIPLIER = 5.0
STABILITY_PROBABILITY_FLOOR = 0.02
STABILITY_MEAN_FLOOR = 0.5

OFFICIAL_SEEDS = [2003, 2010, 2005, 2008, 2004]
OFFICIAL_WINNERS = {
    "wildCard": 2008,
    "semiPlayoff": 2008,
    "playoff": 2010,
    "koreanSeries": 2003,
}

OFFICIAL_REFERENCES = {
    "schedule": (
        "https://www.koreabaseball.com/"
        "MediaNews/Notice/View.aspx?bdSe=11735"
    ),
    "playoff": (
        "https://www.koreabaseball.com/"
        "MediaNews/News/BreakingNews/View.aspx?bdSe=60367"
    ),
    "korean_series": (
        "https://koreabaseball.com/"
        "MediaNews/News/BreakingNews/View.aspx?bdSe=60387"
    ),
    "away_win": (
        "https://www.yna.co.kr/view/AKR20251031176652007"
    ),
}


def require(condition: bool, message: str) -> None:
    """검산 조건이 성립하지 않으면 실패를 명시한다."""
    if not condition:
        raise ValueError(message)


def file_hash(path: Path) -> str:
    """대형 파일도 전체 메모리에 올리지 않고 SHA-256을 계산한다."""
    digest = sha256()

    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def read_json(path: Path) -> JsonObject:
    """UTF-8 JSON 객체를 읽는다."""
    with path.open(encoding="utf-8") as stream:
        result = json.load(stream)

    require(isinstance(result, dict), f"JSON 객체 오류: {path}")

    return result


def ratio(numerator: float, denominator: int) -> float | None:
    """분모가 없는 값을 실제 0과 구별한다."""
    return numerator / denominator if denominator else None


def equal(actual: object, expected: object, label: str) -> None:
    """구조·결측·타입을 유지하면서 계산 오차만 허용한다."""
    if isinstance(expected, dict):
        require(
            isinstance(actual, dict)
            and actual.keys() == expected.keys(),
            f"{label}: 객체 키 불일치",
        )

        for key, value in expected.items():
            equal(actual[key], value, f"{label}.{key}")

        return

    if isinstance(expected, list):
        require(
            isinstance(actual, list)
            and len(actual) == len(expected),
            f"{label}: 배열 길이 불일치",
        )

        for index, (left, right) in enumerate(
            zip(actual, expected, strict=True)
        ):
            equal(left, right, f"{label}[{index}]")

        return

    if type(expected) is float:
        require(
            type(actual) in (int, float)
            and isclose(
                actual,
                expected,
                rel_tol=0.0,
                abs_tol=TOLERANCE,
            ),
            f"{label}: {actual!r} != {expected!r}",
        )
        return

    require(
        type(actual) is type(expected) and actual == expected,
        f"{label}: {actual!r} != {expected!r}",
    )


def check_fields(
    actual: JsonObject,
    expected: JsonObject,
    label: str,
) -> None:
    """기존 출력의 검산 대상 필드를 대조한다."""
    for key, value in expected.items():
        require(key in actual, f"{label}: 필드 누락 {key}")
        equal(actual[key], value, f"{label}.{key}")


@dataclass
class Run:
    """무결성을 확인한 기존 시뮬레이션 실행."""

    directory: Path
    manifest: JsonObject
    scenario: JsonObject
    summary: JsonObject
    rows: list[JsonObject]
    digests: dict[str, str]


def check_sources(
    root: Path,
    package: str,
    recorded: dict[str, str],
) -> None:
    """산출물에서 기록한 소스 판본을 실제 파일과 대조한다."""
    source = root / "src" / "simulation" / package

    expected = {
        path.name
        for path in source.glob("*.py")
    }

    require(
        set(recorded) == expected,
        f"{package}: 소스 파일 목록 불일치",
    )

    for name, digest in recorded.items():
        require(
            file_hash(source / name) == digest,
            f"{package}: 소스 지문 불일치 {name}",
        )


def load_run(root: Path, directory: Path, package: str) -> Run:
    """매니페스트와 전체 결과 파일 지문을 대조한다."""
    manifest = read_json(directory / "manifest.json")

    require(
        manifest["status"] == "completed"
        and manifest["run_id"] == directory.name,
        f"실행 상태 또는 ID 오류: {directory}",
    )
    require(
        manifest["season"] == SEASON
        and manifest["final_test_used"] is False,
        "지원 시즌 또는 최종 Test 사용 상태 오류",
    )

    check_sources(root, package, manifest["source_sha256"])

    filenames = {
        "scenario": "scenario.json",
        "summary": "summary.json",
        "iterations": "iterations.jsonl",
        "diagnostic": "diagnostic.json",
        "baseline_iterations": "baseline_iterations.jsonl",
    }

    files = manifest["files"]

    require(
        {"scenario", "summary", "iterations"} <= set(files)
        <= set(filenames),
        "매니페스트 결과 파일 목록 오류",
    )

    digests = {}

    for key, information in files.items():
        filename = filenames[key]
        digest = file_hash(directory / filename)

        require(
            digest == information["sha256"],
            f"결과 지문 불일치: {filename}",
        )
        digests[filename] = digest

    scenario = read_json(directory / "scenario.json")
    summary = read_json(directory / "summary.json")

    equal(
        scenario["metadata"],
        manifest["metadata"],
        "시나리오 메타데이터",
    )
    equal(
        scenario["schema"],
        manifest["schemas"]["scenario"],
        "시나리오 스키마",
    )
    equal(
        summary["schema"],
        manifest["schemas"]["summary"],
        "요약 스키마",
    )

    rows = []

    with (directory / "iterations.jsonl").open(
        encoding="utf-8"
    ) as stream:
        for index, line in enumerate(stream, start=1):
            row = json.loads(line)

            require(
                type(row["iteration_id"]) is int
                and row["iteration_id"] == index,
                "반복 ID 누락·중복·순서 오류",
            )
            require(
                row["schema"] == manifest["schemas"]["iterations"],
                "반복 스키마 오류",
            )
            rows.append(row)

    require(
        len(rows) == manifest["iterations"]
        == summary["requested_iterations"]
        and len(rows) > 0,
        "요청·실제 반복 수 불일치",
    )

    return Run(
        directory,
        manifest,
        scenario,
        summary,
        rows,
        digests,
    )


def regular_games(
    db: sqlite3.Connection,
) -> list[JsonObject]:
    """표준 완료 정규시즌 경기에서 독립 검산 행을 읽는다."""
    rows = db.execute(
        """
        SELECT game_id, source_key, date, game_number,
               home_team_id, away_team_id, home_score,
               away_score, winner
        FROM eligible_games
        WHERE season = ?
          AND competition = 'regular'
        ORDER BY date, game_number, game_id
        """,
        (SEASON,),
    )

    result = []
    seen = set()

    for game_id, source, day, number, home, away, hs, aws, winner in rows:
        require(
            game_id == source and game_id not in seen,
            "정규시즌 원천 키 중복 또는 불일치",
        )
        require(
            home in TEAMS and away in TEAMS and home != away,
            "정규시즌 참가 구단 오류",
        )
        require(
            type(hs) is int and type(aws) is int
            and min(hs, aws) >= 0,
            "정규시즌 득점 오류",
        )

        outcome = "home" if hs > aws else "away" if hs < aws else "tie"

        equal(winner, outcome, f"{game_id}: 승무패")

        seen.add(game_id)

        result.append({
            "game_id": game_id,
            "date": day,
            "game_number": number,
            "home_team_id": home,
            "away_team_id": away,
            "home_runs": hs,
            "away_runs": aws,
        })

    return result


def fixture_only(game: JsonObject) -> JsonObject:
    """잔여 일정에는 사후 득점을 포함하지 않는다."""
    return {
        key: game[key]
        for key in (
            "game_id",
            "date",
            "game_number",
            "home_team_id",
            "away_team_id",
        )
    }


def team_scores(
    games: Sequence[JsonObject],
    team: int,
) -> list[tuple[int, int]]:
    """경기 결과를 해당 팀의 득점·실점 관점으로 변환한다."""
    result = []

    for game in games:
        if game["home_team_id"] == team:
            result.append((game["home_runs"], game["away_runs"]))
        elif game["away_team_id"] == team:
            result.append((game["away_runs"], game["home_runs"]))

    return result


def record_stats(
    scores: Sequence[tuple[int, int]],
) -> JsonObject:
    """운영 Record 클래스와 독립적으로 관측 성적을 계산한다."""
    games = len(scores)
    wins = sum(scored > allowed for scored, allowed in scores)
    losses = sum(scored < allowed for scored, allowed in scores)
    ties = games - wins - losses
    rf = sum(scored for scored, _ in scores)
    ra = sum(allowed for _, allowed in scores)
    decisions = wins + losses

    status = (
        "no_observations" if not games
        else "limited_sample" if games < 5
        else "supported"
    )

    return {
        "status": status,
        "reason": None if status == "supported" else status,
        "games": games,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "runs_for": rf,
        "runs_against": ra,
        "run_difference": rf - ra,
        "decisions": decisions,
        "win_rate": ratio(wins, decisions),
        "runs_for_per_game": ratio(rf, games),
        "runs_against_per_game": ratio(ra, games),
        "run_difference_per_game": ratio(rf - ra, games),
        "win_rate_denominator": decisions,
        "run_rate_denominator": games,
        "win_rate_reason": None if decisions else "no_decided_games",
        "run_rate_reason": None if games else "no_observations",
    }


def distribution(
    values: Sequence[int | None],
    reasons: Sequence[str | None],
    categories: Sequence[int],
) -> JsonObject:
    """저장된 반복별 값에서 범주 분포와 미결정 수를 재집계한다."""
    require(len(values) == len(reasons), "분포 입력 길이 오류")

    observed = [value for value in values if value is not None]
    counts = Counter(observed)
    valid = len(observed)

    require(
        set(counts) <= set(categories),
        "분포 범주 오류",
    )

    unresolved = Counter(
        reason or "unresolved"
        for value, reason in zip(values, reasons, strict=True)
        if value is None
    )

    return {
        "status": "supported" if valid else "unsupported",
        "requested": len(values),
        "valid": valid,
        "unresolved": len(values) - valid,
        "unresolved_reasons": dict(unresolved),
        "counts": {
            str(category): counts[category]
            for category in categories
        },
        "probabilities": {
            str(category): ratio(counts[category], valid)
            for category in categories
        },
        "expected": ratio(sum(observed), valid),
    }


def regular_events(
    row: JsonObject,
    initial: dict[str, JsonObject],
) -> None:
    """팀 기록·순위·진출·시드의 논리적 일관성을 검사한다."""
    teams = row["teams"]

    require(
        set(teams) == {str(team) for team in TEAMS},
        "정규시즌 반복 구단 집합 오류",
    )

    for team, item in teams.items():
        counts = [item[key] for key in ("wins", "losses", "ties")]

        require(
            all(type(value) is int and value >= 0 for value in counts),
            "승무패 횟수 오류",
        )
        require(
            sum(counts) == item["games"] == GAMES_PER_TEAM,
            "완전 일정의 최종 경기 수 오류",
        )
        equal(item["final_wins"], item["wins"], "최종 승수")

        for key in ("wins", "losses", "ties"):
            require(
                item[key] >= initial[team][key],
                "기준일 확정 성적 소실",
            )

        for key in ("first", "qualifies"):
            require(
                item[key] is None or type(item[key]) is bool,
                "이진 사건 타입 오류",
            )

        rank = item["rank"]

        if rank is not None:
            require(
                type(rank) is int and 1 <= rank <= len(TEAMS),
                "순위 범위 오류",
            )
            equal(item["first"], rank == 1, "1위 상태")
            equal(item["qualifies"], rank <= PLAYOFF_SLOTS, "진출 상태")

    require(
        sum(item["wins"] for item in teams.values())
        == sum(item["losses"] for item in teams.values()),
        "리그 승·패 합계 불일치",
    )
    require(
        sum(item["ties"] for item in teams.values()) % 2 == 0,
        "리그 무승부 합계 오류",
    )

    ranks_resolved = all(
        item["rank"] is not None for item in teams.values()
    )
    qualification_resolved = all(
        item["qualifies"] is not None for item in teams.values()
    )

    equal(row["rank_resolved"], ranks_resolved, "순위 확정 상태")
    equal(
        row["qualification_resolved"],
        qualification_resolved,
        "진출 확정 상태",
    )

    qualifiers = sorted(
        int(team)
        for team, item in teams.items()
        if item["qualifies"] is True
    )

    if qualification_resolved:
        require(len(qualifiers) == PLAYOFF_SLOTS, "진출 구단 수 오류")

    equal(
        row["qualifiers"],
        qualifiers if qualification_resolved else None,
        "진출 구단",
    )

    rank_map = defaultdict(list)

    for team, item in teams.items():
        rank = item["rank"]

        if rank is not None and rank <= PLAYOFF_SLOTS:
            rank_map[rank].append(int(team))

    require(
        all(len(group) == 1 for group in rank_map.values()),
        "포스트시즌 시드 순위 중복",
    )

    expected_seeds = None

    if qualification_resolved and len(rank_map) == PLAYOFF_SLOTS:
        expected_seeds = [
            rank_map[rank][0]
            for rank in range(1, PLAYOFF_SLOTS + 1)
        ]

    equal(row["postseason_seeds"], expected_seeds, "시드")


def audit_regular(
    run: Run,
    games: Sequence[JsonObject],
) -> JsonObject:
    """실자료 완료 경기와 정규시즌 반복 분포를 검산한다."""
    cutoff = run.manifest["as_of_kst"][:10]
    completed = [game for game in games if game["date"] < cutoff]
    remaining = [
        fixture_only(game)
        for game in games
        if game["date"] >= cutoff
    ]

    equal(run.scenario["completed"], completed, "완료 경기 원천")
    equal(run.scenario["remaining"], remaining, "잔여 경기 원천")

    check_fields(
        run.scenario["metadata"],
        {
            "completed_games": len(completed),
            "remaining_games": len(remaining),
            "missing_games": 0,
        },
        "시나리오 경기 수",
    )

    initial = {
        str(team): record_stats(team_scores(completed, team))
        for team in TEAMS
    }

    for team, stats in initial.items():
        check_fields(
            run.scenario["current_teams"][team],
            {
                key: stats[key]
                for key in ("games", "wins", "losses", "ties")
            },
            f"기준 성적 {team}",
        )

    for row in run.rows:
        regular_events(row, initial)

    fields = (
        ("final_wins", "final_wins", range(GAMES_PER_TEAM + 1)),
        ("rank", "rank", range(1, len(TEAMS) + 1)),
        ("regular_season_first", "first", (0, 1)),
        ("postseason_qualification", "qualifies", (0, 1)),
    )

    for team in map(str, TEAMS):
        records = [row["teams"][team] for row in run.rows]

        for output, source, categories in fields:
            values = [
                None if item[source] is None else int(item[source])
                for item in records
            ]
            reasons = [
                "incomplete_schedule"
                if source == "final_wins" else item["reason"]
                for item in records
            ]

            expected = distribution(values, reasons, categories)

            equal(
                run.summary["teams"][team][output],
                expected,
                f"정규시즌 {team}.{output}",
            )

    first = [
        row for row in run.rows
        if all(item["first"] is not None for item in row["teams"].values())
    ]
    qualification = [
        row for row in run.rows if row["qualification_resolved"]
    ]

    for row in first:
        require(
            sum(item["first"] is True for item in row["teams"].values()) == 1,
            "공동 확정 반복의 1위 구단 수 오류",
        )

    for label, source, valid in (
        ("first", "first", first),
        ("qualification", "qualifies", qualification),
    ):
        expected = {
            str(team): ratio(
                sum(row["teams"][str(team)][source] is True for row in valid),
                len(valid),
            )
            for team in TEAMS
        }

        check_fields(
            run.summary["league"],
            {
                f"{label}_joint_valid": len(valid),
                f"{label}_joint_unresolved": len(run.rows) - len(valid),
                f"{label}_joint_probabilities": expected,
            },
            label,
        )

    check_fields(
        run.summary["league"],
        {
            "rank_resolved_iterations": sum(
                row["rank_resolved"] for row in run.rows
            ),
            "postseason_seed_resolved_iterations": sum(
                row["postseason_seeds"] is not None for row in run.rows
            ),
        },
        "순위 및 시드 분모",
    )

    return {
        "completed_games": len(completed),
        "remaining_games": len(remaining),
        "iterations": len(run.rows),
        "league": run.summary["league"],
        "fixed_game_detail_scope": "scenario_source_rows",
        "iteration_scope": "team_aggregate",
    }


def audit_parent(parent: Run, child: Run) -> JsonObject:
    """같은 부모 반복의 순위·진출 상태가 보존되었는지 확인한다."""
    metadata = child.scenario["metadata"]["parent"]

    check_fields(
        metadata,
        {
            "run_id": parent.directory.name,
            "manifest_sha256": file_hash(
                parent.directory / "manifest.json"
            ),
            "seed": parent.manifest["seed"],
            "iterations": len(parent.rows),
            "metadata": parent.scenario["metadata"],
        },
        "부모 실행 계보",
    )

    for filename in ("scenario.json", "summary.json", "iterations.jsonl"):
        equal(
            metadata["files_sha256"][filename],
            parent.digests[filename],
            f"부모 지문 {filename}",
        )

    equal(child.manifest["seed"], parent.manifest["seed"], "Seed 전달")
    equal(
        child.manifest["as_of_kst"],
        parent.manifest["as_of_kst"],
        "부모·자식 기준일",
    )
    equal(len(child.rows), len(parent.rows), "부모·자식 반복 수")

    for left, right in zip(parent.rows, child.rows, strict=True):
        teams = left["teams"]
        first = [
            int(team)
            for team, item in teams.items()
            if item["first"] is True
        ]

        check_fields(
            right,
            {
                "iteration_id": left["iteration_id"],
                "postseason_seeds": left["postseason_seeds"],
                "qualifiers": left["qualifiers"],
                "qualification": {
                    team: item["qualifies"]
                    for team, item in teams.items()
                },
                "regular_ranks": {
                    team: item["rank"]
                    for team, item in teams.items()
                },
                "regular_first_status": {
                    team: item["first"]
                    for team, item in teams.items()
                },
                "regular_first": first[0] if first else None,
                "seed_reasons": sorted({
                    item["reason"]
                    for item in teams.values()
                    if item["reason"] is not None
                }),
            },
            f"부모 연결 {left['iteration_id']}",
        )

    return {
        "parent_run": parent.directory.name,
        "child_run": child.directory.name,
        "matched_iterations": len(parent.rows),
        "seed": parent.manifest["seed"],
        "seed_resolved": sum(
            row["postseason_seeds"] is not None
            for row in parent.rows
        ),
    }


def probability(values: Sequence[bool | None]) -> JsonObject:
    """사건의 유효·미결정 분모와 가능한 확률 범위를 계산한다."""
    total = len(values)

    require(
        all(value is None or type(value) is bool for value in values),
        "사건 값 타입 오류",
    )

    positive = sum(value is True for value in values)
    negative = sum(value is False for value in values)
    valid = positive + negative
    unresolved = total - valid

    return {
        "total": total,
        "valid": valid,
        "unresolved": unresolved,
        "positive": positive,
        "negative": negative,
        "probability": ratio(positive, valid),
        "lower_bound": ratio(positive, total),
        "upper_bound": ratio(positive + unresolved, total),
        "all_iterations_resolved": total > 0 and unresolved == 0,
        "status": (
            "unsupported" if not valid
            else "partial" if unresolved
            else "estimated"
        ),
    }


def first_event(row: JsonObject, team: int) -> bool | None:
    """개별 팀의 확정된 정규시즌 1위 상태를 보존한다."""
    if row["regular_first"] is not None:
        return row["regular_first"] == team

    if row["qualification"][str(team)] is False:
        return False

    return row["regular_first_status"][str(team)]


def round_event(
    row: JsonObject,
    team: int,
    stage: str,
    kind: str,
) -> bool | None:
    """라운드 참가·승리 사건을 원래 반복 행에서 판정한다."""
    require(kind in ("reach", "win"), "지원하지 않는 사건")

    result = row["rounds"][stage]
    participants = result["participants"]

    if participants is not None:
        if kind == "reach":
            return team in participants

        if team not in participants:
            return False

        winner = result["winner"]
        return None if winner is None else winner == team

    if row["qualification"][str(team)] is False:
        return False

    direct_rank = SEED_INDEX[stage] + 1
    seeds = row["postseason_seeds"]

    if seeds is not None:
        if team not in seeds[direct_rank - 1:]:
            return False

        if kind == "reach" and team == seeds[direct_rank - 1]:
            return True

    else:
        rank = row["regular_ranks"][str(team)]

        if rank is not None:
            if rank < direct_rank or rank > PLAYOFF_SLOTS:
                return False

            if kind == "reach" and (
                rank == direct_rank
                or (stage == "wildCard" and rank == PLAYOFF_SLOTS)
            ):
                return True

    if (
        stage == "koreanSeries"
        and kind == "reach"
        and first_event(row, team) is True
    ):
        return True

    for previous_stage in STAGES[:STAGES.index(stage)]:
        previous = row["rounds"][previous_stage]

        if (
            previous["participants"] is not None
            and team in previous["participants"]
            and previous["winner"] is not None
            and previous["winner"] != team
        ):
            return False

    return None


def audit_bracket(row: JsonObject, archive: JsonObject | None) -> None:
    """시드 연결 및 시리즈 규칙 재생을 검사한다."""
    seeds = row["postseason_seeds"]
    advancing = seeds[-1] if seeds is not None else None

    for rule in get_rules(SEASON):
        stage = rule.stage
        result = row["rounds"][stage]

        if seeds is None or advancing is None:
            check_fields(
                result,
                {
                    "participants": None,
                    "winner": None,
                    "games": [],
                    "fixed": False,
                    "reason": (
                        "seed_unresolved"
                        if seeds is None
                        else "previous_round_unresolved"
                    ),
                },
                f"{stage}: 미결정 연결",
            )
            continue

        higher = seeds[rule.seed_index]

        equal(
            result["participants"],
            [higher, advancing],
            f"{stage}: 동일 시드 연결",
        )

        state = SeriesState(rule, higher, advancing)

        observed = []

        for item in result["games"]:
            game = Game(**item)
            state.apply(game)

            if game.observed:
                observed.append(item)

        winner = result["winner"]

        if winner is not None:
            equal(winner, state.winner, f"{stage}: 종료 승자")
            equal(result["reason"], None, f"{stage}: 종료 사유")
        else:
            require(bool(result["reason"]), f"{stage}: 미결정 사유 누락")

            if state.winner is not None:
                require(
                    archive is not None
                    and (
                        stage in archive["blocked"]
                        or result["reason"] == "archive_completion_evidence_missing"
                    ),
                    "확정 승자의 임의 제거",
                )

        if archive is None:
            require(not observed, "부모 모드의 사후 실제 경기 혼입")
            equal(result["fixed"], False, f"{stage}: 고정 상태")

            if winner is None:
                equal(result["reason"], state.unresolved_reason, stage)

        else:
            source = archive["games"].get(stage, [])

            equal(
                observed,
                source[:len(observed)],
                f"{stage}: 관측 경기 접두 구간",
            )

            if result["fixed"]:
                require(
                    winner is not None
                    and stage in archive["completion_evidence"]
                    and len(observed) == len(result["games"]),
                    "완료 시리즈 고정 근거 오류",
                )

        advancing = winner

    equal(
        row["korean_series_champion"],
        row["rounds"]["koreanSeries"]["winner"],
        "한국시리즈 우승 연결",
    )


def audit_postseason(run: Run) -> JsonObject:
    """포스트시즌 전체 및 대진 조건부 분모를 독립 재집계한다."""
    rows = run.rows
    archive = run.scenario["metadata"]["archive"]
    total = len(rows)

    for row in rows:
        audit_bracket(row, archive)

    for team in TEAMS:
        summary = run.summary["teams"][str(team)]

        equal(
            summary["regular_first"],
            probability([first_event(row, team) for row in rows]),
            f"{team}: 정규시즌 1위",
        )

        for stage in STAGES:
            for kind in ("reach", "win"):
                expected = probability([
                    round_event(row, team, stage, kind)
                    for row in rows
                ])

                equal(
                    summary["rounds"][stage][kind],
                    expected,
                    f"{team}.{stage}.{kind}",
                )

        equal(
            summary["korean_series_entry"],
            summary["rounds"]["koreanSeries"]["reach"],
            "한국시리즈 진출 별칭",
        )
        equal(
            summary["korean_series_champion"],
            summary["rounds"]["koreanSeries"]["win"],
            "한국시리즈 우승 별칭",
        )

    for stage in STAGES:
        winners = Counter(
            row["rounds"][stage]["winner"]
            for row in rows
            if row["rounds"][stage]["winner"] is not None
        )

        valid = sum(winners.values())
        groups = defaultdict(list)

        for row in rows:
            participants = row["rounds"][stage]["participants"]

            if participants is not None:
                groups[tuple(participants)].append(row)

        formed = sum(map(len, groups.values()))

        reasons = Counter(
            row["rounds"][stage]["reason"] or "unknown"
            for row in rows
            if row["rounds"][stage]["winner"] is None
        )

        equal(
            run.summary["league"][stage],
            {
                "total": total,
                "valid": valid,
                "unresolved": total - valid,
                "matchup_formed": formed,
                "matchup_unresolved": total - formed,
                "fixed": sum(row["rounds"][stage]["fixed"] for row in rows),
                "joint_win_probabilities": {
                    str(team): ratio(winners[team], valid)
                    for team in TEAMS
                },
                "reasons": dict(reasons),
            },
            f"{stage}: 리그 공동 분모",
        )

        conditionals = run.summary["conditional_matchups"][stage]
        pairs = tuple(permutations(TEAMS, 2))

        equal(
            set(conditionals),
            {f"{higher}:{lower}" for higher, lower in pairs},
            f"{stage}: 대진 집합",
        )

        for higher, lower in pairs:
            group = groups.get((higher, lower), [])
            series_valid = sum(
                row["rounds"][stage]["winner"] is not None
                for row in group
            )

            expected = {
                "higher_seed_team": higher,
                "lower_seed_team": lower,
                "formed": len(group),
                "formation_unresolved": total - formed,
                "series_valid": series_valid,
                "series_unresolved": len(group) - series_valid,
                "teams": {
                    str(team): {
                        "series_win": probability([
                            round_event(row, team, stage, "win")
                            for row in group
                        ]),
                        "champion": probability([
                            round_event(row, team, "koreanSeries", "win")
                            for row in group
                        ]),
                    }
                    for team in (higher, lower)
                },
            }

            equal(
                conditionals[f"{higher}:{lower}"],
                expected,
                f"{stage}: 조건부 대진 {higher}:{lower}",
            )

    seed_resolved = sum(
        row["postseason_seeds"] is not None for row in rows
    )
    seed_reasons = Counter(
        reason
        for row in rows
        if row["postseason_seeds"] is None
        for reason in row["seed_reasons"] or ["seed_unresolved"]
    )

    check_fields(
        run.summary,
        {
            "seed_resolved": seed_resolved,
            "seed_unresolved": total - seed_resolved,
            "seed_reasons": dict(seed_reasons),
        },
        "시드 분모",
    )

    champion = run.summary["league"]["koreanSeries"]

    check_fields(
        run.manifest,
        {
            "champion_valid": champion["valid"],
            "champion_unresolved": champion["unresolved"],
        },
        "매니페스트 우승 분모",
    )

    return {
        "iterations": total,
        "seed_resolved": seed_resolved,
        "league": run.summary["league"],
        "conditional_matchup_groups": len(STAGES) * len(TEAMS) * (len(TEAMS) - 1),
    }


def audit_archive(
    run: Run,
    standard: sqlite3.Connection,
    model: sqlite3.Connection,
) -> JsonObject:
    """종료 재생의 원천 기록과 공식 확인 사실을 대조한다."""
    archive = run.scenario["metadata"]["archive"]
    cutoff = run.manifest["as_of_kst"][:10]

    require(archive is not None, "종료 재생 자료 누락")

    schedule_columns = """
        SELECT source_sha, source_row, stage,
               series_game_number, date, away_name,
               home_name, final_observed, linked_game_id
        FROM {table}
        WHERE substr(date, 1, 4) = ?
          AND date < ?
        ORDER BY date, series_game_number, source_sha, source_row
    """

    schedule = standard.execute(
        schedule_columns.format(table="postseason_schedule"),
        (str(SEASON), cutoff),
    ).fetchall()

    model_schedule = model.execute(
        schedule_columns.format(table="season_schedule_evidence"),
        (str(SEASON), cutoff),
    ).fetchall()

    equal(schedule, model_schedule, "표준·모델 일정 근거")
    equal(
        archive["schedule_records"],
        [list(row) for row in schedule],
        "저장 일정 근거",
    )

    source_rows = standard.execute(
        """
        SELECT game_id, source_key, stage, date, game_number,
               home_team_id, away_team_id, home_score,
               away_score, winner, source_sha, source_row
        FROM eligible_games
        WHERE season = ?
          AND competition = 'postseason'
          AND date < ?
        """,
        (SEASON, cutoff),
    ).fetchall()

    source = {row[0]: row for row in source_rows}

    require(
        len(source) == len(source_rows),
        "포스트시즌 원천 경기 ID 중복",
    )

    for stage, games in archive["games"].items():
        for game in games:
            record = source[game["game_id"]]

            equal(
                list(record[2:10]),
                [
                    stage,
                    game["day"],
                    game["number"],
                    game["home"],
                    game["away"],
                    game["home_score"],
                    game["away_score"],
                    game["outcome"],
                ],
                "포스트시즌 실제 경기",
            )

    fifth = next(
        game
        for game in archive["games"]["koreanSeries"]
        if game["number"] == 5
    )
    raw = source[fifth["game_id"]]

    equal(raw[1], "2025:koreanSeries:5", "한국시리즈 원천 키")
    equal(fifth["day"], "2025-10-31", "한국시리즈 5차전 날짜")
    equal(fifth["home"], 2003, "원천 홈 표기 충돌")
    equal(fifth["away"], 2010, "원천 원정 표기 충돌")

    reason = archive["blocked"]["koreanSeries"]

    require(
        "archive_replay_conflict" in reason,
        "기존 홈 표기 충돌 사유 소실",
    )

    for row in run.rows:
        equal(row["postseason_seeds"], OFFICIAL_SEEDS, "종료 시즌 시드")

        for stage in STAGES[:-1]:
            check_fields(
                row["rounds"][stage],
                {
                    "winner": OFFICIAL_WINNERS[stage],
                    "fixed": True,
                },
                f"{stage}: 공식 사실 대조",
            )

        korean = row["rounds"]["koreanSeries"]

        equal(korean["winner"], None, "충돌 우승 미결정")
        equal(korean["fixed"], False, "충돌 고정 금지")
        equal(len(korean["games"]), 4, "충돌 전 재생 경기 수")
        equal(korean["reason"], reason, "충돌 전파 사유")

    equal(run.manifest["champion_valid"], 0, "종료 재생 우승 분모")
    equal(run.manifest["cache_profiles"], 0, "종료 재생 신규 전망 금지")

    return {
        "source_game": fifth,
        "source_key": raw[1],
        "source_sha256": raw[10],
        "source_row": raw[11],
        "source_schedule_records": [
            list(row)
            for row in schedule
            if row[-1] == fifth["game_id"]
        ],
        "official_game": {
            "date": "2025-10-31",
            "number": 5,
            "home": 2010,
            "away": 2003,
            "home_score": 1,
            "away_score": 4,
            "winner": 2003,
            "series_wins": 4,
            "series_losses": 1,
        },
        "team_score_matches": {
            "LG": fifth["home_score"] == 4,
            "HH": fifth["away_score"] == 1,
        },
        "home_conflict": True,
        "unresolved": len(run.rows),
        "reason": reason,
        "official_references": OFFICIAL_REFERENCES,
        "official_full_game_record_verified": False,
        "raw_file_bytes_rechecked": False,
        "official_champion_reproduced": False,
    }


def starter_metrics(
    samples: Sequence[JsonObject],
    method: str,
) -> JsonObject:
    """선발 정답·후보 제공 및 적중 횟수를 독립 계산한다."""
    truth = [
        item for item in samples
        if item["actual_player_id"] is not None
    ]
    provided = [item for item in samples if item[method]]
    valid = [item for item in truth if item[method]]

    top1 = sum(
        item[method][0] == item["actual_player_id"]
        for item in valid
    )
    topk = sum(
        item["actual_player_id"] in item[method]
        for item in valid
    )

    sizes = Counter(len(item[method]) for item in samples)
    candidate_total = sum(len(item[method]) for item in samples)

    return {
        "method": method,
        "team_game_samples": len(samples),
        "truth_available": len(truth),
        "candidate_provided": len(provided),
        "candidate_coverage": ratio(len(provided), len(samples)),
        "evaluable_samples": len(valid),
        "evaluable_truth_coverage": ratio(len(valid), len(truth)),
        "top1_hits": top1,
        "top1_accuracy": ratio(top1, len(valid)),
        "top_k": TOP_K,
        "top_k_hits": topk,
        "top_k_accuracy": ratio(topk, len(valid)),
        "top1_all_identified_truth": ratio(top1, len(truth)),
        "top_k_all_identified_truth": ratio(topk, len(truth)),
        "candidate_count_total": candidate_total,
        "mean_candidates_when_provided": ratio(
            candidate_total, len(provided)
        ),
        "candidate_size_distribution": {
            str(key): value for key, value in sorted(sizes.items())
        },
    }


def starter_comparison(samples: Sequence[JsonObject]) -> JsonObject:
    """두 규칙을 같은 공통 유효 표본에서 비교한다."""
    common = [
        item for item in samples
        if item["actual_player_id"] is not None
        and item["baseline"]
        and item["proposed"]
    ]

    exclusions = Counter()

    for item in samples:
        if item["actual_player_id"] is None:
            exclusions[
                item["truth_reason"] or "truth_unavailable"
            ] += 1
        elif not item["baseline"] and not item["proposed"]:
            exclusions["neither_method_has_candidates"] += 1
        elif not item["baseline"]:
            exclusions["baseline_candidates_missing"] += 1
        elif not item["proposed"]:
            exclusions["proposed_candidates_missing"] += 1

    return {
        "baseline": starter_metrics(samples, "baseline"),
        "proposed": starter_metrics(samples, "proposed"),
        "common_samples": len(common),
        "common_baseline": starter_metrics(common, "baseline"),
        "common_proposed": starter_metrics(common, "proposed"),
        "common_exclusions": dict(sorted(exclusions.items())),
    }


def before_cutoff(data: StarterData, cutoff: date) -> StarterData:
    """후보 자료에서 기준일 이후 및 당일 관측을 물리적으로 제외한다."""
    return StarterData(
        starts={
            player: tuple(row for row in rows if row.day < cutoff)
            for player, rows in data.starts.items()
        },
        identities={
            player: tuple(row for row in rows if row.day < cutoff)
            for player, rows in data.identities.items()
        },
        appearances={
            player: tuple(row for row in rows if row.day < cutoff)
            for player, rows in data.appearances.items()
        },
    )


def valid_text(value: object) -> bool:
    """식별 가능한 비어 있지 않은 문자열인지 확인한다."""
    return isinstance(value, str) and bool(value.strip())


def audit_starters(
    result: JsonObject,
    standard: sqlite3.Connection,
    model: sqlite3.Connection,
) -> JsonObject:
    """실제 정답과 후보 이력·평가 분모를 검산한다."""
    end = date(SEASON + 1, 1, 1)
    completed = read_completed(standard, model, SEASON, end)
    data = read_starter_data(standard, model, completed, SEASON, end)

    fixtures = {
        game.fixture.game_id: game.fixture
        for game in completed
    }
    expected_keys = {
        (game_id, team)
        for game_id, fixture in fixtures.items()
        for team in (fixture.home, fixture.away)
    }

    truth = {}

    for game_id, side, team, day, player, name, count in model.execute(
        """
        SELECT game_id, side, team_id, date,
               player_id, name, evidence_count
        FROM starter_actual
        WHERE date >= ? AND date < ?
        """,
        (f"{SEASON}-01-01", end.isoformat()),
    ):
        if game_id not in fixtures:
            continue

        fixture = fixtures[game_id]
        key = (game_id, team)

        require(key in expected_keys, "실제 선발 참가 구단 오류")
        require(key not in truth, "실제 선발 정답 중복")
        equal(day, fixture.day.isoformat(), "실제 선발 날짜")
        equal(
            side,
            "home" if team == fixture.home else "away",
            "실제 선발 홈·원정",
        )

        if count is None or count == 0:
            truth[key] = (None, "truth_evidence_missing")
        elif not valid_text(player):
            truth[key] = (None, "truth_player_id_unresolved")
        elif not valid_text(name):
            truth[key] = (None, "truth_name_missing")
        else:
            truth[key] = (player, None)

    samples = result["samples"]
    seen = set()
    by_team = defaultdict(list)
    current_day = None
    truncated = None
    rankings = {}

    for sample in sorted(
        samples,
        key=lambda item: (item["date"], item["game_id"], item["team_id"]),
    ):
        key = (sample["game_id"], sample["team_id"])

        require(key in expected_keys and key not in seen, "선발 표본 키 오류")
        seen.add(key)

        fixture = fixtures[key[0]]
        day = fixture.day
        team = key[1]

        equal(sample["date"], day.isoformat(), "선발 표본 날짜")

        actual, reason = truth.get(
            key,
            (None, "truth_missing_or_conflicted"),
        )

        equal(sample["actual_player_id"], actual, "선발 정답")
        equal(sample["truth_reason"], reason, "선발 정답 사유")

        if day != current_day:
            current_day = day
            truncated = before_cutoff(data, day)
            rankings.clear()

        if team not in rankings:
            require(truncated is not None, "선발 절단 자료 누락")

            full = rank_starters(data, team, day, day)
            cut = rank_starters(truncated, team, day, day)

            equal(full, cut, "선발 미래 자료 영향")
            rankings[team] = cut

        ranking = rankings[team]

        for method in ("baseline", "proposed"):
            candidates = getattr(ranking, method)[:TOP_K]
            ids = [candidate.player_id for candidate in candidates]

            equal(sample[method], ids, "저장 선발 후보 순위")
            require(len(ids) == len(set(ids)), "선발 후보 중복")

            for candidate in candidates:
                history = [
                    row.day
                    for row in data.starts.get(candidate.player_id, ())
                    if row.team_id == team
                    and day - timedelta(days=RECENT_DAYS) <= row.day < day
                ]

                require(bool(history), "후보의 기준일 이전 선발 근거 누락")
                equal(
                    candidate.previous_start_date,
                    max(history).isoformat(),
                    "후보의 이전 선발일",
                )

                for value in (
                    candidate.previous_start_date,
                    candidate.previous_observed_appearance_date,
                    candidate.last_team_evidence_date,
                ):
                    require(value < day.isoformat(), "후보 사후 정보 혼입")

        by_team[team].append(sample)

    equal(seen, expected_keys, "선발 전체 경기×팀 측")
    equal(result["games"], len(completed), "선발 평가 경기 수")
    equal(result["team_game_samples"], len(expected_keys), "선발 평가 측 수")

    comparison = starter_comparison(samples)

    equal(result["comparison"], comparison, "선발 전체 비교")

    for team, group in by_team.items():
        equal(
            result["by_team"][str(team)],
            starter_comparison(group),
            f"선발 팀별 비교 {team}",
        )

    available = comparison["baseline"]["truth_available"]

    return {
        "games": len(completed),
        "team_game_samples": len(samples),
        "truth_available": available,
        "truth_identification_rate": ratio(available, len(samples)),
        "comparison": comparison,
        "candidate_exclusions": result["candidate_exclusions"],
        "candidate_reference_audit": result["candidate_reference_audit"],
        "future_cutoff_comparison": "passed",
        "ranking_algorithm": "existing_fixed_rule",
    }


def check_difficulty(
    row: JsonObject,
    probabilities: JsonObject | None,
) -> float | None:
    """경기별 팀 관점 난이도의 산식을 직접 검증한다."""
    if probabilities is None:
        check_fields(
            row,
            {
                "probabilities": None,
                "difficulty": None,
                "status": "unsupported",
                "reason": "forecast_missing",
                "weight": 1,
            },
            "난이도 미지원",
        )
        return None

    require(
        set(probabilities) == {"win", "tie", "loss"},
        "승무패 확률 키 오류",
    )
    require(
        all(
            type(value) in (int, float)
            and isfinite(value)
            and 0 <= value <= 1
            for value in probabilities.values()
        ),
        "승무패 확률 범위 오류",
    )

    equal(fsum(probabilities.values()), 1.0, "확률 합")

    difficulty = probabilities["loss"] + DRAW_WEIGHT * probabilities["tie"]

    check_fields(
        row,
        {
            "probabilities": probabilities,
            "difficulty": difficulty,
            "status": "supported",
            "reason": None,
            "weight": 1,
        },
        "경기 난이도",
    )

    return difficulty


def audit_support_season(
    result: JsonObject,
    games: Sequence[JsonObject],
) -> JsonObject:
    """팀 누적·최근 성적과 잔여 경기별 난이도를 독립 검산한다."""
    cutoff = date.fromisoformat(result["as_of_kst"][:10])
    completed = [
        game for game in games
        if game["date"] < cutoff.isoformat()
    ]
    remaining = [
        game for game in games
        if game["date"] >= cutoff.isoformat()
    ]

    expected = {
        game["game_id"]: fixture_only(game)
        for game in remaining
    }
    reports = {}
    difficulty_rows = defaultdict(list)
    candidate_reasons = Counter()

    for report in result["games"]:
        fixture = report["fixture"]
        game_id = fixture["game_id"]

        require(
            game_id in expected and game_id not in reports,
            "보조 분석 잔여 경기 키 오류",
        )
        equal(fixture, expected[game_id], "보조 분석 경기 식별 정보")
        reports[game_id] = report

        forecast = report["m1_forecast"]
        probabilities = (
            None if forecast is None else forecast["result_probabilities"]
        )
        home = fixture["home_team_id"]
        away = fixture["away_team_id"]
        values = []

        for team, opponent, venue in (
            (home, away, "home"),
            (away, home, "away"),
        ):
            side = report["difficulty"][str(team)]

            check_fields(
                side,
                {
                    "game_id": game_id,
                    "team_id": team,
                    "opponent_team_id": opponent,
                    "venue": venue,
                    "date": fixture["date"],
                },
                "난이도 경기×팀 식별자",
            )

            team_probabilities = None

            if probabilities is not None:
                team_probabilities = {
                    "win": probabilities[venue],
                    "tie": probabilities["tie"],
                    "loss": probabilities["away" if venue == "home" else "home"],
                }

            values.append(check_difficulty(side, team_probabilities))
            difficulty_rows[team].append(side)

            starters = report["starters"][str(team)]
            candidates = starters["candidates"]

            equal(starters["candidate_count"], len(candidates), "후보 수")
            require(len(candidates) <= TOP_K, "후보 상한 초과")

            for candidate in candidates:
                require(
                    all(
                        candidate[key] < cutoff.isoformat()
                        for key in (
                            "previous_start_date",
                            "previous_observed_appearance_date",
                            "last_team_evidence_date",
                        )
                    ),
                    "보조 분석 후보의 미래 기록 혼입",
                )

            if starters["reason"] is not None:
                candidate_reasons[starters["reason"]] += 1

        if all(value is not None for value in values):
            equal(fsum(values), 1.0, "홈·원정 난이도 합")

    equal(set(reports), set(expected), "잔여 경기 제공 범위")

    for team in TEAMS:
        report = result["teams"][str(team)]
        profile = report["profile"]

        season_start = date(SEASON, 1, 1)
        recent_start = max(season_start, cutoff - timedelta(days=RECENT_DAYS))

        for label, start in (
            ("season", season_start),
            ("recent", recent_start),
        ):
            window = profile[label]

            subset = [
                game for game in completed
                if start.isoformat() <= game["date"] < cutoff.isoformat()
            ]
            home_games = [
                game for game in subset if game["home_team_id"] == team
            ]
            away_games = [
                game for game in subset if game["away_team_id"] == team
            ]

            check_fields(
                window,
                {
                    "start_inclusive": start.isoformat(),
                    "end_exclusive": cutoff.isoformat(),
                    "overall": record_stats(team_scores(subset, team)),
                    "home": record_stats(team_scores(home_games, team)),
                    "away": record_stats(team_scores(away_games, team)),
                },
                f"{team}.{label}",
            )

            for opponent in TEAMS:
                if opponent == team:
                    continue

                versus = [
                    game for game in subset
                    if {
                        game["home_team_id"],
                        game["away_team_id"],
                    } == {team, opponent}
                ]

                equal(
                    window["opponents"][str(opponent)],
                    record_stats(team_scores(versus, team)),
                    f"{team}.{label}.상대 {opponent}",
                )

        schedule = report["schedule"]
        rows = difficulty_rows[team]

        equal(schedule["games"], rows, f"{team}: 난이도 경기 목록")

        valid = [
            row["difficulty"]
            for row in rows
            if row["difficulty"] is not None
        ]
        scheduled = len(rows)
        unassigned = (
            GAMES_PER_TEAM
            - len(team_scores(completed, team))
            - scheduled
        )

        require(unassigned >= 0, "팀 미편성 경기 수 오류")

        expected_games = scheduled + unassigned
        valid_count = len(valid)
        missing = scheduled - valid_count

        status = (
            "no_remaining_games_under_assumption" if not expected_games
            else "schedule_missing" if not scheduled
            else "no_valid_forecast" if not valid_count
            else "partial" if missing or unassigned
            else "supported_under_assumption"
        )

        check_fields(
            schedule,
            {
                "status": status,
                "reason": None if status == "supported_under_assumption" else status,
                "expected_remaining_games": expected_games,
                "valid_scheduled_games": scheduled,
                "independently_confirmed_schedule_games": 0,
                "unverified_schedule_games": scheduled,
                "unassigned_games": unassigned,
                "forecast_games": valid_count,
                "missing_forecasts": missing,
                "forecast_coverage": ratio(valid_count, scheduled),
                "expected_schedule_coverage": ratio(valid_count, expected_games),
                "difficulty_sum": fsum(valid) if valid else None,
                "difficulty": ratio(fsum(valid), valid_count),
                "denominator": valid_count,
                "draw_weight": DRAW_WEIGHT,
                "weighting": "equal_per_valid_fixture",
            },
            f"{team}: 일정 난이도",
        )

    return {
        "cutoff": cutoff.isoformat(),
        "completed_games": len(completed),
        "remaining_games": len(remaining),
        "team_profiles_checked": len(TEAMS),
        "difficulty_sides_checked": len(remaining) * 2,
        "candidate_reasons": dict(candidate_reasons),
        "missing_forecasts": sum(
            row["schedule"]["missing_forecasts"]
            for row in result["teams"].values()
        ),
    }


def audit_support_game(
    game: JsonObject,
    season: JsonObject,
) -> JsonObject:
    """대표 경기와 시즌 분석에서 같은 경기·팀 결과가 일치하는지 확인한다."""
    require(len(game["games"]) == 1, "개별 경기 출력 수 오류")

    selected = game["games"][0]
    game_id = selected["fixture"]["game_id"]

    expected = next(
        row for row in season["games"]
        if row["fixture"]["game_id"] == game_id
    )

    equal(selected, expected, "개별·시즌 경기 계산 일치")

    for team, report in game["teams"].items():
        equal(report, season["teams"][team], f"개별·시즌 팀 {team}")

    return {"game_id": game_id, "matched": True}


def normalize_support(result: JsonObject) -> JsonObject:
    """가변 실행 식별자와 생성 시각만 재현 비교에서 제외한다."""
    return {
        key: value
        for key, value in result.items()
        if key not in {"run_id", "created_utc"}
    }


def stability_metrics(
    summary: JsonObject,
) -> dict[str, tuple[float | None, int, float]]:
    """결과표에서 비교 가능한 확률·기대값 및 분모를 추출한다."""
    metrics = {}

    def visit(node: JsonObject, prefix: str) -> None:
        if "valid" in node and "probability" in node:
            metrics[prefix] = (node["probability"], node["valid"], 1.0)

        if "valid" in node and "probabilities" in node:
            for category, value in node["probabilities"].items():
                metrics[f"{prefix}.{category}"] = (
                    value, node["valid"], 1.0
                )

            scale = (
                float(GAMES_PER_TEAM) if prefix.endswith("final_wins")
                else 9.0 if prefix.endswith(".rank")
                else 1.0
            )

            metrics[f"{prefix}.expected"] = (
                node["expected"], node["valid"], scale
            )

        if "valid" in node and "joint_win_probabilities" in node:
            for team, value in node["joint_win_probabilities"].items():
                metrics[f"{prefix}.joint.{team}"] = (
                    value, node["valid"], 1.0
                )

        for event in ("first", "qualification"):
            key = f"{event}_joint_probabilities"

            if key in node:
                for team, value in node[key].items():
                    metrics[f"{prefix}.{key}.{team}"] = (
                        value,
                        node[f"{event}_joint_valid"],
                        1.0,
                    )

        for key, value in node.items():
            if isinstance(value, dict):
                visit(value, f"{prefix}.{key}")

    visit(summary, "summary")

    return metrics


def stability(
    reference: JsonObject,
    candidate: JsonObject,
) -> JsonObject:
    """Seed·반복 수 변화의 진단적 경고를 사전 기준으로 계산한다."""
    left = stability_metrics(reference)
    right = stability_metrics(candidate)

    equal(set(left), set(right), "안정성 비교 지표 집합")

    comparisons = []

    for key in sorted(left):
        a, na, scale = left[key]
        b, nb, other_scale = right[key]

        equal(scale, other_scale, "안정성 지표 척도")

        difference = None if a is None or b is None else abs(a - b)
        threshold = None
        warning = False

        if (
            difference is not None
            and min(na, nb) >= STABILITY_MIN_VALID
        ):
            floor = (
                STABILITY_PROBABILITY_FLOOR
                if scale == 1.0
                else STABILITY_MEAN_FLOOR
            )
            threshold = max(
                floor,
                STABILITY_MULTIPLIER * scale
                * sqrt(0.25 / na + 0.25 / nb),
            )
            warning = difference > threshold

        comparisons.append({
            "metric": key,
            "reference": a,
            "candidate": b,
            "reference_valid": na,
            "candidate_valid": nb,
            "absolute_change": difference,
            "warning_threshold": threshold,
            "warning": warning,
        })

    return {
        "comparison_count": len(comparisons),
        "warning_count": sum(row["warning"] for row in comparisons),
        "diagnostic_only": True,
        "comparisons": comparisons,
    }


def official_diagnostic(
    regular: Run,
    postseason: Run,
    games: Sequence[JsonObject],
) -> JsonObject:
    """2025년 확인 사실에 대한 개발 시나리오 차이를 기록한다."""
    events = {}

    probabilities = regular.summary["league"]["first_joint_probabilities"]

    if probabilities[str(OFFICIAL_SEEDS[0])] is not None:
        events["regular_first"] = {
            "actual": OFFICIAL_SEEDS[0],
            "probability": probabilities[str(OFFICIAL_SEEDS[0])],
            "valid": regular.summary["league"]["first_joint_valid"],
            "brier": fsum(
                (value - int(int(team) == OFFICIAL_SEEDS[0])) ** 2
                for team, value in probabilities.items()
            ),
        }

    for stage, winner in OFFICIAL_WINNERS.items():
        league = postseason.summary["league"][stage]
        probabilities = league["joint_win_probabilities"]

        events[stage] = {
            "actual": winner,
            "probability": probabilities[str(winner)],
            "valid": league["valid"],
            "brier": (
                fsum(
                    (value - int(int(team) == winner)) ** 2
                    for team, value in probabilities.items()
                )
                if league["valid"] else None
            ),
        }

    errors = {}

    for team in TEAMS:
        actual = record_stats(team_scores(games, team))["wins"]
        predicted = regular.summary["teams"][str(team)]["final_wins"]["expected"]

        errors[str(team)] = {
            "archive_actual_wins": actual,
            "predicted_wins": predicted,
            "absolute_error": abs(actual - predicted) if predicted is not None else None,
        }

    valid = [
        item["absolute_error"]
        for item in errors.values()
        if item["absolute_error"] is not None
    ]

    return {
        "scope": "retrospective_development_diagnostic",
        "formal_backtest_supported": False,
        "joint_valid_conditioning": True,
        "official_events": events,
        "archive_win_errors": errors,
        "archive_win_mae": ratio(fsum(valid), len(valid)),
        "official_references": OFFICIAL_REFERENCES,
    }


def season_availability(db: sqlite3.Connection) -> list[JsonObject]:
    """시즌·라운드별 자료 가용성과 현재 평가 자격을 구분한다."""
    counts = Counter()

    for season, competition, stage, count in db.execute(
        """
        SELECT season, competition, stage, COUNT(*)
        FROM eligible_games
        WHERE season IN (2025, 2026)
        GROUP BY season, competition, stage
        """
    ):
        key = "regular" if competition == "regular" else stage
        counts[(season, key)] += count

    result = []

    for season in (2025, 2026):
        for stage in ("regular", *STAGES):
            result.append({
                "season": season,
                "stage": stage,
                "eligible_archive_games": counts[(season, stage)],
                "rule_supported": season == SEASON,
                "calculation_replay": (
                    "development_assumption"
                    if season == SEASON else "unsupported"
                ),
                "official_fact_comparison": (
                    "selected_event_verified"
                    if season == SEASON else "not_evaluated"
                ),
                "formal_backtest_supported": False,
                "reason": (
                    "archive_home_conflict_and_availability_unverified"
                    if season == SEASON and stage == "koreanSeries"
                    else "historical_information_availability_unverified"
                    if season == SEASON
                    else "unsupported_season_rules"
                ),
            })

    return result
