"""포스트시즌 입력 자료 및 선행 실행 계보 검증."""

from collections import defaultdict
from contextlib import closing
from dataclasses import asdict, dataclass, replace
from datetime import date
import json
from pathlib import Path
import sqlite3
from typing import cast

from ...datasets.published_sources import (
    RUN_PATTERN,
    as_row,
    read_json,
    sha256_file,
)
from ...models.game_outcome import model as m1
from ...settings import Settings
from ...validation.standard_data import team_id

from ..regular_season import (
    ITERATION_SCHEMA as PARENT_ITERATION_SCHEMA,
    SCENARIO_SCHEMA as PARENT_SCENARIO_SCHEMA,
    SUMMARY_SCHEMA as PARENT_SUMMARY_SCHEMA,
    VERSION as PARENT_VERSION,
)
from ..regular_season.data import (
    cutoff_timestamp,
    fixture_from_values,
    fixture_to_dict,
    open_readonly,
    parse_cutoff,
    parse_day,
    read_completed,
    verify_assets,
)
from ..regular_season.engine import (
    Iteration,
    build_state,
    make_iteration,
    summarize as summarize_regular,
)
from ..regular_season.rules import (
    RULE_VERSION as RANK_RULE_VERSION,
    PlayedGame,
    SeasonState,
    get_rules as get_regular_rules,
    resolve_standings,
    validate_schedule,
)

from .engine import ArchiveState, SeedInput
from .rules import (
    STAGES,
    Game,
    Outcome,
    SeriesState,
    get_rules,
)


PARENT_FILES = {
    "scenario": "scenario.json",
    "summary": "summary.json",
    "iterations": "iterations.jsonl",
}

PARENT_SOURCE_FILES = {
    "__init__.py",
    "__main__.py",
    "data.py",
    "engine.py",
    "rules.py",
}

PARTICIPANT_KEYS = {
    "wildCard": ("team4", "team5"),
    "semiPlayoff": ("team3", "adv"),
    "playoff": ("team2", "adv"),
    "koreanSeries": ("team1", "adv"),
}

ARCHIVE_LIMITATIONS = (
    "2025 WC 1차전 점수가 공식 기록과 불일치",
    "2025 PO 5차전 날짜 및 점수가 공식 기록과 불일치",
    "2025 KS 경기 일정 및 5차전 홈 배정이 공식 기록과 불일치",
    "사후 배포 자료이며 당시 공개 시각을 입증하지 못함",
    "원천 판본 재생은 공식 경기 단위 정답 검증이 아님",
)

ARCHIVE_REFERENCE_URLS = (
    "https://www.yna.co.kr/view/AKR20251006019300007",
    "https://www.koreabaseball.com/MediaNews/News/"
    "BreakingNews/View.aspx?bdSe=60367",
    "https://www.koreabaseball.com/MediaNews/Notice/"
    "View.aspx?bdSe=11730",
    "https://www.koreabaseball.com/MediaNews/Notice/"
    "View.aspx?bdSe=11744",
)


@dataclass
class Inputs:
    """검증된 시뮬레이션 입력과 인수 상태."""

    season: int
    cutoff: date
    model: m1.Model
    state: SeasonState
    draws: tuple[SeedInput, ...]
    archive: ArchiveState
    metadata: dict[str, object]


def require(condition: bool, message: str) -> None:
    """입력 계약 위반 시 명확한 오류를 발생시킨다."""
    if not condition:
        raise ValueError(message)


def seed_input(
    row: Iteration,
    teams: tuple[int, ...],
) -> SeedInput:
    """반복별 순위·진출·시드의 의미와 타입을 검사한다."""
    require(
        row["schema"] == PARENT_ITERATION_SCHEMA,
        "부모 반복 스키마 불일치",
    )

    require(
        type(row["iteration_id"]) is int
        and row["iteration_id"] >= 1,
        "부모 반복 ID 오류",
    )

    require(
        set(row["teams"]) == {
            str(team) for team in teams
        },
        "부모 반복 구단 집합 불일치",
    )

    qualifications: dict[int, bool | None] = {}
    ranks: dict[int, int] = {}
    first_teams: list[int] = []
    reasons: set[str] = set()

    for team in teams:
        item = row["teams"][str(team)]

        rank = item["rank"]
        first = item["first"]
        qualifies = item["qualifies"]

        require(
            first is None or type(first) is bool,
            "정규시즌 1위 상태 타입 오류",
        )

        require(
            qualifies is None
            or type(qualifies) is bool,
            "정규시즌 진출 상태 타입 오류",
        )

        qualifications[team] = qualifies

        if first is True:
            first_teams.append(team)

        if rank is not None:
            require(
                type(rank) is int
                and 1 <= rank <= len(teams),
                "부모 반복 순위 범위 오류",
            )

            require(
                rank not in ranks,
                "확정 순위 중복",
            )

            ranks[rank] = team

            require(
                first == (rank == 1)
                and qualifies == (rank <= 5),
                "순위와 1위·진출 상태 불일치",
            )

        reason = item["reason"]

        require(
            reason is None or isinstance(reason, str),
            "부모 반복 미결정 사유 타입 오류",
        )

        if reason is not None:
            reasons.add(reason)

    require(
        len(first_teams) <= 1,
        "정규시즌 1위 구단 중복",
    )

    rank_resolved = len(ranks) == len(teams)

    qualification_resolved = all(
        value is not None
        for value in qualifications.values()
    )

    require(
        type(row["rank_resolved"]) is bool
        and row["rank_resolved"] == rank_resolved,
        "순위 확정 표시 불일치",
    )

    require(
        type(row["qualification_resolved"]) is bool
        and row["qualification_resolved"]
        == qualification_resolved,
        "진출 확정 표시 불일치",
    )

    qualifiers = row["qualifiers"]

    if qualification_resolved:
        expected = sorted(
            team
            for team, status in qualifications.items()
            if status is True
        )

        require(
            isinstance(qualifiers, list)
            and all(type(team) is int for team in qualifiers)
            and qualifiers == expected
            and len(expected) == 5,
            "부모 반복 진출 구단 불일치",
        )

    else:
        require(
            qualifiers is None,
            "미결정 진출 구단의 임의 확정",
        )

    seeds = row["postseason_seeds"]

    expected_seeds = None

    if qualification_resolved and all(
        rank in ranks for rank in range(1, 6)
    ):
        expected_seeds = [
            ranks[rank]
            for rank in range(1, 6)
        ]

    require(
        seeds == expected_seeds,
        "부모 반복 시드와 확정 순위 불일치",
    )

    if seeds is not None:
        require(
            isinstance(seeds, list)
            and all(type(team) is int for team in seeds)
            and len(seeds) == 5
            and sorted(seeds) == qualifiers,
            "부모 반복 시드 구성 오류",
        )

    return SeedInput(
        iteration_id=row["iteration_id"],
        seeds=tuple(seeds) if seeds is not None else None,
        qualifiers=(
            tuple(qualifiers)
            if qualifiers is not None
            else None
        ),
        qualification=qualifications,
        regular_first=(
            first_teams[0] if first_teams else None
        ),
        reasons=tuple(sorted(reasons)),
    )


def verify_parent_provenance(
    parent: dict[str, object],
    current: dict[str, object],
) -> None:
    """선행 실행이 같은 DB·모델 자산을 사용했는지 검사한다."""
    fields = (
        "model_data_sha256",
        "standard_run_id",
        "standard_verification_id",
        "standard_database_sha256",
        "model_run_id",
        "model_schema",
        "model_asset_sha256",
    )

    for name in fields:
        require(
            parent[name] == current[name],
            f"부모 실행 자산 계보 불일치: {name}",
        )

    parent_data = as_row(parent["model_data"])
    current_data = as_row(current["model_data"])

    for name in (
        "model_data_run_id",
        "contract_sha256",
        "manifest_sha256",
    ):
        require(
            parent_data[name] == current_data[name],
            f"부모 실행 모델 자료 계약 불일치: {name}",
        )


def read_parent(
    settings: Settings,
    parent_run: str,
    season: int,
    cutoff: date,
    completed: tuple[PlayedGame, ...],
    provenance: dict[str, object],
) -> tuple[tuple[SeedInput, ...], dict[str, object]]:
    """선행 실행의 파일·메타데이터·반복 결과를 교차 검증한다."""
    require(
        RUN_PATTERN.fullmatch(parent_run) is not None,
        "부모 실행 ID 형식 오류",
    )

    root = (
        settings.results_dir
        / "regular_season"
        / parent_run
    )

    manifest_path = root / "manifest.json"
    manifest = as_row(read_json(manifest_path))

    require(
        manifest["status"] == "completed"
        and manifest["run_id"] == parent_run
        and manifest["version"] == PARENT_VERSION,
        "부모 실행 상태 또는 버전 불일치",
    )

    require(
        manifest["season"] == season
        and manifest["as_of_kst"] == cutoff_timestamp(cutoff),
        "부모 실행 시즌 또는 기준 시각 불일치",
    )

    require(
        manifest["final_test_used"] is False,
        "최종 Test가 사용된 부모 실행은 인수할 수 없습니다.",
    )

    require(
        manifest["schemas"] == {
            "scenario": PARENT_SCENARIO_SCHEMA,
            "summary": PARENT_SUMMARY_SCHEMA,
            "iterations": PARENT_ITERATION_SCHEMA,
        },
        "부모 실행 출력 스키마 불일치",
    )

    files = as_row(manifest["files"])

    digests: dict[str, str] = {}

    for name, filename in PARENT_FILES.items():
        path = root / filename
        actual = sha256_file(path)

        require(
            actual == as_row(files[name])["sha256"],
            f"부모 실행 파일 해시 불일치: {filename}",
        )

        digests[filename] = actual

    source_hashes = as_row(manifest["source_sha256"])

    require(
        set(source_hashes) == PARENT_SOURCE_FILES,
        "부모 실행 소스 파일 목록 불일치",
    )

    source_root = (
        settings.project_root
        / "src"
        / "simulation"
        / "regular_season"
    )

    for name, digest in source_hashes.items():
        require(
            sha256_file(source_root / name) == digest,
            f"부모 실행 소스 판본 불일치: {name}",
        )

    scenario = as_row(
        read_json(root / "scenario.json")
    )

    summary = as_row(
        read_json(root / "summary.json")
    )

    metadata = as_row(scenario["metadata"])

    require(
        scenario["schema"] == PARENT_SCENARIO_SCHEMA
        and summary["schema"] == PARENT_SUMMARY_SCHEMA
        and metadata == manifest["metadata"],
        "부모 실행 시나리오 또는 요약 계약 불일치",
    )

    require(
        metadata["season"] == season
        and metadata["as_of_kst"]
        == cutoff_timestamp(cutoff)
        and metadata["final_test_used"] is False,
        "부모 실행 정보 경계 불일치",
    )

    require(
        as_row(metadata["rules"])["version"]
        == RANK_RULE_VERSION,
        "부모 실행 정규시즌 규칙 판본 불일치",
    )

    verify_parent_provenance(
        as_row(metadata["provenance"]),
        provenance,
    )

    expected_completed = [
        {
            **fixture_to_dict(game.fixture),
            "home_runs": game.home_runs,
            "away_runs": game.away_runs,
        }
        for game in completed
    ]

    require(
        scenario["completed"] == expected_completed
        and metadata["completed_games"] == len(completed),
        "부모 실행 완료 경기와 표준·모델 DB 불일치",
    )

    remaining = []

    require(
        isinstance(scenario["remaining"], list),
        "부모 실행 잔여 일정 형식 오류",
    )

    for raw in scenario["remaining"]:
        game = as_row(raw)

        fixture = fixture_from_values(
            game["game_id"],
            game["date"],
            game["game_number"],
            game["home_team_id"],
            game["away_team_id"],
        )

        require(
            fixture.day >= cutoff,
            "기준일 이전 경기의 잔여 일정 혼입",
        )

        remaining.append(fixture)

    regular_rules = get_regular_rules(season)

    missing = validate_schedule(
        regular_rules,
        [
            *(game.fixture for game in completed),
            *remaining,
        ],
    )

    require(
        metadata["remaining_games"] == len(remaining)
        and metadata["missing_pairs"] == missing
        and metadata["missing_games"] == sum(
            item["missing_games"]
            for item in missing
        ),
        "부모 실행 잔여·누락 경기 수 불일치",
    )

    requested = manifest["iterations"]

    require(
        type(requested) is int and requested > 0,
        "부모 실행 반복 횟수 오류",
    )

    rows: list[Iteration] = []

    with (root / "iterations.jsonl").open(
        encoding="utf-8",
    ) as stream:
        for line in stream:
            row = cast(
                Iteration,
                as_row(json.loads(line)),
            )

            require(
                row["iteration_id"] == len(rows) + 1,
                "부모 실행 반복 ID 누락 또는 중복",
            )

            rows.append(row)

    require(
        len(rows) == requested,
        "부모 실행 반복 수 불일치",
    )

    draws = tuple(
        seed_input(row, regular_rules.teams)
        for row in rows
    )

    require(
        summarize_regular(regular_rules, rows) == summary,
        "부모 실행 요약 재집계 불일치",
    )

    return draws, {
        "run_id": parent_run,
        "manifest_sha256": sha256_file(manifest_path),
        "files_sha256": digests,
        "schemas": manifest["schemas"],
        "seed": manifest["seed"],
        "iterations": requested,
        "metadata": metadata,
        "seed_resolved": sum(
            draw.seeds is not None
            for draw in draws
        ),
    }


def read_schedule_rows(
    db: sqlite3.Connection,
    table: str,
    season: int,
    cutoff: date,
) -> list[tuple]:
    """기준일 이전에 해당하는 일정 근거만 조회한다."""
    require(
        table in (
            "postseason_schedule",
            "season_schedule_evidence",
        ),
        "허용하지 않는 일정 근거 테이블",
    )

    return db.execute(
        f"""
        SELECT source_sha, source_row, stage,
               series_game_number, date,
               away_name, home_name,
               final_observed, linked_game_id
        FROM {table}
        WHERE substr(date, 1, 4) = ?
          AND date < ?
        ORDER BY date, series_game_number,
                 source_sha, source_row
        """,
        (
            str(season),
            cutoff.isoformat(),
        ),
    ).fetchall()


def verify_completion(
    standard: sqlite3.Connection,
    model_db: sqlite3.Connection,
    season: int,
    state: SeriesState,
    seeds: tuple[int, ...],
    sources: set[str],
) -> dict[str, object]:
    """재생으로 종료된 시리즈의 사후 완료 증거를 대조한다."""
    stage = state.rule.stage

    standard_rows = standard.execute(
        """
        SELECT source_sha, series_complete, metadata_json
        FROM postseason_series
        WHERE season = ? AND stage = ?
        ORDER BY source_sha
        """,
        (season, stage),
    ).fetchall()

    model_rows = model_db.execute(
        """
        SELECT source_sha, series_complete, metadata_json
        FROM season_series_evidence
        WHERE season = ? AND stage = ?
        ORDER BY source_sha
        """,
        (season, stage),
    ).fetchall()

    require(
        standard_rows == model_rows
        and len(standard_rows) == 1,
        "시리즈 완료 근거 누락 또는 불일치",
    )

    source, complete, raw = standard_rows[0]

    require(
        complete == 1 and sources == {source},
        "시리즈 완료 또는 출처 연결 불일치",
    )

    evidence = as_row(json.loads(raw))

    require(
        evidence["isComplete"] is True
        and evidence["_score_rows"] == len(state.games),
        "시리즈 종료 근거와 재생 경기 수 불일치",
    )

    archived_seeds = as_row(evidence["seeds"])

    expected = tuple(
        as_row(archived_seeds[str(rank)])["id"]
        for rank in range(1, 6)
    )

    require(
        expected == seeds,
        "원천 시드와 정규시즌 확정 시드 불일치",
    )

    higher_key, lower_key = PARTICIPANT_KEYS[stage]

    higher = as_row(evidence[higher_key])
    lower = as_row(evidence[lower_key])

    require(
        higher["id"] == state.higher
        and lower["id"] == state.lower
        and higher["wins"] == state.higher_wins
        and lower["wins"] == state.lower_wins,
        "시리즈 참가 구단 또는 실제 승수 불일치",
    )

    winner = evidence.get("winnerId")

    if winner is None:
        winner = evidence.get("championId")

    require(
        winner == state.winner
        and higher["isWinner"]
        == (state.winner == state.higher)
        and lower["isWinner"]
        == (state.winner == state.lower),
        "시리즈 재생 승자와 종료 증거 불일치",
    )

    return {
        "source_sha": source,
        "series_complete": True,
        "observed_games": len(state.games),
        "winner": state.winner,
        "higher_wins": state.higher_wins,
        "lower_wins": state.lower_wins,
        "ties": state.ties,
        "evidence_kind": "retrospective_archive",
        "official_record_verified": False,
    }


def read_archive(
    standard: sqlite3.Connection,
    model_db: sqlite3.Connection,
    season: int,
    cutoff: date,
    seeds: tuple[int, ...] | None,
) -> tuple[ArchiveState, dict[str, object]]:
    """사후 원천 판본의 기준일 이전 경기와 일정 근거를 연결한다."""
    archive = ArchiveState()

    schedule = read_schedule_rows(
        standard,
        "postseason_schedule",
        season,
        cutoff,
    )

    model_schedule = read_schedule_rows(
        model_db,
        "season_schedule_evidence",
        season,
        cutoff,
    )

    require(
        schedule == model_schedule,
        "표준·모델 일정 근거 불일치",
    )

    standard_rows = standard.execute(
        """
        SELECT game_id, source_key, stage, date,
               game_number, home_team_id,
               away_team_id, home_score,
               away_score, winner,
               source_sha, source_row
        FROM eligible_games
        WHERE season = ?
          AND competition = 'postseason'
          AND date < ?
        ORDER BY date, game_number, game_id
        """,
        (season, cutoff.isoformat()),
    ).fetchall()

    by_id = {
        row[0]: row
        for row in standard_rows
    }

    require(
        len(by_id) == len(standard_rows),
        "표준 포스트시즌 경기 ID 중복",
    )

    groups: dict[str, list[Game]] = defaultdict(list)
    sources: dict[str, set[str]] = defaultdict(set)
    matched: set[str] = set()

    for item in schedule:
        (
            source,
            source_row,
            stage,
            number,
            day,
            away_name,
            home_name,
            final_observed,
            linked,
        ) = item

        require(
            stage in STAGES,
            f"지원하지 않는 포스트시즌 라운드: {stage}",
        )

        try:
            require(
                parse_day(day) < cutoff
                and type(number) is int
                and number >= 1
                and final_observed in (0, 1),
                "일정 날짜·차전·완료 표시 오류",
            )

            require(
                isinstance(linked, str)
                and linked in by_id,
                "일정만 존재하거나 완료 경기 연결 없음",
            )

            require(
                linked not in matched,
                "일정과 완료 경기 중복 연결",
            )

            matched.add(linked)

            (
                game_id,
                source_key,
                game_stage,
                game_day,
                game_number,
                home,
                away,
                home_score,
                away_score,
                winner,
                game_source,
                game_source_row,
            ) = by_id[linked]

            expected_key = f"{season}:{stage}:{number}"

            require(
                source_key == expected_key
                and game_id
                == f"postseason:dent:{expected_key}",
                "포스트시즌 경기 키 불일치",
            )

            require(
                game_stage == stage
                and game_day == day
                and game_number == number
                and game_source == source
                and game_source_row == source_row
                and home == team_id(home_name)
                and away == team_id(away_name),
                "일정과 표준 경기의 차전·구단·출처 불일치",
            )

            # final_observed=0이어도 유효 표준 결과와
            # 별도 시리즈 완료 근거를 통해 재생할 수 있다.
            game = Game(
                number=number,
                home=home,
                away=away,
                outcome=cast(Outcome, winner),
                game_id=game_id,
                day=day,
                home_score=home_score,
                away_score=away_score,
            )

            groups[stage].append(game)
            sources[stage].add(source)

        except (ValueError, TypeError, KeyError) as exc:
            archive.blocked.setdefault(
                stage,
                f"archive_input_conflict: {exc}",
            )

    for game_id, row in by_id.items():
        if game_id in matched:
            continue

        stage = row[2]

        require(
            stage in STAGES,
            "미지원 라운드의 완료 경기",
        )

        archive.blocked.setdefault(
            stage,
            "completed_game_without_schedule_evidence",
        )

    archive.games = {
        stage: tuple(
            sorted(
                games,
                key=lambda game: game.number,
            )
        )
        for stage, games in groups.items()
    }

    if seeds is not None:
        audit_archive_bracket(
            standard,
            model_db,
            season,
            seeds,
            archive,
            sources,
        )

    return archive, {
        "schedule_rows": len(schedule),
        "eligible_postseason_games": len(standard_rows),
        "matched_games": len(matched),
        "schedule_records": schedule,
        "games": {
            stage: [asdict(game) for game in games]
            for stage, games in archive.games.items()
        },
        "blocked": archive.blocked,
        "completion_evidence": archive.completion_evidence,
        "official_record_verified": False,
        "limitations": ARCHIVE_LIMITATIONS,
        "references": ARCHIVE_REFERENCE_URLS,
    }


def audit_archive_bracket(
    standard: sqlite3.Connection,
    model_db: sqlite3.Connection,
    season: int,
    seeds: tuple[int, ...],
    archive: ArchiveState,
    sources: dict[str, set[str]],
) -> None:
    """기준일 이전 결과만 사용해 확정 시리즈의 연결을 검증한다."""
    rules = get_rules(season)
    advancing = seeds[4]

    for index, rule in enumerate(rules):
        stage = rule.stage

        if stage in archive.blocked:
            return

        state = SeriesState(
            rule,
            seeds[rule.seed_index],
            advancing,
        )

        try:
            for game in archive.games.get(stage, ()):
                state.apply(game)

            if state.winner is None:
                later_stages = STAGES[index + 1:]

                require(
                    not any(
                        archive.games.get(later)
                        or later in archive.blocked
                        for later in later_stages
                    ),
                    "이전 라운드 미종료 상태에서 "
                    "후속 라운드 경기 발견",
                )

                # 완료되지 않은 시리즈의 사후 승자 또는
                # 최종 승수를 조회하지 않는다.
                return

            archive.completion_evidence[stage] = (
                verify_completion(
                    standard,
                    model_db,
                    season,
                    state,
                    seeds,
                    sources[stage],
                )
            )

            advancing = state.winner

        except (ValueError, TypeError, KeyError) as exc:
            archive.blocked[stage] = (
                f"archive_replay_conflict: {exc}"
            )
            return


def load_inputs(
    settings: Settings,
    run_id: str,
    asset: Path,
    season: int,
    as_of: str,
    parent_run: str | None,
    iterations: int | None,
    archive_assumption: bool,
) -> Inputs:
    """실행 모드를 검증하고 기준일 이전 입력만 구성한다."""
    get_rules(season)
    cutoff = parse_cutoff(as_of, season)

    if parent_run is not None:
        require(
            not archive_assumption,
            "부모 실행 모드와 원천 판본 재생 모드는 분리합니다.",
        )

        require(
            iterations is None,
            "부모 실행 반복 수는 임의 변경할 수 없습니다.",
        )

    else:
        require(
            archive_assumption,
            "현재 포스트시즌 자료는 공식 기록과 불일치하므로 "
            "--archive-assumption의 명시적 선택이 필요합니다.",
        )

        require(
            type(iterations) is int and iterations > 0,
            "고정 시드 모드의 반복 횟수 오류",
        )

    standard_path, model_path, model, provenance = (
        verify_assets(settings, run_id, asset)
    )

    regular_rules = get_regular_rules(season)

    with (
        closing(open_readonly(standard_path)) as standard,
        closing(open_readonly(model_path)) as model_db,
    ):
        completed = read_completed(
            standard,
            model_db,
            season,
            cutoff,
        )

        state = build_state(
            regular_rules,
            completed,
        )

        archive = ArchiveState()
        parent_info = None
        archive_info = None

        if parent_run is not None:
            draws, parent_info = read_parent(
                settings,
                parent_run,
                season,
                cutoff,
                completed,
                provenance,
            )

            require(
                len(completed) < regular_rules.total_games
                or all(
                    draw.seeds == draws[0].seeds
                    for draw in draws
                ),
                "정규시즌 종료 후 부모 시드 불일치",
            )

            require(
                len(completed) < regular_rules.total_games,
                "정규시즌 종료 이후의 기준일은 "
                "원천 판본 재생 모드를 사용합니다.",
            )

            mode = "regular_season_joint_scenario"

        else:
            require(
                len(completed) == regular_rules.total_games
                and not validate_schedule(
                    regular_rules,
                    [
                        game.fixture
                        for game in completed
                    ],
                ),
                "정규시즌 완료 및 고정 시드 근거 부족",
            )

            first_draw = seed_input(
                make_iteration(
                    1,
                    state,
                    resolve_standings(state),
                ),
                regular_rules.teams,
            )

            draws = tuple(
                replace(
                    first_draw,
                    iteration_id=index,
                )
                for index in range(1, iterations + 1)
            )

            archive, archive_info = read_archive(
                standard,
                model_db,
                season,
                cutoff,
                first_draw.seeds,
            )

            mode = "retrospective_archive_assumption"

    metadata = {
        "season": season,
        "as_of_kst": cutoff_timestamp(cutoff),
        "timezone": "Asia/Seoul",
        "mode": mode,
        "provenance": provenance,
        "parent": parent_info,
        "archive": archive_info,
        "regular_completed_games": len(completed),
        "profile_policy": (
            "fixed_regular_history_before_cutoff"
        ),
        "prediction_method": (
            "existing_regular_m1_result_probability_transfer"
        ),
        "formal_backtest_supported": False,
        "official_postseason_record_verified": False,
        "final_test_used": False,
        "limitations": [
            "M1의 학습 및 선정은 정규시즌 자료에 한정",
            "운영 M1의 기준일 당시 이용 가능성 미입증",
            "포스트시즌 최대 15회 연장을 직접 모델링하지 않음",
            "2025년 정규시즌 무승부 예측 2.07%, 관측 3.06%",
            "과거 일정 및 경기 결과 공개 시각 미입증",
            "구체적인 미래 개최 날짜 및 우천 순연 예측 제외",
            "2001~2009년 및 2021~2023년 경기 결과 공백",
            "2024년 한국시리즈 이전 세 라운드 점수 공백",
            *ARCHIVE_LIMITATIONS,
        ],
    }

    return Inputs(
        season=season,
        cutoff=cutoff,
        model=model,
        state=state,
        draws=draws,
        archive=archive,
        metadata=metadata,
    )