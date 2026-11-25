"""기존 검증판을 기반으로 포스트시즌 및 정규시즌 완료 근거를 보완한다."""

import argparse
from collections import Counter, defaultdict
from collections.abc import Sequence
from contextlib import closing
from datetime import date, timedelta
import json
import logging
from pathlib import Path
import shutil
import sqlite3
from uuid import uuid4

from . import model_data
from .published_sources import (
    EXPECTED_TARGETS,
    POSTSEASON_STAGES,
    RUN_PATTERN,
    SourceVersion,
    as_row,
    intake_sources,
    read_interim_rows,
    read_json,
    sha256_file,
    write_json,
)
from .standardize import convert_postseason, coverage_report
from ..settings import Settings, load_settings
from ..validation.standard_data import (
    POSTSEASON_NAMES,
    TABLE_NAMES,
    TEAM_CODES,
    StandardStore,
    compare_summaries,
    normalized_date,
    required_integer,
    team_id,
)
from ..validation.standard_quality import (
    AUDIT_TABLES,
    _digest_tables,
    _handoff,
)


LOGGER = logging.getLogger(__name__)
VERSION = "1.0.0"

Row = dict[str, object]

SUPPLEMENT_SEASONS = (2019, 2020, 2024, 2025)

SUPPLEMENT_TARGETS = frozenset(
    f"dent-postseason-{season}"
    for season in SUPPLEMENT_SEASONS
)

SUPPLEMENT_REVISION = "399d27b99322ff6b9a28d4910ac61e9bf140a9ca"

EXPECTED_ROWS = {
    2019: 12,
    2020: 13,
    2024: 5,
    2025: 16,
}

STAGE_TEAMS = {
    "wildCard": ("team4", "team5"),
    "semiPlayoff": ("team3", "adv"),
    "playoff": ("team2", "adv"),
    "koreanSeries": ("team1", "adv"),
}

REGULAR_COMPLETION = {
    2023: {
        "last_date": "2023-10-17",
        "published_date": "2023-10-18",
        "games": 720,
        "notice_url": (
            "https://www.koreabaseball.com/"
            "MediaNews/Notice/View.aspx?bdSe=9846"
        ),
    },
    2024: {
        "last_date": "2024-10-01",
        "published_date": "2024-10-02",
        "games": 720,
        "notice_url": (
            "https://www.koreabaseball.com/"
            "MediaNews/Notice/View.aspx?bdSe=10231"
        ),
    },
    2025: {
        "last_date": "2025-10-04",
        "published_date": "2025-10-05",
        "games": 720,
        "notice_url": (
            "https://www.koreabaseball.com/"
            "MediaNews/Notice/View.aspx?bdSe=11708"
        ),
    },
}

VALIDATION_START = "2025-01-01"
TEST_START = "2026-01-01"
OBSERVATION_END = "2026-11-26"


def require(condition: bool, message: str) -> None:
    """검증 기준을 충족하지 못하면 완료 처리하지 않는다."""
    if not condition:
        raise ValueError(message)


def winner_id(
    away_id: int,
    home_id: int,
    away_score: int,
    home_score: int,
) -> int | None:
    """점수에 근거한 승리 팀을 반환한다."""
    if away_score == home_score:
        return None

    return away_id if away_score > home_score else home_id


def inspect_series(
    stage: str,
    series: Row,
    rows: list[Row],
) -> None:
    """완료 시리즈의 실제 경기 승수와 원본 시리즈 승수를 대조한다."""
    if series.get("isComplete") is not True or not rows:
        return

    participants = [
        as_row(series[key])
        for key in STAGE_TEAMS[stage]
    ]

    ids = [
        required_integer(participant["id"])
        for participant in participants
    ]

    require(
        len(set(ids)) == 2,
        f"시리즈 참가 팀 중복: {stage}",
    )

    wins: Counter[int] = Counter()

    for row in rows:
        require(
            {
                team_id(row["away"]),
                team_id(row["home"]),
            } == set(ids),
            f"시리즈 참가 팀 불일치: {stage}",
        )

        if (
            row["_completion_basis"] != "not_final"
            and row.get("winnerId") is not None
        ):
            wins[required_integer(row["winnerId"])] += 1

    for participant in participants:
        participant_id = required_integer(participant["id"])

        require(
            wins[participant_id]
            == required_integer(participant["wins"]),
            f"시리즈 실제 승수 불일치: {stage}",
        )

    champion = series.get("winnerId")

    if champion is None:
        champion = series.get("championId")

    require(
        champion in ids,
        f"시리즈 승자 미확인: {stage}",
    )

    require(
        {
            participant["id"]
            for participant in participants
            if participant.get("isWinner") is True
        } == {champion},
        f"시리즈 승자 표시 불일치: {stage}",
    )

    effective_wins = dict(wins)

    # 와일드카드의 원본 명시 어드밴티지를 경기 승수와 구분한다.
    for participant in participants:
        participant_id = required_integer(participant["id"])
        advantage = participant.get("advantageWins", 0)

        if (
            stage == "wildCard"
            and participant.get("hasAdvantage") is True
            and "+1 Adv" in str(series.get("format"))
        ):
            advantage = 1

        effective_wins[participant_id] = (
            wins[participant_id]
            + required_integer(advantage)
        )

    require(
        effective_wins[champion]
        >= max(effective_wins.values()),
        f"시리즈 승자와 승수 불일치: {stage}",
    )

    if series.get("winsNeeded") is not None:
        require(
            effective_wins[champion]
            >= required_integer(series["winsNeeded"]),
            f"시리즈 종료 조건 불일치: {stage}",
        )


def flatten_postseason(
    raw: Row,
    season: int,
) -> tuple[list[Row], list[Row]]:
    """실제로 존재하는 포스트시즌 배열만 펼치고 종료 근거를 구분한다."""
    require(raw.get("season") == season, "원본 시즌 불일치")

    postseason = as_row(raw["postseason"])
    rows: list[Row] = []
    series_records: list[Row] = []
    keys: set[tuple[str, int]] = set()

    for stage in POSTSEASON_STAGES:
        if stage not in postseason:
            continue

        series = as_row(postseason[stage])
        games = series.get("games")

        if games is not None:
            require(
                isinstance(games, list),
                f"시리즈 배열 형식 오류: {season}/{stage}",
            )

        stage_rows: list[Row] = []
        previous_date: str | None = None

        for item in games or []:
            game = as_row(item)

            require(
                not any(
                    key.startswith("_")
                    for key in game
                ),
                "파생 근거 필드명과 원본 충돌",
            )

            day = normalized_date(game["date"])
            number = required_integer(game["game"])
            away = team_id(game["away"])
            home = team_id(game["home"])

            require(
                day.startswith(f"{season}-")
                and number >= 1
                and away != home,
                "포스트시즌 경기 식별 정보 오류",
            )

            require(
                previous_date is None or previous_date <= day,
                "시리즈 경기 날짜 순서 오류",
            )

            require(
                (stage, number) not in keys,
                "포스트시즌 경기 차수 중복",
            )

            keys.add((stage, number))
            previous_date = day

            if game.get("isFinal") is True:
                basis = "explicit_is_final"

            elif (
                "isFinal" not in game
                and series.get("isComplete") is True
                and game.get("awayScore") is not None
                and game.get("homeScore") is not None
                and "winnerId" in game
            ):
                basis = "completed_series_score_winner"

            else:
                basis = "not_final"

            if basis != "not_final":
                away_score = required_integer(game["awayScore"])
                home_score = required_integer(game["homeScore"])

                require(
                    game.get("winnerId") == winner_id(
                        away,
                        home,
                        away_score,
                        home_score,
                    ),
                    "포스트시즌 점수와 승자 불일치",
                )

            stage_rows.append({
                **game,
                "_stage": stage,
                "_series_complete": series.get("isComplete"),
                "_completion_basis": basis,
                "_source_row": len(rows) + len(stage_rows),
            })

        inspect_series(stage, series, stage_rows)

        score_rows = sum(
            row["_completion_basis"] != "not_final"
            for row in stage_rows
        )

        metadata = {
            key: value
            for key, value in series.items()
            if key != "games"
        }

        metadata["seeds"] = postseason.get("seeds")
        metadata["_games_array_present"] = isinstance(games, list)
        metadata["_score_rows"] = score_rows
        metadata["_completion_basis_counts"] = dict(
            Counter(
                row["_completion_basis"]
                for row in stage_rows
            )
        )

        series_records.append({
            "stage": stage,
            "series_complete": series.get("isComplete"),
            "rows": len(stage_rows),
            "score_rows": score_rows,
            "metadata": metadata,
        })

        rows.extend(stage_rows)

    all_rounds_observed = (
        {record["stage"] for record in series_records}
        == set(POSTSEASON_STAGES)
        and all(
            record["rows"] > 0
            and record["rows"] == record["score_rows"]
            and record["series_complete"] is True
            for record in series_records
        )
    )

    season_complete = postseason.get("isComplete")

    if season_complete is True and not all_rounds_observed:
        season_complete = None

    for row in rows:
        row["_season_complete"] = season_complete

    for record in series_records:
        record["season_complete"] = season_complete
        record["all_rounds_score_observed"] = all_rounds_observed

    return rows, series_records


def materialize_supplement(
    source: SourceVersion,
    directory: Path,
) -> tuple[Row, list[Row], list[Row]]:
    """불변 원본의 포스트시즌 영역을 별도 중간 자료로 생성한다."""
    season = required_integer(source.target["season"])

    require(
        source.revision == SUPPLEMENT_REVISION
        and season in SUPPLEMENT_SEASONS,
        "승인된 보완 원본 판본과 불일치",
    )

    raw = as_row(read_json(source.path))
    rows, series = flatten_postseason(raw, season)

    inspection_stage = source.target["records_path"][1]
    inspected = as_row(
        as_row(raw["postseason"])[inspection_stage]
    )["games"]

    require(
        len(inspected)
        == as_row(source.result["inspection"])["rows"],
        "기존 최소 파싱 행 수 불일치",
    )

    require(
        len(rows) == EXPECTED_ROWS[season],
        "고정 판본의 포스트시즌 전체 행 수 불일치",
    )

    output = directory / (
        f"{source.target_id}-{source.sha256}.jsonl"
    )

    field_types: dict[str, set[str]] = defaultdict(set)
    null_counts: Counter[str] = Counter()

    with output.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            for field, value in row.items():
                field_types[field].add(type(value).__name__)

                if value is None or value == "":
                    null_counts[field] += 1

            stream.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )

    require(
        list(read_interim_rows(output)) == rows,
        "포스트시즌 원본과 중간 자료 불일치",
    )

    evidence = {
        "target_id": source.target_id,
        "source_sha": source.sha256,
        "revision": source.revision,
        "raw_path": f"objects/{source.sha256}.raw",
        "interim_file": output.name,
        "interim_sha256": sha256_file(output),
        "rows": len(rows),
        "schema": {
            field: {
                "types": sorted(types),
                "null_or_empty_count": null_counts[field],
            }
            for field, types in field_types.items()
        },
        "target": source.target,
        "collection_result": source.result,
        "selected": True,
        "row_namespace": "postseason",
        "minimal_inspection_rows": len(inspected),
        "series_evidence": series,
    }

    return evidence, rows, series


def regular_completion_evidence(
    db: sqlite3.Connection,
    season: int,
) -> Row:
    """공식 종료 근거를 완료 경기 및 PBP 경기 집합과 대조한다."""
    official = REGULAR_COMPLETION[season]

    games = db.execute(
        """
        SELECT game_id,date,home_team_id,away_team_id,source_sha
        FROM games
        WHERE season=? AND competition='regular'
          AND result_valid=1 AND status='Final'
        ORDER BY game_id
        """,
        (season,),
    ).fetchall()

    pbp = db.execute(
        """
        SELECT c.game_id,c.final_score_match,c.source_sha
        FROM game_checks c
        JOIN games g USING(game_id)
        WHERE g.season=? AND g.competition='regular'
        ORDER BY c.game_id
        """,
        (season,),
    ).fetchall()

    game_ids = {row[0] for row in games}
    pbp_ids = {row[0] for row in pbp}

    require(
        len(games)
        == len(game_ids)
        == len(pbp)
        == official["games"]
        and game_ids == pbp_ids,
        f"정규시즌/PBP 완료 경기 집합 불일치: {season}",
    )

    require(
        max(row[1] for row in games) == official["last_date"],
        f"공식 정규시즌 종료일 불일치: {season}",
    )

    require(
        all(row[1] == 1 for row in pbp),
        f"PBP 최종 점수 대조 실패: {season}",
    )

    team_games: Counter[int] = Counter()

    for row in games:
        team_games[row[2]] += 1
        team_games[row[3]] += 1

    require(
        len(team_games) == 10
        and all(count == 144 for count in team_games.values()),
        f"정규시즌 팀별 완료 경기 수 불일치: {season}",
    )

    ready_date = (
        date.fromisoformat(official["last_date"])
        + timedelta(days=1)
    ).isoformat()

    require(
        ready_date == official["published_date"],
        "시즌 종료와 공지일의 정보 경계 불일치",
    )

    return {
        "season": season,
        **official,
        "competition": "regular",
        "completed_games": len(games),
        "pbp_games": len(pbp),
        "game_sets_match": True,
        "all_final_scores_match": True,
        "team_games": {
            str(key): value
            for key, value in sorted(team_games.items())
        },
        "game_ids_sha256": model_data.fingerprint(
            sorted(game_ids)
        ),
        "regular_source_sha256": sorted({
            row[4] for row in games
        }),
        "pbp_source_sha256": sorted({
            row[2] for row in pbp
        }),
        "label_ready_date": ready_date,
        "retrospective_label_evidence_only": True,
    }


def schedule_matches(
    db: sqlite3.Connection,
    row: Row,
) -> list[Row]:
    """기존 일정 참조의 날짜·팀·대회명 일치 여부를 기록한다."""
    away_id = team_id(row["away"])
    home_id = team_id(row["home"])

    result = []

    for item in db.execute(
        """
        SELECT *
        FROM schedule_reference
        WHERE date=?
        ORDER BY source_sha,source_row
        """,
        (normalized_date(row["date"]),),
    ):
        if (
            TEAM_CODES.get(item["away_code"]) != away_id
            or TEAM_CODES.get(item["home_code"]) != home_id
        ):
            continue

        result.append({
            "source_sha": item["source_sha"],
            "source_row": item["source_row"],
            "source_key": item["source_key"],
            "series_name": item["series_name"],
            "classification": item["classification"],
            "stage_label_matches": (
                POSTSEASON_NAMES.get(item["series_name"])
                == row["_stage"]
            ),
        })

    return result


def apply_supplement(
    store: StandardStore,
    source: SourceVersion,
    rows: list[Row],
    series: list[Row],
    *,
    verified: bool,
) -> Row:
    """기존 경기 변환기를 사용하고 보완 원본의 대응 행을 대조한다."""
    db = store.connection
    transformed = []

    for row in rows:
        day = normalized_date(row["date"])
        away_id = team_id(row["away"])
        home_id = team_id(row["home"])

        require(
            db.execute(
                """
                SELECT 1
                FROM games
                WHERE competition='regular'
                  AND result_valid=1
                  AND date=?
                  AND away_team_id=?
                  AND home_team_id=?
                """,
                (day, away_id, home_id),
            ).fetchone() is None,
            "정규시즌 완료 경기와 포스트시즌 경기 중복",
        )

        # 원본의 isFinal 부재는 중간 자료에 유지한다.
        # 기존 경기 변환기에만 검증된 파생 종료 상태를 전달한다.
        transformed.append({
            **row,
            "isFinal": row["_completion_basis"] != "not_final",
        })

    convert_postseason(
        store,
        source,
        iter(transformed),
    )

    stored = {
        row["source_row"]: row
        for row in db.execute(
            """
            SELECT *
            FROM games
            WHERE source_sha=? AND competition='postseason'
            """,
            (source.sha256,),
        )
    }

    require(
        set(stored) == set(range(len(rows))),
        "포스트시즌 원본 행 변환 또는 추적 누락",
    )

    observations = []

    for row in rows:
        index = row["_source_row"]
        game = stored[index]
        final = row["_completion_basis"] != "not_final"

        require(
            game["season"] == source.target["season"]
            and game["stage"] == row["_stage"]
            and game["date"] == normalized_date(row["date"])
            and game["game_number"] == row["game"]
            and game["away_team_id"] == team_id(row["away"])
            and game["home_team_id"] == team_id(row["home"])
            and game["result_valid"] == int(final),
            "표준 경기와 원본 정보 불일치",
        )

        if final:
            require(
                game["away_score"] == row["awayScore"]
                and game["home_score"] == row["homeScore"],
                "표준 경기와 원본 점수 불일치",
            )

        matches = schedule_matches(db, row)

        observations.append({
            "source_row": index,
            "game_id": game["game_id"],
            "stage": row["_stage"],
            "date": game["date"],
            "completion_basis": row["_completion_basis"],
            "valid_result": game["result_valid"],
            "schedule_matches": matches,
            "schedule_match_status": (
                "missing_reference"
                if not matches
                else "stage_label_match"
                if any(item["stage_label_matches"] for item in matches)
                else "date_teams_match_stage_label_differs"
            ),
        })

        if verified:
            db.execute(
                """
                INSERT INTO postseason_schedule
                VALUES (?,?,?,?,?,?,?,?,?)
                """,
                (
                    source.sha256,
                    index,
                    row["_stage"],
                    row["game"],
                    game["date"],
                    row["away"],
                    row["home"],
                    int(row.get("isFinal") is True),
                    game["game_id"],
                ),
            )

    if verified:
        for record in series:
            db.execute(
                """
                INSERT INTO postseason_series
                VALUES (?,?,?,?,?,?)
                """,
                (
                    source.sha256,
                    source.target["season"],
                    record["stage"],
                    record["series_complete"],
                    record["season_complete"],
                    json.dumps(
                        record["metadata"],
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ),
                ),
            )

    return {
        "target_id": source.target_id,
        "source_sha": source.sha256,
        "revision": source.revision,
        "competition": "postseason",
        "expanded_rows": len(rows),
        "mapped_rows": len(stored),
        "valid_results": sum(
            game["result_valid"]
            for game in stored.values()
        ),
        "series": series,
        "observations": observations,
    }


def unchanged_tables(
    previous: Row,
    current: Row,
    allowed: set[str],
) -> list[str]:
    """보완 범위 외 테이블의 변경을 차단한다."""
    require(
        previous.keys() == current.keys(),
        "테이블 목록 변경",
    )

    changed = sorted(
        name
        for name in current
        if current[name] != previous[name]
    )

    require(
        set(changed).issubset(allowed),
        f"보완 범위 외 테이블 변경: {changed}",
    )

    return changed


def build_standard_supplement(
    settings: Settings,
    collection_run: str,
    base_run: str,
    base_verification: str,
    *,
    compare_with: str | None = None,
) -> Row:
    """이전 검증판을 보존하며 신규 표준판과 최종 검증판을 생성한다."""
    require(
        bool(RUN_PATTERN.fullmatch(collection_run)),
        "수집 실행 ID 오류",
    )

    base_verified, provenance = model_data.verify_source(
        settings,
        base_run,
        base_verification,
    )

    base_standard_dir = (
        settings.processed_dir / "standard" / base_run
    )

    base_interim = (
        settings.interim_dir / "standard" / base_run
    )

    base_manifest = as_row(
        read_json(base_interim / "manifest.json")
    )

    base_quality = as_row(
        read_json(base_standard_dir / "quality.json")
    )

    base_report = as_row(
        read_json(
            base_standard_dir
            / f"verified-{base_verification}.json"
        )
    )

    intake = intake_sources(
        settings.raw_dir / "published",
        as_of_run=collection_run,
        expected_targets=EXPECTED_TARGETS | SUPPLEMENT_TARGETS,
    )

    old_versions = {
        (record["target_id"], record["source_sha"]): record
        for record in base_manifest["versions"]
    }

    current_old_versions = {
        (source.target_id, source.sha256)
        for source in intake.versions
        if source.target_id in EXPECTED_TARGETS
    }

    require(
        current_old_versions == old_versions.keys(),
        "기존 원본 판본이 변경돼 증분 적용할 수 없습니다.",
    )

    for source in intake.versions:
        if source.target_id not in EXPECTED_TARGETS:
            continue

        previous = old_versions[
            (source.target_id, source.sha256)
        ]

        require(
            previous["target"] == source.target
            and previous["collection_result"] == source.result
            and previous["selected"] == (
                intake.selected[source.target_id].sha256
                == source.sha256
            ),
            "기존 원본 선택 판본 또는 수집 근거 변경",
        )

    new_sources = [
        intake.selected[target]
        for target in sorted(SUPPLEMENT_TARGETS)
    ]

    require(
        len([
            source
            for source in intake.versions
            if source.target_id in SUPPLEMENT_TARGETS
        ]) == len(new_sources),
        "추가 원본의 예상 외 판본",
    )

    for source in new_sources:
        regular_id = f"dent-results-{source.target['season']}"

        require(
            source.sha256 == intake.selected[regular_id].sha256,
            "추가 포스트시즌과 기존 정규시즌 원본의 해시 불일치",
        )

    with closing(
        sqlite3.connect(
            base_verified.as_uri() + "?mode=ro",
            uri=True,
        )
    ) as db:
        regular_evidence = [
            regular_completion_evidence(db, season)
            for season in sorted(REGULAR_COMPLETION)
        ]

    inputs = {
        "version": VERSION,
        "base_run": base_run,
        "base_verification": base_verification,
        "base_verification_fingerprint": provenance[
            "verification_fingerprint"
        ],
        "base_database_sha256": provenance["database_sha256"],
        "base_parent_database_sha256": provenance[
            "parent_db_sha256"
        ],
        "collection_run": collection_run,
        "collection_history_sha256": model_data.fingerprint(
            intake.history
        ),
        "code_and_settings_sha256": {
            name: sha256_file(settings.project_root / name)
            for name in (
                "src/datasets/postseason_supplement.py",
                "src/datasets/published_sources.py",
                "src/datasets/standardize.py",
                "src/datasets/model_data.py",
                "src/validation/standard_data.py",
                "src/validation/standard_quality.py",
                "kbo.toml",
            )
        },
        "selected_sources": {
            target: {
                "sha256": source.sha256,
                "revision": source.revision,
            }
            for target, source in sorted(intake.selected.items())
        },
        "regular_completion": regular_evidence,
    }

    fingerprint = model_data.fingerprint(inputs)
    run_id = uuid4().hex
    verification_id = uuid4().hex

    interim = settings.interim_dir / "standard" / run_id
    directory = settings.processed_dir / "standard" / run_id

    interim.mkdir(parents=True, exist_ok=False)
    directory.mkdir(parents=True, exist_ok=False)

    records = []

    # 이전에 실제 원본과 대조한 중간 자료를 신규 실행에 복제한다.
    for record in base_manifest["versions"]:
        original = base_interim / record["interim_file"]
        destination = interim / record["interim_file"]

        require(
            sha256_file(original) == record["interim_sha256"],
            "기존 중간 자료 해시 불일치",
        )

        shutil.copyfile(original, destination)

        require(
            sha256_file(destination) == record["interim_sha256"],
            "증분 중간 자료 복제 해시 불일치",
        )

        records.append(record)

    supplements = []

    for source in new_sources:
        evidence, rows, series = materialize_supplement(
            source,
            interim,
        )

        records.append(evidence)
        supplements.append((source, rows, series))

    manifest = {
        "run_id": run_id,
        "as_of_run": collection_run,
        "input_fingerprint": fingerprint,
        "inputs": inputs,
        "object_count": intake.object_count,
        "raw_bytes": intake.raw_bytes,
        "logical_target_count": len(intake.selected),
        "versions": records,
        "collection_history": intake.history,
    }

    manifest_path = interim / "manifest.json"
    write_json(manifest_path, manifest)

    parent_path = directory / "standard.sqlite3"
    verified_path = (
        directory / f"verified-{verification_id}.sqlite3"
    )

    parent_source = base_standard_dir / "standard.sqlite3"

    outputs = []

    for original, destination, verified in (
        (parent_source, parent_path, False),
        (base_verified, verified_path, True),
    ):
        store = StandardStore(destination)

        try:
            with closing(
                sqlite3.connect(
                    original.as_uri() + "?mode=ro",
                    uri=True,
                )
            ) as previous:
                previous.backup(store.connection)

            db = store.connection
            db.execute("PRAGMA foreign_keys=ON")

            evidence = [
                regular_completion_evidence(db, season)
                for season in sorted(REGULAR_COMPLETION)
            ]

            require(
                evidence == regular_evidence,
                "부모 표준판과 최종 검증판의 정규시즌 근거 불일치",
            )

            for item in evidence:
                db.execute(
                    """
                    UPDATE games
                    SET season_complete=1
                    WHERE competition='regular' AND season=?
                    """,
                    (item["season"],),
                )

            source_evidence = [
                apply_supplement(
                    store,
                    source,
                    rows,
                    series,
                    verified=verified,
                )
                for source, rows, series in supplements
            ]

            require(
                db.execute(
                    "PRAGMA integrity_check"
                ).fetchone()[0] == "ok"
                and db.execute(
                    "PRAGMA foreign_key_check"
                ).fetchone() is None,
                "보완 표준 DB 무결성 검사 실패",
            )

            if verified:
                hashes = _digest_tables(
                    db,
                    TABLE_NAMES + AUDIT_TABLES,
                )

                changed = unchanged_tables(
                    as_row(base_report["tables"]),
                    hashes,
                    {
                        "games",
                        "postseason_series",
                        "postseason_schedule",
                    },
                )

                handoff = _handoff(db)

                handoff["regular_completion"] = evidence
                handoff["postseason_supplement"] = source_evidence

                contract = handoff["model_contract"]

                contract["M1"]["period"] = (
                    "2001-2026 regular; "
                    "2010-2020,2024-2026 postseason observed; "
                    "2024 Korean Series scores only"
                )

                contract["postseason"]["limit"] = (
                    "2021-2023 및 2024 앞 세 라운드 점수 미확보; "
                    "일정 참조와 실제 점수 구분; "
                    "공식 순위·대진·시리즈 규칙 별도 검증"
                )

                outputs.append({
                    "hashes": hashes,
                    "handoff": handoff,
                    "changed_tables": changed,
                    "supplement": source_evidence,
                })

            else:
                summary = store.summary()

                previous_hashes = {
                    name: {
                        "rows": info["rows"],
                        "sha256": info["sha256"],
                    }
                    for name, info in base_quality["tables"].items()
                }

                hashes = _digest_tables(db, TABLE_NAMES)

                changed = unchanged_tables(
                    previous_hashes,
                    hashes,
                    {"games"},
                )

                coverage = coverage_report(store)
                coverage["regular_completion"] = evidence
                coverage["postseason_supplement"] = source_evidence

                quality = {
                    "status": "completed",
                    "run_id": run_id,
                    "as_of_run": collection_run,
                    "input_fingerprint": fingerprint,
                    "transform_version": f"supplement-{VERSION}",
                    "source_objects": intake.object_count,
                    "source_bytes": intake.raw_bytes,
                    "logical_targets": len(intake.selected),
                    "source_versions": len(records),
                    "interim_manifest_sha256": sha256_file(
                        manifest_path
                    ),
                    "coverage": coverage,
                    "changed_tables_from_base": changed,
                    **summary,
                }

                quality["comparison"] = compare_summaries(
                    base_quality,
                    quality,
                )

                outputs.append({
                    "hashes": hashes,
                    "quality": quality,
                    "supplement": source_evidence,
                })

            db.commit()

        except BaseException:
            store.connection.rollback()
            raise

        finally:
            store.close()

    parent, verified = outputs

    require(
        parent["hashes"]["games"]
        == verified["hashes"]["games"]
        and parent["supplement"] == verified["supplement"],
        "표준판과 최종 검증판의 보완 결과 불일치",
    )

    quality = parent["quality"]

    report_inputs = {
        "parent_input_fingerprint": fingerprint,
        "parent_tables": parent["hashes"],
        "quality_version": f"supplement-{VERSION}",
        "quality_code_sha256": sha256_file(Path(__file__)),
    }

    report_fingerprint = model_data.fingerprint(report_inputs)

    row_accounting = list(base_report["row_accounting"])

    for item in verified["supplement"]:
        row_accounting.append({
            "target_id": item["target_id"],
            "source_sha": item["source_sha"],
            "competition": "postseason",
            "physical_rows": item["expanded_rows"],
            "mapped_rows": item["mapped_rows"],
            "excluded_only": 0,
            "mapped_with_restriction": (
                item["mapped_rows"] - item["valid_results"]
            ),
        })

    report_path = (
        directory / f"verified-{verification_id}.json"
    )

    report = {
        "status": "completed",
        "run_id": run_id,
        "verification_id": verification_id,
        "verification_key": f"{run_id}:{verification_id}",
        "input_fingerprint": report_fingerprint,
        "inputs": report_inputs,
        "source_objects": intake.object_count,
        "logical_targets": len(intake.selected),
        "parent_run": run_id,
        "parent_coverage": quality["coverage"],
        "source_specs": {
            path.name: sha256_file(path)
            for path in sorted(
                (settings.raw_dir / "sources").glob("*.json")
            )
        },
        "row_accounting": row_accounting,
        "tables": verified["hashes"],
        "handoff": verified["handoff"],
        "comparison": {
            "same_input": False,
            "changed_tables": verified["changed_tables"],
            "reproducible": None,
        },
        "database_path": str(verified_path),
        "report_path": str(report_path),
    }

    if compare_with:
        parts = compare_with.split(":")

        require(
            len(parts) == 2
            and all(RUN_PATTERN.fullmatch(part) for part in parts),
            "동일 입력 비교 실행 키 오류",
        )

        old_directory = (
            settings.processed_dir / "standard" / parts[0]
        )

        previous_quality = as_row(
            read_json(old_directory / "quality.json")
        )

        previous_report = as_row(
            read_json(
                old_directory / f"verified-{parts[1]}.json"
            )
        )

        parent_comparison = compare_summaries(
            previous_quality,
            quality,
        )

        changed = [
            name
            for name in report["tables"]
            if report["tables"][name]
            != previous_report["tables"].get(name)
        ]

        require(
            parent_comparison["reproducible"] is True
            and previous_report["input_fingerprint"]
            == report["input_fingerprint"]
            and not changed,
            "보완 표준 자료 동일 입력 재생성 실패",
        )

        quality["comparison"] = parent_comparison
        report["comparison"] = {
            "same_input": True,
            "changed_tables": changed,
            "reproducible": True,
        }

    require(
        sha256_file(base_verified)
        == provenance["database_sha256"]
        and sha256_file(parent_source)
        == provenance["parent_db_sha256"],
        "기존 표준 산출물 변경 감지",
    )

    write_json(directory / "quality.json", quality)
    write_json(report_path, report)

    LOGGER.info(
        "보완 표준 검증판 완료: %s:%s",
        run_id,
        verification_id,
    )

    return {
        "run_id": run_id,
        "verification_id": verification_id,
        "verification_key": report["verification_key"],
        "database_path": str(verified_path),
        "report_path": str(report_path),
        "object_count": intake.object_count,
        "logical_targets": len(intake.selected),
        "source_versions": len(records),
        "regular_completion": regular_evidence,
        "postseason": verified["supplement"],
        "comparison": report["comparison"],
    }


def model_diagnostics(path: Path) -> Row:
    """Train/Validation의 기간별 정답 범위와 관측 위험을 집계한다."""
    result = {}

    with closing(
        sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    ) as db:
        db.row_factory = sqlite3.Row

        for role in ("hitter", "pitcher"):
            model = f"m3_{role}"

            denominator = (
                "target_pa"
                if role == "hitter"
                else "target_bf"
            )

            result[role] = {
                "targets": [
                    dict(row)
                    for row in db.execute(
                        f"""
                        SELECT i.season,i.split,t.horizon,
                               t.status,t.usable,t.ratio_status,
                               COUNT(*) samples,
                               COUNT(t.target_games) valid_counts,
                               COUNT(t.target_hit_rate) valid_hit_rates,
                               COUNT(t.target_strikeout_rate)
                                   valid_strikeout_rates,
                               SUM(COALESCE(t.target_ab,0))
                                   valid_at_bats,
                               SUM(COALESCE(t.{denominator},0))
                                   valid_opportunities,
                               SUM(t.missing_games>0)
                                   missing_game_windows,
                               SUM(t.team_risky_games>0)
                                   team_risky_windows,
                               SUM(t.player_rejected>0)
                                   player_rejected_windows,
                               SUM(t.league_rejected_rows>0)
                                   league_rejected_windows
                        FROM {model}_targets t
                        JOIN {model}_index i USING(sample_id)
                        WHERE i.split IN ('train','validation')
                        GROUP BY 1,2,3,4,5,6
                        ORDER BY 1,2,3,4,5,6
                        """
                    )
                ],
                "team_role_quality": [
                    dict(row)
                    for row in db.execute(
                        """
                        SELECT season,team_id,role,
                               COUNT(*) games,
                               SUM(risky) risky,
                               SUM(missing_game) missing,
                               SUM(state_uncertain) uncertain,
                               SUM(team_mismatches>0)
                                   aggregate_conflicts,
                               SUM(identity_conflicts>0)
                                   identity_conflicts,
                               SUM(rejected_rows>0)
                                   rejected_game_rows
                        FROM player_game_role_quality
                        WHERE season<2026 AND role=?
                        GROUP BY 1,2,3
                        ORDER BY 1,2,3
                        """,
                        (role,),
                    )
                ],
            }

    return result


def compare_model_runs(
    settings: Settings,
    previous_run: str,
    current_run: str,
) -> Row:
    """이전 및 최신 모델 자료의 논리 해시와 기간별 정답을 비교한다."""
    require(
        bool(RUN_PATTERN.fullmatch(previous_run))
        and bool(RUN_PATTERN.fullmatch(current_run)),
        "모델 자료 실행 ID 오류",
    )

    root = settings.processed_dir / "model_data"
    previous_dir = root / previous_run
    current_dir = root / current_run

    previous = as_row(
        read_json(previous_dir / "manifest.json")
    )

    current = as_row(
        read_json(current_dir / "manifest.json")
    )

    previous_db = previous_dir / "model_data.sqlite3"
    current_db = current_dir / "model_data.sqlite3"

    require(
        previous["status"] == current["status"] == "completed"
        and sha256_file(previous_db)
        == previous["database_sha256"]
        and sha256_file(current_db)
        == current["database_sha256"],
        "모델 자료 변경 또는 미완료",
    )

    allowed_changes = {
        "m1_index",
        "m1_inputs",
        "m1_targets",
        "m3_hitter_targets",
        "m3_pitcher_targets",
        "m3_window_games",
        "m3_window_teams",
        "season_coverage",
        "season_series_evidence",
        "season_schedule_evidence",
    }

    changes = unchanged_tables(
        as_row(previous["tables"]),
        as_row(current["tables"]),
        allowed_changes,
    )

    result = {
        "previous_run": previous_run,
        "current_run": current_run,
        "changed_tables": changes,
        "table_comparison": {
            name: {
                "previous_rows": previous["tables"][name]["rows"],
                "current_rows": current["tables"][name]["rows"],
                "row_delta": (
                    current["tables"][name]["rows"]
                    - previous["tables"][name]["rows"]
                ),
                "content_changed": name in changes,
            }
            for name in sorted(current["tables"])
        },
        "previous_diagnostics": model_diagnostics(previous_db),
        "current_diagnostics": model_diagnostics(current_db),
        "final_test_usage": "integrity_and_split_checks_only",
    }

    write_json(
        current_dir / "supplement-impact.json",
        result,
    )

    return result


def main(argv: Sequence[str] | None = None) -> int:
    """포스트시즌 보완부터 모델 자료 생성 및 재현 검사까지 실행한다."""
    parser = argparse.ArgumentParser(
        description="포스트시즌 및 정규시즌 완료 근거 보완"
    )

    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--collection-run", required=True)
    parser.add_argument("--base-run", required=True)
    parser.add_argument("--base-verification", required=True)
    parser.add_argument("--base-model-run", required=True)
    parser.add_argument("--reproduce", action="store_true")

    args = parser.parse_args(argv)

    try:
        settings = load_settings(args.project_root)

        logging.basicConfig(
            level=settings.log_level,
            format="%(levelname)s %(name)s: %(message)s",
        )

        standard = build_standard_supplement(
            settings,
            args.collection_run,
            args.base_run,
            args.base_verification,
        )

        model = model_data.build_model_data(
            settings,
            standard["run_id"],
            standard["verification_id"],
            VALIDATION_START,
            TEST_START,
            OBSERVATION_END,
        )

        result = {
            "standard": standard,
            "model": model,
            "impact": compare_model_runs(
                settings,
                args.base_model_run,
                model["run_id"],
            ),
        }

        if args.reproduce:
            result["standard_reproduction"] = (
                build_standard_supplement(
                    settings,
                    args.collection_run,
                    args.base_run,
                    args.base_verification,
                    compare_with=standard["verification_key"],
                )
            )

            result["model_reproduction"] = (
                model_data.build_model_data(
                    settings,
                    standard["run_id"],
                    standard["verification_id"],
                    VALIDATION_START,
                    TEST_START,
                    OBSERVATION_END,
                    compare_run=model["run_id"],
                )
            )

        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        )

    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        sqlite3.DatabaseError,
    ) as exc:
        LOGGER.error("자료 보완 및 검증 실패: %s", exc)
        return 2

    except KeyboardInterrupt:
        LOGGER.warning("자료 보완 중단")
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())