"""인수 자료·일정·저장 모델의 무결성과 기준 시점 입력을 검사한다."""

from collections.abc import Mapping
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
import sqlite3

from ...datasets.published_sources import (
    RUN_PATTERN,
    as_row,
    read_json,
    sha256_file,
)
from ...models.game_outcome import model as m1
from ...models.game_outcome.__main__ import load_model, source
from ...settings import Settings
from ...validation.standard_data import GAME_KEY_PATTERN, TEAM_CODES

from . import SNAPSHOT_SCHEMA
from .rules import (
    PREVIOUS_RANK_SOURCE,
    RULE_SOURCE,
    RULE_VERSION,
    Fixture,
    PlayedGame,
    SeasonRules,
    fixture_order,
    get_rules,
    validate_schedule,
)


KST = timezone(timedelta(hours=9), name="KST")

SUPPORTED_FIRST_DAY = date(2025, 3, 22)
SUPPORTED_LAST_DAY = date(2025, 12, 31)

SCHEDULE_STATUSES = (
    "scheduled",
    "postponed",
    "cancelled",
)

FIXTURE_FIELDS = {
    "game_id",
    "date",
    "home_team_id",
    "away_team_id",
    "game_number",
    "status",
}


@dataclass
class Scenario:
    """검증된 실제 이력과 미래 경기 식별 정보를 분리한 실행 입력."""

    rules: SeasonRules
    cutoff: date
    completed: tuple[PlayedGame, ...]
    remaining: tuple[Fixture, ...]
    model: m1.Model
    standard_db: Path
    model_db: Path
    metadata: dict[str, object]


def parse_day(value: object) -> date:
    """날짜 형식과 실제 달력의 유효성을 검사한다."""

    if not isinstance(value, str):
        raise ValueError("날짜는 YYYY-MM-DD 문자열이어야 합니다.")

    day = date.fromisoformat(value)

    if day.isoformat() != value:
        raise ValueError(f"정규화되지 않은 날짜: {value}")

    return day


def parse_cutoff(value: str, season: int) -> date:
    """공개 완료 시각이 없는 자료에서는 KST 날짜 경계를 사용한다."""

    day = parse_day(value)

    if day.year != season:
        raise ValueError("기준 날짜와 지원 시즌이 다릅니다.")

    if not SUPPORTED_FIRST_DAY <= day <= SUPPORTED_LAST_DAY:
        raise ValueError("지원하는 2025년 기준 날짜 범위를 벗어났습니다.")

    return day


def cutoff_timestamp(day: date) -> str:
    """KST 자정 기준 시각을 반환한다."""

    return datetime.combine(
        day,
        time.min,
        tzinfo=KST,
    ).isoformat()


def open_readonly(path: Path) -> sqlite3.Connection:
    """자료 원본을 변경할 수 없는 SQLite 연결을 반환한다."""

    if not path.is_file():
        raise FileNotFoundError(path)

    connection = sqlite3.connect(
        path.resolve().as_uri() + "?mode=ro",
        uri=True,
    )

    try:
        connection.execute("PRAGMA query_only = ON")
    except sqlite3.DatabaseError:
        connection.close()
        raise

    return connection


def fixture_from_values(
    game_id: object,
    day_value: object,
    number: object,
    home: object,
    away: object,
) -> Fixture:
    """표준 정규시즌 경기 키와 날짜·차수·팀 ID를 교차 검사한다."""

    if (
        not isinstance(game_id, str)
        or not GAME_KEY_PATTERN.fullmatch(game_id)
    ):
        raise ValueError(f"정규시즌 경기 ID 형식 오류: {game_id}")

    day = parse_day(day_value)

    if game_id[:8] != day.strftime("%Y%m%d"):
        raise ValueError(f"경기 ID와 날짜 불일치: {game_id}")

    if type(home) is not int or type(away) is not int:
        raise ValueError(f"구단 ID 타입 오류: {game_id}")

    if (
        TEAM_CODES.get(game_id[8:10]) != away
        or TEAM_CODES.get(game_id[10:12]) != home
    ):
        raise ValueError(f"경기 ID와 홈·원정 구단 불일치: {game_id}")

    if type(number) is not int or number != int(game_id[-1]):
        raise ValueError(f"경기 ID와 차수 불일치: {game_id}")

    return Fixture(
        game_id=game_id,
        day=day,
        home=home,
        away=away,
        number=number,
    )


def fixture_to_dict(game: Fixture) -> dict[str, object]:
    """반복 실행과 후속 검산에서 사용할 일정 표현."""

    return {
        "game_id": game.game_id,
        "date": game.day.isoformat(),
        "home_team_id": game.home,
        "away_team_id": game.away,
        "game_number": game.number,
    }


def read_completed(
    standard: sqlite3.Connection,
    model_data: sqlite3.Connection,
    season: int,
    cutoff: date,
) -> tuple[PlayedGame, ...]:
    """기준 날짜 이전의 완료 경기만 조회하고 두 자료판을 대조한다.

    모델 데이터의 누적 변수는 사용하지 않는다.
    기준 이후 경기의 정답도 이 함수에서 조회하지 않는다.
    """

    standard_rows = standard.execute(
        """
        SELECT game_id, source_key, date, game_number,
               home_team_id, away_team_id,
               home_score, away_score, winner
        FROM eligible_games
        WHERE season = ?
          AND competition = 'regular'
          AND date < ?
        ORDER BY date, game_number, game_id
        """,
        (season, cutoff.isoformat()),
    )

    results: dict[str, PlayedGame] = {}

    for (
        game_id,
        source_key,
        day,
        number,
        home,
        away,
        home_runs,
        away_runs,
        winner,
    ) in standard_rows:

        fixture = fixture_from_values(
            game_id, day, number, home, away
        )

        if source_key != game_id:
            raise ValueError(f"표준 경기 키 불일치: {game_id}")

        if game_id in results:
            raise ValueError(f"완료 경기 중복: {game_id}")

        result = PlayedGame(
            fixture,
            home_runs,
            away_runs,
        )

        if winner != m1.score_outcome(home_runs, away_runs):
            raise ValueError(f"득점과 경기 결과 불일치: {game_id}")

        results[game_id] = result

    model_rows = model_data.execute(
        """
        SELECT x.game_id, x.sample_id, x.sample_date,
               i.home_team_id, i.away_team_id,
               t.target_home_runs, t.target_away_runs,
               t.target_result
        FROM m1_index x
        JOIN m1_inputs i ON x.sample_id = i.sample_id
        JOIN m1_targets t ON x.sample_id = t.sample_id
        WHERE i.season = ?
          AND i.competition = 'regular'
          AND x.status = 'eligible'
          AND x.sample_date < ?
        ORDER BY x.sample_date, x.sample_id
        """,
        (season, cutoff.isoformat()),
    )

    seen: set[str] = set()

    for (
        game_id,
        sample_id,
        day,
        home,
        away,
        home_runs,
        away_runs,
        outcome,
    ) in model_rows:

        if game_id in seen or game_id != sample_id:
            raise ValueError(f"M1 경기 키 오류: {game_id}")

        seen.add(game_id)

        if game_id not in results:
            raise ValueError(f"표준 자료에 없는 M1 경기: {game_id}")

        result = results[game_id]
        fixture = result.fixture

        if (
            day != fixture.day.isoformat()
            or home != fixture.home
            or away != fixture.away
            or home_runs != result.home_runs
            or away_runs != result.away_runs
            or outcome != m1.score_outcome(home_runs, away_runs)
        ):
            raise ValueError(f"M1과 표준 경기 자료 불일치: {game_id}")

    if seen != set(results):
        raise ValueError("표준 자료와 M1 완료 경기 집합 불일치")

    return tuple(
        sorted(
            results.values(),
            key=lambda game: fixture_order(game.fixture),
        )
    )


def read_archive_schedule(
    standard: sqlite3.Connection,
    season: int,
    cutoff: date,
) -> tuple[Fixture, ...]:
    """사후 확정된 과거 일정에서 미래 경기 식별 정보만 가져온다.

    이 일정은 당시 공개 일정으로 간주하지 않는다.
    미래 실제 득점과 승무패 컬럼은 조회하지 않는다.
    """

    rows = standard.execute(
        """
        SELECT game_id, source_key, date, game_number,
               home_team_id, away_team_id
        FROM eligible_games
        WHERE season = ?
          AND competition = 'regular'
          AND date >= ?
        ORDER BY date, game_number, game_id
        """,
        (season, cutoff.isoformat()),
    )

    fixtures = []

    for game_id, source_key, day, number, home, away in rows:
        if source_key != game_id:
            raise ValueError(f"잔여 일정 경기 키 불일치: {game_id}")

        fixtures.append(
            fixture_from_values(
                game_id, day, number, home, away
            )
        )

    return tuple(fixtures)


def archive_status_counts(
    standard: sqlite3.Connection,
    season: int,
) -> list[dict[str, object]]:
    """사후 판본의 연기·취소 등 상태별 건수를 기록한다."""

    rows = standard.execute(
        """
        SELECT status, result_valid, COUNT(*)
        FROM games
        WHERE season = ?
          AND competition = 'regular'
        GROUP BY status, result_valid
        ORDER BY status, result_valid
        """,
        (season,),
    )

    return [
        {
            "status": status,
            "result_valid": bool(valid),
            "count": count,
        }
        for status, valid, count in rows
    ]


def load_schedule_snapshot(
    path: Path,
    season: int,
    cutoff: date,
) -> tuple[tuple[Fixture, ...], dict[str, object]]:
    """자동 확보한 외부 일정 스냅샷을 검사한다.

    외부 파일의 공개 시각 주장은 검증할 수 있어야 하지만,
    이 로더 자체가 공개 출처의 진위를 독립적으로 인증하지는 않는다.
    """

    payload = as_row(read_json(path))

    if payload.get("schema") != SNAPSHOT_SCHEMA:
        raise ValueError("일정 스냅샷 스키마 불일치")

    if payload.get("season") != season:
        raise ValueError("일정 스냅샷 시즌 불일치")

    published_value = payload.get("published_at")
    source_url = payload.get("source_url")
    version = payload.get("version")

    if not isinstance(published_value, str):
        raise ValueError("일정 공개 시각이 없습니다.")

    published = datetime.fromisoformat(published_value)

    if published.utcoffset() is None:
        raise ValueError("공개 시각에 시간대가 필요합니다.")

    boundary = datetime.combine(
        cutoff,
        time.min,
        tzinfo=KST,
    )

    if published > boundary:
        raise ValueError("기준 시점 이후 공개 일정은 사용할 수 없습니다.")

    if not isinstance(source_url, str) or not source_url.startswith("https://"):
        raise ValueError("일정 출처 HTTPS URL이 필요합니다.")

    if not isinstance(version, str) or not version.strip():
        raise ValueError("일정 판본이 없습니다.")

    rows = payload.get("fixtures")

    if not isinstance(rows, list):
        raise ValueError("일정 행 목록이 필요합니다.")

    entries: dict[
        str,
        tuple[Fixture, str, str | None],
    ] = {}

    for raw in rows:
        row = as_row(raw)

        if (
            not FIXTURE_FIELDS.issubset(row)
            or not set(row).issubset(
                FIXTURE_FIELDS | {"rescheduled_from"}
            )
        ):
            raise ValueError("일정에 누락 또는 허용되지 않는 필드가 있습니다.")

        fixture = fixture_from_values(
            row["game_id"],
            row["date"],
            row["game_number"],
            row["home_team_id"],
            row["away_team_id"],
        )

        if fixture.day.year != season:
            raise ValueError("일정 경기의 시즌 불일치")

        status = row["status"]

        if not isinstance(status, str) or status not in SCHEDULE_STATUSES:
            raise ValueError(f"지원하지 않는 일정 상태: {status}")

        if fixture.game_id in entries:
            raise ValueError(f"일정 스냅샷 중복: {fixture.game_id}")

        if status == "scheduled" and fixture.day < cutoff:
            raise ValueError("잔여 예정 경기의 날짜가 기준 시점 이전입니다.")

        previous = row.get("rescheduled_from")

        if previous is not None and (
            not isinstance(previous, str)
            or status != "scheduled"
        ):
            raise ValueError("재편성 연결 형식 오류")

        entries[fixture.game_id] = (
            fixture,
            status,
            previous,
        )

    fixtures = []
    replaced: set[str] = set()
    evidence = []

    for fixture, status, previous in entries.values():
        if previous is not None:
            if previous not in entries or previous in replaced:
                raise ValueError("재편성 대상 누락 또는 중복")

            old, old_status, _ = entries[previous]

            if (
                old_status not in ("postponed", "cancelled")
                or fixture.day < old.day
                or {fixture.home, fixture.away} != {old.home, old.away}
            ):
                raise ValueError("재편성 대상 경기 불일치")

            replaced.add(previous)

        if status == "scheduled":
            fixtures.append(fixture)

        evidence.append({
            **fixture_to_dict(fixture),
            "status": status,
            "rescheduled_from": previous,
        })

    metadata = {
        "mode": "provided_snapshot_unverified",
        "source_url": source_url,
        "published_at": published.isoformat(),
        "version": version,
        "snapshot_path": str(path.resolve()),
        "snapshot_sha256": sha256_file(path),
        "publication_independently_verified": False,
        "entries": evidence,
    }

    return (
        tuple(sorted(fixtures, key=fixture_order)),
        metadata,
    )


def verify_assets(
    settings: Settings,
    run_id: str,
    asset: Path,
) -> tuple[Path, Path, m1.Model, dict[str, object]]:
    """인수 DB와 모델 자산의 계보 및 실제 파일 해시를 검증한다."""

    model_db, data_provenance = source(settings, run_id)

    manifest_path = model_db.parent / "manifest.json"
    manifest = as_row(read_json(manifest_path))
    standard_source = as_row(manifest["source"])

    standard_run = standard_source["run_id"]
    verification = standard_source["verification_id"]

    if (
        not isinstance(standard_run, str)
        or not RUN_PATTERN.fullmatch(standard_run)
        or not isinstance(verification, str)
        or not RUN_PATTERN.fullmatch(verification)
    ):
        raise ValueError("표준 자료 식별자 오류")

    standard_db = (
        settings.processed_dir
        / "standard"
        / standard_run
        / f"verified-{verification}.sqlite3"
    )

    model_db_hash = sha256_file(model_db)
    standard_db_hash = sha256_file(standard_db)

    if model_db_hash != manifest["database_sha256"]:
        raise ValueError("모델 데이터 DB 해시 불일치")

    if standard_db_hash != standard_source["database_sha256"]:
        raise ValueError("표준 검증 DB 해시 불일치")

    asset = asset.resolve()

    envelope = as_row(read_json(asset))
    payload = as_row(envelope["payload"])
    model_source = as_row(payload["data"])

    if (
        model_source["model_data_run_id"] != run_id
        or model_source["database_sha256_from_manifest"] != model_db_hash
        or model_source["contract_sha256"] != manifest["contract_sha256"]
    ):
        raise ValueError("M1 학습 자료 계보 불일치")

    splits = as_row(payload["splits"])

    if set(splits) != {"train", "validation"}:
        raise ValueError("지원하는 M1 개발 자료 분할이 아닙니다.")

    validation = as_row(splits["validation"])

    if parse_day(validation["last_date"]) >= date(2026, 1, 1):
        raise ValueError("최종 Test 자료가 모델 검증 범위에 포함됩니다.")

    model = load_model(asset)

    provenance = {
        "model_data": data_provenance,
        "model_data_sha256": model_db_hash,
        "standard_run_id": standard_run,
        "standard_verification_id": verification,
        "standard_database": str(standard_db),
        "standard_database_sha256": standard_db_hash,
        "model_run_id": payload["training_run_id"],
        "model_schema": payload["schema"],
        "model_asset": str(asset),
        "model_asset_sha256": sha256_file(asset),
        "model_created_utc": payload["created_utc"],
        "model_splits": splits,
    }

    return standard_db, model_db, model, provenance


def load_scenario(
    settings: Settings,
    run_id: str,
    asset: Path,
    season: int,
    cutoff: date,
    schedule_snapshot: Path | None = None,
) -> Scenario:
    """시즌 입력과 일정 가정을 구성하되 실제 미래 정답은 배제한다."""

    rules = get_rules(season)
    parse_cutoff(cutoff.isoformat(), season)

    standard_db, model_db, model, provenance = verify_assets(
        settings,
        run_id,
        asset,
    )

    with (
        closing(open_readonly(standard_db)) as standard,
        closing(open_readonly(model_db)) as model_data,
    ):
        completed = read_completed(
            standard,
            model_data,
            season,
            cutoff,
        )

        if schedule_snapshot is None:
            remaining = read_archive_schedule(
                standard,
                season,
                cutoff,
            )

            schedule_metadata: dict[str, object] = {
                "mode": "retrospective_schedule_assumption",
                "source_database": str(standard_db),
                "version": provenance["standard_verification_id"],
                "publication_independently_verified": False,
                "archived_status_counts": archive_status_counts(
                    standard,
                    season,
                ),
                "reschedule_mapping": "not_verified",
                "assumptions": [
                    "실제 완료 경기의 식별 정보로 잔여 일정을 가정",
                    "사후 판본의 연기·취소 기록은 별도 경기로 합산하지 않음",
                    "연기 기록과 재편성 경기의 개별 연결은 미확인",
                ],
            }

        else:
            remaining, schedule_metadata = load_schedule_snapshot(
                schedule_snapshot,
                season,
                cutoff,
            )

    fixtures = [
        *(game.fixture for game in completed),
        *remaining,
    ]

    missing = validate_schedule(rules, fixtures)

    missing_total = sum(
        row["missing_games"]
        for row in missing
    )

    if len(fixtures) + missing_total != rules.total_games:
        raise ValueError("시즌 경기 수 검산 실패")

    metadata = {
        "season": season,
        "as_of_kst": cutoff_timestamp(cutoff),
        "timezone": "Asia/Seoul",
        "rules": {
            "version": RULE_VERSION,
            "source": RULE_SOURCE,
            "previous_rank_source": PREVIOUS_RANK_SOURCE,
            "previous_rank_publication": "2025-03-07",
            **asdict(rules),
        },
        "provenance": provenance,
        "schedule": schedule_metadata,
        "completed_games": len(completed),
        "remaining_games": len(remaining),
        "missing_games": missing_total,
        "missing_pairs": missing,
        "expected_games": rules.total_games,
        "schedule_complete_under_assumption": missing_total == 0,
        "historical_result_availability": (
            "이전 KST 날짜의 공식 완료 결과가 "
            "다음 날짜 자정까지 이용 가능했다고 가정"
        ),
        "formal_backtest_supported": False,
        "formal_backtest_reasons": [
            "기준 당시 일정·정답 공개 이력 미입증",
            "운영 M1의 생성 및 선정 시점이 기준 시점 이전임을 입증하지 못함",
        ],
        "final_test_used": False,
        "m1_limitations": [
            "2025년 평균 예측 무승부율 0.0207, 관측 무승부율 0.0306",
            "더블헤더 1차전의 별도 무승부 보정 없음",
            "선수·라인업·선발 변경은 기존 M1 입력 범위 밖",
        ],
    }

    return Scenario(
        rules=rules,
        cutoff=cutoff,
        completed=completed,
        remaining=remaining,
        model=model,
        standard_db=standard_db,
        model_db=model_db,
        metadata=metadata,
    )


def read_diagnostic_truth(
    scenario: Scenario,
) -> tuple[PlayedGame, ...]:
    """개발 시나리오 계산 종료 후 별도로 실제 완료 시즌을 읽는다.

    미래 정답은 이 함수의 반환 이후 진단에만 사용한다.
    예측 입력 또는 모델 선정에 연결하지 않는다.
    """

    season = scenario.rules.season

    with (
        closing(open_readonly(scenario.standard_db)) as standard,
        closing(open_readonly(scenario.model_db)) as model_data,
    ):
        completion = standard.execute(
            """
            SELECT COUNT(*), SUM(season_complete = 1)
            FROM eligible_games
            WHERE season = ?
              AND competition = 'regular'
            """,
            (season,),
        ).fetchone()

        expected = scenario.rules.total_games

        if completion != (expected, expected):
            raise ValueError("실제 완료 시즌 정답 근거가 부족합니다.")

        actual = read_completed(
            standard,
            model_data,
            season,
            date(season + 1, 1, 1),
        )

    missing = validate_schedule(
        scenario.rules,
        [game.fixture for game in actual],
    )

    if missing:
        raise ValueError("실제 완료 시즌 정답에 누락이 있습니다.")

    return actual