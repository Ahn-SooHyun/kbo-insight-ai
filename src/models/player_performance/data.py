"""M3 인수 자료·예측 시점·관측 정답 및 입력 계약을 검증한다."""

from collections import Counter
from collections.abc import Mapping
from contextlib import closing
from dataclasses import dataclass
from datetime import date, timedelta
import math
from pathlib import Path
import sqlite3

from ...datasets.model_data import fingerprint, inside, require
from ...datasets.model_players import HORIZONS, ROLE_COUNTS, ROLES
from ...datasets.published_sources import (
    as_row,
    read_json,
    sha256_file,
)
from ...settings import Settings


INTAKE_RUN = "259b38decaac4ed68dd5a596cfbe4849"
INTAKE_SHA256 = (
    "c8383a9d3e8930830a137d1b6791cbb26"
    "e2316dbbbdf4f0f436ceda9a43890d0"
)
STANDARD_RUN = "e020dafedc8b4fd9ac6696d0e4e4a92e"
STANDARD_VERIFICATION = "37b56c39caf5435e8fb2a17f993ec4df"
DATA_VERSION = "1.2.0"

CALIBRATION_START = date(2024, 1, 1)
VALIDATION_START = date(2025, 1, 1)
TEST_START = date(2026, 1, 1)
OBSERVATION_END = date(2026, 11, 26)

VALID_STATUSES = frozenset({"valid_observed", "zero_observed"})
ALL_STATUSES = VALID_STATUSES | {
    "partial_unfinished",
    "unobserved",
    "partial",
    "missing",
}
RISK_FIELDS = (
    "missing_games",
    "league_risky_games",
    "team_risky_games",
    "league_identity_conflicts",
    "league_rejected_rows",
    "team_rejected_rows",
    "player_rejected",
)
EXPECTED = {
    "hitter": {
        "train": (6263, 3435, 945),
        "validation": (1114, 204, 171),
    },
    "pitcher": {
        "train": (5306, 2926, 754),
        "validation": (934, 147, 116),
    },
}
EXPECTED_VALID_RATIOS = {
    "hitter": (202, 169),
    "pitcher": (147, 116),
}

Count = int | float
OptionalCounts = tuple[int | None, ...]


def metrics(role: str) -> tuple[str, ...]:
    """역할별 출전량 및 누적 기록의 고정 출력 순서."""
    require(role in ROLES, "지원하지 않는 선수 역할")
    return ("games", *ROLE_COUNTS[role])


def exposure(role: str) -> str:
    """역할별 타석 또는 상대 타자 분모."""
    require(role in ROLES, "지원하지 않는 선수 역할")
    return "pa" if role == "hitter" else "bf"


def output_keys(role: str) -> tuple[str, ...]:
    """기간과 기록을 결합한 공동 출력 사전."""
    return tuple(
        f"{horizon}.{metric}"
        for horizon in HORIZONS
        for metric in metrics(role)
    )


def history_columns(role: str) -> tuple[str, ...]:
    """데이터 생성기와 동일한 과거 기록 열."""
    return tuple(
        f"{prefix}_{metric}"
        for prefix in ("season", "recent28")
        for metric in metrics(role)
    )


def input_columns(role: str) -> tuple[str, ...]:
    """실제 모델 자료의 입력 테이블 열."""
    return (
        "player_id",
        "season",
        "known_team_id",
        "days_since_observed",
        "prior_rejected",
        *history_columns(role),
    )


def day(value: object) -> date:
    """ISO 날짜를 검증한다. 시각과 비표준 표기는 허용하지 않는다."""
    require(isinstance(value, str), "날짜 문자열이 필요합니다")
    parsed = date.fromisoformat(value)
    require(parsed.isoformat() == value, "ISO 날짜 형식 오류")
    return parsed


def integer(
    value: object,
    name: str,
    *,
    nullable: bool = False,
) -> int | None:
    """실제 Count의 정수·비음수 계약을 확인한다."""
    if nullable and value is None:
        return None

    require(
        type(value) is int and value >= 0,
        f"비음수 정수 오류: {name}",
    )
    return value


def check_counts(
    role: str,
    counts: Mapping[str, Count | None],
) -> None:
    """확인 가능한 구성 기록 관계를 검사한다.

    아웃은 주루 사건에서도 발생할 수 있고 무투구 타석도 존재하므로
    outs<=bf 또는 pitches>=bf 같은 미확정 제약은 강제하지 않는다.
    """
    require(
        set(counts) == set(metrics(role)),
        "Count 기록 사전 불일치",
    )

    for name, value in counts.items():
        require(
            value is None
            or (
                type(value) in (int, float)
                and math.isfinite(value)
                and value >= 0
            ),
            f"Count 범위 오류: {name}",
        )

    def bounded(
        children: tuple[str, ...],
        parent: str,
    ) -> None:
        parent_value = counts[parent]
        values = [counts[name] for name in children]

        if parent_value is None or any(
            value is None for value in values
        ):
            return

        total = sum(
            value for value in values if value is not None
        )

        require(
            total <= parent_value + 1e-8,
            f"Count 구성 오류: {children} > {parent}",
        )

    opportunity = exposure(role)

    bounded(("ab",), opportunity)
    bounded(("walks",), opportunity)
    bounded(("hbp",), opportunity)
    bounded(("ab", "walks", "hbp"), opportunity)
    bounded(("hits",), "ab")
    bounded(("home_runs",), "hits")
    bounded(("strikeouts",), "ab")
    bounded(("hits", "strikeouts"), "ab")

    if counts["games"] == 0:
        require(
            all(
                value is None or value <= 1e-8
                for name, value in counts.items()
                if name != "games"
            ),
            "경기 수 0과 양의 기록이 함께 존재합니다",
        )


@dataclass(frozen=True, slots=True)
class PlayerInput:
    """기준 월요일 이전에 알 수 있는 역할별 선수 자료."""

    role: str
    sample_id: str
    player_id: str
    season: int
    anchor: date
    last_observed: date
    known_team_id: int | None
    days_since_observed: int
    prior_rejected: int
    season_counts: OptionalCounts
    recent_counts: OptionalCounts
    current_counts: tuple[int, ...] | None
    current_scope: str
    known_end: date | None
    end_evidence: str | None


@dataclass(frozen=True, slots=True)
class Target:
    """기간별 Count 정답과 실제 관측 완료 근거."""

    horizon: str
    window_end: date | None
    observed_end: date
    ready: date | None
    status: str
    usable: bool
    ratio_status: str
    counts: OptionalCounts
    hit_rate: float | None
    strikeout_rate: float | None
    risks: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Sample:
    """한 선수·역할·기준 월요일의 공동 학습 표본."""

    request: dict[str, object]
    player: PlayerInput
    split: str
    targets: tuple[Target, ...]


@dataclass(slots=True)
class Dataset:
    """Train·Validation 자료와 인수 및 품질 감사 결과."""

    train: list[Sample]
    validation: list[Sample]
    provenance: dict[str, object]
    audit: dict[str, object]


def parse_input(value: object) -> PlayerInput:
    """외부 추론에서도 동일한 사전 입력 계약을 검증한다.

    등록 여부·자료 진위·실제 시즌 진행 상태는 호출자가
    기준 시점의 근거로 확인해야 한다.
    """
    row = as_row(value)
    role = row.get("role")
    require(isinstance(role, str) and role in ROLES, "역할 오류")

    required = {
        "role",
        "sample_id",
        "sample_date",
        "last_observed_date",
        "season_active",
        *input_columns(role),
    }
    optional = {
        "current_counts",
        "current_scope",
        "current_as_of",
        "known_end_exclusive",
        "known_end_as_of",
        "known_end_source",
    }

    require(
        required <= set(row) <= required | optional,
        "M3 입력 열 누락 또는 허용하지 않은 입력",
    )
    require(
        row["season_active"] is True,
        "기준 시점에 정규시즌이 진행 중이어야 합니다",
    )

    anchor = day(row["sample_date"])
    last = day(row["last_observed_date"])
    season = integer(row["season"], "season")
    player_id = row["player_id"]

    require(
        anchor.weekday() == 0
        and anchor.year == season
        and last.year == season
        and last < anchor,
        "기준 월요일 또는 과거 기록 날짜 오류",
    )
    require(
        isinstance(player_id, str) and bool(player_id.strip()),
        "선수 식별자 오류",
    )
    require(
        row["sample_id"]
        == f"{season}:{anchor.isoformat()}:{player_id}",
        "선수·시즌·기준 월요일 식별자 불일치",
    )

    days_since = integer(
        row["days_since_observed"],
        "days_since_observed",
    )
    prior_rejected = integer(
        row["prior_rejected"],
        "prior_rejected",
    )

    require(
        days_since == (anchor - last).days,
        "최근 관측 이후 경과일 불일치",
    )

    known_team = integer(
        row["known_team_id"],
        "known_team_id",
        nullable=True,
    )

    names = metrics(role)
    histories: list[OptionalCounts] = []

    for prefix in ("season", "recent28"):
        counts = tuple(
            integer(
                row[f"{prefix}_{name}"],
                f"{prefix}_{name}",
                nullable=name != "games",
            )
            for name in names
        )
        check_counts(role, dict(zip(names, counts)))
        histories.append(counts)

    season_counts, recent_counts = histories

    for total, recent in zip(season_counts, recent_counts):
        require(
            total is None or recent is None or recent <= total,
            "최근 28일 기록이 시즌 누적을 초과합니다",
        )

    # 관측된 Count와 호출자가 별도 확인한 전체 누적을 구분한다.
    current = row.get("current_counts")
    scope = row.get("current_scope")
    current_counts = None

    if current is not None:
        require(
            scope in {"caller_verified", "observed_pbp"},
            "현재 누적의 확인 범위가 필요합니다",
        )
        require(
            day(row.get("current_as_of")) == anchor,
            "현재 누적의 종료 경계가 기준 월요일과 다릅니다",
        )

        current_row = as_row(current)
        require(
            set(current_row) == set(names),
            "현재 누적 기록 사전 불일치",
        )

        current_counts = tuple(
            int(integer(current_row[name], name))
            for name in names
        )
        check_counts(role, dict(zip(names, current_counts)))

        if scope == "observed_pbp":
            require(
                current_counts == season_counts,
                "관측 PBP 누적과 입력 누적이 다릅니다",
            )

    else:
        require(
            scope is None and row.get("current_as_of") is None,
            "현재 누적이 없으면 누적 범위도 지정하지 않습니다",
        )
        if all(value is not None for value in season_counts):
            current_counts = tuple(
                int(value)
                for value in season_counts
                if value is not None
            )
            scope = "observed_pbp"
        else:
            scope = "unavailable"

    known_end = None
    end_evidence = None
    end_fields = {
        "known_end_exclusive",
        "known_end_as_of",
        "known_end_source",
    }

    if any(row.get(name) is not None for name in end_fields):
        known_end = day(row.get("known_end_exclusive"))
        evidence_day = day(row.get("known_end_as_of"))
        end_evidence = row.get("known_end_source")

        require(
            evidence_day < anchor
            and anchor < known_end
            and known_end.year == season
            and isinstance(end_evidence, str)
            and bool(end_evidence.strip()),
            "정규시즌 종료 경계의 사전 이용 가능 근거 오류",
        )

    return PlayerInput(
        role=role,
        sample_id=str(row["sample_id"]),
        player_id=player_id,
        season=int(season),
        anchor=anchor,
        last_observed=last,
        known_team_id=known_team,
        days_since_observed=int(days_since),
        prior_rejected=int(prior_rejected),
        season_counts=season_counts,
        recent_counts=recent_counts,
        current_counts=current_counts,
        current_scope=str(scope),
        known_end=known_end,
        end_evidence=end_evidence,
    )


def split_for(anchor: date) -> str:
    """최종 Test를 포함한 기존 시간 분할을 확인한다."""
    require(anchor < OBSERVATION_END, "관측 종료일 이후 표본")

    if anchor < VALIDATION_START:
        return "train"

    if anchor < TEST_START:
        return "validation"

    return "test"


def parse_target(
    row: Mapping[str, object],
    player: PlayerInput,
    split: str,
) -> Target:
    """부분 관측값을 정답으로 대체하지 않고 마스크를 구성한다."""
    role = player.role
    names = metrics(role)
    horizon = row["horizon"]

    require(horizon in HORIZONS, "정답 기간 오류")
    require(
        row["sample_id"] == player.sample_id
        and day(row["window_start"]) == player.anchor,
        "선수 정답 키 또는 시작일 불일치",
    )

    end = (
        day(row["window_end"])
        if row["window_end"] is not None
        else None
    )
    ready = (
        day(row["ready_date"])
        if row["ready_date"] is not None
        else None
    )
    observed_end = day(row["observed_through_exclusive"])

    if horizon != "remaining":
        expected = player.anchor + timedelta(
            days=7 if horizon == "7d" else 28
        )
        require(end == expected, "고정 기간의 종료일 오류")

    require(
        player.anchor <= observed_end <= OBSERVATION_END
        and (end is None or player.anchor < end)
        and (end is None or observed_end <= end)
        and (ready is None or ready == end),
        "관측 기간 또는 정답 완료일 오류",
    )

    status = row["status"]
    require(status in ALL_STATUSES, "정답 상태 오류")

    flag = row["usable"]
    require(type(flag) is int and flag in (0, 1), "usable 오류")

    cutoff = (
        VALIDATION_START
        if split == "train"
        else TEST_START
    )
    valid = (
        status in VALID_STATUSES
        and end is not None
        and ready is not None
        and ready <= cutoff
    )

    require(bool(flag) == valid, "정답 완료·분할 마스크 불일치")

    counts = tuple(
        integer(
            row[f"target_{name}"],
            f"target_{name}",
            nullable=True,
        )
        for name in names
    )

    if valid:
        require(
            any(value is not None for value in counts),
            "유효 정답에 기록이 없습니다",
        )
        check_counts(role, dict(zip(names, counts)))

        for name, count in zip(names, counts):
            if count is not None:
                require(
                    row[f"observed_{name}"] == count,
                    "확정 정답과 관측 기록 불일치",
                )

        if status == "zero_observed":
            require(
                all(
                    value is None or value == 0
                    for value in counts
                ),
                "관측 0 정답에 양의 기록 존재",
            )
    else:
        require(
            all(value is None for value in counts),
            "사용 불가 정답에 Count가 남아 있습니다",
        )

    risks = tuple(
        int(integer(row[name], name))
        for name in RISK_FIELDS
    )
    risk = dict(zip(RISK_FIELDS, risks))

    if valid:
        require(
            not risk["missing_games"]
            and not risk["team_risky_games"]
            and not risk["player_rejected"],
            "품질 위험이 있는 정답 사용",
        )

        if status == "zero_observed":
            require(
                not risk["league_risky_games"]
                and not risk["league_identity_conflicts"]
                and not risk["league_rejected_rows"],
                "미확인 0 정답 사용",
            )

    ratio_status = row["ratio_status"]
    actual = dict(zip(names, counts))
    rate_values: list[float | None] = []

    if horizon == "7d":
        expected_ratio = "not_requested"
    elif not valid:
        expected_ratio = "unavailable"
    elif actual["ab"] and actual[exposure(role)]:
        expected_ratio = "valid"
    else:
        expected_ratio = "zero_denominator"

    require(
        ratio_status == expected_ratio,
        "비율 정답 상태 불일치",
    )

    for key, numerator, denominator in (
        ("target_hit_rate", "hits", "ab"),
        ("target_strikeout_rate", "strikeouts", exposure(role)),
    ):
        value = row[key]
        n = actual[numerator]
        d = actual[denominator]

        if horizon == "7d" or not valid or not d or n is None:
            require(value is None, f"비율 마스크 오류: {key}")
            rate_values.append(None)
        else:
            require(
                type(value) in (int, float)
                and math.isfinite(value)
                and math.isclose(
                    value,
                    n / d,
                    rel_tol=1e-10,
                    abs_tol=1e-10,
                ),
                f"비율 정답 불일치: {key}",
            )
            rate_values.append(float(value))

    return Target(
        horizon=str(horizon),
        window_end=end,
        observed_end=observed_end,
        ready=ready,
        status=str(status),
        usable=valid,
        ratio_status=str(ratio_status),
        counts=counts,
        hit_rate=rate_values[0],
        strikeout_rate=rate_values[1],
        risks=risks,
    )


def labels_for(
    sample: Sample,
    cutoff: date,
) -> tuple[int | None, ...]:
    """논리적 적합·평가 시각까지 완료된 기록만 반환한다."""
    labels: list[int | None] = []

    for target in sample.targets:
        available = (
            sample.player.anchor < cutoff
            and target.usable
            and target.ready is not None
            and target.ready <= cutoff
            and target.window_end is not None
            and target.window_end <= cutoff
        )

        if available:
            labels.extend(target.counts)
        else:
            labels.extend(None for _ in target.counts)

    return tuple(labels)


def verify_target_relationships(
    targets: tuple[Target, ...],
) -> None:
    """정답 기간 관계와 시즌 종료에 의한 관측 절단을 검사한다."""
    require(
        tuple(target.horizon for target in targets) == HORIZONS,
        "기간별 정답 누락 또는 순서 오류",
    )

    season_end = targets[-1].window_end

    for target in targets:
        if target.usable:
            expected = target.window_end
            if season_end is not None:
                expected = min(expected, season_end)

            require(
                target.observed_end == expected,
                "정답 관측 완료 범위 불일치",
            )

    for i, earlier in enumerate(targets):
        for later in targets[i + 1:]:
            if not earlier.usable or not later.usable:
                continue

            for a, b in zip(earlier.counts, later.counts):
                if a is None or b is None:
                    continue

                require(a <= b, "정답 기간별 누적 관계 오류")

                if (
                    earlier.window_end is not None
                    and season_end is not None
                    and earlier.window_end >= season_end
                ):
                    require(a == b, "시즌 종료 후 Count 불일치")


def intake(
    settings: Settings,
    run_id: str,
    role: str,
) -> tuple[Path, Path, dict[str, object]]:
    """승인된 모델 자료·계약·표준 자료의 지문을 대조한다."""
    require(run_id == INTAKE_RUN, "승인되지 않은 자료 실행 ID")

    directory = settings.processed_dir / "model_data" / run_id
    database = inside(directory, "model_data.sqlite3")
    manifest_path = inside(directory, "manifest.json")
    contract_path = inside(directory, "feature_target_contract.json")
    eda_path = inside(directory, "train_eda.json")
    supplement_path = inside(directory, "supplement-impact.json")

    manifest = as_row(read_json(manifest_path))
    definitions = as_row(read_json(contract_path))
    config = as_row(manifest["config"])
    source = as_row(manifest["source"])
    contract = as_row(definitions[ROLES[role]])

    require(
        manifest["status"] == "completed"
        and manifest["run_id"] == run_id
        and config["version"] == DATA_VERSION
        and config["timezone"] == "Asia/Seoul"
        and config["seed"] == settings.seed
        and config["validation_start"]
        == VALIDATION_START.isoformat()
        and config["test_start"] == TEST_START.isoformat()
        and config["observation_end"]
        == OBSERVATION_END.isoformat()
        and source["run_id"] == STANDARD_RUN
        and source["verification_id"] == STANDARD_VERIFICATION,
        "M3 인수 매니페스트 계약 불일치",
    )

    require(
        contract["index"] == f"{ROLES[role]}_index"
        and set(as_row(contract["inputs"]))
        == set(input_columns(role)),
        "M3 입력 계약 불일치",
    )

    require(
        fingerprint(definitions) == manifest["contract_sha256"]
        and fingerprint(read_json(eda_path))
        == manifest["eda_sha256"],
        "자료 계약 또는 Train EDA 지문 불일치",
    )

    require(
        manifest["database_sha256"] == INTAKE_SHA256
        and sha256_file(database) == INTAKE_SHA256,
        "인수 모델 DB SHA-256 불일치",
    )

    standard = inside(
        settings.processed_dir / "standard" / STANDARD_RUN,
        f"verified-{STANDARD_VERIFICATION}.sqlite3",
    )

    standard_hash = sha256_file(standard)
    require(
        standard_hash == source["database_sha256"],
        "표준 검증 DB SHA-256 불일치",
    )

    as_row(read_json(supplement_path))

    return database, standard, {
        "model_data_run_id": run_id,
        "model_data_version": DATA_VERSION,
        "database_sha256": INTAKE_SHA256,
        "standard_run_id": STANDARD_RUN,
        "standard_verification_id": STANDARD_VERIFICATION,
        "standard_database_sha256": standard_hash,
        "manifest_sha256": sha256_file(manifest_path),
        "contract_sha256": manifest["contract_sha256"],
        "eda_sha256": manifest["eda_sha256"],
        "supplement_impact_sha256": sha256_file(supplement_path),
        "timezone": "Asia/Seoul",
        "contract": contract,
        "final_test_used": False,
    }


def verify_keys(
    db: sqlite3.Connection,
    role: str,
    contract: Mapping[str, object],
) -> None:
    """전체 키와 개발 분할 연결을 검사한다.

    Test의 정답 값은 조회하지 않는다.
    """
    table = ROLES[role]
    schemas = {
        f"{table}_index": {
            "sample_id", "player_id", "role", "season",
            "sample_date", "split", "status", "last_observed_date",
        },
        f"{table}_inputs": {"sample_id", *input_columns(role)},
        f"{table}_targets": set(contract["target_columns"]),
    }

    for name, expected in schemas.items():
        actual = {
            row[1] for row in db.execute(
                f"PRAGMA table_info({name})"
            )
        }
        require(actual == expected, f"테이블 스키마 오류: {name}")

    queries = (
        f"""
        SELECT 1 FROM {table}_index i
        LEFT JOIN {table}_inputs x USING(sample_id)
        WHERE x.sample_id IS NULL OR i.role<>?
           OR i.status<>'candidate'
           OR x.player_id IS NOT i.player_id
           OR x.season IS NOT i.season
        LIMIT 1
        """,
        f"""
        SELECT 1 FROM {table}_inputs x
        LEFT JOIN {table}_index i USING(sample_id)
        WHERE i.sample_id IS NULL LIMIT 1
        """,
        f"""
        SELECT 1 FROM {table}_targets t
        LEFT JOIN {table}_index i USING(sample_id)
        WHERE i.sample_id IS NULL LIMIT 1
        """,
        f"""
        SELECT 1 FROM {table}_index i
        LEFT JOIN {table}_targets t USING(sample_id)
        GROUP BY i.sample_id
        HAVING COUNT(t.horizon)<>3
          OR COUNT(DISTINCT t.horizon)<>3
          OR SUM(t.horizon NOT IN ('7d','28d','remaining'))<>0
        LIMIT 1
        """,
    )

    for i, sql in enumerate(queries):
        params = (role,) if i == 0 else ()
        require(
            db.execute(sql, params).fetchone() is None,
            "M3 선수 입력·정답 연결 오류",
        )


def verify_games(
    db: sqlite3.Connection,
    role: str,
) -> dict[int, date]:
    """개발 정답 창의 정규경기 귀속과 공식 종료 근거를 대조한다."""
    table = ROLES[role]

    windows = f"""
        SELECT i.season, i.sample_date, t.horizon,
               t.window_start, t.observed_through_exclusive,
               MIN(t.league_games) league_games,
               MAX(t.league_games) maximum_games
        FROM {table}_index i
        JOIN {table}_targets t USING(sample_id)
        WHERE i.sample_date<?
          AND i.split IN ('train','validation')
          AND t.usable=1
        GROUP BY 1,2,3,4,5
    """

    invalid = db.execute(
        f"""
        WITH windows AS ({windows})
        SELECT 1
        FROM windows a
        JOIN m3_window_games w
          ON w.season=a.season
         AND w.anchor_date=a.sample_date
         AND w.horizon=a.horizon
        LEFT JOIN standard.eligible_games g USING(game_id)
        WHERE g.game_id IS NULL
           OR g.competition<>'regular'
           OR g.season<>a.season
           OR g.date<a.window_start
           OR g.date>=a.observed_through_exclusive
        LIMIT 1
        """,
        (TEST_START.isoformat(),),
    ).fetchone()

    require(invalid is None, "정규시즌 정답 창 경기 귀속 오류")

    invalid_count = db.execute(
        f"""
        WITH windows AS ({windows})
        SELECT 1
        FROM windows a
        LEFT JOIN m3_window_games w
          ON w.season=a.season
         AND w.anchor_date=a.sample_date
         AND w.horizon=a.horizon
        GROUP BY a.season,a.sample_date,a.horizon,
                 a.window_start,a.observed_through_exclusive
        HAVING COUNT(w.game_id)<>a.league_games
            OR a.league_games<>a.maximum_games
        LIMIT 1
        """,
        (TEST_START.isoformat(),),
    ).fetchone()

    require(invalid_count is None, "정답 창 경기 수 불일치")

    return {
        row["season"]: day(row["last_date"]) + timedelta(days=1)
        for row in db.execute(
            """
            SELECT season, MAX(date) last_date
            FROM standard.eligible_games
            WHERE competition='regular' AND season<?
            GROUP BY season
            HAVING MIN(COALESCE(season_complete,0))=1
            """,
            (TEST_START.year,),
        )
    }


def audit_samples(
    samples: list[Sample],
    role: str,
    cutoff: date,
) -> dict[str, object]:
    """모집단·상태·마스크·리그 및 팀 위험을 집계한다."""
    horizons: dict[str, object] = {}
    names = metrics(role)

    for i, horizon in enumerate(HORIZONS):
        status: Counter[str] = Counter()
        ratios: Counter[str] = Counter()
        masks: Counter[str] = Counter()
        risks: Counter[str] = Counter()
        teams: dict[str, Counter[str]] = {}
        players: set[str] = set()
        weeks: set[str] = set()
        usable = 0
        boundary_excluded = 0

        for sample in samples:
            target = sample.targets[i]
            status[target.status] += 1
            ratios[target.ratio_status] += 1

            team_key = str(sample.player.known_team_id)
            team = teams.setdefault(team_key, Counter())
            team["population"] += 1

            if target.usable:
                usable += 1
                team["usable"] += 1

            if (
                target.usable
                and target.ready is not None
                and target.ready > cutoff
            ):
                boundary_excluded += 1

            start = i * len(names)
            selected = labels_for(sample, cutoff)[
                start:start + len(names)
            ]

            if any(value is not None for value in selected):
                players.add(sample.player.player_id)
                weeks.add(sample.player.anchor.isoformat())
                team["available"] += 1

            for name, value in zip(names, selected):
                masks[f"{name}.valid" if value is not None
                      else f"{name}.masked"] += 1

                if value == 0:
                    masks[f"{name}.observed_zero"] += 1

            for name, value in zip(RISK_FIELDS, target.risks):
                if value:
                    risks[name] += 1
                    team[name] += 1

        horizons[horizon] = {
            "population": len(samples),
            "usable": usable,
            "additional_boundary_excluded": boundary_excluded,
            "statuses": dict(status),
            "ratio_statuses": dict(ratios),
            "record_masks": dict(masks),
            "risk_affected_rows": dict(risks),
            "known_team_groups": {
                key: dict(value) for key, value in teams.items()
            },
            "unique_available_players": len(players),
            "available_anchor_weeks": len(weeks),
        }

    return {
        "samples": len(samples),
        "unique_players": len({
            sample.player.player_id for sample in samples
        }),
        "anchor_weeks": len({
            sample.player.anchor for sample in samples
        }),
        "no_admitted_season_games": sum(
            sample.player.season_counts[0] == 0
            for sample in samples
        ),
        "unknown_known_team": sum(
            sample.player.known_team_id is None
            for sample in samples
        ),
        "prior_rejected_affected": sum(
            sample.player.prior_rejected > 0
            for sample in samples
        ),
        "horizons": horizons,
    }


def load_dataset(
    settings: Settings,
    run_id: str,
    role: str,
) -> Dataset:
    """개발 자료만 로드하고 전체 Test 정답은 격리한다."""
    database, standard, provenance = intake(
        settings, run_id, role
    )
    table = ROLES[role]
    contract = as_row(provenance["contract"])
    samples: dict[str, list[Sample]] = {
        "train": [],
        "validation": [],
    }

    with closing(sqlite3.connect(
        database.resolve().as_uri() + "?mode=ro",
        uri=True,
    )) as db:
        db.row_factory = sqlite3.Row
        db.execute(
            "ATTACH DATABASE ? AS standard",
            (standard.resolve().as_uri() + "?mode=ro",),
        )
        db.execute("PRAGMA query_only=ON")

        verify_keys(db, role, contract)
        season_ends = verify_games(db, role)

        target_rows: dict[str, dict[str, dict[str, object]]] = {}

        for row in db.execute(
            f"""
            SELECT t.*
            FROM {table}_targets t
            JOIN {table}_index i USING(sample_id)
            WHERE i.sample_date<?
              AND i.split IN ('train','validation')
            ORDER BY t.sample_id,t.horizon
            """,
            (TEST_START.isoformat(),),
        ):
            target_rows.setdefault(row["sample_id"], {})[
                row["horizon"]
            ] = dict(row)

        selected = ",".join(
            f"x.{name}" for name in input_columns(role)
        )

        for row in db.execute(
            f"""
            SELECT i.sample_id,i.role,i.sample_date,
                   i.last_observed_date,i.split,{selected}
            FROM {table}_index i
            JOIN {table}_inputs x USING(sample_id)
            WHERE i.sample_date<?
            ORDER BY i.sample_date,i.player_id
            """,
            (TEST_START.isoformat(),),
        ):
            request = dict(row)
            split = request.pop("split")
            request["season_active"] = True
            player = parse_input(request)

            require(
                split == split_for(player.anchor)
                and split in samples,
                "M3 개발 분할 오류",
            )

            rows = target_rows.pop(player.sample_id)

            targets = tuple(
                parse_target(rows[horizon], player, split)
                for horizon in HORIZONS
            )

            require(
                targets[-1].window_end
                == season_ends.get(player.season),
                "잔여 정규시즌 종료 근거 불일치",
            )

            verify_target_relationships(targets)
            samples[split].append(
                Sample(request, player, split, targets)
            )

        require(not target_rows, "연결되지 않은 개발 정답")

    for split, rows in samples.items():
        actual = tuple(
            sum(sample.targets[i].usable for sample in rows)
            for i in range(len(HORIZONS))
        )
        require(
            actual == EXPECTED[role][split],
            f"인수 유효 정답 수 불일치: {role}/{split}/{actual}",
        )

    ratios = tuple(
        sum(
            sample.targets[i].ratio_status == "valid"
            for sample in samples["validation"]
        )
        for i in (1, 2)
    )
    require(
        ratios == EXPECTED_VALID_RATIOS[role],
        "인수 Validation 비율 정답 수 불일치",
    )

    provenance["verified_season_ends"] = {
        str(year): value.isoformat()
        for year, value in season_ends.items()
    }
    provenance["database_open_mode"] = "read_only"

    return Dataset(
        train=samples["train"],
        validation=samples["validation"],
        provenance=provenance,
        audit={
            "train": audit_samples(
                samples["train"], role, VALIDATION_START
            ),
            "validation": audit_samples(
                samples["validation"], role, TEST_START
            ),
        },
    )