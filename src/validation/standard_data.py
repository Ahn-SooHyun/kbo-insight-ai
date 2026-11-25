"""표준 데이터의 스키마, 상태 검사 및 품질 증거를 관리한다."""

from collections import Counter
from datetime import date
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3


Row = dict[str, object]

TRANSFORM_VERSION = "1.0.0"

TEAM_CODES = {
    "HT": 2001,
    "KT": 2002,
    "LG": 2003,
    "NC": 2004,
    "SK": 2005,
    "OB": 2006,
    "LT": 2007,
    "SS": 2008,
    "WO": 2009,
    "HH": 2010,
    "HD": 2011,
}

TEAM_ALIASES = {
    **TEAM_CODES,
    "KIA": 2001,
    "기아": 2001,
    "KIA Tigers": 2001,
    "KT Wiz": 2002,
    "LG Twins": 2003,
    "NC Dinos": 2004,
    "SSG": 2005,
    "SSG Landers": 2005,
    "두산": 2006,
    "Doosan Bears": 2006,
    "롯데": 2007,
    "Lotte Giants": 2007,
    "삼성": 2008,
    "Samsung Lions": 2008,
    "넥센": 2009,
    "키움": 2009,
    "Kiwoom Heroes": 2009,
    "한화": 2010,
    "Hanwha Eagles": 2010,
    "현대": 2011,
    "Hyundai Unicorns": 2011,
}

PITCH_TYPES = {
    "FF": "4-Seam Fastball",
    "CH": "Changeup",
    "SL": "Slider",
    "CU": "Curveball",
    "SI": "Sinker",
    "FO": "Forkball",
    "FC": "Cutter",
    "KN": "Knuckleball",
    "ST": "Sweeper",
    "SV": "Slurve",
}

PITCH_RESULT_TYPES = {
    "B": "B",
    "H": "X",
    "T": "S",
    "F": "S",
    "S": "S",
    "W": "S",
    "V": "S",
}

HIT_EVENTS = {
    "single",
    "double",
    "triple",
    "home_run",
}

NON_AB_EVENTS = {
    "walk",
    "hit_by_pitch",
    "sac_fly",
    "sac_bunt",
    "catcher_interference",
}

PA_EVENTS = frozenset({
    *HIT_EVENTS,
    *NON_AB_EVENTS,
    "field_out",
    "fielders_choice",
    "field_error",
    "strikeout",
    "double_play",
    "triple_play",
})

POSTSEASON_NAMES = {
    "와일드카드": "wildCard",
    "준플레이오프": "semiPlayoff",
    "플레이오프": "playoff",
    "한국시리즈": "koreanSeries",
}

GAME_KEY_PATTERN = re.compile(r"\d{8}[A-Z]{4}\d")

SCHEMA = """
CREATE TABLE games (
    game_id TEXT PRIMARY KEY,
    source_key TEXT NOT NULL,
    season INTEGER NOT NULL,
    competition TEXT NOT NULL,
    stage TEXT,
    date TEXT NOT NULL,
    start_time TEXT,
    game_number INTEGER,
    away_team_id INTEGER,
    home_team_id INTEGER,
    away_score INTEGER,
    home_score INTEGER,
    status TEXT NOT NULL,
    winner TEXT,
    is_tie INTEGER,
    result_valid INTEGER NOT NULL,
    season_complete INTEGER,
    source_sha TEXT NOT NULL,
    source_row INTEGER NOT NULL
) STRICT;

CREATE TABLE schedule_reference (
    source_sha TEXT NOT NULL,
    source_row INTEGER NOT NULL,
    source_key TEXT,
    date TEXT NOT NULL,
    start_time TEXT,
    away_code TEXT,
    home_code TEXT,
    series_code TEXT,
    series_name TEXT,
    cancellation_code TEXT,
    cancellation_name TEXT,
    classification TEXT NOT NULL,
    PRIMARY KEY (source_sha, source_row)
) STRICT;

CREATE TABLE plate_appearances (
    game_id TEXT NOT NULL REFERENCES games(game_id),
    pa_number INTEGER NOT NULL,
    inning INTEGER NOT NULL,
    half TEXT NOT NULL,
    start_batter_id TEXT NOT NULL,
    end_batter_id TEXT NOT NULL,
    start_pitcher_id TEXT NOT NULL,
    end_pitcher_id TEXT NOT NULL,
    credited_batter_candidate TEXT,
    credited_pitcher_candidate TEXT,
    participant_changed INTEGER NOT NULL,
    event TEXT,
    result_valid INTEGER NOT NULL,
    state_valid INTEGER NOT NULL,
    physical_pitch_rows INTEGER NOT NULL,
    pitch_order_valid INTEGER NOT NULL,
    pre_home_score INTEGER,
    pre_away_score INTEGER,
    pre_outs INTEGER,
    pre_on_1b TEXT,
    pre_on_2b TEXT,
    pre_on_3b TEXT,
    post_home_score INTEGER,
    post_away_score INTEGER,
    post_outs INTEGER,
    post_on_1b TEXT,
    post_on_2b TEXT,
    post_on_3b TEXT,
    source_sha TEXT NOT NULL,
    start_source_row INTEGER NOT NULL,
    end_source_row INTEGER NOT NULL,
    PRIMARY KEY (game_id, pa_number)
) STRICT;

CREATE TABLE pitches (
    game_id TEXT NOT NULL,
    pa_number INTEGER NOT NULL,
    pitch_number INTEGER NOT NULL,
    batter_id TEXT NOT NULL,
    pitcher_id TEXT NOT NULL,
    pitch_type TEXT,
    pitch_result TEXT,
    pitch_result_type TEXT,
    release_speed_kmh REAL,
    plate_x REAL,
    plate_z REAL,
    balls INTEGER,
    strikes INTEGER,
    outs_when_up INTEGER,
    home_score INTEGER,
    away_score INTEGER,
    on_1b TEXT,
    on_2b TEXT,
    on_3b TEXT,
    state_valid INTEGER NOT NULL,
    order_valid INTEGER NOT NULL,
    target_valid INTEGER NOT NULL,
    source_sha TEXT NOT NULL,
    source_row INTEGER NOT NULL,
    PRIMARY KEY (game_id, pa_number, pitch_number),
    FOREIGN KEY (game_id, pa_number)
        REFERENCES plate_appearances(game_id, pa_number)
) STRICT;

CREATE TABLE non_pitch (
    source_sha TEXT NOT NULL,
    source_row INTEGER NOT NULL,
    game_id TEXT NOT NULL,
    pa_number INTEGER NOT NULL,
    event TEXT,
    PRIMARY KEY (source_sha, source_row)
) STRICT;

CREATE TABLE player_observations (
    player_id TEXT NOT NULL,
    season INTEGER NOT NULL,
    team_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    source_sha TEXT NOT NULL,
    source_row INTEGER NOT NULL,
    PRIMARY KEY (player_id, season, team_id, name)
) STRICT;

CREATE TABLE player_games (
    game_id TEXT NOT NULL REFERENCES games(game_id),
    team_id INTEGER NOT NULL,
    player_id TEXT NOT NULL,
    role TEXT NOT NULL,
    observed_pa INTEGER,
    observed_ab INTEGER,
    observed_hits INTEGER,
    observed_home_runs INTEGER,
    observed_walks INTEGER,
    observed_hbp INTEGER,
    observed_strikeouts INTEGER,
    observed_bf INTEGER,
    observed_outs INTEGER,
    observed_pitches INTEGER,
    ambiguous_pa INTEGER NOT NULL,
    ambiguous_outs INTEGER NOT NULL,
    coverage_candidate INTEGER NOT NULL,
    earned_runs INTEGER,
    source_sha TEXT NOT NULL,
    PRIMARY KEY (game_id, team_id, player_id, role)
) STRICT;

CREATE TABLE starters (
    source_sha TEXT NOT NULL,
    source_row INTEGER NOT NULL,
    game_id TEXT NOT NULL REFERENCES games(game_id),
    side TEXT NOT NULL,
    team_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    player_id TEXT,
    method TEXT NOT NULL,
    valid INTEGER NOT NULL,
    PRIMARY KEY (source_sha, source_row, side, method)
) STRICT;

CREATE TABLE pitching_records (
    source_sha TEXT NOT NULL,
    source_row INTEGER NOT NULL,
    game_id TEXT NOT NULL REFERENCES games(game_id),
    side TEXT NOT NULL,
    team_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    raw_player_id TEXT,
    hf_candidate_id TEXT,
    is_starter INTEGER NOT NULL,
    outs INTEGER,
    batters_faced INTEGER,
    pitches INTEGER,
    at_bats INTEGER,
    hits INTEGER,
    home_runs INTEGER,
    walks_hbp INTEGER,
    strikeouts INTEGER,
    runs INTEGER,
    earned_runs INTEGER,
    valid INTEGER NOT NULL,
    PRIMARY KEY (source_sha, source_row)
) STRICT;

CREATE TABLE game_checks (
    game_id TEXT PRIMARY KEY REFERENCES games(game_id),
    source_sha TEXT NOT NULL,
    physical_rows INTEGER NOT NULL,
    pa_count INTEGER NOT NULL,
    positive_pitch_rows INTEGER NOT NULL,
    non_pitch_rows INTEGER NOT NULL,
    final_score_match INTEGER NOT NULL,
    order_valid INTEGER NOT NULL,
    boundary_valid INTEGER NOT NULL,
    terminal_state TEXT NOT NULL
) STRICT;

CREATE TABLE comparisons (
    source_sha TEXT NOT NULL,
    source_row INTEGER NOT NULL,
    metric TEXT NOT NULL,
    game_id TEXT,
    direct_value TEXT,
    derived_value TEXT,
    status TEXT NOT NULL,
    PRIMARY KEY (source_sha, source_row, metric)
) STRICT;

CREATE TABLE issues (
    id INTEGER PRIMARY KEY,
    severity TEXT NOT NULL,
    reason TEXT NOT NULL,
    entity TEXT NOT NULL,
    entity_key TEXT,
    source_sha TEXT,
    source_row INTEGER,
    impact TEXT NOT NULL
) STRICT;

CREATE INDEX player_games_player_idx
    ON player_games(player_id, role);

CREATE INDEX games_season_idx
    ON games(season, competition, status);

CREATE INDEX starters_game_idx
    ON starters(game_id, side, method);

CREATE INDEX pitching_records_game_idx
    ON pitching_records(game_id, side);
"""

TABLE_NAMES = (
    "games",
    "schedule_reference",
    "plate_appearances",
    "pitches",
    "non_pitch",
    "player_observations",
    "player_games",
    "starters",
    "pitching_records",
    "game_checks",
    "comparisons",
    "issues",
)


class StandardDataError(ValueError):
    """표준화 또는 재생성 과정에서 발견된 검사 오류."""


def valid_integer(
    value: object,
    minimum: int = 0,
    maximum: int | None = None,
) -> bool:
    """bool을 정수로 허용하지 않고 값 범위를 검사한다."""
    return (
        type(value) is int
        and value >= minimum
        and (maximum is None or value <= maximum)
    )


def required_integer(value: object) -> int:
    if not valid_integer(value):
        raise StandardDataError("음이 아닌 정수가 필요합니다.")

    return value


def required_text(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StandardDataError("비어 있지 않은 문자열이 필요합니다.")

    return value.strip()


def normalized_date(value: object) -> str:
    """ISO 날짜 또는 YYYYMMDD를 표준 날짜로 변환한다."""
    text = required_text(value)

    if len(text) == 8 and text.isdigit():
        text = f"{text[:4]}-{text[4:6]}-{text[6:]}"

    try:
        result = date.fromisoformat(text)
    except ValueError as exc:
        raise StandardDataError("날짜 형식 오류") from exc

    if result.isoformat() != text:
        raise StandardDataError("날짜 형식 오류")

    return result.isoformat()


def optional_text(value: object) -> str | None:
    if value is None or value == "":
        return None

    return required_text(value)


def optional_number(value: object) -> float | None:
    if value is None or value == "":
        return None

    if type(value) not in (int, float):
        raise StandardDataError("수치 타입 오류")

    if not math.isfinite(value):
        raise StandardDataError("유한한 수치가 필요합니다.")

    return float(value)


def team_id(value: object) -> int:
    name = required_text(value)

    if name not in TEAM_ALIASES:
        raise StandardDataError(f"미연결 팀 코드: {name}")

    return TEAM_ALIASES[name]


def hf_player_id(value: object) -> str:
    """HF 원천 내부 선수 ID에 출처 범위를 명시한다."""
    return f"hf:{required_text(value)}"


def optional_hf_player_id(value: object) -> str | None:
    if value is None or value == "":
        return None

    return hf_player_id(value)


def innings_to_outs(value: object) -> int:
    """정수 및 1/3, 2/3 혼합 분수 이닝을 아웃 수로 변환한다."""
    text = required_text(value)
    parts = text.split()

    try:
        if len(parts) == 1:
            if not re.fullmatch(r"\d+|\d/[3]", parts[0]):
                raise ValueError

            innings = Fraction(parts[0])

        elif len(parts) == 2:
            if not parts[0].isdigit():
                raise ValueError

            if parts[1] not in {"1/3", "2/3"}:
                raise ValueError

            innings = Fraction(int(parts[0])) + Fraction(parts[1])

        else:
            raise ValueError

    except (ValueError, ZeroDivisionError) as exc:
        raise StandardDataError("이닝 표기 오류") from exc

    outs = innings * 3

    if innings < 0 or outs.denominator != 1:
        raise StandardDataError("아웃 단위 이닝이 아닙니다.")

    return int(outs)


def regular_game_key(
    value: object,
    game_date: object,
    away_code: object,
    home_code: object,
    *,
    hf: bool = False,
) -> str:
    """날짜·홈/원정과 실제 접미부 근거로 경기 키를 확인한다."""
    raw_key = required_text(value)
    day = normalized_date(game_date)
    away = required_text(away_code)
    home = required_text(home_code)

    if hf:
        if len(raw_key) != 17 or raw_key[-4:] != day[:4]:
            raise StandardDataError("HF 경기 ID 연도 접미부 불일치")

        raw_key = raw_key[:-4]

    if not GAME_KEY_PATTERN.fullmatch(raw_key):
        raise StandardDataError("정규시즌 경기 ID 형식 오류")

    if raw_key[:8] != day.replace("-", ""):
        raise StandardDataError("경기 ID와 날짜 불일치")

    if raw_key[8:12] != away + home:
        raise StandardDataError("홈·원정 팀과 경기 ID 불일치")

    return raw_key


def valid_pbp_row(row: Row) -> bool:
    """HF 원본의 핵심 참가자·순서·상태 필드를 검사한다."""
    try:
        normalized_date(row["game_date"])

        for field in (
            "game_pk",
            "batter",
            "pitcher",
            "batter_name",
            "pitcher_name",
            "home_team",
            "away_team",
        ):
            required_text(row[field])

        limits = {
            "inning": (1, None),
            "at_bat_number": (1, None),
            "pitch_number": (0, None),
            "balls": (0, 3),
            "strikes": (0, 2),
            "outs_when_up": (0, 2),
            "post_outs": (0, 3),
            "home_score": (0, None),
            "away_score": (0, None),
            "post_home_score": (0, None),
            "post_away_score": (0, None),
            "runs_scored": (0, None),
        }

        if row["inning_topbot"] not in {"top", "bot"}:
            return False

        if not all(
            valid_integer(row[field], minimum, maximum)
            for field, (minimum, maximum) in limits.items()
        ):
            return False

        for field in (
            "on_1b",
            "on_2b",
            "on_3b",
            "post_on_1b",
            "post_on_2b",
            "post_on_3b",
        ):
            optional_hf_player_id(row[field])

        return True

    except (KeyError, StandardDataError):
        return False


def before_state(row: Row) -> tuple[object, ...]:
    """원본 행에서 관측한 투구 또는 타석 시작 상태."""
    return (
        row["home_score"],
        row["away_score"],
        row["outs_when_up"],
        row["on_1b"],
        row["on_2b"],
        row["on_3b"],
    )


def after_state(row: Row) -> tuple[object, ...]:
    """투구 직후가 아닌 타석 종료 상태."""
    return (
        row["post_home_score"],
        row["post_away_score"],
        row["post_outs"],
        row["post_on_1b"],
        row["post_on_2b"],
        row["post_on_3b"],
    )


def half_key(row: Row) -> tuple[int, str]:
    return int(row["inning"]), str(row["inning_topbot"])


def pitch_order_valid(rows: list[Row]) -> bool:
    """무투구 한 행 또는 1부터 연속된 실제 투구인지 확인한다."""
    numbers = [int(row["pitch_number"]) for row in rows]

    return numbers == [0] or numbers == list(
        range(1, len(rows) + 1)
    )


def pa_state_valid(rows: list[Row]) -> bool:
    """타석 내부의 상태 의미와 비정상 전이를 확인한다."""
    if not rows or not all(valid_pbp_row(row) for row in rows):
        return False

    if len({half_key(row) for row in rows}) != 1:
        return False

    if len({after_state(row) for row in rows}) != 1:
        return False

    first = before_state(rows[0])
    last = after_state(rows[-1])

    if any(last[index] < first[index] for index in range(3)):
        return False

    for previous, following in zip(rows, rows[1:]):
        before = before_state(previous)
        after = before_state(following)

        if any(after[index] < before[index] for index in range(3)):
            return False

    return True


def boundary_valid(previous: Row, following: Row) -> bool:
    """타석 사이 및 반이닝 전환의 관측 연속성을 검사한다."""
    end = after_state(previous)
    start = before_state(following)

    if end[:2] != start[:2]:
        return False

    previous_half = half_key(previous)
    following_half = half_key(following)

    if previous_half == following_half:
        return end == start

    inning, half = previous_half
    expected = (
        (inning, "bot")
        if half == "top"
        else (inning + 1, "top")
    )

    return (
        following_half == expected
        and end[2] == 3
        and start[2] == 0
    )


def valid_pitch_target(row: Row) -> bool:
    """실제 투구의 관측 구종·결과 코드만 정답 후보로 인정한다."""
    if not valid_integer(row.get("pitch_number"), 1):
        return False

    pitch_type = row.get("pitch_type")
    pitch_result = row.get("pitch_result")

    return (
        pitch_type in PITCH_TYPES
        and row.get("pitch_name") == PITCH_TYPES[pitch_type]
        and pitch_result in PITCH_RESULT_TYPES
        and row.get("type") == PITCH_RESULT_TYPES[pitch_result]
    )


class StandardStore:
    """표준 SQLite 저장과 검사 결과 기록을 담당한다."""

    def __init__(self, path: Path) -> None:
        if path.exists():
            raise StandardDataError("기존 표준 DB를 덮어쓸 수 없습니다.")

        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.executescript(SCHEMA)

    def put(
        self,
        table: str,
        row: Row,
        *,
        ignore_existing: bool = False,
    ) -> None:
        """정해진 표준 테이블에 타입 제약을 적용해 저장한다."""
        if table not in TABLE_NAMES:
            raise StandardDataError("등록되지 않은 표준 테이블")

        columns = ", ".join(row)
        placeholders = ", ".join("?" for _ in row)
        operation = (
            "INSERT OR IGNORE" if ignore_existing else "INSERT"
        )

        self.connection.execute(
            f"{operation} INTO {table} ({columns}) "
            f"VALUES ({placeholders})",
            tuple(row.values()),
        )

    def issue(
        self,
        reason: str,
        entity: str,
        impact: str,
        *,
        entity_key: str | None = None,
        source_sha: str | None = None,
        source_row: int | None = None,
        severity: str = "warning",
    ) -> None:
        self.put("issues", {
            "severity": severity,
            "reason": reason,
            "entity": entity,
            "entity_key": entity_key,
            "source_sha": source_sha,
            "source_row": source_row,
            "impact": impact,
        })

    def comparison(
        self,
        source_sha: str,
        source_row: int,
        metric: str,
        game_id: str,
        direct: object,
        derived: object,
    ) -> None:
        if direct is None or derived is None:
            status = "unavailable"
        elif direct == derived:
            status = "match"
        else:
            status = "mismatch"

        self.put("comparisons", {
            "source_sha": source_sha,
            "source_row": source_row,
            "metric": metric,
            "game_id": game_id,
            "direct_value": (
                None if direct is None else str(direct)
            ),
            "derived_value": (
                None if derived is None else str(derived)
            ),
            "status": status,
        })

    def summary(self) -> Row:
        """각 테이블의 행 수, NULL 및 재생성 비교용 해시를 계산한다."""
        tables: Row = {}

        for table in TABLE_NAMES:
            cursor = self.connection.execute(
                f"SELECT * FROM {table} ORDER BY rowid"
            )

            digest = hashlib.sha256()
            null_counts: Counter[str] = Counter()
            columns = [item[0] for item in cursor.description]
            count = 0

            for record in cursor:
                values = list(record)

                for column, value in zip(columns, values):
                    if value is None:
                        null_counts[column] += 1

                payload = json.dumps(
                    values,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )

                digest.update(payload.encode("utf-8"))
                digest.update(b"\n")
                count += 1

            tables[table] = {
                "rows": count,
                "sha256": digest.hexdigest(),
                "null_counts": dict(null_counts),
            }

        issues = [
            dict(row)
            for row in self.connection.execute(
                """
                SELECT severity, entity, reason, COUNT(*) AS count
                FROM issues
                GROUP BY severity, entity, reason
                ORDER BY severity, entity, reason
                """
            )
        ]

        integrity = self.connection.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0]

        foreign_key_errors = list(
            self.connection.execute("PRAGMA foreign_key_check")
        )

        if integrity != "ok" or foreign_key_errors:
            raise StandardDataError("표준 DB 무결성 검사 실패")

        return {
            "tables": tables,
            "issues": issues,
            "integrity_check": integrity,
            "foreign_key_error_count": len(foreign_key_errors),
        }

    def close(self) -> None:
        self.connection.close()


def compare_summaries(
    previous: Row,
    current: Row,
) -> Row:
    """같은 입력 재생성과 정정 판본 간 표준 자료 차이를 구분한다."""
    same_input = (
        previous["input_fingerprint"]
        == current["input_fingerprint"]
    )

    old_tables = previous["tables"]
    new_tables = current["tables"]

    changes = {}

    for name in TABLE_NAMES:
        old = old_tables[name]
        new = new_tables[name]

        changes[name] = {
            "previous_rows": old["rows"],
            "current_rows": new["rows"],
            "row_delta": new["rows"] - old["rows"],
            "content_changed": old["sha256"] != new["sha256"],
        }

    identical = not any(
        row["content_changed"]
        for row in changes.values()
    )

    return {
        "same_input": same_input,
        "identical_tables": identical,
        "reproducible": identical if same_input else None,
        "changes": changes,
    }