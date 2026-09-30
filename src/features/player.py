from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import pandas as pd


FEATURE_VERSION = "player_pregame_v1"
RATE_DECIMAL_PLACES = 3
ROLLING_WINDOWS = (5, 10, 20)
VALID_ROLES = ("batting", "pitching")

REQUEST_KEY = (
    "player_id",
    "role",
    "prediction_date",
)

AUDIT_COLUMNS = (
    "max_source_game_date",
    "history_game_count",
    "has_history",
    "feature_version",
)

COMMON_FEATURE_COLUMNS = (
    "days_since_last_observed_appearance",
)

BATTING_SOURCE_COUNTS = (
    "pa",
    "ab",
    "h",
    "single",
    "double",
    "triple",
    "hr",
    "bb",
    "hbp",
    "so",
    "sf",
    "sh",
    "tb",
    "double_play",
    "triple_play",
    "field_error",
    "fielders_choice",
    "catcher_interference",
)

BATTING_RATE_NAMES = (
    "avg",
    "obp",
    "slg",
    "ops",
)

PITCHING_SOURCE_COUNTS = (
    "pitch_rows",
    "pitches",
    "batters_faced_completed",
    "hits_allowed",
    "single_allowed",
    "double_allowed",
    "triple_allowed",
    "hr_allowed",
    "bb_allowed",
    "hbp_allowed",
    "so",
    "sf",
    "sh",
    "outs_recorded",
    "ball_pitch_count",
    "strike_pitch_count",
    "in_play_pitch_count",
)

BATTING_REQUIRED_SOURCE_COLUMNS = (
    "game_pk",
    "game_date",
    "season",
    "batter",
) + BATTING_SOURCE_COUNTS

PITCHING_REQUIRED_SOURCE_COLUMNS = (
    "game_pk",
    "game_date",
    "season",
    "pitcher",
) + PITCHING_SOURCE_COUNTS


def _ordered_union(*groups: Sequence[str]) -> tuple[str, ...]:
    """입력 순서를 유지하면서 중복 없는 컬럼 이름 튜플을 만든다."""
    seen: set[str] = set()
    result: list[str] = []
    for group in groups:
        for value in group:
            if value in seen:
                continue
            seen.add(value)
            result.append(value)
    return tuple(result)


def _prefixed(prefix: str, names: Sequence[str]) -> tuple[str, ...]:
    """통계 이름에 Historical Feature 접두사를 붙인다."""
    return tuple(f"{prefix}{name}" for name in names)


BATTING_EXPANDING_COLUMNS = _prefixed("hist_", BATTING_SOURCE_COUNTS) + _prefixed(
    "hist_", BATTING_RATE_NAMES
)
PITCHING_EXPANDING_COLUMNS = _prefixed("hist_", PITCHING_SOURCE_COUNTS)

BATTING_ROLLING_COLUMNS = tuple(
    column
    for window in ROLLING_WINDOWS
    for column in (
        _prefixed(f"last_{window}g_", BATTING_SOURCE_COUNTS)
        + _prefixed(f"last_{window}g_", BATTING_RATE_NAMES)
    )
)
PITCHING_ROLLING_COLUMNS = tuple(
    column
    for window in ROLLING_WINDOWS
    for column in _prefixed(f"last_{window}g_", PITCHING_SOURCE_COUNTS)
)
ROLLING_AUDIT_COLUMNS = tuple(f"last_{window}g_games" for window in ROLLING_WINDOWS)
OUTS_QUALITY_AUDIT_COLUMNS = (
    "history_uncertain_outs_game_count",
) + tuple(
    f"last_{window}g_uncertain_outs_game_count"
    for window in ROLLING_WINDOWS
)

ROLE_FEATURE_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "batting": (
        COMMON_FEATURE_COLUMNS
        + BATTING_EXPANDING_COLUMNS
        + BATTING_ROLLING_COLUMNS
    ),
    "pitching": (
        COMMON_FEATURE_COLUMNS
        + PITCHING_EXPANDING_COLUMNS
        + PITCHING_ROLLING_COLUMNS
    ),
}

FEATURE_COLUMNS = _ordered_union(
    ROLE_FEATURE_COLUMNS["batting"],
    ROLE_FEATURE_COLUMNS["pitching"],
)

OUTPUT_COLUMNS = (
    REQUEST_KEY
    + AUDIT_COLUMNS
    + ROLLING_AUDIT_COLUMNS
    + OUTS_QUALITY_AUDIT_COLUMNS
    + FEATURE_COLUMNS
)

INTEGER_FEATURE_COLUMNS = _ordered_union(
    _prefixed("hist_", BATTING_SOURCE_COUNTS),
    _prefixed("hist_", PITCHING_SOURCE_COUNTS),
    tuple(
        column
        for window in ROLLING_WINDOWS
        for column in _ordered_union(
            _prefixed(f"last_{window}g_", BATTING_SOURCE_COUNTS),
            _prefixed(f"last_{window}g_", PITCHING_SOURCE_COUNTS),
        )
    ),
)
FLOAT_FEATURE_COLUMNS = _ordered_union(
    _prefixed("hist_", BATTING_RATE_NAMES),
    tuple(
        column
        for window in ROLLING_WINDOWS
        for column in _prefixed(f"last_{window}g_", BATTING_RATE_NAMES)
    ),
)

PROHIBITED_FEATURE_NAMES = frozenset(
    {
        "hist_avg_release_speed_kmh",
        "hist_era",
        "hist_earned_runs",
        "hist_runs_allowed",
        "hist_rbi",
        "hist_runs",
        "hist_batter_runs",
    }
)


class PlayerFeatureBuildError(RuntimeError):
    """Player Pregame Historical Feature 계약 위반으로 빌드를 중단해야 할 때 사용한다."""


def _require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    source_name: str,
) -> None:
    """필수 컬럼이 모두 존재하는지 검증한다."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise PlayerFeatureBuildError(
            f"{source_name}에 필수 컬럼이 없습니다: {missing}"
        )


def _normalize_string_id(
    series: pd.Series,
    *,
    source_name: str,
    column: str,
) -> pd.Series:
    """숫자처럼 보이는 ID도 문자열 의미를 잃지 않도록 문자열 입력만 허용한다."""
    non_null = series.loc[series.notna()]
    invalid_type = non_null.map(lambda value: not isinstance(value, str))
    if invalid_type.any():
        examples = non_null.loc[invalid_type].head(5).tolist()
        raise PlayerFeatureBuildError(
            f"{source_name}.{column}은 문자열 ID여야 합니다: {examples}"
        )

    result = series.astype("string")
    invalid_value = result.isna() | result.str.strip().eq("")
    if invalid_value.any():
        raise PlayerFeatureBuildError(
            f"{source_name}.{column}에 null 또는 빈 문자열이 있습니다."
        )
    return result.str.strip()


def _normalize_date(
    series: pd.Series,
    *,
    source_name: str,
) -> pd.Series:
    """날짜를 시간대 없는 자정 `datetime64[us]`로 정규화한다."""
    try:
        parsed = pd.to_datetime(series, errors="raise")
    except (TypeError, ValueError) as exc:
        raise PlayerFeatureBuildError(
            f"{source_name} 날짜를 해석할 수 없습니다."
        ) from exc

    if parsed.isna().any():
        raise PlayerFeatureBuildError(
            f"{source_name} 날짜에 결측값이 있습니다."
        )

    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_localize(None)

    return parsed.dt.normalize().astype("datetime64[us]")


def _coerce_integer(
    series: pd.Series,
    *,
    source_name: str,
    column: str,
    allow_null: bool = False,
) -> pd.Series:
    """정수 의미 컬럼을 손실 없이 pandas nullable `Int64`로 변환한다."""
    try:
        numeric = pd.to_numeric(series, errors="raise")
    except (TypeError, ValueError) as exc:
        raise PlayerFeatureBuildError(
            f"{source_name}.{column}을 숫자로 변환할 수 없습니다."
        ) from exc

    if not allow_null and numeric.isna().any():
        raise PlayerFeatureBuildError(
            f"{source_name}.{column}에 결측값이 있습니다."
        )

    non_integer = numeric.notna() & numeric.mod(1).ne(0)
    if non_integer.any():
        raise PlayerFeatureBuildError(
            f"{source_name}.{column}에 정수가 아닌 값이 있습니다."
        )

    result = numeric.astype("Int64")
    if result.dropna().lt(0).any():
        raise PlayerFeatureBuildError(
            f"{source_name}.{column}에 음수가 있습니다."
        )
    return result


def _validate_source_season(
    frame: pd.DataFrame,
    *,
    source_name: str,
) -> None:
    """Calendar-year season 계약과 `game_date` 연도가 일치하는지 확인한다."""
    expected = frame["game_date"].dt.year.astype("Int64")
    if frame["season"].ne(expected).any():
        raise PlayerFeatureBuildError(
            f"{source_name}.season이 game_date 연도와 일치하지 않습니다."
        )


def normalize_requests(requests: pd.DataFrame) -> pd.DataFrame:
    """중복 없는 `(player_id, role, prediction_date)` 요청으로 정규화한다."""
    _require_columns(
        requests,
        REQUEST_KEY,
        source_name="requests",
    )
    result = requests.loc[:, REQUEST_KEY].copy()
    if result.empty:
        result["player_id"] = result["player_id"].astype("string")
        result["role"] = result["role"].astype("string")
        result["prediction_date"] = pd.Series(
            pd.array([], dtype="datetime64[us]"),
            index=result.index,
        )
        return result.reset_index(drop=True)

    result["player_id"] = _normalize_string_id(
        result["player_id"],
        source_name="requests",
        column="player_id",
    )
    result["role"] = _normalize_string_id(
        result["role"],
        source_name="requests",
        column="role",
    )
    invalid_roles = sorted(
        result.loc[~result["role"].isin(VALID_ROLES), "role"].unique().tolist()
    )
    if invalid_roles:
        raise PlayerFeatureBuildError(
            f"지원하지 않는 role이 있습니다: {invalid_roles}; 허용값={VALID_ROLES}"
        )
    result["prediction_date"] = _normalize_date(
        result["prediction_date"],
        source_name="requests.prediction_date",
    )

    duplicate_count = int(
        result.duplicated(subset=list(REQUEST_KEY), keep=False).sum()
    )
    if duplicate_count:
        raise PlayerFeatureBuildError(
            "requests의 (player_id, role, prediction_date) 중복 Row가 "
            f"{duplicate_count}건 있습니다."
        )

    return (
        result.sort_values(
            ["prediction_date", "role", "player_id"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def normalize_batting_source(frame: pd.DataFrame) -> pd.DataFrame:
    """Canonical `player_game_batting`을 Historical Source 계약으로 검증한다."""
    _require_columns(
        frame,
        BATTING_REQUIRED_SOURCE_COLUMNS,
        source_name="player_game_batting",
    )
    result = frame.loc[:, BATTING_REQUIRED_SOURCE_COLUMNS].copy()
    result["game_pk"] = _normalize_string_id(
        result["game_pk"],
        source_name="player_game_batting",
        column="game_pk",
    )
    result["batter"] = _normalize_string_id(
        result["batter"],
        source_name="player_game_batting",
        column="batter",
    )
    result = result.rename(columns={"batter": "player_id"})
    result["game_date"] = _normalize_date(
        result["game_date"],
        source_name="player_game_batting.game_date",
    )
    result["season"] = _coerce_integer(
        result["season"],
        source_name="player_game_batting",
        column="season",
    )
    for column in BATTING_SOURCE_COUNTS:
        result[column] = _coerce_integer(
            result[column],
            source_name="player_game_batting",
            column=column,
        )

    duplicate_count = int(
        result.duplicated(subset=["game_pk", "player_id"], keep=False).sum()
    )
    if duplicate_count:
        raise PlayerFeatureBuildError(
            "player_game_batting (game_pk, batter) 중복 Row가 "
            f"{duplicate_count}건 있습니다."
        )

    _validate_source_season(result, source_name="player_game_batting")

    expected_h = (
        result["single"]
        + result["double"]
        + result["triple"]
        + result["hr"]
    )
    if result["h"].ne(expected_h).any():
        raise PlayerFeatureBuildError(
            "player_game_batting.h가 1B+2B+3B+HR 합과 일치하지 않습니다."
        )

    expected_ab = (
        result["pa"]
        - result["bb"]
        - result["hbp"]
        - result["sh"]
        - result["sf"]
        - result["catcher_interference"]
    )
    if result["ab"].ne(expected_ab).any():
        raise PlayerFeatureBuildError(
            "player_game_batting.ab가 Catalog 산식과 일치하지 않습니다."
        )

    expected_tb = (
        result["single"]
        + 2 * result["double"]
        + 3 * result["triple"]
        + 4 * result["hr"]
    )
    if result["tb"].ne(expected_tb).any():
        raise PlayerFeatureBuildError(
            "player_game_batting.tb가 Catalog 산식과 일치하지 않습니다."
        )

    return (
        result.sort_values(
            ["player_id", "season", "game_date", "game_pk"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def normalize_pitching_source(frame: pd.DataFrame) -> pd.DataFrame:
    """Canonical `player_game_pitching`을 nullable outs 계약까지 검증한다."""
    _require_columns(
        frame,
        PITCHING_REQUIRED_SOURCE_COLUMNS,
        source_name="player_game_pitching",
    )
    result = frame.loc[:, PITCHING_REQUIRED_SOURCE_COLUMNS].copy()
    result["game_pk"] = _normalize_string_id(
        result["game_pk"],
        source_name="player_game_pitching",
        column="game_pk",
    )
    result["pitcher"] = _normalize_string_id(
        result["pitcher"],
        source_name="player_game_pitching",
        column="pitcher",
    )
    result = result.rename(columns={"pitcher": "player_id"})
    result["game_date"] = _normalize_date(
        result["game_date"],
        source_name="player_game_pitching.game_date",
    )
    result["season"] = _coerce_integer(
        result["season"],
        source_name="player_game_pitching",
        column="season",
    )
    for column in PITCHING_SOURCE_COUNTS:
        result[column] = _coerce_integer(
            result[column],
            source_name="player_game_pitching",
            column=column,
            allow_null=column == "outs_recorded",
        )

    duplicate_count = int(
        result.duplicated(subset=["game_pk", "player_id"], keep=False).sum()
    )
    if duplicate_count:
        raise PlayerFeatureBuildError(
            "player_game_pitching (game_pk, pitcher) 중복 Row가 "
            f"{duplicate_count}건 있습니다."
        )

    _validate_source_season(result, source_name="player_game_pitching")

    expected_hits = (
        result["single_allowed"]
        + result["double_allowed"]
        + result["triple_allowed"]
        + result["hr_allowed"]
    )
    if result["hits_allowed"].ne(expected_hits).any():
        raise PlayerFeatureBuildError(
            "player_game_pitching.hits_allowed가 안타 유형 합과 일치하지 않습니다."
        )

    if result["pitch_rows"].lt(result["pitches"]).any():
        raise PlayerFeatureBuildError(
            "player_game_pitching.pitch_rows는 pitches보다 작을 수 없습니다."
        )

    expected_pitches = (
        result["ball_pitch_count"]
        + result["strike_pitch_count"]
        + result["in_play_pitch_count"]
    )
    if result["pitches"].ne(expected_pitches).any():
        raise PlayerFeatureBuildError(
            "player_game_pitching.pitches가 B/S/X 실제 Pitch 수 합과 일치하지 않습니다."
        )

    return (
        result.sort_values(
            ["player_id", "season", "game_date", "game_pk"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def _validate_windows(windows: Sequence[int]) -> tuple[int, ...]:
    """Rolling window를 양의 정수이며 중복 없는 값으로 검증한다."""
    normalized = tuple(int(value) for value in windows)
    if not normalized:
        raise PlayerFeatureBuildError("Rolling window가 비어 있습니다.")
    if any(value <= 0 for value in normalized):
        raise PlayerFeatureBuildError(
            f"Rolling window는 양의 정수여야 합니다: {normalized}"
        )
    if len(normalized) != len(set(normalized)):
        raise PlayerFeatureBuildError(
            f"Rolling window에 중복이 있습니다: {normalized}"
        )
    if normalized != ROLLING_WINDOWS:
        raise PlayerFeatureBuildError(
            f"feature_catalog_v1의 고정 Rolling window는 {ROLLING_WINDOWS}입니다."
        )
    return normalized


def _safe_rate(numerator: int, denominator: int) -> object:
    """0 denominator는 null로 유지하고 유효한 Rate는 소수점 3자리로 반올림한다."""
    if denominator == 0:
        return pd.NA
    return round(
        float(numerator) / float(denominator),
        RATE_DECIMAL_PLACES,
    )


def _batting_aggregate(history: pd.DataFrame, *, prefix: str) -> dict[str, object]:
    """타격 Count를 합산하고 AVG/OBP/SLG/OPS를 합산 Count에서 다시 계산한다."""
    counts = {
        column: int(history[column].sum()) if not history.empty else 0
        for column in BATTING_SOURCE_COUNTS
    }
    avg = _safe_rate(counts["h"], counts["ab"])
    obp_denominator = (
        counts["ab"]
        + counts["bb"]
        + counts["hbp"]
        + counts["sf"]
    )
    obp = _safe_rate(
        counts["h"] + counts["bb"] + counts["hbp"],
        obp_denominator,
    )
    slg = _safe_rate(counts["tb"], counts["ab"])
    ops = (
        pd.NA
        if pd.isna(obp) or pd.isna(slg)
        else round(
            float(obp) + float(slg),
            RATE_DECIMAL_PLACES,
        )
    )

    result: dict[str, object] = {
        f"{prefix}{column}": value
        for column, value in counts.items()
    }
    result.update(
        {
            f"{prefix}avg": avg,
            f"{prefix}obp": obp,
            f"{prefix}slg": slg,
            f"{prefix}ops": ops,
        }
    )
    return result


def _pitching_aggregate(history: pd.DataFrame, *, prefix: str) -> dict[str, object]:
    """투구 Count를 합산하되 불확실 outs가 하나라도 있으면 합계도 null로 전파한다."""
    result: dict[str, object] = {}
    for column in PITCHING_SOURCE_COUNTS:
        feature_name = f"{prefix}{column}"
        if history.empty:
            result[feature_name] = 0
            continue
        if column == "outs_recorded":
            result[feature_name] = (
                pd.NA
                if history[column].isna().any()
                else int(history[column].sum())
            )
            continue
        result[feature_name] = int(history[column].sum())
    return result


def _select_rolling_history(
    history: pd.DataFrame,
    *,
    window: int,
) -> pd.DataFrame:
    """최근 N경기 경계 날짜의 Player Game을 같은 날짜 묶음 전체로 포함한다."""
    if history.empty:
        return history.copy()

    ordered = history.sort_values(
        ["game_date", "game_pk"],
        kind="mergesort",
    )
    date_counts = ordered.groupby(
        "game_date",
        sort=True,
        dropna=False,
    ).size()

    selected_dates: list[pd.Timestamp] = []
    selected_count = 0
    for game_date, count in reversed(list(date_counts.items())):
        selected_dates.append(game_date)
        selected_count += int(count)
        if selected_count >= window:
            break

    return ordered.loc[ordered["game_date"].isin(selected_dates)].copy()


def _group_source(
    source: pd.DataFrame,
) -> dict[tuple[str, int], pd.DataFrame]:
    """요청별 전체 Source 재필터링을 피하도록 선수·시즌 단위 인덱스를 만든다."""
    grouped: dict[tuple[str, int], pd.DataFrame] = {}
    for (player_id, season), group in source.groupby(
        ["player_id", "season"],
        sort=False,
        dropna=False,
    ):
        grouped[(str(player_id), int(season))] = group.reset_index(drop=True)
    return grouped


def _empty_history_like(source: pd.DataFrame) -> pd.DataFrame:
    """Source와 같은 Schema를 가진 빈 History를 반환한다."""
    return source.iloc[0:0].copy()


def _build_one_request(
    request: pd.Series,
    *,
    batting_groups: Mapping[tuple[str, int], pd.DataFrame],
    pitching_groups: Mapping[tuple[str, int], pd.DataFrame],
    empty_batting: pd.DataFrame,
    empty_pitching: pd.DataFrame,
    windows: Sequence[int],
) -> dict[str, object]:
    """유효 요청 하나에 대해 strict cutoff Historical Feature 한 행을 계산한다."""
    player_id = str(request["player_id"])
    role = str(request["role"])
    prediction_date = pd.Timestamp(request["prediction_date"])
    season = int(prediction_date.year)

    if role == "batting":
        season_source = batting_groups.get(
            (player_id, season),
            empty_batting,
        )
        aggregate = _batting_aggregate
    else:
        season_source = pitching_groups.get(
            (player_id, season),
            empty_pitching,
        )
        aggregate = _pitching_aggregate

    # 날짜 단위 availability만 검증되었으므로 당일 Player Game은 전부 제외한다.
    history = season_source.loc[
        season_source["game_date"].lt(prediction_date)
    ].copy()
    history = history.sort_values(
        ["game_date", "game_pk"],
        kind="mergesort",
    ).reset_index(drop=True)

    history_count = int(len(history))
    max_source_date = history["game_date"].max() if history_count else pd.NaT
    days_since_last = (
        int((prediction_date - max_source_date).days)
        if history_count
        else pd.NA
    )

    row: dict[str, object] = {
        "player_id": player_id,
        "role": role,
        "prediction_date": prediction_date,
        "max_source_game_date": max_source_date,
        "history_game_count": history_count,
        "has_history": history_count > 0,
        "days_since_last_observed_appearance": days_since_last,
        "feature_version": FEATURE_VERSION,
    }
    row.update({column: pd.NA for column in FEATURE_COLUMNS if column not in row})
    if role == "pitching":
        row["history_uncertain_outs_game_count"] = int(
            history["outs_recorded"].isna().sum()
        )
    else:
        row["history_uncertain_outs_game_count"] = pd.NA
    row.update(aggregate(history, prefix="hist_"))

    for window in windows:
        rolling = _select_rolling_history(history, window=window)
        row[f"last_{window}g_games"] = int(len(rolling))
        row[f"last_{window}g_uncertain_outs_game_count"] = (
            int(rolling["outs_recorded"].isna().sum())
            if role == "pitching"
            else pd.NA
        )
        row.update(aggregate(rolling, prefix=f"last_{window}g_"))

    return row


def normalize_feature_output(frame: pd.DataFrame) -> pd.DataFrame:
    """Output 컬럼 순서·nullable dtype·결정적 정렬을 고정한다."""
    missing = [column for column in OUTPUT_COLUMNS if column not in frame.columns]
    if missing:
        raise PlayerFeatureBuildError(
            f"Player Feature Output Schema에 필요한 컬럼이 없습니다: {missing}"
        )

    result = frame.loc[:, OUTPUT_COLUMNS].copy()
    result["player_id"] = result["player_id"].astype("string")
    result["role"] = result["role"].astype("string")
    result["feature_version"] = result["feature_version"].astype("string")

    for column in (
        "history_game_count",
        "days_since_last_observed_appearance",
    ) + ROLLING_AUDIT_COLUMNS + OUTS_QUALITY_AUDIT_COLUMNS + INTEGER_FEATURE_COLUMNS:
        result[column] = pd.array(result[column], dtype="Int64")

    for column in FLOAT_FEATURE_COLUMNS:
        result[column] = pd.array(result[column], dtype="Float64")

    result["has_history"] = pd.array(result["has_history"], dtype="boolean")
    result["prediction_date"] = pd.to_datetime(
        result["prediction_date"],
        errors="raise",
    ).astype("datetime64[us]")
    result["max_source_game_date"] = pd.to_datetime(
        result["max_source_game_date"],
        errors="coerce",
    ).astype("datetime64[us]")

    return (
        result.sort_values(
            ["prediction_date", "role", "player_id"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def _empty_output() -> pd.DataFrame:
    """빈 Request도 고정 Output Schema와 dtype으로 보존한다."""
    seed = pd.DataFrame({column: pd.Series(dtype="object") for column in OUTPUT_COLUMNS})
    return normalize_feature_output(seed)


def validate_feature_output(
    features: pd.DataFrame,
    requests: pd.DataFrame,
    *,
    rolling_windows: Sequence[int] = ROLLING_WINDOWS,
) -> None:
    """Request 보존, cutoff, cold start, role별 Feature 계약을 검증한다."""
    windows = _validate_windows(rolling_windows)
    normalized_requests = normalize_requests(requests)

    if tuple(features.columns) != OUTPUT_COLUMNS:
        raise PlayerFeatureBuildError(
            "Player Feature Output Column 순서가 계약과 일치하지 않습니다."
        )
    if len(features) != len(normalized_requests):
        raise PlayerFeatureBuildError(
            "Player Feature Row 수가 유효 Request Row 수와 일치하지 않습니다."
        )

    duplicate_count = int(
        features.duplicated(subset=list(REQUEST_KEY), keep=False).sum()
    )
    if duplicate_count:
        raise PlayerFeatureBuildError(
            f"Player Feature Request Key 중복 Row가 {duplicate_count}건 있습니다."
        )

    actual_keys = features.loc[:, REQUEST_KEY].sort_values(
        list(REQUEST_KEY),
        kind="mergesort",
    ).reset_index(drop=True)
    expected_keys = normalized_requests.loc[:, REQUEST_KEY].sort_values(
        list(REQUEST_KEY),
        kind="mergesort",
    ).reset_index(drop=True)
    if not actual_keys.equals(expected_keys):
        raise PlayerFeatureBuildError(
            "Player Feature Output이 Request Key를 정확히 보존하지 않습니다."
        )

    if not features["role"].isin(VALID_ROLES).all():
        raise PlayerFeatureBuildError("Output에 지원하지 않는 role이 있습니다.")
    if features["feature_version"].ne(FEATURE_VERSION).any():
        raise PlayerFeatureBuildError("Output feature_version이 고정 계약과 다릅니다.")

    has_history = features["has_history"].astype("boolean")
    if has_history.ne(features["history_game_count"].gt(0)).any():
        raise PlayerFeatureBuildError(
            "has_history와 history_game_count 의미가 일치하지 않습니다."
        )

    invalid_with_history = has_history & (
        features["max_source_game_date"].isna()
        | features["max_source_game_date"].ge(features["prediction_date"])
    )
    invalid_without_history = (
        ~has_history
        & features["max_source_game_date"].notna()
    )
    if invalid_with_history.any() or invalid_without_history.any():
        raise PlayerFeatureBuildError(
            "max_source_game_date가 strict historical cutoff와 일치하지 않습니다."
        )

    expected_days = (
        features["prediction_date"] - features["max_source_game_date"]
    ).dt.days.astype("Int64")
    if not features.loc[has_history, "days_since_last_observed_appearance"].equals(
        expected_days.loc[has_history]
    ):
        raise PlayerFeatureBuildError(
            "days_since_last_observed_appearance가 최근 관측일과 일치하지 않습니다."
        )
    if features.loc[~has_history, "days_since_last_observed_appearance"].notna().any():
        raise PlayerFeatureBuildError(
            "History가 없는 요청의 activity context는 null이어야 합니다."
        )

    for window in windows:
        count_column = f"last_{window}g_games"
        if features[count_column].gt(features["history_game_count"]).any():
            raise PlayerFeatureBuildError(
                f"{count_column}가 전체 history_game_count보다 클 수 없습니다."
            )

    batting_mask = features["role"].eq("batting")
    pitching_mask = features["role"].eq("pitching")
    cold_batting = batting_mask & ~has_history
    cold_pitching = pitching_mask & ~has_history

    if features.loc[batting_mask, list(OUTS_QUALITY_AUDIT_COLUMNS)].notna().any().any():
        raise PlayerFeatureBuildError(
            "Batting Row의 Pitching outs 품질 Audit은 null이어야 합니다."
        )
    pitching_uncertain = features.loc[
        pitching_mask, "history_uncertain_outs_game_count"
    ]
    pitching_hist_outs = features.loc[pitching_mask, "hist_outs_recorded"]
    expected_hist_outs_null = pitching_uncertain.gt(0)
    if pitching_hist_outs.isna().ne(expected_hist_outs_null).any():
        raise PlayerFeatureBuildError(
            "hist_outs_recorded null과 불확실 Player Game Audit가 일치하지 않습니다."
        )
    for window in windows:
        uncertain_column = f"last_{window}g_uncertain_outs_game_count"
        outs_column = f"last_{window}g_outs_recorded"
        expected_window_outs_null = features.loc[
            pitching_mask, uncertain_column
        ].gt(0)
        if features.loc[pitching_mask, outs_column].isna().ne(
            expected_window_outs_null
        ).any():
            raise PlayerFeatureBuildError(
                f"{outs_column} null과 Window 불확실 outs Audit가 일치하지 않습니다."
            )

    batting_count_columns = _ordered_union(
        _prefixed("hist_", BATTING_SOURCE_COUNTS),
        tuple(
            column
            for window in windows
            for column in _prefixed(f"last_{window}g_", BATTING_SOURCE_COUNTS)
        ),
    )
    pitching_count_columns = _ordered_union(
        _prefixed("hist_", PITCHING_SOURCE_COUNTS),
        tuple(
            column
            for window in windows
            for column in _prefixed(f"last_{window}g_", PITCHING_SOURCE_COUNTS)
        ),
    )
    batting_rate_columns = _ordered_union(
        _prefixed("hist_", BATTING_RATE_NAMES),
        tuple(
            column
            for window in windows
            for column in _prefixed(f"last_{window}g_", BATTING_RATE_NAMES)
        ),
    )

    for column in batting_rate_columns:
        values = features.loc[batting_mask, column].dropna().astype("Float64")
        if not values.eq(values.round(RATE_DECIMAL_PLACES)).all():
            raise PlayerFeatureBuildError(
                f"{column}은 소수점 {RATE_DECIMAL_PLACES}자리 Rate 계약을 따라야 합니다."
            )

    if features.loc[cold_batting, list(batting_count_columns)].fillna(-1).ne(0).any().any():
        raise PlayerFeatureBuildError(
            "Batting Cold Start의 Count Feature는 0이어야 합니다."
        )
    if features.loc[cold_pitching, list(pitching_count_columns)].fillna(-1).ne(0).any().any():
        raise PlayerFeatureBuildError(
            "Pitching Cold Start의 Count Feature는 0이어야 합니다."
        )
    if features.loc[cold_batting, list(batting_rate_columns)].notna().any().any():
        raise PlayerFeatureBuildError(
            "Batting Cold Start의 Rate Feature는 null이어야 합니다."
        )

    batting_only = sorted(
        set(ROLE_FEATURE_COLUMNS["batting"])
        - set(ROLE_FEATURE_COLUMNS["pitching"])
    )
    pitching_only = sorted(
        set(ROLE_FEATURE_COLUMNS["pitching"])
        - set(ROLE_FEATURE_COLUMNS["batting"])
    )
    if features.loc[pitching_mask, batting_only].notna().any().any():
        raise PlayerFeatureBuildError(
            "Pitching Row에 Batting 전용 Feature 값이 존재합니다."
        )
    if features.loc[batting_mask, pitching_only].notna().any().any():
        raise PlayerFeatureBuildError(
            "Batting Row에 Pitching 전용 Feature 값이 존재합니다."
        )

    prohibited = PROHIBITED_FEATURE_NAMES.intersection(features.columns)
    if prohibited:
        raise PlayerFeatureBuildError(
            f"금지된 Historical Feature가 Output에 포함되었습니다: {sorted(prohibited)}"
        )


def build_player_pregame_features(
    requests: pd.DataFrame,
    batting_source: pd.DataFrame,
    pitching_source: pd.DataFrame,
    *,
    rolling_windows: Sequence[int] = ROLLING_WINDOWS,
) -> pd.DataFrame:
    """
    요청된 선수·역할·날짜만 보존해 전일까지의 Player Historical X를 계산한다.

    요청 모집단은 외부 호출자가 결정하며, 이 함수는 미래 출장이나 Player Master를
    조회해 요청을 자동 생성하지 않는다. 시즌 이력은 `prediction_date.year` 기준으로
    초기화하고 같은 날짜 Player Game은 어떤 요청에도 Historical Source로 사용하지 않는다.
    """
    windows = _validate_windows(rolling_windows)
    normalized_requests = normalize_requests(requests)
    normalized_batting = normalize_batting_source(batting_source)
    normalized_pitching = normalize_pitching_source(pitching_source)

    if normalized_requests.empty:
        return _empty_output()

    batting_groups = _group_source(normalized_batting)
    pitching_groups = _group_source(normalized_pitching)
    empty_batting = _empty_history_like(normalized_batting)
    empty_pitching = _empty_history_like(normalized_pitching)

    rows = [
        _build_one_request(
            request,
            batting_groups=batting_groups,
            pitching_groups=pitching_groups,
            empty_batting=empty_batting,
            empty_pitching=empty_pitching,
            windows=windows,
        )
        for _, request in normalized_requests.iterrows()
    ]

    result = normalize_feature_output(pd.DataFrame(rows))
    validate_feature_output(
        result,
        normalized_requests,
        rolling_windows=windows,
    )
    return result
