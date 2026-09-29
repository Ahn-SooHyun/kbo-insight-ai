from __future__ import annotations

import argparse
import hashlib
import json
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_GAMES_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "hf_kbo_pbp"
    / "derived"
    / "games.parquet"
)
DEFAULT_TEAM_GAMES_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "hf_kbo_pbp"
    / "derived"
    / "team_games.parquet"
)
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hf_kbo_pbp"
    / "features"
    / "team_pregame_features.parquet"
)

FEATURE_VERSION = "team_pregame_v1"
SOURCE_CONTRACT_VERSION = "feature_catalog_v1"
ROLLING_WINDOWS = (5, 10, 20)

GAME_KEY = ("game_pk",)
TEAM_GAME_KEY = ("game_pk", "team")

REQUIRED_GAMES_COLUMNS = (
    "game_pk",
    "game_date",
    "season",
    "home_team",
    "away_team",
    "final_home_score",
    "final_away_score",
    "home_win",
    "away_win",
    "is_tie",
)

REQUIRED_TEAM_GAMES_COLUMNS = (
    "game_pk",
    "game_date",
    "season",
    "team",
    "opponent",
    "is_home",
    "runs_for",
    "runs_against",
    "run_diff",
    "win",
    "loss",
    "tie",
    "plate_appearances",
    "opponent_plate_appearances",
)

KEY_COLUMNS = (
    "game_pk",
    "team",
)

AUDIT_COLUMNS = (
    "opponent",
    "prediction_date",
    "season",
    "is_home",
    "max_source_game_date",
    "history_game_count",
    "has_history",
    "feature_version",
)

EXPANDING_FEATURE_COLUMNS = (
    "hist_games",
    "hist_wins",
    "hist_losses",
    "hist_ties",
    "hist_runs_for",
    "hist_runs_against",
    "hist_run_diff",
    "hist_win_pct",
    "hist_runs_for_per_game",
    "hist_runs_against_per_game",
    "hist_run_diff_per_game",
    "days_since_last_observed_game",
)

ROLLING_SUFFIXES = (
    "games",
    "wins",
    "losses",
    "ties",
    "runs_for",
    "runs_against",
    "run_diff",
    "win_pct",
    "runs_for_per_game",
    "runs_against_per_game",
    "run_diff_per_game",
)


def _rolling_columns(
    windows: Sequence[int],
) -> tuple[str, ...]:
    """Rolling window 설정으로 고정 Feature 이름을 만든다."""
    return tuple(
        f"last_{window}g_{suffix}"
        for window in windows
        for suffix in ROLLING_SUFFIXES
    )


ROLLING_FEATURE_COLUMNS = _rolling_columns(ROLLING_WINDOWS)
X_COLUMNS = EXPANDING_FEATURE_COLUMNS + ROLLING_FEATURE_COLUMNS
OUTPUT_COLUMNS = KEY_COLUMNS + AUDIT_COLUMNS + X_COLUMNS

STRING_OUTPUT_COLUMNS = (
    "game_pk",
    "team",
    "opponent",
    "feature_version",
)
INTEGER_OUTPUT_COLUMNS = (
    "season",
    "history_game_count",
    "hist_games",
    "hist_wins",
    "hist_losses",
    "hist_ties",
    "hist_runs_for",
    "hist_runs_against",
    "hist_run_diff",
    "days_since_last_observed_game",
) + tuple(
    column
    for column in ROLLING_FEATURE_COLUMNS
    if not column.endswith(
        (
            "win_pct",
            "per_game",
        )
    )
)
FLOAT_OUTPUT_COLUMNS = (
    "hist_win_pct",
    "hist_runs_for_per_game",
    "hist_runs_against_per_game",
    "hist_run_diff_per_game",
) + tuple(
    column
    for column in ROLLING_FEATURE_COLUMNS
    if column.endswith(
        (
            "win_pct",
            "per_game",
        )
    )
)
BOOLEAN_OUTPUT_COLUMNS = (
    "is_home",
    "has_history",
)
DATETIME_OUTPUT_COLUMNS = (
    "prediction_date",
    "max_source_game_date",
)


class TeamFeatureBuildError(RuntimeError):
    """Team Pregame Feature 생성을 중단해야 하는 오류를 나타낸다."""


def parse_args() -> argparse.Namespace:
    """Team Pregame Feature Builder의 CLI 인자를 파싱한다."""
    parser = argparse.ArgumentParser(
        description=(
            "Canonical games/team_games에서 strict historical cutoff를 적용한 "
            "Team Pregame Feature를 생성합니다."
        )
    )
    parser.add_argument(
        "--games-path",
        type=Path,
        default=DEFAULT_GAMES_PATH,
        help=f"Canonical games.parquet 경로입니다. 기본값: {DEFAULT_GAMES_PATH}",
    )
    parser.add_argument(
        "--team-games-path",
        type=Path,
        default=DEFAULT_TEAM_GAMES_PATH,
        help=(
            "Canonical team_games.parquet 경로입니다. "
            f"기본값: {DEFAULT_TEAM_GAMES_PATH}"
        ),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Feature Parquet 경로입니다. 기본값: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help=(
            "Manifest JSON 경로입니다. 생략하면 output과 같은 디렉터리에 "
            "team_pregame_features.manifest.json을 생성합니다."
        ),
    )
    return parser.parse_args()


def configure_logging() -> None:
    """독립 실행 시 사용할 logging 형식을 설정한다."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def _require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    source_name: str,
) -> None:
    """필수 컬럼 존재 여부를 확인한다."""
    missing = [
        column
        for column in columns
        if column not in frame.columns
    ]
    if missing:
        raise TeamFeatureBuildError(
            f"{source_name}에 필수 컬럼이 없습니다: {missing}"
        )


def _normalize_date_column(
    series: pd.Series,
    *,
    source_name: str,
) -> pd.Series:
    """날짜 컬럼을 날짜 단위의 timezone-naive datetime64[us]로 정규화한다."""
    try:
        parsed = pd.to_datetime(
            series,
            errors="raise",
        )
    except (TypeError, ValueError) as exc:
        raise TeamFeatureBuildError(
            f"{source_name} 날짜 컬럼을 해석할 수 없습니다."
        ) from exc

    if parsed.isna().any():
        raise TeamFeatureBuildError(
            f"{source_name} 날짜 컬럼에 결측값이 있습니다."
        )

    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_convert(None)

    return (
        parsed
        .dt.normalize()
        .astype("datetime64[us]")
    )


def _coerce_non_null_integer(
    series: pd.Series,
    *,
    source_name: str,
    column: str,
) -> pd.Series:
    """정수 의미 컬럼을 손실 없이 nullable Int64로 정규화한다."""
    try:
        numeric = pd.to_numeric(
            series,
            errors="raise",
        )
    except (TypeError, ValueError) as exc:
        raise TeamFeatureBuildError(
            f"{source_name}.{column}을 숫자로 변환할 수 없습니다."
        ) from exc

    if numeric.isna().any():
        raise TeamFeatureBuildError(
            f"{source_name}.{column}에 결측값이 있습니다."
        )

    if numeric.mod(1).ne(0).any():
        raise TeamFeatureBuildError(
            f"{source_name}.{column}에 정수가 아닌 값이 있습니다."
        )

    return numeric.astype("Int64")


def _coerce_non_null_boolean(
    series: pd.Series,
    *,
    source_name: str,
    column: str,
) -> pd.Series:
    """Boolean 의미 컬럼을 nullable boolean으로 정규화한다."""
    try:
        result = series.astype("boolean")
    except (TypeError, ValueError) as exc:
        raise TeamFeatureBuildError(
            f"{source_name}.{column}을 boolean으로 변환할 수 없습니다."
        ) from exc

    if result.isna().any():
        raise TeamFeatureBuildError(
            f"{source_name}.{column}에 결측값이 있습니다."
        )

    return result


def _normalize_string(
    series: pd.Series,
    *,
    source_name: str,
    column: str,
) -> pd.Series:
    """식별자 문자열을 pandas string dtype으로 정규화하고 빈 값을 차단한다."""
    result = series.astype("string")
    invalid = (
        result.isna()
        | result.str.strip().eq("")
    )
    if invalid.any():
        raise TeamFeatureBuildError(
            f"{source_name}.{column}에 null 또는 빈 문자열이 있습니다."
        )
    return result


def _normalize_games(
    games: pd.DataFrame,
) -> pd.DataFrame:
    """games 입력을 #23에서 필요한 최소 Canonical 경계로 검증·정규화한다."""
    if games.empty:
        raise TeamFeatureBuildError(
            "games 입력이 비어 있습니다."
        )

    _require_columns(
        games,
        REQUIRED_GAMES_COLUMNS,
        source_name="games",
    )
    result = games.loc[:, REQUIRED_GAMES_COLUMNS].copy()
    result["game_pk"] = _normalize_string(
        result["game_pk"],
        source_name="games",
        column="game_pk",
    )
    result["home_team"] = _normalize_string(
        result["home_team"],
        source_name="games",
        column="home_team",
    )
    result["away_team"] = _normalize_string(
        result["away_team"],
        source_name="games",
        column="away_team",
    )
    result["game_date"] = _normalize_date_column(
        result["game_date"],
        source_name="games.game_date",
    )
    result["season"] = _coerce_non_null_integer(
        result["season"],
        source_name="games",
        column="season",
    )
    for column in (
        "final_home_score",
        "final_away_score",
    ):
        result[column] = _coerce_non_null_integer(
            result[column],
            source_name="games",
            column=column,
        )
        if result[column].lt(0).any():
            raise TeamFeatureBuildError(
                f"games.{column}에 음수가 있습니다."
            )

    for column in (
        "home_win",
        "away_win",
        "is_tie",
    ):
        result[column] = _coerce_non_null_boolean(
            result[column],
            source_name="games",
            column=column,
        )

    duplicate_count = int(
        result.duplicated(
            subset=list(GAME_KEY),
            keep=False,
        ).sum()
    )
    if duplicate_count:
        raise TeamFeatureBuildError(
            f"games.game_pk 중복 Row가 {duplicate_count}건 있습니다."
        )

    if result["home_team"].eq(result["away_team"]).any():
        raise TeamFeatureBuildError(
            "games에 home_team과 away_team이 같은 Row가 있습니다."
        )

    expected_home_win = result["final_home_score"].gt(
        result["final_away_score"]
    )
    expected_away_win = result["final_away_score"].gt(
        result["final_home_score"]
    )
    expected_tie = result["final_home_score"].eq(
        result["final_away_score"]
    )
    checks = (
        ("home_win", expected_home_win),
        ("away_win", expected_away_win),
        ("is_tie", expected_tie),
    )
    for column, expected in checks:
        if result[column].ne(expected).any():
            raise TeamFeatureBuildError(
                f"games.{column}이 Final Score와 일치하지 않습니다."
            )

    return (
        result
        .sort_values(
            [
                "season",
                "game_date",
                "game_pk",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def _normalize_team_games(
    team_games: pd.DataFrame,
) -> pd.DataFrame:
    """team_games 입력을 Historical Source로 안전하게 사용할 수 있도록 검증한다."""
    if team_games.empty:
        raise TeamFeatureBuildError(
            "team_games 입력이 비어 있습니다."
        )

    _require_columns(
        team_games,
        REQUIRED_TEAM_GAMES_COLUMNS,
        source_name="team_games",
    )
    result = team_games.loc[:, REQUIRED_TEAM_GAMES_COLUMNS].copy()

    for column in (
        "game_pk",
        "team",
        "opponent",
    ):
        result[column] = _normalize_string(
            result[column],
            source_name="team_games",
            column=column,
        )

    result["game_date"] = _normalize_date_column(
        result["game_date"],
        source_name="team_games.game_date",
    )

    for column in (
        "season",
        "runs_for",
        "runs_against",
        "run_diff",
        "plate_appearances",
        "opponent_plate_appearances",
    ):
        result[column] = _coerce_non_null_integer(
            result[column],
            source_name="team_games",
            column=column,
        )

    for column in (
        "runs_for",
        "runs_against",
        "plate_appearances",
        "opponent_plate_appearances",
    ):
        if result[column].lt(0).any():
            raise TeamFeatureBuildError(
                f"team_games.{column}에 음수가 있습니다."
            )

    for column in (
        "is_home",
        "win",
        "loss",
        "tie",
    ):
        result[column] = _coerce_non_null_boolean(
            result[column],
            source_name="team_games",
            column=column,
        )

    duplicate_count = int(
        result.duplicated(
            subset=list(TEAM_GAME_KEY),
            keep=False,
        ).sum()
    )
    if duplicate_count:
        raise TeamFeatureBuildError(
            "team_games (game_pk, team) 중복 Row가 "
            f"{duplicate_count}건 있습니다."
        )

    if result["team"].eq(result["opponent"]).any():
        raise TeamFeatureBuildError(
            "team_games에 team과 opponent가 같은 Row가 있습니다."
        )

    if result["run_diff"].ne(
        result["runs_for"] - result["runs_against"]
    ).any():
        raise TeamFeatureBuildError(
            "team_games.run_diff가 runs_for - runs_against와 일치하지 않습니다."
        )

    outcome_count = (
        result[["win", "loss", "tie"]]
        .astype("Int64")
        .sum(axis=1)
    )
    if outcome_count.ne(1).any():
        raise TeamFeatureBuildError(
            "team_games의 win/loss/tie 중 정확히 하나만 True여야 합니다."
        )

    rows_per_game = result.groupby(
        "game_pk",
        sort=False,
        dropna=False,
    ).size()
    if rows_per_game.ne(2).any():
        invalid = int(rows_per_game.ne(2).sum())
        raise TeamFeatureBuildError(
            "team_games가 정확히 2행이 아닌 경기가 "
            f"{invalid}건 있습니다."
        )

    home_per_game = result.groupby(
        "game_pk",
        sort=False,
        dropna=False,
    )["is_home"].sum()
    if home_per_game.ne(1).any():
        invalid = int(home_per_game.ne(1).sum())
        raise TeamFeatureBuildError(
            "team_games의 Home Row가 정확히 1개가 아닌 경기가 "
            f"{invalid}건 있습니다."
        )

    return (
        result
        .sort_values(
            [
                "season",
                "game_date",
                "game_pk",
                "is_home",
                "team",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def _validate_game_team_relationship(
    games: pd.DataFrame,
    team_games: pd.DataFrame,
) -> None:
    """games와 team_games의 경기당 두 팀 Mirror 계약을 검증한다."""
    if len(team_games) != 2 * len(games):
        raise TeamFeatureBuildError(
            "team_games Row 수가 games의 정확히 2배가 아닙니다: "
            f"games={len(games)}, team_games={len(team_games)}"
        )

    merged = team_games.merge(
        games,
        on="game_pk",
        how="left",
        validate="many_to_one",
        suffixes=("", "_game"),
        indicator=True,
        sort=False,
    )
    if merged["_merge"].ne("both").any():
        raise TeamFeatureBuildError(
            "team_games에 games에 존재하지 않는 game_pk가 있습니다."
        )

    is_home = merged["is_home"].astype("boolean")
    expected_team = merged["home_team"].where(
        is_home,
        merged["away_team"],
    )
    expected_opponent = merged["away_team"].where(
        is_home,
        merged["home_team"],
    )
    expected_runs_for = merged["final_home_score"].where(
        is_home,
        merged["final_away_score"],
    )
    expected_runs_against = merged["final_away_score"].where(
        is_home,
        merged["final_home_score"],
    )
    expected_win = merged["home_win"].where(
        is_home,
        merged["away_win"],
    )
    expected_loss = merged["away_win"].where(
        is_home,
        merged["home_win"],
    )

    checks = (
        ("game_date", merged["game_date"], merged["game_date_game"]),
        ("season", merged["season"], merged["season_game"]),
        ("team", merged["team"], expected_team),
        ("opponent", merged["opponent"], expected_opponent),
        ("runs_for", merged["runs_for"], expected_runs_for),
        ("runs_against", merged["runs_against"], expected_runs_against),
        ("win", merged["win"], expected_win),
        ("loss", merged["loss"], expected_loss),
        ("tie", merged["tie"], merged["is_tie"]),
    )
    for name, actual, expected in checks:
        if actual.ne(expected).fillna(True).any():
            raise TeamFeatureBuildError(
                f"team_games.{name}이 games Mirror 계약과 일치하지 않습니다."
            )


def _validate_windows(
    windows: Sequence[int],
) -> tuple[int, ...]:
    """Rolling window 설정을 양의 정수·중복 없음으로 검증한다."""
    normalized = tuple(
        int(value)
        for value in windows
    )
    if not normalized:
        raise TeamFeatureBuildError(
            "Rolling window가 비어 있습니다."
        )
    if any(value <= 0 for value in normalized):
        raise TeamFeatureBuildError(
            f"Rolling window는 양의 정수여야 합니다: {normalized}"
        )
    if len(set(normalized)) != len(normalized):
        raise TeamFeatureBuildError(
            f"Rolling window에 중복이 있습니다: {normalized}"
        )
    return normalized


def prepare_inputs(
    games: pd.DataFrame,
    team_games: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """#23 Feature 계산 전에 Canonical 입력 경계를 한 번에 검증한다."""
    normalized_games = _normalize_games(games)
    normalized_team_games = _normalize_team_games(team_games)
    _validate_game_team_relationship(
        normalized_games,
        normalized_team_games,
    )
    return normalized_games, normalized_team_games


def _safe_rate(
    numerator: int,
    denominator: int,
) -> object:
    """0 denominator를 임의의 0 rate로 바꾸지 않고 null을 반환한다."""
    if denominator == 0:
        return pd.NA
    return float(numerator) / float(denominator)


def _aggregate_history(
    history: pd.DataFrame,
    *,
    prefix: str,
) -> dict[str, object]:
    """과거 Post-game Fact를 Count 우선 방식으로 집계한다."""
    games = int(len(history))
    wins = int(history["win"].astype("Int64").sum()) if games else 0
    losses = int(history["loss"].astype("Int64").sum()) if games else 0
    ties = int(history["tie"].astype("Int64").sum()) if games else 0
    runs_for = int(history["runs_for"].sum()) if games else 0
    runs_against = int(history["runs_against"].sum()) if games else 0
    run_diff = runs_for - runs_against

    return {
        f"{prefix}games": games,
        f"{prefix}wins": wins,
        f"{prefix}losses": losses,
        f"{prefix}ties": ties,
        f"{prefix}runs_for": runs_for,
        f"{prefix}runs_against": runs_against,
        f"{prefix}run_diff": run_diff,
        f"{prefix}win_pct": _safe_rate(
            wins,
            wins + losses,
        ),
        f"{prefix}runs_for_per_game": _safe_rate(
            runs_for,
            games,
        ),
        f"{prefix}runs_against_per_game": _safe_rate(
            runs_against,
            games,
        ),
        f"{prefix}run_diff_per_game": _safe_rate(
            run_diff,
            games,
        ),
    }


def _select_rolling_history(
    history: pd.DataFrame,
    *,
    window: int,
) -> pd.DataFrame:
    """
    최근 N경기 경계에 같은 날짜 묶음이 걸리면 그 날짜 전체를 포함한다.

    정확한 경기 시각 provenance가 없으므로 같은 날짜를 game_pk 순서로 쪼개지 않는다.
    """
    if history.empty:
        return history.copy()

    ordered = history.sort_values(
        [
            "game_date",
            "game_pk",
        ],
        kind="mergesort",
    )

    selected_dates: list[pd.Timestamp] = []
    selected_count = 0
    date_counts = (
        ordered.groupby(
            "game_date",
            sort=True,
            dropna=False,
        )
        .size()
    )
    for game_date, count in reversed(list(date_counts.items())):
        selected_dates.append(game_date)
        selected_count += int(count)
        if selected_count >= window:
            break

    selected = ordered.loc[
        ordered["game_date"].isin(selected_dates)
    ].copy()
    return selected


def _build_group_features(
    group: pd.DataFrame,
    *,
    windows: Sequence[int],
) -> list[dict[str, object]]:
    """한 `(season, team)`의 날짜별 prefix history로 Pregame Feature를 만든다."""
    rows: list[dict[str, object]] = []
    history = group.iloc[0:0].copy()

    for prediction_date, current_day in group.groupby(
        "game_date",
        sort=True,
        dropna=False,
    ):
        expanding = _aggregate_history(
            history,
            prefix="hist_",
        )
        history_count = int(len(history))
        max_source_date = (
            history["game_date"].max()
            if history_count
            else pd.NaT
        )
        days_since_last = (
            int((prediction_date - max_source_date).days)
            if history_count
            else pd.NA
        )

        rolling_values: dict[str, object] = {}
        for window in windows:
            rolling = _select_rolling_history(
                history,
                window=window,
            )
            rolling_values.update(
                _aggregate_history(
                    rolling,
                    prefix=f"last_{window}g_",
                )
            )

        current_day = current_day.sort_values(
            [
                "game_pk",
                "is_home",
                "team",
            ],
            kind="mergesort",
        )
        for _, current in current_day.iterrows():
            row: dict[str, object] = {
                "game_pk": current["game_pk"],
                "team": current["team"],
                "opponent": current["opponent"],
                "prediction_date": prediction_date,
                "season": current["season"],
                "is_home": current["is_home"],
                "max_source_game_date": max_source_date,
                "history_game_count": history_count,
                "has_history": history_count > 0,
                "feature_version": FEATURE_VERSION,
                "days_since_last_observed_game": days_since_last,
            }
            row.update(expanding)
            row.update(rolling_values)
            rows.append(row)

        # 같은 날짜의 모든 Target Row에 Feature를 계산한 뒤에만
        # 해당 날짜 결과를 이후 날짜의 Historical Source에 추가한다.
        history = pd.concat(
            [
                history,
                current_day,
            ],
            ignore_index=True,
        )

    return rows


def _normalize_output(
    frame: pd.DataFrame,
    *,
    windows: Sequence[int],
) -> pd.DataFrame:
    """Feature Output의 컬럼 순서, dtype, 결정적 정렬을 고정한다."""
    expected_rolling = _rolling_columns(windows)
    expected_x = EXPANDING_FEATURE_COLUMNS + expected_rolling
    expected_columns = KEY_COLUMNS + AUDIT_COLUMNS + expected_x

    missing = [
        column
        for column in expected_columns
        if column not in frame.columns
    ]
    if missing:
        raise TeamFeatureBuildError(
            f"Feature Output Schema에 필요한 컬럼이 없습니다: {missing}"
        )

    result = frame.loc[:, expected_columns].copy()

    for column in STRING_OUTPUT_COLUMNS:
        result[column] = result[column].astype("string")

    integer_columns = (
        "season",
        "history_game_count",
        "hist_games",
        "hist_wins",
        "hist_losses",
        "hist_ties",
        "hist_runs_for",
        "hist_runs_against",
        "hist_run_diff",
        "days_since_last_observed_game",
    ) + tuple(
        column
        for column in expected_rolling
        if not column.endswith(
            (
                "win_pct",
                "per_game",
            )
        )
    )
    float_columns = (
        "hist_win_pct",
        "hist_runs_for_per_game",
        "hist_runs_against_per_game",
        "hist_run_diff_per_game",
    ) + tuple(
        column
        for column in expected_rolling
        if column.endswith(
            (
                "win_pct",
                "per_game",
            )
        )
    )

    for column in integer_columns:
        result[column] = pd.array(
            result[column],
            dtype="Int64",
        )
    for column in float_columns:
        result[column] = pd.array(
            result[column],
            dtype="Float64",
        )
    for column in BOOLEAN_OUTPUT_COLUMNS:
        result[column] = pd.array(
            result[column],
            dtype="boolean",
        )

    result["prediction_date"] = pd.to_datetime(
        result["prediction_date"],
        errors="raise",
    ).astype("datetime64[us]")
    result["max_source_game_date"] = pd.to_datetime(
        result["max_source_game_date"],
        errors="coerce",
    ).astype("datetime64[us]")

    return (
        result
        .sort_values(
            [
                "season",
                "prediction_date",
                "game_pk",
                "is_home",
                "team",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def validate_feature_output(
    features: pd.DataFrame,
    team_games: pd.DataFrame,
    *,
    windows: Sequence[int] = ROLLING_WINDOWS,
) -> None:
    """Pregame Feature의 grain, cutoff, 산식과 입력 Key 보존을 검증한다."""
    windows = _validate_windows(windows)
    expected_columns = KEY_COLUMNS + AUDIT_COLUMNS + (
        EXPANDING_FEATURE_COLUMNS + _rolling_columns(windows)
    )
    if tuple(features.columns) != expected_columns:
        raise TeamFeatureBuildError(
            "Feature Output Column 순서가 계약과 일치하지 않습니다."
        )

    if len(features) != len(team_games):
        raise TeamFeatureBuildError(
            "Feature Row 수가 team_games Row 수와 일치하지 않습니다."
        )

    duplicate_count = int(
        features.duplicated(
            subset=list(KEY_COLUMNS),
            keep=False,
        ).sum()
    )
    if duplicate_count:
        raise TeamFeatureBuildError(
            f"Feature (game_pk, team) 중복 Row가 {duplicate_count}건 있습니다."
        )

    rows_per_game = features.groupby(
        "game_pk",
        sort=False,
        dropna=False,
    ).size()
    if rows_per_game.ne(2).any():
        raise TeamFeatureBuildError(
            "Feature가 정확히 2행이 아닌 game_pk가 있습니다."
        )

    source_context = team_games.loc[
        :,
        [
            "game_pk",
            "team",
            "opponent",
            "game_date",
            "season",
            "is_home",
        ],
    ].rename(
        columns={
            "game_date": "prediction_date_expected",
            "opponent": "opponent_expected",
            "season": "season_expected",
            "is_home": "is_home_expected",
        }
    )
    merged = features.merge(
        source_context,
        on=["game_pk", "team"],
        how="left",
        validate="one_to_one",
        indicator=True,
        sort=False,
    )
    if merged["_merge"].ne("both").any():
        raise TeamFeatureBuildError(
            "Feature Key가 team_games Key와 일치하지 않습니다."
        )

    context_checks = (
        ("opponent", "opponent_expected"),
        ("prediction_date", "prediction_date_expected"),
        ("season", "season_expected"),
        ("is_home", "is_home_expected"),
    )
    for actual_name, expected_name in context_checks:
        if merged[actual_name].ne(merged[expected_name]).fillna(True).any():
            raise TeamFeatureBuildError(
                f"Feature {actual_name}가 team_games Context와 일치하지 않습니다."
            )

    history_count_mismatch = features["history_game_count"].ne(
        features["hist_games"]
    )
    if history_count_mismatch.any():
        raise TeamFeatureBuildError(
            "history_game_count와 hist_games가 일치하지 않습니다."
        )

    has_history_mismatch = features["has_history"].ne(
        features["hist_games"].gt(0)
    )
    if has_history_mismatch.any():
        raise TeamFeatureBuildError(
            "has_history와 hist_games 의미가 일치하지 않습니다."
        )

    source_date = features["max_source_game_date"]
    has_history = features["has_history"].astype("boolean")
    invalid_with_history = (
        has_history
        & (
            source_date.isna()
            | source_date.ge(features["prediction_date"])
        )
    )
    invalid_without_history = (
        ~has_history
        & source_date.notna()
    )
    if invalid_with_history.any() or invalid_without_history.any():
        raise TeamFeatureBuildError(
            "max_source_game_date가 strict historical cutoff와 일치하지 않습니다."
        )

    if features["hist_games"].ne(
        features["hist_wins"]
        + features["hist_losses"]
        + features["hist_ties"]
    ).any():
        raise TeamFeatureBuildError(
            "hist_games가 W/L/T 합과 일치하지 않습니다."
        )

    if features["hist_run_diff"].ne(
        features["hist_runs_for"] - features["hist_runs_against"]
    ).any():
        raise TeamFeatureBuildError(
            "hist_run_diff가 득점-실점과 일치하지 않습니다."
        )

    cold_start = features["hist_games"].eq(0)
    if features.loc[cold_start, "hist_win_pct"].notna().any():
        raise TeamFeatureBuildError(
            "Cold Start의 hist_win_pct는 null이어야 합니다."
        )

    for window in windows:
        count_column = f"last_{window}g_games"
        if features[count_column].gt(features["history_game_count"]).any():
            raise TeamFeatureBuildError(
                f"{count_column}가 전체 history_game_count보다 클 수 없습니다."
            )


def build_team_pregame_features(
    games: pd.DataFrame,
    team_games: pd.DataFrame,
    *,
    rolling_windows: Sequence[int] = ROLLING_WINDOWS,
) -> pd.DataFrame:
    """
    Canonical Post-game Fact에서 strict cutoff를 적용한 Team Pregame Feature를 만든다.

    Season별로 이력을 초기화하고 같은 날짜 결과는 그 날짜의 어떤 Target Row에도
    Source로 사용하지 않는다.
    """
    windows = _validate_windows(rolling_windows)
    _, source = prepare_inputs(
        games,
        team_games,
    )

    rows: list[dict[str, object]] = []
    for _, group in source.groupby(
        ["season", "team"],
        sort=True,
        dropna=False,
    ):
        ordered_group = group.sort_values(
            [
                "game_date",
                "game_pk",
                "is_home",
            ],
            kind="mergesort",
        ).reset_index(drop=True)
        rows.extend(
            _build_group_features(
                ordered_group,
                windows=windows,
            )
        )

    result = _normalize_output(
        pd.DataFrame(rows),
        windows=windows,
    )
    validate_feature_output(
        result,
        source,
        windows=windows,
    )
    return result


def calculate_sha256(
    path: Path,
) -> str:
    """입력 파일 불변성 확인용 SHA256을 계산한다."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as file:
            for chunk in iter(
                lambda: file.read(1024 * 1024),
                b"",
            ):
                digest.update(chunk)
    except OSError as exc:
        raise TeamFeatureBuildError(
            f"SHA256 계산에 실패했습니다: {path}"
        ) from exc
    return digest.hexdigest()


def _json_value(value: object) -> object:
    """Content Fingerprint JSON에 사용할 결정적 scalar 표현으로 변환한다."""
    if value is pd.NA or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            value = value.item()
        except ValueError:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def content_fingerprint(
    frame: pd.DataFrame,
) -> str:
    """Column 순서·dtype·결정적 Row 정렬을 포함한 SHA256 Fingerprint를 계산한다."""
    payload = {
        "schema": [
            {
                "name": column,
                "dtype": str(frame[column].dtype),
            }
            for column in frame.columns
        ],
        "rows": [
            [
                _json_value(value)
                for value in row
            ]
            for row in frame.itertuples(
                index=False,
                name=None,
            )
        ],
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _path_is_within(
    path: Path,
    parent: Path,
) -> bool:
    """path가 parent 자신 또는 하위 경로인지 확인한다."""
    resolved_path = path.resolve(strict=False)
    resolved_parent = parent.resolve(strict=False)
    return (
        resolved_path == resolved_parent
        or resolved_parent in resolved_path.parents
    )


def validate_path_policy(
    *,
    games_path: Path,
    team_games_path: Path,
    output_path: Path,
    manifest_path: Path,
    project_root: Path = PROJECT_ROOT,
) -> None:
    """입력 덮어쓰기와 프로젝트 Raw/Interim 영역으로의 Output 생성을 차단한다."""
    inputs = {
        games_path.resolve(strict=False),
        team_games_path.resolve(strict=False),
    }
    outputs = {
        output_path.resolve(strict=False),
        manifest_path.resolve(strict=False),
    }

    if len(inputs) != 2:
        raise TeamFeatureBuildError(
            "games_path와 team_games_path가 같은 경로일 수 없습니다."
        )
    if len(outputs) != 2:
        raise TeamFeatureBuildError(
            "Feature Output과 Manifest 경로가 같을 수 없습니다."
        )
    if inputs & outputs:
        raise TeamFeatureBuildError(
            "Output 경로가 입력 Canonical 파일과 충돌합니다."
        )

    protected_roots = (
        project_root / "data" / "raw",
        project_root / "data" / "interim",
    )
    for path in outputs:
        for protected_root in protected_roots:
            if _path_is_within(path, protected_root):
                raise TeamFeatureBuildError(
                    "Feature Output/Manifest를 Raw 또는 Interim 영역에 쓸 수 없습니다: "
                    f"{path}"
                )


def _read_parquet(
    path: Path,
    *,
    label: str,
) -> pd.DataFrame:
    """Canonical Parquet을 명시적 pyarrow 엔진으로 읽는다."""
    if not path.is_file():
        raise TeamFeatureBuildError(
            f"{label} Parquet이 없습니다: {path}"
        )
    try:
        return pd.read_parquet(
            path,
            engine="pyarrow",
        )
    except Exception as exc:
        raise TeamFeatureBuildError(
            f"{label} Parquet을 읽지 못했습니다: {path}"
        ) from exc


def _write_parquet_atomic(
    frame: pd.DataFrame,
    output_path: Path,
) -> None:
    """Parquet을 임시 파일에 쓴 뒤 최종 경로로 원자적으로 교체한다."""
    if output_path.exists() and output_path.is_dir():
        raise TeamFeatureBuildError(
            f"Feature Output 경로가 디렉터리입니다: {output_path}"
        )
    try:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
    except OSError as exc:
        raise TeamFeatureBuildError(
            f"Output 디렉터리를 만들 수 없습니다: {output_path.parent}"
        ) from exc

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{output_path.stem}.",
            suffix=".parquet",
            dir=output_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        frame.to_parquet(
            temporary_path,
            engine="pyarrow",
            index=False,
        )
        temporary_path.replace(output_path)
    except Exception as exc:
        raise TeamFeatureBuildError(
            f"Feature Parquet 저장에 실패했습니다: {output_path}"
        ) from exc
    finally:
        if temporary_path is not None and temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError:
                LOGGER.warning(
                    "임시 Parquet 파일을 삭제하지 못했습니다: %s",
                    temporary_path,
                )


def _write_json_atomic(
    payload: Mapping[str, object],
    output_path: Path,
) -> None:
    """Manifest JSON을 임시 파일에 쓴 뒤 원자적으로 교체한다."""
    if output_path.exists() and output_path.is_dir():
        raise TeamFeatureBuildError(
            f"Manifest 경로가 디렉터리입니다: {output_path}"
        )
    try:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
    except OSError as exc:
        raise TeamFeatureBuildError(
            f"Manifest 디렉터리를 만들 수 없습니다: {output_path.parent}"
        ) from exc

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{output_path.stem}.",
            suffix=".json",
            dir=output_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(
                payload,
                temporary_file,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            temporary_file.write("\n")
        temporary_path.replace(output_path)
    except Exception as exc:
        raise TeamFeatureBuildError(
            f"Manifest 저장에 실패했습니다: {output_path}"
        ) from exc
    finally:
        if temporary_path is not None and temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError:
                LOGGER.warning(
                    "임시 Manifest 파일을 삭제하지 못했습니다: %s",
                    temporary_path,
                )


def _schema_manifest(
    frame: pd.DataFrame,
) -> list[dict[str, str]]:
    """Output Column과 dtype을 Manifest용 Schema로 직렬화한다."""
    return [
        {
            "name": column,
            "dtype": str(frame[column].dtype),
        }
        for column in frame.columns
    ]


def _build_manifest(
    *,
    features: pd.DataFrame,
    games_path: Path,
    team_games_path: Path,
    output_path: Path,
    games_sha256: str,
    team_games_sha256: str,
    rolling_windows: Sequence[int],
) -> dict[str, object]:
    """Feature Artifact의 provenance와 결정적 content identity를 구성한다."""
    return {
        "artifact": "team_pregame_features",
        "feature_version": FEATURE_VERSION,
        "source_contract_version": SOURCE_CONTRACT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "grain": list(KEY_COLUMNS),
        "historical_cutoff": "source.game_date < prediction_date",
        "same_day_policy": "exclude_all_same_date_source_results",
        "season_reset": True,
        "rolling_windows": list(rolling_windows),
        "rolling_boundary_policy": "include_entire_boundary_date_group",
        "key_columns": list(KEY_COLUMNS),
        "audit_columns": list(AUDIT_COLUMNS),
        "x_columns": list(
            EXPANDING_FEATURE_COLUMNS
            + _rolling_columns(rolling_windows)
        ),
        "inputs": {
            "games": {
                "path": str(games_path),
                "sha256": games_sha256,
            },
            "team_games": {
                "path": str(team_games_path),
                "sha256": team_games_sha256,
            },
        },
        "output": {
            "path": str(output_path),
            "row_count": int(len(features)),
            "content_fingerprint": content_fingerprint(features),
            "schema": _schema_manifest(features),
        },
    }


def build_team_feature_files(
    *,
    games_path: Path = DEFAULT_GAMES_PATH,
    team_games_path: Path = DEFAULT_TEAM_GAMES_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    manifest_path: Path | None = None,
    rolling_windows: Sequence[int] = ROLLING_WINDOWS,
    project_root: Path = PROJECT_ROOT,
) -> tuple[pd.DataFrame, dict[str, object], Path]:
    """
    Canonical 입력을 읽어 Feature Parquet/Manifest를 생성하고 round-trip까지 검증한다.
    """
    windows = _validate_windows(rolling_windows)
    resolved_manifest_path = (
        manifest_path
        if manifest_path is not None
        else output_path.with_suffix(".manifest.json")
    )

    validate_path_policy(
        games_path=games_path,
        team_games_path=team_games_path,
        output_path=output_path,
        manifest_path=resolved_manifest_path,
        project_root=project_root,
    )

    games_sha_before = calculate_sha256(games_path)
    team_games_sha_before = calculate_sha256(team_games_path)

    games = _read_parquet(
        games_path,
        label="games",
    )
    team_games = _read_parquet(
        team_games_path,
        label="team_games",
    )
    features = build_team_pregame_features(
        games,
        team_games,
        rolling_windows=windows,
    )

    expected_fingerprint = content_fingerprint(features)
    _write_parquet_atomic(
        features,
        output_path,
    )

    round_trip = _read_parquet(
        output_path,
        label="team_pregame_features",
    )
    round_trip = _normalize_output(
        round_trip,
        windows=windows,
    )
    validate_feature_output(
        round_trip,
        _normalize_team_games(team_games),
        windows=windows,
    )
    actual_fingerprint = content_fingerprint(round_trip)
    if actual_fingerprint != expected_fingerprint:
        raise TeamFeatureBuildError(
            "Parquet round-trip 후 content fingerprint가 변경되었습니다."
        )

    games_sha_after = calculate_sha256(games_path)
    team_games_sha_after = calculate_sha256(team_games_path)
    if games_sha_before != games_sha_after:
        raise TeamFeatureBuildError(
            "Builder 실행 중 games 입력 SHA256이 변경되었습니다."
        )
    if team_games_sha_before != team_games_sha_after:
        raise TeamFeatureBuildError(
            "Builder 실행 중 team_games 입력 SHA256이 변경되었습니다."
        )

    manifest = _build_manifest(
        features=round_trip,
        games_path=games_path,
        team_games_path=team_games_path,
        output_path=output_path,
        games_sha256=games_sha_after,
        team_games_sha256=team_games_sha_after,
        rolling_windows=windows,
    )
    _write_json_atomic(
        manifest,
        resolved_manifest_path,
    )

    return round_trip, manifest, resolved_manifest_path


def log_summary(
    features: pd.DataFrame,
    *,
    manifest_path: Path,
) -> None:
    """Feature 생성 결과의 핵심 검증값을 운영 로그에 남긴다."""
    cold_start_count = int(
        (~features["has_history"].astype(bool)).sum()
    )
    LOGGER.info(
        (
            "Team Pregame Feature 생성 완료: rows=%d, key_duplicates=%d, "
            "cold_start=%d, content_fingerprint=%s"
        ),
        len(features),
        int(
            features.duplicated(
                subset=list(KEY_COLUMNS),
                keep=False,
            ).sum()
        ),
        cold_start_count,
        content_fingerprint(features),
    )
    LOGGER.info(
        "Manifest: %s",
        manifest_path,
    )


def main() -> None:
    """CLI Entrypoint를 실행한다."""
    configure_logging()
    args = parse_args()
    try:
        features, _, manifest_path = build_team_feature_files(
            games_path=args.games_path,
            team_games_path=args.team_games_path,
            output_path=args.output_path,
            manifest_path=args.manifest_path,
        )
    except TeamFeatureBuildError:
        LOGGER.exception(
            "Team Pregame Feature 생성에 실패했습니다."
        )
        raise SystemExit(1)

    log_summary(
        features,
        manifest_path=manifest_path,
    )


if __name__ == "__main__":
    main()
