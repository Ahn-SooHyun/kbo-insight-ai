from __future__ import annotations

import argparse
import hashlib
import logging
import tempfile
from pathlib import Path
from typing import Iterable

import pandas as pd
from pandas.api.types import is_bool_dtype


LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

DEFAULT_DERIVED_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "hf_kbo_pbp"
    / "derived"
)

DEFAULT_BATTING_PATH = (
    DEFAULT_DERIVED_DIR
    / "player_game_batting.parquet"
)

DEFAULT_PITCHING_PATH = (
    DEFAULT_DERIVED_DIR
    / "player_game_pitching.parquet"
)

DEFAULT_TEAM_GAMES_PATH = (
    DEFAULT_DERIVED_DIR
    / "team_games.parquet"
)

DEFAULT_OUTPUT_DIR = DEFAULT_DERIVED_DIR

BATTING_SNAPSHOT_FILENAME = (
    "player_season_batting_snapshot.parquet"
)

PITCHING_SNAPSHOT_FILENAME = (
    "player_season_pitching_snapshot.parquet"
)

TEAM_SNAPSHOT_FILENAME = (
    "team_season_snapshot.parquet"
)

BATTING_SOURCE_KEY = (
    "game_pk",
    "batter",
)

PITCHING_SOURCE_KEY = (
    "game_pk",
    "pitcher",
)

TEAM_GAME_SOURCE_KEY = (
    "game_pk",
    "team",
)

BATTING_SNAPSHOT_KEY = (
    "season",
    "batter",
)

PITCHING_SNAPSHOT_KEY = (
    "season",
    "pitcher",
)

TEAM_SNAPSHOT_KEY = (
    "season",
    "team",
)

BATTING_DIRECT_SUM_COLUMNS = (
    "pa",
    "single",
    "double",
    "triple",
    "hr",
    "bb",
    "hbp",
    "so",
    "sf",
    "sh",
    "double_play",
    "triple_play",
    "field_error",
    "fielders_choice",
    "catcher_interference",
)

BATTING_RECONCILIATION_COLUMNS = (
    *BATTING_DIRECT_SUM_COLUMNS,
    "h",
    "ab",
    "tb",
)

REQUIRED_BATTING_COLUMNS = (
    "game_pk",
    "game_date",
    "season",
    "batter",
    *BATTING_RECONCILIATION_COLUMNS,
)

PITCHING_DIRECT_SUM_COLUMNS = (
    "pitch_rows",
    "pitches",
    "batters_faced_completed",
    "single_allowed",
    "double_allowed",
    "triple_allowed",
    "hr_allowed",
    "bb_allowed",
    "hbp_allowed",
    "so",
    "sf",
    "sh",
    "ball_pitch_count",
    "strike_pitch_count",
    "in_play_pitch_count",
)

PITCHING_RECONCILIATION_COLUMNS = (
    *PITCHING_DIRECT_SUM_COLUMNS,
    "hits_allowed",
)

REQUIRED_PITCHING_COLUMNS = (
    "game_pk",
    "game_date",
    "season",
    "pitcher",
    *PITCHING_RECONCILIATION_COLUMNS,
    "outs_recorded",
)

REQUIRED_TEAM_GAME_COLUMNS = (
    "game_pk",
    "game_date",
    "season",
    "team",
    "runs_for",
    "runs_against",
    "run_diff",
    "win",
    "loss",
    "tie",
)

BATTING_OUTPUT_COLUMNS = (
    "season",
    "through_date",
    "batter",
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
    "avg",
    "obp",
    "slg",
    "ops",
)

PITCHING_OUTPUT_COLUMNS = (
    "season",
    "through_date",
    "pitcher",
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

TEAM_OUTPUT_COLUMNS = (
    "season",
    "through_date",
    "team",
    "games",
    "wins",
    "losses",
    "ties",
    "runs_for",
    "runs_against",
    "run_diff",
)

BATTING_INTEGER_OUTPUT_COLUMNS = (
    "season",
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

BATTING_FLOAT_OUTPUT_COLUMNS = (
    "avg",
    "obp",
    "slg",
    "ops",
)

PITCHING_INTEGER_OUTPUT_COLUMNS = (
    "season",
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

TEAM_INTEGER_OUTPUT_COLUMNS = (
    "season",
    "games",
    "wins",
    "losses",
    "ties",
    "runs_for",
    "runs_against",
    "run_diff",
)

PROHIBITED_PLAYER_TEAM_COLUMNS = (
    "team",
    "current_team",
    "latest_team",
    "first_team",
    "last_team",
)

PROHIBITED_PITCHING_COLUMNS = (
    "earned_runs",
    "era",
    "runs_allowed",
    "innings_pitched",
    "ip",
    "avg_release_speed_kmh",
)


class SeasonSnapshotsBuildError(RuntimeError):
    """Season Snapshot 생성을 중단해야 하는 검증 오류를 나타낸다."""


def parse_args() -> argparse.Namespace:
    """Season Snapshot 생성 스크립트의 CLI 인자를 파싱한다."""
    parser = argparse.ArgumentParser(
        description=(
            "Player Game Batting, Player Game Pitching, Team Game에서 "
            "현재 Dataset Snapshot 기준 Season 누적 테이블을 생성합니다."
        )
    )

    parser.add_argument(
        "--batting-path",
        type=Path,
        default=DEFAULT_BATTING_PATH,
        help=(
            "Player Game Batting Parquet 경로입니다. "
            f"기본값: {DEFAULT_BATTING_PATH}"
        ),
    )

    parser.add_argument(
        "--pitching-path",
        type=Path,
        default=DEFAULT_PITCHING_PATH,
        help=(
            "Player Game Pitching Parquet 경로입니다. "
            f"기본값: {DEFAULT_PITCHING_PATH}"
        ),
    )

    parser.add_argument(
        "--team-games-path",
        type=Path,
        default=DEFAULT_TEAM_GAMES_PATH,
        help=(
            "Team Game Parquet 경로입니다. "
            f"기본값: {DEFAULT_TEAM_GAMES_PATH}"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Season Snapshot 출력 디렉터리입니다. "
            f"기본값: {DEFAULT_OUTPUT_DIR}"
        ),
    )

    return parser.parse_args()


def configure_logging() -> None:
    """독립 실행 시 사용할 기본 logging 형식을 설정한다."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def calculate_sha256(path: Path) -> str:
    """Input 파일 불변성 검증을 위해 SHA256을 계산한다."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _path_is_within(
    path: Path,
    parent: Path,
) -> bool:
    """path가 parent 자신 또는 하위 경로인지 확인한다."""
    resolved_path = path.resolve(
        strict=False
    )
    resolved_parent = parent.resolve(
        strict=False
    )

    return (
        resolved_path == resolved_parent
        or resolved_parent in resolved_path.parents
    )


def _require_columns(
    frame: pd.DataFrame,
    required: Iterable[str],
    *,
    source_name: str,
) -> None:
    """Source DataFrame의 필수 컬럼 존재 여부를 검증한다."""
    missing = [
        column
        for column in required
        if column not in frame.columns
    ]

    if missing:
        raise SeasonSnapshotsBuildError(
            f"{source_name}에 필요한 컬럼이 없습니다: {missing}"
        )


def _normalize_game_date(
    series: pd.Series,
    *,
    column_name: str,
) -> pd.Series:
    """날짜를 timezone 없는 datetime64[us] 날짜로 정규화한다."""
    parsed = pd.to_datetime(
        series,
        errors="coerce",
        utc=True,
    )

    invalid = parsed.isna()

    if invalid.any():
        examples = (
            series.loc[invalid]
            .head(5)
            .tolist()
        )

        raise SeasonSnapshotsBuildError(
            f"{column_name}을 datetime으로 변환할 수 없습니다: "
            f"{examples}"
        )

    try:
        return (
            parsed
            .dt
            .tz_convert(None)
            .dt
            .normalize()
            .astype("datetime64[us]")
        )
    except (TypeError, ValueError) as exc:
        raise SeasonSnapshotsBuildError(
            f"{column_name}을 datetime64[us]로 "
            "정규화할 수 없습니다."
        ) from exc


def _coerce_integer_column(
    series: pd.Series,
    *,
    column_name: str,
    allow_null: bool = False,
) -> pd.Series:
    """정수 의미의 컬럼을 nullable Int64로 손실 없이 정규화한다."""
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    invalid = (
        series.notna()
        & numeric.isna()
    )

    if invalid.any():
        examples = (
            series.loc[invalid]
            .head(5)
            .tolist()
        )

        raise SeasonSnapshotsBuildError(
            f"{column_name}에 숫자로 변환할 수 없는 값이 있습니다: "
            f"{examples}"
        )

    non_integer = (
        numeric.notna()
        & numeric.mod(1).ne(0)
    )

    if non_integer.any():
        examples = (
            numeric.loc[non_integer]
            .head(5)
            .tolist()
        )

        raise SeasonSnapshotsBuildError(
            f"{column_name}에 정수가 아닌 값이 있습니다: "
            f"{examples}"
        )

    result = numeric.astype("Int64")

    if (
        not allow_null
        and result.isna().any()
    ):
        raise SeasonSnapshotsBuildError(
            f"{column_name}에는 null을 허용하지 않습니다."
        )

    return result


def _normalize_required_string(
    series: pd.Series,
    *,
    column_name: str,
) -> pd.Series:
    """필수 식별자 문자열에 null 또는 빈 값이 없는지 검증한다."""
    values = series.astype("string")

    invalid = (
        values.isna()
        | values.str.strip().eq("").fillna(False)
    )

    if invalid.any():
        examples = (
            series.loc[invalid]
            .head(5)
            .tolist()
        )

        raise SeasonSnapshotsBuildError(
            f"{column_name}에 null 또는 빈 문자열이 있습니다: "
            f"{examples}"
        )

    return values


def _normalize_boolean_column(
    series: pd.Series,
    *,
    column_name: str,
) -> pd.Series:
    """Canonical boolean 컬럼을 pandas nullable boolean으로 정규화한다."""
    if not is_bool_dtype(
        series.dtype
    ):
        raise SeasonSnapshotsBuildError(
            f"{column_name}은 boolean dtype이어야 합니다. "
            f"현재 dtype={series.dtype}"
        )

    result = series.astype("boolean")

    if result.isna().any():
        raise SeasonSnapshotsBuildError(
            f"{column_name}에는 null을 허용하지 않습니다."
        )

    return result


def _validate_non_negative(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    source_name: str,
) -> None:
    """Count 계열 컬럼에 음수가 없는지 검증한다."""
    for column in columns:
        invalid = (
            frame[column]
            .dropna()
            .lt(0)
        )

        if invalid.any():
            raise SeasonSnapshotsBuildError(
                f"{source_name}.{column}에 음수 값이 있습니다."
            )


def _validate_unique_key(
    frame: pd.DataFrame,
    key: tuple[str, ...],
    *,
    source_name: str,
) -> None:
    """Source 또는 Output의 Grain Key가 Unique인지 검증한다."""
    duplicate_mask = frame.duplicated(
        subset=list(key),
        keep=False,
    )

    duplicate_count = int(
        duplicate_mask.sum()
    )

    if duplicate_count > 0:
        examples = (
            frame.loc[
                duplicate_mask,
                list(key),
            ]
            .head(10)
            .to_dict("records")
        )

        raise SeasonSnapshotsBuildError(
            f"{source_name} Key {key} 중복 Row가 "
            f"{duplicate_count}건 있습니다. 예시={examples}"
        )


def prepare_batting_source(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Player Game Batting Source를 검증하고 정규화한다."""
    _require_columns(
        frame,
        REQUIRED_BATTING_COLUMNS,
        source_name="Player Game Batting",
    )

    if frame.empty:
        raise SeasonSnapshotsBuildError(
            "Player Game Batting이 비어 있습니다."
        )

    result = frame.loc[
        :,
        REQUIRED_BATTING_COLUMNS,
    ].copy()

    result["game_pk"] = _normalize_required_string(
        result["game_pk"],
        column_name="Player Game Batting.game_pk",
    )

    result["batter"] = _normalize_required_string(
        result["batter"],
        column_name="Player Game Batting.batter",
    )

    result["game_date"] = _normalize_game_date(
        result["game_date"],
        column_name="Player Game Batting.game_date",
    )

    integer_columns = (
        "season",
        *BATTING_RECONCILIATION_COLUMNS,
    )

    for column in integer_columns:
        result[column] = _coerce_integer_column(
            result[column],
            column_name=f"Player Game Batting.{column}",
        )

    _validate_non_negative(
        result,
        BATTING_RECONCILIATION_COLUMNS,
        source_name="Player Game Batting",
    )

    _validate_unique_key(
        result,
        BATTING_SOURCE_KEY,
        source_name="Player Game Batting",
    )

    return (
        result
        .sort_values(
            by=[
                "season",
                "game_date",
                "game_pk",
                "batter",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def prepare_pitching_source(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Player Game Pitching Source를 검증하고 정규화한다."""
    _require_columns(
        frame,
        REQUIRED_PITCHING_COLUMNS,
        source_name="Player Game Pitching",
    )

    if frame.empty:
        raise SeasonSnapshotsBuildError(
            "Player Game Pitching이 비어 있습니다."
        )

    result = frame.loc[
        :,
        REQUIRED_PITCHING_COLUMNS,
    ].copy()

    result["game_pk"] = _normalize_required_string(
        result["game_pk"],
        column_name="Player Game Pitching.game_pk",
    )

    result["pitcher"] = _normalize_required_string(
        result["pitcher"],
        column_name="Player Game Pitching.pitcher",
    )

    result["game_date"] = _normalize_game_date(
        result["game_date"],
        column_name="Player Game Pitching.game_date",
    )

    result["season"] = _coerce_integer_column(
        result["season"],
        column_name="Player Game Pitching.season",
    )

    for column in PITCHING_RECONCILIATION_COLUMNS:
        result[column] = _coerce_integer_column(
            result[column],
            column_name=f"Player Game Pitching.{column}",
        )

    result["outs_recorded"] = _coerce_integer_column(
        result["outs_recorded"],
        column_name="Player Game Pitching.outs_recorded",
        allow_null=True,
    )

    _validate_non_negative(
        result,
        (
            *PITCHING_RECONCILIATION_COLUMNS,
            "outs_recorded",
        ),
        source_name="Player Game Pitching",
    )

    _validate_unique_key(
        result,
        PITCHING_SOURCE_KEY,
        source_name="Player Game Pitching",
    )

    return (
        result
        .sort_values(
            by=[
                "season",
                "game_date",
                "game_pk",
                "pitcher",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def prepare_team_games_source(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Team Game Source의 Grain과 최소 Context 계약을 검증한다."""
    _require_columns(
        frame,
        REQUIRED_TEAM_GAME_COLUMNS,
        source_name="Team Game",
    )

    if frame.empty:
        raise SeasonSnapshotsBuildError(
            "Team Game이 비어 있습니다."
        )

    result = frame.loc[
        :,
        REQUIRED_TEAM_GAME_COLUMNS,
    ].copy()

    result["game_pk"] = _normalize_required_string(
        result["game_pk"],
        column_name="Team Game.game_pk",
    )

    result["team"] = _normalize_required_string(
        result["team"],
        column_name="Team Game.team",
    )

    result["game_date"] = _normalize_game_date(
        result["game_date"],
        column_name="Team Game.game_date",
    )

    for column in (
        "season",
        "runs_for",
        "runs_against",
        "run_diff",
    ):
        result[column] = _coerce_integer_column(
            result[column],
            column_name=f"Team Game.{column}",
        )

    for column in (
        "win",
        "loss",
        "tie",
    ):
        result[column] = _normalize_boolean_column(
            result[column],
            column_name=f"Team Game.{column}",
        )

    _validate_non_negative(
        result,
        (
            "runs_for",
            "runs_against",
        ),
        source_name="Team Game",
    )

    _validate_unique_key(
        result,
        TEAM_GAME_SOURCE_KEY,
        source_name="Team Game",
    )

    rows_per_game = (
        result.groupby(
            "game_pk",
            sort=False,
            dropna=False,
        )
        .size()
    )

    invalid_row_count = (
        rows_per_game.ne(2)
    )

    if invalid_row_count.any():
        examples = (
            rows_per_game.loc[
                invalid_row_count
            ]
            .head(10)
            .to_dict()
        )

        raise SeasonSnapshotsBuildError(
            "Team Game은 각 game_pk마다 정확히 2행이어야 합니다. "
            f"예시={examples}"
        )

    teams_per_game = (
        result.groupby(
            "game_pk",
            sort=False,
            dropna=False,
        )["team"]
        .nunique(dropna=False)
    )

    if teams_per_game.ne(2).any():
        raise SeasonSnapshotsBuildError(
            "Team Game의 각 game_pk에는 서로 다른 두 Team이 필요합니다."
        )

    context_counts = (
        result.groupby(
            "game_pk",
            sort=False,
            dropna=False,
        )[
            [
                "game_date",
                "season",
            ]
        ]
        .nunique(dropna=False)
    )

    if context_counts.ne(1).any().any():
        raise SeasonSnapshotsBuildError(
            "Team Game의 game_pk별 game_date/season Context가 "
            "일관되지 않습니다."
        )

    expected_run_diff = (
        result["runs_for"]
        - result["runs_against"]
    )

    if result["run_diff"].ne(
        expected_run_diff
    ).any():
        raise SeasonSnapshotsBuildError(
            "Team Game.run_diff 공식식이 일치하지 않습니다."
        )

    result_count = (
        result["win"].astype("Int64")
        + result["loss"].astype("Int64")
        + result["tie"].astype("Int64")
    )

    if result_count.ne(1).any():
        raise SeasonSnapshotsBuildError(
            "Team Game의 각 Row는 win/loss/tie 중 "
            "정확히 하나만 True여야 합니다."
        )

    return (
        result
        .sort_values(
            by=[
                "season",
                "game_date",
                "game_pk",
                "team",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def build_through_dates(
    team_games: pd.DataFrame,
) -> pd.DataFrame:
    """Team Game Coverage에서 시즌 공통 through_date를 계산한다."""
    result = (
        team_games.groupby(
            "season",
            sort=True,
            dropna=False,
            as_index=False,
        )["game_date"]
        .max()
        .rename(
            columns={
                "game_date": "through_date",
            }
        )
    )

    result["season"] = _coerce_integer_column(
        result["season"],
        column_name="through_date.season",
    )

    result["through_date"] = _normalize_game_date(
        result["through_date"],
        column_name="through_date",
    )

    return result


def _validate_player_source_coverage(
    player_source: pd.DataFrame,
    *,
    team_games: pd.DataFrame,
    through_dates: pd.DataFrame,
    source_name: str,
) -> None:
    """Player Game이 Team Game의 Season/Game Coverage 안에 있는지 검증한다."""
    team_seasons = set(
        team_games["season"]
        .astype(int)
        .tolist()
    )

    player_seasons = set(
        player_source["season"]
        .astype(int)
        .tolist()
    )

    missing_seasons = sorted(
        player_seasons
        - team_seasons
    )

    if missing_seasons:
        raise SeasonSnapshotsBuildError(
            f"{source_name}에 Team Game Coverage가 없는 "
            f"시즌이 있습니다: {missing_seasons}"
        )

    through_map = (
        through_dates
        .set_index("season")["through_date"]
        .to_dict()
    )

    expected_through = (
        player_source["season"]
        .map(through_map)
    )

    late_mask = (
        player_source["game_date"]
        .gt(expected_through)
    )

    if late_mask.any():
        examples = (
            player_source.loc[
                late_mask,
                [
                    "game_pk",
                    "game_date",
                    "season",
                ],
            ]
            .head(10)
            .to_dict("records")
        )

        raise SeasonSnapshotsBuildError(
            f"{source_name}에 해당 시즌 through_date보다 "
            f"늦은 game_date가 있습니다: {examples}"
        )

    team_context = (
        team_games.loc[
            :,
            [
                "game_pk",
                "game_date",
                "season",
            ],
        ]
        .drop_duplicates(
            subset=["game_pk"],
            keep="first",
        )
    )

    merged = (
        player_source.loc[
            :,
            [
                "game_pk",
                "game_date",
                "season",
            ],
        ]
        .merge(
            team_context,
            on="game_pk",
            how="left",
            suffixes=(
                "_player",
                "_team",
            ),
            indicator=True,
            validate="many_to_one",
            sort=False,
        )
    )

    missing_game = (
        merged["_merge"]
        .ne("both")
    )

    if missing_game.any():
        examples = (
            merged.loc[
                missing_game,
                ["game_pk"],
            ]
            .head(10)
            .to_dict("records")
        )

        raise SeasonSnapshotsBuildError(
            f"{source_name}의 game_pk가 Team Game에 없습니다: "
            f"{examples}"
        )

    season_mismatch = (
        merged["season_player"]
        .ne(merged["season_team"])
    )

    date_mismatch = (
        merged["game_date_player"]
        .ne(merged["game_date_team"])
    )

    if (
        season_mismatch.any()
        or date_mismatch.any()
    ):
        examples = (
            merged.loc[
                season_mismatch
                | date_mismatch,
                [
                    "game_pk",
                    "game_date_player",
                    "game_date_team",
                    "season_player",
                    "season_team",
                ],
            ]
            .head(10)
            .to_dict("records")
        )

        raise SeasonSnapshotsBuildError(
            f"{source_name}와 Team Game의 Game Context가 "
            f"일치하지 않습니다: {examples}"
        )


def calculate_rate(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """0 denominator를 <NA>로 유지하며 Rate를 3자리 반올림한다."""
    result = pd.Series(
        pd.NA,
        index=numerator.index,
        dtype="Float64",
    )

    valid = (
        denominator.notna()
        & denominator.ne(0)
    )

    result.loc[valid] = (
        numerator.loc[valid]
        .astype("Float64")
        .div(
            denominator.loc[valid]
            .astype("Float64")
        )
        .round(3)
    )

    return result


def calculate_ops(
    obp: pd.Series,
    slg: pd.Series,
) -> pd.Series:
    """OBP와 SLG가 모두 정의된 경우에만 OPS를 계산한다."""
    result = pd.Series(
        pd.NA,
        index=obp.index,
        dtype="Float64",
    )

    valid = (
        obp.notna()
        & slg.notna()
    )

    result.loc[valid] = (
        obp.loc[valid]
        .astype("Float64")
        .add(
            slg.loc[valid]
            .astype("Float64")
        )
        .round(3)
    )

    return result


def build_batting_snapshot(
    batting: pd.DataFrame,
    through_dates: pd.DataFrame,
) -> pd.DataFrame:
    """Player Game Batting을 (season, batter) 누적 Snapshot으로 집계한다."""
    aggregated = (
        batting.groupby(
            list(BATTING_SNAPSHOT_KEY),
            sort=False,
            dropna=False,
        )[
            list(
                BATTING_DIRECT_SUM_COLUMNS
            )
        ]
        .sum()
        .reset_index()
    )

    aggregated["h"] = (
        aggregated["single"]
        + aggregated["double"]
        + aggregated["triple"]
        + aggregated["hr"]
    )

    aggregated["ab"] = (
        aggregated["pa"]
        - aggregated["bb"]
        - aggregated["hbp"]
        - aggregated["sh"]
        - aggregated["sf"]
        - aggregated[
            "catcher_interference"
        ]
    )

    aggregated["tb"] = (
        aggregated["single"]
        + 2 * aggregated["double"]
        + 3 * aggregated["triple"]
        + 4 * aggregated["hr"]
    )

    aggregated["avg"] = calculate_rate(
        numerator=aggregated["h"],
        denominator=aggregated["ab"],
    )

    on_base_denominator = (
        aggregated["ab"]
        + aggregated["bb"]
        + aggregated["hbp"]
        + aggregated["sf"]
    )

    aggregated["obp"] = calculate_rate(
        numerator=(
            aggregated["h"]
            + aggregated["bb"]
            + aggregated["hbp"]
        ),
        denominator=on_base_denominator,
    )

    aggregated["slg"] = calculate_rate(
        numerator=aggregated["tb"],
        denominator=aggregated["ab"],
    )

    aggregated["ops"] = calculate_ops(
        obp=aggregated["obp"],
        slg=aggregated["slg"],
    )

    return (
        aggregated.merge(
            through_dates,
            on="season",
            how="left",
            validate="many_to_one",
            sort=False,
        )
    )


def _build_nullable_season_outs(
    pitching: pd.DataFrame,
) -> pd.DataFrame:
    """
    Player Game 중 하나라도 outs_recorded가 null이면
    해당 Player-Season의 Season Outs 전체를 <NA>로 전파한다.
    """
    records: list[
        dict[str, object]
    ] = []

    grouped = pitching.groupby(
        list(PITCHING_SNAPSHOT_KEY),
        sort=False,
        dropna=False,
    )

    for (
        season,
        pitcher,
    ), group in grouped:
        if group[
            "outs_recorded"
        ].isna().any():
            outs_recorded: object = pd.NA
        else:
            outs_recorded = int(
                group[
                    "outs_recorded"
                ].sum()
            )

        records.append(
            {
                "season": season,
                "pitcher": pitcher,
                "outs_recorded": outs_recorded,
            }
        )

    result = pd.DataFrame(
        records
    )

    result["outs_recorded"] = (
        result["outs_recorded"]
        .astype("Int64")
    )

    return result


def build_pitching_snapshot(
    pitching: pd.DataFrame,
    through_dates: pd.DataFrame,
) -> pd.DataFrame:
    """Player Game Pitching을 (season, pitcher) 누적 Snapshot으로 집계한다."""
    counts = (
        pitching.groupby(
            list(PITCHING_SNAPSHOT_KEY),
            sort=False,
            dropna=False,
        )[
            list(
                PITCHING_DIRECT_SUM_COLUMNS
            )
        ]
        .sum()
        .reset_index()
    )

    counts["hits_allowed"] = (
        counts["single_allowed"]
        + counts["double_allowed"]
        + counts["triple_allowed"]
        + counts["hr_allowed"]
    )

    outs = _build_nullable_season_outs(
        pitching
    )

    result = counts.merge(
        outs,
        on=list(
            PITCHING_SNAPSHOT_KEY
        ),
        how="inner",
        validate="one_to_one",
        sort=False,
    )

    return (
        result.merge(
            through_dates,
            on="season",
            how="left",
            validate="many_to_one",
            sort=False,
        )
    )


def build_team_snapshot(
    team_games: pd.DataFrame,
    through_dates: pd.DataFrame,
) -> pd.DataFrame:
    """Team Game을 (season, team) 누적 Snapshot으로 집계한다."""
    grouped = team_games.groupby(
        list(TEAM_SNAPSHOT_KEY),
        sort=False,
        dropna=False,
    )

    games = (
        grouped
        .size()
        .rename("games")
        .reset_index()
    )

    stats_source = team_games.copy()

    stats_source["wins"] = (
        stats_source["win"]
        .astype("Int64")
    )
    stats_source["losses"] = (
        stats_source["loss"]
        .astype("Int64")
    )
    stats_source["ties"] = (
        stats_source["tie"]
        .astype("Int64")
    )

    stats = (
        stats_source.groupby(
            list(TEAM_SNAPSHOT_KEY),
            sort=False,
            dropna=False,
        )[
            [
                "wins",
                "losses",
                "ties",
                "runs_for",
                "runs_against",
            ]
        ]
        .sum()
        .reset_index()
    )

    result = games.merge(
        stats,
        on=list(
            TEAM_SNAPSHOT_KEY
        ),
        how="inner",
        validate="one_to_one",
        sort=False,
    )

    result["run_diff"] = (
        result["runs_for"]
        - result["runs_against"]
    )

    return (
        result.merge(
            through_dates,
            on="season",
            how="left",
            validate="many_to_one",
            sort=False,
        )
    )


def normalize_batting_snapshot_dtypes(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Batting Snapshot의 컬럼 순서, dtype, 정렬을 고정한다."""
    _require_columns(
        frame,
        BATTING_OUTPUT_COLUMNS,
        source_name="Batting Snapshot",
    )

    result = frame.loc[
        :,
        BATTING_OUTPUT_COLUMNS,
    ].copy()

    result["batter"] = (
        result["batter"]
        .astype("string")
    )

    result["through_date"] = _normalize_game_date(
        result["through_date"],
        column_name="Batting Snapshot.through_date",
    )

    for column in BATTING_INTEGER_OUTPUT_COLUMNS:
        result[column] = _coerce_integer_column(
            result[column],
            column_name=f"Batting Snapshot.{column}",
        )

    for column in BATTING_FLOAT_OUTPUT_COLUMNS:
        try:
            result[column] = (
                result[column]
                .astype("Float64")
            )
        except (TypeError, ValueError) as exc:
            raise SeasonSnapshotsBuildError(
                f"Batting Snapshot.{column}을 "
                "Float64로 변환할 수 없습니다."
            ) from exc

    return (
        result
        .sort_values(
            by=[
                "season",
                "batter",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def normalize_pitching_snapshot_dtypes(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Pitching Snapshot의 컬럼 순서, dtype, 정렬을 고정한다."""
    _require_columns(
        frame,
        PITCHING_OUTPUT_COLUMNS,
        source_name="Pitching Snapshot",
    )

    result = frame.loc[
        :,
        PITCHING_OUTPUT_COLUMNS,
    ].copy()

    result["pitcher"] = (
        result["pitcher"]
        .astype("string")
    )

    result["through_date"] = _normalize_game_date(
        result["through_date"],
        column_name="Pitching Snapshot.through_date",
    )

    for column in PITCHING_INTEGER_OUTPUT_COLUMNS:
        result[column] = _coerce_integer_column(
            result[column],
            column_name=f"Pitching Snapshot.{column}",
            allow_null=(
                column
                == "outs_recorded"
            ),
        )

    return (
        result
        .sort_values(
            by=[
                "season",
                "pitcher",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def normalize_team_snapshot_dtypes(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Team Snapshot의 컬럼 순서, dtype, 정렬을 고정한다."""
    _require_columns(
        frame,
        TEAM_OUTPUT_COLUMNS,
        source_name="Team Snapshot",
    )

    result = frame.loc[
        :,
        TEAM_OUTPUT_COLUMNS,
    ].copy()

    result["team"] = (
        result["team"]
        .astype("string")
    )

    result["through_date"] = _normalize_game_date(
        result["through_date"],
        column_name="Team Snapshot.through_date",
    )

    for column in TEAM_INTEGER_OUTPUT_COLUMNS:
        result[column] = _coerce_integer_column(
            result[column],
            column_name=f"Team Snapshot.{column}",
        )

    return (
        result
        .sort_values(
            by=[
                "season",
                "team",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def _expected_dtype_map(
    kind: str,
) -> dict[str, str]:
    """Snapshot 종류별 dtype 계약을 반환한다."""
    if kind == "batting":
        return {
            "season": "Int64",
            "through_date": "datetime64[us]",
            "batter": "string",
            **{
                column: "Int64"
                for column
                in BATTING_INTEGER_OUTPUT_COLUMNS
                if column != "season"
            },
            **{
                column: "Float64"
                for column
                in BATTING_FLOAT_OUTPUT_COLUMNS
            },
        }

    if kind == "pitching":
        return {
            "season": "Int64",
            "through_date": "datetime64[us]",
            "pitcher": "string",
            **{
                column: "Int64"
                for column
                in PITCHING_INTEGER_OUTPUT_COLUMNS
                if column != "season"
            },
        }

    if kind == "team":
        return {
            "season": "Int64",
            "through_date": "datetime64[us]",
            "team": "string",
            **{
                column: "Int64"
                for column
                in TEAM_INTEGER_OUTPUT_COLUMNS
                if column != "season"
            },
        }

    raise SeasonSnapshotsBuildError(
        f"지원하지 않는 Snapshot 종류입니다: {kind}"
    )


def _validate_exact_schema(
    frame: pd.DataFrame,
    expected_columns: tuple[str, ...],
    *,
    snapshot_name: str,
) -> None:
    """Output에 누락/추가 컬럼이 없는지 검증한다."""
    actual = tuple(
        frame.columns
    )

    if actual != expected_columns:
        raise SeasonSnapshotsBuildError(
            f"{snapshot_name} Schema가 일치하지 않습니다. "
            f"expected={expected_columns}, actual={actual}"
        )


def _validate_output_dtypes(
    frame: pd.DataFrame,
    *,
    kind: str,
    snapshot_name: str,
) -> None:
    """Snapshot dtype 계약을 검증한다."""
    expected = _expected_dtype_map(
        kind
    )

    mismatch = {}

    for column, expected_dtype in expected.items():
        actual_dtype = str(
            frame[column].dtype
        )

        if actual_dtype != expected_dtype:
            mismatch[column] = {
                "expected": expected_dtype,
                "actual": actual_dtype,
            }

    if mismatch:
        raise SeasonSnapshotsBuildError(
            f"{snapshot_name} dtype 계약이 일치하지 않습니다: "
            f"{mismatch}"
        )


def _nullable_float_series_equal(
    left: pd.Series,
    right: pd.Series,
) -> pd.Series:
    """nullable Float Series를 <NA>까지 포함해 비교한다."""
    return (
        left.eq(right)
        | (
            left.isna()
            & right.isna()
        )
    ).fillna(False)


def _validate_through_date_contract(
    output: pd.DataFrame,
    through_dates: pd.DataFrame,
    *,
    snapshot_name: str,
) -> None:
    """Snapshot의 season -> through_date가 Team Game Coverage와 일치하는지 검증한다."""
    expected = (
        through_dates
        .set_index("season")["through_date"]
        .to_dict()
    )

    output_seasons = set(
        output["season"]
        .astype(int)
        .tolist()
    )

    expected_seasons = set(
        int(season)
        for season in expected
    )

    if not output_seasons.issubset(
        expected_seasons
    ):
        raise SeasonSnapshotsBuildError(
            f"{snapshot_name}에 알 수 없는 season이 있습니다."
        )

    mapped = (
        output["season"]
        .map(expected)
    )

    mismatch = (
        output["through_date"]
        .ne(mapped)
    )

    if mismatch.any():
        raise SeasonSnapshotsBuildError(
            f"{snapshot_name}의 through_date가 "
            "Team Game Coverage와 일치하지 않습니다."
        )

    per_season = (
        output.groupby(
            "season",
            sort=True,
            dropna=False,
        )["through_date"]
        .nunique(dropna=False)
    )

    if per_season.ne(1).any():
        raise SeasonSnapshotsBuildError(
            f"{snapshot_name}의 season -> through_date가 "
            "1:1이 아닙니다."
        )


def _validate_season_column_sums(
    source: pd.DataFrame,
    output: pd.DataFrame,
    columns: Iterable[str],
    *,
    context: str,
) -> None:
    """동일 이름 Count 컬럼의 Source ↔ Snapshot 시즌별 합계를 검증한다."""
    for column in columns:
        source_sum = (
            source.groupby(
                "season",
                sort=True,
                dropna=False,
            )[column]
            .sum()
        )

        output_sum = (
            output.groupby(
                "season",
                sort=True,
                dropna=False,
            )[column]
            .sum()
        )

        reconciled = (
            source_sum
            .to_frame("source")
            .join(
                output_sum.rename(
                    "output"
                ),
                how="outer",
            )
        )

        mismatch = (
            reconciled["source"]
            .ne(
                reconciled["output"]
            )
            .fillna(True)
        )

        if mismatch.any():
            examples = (
                reconciled.loc[
                    mismatch
                ]
                .head(10)
                .reset_index()
                .to_dict("records")
            )

            raise SeasonSnapshotsBuildError(
                f"{context}.{column} 시즌별 Reconciliation이 "
                f"일치하지 않습니다: {examples}"
            )


def validate_batting_snapshot(
    source: pd.DataFrame,
    output: pd.DataFrame,
    through_dates: pd.DataFrame,
) -> None:
    """Batting Snapshot Grain, Count, 공식식, Rate를 검증한다."""
    _validate_exact_schema(
        output,
        BATTING_OUTPUT_COLUMNS,
        snapshot_name="Batting Snapshot",
    )

    _validate_output_dtypes(
        output,
        kind="batting",
        snapshot_name="Batting Snapshot",
    )

    _validate_unique_key(
        output,
        BATTING_SNAPSHOT_KEY,
        source_name="Batting Snapshot",
    )

    _validate_through_date_contract(
        output,
        through_dates,
        snapshot_name="Batting Snapshot",
    )

    prohibited = [
        column
        for column
        in PROHIBITED_PLAYER_TEAM_COLUMNS
        if column in output.columns
    ]

    if prohibited:
        raise SeasonSnapshotsBuildError(
            "Batting Snapshot에 단일 Team 컬럼이 "
            f"포함되어 있습니다: {prohibited}"
        )

    _validate_non_negative(
        output,
        (
            column
            for column
            in BATTING_INTEGER_OUTPUT_COLUMNS
            if column != "season"
        ),
        source_name="Batting Snapshot",
    )

    _validate_season_column_sums(
        source,
        output,
        BATTING_RECONCILIATION_COLUMNS,
        context="Batting Snapshot",
    )

    h_expected = (
        output["single"]
        + output["double"]
        + output["triple"]
        + output["hr"]
    )

    ab_expected = (
        output["pa"]
        - output["bb"]
        - output["hbp"]
        - output["sh"]
        - output["sf"]
        - output[
            "catcher_interference"
        ]
    )

    tb_expected = (
        output["single"]
        + 2 * output["double"]
        + 3 * output["triple"]
        + 4 * output["hr"]
    )

    if output["h"].ne(
        h_expected
    ).any():
        raise SeasonSnapshotsBuildError(
            "Batting Snapshot H 공식식이 일치하지 않습니다."
        )

    if output["ab"].ne(
        ab_expected
    ).any():
        raise SeasonSnapshotsBuildError(
            "Batting Snapshot AB 공식식이 일치하지 않습니다."
        )

    if output["tb"].ne(
        tb_expected
    ).any():
        raise SeasonSnapshotsBuildError(
            "Batting Snapshot TB 공식식이 일치하지 않습니다."
        )

    avg_expected = calculate_rate(
        output["h"],
        output["ab"],
    )

    obp_expected = calculate_rate(
        (
            output["h"]
            + output["bb"]
            + output["hbp"]
        ),
        (
            output["ab"]
            + output["bb"]
            + output["hbp"]
            + output["sf"]
        ),
    )

    slg_expected = calculate_rate(
        output["tb"],
        output["ab"],
    )

    ops_expected = calculate_ops(
        obp_expected,
        slg_expected,
    )

    rate_checks = (
        (
            "avg",
            output["avg"],
            avg_expected,
        ),
        (
            "obp",
            output["obp"],
            obp_expected,
        ),
        (
            "slg",
            output["slg"],
            slg_expected,
        ),
        (
            "ops",
            output["ops"],
            ops_expected,
        ),
    )

    for name, actual, expected in rate_checks:
        matches = _nullable_float_series_equal(
            actual,
            expected,
        )

        if not matches.all():
            raise SeasonSnapshotsBuildError(
                f"Batting Snapshot {name} 공식식이 "
                "일치하지 않습니다."
            )


def validate_pitching_snapshot(
    source: pd.DataFrame,
    output: pd.DataFrame,
    through_dates: pd.DataFrame,
) -> None:
    """Pitching Snapshot의 Count와 Outs Null 전파 계약을 검증한다."""
    _validate_exact_schema(
        output,
        PITCHING_OUTPUT_COLUMNS,
        snapshot_name="Pitching Snapshot",
    )

    _validate_output_dtypes(
        output,
        kind="pitching",
        snapshot_name="Pitching Snapshot",
    )

    _validate_unique_key(
        output,
        PITCHING_SNAPSHOT_KEY,
        source_name="Pitching Snapshot",
    )

    _validate_through_date_contract(
        output,
        through_dates,
        snapshot_name="Pitching Snapshot",
    )

    prohibited = [
        column
        for column
        in (
            *PROHIBITED_PLAYER_TEAM_COLUMNS,
            *PROHIBITED_PITCHING_COLUMNS,
        )
        if column in output.columns
    ]

    if prohibited:
        raise SeasonSnapshotsBuildError(
            "Pitching Snapshot에 범위 제외 컬럼이 "
            f"포함되어 있습니다: {prohibited}"
        )

    _validate_non_negative(
        output,
        (
            column
            for column
            in PITCHING_INTEGER_OUTPUT_COLUMNS
            if column != "season"
        ),
        source_name="Pitching Snapshot",
    )

    _validate_season_column_sums(
        source,
        output,
        PITCHING_RECONCILIATION_COLUMNS,
        context="Pitching Snapshot",
    )

    hits_expected = (
        output["single_allowed"]
        + output["double_allowed"]
        + output["triple_allowed"]
        + output["hr_allowed"]
    )

    if output[
        "hits_allowed"
    ].ne(
        hits_expected
    ).any():
        raise SeasonSnapshotsBuildError(
            "Pitching Snapshot hits_allowed 공식식이 "
            "일치하지 않습니다."
        )

    pitch_type_expected = (
        output["ball_pitch_count"]
        + output["strike_pitch_count"]
        + output["in_play_pitch_count"]
    )

    if output["pitches"].ne(
        pitch_type_expected
    ).any():
        raise SeasonSnapshotsBuildError(
            "Pitching Snapshot에서 "
            "pitches = B + S + X 공식식이 일치하지 않습니다."
        )

    output_index = (
        output
        .set_index(
            list(
                PITCHING_SNAPSHOT_KEY
            )
        )
    )

    for (
        season,
        pitcher,
    ), group in source.groupby(
        list(
            PITCHING_SNAPSHOT_KEY
        ),
        sort=False,
        dropna=False,
    ):
        actual = output_index.loc[
            (
                season,
                pitcher,
            ),
            "outs_recorded",
        ]

        has_null = (
            group[
                "outs_recorded"
            ]
            .isna()
            .any()
        )

        if has_null:
            if not pd.isna(
                actual
            ):
                raise SeasonSnapshotsBuildError(
                    "Pitching Snapshot outs_recorded Null "
                    "전파 계약이 위반되었습니다: "
                    f"season={season}, pitcher={pitcher}"
                )
            continue

        expected = int(
            group[
                "outs_recorded"
            ].sum()
        )

        if (
            pd.isna(actual)
            or int(actual) != expected
        ):
            raise SeasonSnapshotsBuildError(
                "Pitching Snapshot outs_recorded 합계가 "
                "일치하지 않습니다: "
                f"season={season}, pitcher={pitcher}, "
                f"expected={expected}, actual={actual}"
            )


def validate_team_snapshot(
    source: pd.DataFrame,
    output: pd.DataFrame,
    through_dates: pd.DataFrame,
) -> None:
    """Team Snapshot Grain, 결과 합계, 대칭성을 검증한다."""
    _validate_exact_schema(
        output,
        TEAM_OUTPUT_COLUMNS,
        snapshot_name="Team Snapshot",
    )

    _validate_output_dtypes(
        output,
        kind="team",
        snapshot_name="Team Snapshot",
    )

    _validate_unique_key(
        output,
        TEAM_SNAPSHOT_KEY,
        source_name="Team Snapshot",
    )

    _validate_through_date_contract(
        output,
        through_dates,
        snapshot_name="Team Snapshot",
    )

    _validate_non_negative(
        output,
        (
            "games",
            "wins",
            "losses",
            "ties",
            "runs_for",
            "runs_against",
        ),
        source_name="Team Snapshot",
    )

    expected_games = (
        output["wins"]
        + output["losses"]
        + output["ties"]
    )

    if output["games"].ne(
        expected_games
    ).any():
        raise SeasonSnapshotsBuildError(
            "Team Snapshot에서 "
            "games = wins + losses + ties가 성립하지 않습니다."
        )

    expected_run_diff = (
        output["runs_for"]
        - output["runs_against"]
    )

    if output["run_diff"].ne(
        expected_run_diff
    ).any():
        raise SeasonSnapshotsBuildError(
            "Team Snapshot run_diff 공식식이 일치하지 않습니다."
        )

    source_result = source.copy()
    source_result["wins"] = (
        source_result["win"]
        .astype("Int64")
    )
    source_result["losses"] = (
        source_result["loss"]
        .astype("Int64")
    )
    source_result["ties"] = (
        source_result["tie"]
        .astype("Int64")
    )

    _validate_season_column_sums(
        source_result,
        output,
        (
            "wins",
            "losses",
            "ties",
            "runs_for",
            "runs_against",
        ),
        context="Team Snapshot",
    )

    source_rows = (
        source.groupby(
            "season",
            sort=True,
            dropna=False,
        )
        .size()
    )

    output_games = (
        output.groupby(
            "season",
            sort=True,
            dropna=False,
        )["games"]
        .sum()
    )

    game_reconciliation = (
        source_rows
        .rename("source_rows")
        .to_frame()
        .join(
            output_games.rename(
                "output_games"
            ),
            how="outer",
        )
    )

    mismatch = (
        game_reconciliation[
            "source_rows"
        ]
        .ne(
            game_reconciliation[
                "output_games"
            ]
        )
        .fillna(True)
    )

    if mismatch.any():
        raise SeasonSnapshotsBuildError(
            "Team Game Row Count와 Team Snapshot games 합계가 "
            "일치하지 않습니다."
        )

    unique_games = (
        source.groupby(
            "season",
            sort=True,
            dropna=False,
        )["game_pk"]
        .nunique()
        .mul(2)
    )

    if not output_games.astype(
        "Int64"
    ).equals(
        unique_games.astype(
            "Int64"
        )
    ):
        raise SeasonSnapshotsBuildError(
            "Team Snapshot games 합계가 "
            "2 * unique game_pk와 일치하지 않습니다."
        )

    season_summary = (
        output.groupby(
            "season",
            sort=True,
            dropna=False,
        )[
            [
                "wins",
                "losses",
                "ties",
                "runs_for",
                "runs_against",
            ]
        ]
        .sum()
    )

    if season_summary[
        "wins"
    ].ne(
        season_summary[
            "losses"
        ]
    ).any():
        raise SeasonSnapshotsBuildError(
            "시즌 전체 sum(wins) != sum(losses)입니다."
        )

    if season_summary[
        "runs_for"
    ].ne(
        season_summary[
            "runs_against"
        ]
    ).any():
        raise SeasonSnapshotsBuildError(
            "시즌 전체 sum(runs_for) != sum(runs_against)입니다."
        )


def build_season_snapshots(
    player_game_batting: pd.DataFrame,
    player_game_pitching: pd.DataFrame,
    team_games: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """세 Canonical Fact Table에서 Season Snapshot 세 개를 생성한다."""
    batting_source = (
        prepare_batting_source(
            player_game_batting
        )
    )

    pitching_source = (
        prepare_pitching_source(
            player_game_pitching
        )
    )

    team_source = (
        prepare_team_games_source(
            team_games
        )
    )

    through_dates = (
        build_through_dates(
            team_source
        )
    )

    _validate_player_source_coverage(
        batting_source,
        team_games=team_source,
        through_dates=through_dates,
        source_name="Player Game Batting",
    )

    _validate_player_source_coverage(
        pitching_source,
        team_games=team_source,
        through_dates=through_dates,
        source_name="Player Game Pitching",
    )

    batting_snapshot = (
        build_batting_snapshot(
            batting_source,
            through_dates,
        )
    )

    pitching_snapshot = (
        build_pitching_snapshot(
            pitching_source,
            through_dates,
        )
    )

    team_snapshot = (
        build_team_snapshot(
            team_source,
            through_dates,
        )
    )

    batting_snapshot = (
        normalize_batting_snapshot_dtypes(
            batting_snapshot
        )
    )

    pitching_snapshot = (
        normalize_pitching_snapshot_dtypes(
            pitching_snapshot
        )
    )

    team_snapshot = (
        normalize_team_snapshot_dtypes(
            team_snapshot
        )
    )

    validate_batting_snapshot(
        batting_source,
        batting_snapshot,
        through_dates,
    )

    validate_pitching_snapshot(
        pitching_source,
        pitching_snapshot,
        through_dates,
    )

    validate_team_snapshot(
        team_source,
        team_snapshot,
        through_dates,
    )

    return (
        batting_snapshot,
        pitching_snapshot,
        team_snapshot,
    )


def ensure_output_paths_safe(
    *,
    batting_path: Path,
    pitching_path: Path,
    team_games_path: Path,
    batting_output_path: Path,
    pitching_output_path: Path,
    team_output_path: Path,
) -> None:
    """Snapshot Output의 Raw 저장, Input 덮어쓰기, 경로 충돌을 방지한다."""
    input_paths = {
        batting_path.resolve(
            strict=False
        ),
        pitching_path.resolve(
            strict=False
        ),
        team_games_path.resolve(
            strict=False
        ),
    }

    output_paths = [
        batting_output_path,
        pitching_output_path,
        team_output_path,
    ]

    resolved_outputs = [
        path.resolve(
            strict=False
        )
        for path in output_paths
    ]

    if len(
        set(
            resolved_outputs
        )
    ) != len(
        resolved_outputs
    ):
        raise SeasonSnapshotsBuildError(
            "세 Season Snapshot Output 경로는 서로 달라야 합니다."
        )

    for output_path, resolved_output in zip(
        output_paths,
        resolved_outputs,
    ):
        if _path_is_within(
            output_path,
            RAW_DATA_DIR,
        ):
            raise SeasonSnapshotsBuildError(
                "Season Snapshot Output을 data/raw/ 내부에 "
                f"생성할 수 없습니다: {output_path}"
            )

        if resolved_output in input_paths:
            raise SeasonSnapshotsBuildError(
                "Season Snapshot Output이 Canonical Input을 "
                f"덮어쓸 수 없습니다: {output_path}"
            )


def _read_parquet(
    path: Path,
    *,
    source_name: str,
) -> pd.DataFrame:
    """Canonical Input Parquet을 읽는다."""
    if not path.is_file():
        raise SeasonSnapshotsBuildError(
            f"{source_name} 파일이 없습니다: {path}"
        )

    try:
        return pd.read_parquet(
            path,
            engine="pyarrow",
        )
    except (
        OSError,
        ValueError,
        TypeError,
    ) as exc:
        raise SeasonSnapshotsBuildError(
            f"{source_name} Parquet을 읽을 수 없습니다: "
            f"{path}"
        ) from exc


def _normalize_round_trip(
    frame: pd.DataFrame,
    *,
    kind: str,
) -> pd.DataFrame:
    """Parquet read-back 결과를 해당 Snapshot dtype 계약으로 정규화한다."""
    if kind == "batting":
        return normalize_batting_snapshot_dtypes(
            frame
        )

    if kind == "pitching":
        return normalize_pitching_snapshot_dtypes(
            frame
        )

    if kind == "team":
        return normalize_team_snapshot_dtypes(
            frame
        )

    raise SeasonSnapshotsBuildError(
        f"지원하지 않는 Snapshot 종류입니다: {kind}"
    )


def _write_snapshot_atomically(
    frame: pd.DataFrame,
    output_path: Path,
    *,
    kind: str,
) -> None:
    """
    검증된 Snapshot을 임시 Parquet에 기록하고
    round-trip 검증 후 최종 경로로 원자적으로 교체한다.
    """
    if (
        output_path.exists()
        and output_path.is_dir()
    ):
        raise SeasonSnapshotsBuildError(
            "Snapshot Output 경로가 디렉터리입니다: "
            f"{output_path}"
        )

    try:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
    except OSError as exc:
        raise SeasonSnapshotsBuildError(
            "Snapshot Output 디렉터리를 생성할 수 없습니다: "
            f"{output_path.parent}"
        ) from exc

    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            dir=output_path.parent,
            prefix=f".{output_path.stem}.",
            suffix=".tmp.parquet",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(
                temporary_file.name
            )

        frame.to_parquet(
            temporary_path,
            engine="pyarrow",
            index=False,
        )

        round_trip = pd.read_parquet(
            temporary_path,
            engine="pyarrow",
        )

        round_trip = _normalize_round_trip(
            round_trip,
            kind=kind,
        )

        pd.testing.assert_frame_equal(
            frame,
            round_trip,
            check_dtype=True,
            check_like=False,
        )

        LOGGER.info(
            "Output dtype round-trip 검증 통과 | "
            "kind=%s | output=%s | dtypes=%s",
            kind,
            output_path,
            {
                column: str(dtype)
                for column, dtype
                in round_trip.dtypes.items()
            },
        )

        temporary_path.replace(
            output_path
        )

    except (
        OSError,
        ValueError,
        TypeError,
        AssertionError,
    ) as exc:
        raise SeasonSnapshotsBuildError(
            "Season Snapshot Parquet 저장/round-trip "
            f"검증에 실패했습니다: {output_path}"
        ) from exc

    finally:
        if (
            temporary_path is not None
            and temporary_path.exists()
        ):
            try:
                temporary_path.unlink()
            except OSError:
                LOGGER.warning(
                    "임시 Snapshot Parquet을 삭제하지 못했습니다: %s",
                    temporary_path,
                )


def _season_row_count_text(
    frame: pd.DataFrame,
) -> str:
    """Production Logging용 시즌별 Row Count 문자열을 만든다."""
    counts = (
        frame.groupby(
            "season",
            sort=True,
            dropna=False,
        )
        .size()
    )

    return ", ".join(
        f"{int(season)}={int(count)}"
        for season, count
        in counts.items()
    )


def log_production_validation(
    *,
    batting_source: pd.DataFrame,
    pitching_source: pd.DataFrame,
    team_source: pd.DataFrame,
    batting_snapshot: pd.DataFrame,
    pitching_snapshot: pd.DataFrame,
    team_snapshot: pd.DataFrame,
) -> None:
    """Issue #13 Production Validation 핵심 지표를 logging으로 기록한다."""
    through_dates = (
        team_snapshot.loc[
            :,
            [
                "season",
                "through_date",
            ],
        ]
        .drop_duplicates()
        .sort_values(
            "season",
            kind="mergesort",
        )
    )

    LOGGER.info(
        "Season별 through_date: %s",
        through_dates.to_dict(
            "records"
        ),
    )

    LOGGER.info(
        "시즌별 Player Season Batting Row Count: %s",
        _season_row_count_text(
            batting_snapshot
        ),
    )

    LOGGER.info(
        "Batting Validation | "
        "output_rows=%d | duplicate_keys=%d | "
        "source_pa_sum=%d | snapshot_pa_sum=%d | "
        "source_h_sum=%d | snapshot_h_sum=%d | "
        "source_ab_sum=%d | snapshot_ab_sum=%d | "
        "source_tb_sum=%d | snapshot_tb_sum=%d | "
        "count_reconciliation_errors=0 | "
        "formula_errors=0 | rate_errors=0",
        len(
            batting_snapshot
        ),
        int(
            batting_snapshot.duplicated(
                subset=list(
                    BATTING_SNAPSHOT_KEY
                )
            ).sum()
        ),
        int(
            batting_source["pa"].sum()
        ),
        int(
            batting_snapshot["pa"].sum()
        ),
        int(
            batting_source["h"].sum()
        ),
        int(
            batting_snapshot["h"].sum()
        ),
        int(
            batting_source["ab"].sum()
        ),
        int(
            batting_snapshot["ab"].sum()
        ),
        int(
            batting_source["tb"].sum()
        ),
        int(
            batting_snapshot["tb"].sum()
        ),
    )

    source_out_null_groups = int(
        pitching_source.groupby(
            list(
                PITCHING_SNAPSHOT_KEY
            ),
            sort=True,
            dropna=False,
        )["outs_recorded"]
        .apply(
            lambda values: (
                values.isna().any()
            )
        )
        .sum()
    )

    LOGGER.info(
        "시즌별 Player Season Pitching Row Count: %s",
        _season_row_count_text(
            pitching_snapshot
        ),
    )

    LOGGER.info(
        "Pitching Validation | "
        "output_rows=%d | duplicate_keys=%d | "
        "source_pitch_rows_sum=%d | snapshot_pitch_rows_sum=%d | "
        "source_pitches_sum=%d | snapshot_pitches_sum=%d | "
        "completed_bf_reconciliation_errors=0 | "
        "allowed_event_reconciliation_errors=0 | "
        "pitch_type_reconciliation_errors=0 | "
        "source_outs_na_player_seasons=%d | "
        "snapshot_outs_na_player_seasons=%d | "
        "outs_null_propagation_errors=0",
        len(
            pitching_snapshot
        ),
        int(
            pitching_snapshot.duplicated(
                subset=list(
                    PITCHING_SNAPSHOT_KEY
                )
            ).sum()
        ),
        int(
            pitching_source[
                "pitch_rows"
            ].sum()
        ),
        int(
            pitching_snapshot[
                "pitch_rows"
            ].sum()
        ),
        int(
            pitching_source[
                "pitches"
            ].sum()
        ),
        int(
            pitching_snapshot[
                "pitches"
            ].sum()
        ),
        source_out_null_groups,
        int(
            pitching_snapshot[
                "outs_recorded"
            ]
            .isna()
            .sum()
        ),
    )

    LOGGER.info(
        "시즌별 Team Snapshot Row Count: %s",
        _season_row_count_text(
            team_snapshot
        ),
    )

    LOGGER.info(
        "Team Validation | "
        "output_rows=%d | duplicate_keys=%d | "
        "source_team_game_rows=%d | snapshot_games_sum=%d | "
        "wlt_reconciliation_errors=0 | "
        "runs_reconciliation_errors=0 | "
        "run_diff_formula_errors=0 | "
        "games_result_formula_errors=0",
        len(
            team_snapshot
        ),
        int(
            team_snapshot.duplicated(
                subset=list(
                    TEAM_SNAPSHOT_KEY
                )
            ).sum()
        ),
        len(
            team_source
        ),
        int(
            team_snapshot[
                "games"
            ].sum()
        ),
    )


def build_season_snapshots_file(
    *,
    batting_path: Path = DEFAULT_BATTING_PATH,
    pitching_path: Path = DEFAULT_PITCHING_PATH,
    team_games_path: Path = DEFAULT_TEAM_GAMES_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """세 Canonical Parquet을 읽어 Season Snapshot 세 개를 안전하게 생성한다."""
    batting_output_path = (
        output_dir
        / BATTING_SNAPSHOT_FILENAME
    )

    pitching_output_path = (
        output_dir
        / PITCHING_SNAPSHOT_FILENAME
    )

    team_output_path = (
        output_dir
        / TEAM_SNAPSHOT_FILENAME
    )

    ensure_output_paths_safe(
        batting_path=batting_path,
        pitching_path=pitching_path,
        team_games_path=team_games_path,
        batting_output_path=batting_output_path,
        pitching_output_path=pitching_output_path,
        team_output_path=team_output_path,
    )

    input_paths = {
        "Player Game Batting": batting_path,
        "Player Game Pitching": pitching_path,
        "Team Game": team_games_path,
    }

    for source_name, path in input_paths.items():
        if not path.is_file():
            raise SeasonSnapshotsBuildError(
                f"{source_name} 파일이 없습니다: {path}"
            )

    hashes_before = {
        source_name: calculate_sha256(
            path
        )
        for source_name, path
        in input_paths.items()
    }

    for source_name, path in input_paths.items():
        LOGGER.info(
            "Input SHA256 BEFORE | source=%s | path=%s | sha256=%s",
            source_name,
            path,
            hashes_before[
                source_name
            ],
        )

    batting_raw = _read_parquet(
        batting_path,
        source_name="Player Game Batting",
    )

    pitching_raw = _read_parquet(
        pitching_path,
        source_name="Player Game Pitching",
    )

    team_raw = _read_parquet(
        team_games_path,
        source_name="Team Game",
    )

    (
        batting_snapshot,
        pitching_snapshot,
        team_snapshot,
    ) = build_season_snapshots(
        batting_raw,
        pitching_raw,
        team_raw,
    )

    batting_source = (
        prepare_batting_source(
            batting_raw
        )
    )

    pitching_source = (
        prepare_pitching_source(
            pitching_raw
        )
    )

    team_source = (
        prepare_team_games_source(
            team_raw
        )
    )

    log_production_validation(
        batting_source=batting_source,
        pitching_source=pitching_source,
        team_source=team_source,
        batting_snapshot=batting_snapshot,
        pitching_snapshot=pitching_snapshot,
        team_snapshot=team_snapshot,
    )

    # 세 DataFrame의 Build와 Validation이 모두 끝난 뒤에만 저장한다.
    _write_snapshot_atomically(
        batting_snapshot,
        batting_output_path,
        kind="batting",
    )

    _write_snapshot_atomically(
        pitching_snapshot,
        pitching_output_path,
        kind="pitching",
    )

    _write_snapshot_atomically(
        team_snapshot,
        team_output_path,
        kind="team",
    )

    for source_name, path in input_paths.items():
        hash_after = calculate_sha256(
            path
        )

        LOGGER.info(
            "Input SHA256 AFTER | source=%s | path=%s | sha256=%s",
            source_name,
            path,
            hash_after,
        )

        if (
            hashes_before[
                source_name
            ]
            != hash_after
        ):
            raise SeasonSnapshotsBuildError(
                f"{source_name} 입력 파일이 Build 중 변경되었습니다: "
                f"{path}"
            )

    LOGGER.info(
        "Season Snapshot 생성 완료 | "
        "batting=%s | pitching=%s | team=%s",
        batting_output_path,
        pitching_output_path,
        team_output_path,
    )

    return (
        batting_snapshot,
        pitching_snapshot,
        team_snapshot,
    )


def main() -> None:
    """CLI 진입점."""
    configure_logging()
    args = parse_args()

    try:
        build_season_snapshots_file(
            batting_path=args.batting_path,
            pitching_path=args.pitching_path,
            team_games_path=args.team_games_path,
            output_dir=args.output_dir,
        )
    except SeasonSnapshotsBuildError as exc:
        LOGGER.error(
            "%s",
            exc,
        )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()