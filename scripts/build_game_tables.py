from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path

import pandas as pd
from pandas.api.types import is_bool_dtype


LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "hf_kbo_pbp"
    / "derived"
    / "plate_appearances.parquet"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "hf_kbo_pbp"
    / "derived"
)

GAMES_FILENAME = "games.parquet"
TEAM_GAMES_FILENAME = "team_games.parquet"

PA_KEY = (
    "game_pk",
    "at_bat_number",
)

GAME_KEY = (
    "game_pk",
)

TEAM_GAME_KEY = (
    "game_pk",
    "team",
)

REQUIRED_PA_COLUMNS = (
    "game_pk",
    "game_date",
    "season",
    "at_bat_number",
    "inning",
    "batting_team",
    "fielding_team",
    "is_home_batting",
    "home_score_before",
    "away_score_before",
    "runs_scored",
    "post_home_score",
    "post_away_score",
    "pitch_rows",
    "pitch_count",
)

PA_STRING_COLUMNS = (
    "game_pk",
    "batting_team",
    "fielding_team",
)

PA_INTEGER_COLUMNS = (
    "season",
    "at_bat_number",
    "inning",
    "home_score_before",
    "away_score_before",
    "runs_scored",
    "post_home_score",
    "post_away_score",
    "pitch_rows",
    "pitch_count",
)

GAME_CONTEXT_COLUMNS = (
    "game_date",
    "season",
    "home_team",
    "away_team",
)

GAMES_COLUMNS = (
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
    "winner_team",
    "loser_team",
    "innings_played",
    "pitch_rows",
    "pitch_count",
    "plate_appearances",
    "home_plate_appearances",
    "away_plate_appearances",
    "total_runs",
)

GAMES_STRING_COLUMNS = (
    "game_pk",
    "home_team",
    "away_team",
    "winner_team",
    "loser_team",
)

GAMES_INTEGER_COLUMNS = (
    "season",
    "final_home_score",
    "final_away_score",
    "innings_played",
    "pitch_rows",
    "pitch_count",
    "plate_appearances",
    "home_plate_appearances",
    "away_plate_appearances",
    "total_runs",
)

GAMES_BOOLEAN_COLUMNS = (
    "home_win",
    "away_win",
    "is_tie",
)

TEAM_GAMES_COLUMNS = (
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

TEAM_GAMES_STRING_COLUMNS = (
    "game_pk",
    "team",
    "opponent",
)

TEAM_GAMES_INTEGER_COLUMNS = (
    "season",
    "runs_for",
    "runs_against",
    "run_diff",
    "plate_appearances",
    "opponent_plate_appearances",
)

TEAM_GAMES_BOOLEAN_COLUMNS = (
    "is_home",
    "win",
    "loss",
    "tie",
)


class GameTableBuildError(RuntimeError):
    """Game 및 Team Game 파생 테이블 생성을 중단해야 하는 오류를 나타낸다."""


def parse_args() -> argparse.Namespace:
    """Game 및 Team Game 생성 스크립트의 CLI 인자를 파싱한다."""
    parser = argparse.ArgumentParser(
        description=(
            "Canonical Plate Appearance Parquet에서 "
            "Game 및 Team Game 파생 테이블을 생성합니다."
        )
    )

    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=(
            "Canonical Plate Appearance Parquet 경로입니다. "
            f"기본값: {DEFAULT_INPUT_PATH}"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Game 파생 테이블 출력 디렉터리입니다. "
            f"기본값: {DEFAULT_OUTPUT_DIR}"
        ),
    )

    return parser.parse_args()


def configure_logging() -> None:
    """독립 실행 시 사용할 기본 logging 형식을 설정한다."""
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
    )


def coerce_integer_column(
    series: pd.Series,
    column: str,
) -> pd.Series:
    """정수 의미의 컬럼을 손실 없이 nullable Int64로 정규화한다."""
    try:
        numeric = pd.to_numeric(
            series,
            errors="raise",
        )
    except (TypeError, ValueError) as exc:
        raise GameTableBuildError(
            f"{column} 컬럼을 숫자로 변환할 수 없습니다."
        ) from exc

    if numeric.isna().any():
        raise GameTableBuildError(
            f"{column} 컬럼에 결측값이 있습니다."
        )

    if numeric.mod(1).ne(0).any():
        raise GameTableBuildError(
            f"{column} 컬럼에 정수가 아닌 값이 있습니다."
        )

    return numeric.astype("Int64")


def normalize_game_date(
    series: pd.Series,
    context: str,
) -> pd.Series:
    """game_date를 timezone 없는 datetime64[us] 날짜로 정규화한다."""
    parsed = pd.to_datetime(
        series,
        errors="coerce",
        utc=True,
    )

    invalid_count = int(
        parsed.isna().sum()
    )

    if invalid_count > 0:
        raise GameTableBuildError(
            f"{context} game_date 변환 실패가 "
            f"{invalid_count}건 있습니다."
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
        raise GameTableBuildError(
            f"{context} game_date를 "
            "datetime64[us]로 정규화할 수 없습니다."
        ) from exc


def validate_non_negative(
    df: pd.DataFrame,
    columns: tuple[str, ...],
) -> None:
    """Score 및 Count 계열에 음수가 존재하지 않는지 검증한다."""
    for column in columns:
        invalid_count = int(
            df[column]
            .lt(0)
            .sum()
        )

        if invalid_count > 0:
            raise GameTableBuildError(
                f"{column} 컬럼에 음수 값이 "
                f"{invalid_count}건 있습니다."
            )


def restore_home_away_teams(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """PA 공격/수비 관점에서 경기의 Home/Away Team을 복원한다."""
    result = df.copy()

    is_home_batting = (
        result["is_home_batting"]
        .astype("boolean")
    )

    result["home_team"] = (
        result["batting_team"]
        .where(
            is_home_batting,
            result["fielding_team"],
        )
        .astype("string")
    )

    result["away_team"] = (
        result["fielding_team"]
        .where(
            is_home_batting,
            result["batting_team"],
        )
        .astype("string")
    )

    same_team_mask = (
        result["home_team"]
        .eq(
            result["away_team"]
        )
        .fillna(False)
    )

    same_team_count = int(
        same_team_mask.sum()
    )

    if same_team_count > 0:
        raise GameTableBuildError(
            "복원된 home_team과 away_team이 동일한 PA가 "
            f"{same_team_count}건 있습니다."
        )

    return result


def validate_game_context(
    df: pd.DataFrame,
) -> None:
    """각 game_pk의 날짜, 시즌, Home/Away Team이 하나로 일관되는지 검증한다."""
    for column in GAME_CONTEXT_COLUMNS:
        counts = (
            df.groupby(
                "game_pk",
                sort=False,
                dropna=False,
            )[column]
            .nunique(
                dropna=False
            )
        )

        invalid = counts.loc[
            counts.ne(1)
        ]

        if not invalid.empty:
            sample_game_pks = (
                invalid
                .index
                .astype(str)
                .tolist()[:5]
            )

            raise GameTableBuildError(
                f"game_pk별 {column} 값이 일관되지 않습니다. "
                f"오류 경기 수={len(invalid)}, "
                f"예시={sample_game_pks}"
            )


def prepare_plate_appearances(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Canonical PA의 필수 Schema와 무결성을 검증하고
    Game Table 생성용 결정적 순서로 정규화한다.
    """
    missing_columns = [
        column
        for column in REQUIRED_PA_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise GameTableBuildError(
            "Game Table 생성에 필요한 "
            "Plate Appearance 컬럼이 없습니다: "
            f"{missing_columns}"
        )

    if df.empty:
        raise GameTableBuildError(
            "입력 Plate Appearance DataFrame이 비어 있습니다."
        )

    working = df.loc[
        :,
        REQUIRED_PA_COLUMNS,
    ].copy()

    for column in REQUIRED_PA_COLUMNS:
        null_count = int(
            working[column]
            .isna()
            .sum()
        )

        if null_count > 0:
            raise GameTableBuildError(
                f"{column} 컬럼에 "
                f"{null_count}개의 결측값이 있습니다."
            )

    for column in PA_STRING_COLUMNS:
        working[column] = (
            working[column]
            .astype("string")
        )

    for column in PA_INTEGER_COLUMNS:
        working[column] = (
            coerce_integer_column(
                working[column],
                column,
            )
        )

    if not is_bool_dtype(
        working["is_home_batting"].dtype
    ):
        raise GameTableBuildError(
            "is_home_batting 컬럼은 "
            "boolean dtype이어야 합니다."
        )

    working["is_home_batting"] = (
        working["is_home_batting"]
        .astype("boolean")
    )

    working["game_date"] = (
        normalize_game_date(
            working["game_date"],
            context="Plate Appearance",
        )
    )

    if working["at_bat_number"].le(0).any():
        raise GameTableBuildError(
            "at_bat_number는 1 이상이어야 합니다."
        )

    if working["inning"].le(0).any():
        raise GameTableBuildError(
            "inning은 1 이상이어야 합니다."
        )

    validate_non_negative(
        working,
        (
            "home_score_before",
            "away_score_before",
            "runs_scored",
            "post_home_score",
            "post_away_score",
            "pitch_rows",
            "pitch_count",
        ),
    )

    invalid_pitch_summary = (
        working["pitch_rows"]
        .lt(
            working["pitch_count"]
        )
    )

    if invalid_pitch_summary.any():
        raise GameTableBuildError(
            "pitch_rows보다 pitch_count가 큰 "
            "Plate Appearance가 있습니다."
        )

    duplicate_count = int(
        working.duplicated(
            subset=list(PA_KEY),
            keep=False,
        ).sum()
    )

    if duplicate_count > 0:
        raise GameTableBuildError(
            "입력 Plate Appearance Key "
            "(game_pk, at_bat_number) 중복 Row가 "
            f"{duplicate_count}건 있습니다."
        )

    working = (
        working
        .sort_values(
            by=[
                "game_pk",
                "at_bat_number",
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    working = restore_home_away_teams(
        working
    )

    validate_game_context(
        working
    )

    return working


def select_game_boundary_rows(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    at_bat_number로 정렬된 PA에서 각 경기의
    실제 첫 PA Row와 실제 마지막 PA Row를 선택한다.

    nullable 컬럼별 마지막 non-null 값을 조합할 수 있는
    groupby().last()는 사용하지 않는다.
    """
    first_rows = (
        df.drop_duplicates(
            subset=list(GAME_KEY),
            keep="first",
        )
        .copy()
    )

    last_rows = (
        df.drop_duplicates(
            subset=list(GAME_KEY),
            keep="last",
        )
        .copy()
    )

    return (
        first_rows,
        last_rows,
    )


def build_game_aggregation(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    검증·정규화된 PA에서 Game 단위 Context, Final State,
    Count 및 Score Reconciliation 값을 생성한다.
    """
    first_rows, last_rows = (
        select_game_boundary_rows(
            df
        )
    )

    first_projection = (
        first_rows.loc[
            :,
            [
                "game_pk",
                "game_date",
                "season",
                "home_team",
                "away_team",
                "home_score_before",
                "away_score_before",
            ],
        ]
        .rename(
            columns={
                "home_score_before": (
                    "_first_home_score_before"
                ),
                "away_score_before": (
                    "_first_away_score_before"
                ),
            }
        )
    )

    last_projection = (
        last_rows.loc[
            :,
            [
                "game_pk",
                "post_home_score",
                "post_away_score",
            ],
        ]
        .rename(
            columns={
                "post_home_score": "final_home_score",
                "post_away_score": "final_away_score",
            }
        )
    )

    summary_source = df.loc[
        :,
        [
            "game_pk",
            "at_bat_number",
            "inning",
            "is_home_batting",
            "runs_scored",
            "pitch_rows",
            "pitch_count",
        ],
    ].copy()

    is_home_batting = (
        summary_source["is_home_batting"]
        .astype("boolean")
    )

    summary_source["_home_runs"] = (
        summary_source["runs_scored"]
        .where(
            is_home_batting,
            0,
        )
    )

    summary_source["_away_runs"] = (
        summary_source["runs_scored"]
        .where(
            ~is_home_batting,
            0,
        )
    )

    summary_source["_home_pa"] = (
        is_home_batting
        .astype("Int64")
    )

    summary_source["_away_pa"] = (
        (~is_home_batting)
        .astype("Int64")
    )

    summary = (
        summary_source
        .groupby(
            "game_pk",
            sort=False,
            dropna=False,
        )
        .agg(
            innings_played=(
                "inning",
                "max",
            ),
            pitch_rows=(
                "pitch_rows",
                "sum",
            ),
            pitch_count=(
                "pitch_count",
                "sum",
            ),
            plate_appearances=(
                "at_bat_number",
                "size",
            ),
            home_plate_appearances=(
                "_home_pa",
                "sum",
            ),
            away_plate_appearances=(
                "_away_pa",
                "sum",
            ),
            _home_runs=(
                "_home_runs",
                "sum",
            ),
            _away_runs=(
                "_away_runs",
                "sum",
            ),
        )
        .reset_index()
    )

    aggregated = (
        first_projection
        .merge(
            last_projection,
            on="game_pk",
            how="inner",
            validate="one_to_one",
            sort=False,
        )
        .merge(
            summary,
            on="game_pk",
            how="inner",
            validate="one_to_one",
            sort=False,
        )
    )

    aggregated[
        "_expected_final_home_score"
    ] = (
        aggregated[
            "_first_home_score_before"
        ]
        + aggregated["_home_runs"]
    )

    aggregated[
        "_expected_final_away_score"
    ] = (
        aggregated[
            "_first_away_score_before"
        ]
        + aggregated["_away_runs"]
    )

    home_mismatch = (
        aggregated["final_home_score"]
        .ne(
            aggregated[
                "_expected_final_home_score"
            ]
        )
        .fillna(True)
    )

    away_mismatch = (
        aggregated["final_away_score"]
        .ne(
            aggregated[
                "_expected_final_away_score"
            ]
        )
        .fillna(True)
    )

    reconciliation_mismatch = (
        home_mismatch
        | away_mismatch
    )

    mismatch_count = int(
        reconciliation_mismatch.sum()
    )

    if mismatch_count > 0:
        sample = (
            aggregated.loc[
                reconciliation_mismatch,
                [
                    "game_pk",
                    "final_home_score",
                    "_expected_final_home_score",
                    "final_away_score",
                    "_expected_final_away_score",
                ],
            ]
            .head(5)
            .to_dict(
                orient="records"
            )
        )

        raise GameTableBuildError(
            "Final Score와 runs_scored 기반 Score "
            "Reconciliation이 일치하지 않습니다. "
            f"오류 경기 수={mismatch_count}, "
            f"예시={sample}"
        )

    return aggregated


def normalize_games_dtypes(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """games Output의 컬럼 순서와 dtype 및 정렬을 고정한다."""
    missing_columns = [
        column
        for column in GAMES_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise GameTableBuildError(
            "games Output Schema에 필요한 컬럼이 없습니다: "
            f"{missing_columns}"
        )

    result = df.loc[
        :,
        GAMES_COLUMNS,
    ].copy()

    result["game_date"] = (
        normalize_game_date(
            result["game_date"],
            context="games Output",
        )
    )

    for column in GAMES_STRING_COLUMNS:
        result[column] = (
            result[column]
            .astype("string")
        )

    for column in GAMES_INTEGER_COLUMNS:
        result[column] = (
            coerce_integer_column(
                result[column],
                column,
            )
        )

    for column in GAMES_BOOLEAN_COLUMNS:
        try:
            result[column] = (
                result[column]
                .astype("boolean")
            )
        except (TypeError, ValueError) as exc:
            raise GameTableBuildError(
                f"games Output {column} 컬럼을 "
                "boolean으로 변환할 수 없습니다."
            ) from exc

    result = (
        result
        .sort_values(
            by=[
                "season",
                "game_date",
                "game_pk",
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    return result


def validate_games_output(
    source: pd.DataFrame,
    games: pd.DataFrame,
) -> None:
    """Canonical PA와 games 사이의 Grain 및 집계 불변식을 검증한다."""
    expected_game_count = int(
        source["game_pk"]
        .nunique(
            dropna=False
        )
    )

    if len(games) != expected_game_count:
        raise GameTableBuildError(
            "games Row 수가 입력 PA의 unique game_pk 수와 "
            "일치하지 않습니다: "
            f"expected={expected_game_count}, "
            f"actual={len(games)}"
        )

    duplicate_count = int(
        games.duplicated(
            subset=list(GAME_KEY),
            keep=False,
        ).sum()
    )

    if duplicate_count > 0:
        raise GameTableBuildError(
            "games.game_pk 중복 Row가 "
            f"{duplicate_count}건 있습니다."
        )

    pa_count_mismatch = (
        games["plate_appearances"]
        .ne(
            games["home_plate_appearances"]
            + games["away_plate_appearances"]
        )
    )

    if pa_count_mismatch.any():
        raise GameTableBuildError(
            "plate_appearances가 Home/Away PA 합과 "
            "일치하지 않는 경기가 있습니다."
        )

    total_run_mismatch = (
        games["total_runs"]
        .ne(
            games["final_home_score"]
            + games["final_away_score"]
        )
    )

    if total_run_mismatch.any():
        raise GameTableBuildError(
            "total_runs가 Final Home/Away Score 합과 "
            "일치하지 않는 경기가 있습니다."
        )

    outcome_count = (
        games[
            [
                "home_win",
                "away_win",
                "is_tie",
            ]
        ]
        .astype("Int64")
        .sum(
            axis=1
        )
    )

    if outcome_count.ne(1).any():
        raise GameTableBuildError(
            "home_win + away_win + is_tie가 "
            "1이 아닌 경기가 있습니다."
        )

    tie_mask = (
        games["is_tie"]
        .astype("boolean")
    )

    invalid_tie_team = (
        tie_mask
        & (
            games["winner_team"].notna()
            | games["loser_team"].notna()
        )
    )

    if invalid_tie_team.any():
        raise GameTableBuildError(
            "Tie 경기의 winner_team 또는 loser_team이 "
            "null이 아닙니다."
        )

    non_tie_mask = ~tie_mask

    invalid_decision_team = (
        non_tie_mask
        & (
            games["winner_team"].isna()
            | games["loser_team"].isna()
        )
    )

    if invalid_decision_team.any():
        raise GameTableBuildError(
            "승패가 결정된 경기의 winner_team 또는 "
            "loser_team이 null입니다."
        )

    aggregated = (
        build_game_aggregation(
            source
        )
    )

    validation_frame = (
        games.merge(
            aggregated.loc[
                :,
                [
                    "game_pk",
                    "final_home_score",
                    "final_away_score",
                    "_expected_final_home_score",
                    "_expected_final_away_score",
                    "innings_played",
                    "pitch_rows",
                    "pitch_count",
                    "plate_appearances",
                    "home_plate_appearances",
                    "away_plate_appearances",
                ],
            ],
            on="game_pk",
            how="left",
            validate="one_to_one",
            suffixes=(
                "",
                "_expected",
            ),
            sort=False,
        )
    )

    compare_columns = (
        "final_home_score",
        "final_away_score",
        "innings_played",
        "pitch_rows",
        "pitch_count",
        "plate_appearances",
        "home_plate_appearances",
        "away_plate_appearances",
    )

    for column in compare_columns:
        expected_column = (
            f"{column}_expected"
        )

        if (
            validation_frame[column]
            .ne(
                validation_frame[
                    expected_column
                ]
            )
            .any()
        ):
            raise GameTableBuildError(
                f"games.{column}이 입력 PA 집계와 "
                "일치하지 않습니다."
            )

    final_home_reconciliation = (
        validation_frame[
            "final_home_score"
        ]
        .ne(
            validation_frame[
                "_expected_final_home_score"
            ]
        )
    )

    final_away_reconciliation = (
        validation_frame[
            "final_away_score"
        ]
        .ne(
            validation_frame[
                "_expected_final_away_score"
            ]
        )
    )

    if (
        final_home_reconciliation.any()
        or final_away_reconciliation.any()
    ):
        raise GameTableBuildError(
            "games Final Score가 runs_scored 기반 "
            "Reconciliation 결과와 일치하지 않습니다."
        )


def build_games(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Canonical PA DataFrame을 game_pk Grain의 games로 변환한다."""
    source = prepare_plate_appearances(
        df
    )

    aggregated = (
        build_game_aggregation(
            source
        )
    )

    games = aggregated.copy()

    games["home_win"] = (
        games["final_home_score"]
        .gt(
            games["final_away_score"]
        )
    )

    games["away_win"] = (
        games["final_away_score"]
        .gt(
            games["final_home_score"]
        )
    )

    games["is_tie"] = (
        games["final_home_score"]
        .eq(
            games["final_away_score"]
        )
    )

    games["winner_team"] = pd.Series(
        pd.NA,
        index=games.index,
        dtype="string",
    )

    games["loser_team"] = pd.Series(
        pd.NA,
        index=games.index,
        dtype="string",
    )

    home_win_mask = (
        games["home_win"]
        .astype(bool)
    )

    away_win_mask = (
        games["away_win"]
        .astype(bool)
    )

    games.loc[
        home_win_mask,
        "winner_team",
    ] = games.loc[
        home_win_mask,
        "home_team",
    ]

    games.loc[
        home_win_mask,
        "loser_team",
    ] = games.loc[
        home_win_mask,
        "away_team",
    ]

    games.loc[
        away_win_mask,
        "winner_team",
    ] = games.loc[
        away_win_mask,
        "away_team",
    ]

    games.loc[
        away_win_mask,
        "loser_team",
    ] = games.loc[
        away_win_mask,
        "home_team",
    ]

    games["total_runs"] = (
        games["final_home_score"]
        + games["final_away_score"]
    )

    result = normalize_games_dtypes(
        games
    )

    validate_games_output(
        source=source,
        games=result,
    )

    return result


def normalize_team_games_dtypes(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """team_games Output의 컬럼 순서와 dtype 및 정렬을 고정한다."""
    missing_columns = [
        column
        for column in TEAM_GAMES_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise GameTableBuildError(
            "team_games Output Schema에 필요한 컬럼이 없습니다: "
            f"{missing_columns}"
        )

    result = df.loc[
        :,
        TEAM_GAMES_COLUMNS,
    ].copy()

    result["game_date"] = (
        normalize_game_date(
            result["game_date"],
            context="team_games Output",
        )
    )

    for column in TEAM_GAMES_STRING_COLUMNS:
        result[column] = (
            result[column]
            .astype("string")
        )

    for column in TEAM_GAMES_INTEGER_COLUMNS:
        result[column] = (
            coerce_integer_column(
                result[column],
                column,
            )
        )

    for column in TEAM_GAMES_BOOLEAN_COLUMNS:
        try:
            result[column] = (
                result[column]
                .astype("boolean")
            )
        except (TypeError, ValueError) as exc:
            raise GameTableBuildError(
                f"team_games Output {column} 컬럼을 "
                "boolean으로 변환할 수 없습니다."
            ) from exc

    result = (
        result
        .sort_values(
            by=[
                "season",
                "game_date",
                "game_pk",
                "is_home",
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    return result


def validate_team_games_output(
    games: pd.DataFrame,
    team_games: pd.DataFrame,
) -> None:
    """games와 team_games 사이의 2행 Mirror 불변식을 검증한다."""
    expected_row_count = (
        2
        * len(games)
    )

    if len(team_games) != expected_row_count:
        raise GameTableBuildError(
            "team_games Row 수가 games의 2배가 아닙니다: "
            f"expected={expected_row_count}, "
            f"actual={len(team_games)}"
        )

    duplicate_count = int(
        team_games.duplicated(
            subset=list(TEAM_GAME_KEY),
            keep=False,
        ).sum()
    )

    if duplicate_count > 0:
        raise GameTableBuildError(
            "team_games (game_pk, team) 중복 Row가 "
            f"{duplicate_count}건 있습니다."
        )

    rows_per_game = (
        team_games.groupby(
            "game_pk",
            sort=False,
            dropna=False,
        )
        .size()
    )

    invalid_row_count = int(
        rows_per_game
        .ne(2)
        .sum()
    )

    if invalid_row_count > 0:
        raise GameTableBuildError(
            "team_games가 정확히 2행이 아닌 경기가 "
            f"{invalid_row_count}건 있습니다."
        )

    home_rows_per_game = (
        team_games.groupby(
            "game_pk",
            sort=False,
            dropna=False,
        )["is_home"]
        .sum()
    )

    invalid_home_count = int(
        home_rows_per_game
        .ne(1)
        .sum()
    )

    if invalid_home_count > 0:
        raise GameTableBuildError(
            "team_games의 Home Row가 정확히 1개가 아닌 "
            f"경기가 {invalid_home_count}건 있습니다."
        )

    same_team_mask = (
        team_games["team"]
        .eq(
            team_games["opponent"]
        )
        .fillna(False)
    )

    if same_team_mask.any():
        raise GameTableBuildError(
            "team_games에 team과 opponent가 "
            "동일한 Row가 있습니다."
        )

    merged = (
        team_games.merge(
            games.loc[
                :,
                [
                    "game_pk",
                    "home_team",
                    "away_team",
                    "final_home_score",
                    "final_away_score",
                    "home_win",
                    "away_win",
                    "is_tie",
                    "home_plate_appearances",
                    "away_plate_appearances",
                ],
            ],
            on="game_pk",
            how="left",
            validate="many_to_one",
            sort=False,
        )
    )

    is_home = (
        merged["is_home"]
        .astype("boolean")
    )

    expected_team = (
        merged["home_team"]
        .where(
            is_home,
            merged["away_team"],
        )
    )

    expected_opponent = (
        merged["away_team"]
        .where(
            is_home,
            merged["home_team"],
        )
    )

    expected_runs_for = (
        merged["final_home_score"]
        .where(
            is_home,
            merged["final_away_score"],
        )
    )

    expected_runs_against = (
        merged["final_away_score"]
        .where(
            is_home,
            merged["final_home_score"],
        )
    )

    expected_win = (
        merged["home_win"]
        .where(
            is_home,
            merged["away_win"],
        )
    )

    expected_loss = (
        merged["away_win"]
        .where(
            is_home,
            merged["home_win"],
        )
    )

    expected_pa = (
        merged["home_plate_appearances"]
        .where(
            is_home,
            merged["away_plate_appearances"],
        )
    )

    expected_opponent_pa = (
        merged["away_plate_appearances"]
        .where(
            is_home,
            merged["home_plate_appearances"],
        )
    )

    checks = (
        (
            "team",
            merged["team"],
            expected_team,
        ),
        (
            "opponent",
            merged["opponent"],
            expected_opponent,
        ),
        (
            "runs_for",
            merged["runs_for"],
            expected_runs_for,
        ),
        (
            "runs_against",
            merged["runs_against"],
            expected_runs_against,
        ),
        (
            "win",
            merged["win"],
            expected_win,
        ),
        (
            "loss",
            merged["loss"],
            expected_loss,
        ),
        (
            "tie",
            merged["tie"],
            merged["is_tie"],
        ),
        (
            "plate_appearances",
            merged["plate_appearances"],
            expected_pa,
        ),
        (
            "opponent_plate_appearances",
            merged[
                "opponent_plate_appearances"
            ],
            expected_opponent_pa,
        ),
    )

    for name, actual, expected in checks:
        mismatch = (
            actual
            .ne(
                expected
            )
            .fillna(True)
        )

        if mismatch.any():
            raise GameTableBuildError(
                f"team_games.{name}이 games Mirror 규칙과 "
                "일치하지 않습니다."
            )

    run_diff_mismatch = (
        merged["run_diff"]
        .ne(
            merged["runs_for"]
            - merged["runs_against"]
        )
    )

    if run_diff_mismatch.any():
        raise GameTableBuildError(
            "team_games.run_diff가 "
            "runs_for - runs_against와 일치하지 않습니다."
        )


def build_team_games(
    games: pd.DataFrame,
) -> pd.DataFrame:
    """검증된 games를 경기당 Home/Away 2행의 Long Format으로 변환한다."""
    normalized_games = (
        normalize_games_dtypes(
            games
        )
    )

    duplicate_game_count = int(
        normalized_games.duplicated(
            subset=list(GAME_KEY),
            keep=False,
        ).sum()
    )

    if duplicate_game_count > 0:
        raise GameTableBuildError(
            "team_games 생성 입력 games에 "
            f"game_pk 중복이 {duplicate_game_count}건 있습니다."
        )

    home_rows = pd.DataFrame(
        {
            "game_pk": (
                normalized_games["game_pk"]
            ),
            "game_date": (
                normalized_games["game_date"]
            ),
            "season": (
                normalized_games["season"]
            ),
            "team": (
                normalized_games["home_team"]
            ),
            "opponent": (
                normalized_games["away_team"]
            ),
            "is_home": True,
            "runs_for": (
                normalized_games[
                    "final_home_score"
                ]
            ),
            "runs_against": (
                normalized_games[
                    "final_away_score"
                ]
            ),
            "run_diff": (
                normalized_games[
                    "final_home_score"
                ]
                - normalized_games[
                    "final_away_score"
                ]
            ),
            "win": (
                normalized_games["home_win"]
            ),
            "loss": (
                normalized_games["away_win"]
            ),
            "tie": (
                normalized_games["is_tie"]
            ),
            "plate_appearances": (
                normalized_games[
                    "home_plate_appearances"
                ]
            ),
            "opponent_plate_appearances": (
                normalized_games[
                    "away_plate_appearances"
                ]
            ),
        }
    )

    away_rows = pd.DataFrame(
        {
            "game_pk": (
                normalized_games["game_pk"]
            ),
            "game_date": (
                normalized_games["game_date"]
            ),
            "season": (
                normalized_games["season"]
            ),
            "team": (
                normalized_games["away_team"]
            ),
            "opponent": (
                normalized_games["home_team"]
            ),
            "is_home": False,
            "runs_for": (
                normalized_games[
                    "final_away_score"
                ]
            ),
            "runs_against": (
                normalized_games[
                    "final_home_score"
                ]
            ),
            "run_diff": (
                normalized_games[
                    "final_away_score"
                ]
                - normalized_games[
                    "final_home_score"
                ]
            ),
            "win": (
                normalized_games["away_win"]
            ),
            "loss": (
                normalized_games["home_win"]
            ),
            "tie": (
                normalized_games["is_tie"]
            ),
            "plate_appearances": (
                normalized_games[
                    "away_plate_appearances"
                ]
            ),
            "opponent_plate_appearances": (
                normalized_games[
                    "home_plate_appearances"
                ]
            ),
        }
    )

    team_games = pd.concat(
        [
            home_rows,
            away_rows,
        ],
        ignore_index=True,
    )

    result = (
        normalize_team_games_dtypes(
            team_games
        )
    )

    validate_team_games_output(
        games=normalized_games,
        team_games=result,
    )

    return result


def build_game_tables(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Canonical PA에서 games와 team_games를 순서대로 생성한다."""
    games = build_games(
        df
    )

    team_games = build_team_games(
        games
    )

    return (
        games,
        team_games,
    )


def load_plate_appearances(
    input_path: Path,
) -> pd.DataFrame:
    """Canonical Plate Appearance Parquet을 읽는다."""
    if not input_path.is_file():
        raise GameTableBuildError(
            "Canonical Plate Appearance Parquet이 없습니다: "
            f"{input_path}"
        )

    try:
        return pd.read_parquet(
            input_path,
            engine="pyarrow",
        )
    except Exception as exc:
        raise GameTableBuildError(
            "Canonical Plate Appearance Parquet을 "
            f"읽지 못했습니다: {input_path}"
        ) from exc


def write_parquet_atomic(
    df: pd.DataFrame,
    output_path: Path,
    label: str,
) -> None:
    """
    DataFrame을 임시 Parquet에 먼저 작성한 뒤
    최종 경로로 원자적으로 교체한다.
    """
    if (
        output_path.exists()
        and output_path.is_dir()
    ):
        raise GameTableBuildError(
            f"{label} Output 경로가 "
            "파일이 아닌 디렉터리입니다: "
            f"{output_path}"
        )

    try:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
    except OSError as exc:
        raise GameTableBuildError(
            "Output 디렉터리를 생성할 수 없습니다: "
            f"{output_path.parent}"
        ) from exc

    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=(
                f".{output_path.stem}."
            ),
            suffix=".parquet",
            dir=output_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(
                temporary_file.name
            )

        df.to_parquet(
            temporary_path,
            engine="pyarrow",
            index=False,
        )

        temporary_path.replace(
            output_path
        )

    except Exception as exc:
        raise GameTableBuildError(
            f"{label} Parquet 저장에 실패했습니다: "
            f"{output_path}"
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
                    "임시 Parquet 파일을 삭제하지 못했습니다: %s",
                    temporary_path,
                )


def log_validation_summary(
    games: pd.DataFrame,
    team_games: pd.DataFrame,
) -> None:
    """성공한 Production Validation의 핵심 결과를 로그에 남긴다."""
    season_counts = (
        games.groupby(
            "season",
            sort=True,
            dropna=False,
        )
        .size()
    )

    season_count_text = ", ".join(
        f"{int(season)}={int(count)}"
        for season, count
        in season_counts.items()
    )

    games_duplicate_count = int(
        games.duplicated(
            subset=list(GAME_KEY),
            keep=False,
        ).sum()
    )

    team_games_duplicate_count = int(
        team_games.duplicated(
            subset=list(TEAM_GAME_KEY),
            keep=False,
        ).sum()
    )

    invalid_team_game_count = int(
        team_games.groupby(
            "game_pk",
            sort=False,
        )
        .size()
        .ne(2)
        .sum()
    )

    pa_count_error_count = int(
        games["plate_appearances"]
        .ne(
            games["home_plate_appearances"]
            + games["away_plate_appearances"]
        )
        .sum()
    )

    tie_count = int(
        games["is_tie"]
        .sum()
    )

    LOGGER.info(
        "시즌별 Game Count: %s",
        season_count_text,
    )

    LOGGER.info(
        (
            "Production Validation 통과: "
            "games_rows=%d, "
            "games_game_pk_duplicates=%d, "
            "team_games_rows=%d, "
            "team_games_key_duplicates=%d, "
            "team_games_row_count_errors=%d, "
            "final_score_reconciliation_errors=0, "
            "pa_count_reconciliation_errors=%d, "
            "ties=%d"
        ),
        len(games),
        games_duplicate_count,
        len(team_games),
        team_games_duplicate_count,
        invalid_team_game_count,
        pa_count_error_count,
        tie_count,
    )


def build_game_table_files(
    input_path: Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """PA 읽기부터 Game Table 검증 및 Parquet 저장까지 수행한다."""
    if (
        output_dir.exists()
        and not output_dir.is_dir()
    ):
        raise GameTableBuildError(
            "Output 경로가 디렉터리가 아닙니다: "
            f"{output_dir}"
        )

    source = load_plate_appearances(
        input_path
    )

    games, team_games = (
        build_game_tables(
            source
        )
    )

    log_validation_summary(
        games=games,
        team_games=team_games,
    )

    games_path = (
        output_dir
        / GAMES_FILENAME
    )

    team_games_path = (
        output_dir
        / TEAM_GAMES_FILENAME
    )

    write_parquet_atomic(
        df=games,
        output_path=games_path,
        label="games",
    )

    write_parquet_atomic(
        df=team_games,
        output_path=team_games_path,
        label="team_games",
    )

    return (
        games,
        team_games,
    )


def main() -> None:
    """CLI 진입점."""
    configure_logging()
    args = parse_args()

    try:
        games, team_games = (
            build_game_table_files(
                input_path=args.input_path,
                output_dir=args.output_dir,
            )
        )
    except GameTableBuildError as exc:
        LOGGER.error(
            "%s",
            exc,
        )
        raise SystemExit(1) from exc

    LOGGER.info(
        (
            "Game 파생 테이블 생성 완료: "
            "games_rows=%d, "
            "team_games_rows=%d, "
            "games_path=%s, "
            "team_games_path=%s"
        ),
        len(games),
        len(team_games),
        args.output_dir / GAMES_FILENAME,
        args.output_dir / TEAM_GAMES_FILENAME,
    )


if __name__ == "__main__":
    main()