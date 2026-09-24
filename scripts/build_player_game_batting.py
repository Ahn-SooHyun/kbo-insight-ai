from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path

import pandas as pd
from pandas.api.types import is_bool_dtype


LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
)

DEFAULT_INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "hf_kbo_pbp"
    / "derived"
    / "plate_appearances.parquet"
)

DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "hf_kbo_pbp"
    / "derived"
    / "player_game_batting.parquet"
)

PA_KEY = (
    "game_pk",
    "at_bat_number",
)

PLAYER_GAME_KEY = (
    "game_pk",
    "batter",
)

REQUIRED_PA_COLUMNS = (
    "game_pk",
    "game_date",
    "season",
    "at_bat_number",
    "batting_team",
    "fielding_team",
    "is_home_batting",
    "batter",
    "batter_name",
    "event",
)

PA_STRING_COLUMNS = (
    "game_pk",
    "batting_team",
    "fielding_team",
    "batter",
    "batter_name",
    "event",
)

PA_INTEGER_COLUMNS = (
    "season",
    "at_bat_number",
)

EVENT_VALUES = (
    "single",
    "double",
    "triple",
    "home_run",
    "walk",
    "hit_by_pitch",
    "strikeout",
    "field_out",
    "double_play",
    "triple_play",
    "sac_bunt",
    "sac_fly",
    "field_error",
    "fielders_choice",
    "catcher_interference",
)

EVENT_TO_COUNT_COLUMN = {
    "single": "single",
    "double": "double",
    "triple": "triple",
    "home_run": "hr",
    "walk": "bb",
    "hit_by_pitch": "hbp",
    "strikeout": "so",
    "sac_fly": "sf",
    "sac_bunt": "sh",
    "double_play": "double_play",
    "triple_play": "triple_play",
    "field_error": "field_error",
    "fielders_choice": "fielders_choice",
    "catcher_interference": "catcher_interference",
}

PLAYER_GAME_CONTEXT_COLUMNS = (
    "game_date",
    "season",
    "batter_name",
    "team",
    "opponent",
    "is_home",
)

OUTPUT_COLUMNS = (
    "game_pk",
    "game_date",
    "season",
    "batter",
    "batter_name",
    "team",
    "opponent",
    "is_home",
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

STRING_OUTPUT_COLUMNS = (
    "game_pk",
    "batter",
    "batter_name",
    "team",
    "opponent",
)

INTEGER_OUTPUT_COLUMNS = (
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

BOOLEAN_OUTPUT_COLUMNS = (
    "is_home",
)

FLOAT_OUTPUT_COLUMNS = (
    "avg",
    "obp",
    "slg",
    "ops",
)

PROHIBITED_OUTPUT_COLUMNS = (
    "rbi",
    "runs",
    "batter_runs",
)


class PlayerGameBattingBuildError(RuntimeError):
    """Player Game Batting 파생 테이블 생성을 중단해야 하는 오류를 나타낸다."""


def parse_args() -> argparse.Namespace:
    """Player Game Batting 생성 스크립트의 CLI 인자를 파싱한다."""
    parser = argparse.ArgumentParser(
        description=(
            "Canonical Plate Appearance Parquet에서 "
            "Player Game Batting 파생 테이블을 생성합니다."
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
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=(
            "Player Game Batting Parquet 출력 경로입니다. "
            f"기본값: {DEFAULT_OUTPUT_PATH}"
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
        raise PlayerGameBattingBuildError(
            f"{column} 컬럼을 숫자로 변환할 수 없습니다."
        ) from exc

    if numeric.isna().any():
        raise PlayerGameBattingBuildError(
            f"{column} 컬럼에 결측값이 있습니다."
        )

    if numeric.mod(1).ne(0).any():
        raise PlayerGameBattingBuildError(
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
        raise PlayerGameBattingBuildError(
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
        raise PlayerGameBattingBuildError(
            f"{context} game_date를 "
            "datetime64[us]로 정규화할 수 없습니다."
        ) from exc


def ensure_non_empty_string(
    series: pd.Series,
    column: str,
) -> None:
    """식별자 및 팀 컬럼에 null 또는 빈 문자열이 없는지 검증한다."""
    normalized = (
        series
        .astype("string")
    )

    invalid_mask = (
        normalized.isna()
        | normalized.str.strip().eq("").fillna(False)
    )

    invalid_count = int(
        invalid_mask.sum()
    )

    if invalid_count > 0:
        raise PlayerGameBattingBuildError(
            f"{column} 컬럼에 null 또는 빈 문자열이 "
            f"{invalid_count}건 있습니다."
        )


def ensure_output_path_safe(
    input_path: Path,
    output_path: Path,
) -> None:
    """
    Canonical PA 덮어쓰기와 data/raw 내부 Derived Output 생성을 방지한다.
    """
    resolved_input = input_path.resolve()
    resolved_output = output_path.resolve()
    resolved_raw_dir = RAW_DATA_DIR.resolve()

    if resolved_output == resolved_input:
        raise PlayerGameBattingBuildError(
            "Output 경로는 입력 Plate Appearance Parquet과 "
            f"같을 수 없습니다: {output_path}"
        )

    try:
        resolved_output.relative_to(
            resolved_raw_dir
        )
    except ValueError:
        return

    raise PlayerGameBattingBuildError(
        "Player Game Batting Output은 data/raw 내부에 "
        f"생성할 수 없습니다: {output_path}"
    )


def prepare_plate_appearances(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Canonical PA의 최소 Schema와 무결성을 검증하고
    Player Game 집계용 결정적 순서로 정규화한다.
    """
    missing_columns = [
        column
        for column in REQUIRED_PA_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise PlayerGameBattingBuildError(
            "Player Game Batting 생성에 필요한 "
            "Plate Appearance 컬럼이 없습니다: "
            f"{missing_columns}"
        )

    if df.empty:
        raise PlayerGameBattingBuildError(
            "입력 Plate Appearance DataFrame이 비어 있습니다."
        )

    working = df.loc[
        :,
        REQUIRED_PA_COLUMNS,
    ].copy()

    non_null_columns = (
        "game_pk",
        "game_date",
        "season",
        "at_bat_number",
        "batting_team",
        "fielding_team",
        "is_home_batting",
        "batter",
    )

    for column in non_null_columns:
        null_count = int(
            working[column]
            .isna()
            .sum()
        )

        if null_count > 0:
            raise PlayerGameBattingBuildError(
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
        raise PlayerGameBattingBuildError(
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

    ensure_non_empty_string(
        working["game_pk"],
        "game_pk",
    )

    ensure_non_empty_string(
        working["batter"],
        "batter",
    )

    ensure_non_empty_string(
        working["batting_team"],
        "batting_team",
    )

    ensure_non_empty_string(
        working["fielding_team"],
        "fielding_team",
    )

    if working["at_bat_number"].le(0).any():
        raise PlayerGameBattingBuildError(
            "at_bat_number는 1 이상이어야 합니다."
        )

    duplicate_count = int(
        working.duplicated(
            subset=list(PA_KEY),
            keep=False,
        ).sum()
    )

    if duplicate_count > 0:
        raise PlayerGameBattingBuildError(
            "입력 Plate Appearance Key "
            "(game_pk, at_bat_number) 중복 Row가 "
            f"{duplicate_count}건 있습니다."
        )

    same_team_mask = (
        working["batting_team"]
        .eq(
            working["fielding_team"]
        )
        .fillna(False)
    )

    same_team_count = int(
        same_team_mask.sum()
    )

    if same_team_count > 0:
        raise PlayerGameBattingBuildError(
            "batting_team과 fielding_team이 동일한 "
            f"Plate Appearance가 {same_team_count}건 있습니다."
        )

    invalid_event_mask = (
        working["event"].notna()
        & ~working["event"].isin(
            EVENT_VALUES
        )
    )

    invalid_event_count = int(
        invalid_event_mask.sum()
    )

    if invalid_event_count > 0:
        invalid_values = sorted(
            {
                str(value)
                for value in working.loc[
                    invalid_event_mask,
                    "event",
                ].tolist()
            }
        )

        raise PlayerGameBattingBuildError(
            "완료 Plate Appearance에 지원하지 않는 "
            "event 값이 있습니다: "
            f"{invalid_values}"
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

    return working


def build_completed_plate_appearances(
    source: pd.DataFrame,
) -> pd.DataFrame:
    """
    공식 타격 집계 대상인 event가 존재하는 완료 PA만 선택하고
    Player Game Context 컬럼을 생성한다.
    """
    completed = (
        source.loc[
            source["event"].notna()
        ]
        .copy()
    )

    completed["team"] = (
        completed["batting_team"]
        .astype("string")
    )

    completed["opponent"] = (
        completed["fielding_team"]
        .astype("string")
    )

    completed["is_home"] = (
        completed["is_home_batting"]
        .astype("boolean")
    )

    return completed


def validate_player_game_context(
    completed: pd.DataFrame,
) -> None:
    """
    동일 (game_pk, batter) 안에서 날짜, 시즌, 선수명,
    Team/Opponent/Home Context가 하나로 일관되는지 검증한다.
    """
    if completed.empty:
        return

    ensure_non_empty_string(
        completed["batter"],
        "batter",
    )

    ensure_non_empty_string(
        completed["team"],
        "team",
    )

    ensure_non_empty_string(
        completed["opponent"],
        "opponent",
    )

    same_team_mask = (
        completed["team"]
        .eq(
            completed["opponent"]
        )
        .fillna(False)
    )

    if same_team_mask.any():
        raise PlayerGameBattingBuildError(
            "완료 Plate Appearance에 team과 opponent가 "
            "동일한 Row가 있습니다."
        )

    for column in PLAYER_GAME_CONTEXT_COLUMNS:
        counts = (
            completed.groupby(
                list(PLAYER_GAME_KEY),
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
            examples = [
                {
                    "game_pk": str(game_pk),
                    "batter": str(batter),
                }
                for game_pk, batter
                in invalid.index.tolist()[:5]
            ]

            raise PlayerGameBattingBuildError(
                "(game_pk, batter)별 "
                f"{column} 값이 일관되지 않습니다. "
                f"오류 Player Game 수={len(invalid)}, "
                f"예시={examples}"
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

    valid_mask = (
        denominator.notna()
        & denominator.ne(0)
    )

    result.loc[
        valid_mask
    ] = (
        numerator.loc[
            valid_mask
        ]
        .astype("Float64")
        .div(
            denominator.loc[
                valid_mask
            ]
            .astype("Float64")
        )
        .round(3)
    )

    return result


def calculate_ops(
    obp: pd.Series,
    slg: pd.Series,
) -> pd.Series:
    """OBP와 SLG가 모두 정의된 Row에서만 OPS를 계산한다."""
    result = pd.Series(
        pd.NA,
        index=obp.index,
        dtype="Float64",
    )

    valid_mask = (
        obp.notna()
        & slg.notna()
    )

    result.loc[
        valid_mask
    ] = (
        obp.loc[
            valid_mask
        ]
        .astype("Float64")
        .add(
            slg.loc[
                valid_mask
            ]
            .astype("Float64")
        )
        .round(3)
    )

    return result


def build_player_game_aggregation(
    completed: pd.DataFrame,
) -> pd.DataFrame:
    """완료 PA를 (game_pk, batter) Grain의 타격 집계로 변환한다."""
    if completed.empty:
        return pd.DataFrame(
            columns=OUTPUT_COLUMNS
        )

    validate_player_game_context(
        completed
    )

    context = (
        completed.loc[
            :,
            [
                "game_pk",
                "game_date",
                "season",
                "batter",
                "batter_name",
                "team",
                "opponent",
                "is_home",
            ],
        ]
        .drop_duplicates(
            subset=list(
                PLAYER_GAME_KEY
            ),
            keep="first",
        )
        .copy()
    )

    event_counts = (
        completed.groupby(
            [
                "game_pk",
                "batter",
                "event",
            ],
            sort=False,
            dropna=False,
        )
        .size()
        .unstack(
            "event",
            fill_value=0,
        )
        .reindex(
            columns=EVENT_VALUES,
            fill_value=0,
        )
        .reset_index()
    )

    event_counts["pa"] = (
        event_counts[
            list(EVENT_VALUES)
        ]
        .sum(
            axis=1
        )
    )

    for event, output_column in EVENT_TO_COUNT_COLUMN.items():
        event_counts[
            output_column
        ] = event_counts[
            event
        ]

    event_counts["h"] = (
        event_counts["single"]
        + event_counts["double"]
        + event_counts["triple"]
        + event_counts["hr"]
    )

    event_counts["ab"] = (
        event_counts["pa"]
        - event_counts["bb"]
        - event_counts["hbp"]
        - event_counts["sh"]
        - event_counts["sf"]
        - event_counts[
            "catcher_interference"
        ]
    )

    event_counts["tb"] = (
        event_counts["single"]
        + 2 * event_counts["double"]
        + 3 * event_counts["triple"]
        + 4 * event_counts["hr"]
    )

    stat_columns = [
        "game_pk",
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
    ]

    stats = (
        event_counts.loc[
            :,
            stat_columns,
        ]
        .copy()
    )

    result = (
        context.merge(
            stats,
            on=list(
                PLAYER_GAME_KEY
            ),
            how="inner",
            validate="one_to_one",
            sort=False,
        )
    )

    result["avg"] = calculate_rate(
        numerator=result["h"],
        denominator=result["ab"],
    )

    on_base_denominator = (
        result["ab"]
        + result["bb"]
        + result["hbp"]
        + result["sf"]
    )

    result["obp"] = calculate_rate(
        numerator=(
            result["h"]
            + result["bb"]
            + result["hbp"]
        ),
        denominator=on_base_denominator,
    )

    result["slg"] = calculate_rate(
        numerator=result["tb"],
        denominator=result["ab"],
    )

    result["ops"] = calculate_ops(
        obp=result["obp"],
        slg=result["slg"],
    )

    return result


def normalize_player_game_batting_dtypes(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Output 컬럼 순서, dtype 및 결정적 정렬을 고정한다."""
    missing_columns = [
        column
        for column in OUTPUT_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise PlayerGameBattingBuildError(
            "Player Game Batting Output Schema에 "
            "필요한 컬럼이 없습니다: "
            f"{missing_columns}"
        )

    result = df.loc[
        :,
        OUTPUT_COLUMNS,
    ].copy()

    result["game_date"] = (
        normalize_game_date(
            result["game_date"],
            context="Player Game Batting Output",
        )
    )

    for column in STRING_OUTPUT_COLUMNS:
        result[column] = (
            result[column]
            .astype("string")
        )

    for column in INTEGER_OUTPUT_COLUMNS:
        result[column] = (
            coerce_integer_column(
                result[column],
                column,
            )
        )

    for column in BOOLEAN_OUTPUT_COLUMNS:
        try:
            result[column] = (
                result[column]
                .astype("boolean")
            )
        except (TypeError, ValueError) as exc:
            raise PlayerGameBattingBuildError(
                f"{column} 컬럼을 boolean으로 "
                "변환할 수 없습니다."
            ) from exc

    for column in FLOAT_OUTPUT_COLUMNS:
        try:
            result[column] = (
                result[column]
                .astype("Float64")
            )
        except (TypeError, ValueError) as exc:
            raise PlayerGameBattingBuildError(
                f"{column} 컬럼을 Float64로 "
                "변환할 수 없습니다."
            ) from exc

    result = (
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
        .reset_index(
            drop=True
        )
    )

    return result


def nullable_float_series_equal(
    left: pd.Series,
    right: pd.Series,
) -> pd.Series:
    """nullable Float Series를 <NA>까지 포함해 Row 단위로 비교한다."""
    return (
        left.eq(
            right
        )
        | (
            left.isna()
            & right.isna()
        )
    ).fillna(False)


def validate_player_game_output(
    source: pd.DataFrame,
    output: pd.DataFrame,
) -> None:
    """
    Source 완료 PA와 Player Game Output 사이의
    Grain, Count, 공식식 및 Rate 불변식을 검증한다.
    """
    completed = (
        source.loc[
            source["event"].notna()
        ]
        .copy()
    )

    duplicate_count = int(
        output.duplicated(
            subset=list(
                PLAYER_GAME_KEY
            ),
            keep=False,
        ).sum()
    )

    if duplicate_count > 0:
        raise PlayerGameBattingBuildError(
            "Player Game Batting Key "
            "(game_pk, batter) 중복 Row가 "
            f"{duplicate_count}건 있습니다."
        )

    expected_keys = {
        tuple(row)
        for row in completed.loc[
            :,
            PLAYER_GAME_KEY,
        ].itertuples(
            index=False,
            name=None,
        )
    }

    actual_keys = {
        tuple(row)
        for row in output.loc[
            :,
            PLAYER_GAME_KEY,
        ].itertuples(
            index=False,
            name=None,
        )
    }

    if actual_keys != expected_keys:
        raise PlayerGameBattingBuildError(
            "Player Game Batting Key 집합이 "
            "완료 PA의 (game_pk, batter) 집합과 "
            "일치하지 않습니다."
        )

    completed_count = len(
        completed
    )

    output_pa_sum = int(
        output["pa"].sum()
    )

    if output_pa_sum != completed_count:
        raise PlayerGameBattingBuildError(
            "Completed PA Reconciliation이 "
            "일치하지 않습니다: "
            f"input={completed_count}, "
            f"output_pa_sum={output_pa_sum}"
        )

    expected_game_counts = (
        completed.groupby(
            "game_pk",
            sort=True,
            dropna=False,
        )
        .size()
        .rename(
            "expected"
        )
    )

    actual_game_counts = (
        output.groupby(
            "game_pk",
            sort=True,
            dropna=False,
        )["pa"]
        .sum()
        .rename(
            "actual"
        )
    )

    game_reconciliation = (
        expected_game_counts
        .to_frame()
        .join(
            actual_game_counts,
            how="outer",
        )
    )

    game_mismatch = (
        game_reconciliation["expected"]
        .ne(
            game_reconciliation["actual"]
        )
        .fillna(True)
    )

    if game_mismatch.any():
        sample = (
            game_reconciliation.loc[
                game_mismatch
            ]
            .head(5)
            .reset_index()
            .to_dict(
                orient="records"
            )
        )

        raise PlayerGameBattingBuildError(
            "경기별 Completed PA Reconciliation이 "
            f"일치하지 않습니다. 예시={sample}"
        )

    for event, output_column in EVENT_TO_COUNT_COLUMN.items():
        expected_count = int(
            completed["event"]
            .eq(
                event
            )
            .sum()
        )

        actual_count = int(
            output[
                output_column
            ]
            .sum()
        )

        if actual_count != expected_count:
            raise PlayerGameBattingBuildError(
                "Event Count Reconciliation이 "
                f"일치하지 않습니다: event={event}, "
                f"input={expected_count}, "
                f"output={actual_count}"
            )

    exposed_event_columns = list(
        EVENT_TO_COUNT_COLUMN.values()
    )

    reconstructed_field_out = (
        output["pa"]
        - output[
            exposed_event_columns
        ]
        .sum(
            axis=1
        )
    )

    expected_field_out = int(
        completed["event"]
        .eq(
            "field_out"
        )
        .sum()
    )

    actual_field_out = int(
        reconstructed_field_out.sum()
    )

    if actual_field_out != expected_field_out:
        raise PlayerGameBattingBuildError(
            "field_out Count Reconciliation이 "
            "일치하지 않습니다: "
            f"input={expected_field_out}, "
            f"output={actual_field_out}"
        )

    h_expected = (
        output["single"]
        + output["double"]
        + output["triple"]
        + output["hr"]
    )

    if output["h"].ne(
        h_expected
    ).any():
        raise PlayerGameBattingBuildError(
            "H 공식식 "
            "h = single + double + triple + hr이 "
            "일치하지 않는 Row가 있습니다."
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

    if output["ab"].ne(
        ab_expected
    ).any():
        raise PlayerGameBattingBuildError(
            "AB 공식식이 일치하지 않는 Row가 있습니다."
        )

    tb_expected = (
        output["single"]
        + 2 * output["double"]
        + 3 * output["triple"]
        + 4 * output["hr"]
    )

    if output["tb"].ne(
        tb_expected
    ).any():
        raise PlayerGameBattingBuildError(
            "TB 공식식이 일치하지 않는 Row가 있습니다."
        )

    avg_expected = calculate_rate(
        numerator=output["h"],
        denominator=output["ab"],
    )

    obp_expected = calculate_rate(
        numerator=(
            output["h"]
            + output["bb"]
            + output["hbp"]
        ),
        denominator=(
            output["ab"]
            + output["bb"]
            + output["hbp"]
            + output["sf"]
        ),
    )

    slg_expected = calculate_rate(
        numerator=output["tb"],
        denominator=output["ab"],
    )

    ops_expected = calculate_ops(
        obp=obp_expected,
        slg=slg_expected,
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
        matches = nullable_float_series_equal(
            actual,
            expected,
        )

        if not matches.all():
            raise PlayerGameBattingBuildError(
                f"{name} Rate 공식식이 "
                "일치하지 않는 Row가 있습니다."
            )

    prohibited = [
        column
        for column in PROHIBITED_OUTPUT_COLUMNS
        if column in output.columns
    ]

    if prohibited:
        raise PlayerGameBattingBuildError(
            "범위 제외 컬럼이 Player Game Batting "
            f"Output에 포함되어 있습니다: {prohibited}"
        )


def build_player_game_batting(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Canonical PA를 Player Game 타격 Fact Table로 변환한다."""
    source = prepare_plate_appearances(
        df
    )

    completed = (
        build_completed_plate_appearances(
            source
        )
    )

    aggregated = (
        build_player_game_aggregation(
            completed
        )
    )

    result = (
        normalize_player_game_batting_dtypes(
            aggregated
        )
    )

    validate_player_game_output(
        source=source,
        output=result,
    )

    return result


def load_plate_appearances(
    input_path: Path,
) -> pd.DataFrame:
    """Canonical Plate Appearance Parquet을 읽는다."""
    if not input_path.is_file():
        raise PlayerGameBattingBuildError(
            "Canonical Plate Appearance Parquet이 없습니다: "
            f"{input_path}"
        )

    try:
        return pd.read_parquet(
            input_path,
            engine="pyarrow",
        )
    except Exception as exc:
        raise PlayerGameBattingBuildError(
            "Canonical Plate Appearance Parquet을 "
            f"읽지 못했습니다: {input_path}"
        ) from exc


def write_parquet_atomic(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    DataFrame을 임시 Parquet에 먼저 작성한 뒤
    최종 경로로 원자적으로 교체한다.
    """
    if (
        output_path.exists()
        and output_path.is_dir()
    ):
        raise PlayerGameBattingBuildError(
            "Player Game Batting Output 경로가 "
            "파일이 아닌 디렉터리입니다: "
            f"{output_path}"
        )

    try:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
    except OSError as exc:
        raise PlayerGameBattingBuildError(
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
        raise PlayerGameBattingBuildError(
            "Player Game Batting Parquet 저장에 "
            f"실패했습니다: {output_path}"
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
    source: pd.DataFrame,
    output: pd.DataFrame,
) -> None:
    """성공한 Production Validation의 핵심 결과를 로그에 남긴다."""
    completed = (
        source.loc[
            source["event"].notna()
        ]
    )

    season_counts = (
        output.groupby(
            "season",
            sort=True,
            dropna=False,
        )
        .size()
    )

    if season_counts.empty:
        season_count_text = "<none>"
    else:
        season_count_text = ", ".join(
            f"{int(season)}={int(count)}"
            for season, count
            in season_counts.items()
        )

    duplicate_count = int(
        output.duplicated(
            subset=list(
                PLAYER_GAME_KEY
            ),
            keep=False,
        ).sum()
    )

    completed_count = len(
        completed
    )

    output_pa_sum = int(
        output["pa"].sum()
    )

    completed_reconciliation_errors = int(
        completed_count
        != output_pa_sum
    )

    event_reconciliation_errors = 0

    for event, output_column in EVENT_TO_COUNT_COLUMN.items():
        expected = int(
            completed["event"]
            .eq(
                event
            )
            .sum()
        )

        actual = int(
            output[
                output_column
            ]
            .sum()
        )

        event_reconciliation_errors += int(
            expected != actual
        )

    reconstructed_field_out = (
        output["pa"]
        - output[
            list(
                EVENT_TO_COUNT_COLUMN.values()
            )
        ]
        .sum(
            axis=1
        )
    )

    event_reconciliation_errors += int(
        int(
            completed["event"]
            .eq(
                "field_out"
            )
            .sum()
        )
        != int(
            reconstructed_field_out.sum()
        )
    )

    h_errors = (
        output["h"]
        .ne(
            output["single"]
            + output["double"]
            + output["triple"]
            + output["hr"]
        )
    )

    ab_errors = (
        output["ab"]
        .ne(
            output["pa"]
            - output["bb"]
            - output["hbp"]
            - output["sh"]
            - output["sf"]
            - output[
                "catcher_interference"
            ]
        )
    )

    tb_errors = (
        output["tb"]
        .ne(
            output["single"]
            + 2 * output["double"]
            + 3 * output["triple"]
            + 4 * output["hr"]
        )
    )

    formula_error_count = int(
        (
            h_errors
            | ab_errors
            | tb_errors
        ).sum()
    )

    LOGGER.info(
        "시즌별 Player Game Batting Row Count: %s",
        season_count_text,
    )

    LOGGER.info(
        (
            "Production Validation 통과: "
            "output_rows=%d, "
            "player_game_key_duplicates=%d, "
            "completed_pa_input=%d, "
            "output_pa_sum=%d, "
            "completed_pa_reconciliation_errors=%d, "
            "event_count_reconciliation_errors=%d, "
            "h_ab_tb_formula_errors=%d"
        ),
        len(output),
        duplicate_count,
        completed_count,
        output_pa_sum,
        completed_reconciliation_errors,
        event_reconciliation_errors,
        formula_error_count,
    )


def build_player_game_batting_file(
    input_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """
    Canonical PA 읽기부터 Player Game Batting 검증 및
    Parquet 저장까지 수행한다.
    """
    ensure_output_path_safe(
        input_path=input_path,
        output_path=output_path,
    )

    source = load_plate_appearances(
        input_path
    )

    result = build_player_game_batting(
        source
    )

    normalized_source = (
        prepare_plate_appearances(
            source
        )
    )

    log_validation_summary(
        source=normalized_source,
        output=result,
    )

    write_parquet_atomic(
        df=result,
        output_path=output_path,
    )

    return result


def main() -> None:
    """CLI 진입점."""
    configure_logging()
    args = parse_args()

    try:
        result = (
            build_player_game_batting_file(
                input_path=args.input_path,
                output_path=args.output_path,
            )
        )
    except PlayerGameBattingBuildError as exc:
        LOGGER.error(
            "%s",
            exc,
        )
        raise SystemExit(1) from exc

    LOGGER.info(
        (
            "Player Game Batting 파생 테이블 생성 완료: "
            "rows=%d, output_path=%s"
        ),
        len(result),
        args.output_path,
    )


if __name__ == "__main__":
    main()