from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path
from typing import Sequence

import pandas as pd


LOGGER = logging.getLogger(__name__)

DEFAULT_SEASONS = (2023, 2024, 2025, 2026)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "hf_kbo_pbp"
)
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "hf_kbo_pbp"
    / "derived"
    / "plate_appearances.parquet"
)

PA_KEY = (
    "game_pk",
    "at_bat_number",
)

PITCH_KEY = (
    "game_pk",
    "at_bat_number",
    "pitch_number",
)

PITCH_TYPE_VALUES = (
    "B",
    "S",
    "X",
)

REQUIRED_SOURCE_COLUMNS = (
    "game_pk",
    "game_date",
    "home_team",
    "away_team",
    "inning",
    "inning_topbot",
    "at_bat_number",
    "pitch_number",
    "batter",
    "pitcher",
    "batter_name",
    "pitcher_name",
    "strikes",
    "outs_when_up",
    "on_1b",
    "on_2b",
    "on_3b",
    "home_score",
    "away_score",
    "type",
    "stand",
    "events",
    "post_home_score",
    "post_away_score",
    "post_outs",
    "runs_scored",
    "post_on_1b",
    "post_on_2b",
    "post_on_3b",
)

NON_NULL_SOURCE_COLUMNS = (
    "game_pk",
    "game_date",
    "home_team",
    "away_team",
    "inning",
    "inning_topbot",
    "at_bat_number",
    "pitch_number",
    "batter",
    "pitcher",
    "strikes",
    "outs_when_up",
    "home_score",
    "away_score",
    "post_home_score",
    "post_away_score",
    "post_outs",
    "runs_scored",
)

SOURCE_INTEGER_COLUMNS = (
    "inning",
    "at_bat_number",
    "pitch_number",
    "strikes",
    "outs_when_up",
    "home_score",
    "away_score",
    "post_home_score",
    "post_away_score",
    "post_outs",
    "runs_scored",
)

SOURCE_STRING_COLUMNS = (
    "game_pk",
    "home_team",
    "away_team",
    "inning_topbot",
    "batter",
    "pitcher",
    "batter_name",
    "pitcher_name",
    "on_1b",
    "on_2b",
    "on_3b",
    "type",
    "stand",
    "events",
    "post_on_1b",
    "post_on_2b",
    "post_on_3b",
)

OUTPUT_COLUMNS = (
    "game_pk",
    "game_date",
    "season",
    "at_bat_number",
    "inning",
    "inning_topbot",
    "batting_team",
    "fielding_team",
    "is_home_batting",
    "batter",
    "batter_name",
    "pitcher",
    "pitcher_name",
    "stand",
    "outs_before",
    "on_1b_before",
    "on_2b_before",
    "on_3b_before",
    "home_score_before",
    "away_score_before",
    "batting_score_before",
    "fielding_score_before",
    "score_diff_before",
    "event",
    "pa_completed",
    "runs_scored",
    "post_outs",
    "post_on_1b",
    "post_on_2b",
    "post_on_3b",
    "post_home_score",
    "post_away_score",
    "pitch_rows",
    "pitch_count",
    "ball_pitch_count",
    "strike_pitch_count",
    "in_play_pitch_count",
)

STRING_OUTPUT_COLUMNS = (
    "game_pk",
    "inning_topbot",
    "batting_team",
    "fielding_team",
    "batter",
    "batter_name",
    "pitcher",
    "pitcher_name",
    "stand",
    "on_1b_before",
    "on_2b_before",
    "on_3b_before",
    "event",
    "post_on_1b",
    "post_on_2b",
    "post_on_3b",
)

INTEGER_OUTPUT_COLUMNS = (
    "season",
    "at_bat_number",
    "inning",
    "outs_before",
    "home_score_before",
    "away_score_before",
    "batting_score_before",
    "fielding_score_before",
    "score_diff_before",
    "runs_scored",
    "post_outs",
    "post_home_score",
    "post_away_score",
    "pitch_rows",
    "pitch_count",
    "ball_pitch_count",
    "strike_pitch_count",
    "in_play_pitch_count",
)

BOOLEAN_OUTPUT_COLUMNS = (
    "is_home_batting",
    "pa_completed",
)


class PlateAppearanceBuildError(RuntimeError):
    """Plate Appearance 파생 테이블 생성을 중단해야 하는 오류를 나타낸다."""


def parse_args() -> argparse.Namespace:
    """Plate Appearance 생성 스크립트의 CLI 인자를 파싱한다."""
    parser = argparse.ArgumentParser(
        description=(
            "검증된 KBO Pitch-level Raw Dataset에서 "
            "Plate Appearance 파생 테이블을 생성합니다."
        )
    )

    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help=(
            "시즌별 Raw Parquet 디렉터리입니다. "
            f"기본값: {DEFAULT_RAW_DIR}"
        ),
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=(
            "Plate Appearance Parquet 출력 경로입니다. "
            f"기본값: {DEFAULT_OUTPUT_PATH}"
        ),
    )

    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEASONS),
        help=(
            "처리할 시즌입니다. "
            "기본값: 2023 2024 2025 2026"
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


def normalize_seasons(
    values: Sequence[int],
) -> tuple[int, ...]:
    """중복 시즌을 제거하고 2023~2026을 오름차순으로 정규화한다."""
    seasons = tuple(
        sorted(
            set(values)
        )
    )

    if not seasons:
        raise PlateAppearanceBuildError(
            "최소 1개 이상의 시즌을 지정해야 합니다."
        )

    unsupported = sorted(
        set(seasons)
        - set(DEFAULT_SEASONS)
    )

    if unsupported:
        raise PlateAppearanceBuildError(
            f"지원하지 않는 시즌입니다: {unsupported}. "
            f"지원 시즌: {list(DEFAULT_SEASONS)}"
        )

    return seasons


def ensure_output_outside_raw(
    raw_dir: Path,
    output_path: Path,
) -> None:
    """파생 결과가 Raw 디렉터리 내부에 기록되는 것을 방지한다."""
    resolved_raw_dir = raw_dir.resolve()
    resolved_output_path = output_path.resolve()

    try:
        resolved_output_path.relative_to(
            resolved_raw_dir
        )
    except ValueError:
        return

    raise PlateAppearanceBuildError(
        "출력 경로는 Raw 디렉터리 내부일 수 없습니다: "
        f"{output_path}"
    )


def coerce_integer_column(
    series: pd.Series,
    column: str,
) -> pd.Series:
    """정수 의미의 Raw 컬럼을 손실 없이 int64로 정규화한다."""
    try:
        numeric = pd.to_numeric(
            series,
            errors="raise",
        )
    except (TypeError, ValueError) as exc:
        raise PlateAppearanceBuildError(
            f"{column} 컬럼을 숫자로 변환할 수 없습니다."
        ) from exc

    if numeric.isna().any():
        raise PlateAppearanceBuildError(
            f"{column} 컬럼에 결측값이 있습니다."
        )

    if numeric.mod(1).ne(0).any():
        raise PlateAppearanceBuildError(
            f"{column} 컬럼에 정수가 아닌 값이 있습니다."
        )

    return numeric.astype("int64")


def prepare_source_dataframe(
    df: pd.DataFrame,
    season: int,
) -> pd.DataFrame:
    """
    Raw DataFrame의 필수 컬럼과 기본 무결성을 검증하고
    PA 변환에 사용할 결정적 순서로 정렬한다.
    """
    missing_columns = [
        column
        for column in REQUIRED_SOURCE_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise PlateAppearanceBuildError(
            "Plate Appearance 생성에 필요한 "
            "Raw 컬럼이 없습니다: "
            f"{missing_columns}"
        )

    working = df.loc[
        :,
        REQUIRED_SOURCE_COLUMNS,
    ].copy()

    for column in NON_NULL_SOURCE_COLUMNS:
        null_count = int(
            working[column]
            .isna()
            .sum()
        )

        if null_count > 0:
            raise PlateAppearanceBuildError(
                f"{season} 시즌 {column} 컬럼에 "
                f"{null_count}개의 결측값이 있습니다."
            )

    for column in SOURCE_INTEGER_COLUMNS:
        working[column] = (
            coerce_integer_column(
                working[column],
                column,
            )
        )

    for column in SOURCE_STRING_COLUMNS:
        working[column] = (
            working[column]
            .astype("string")
        )

    parsed_dates = pd.to_datetime(
        working["game_date"],
        errors="coerce",
        utc=True,
    )

    invalid_date_count = int(
        parsed_dates.isna().sum()
    )

    if invalid_date_count > 0:
        raise PlateAppearanceBuildError(
            f"{season} 시즌 game_date 변환 실패가 "
            f"{invalid_date_count}건 있습니다."
        )

    season_mismatch_count = int(
        parsed_dates
        .dt
        .year
        .ne(season)
        .sum()
    )

    if season_mismatch_count > 0:
        raise PlateAppearanceBuildError(
            f"{season} 시즌 파일에서 game_date 기준 "
            f"시즌 불일치가 {season_mismatch_count}건 있습니다."
        )

    working["game_date"] = (
        parsed_dates
        .dt
        .tz_convert(None)
        .dt
        .normalize()
    )

    invalid_topbot = (
        ~working["inning_topbot"]
        .isin(
            (
                "top",
                "bot",
            )
        )
    )

    if invalid_topbot.any():
        values = sorted(
            {
                str(value)
                for value
                in working.loc[
                    invalid_topbot,
                    "inning_topbot",
                ].tolist()
            }
        )

        raise PlateAppearanceBuildError(
            "inning_topbot에 지원하지 않는 값이 있습니다: "
            f"{values}"
        )

    if working["at_bat_number"].le(0).any():
        raise PlateAppearanceBuildError(
            "at_bat_number는 1 이상이어야 합니다."
        )

    if working["pitch_number"].lt(0).any():
        raise PlateAppearanceBuildError(
            "pitch_number는 0 이상이어야 합니다."
        )

    duplicate_pitch_rows = int(
        working.duplicated(
            subset=list(
                PITCH_KEY
            ),
            keep=False,
        ).sum()
    )

    if duplicate_pitch_rows > 0:
        raise PlateAppearanceBuildError(
            "Raw Pitch Key "
            "(game_pk, at_bat_number, pitch_number) "
            f"중복 Row가 {duplicate_pitch_rows}건 있습니다."
        )

    actual_pitch_mask = (
        working["pitch_number"]
        .gt(0)
    )

    invalid_pitch_type_mask = (
        actual_pitch_mask
        & ~working["type"].isin(
            PITCH_TYPE_VALUES
        )
    )

    if invalid_pitch_type_mask.any():
        invalid_values = sorted(
            {
                str(value)
                for value
                in working.loc[
                    invalid_pitch_type_mask,
                    "type",
                ].tolist()
            }
        )

        raise PlateAppearanceBuildError(
            "실제 Pitch(pitch_number > 0)의 type은 "
            "B/S/X 중 하나여야 합니다. "
            f"확인된 값: {invalid_values}"
        )

    working = (
        working
        .sort_values(
            by=list(PITCH_KEY),
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    return working


def select_pa_boundary_rows(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    각 PA의 실제 첫 Row와 실제 마지막 Row를 명시적으로 선택한다.

    nullable 컬럼별 마지막 non-null 값을 섞을 수 있는
    groupby().last()는 사용하지 않는다.
    """
    first_rows = (
        df.drop_duplicates(
            subset=list(PA_KEY),
            keep="first",
        )
        .copy()
    )

    last_rows = (
        df.drop_duplicates(
            subset=list(PA_KEY),
            keep="last",
        )
        .copy()
    )

    return (
        first_rows,
        last_rows,
    )


def build_pitch_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    PA별 Raw Row 수와 실제 Pitch 수,
    B/S/X Pitch 수를 계산한다.
    """
    working = df.loc[
        :,
        [
            *PA_KEY,
            "pitch_number",
            "type",
        ],
    ].copy()

    actual_pitch = (
        working["pitch_number"]
        .gt(0)
    )

    working["_is_pitch"] = (
        actual_pitch
        .astype("int64")
    )

    working["_is_ball_pitch"] = (
        actual_pitch
        & working["type"].eq("B")
    ).astype("int64")

    working["_is_strike_pitch"] = (
        actual_pitch
        & working["type"].eq("S")
    ).astype("int64")

    working["_is_in_play_pitch"] = (
        actual_pitch
        & working["type"].eq("X")
    ).astype("int64")

    summary = (
        working
        .groupby(
            list(PA_KEY),
            sort=False,
            dropna=False,
        )
        .agg(
            pitch_rows=(
                "pitch_number",
                "size",
            ),
            pitch_count=(
                "_is_pitch",
                "sum",
            ),
            ball_pitch_count=(
                "_is_ball_pitch",
                "sum",
            ),
            strike_pitch_count=(
                "_is_strike_pitch",
                "sum",
            ),
            in_play_pitch_count=(
                "_is_in_play_pitch",
                "sum",
            ),
        )
        .reset_index()
    )

    return summary


def build_first_takeover_summary(
    df: pd.DataFrame,
    first_rows: pd.DataFrame,
) -> pd.DataFrame:
    """
    Batter가 교체된 PA에서 첫 교체 Row의 Strike Count를 찾는다.

    Two-strike 교체 후 Strikeout의 공식 기록 귀속을
    판정할 때만 사용한다.
    """
    starting_batters = (
        first_rows.loc[
            :,
            [
                *PA_KEY,
                "batter",
            ],
        ]
        .rename(
            columns={
                "batter": "_starting_batter",
            }
        )
    )

    with_starting_batter = (
        df.merge(
            starting_batters,
            on=list(PA_KEY),
            how="left",
            validate="many_to_one",
            sort=False,
        )
    )

    batter_changed = (
        with_starting_batter[
            "batter"
        ]
        .ne(
            with_starting_batter[
                "_starting_batter"
            ]
        )
    )

    takeover_rows = (
        with_starting_batter.loc[
            batter_changed,
            [
                *PA_KEY,
                "strikes",
            ],
        ]
        .drop_duplicates(
            subset=list(PA_KEY),
            keep="first",
        )
        .rename(
            columns={
                "strikes": "_handover_strikes",
            }
        )
    )

    return takeover_rows


def empty_output_dataframe() -> pd.DataFrame:
    """고정 Output Schema를 가진 빈 DataFrame을 생성한다."""
    frame = pd.DataFrame(
        {
            column: pd.Series(
                dtype="object"
            )
            for column
            in OUTPUT_COLUMNS
        }
    )

    return normalize_output_dtypes(
        frame
    )


def normalize_output_dtypes(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Plate Appearance Output 컬럼 순서와 dtype을 고정한다."""
    missing_columns = [
        column
        for column in OUTPUT_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise PlateAppearanceBuildError(
            "Output Schema에 필요한 컬럼이 없습니다: "
            f"{missing_columns}"
        )

    result = df.loc[
        :,
        OUTPUT_COLUMNS,
    ].copy()

    try:
        result["game_date"] = (
            pd.to_datetime(
                result["game_date"],
                errors="raise",
            )
            .dt
            .normalize()
        )
    except (TypeError, ValueError) as exc:
        raise PlateAppearanceBuildError(
            "Output game_date dtype 정규화에 실패했습니다."
        ) from exc

    for column in STRING_OUTPUT_COLUMNS:
        result[column] = (
            result[column]
            .astype("string")
        )

    for column in INTEGER_OUTPUT_COLUMNS:
        try:
            result[column] = (
                pd.to_numeric(
                    result[column],
                    errors="raise",
                )
                .astype("Int64")
            )
        except (TypeError, ValueError) as exc:
            raise PlateAppearanceBuildError(
                f"Output {column} 컬럼을 "
                "Int64로 변환할 수 없습니다."
            ) from exc

    for column in BOOLEAN_OUTPUT_COLUMNS:
        try:
            result[column] = (
                result[column]
                .astype("boolean")
            )
        except (TypeError, ValueError) as exc:
            raise PlateAppearanceBuildError(
                f"Output {column} 컬럼을 "
                "boolean으로 변환할 수 없습니다."
            ) from exc

    result = (
        result
        .sort_values(
            by=[
                "season",
                "game_date",
                "game_pk",
                "at_bat_number",
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    return result


def validate_plate_appearance_output(
    source: pd.DataFrame,
    output: pd.DataFrame,
    season: int,
) -> None:
    """Raw Unique PA와 파생 PA의 Grain 및 Pitch Summary를 검증한다."""
    expected_pa_count = int(
        source.loc[
            :,
            PA_KEY,
        ]
        .drop_duplicates()
        .shape[0]
    )

    actual_pa_count = len(
        output
    )

    if actual_pa_count != expected_pa_count:
        raise PlateAppearanceBuildError(
            f"{season} 시즌 PA 수 불일치: "
            f"raw_unique={expected_pa_count}, "
            f"derived={actual_pa_count}"
        )

    duplicate_count = int(
        output.duplicated(
            subset=list(PA_KEY),
            keep=False,
        ).sum()
    )

    if duplicate_count > 0:
        raise PlateAppearanceBuildError(
            f"{season} 시즌 파생 PA Key 중복이 "
            f"{duplicate_count}건 있습니다."
        )

    invalid_row_count = (
        output["pitch_rows"]
        .lt(
            output["pitch_count"]
        )
    )

    if invalid_row_count.any():
        raise PlateAppearanceBuildError(
            f"{season} 시즌에 pitch_rows보다 "
            "pitch_count가 큰 PA가 있습니다."
        )

    classified_pitch_count = (
        output[
            [
                "ball_pitch_count",
                "strike_pitch_count",
                "in_play_pitch_count",
            ]
        ]
        .sum(
            axis=1
        )
    )

    pitch_count_mismatch = (
        classified_pitch_count
        .ne(
            output["pitch_count"]
        )
    )

    if pitch_count_mismatch.any():
        raise PlateAppearanceBuildError(
            f"{season} 시즌 실제 Pitch 수와 "
            "B/S/X 분류 합계가 일치하지 않습니다."
        )

    if not output["season"].eq(season).all():
        raise PlateAppearanceBuildError(
            f"{season} 시즌 Output에 "
            "다른 season 값이 포함되어 있습니다."
        )


def build_plate_appearances(
    df: pd.DataFrame,
    season: int,
) -> pd.DataFrame:
    """
    한 시즌 Pitch-level Raw DataFrame을
    Canonical Plate Appearance DataFrame으로 변환한다.
    """
    working = prepare_source_dataframe(
        df=df,
        season=season,
    )

    if working.empty:
        return empty_output_dataframe()

    first_rows, last_rows = (
        select_pa_boundary_rows(
            working
        )
    )

    first_projection = (
        first_rows.loc[
            :,
            [
                *PA_KEY,
                "game_date",
                "inning",
                "inning_topbot",
                "home_team",
                "away_team",
                "batter",
                "batter_name",
                "stand",
                "outs_when_up",
                "on_1b",
                "on_2b",
                "on_3b",
                "home_score",
                "away_score",
            ],
        ]
        .rename(
            columns={
                "batter": "_starting_batter",
                "batter_name": "_starting_batter_name",
                "stand": "_starting_stand",
                "outs_when_up": "outs_before",
                "on_1b": "on_1b_before",
                "on_2b": "on_2b_before",
                "on_3b": "on_3b_before",
                "home_score": "home_score_before",
                "away_score": "away_score_before",
            }
        )
    )

    last_projection = (
        last_rows.loc[
            :,
            [
                *PA_KEY,
                "batter",
                "batter_name",
                "pitcher",
                "pitcher_name",
                "stand",
                "events",
                "runs_scored",
                "post_outs",
                "post_on_1b",
                "post_on_2b",
                "post_on_3b",
                "post_home_score",
                "post_away_score",
            ],
        ]
        .rename(
            columns={
                "batter": "_finishing_batter",
                "batter_name": "_finishing_batter_name",
                "stand": "_finishing_stand",
                "events": "event",
            }
        )
    )

    pa = (
        first_projection
        .merge(
            last_projection,
            on=list(PA_KEY),
            how="inner",
            validate="one_to_one",
            sort=False,
        )
    )

    is_home_batting = (
        pa["inning_topbot"]
        .eq("bot")
    )

    pa["season"] = season
    pa["is_home_batting"] = (
        is_home_batting
    )

    pa["batting_team"] = (
        pa["home_team"]
        .where(
            is_home_batting,
            pa["away_team"],
        )
    )

    pa["fielding_team"] = (
        pa["away_team"]
        .where(
            is_home_batting,
            pa["home_team"],
        )
    )

    pa["batting_score_before"] = (
        pa["home_score_before"]
        .where(
            is_home_batting,
            pa["away_score_before"],
        )
    )

    pa["fielding_score_before"] = (
        pa["away_score_before"]
        .where(
            is_home_batting,
            pa["home_score_before"],
        )
    )

    pa["score_diff_before"] = (
        pa["batting_score_before"]
        - pa["fielding_score_before"]
    )

    pa["batter"] = (
        pa["_finishing_batter"]
    )
    pa["batter_name"] = (
        pa["_finishing_batter_name"]
    )
    pa["stand"] = (
        pa["_finishing_stand"]
    )

    pa["pa_completed"] = (
        pa["event"]
        .notna()
    )

    takeover_summary = (
        build_first_takeover_summary(
            df=working,
            first_rows=first_rows,
        )
    )

    pa = (
        pa.merge(
            takeover_summary,
            on=list(PA_KEY),
            how="left",
            validate="one_to_one",
            sort=False,
        )
    )

    two_strike_substitution_strikeout = (
        pa["event"]
        .eq("strikeout")
        & pa["_handover_strikes"]
        .eq(2)
    ).fillna(False)

    pa.loc[
        two_strike_substitution_strikeout,
        "batter",
    ] = pa.loc[
        two_strike_substitution_strikeout,
        "_starting_batter",
    ]

    pa.loc[
        two_strike_substitution_strikeout,
        "batter_name",
    ] = pa.loc[
        two_strike_substitution_strikeout,
        "_starting_batter_name",
    ]

    pa.loc[
        two_strike_substitution_strikeout,
        "stand",
    ] = pa.loc[
        two_strike_substitution_strikeout,
        "_starting_stand",
    ]

    pitch_summary = (
        build_pitch_summary(
            working
        )
    )

    pa = (
        pa.merge(
            pitch_summary,
            on=list(PA_KEY),
            how="left",
            validate="one_to_one",
            sort=False,
        )
    )

    result = normalize_output_dtypes(
        pa
    )

    validate_plate_appearance_output(
        source=working,
        output=result,
        season=season,
    )

    return result


def load_season_dataframe(
    raw_dir: Path,
    season: int,
) -> pd.DataFrame:
    """한 시즌 Raw Parquet을 읽는다."""
    raw_path = (
        raw_dir
        / f"{season}.parquet"
    )

    if not raw_path.is_file():
        raise PlateAppearanceBuildError(
            f"{season} 시즌 Raw Parquet이 없습니다: "
            f"{raw_path}"
        )

    try:
        return pd.read_parquet(
            raw_path,
            engine="pyarrow",
        )
    except Exception as exc:
        raise PlateAppearanceBuildError(
            f"{season} 시즌 Raw Parquet을 "
            f"읽지 못했습니다: {raw_path}"
        ) from exc


def build_plate_appearance_dataset(
    raw_dir: Path,
    seasons: Sequence[int],
) -> pd.DataFrame:
    """
    시즌별 Raw를 순차 처리하여
    하나의 Plate Appearance Dataset으로 결합한다.
    """
    normalized_seasons = (
        normalize_seasons(
            seasons
        )
    )

    season_frames: list[
        pd.DataFrame
    ] = []

    for season in normalized_seasons:
        LOGGER.info(
            "%s 시즌 Plate Appearance 변환 시작",
            season,
        )

        raw_df = load_season_dataframe(
            raw_dir=raw_dir,
            season=season,
        )

        pa_df = build_plate_appearances(
            df=raw_df,
            season=season,
        )

        completed_count = int(
            pa_df["pa_completed"]
            .sum()
        )

        incomplete_count = (
            len(pa_df)
            - completed_count
        )

        pitchless_count = int(
            pa_df["pitch_count"]
            .eq(0)
            .sum()
        )

        LOGGER.info(
            (
                "%s 시즌 Plate Appearance 변환 완료: "
                "raw_rows=%d, pa_rows=%d, "
                "completed=%d, incomplete=%d, "
                "pitchless=%d"
            ),
            season,
            len(raw_df),
            len(pa_df),
            completed_count,
            incomplete_count,
            pitchless_count,
        )

        season_frames.append(
            pa_df
        )

    combined = pd.concat(
        season_frames,
        ignore_index=True,
    )

    combined = (
        normalize_output_dtypes(
            combined
        )
    )

    duplicate_count = int(
        combined.duplicated(
            subset=list(PA_KEY),
            keep=False,
        ).sum()
    )

    if duplicate_count > 0:
        raise PlateAppearanceBuildError(
            "전체 시즌 결합 후 "
            "(game_pk, at_bat_number) 중복이 "
            f"{duplicate_count}건 있습니다."
        )

    return combined


def write_plate_appearance_parquet(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    완성된 PA DataFrame을 임시 파일에 먼저 작성한 뒤
    최종 경로로 원자적으로 교체한다.
    """
    if (
        output_path.exists()
        and output_path.is_dir()
    ):
        raise PlateAppearanceBuildError(
            "Output 경로가 파일이 아닌 디렉터리입니다: "
            f"{output_path}"
        )

    try:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
    except OSError as exc:
        raise PlateAppearanceBuildError(
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
        raise PlateAppearanceBuildError(
            "Plate Appearance Parquet 저장에 실패했습니다: "
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


def build_plate_appearance_file(
    raw_dir: Path,
    output_path: Path,
    seasons: Sequence[int],
) -> pd.DataFrame:
    """Raw 읽기부터 최종 Parquet 저장까지 전체 생성 절차를 수행한다."""
    ensure_output_outside_raw(
        raw_dir=raw_dir,
        output_path=output_path,
    )

    result = (
        build_plate_appearance_dataset(
            raw_dir=raw_dir,
            seasons=seasons,
        )
    )

    write_plate_appearance_parquet(
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
            build_plate_appearance_file(
                raw_dir=args.raw_dir,
                output_path=args.output_path,
                seasons=args.seasons,
            )
        )
    except PlateAppearanceBuildError as exc:
        LOGGER.error(
            "%s",
            exc,
        )
        raise SystemExit(1) from exc

    LOGGER.info(
        "Plate Appearance 파생 테이블 생성 완료: rows=%d, path=%s",
        len(result),
        args.output_path,
    )


if __name__ == "__main__":
    main()