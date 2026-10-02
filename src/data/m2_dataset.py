from __future__ import annotations

import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from src.data.processed_contract import (
    FEATURE_CATALOG_VERSION,
    PREDICTION_CONTRACT_VERSION,
    SPLIT_VERSION,
    ProcessedContractError,
    TemporalSplitRange,
    assign_temporal_split_series,
    assert_input_hashes_unchanged,
    calculate_input_hashes,
    calculate_sha256,
    content_fingerprint,
    invalidate_completion_marker,
    parse_split_ranges,
    read_json,
    read_parquet,
    schema_manifest,
    stable_json_fingerprint,
    validate_output_paths,
    validate_x_allowlist,
    write_json_atomic,
    write_parquet_atomic,
)
from src.features.player import (
    FEATURE_VERSION as PLAYER_FEATURE_VERSION,
    PlayerFeatureBuildError,
    build_player_pregame_features,
)


DATASET_VERSION = "m2_matchup_dataset_v1"
TARGET_VERSION = "m2_target_v1"
OUTPUT_SCHEMA_VERSION = "m2_matchup_dataset_schema_v1"
M2_FEATURE_VERSION = "extended_v1"
AVAILABILITY_BASIS = "conservative_next_date"

RAW_SEASONS = (2023, 2024, 2025, 2026)
ROLLING_WINDOWS = (5, 10, 20)

PA_KEY = (
    "game_pk",
    "at_bat_number",
)

PITCH_KEY = (
    "game_pk",
    "at_bat_number",
    "pitch_number",
)

TARGET_CLASSES = (
    "1B",
    "2B",
    "3B",
    "HR",
    "BB",
    "HBP",
    "SO",
    "OUT",
    "ROE",
    "FC",
    "CI",
)

TARGET_CODES = {
    name: code
    for code, name in enumerate(TARGET_CLASSES)
}

EVENT_TO_CLASS = {
    "single": "1B",
    "double": "2B",
    "triple": "3B",
    "home_run": "HR",
    "walk": "BB",
    "hit_by_pitch": "HBP",
    "strikeout": "SO",
    "field_out": "OUT",
    "double_play": "OUT",
    "triple_play": "OUT",
    "sac_bunt": "OUT",
    "sac_fly": "OUT",
    "field_error": "ROE",
    "fielders_choice": "FC",
    "catcher_interference": "CI",
}

REQUIRED_PA_COLUMNS = (
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
    "pitcher",
    "stand",
    "outs_before",
    "on_1b_before",
    "on_2b_before",
    "on_3b_before",
    "home_score_before",
    "away_score_before",
    "score_diff_before",
    "event",
    "pa_completed",
    "pitch_count",
)

RAW_REQUIRED_COLUMNS = (
    "game_pk",
    "game_date",
    "inning",
    "inning_topbot",
    "at_bat_number",
    "pitch_number",
    "batter",
    "pitcher",
    "stand",
    "outs_when_up",
    "on_1b",
    "on_2b",
    "on_3b",
    "home_score",
    "away_score",
)

CONTEXT_X_COLUMNS = (
    "inning",
    "inning_topbot",
    "outs_before",
    "on_1b_occupied",
    "on_2b_occupied",
    "on_3b_occupied",
    "score_diff_before",
    "starting_stand",
)

BATTER_BASE_SOURCE_FEATURES = (
    "hist_pa",
    "hist_ab",
    "hist_h",
    "hist_hr",
    "hist_bb",
    "hist_hbp",
    "hist_so",
    "hist_tb",
    "hist_avg",
    "hist_obp",
    "hist_slg",
    "hist_ops",
)

PITCHER_BASE_SOURCE_FEATURES = (
    "hist_pitches",
    "hist_batters_faced_completed",
    "hist_hits_allowed",
    "hist_hr_allowed",
    "hist_bb_allowed",
    "hist_hbp_allowed",
    "hist_so",
    "hist_outs_recorded",
)

BATTER_ROLLING_FAMILIES = (
    "pa",
    "ab",
    "h",
    "hr",
    "bb",
    "hbp",
    "so",
    "tb",
    "avg",
    "obp",
    "slg",
    "ops",
)

PITCHER_ROLLING_FAMILIES = (
    "pitches",
    "batters_faced_completed",
    "hits_allowed",
    "hr_allowed",
    "bb_allowed",
    "hbp_allowed",
    "so",
    "outs_recorded",
)

BATTER_ROLLING_SOURCE_FEATURES = tuple(
    f"last_{window}g_{stat}"
    for window in ROLLING_WINDOWS
    for stat in BATTER_ROLLING_FAMILIES
)

PITCHER_ROLLING_SOURCE_FEATURES = tuple(
    f"last_{window}g_{stat}"
    for window in ROLLING_WINDOWS
    for stat in PITCHER_ROLLING_FAMILIES
)

BATTER_X_COLUMNS = (
    tuple(f"batter_{name}" for name in BATTER_BASE_SOURCE_FEATURES)
    + ("batter_has_history",)
    + tuple(
        f"batter_{name}"
        for name in BATTER_ROLLING_SOURCE_FEATURES
    )
)

PITCHER_X_COLUMNS = (
    tuple(f"pitcher_{name}" for name in PITCHER_BASE_SOURCE_FEATURES)
    + ("pitcher_has_history",)
    + tuple(
        f"pitcher_{name}"
        for name in PITCHER_ROLLING_SOURCE_FEATURES
    )
)

M2_X_COLUMNS = (
    CONTEXT_X_COLUMNS
    + BATTER_X_COLUMNS
    + PITCHER_X_COLUMNS
)

KEY_COLUMNS = (
    "game_pk",
    "at_bat_number",
)

AUDIT_COLUMNS = (
    "game_date",
    "season",
    "batting_team",
    "fielding_team",
    "is_home_batting",
    "prediction_date",
    "label_available_at",
    "availability_basis",
    "first_observed_pitch_number",
    "starting_batter",
    "starting_pitcher",
    "credited_batter",
    "final_pitcher",
    "credited_stand",
    "source_event",
    "pa_completed",
    "pitch_count",
    "batter_max_source_game_date",
    "batter_history_game_count",
    "pitcher_max_source_game_date",
    "pitcher_history_game_count",
    "contract_version",
    "feature_catalog_version",
    "feature_version",
    "player_feature_version",
    "target_version",
    "split_version",
    "output_schema_version",
)

Y_COLUMNS = (
    "target_class",
    "target_code",
)

SPLIT_COLUMNS = (
    "split",
)

QUALITY_COLUMNS = (
    "start_context_quality",
    "start_context_quality_reason",
    "starting_stand_missing",
    "pitchless_pa",
    "batter_identity_diff",
    "pitcher_identity_diff",
    "stand_identity_diff",
)

CONTROL_COLUMNS = (
    "target_eligible",
    "is_excluded",
    "exclusion_reason",
    "diagnostic_reasons",
    "is_censored",
    "censor_reason",
    "coverage_status",
    "is_purged",
    "purge_reason",
)

LABEL_STATUS_COLUMNS = (
    "supervised_usable",
    "label_status",
)

FORBIDDEN_X_COLUMNS = set(
    KEY_COLUMNS
    + AUDIT_COLUMNS
    + Y_COLUMNS
    + SPLIT_COLUMNS
    + QUALITY_COLUMNS
    + CONTROL_COLUMNS
    + LABEL_STATUS_COLUMNS
    + (
        "event",
        "runs_scored",
        "post_outs",
        "post_on_1b",
        "post_on_2b",
        "post_on_3b",
        "post_home_score",
        "post_away_score",
        "pitch_rows",
        "ball_pitch_count",
        "strike_pitch_count",
        "in_play_pitch_count",
    )
)

FORBIDDEN_X_PREFIXES = (
    "target_",
    "label_",
    "quality_",
    "exclusion_",
    "censor_",
    "purge_",
    "post_",
    "credited_",
    "final_",
)


class M2DatasetBuildError(RuntimeError):
    """M2 Model-ready Dataset 계약 위반으로 생성을 중단할 때 사용한다."""


def _wrap_contract_error(
    exc: ProcessedContractError,
) -> M2DatasetBuildError:
    """공통 Processed 계약 오류를 M2 Builder 오류로 변환한다."""
    return M2DatasetBuildError(str(exc))


def _require_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    source_name: str,
) -> None:
    """필수 컬럼 존재 여부를 검증한다."""
    missing = [
        column
        for column in columns
        if column not in frame.columns
    ]
    if missing:
        raise M2DatasetBuildError(
            f"{source_name}에 필수 컬럼이 없습니다: {missing}"
        )


def _normalize_string(
    series: pd.Series,
    *,
    source_name: str,
    column: str,
    allow_null: bool = False,
) -> pd.Series:
    """문자열 의미 컬럼을 pandas string dtype으로 정규화한다."""
    result = series.astype("string")

    if not allow_null:
        invalid = (
            result.isna()
            | result.str.strip().eq("")
        )
        if invalid.any():
            raise M2DatasetBuildError(
                f"{source_name}.{column}에 null 또는 빈 문자열이 있습니다."
            )

    non_null = result.notna()
    result.loc[non_null] = (
        result.loc[non_null]
        .str.strip()
    )
    return result


def _normalize_integer(
    series: pd.Series,
    *,
    source_name: str,
    column: str,
    allow_null: bool = False,
) -> pd.Series:
    """정수 의미 컬럼을 nullable Int64로 정규화한다."""
    try:
        numeric = pd.to_numeric(
            series,
            errors="raise",
        )
    except (TypeError, ValueError) as exc:
        raise M2DatasetBuildError(
            f"{source_name}.{column}을 숫자로 변환할 수 없습니다."
        ) from exc

    if not allow_null and numeric.isna().any():
        raise M2DatasetBuildError(
            f"{source_name}.{column}에 결측값이 있습니다."
        )

    non_null = numeric.dropna()
    if non_null.mod(1).ne(0).any():
        raise M2DatasetBuildError(
            f"{source_name}.{column}에 정수가 아닌 값이 있습니다."
        )

    return numeric.astype("Int64")


def _normalize_date(
    series: pd.Series,
    *,
    source_name: str,
) -> pd.Series:
    """날짜를 timezone-naive 자정 datetime64[us]로 정규화한다."""
    try:
        parsed = pd.to_datetime(
            series,
            errors="raise",
        )
    except (TypeError, ValueError) as exc:
        raise M2DatasetBuildError(
            f"{source_name} 날짜를 해석할 수 없습니다."
        ) from exc

    if parsed.isna().any():
        raise M2DatasetBuildError(
            f"{source_name} 날짜에 결측값이 있습니다."
        )

    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_convert(None)

    return (
        parsed
        .dt
        .normalize()
        .astype("datetime64[us]")
    )


def _different(
    left: pd.Series,
    right: pd.Series,
) -> pd.Series:
    """Nullable 문자열 두 Series의 값 차이를 Boolean으로 반환한다."""
    left_string = left.astype("string")
    right_string = right.astype("string")

    same = (
        left_string
        .eq(right_string)
        .fillna(False)
    )
    both_null = (
        left_string.isna()
        & right_string.isna()
    )

    return ~(same | both_null)


def _nullable_string_equal(
    left: pd.Series,
    right: pd.Series,
) -> pd.Series:
    """Nullable 문자열의 null-safe equality를 계산한다."""
    return ~_different(left, right)


def _normalize_plate_appearances(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """M2 조립에 필요한 Canonical PA 계약을 검증한다."""
    if frame.empty:
        raise M2DatasetBuildError(
            "plate_appearances 입력이 비어 있습니다."
        )

    _require_columns(
        frame,
        REQUIRED_PA_COLUMNS,
        source_name="plate_appearances",
    )

    result = frame.loc[
        :,
        REQUIRED_PA_COLUMNS,
    ].copy()

    for column in (
        "game_pk",
        "inning_topbot",
        "batting_team",
        "fielding_team",
        "batter",
        "pitcher",
    ):
        result[column] = _normalize_string(
            result[column],
            source_name="plate_appearances",
            column=column,
        )

    for column in (
        "stand",
        "on_1b_before",
        "on_2b_before",
        "on_3b_before",
        "event",
    ):
        result[column] = _normalize_string(
            result[column],
            source_name="plate_appearances",
            column=column,
            allow_null=True,
        )

    result["game_date"] = _normalize_date(
        result["game_date"],
        source_name="plate_appearances.game_date",
    )

    for column in (
        "season",
        "at_bat_number",
        "inning",
        "outs_before",
        "home_score_before",
        "away_score_before",
        "score_diff_before",
        "pitch_count",
    ):
        result[column] = _normalize_integer(
            result[column],
            source_name="plate_appearances",
            column=column,
        )

    try:
        result["is_home_batting"] = (
            result["is_home_batting"]
            .astype("boolean")
        )
        result["pa_completed"] = (
            result["pa_completed"]
            .astype("boolean")
        )
    except (TypeError, ValueError) as exc:
        raise M2DatasetBuildError(
            "plate_appearances boolean dtype 정규화에 실패했습니다."
        ) from exc

    duplicate_count = int(
        result
        .duplicated(
            subset=list(PA_KEY),
            keep=False,
        )
        .sum()
    )
    if duplicate_count:
        raise M2DatasetBuildError(
            "plate_appearances "
            "(game_pk, at_bat_number) 중복 Row가 "
            f"{duplicate_count}건 있습니다."
        )

    invalid_topbot = (
        ~result["inning_topbot"]
        .isin(("top", "bot"))
    )
    if invalid_topbot.any():
        raise M2DatasetBuildError(
            "plate_appearances.inning_topbot에 "
            "top/bot 이외 값이 있습니다."
        )

    if result["at_bat_number"].le(0).any():
        raise M2DatasetBuildError(
            "plate_appearances.at_bat_number는 1 이상이어야 합니다."
        )

    if result["pitch_count"].lt(0).any():
        raise M2DatasetBuildError(
            "plate_appearances.pitch_count에 음수가 있습니다."
        )

    expected_completed = (
        result["event"]
        .notna()
        .astype("boolean")
    )
    if result["pa_completed"].ne(expected_completed).any():
        raise M2DatasetBuildError(
            "plate_appearances.pa_completed와 event null 여부가 일치하지 않습니다."
        )

    expected_season = (
        result["game_date"]
        .dt
        .year
        .astype("Int64")
    )
    if result["season"].ne(expected_season).any():
        raise M2DatasetBuildError(
            "plate_appearances.season이 game_date 연도와 일치하지 않습니다."
        )

    return (
        result
        .sort_values(
            [
                "season",
                "game_date",
                "game_pk",
                "at_bat_number",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def _normalize_raw(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Raw Pitch-level 입력을 첫 관측 Row 복원에 필요한 범위만 검증한다."""
    if frame.empty:
        raise M2DatasetBuildError(
            "Raw PBP 입력이 비어 있습니다."
        )

    _require_columns(
        frame,
        RAW_REQUIRED_COLUMNS,
        source_name="raw_pbp",
    )

    result = frame.loc[
        :,
        RAW_REQUIRED_COLUMNS,
    ].copy()

    for column in (
        "game_pk",
        "inning_topbot",
        "batter",
        "pitcher",
    ):
        result[column] = _normalize_string(
            result[column],
            source_name="raw_pbp",
            column=column,
        )

    for column in (
        "stand",
        "on_1b",
        "on_2b",
        "on_3b",
    ):
        result[column] = _normalize_string(
            result[column],
            source_name="raw_pbp",
            column=column,
            allow_null=True,
        )

    result["game_date"] = _normalize_date(
        result["game_date"],
        source_name="raw_pbp.game_date",
    )

    for column in (
        "inning",
        "at_bat_number",
        "pitch_number",
        "outs_when_up",
        "home_score",
        "away_score",
    ):
        result[column] = _normalize_integer(
            result[column],
            source_name="raw_pbp",
            column=column,
        )

    if result["at_bat_number"].le(0).any():
        raise M2DatasetBuildError(
            "raw_pbp.at_bat_number는 1 이상이어야 합니다."
        )

    if result["pitch_number"].lt(0).any():
        raise M2DatasetBuildError(
            "raw_pbp.pitch_number에 음수가 있습니다."
        )

    duplicate_count = int(
        result
        .duplicated(
            subset=list(PITCH_KEY),
            keep=False,
        )
        .sum()
    )
    if duplicate_count:
        raise M2DatasetBuildError(
            "Raw Pitch Key "
            "(game_pk, at_bat_number, pitch_number) 중복 Row가 "
            f"{duplicate_count}건 있습니다."
        )

    invalid_topbot = (
        ~result["inning_topbot"]
        .isin(("top", "bot"))
    )
    if invalid_topbot.any():
        raise M2DatasetBuildError(
            "raw_pbp.inning_topbot에 top/bot 이외 값이 있습니다."
        )

    return (
        result
        .sort_values(
            list(PITCH_KEY),
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def build_raw_start_context(
    raw_pbp: pd.DataFrame,
) -> pd.DataFrame:
    """실제 첫 관측 Raw Row 하나에서 PA-start 식별자와 Context를 복원한다."""
    raw = _normalize_raw(raw_pbp)

    first_rows = (
        raw
        .drop_duplicates(
            subset=list(PA_KEY),
            keep="first",
        )
        .copy()
    )

    pitch_summary = (
        raw
        .assign(
            _is_actual_pitch=(
                raw["pitch_number"]
                .gt(0)
            )
            .astype("int64")
        )
        .groupby(
            list(PA_KEY),
            sort=False,
            dropna=False,
        )
        .agg(
            actual_pitch_count=(
                "_is_actual_pitch",
                "sum",
            ),
        )
        .reset_index()
    )

    first_rows = (
        first_rows
        .merge(
            pitch_summary,
            on=list(PA_KEY),
            how="left",
            validate="one_to_one",
            sort=False,
        )
    )

    invalid_zero_prefix = (
        first_rows["pitch_number"].eq(0)
        & first_rows["actual_pitch_count"].gt(0)
    )
    if invalid_zero_prefix.any():
        raise M2DatasetBuildError(
            "pitch_number=0 첫 Row 뒤에 실제 Pitch가 존재하는 PA가 있습니다. "
            "현재 pitch-less 계약으로 시작 Context를 확정할 수 없습니다."
        )

    first_rows["pitchless_pa"] = (
        first_rows["actual_pitch_count"]
        .eq(0)
        .astype("boolean")
    )

    first_rows["start_context_quality"] = pd.Series(
        pd.NA,
        index=first_rows.index,
        dtype="string",
    )
    first_rows["start_context_quality_reason"] = pd.Series(
        pd.NA,
        index=first_rows.index,
        dtype="string",
    )

    pitchless_mask = first_rows["pitchless_pa"].fillna(False)
    first_pitch_mask = first_rows["pitch_number"].eq(1)
    partial_mask = first_rows["pitch_number"].gt(1)

    first_rows.loc[
        pitchless_mask,
        "start_context_quality",
    ] = "verified_pitchless_start"

    first_rows.loc[
        first_pitch_mask,
        "start_context_quality",
    ] = "verified_first_pitch_start"

    first_rows.loc[
        partial_mask,
        "start_context_quality",
    ] = "unverified_missing_initial_pitch_rows"

    first_rows.loc[
        partial_mask,
        "start_context_quality_reason",
    ] = "first_observed_pitch_number_gt_1"

    unresolved = first_rows["start_context_quality"].isna()
    if unresolved.any():
        raise M2DatasetBuildError(
            "첫 관측 Raw Row의 시작 Context 품질 상태를 결정할 수 없는 PA가 있습니다."
        )

    return (
        first_rows.loc[
            :,
            [
                *PA_KEY,
                "game_date",
                "inning",
                "inning_topbot",
                "pitch_number",
                "batter",
                "pitcher",
                "stand",
                "outs_when_up",
                "on_1b",
                "on_2b",
                "on_3b",
                "home_score",
                "away_score",
                "actual_pitch_count",
                "pitchless_pa",
                "start_context_quality",
                "start_context_quality_reason",
            ],
        ]
        .rename(
            columns={
                "pitch_number": "first_observed_pitch_number",
                "batter": "starting_batter",
                "pitcher": "starting_pitcher",
                "stand": "starting_stand",
                "outs_when_up": "raw_outs_before",
                "on_1b": "raw_on_1b_before",
                "on_2b": "raw_on_2b_before",
                "on_3b": "raw_on_3b_before",
                "home_score": "raw_home_score_before",
                "away_score": "raw_away_score_before",
            }
        )
        .sort_values(
            list(PA_KEY),
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def _validate_raw_canonical_context(
    merged: pd.DataFrame,
) -> None:
    """Raw 첫 Row와 Canonical PA-start Context가 동일한지 검증한다."""
    integer_pairs = (
        ("inning", "raw_inning"),
        ("outs_before", "raw_outs_before"),
        ("home_score_before", "raw_home_score_before"),
        ("away_score_before", "raw_away_score_before"),
    )

    for canonical_column, raw_column in integer_pairs:
        mismatch = (
            merged[canonical_column]
            .ne(merged[raw_column])
        )
        if mismatch.any():
            raise M2DatasetBuildError(
                "Raw 첫 Row와 Canonical PA-start Context가 일치하지 않습니다: "
                f"{canonical_column}"
            )

    if (
        merged["inning_topbot"]
        .ne(merged["raw_inning_topbot"])
        .any()
    ):
        raise M2DatasetBuildError(
            "Raw 첫 Row와 Canonical inning_topbot이 일치하지 않습니다."
        )

    runner_pairs = (
        ("on_1b_before", "raw_on_1b_before"),
        ("on_2b_before", "raw_on_2b_before"),
        ("on_3b_before", "raw_on_3b_before"),
    )

    for canonical_column, raw_column in runner_pairs:
        equal = _nullable_string_equal(
            merged[canonical_column],
            merged[raw_column],
        )
        if not equal.all():
            raise M2DatasetBuildError(
                "Raw 첫 Row와 Canonical Runner Context가 일치하지 않습니다: "
                f"{canonical_column}"
            )


def _merge_raw_start_context(
    plate_appearances: pd.DataFrame,
    raw_start: pd.DataFrame,
) -> pd.DataFrame:
    """Canonical PA와 실제 첫 Raw Row를 one-to-one으로 결합한다."""
    raw_projection = raw_start.rename(
        columns={
            "game_date": "raw_game_date",
            "inning": "raw_inning",
            "inning_topbot": "raw_inning_topbot",
        }
    )

    before_rows = len(plate_appearances)

    merged = (
        plate_appearances
        .merge(
            raw_projection,
            on=list(PA_KEY),
            how="left",
            validate="one_to_one",
            sort=False,
        )
    )

    if len(merged) != before_rows:
        raise M2DatasetBuildError(
            "Raw 시작 Context Join으로 PA Row가 증식했습니다."
        )

    if merged["raw_game_date"].isna().any():
        raise M2DatasetBuildError(
            "Canonical PA에 대응하는 Raw 첫 Row가 없는 PA가 있습니다."
        )

    raw_only_check = (
        raw_start.loc[:, PA_KEY]
        .merge(
            plate_appearances.loc[:, PA_KEY],
            on=list(PA_KEY),
            how="left",
            indicator=True,
            validate="one_to_one",
        )
    )
    if raw_only_check["_merge"].ne("both").any():
        raise M2DatasetBuildError(
            "Canonical PA에 존재하지 않는 Raw PA Key가 있습니다."
        )

    canonical_date = (
        pd.to_datetime(
            merged["game_date"],
            errors="raise",
        )
        .dt
        .normalize()
    )
    raw_date = (
        pd.to_datetime(
            merged["raw_game_date"],
            errors="raise",
        )
        .dt
        .normalize()
    )
    if canonical_date.ne(raw_date).any():
        raise M2DatasetBuildError(
            "Raw와 Canonical PA의 game_date가 일치하지 않습니다."
        )

    _validate_raw_canonical_context(merged)

    if (
        merged["pitch_count"]
        .ne(merged["actual_pitch_count"])
        .any()
    ):
        raise M2DatasetBuildError(
            "Raw actual Pitch 수와 Canonical pitch_count가 일치하지 않습니다."
        )

    return merged


def _map_targets(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Canonical event를 고정된 15→11 Target으로 매핑한다."""
    result = frame.copy()

    unknown_events = sorted(
        {
            str(value)
            for value in result["event"]
            .dropna()
            .unique()
            .tolist()
            if str(value) not in EVENT_TO_CLASS
        }
    )

    if unknown_events:
        raise M2DatasetBuildError(
            "알 수 없는 non-null event가 있습니다. "
            "자동 OUT/OTHER 병합은 허용하지 않습니다: "
            f"{unknown_events}"
        )

    result["source_event"] = (
        result["event"]
        .astype("string")
    )

    result["target_class"] = (
        result["source_event"]
        .map(EVENT_TO_CLASS)
        .astype("string")
    )

    result["target_code"] = (
        result["target_class"]
        .map(TARGET_CODES)
        .astype("Int64")
    )

    return result


def _build_player_features(
    frame: pd.DataFrame,
    *,
    batting_source: pd.DataFrame,
    pitching_source: pd.DataFrame,
) -> pd.DataFrame:
    """시작 타자·투수 요청을 #24 공통 Player Feature API로 한 번 계산한다."""
    batter_requests = (
        frame.loc[
            :,
            [
                "starting_batter",
                "game_date",
            ],
        ]
        .rename(
            columns={
                "starting_batter": "player_id",
                "game_date": "prediction_date",
            }
        )
        .assign(role="batting")
    )

    pitcher_requests = (
        frame.loc[
            :,
            [
                "starting_pitcher",
                "game_date",
            ],
        ]
        .rename(
            columns={
                "starting_pitcher": "player_id",
                "game_date": "prediction_date",
            }
        )
        .assign(role="pitching")
    )

    requests = (
        pd.concat(
            [
                batter_requests,
                pitcher_requests,
            ],
            ignore_index=True,
        )
        .loc[
            :,
            [
                "player_id",
                "role",
                "prediction_date",
            ],
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    try:
        features = build_player_pregame_features(
            requests,
            batting_source,
            pitching_source,
        )
    except PlayerFeatureBuildError as exc:
        raise M2DatasetBuildError(
            f"#24 Player Feature 계산에 실패했습니다: {exc}"
        ) from exc

    if (
        features["feature_version"]
        .ne(PLAYER_FEATURE_VERSION)
        .any()
    ):
        raise M2DatasetBuildError(
            "#24 Player Feature Version이 current-main 계약과 다릅니다."
        )

    batter_source_columns = (
        BATTER_BASE_SOURCE_FEATURES
        + ("has_history",)
        + BATTER_ROLLING_SOURCE_FEATURES
    )

    pitcher_source_columns = (
        PITCHER_BASE_SOURCE_FEATURES
        + ("has_history",)
        + PITCHER_ROLLING_SOURCE_FEATURES
    )

    batter = (
        features.loc[
            features["role"].eq("batting"),
            [
                "player_id",
                "prediction_date",
                "max_source_game_date",
                "history_game_count",
                *batter_source_columns,
            ],
        ]
        .copy()
    )

    pitcher = (
        features.loc[
            features["role"].eq("pitching"),
            [
                "player_id",
                "prediction_date",
                "max_source_game_date",
                "history_game_count",
                *pitcher_source_columns,
            ],
        ]
        .copy()
    )

    batter_rename = {
        "player_id": "starting_batter",
        "prediction_date": "game_date",
        "max_source_game_date": "batter_max_source_game_date",
        "history_game_count": "batter_history_game_count",
        "has_history": "batter_has_history",
    }
    batter_rename.update(
        {
            name: f"batter_{name}"
            for name in (
                BATTER_BASE_SOURCE_FEATURES
                + BATTER_ROLLING_SOURCE_FEATURES
            )
        }
    )

    pitcher_rename = {
        "player_id": "starting_pitcher",
        "prediction_date": "game_date",
        "max_source_game_date": "pitcher_max_source_game_date",
        "history_game_count": "pitcher_history_game_count",
        "has_history": "pitcher_has_history",
    }
    pitcher_rename.update(
        {
            name: f"pitcher_{name}"
            for name in (
                PITCHER_BASE_SOURCE_FEATURES
                + PITCHER_ROLLING_SOURCE_FEATURES
            )
        }
    )

    batter = batter.rename(
        columns=batter_rename
    )
    pitcher = pitcher.rename(
        columns=pitcher_rename
    )

    before_rows = len(frame)

    result = (
        frame
        .merge(
            batter,
            on=[
                "starting_batter",
                "game_date",
            ],
            how="left",
            validate="many_to_one",
            sort=False,
        )
        .merge(
            pitcher,
            on=[
                "starting_pitcher",
                "game_date",
            ],
            how="left",
            validate="many_to_one",
            sort=False,
        )
    )

    if len(result) != before_rows:
        raise M2DatasetBuildError(
            "Player Feature Join으로 PA Row가 증식했습니다."
        )

    if (
        result["batter_has_history"].isna().any()
        or result["pitcher_has_history"].isna().any()
    ):
        raise M2DatasetBuildError(
            "#24 Player Feature 요청 Row가 M2 PA에 정확히 보존되지 않았습니다."
        )

    return result


def _build_player_feature_provenance(
    frame: pd.DataFrame,
) -> dict[str, object]:
    """M2에 실제 결합된 #24 Player Feature 요청 집합의 결정적 provenance를 계산한다."""
    batter_columns = [
        "starting_batter",
        "game_date",
        "batter_max_source_game_date",
        "batter_history_game_count",
        *BATTER_X_COLUMNS,
    ]
    pitcher_columns = [
        "starting_pitcher",
        "game_date",
        "pitcher_max_source_game_date",
        "pitcher_history_game_count",
        *PITCHER_X_COLUMNS,
    ]

    batter_requests = (
        frame.loc[
            :,
            batter_columns,
        ]
        .drop_duplicates()
        .sort_values(
            [
                "game_date",
                "starting_batter",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    pitcher_requests = (
        frame.loc[
            :,
            pitcher_columns,
        ]
        .drop_duplicates()
        .sort_values(
            [
                "game_date",
                "starting_pitcher",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    batter_duplicate_keys = (
        batter_requests
        .duplicated(
            subset=[
                "starting_batter",
                "game_date",
            ],
            keep=False,
        )
    )
    if batter_duplicate_keys.any():
        raise M2DatasetBuildError(
            "동일 Batter Player Feature Request Key에 서로 다른 Feature 값이 있습니다."
        )

    pitcher_duplicate_keys = (
        pitcher_requests
        .duplicated(
            subset=[
                "starting_pitcher",
                "game_date",
            ],
            keep=False,
        )
    )
    if pitcher_duplicate_keys.any():
        raise M2DatasetBuildError(
            "동일 Pitcher Player Feature Request Key에 서로 다른 Feature 값이 있습니다."
        )

    batter_fingerprint = content_fingerprint(
        batter_requests,
        sort_columns=[
            "game_date",
            "starting_batter",
        ],
    )
    pitcher_fingerprint = content_fingerprint(
        pitcher_requests,
        sort_columns=[
            "game_date",
            "starting_pitcher",
        ],
    )

    payload: dict[str, object] = {
        "feature_version": PLAYER_FEATURE_VERSION,
        "request_count": int(
            len(batter_requests)
            + len(pitcher_requests)
        ),
        "batting_request_count": int(
            len(batter_requests)
        ),
        "pitching_request_count": int(
            len(pitcher_requests)
        ),
        "batting_content_fingerprint": batter_fingerprint,
        "pitching_content_fingerprint": pitcher_fingerprint,
    }

    payload["content_fingerprint"] = stable_json_fingerprint(
        payload
    )

    return payload


def _validate_config(
    config: Mapping[str, object],
) -> tuple[
    tuple[TemporalSplitRange, ...],
    dict[str, object],
]:
    """M2 Version, Mapping, Raw Season, Split 설정을 current-main 계약과 대조한다."""
    if (
        config.get("prediction_contract_version")
        != PREDICTION_CONTRACT_VERSION
    ):
        raise M2DatasetBuildError(
            "prediction_contract_version이 current-main 계약과 다릅니다."
        )

    if (
        config.get("feature_catalog_version")
        != FEATURE_CATALOG_VERSION
    ):
        raise M2DatasetBuildError(
            "feature_catalog_version이 current-main 계약과 다릅니다."
        )

    m2 = config.get("m2")
    if not isinstance(m2, dict):
        raise M2DatasetBuildError(
            "config.m2 object가 없습니다."
        )

    expected_versions = {
        "dataset_version": DATASET_VERSION,
        "feature_version": M2_FEATURE_VERSION,
        "target_version": TARGET_VERSION,
        "split_version": SPLIT_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "timestamp_resolution": "date",
    }

    for key, expected in expected_versions.items():
        if m2.get(key) != expected:
            raise M2DatasetBuildError(
                f"config.m2.{key}={m2.get(key)!r}가 "
                f"기대값 {expected!r}와 다릅니다."
            )

    if m2.get("raw_seasons") != list(RAW_SEASONS):
        raise M2DatasetBuildError(
            f"config.m2.raw_seasons는 {list(RAW_SEASONS)}여야 합니다."
        )

    expected_classes = [
        {
            "class": name,
            "code": TARGET_CODES[name],
        }
        for name in TARGET_CLASSES
    ]

    if m2.get("target_classes") != expected_classes:
        raise M2DatasetBuildError(
            "config.m2.target_classes가 고정 11-class 순서와 다릅니다."
        )

    if m2.get("event_mapping") != EVENT_TO_CLASS:
        raise M2DatasetBuildError(
            "config.m2.event_mapping이 고정 15→11 mapping과 다릅니다."
        )

    raw_ranges = m2.get("split_ranges")
    if not isinstance(raw_ranges, list):
        raise M2DatasetBuildError(
            "config.m2.split_ranges가 list가 아닙니다."
        )

    try:
        split_ranges = parse_split_ranges(
            raw_ranges
        )
    except ProcessedContractError as exc:
        raise _wrap_contract_error(exc) from exc

    expected_names = [
        "train",
        "validation",
        "test",
        "snapshot",
    ]
    if (
        [item.name for item in split_ranges]
        != expected_names
    ):
        raise M2DatasetBuildError(
            "M2 Split 순서는 train/validation/test/snapshot이어야 합니다."
        )

    expected_bounds = (
        ("2023-01-01", "2024-01-01"),
        ("2024-01-01", "2025-01-01"),
        ("2025-01-01", "2026-01-01"),
        ("2026-01-01", "2027-01-01"),
    )

    for split_range, (
        start,
        end_exclusive,
    ) in zip(
        split_ranges,
        expected_bounds,
    ):
        if (
            split_range.start != pd.Timestamp(start)
            or split_range.end_exclusive
            != pd.Timestamp(end_exclusive)
        ):
            raise M2DatasetBuildError(
                f"{split_range.name} Split 경계가 current-main #22 계약과 다릅니다."
            )

    return split_ranges, m2


def _build_diagnostic_reasons(
    frame: pd.DataFrame,
) -> pd.Series:
    """행별 진단 원인을 결정적인 문자열로 구성한다."""
    values: list[str | pd._libs.missing.NAType] = []

    for row in frame.itertuples(index=False):
        reasons: list[str] = []

        if bool(row.pitchless_pa):
            reasons.append("pitchless_pa")

        if bool(row.starting_stand_missing):
            reasons.append("starting_stand_missing")

        if bool(row.batter_identity_diff):
            reasons.append("starting_batter_differs_from_credited")

        if bool(row.pitcher_identity_diff):
            reasons.append("starting_pitcher_differs_from_final")

        if bool(row.stand_identity_diff):
            reasons.append("starting_stand_differs_from_credited")

        if row.start_context_quality == "unverified_missing_initial_pitch_rows":
            reasons.append("start_context_not_fully_observed")

        if pd.isna(row.source_event):
            reasons.append("incomplete_pa_event_null")

        values.append(
            ";".join(reasons)
            if reasons
            else pd.NA
        )

    return pd.Series(
        values,
        index=frame.index,
        dtype="string",
    )


def _build_output_roles() -> dict[str, list[str]]:
    """M2 Output의 물리적 역할 분리를 반환한다."""
    return {
        "key_columns": list(KEY_COLUMNS),
        "audit_columns": list(AUDIT_COLUMNS),
        "x_columns": list(M2_X_COLUMNS),
        "y_columns": list(Y_COLUMNS),
        "split_columns": list(SPLIT_COLUMNS),
        "quality_columns": list(QUALITY_COLUMNS),
        "control_columns": list(CONTROL_COLUMNS),
        "label_status_columns": list(LABEL_STATUS_COLUMNS),
    }


def build_m2_dataset(
    plate_appearances: pd.DataFrame,
    raw_pbp: pd.DataFrame,
    batting_source: pd.DataFrame,
    pitching_source: pd.DataFrame,
    *,
    config: Mapping[str, object],
) -> tuple[
    pd.DataFrame,
    dict[str, list[str]],
    dict[str, object],
]:
    """PA-start Context와 시작 선수 Historical Feature를 M2 Dataset으로 조립한다."""
    split_ranges, _ = _validate_config(
        config
    )

    pa = _normalize_plate_appearances(
        plate_appearances
    )

    raw_start = build_raw_start_context(
        raw_pbp
    )

    base = _merge_raw_start_context(
        pa,
        raw_start,
    )

    base = _map_targets(
        base
    )

    base = _build_player_features(
        base,
        batting_source=batting_source,
        pitching_source=pitching_source,
    )

    base["credited_batter"] = (
        base["batter"]
        .astype("string")
    )
    base["final_pitcher"] = (
        base["pitcher"]
        .astype("string")
    )
    base["credited_stand"] = (
        base["stand"]
        .astype("string")
    )

    base["on_1b_occupied"] = (
        base["on_1b_before"]
        .notna()
        .astype("boolean")
    )
    base["on_2b_occupied"] = (
        base["on_2b_before"]
        .notna()
        .astype("boolean")
    )
    base["on_3b_occupied"] = (
        base["on_3b_before"]
        .notna()
        .astype("boolean")
    )

    base["starting_stand_missing"] = (
        base["starting_stand"]
        .isna()
        .astype("boolean")
    )

    base["batter_identity_diff"] = (
        _different(
            base["starting_batter"],
            base["credited_batter"],
        )
        .astype("boolean")
    )
    base["pitcher_identity_diff"] = (
        _different(
            base["starting_pitcher"],
            base["final_pitcher"],
        )
        .astype("boolean")
    )
    base["stand_identity_diff"] = (
        _different(
            base["starting_stand"],
            base["credited_stand"],
        )
        .astype("boolean")
    )

    base["prediction_date"] = (
        base["game_date"]
        .astype("datetime64[us]")
    )

    game_date_counts = (
        base.groupby(
            "game_pk",
            sort=False,
            dropna=False,
        )["prediction_date"]
        .nunique(
            dropna=False
        )
    )
    inconsistent_game_dates = (
        game_date_counts.loc[
            game_date_counts.ne(1)
        ]
    )
    if not inconsistent_game_dates.empty:
        sample = (
            inconsistent_game_dates
            .head(10)
            .index
            .astype(str)
            .tolist()
        )
        raise M2DatasetBuildError(
            "동일 game_pk에 서로 다른 prediction_date가 있습니다: "
            f"sample={sample}"
        )

    label_available = pd.Series(
        pd.array(
            [pd.NaT] * len(base),
            dtype="datetime64[us]",
        ),
        index=base.index,
    )

    completed_target = (
        base["target_class"]
        .notna()
    )
    label_available.loc[
        completed_target
    ] = (
        base.loc[
            completed_target,
            "game_date",
        ]
        + pd.Timedelta(days=1)
    )

    base["label_available_at"] = (
        label_available
        .astype("datetime64[us]")
    )

    base["availability_basis"] = pd.Series(
        AVAILABILITY_BASIS,
        index=base.index,
        dtype="string",
    )

    base["contract_version"] = pd.Series(
        PREDICTION_CONTRACT_VERSION,
        index=base.index,
        dtype="string",
    )
    base["feature_catalog_version"] = pd.Series(
        FEATURE_CATALOG_VERSION,
        index=base.index,
        dtype="string",
    )
    base["feature_version"] = pd.Series(
        M2_FEATURE_VERSION,
        index=base.index,
        dtype="string",
    )
    base["player_feature_version"] = pd.Series(
        PLAYER_FEATURE_VERSION,
        index=base.index,
        dtype="string",
    )
    base["target_version"] = pd.Series(
        TARGET_VERSION,
        index=base.index,
        dtype="string",
    )
    base["split_version"] = pd.Series(
        SPLIT_VERSION,
        index=base.index,
        dtype="string",
    )
    base["output_schema_version"] = pd.Series(
        OUTPUT_SCHEMA_VERSION,
        index=base.index,
        dtype="string",
    )

    try:
        base["split"] = assign_temporal_split_series(
            base["prediction_date"],
            split_ranges,
        )
    except ProcessedContractError as exc:
        raise _wrap_contract_error(exc) from exc

    game_split_counts = (
        base.groupby(
            "game_pk",
            sort=False,
            dropna=False,
        )["split"]
        .nunique(
            dropna=False
        )
    )
    inconsistent_game_splits = (
        game_split_counts.loc[
            game_split_counts.ne(1)
        ]
    )
    if not inconsistent_game_splits.empty:
        sample = (
            inconsistent_game_splits
            .head(10)
            .index
            .astype(str)
            .tolist()
        )
        raise M2DatasetBuildError(
            "동일 game_pk의 PA가 서로 다른 Split에 배정되었습니다: "
            f"sample={sample}"
        )

    partial_start = (
        base["start_context_quality"]
        .eq("unverified_missing_initial_pitch_rows")
    )

    base["is_excluded"] = (
        partial_start
        .astype("boolean")
    )
    base["exclusion_reason"] = pd.Series(
        pd.NA,
        index=base.index,
        dtype="string",
    )
    base.loc[
        partial_start,
        "exclusion_reason",
    ] = "start_context_not_fully_observed"

    incomplete_target = (
        base["source_event"]
        .isna()
    )

    base["target_eligible"] = (
        ~base["is_excluded"].fillna(True)
        & base["target_class"].notna()
    ).astype("boolean")

    base["is_censored"] = (
        ~base["is_excluded"].fillna(True)
        & incomplete_target
    ).astype("boolean")

    base["censor_reason"] = pd.Series(
        pd.NA,
        index=base.index,
        dtype="string",
    )
    base.loc[
        base["is_censored"].fillna(False),
        "censor_reason",
    ] = "event_null_incomplete_pa"

    base["coverage_status"] = pd.Series(
        "complete_pa",
        index=base.index,
        dtype="string",
    )
    base.loc[
        incomplete_target,
        "coverage_status",
    ] = "incomplete_pa"

    base["is_purged"] = pd.Series(
        False,
        index=base.index,
        dtype="boolean",
    )
    base["purge_reason"] = pd.Series(
        pd.NA,
        index=base.index,
        dtype="string",
    )

    base["supervised_usable"] = (
        base["target_eligible"]
        & ~base["is_censored"]
        & ~base["is_purged"]
    ).astype("boolean")

    label_status: list[str] = []

    for row in base.itertuples(index=False):
        if bool(row.is_excluded):
            label_status.append("excluded")
        elif bool(row.is_censored):
            label_status.append("target_pending")
        elif bool(row.supervised_usable):
            label_status.append("usable")
        else:
            label_status.append("held")

    base["label_status"] = pd.Series(
        label_status,
        index=base.index,
        dtype="string",
    )

    base["diagnostic_reasons"] = (
        _build_diagnostic_reasons(
            base
        )
    )

    roles = _build_output_roles()

    output_columns = (
        KEY_COLUMNS
        + AUDIT_COLUMNS
        + M2_X_COLUMNS
        + Y_COLUMNS
        + SPLIT_COLUMNS
        + QUALITY_COLUMNS
        + CONTROL_COLUMNS
        + LABEL_STATUS_COLUMNS
    )

    missing_output = [
        column
        for column in output_columns
        if column not in base.columns
    ]
    if missing_output:
        raise M2DatasetBuildError(
            "내부 오류: M2 Output 컬럼이 누락되었습니다: "
            f"{missing_output}"
        )

    result = (
        base.loc[
            :,
            output_columns,
        ]
        .sort_values(
            [
                "season",
                "game_date",
                "game_pk",
                "at_bat_number",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    if len(result) != len(pa):
        raise M2DatasetBuildError(
            "M2 Output Row 수가 Canonical PA 수와 일치하지 않습니다."
        )

    duplicate_count = int(
        result
        .duplicated(
            subset=list(PA_KEY),
            keep=False,
        )
        .sum()
    )
    if duplicate_count:
        raise M2DatasetBuildError(
            f"M2 Output PA Key 중복이 {duplicate_count}건 있습니다."
        )

    all_role_columns: list[str] = []
    for columns in roles.values():
        all_role_columns.extend(
            columns
        )

    if len(all_role_columns) != len(
        set(all_role_columns)
    ):
        duplicates = [
            column
            for column, count in Counter(
                all_role_columns
            ).items()
            if count > 1
        ]
        raise M2DatasetBuildError(
            f"M2 Output Schema 역할이 중복되었습니다: {duplicates}"
        )

    try:
        validate_x_allowlist(
            available_columns=result.columns,
            x_columns=M2_X_COLUMNS,
            forbidden_columns=FORBIDDEN_X_COLUMNS,
            forbidden_prefixes=FORBIDDEN_X_PREFIXES,
        )
    except ProcessedContractError as exc:
        raise _wrap_contract_error(exc) from exc

    summary = build_quality_summary(
        result
    )
    summary["player_feature_provenance"] = (
        _build_player_feature_provenance(
            result
        )
    )

    return (
        result,
        roles,
        summary,
    )


def _value_counts(
    series: pd.Series,
) -> dict[str, int]:
    """Nullable 값을 포함한 결정적 value_counts Mapping을 만든다."""
    result: dict[str, int] = {}

    counts = (
        series
        .value_counts(
            dropna=False,
        )
        .sort_index(
            key=lambda index: index.astype(str)
        )
    )

    for key, value in counts.items():
        label = (
            "<NULL>"
            if pd.isna(key)
            else str(key)
        )
        result[label] = int(value)

    return result


def _build_event_mapping_reconciliation(
    dataset: pd.DataFrame,
) -> dict[str, dict[str, object]]:
    """Source Event별 Target Mapping과 Row 상태 수량을 결정적으로 대사한다."""
    reconciliation: dict[
        str,
        dict[str, object],
    ] = {}

    reconciled_total = 0

    for source_event in (
        *EVENT_TO_CLASS.keys(),
        "<NULL>",
    ):
        if source_event == "<NULL>":
            mask = (
                dataset["source_event"]
                .isna()
            )
            expected_target: str | None = None
            expected_code: int | None = None
        else:
            mask = (
                dataset["source_event"]
                .eq(source_event)
                .fillna(False)
            )
            expected_target = EVENT_TO_CLASS[
                source_event
            ]
            expected_code = TARGET_CODES[
                expected_target
            ]

        rows = dataset.loc[
            mask
        ]

        input_rows = int(
            len(rows)
        )
        supervised = int(
            rows["supervised_usable"]
            .fillna(False)
            .sum()
        )
        pending = int(
            rows["is_censored"]
            .fillna(False)
            .sum()
        )
        excluded = int(
            rows["is_excluded"]
            .fillna(False)
            .sum()
        )

        reconciled_rows = (
            supervised
            + pending
            + excluded
        )

        if reconciled_rows != input_rows:
            raise M2DatasetBuildError(
                "Source Event별 Row 상태 대사가 실패했습니다: "
                f"event={source_event!r}, input={input_rows}, "
                f"supervised={supervised}, pending={pending}, "
                f"excluded={excluded}"
            )

        if expected_target is None:
            if (
                rows["target_class"]
                .notna()
                .any()
                or rows["target_code"]
                .notna()
                .any()
            ):
                raise M2DatasetBuildError(
                    "event IS NULL Row는 Target Class/Code도 null이어야 합니다."
                )
        else:
            invalid_target = (
                rows["target_class"]
                .isna()
                | rows["target_class"]
                .ne(expected_target)
            )
            invalid_code = (
                rows["target_code"]
                .isna()
                | rows["target_code"]
                .ne(expected_code)
            )
            if (
                invalid_target.any()
                or invalid_code.any()
            ):
                raise M2DatasetBuildError(
                    "Source Event와 고정 Target Mapping이 일치하지 않습니다: "
                    f"event={source_event!r}, "
                    f"target={expected_target!r}, "
                    f"code={expected_code!r}"
                )

        reconciliation[
            source_event
        ] = {
            "target_class": expected_target,
            "target_code": expected_code,
            "input_rows": input_rows,
            "supervised_usable": supervised,
            "target_pending_or_held": pending,
            "excluded": excluded,
            "reconciled_rows": reconciled_rows,
        }

        reconciled_total += input_rows

    if reconciled_total != len(
        dataset
    ):
        raise M2DatasetBuildError(
            "Source Event별 입력 수량 합이 전체 PA Row 수와 일치하지 않습니다: "
            f"{reconciled_total} != {len(dataset)}"
        )

    return reconciliation


def build_quality_summary(
    dataset: pd.DataFrame,
) -> dict[str, object]:
    """M2 입력·Supervised·Pending·Excluded 및 Event Mapping 수량을 대사한다."""
    total = int(
        len(dataset)
    )
    supervised = int(
        dataset["supervised_usable"]
        .fillna(False)
        .sum()
    )
    excluded = int(
        dataset["is_excluded"]
        .fillna(False)
        .sum()
    )
    pending = int(
        dataset["is_censored"]
        .fillna(False)
        .sum()
    )

    reconciled = (
        supervised
        + excluded
        + pending
    )

    if reconciled != total:
        raise M2DatasetBuildError(
            "M2 Row 상태 대사가 실패했습니다: "
            f"total={total}, supervised={supervised}, "
            f"pending={pending}, excluded={excluded}"
        )

    event_mapping_reconciliation = (
        _build_event_mapping_reconciliation(
            dataset
        )
    )

    return {
        "input_plate_appearances": total,
        "supervised_usable": supervised,
        "target_pending_or_held": pending,
        "excluded": excluded,
        "reconciled_total": reconciled,
        "pitchless_pa": int(
            dataset["pitchless_pa"]
            .fillna(False)
            .sum()
        ),
        "starting_stand_null": int(
            dataset["starting_stand"]
            .isna()
            .sum()
        ),
        "batter_identity_diff": int(
            dataset["batter_identity_diff"]
            .fillna(False)
            .sum()
        ),
        "pitcher_identity_diff": int(
            dataset["pitcher_identity_diff"]
            .fillna(False)
            .sum()
        ),
        "stand_identity_diff": int(
            dataset["stand_identity_diff"]
            .fillna(False)
            .sum()
        ),
        "split_counts": _value_counts(
            dataset["split"]
        ),
        "source_event_counts": _value_counts(
            dataset["source_event"]
        ),
        "target_class_counts": _value_counts(
            dataset["target_class"]
        ),
        "event_mapping_reconciliation": (
            event_mapping_reconciliation
        ),
        "start_context_quality_counts": _value_counts(
            dataset["start_context_quality"]
        ),
        "primary_exclusion_reason_counts": _value_counts(
            dataset.loc[
                dataset["is_excluded"]
                .fillna(False),
                "exclusion_reason",
            ]
        ),
    }


def build_schema_payload(
    dataset: pd.DataFrame,
    roles: Mapping[
        str,
        Sequence[str],
    ],
) -> dict[str, object]:
    """M2 Output 역할·dtype·Version·Target Mapping을 Schema Artifact로 직렬화한다."""
    return {
        "artifact": "m2_matchup_dataset_schema",
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "dataset_version": DATASET_VERSION,
        "prediction_contract_version": PREDICTION_CONTRACT_VERSION,
        "feature_catalog_version": FEATURE_CATALOG_VERSION,
        "feature_version": M2_FEATURE_VERSION,
        "player_feature_version": PLAYER_FEATURE_VERSION,
        "target_version": TARGET_VERSION,
        "split_version": SPLIT_VERSION,
        "grain": list(PA_KEY),
        "roles": {
            key: list(value)
            for key, value in roles.items()
        },
        "columns": schema_manifest(
            dataset
        ),
        "source_event_domain": list(
            EVENT_TO_CLASS.keys()
        ),
        "event_mapping": dict(
            EVENT_TO_CLASS
        ),
        "target_classes": [
            {
                "class": name,
                "code": TARGET_CODES[name],
            }
            for name in TARGET_CLASSES
        ],
        "model_ready_semantics": (
            "PA 시작 Context와 시작 선수의 전일까지 Historical Feature를 "
            "X로 사용하는 Processed Dataset이며 모델 학습·성능 검증 완료를 의미하지 않음"
        ),
    }


def build_quality_report(
    dataset: pd.DataFrame,
    summary: Mapping[str, object],
) -> dict[str, object]:
    """제외·미완료·Context 품질과 Event Mapping 대사를 Quality Report로 만든다."""
    excluded_rows = dataset.loc[
        dataset["is_excluded"]
        .fillna(False),
        [
            "game_pk",
            "at_bat_number",
            "first_observed_pitch_number",
            "exclusion_reason",
            "diagnostic_reasons",
        ],
    ]

    pending_rows = dataset.loc[
        dataset["is_censored"]
        .fillna(False),
        [
            "game_pk",
            "at_bat_number",
            "censor_reason",
            "diagnostic_reasons",
        ],
    ]

    return {
        "artifact": "m2_matchup_dataset_quality_report",
        "dataset_version": DATASET_VERSION,
        "prediction_contract_version": PREDICTION_CONTRACT_VERSION,
        "feature_catalog_version": FEATURE_CATALOG_VERSION,
        "feature_version": M2_FEATURE_VERSION,
        "player_feature_version": PLAYER_FEATURE_VERSION,
        "target_version": TARGET_VERSION,
        "split_version": SPLIT_VERSION,
        "summary": dict(
            summary
        ),
        "excluded_rows": excluded_rows.to_dict(
            "records"
        ),
        "pending_rows": pending_rows.to_dict(
            "records"
        ),
    }


def _coerce_round_trip_dtypes(
    frame: pd.DataFrame,
    reference: pd.DataFrame,
) -> pd.DataFrame:
    """Parquet round-trip 결과를 생성 직전 dtype 계약으로 복원한다."""
    if list(frame.columns) != list(
        reference.columns
    ):
        raise M2DatasetBuildError(
            "Parquet round-trip 후 Column 순서가 변경되었습니다."
        )

    if len(frame) != len(reference):
        raise M2DatasetBuildError(
            "Parquet round-trip 후 Row 수가 변경되었습니다."
        )

    result = frame.copy()

    for column in reference.columns:
        expected_dtype = str(
            reference[column].dtype
        )

        try:
            if expected_dtype.startswith(
                "datetime64"
            ):
                result[column] = (
                    pd.to_datetime(
                        result[column],
                        errors="raise",
                    )
                    .astype("datetime64[us]")
                )
            else:
                result[column] = (
                    result[column]
                    .astype(expected_dtype)
                )
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise M2DatasetBuildError(
                "Parquet round-trip dtype 복원에 실패했습니다: "
                f"{column} -> {expected_dtype}"
            ) from exc

    return result


def _validate_raw_manifest(
    manifest: pd.DataFrame,
    *,
    raw_paths: Mapping[int, Path],
    raw_schema_path: Path,
) -> str:
    """Raw 시즌 파일과 schema가 하나의 fixed revision 및 SHA256과 일치하는지 검증한다."""
    required = (
        "remote_filename",
        "season",
        "resolved_revision",
        "sha256",
    )
    _require_columns(
        manifest,
        required,
        source_name="download_manifest",
    )

    season_values = pd.to_numeric(
        manifest["season"],
        errors="coerce",
    )

    revision_values: list[str] = []

    for season in RAW_SEASONS:
        rows = manifest.loc[
            season_values.eq(season)
        ]

        if len(rows) != 1:
            raise M2DatasetBuildError(
                f"download_manifest의 {season} 시즌 Row는 정확히 1개여야 합니다."
            )

        row = rows.iloc[0]
        expected_hash = str(
            row["sha256"]
        ).lower()
        actual_hash = calculate_sha256(
            raw_paths[season]
        ).lower()

        if actual_hash != expected_hash:
            raise M2DatasetBuildError(
                f"{season} Raw SHA256이 download_manifest와 다릅니다."
            )

        revision_values.append(
            str(
                row["resolved_revision"]
            )
        )

    schema_rows = manifest.loc[
        manifest["remote_filename"]
        .astype("string")
        .str
        .endswith("schema.yaml")
        .fillna(False)
    ]

    if len(schema_rows) != 1:
        raise M2DatasetBuildError(
            "download_manifest의 schema.yaml Row는 정확히 1개여야 합니다."
        )

    schema_row = schema_rows.iloc[0]
    expected_schema_hash = str(
        schema_row["sha256"]
    ).lower()
    actual_schema_hash = calculate_sha256(
        raw_schema_path
    ).lower()

    if actual_schema_hash != expected_schema_hash:
        raise M2DatasetBuildError(
            "Raw schema.yaml SHA256이 download_manifest와 다릅니다."
        )

    revision_values.append(
        str(
            schema_row["resolved_revision"]
        )
    )

    revisions = sorted(
        set(revision_values)
    )

    if len(revisions) != 1:
        raise M2DatasetBuildError(
            "Raw 시즌 파일과 schema가 하나의 fixed revision을 공유하지 않습니다: "
            f"{revisions}"
        )

    return revisions[0]


def build_deterministic_identity(
    *,
    raw_revision: str,
    input_hashes: Mapping[str, str],
    dataset_content_fingerprint: str,
    schema_payload: Mapping[str, object],
    quality_payload: Mapping[str, object],
) -> dict[str, object]:
    """실행 시각과 독립적인 M2 Artifact Set Identity를 계산한다."""
    identity: dict[str, object] = {
        "dataset_version": DATASET_VERSION,
        "prediction_contract_version": PREDICTION_CONTRACT_VERSION,
        "feature_catalog_version": FEATURE_CATALOG_VERSION,
        "feature_version": M2_FEATURE_VERSION,
        "player_feature_version": PLAYER_FEATURE_VERSION,
        "target_version": TARGET_VERSION,
        "split_version": SPLIT_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "raw_revision": raw_revision,
        "input_sha256": {
            name: input_hashes[name]
            for name in sorted(
                input_hashes
            )
        },
        "dataset_content_fingerprint": dataset_content_fingerprint,
        "schema_fingerprint": stable_json_fingerprint(
            schema_payload
        ),
        "quality_report_fingerprint": stable_json_fingerprint(
            quality_payload
        ),
    }

    identity["fingerprint"] = stable_json_fingerprint(
        identity
    )

    return identity


def resolve_code_revision(
    project_root: Path,
) -> str:
    """Read-only git rev-parse로 Code Revision을 읽는다."""
    try:
        completed = subprocess.run(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (
        OSError,
        subprocess.SubprocessError,
    ):
        return "unknown"

    revision = (
        completed.stdout
        .strip()
    )
    return revision or "unknown"


def build_m2_dataset_files(
    *,
    config_path: Path,
    raw_manifest_path: Path,
    raw_schema_path: Path,
    raw_paths: Mapping[int, Path],
    plate_appearances_path: Path,
    batting_path: Path,
    pitching_path: Path,
    output_path: Path,
    manifest_path: Path,
    schema_path: Path,
    quality_report_path: Path,
    project_root: Path,
    code_revision: str | None = None,
) -> tuple[
    pd.DataFrame,
    dict[str, object],
]:
    """M2 Artifact Set을 생성하고 round-trip 검증 후 완료 Manifest를 마지막에 기록한다."""
    expected_seasons = set(
        RAW_SEASONS
    )
    if set(raw_paths) != expected_seasons:
        raise M2DatasetBuildError(
            "raw_paths 시즌 Key가 "
            f"{sorted(expected_seasons)}와 일치해야 합니다."
        )

    input_paths: dict[str, Path] = {
        "config": config_path,
        "raw_manifest": raw_manifest_path,
        "raw_schema": raw_schema_path,
        "plate_appearances": plate_appearances_path,
        "player_game_batting": batting_path,
        "player_game_pitching": pitching_path,
    }

    for season in RAW_SEASONS:
        input_paths[
            f"raw_{season}"
        ] = raw_paths[
            season
        ]

    try:
        validate_output_paths(
            input_paths=list(
                input_paths.values()
            ),
            output_paths=[
                output_path,
                manifest_path,
                schema_path,
                quality_report_path,
            ],
            project_root=project_root,
        )
        invalidate_completion_marker(
            manifest_path
        )
    except ProcessedContractError as exc:
        raise _wrap_contract_error(exc) from exc

    before_hashes = calculate_input_hashes(
        input_paths
    )

    config = read_json(
        config_path,
        label="processed_dataset config",
    )
    _validate_config(
        config
    )

    try:
        raw_manifest = pd.read_csv(
            raw_manifest_path
        )
    except Exception as exc:
        raise M2DatasetBuildError(
            f"Raw download_manifest를 읽지 못했습니다: {raw_manifest_path}"
        ) from exc

    raw_revision = _validate_raw_manifest(
        raw_manifest,
        raw_paths=raw_paths,
        raw_schema_path=raw_schema_path,
    )

    raw_frames: list[
        pd.DataFrame
    ] = []
    raw_row_counts: dict[
        str,
        int,
    ] = {}

    for season in RAW_SEASONS:
        path = raw_paths[
            season
        ]

        if not path.is_file():
            raise M2DatasetBuildError(
                f"{season} Raw Parquet이 없습니다: {path}"
            )

        try:
            frame = pd.read_parquet(
                path,
                columns=list(
                    RAW_REQUIRED_COLUMNS
                ),
                engine="pyarrow",
            )
        except Exception as exc:
            raise M2DatasetBuildError(
                f"{season} Raw Parquet을 읽지 못했습니다: {path}"
            ) from exc

        parsed_year = (
            _normalize_date(
                frame["game_date"],
                source_name=f"raw_{season}.game_date",
            )
            .dt
            .year
        )
        if parsed_year.ne(season).any():
            raise M2DatasetBuildError(
                f"{season} Raw 파일에 다른 연도 game_date가 있습니다."
            )

        raw_row_counts[
            str(season)
        ] = int(
            len(frame)
        )
        raw_frames.append(
            frame
        )

    raw_pbp = pd.concat(
        raw_frames,
        ignore_index=True,
    )

    plate_appearances = read_parquet(
        plate_appearances_path,
        label="plate_appearances",
    )
    batting_source = read_parquet(
        batting_path,
        label="player_game_batting",
    )
    pitching_source = read_parquet(
        pitching_path,
        label="player_game_pitching",
    )

    dataset, roles, summary = build_m2_dataset(
        plate_appearances,
        raw_pbp,
        batting_source,
        pitching_source,
        config=config,
    )

    expected_fingerprint = content_fingerprint(
        dataset,
        sort_columns=list(
            PA_KEY
        ),
    )

    write_parquet_atomic(
        dataset,
        output_path,
    )

    round_trip_raw = read_parquet(
        output_path,
        label="m2 matchup_dataset",
    )
    round_trip = _coerce_round_trip_dtypes(
        round_trip_raw,
        dataset,
    )

    duplicate_count = int(
        round_trip
        .duplicated(
            subset=list(PA_KEY),
            keep=False,
        )
        .sum()
    )
    if duplicate_count:
        raise M2DatasetBuildError(
            "Parquet round-trip 후 M2 PA Key 중복이 있습니다."
        )

    actual_fingerprint = content_fingerprint(
        round_trip,
        sort_columns=list(
            PA_KEY
        ),
    )

    if (
        actual_fingerprint
        != expected_fingerprint
    ):
        raise M2DatasetBuildError(
            "Parquet round-trip 후 M2 content fingerprint가 변경되었습니다."
        )

    try:
        validate_x_allowlist(
            available_columns=round_trip.columns,
            x_columns=roles["x_columns"],
            forbidden_columns=FORBIDDEN_X_COLUMNS,
            forbidden_prefixes=FORBIDDEN_X_PREFIXES,
        )
    except ProcessedContractError as exc:
        raise _wrap_contract_error(exc) from exc

    schema_payload = build_schema_payload(
        round_trip,
        roles,
    )
    quality_payload = build_quality_report(
        round_trip,
        summary,
    )

    write_json_atomic(
        schema_payload,
        schema_path,
    )
    write_json_atomic(
        quality_payload,
        quality_report_path,
    )

    after_hashes = assert_input_hashes_unchanged(
        before_hashes,
        input_paths,
    )

    deterministic_identity = build_deterministic_identity(
        raw_revision=raw_revision,
        input_hashes=after_hashes,
        dataset_content_fingerprint=actual_fingerprint,
        schema_payload=schema_payload,
        quality_payload=quality_payload,
    )

    resolved_revision = (
        code_revision
        or resolve_code_revision(
            project_root
        )
    )

    created_at = datetime.now(
        timezone.utc
    )
    created_at_text = (
        created_at.isoformat()
    )

    run_id = (
        f"m2-{created_at.strftime('%Y%m%dT%H%M%S%fZ')}-"
        f"{str(deterministic_identity['fingerprint'])[:12]}"
    )

    prediction_min = (
        round_trip["prediction_date"]
        .min()
    )
    prediction_max = (
        round_trip["prediction_date"]
        .max()
    )

    manifest = {
        "artifact": "m2_matchup_dataset",
        "dataset_name": "m2_matchup_dataset",
        "run_id": run_id,
        "run_status": "complete",
        "created_at": created_at_text,
        "created_at_utc": created_at_text,
        "code_revision": resolved_revision,
        "raw_revision": raw_revision,
        "dataset_version": DATASET_VERSION,
        "contract_version": PREDICTION_CONTRACT_VERSION,
        "prediction_contract_version": PREDICTION_CONTRACT_VERSION,
        "feature_catalog_version": FEATURE_CATALOG_VERSION,
        "feature_version": M2_FEATURE_VERSION,
        "player_feature_version": PLAYER_FEATURE_VERSION,
        "player_feature_provenance": summary[
            "player_feature_provenance"
        ],
        "target_version": TARGET_VERSION,
        "split_version": SPLIT_VERSION,
        "timestamp_resolution": config["m2"]["timestamp_resolution"],
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "grain": list(PA_KEY),
        "config": config,
        "roles": roles,
        "input_paths": {
            name: str(path)
            for name, path in input_paths.items()
        },
        "input_sha256": {
            name: after_hashes[name]
            for name in sorted(
                after_hashes
            )
        },
        "prediction_range": {
            "min": (
                prediction_min.isoformat()
                if not pd.isna(
                    prediction_min
                )
                else None
            ),
            "max": (
                prediction_max.isoformat()
                if not pd.isna(
                    prediction_max
                )
                else None
            ),
        },
        "source_coverage": {
            "raw_row_counts": raw_row_counts,
            "raw_total_rows": int(
                len(raw_pbp)
            ),
            "plate_appearance_rows": int(
                len(plate_appearances)
            ),
            "player_game_batting_rows": int(
                len(batting_source)
            ),
            "player_game_pitching_rows": int(
                len(pitching_source)
            ),
            "player_feature_request_count": int(
                summary[
                    "player_feature_provenance"
                ][
                    "request_count"
                ]
            ),
            "split_counts": summary[
                "split_counts"
            ],
        },
        "row_count": int(
            len(round_trip)
        ),
        "usable_row_count": int(
            summary[
                "supervised_usable"
            ]
        ),
        "excluded_row_count": int(
            summary[
                "excluded"
            ]
        ),
        "censored_row_count": int(
            summary[
                "target_pending_or_held"
            ]
        ),
        "purged_row_count": int(
            round_trip[
                "is_purged"
            ]
            .fillna(False)
            .sum()
        ),
        "null_summary": {
            column: int(
                round_trip[column]
                .isna()
                .sum()
            )
            for column in round_trip.columns
        },
        "quality_summary": summary,
        "output_content_fingerprint": actual_fingerprint,
        "output": {
            "dataset_path": str(
                output_path
            ),
            "schema_path": str(
                schema_path
            ),
            "quality_report_path": str(
                quality_report_path
            ),
            "row_count": int(
                len(round_trip)
            ),
            "content_fingerprint": actual_fingerprint,
            "schema": schema_manifest(
                round_trip
            ),
        },
        "summary": summary,
        "deterministic_identity": deterministic_identity,
    }

    write_json_atomic(
        manifest,
        manifest_path,
    )

    return (
        round_trip,
        manifest,
    )