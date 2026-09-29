from __future__ import annotations

import json
import logging
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


LOGGER = logging.getLogger(__name__)

DATASET_VERSION = "m1_game_dataset_v1"
TARGET_VERSION = "m1_target_v1"
OUTPUT_SCHEMA_VERSION = "m1_game_dataset_schema_v1"
TARGET_SEMANTICS = "canonical_observed_terminal"
AVAILABILITY_BASIS = "conservative_next_date"
M1_FEATURE_VERSION = "extended_v1"
TEAM_FEATURE_VERSION = "team_pregame_v1"

TEAM_BASELINE_X_COLUMNS = (
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
TEAM_ROLLING_WINDOWS = (5, 10, 20)
TEAM_ROLLING_SUFFIXES = (
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
TEAM_ROLLING_X_COLUMNS = tuple(
    f"last_{window}g_{suffix}"
    for window in TEAM_ROLLING_WINDOWS
    for suffix in TEAM_ROLLING_SUFFIXES
)
TEAM_SOURCE_X_COLUMNS = TEAM_BASELINE_X_COLUMNS + TEAM_ROLLING_X_COLUMNS
M1_DIFFERENCE_SOURCES = (
    ("diff_hist_win_pct", "hist_win_pct"),
    ("diff_hist_runs_for_per_game", "hist_runs_for_per_game"),
    ("diff_hist_runs_against_per_game", "hist_runs_against_per_game"),
    ("diff_hist_run_diff_per_game", "hist_run_diff_per_game"),
)
M1_DIFFERENCE_COLUMNS = tuple(name for name, _ in M1_DIFFERENCE_SOURCES)

HOME_RESULT_CLASSES = ("loss", "tie", "win")
HOME_RESULT_CODES = {"loss": 0, "tie": 1, "win": 2}

GAME_KEY = ("game_pk",)
TEAM_FEATURE_KEY = ("game_pk", "team")

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

REQUIRED_TEAM_FEATURE_AUDIT_COLUMNS = (
    "opponent",
    "prediction_date",
    "season",
    "is_home",
    "max_source_game_date",
    "history_game_count",
    "has_history",
    "feature_version",
)

REQUIRED_PA_AUDIT_COLUMNS = (
    "game_pk",
    "at_bat_number",
    "post_home_score",
    "post_away_score",
    "pa_completed",
)

KEY_COLUMNS = (
    "game_pk",
)

AUDIT_COLUMNS = (
    "game_date",
    "season",
    "home_team",
    "away_team",
    "prediction_date",
    "label_available_at",
    "availability_basis",
    "target_semantics",
    "contract_version",
    "feature_catalog_version",
    "team_feature_version",
    "target_version",
    "split_version",
    "output_schema_version",
    "home_max_source_game_date",
    "away_max_source_game_date",
    "home_history_game_count",
    "away_history_game_count",
)

Y_COLUMNS = (
    "home_result",
    "home_result_code",
    "home_runs",
    "away_runs",
)

SPLIT_COLUMNS = (
    "split",
)

QUALITY_COLUMNS = (
    "observation_quality",
    "observation_quality_reason",
    "official_status_verified",
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
        "pa_completed",
    )
)

FORBIDDEN_X_PREFIXES = (
    "target_",
    "label_",
    "quality_",
    "observation_quality",
    "exclusion_",
    "censor_",
    "purge_",
    "manifest_",
    "terminal_",
)


class M1DatasetBuildError(RuntimeError):
    """M1 Model-ready Dataset 생성을 중단해야 하는 오류다."""


def _wrap_contract_error(exc: ProcessedContractError) -> M1DatasetBuildError:
    """공통 Processed 계약 오류를 M1 Builder 오류로 통일한다."""
    return M1DatasetBuildError(str(exc))


def _require_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    source_name: str,
) -> None:
    """입력 DataFrame의 필수 컬럼 존재 여부를 검증한다."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise M1DatasetBuildError(f"{source_name}에 필수 컬럼이 없습니다: {missing}")


def _normalize_string(
    series: pd.Series,
    *,
    source_name: str,
    column: str,
) -> pd.Series:
    """식별자 문자열을 pandas string dtype으로 정규화하고 빈 값을 차단한다."""
    result = series.astype("string")
    invalid = result.isna() | result.str.strip().eq("")
    if invalid.any():
        raise M1DatasetBuildError(
            f"{source_name}.{column}에 null 또는 빈 문자열이 있습니다."
        )
    return result


def _normalize_date(
    series: pd.Series,
    *,
    source_name: str,
    allow_null: bool = False,
) -> pd.Series:
    """날짜 컬럼을 timezone-naive 날짜 단위 datetime64[us]로 정규화한다."""
    try:
        parsed = pd.to_datetime(series, errors="raise")
    except (TypeError, ValueError) as exc:
        raise M1DatasetBuildError(f"{source_name} 날짜 컬럼을 해석할 수 없습니다.") from exc

    if not allow_null and parsed.isna().any():
        raise M1DatasetBuildError(f"{source_name} 날짜 컬럼에 결측값이 있습니다.")
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_convert(None)
    return parsed.dt.normalize().astype("datetime64[us]")


def _normalize_integer(
    series: pd.Series,
    *,
    source_name: str,
    column: str,
    allow_null: bool = False,
) -> pd.Series:
    """정수 의미 컬럼을 손실 없이 nullable Int64로 정규화한다."""
    try:
        numeric = pd.to_numeric(series, errors="raise")
    except (TypeError, ValueError) as exc:
        raise M1DatasetBuildError(
            f"{source_name}.{column}을 숫자로 변환할 수 없습니다."
        ) from exc
    if not allow_null and numeric.isna().any():
        raise M1DatasetBuildError(f"{source_name}.{column}에 결측값이 있습니다.")
    non_null = numeric.dropna()
    if non_null.mod(1).ne(0).any():
        raise M1DatasetBuildError(f"{source_name}.{column}에 정수가 아닌 값이 있습니다.")
    return numeric.astype("Int64")


def _normalize_boolean(
    series: pd.Series,
    *,
    source_name: str,
    column: str,
    allow_null: bool = False,
) -> pd.Series:
    """Boolean 의미 컬럼을 nullable boolean dtype으로 정규화한다."""
    try:
        result = series.astype("boolean")
    except (TypeError, ValueError) as exc:
        raise M1DatasetBuildError(
            f"{source_name}.{column}을 boolean으로 변환할 수 없습니다."
        ) from exc
    if not allow_null and result.isna().any():
        raise M1DatasetBuildError(f"{source_name}.{column}에 결측값이 있습니다.")
    return result


def _normalize_games(games: pd.DataFrame) -> pd.DataFrame:
    """M1 Target과 Join에 필요한 Canonical games 경계를 검증·정규화한다."""
    if games.empty:
        raise M1DatasetBuildError("games 입력이 비어 있습니다.")
    _require_columns(games, REQUIRED_GAMES_COLUMNS, source_name="games")

    result = games.loc[:, REQUIRED_GAMES_COLUMNS].copy()
    for column in ("game_pk", "home_team", "away_team"):
        result[column] = _normalize_string(
            result[column],
            source_name="games",
            column=column,
        )
    result["game_date"] = _normalize_date(
        result["game_date"],
        source_name="games.game_date",
    )
    result["season"] = _normalize_integer(
        result["season"],
        source_name="games",
        column="season",
    )
    for column in ("final_home_score", "final_away_score"):
        result[column] = _normalize_integer(
            result[column],
            source_name="games",
            column=column,
            allow_null=True,
        )
        if result[column].dropna().lt(0).any():
            raise M1DatasetBuildError(f"games.{column}에 음수가 있습니다.")
    for column in ("home_win", "away_win", "is_tie"):
        result[column] = _normalize_boolean(
            result[column],
            source_name="games",
            column=column,
            allow_null=True,
        )

    duplicate_count = int(result.duplicated(subset=list(GAME_KEY), keep=False).sum())
    if duplicate_count:
        raise M1DatasetBuildError(
            f"games.game_pk 중복 Row가 {duplicate_count}건 있습니다."
        )
    if result["home_team"].eq(result["away_team"]).any():
        raise M1DatasetBuildError("games에 home_team과 away_team이 같은 Row가 있습니다.")

    return (
        result.sort_values(["season", "game_date", "game_pk"], kind="mergesort")
        .reset_index(drop=True)
    )


def _validate_team_feature_manifest(
    manifest: Mapping[str, object],
) -> tuple[tuple[str, ...], str]:
    """#23 Team Pregame Feature Manifest의 역할 계약을 검증한다."""
    if manifest.get("artifact") != "team_pregame_features":
        raise M1DatasetBuildError(
            "Team Feature Manifest artifact가 team_pregame_features가 아닙니다."
        )

    key_columns = tuple(str(value) for value in manifest.get("key_columns", []))
    if key_columns != TEAM_FEATURE_KEY:
        raise M1DatasetBuildError(
            "Team Feature Manifest key_columns가 (game_pk, team) 계약과 다릅니다: "
            f"{key_columns}"
        )

    audit_columns = tuple(str(value) for value in manifest.get("audit_columns", []))
    missing_audit = [
        column for column in REQUIRED_TEAM_FEATURE_AUDIT_COLUMNS if column not in audit_columns
    ]
    if missing_audit:
        raise M1DatasetBuildError(
            f"Team Feature Manifest audit_columns에 필수 컬럼이 없습니다: {missing_audit}"
        )

    x_columns = tuple(str(value) for value in manifest.get("x_columns", []))
    if not x_columns:
        raise M1DatasetBuildError("Team Feature Manifest x_columns가 비어 있습니다.")
    if len(x_columns) != len(set(x_columns)):
        raise M1DatasetBuildError("Team Feature Manifest x_columns에 중복이 있습니다.")
    if x_columns != TEAM_SOURCE_X_COLUMNS:
        missing = [column for column in TEAM_SOURCE_X_COLUMNS if column not in x_columns]
        unexpected = [column for column in x_columns if column not in TEAM_SOURCE_X_COLUMNS]
        raise M1DatasetBuildError(
            "Team Feature Manifest x_columns가 current-main #23/Feature Catalog 계약과 "
            f"다릅니다: missing={missing}, unexpected={unexpected}"
        )
    role_overlap = sorted(set(x_columns) & (set(key_columns) | set(audit_columns)))
    if role_overlap:
        raise M1DatasetBuildError(
            f"Team Feature Manifest에서 X와 key/audit 역할이 겹칩니다: {role_overlap}"
        )
    try:
        validate_x_allowlist(
            available_columns=x_columns,
            x_columns=x_columns,
            forbidden_columns=FORBIDDEN_X_COLUMNS,
            forbidden_prefixes=FORBIDDEN_X_PREFIXES,
        )
    except ProcessedContractError as exc:
        raise _wrap_contract_error(exc) from exc

    source_contract_version = str(
        manifest.get("source_contract_version", "")
    ).strip()
    if source_contract_version != FEATURE_CATALOG_VERSION:
        raise M1DatasetBuildError(
            "Team Feature Manifest source_contract_version이 current-main "
            f"Feature Catalog와 다릅니다: {source_contract_version!r}"
        )

    feature_version = str(manifest.get("feature_version", "")).strip()
    if feature_version != TEAM_FEATURE_VERSION:
        raise M1DatasetBuildError(
            "Team Feature Manifest feature_version이 current-main #23 계약과 다릅니다: "
            f"{feature_version!r}"
        )

    expected_manifest_contract = {
        "historical_cutoff": "source.game_date < prediction_date",
        "same_day_policy": "exclude_all_same_date_source_results",
        "season_reset": True,
        "rolling_windows": list(TEAM_ROLLING_WINDOWS),
    }
    for key, expected in expected_manifest_contract.items():
        if manifest.get(key) != expected:
            raise M1DatasetBuildError(
                f"Team Feature Manifest {key}={manifest.get(key)!r}가 "
                f"current-main #23 계약 {expected!r}와 다릅니다."
            )
    return x_columns, feature_version


def _normalize_team_features(
    team_features: pd.DataFrame,
    *,
    manifest: Mapping[str, object],
) -> tuple[pd.DataFrame, tuple[str, ...], str]:
    """#23 Team Feature를 Manifest 역할 계약에 맞춰 검증한다."""
    x_columns, feature_version = _validate_team_feature_manifest(manifest)
    required = TEAM_FEATURE_KEY + REQUIRED_TEAM_FEATURE_AUDIT_COLUMNS + x_columns
    _require_columns(team_features, required, source_name="team_pregame_features")

    if team_features.empty:
        raise M1DatasetBuildError("team_pregame_features 입력이 비어 있습니다.")
    result = team_features.loc[:, required].copy()

    output_metadata = manifest.get("output")
    if isinstance(output_metadata, Mapping) and "row_count" in output_metadata:
        try:
            manifest_row_count = int(output_metadata["row_count"])
        except (TypeError, ValueError) as exc:
            raise M1DatasetBuildError(
                "Team Feature Manifest output.row_count를 정수로 해석할 수 없습니다."
            ) from exc
        if manifest_row_count != len(result):
            raise M1DatasetBuildError(
                "Team Feature Manifest output.row_count와 실제 Row 수가 다릅니다: "
                f"manifest={manifest_row_count}, actual={len(result)}"
            )

    for column in ("game_pk", "team", "opponent", "feature_version"):
        result[column] = _normalize_string(
            result[column],
            source_name="team_pregame_features",
            column=column,
        )
    result["prediction_date"] = _normalize_date(
        result["prediction_date"],
        source_name="team_pregame_features.prediction_date",
    )
    result["max_source_game_date"] = _normalize_date(
        result["max_source_game_date"],
        source_name="team_pregame_features.max_source_game_date",
        allow_null=True,
    )
    for column in ("season", "history_game_count"):
        result[column] = _normalize_integer(
            result[column],
            source_name="team_pregame_features",
            column=column,
        )
    for column in ("is_home", "has_history"):
        result[column] = _normalize_boolean(
            result[column],
            source_name="team_pregame_features",
            column=column,
        )

    duplicate_count = int(
        result.duplicated(subset=list(TEAM_FEATURE_KEY), keep=False).sum()
    )
    if duplicate_count:
        raise M1DatasetBuildError(
            "team_pregame_features에 (game_pk, team) 중복 Row가 "
            f"{duplicate_count}건 있습니다."
        )
    if result["feature_version"].ne(feature_version).any():
        raise M1DatasetBuildError(
            "Team Feature Row의 feature_version이 Manifest와 일치하지 않습니다."
        )
    if result.loc[result["has_history"], "max_source_game_date"].ge(
        result.loc[result["has_history"], "prediction_date"]
    ).any():
        raise M1DatasetBuildError(
            "Team Feature에 max_source_game_date >= prediction_date인 Row가 있습니다."
        )

    return result, x_columns, feature_version


def _expected_team_context(games: pd.DataFrame) -> pd.DataFrame:
    """games에서 경기당 Home/Away 두 개의 기대 Team Feature Key/Context를 만든다."""
    home = games.loc[:, ["game_pk", "game_date", "season", "home_team", "away_team"]].copy()
    home = home.rename(columns={"home_team": "team", "away_team": "opponent"})
    home["is_home"] = pd.Series(True, index=home.index, dtype="boolean")

    away = games.loc[:, ["game_pk", "game_date", "season", "home_team", "away_team"]].copy()
    away = away.rename(columns={"away_team": "team", "home_team": "opponent"})
    away["is_home"] = pd.Series(False, index=away.index, dtype="boolean")

    expected = pd.concat([home, away], ignore_index=True)
    expected = expected.rename(columns={"game_date": "prediction_date"})
    return expected.loc[:, [
        "game_pk",
        "team",
        "opponent",
        "prediction_date",
        "season",
        "is_home",
    ]]


def _validate_team_feature_context(
    games: pd.DataFrame,
    team_features: pd.DataFrame,
) -> None:
    """Home/Away Team Feature가 정확히 1개씩이며 방향·상대·날짜가 일치하는지 검증한다."""
    expected = _expected_team_context(games)
    expected_keys = set(expected.loc[:, list(TEAM_FEATURE_KEY)].itertuples(index=False, name=None))
    actual_keys = set(
        team_features.loc[:, list(TEAM_FEATURE_KEY)].itertuples(index=False, name=None)
    )

    missing = sorted(expected_keys - actual_keys)
    orphan = sorted(actual_keys - expected_keys)
    if missing:
        raise M1DatasetBuildError(
            f"Home/Away Team Feature가 누락되었습니다. 예시: {missing[:10]}"
        )
    if orphan:
        raise M1DatasetBuildError(
            f"games에 없는 orphan Team Feature가 있습니다. 예시: {orphan[:10]}"
        )
    if len(team_features) != 2 * len(games):
        raise M1DatasetBuildError(
            "Team Feature Row 수가 경기당 정확히 2행이 아닙니다: "
            f"games={len(games)}, team_features={len(team_features)}"
        )

    actual = team_features.loc[:, [
        "game_pk",
        "team",
        "opponent",
        "prediction_date",
        "season",
        "is_home",
    ]]
    merged = expected.merge(
        actual,
        on=["game_pk", "team"],
        how="left",
        suffixes=("_expected", "_actual"),
        validate="one_to_one",
    )
    for column in ("opponent", "prediction_date", "season", "is_home"):
        expected_column = merged[f"{column}_expected"]
        actual_column = merged[f"{column}_actual"]
        mismatch = expected_column.ne(actual_column)
        if mismatch.any():
            sample = merged.loc[mismatch, ["game_pk", "team"]].head(10).to_dict("records")
            raise M1DatasetBuildError(
                f"Team Feature {column} Context가 games와 일치하지 않습니다. 예시: {sample}"
            )


def _prefix_team_side(
    team_features: pd.DataFrame,
    *,
    side: str,
    x_columns: Sequence[str],
) -> pd.DataFrame:
    """한 팀 방향의 Feature/Audit에 명확한 home_/away_ prefix를 붙인다."""
    if side not in {"home", "away"}:
        raise M1DatasetBuildError(f"지원하지 않는 team side입니다: {side}")
    is_home = side == "home"
    selected = team_features.loc[team_features["is_home"].eq(is_home)].copy()
    selected = selected.loc[:, [
        "game_pk",
        "team",
        "max_source_game_date",
        "history_game_count",
        "has_history",
        *x_columns,
    ]]
    rename = {
        "team": f"{side}_feature_team",
        "max_source_game_date": f"{side}_max_source_game_date",
        "history_game_count": f"{side}_history_game_count",
        "has_history": f"{side}_has_history",
    }
    rename.update({column: f"{side}_{column}" for column in x_columns})
    return selected.rename(columns=rename)


def _normalize_pa_audit(plate_appearances: pd.DataFrame) -> pd.DataFrame:
    """Terminal observation quality 확인에 필요한 최소 Canonical PA 컬럼만 검증한다."""
    if plate_appearances.empty:
        raise M1DatasetBuildError("plate_appearances audit 입력이 비어 있습니다.")
    _require_columns(
        plate_appearances,
        REQUIRED_PA_AUDIT_COLUMNS,
        source_name="plate_appearances",
    )
    result = plate_appearances.loc[:, REQUIRED_PA_AUDIT_COLUMNS].copy()
    result["game_pk"] = _normalize_string(
        result["game_pk"],
        source_name="plate_appearances",
        column="game_pk",
    )
    result["at_bat_number"] = _normalize_integer(
        result["at_bat_number"],
        source_name="plate_appearances",
        column="at_bat_number",
    )
    for column in ("post_home_score", "post_away_score"):
        result[column] = _normalize_integer(
            result[column],
            source_name="plate_appearances",
            column=column,
            allow_null=True,
        )
    result["pa_completed"] = _normalize_boolean(
        result["pa_completed"],
        source_name="plate_appearances",
        column="pa_completed",
        allow_null=True,
    )
    duplicate_count = int(
        result.duplicated(subset=["game_pk", "at_bat_number"], keep=False).sum()
    )
    if duplicate_count:
        raise M1DatasetBuildError(
            "plate_appearances에 (game_pk, at_bat_number) 중복 Row가 "
            f"{duplicate_count}건 있습니다."
        )
    return result


def _terminal_pa_audit(
    games: pd.DataFrame,
    plate_appearances: pd.DataFrame | None,
) -> pd.DataFrame:
    """마지막 관측 PA의 Score/완료 상태를 Target과 분리된 Quality Audit로 만든다."""
    base = games.loc[:, ["game_pk", "final_home_score", "final_away_score"]].copy()
    if plate_appearances is None:
        base["terminal_post_home_score"] = pd.Series(pd.NA, index=base.index, dtype="Int64")
        base["terminal_post_away_score"] = pd.Series(pd.NA, index=base.index, dtype="Int64")
        base["terminal_pa_completed"] = pd.Series(pd.NA, index=base.index, dtype="boolean")
        base["terminal_reconciled"] = pd.Series(pd.NA, index=base.index, dtype="boolean")
        return base

    pa = _normalize_pa_audit(plate_appearances)
    game_keys = set(games["game_pk"].tolist())
    pa_keys = set(pa["game_pk"].tolist())
    orphan = sorted(pa_keys - game_keys)
    missing = sorted(game_keys - pa_keys)
    if orphan:
        raise M1DatasetBuildError(
            f"games에 없는 orphan PA가 있습니다. 예시 game_pk: {orphan[:10]}"
        )
    if missing:
        raise M1DatasetBuildError(
            f"Terminal quality audit에 필요한 PA가 없는 경기가 있습니다: {missing[:10]}"
        )

    terminal = (
        pa.sort_values(["game_pk", "at_bat_number"], kind="mergesort")
        .groupby("game_pk", as_index=False, sort=False)
        .tail(1)
        .loc[:, [
            "game_pk",
            "post_home_score",
            "post_away_score",
            "pa_completed",
        ]]
        .rename(
            columns={
                "post_home_score": "terminal_post_home_score",
                "post_away_score": "terminal_post_away_score",
                "pa_completed": "terminal_pa_completed",
            }
        )
    )
    base = base.merge(terminal, on="game_pk", how="left", validate="one_to_one")
    scores_present = (
        base["final_home_score"].notna()
        & base["final_away_score"].notna()
        & base["terminal_post_home_score"].notna()
        & base["terminal_post_away_score"].notna()
    )
    reconciled = (
        base["final_home_score"].eq(base["terminal_post_home_score"])
        & base["final_away_score"].eq(base["terminal_post_away_score"])
    )
    base["terminal_reconciled"] = reconciled.where(scores_present, pd.NA).astype("boolean")
    return base


def _diagnose_rows(
    games: pd.DataFrame,
    terminal_audit: pd.DataFrame,
) -> pd.DataFrame:
    """Target/관측 품질의 primary reason과 상세 diagnostics를 상호 배타적으로 정리한다."""
    frame = games.loc[:, [
        "game_pk",
        "final_home_score",
        "final_away_score",
        "home_win",
        "away_win",
        "is_tie",
    ]].merge(
        terminal_audit.loc[:, [
            "game_pk",
            "terminal_pa_completed",
            "terminal_reconciled",
        ]],
        on="game_pk",
        how="left",
        validate="one_to_one",
    )

    quality: list[str] = []
    quality_reason: list[str] = []
    excluded: list[bool] = []
    exclusion_reason: list[object] = []
    diagnostics_json: list[str] = []

    for row in frame.itertuples(index=False):
        reasons: list[str] = []
        missing_score = pd.isna(row.final_home_score) or pd.isna(row.final_away_score)
        if missing_score:
            reasons.append("missing_terminal_score")

        flag_mismatch = False
        if not missing_score:
            home_runs = int(row.final_home_score)
            away_runs = int(row.final_away_score)
            expected_home_win = home_runs > away_runs
            expected_away_win = away_runs > home_runs
            expected_tie = home_runs == away_runs
            flags = (row.home_win, row.away_win, row.is_tie)
            if any(pd.isna(value) for value in flags):
                flag_mismatch = True
            else:
                flag_mismatch = (
                    bool(row.home_win) != expected_home_win
                    or bool(row.away_win) != expected_away_win
                    or bool(row.is_tie) != expected_tie
                )
        if flag_mismatch:
            reasons.append("game_result_flag_reconciliation")

        if row.terminal_reconciled is not pd.NA and not pd.isna(row.terminal_reconciled):
            if not bool(row.terminal_reconciled):
                reasons.append("terminal_pa_score_reconciliation")

        if missing_score:
            primary = "missing_terminal_score"
            quality.append("invalid_missing_score")
            quality_reason.append("required observed terminal score is missing")
            is_excluded = True
        elif "game_result_flag_reconciliation" in reasons or "terminal_pa_score_reconciliation" in reasons:
            primary = (
                "terminal_pa_score_reconciliation"
                if "terminal_pa_score_reconciliation" in reasons
                else "game_result_flag_reconciliation"
            )
            quality.append("invalid_reconciliation")
            quality_reason.append("canonical result/score reconciliation failed")
            is_excluded = True
        elif row.terminal_pa_completed is not pd.NA and not pd.isna(row.terminal_pa_completed) and not bool(row.terminal_pa_completed):
            primary = None
            reasons.append("terminal_pa_incomplete_warning")
            quality.append("warning_terminal_pa_incomplete")
            quality_reason.append(
                "terminal PA is incomplete, but that alone does not prove the game was incomplete"
            )
            is_excluded = False
        else:
            primary = None
            quality.append("verified_canonical")
            quality_reason.append(
                "canonical key/score contract accepted; official game status is not verified"
            )
            is_excluded = False

        excluded.append(is_excluded)
        exclusion_reason.append(primary if is_excluded else pd.NA)
        diagnostics_json.append(
            json.dumps(sorted(set(reasons)), ensure_ascii=False, separators=(",", ":"))
        )

    return pd.DataFrame(
        {
            "game_pk": frame["game_pk"].astype("string"),
            "observation_quality": pd.Series(quality, dtype="string"),
            "observation_quality_reason": pd.Series(quality_reason, dtype="string"),
            "is_excluded": pd.Series(excluded, dtype="boolean"),
            "exclusion_reason": pd.Series(exclusion_reason, dtype="string"),
            "diagnostic_reasons": pd.Series(diagnostics_json, dtype="string"),
        }
    )


def _target_columns(games: pd.DataFrame) -> pd.DataFrame:
    """Canonical observed terminal score에서 M1 승/무/패 및 득점 Target을 생성한다."""
    result = games.loc[:, ["game_pk", "final_home_score", "final_away_score"]].copy()
    result = result.rename(
        columns={
            "final_home_score": "home_runs",
            "final_away_score": "away_runs",
        }
    )

    home_result = pd.Series(pd.NA, index=result.index, dtype="string")
    usable = result["home_runs"].notna() & result["away_runs"].notna()
    home_result.loc[usable & result["home_runs"].lt(result["away_runs"])] = "loss"
    home_result.loc[usable & result["home_runs"].eq(result["away_runs"])] = "tie"
    home_result.loc[usable & result["home_runs"].gt(result["away_runs"])] = "win"
    result["home_result"] = home_result
    result["home_result_code"] = home_result.map(HOME_RESULT_CODES).astype("Int64")
    return result.loc[:, [
        "game_pk",
        "home_result",
        "home_result_code",
        "home_runs",
        "away_runs",
    ]]


def _validate_config(config: Mapping[str, object]) -> tuple[tuple[TemporalSplitRange, ...], dict[str, object]]:
    """current-main #22 계약과 설정 버전/Target/Split 정의가 일치하는지 검증한다."""
    if config.get("prediction_contract_version") != PREDICTION_CONTRACT_VERSION:
        raise M1DatasetBuildError(
            "prediction_contract_version이 current-main 계약과 다릅니다."
        )
    if config.get("feature_catalog_version") != FEATURE_CATALOG_VERSION:
        raise M1DatasetBuildError("feature_catalog_version이 current-main 계약과 다릅니다.")

    m1 = config.get("m1")
    if not isinstance(m1, dict):
        raise M1DatasetBuildError("config.m1 object가 없습니다.")
    expected_versions = {
        "dataset_version": DATASET_VERSION,
        "feature_version": M1_FEATURE_VERSION,
        "target_version": TARGET_VERSION,
        "split_version": SPLIT_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "target_semantics": TARGET_SEMANTICS,
        "timestamp_resolution": "date",
    }
    for key, expected in expected_versions.items():
        if m1.get(key) != expected:
            raise M1DatasetBuildError(
                f"config.m1.{key}={m1.get(key)!r}가 기대값 {expected!r}와 다릅니다."
            )

    classes = m1.get("home_result_classes")
    expected_classes = [
        {"class": "loss", "code": 0},
        {"class": "tie", "code": 1},
        {"class": "win", "code": 2},
    ]
    if classes != expected_classes:
        raise M1DatasetBuildError(
            "home_result class/code 순서는 loss=0, tie=1, win=2 계약과 같아야 합니다."
        )

    expected_difference_features = list(M1_DIFFERENCE_COLUMNS)
    if m1.get("difference_features") != expected_difference_features:
        raise M1DatasetBuildError(
            "M1 difference_features가 current-main Feature Catalog의 최소 Difference와 "
            f"다릅니다: {m1.get('difference_features')!r}"
        )

    raw_ranges = m1.get("split_ranges")
    if not isinstance(raw_ranges, list):
        raise M1DatasetBuildError("config.m1.split_ranges가 list가 아닙니다.")
    try:
        split_ranges = parse_split_ranges(raw_ranges)
    except ProcessedContractError as exc:
        raise _wrap_contract_error(exc) from exc

    expected_split_names = ["train", "validation", "test", "snapshot"]
    if [item.name for item in split_ranges] != expected_split_names:
        raise M1DatasetBuildError(
            "M1 Split 순서는 train/validation/test/snapshot이어야 합니다."
        )
    expected_bounds = [
        ("2023-01-01", "2024-01-01"),
        ("2024-01-01", "2025-01-01"),
        ("2025-01-01", "2026-01-01"),
        ("2026-01-01", "2027-01-01"),
    ]
    for item, (start, end) in zip(split_ranges, expected_bounds):
        if item.start != pd.Timestamp(start) or item.end_exclusive != pd.Timestamp(end):
            raise M1DatasetBuildError(
                f"{item.name} Split 경계가 current-main #22 계약과 다릅니다."
            )
    return split_ranges, m1


def _build_output_roles(
    x_columns: Sequence[str],
) -> dict[str, list[str]]:
    """Output Schema의 물리적 역할 분리를 직렬화한다."""
    return {
        "key_columns": list(KEY_COLUMNS),
        "audit_columns": list(AUDIT_COLUMNS),
        "x_columns": list(x_columns),
        "y_columns": list(Y_COLUMNS),
        "split_columns": list(SPLIT_COLUMNS),
        "quality_columns": list(QUALITY_COLUMNS),
        "control_columns": list(CONTROL_COLUMNS),
        "label_status_columns": list(LABEL_STATUS_COLUMNS),
    }


def build_m1_dataset(
    games: pd.DataFrame,
    team_features: pd.DataFrame,
    *,
    team_feature_manifest: Mapping[str, object],
    config: Mapping[str, object],
    plate_appearances: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, object]]:
    """Home/Away Team Pregame Feature와 M1 Target을 1 game_pk=1 row로 조립한다."""
    split_ranges, _ = _validate_config(config)
    normalized_games = _normalize_games(games)
    normalized_features, team_x_columns, feature_version = _normalize_team_features(
        team_features,
        manifest=team_feature_manifest,
    )
    _validate_team_feature_context(normalized_games, normalized_features)

    home_features = _prefix_team_side(
        normalized_features,
        side="home",
        x_columns=team_x_columns,
    )
    away_features = _prefix_team_side(
        normalized_features,
        side="away",
        x_columns=team_x_columns,
    )

    base = normalized_games.loc[:, [
        "game_pk",
        "game_date",
        "season",
        "home_team",
        "away_team",
    ]].copy()
    before_rows = len(base)
    base = base.merge(home_features, on="game_pk", how="left", validate="one_to_one")
    base = base.merge(away_features, on="game_pk", how="left", validate="one_to_one")
    if len(base) != before_rows:
        raise M1DatasetBuildError("Home/Away Feature Join으로 Row가 증식했습니다.")
    if base["home_feature_team"].ne(base["home_team"]).any():
        raise M1DatasetBuildError("Home Feature Team 방향이 home_team과 일치하지 않습니다.")
    if base["away_feature_team"].ne(base["away_team"]).any():
        raise M1DatasetBuildError("Away Feature Team 방향이 away_team과 일치하지 않습니다.")
    base = base.drop(columns=["home_feature_team", "away_feature_team"])

    targets = _target_columns(normalized_games)
    terminal_audit = _terminal_pa_audit(normalized_games, plate_appearances)
    diagnostics = _diagnose_rows(normalized_games, terminal_audit)
    base = base.merge(targets, on="game_pk", how="left", validate="one_to_one")
    base = base.merge(diagnostics, on="game_pk", how="left", validate="one_to_one")

    # 실제 score/key reconciliation 오류는 학습 Target으로 소비되지 않도록 null 처리한다.
    excluded_mask = base["is_excluded"].fillna(True)
    base.loc[excluded_mask, "home_result"] = pd.NA
    base.loc[excluded_mask, "home_result_code"] = pd.NA
    base.loc[excluded_mask, "home_runs"] = pd.NA
    base.loc[excluded_mask, "away_runs"] = pd.NA
    base["home_result"] = base["home_result"].astype("string")
    base["home_result_code"] = base["home_result_code"].astype("Int64")
    base["home_runs"] = base["home_runs"].astype("Int64")
    base["away_runs"] = base["away_runs"].astype("Int64")

    base["prediction_date"] = base["game_date"].astype("datetime64[us]")
    base["label_available_at"] = (
        base["game_date"] + pd.Timedelta(days=1)
    ).astype("datetime64[us]")
    base["availability_basis"] = pd.Series(
        AVAILABILITY_BASIS, index=base.index, dtype="string"
    )
    base["target_semantics"] = pd.Series(
        TARGET_SEMANTICS, index=base.index, dtype="string"
    )
    base["contract_version"] = pd.Series(
        PREDICTION_CONTRACT_VERSION, index=base.index, dtype="string"
    )
    base["feature_catalog_version"] = pd.Series(
        FEATURE_CATALOG_VERSION, index=base.index, dtype="string"
    )
    base["team_feature_version"] = pd.Series(
        feature_version, index=base.index, dtype="string"
    )
    base["target_version"] = pd.Series(TARGET_VERSION, index=base.index, dtype="string")
    base["split_version"] = pd.Series(SPLIT_VERSION, index=base.index, dtype="string")
    base["output_schema_version"] = pd.Series(
        OUTPUT_SCHEMA_VERSION, index=base.index, dtype="string"
    )
    try:
        base["split"] = assign_temporal_split_series(base["prediction_date"], split_ranges)
    except ProcessedContractError as exc:
        raise _wrap_contract_error(exc) from exc

    base["official_status_verified"] = pd.Series(False, index=base.index, dtype="boolean")
    base["target_eligible"] = (
        ~base["is_excluded"].fillna(True)
        & base["home_result"].notna()
        & base["home_runs"].notna()
        & base["away_runs"].notna()
    ).astype("boolean")
    base["is_censored"] = pd.Series(False, index=base.index, dtype="boolean")
    base["censor_reason"] = pd.Series(pd.NA, index=base.index, dtype="string")
    base["coverage_status"] = pd.Series(
        "not_applicable_m1", index=base.index, dtype="string"
    )
    base["is_purged"] = pd.Series(False, index=base.index, dtype="boolean")
    base["purge_reason"] = pd.Series(pd.NA, index=base.index, dtype="string")
    base["supervised_usable"] = (
        base["target_eligible"]
        & ~base["is_censored"]
        & ~base["is_purged"]
    ).astype("boolean")
    base["label_status"] = pd.Series(
        ["usable" if bool(value) else "excluded" for value in base["supervised_usable"]],
        index=base.index,
        dtype="string",
    )

    for difference_column, source_column in M1_DIFFERENCE_SOURCES:
        base[difference_column] = (
            base[f"home_{source_column}"] - base[f"away_{source_column}"]
        ).astype("Float64")

    m1_x_columns = tuple(
        [f"home_{column}" for column in team_x_columns]
        + ["home_has_history"]
        + [f"away_{column}" for column in team_x_columns]
        + ["away_has_history"]
        + list(M1_DIFFERENCE_COLUMNS)
    )
    roles = _build_output_roles(m1_x_columns)

    output_columns = (
        KEY_COLUMNS
        + AUDIT_COLUMNS
        + m1_x_columns
        + Y_COLUMNS
        + SPLIT_COLUMNS
        + QUALITY_COLUMNS
        + CONTROL_COLUMNS
        + LABEL_STATUS_COLUMNS
    )
    missing_output = [column for column in output_columns if column not in base.columns]
    if missing_output:
        raise M1DatasetBuildError(f"내부 오류: Output 컬럼이 누락되었습니다: {missing_output}")

    result = base.loc[:, output_columns].copy()
    result = (
        result.sort_values(["season", "game_date", "game_pk"], kind="mergesort")
        .reset_index(drop=True)
    )

    # Join 증식/방향 오류와 역할 중복을 마지막 경계에서 다시 차단한다.
    if len(result) != len(normalized_games):
        raise M1DatasetBuildError(
            f"M1 Output Row 수가 games와 다릅니다: {len(result)} != {len(normalized_games)}"
        )
    if result["game_pk"].duplicated().any():
        raise M1DatasetBuildError("M1 Output에 duplicate game_pk가 있습니다.")

    all_role_columns: list[str] = []
    for columns in roles.values():
        all_role_columns.extend(columns)
    if len(all_role_columns) != len(set(all_role_columns)):
        duplicates = [
            column for column, count in Counter(all_role_columns).items() if count > 1
        ]
        raise M1DatasetBuildError(f"Output Schema 역할이 중복되었습니다: {duplicates}")

    try:
        validate_x_allowlist(
            available_columns=result.columns,
            x_columns=m1_x_columns,
            forbidden_columns=FORBIDDEN_X_COLUMNS,
            forbidden_prefixes=FORBIDDEN_X_PREFIXES,
        )
    except ProcessedContractError as exc:
        raise _wrap_contract_error(exc) from exc

    summary = build_quality_summary(result)
    return result, roles, summary


def build_quality_summary(dataset: pd.DataFrame) -> dict[str, object]:
    """상호 배타적 Row 상태와 Quality/Target/Split 수량 대사를 생성한다."""
    total = int(len(dataset))
    supervised = int(dataset["supervised_usable"].fillna(False).sum())
    excluded = int(dataset["is_excluded"].fillna(False).sum())
    target_pending = total - supervised - excluded
    if target_pending < 0:
        raise M1DatasetBuildError("Row 상태 수량 대사가 음수가 되어 계약을 위반했습니다.")

    exclusion_counts = {
        str(key): int(value)
        for key, value in dataset.loc[dataset["is_excluded"], "exclusion_reason"]
        .value_counts(dropna=False)
        .sort_index()
        .items()
    }
    quality_counts = {
        str(key): int(value)
        for key, value in dataset["observation_quality"].value_counts(dropna=False).sort_index().items()
    }
    split_counts = {
        str(key): int(value)
        for key, value in dataset["split"].value_counts(dropna=False).sort_index().items()
    }
    class_counts = {
        str(key): int(value)
        for key, value in dataset["home_result"].value_counts(dropna=False).sort_index().items()
    }
    return {
        "input_games": total,
        "supervised_usable": supervised,
        "target_pending_or_held": target_pending,
        "excluded": excluded,
        "reconciled_total": supervised + target_pending + excluded,
        "split_counts": split_counts,
        "home_result_counts": class_counts,
        "observation_quality_counts": quality_counts,
        "primary_exclusion_reason_counts": exclusion_counts,
        "target_null_counts": {
            column: int(dataset[column].isna().sum()) for column in Y_COLUMNS
        },
    }


def build_schema_payload(
    dataset: pd.DataFrame,
    roles: Mapping[str, Sequence[str]],
) -> dict[str, object]:
    """M1 Output의 역할·dtype·Target Class 계약을 Schema Artifact로 만든다."""
    return {
        "artifact": "m1_game_dataset_schema",
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "dataset_version": DATASET_VERSION,
        "feature_version": M1_FEATURE_VERSION,
        "grain": list(GAME_KEY),
        "roles": {key: list(value) for key, value in roles.items()},
        "columns": schema_manifest(dataset),
        "home_result_classes": [
            {"class": name, "code": HOME_RESULT_CODES[name]}
            for name in HOME_RESULT_CLASSES
        ],
        "target_semantics": TARGET_SEMANTICS,
        "model_ready_semantics": (
            "X/y/split/schema가 고정된 Processed Dataset이며 모델 학습·성능 검증 완료를 의미하지 않음"
        ),
    }


def build_quality_report(
    dataset: pd.DataFrame,
    summary: Mapping[str, object],
) -> dict[str, object]:
    """Primary reason 합계와 행별 diagnostics를 분리한 Quality Report를 만든다."""
    excluded_rows = dataset.loc[
        dataset["is_excluded"].fillna(False),
        ["game_pk", "exclusion_reason", "diagnostic_reasons"],
    ]
    warning_rows = dataset.loc[
        dataset["observation_quality"].eq("warning_terminal_pa_incomplete"),
        ["game_pk", "observation_quality", "observation_quality_reason", "diagnostic_reasons"],
    ]
    return {
        "artifact": "m1_game_dataset_quality_report",
        "dataset_version": DATASET_VERSION,
        "target_semantics": TARGET_SEMANTICS,
        "official_status_verified": False,
        "summary": dict(summary),
        "excluded_rows": excluded_rows.to_dict("records"),
        "warning_rows": warning_rows.to_dict("records"),
    }


def _coerce_round_trip_dtypes(
    frame: pd.DataFrame,
    reference: pd.DataFrame,
) -> pd.DataFrame:
    """Parquet round-trip 결과를 생성 직전 nullable dtype 계약으로 복원한다."""
    if list(frame.columns) != list(reference.columns):
        raise M1DatasetBuildError("Parquet round-trip 후 Column 순서가 변경되었습니다.")
    if len(frame) != len(reference):
        raise M1DatasetBuildError("Parquet round-trip 후 Row 수가 변경되었습니다.")

    result = frame.copy()
    for column in reference.columns:
        expected_dtype = str(reference[column].dtype)
        try:
            if expected_dtype.startswith("datetime64"):
                result[column] = pd.to_datetime(result[column], errors="raise").astype("datetime64[us]")
            else:
                result[column] = result[column].astype(expected_dtype)
        except (TypeError, ValueError) as exc:
            raise M1DatasetBuildError(
                f"Parquet round-trip dtype 복원에 실패했습니다: {column} -> {expected_dtype}"
            ) from exc
    return result


def build_deterministic_identity(
    *,
    input_hashes: Mapping[str, str],
    team_feature_content_fingerprint: object,
    dataset_content_fingerprint: str,
    schema_payload: Mapping[str, object],
    quality_payload: Mapping[str, object],
) -> dict[str, object]:
    """실행 시각과 분리된 M1 Artifact Set의 결정적 Content Identity를 만든다."""
    identity: dict[str, object] = {
        "dataset_version": DATASET_VERSION,
        "prediction_contract_version": PREDICTION_CONTRACT_VERSION,
        "feature_catalog_version": FEATURE_CATALOG_VERSION,
        "feature_version": M1_FEATURE_VERSION,
        "target_version": TARGET_VERSION,
        "split_version": SPLIT_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "input_sha256": {name: input_hashes[name] for name in sorted(input_hashes)},
        "team_feature_content_fingerprint": team_feature_content_fingerprint,
        "dataset_content_fingerprint": dataset_content_fingerprint,
        "schema_fingerprint": stable_json_fingerprint(schema_payload),
        "quality_report_fingerprint": stable_json_fingerprint(quality_payload),
    }
    identity["fingerprint"] = stable_json_fingerprint(identity)
    return identity


def resolve_code_revision(project_root: Path) -> str:
    """Read-only git rev-parse로 Code Revision을 읽고 불가하면 unknown을 반환한다."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    revision = completed.stdout.strip()
    return revision or "unknown"


def build_m1_dataset_files(
    *,
    config_path: Path,
    games_path: Path,
    team_features_path: Path,
    team_feature_manifest_path: Path,
    plate_appearances_path: Path | None,
    output_path: Path,
    manifest_path: Path,
    schema_path: Path,
    quality_report_path: Path,
    project_root: Path,
    code_revision: str | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """M1 Dataset Artifact Set을 생성·round-trip 검증하고 완료 Manifest를 마지막에 기록한다."""
    input_paths: dict[str, Path] = {
        "config": config_path,
        "games": games_path,
        "team_features": team_features_path,
        "team_feature_manifest": team_feature_manifest_path,
    }
    if plate_appearances_path is not None:
        input_paths["plate_appearances"] = plate_appearances_path

    try:
        validate_output_paths(
            input_paths=list(input_paths.values()),
            output_paths=[output_path, manifest_path, schema_path, quality_report_path],
            project_root=project_root,
        )
    except ProcessedContractError as exc:
        raise _wrap_contract_error(exc) from exc

    # 이전 성공 Run의 Manifest를 먼저 지워 새 실행 실패 시 stale 완료 상태가 남지 않게 한다.
    try:
        invalidate_completion_marker(manifest_path)
    except ProcessedContractError as exc:
        raise _wrap_contract_error(exc) from exc

    before_hashes = calculate_input_hashes(input_paths)
    config = read_json(config_path, label="processed_dataset config")
    team_manifest = read_json(team_feature_manifest_path, label="team feature manifest")
    games = read_parquet(games_path, label="games")
    team_features = read_parquet(team_features_path, label="team_pregame_features")
    plate_appearances = (
        read_parquet(plate_appearances_path, label="plate_appearances")
        if plate_appearances_path is not None
        else None
    )

    dataset, roles, summary = build_m1_dataset(
        games,
        team_features,
        team_feature_manifest=team_manifest,
        config=config,
        plate_appearances=plate_appearances,
    )
    expected_fingerprint = content_fingerprint(dataset, sort_columns=["game_pk"])

    # Manifest는 Artifact Set이 검증된 뒤 마지막에 기록한다.
    write_parquet_atomic(dataset, output_path)
    round_trip_raw = read_parquet(output_path, label="m1 game_dataset")
    round_trip = _coerce_round_trip_dtypes(round_trip_raw, dataset)
    if round_trip["game_pk"].duplicated().any():
        raise M1DatasetBuildError("Parquet round-trip 후 duplicate game_pk가 있습니다.")
    actual_fingerprint = content_fingerprint(round_trip, sort_columns=["game_pk"])
    if actual_fingerprint != expected_fingerprint:
        raise M1DatasetBuildError(
            "Parquet round-trip 후 Dataset content fingerprint가 변경되었습니다."
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

    schema_payload = build_schema_payload(round_trip, roles)
    quality_payload = build_quality_report(round_trip, summary)
    write_json_atomic(schema_payload, schema_path)
    write_json_atomic(quality_payload, quality_report_path)

    after_hashes = assert_input_hashes_unchanged(before_hashes, input_paths)
    resolved_revision = code_revision or resolve_code_revision(project_root)

    team_output = team_manifest.get("output")
    team_content_fingerprint = None
    if isinstance(team_output, dict):
        team_content_fingerprint = team_output.get("content_fingerprint")

    deterministic_identity = build_deterministic_identity(
        input_hashes=after_hashes,
        team_feature_content_fingerprint=team_content_fingerprint,
        dataset_content_fingerprint=actual_fingerprint,
        schema_payload=schema_payload,
        quality_payload=quality_payload,
    )

    created_at = datetime.now(timezone.utc)
    created_at_text = created_at.isoformat()
    run_id = (
        f"m1-{created_at.strftime('%Y%m%dT%H%M%S%fZ')}-"
        f"{str(deterministic_identity['fingerprint'])[:12]}"
    )
    prediction_min = round_trip["prediction_date"].min()
    prediction_max = round_trip["prediction_date"].max()
    input_paths_payload = {name: str(path) for name, path in input_paths.items()}
    input_revision = {name: None for name in input_paths}
    input_revision["team_features"] = team_manifest.get("feature_version")
    input_revision["team_feature_manifest"] = team_manifest.get("feature_version")
    input_revision_basis = {
        name: "independent artifact revision unavailable; sha256 is authoritative"
        for name in input_paths
    }
    input_revision_basis["team_features"] = "team feature manifest.feature_version"
    input_revision_basis["team_feature_manifest"] = "manifest.feature_version"
    input_fingerprints: dict[str, object] = {
        name: {"sha256": after_hashes[name]}
        for name in sorted(input_paths)
    }
    if team_content_fingerprint is not None:
        team_entry = input_fingerprints["team_features"]
        if isinstance(team_entry, dict):
            team_entry["content_fingerprint"] = team_content_fingerprint

    manifest = {
        "artifact": "m1_game_dataset",
        "dataset_name": "m1_game_dataset",
        "run_id": run_id,
        "run_status": "complete",
        "created_at": created_at_text,
        "created_at_utc": created_at_text,
        "code_revision": resolved_revision,
        "dataset_version": DATASET_VERSION,
        "contract_version": PREDICTION_CONTRACT_VERSION,
        "prediction_contract_version": PREDICTION_CONTRACT_VERSION,
        "feature_catalog_version": FEATURE_CATALOG_VERSION,
        "feature_version": M1_FEATURE_VERSION,
        "team_feature_version": team_manifest.get("feature_version"),
        "target_version": TARGET_VERSION,
        "target_semantics": TARGET_SEMANTICS,
        "split_version": SPLIT_VERSION,
        "timestamp_resolution": config["m1"]["timestamp_resolution"],
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "grain": list(GAME_KEY),
        "config": config,
        "roles": roles,
        "input_paths": input_paths_payload,
        "input_revision": input_revision,
        "input_revision_basis": input_revision_basis,
        "input_sha256_or_content_fingerprint": input_fingerprints,
        "inputs": {
            name: {
                "path": str(path),
                "revision": input_revision[name],
                "revision_basis": input_revision_basis[name],
                "sha256": after_hashes[name],
            }
            for name, path in input_paths.items()
        },
        "prediction_range": {
            "min": prediction_min.isoformat() if not pd.isna(prediction_min) else None,
            "max": prediction_max.isoformat() if not pd.isna(prediction_max) else None,
        },
        "source_coverage": {
            "games_row_count": int(len(games)),
            "team_feature_row_count": int(len(team_features)),
            "plate_appearance_row_count": (
                int(len(plate_appearances)) if plate_appearances is not None else None
            ),
            "split_counts": summary["split_counts"],
        },
        "row_count": int(len(round_trip)),
        "usable_row_count": int(summary["supervised_usable"]),
        "excluded_row_count": int(summary["excluded"]),
        "censored_row_count": int(round_trip["is_censored"].fillna(False).sum()),
        "purged_row_count": int(round_trip["is_purged"].fillna(False).sum()),
        "null_summary": {
            column: int(round_trip[column].isna().sum()) for column in round_trip.columns
        },
        "quality_summary": summary,
        "output_content_fingerprint": actual_fingerprint,
        "output": {
            "dataset_path": str(output_path),
            "schema_path": str(schema_path),
            "quality_report_path": str(quality_report_path),
            "row_count": int(len(round_trip)),
            "content_fingerprint": actual_fingerprint,
            "schema": schema_manifest(round_trip),
        },
        "summary": summary,
        "deterministic_identity": deterministic_identity,
    }
    write_json_atomic(manifest, manifest_path)
    return round_trip, manifest
