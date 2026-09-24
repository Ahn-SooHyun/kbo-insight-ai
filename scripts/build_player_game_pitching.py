from __future__ import annotations

import argparse
import hashlib
import logging
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


LOGGER = logging.getLogger(__name__)

DEFAULT_SEASONS = (2023, 2024, 2025, 2026)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "hf_kbo_pbp"
DEFAULT_PA_PATH = (
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
    / "player_game_pitching.parquet"
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

PLAYER_GAME_KEY = (
    "game_pk",
    "pitcher",
)

PITCH_TYPE_VALUES = frozenset({"B", "S", "X"})

EVENT_VALUES = frozenset(
    {
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
    }
)

# 여러 투수가 한 PA에 등장한 경우 PA 종료 Event 자체로 안전하게 설명할 수 있는
# 최소 Out 수를 정의한다. PA 전체 Out Delta가 이 값과 정확히 일치하면 마지막
# Raw Pitcher(= Canonical PA Pitcher)에게 귀속할 수 있다. 더 큰 Delta이거나 Event가
# 이 표에 없으면 투수 교체 전/후 어느 시점의 Out인지 공개 Raw만으로 복원할 수
# 없으므로 해당 PA에 참여한 Player Game의 outs_recorded를 nullable <NA>로 남긴다.
# 이 규칙은 공식 실점 책임 또는 승계 주자 책임을 복원하는 규칙이 아니다.
MULTI_PITCHER_TERMINAL_OUTS: dict[str, int] = {
    "strikeout": 1,
    "field_out": 1,
    "sac_bunt": 1,
    "sac_fly": 1,
    "fielders_choice": 1,
    "double_play": 2,
    "triple_play": 3,
}

REQUIRED_RAW_COLUMNS = (
    "game_pk",
    "game_date",
    "home_team",
    "away_team",
    "inning_topbot",
    "at_bat_number",
    "pitch_number",
    "pitcher",
    "pitcher_name",
    "type",
    "release_speed_kmh",
)

REQUIRED_PA_COLUMNS = (
    "game_pk",
    "game_date",
    "season",
    "at_bat_number",
    "batting_team",
    "fielding_team",
    "is_home_batting",
    "pitcher",
    "pitcher_name",
    "event",
    "outs_before",
    "post_outs",
    "pitch_rows",
    "pitch_count",
)

OUTPUT_COLUMNS = (
    "game_pk",
    "game_date",
    "season",
    "pitcher",
    "pitcher_name",
    "team",
    "opponent",
    "is_home",
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
    "avg_release_speed_kmh",
)

STRING_OUTPUT_COLUMNS = (
    "game_pk",
    "pitcher",
    "pitcher_name",
    "team",
    "opponent",
)

INTEGER_OUTPUT_COLUMNS = (
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

BOOLEAN_OUTPUT_COLUMNS = (
    "is_home",
)

FLOAT_OUTPUT_COLUMNS = (
    "avg_release_speed_kmh",
)

COUNT_EVENT_MAPPING = {
    "single": "single_allowed",
    "double": "double_allowed",
    "triple": "triple_allowed",
    "home_run": "hr_allowed",
    "walk": "bb_allowed",
    "hit_by_pitch": "hbp_allowed",
    "strikeout": "so",
    "sac_fly": "sf",
    "sac_bunt": "sh",
}

PA_COUNT_COLUMNS = (
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
    "outs_recorded",
)

PROHIBITED_OUTPUT_COLUMNS = (
    "earned_runs",
    "era",
    "runs_allowed",
)


class PlayerGamePitchingBuildError(RuntimeError):
    """Player Game Pitching 생성을 중단해야 하는 검증 오류를 나타낸다."""


def parse_args() -> argparse.Namespace:
    """Player Game Pitching 생성 스크립트의 CLI 인자를 파싱한다."""
    parser = argparse.ArgumentParser(
        description=(
            "KBO PBP Raw 및 Canonical Plate Appearance에서 "
            "Player Game Pitching 파생 테이블을 생성합니다."
        )
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help=f"시즌별 Raw Parquet 디렉터리입니다. 기본값: {DEFAULT_RAW_DIR}",
    )
    parser.add_argument(
        "--pa-path",
        type=Path,
        default=DEFAULT_PA_PATH,
        help=f"Canonical Plate Appearance Parquet 경로입니다. 기본값: {DEFAULT_PA_PATH}",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Player Game Pitching Parquet 출력 경로입니다. 기본값: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEASONS),
        help="처리할 시즌입니다. 기본값: 2023 2024 2025 2026",
    )
    return parser.parse_args()


def configure_logging() -> None:
    """독립 실행 시 사용할 기본 logging 형식을 설정한다."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def normalize_seasons(seasons: Sequence[int]) -> tuple[int, ...]:
    """지원 시즌을 중복 제거 후 오름차순으로 정규화한다."""
    normalized = tuple(sorted(set(int(season) for season in seasons)))
    unsupported = [season for season in normalized if season not in DEFAULT_SEASONS]
    if unsupported:
        raise PlayerGamePitchingBuildError(
            f"지원하지 않는 시즌이 포함되어 있습니다: {unsupported}"
        )
    if not normalized:
        raise PlayerGamePitchingBuildError("처리할 시즌이 하나 이상 필요합니다.")
    return normalized


def calculate_sha256(path: Path) -> str:
    """파일 변경 여부 확인을 위해 SHA256을 계산한다."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_columns(
    frame: pd.DataFrame,
    required: Iterable[str],
    *,
    source_name: str,
) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise PlayerGamePitchingBuildError(
            f"{source_name}에 필요한 컬럼이 없습니다: {missing}"
        )


def _coerce_integer_column(
    series: pd.Series,
    *,
    column_name: str,
    allow_null: bool = False,
) -> pd.Series:
    """정수 컬럼을 nullable Int64로 변환하고 비정수 값을 거부한다."""
    numeric = pd.to_numeric(series, errors="coerce")
    invalid = series.notna() & numeric.isna()
    if invalid.any():
        examples = series.loc[invalid].head(5).tolist()
        raise PlayerGamePitchingBuildError(
            f"{column_name}에 숫자로 변환할 수 없는 값이 있습니다: {examples}"
        )

    non_integer = numeric.notna() & numeric.mod(1).ne(0)
    if non_integer.any():
        examples = numeric.loc[non_integer].head(5).tolist()
        raise PlayerGamePitchingBuildError(
            f"{column_name}에 정수가 아닌 값이 있습니다: {examples}"
        )

    result = numeric.astype("Int64")
    if not allow_null and result.isna().any():
        raise PlayerGamePitchingBuildError(
            f"{column_name}에는 null을 허용하지 않습니다."
        )
    return result


def _coerce_nullable_float_column(
    series: pd.Series,
    *,
    column_name: str,
) -> pd.Series:
    """실수 컬럼을 nullable Float64로 변환한다."""
    numeric = pd.to_numeric(series, errors="coerce")
    invalid = series.notna() & numeric.isna()
    if invalid.any():
        examples = series.loc[invalid].head(5).tolist()
        raise PlayerGamePitchingBuildError(
            f"{column_name}에 숫자로 변환할 수 없는 값이 있습니다: {examples}"
        )
    return numeric.astype("Float64")


def _normalize_game_date(series: pd.Series) -> pd.Series:
    """game_date를 timezone 없는 datetime64[us]로 정규화한다."""
    try:
        normalized = pd.to_datetime(series, errors="raise")
    except (TypeError, ValueError) as exc:
        raise PlayerGamePitchingBuildError(
            "game_date를 datetime으로 변환할 수 없습니다."
        ) from exc

    if getattr(normalized.dt, "tz", None) is not None:
        normalized = normalized.dt.tz_localize(None)
    return normalized.astype("datetime64[us]")


def _normalize_string_column(series: pd.Series) -> pd.Series:
    return series.astype("string")


def _ensure_non_empty_string(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    source_name: str,
) -> None:
    for column in columns:
        values = frame[column].astype("string")
        invalid = values.isna() | values.str.strip().eq("")
        if invalid.any():
            examples = frame.loc[invalid, column].head(5).tolist()
            raise PlayerGamePitchingBuildError(
                f"{source_name}.{column}에 null 또는 빈 문자열이 있습니다: {examples}"
            )


def _path_is_within(path: Path, parent: Path) -> bool:
    """path가 parent 자신 또는 하위 경로인지 확인한다."""
    path_resolved = path.resolve(strict=False)
    parent_resolved = parent.resolve(strict=False)
    return path_resolved == parent_resolved or parent_resolved in path_resolved.parents


def ensure_output_path_safe(
    *,
    raw_dir: Path,
    pa_path: Path,
    output_path: Path,
    seasons: Sequence[int],
) -> None:
    """Derived Output이 Raw 또는 입력 파일을 덮어쓰지 않도록 검증한다."""
    output_resolved = output_path.resolve(strict=False)
    pa_resolved = pa_path.resolve(strict=False)

    if _path_is_within(output_path, PROJECT_ROOT / "data" / "raw"):
        raise PlayerGamePitchingBuildError(
            "Player Game Pitching Output을 data/raw/ 내부에 생성할 수 없습니다."
        )
    if _path_is_within(output_path, raw_dir):
        raise PlayerGamePitchingBuildError(
            "Player Game Pitching Output을 Raw 입력 디렉터리 내부에 생성할 수 없습니다."
        )
    if output_resolved == pa_resolved:
        raise PlayerGamePitchingBuildError(
            "Player Game Pitching Output이 plate_appearances.parquet을 덮어쓸 수 없습니다."
        )

    for season in seasons:
        raw_path = (raw_dir / f"{season}.parquet").resolve(strict=False)
        if output_resolved == raw_path:
            raise PlayerGamePitchingBuildError(
                f"Player Game Pitching Output이 {season} Raw Parquet을 덮어쓸 수 없습니다."
            )


def _validate_unique_key(
    frame: pd.DataFrame,
    key_columns: Sequence[str],
    *,
    source_name: str,
) -> None:
    duplicates = frame.duplicated(list(key_columns), keep=False)
    if duplicates.any():
        examples = (
            frame.loc[duplicates, list(key_columns)]
            .head(10)
            .to_dict("records")
        )
        raise PlayerGamePitchingBuildError(
            f"{source_name}에 중복 Key가 있습니다: {examples}"
        )


def _dataframe_key_set(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> set[tuple[object, ...]]:
    return set(frame.loc[:, list(columns)].itertuples(index=False, name=None))


def prepare_raw_pitching_source(
    raw: pd.DataFrame,
    *,
    season: int,
) -> pd.DataFrame:
    """한 시즌 Raw Pitch-level Data를 투구 집계용 형태로 검증·정규화한다."""
    _require_columns(raw, REQUIRED_RAW_COLUMNS, source_name="Raw")
    working = raw.copy()

    for column in (
        "game_pk",
        "home_team",
        "away_team",
        "inning_topbot",
        "pitcher",
        "pitcher_name",
        "type",
    ):
        working[column] = _normalize_string_column(working[column])

    working["game_date"] = _normalize_game_date(working["game_date"])
    working["at_bat_number"] = _coerce_integer_column(
        working["at_bat_number"],
        column_name="Raw.at_bat_number",
    )
    working["pitch_number"] = _coerce_integer_column(
        working["pitch_number"],
        column_name="Raw.pitch_number",
    )
    working["release_speed_kmh"] = _coerce_nullable_float_column(
        working["release_speed_kmh"],
        column_name="Raw.release_speed_kmh",
    )

    _ensure_non_empty_string(
        working,
        ("game_pk", "home_team", "away_team", "inning_topbot", "pitcher"),
        source_name="Raw",
    )

    if working["pitch_number"].lt(0).any():
        raise PlayerGamePitchingBuildError("Raw.pitch_number에는 음수를 허용하지 않습니다.")

    _validate_unique_key(working, PITCH_KEY, source_name="Raw Pitch")

    invalid_topbot = ~working["inning_topbot"].isin(["top", "bot"])
    if invalid_topbot.any():
        values = working.loc[invalid_topbot, "inning_topbot"].drop_duplicates().tolist()
        raise PlayerGamePitchingBuildError(
            f"Raw.inning_topbot에 지원하지 않는 값이 있습니다: {values}"
        )

    actual_pitch = working["pitch_number"].gt(0).fillna(False)
    invalid_pitch_type = actual_pitch & ~working["type"].isin(PITCH_TYPE_VALUES)
    if invalid_pitch_type.any():
        examples = (
            working.loc[
                invalid_pitch_type,
                ["game_pk", "at_bat_number", "pitch_number", "type"],
            ]
            .head(10)
            .to_dict("records")
        )
        raise PlayerGamePitchingBuildError(
            "실제 Pitch의 type은 B/S/X 중 하나여야 합니다. "
            f"예시={examples}"
        )

    working["season"] = pd.Series(season, index=working.index, dtype="Int64")
    is_top = working["inning_topbot"].eq("top")
    working["team"] = working["away_team"].where(~is_top, working["home_team"])
    working["opponent"] = working["home_team"].where(~is_top, working["away_team"])
    working["is_home"] = is_top.astype("boolean")

    if working["team"].eq(working["opponent"]).any():
        raise PlayerGamePitchingBuildError("Raw에서 team과 opponent가 같은 Row가 있습니다.")

    if not working["game_date"].dt.year.eq(season).all():
        examples = (
            working.loc[
                ~working["game_date"].dt.year.eq(season),
                ["game_pk", "game_date"],
            ]
            .head(5)
            .to_dict("records")
        )
        raise PlayerGamePitchingBuildError(
            f"Raw.game_date의 연도가 처리 시즌 {season}과 다릅니다: {examples}"
        )

    return working


def prepare_plate_appearances(
    plate_appearances: pd.DataFrame,
    *,
    season: int,
) -> pd.DataFrame:
    """Canonical PA를 투구 집계용 형태로 검증·정규화한다."""
    _require_columns(
        plate_appearances,
        REQUIRED_PA_COLUMNS,
        source_name="Plate Appearance",
    )
    working = plate_appearances.copy()

    for column in (
        "game_pk",
        "batting_team",
        "fielding_team",
        "pitcher",
        "pitcher_name",
        "event",
    ):
        working[column] = _normalize_string_column(working[column])

    working["game_date"] = _normalize_game_date(working["game_date"])
    for column in (
        "season",
        "at_bat_number",
        "outs_before",
        "post_outs",
        "pitch_rows",
        "pitch_count",
    ):
        working[column] = _coerce_integer_column(
            working[column],
            column_name=f"Plate Appearance.{column}",
        )

    try:
        working["is_home_batting"] = working["is_home_batting"].astype("boolean")
    except (TypeError, ValueError) as exc:
        raise PlayerGamePitchingBuildError(
            "Plate Appearance.is_home_batting을 boolean으로 변환할 수 없습니다."
        ) from exc

    if working["is_home_batting"].isna().any():
        raise PlayerGamePitchingBuildError(
            "Plate Appearance.is_home_batting에는 null을 허용하지 않습니다."
        )

    _ensure_non_empty_string(
        working,
        ("game_pk", "batting_team", "fielding_team", "pitcher"),
        source_name="Plate Appearance",
    )
    _validate_unique_key(working, PA_KEY, source_name="Plate Appearance")

    if not working["season"].eq(season).all():
        raise PlayerGamePitchingBuildError(
            f"Plate Appearance.season이 처리 시즌 {season}과 일치하지 않습니다."
        )
    if not working["game_date"].dt.year.eq(season).all():
        raise PlayerGamePitchingBuildError(
            f"Plate Appearance.game_date의 연도가 처리 시즌 {season}과 일치하지 않습니다."
        )
    if working["pitch_rows"].lt(1).any():
        raise PlayerGamePitchingBuildError("Plate Appearance.pitch_rows는 1 이상이어야 합니다.")
    if working["pitch_count"].lt(0).any():
        raise PlayerGamePitchingBuildError("Plate Appearance.pitch_count는 0 이상이어야 합니다.")
    if working["pitch_count"].gt(working["pitch_rows"]).any():
        raise PlayerGamePitchingBuildError(
            "Plate Appearance.pitch_count가 pitch_rows보다 큰 Row가 있습니다."
        )

    invalid_event = working["event"].notna() & ~working["event"].isin(EVENT_VALUES)
    if invalid_event.any():
        values = working.loc[invalid_event, "event"].drop_duplicates().tolist()
        raise PlayerGamePitchingBuildError(
            f"지원하지 않는 Canonical PA event가 있습니다: {values}"
        )

    working["_pa_outs_recorded"] = working["post_outs"] - working["outs_before"]
    invalid_out_delta = (
        working["_pa_outs_recorded"].lt(0)
        | working["_pa_outs_recorded"].gt(3)
    )
    if invalid_out_delta.any():
        examples = (
            working.loc[
                invalid_out_delta,
                ["game_pk", "at_bat_number", "outs_before", "post_outs"],
            ]
            .head(10)
            .to_dict("records")
        )
        raise PlayerGamePitchingBuildError(
            f"PA Out Delta는 0~3이어야 합니다: {examples}"
        )

    working["team"] = working["fielding_team"]
    working["opponent"] = working["batting_team"]
    working["is_home"] = (~working["is_home_batting"]).astype("boolean")

    if working["team"].eq(working["opponent"]).any():
        raise PlayerGamePitchingBuildError(
            "Plate Appearance에서 team과 opponent가 같은 Row가 있습니다."
        )

    return working


def validate_pa_raw_alignment(
    raw: pd.DataFrame,
    plate_appearances: pd.DataFrame,
) -> None:
    """Raw와 Canonical PA가 동일한 PA 집합과 Pitch 수를 표현하는지 검증한다."""
    raw_keys = _dataframe_key_set(raw, PA_KEY)
    pa_keys = _dataframe_key_set(plate_appearances, PA_KEY)
    if raw_keys != pa_keys:
        missing_from_pa = list(raw_keys - pa_keys)[:10]
        missing_from_raw = list(pa_keys - raw_keys)[:10]
        raise PlayerGamePitchingBuildError(
            "Raw와 Canonical PA의 PA Key 집합이 다릅니다. "
            f"PA 누락 예시={missing_from_pa}, Raw 누락 예시={missing_from_raw}"
        )

    raw_counts = (
        raw.assign(_actual_pitch=raw["pitch_number"].gt(0).astype("Int64"))
        .groupby(list(PA_KEY), sort=False, dropna=False)
        .agg(
            _raw_pitch_rows=("pitch_number", "size"),
            _raw_pitch_count=("_actual_pitch", "sum"),
        )
        .reset_index()
    )

    checked = plate_appearances.merge(
        raw_counts,
        on=list(PA_KEY),
        how="left",
        validate="one_to_one",
        sort=False,
    )

    row_mismatch = checked["pitch_rows"].ne(checked["_raw_pitch_rows"])
    pitch_mismatch = checked["pitch_count"].ne(checked["_raw_pitch_count"])
    if row_mismatch.any() or pitch_mismatch.any():
        examples = (
            checked.loc[
                row_mismatch | pitch_mismatch,
                [
                    "game_pk",
                    "at_bat_number",
                    "pitch_rows",
                    "_raw_pitch_rows",
                    "pitch_count",
                    "_raw_pitch_count",
                ],
            ]
            .head(10)
            .to_dict("records")
        )
        raise PlayerGamePitchingBuildError(
            f"Raw와 Canonical PA의 Pitch Count가 일치하지 않습니다: {examples}"
        )


def _single_non_null_value(
    series: pd.Series,
    *,
    label: str,
) -> object:
    """null을 제외한 Context 값이 하나인지 확인하고 해당 값을 반환한다."""
    non_null = series.dropna()
    unique = pd.unique(non_null)
    if len(unique) > 1:
        raise PlayerGamePitchingBuildError(
            f"동일 Player Game에서 {label} 값이 충돌합니다: {unique[:5].tolist()}"
        )
    if len(unique) == 0:
        return pd.NA
    return unique[0]


def validate_player_game_context(raw: pd.DataFrame) -> None:
    """Raw 기준 Player Game Context가 하나의 값으로 일관되는지 검증한다."""
    grouped = raw.groupby(list(PLAYER_GAME_KEY), sort=False, dropna=False)

    required_context = ("game_date", "season", "team", "opponent", "is_home")
    for column in required_context:
        counts = grouped[column].nunique(dropna=False)
        invalid = counts.gt(1)
        if invalid.any():
            examples = list(counts.loc[invalid].index[:10])
            raise PlayerGamePitchingBuildError(
                f"동일 (game_pk, pitcher)에서 {column} Context가 충돌합니다: {examples}"
            )

    # 이름은 일부 Row에서 null일 수 있으므로 non-null 값끼리만 충돌 여부를 본다.
    name_counts = grouped["pitcher_name"].nunique(dropna=True)
    invalid_names = name_counts.gt(1)
    if invalid_names.any():
        examples = list(name_counts.loc[invalid_names].index[:10])
        raise PlayerGamePitchingBuildError(
            "동일 (game_pk, pitcher)에서 pitcher_name이 충돌합니다: "
            f"{examples}"
        )


def validate_raw_pa_pitcher_context(
    raw: pd.DataFrame,
    plate_appearances: pd.DataFrame,
) -> None:
    """Canonical PA 투수가 Raw Player Game Key 및 수비 Context와 일치하는지 검증한다."""
    raw_context = (
        raw.groupby(list(PLAYER_GAME_KEY), sort=False, dropna=False)
        .agg(
            raw_game_date=("game_date", "first"),
            raw_season=("season", "first"),
            raw_team=("team", "first"),
            raw_opponent=("opponent", "first"),
            raw_is_home=("is_home", "first"),
        )
        .reset_index()
    )

    checked = plate_appearances.merge(
        raw_context,
        on=list(PLAYER_GAME_KEY),
        how="left",
        validate="many_to_one",
        sort=False,
    )
    missing_key = checked["raw_team"].isna()
    if missing_key.any():
        examples = (
            checked.loc[missing_key, ["game_pk", "at_bat_number", "pitcher"]]
            .head(10)
            .to_dict("records")
        )
        raise PlayerGamePitchingBuildError(
            f"Canonical PA pitcher Key가 Raw Player Game Key에 없습니다: {examples}"
        )

    mismatch = (
        checked["game_date"].ne(checked["raw_game_date"])
        | checked["season"].ne(checked["raw_season"])
        | checked["team"].ne(checked["raw_team"])
        | checked["opponent"].ne(checked["raw_opponent"])
        | checked["is_home"].ne(checked["raw_is_home"])
    )
    if mismatch.any():
        examples = (
            checked.loc[
                mismatch,
                ["game_pk", "at_bat_number", "pitcher", "team", "raw_team"],
            ]
            .head(10)
            .to_dict("records")
        )
        raise PlayerGamePitchingBuildError(
            f"Raw와 Canonical PA의 Pitcher Context가 일치하지 않습니다: {examples}"
        )


def build_multi_pitcher_pa_table(
    raw: pd.DataFrame,
    plate_appearances: pd.DataFrame,
) -> pd.DataFrame:
    """PA별 Raw 투수 수와 실제 마지막 Raw 투수를 Canonical PA에 결합한다."""
    ordered_raw = raw.sort_values(
        by=["game_pk", "at_bat_number", "pitch_number"],
        kind="mergesort",
    )

    pitcher_counts = (
        ordered_raw.groupby(list(PA_KEY), sort=False, dropna=False)["pitcher"]
        .nunique(dropna=False)
        .rename("unique_pitchers_per_pa")
        .reset_index()
    )
    last_pitcher = (
        ordered_raw.groupby(list(PA_KEY), sort=False, as_index=False)
        .tail(1)
        .loc[:, ["game_pk", "at_bat_number", "pitcher"]]
        .rename(columns={"pitcher": "_last_raw_pitcher"})
    )

    return (
        plate_appearances.merge(
            pitcher_counts,
            on=list(PA_KEY),
            how="left",
            validate="one_to_one",
            sort=False,
        )
        .merge(
            last_pitcher,
            on=list(PA_KEY),
            how="left",
            validate="one_to_one",
            sort=False,
        )
    )


def identify_uncertain_multi_pitcher_out_keys(
    raw: pd.DataFrame,
    multi_pitcher_pa: pd.DataFrame,
) -> pd.DataFrame:
    """Out 발생 시점을 복원할 수 없는 Multi-pitcher PA의 Player Game Key를 반환한다.

    마지막 Raw Pitcher와 Canonical PA Pitcher가 다르면 PA 결과 귀속 자체가
    불일치하므로 실패시킨다. 반면 Event와 PA 전체 Out Delta만으로 투수별 Out
    발생 시점을 안전하게 구분할 수 없는 경우에는 빌드를 중단하지 않고 해당
    PA에 참여한 모든 `(game_pk, pitcher)`의 `outs_recorded`를 nullable `<NA>`로
    표시하기 위한 Key 집합을 반환한다.
    """
    candidates = multi_pitcher_pa.loc[
        multi_pitcher_pa["unique_pitchers_per_pa"].gt(1)
        & multi_pitcher_pa["_pa_outs_recorded"].gt(0)
    ].copy()

    if candidates.empty:
        return pd.DataFrame(columns=list(PLAYER_GAME_KEY)).astype(
            {"game_pk": "string", "pitcher": "string"}
        )

    pitcher_mismatch = candidates["_last_raw_pitcher"].ne(candidates["pitcher"])
    if pitcher_mismatch.any():
        examples = (
            candidates.loc[
                pitcher_mismatch,
                [
                    "game_pk",
                    "at_bat_number",
                    "pitcher",
                    "_last_raw_pitcher",
                    "event",
                    "_pa_outs_recorded",
                ],
            ]
            .head(10)
            .to_dict("records")
        )
        raise PlayerGamePitchingBuildError(
            "Multi-pitcher PA의 마지막 Raw pitcher와 Canonical PA pitcher가 "
            f"일치하지 않습니다. 예시={examples}"
        )

    terminal_outs = candidates["event"].map(MULTI_PITCHER_TERMINAL_OUTS).astype("Int64")
    safely_attributable = (
        terminal_outs.notna()
        & candidates["_pa_outs_recorded"].eq(terminal_outs)
    )
    uncertain_pa = candidates.loc[~safely_attributable].copy()

    if uncertain_pa.empty:
        LOGGER.info(
            "Multi-pitcher terminal out validation PASS | exact_pa=%d | uncertain_pa=0",
            len(candidates),
        )
        return pd.DataFrame(columns=list(PLAYER_GAME_KEY)).astype(
            {"game_pk": "string", "pitcher": "string"}
        )

    uncertain_pa_keys = uncertain_pa.loc[:, list(PA_KEY)].drop_duplicates()
    uncertain_player_game_keys = (
        raw.merge(
            uncertain_pa_keys,
            on=list(PA_KEY),
            how="inner",
            validate="many_to_one",
            sort=False,
        )
        .loc[:, list(PLAYER_GAME_KEY)]
        .drop_duplicates()
        .sort_values(list(PLAYER_GAME_KEY), kind="mergesort", ignore_index=True)
    )

    examples = (
        uncertain_pa.loc[
            :,
            [
                "game_pk",
                "at_bat_number",
                "event",
                "_pa_outs_recorded",
                "pitcher",
                "_last_raw_pitcher",
            ],
        ]
        .head(10)
        .to_dict("records")
    )
    LOGGER.warning(
        "Multi-pitcher PA의 Out 발생 시점을 공개 Raw만으로 안전하게 분리할 수 없어 "
        "관련 Player Game의 outs_recorded를 <NA>로 유지합니다. "
        "uncertain_pa=%d | affected_player_games=%d | 예시=%s",
        len(uncertain_pa),
        len(uncertain_player_game_keys),
        examples,
    )
    LOGGER.info(
        "Multi-pitcher terminal out validation PASS | exact_pa=%d | uncertain_pa=%d",
        int(safely_attributable.sum()),
        len(uncertain_pa),
    )
    return uncertain_player_game_keys


def apply_uncertain_outs_mask(
    frame: pd.DataFrame,
    uncertain_player_game_keys: pd.DataFrame,
) -> pd.DataFrame:
    """Out 책임 시점을 복원할 수 없는 Player Game의 outs_recorded를 <NA>로 만든다."""
    result = frame.copy()
    if uncertain_player_game_keys.empty:
        return result

    uncertain_index = pd.MultiIndex.from_frame(
        uncertain_player_game_keys.loc[:, list(PLAYER_GAME_KEY)]
    )
    result_index = pd.MultiIndex.from_frame(result.loc[:, list(PLAYER_GAME_KEY)])
    mask = result_index.isin(uncertain_index)
    result.loc[mask, "outs_recorded"] = pd.NA
    return result


def build_raw_pitcher_summary(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw Pitch Row에서 Player Game 투구량·구속 집계를 생성한다."""
    working = raw.copy()
    actual_pitch = working["pitch_number"].gt(0).fillna(False)
    normalized_pitch_type = working["type"].fillna("")

    working["_is_pitch"] = actual_pitch.astype("Int64")
    working["_is_ball_pitch"] = (
        actual_pitch & normalized_pitch_type.eq("B")
    ).astype("Int64")
    working["_is_strike_pitch"] = (
        actual_pitch & normalized_pitch_type.eq("S")
    ).astype("Int64")
    working["_is_in_play_pitch"] = (
        actual_pitch & normalized_pitch_type.eq("X")
    ).astype("Int64")
    working["_speed_for_average"] = working["release_speed_kmh"].where(actual_pitch)

    grouped = working.groupby(list(PLAYER_GAME_KEY), sort=False, dropna=False)
    summary = grouped.agg(
        game_date=("game_date", "first"),
        season=("season", "first"),
        team=("team", "first"),
        opponent=("opponent", "first"),
        is_home=("is_home", "first"),
        pitch_rows=("pitch_number", "size"),
        pitches=("_is_pitch", "sum"),
        ball_pitch_count=("_is_ball_pitch", "sum"),
        strike_pitch_count=("_is_strike_pitch", "sum"),
        in_play_pitch_count=("_is_in_play_pitch", "sum"),
        avg_release_speed_kmh=("_speed_for_average", "mean"),
    ).reset_index()

    names = (
        grouped["pitcher_name"]
        .apply(lambda series: _single_non_null_value(series, label="pitcher_name"))
        .rename("pitcher_name")
        .reset_index()
    )
    summary = summary.merge(
        names,
        on=list(PLAYER_GAME_KEY),
        how="left",
        validate="one_to_one",
        sort=False,
    )
    return summary


def build_pa_pitcher_summary(plate_appearances: pd.DataFrame) -> pd.DataFrame:
    """Canonical PA에서 완료 BF, 허용 Event 및 PA Out Delta를 집계한다."""
    working = plate_appearances.copy()
    completed = working["event"].notna()
    working["batters_faced_completed"] = completed.astype("Int64")

    for event, column in COUNT_EVENT_MAPPING.items():
        working[column] = (completed & working["event"].eq(event)).astype("Int64")

    grouped = (
        working.groupby(list(PLAYER_GAME_KEY), sort=False, dropna=False)
        .agg(
            batters_faced_completed=("batters_faced_completed", "sum"),
            single_allowed=("single_allowed", "sum"),
            double_allowed=("double_allowed", "sum"),
            triple_allowed=("triple_allowed", "sum"),
            hr_allowed=("hr_allowed", "sum"),
            bb_allowed=("bb_allowed", "sum"),
            hbp_allowed=("hbp_allowed", "sum"),
            so=("so", "sum"),
            sf=("sf", "sum"),
            sh=("sh", "sum"),
            outs_recorded=("_pa_outs_recorded", "sum"),
        )
        .reset_index()
    )
    return grouped


def validate_uncertain_outs_mask(
    frame: pd.DataFrame,
    uncertain_player_game_keys: pd.DataFrame,
) -> None:
    """outs_recorded의 null Key가 사전에 식별한 불확실 Player Game과 일치하는지 검증한다."""
    expected_keys = _dataframe_key_set(
        uncertain_player_game_keys,
        PLAYER_GAME_KEY,
    )
    actual_keys = _dataframe_key_set(
        frame.loc[frame["outs_recorded"].isna(), list(PLAYER_GAME_KEY)],
        PLAYER_GAME_KEY,
    )
    if expected_keys != actual_keys:
        missing_null = list(expected_keys - actual_keys)[:10]
        unexpected_null = list(actual_keys - expected_keys)[:10]
        raise PlayerGamePitchingBuildError(
            "outs_recorded 불확실성 Mask가 예상 Key와 일치하지 않습니다. "
            f"null 누락 예시={missing_null}, 예상 밖 null 예시={unexpected_null}"
        )


def normalize_player_game_pitching_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    """최종 Player Game Pitching Output dtype과 컬럼 순서를 정규화한다."""
    result = frame.copy()
    for column in STRING_OUTPUT_COLUMNS:
        result[column] = result[column].astype("string")
    result["game_date"] = _normalize_game_date(result["game_date"])
    for column in INTEGER_OUTPUT_COLUMNS:
        result[column] = _coerce_integer_column(
            result[column],
            column_name=f"Output.{column}",
            allow_null=column == "outs_recorded",
        )
    for column in BOOLEAN_OUTPUT_COLUMNS:
        result[column] = result[column].astype("boolean")
    for column in FLOAT_OUTPUT_COLUMNS:
        result[column] = _coerce_nullable_float_column(
            result[column],
            column_name=f"Output.{column}",
        ).round(3)
    return result.loc[:, list(OUTPUT_COLUMNS)]


def validate_player_game_output(frame: pd.DataFrame) -> None:
    """최종 Grain, 공식식, 금지 컬럼 및 기본 값 범위를 검증한다."""
    if tuple(frame.columns) != OUTPUT_COLUMNS:
        raise PlayerGamePitchingBuildError(
            "Player Game Pitching Output 컬럼 순서가 기대값과 다릅니다."
        )
    _validate_unique_key(frame, PLAYER_GAME_KEY, source_name="Player Game Pitching")
    _ensure_non_empty_string(
        frame,
        ("game_pk", "pitcher", "team", "opponent"),
        source_name="Player Game Pitching",
    )
    if frame["team"].eq(frame["opponent"]).any():
        raise PlayerGamePitchingBuildError("Output에서 team과 opponent가 같은 Row가 있습니다.")

    for column in INTEGER_OUTPUT_COLUMNS:
        if frame[column].lt(0).any():
            raise PlayerGamePitchingBuildError(
                f"Output.{column}에는 음수를 허용하지 않습니다."
            )

    pitch_formula = (
        frame["ball_pitch_count"]
        + frame["strike_pitch_count"]
        + frame["in_play_pitch_count"]
    )
    if frame["pitches"].ne(pitch_formula).any():
        raise PlayerGamePitchingBuildError(
            "pitches가 B/S/X Pitch Count 합과 일치하지 않습니다."
        )
    if frame["pitch_rows"].lt(frame["pitches"]).any():
        raise PlayerGamePitchingBuildError("pitch_rows가 pitches보다 작은 Row가 있습니다.")

    hits_formula = (
        frame["single_allowed"]
        + frame["double_allowed"]
        + frame["triple_allowed"]
        + frame["hr_allowed"]
    )
    if frame["hits_allowed"].ne(hits_formula).any():
        raise PlayerGamePitchingBuildError("hits_allowed 공식이 일치하지 않습니다.")

    prohibited = [column for column in PROHIBITED_OUTPUT_COLUMNS if column in frame.columns]
    if prohibited:
        raise PlayerGamePitchingBuildError(
            f"공식 책임을 검증하지 않은 금지 컬럼이 포함되어 있습니다: {prohibited}"
        )


def validate_grouped_reconciliation(
    raw: pd.DataFrame,
    plate_appearances: pd.DataFrame,
    output: pd.DataFrame,
) -> None:
    """Raw, Canonical PA, Player Game Output 사이의 핵심 Count를 교차 검증한다."""
    expected_raw_rows = len(raw)
    actual_raw_rows = int(output["pitch_rows"].sum())
    if actual_raw_rows != expected_raw_rows:
        raise PlayerGamePitchingBuildError(
            f"Raw Row reconciliation 실패: expected={expected_raw_rows}, actual={actual_raw_rows}"
        )

    expected_pitches = int(raw["pitch_number"].gt(0).sum())
    actual_pitches = int(output["pitches"].sum())
    if actual_pitches != expected_pitches:
        raise PlayerGamePitchingBuildError(
            f"Actual Pitch reconciliation 실패: expected={expected_pitches}, actual={actual_pitches}"
        )

    expected_completed = int(plate_appearances["event"].notna().sum())
    actual_completed = int(output["batters_faced_completed"].sum())
    if actual_completed != expected_completed:
        raise PlayerGamePitchingBuildError(
            "Completed BF reconciliation 실패: "
            f"expected={expected_completed}, actual={actual_completed}"
        )

    expected_outs_by_key = (
        plate_appearances.groupby(list(PLAYER_GAME_KEY), sort=False, dropna=False)[
            "_pa_outs_recorded"
        ]
        .sum()
        .rename("_expected_outs_recorded")
        .reset_index()
    )
    checked_outs = output.loc[
        :,
        ["game_pk", "pitcher", "outs_recorded"],
    ].merge(
        expected_outs_by_key,
        on=list(PLAYER_GAME_KEY),
        how="left",
        validate="one_to_one",
        sort=False,
    )
    checked_outs["_expected_outs_recorded"] = (
        checked_outs["_expected_outs_recorded"].fillna(0).astype("Int64")
    )
    known_outs = checked_outs["outs_recorded"].notna()
    outs_mismatch = known_outs & checked_outs["outs_recorded"].ne(
        checked_outs["_expected_outs_recorded"]
    )
    if outs_mismatch.any():
        examples = (
            checked_outs.loc[
                outs_mismatch,
                [
                    "game_pk",
                    "pitcher",
                    "outs_recorded",
                    "_expected_outs_recorded",
                ],
            ]
            .head(10)
            .to_dict("records")
        )
        raise PlayerGamePitchingBuildError(
            "Known outs_recorded reconciliation 실패: "
            f"예시={examples}"
        )

    event_expected = plate_appearances.loc[
        plate_appearances["event"].notna(), "event"
    ].value_counts()
    event_output_mapping = {
        "single": "single_allowed",
        "double": "double_allowed",
        "triple": "triple_allowed",
        "home_run": "hr_allowed",
        "walk": "bb_allowed",
        "hit_by_pitch": "hbp_allowed",
        "strikeout": "so",
        "sac_fly": "sf",
        "sac_bunt": "sh",
    }
    for event, column in event_output_mapping.items():
        expected = int(event_expected.get(event, 0))
        actual = int(output[column].sum())
        if expected != actual:
            raise PlayerGamePitchingBuildError(
                f"Event reconciliation 실패: event={event}, expected={expected}, actual={actual}"
            )

    # 경기별 Count도 다시 비교하여 전체 합계 상쇄로 인한 오류를 방지한다.
    raw_by_game = raw.groupby("game_pk", sort=False).size().astype("Int64")
    output_by_game = output.groupby("game_pk", sort=False)["pitch_rows"].sum()
    if not raw_by_game.sort_index().equals(output_by_game.sort_index()):
        raise PlayerGamePitchingBuildError("경기별 Raw Row reconciliation이 실패했습니다.")

    pa_by_game = (
        plate_appearances.assign(
            _completed=plate_appearances["event"].notna().astype("Int64")
        )
        .groupby("game_pk", sort=False)["_completed"]
        .sum()
    )
    bf_by_game = output.groupby("game_pk", sort=False)["batters_faced_completed"].sum()
    if not pa_by_game.sort_index().equals(bf_by_game.sort_index()):
        raise PlayerGamePitchingBuildError("경기별 Completed BF reconciliation이 실패했습니다.")


def build_player_game_pitching_from_prepared(
    raw: pd.DataFrame,
    plate_appearances: pd.DataFrame,
) -> pd.DataFrame:
    """검증·정규화된 Raw/PA에서 Player Game Pitching을 생성한다."""
    validate_pa_raw_alignment(raw, plate_appearances)
    validate_player_game_context(raw)
    validate_raw_pa_pitcher_context(raw, plate_appearances)

    multi_pitcher_pa = build_multi_pitcher_pa_table(raw, plate_appearances)
    uncertain_out_keys = identify_uncertain_multi_pitcher_out_keys(
        raw,
        multi_pitcher_pa,
    )

    raw_summary = build_raw_pitcher_summary(raw)
    pa_summary = build_pa_pitcher_summary(plate_appearances)

    result = raw_summary.merge(
        pa_summary,
        on=list(PLAYER_GAME_KEY),
        how="left",
        validate="one_to_one",
        sort=False,
    )

    for column in PA_COUNT_COLUMNS:
        result[column] = result[column].fillna(0).astype("Int64")

    result["hits_allowed"] = (
        result["single_allowed"]
        + result["double_allowed"]
        + result["triple_allowed"]
        + result["hr_allowed"]
    ).astype("Int64")
    result = apply_uncertain_outs_mask(result, uncertain_out_keys)

    result = normalize_player_game_pitching_dtypes(result)
    result = result.sort_values(
        by=["season", "game_date", "game_pk", "pitcher"],
        kind="mergesort",
        ignore_index=True,
    )

    validate_player_game_output(result)
    validate_uncertain_outs_mask(result, uncertain_out_keys)
    validate_grouped_reconciliation(raw, plate_appearances, result)
    return result


def build_player_game_pitching(
    raw: pd.DataFrame,
    plate_appearances: pd.DataFrame,
    season: int | None = None,
) -> pd.DataFrame:
    """Raw/Canonical PA DataFrame에서 한 시즌 Player Game Pitching을 생성한다."""
    if season is None:
        if "season" not in plate_appearances.columns:
            raise PlayerGamePitchingBuildError(
                "season 인자를 생략하려면 Plate Appearance에 season 컬럼이 필요합니다."
            )
        season_values = pd.to_numeric(
            plate_appearances["season"], errors="coerce"
        ).dropna().unique()
        if len(season_values) != 1:
            raise PlayerGamePitchingBuildError(
                "한 번의 build_player_game_pitching 호출에는 하나의 시즌만 허용합니다."
            )
        season = int(season_values[0])

    if int(season) not in DEFAULT_SEASONS:
        raise PlayerGamePitchingBuildError(f"지원하지 않는 시즌입니다: {season}")

    prepared_raw = prepare_raw_pitching_source(raw, season=int(season))
    prepared_pa = prepare_plate_appearances(plate_appearances, season=int(season))
    return build_player_game_pitching_from_prepared(prepared_raw, prepared_pa)


def _load_raw_season(raw_dir: Path, season: int) -> pd.DataFrame:
    path = raw_dir / f"{season}.parquet"
    if not path.is_file():
        raise PlayerGamePitchingBuildError(f"Raw Parquet 파일이 없습니다: {path}")
    try:
        return pd.read_parquet(path, engine="pyarrow")
    except (OSError, ValueError) as exc:
        raise PlayerGamePitchingBuildError(
            f"Raw Parquet을 읽을 수 없습니다: {path}"
        ) from exc


def _load_plate_appearances(pa_path: Path) -> pd.DataFrame:
    if not pa_path.is_file():
        raise PlayerGamePitchingBuildError(
            f"Canonical Plate Appearance 파일이 없습니다: {pa_path}"
        )
    try:
        return pd.read_parquet(pa_path, engine="pyarrow")
    except (OSError, ValueError) as exc:
        raise PlayerGamePitchingBuildError(
            f"Canonical Plate Appearance를 읽을 수 없습니다: {pa_path}"
        ) from exc


def _write_parquet_atomically(frame: pd.DataFrame, output_path: Path) -> None:
    """검증된 Output을 임시 파일에 쓴 뒤 최종 경로로 원자적으로 교체한다."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_path.parent,
            prefix=f".{output_path.stem}.",
            suffix=".tmp.parquet",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

        frame.to_parquet(temporary_path, index=False, engine="pyarrow")
        round_trip = pd.read_parquet(temporary_path, engine="pyarrow")
        round_trip = normalize_player_game_pitching_dtypes(round_trip)
        round_trip = round_trip.sort_values(
            by=["season", "game_date", "game_pk", "pitcher"],
            kind="mergesort",
            ignore_index=True,
        )
        validate_player_game_output(round_trip)
        pd.testing.assert_frame_equal(frame, round_trip, check_dtype=True)
        temporary_path.replace(output_path)
    except (OSError, ValueError, AssertionError) as exc:
        raise PlayerGamePitchingBuildError(
            f"Player Game Pitching Parquet 저장/round-trip 검증에 실패했습니다: {output_path}"
        ) from exc
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def build_player_game_pitching_file(
    *,
    raw_dir: Path = DEFAULT_RAW_DIR,
    pa_path: Path = DEFAULT_PA_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    seasons: Sequence[int] = DEFAULT_SEASONS,
) -> pd.DataFrame:
    """시즌별 Raw/Canonical PA를 읽어 최종 Parquet을 안전하게 생성한다."""
    normalized_seasons = normalize_seasons(seasons)
    ensure_output_path_safe(
        raw_dir=raw_dir,
        pa_path=pa_path,
        output_path=output_path,
        seasons=normalized_seasons,
    )

    raw_paths = [raw_dir / f"{season}.parquet" for season in normalized_seasons]
    for raw_path in raw_paths:
        if not raw_path.is_file():
            raise PlayerGamePitchingBuildError(f"Raw Parquet 파일이 없습니다: {raw_path}")
    if not pa_path.is_file():
        raise PlayerGamePitchingBuildError(
            f"Canonical Plate Appearance 파일이 없습니다: {pa_path}"
        )

    raw_hashes_before = {path: calculate_sha256(path) for path in raw_paths}
    pa_hash_before = calculate_sha256(pa_path)

    plate_appearances = _load_plate_appearances(pa_path)
    _require_columns(
        plate_appearances,
        REQUIRED_PA_COLUMNS,
        source_name="Plate Appearance",
    )
    outputs: list[pd.DataFrame] = []

    for season in normalized_seasons:
        LOGGER.info("%d 시즌 Player Game Pitching 변환 시작", season)
        raw = _load_raw_season(raw_dir, season)
        season_pa = plate_appearances.loc[
            pd.to_numeric(plate_appearances["season"], errors="coerce").eq(season)
        ].copy()
        if season_pa.empty:
            raise PlayerGamePitchingBuildError(
                f"Canonical Plate Appearance에 {season} 시즌 Row가 없습니다."
            )

        output = build_player_game_pitching(raw, season_pa, season=season)
        outputs.append(output)

        multi_pitcher_count = int(
            prepare_raw_pitching_source(raw, season=season)
            .groupby(list(PA_KEY))["pitcher"]
            .nunique(dropna=False)
            .gt(1)
            .sum()
        )
        pitchless_count = int((raw["pitch_number"] == 0).sum())
        known_outs = int(output["outs_recorded"].sum(skipna=True))
        unknown_out_player_games = int(output["outs_recorded"].isna().sum())
        LOGGER.info(
            "%d 시즌 PASS | player_game_rows=%d | raw_rows=%d | pitches=%d | "
            "completed_bf=%d | known_outs=%d | unknown_out_player_games=%d | "
            "multi_pitcher_pa=%d | pitchless_rows=%d",
            season,
            len(output),
            int(output["pitch_rows"].sum()),
            int(output["pitches"].sum()),
            int(output["batters_faced_completed"].sum()),
            known_outs,
            unknown_out_player_games,
            multi_pitcher_count,
            pitchless_count,
        )

    final = pd.concat(outputs, ignore_index=True)
    final = normalize_player_game_pitching_dtypes(final)
    final = final.sort_values(
        by=["season", "game_date", "game_pk", "pitcher"],
        kind="mergesort",
        ignore_index=True,
    )
    validate_player_game_output(final)

    _write_parquet_atomically(final, output_path)

    for path, hash_before in raw_hashes_before.items():
        hash_after = calculate_sha256(path)
        if hash_before != hash_after:
            raise PlayerGamePitchingBuildError(f"Raw 입력 파일이 변경되었습니다: {path}")
    if pa_hash_before != calculate_sha256(pa_path):
        raise PlayerGamePitchingBuildError(
            f"Canonical Plate Appearance 입력 파일이 변경되었습니다: {pa_path}"
        )

    LOGGER.info(
        "Player Game Pitching 생성 완료 | output=%s | rows=%d | duplicate_key=%d",
        output_path,
        len(final),
        int(final.duplicated(list(PLAYER_GAME_KEY)).sum()),
    )
    return final


def main() -> None:
    """CLI 진입점."""
    configure_logging()
    args = parse_args()
    try:
        build_player_game_pitching_file(
            raw_dir=args.raw_dir,
            pa_path=args.pa_path,
            output_path=args.output_path,
            seasons=args.seasons,
        )
    except PlayerGamePitchingBuildError as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
