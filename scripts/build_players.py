from __future__ import annotations

import argparse
import hashlib
import json
import logging
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PA_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "hf_kbo_pbp"
    / "derived"
    / "plate_appearances.parquet"
)

DEFAULT_BATTING_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "hf_kbo_pbp"
    / "derived"
    / "player_game_batting.parquet"
)

DEFAULT_PITCHING_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "hf_kbo_pbp"
    / "derived"
    / "player_game_pitching.parquet"
)

DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "hf_kbo_pbp"
    / "derived"
    / "players.parquet"
)

REQUIRED_PA_COLUMNS = (
    "game_pk",
    "game_date",
    "season",
    "batter",
    "batter_name",
    "pitcher",
    "pitcher_name",
)

REQUIRED_BATTING_COLUMNS = (
    "game_pk",
    "game_date",
    "season",
    "batter",
    "batter_name",
)

REQUIRED_PITCHING_COLUMNS = (
    "game_pk",
    "game_date",
    "season",
    "pitcher",
    "pitcher_name",
)

OBSERVATION_COLUMNS = (
    "player_id",
    "player_name",
    "game_pk",
    "game_date",
    "season",
    "role",
    "source",
)

FACT_COLUMNS = (
    "player_id",
    "game_pk",
    "game_date",
    "season",
    "role",
    "player_name",
)

OUTPUT_COLUMNS = (
    "player_id",
    "display_name",
    "name_variants",
    "is_batter",
    "is_pitcher",
    "first_seen_date",
    "last_seen_date",
    "first_seen_season",
    "last_seen_season",
)

STRING_OUTPUT_COLUMNS = (
    "player_id",
    "display_name",
    "name_variants",
)

BOOLEAN_OUTPUT_COLUMNS = (
    "is_batter",
    "is_pitcher",
)

INTEGER_OUTPUT_COLUMNS = (
    "first_seen_season",
    "last_seen_season",
)

PROHIBITED_OUTPUT_COLUMNS = (
    "team",
    "current_team",
    "latest_team",
    "first_team",
    "last_team",
)

ROLE_BATTER = "batter"
ROLE_PITCHER = "pitcher"

SOURCE_PA = "plate_appearance"
SOURCE_BATTING = "player_game_batting"
SOURCE_PITCHING = "player_game_pitching"


class PlayersBuildError(RuntimeError):
    """Player Metadata 생성을 중단해야 하는 검증 오류를 나타낸다."""


def parse_args() -> argparse.Namespace:
    """Player Metadata 생성 스크립트의 CLI 인자를 파싱한다."""
    parser = argparse.ArgumentParser(
        description=(
            "Canonical Plate Appearance, Player Game Batting, "
            "Player Game Pitching에서 Player Metadata를 생성합니다."
        )
    )
    parser.add_argument(
        "--pa-path",
        type=Path,
        default=DEFAULT_PA_PATH,
        help=f"Canonical Plate Appearance 경로입니다. 기본값: {DEFAULT_PA_PATH}",
    )
    parser.add_argument(
        "--batting-path",
        type=Path,
        default=DEFAULT_BATTING_PATH,
        help=f"Player Game Batting 경로입니다. 기본값: {DEFAULT_BATTING_PATH}",
    )
    parser.add_argument(
        "--pitching-path",
        type=Path,
        default=DEFAULT_PITCHING_PATH,
        help=f"Player Game Pitching 경로입니다. 기본값: {DEFAULT_PITCHING_PATH}",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Player Metadata 출력 경로입니다. 기본값: {DEFAULT_OUTPUT_PATH}",
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
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        raise PlayersBuildError(
            f"{source_name}에 필요한 컬럼이 없습니다: {missing}"
        )


def _coerce_integer_column(
    series: pd.Series,
    *,
    column_name: str,
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
        raise PlayersBuildError(
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
        raise PlayersBuildError(
            f"{column_name}에 정수가 아닌 값이 있습니다: "
            f"{examples}"
        )

    result = numeric.astype("Int64")

    if result.isna().any():
        raise PlayersBuildError(
            f"{column_name}에는 null을 허용하지 않습니다."
        )

    return result


def _normalize_game_date(
    series: pd.Series,
    *,
    column_name: str,
) -> pd.Series:
    """날짜를 timezone 없는 datetime64[us]로 정규화한다."""
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
        raise PlayersBuildError(
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
        raise PlayersBuildError(
            f"{column_name}을 datetime64[us]로 "
            "정규화할 수 없습니다."
        ) from exc


def _normalize_required_string(
    series: pd.Series,
    *,
    column_name: str,
) -> pd.Series:
    """
    필수 문자열을 pandas string으로 변환한다.

    Player ID의 원래 문자열 의미를 보존하기 위해 값 자체를 strip하거나
    정수로 변환하지 않고, null/빈 문자열/공백-only 여부만 검증한다.
    """
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
        raise PlayersBuildError(
            f"{column_name}에 null 또는 빈 문자열이 있습니다: "
            f"{examples}"
        )

    return values


def _normalize_player_name(
    series: pd.Series,
) -> pd.Series:
    """
    Player Name을 nullable string으로 정규화한다.

    이름 자체를 추정 보정하지 않고 leading/trailing whitespace만 제거한다.
    빈 문자열 또는 공백-only 값은 valid Name이 아니므로 <NA>로 변환한다.
    """
    values = (
        series
        .astype("string")
        .str
        .strip()
    )

    empty = (
        values.notna()
        & values.eq("")
    )

    return values.mask(
        empty,
        pd.NA,
    )


def _validate_source_game_context(
    frame: pd.DataFrame,
    *,
    source_name: str,
) -> None:
    """
    한 Source 내부에서 동일 game_pk의 game_date/season이
    하나의 Context로 일관되는지 검증한다.
    """
    if frame.empty:
        return

    context = (
        frame.loc[
            :,
            [
                "game_pk",
                "game_date",
                "season",
            ],
        ]
        .drop_duplicates()
    )

    grouped = context.groupby(
        "game_pk",
        sort=False,
        dropna=False,
    )

    date_counts = grouped["game_date"].nunique(
        dropna=False
    )
    season_counts = grouped["season"].nunique(
        dropna=False
    )

    invalid_game_pk = set(
        date_counts.loc[
            date_counts.ne(1)
        ].index.tolist()
    )
    invalid_game_pk.update(
        season_counts.loc[
            season_counts.ne(1)
        ].index.tolist()
    )

    if invalid_game_pk:
        examples = (
            context.loc[
                context["game_pk"].isin(
                    list(invalid_game_pk)
                )
            ]
            .sort_values(
                by=[
                    "game_pk",
                    "game_date",
                    "season",
                ],
                kind="mergesort",
            )
            .head(10)
            .to_dict("records")
        )

        raise PlayersBuildError(
            f"{source_name}에서 동일 game_pk의 "
            "game_date/season Context가 충돌합니다: "
            f"{examples}"
        )


def _prepare_role_observations(
    frame: pd.DataFrame,
    *,
    required_columns: Sequence[str],
    id_column: str,
    name_column: str,
    role: str,
    source: str,
    source_name: str,
) -> pd.DataFrame:
    """한 Source의 Batter 또는 Pitcher 관측을 공통 Schema로 변환한다."""
    _require_columns(
        frame,
        required_columns,
        source_name=source_name,
    )

    working = frame.loc[
        :,
        [
            "game_pk",
            "game_date",
            "season",
            id_column,
            name_column,
        ],
    ].copy()

    working["game_pk"] = _normalize_required_string(
        working["game_pk"],
        column_name=f"{source_name}.game_pk",
    )
    working["game_date"] = _normalize_game_date(
        working["game_date"],
        column_name=f"{source_name}.game_date",
    )
    working["season"] = _coerce_integer_column(
        working["season"],
        column_name=f"{source_name}.season",
    )
    working[id_column] = _normalize_required_string(
        working[id_column],
        column_name=f"{source_name}.{id_column}",
    )
    working[name_column] = _normalize_player_name(
        working[name_column]
    )

    _validate_source_game_context(
        working,
        source_name=source_name,
    )

    result = (
        working.rename(
            columns={
                id_column: "player_id",
                name_column: "player_name",
            }
        )
        .assign(
            role=role,
            source=source,
        )
    )

    result["role"] = result["role"].astype(
        "string"
    )
    result["source"] = result["source"].astype(
        "string"
    )

    return result.loc[
        :,
        OBSERVATION_COLUMNS,
    ]


def prepare_plate_appearance_observations(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """PA의 Batter/Pitcher를 두 개의 Role Observation으로 변환한다."""
    batter = _prepare_role_observations(
        frame,
        required_columns=REQUIRED_PA_COLUMNS,
        id_column="batter",
        name_column="batter_name",
        role=ROLE_BATTER,
        source=SOURCE_PA,
        source_name="Plate Appearance",
    )

    pitcher = _prepare_role_observations(
        frame,
        required_columns=REQUIRED_PA_COLUMNS,
        id_column="pitcher",
        name_column="pitcher_name",
        role=ROLE_PITCHER,
        source=SOURCE_PA,
        source_name="Plate Appearance",
    )

    return pd.concat(
        [
            batter,
            pitcher,
        ],
        ignore_index=True,
    )


def prepare_batting_observations(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Player Game Batting을 Batter Observation으로 변환한다."""
    return _prepare_role_observations(
        frame,
        required_columns=REQUIRED_BATTING_COLUMNS,
        id_column="batter",
        name_column="batter_name",
        role=ROLE_BATTER,
        source=SOURCE_BATTING,
        source_name="Player Game Batting",
    )


def prepare_pitching_observations(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Player Game Pitching을 Pitcher Observation으로 변환한다."""
    return _prepare_role_observations(
        frame,
        required_columns=REQUIRED_PITCHING_COLUMNS,
        id_column="pitcher",
        name_column="pitcher_name",
        role=ROLE_PITCHER,
        source=SOURCE_PITCHING,
        source_name="Player Game Pitching",
    )


def validate_cross_source_game_context(
    observations: pd.DataFrame,
) -> None:
    """
    세 Source에 공통으로 존재하는 game_pk의 game_date/season이
    서로 일치하는지 교차 검증한다.
    """
    if observations.empty:
        return

    context = (
        observations.loc[
            :,
            [
                "source",
                "game_pk",
                "game_date",
                "season",
            ],
        ]
        .drop_duplicates()
    )

    grouped = context.groupby(
        "game_pk",
        sort=False,
        dropna=False,
    )

    date_counts = grouped["game_date"].nunique(
        dropna=False
    )
    season_counts = grouped["season"].nunique(
        dropna=False
    )

    invalid_game_pk = set(
        date_counts.loc[
            date_counts.ne(1)
        ].index.tolist()
    )
    invalid_game_pk.update(
        season_counts.loc[
            season_counts.ne(1)
        ].index.tolist()
    )

    if invalid_game_pk:
        examples = (
            context.loc[
                context["game_pk"].isin(
                    list(invalid_game_pk)
                )
            ]
            .sort_values(
                by=[
                    "game_pk",
                    "source",
                    "game_date",
                    "season",
                ],
                kind="mergesort",
            )
            .head(15)
            .to_dict("records")
        )

        raise PlayersBuildError(
            "Canonical Source 간 동일 game_pk의 "
            "game_date/season Context가 충돌합니다: "
            f"{examples}"
        )


def build_player_observations(
    plate_appearances: pd.DataFrame,
    player_game_batting: pd.DataFrame,
    player_game_pitching: pd.DataFrame,
) -> pd.DataFrame:
    """세 Canonical Input을 공통 Player Observation으로 통합한다."""
    pa = prepare_plate_appearance_observations(
        plate_appearances
    )
    batting = prepare_batting_observations(
        player_game_batting
    )
    pitching = prepare_pitching_observations(
        player_game_pitching
    )

    combined = pd.concat(
        [
            pa,
            batting,
            pitching,
        ],
        ignore_index=True,
    )

    if combined.empty:
        raise PlayersBuildError(
            "Player Metadata를 생성할 Player Observation이 없습니다."
        )

    validate_cross_source_game_context(
        combined
    )

    return combined


def deduplicate_observations(
    observations: pd.DataFrame,
) -> pd.DataFrame:
    """
    동일 사실이 여러 Source에 존재할 때 Source Row 빈도가
    Name Metadata에 가중치로 작동하지 않도록 중복을 제거한다.
    """
    working = (
        observations
        .sort_values(
            by=[
                "player_id",
                "game_date",
                "game_pk",
                "role",
                "player_name",
                "source",
            ],
            kind="mergesort",
            na_position="last",
        )
        .drop_duplicates(
            subset=list(FACT_COLUMNS),
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    return working


def _resolve_boundary_season(
    player_group: pd.DataFrame,
    *,
    boundary_date: pd.Timestamp,
    boundary_label: str,
) -> int:
    """
    first/last date에 실제 연결된 Season 하나를 반환한다.

    같은 boundary date에서 서로 다른 Season이 관측되면
    Date/Season 의미가 충돌하므로 실패시킨다.
    """
    seasons = (
        player_group.loc[
            player_group["game_date"].eq(
                boundary_date
            ),
            "season",
        ]
        .drop_duplicates()
        .sort_values(
            kind="mergesort"
        )
        .tolist()
    )

    if len(seasons) != 1:
        player_id = str(
            player_group["player_id"].iloc[0]
        )
        raise PlayersBuildError(
            f"player_id={player_id}의 {boundary_label} "
            f"{boundary_date}에서 season Context가 충돌합니다: "
            f"{seasons}"
        )

    return int(
        seasons[0]
    )


def _metadata_from_player_group(
    player_group: pd.DataFrame,
) -> dict[str, object]:
    """한 Player의 Role, Name Variant, 관측 기간 Metadata를 계산한다."""
    player_id = str(
        player_group["player_id"].iloc[0]
    )

    roles = set(
        player_group["role"]
        .dropna()
        .astype(str)
        .tolist()
    )

    is_batter = (
        ROLE_BATTER in roles
    )
    is_pitcher = (
        ROLE_PITCHER in roles
    )

    if not is_batter and not is_pitcher:
        raise PlayersBuildError(
            f"player_id={player_id}에 유효한 Batter/Pitcher Role이 없습니다."
        )

    valid_names = (
        player_group.loc[
            player_group["player_name"].notna()
        ]
        .copy()
    )

    name_variants = sorted(
        set(
            valid_names["player_name"]
            .astype(str)
            .tolist()
        )
    )

    if valid_names.empty:
        display_name: object = pd.NA
    else:
        latest_name_date = (
            valid_names["game_date"]
            .max()
        )

        latest_names = sorted(
            set(
                valid_names.loc[
                    valid_names["game_date"].eq(
                        latest_name_date
                    ),
                    "player_name",
                ]
                .astype(str)
                .tolist()
            )
        )

        display_name = latest_names[0]

    first_seen_date = (
        player_group["game_date"]
        .min()
    )
    last_seen_date = (
        player_group["game_date"]
        .max()
    )

    first_seen_season = _resolve_boundary_season(
        player_group,
        boundary_date=first_seen_date,
        boundary_label="first_seen_date",
    )
    last_seen_season = _resolve_boundary_season(
        player_group,
        boundary_date=last_seen_date,
        boundary_label="last_seen_date",
    )

    return {
        "player_id": player_id,
        "display_name": display_name,
        "name_variants": json.dumps(
            name_variants,
            ensure_ascii=False,
        ),
        "is_batter": is_batter,
        "is_pitcher": is_pitcher,
        "first_seen_date": first_seen_date,
        "last_seen_date": last_seen_date,
        "first_seen_season": first_seen_season,
        "last_seen_season": last_seen_season,
    }


def _summarize_players(
    observations: pd.DataFrame,
) -> pd.DataFrame:
    """중복 제거된 Observation을 player_id Grain으로 집계한다."""
    rows: list[dict[str, object]] = []

    for _, player_group in observations.groupby(
        "player_id",
        sort=False,
        dropna=False,
    ):
        rows.append(
            _metadata_from_player_group(
                player_group
            )
        )

    return pd.DataFrame(
        rows,
        columns=OUTPUT_COLUMNS,
    )


def normalize_players_dtypes(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Player Metadata의 컬럼 순서와 dtype을 고정한다."""
    missing = [
        column
        for column in OUTPUT_COLUMNS
        if column not in frame.columns
    ]
    if missing:
        raise PlayersBuildError(
            f"Player Output에 필요한 컬럼이 없습니다: {missing}"
        )

    unexpected = [
        column
        for column in frame.columns
        if column not in OUTPUT_COLUMNS
    ]
    if unexpected:
        raise PlayersBuildError(
            f"Player Output에 정의되지 않은 컬럼이 있습니다: "
            f"{unexpected}"
        )

    result = frame.loc[
        :,
        OUTPUT_COLUMNS,
    ].copy()

    for column in STRING_OUTPUT_COLUMNS:
        result[column] = (
            result[column]
            .astype("string")
        )

    result["first_seen_date"] = _normalize_game_date(
        result["first_seen_date"],
        column_name="Output.first_seen_date",
    )
    result["last_seen_date"] = _normalize_game_date(
        result["last_seen_date"],
        column_name="Output.last_seen_date",
    )

    for column in BOOLEAN_OUTPUT_COLUMNS:
        try:
            result[column] = (
                result[column]
                .astype("boolean")
            )
        except (TypeError, ValueError) as exc:
            raise PlayersBuildError(
                f"Output.{column}을 boolean으로 변환할 수 없습니다."
            ) from exc

        if result[column].isna().any():
            raise PlayersBuildError(
                f"Output.{column}에는 null을 허용하지 않습니다."
            )

    for column in INTEGER_OUTPUT_COLUMNS:
        result[column] = _coerce_integer_column(
            result[column],
            column_name=f"Output.{column}",
        )

    result = (
        result
        .sort_values(
            by=["player_id"],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    return result


def _parse_name_variants(
    value: object,
    *,
    player_id: str,
) -> list[str]:
    """name_variants JSON Array 계약을 검증하고 Python list로 반환한다."""
    if pd.isna(value):
        raise PlayersBuildError(
            f"player_id={player_id}의 name_variants가 null입니다."
        )

    try:
        parsed = json.loads(
            str(value)
        )
    except json.JSONDecodeError as exc:
        raise PlayersBuildError(
            f"player_id={player_id}의 name_variants가 "
            "유효한 JSON이 아닙니다."
        ) from exc

    if not isinstance(parsed, list):
        raise PlayersBuildError(
            f"player_id={player_id}의 name_variants는 "
            "JSON Array여야 합니다."
        )

    if not all(
        isinstance(name, str)
        for name in parsed
    ):
        raise PlayersBuildError(
            f"player_id={player_id}의 name_variants에는 "
            "문자열만 허용합니다."
        )

    invalid_name = [
        name
        for name in parsed
        if (
            not name.strip()
            or name != name.strip()
        )
    ]
    if invalid_name:
        raise PlayersBuildError(
            f"player_id={player_id}의 name_variants에 "
            "빈 이름 또는 정규화되지 않은 이름이 있습니다: "
            f"{invalid_name[:5]}"
        )

    expected_order = sorted(
        set(parsed)
    )
    if parsed != expected_order:
        raise PlayersBuildError(
            f"player_id={player_id}의 name_variants가 "
            "unique deterministic sort 계약을 만족하지 않습니다."
        )

    return parsed


def _validate_output_dtypes(
    frame: pd.DataFrame,
) -> None:
    """Output dtype 계약을 검증한다."""
    expected = {
        "player_id": "string",
        "display_name": "string",
        "name_variants": "string",
        "is_batter": "boolean",
        "is_pitcher": "boolean",
        "first_seen_date": "datetime64[us]",
        "last_seen_date": "datetime64[us]",
        "first_seen_season": "Int64",
        "last_seen_season": "Int64",
    }

    actual = {
        column: str(
            frame[column].dtype
        )
        for column in OUTPUT_COLUMNS
    }

    mismatch = {
        column: {
            "expected": expected[column],
            "actual": actual[column],
        }
        for column in OUTPUT_COLUMNS
        if actual[column] != expected[column]
    }

    if mismatch:
        raise PlayersBuildError(
            f"Player Output dtype 계약이 일치하지 않습니다: "
            f"{mismatch}"
        )


def validate_players_output(
    frame: pd.DataFrame,
    *,
    observations: pd.DataFrame | None = None,
) -> None:
    """
    Player Output의 Grain, Role, Name, 관측기간 및
    Source Reconciliation을 검증한다.
    """
    missing = [
        column
        for column in OUTPUT_COLUMNS
        if column not in frame.columns
    ]
    if missing:
        raise PlayersBuildError(
            f"Player Output에 필요한 컬럼이 없습니다: {missing}"
        )

    unexpected = [
        column
        for column in frame.columns
        if column not in OUTPUT_COLUMNS
    ]
    if unexpected:
        raise PlayersBuildError(
            f"Player Output에 정의되지 않은 컬럼이 있습니다: "
            f"{unexpected}"
        )

    prohibited = [
        column
        for column in PROHIBITED_OUTPUT_COLUMNS
        if column in frame.columns
    ]
    if prohibited:
        raise PlayersBuildError(
            f"Player Master에 금지된 Team 컬럼이 있습니다: "
            f"{prohibited}"
        )

    _validate_output_dtypes(
        frame
    )

    invalid_id = (
        frame["player_id"].isna()
        | frame["player_id"]
        .str
        .strip()
        .eq("")
        .fillna(False)
    )
    if invalid_id.any():
        raise PlayersBuildError(
            "Player Output.player_id에 null 또는 빈 문자열이 있습니다."
        )

    duplicate_count = int(
        frame.duplicated(
            subset=["player_id"],
            keep=False,
        ).sum()
    )
    if duplicate_count > 0:
        examples = (
            frame.loc[
                frame.duplicated(
                    subset=["player_id"],
                    keep=False,
                ),
                ["player_id"],
            ]
            .head(10)
            .to_dict("records")
        )
        raise PlayersBuildError(
            "Player Output.player_id가 Unique하지 않습니다: "
            f"duplicate_rows={duplicate_count}, 예시={examples}"
        )

    no_role = (
        ~frame["is_batter"]
        & ~frame["is_pitcher"]
    )
    if no_role.any():
        examples = (
            frame.loc[
                no_role,
                ["player_id"],
            ]
            .head(10)
            .to_dict("records")
        )
        raise PlayersBuildError(
            "Batter/Pitcher Role이 모두 False인 Player가 있습니다: "
            f"{examples}"
        )

    invalid_date_range = (
        frame["first_seen_date"]
        .gt(
            frame["last_seen_date"]
        )
    )
    if invalid_date_range.any():
        examples = (
            frame.loc[
                invalid_date_range,
                [
                    "player_id",
                    "first_seen_date",
                    "last_seen_date",
                ],
            ]
            .head(10)
            .to_dict("records")
        )
        raise PlayersBuildError(
            "first_seen_date가 last_seen_date보다 늦은 "
            f"Player가 있습니다: {examples}"
        )

    for row in frame.itertuples(
        index=False
    ):
        variants = _parse_name_variants(
            row.name_variants,
            player_id=str(
                row.player_id
            ),
        )

        if not variants:
            if not pd.isna(
                row.display_name
            ):
                raise PlayersBuildError(
                    f"player_id={row.player_id}는 "
                    "Name Variant가 없지만 display_name이 존재합니다."
                )
        else:
            if pd.isna(
                row.display_name
            ):
                raise PlayersBuildError(
                    f"player_id={row.player_id}는 "
                    "Name Variant가 있지만 display_name이 null입니다."
                )

            if str(
                row.display_name
            ) not in variants:
                raise PlayersBuildError(
                    f"player_id={row.player_id}의 display_name이 "
                    "name_variants에 존재하지 않습니다."
                )

    sorted_ids = (
        frame["player_id"]
        .sort_values(
            kind="mergesort"
        )
        .reset_index(
            drop=True
        )
    )
    actual_ids = (
        frame["player_id"]
        .reset_index(
            drop=True
        )
    )
    if not actual_ids.equals(
        sorted_ids
    ):
        raise PlayersBuildError(
            "Player Output이 player_id ascending 순으로 "
            "정렬되어 있지 않습니다."
        )

    if observations is None:
        return

    deduplicated = deduplicate_observations(
        observations
    )

    expected = _summarize_players(
        deduplicated
    )
    expected = normalize_players_dtypes(
        expected
    )

    expected_ids = set(
        deduplicated["player_id"]
        .astype(str)
        .tolist()
    )
    actual_id_set = set(
        frame["player_id"]
        .astype(str)
        .tolist()
    )

    if expected_ids != actual_id_set:
        missing_ids = sorted(
            expected_ids
            - actual_id_set
        )[:10]
        unexpected_ids = sorted(
            actual_id_set
            - expected_ids
        )[:10]

        raise PlayersBuildError(
            "Player ID Union Reconciliation이 실패했습니다. "
            f"missing={missing_ids}, unexpected={unexpected_ids}"
        )

    try:
        pd.testing.assert_frame_equal(
            expected,
            frame.reset_index(
                drop=True
            ),
            check_dtype=True,
            check_like=False,
        )
    except AssertionError as exc:
        raise PlayersBuildError(
            "Player Metadata가 Source Observation에서 "
            "재계산한 기대값과 일치하지 않습니다."
        ) from exc


def build_players(
    plate_appearances: pd.DataFrame,
    player_game_batting: pd.DataFrame,
    player_game_pitching: pd.DataFrame,
) -> pd.DataFrame:
    """세 Canonical DataFrame에서 Player Metadata를 생성한다."""
    observations = build_player_observations(
        plate_appearances,
        player_game_batting,
        player_game_pitching,
    )

    deduplicated = deduplicate_observations(
        observations
    )

    result = _summarize_players(
        deduplicated
    )
    result = normalize_players_dtypes(
        result
    )

    validate_players_output(
        result,
        observations=observations,
    )

    return result


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


def ensure_output_path_safe(
    *,
    pa_path: Path,
    batting_path: Path,
    pitching_path: Path,
    output_path: Path,
) -> None:
    """Player Output이 Raw 또는 세 Canonical Input을 덮어쓰지 않도록 검증한다."""
    output_resolved = output_path.resolve(
        strict=False
    )

    if _path_is_within(
        output_path,
        PROJECT_ROOT / "data" / "raw",
    ):
        raise PlayersBuildError(
            "Player Metadata Output을 data/raw/ 내부에 "
            "생성할 수 없습니다."
        )

    input_paths = {
        "plate_appearances.parquet": pa_path,
        "player_game_batting.parquet": batting_path,
        "player_game_pitching.parquet": pitching_path,
    }

    for label, input_path in input_paths.items():
        if (
            output_resolved
            == input_path.resolve(
                strict=False
            )
        ):
            raise PlayersBuildError(
                f"Player Metadata Output이 {label}을 "
                "덮어쓸 수 없습니다."
            )


def _read_parquet(
    path: Path,
    *,
    source_name: str,
) -> pd.DataFrame:
    """Canonical Parquet을 읽고 읽기 실패를 명시적인 BuildError로 변환한다."""
    if not path.is_file():
        raise PlayersBuildError(
            f"{source_name} 파일이 없습니다: {path}"
        )

    try:
        return pd.read_parquet(
            path,
            engine="pyarrow",
        )
    except (OSError, ValueError) as exc:
        raise PlayersBuildError(
            f"{source_name} Parquet을 읽을 수 없습니다: "
            f"{path}"
        ) from exc


def _write_parquet_atomically(
    frame: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    검증된 Player Output을 임시 Parquet에 기록한 뒤
    round-trip 검증 후 최종 경로로 원자적으로 교체한다.
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
            index=False,
            engine="pyarrow",
        )

        round_trip = pd.read_parquet(
            temporary_path,
            engine="pyarrow",
        )
        round_trip = normalize_players_dtypes(
            round_trip
        )

        validate_players_output(
            round_trip
        )

        pd.testing.assert_frame_equal(
            frame,
            round_trip,
            check_dtype=True,
            check_like=False,
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
        raise PlayersBuildError(
            "Player Metadata Parquet 저장/round-trip "
            f"검증에 실패했습니다: {output_path}"
        ) from exc
    finally:
        if (
            temporary_path is not None
            and temporary_path.exists()
        ):
            temporary_path.unlink()


def _count_json_validation_errors(
    frame: pd.DataFrame,
) -> int:
    """Production Logging용 name_variants JSON 오류 수를 계산한다."""
    error_count = 0

    for row in frame.itertuples(
        index=False
    ):
        try:
            _parse_name_variants(
                row.name_variants,
                player_id=str(
                    row.player_id
                ),
            )
        except PlayersBuildError:
            error_count += 1

    return error_count


def log_production_validation(
    *,
    observations: pd.DataFrame,
    output: pd.DataFrame,
) -> None:
    """Issue #12의 Production Validation 지표를 logging으로 기록한다."""
    pa_batter_ids = set(
        observations.loc[
            observations["source"].eq(
                SOURCE_PA
            )
            & observations["role"].eq(
                ROLE_BATTER
            ),
            "player_id",
        ]
        .astype(str)
        .tolist()
    )

    pa_pitcher_ids = set(
        observations.loc[
            observations["source"].eq(
                SOURCE_PA
            )
            & observations["role"].eq(
                ROLE_PITCHER
            ),
            "player_id",
        ]
        .astype(str)
        .tolist()
    )

    batting_ids = set(
        observations.loc[
            observations["source"].eq(
                SOURCE_BATTING
            ),
            "player_id",
        ]
        .astype(str)
        .tolist()
    )

    pitching_ids = set(
        observations.loc[
            observations["source"].eq(
                SOURCE_PITCHING
            ),
            "player_id",
        ]
        .astype(str)
        .tolist()
    )

    union_ids = set(
        observations["player_id"]
        .astype(str)
        .tolist()
    )

    output_ids = set(
        output["player_id"]
        .astype(str)
        .tolist()
    )

    missing_batting = (
        batting_ids
        - output_ids
    )
    missing_pitching = (
        pitching_ids
        - output_ids
    )

    batter_only = int(
        (
            output["is_batter"]
            & ~output["is_pitcher"]
        ).sum()
    )
    pitcher_only = int(
        (
            ~output["is_batter"]
            & output["is_pitcher"]
        ).sum()
    )
    both_roles = int(
        (
            output["is_batter"]
            & output["is_pitcher"]
        ).sum()
    )

    variant_lengths = (
        output["name_variants"]
        .map(
            lambda value: len(
                json.loads(
                    str(value)
                )
            )
        )
        .astype("Int64")
    )

    single_variant_count = int(
        variant_lengths.eq(1).sum()
    )
    multi_variant_count = int(
        variant_lengths.ge(2).sum()
    )

    multi_examples = (
        output.loc[
            variant_lengths.ge(2),
            [
                "player_id",
                "display_name",
                "name_variants",
            ],
        ]
        .head(5)
        .to_dict("records")
    )

    prohibited_count = sum(
        int(
            column in output.columns
        )
        for column in PROHIBITED_OUTPUT_COLUMNS
    )

    LOGGER.info(
        "Production Validation | 전체 Player Count=%d | "
        "player_id Duplicate Count=%d",
        len(output),
        int(
            output.duplicated(
                subset=["player_id"]
            ).sum()
        ),
    )

    LOGGER.info(
        "Production Validation | "
        "PA Batter unique ID Count=%d | "
        "PA Pitcher unique ID Count=%d | "
        "Player Game Batting unique ID Count=%d | "
        "Player Game Pitching unique ID Count=%d",
        len(pa_batter_ids),
        len(pa_pitcher_ids),
        len(batting_ids),
        len(pitching_ids),
    )

    LOGGER.info(
        "Production Validation | "
        "Union unique Player ID Count=%d | "
        "Output Player ID Count=%d",
        len(union_ids),
        len(output_ids),
    )

    LOGGER.info(
        "Production Validation | "
        "Missing Batting Player Coverage Count=%d | "
        "Missing Pitching Player Coverage Count=%d",
        len(missing_batting),
        len(missing_pitching),
    )

    LOGGER.info(
        "Production Validation | "
        "Batter-only Count=%d | "
        "Pitcher-only Count=%d | "
        "Both-role Count=%d",
        batter_only,
        pitcher_only,
        both_roles,
    )

    LOGGER.info(
        "Production Validation | "
        "Display Name Null Player Count=%d | "
        "Single Name Variant Player Count=%d | "
        "Multi Name Variant Player Count=%d",
        int(
            output["display_name"]
            .isna()
            .sum()
        ),
        single_variant_count,
        multi_variant_count,
    )

    LOGGER.info(
        "Production Validation | "
        "Multi Name Variant 대표 예시=%s",
        multi_examples,
    )

    LOGGER.info(
        "Production Validation | "
        "Earliest first_seen_date=%s | "
        "Latest last_seen_date=%s",
        output["first_seen_date"].min(),
        output["last_seen_date"].max(),
    )

    LOGGER.info(
        "Production Validation | "
        "name_variants JSON Validation Error Count=%d | "
        "Prohibited Team Column Count=%d",
        _count_json_validation_errors(
            output
        ),
        prohibited_count,
    )


def build_players_file(
    *,
    pa_path: Path = DEFAULT_PA_PATH,
    batting_path: Path = DEFAULT_BATTING_PATH,
    pitching_path: Path = DEFAULT_PITCHING_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """세 Canonical Parquet을 읽어 players.parquet을 안전하게 생성한다."""
    ensure_output_path_safe(
        pa_path=pa_path,
        batting_path=batting_path,
        pitching_path=pitching_path,
        output_path=output_path,
    )

    input_paths = {
        "Plate Appearance": pa_path,
        "Player Game Batting": batting_path,
        "Player Game Pitching": pitching_path,
    }

    for source_name, path in input_paths.items():
        if not path.is_file():
            raise PlayersBuildError(
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

    plate_appearances = _read_parquet(
        pa_path,
        source_name="Plate Appearance",
    )
    player_game_batting = _read_parquet(
        batting_path,
        source_name="Player Game Batting",
    )
    player_game_pitching = _read_parquet(
        pitching_path,
        source_name="Player Game Pitching",
    )

    observations = build_player_observations(
        plate_appearances,
        player_game_batting,
        player_game_pitching,
    )

    deduplicated = deduplicate_observations(
        observations
    )

    output = _summarize_players(
        deduplicated
    )
    output = normalize_players_dtypes(
        output
    )

    validate_players_output(
        output,
        observations=observations,
    )

    log_production_validation(
        observations=observations,
        output=output,
    )

    _write_parquet_atomically(
        output,
        output_path,
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
            raise PlayersBuildError(
                f"{source_name} 입력 파일이 변경되었습니다: {path}"
            )

    LOGGER.info(
        "Player Metadata 생성 완료 | "
        "output=%s | rows=%d | duplicate_player_id=%d",
        output_path,
        len(output),
        int(
            output.duplicated(
                subset=["player_id"]
            ).sum()
        ),
    )

    return output


def main() -> None:
    """CLI 진입점."""
    configure_logging()
    args = parse_args()

    try:
        build_players_file(
            pa_path=args.pa_path,
            batting_path=args.batting_path,
            pitching_path=args.pitching_path,
            output_path=args.output_path,
        )
    except PlayersBuildError as exc:
        LOGGER.error(
            "%s",
            exc,
        )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()