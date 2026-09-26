from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from scripts import build_game_tables as game_builder  # noqa: E402
from scripts import build_plate_appearances as pa_builder  # noqa: E402
from scripts import build_player_game_batting as batting_builder  # noqa: E402
from scripts import build_player_game_pitching as pitching_builder  # noqa: E402
from scripts import build_players as players_builder  # noqa: E402
from scripts import build_season_snapshots as season_builder  # noqa: E402


LOGGER = logging.getLogger(__name__)

DEFAULT_SEASONS = (
    2023,
    2024,
    2025,
    2026,
)

DEFAULT_RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "hf_kbo_pbp"
)

DEFAULT_DERIVED_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "hf_kbo_pbp"
    / "derived"
)

RAW_REQUIRED_COLUMNS = (
    "game_pk",
    "game_date",
    "at_bat_number",
    "pitch_number",
    "pitcher",
    "pitcher_name",
    "type",
    "events",
    "home_team",
    "away_team",
)

RAW_REQUIRED_NON_NULL_COLUMNS = (
    "game_pk",
    "game_date",
    "at_bat_number",
    "pitch_number",
    "pitcher",
    "home_team",
    "away_team",
)

RAW_STRING_COLUMNS = (
    "game_pk",
    "pitcher",
    "pitcher_name",
    "type",
    "events",
    "home_team",
    "away_team",
)

RAW_INTEGER_COLUMNS = (
    "at_bat_number",
    "pitch_number",
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

PITCHER_GAME_KEY = (
    "game_pk",
    "pitcher",
)

CROSS_ERROR_CATEGORIES = (
    "raw_pa_key",
    "raw_game_key",
    "raw_context",
    "game_team_game",
    "score_last_pa",
    "score_runs",
    "batting_pa",
    "batting_event",
    "season_batting",
    "raw_pitch",
    "pitch_type",
    "bf_allowed_event",
    "pitching_outs",
    "season_pitching",
    "outs_null_propagation",
    "player_union",
    "player_role",
    "team_season",
    "through_date",
    "partial_2026",
    "cross_table_context",
)


class DerivedValidationError(RuntimeError):
    """Canonical Derived 통합 검증을 실패시켜야 하는 오류를 나타낸다."""


@dataclass(frozen=True)
class TableContract:
    """한 Canonical Derived Table의 Schema와 Grain 계약을 보관한다."""

    name: str
    filename: str
    key: tuple[str, ...]
    columns: tuple[str, ...]
    expected_dtypes: Mapping[str, str]


@dataclass(frozen=True)
class RawScanSummary:
    """시즌별 Raw를 축약하여 통합 검증에 필요한 정보만 보관한다."""

    row_count: int
    actual_pitch_count: int
    pa_summary: pd.DataFrame
    game_context: pd.DataFrame
    pitcher_summary: pd.DataFrame
    multi_pitcher_participants: pd.DataFrame


@dataclass(frozen=True)
class DerivedValidationSummary:
    """성공한 통합 검증의 결정적 Summary를 보관한다."""

    raw_rows: int
    raw_actual_pitches: int
    raw_unique_pa: int
    raw_unique_games: int
    table_row_counts: dict[str, int]
    duplicate_key_counts: dict[str, int]
    schema_error_counts: dict[str, int]
    cross_table_error_counts: dict[str, int]
    through_dates: dict[int, str]
    content_fingerprints: dict[str, str]
    raw_sha256_before: dict[str, str | None]
    raw_sha256_after: dict[str, str | None]
    derived_sha256_before: dict[str, str | None]
    derived_sha256_after: dict[str, str | None]


def parse_args() -> argparse.Namespace:
    """통합 검증 CLI 인자를 파싱한다."""
    parser = argparse.ArgumentParser(
        description=(
            "KBO Canonical Derived Layer 9개 Table을 "
            "Raw 및 상호 계약 기준으로 Read-only 검증합니다."
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
        "--derived-dir",
        type=Path,
        default=DEFAULT_DERIVED_DIR,
        help=(
            "Canonical Derived Parquet 디렉터리입니다. "
            f"기본값: {DEFAULT_DERIVED_DIR}"
        ),
    )

    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEASONS),
        help="검증할 시즌입니다. 기본값: 2023 2024 2025 2026",
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
    """지원 시즌을 중복 제거 후 오름차순으로 정규화한다."""
    seasons = tuple(
        sorted(
            {
                int(value)
                for value in values
            }
        )
    )

    if not seasons:
        raise DerivedValidationError(
            "최소 1개 이상의 시즌을 지정해야 합니다."
        )

    unsupported = sorted(
        set(seasons)
        - set(DEFAULT_SEASONS)
    )

    if unsupported:
        raise DerivedValidationError(
            "지원하지 않는 시즌입니다: "
            f"{unsupported}. "
            f"지원 시즌={list(DEFAULT_SEASONS)}"
        )

    return seasons


def _build_dtype_map(
    columns: Sequence[str],
    *,
    string_columns: Iterable[str] = (),
    integer_columns: Iterable[str] = (),
    boolean_columns: Iterable[str] = (),
    float_columns: Iterable[str] = (),
    datetime_columns: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Builder 상수 기준으로 Output dtype 계약을 구성한다."""
    result: dict[str, str] = {}

    groups = (
        (string_columns, "string"),
        (integer_columns, "Int64"),
        (boolean_columns, "boolean"),
        (float_columns, "Float64"),
    )

    for group_columns, dtype in groups:
        for column in group_columns:
            if column in result:
                raise RuntimeError(
                    f"dtype 계약이 중복 정의되었습니다: {column}"
                )

            result[column] = dtype

    for column, dtype in (
        datetime_columns
        or {}
    ).items():
        if column in result:
            raise RuntimeError(
                f"dtype 계약이 중복 정의되었습니다: {column}"
            )

        result[column] = dtype

    missing = [
        column
        for column in columns
        if column not in result
    ]

    if missing:
        raise RuntimeError(
            "dtype 계약이 정의되지 않은 Output Column이 있습니다: "
            f"{missing}"
        )

    return result


def build_table_contracts() -> tuple[TableContract, ...]:
    """
    현재 Builder의 Output 상수와 dtype 정책으로
    아홉 Canonical Table 계약을 구성한다.
    """
    return (
        TableContract(
            name="plate_appearances",
            filename=pa_builder.DEFAULT_OUTPUT_PATH.name,
            key=tuple(pa_builder.PA_KEY),
            columns=tuple(pa_builder.OUTPUT_COLUMNS),
            expected_dtypes=_build_dtype_map(
                pa_builder.OUTPUT_COLUMNS,
                string_columns=pa_builder.STRING_OUTPUT_COLUMNS,
                integer_columns=pa_builder.INTEGER_OUTPUT_COLUMNS,
                boolean_columns=pa_builder.BOOLEAN_OUTPUT_COLUMNS,
                datetime_columns={
                    "game_date": "datetime64[us]",
                },
            ),
        ),
        TableContract(
            name="games",
            filename=game_builder.GAMES_FILENAME,
            key=tuple(game_builder.GAME_KEY),
            columns=tuple(game_builder.GAMES_COLUMNS),
            expected_dtypes=_build_dtype_map(
                game_builder.GAMES_COLUMNS,
                string_columns=game_builder.GAMES_STRING_COLUMNS,
                integer_columns=game_builder.GAMES_INTEGER_COLUMNS,
                boolean_columns=game_builder.GAMES_BOOLEAN_COLUMNS,
                datetime_columns={
                    "game_date": "datetime64[us]",
                },
            ),
        ),
        TableContract(
            name="team_games",
            filename=game_builder.TEAM_GAMES_FILENAME,
            key=tuple(game_builder.TEAM_GAME_KEY),
            columns=tuple(game_builder.TEAM_GAMES_COLUMNS),
            expected_dtypes=_build_dtype_map(
                game_builder.TEAM_GAMES_COLUMNS,
                string_columns=game_builder.TEAM_GAMES_STRING_COLUMNS,
                integer_columns=game_builder.TEAM_GAMES_INTEGER_COLUMNS,
                boolean_columns=game_builder.TEAM_GAMES_BOOLEAN_COLUMNS,
                datetime_columns={
                    "game_date": "datetime64[us]",
                },
            ),
        ),
        TableContract(
            name="player_game_batting",
            filename=batting_builder.DEFAULT_OUTPUT_PATH.name,
            key=tuple(batting_builder.PLAYER_GAME_KEY),
            columns=tuple(batting_builder.OUTPUT_COLUMNS),
            expected_dtypes=_build_dtype_map(
                batting_builder.OUTPUT_COLUMNS,
                string_columns=batting_builder.STRING_OUTPUT_COLUMNS,
                integer_columns=batting_builder.INTEGER_OUTPUT_COLUMNS,
                boolean_columns=batting_builder.BOOLEAN_OUTPUT_COLUMNS,
                float_columns=batting_builder.FLOAT_OUTPUT_COLUMNS,
                datetime_columns={
                    "game_date": "datetime64[us]",
                },
            ),
        ),
        TableContract(
            name="player_game_pitching",
            filename=pitching_builder.DEFAULT_OUTPUT_PATH.name,
            key=tuple(pitching_builder.PLAYER_GAME_KEY),
            columns=tuple(pitching_builder.OUTPUT_COLUMNS),
            expected_dtypes=_build_dtype_map(
                pitching_builder.OUTPUT_COLUMNS,
                string_columns=pitching_builder.STRING_OUTPUT_COLUMNS,
                integer_columns=pitching_builder.INTEGER_OUTPUT_COLUMNS,
                boolean_columns=pitching_builder.BOOLEAN_OUTPUT_COLUMNS,
                float_columns=pitching_builder.FLOAT_OUTPUT_COLUMNS,
                datetime_columns={
                    "game_date": "datetime64[us]",
                },
            ),
        ),
        TableContract(
            name="players",
            filename=players_builder.DEFAULT_OUTPUT_PATH.name,
            key=("player_id",),
            columns=tuple(players_builder.OUTPUT_COLUMNS),
            expected_dtypes=_build_dtype_map(
                players_builder.OUTPUT_COLUMNS,
                string_columns=players_builder.STRING_OUTPUT_COLUMNS,
                integer_columns=players_builder.INTEGER_OUTPUT_COLUMNS,
                boolean_columns=players_builder.BOOLEAN_OUTPUT_COLUMNS,
                datetime_columns={
                    "first_seen_date": "datetime64[us]",
                    "last_seen_date": "datetime64[us]",
                },
            ),
        ),
        TableContract(
            name="player_season_batting_snapshot",
            filename=season_builder.BATTING_SNAPSHOT_FILENAME,
            key=tuple(season_builder.BATTING_SNAPSHOT_KEY),
            columns=tuple(season_builder.BATTING_OUTPUT_COLUMNS),
            expected_dtypes=_build_dtype_map(
                season_builder.BATTING_OUTPUT_COLUMNS,
                string_columns=("batter",),
                integer_columns=(
                    season_builder.BATTING_INTEGER_OUTPUT_COLUMNS
                ),
                float_columns=(
                    season_builder.BATTING_FLOAT_OUTPUT_COLUMNS
                ),
                datetime_columns={
                    "through_date": "datetime64[us]",
                },
            ),
        ),
        TableContract(
            name="player_season_pitching_snapshot",
            filename=season_builder.PITCHING_SNAPSHOT_FILENAME,
            key=tuple(season_builder.PITCHING_SNAPSHOT_KEY),
            columns=tuple(season_builder.PITCHING_OUTPUT_COLUMNS),
            expected_dtypes=_build_dtype_map(
                season_builder.PITCHING_OUTPUT_COLUMNS,
                string_columns=("pitcher",),
                integer_columns=(
                    season_builder.PITCHING_INTEGER_OUTPUT_COLUMNS
                ),
                datetime_columns={
                    "through_date": "datetime64[us]",
                },
            ),
        ),
        TableContract(
            name="team_season_snapshot",
            filename=season_builder.TEAM_SNAPSHOT_FILENAME,
            key=tuple(season_builder.TEAM_SNAPSHOT_KEY),
            columns=tuple(season_builder.TEAM_OUTPUT_COLUMNS),
            expected_dtypes=_build_dtype_map(
                season_builder.TEAM_OUTPUT_COLUMNS,
                string_columns=("team",),
                integer_columns=(
                    season_builder.TEAM_INTEGER_OUTPUT_COLUMNS
                ),
                datetime_columns={
                    "through_date": "datetime64[us]",
                },
            ),
        ),
    )


TABLE_CONTRACTS = build_table_contracts()


def _validation_error(
    category: str,
    message: str,
    *,
    count: int = 1,
    examples: object | None = None,
) -> DerivedValidationError:
    """오류 Category, Count, 예시를 일관된 메시지로 변환한다."""
    detail = (
        f"{message} | "
        f"category={category} | "
        f"count={count}"
    )

    if examples is not None:
        detail += (
            " | "
            f"examples={examples}"
        )

    return DerivedValidationError(
        detail
    )


def calculate_sha256(
    path: Path,
) -> str:
    """파일 불변성 확인을 위해 SHA256을 계산한다."""
    digest = hashlib.sha256()

    try:
        with path.open(
            "rb"
        ) as file:
            for chunk in iter(
                lambda: file.read(
                    1024 * 1024
                ),
                b"",
            ):
                digest.update(
                    chunk
                )
    except OSError as exc:
        raise DerivedValidationError(
            f"SHA256 계산에 실패했습니다: {path}"
        ) from exc

    return digest.hexdigest()


def _capture_named_sha256(
    paths: Mapping[str, Path],
) -> dict[str, str | None]:
    """기대 파일의 존재 상태와 SHA256을 이름 기준으로 기록한다."""
    result: dict[
        str,
        str | None,
    ] = {}

    for name, path in paths.items():
        if path.is_file():
            result[name] = (
                calculate_sha256(
                    path
                )
            )

        elif path.exists():
            raise DerivedValidationError(
                "기대 파일 경로가 일반 파일이 아닙니다: "
                f"{path}"
            )

        else:
            result[name] = None

    return result


def _validate_snapshot_unchanged(
    before: Mapping[str, str | None],
    after: Mapping[str, str | None],
    *,
    label: str,
) -> None:
    """Validator 전후 File Snapshot이 완전히 같은지 확인한다."""
    if dict(before) == dict(after):
        return

    changed = {
        name: {
            "before": before.get(name),
            "after": after.get(name),
        }
        for name in sorted(
            set(before)
            | set(after)
        )
        if before.get(name)
        != after.get(name)
    }

    raise _validation_error(
        "file_immutability",
        f"{label} 파일이 Validator 실행 중 변경되었습니다.",
        count=len(changed),
        examples=changed,
    )


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
        resolved_path
        == resolved_parent
        or resolved_parent
        in resolved_path.parents
    )


def validate_derived_path_policy(
    derived_dir: Path,
    *,
    project_root: Path = PROJECT_ROOT,
) -> None:
    """
    Canonical Derived Output 위치와
    현재 .gitignore의 data/interim 정책을 확인한다.
    """
    interim_root = (
        project_root
        / "data"
        / "interim"
    )

    if not _path_is_within(
        derived_dir,
        interim_root,
    ):
        raise _validation_error(
            "derived_path",
            "Derived 디렉터리가 data/interim 아래에 있지 않습니다.",
            examples={
                "derived_dir": str(
                    derived_dir
                ),
                "interim_root": str(
                    interim_root
                ),
            },
        )

    for contract in TABLE_CONTRACTS:
        path = (
            derived_dir
            / contract.filename
        )

        if not _path_is_within(
            path,
            interim_root,
        ):
            raise _validation_error(
                "derived_path",
                "Canonical Derived 파일이 data/interim 밖에 있습니다.",
                examples=str(path),
            )

    gitignore_path = (
        project_root
        / ".gitignore"
    )

    if not gitignore_path.is_file():
        raise _validation_error(
            "gitignore",
            ".gitignore 파일이 없습니다.",
            examples=str(
                gitignore_path
            ),
        )

    try:
        lines = {
            line.strip()
            for line in gitignore_path.read_text(
                encoding="utf-8",
            ).splitlines()
            if (
                line.strip()
                and not line.lstrip().startswith("#")
            )
        }
    except OSError as exc:
        raise DerivedValidationError(
            f".gitignore 읽기에 실패했습니다: {gitignore_path}"
        ) from exc

    if "/data/interim/**" not in lines:
        raise _validation_error(
            "gitignore",
            "현재 프로젝트의 /data/interim/** Ignore 정책이 없습니다.",
            examples=str(
                gitignore_path
            ),
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
        raise _validation_error(
            "schema",
            f"{source_name}에 필수 컬럼이 없습니다.",
            count=len(missing),
            examples=missing,
        )


def _coerce_raw_integer(
    series: pd.Series,
    *,
    column: str,
) -> pd.Series:
    """Raw 정수 의미 컬럼을 손실 없이 nullable Int64로 정규화한다."""
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    invalid_numeric = (
        series.notna()
        & numeric.isna()
    )

    if invalid_numeric.any():
        examples = (
            series.loc[
                invalid_numeric
            ]
            .head(5)
            .tolist()
        )

        raise _validation_error(
            "raw_schema",
            f"Raw.{column}을 숫자로 변환할 수 없습니다.",
            count=int(
                invalid_numeric.sum()
            ),
            examples=examples,
        )

    non_integer = (
        numeric.notna()
        & numeric.mod(1).ne(0)
    )

    if non_integer.any():
        examples = (
            numeric.loc[
                non_integer
            ]
            .head(5)
            .tolist()
        )

        raise _validation_error(
            "raw_schema",
            f"Raw.{column}에 정수가 아닌 값이 있습니다.",
            count=int(
                non_integer.sum()
            ),
            examples=examples,
        )

    result = numeric.astype(
        "Int64"
    )

    if result.isna().any():
        raise _validation_error(
            "raw_schema",
            f"Raw.{column}에는 null을 허용하지 않습니다.",
            count=int(
                result.isna().sum()
            ),
        )

    return result


def _normalize_raw_frame(
    frame: pd.DataFrame,
    *,
    season: int,
) -> pd.DataFrame:
    """한 시즌 Raw를 통합 검증에 필요한 최소 형태로 정규화한다."""
    _require_columns(
        frame,
        RAW_REQUIRED_COLUMNS,
        source_name=f"Raw {season}",
    )

    working = frame.loc[
        :,
        RAW_REQUIRED_COLUMNS,
    ].copy()

    for column in RAW_REQUIRED_NON_NULL_COLUMNS:
        null_count = int(
            working[column]
            .isna()
            .sum()
        )

        if null_count > 0:
            raise _validation_error(
                "raw_schema",
                f"{season} Raw.{column}에 null이 있습니다.",
                count=null_count,
            )

    for column in RAW_STRING_COLUMNS:
        working[column] = (
            working[column]
            .astype("string")
        )

    for column in RAW_INTEGER_COLUMNS:
        working[column] = (
            _coerce_raw_integer(
                working[column],
                column=column,
            )
        )

    parsed_date = pd.to_datetime(
        working["game_date"],
        errors="coerce",
        utc=True,
    )

    invalid_date = (
        parsed_date.isna()
    )

    if invalid_date.any():
        raise _validation_error(
            "raw_context",
            f"{season} Raw.game_date 변환 실패가 있습니다.",
            count=int(
                invalid_date.sum()
            ),
            examples=(
                working.loc[
                    invalid_date,
                    "game_date",
                ]
                .head(5)
                .tolist()
            ),
        )

    working["game_date"] = (
        parsed_date
        .dt
        .tz_convert(None)
        .dt
        .normalize()
        .astype("datetime64[us]")
    )

    season_mismatch = (
        working["game_date"]
        .dt
        .year
        .ne(season)
    )

    if season_mismatch.any():
        raise _validation_error(
            "raw_context",
            f"{season} Raw에 game_date 기준 시즌 불일치가 있습니다.",
            count=int(
                season_mismatch.sum()
            ),
            examples=(
                working.loc[
                    season_mismatch,
                    [
                        "game_pk",
                        "game_date",
                    ],
                ]
                .head(5)
                .to_dict("records")
            ),
        )

    if working["at_bat_number"].le(0).any():
        raise _validation_error(
            "raw_schema",
            "Raw.at_bat_number는 1 이상이어야 합니다.",
            count=int(
                working[
                    "at_bat_number"
                ]
                .le(0)
                .sum()
            ),
        )

    if working["pitch_number"].lt(0).any():
        raise _validation_error(
            "raw_schema",
            "Raw.pitch_number는 0 이상이어야 합니다.",
            count=int(
                working[
                    "pitch_number"
                ]
                .lt(0)
                .sum()
            ),
        )

    duplicate_pitch = (
        working.duplicated(
            subset=list(
                PITCH_KEY
            ),
            keep=False,
        )
    )

    if duplicate_pitch.any():
        raise _validation_error(
            "raw_grain",
            "Raw Pitch Key가 중복되었습니다.",
            count=int(
                duplicate_pitch.sum()
            ),
            examples=(
                working.loc[
                    duplicate_pitch,
                    list(PITCH_KEY),
                ]
                .head(10)
                .to_dict("records")
            ),
        )

    actual_pitch = (
        working["pitch_number"]
        .gt(0)
    )

    invalid_pitch_type = (
        actual_pitch
        & ~working["type"].isin(
            pitching_builder.PITCH_TYPE_VALUES
        )
    )

    if invalid_pitch_type.any():
        raise _validation_error(
            "pitch_type",
            "실제 Pitch의 Raw.type은 B/S/X 중 하나여야 합니다.",
            count=int(
                invalid_pitch_type.sum()
            ),
            examples=(
                working.loc[
                    invalid_pitch_type,
                    [
                        "game_pk",
                        "at_bat_number",
                        "pitch_number",
                        "type",
                    ],
                ]
                .head(10)
                .to_dict("records")
            ),
        )

    return working


def scan_raw(
    raw_dir: Path,
    seasons: Sequence[int],
) -> RawScanSummary:
    """
    Raw를 시즌별 최소 Column로 읽고
    PA/Game/Pitcher 단위 Summary만 축적한다.
    """
    row_count = 0
    actual_pitch_count = 0

    pa_summaries: list[
        pd.DataFrame
    ] = []
    game_contexts: list[
        pd.DataFrame
    ] = []
    pitcher_summaries: list[
        pd.DataFrame
    ] = []
    multi_pitcher_participants: list[
        pd.DataFrame
    ] = []

    for season in seasons:
        path = (
            raw_dir
            / f"{season}.parquet"
        )

        if not path.is_file():
            raise _validation_error(
                "raw_file",
                f"{season} Raw Parquet이 없습니다.",
                examples=str(path),
            )

        try:
            raw = pd.read_parquet(
                path,
                columns=list(
                    RAW_REQUIRED_COLUMNS
                ),
            )
        except Exception as exc:
            raise DerivedValidationError(
                f"Raw Parquet 읽기에 실패했습니다: {path}"
            ) from exc

        raw = _normalize_raw_frame(
            raw,
            season=season,
        )

        row_count += len(
            raw
        )

        actual_pitch = (
            raw["pitch_number"]
            .gt(0)
        )

        actual_pitch_count += int(
            actual_pitch.sum()
        )

        ordered = (
            raw
            .sort_values(
                by=list(
                    PITCH_KEY
                ),
                kind="mergesort",
            )
            .reset_index(
                drop=True
            )
        )

        pa_work = (
            ordered
            .assign(
                _actual_pitch=(
                    ordered[
                        "pitch_number"
                    ]
                    .gt(0)
                    .astype("Int64")
                )
            )
        )

        grouped_pa = (
            pa_work
            .groupby(
                list(
                    PA_KEY
                ),
                sort=False,
                dropna=False,
            )
        )

        pa_summary = (
            grouped_pa
            .agg(
                pitch_rows=(
                    "pitch_number",
                    "size",
                ),
                pitch_count=(
                    "_actual_pitch",
                    "sum",
                ),
                unique_pitchers=(
                    "pitcher",
                    "nunique",
                ),
            )
            .reset_index()
        )

        last_pitcher = (
            ordered
            .drop_duplicates(
                subset=list(
                    PA_KEY
                ),
                keep="last",
            )
            .loc[
                :,
                [
                    *PA_KEY,
                    "pitcher",
                ],
            ]
            .rename(
                columns={
                    "pitcher": (
                        "last_raw_pitcher"
                    ),
                }
            )
        )

        pa_summary = (
            pa_summary
            .merge(
                last_pitcher,
                on=list(
                    PA_KEY
                ),
                how="left",
                validate="one_to_one",
                sort=False,
            )
        )

        for column in (
            "pitch_rows",
            "pitch_count",
            "unique_pitchers",
        ):
            pa_summary[column] = (
                pa_summary[column]
                .astype("Int64")
            )

        pa_summaries.append(
            pa_summary
        )

        multi_keys = (
            pa_summary.loc[
                pa_summary[
                    "unique_pitchers"
                ].gt(1),
                list(
                    PA_KEY
                ),
            ]
        )

        if not multi_keys.empty:
            participants = (
                ordered.merge(
                    multi_keys,
                    on=list(
                        PA_KEY
                    ),
                    how="inner",
                    validate="many_to_one",
                    sort=False,
                )
                .loc[
                    :,
                    [
                        *PA_KEY,
                        "pitcher",
                    ],
                ]
                .drop_duplicates()
            )

            multi_pitcher_participants.append(
                participants
            )

        context = (
            ordered.loc[
                :,
                [
                    "game_pk",
                    "game_date",
                    "home_team",
                    "away_team",
                ],
            ]
            .drop_duplicates()
        )

        context_count = (
            context.groupby(
                "game_pk",
                sort=False,
                dropna=False,
            )
            .size()
        )

        invalid_context = (
            context_count.gt(1)
        )

        if invalid_context.any():
            invalid_games = (
                invalid_context.loc[
                    invalid_context
                ]
                .index
                .tolist()
            )

            raise _validation_error(
                "raw_context",
                "동일 Raw game_pk의 날짜/Home/Away Context가 충돌합니다.",
                count=len(
                    invalid_games
                ),
                examples=(
                    invalid_games[:10]
                ),
            )

        context["season"] = (
            pd.Series(
                season,
                index=context.index,
                dtype="Int64",
            )
        )

        game_contexts.append(
            context.loc[
                :,
                [
                    "game_pk",
                    "game_date",
                    "season",
                    "home_team",
                    "away_team",
                ],
            ]
        )

        pitcher_work = ordered.copy()

        is_pitch = (
            pitcher_work[
                "pitch_number"
            ]
            .gt(0)
            .astype("boolean")
        )

        pitch_type = (
            pitcher_work[
                "type"
            ]
            .astype("string")
        )

        pitcher_work[
            "_is_pitch"
        ] = is_pitch

        pitcher_work[
            "_ball"
        ] = (
            is_pitch
            & pitch_type
            .eq("B")
            .fillna(False)
            .astype("boolean")
        )

        pitcher_work[
            "_strike"
        ] = (
            is_pitch
            & pitch_type
            .eq("S")
            .fillna(False)
            .astype("boolean")
        )

        pitcher_work[
            "_in_play"
        ] = (
            is_pitch
            & pitch_type
            .eq("X")
            .fillna(False)
            .astype("boolean")
        )

        for column in (
            "_is_pitch",
            "_ball",
            "_strike",
            "_in_play",
        ):
            pitcher_work[column] = (
                pitcher_work[column]
                .astype("Int64")
            )

        pitcher_summary = (
            pitcher_work
            .groupby(
                list(
                    PITCHER_GAME_KEY
                ),
                sort=False,
                dropna=False,
            )
            .agg(
                pitch_rows=(
                    "pitch_number",
                    "size",
                ),
                pitches=(
                    "_is_pitch",
                    "sum",
                ),
                ball_pitch_count=(
                    "_ball",
                    "sum",
                ),
                strike_pitch_count=(
                    "_strike",
                    "sum",
                ),
                in_play_pitch_count=(
                    "_in_play",
                    "sum",
                ),
            )
            .reset_index()
        )

        for column in (
            "pitch_rows",
            "pitches",
            "ball_pitch_count",
            "strike_pitch_count",
            "in_play_pitch_count",
        ):
            pitcher_summary[column] = (
                pitcher_summary[column]
                .astype("Int64")
            )

        pitcher_summaries.append(
            pitcher_summary
        )

    pa_summary_all = pd.concat(
        pa_summaries,
        ignore_index=True,
    )

    game_context_all = pd.concat(
        game_contexts,
        ignore_index=True,
    )

    pitcher_summary_all = (
        pd.concat(
            pitcher_summaries,
            ignore_index=True,
        )
    )

    if multi_pitcher_participants:
        participants_all = pd.concat(
            multi_pitcher_participants,
            ignore_index=True,
        )
    else:
        participants_all = pd.DataFrame(
            {
                "game_pk": pd.Series(
                    dtype="string"
                ),
                "at_bat_number": pd.Series(
                    dtype="Int64"
                ),
                "pitcher": pd.Series(
                    dtype="string"
                ),
            }
        )

    duplicate_pa = (
        pa_summary_all.duplicated(
            subset=list(
                PA_KEY
            ),
            keep=False,
        )
    )

    if duplicate_pa.any():
        raise _validation_error(
            "raw_grain",
            "서로 다른 시즌 Raw 사이에서 PA Key가 중복되었습니다.",
            count=int(
                duplicate_pa.sum()
            ),
            examples=(
                pa_summary_all.loc[
                    duplicate_pa,
                    list(PA_KEY),
                ]
                .head(10)
                .to_dict("records")
            ),
        )

    duplicate_game = (
        game_context_all.duplicated(
            subset=[
                "game_pk"
            ],
            keep=False,
        )
    )

    if duplicate_game.any():
        raise _validation_error(
            "raw_grain",
            "서로 다른 시즌 Raw 사이에서 game_pk가 중복되었습니다.",
            count=int(
                duplicate_game.sum()
            ),
            examples=(
                game_context_all.loc[
                    duplicate_game,
                    [
                        "game_pk",
                        "season",
                    ],
                ]
                .head(10)
                .to_dict("records")
            ),
        )

    return RawScanSummary(
        row_count=row_count,
        actual_pitch_count=actual_pitch_count,
        pa_summary=pa_summary_all,
        game_context=game_context_all,
        pitcher_summary=pitcher_summary_all,
        multi_pitcher_participants=participants_all,
    )


def _load_derived_tables(
    derived_dir: Path,
) -> dict[str, pd.DataFrame]:
    """아홉 Canonical Derived Parquet을 읽는다."""
    tables: dict[
        str,
        pd.DataFrame,
    ] = {}

    missing_files = [
        contract.filename
        for contract in TABLE_CONTRACTS
        if not (
            derived_dir
            / contract.filename
        ).is_file()
    ]

    if missing_files:
        raise _validation_error(
            "derived_file",
            "Canonical Derived 파일이 누락되었습니다.",
            count=len(
                missing_files
            ),
            examples=missing_files,
        )

    for contract in TABLE_CONTRACTS:
        path = (
            derived_dir
            / contract.filename
        )

        try:
            tables[
                contract.name
            ] = pd.read_parquet(
                path
            )
        except Exception as exc:
            raise DerivedValidationError(
                f"Derived Parquet 읽기에 실패했습니다: {path}"
            ) from exc

    return tables


def validate_schema_and_grain(
    tables: Mapping[str, pd.DataFrame],
) -> dict[str, int]:
    """Column 순서, dtype, Grain Key null/중복을 검증한다."""
    duplicate_counts: dict[
        str,
        int,
    ] = {}

    for contract in TABLE_CONTRACTS:
        frame = tables[
            contract.name
        ]

        actual_columns = tuple(
            str(column)
            for column in frame.columns
        )

        if actual_columns != contract.columns:
            missing = [
                column
                for column in contract.columns
                if column
                not in actual_columns
            ]

            unexpected = [
                column
                for column in actual_columns
                if column
                not in contract.columns
            ]

            raise _validation_error(
                "schema",
                f"{contract.name} Column 순서/구성이 Builder 계약과 다릅니다.",
                examples={
                    "missing": missing,
                    "unexpected": unexpected,
                    "expected": list(
                        contract.columns
                    ),
                    "actual": list(
                        actual_columns
                    ),
                },
            )

        dtype_mismatches: list[
            dict[str, str]
        ] = []

        for column in contract.columns:
            actual_dtype = str(
                frame[column].dtype
            )

            expected_dtype = (
                contract
                .expected_dtypes[
                    column
                ]
            )

            if (
                actual_dtype
                != expected_dtype
            ):
                dtype_mismatches.append(
                    {
                        "column": column,
                        "expected": (
                            expected_dtype
                        ),
                        "actual": (
                            actual_dtype
                        ),
                    }
                )

        if dtype_mismatches:
            raise _validation_error(
                "dtype",
                f"{contract.name} dtype이 Builder 계약과 다릅니다.",
                count=len(
                    dtype_mismatches
                ),
                examples=(
                    dtype_mismatches[:10]
                ),
            )

        null_key = (
            frame.loc[
                :,
                list(
                    contract.key
                ),
            ]
            .isna()
            .any(
                axis=1
            )
        )

        if null_key.any():
            raise _validation_error(
                "grain",
                f"{contract.name} Grain Key에 null이 있습니다.",
                count=int(
                    null_key.sum()
                ),
                examples=(
                    frame.loc[
                        null_key,
                        list(
                            contract.key
                        ),
                    ]
                    .head(10)
                    .to_dict("records")
                ),
            )

        duplicate = (
            frame.duplicated(
                subset=list(
                    contract.key
                ),
                keep=False,
            )
        )

        duplicate_count = int(
            duplicate.sum()
        )

        duplicate_counts[
            contract.name
        ] = duplicate_count

        if duplicate_count > 0:
            raise _validation_error(
                "grain",
                f"{contract.name} Grain Key가 중복되었습니다.",
                count=duplicate_count,
                examples=(
                    frame.loc[
                        duplicate,
                        list(
                            contract.key
                        ),
                    ]
                    .head(10)
                    .to_dict("records")
                ),
            )

    return duplicate_counts


def _assert_no_null(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    label: str,
) -> None:
    """공식 Count/Context 컬럼의 예기치 않은 null을 검증한다."""
    for column in columns:
        null_count = int(
            frame[column]
            .isna()
            .sum()
        )

        if null_count > 0:
            raise _validation_error(
                "null_policy",
                f"{label}.{column}에 허용되지 않은 null이 있습니다.",
                count=null_count,
            )


def _assert_key_set_equal(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    key: Sequence[str],
    label: str,
) -> None:
    """두 DataFrame의 Key Set이 정확히 같은지 검증한다."""
    actual_key = (
        actual.loc[
            :,
            list(key),
        ]
        .drop_duplicates()
    )

    expected_key = (
        expected.loc[
            :,
            list(key),
        ]
        .drop_duplicates()
    )

    checked = (
        actual_key
        .merge(
            expected_key,
            on=list(
                key
            ),
            how="outer",
            indicator=True,
            validate="one_to_one",
            sort=False,
        )
    )

    mismatch = (
        checked["_merge"]
        .ne("both")
    )

    if mismatch.any():
        raise _validation_error(
            label,
            "Key Set이 일치하지 않습니다.",
            count=int(
                mismatch.sum()
            ),
            examples=(
                checked.loc[
                    mismatch,
                    [
                        *key,
                        "_merge",
                    ],
                ]
                .head(10)
                .to_dict("records")
            ),
        )


def _series_equal_mask(
    actual: pd.Series,
    expected: pd.Series,
    *,
    floating: bool,
) -> pd.Series:
    """nullable 값과 Float 허용오차를 고려한 값 일치 Mask를 만든다."""
    both_null = (
        actual.isna()
        & expected.isna()
    )

    if floating:
        left = pd.to_numeric(
            actual,
            errors="coerce",
        ).astype(
            "Float64"
        )

        right = pd.to_numeric(
            expected,
            errors="coerce",
        ).astype(
            "Float64"
        )

        equal_value = (
            left.sub(
                right
            )
            .abs()
            .le(
                1e-12
            )
            .fillna(False)
        )

    else:
        equal_value = (
            actual.eq(
                expected
            )
            .fillna(False)
        )

    return (
        both_null
        | equal_value
    )


def _assert_frame_matches(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    key: Sequence[str],
    columns: Sequence[str],
    label: str,
    float_columns: Iterable[str] = (),
) -> None:
    """Key와 지정 Column 값이 Source 간 정확히 일치하는지 검증한다."""
    _assert_key_set_equal(
        actual,
        expected,
        key=key,
        label=label,
    )

    checked = (
        actual.loc[
            :,
            [
                *key,
                *columns,
            ],
        ]
        .merge(
            expected.loc[
                :,
                [
                    *key,
                    *columns,
                ],
            ],
            on=list(
                key
            ),
            how="inner",
            validate="one_to_one",
            suffixes=(
                "_actual",
                "_expected",
            ),
            sort=False,
        )
    )

    float_set = set(
        float_columns
    )

    mismatched_columns: list[
        tuple[
            str,
            pd.Series,
        ]
    ] = []

    for column in columns:
        equal = _series_equal_mask(
            checked[
                f"{column}_actual"
            ],
            checked[
                f"{column}_expected"
            ],
            floating=(
                column
                in float_set
            ),
        )

        mismatch = ~equal

        if mismatch.any():
            mismatched_columns.append(
                (
                    column,
                    mismatch,
                )
            )

    if not mismatched_columns:
        return

    column, mismatch = (
        mismatched_columns[0]
    )

    example_columns = [
        *key,
        f"{column}_actual",
        f"{column}_expected",
    ]

    raise _validation_error(
        label,
        f"{column} 값이 일치하지 않습니다.",
        count=int(
            mismatch.sum()
        ),
        examples=(
            checked.loc[
                mismatch,
                example_columns,
            ]
            .head(10)
            .to_dict("records")
        ),
    )


def _unique_game_context(
    frame: pd.DataFrame,
    *,
    source_name: str,
) -> pd.DataFrame:
    """한 Table 안에서 game_pk별 game_date/season Context를 하나로 축약한다."""
    _require_columns(
        frame,
        (
            "game_pk",
            "game_date",
            "season",
        ),
        source_name=source_name,
    )

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

    counts = (
        context.groupby(
            "game_pk",
            sort=False,
            dropna=False,
        )
        .size()
    )

    invalid = (
        counts.gt(1)
    )

    if invalid.any():
        invalid_games = (
            invalid.loc[
                invalid
            ]
            .index
            .tolist()
        )

        raise _validation_error(
            "cross_table_context",
            f"{source_name}의 game_pk별 game_date/season이 일관되지 않습니다.",
            count=len(
                invalid_games
            ),
            examples=(
                invalid_games[:10]
            ),
        )

    return context


def validate_cross_table_context(
    tables: Mapping[str, pd.DataFrame],
) -> None:
    """Canonical Table 간 game_pk의 game_date/season을 교차 검증한다."""
    games = tables[
        "games"
    ]

    base = (
        games.loc[
            :,
            [
                "game_pk",
                "game_date",
                "season",
            ],
        ]
    )

    specifications = (
        (
            "plate_appearances",
            True,
        ),
        (
            "team_games",
            True,
        ),
        (
            "player_game_batting",
            False,
        ),
        (
            "player_game_pitching",
            False,
        ),
    )

    for table_name, require_exact in specifications:
        context = _unique_game_context(
            tables[
                table_name
            ],
            source_name=table_name,
        )

        if require_exact:
            _assert_key_set_equal(
                context,
                base,
                key=("game_pk",),
                label="cross_table_context",
            )

        checked = (
            context
            .merge(
                base,
                on="game_pk",
                how="left",
                validate="one_to_one",
                suffixes=(
                    "_actual",
                    "_games",
                ),
                sort=False,
            )
        )

        missing_game = (
            checked[
                "game_date_games"
            ]
            .isna()
        )

        if missing_game.any():
            raise _validation_error(
                "cross_table_context",
                f"{table_name}에 games Coverage 밖의 game_pk가 있습니다.",
                count=int(
                    missing_game.sum()
                ),
                examples=(
                    checked.loc[
                        missing_game,
                        [
                            "game_pk"
                        ],
                    ]
                    .head(10)
                    .to_dict("records")
                ),
            )

        date_match = _series_equal_mask(
            checked[
                "game_date_actual"
            ],
            checked[
                "game_date_games"
            ],
            floating=False,
        )

        season_match = _series_equal_mask(
            checked[
                "season_actual"
            ],
            checked[
                "season_games"
            ],
            floating=False,
        )

        mismatch = (
            ~date_match
            | ~season_match
        )

        if mismatch.any():
            raise _validation_error(
                "cross_table_context",
                f"{table_name}의 game_date/season이 games와 다릅니다.",
                count=int(
                    mismatch.sum()
                ),
                examples=(
                    checked.loc[
                        mismatch,
                        [
                            "game_pk",
                            "game_date_actual",
                            "game_date_games",
                            "season_actual",
                            "season_games",
                        ],
                    ]
                    .head(10)
                    .to_dict("records")
                ),
            )


def validate_raw_pa_and_games(
    raw: RawScanSummary,
    tables: Mapping[str, pd.DataFrame],
) -> None:
    """Raw ↔ PA 및 Raw ↔ Games의 Key/Context/Pitch Count를 검증한다."""
    plate_appearances = tables[
        "plate_appearances"
    ]

    games = tables[
        "games"
    ]

    _assert_frame_matches(
        plate_appearances,
        raw.pa_summary,
        key=PA_KEY,
        columns=(
            "pitch_rows",
            "pitch_count",
        ),
        label="raw_pa_key",
    )

    _assert_frame_matches(
        games,
        raw.game_context,
        key=("game_pk",),
        columns=(
            "game_date",
            "season",
            "home_team",
            "away_team",
        ),
        label="raw_game_key",
    )


def _build_expected_team_games(
    games: pd.DataFrame,
) -> pd.DataFrame:
    """Games 한 행에서 기대 Team Game Home/Away Mirror 두 행을 만든다."""
    home = pd.DataFrame(
        {
            "game_pk": games[
                "game_pk"
            ],
            "game_date": games[
                "game_date"
            ],
            "season": games[
                "season"
            ],
            "team": games[
                "home_team"
            ],
            "opponent": games[
                "away_team"
            ],
            "is_home": pd.Series(
                True,
                index=games.index,
                dtype="boolean",
            ),
            "runs_for": games[
                "final_home_score"
            ],
            "runs_against": games[
                "final_away_score"
            ],
            "run_diff": (
                games[
                    "final_home_score"
                ]
                - games[
                    "final_away_score"
                ]
            ),
            "win": games[
                "home_win"
            ],
            "loss": games[
                "away_win"
            ],
            "tie": games[
                "is_tie"
            ],
            "plate_appearances": games[
                "home_plate_appearances"
            ],
            "opponent_plate_appearances": games[
                "away_plate_appearances"
            ],
        }
    )

    away = pd.DataFrame(
        {
            "game_pk": games[
                "game_pk"
            ],
            "game_date": games[
                "game_date"
            ],
            "season": games[
                "season"
            ],
            "team": games[
                "away_team"
            ],
            "opponent": games[
                "home_team"
            ],
            "is_home": pd.Series(
                False,
                index=games.index,
                dtype="boolean",
            ),
            "runs_for": games[
                "final_away_score"
            ],
            "runs_against": games[
                "final_home_score"
            ],
            "run_diff": (
                games[
                    "final_away_score"
                ]
                - games[
                    "final_home_score"
                ]
            ),
            "win": games[
                "away_win"
            ],
            "loss": games[
                "home_win"
            ],
            "tie": games[
                "is_tie"
            ],
            "plate_appearances": games[
                "away_plate_appearances"
            ],
            "opponent_plate_appearances": games[
                "home_plate_appearances"
            ],
        }
    )

    return pd.concat(
        [
            home,
            away,
        ],
        ignore_index=True,
    )


def validate_games_team_games(
    tables: Mapping[str, pd.DataFrame],
) -> None:
    """Games ↔ Team Games의 2행 및 Home/Away Mirror 계약을 검증한다."""
    games = tables[
        "games"
    ]

    team_games = tables[
        "team_games"
    ]

    if (
        len(team_games)
        != 2 * len(games)
    ):
        raise _validation_error(
            "game_team_game",
            "Team Game Row 수가 Games의 정확히 2배가 아닙니다.",
            count=abs(
                len(team_games)
                - 2 * len(games)
            ),
            examples={
                "games": len(
                    games
                ),
                "team_games": len(
                    team_games
                ),
            },
        )

    rows_per_game = (
        team_games.groupby(
            "game_pk",
            sort=False,
            dropna=False,
        )
        .size()
    )

    invalid = (
        rows_per_game.ne(2)
    )

    if invalid.any():
        raise _validation_error(
            "game_team_game",
            "경기당 Team Game Row가 2개가 아닙니다.",
            count=int(
                invalid.sum()
            ),
            examples=(
                rows_per_game.loc[
                    invalid
                ]
                .head(10)
                .to_dict()
            ),
        )

    expected = (
        _build_expected_team_games(
            games
        )
    )

    compare_columns = tuple(
        column
        for column
        in game_builder.TEAM_GAMES_COLUMNS
        if column
        not in game_builder.TEAM_GAME_KEY
    )

    _assert_frame_matches(
        team_games,
        expected,
        key=game_builder.TEAM_GAME_KEY,
        columns=compare_columns,
        label="game_team_game",
    )


def validate_score_chain(
    plate_appearances: pd.DataFrame,
    games: pd.DataFrame,
) -> None:
    """
    실제 마지막 PA Post-score와
    첫 Pre-score + runs_scored를 독립적으로 검증한다.
    """
    ordered = (
        plate_appearances
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

    first_rows = (
        ordered
        .drop_duplicates(
            subset=[
                "game_pk"
            ],
            keep="first",
        )
        .copy()
    )

    last_rows = (
        ordered
        .drop_duplicates(
            subset=[
                "game_pk"
            ],
            keep="last",
        )
        .copy()
    )

    last_expected = (
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
                "post_home_score": (
                    "final_home_score"
                ),
                "post_away_score": (
                    "final_away_score"
                ),
            }
        )
    )

    _assert_frame_matches(
        games,
        last_expected,
        key=("game_pk",),
        columns=(
            "final_home_score",
            "final_away_score",
        ),
        label="score_last_pa",
    )

    run_source = (
        ordered.loc[
            :,
            [
                "game_pk",
                "is_home_batting",
                "runs_scored",
            ],
        ]
        .copy()
    )

    is_home = (
        run_source[
            "is_home_batting"
        ]
        .astype("boolean")
    )

    run_source[
        "_home_runs"
    ] = (
        run_source[
            "runs_scored"
        ]
        .where(
            is_home,
            0,
        )
    )

    run_source[
        "_away_runs"
    ] = (
        run_source[
            "runs_scored"
        ]
        .where(
            ~is_home,
            0,
        )
    )

    run_totals = (
        run_source
        .groupby(
            "game_pk",
            sort=False,
            dropna=False,
        )
        .agg(
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

    expected = (
        first_rows.loc[
            :,
            [
                "game_pk",
                "home_score_before",
                "away_score_before",
            ],
        ]
        .merge(
            run_totals,
            on="game_pk",
            how="inner",
            validate="one_to_one",
            sort=False,
        )
    )

    expected[
        "final_home_score"
    ] = (
        expected[
            "home_score_before"
        ]
        + expected[
            "_home_runs"
        ]
    )

    expected[
        "final_away_score"
    ] = (
        expected[
            "away_score_before"
        ]
        + expected[
            "_away_runs"
        ]
    )

    _assert_frame_matches(
        games,
        expected,
        key=("game_pk",),
        columns=(
            "final_home_score",
            "final_away_score",
        ),
        label="score_runs",
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

    result.loc[
        valid
    ] = (
        numerator.loc[
            valid
        ]
        .astype("Float64")
        .div(
            denominator.loc[
                valid
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

    result.loc[
        valid
    ] = (
        obp.loc[
            valid
        ]
        .astype("Float64")
        .add(
            slg.loc[
                valid
            ]
            .astype("Float64")
        )
        .round(3)
    )

    return result


def build_expected_player_game_batting(
    plate_appearances: pd.DataFrame,
) -> pd.DataFrame:
    """Canonical PA Event를 독립 cross-tab하여 Player Game Batting 기대값을 만든다."""
    completed = (
        plate_appearances.loc[
            plate_appearances[
                "event"
            ].notna()
        ]
        .sort_values(
            by=[
                "game_pk",
                "at_bat_number",
            ],
            kind="mergesort",
        )
        .copy()
    )

    if completed.empty:
        return pd.DataFrame(
            columns=batting_builder.OUTPUT_COLUMNS
        )

    completed[
        "team"
    ] = completed[
        "batting_team"
    ]

    completed[
        "opponent"
    ] = completed[
        "fielding_team"
    ]

    completed[
        "is_home"
    ] = completed[
        "is_home_batting"
    ]

    completed[
        "_pa"
    ] = pd.Series(
        1,
        index=completed.index,
        dtype="Int64",
    )

    count_columns = tuple(
        dict.fromkeys(
            batting_builder
            .EVENT_TO_COUNT_COLUMN
            .values()
        )
    )

    for event, column in (
        batting_builder
        .EVENT_TO_COUNT_COLUMN
        .items()
    ):
        completed[column] = (
            completed[
                "event"
            ]
            .eq(event)
            .astype("Int64")
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
                batting_builder
                .PLAYER_GAME_KEY
            ),
            keep="first",
        )
    )

    aggregation = {
        "pa": (
            "_pa",
            "sum",
        ),
    }

    aggregation.update(
        {
            column: (
                column,
                "sum",
            )
            for column
            in count_columns
        }
    )

    counts = (
        completed
        .groupby(
            list(
                batting_builder
                .PLAYER_GAME_KEY
            ),
            sort=False,
            dropna=False,
        )
        .agg(
            **aggregation
        )
        .reset_index()
    )

    result = (
        context.merge(
            counts,
            on=list(
                batting_builder
                .PLAYER_GAME_KEY
            ),
            how="inner",
            validate="one_to_one",
            sort=False,
        )
    )

    result[
        "h"
    ] = (
        result[
            "single"
        ]
        + result[
            "double"
        ]
        + result[
            "triple"
        ]
        + result[
            "hr"
        ]
    ).astype(
        "Int64"
    )

    result[
        "ab"
    ] = (
        result[
            "pa"
        ]
        - result[
            "bb"
        ]
        - result[
            "hbp"
        ]
        - result[
            "sh"
        ]
        - result[
            "sf"
        ]
        - result[
            "catcher_interference"
        ]
    ).astype(
        "Int64"
    )

    result[
        "tb"
    ] = (
        result[
            "single"
        ]
        + 2
        * result[
            "double"
        ]
        + 3
        * result[
            "triple"
        ]
        + 4
        * result[
            "hr"
        ]
    ).astype(
        "Int64"
    )

    result[
        "avg"
    ] = calculate_rate(
        result[
            "h"
        ],
        result[
            "ab"
        ],
    )

    obp_denominator = (
        result[
            "ab"
        ]
        + result[
            "bb"
        ]
        + result[
            "hbp"
        ]
        + result[
            "sf"
        ]
    )

    result[
        "obp"
    ] = calculate_rate(
        (
            result[
                "h"
            ]
            + result[
                "bb"
            ]
            + result[
                "hbp"
            ]
        ),
        obp_denominator,
    )

    result[
        "slg"
    ] = calculate_rate(
        result[
            "tb"
        ],
        result[
            "ab"
        ],
    )

    result[
        "ops"
    ] = calculate_ops(
        result[
            "obp"
        ],
        result[
            "slg"
        ],
    )

    return result.loc[
        :,
        batting_builder.OUTPUT_COLUMNS,
    ]


def validate_player_game_batting(
    plate_appearances: pd.DataFrame,
    player_game_batting: pd.DataFrame,
) -> None:
    """PA ↔ Player Game Batting Event/Formula 계약을 검증한다."""
    count_columns = (
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

    _assert_no_null(
        player_game_batting,
        count_columns,
        label=(
            "player_game_batting"
        ),
    )

    expected = (
        build_expected_player_game_batting(
            plate_appearances
        )
    )

    completed_pa_count = int(
        plate_appearances[
            "event"
        ]
        .notna()
        .sum()
    )

    actual_pa_count = int(
        player_game_batting[
            "pa"
        ]
        .sum()
    )

    if (
        actual_pa_count
        != completed_pa_count
    ):
        raise _validation_error(
            "batting_pa",
            "Completed PA 수와 Player Game Batting pa 합이 다릅니다.",
            examples={
                "completed_pa": (
                    completed_pa_count
                ),
                "batting_pa": (
                    actual_pa_count
                ),
            },
        )

    compare_columns = tuple(
        column
        for column
        in batting_builder.OUTPUT_COLUMNS
        if column
        not in batting_builder.PLAYER_GAME_KEY
    )

    _assert_frame_matches(
        player_game_batting,
        expected,
        key=batting_builder.PLAYER_GAME_KEY,
        columns=compare_columns,
        label="batting_event",
        float_columns=(
            "avg",
            "obp",
            "slg",
            "ops",
        ),
    )


def _season_through_dates(
    team_games: pd.DataFrame,
) -> pd.DataFrame:
    """Team Games에서 시즌 공통 through_date를 계산한다."""
    return (
        team_games
        .groupby(
            "season",
            sort=True,
            dropna=False,
        )[
            "game_date"
        ]
        .max()
        .rename(
            "through_date"
        )
        .reset_index()
    )


def build_expected_season_batting(
    player_game_batting: pd.DataFrame,
    team_games: pd.DataFrame,
) -> pd.DataFrame:
    """Player Game Count만 합산하여 Season Batting 기대값을 만든다."""
    direct_columns = (
        season_builder
        .BATTING_DIRECT_SUM_COLUMNS
    )

    grouped = (
        player_game_batting
        .groupby(
            [
                "season",
                "batter",
            ],
            sort=True,
            dropna=False,
        )[
            list(
                direct_columns
            )
        ]
        .sum()
        .reset_index()
    )

    grouped[
        "h"
    ] = (
        grouped[
            "single"
        ]
        + grouped[
            "double"
        ]
        + grouped[
            "triple"
        ]
        + grouped[
            "hr"
        ]
    ).astype(
        "Int64"
    )

    grouped[
        "ab"
    ] = (
        grouped[
            "pa"
        ]
        - grouped[
            "bb"
        ]
        - grouped[
            "hbp"
        ]
        - grouped[
            "sh"
        ]
        - grouped[
            "sf"
        ]
        - grouped[
            "catcher_interference"
        ]
    ).astype(
        "Int64"
    )

    grouped[
        "tb"
    ] = (
        grouped[
            "single"
        ]
        + 2
        * grouped[
            "double"
        ]
        + 3
        * grouped[
            "triple"
        ]
        + 4
        * grouped[
            "hr"
        ]
    ).astype(
        "Int64"
    )

    grouped[
        "avg"
    ] = calculate_rate(
        grouped[
            "h"
        ],
        grouped[
            "ab"
        ],
    )

    grouped[
        "obp"
    ] = calculate_rate(
        (
            grouped[
                "h"
            ]
            + grouped[
                "bb"
            ]
            + grouped[
                "hbp"
            ]
        ),
        (
            grouped[
                "ab"
            ]
            + grouped[
                "bb"
            ]
            + grouped[
                "hbp"
            ]
            + grouped[
                "sf"
            ]
        ),
    )

    grouped[
        "slg"
    ] = calculate_rate(
        grouped[
            "tb"
        ],
        grouped[
            "ab"
        ],
    )

    grouped[
        "ops"
    ] = calculate_ops(
        grouped[
            "obp"
        ],
        grouped[
            "slg"
        ],
    )

    result = (
        grouped.merge(
            _season_through_dates(
                team_games
            ),
            on="season",
            how="left",
            validate="many_to_one",
            sort=False,
        )
    )

    return result.loc[
        :,
        season_builder.BATTING_OUTPUT_COLUMNS,
    ]


def validate_season_batting(
    player_game_batting: pd.DataFrame,
    snapshot: pd.DataFrame,
    team_games: pd.DataFrame,
) -> None:
    """Player Game Batting ↔ Season Batting Snapshot을 검증한다."""
    expected = (
        build_expected_season_batting(
            player_game_batting,
            team_games,
        )
    )

    compare_columns = tuple(
        column
        for column
        in season_builder.BATTING_OUTPUT_COLUMNS
        if column
        not in season_builder.BATTING_SNAPSHOT_KEY
    )

    _assert_frame_matches(
        snapshot,
        expected,
        key=season_builder.BATTING_SNAPSHOT_KEY,
        columns=compare_columns,
        label="season_batting",
        float_columns=(
            "avg",
            "obp",
            "slg",
            "ops",
        ),
    )


def _build_expected_pitching_pa_counts(
    plate_appearances: pd.DataFrame,
    raw_pitcher_keys: pd.DataFrame,
) -> pd.DataFrame:
    """Canonical PA에서 Pitcher BF와 Allowed Event를 독립 집계한다."""
    working = (
        plate_appearances.loc[
            :,
            [
                "game_pk",
                "pitcher",
                "event",
            ],
        ]
        .copy()
    )

    completed = (
        working[
            "event"
        ]
        .notna()
    )

    working[
        "batters_faced_completed"
    ] = completed.astype(
        "Int64"
    )

    count_columns = tuple(
        dict.fromkeys(
            pitching_builder
            .COUNT_EVENT_MAPPING
            .values()
        )
    )

    for event, column in (
        pitching_builder
        .COUNT_EVENT_MAPPING
        .items()
    ):
        working[column] = (
            completed
            & working[
                "event"
            ].eq(event)
        ).astype(
            "Int64"
        )

    aggregation = {
        "batters_faced_completed": (
            "batters_faced_completed",
            "sum",
        ),
    }

    aggregation.update(
        {
            column: (
                column,
                "sum",
            )
            for column
            in count_columns
        }
    )

    grouped = (
        working
        .groupby(
            list(
                PITCHER_GAME_KEY
            ),
            sort=False,
            dropna=False,
        )
        .agg(
            **aggregation
        )
        .reset_index()
    )

    expected = (
        raw_pitcher_keys.loc[
            :,
            list(
                PITCHER_GAME_KEY
            ),
        ]
        .drop_duplicates()
        .merge(
            grouped,
            on=list(
                PITCHER_GAME_KEY
            ),
            how="left",
            validate="one_to_one",
            sort=False,
        )
    )

    fill_columns = (
        "batters_faced_completed",
        *count_columns,
    )

    for column in fill_columns:
        expected[column] = (
            expected[column]
            .fillna(0)
            .astype("Int64")
        )

    expected[
        "hits_allowed"
    ] = (
        expected[
            "single_allowed"
        ]
        + expected[
            "double_allowed"
        ]
        + expected[
            "triple_allowed"
        ]
        + expected[
            "hr_allowed"
        ]
    ).astype(
        "Int64"
    )

    return expected


def _build_expected_outs(
    plate_appearances: pd.DataFrame,
    raw: RawScanSummary,
) -> pd.DataFrame:
    """
    Issue #11 공개 계약을 독립 적용해
    Player Game별 기대 outs_recorded를 계산한다.
    """
    pa_outs = (
        plate_appearances.loc[
            :,
            [
                "game_pk",
                "at_bat_number",
                "pitcher",
                "event",
                "outs_before",
                "post_outs",
            ],
        ]
        .merge(
            raw.pa_summary.loc[
                :,
                [
                    *PA_KEY,
                    "unique_pitchers",
                    "last_raw_pitcher",
                ],
            ],
            on=list(
                PA_KEY
            ),
            how="left",
            validate="one_to_one",
            sort=False,
        )
    )

    missing_raw = (
        pa_outs[
            "unique_pitchers"
        ]
        .isna()
    )

    if missing_raw.any():
        raise _validation_error(
            "pitching_outs",
            "Out 검증 중 Canonical PA에 대응하는 Raw PA를 찾지 못했습니다.",
            count=int(
                missing_raw.sum()
            ),
            examples=(
                pa_outs.loc[
                    missing_raw,
                    list(
                        PA_KEY
                    ),
                ]
                .head(10)
                .to_dict("records")
            ),
        )

    pitcher_mismatch = (
        pa_outs[
            "pitcher"
        ]
        .ne(
            pa_outs[
                "last_raw_pitcher"
            ]
        )
        .fillna(True)
    )

    if pitcher_mismatch.any():
        raise _validation_error(
            "pitching_outs",
            "Canonical PA pitcher와 실제 마지막 Raw pitcher가 다릅니다.",
            count=int(
                pitcher_mismatch.sum()
            ),
            examples=(
                pa_outs.loc[
                    pitcher_mismatch,
                    [
                        *PA_KEY,
                        "pitcher",
                        "last_raw_pitcher",
                    ],
                ]
                .head(10)
                .to_dict("records")
            ),
        )

    participants_by_pa: dict[
        tuple[object, object],
        list[str],
    ] = {}

    if not (
        raw
        .multi_pitcher_participants
        .empty
    ):
        for key, group in (
            raw.multi_pitcher_participants
            .groupby(
                list(
                    PA_KEY
                ),
                sort=False,
                dropna=False,
            )
        ):
            normalized_key = (
                key
                if isinstance(
                    key,
                    tuple,
                )
                else (
                    key,
                )
            )

            participants_by_pa[
                (
                    normalized_key[0],
                    normalized_key[1],
                )
            ] = (
                group[
                    "pitcher"
                ]
                .astype("string")
                .tolist()
            )

    totals: dict[
        tuple[str, str],
        int,
    ] = {
        (
            str(game_pk),
            str(pitcher),
        ): 0
        for game_pk, pitcher
        in raw.pitcher_summary.loc[
            :,
            list(
                PITCHER_GAME_KEY
            ),
        ].itertuples(
            index=False,
            name=None,
        )
    }

    ambiguous: set[
        tuple[str, str]
    ] = set()

    for row in pa_outs.itertuples(
        index=False
    ):
        delta = (
            int(
                row.post_outs
            )
            - int(
                row.outs_before
            )
        )

        if (
            delta < 0
            or delta > 3
        ):
            raise _validation_error(
                "pitching_outs",
                "PA Out Delta가 0~3 범위를 벗어났습니다.",
                examples={
                    "game_pk": (
                        row.game_pk
                    ),
                    "at_bat_number": (
                        row.at_bat_number
                    ),
                    "delta": delta,
                },
            )

        game_key = str(
            row.game_pk
        )

        final_pitcher = str(
            row.pitcher
        )

        unique_pitchers = int(
            row.unique_pitchers
        )

        if (
            unique_pitchers > 1
            and delta > 0
        ):
            terminal_outs = (
                pitching_builder
                .MULTI_PITCHER_TERMINAL_OUTS
                .get(
                    row.event
                )
            )

            safely_attributable = (
                terminal_outs
                is not None
                and int(
                    terminal_outs
                )
                == delta
            )

            if not safely_attributable:
                pa_key = (
                    row.game_pk,
                    row.at_bat_number,
                )

                participants = (
                    participants_by_pa
                    .get(
                        pa_key,
                        [],
                    )
                )

                if (
                    len(
                        participants
                    )
                    != unique_pitchers
                ):
                    raise _validation_error(
                        "pitching_outs",
                        "Multi-pitcher PA 참여자 정보를 복원할 수 없습니다.",
                        examples={
                            "pa_key": (
                                pa_key
                            ),
                            "expected_pitchers": (
                                unique_pitchers
                            ),
                            "participants": (
                                participants
                            ),
                        },
                    )

                ambiguous.update(
                    (
                        game_key,
                        str(
                            pitcher
                        ),
                    )
                    for pitcher
                    in participants
                )

                continue

        if delta > 0:
            key = (
                game_key,
                final_pitcher,
            )

            if key not in totals:
                raise _validation_error(
                    "pitching_outs",
                    "Canonical PA pitcher가 Raw Player Game Key에 없습니다.",
                    examples=key,
                )

            totals[key] += delta

    rows: list[
        dict[str, object]
    ] = []

    for key in sorted(
        totals
    ):
        value: object

        if key in ambiguous:
            value = pd.NA
        else:
            value = totals[
                key
            ]

        rows.append(
            {
                "game_pk": key[0],
                "pitcher": key[1],
                "outs_recorded": value,
            }
        )

    result = pd.DataFrame(
        rows
    )

    result[
        "game_pk"
    ] = result[
        "game_pk"
    ].astype(
        "string"
    )

    result[
        "pitcher"
    ] = result[
        "pitcher"
    ].astype(
        "string"
    )

    result[
        "outs_recorded"
    ] = result[
        "outs_recorded"
    ].astype(
        "Int64"
    )

    return result


def validate_player_game_pitching(
    raw: RawScanSummary,
    plate_appearances: pd.DataFrame,
    player_game_pitching: pd.DataFrame,
) -> None:
    """Raw/PA ↔ Player Game Pitching 계약을 검증한다."""
    non_nullable_count_columns = (
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
        "ball_pitch_count",
        "strike_pitch_count",
        "in_play_pitch_count",
    )

    _assert_no_null(
        player_game_pitching,
        non_nullable_count_columns,
        label=(
            "player_game_pitching"
        ),
    )

    _assert_key_set_equal(
        player_game_pitching,
        raw.pitcher_summary,
        key=PITCHER_GAME_KEY,
        label="raw_pitch",
    )

    _assert_frame_matches(
        player_game_pitching,
        raw.pitcher_summary,
        key=PITCHER_GAME_KEY,
        columns=(
            "pitch_rows",
            "pitches",
            "ball_pitch_count",
            "strike_pitch_count",
            "in_play_pitch_count",
        ),
        label="raw_pitch",
    )

    pitch_type_sum = (
        player_game_pitching[
            "ball_pitch_count"
        ]
        + player_game_pitching[
            "strike_pitch_count"
        ]
        + player_game_pitching[
            "in_play_pitch_count"
        ]
    )

    pitch_type_mismatch = (
        pitch_type_sum
        .ne(
            player_game_pitching[
                "pitches"
            ]
        )
    )

    if pitch_type_mismatch.any():
        raise _validation_error(
            "pitch_type",
            "Player Game Pitching의 B/S/X 합이 pitches와 다릅니다.",
            count=int(
                pitch_type_mismatch.sum()
            ),
            examples=(
                player_game_pitching.loc[
                    pitch_type_mismatch,
                    [
                        "game_pk",
                        "pitcher",
                        "pitches",
                        "ball_pitch_count",
                        "strike_pitch_count",
                        "in_play_pitch_count",
                    ],
                ]
                .head(10)
                .to_dict("records")
            ),
        )

    expected_pa_counts = (
        _build_expected_pitching_pa_counts(
            plate_appearances,
            raw.pitcher_summary,
        )
    )

    pa_compare_columns = (
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
    )

    _assert_frame_matches(
        player_game_pitching,
        expected_pa_counts,
        key=PITCHER_GAME_KEY,
        columns=pa_compare_columns,
        label="bf_allowed_event",
    )

    expected_outs = (
        _build_expected_outs(
            plate_appearances,
            raw,
        )
    )

    _assert_frame_matches(
        player_game_pitching,
        expected_outs,
        key=PITCHER_GAME_KEY,
        columns=(
            "outs_recorded",
        ),
        label="pitching_outs",
    )

    negative_non_null_outs = (
        player_game_pitching[
            "outs_recorded"
        ]
        .dropna()
        .lt(0)
    )

    if negative_non_null_outs.any():
        raise _validation_error(
            "pitching_outs",
            "non-null outs_recorded에 음수가 있습니다.",
            count=int(
                negative_non_null_outs.sum()
            ),
        )


def build_expected_season_pitching(
    player_game_pitching: pd.DataFrame,
    team_games: pd.DataFrame,
) -> pd.DataFrame:
    """Player Game Pitching Count와 nullable Outs로 Season 기대값을 만든다."""
    direct_columns = (
        season_builder
        .PITCHING_DIRECT_SUM_COLUMNS
    )

    grouped = (
        player_game_pitching
        .groupby(
            [
                "season",
                "pitcher",
            ],
            sort=True,
            dropna=False,
        )[
            list(
                direct_columns
            )
        ]
        .sum()
        .reset_index()
    )

    grouped[
        "hits_allowed"
    ] = (
        grouped[
            "single_allowed"
        ]
        + grouped[
            "double_allowed"
        ]
        + grouped[
            "triple_allowed"
        ]
        + grouped[
            "hr_allowed"
        ]
    ).astype(
        "Int64"
    )

    def aggregate_outs(
        series: pd.Series,
    ) -> object:
        if series.isna().any():
            return pd.NA

        return int(
            series.sum()
        )

    outs = (
        player_game_pitching
        .groupby(
            [
                "season",
                "pitcher",
            ],
            sort=True,
            dropna=False,
        )[
            "outs_recorded"
        ]
        .apply(
            aggregate_outs
        )
        .rename(
            "outs_recorded"
        )
        .reset_index()
    )

    outs[
        "outs_recorded"
    ] = (
        outs[
            "outs_recorded"
        ]
        .astype("Int64")
    )

    result = (
        grouped
        .merge(
            outs,
            on=[
                "season",
                "pitcher",
            ],
            how="inner",
            validate="one_to_one",
            sort=False,
        )
        .merge(
            _season_through_dates(
                team_games
            ),
            on="season",
            how="left",
            validate="many_to_one",
            sort=False,
        )
    )

    return result.loc[
        :,
        season_builder.PITCHING_OUTPUT_COLUMNS,
    ]


def validate_season_pitching(
    player_game_pitching: pd.DataFrame,
    snapshot: pd.DataFrame,
    team_games: pd.DataFrame,
) -> None:
    """Player Game Pitching ↔ Season Pitching Snapshot을 검증한다."""
    expected = (
        build_expected_season_pitching(
            player_game_pitching,
            team_games,
        )
    )

    compare_columns = tuple(
        column
        for column
        in season_builder.PITCHING_OUTPUT_COLUMNS
        if column
        not in season_builder.PITCHING_SNAPSHOT_KEY
    )

    _assert_frame_matches(
        snapshot,
        expected,
        key=season_builder.PITCHING_SNAPSHOT_KEY,
        columns=compare_columns,
        label="season_pitching",
    )


def validate_players(
    plate_appearances: pd.DataFrame,
    player_game_batting: pd.DataFrame,
    player_game_pitching: pd.DataFrame,
    players: pd.DataFrame,
) -> None:
    """Player Source Union exact equality와 역할을 검증한다."""
    prohibited = [
        column
        for column
        in players_builder
        .PROHIBITED_OUTPUT_COLUMNS
        if column
        in players.columns
    ]

    if prohibited:
        raise _validation_error(
            "player_union",
            "Player Master에 단일 Team 속성이 존재합니다.",
            count=len(
                prohibited
            ),
            examples=prohibited,
        )

    batter_ids = set(
        pd.concat(
            [
                plate_appearances[
                    "batter"
                ],
                player_game_batting[
                    "batter"
                ],
            ],
            ignore_index=True,
        )
        .dropna()
        .astype("string")
        .tolist()
    )

    pitcher_ids = set(
        pd.concat(
            [
                plate_appearances[
                    "pitcher"
                ],
                player_game_pitching[
                    "pitcher"
                ],
            ],
            ignore_index=True,
        )
        .dropna()
        .astype("string")
        .tolist()
    )

    all_ids = sorted(
        batter_ids
        | pitcher_ids
    )

    expected = pd.DataFrame(
        {
            "player_id": pd.Series(
                all_ids,
                dtype="string",
            ),
            "is_batter": pd.Series(
                [
                    player_id
                    in batter_ids
                    for player_id
                    in all_ids
                ],
                dtype="boolean",
            ),
            "is_pitcher": pd.Series(
                [
                    player_id
                    in pitcher_ids
                    for player_id
                    in all_ids
                ],
                dtype="boolean",
            ),
        }
    )

    _assert_frame_matches(
        players,
        expected,
        key=("player_id",),
        columns=(
            "is_batter",
            "is_pitcher",
        ),
        label="player_role",
    )


def build_expected_team_season(
    team_games: pd.DataFrame,
) -> pd.DataFrame:
    """Team Games를 독립 집계하여 Team Season 기대값을 만든다."""
    working = (
        team_games
        .loc[
            :,
            [
                "game_pk",
                "game_date",
                "season",
                "team",
                "runs_for",
                "runs_against",
                "win",
                "loss",
                "tie",
            ],
        ]
        .copy()
    )

    working[
        "_win"
    ] = working[
        "win"
    ].astype(
        "Int64"
    )

    working[
        "_loss"
    ] = working[
        "loss"
    ].astype(
        "Int64"
    )

    working[
        "_tie"
    ] = working[
        "tie"
    ].astype(
        "Int64"
    )

    result = (
        working
        .groupby(
            [
                "season",
                "team",
            ],
            sort=True,
            dropna=False,
        )
        .agg(
            games=(
                "game_pk",
                "size",
            ),
            wins=(
                "_win",
                "sum",
            ),
            losses=(
                "_loss",
                "sum",
            ),
            ties=(
                "_tie",
                "sum",
            ),
            runs_for=(
                "runs_for",
                "sum",
            ),
            runs_against=(
                "runs_against",
                "sum",
            ),
        )
        .reset_index()
    )

    for column in (
        "games",
        "wins",
        "losses",
        "ties",
        "runs_for",
        "runs_against",
    ):
        result[column] = (
            result[column]
            .astype("Int64")
        )

    result[
        "run_diff"
    ] = (
        result[
            "runs_for"
        ]
        - result[
            "runs_against"
        ]
    ).astype(
        "Int64"
    )

    result = (
        result
        .merge(
            _season_through_dates(
                team_games
            ),
            on="season",
            how="left",
            validate="many_to_one",
            sort=False,
        )
    )

    return result.loc[
        :,
        season_builder.TEAM_OUTPUT_COLUMNS,
    ]


def validate_team_season(
    team_games: pd.DataFrame,
    snapshot: pd.DataFrame,
) -> None:
    """Team Games ↔ Team Season 및 시즌 대칭성을 검증한다."""
    expected = (
        build_expected_team_season(
            team_games
        )
    )

    compare_columns = tuple(
        column
        for column
        in season_builder.TEAM_OUTPUT_COLUMNS
        if column
        not in season_builder.TEAM_SNAPSHOT_KEY
    )

    _assert_frame_matches(
        snapshot,
        expected,
        key=season_builder.TEAM_SNAPSHOT_KEY,
        columns=compare_columns,
        label="team_season",
    )

    invalid_record = (
        snapshot[
            "games"
        ]
        .ne(
            snapshot[
                "wins"
            ]
            + snapshot[
                "losses"
            ]
            + snapshot[
                "ties"
            ]
        )
    )

    if invalid_record.any():
        raise _validation_error(
            "team_season",
            "Team Season games != wins + losses + ties 입니다.",
            count=int(
                invalid_record.sum()
            ),
        )

    symmetry_source = (
        team_games
        .assign(
            _win=(
                team_games[
                    "win"
                ]
                .astype("Int64")
            ),
            _loss=(
                team_games[
                    "loss"
                ]
                .astype("Int64")
            ),
        )
        .groupby(
            "season",
            sort=True,
            dropna=False,
        )
        .agg(
            wins=(
                "_win",
                "sum",
            ),
            losses=(
                "_loss",
                "sum",
            ),
            runs_for=(
                "runs_for",
                "sum",
            ),
            runs_against=(
                "runs_against",
                "sum",
            ),
        )
    )

    invalid_wins = (
        symmetry_source[
            "wins"
        ]
        .ne(
            symmetry_source[
                "losses"
            ]
        )
    )

    invalid_runs = (
        symmetry_source[
            "runs_for"
        ]
        .ne(
            symmetry_source[
                "runs_against"
            ]
        )
    )

    if (
        invalid_wins.any()
        or invalid_runs.any()
    ):
        raise _validation_error(
            "team_season",
            "시즌 전체 Team Game의 W/L 또는 runs_for/runs_against 대칭성이 깨졌습니다.",
            count=int(
                (
                    invalid_wins
                    | invalid_runs
                ).sum()
            ),
        )

    if (
        int(
            snapshot[
                "games"
            ]
            .sum()
        )
        != len(
            team_games
        )
    ):
        raise _validation_error(
            "team_season",
            "Team Season games 합이 Team Game Row 수와 다릅니다.",
            examples={
                "snapshot_games": int(
                    snapshot[
                        "games"
                    ]
                    .sum()
                ),
                "team_games": len(
                    team_games
                ),
            },
        )


def validate_snapshot_through_dates(
    team_games: pd.DataFrame,
    batting_snapshot: pd.DataFrame,
    pitching_snapshot: pd.DataFrame,
    team_snapshot: pd.DataFrame,
) -> dict[int, str]:
    """세 Snapshot의 through_date를 Team Games 시즌 max date와 비교한다."""
    through = (
        _season_through_dates(
            team_games
        )
    )

    for name, frame in (
        (
            "player_season_batting_snapshot",
            batting_snapshot,
        ),
        (
            "player_season_pitching_snapshot",
            pitching_snapshot,
        ),
        (
            "team_season_snapshot",
            team_snapshot,
        ),
    ):
        checked = (
            frame.loc[
                :,
                [
                    "season",
                    "through_date",
                ],
            ]
            .merge(
                through,
                on="season",
                how="left",
                validate="many_to_one",
                suffixes=(
                    "_actual",
                    "_expected",
                ),
                sort=False,
            )
        )

        equal = _series_equal_mask(
            checked[
                "through_date_actual"
            ],
            checked[
                "through_date_expected"
            ],
            floating=False,
        )

        mismatch = ~equal

        if mismatch.any():
            category = (
                "partial_2026"
                if (
                    checked.loc[
                        mismatch,
                        "season",
                    ]
                    .eq(2026)
                    .any()
                )
                else "through_date"
            )

            raise _validation_error(
                category,
                f"{name}.through_date가 Team Games 시즌 max date와 다릅니다.",
                count=int(
                    mismatch.sum()
                ),
                examples=(
                    checked.loc[
                        mismatch,
                        [
                            "season",
                            "through_date_actual",
                            "through_date_expected",
                        ],
                    ]
                    .head(10)
                    .to_dict("records")
                ),
            )

    return {
        int(
            row.season
        ): (
            pd.Timestamp(
                row.through_date
            )
            .date()
            .isoformat()
        )
        for row
        in through.itertuples(
            index=False
        )
    }


def content_fingerprint(
    frame: pd.DataFrame,
    *,
    sort_columns: Sequence[str],
    columns: Sequence[str] | None = None,
) -> str:
    """
    Canonical 정렬 후 DataFrame 내용과 Schema를 SHA256으로 요약한다.

    Parquet Byte Hash가 아니라 동일 환경/동일 Snapshot에서
    결정성 진단에 사용하는 Content Fingerprint다.
    """
    selected_columns = (
        list(
            columns
        )
        if columns
        is not None
        else list(
            frame.columns
        )
    )

    canonical = (
        frame.loc[
            :,
            selected_columns,
        ]
        .sort_values(
            by=list(
                sort_columns
            ),
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    digest = hashlib.sha256()

    digest.update(
        "\x1f".join(
            selected_columns
        ).encode(
            "utf-8"
        )
    )

    digest.update(
        "\x1f".join(
            str(
                canonical[
                    column
                ].dtype
            )
            for column
            in selected_columns
        ).encode(
            "utf-8"
        )
    )

    row_hash = (
        pd.util
        .hash_pandas_object(
            canonical,
            index=False,
            categorize=True,
        )
        .to_numpy(
            dtype="uint64",
            copy=False,
        )
    )

    digest.update(
        row_hash.tobytes()
    )

    return digest.hexdigest()


def _build_content_fingerprints(
    tables: Mapping[str, pd.DataFrame],
) -> dict[str, str]:
    """아홉 Table의 Canonical Content Fingerprint를 계산한다."""
    return {
        contract.name: (
            content_fingerprint(
                tables[
                    contract.name
                ],
                sort_columns=(
                    contract.key
                ),
                columns=(
                    contract.columns
                ),
            )
        )
        for contract
        in TABLE_CONTRACTS
    }


def _log_summary(
    summary: DerivedValidationSummary,
) -> None:
    """성공한 Production Validation의 핵심 Summary를 logging으로 출력한다."""
    LOGGER.info(
        "Derived Validation PASS | "
        "raw_rows=%d | "
        "raw_actual_pitches=%d | "
        "raw_unique_pa=%d | "
        "raw_unique_games=%d",
        summary.raw_rows,
        summary.raw_actual_pitches,
        summary.raw_unique_pa,
        summary.raw_unique_games,
    )

    LOGGER.info(
        "Derived row counts | %s",
        summary.table_row_counts,
    )

    LOGGER.info(
        "Duplicate key counts | %s",
        summary.duplicate_key_counts,
    )

    LOGGER.info(
        "Schema error counts | %s",
        summary.schema_error_counts,
    )

    LOGGER.info(
        "Cross-table error counts | %s",
        summary.cross_table_error_counts,
    )

    LOGGER.info(
        "Through dates | %s",
        summary.through_dates,
    )

    LOGGER.info(
        "Content fingerprints | %s",
        summary.content_fingerprints,
    )

    LOGGER.info(
        "Raw SHA256 before | %s",
        summary.raw_sha256_before,
    )

    LOGGER.info(
        "Raw SHA256 after | %s",
        summary.raw_sha256_after,
    )

    LOGGER.info(
        "Derived SHA256 before | %s",
        summary.derived_sha256_before,
    )

    LOGGER.info(
        "Derived SHA256 after | %s",
        summary.derived_sha256_after,
    )


def validate_derived_tables(
    *,
    raw_dir: Path = DEFAULT_RAW_DIR,
    derived_dir: Path = DEFAULT_DERIVED_DIR,
    seasons: Sequence[int] = DEFAULT_SEASONS,
    project_root: Path = PROJECT_ROOT,
) -> DerivedValidationSummary:
    """
    Raw와 아홉 Canonical Derived Artifact를 Read-only 통합 검증한다.

    어떤 Raw/Derived 파일도 생성하거나 수정하지 않는다.
    """
    normalized_seasons = (
        normalize_seasons(
            seasons
        )
    )

    raw_paths = {
        f"{season}.parquet": (
            raw_dir
            / f"{season}.parquet"
        )
        for season
        in normalized_seasons
    }

    derived_paths = {
        contract.filename: (
            derived_dir
            / contract.filename
        )
        for contract
        in TABLE_CONTRACTS
    }

    raw_before = (
        _capture_named_sha256(
            raw_paths
        )
    )

    derived_before = (
        _capture_named_sha256(
            derived_paths
        )
    )

    try:
        validate_derived_path_policy(
            derived_dir,
            project_root=project_root,
        )

        tables = (
            _load_derived_tables(
                derived_dir
            )
        )

        duplicate_counts = (
            validate_schema_and_grain(
                tables
            )
        )

        raw = scan_raw(
            raw_dir,
            normalized_seasons,
        )

        validate_cross_table_context(
            tables
        )

        validate_raw_pa_and_games(
            raw,
            tables,
        )

        validate_games_team_games(
            tables
        )

        validate_score_chain(
            tables[
                "plate_appearances"
            ],
            tables[
                "games"
            ],
        )

        validate_player_game_batting(
            tables[
                "plate_appearances"
            ],
            tables[
                "player_game_batting"
            ],
        )

        validate_season_batting(
            tables[
                "player_game_batting"
            ],
            tables[
                "player_season_batting_snapshot"
            ],
            tables[
                "team_games"
            ],
        )

        validate_player_game_pitching(
            raw,
            tables[
                "plate_appearances"
            ],
            tables[
                "player_game_pitching"
            ],
        )

        validate_season_pitching(
            tables[
                "player_game_pitching"
            ],
            tables[
                "player_season_pitching_snapshot"
            ],
            tables[
                "team_games"
            ],
        )

        validate_players(
            tables[
                "plate_appearances"
            ],
            tables[
                "player_game_batting"
            ],
            tables[
                "player_game_pitching"
            ],
            tables[
                "players"
            ],
        )

        validate_team_season(
            tables[
                "team_games"
            ],
            tables[
                "team_season_snapshot"
            ],
        )

        through_dates = (
            validate_snapshot_through_dates(
                tables[
                    "team_games"
                ],
                tables[
                    "player_season_batting_snapshot"
                ],
                tables[
                    "player_season_pitching_snapshot"
                ],
                tables[
                    "team_season_snapshot"
                ],
            )
        )

        fingerprints = (
            _build_content_fingerprints(
                tables
            )
        )

        table_row_counts = {
            contract.name: len(
                tables[
                    contract.name
                ]
            )
            for contract
            in TABLE_CONTRACTS
        }

    except Exception as exc:
        raw_after = (
            _capture_named_sha256(
                raw_paths
            )
        )

        derived_after = (
            _capture_named_sha256(
                derived_paths
            )
        )

        try:
            _validate_snapshot_unchanged(
                raw_before,
                raw_after,
                label="Raw",
            )

            _validate_snapshot_unchanged(
                derived_before,
                derived_after,
                label="Derived",
            )

        except DerivedValidationError as immutability_error:
            raise immutability_error from exc

        raise

    raw_after = (
        _capture_named_sha256(
            raw_paths
        )
    )

    derived_after = (
        _capture_named_sha256(
            derived_paths
        )
    )

    _validate_snapshot_unchanged(
        raw_before,
        raw_after,
        label="Raw",
    )

    _validate_snapshot_unchanged(
        derived_before,
        derived_after,
        label="Derived",
    )

    summary = DerivedValidationSummary(
        raw_rows=raw.row_count,
        raw_actual_pitches=(
            raw.actual_pitch_count
        ),
        raw_unique_pa=len(
            raw.pa_summary
        ),
        raw_unique_games=len(
            raw.game_context
        ),
        table_row_counts=(
            table_row_counts
        ),
        duplicate_key_counts=(
            duplicate_counts
        ),
        schema_error_counts={
            "missing_file": 0,
            "column_order": 0,
            "dtype": 0,
            "null_policy": 0,
        },
        cross_table_error_counts={
            category: 0
            for category
            in CROSS_ERROR_CATEGORIES
        },
        through_dates=through_dates,
        content_fingerprints=(
            fingerprints
        ),
        raw_sha256_before=(
            raw_before
        ),
        raw_sha256_after=(
            raw_after
        ),
        derived_sha256_before=(
            derived_before
        ),
        derived_sha256_after=(
            derived_after
        ),
    )

    _log_summary(
        summary
    )

    return summary


def main() -> None:
    """CLI 진입점."""
    configure_logging()

    args = parse_args()

    try:
        validate_derived_tables(
            raw_dir=args.raw_dir,
            derived_dir=(
                args.derived_dir
            ),
            seasons=(
                args.seasons
            ),
        )

    except DerivedValidationError as exc:
        LOGGER.error(
            "Derived Validation FAIL | %s",
            exc,
        )
        raise SystemExit(
            1
        ) from exc

    except Exception as exc:
        LOGGER.exception(
            "Derived Validation 중 예상하지 못한 오류가 발생했습니다."
        )
        raise SystemExit(
            1
        ) from exc


if __name__ == "__main__":
    main()