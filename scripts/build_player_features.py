from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # 독립 실행에서도 src namespace package를 Project Root 기준으로 찾게 한다.
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.processed_contract import (
    FEATURE_CATALOG_VERSION,
    PREDICTION_CONTRACT_VERSION,
    ProcessedContractError,
    assert_input_hashes_unchanged,
    calculate_input_hashes,
    content_fingerprint,
    invalidate_completion_marker,
    read_parquet,
    schema_manifest,
    validate_output_paths,
    write_json_atomic,
    write_parquet_atomic,
)
from src.features.player import (
    AUDIT_COLUMNS,
    BATTING_REQUIRED_SOURCE_COLUMNS,
    FEATURE_VERSION,
    OUTPUT_COLUMNS,
    PITCHING_REQUIRED_SOURCE_COLUMNS,
    REQUEST_KEY,
    ROLE_FEATURE_COLUMNS,
    ROLLING_AUDIT_COLUMNS,
    OUTS_QUALITY_AUDIT_COLUMNS,
    ROLLING_WINDOWS,
    PlayerFeatureBuildError,
    normalize_feature_output,
    build_player_pregame_features,
    normalize_requests,
    validate_feature_output,
)


LOGGER = logging.getLogger(__name__)

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
    / "processed"
    / "hf_kbo_pbp"
    / "features"
    / "player_pregame_features.parquet"
)


def parse_args() -> argparse.Namespace:
    """Player Pregame Historical Feature Builder의 CLI 인자를 파싱한다."""
    parser = argparse.ArgumentParser(
        description=(
            "명시적 (player_id, role, prediction_date) Request와 Canonical "
            "Player Game Fact를 분리해 Player Pregame Historical Feature를 생성합니다."
        )
    )
    parser.add_argument(
        "--requests-path",
        type=Path,
        required=True,
        help=(
            "중복 없는 (player_id, role, prediction_date) Request Parquet 경로입니다. "
            "Builder는 미래 출장 여부로 이 파일을 자동 생성하지 않습니다."
        ),
    )
    parser.add_argument(
        "--batting-path",
        type=Path,
        default=DEFAULT_BATTING_PATH,
        help=f"Canonical player_game_batting.parquet 경로입니다. 기본값: {DEFAULT_BATTING_PATH}",
    )
    parser.add_argument(
        "--pitching-path",
        type=Path,
        default=DEFAULT_PITCHING_PATH,
        help=f"Canonical player_game_pitching.parquet 경로입니다. 기본값: {DEFAULT_PITCHING_PATH}",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Player Feature Parquet 경로입니다. 기본값: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help=(
            "Manifest JSON 경로입니다. 생략하면 output과 같은 디렉터리에 "
            "player_pregame_features.manifest.json을 생성합니다."
        ),
    )
    return parser.parse_args()


def configure_logging() -> None:
    """독립 실행 시 사용할 logging 형식을 설정한다."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def _null_summary(frame: pd.DataFrame) -> dict[str, int]:
    """Manifest Audit용 컬럼별 null 수를 결정적인 컬럼 순서로 계산한다."""
    return {
        column: int(frame[column].isna().sum())
        for column in frame.columns
    }


def _role_row_counts(frame: pd.DataFrame) -> dict[str, int]:
    """Role별 Request 보존 Row 수를 Manifest에 기록한다."""
    return {
        role: int(frame["role"].eq(role).sum())
        for role in ("batting", "pitching")
    }


def _build_manifest(
    *,
    requests: pd.DataFrame,
    features: pd.DataFrame,
    requests_path: Path,
    batting_path: Path,
    pitching_path: Path,
    output_path: Path,
    input_hashes: Mapping[str, str],
    rolling_windows: Sequence[int],
) -> dict[str, object]:
    """Request/Source provenance와 결정적 Output identity를 Manifest로 구성한다."""
    request_fingerprint = content_fingerprint(
        requests,
        sort_columns=list(REQUEST_KEY),
    )
    output_fingerprint = content_fingerprint(
        features,
        sort_columns=list(REQUEST_KEY),
    )
    cold_start_count = int((~features["has_history"].fillna(False)).sum())

    return {
        "artifact": "player_pregame_features",
        "feature_version": FEATURE_VERSION,
        "prediction_contract_version": PREDICTION_CONTRACT_VERSION,
        "source_contract_version": FEATURE_CATALOG_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_status": "complete",
        "grain": list(REQUEST_KEY),
        "timestamp_resolution": "date",
        "historical_cutoff": "source.game_date < prediction_date",
        "same_day_policy": "exclude_all_same_date_source_results",
        "season_reset": True,
        "season_reset_basis": "prediction_date.year",
        "rolling_windows": list(rolling_windows),
        "rolling_boundary_policy": "include_entire_boundary_date_group",
        "key_columns": list(REQUEST_KEY),
        "audit_columns": list(
            AUDIT_COLUMNS + ROLLING_AUDIT_COLUMNS + OUTS_QUALITY_AUDIT_COLUMNS
        ),
        "role_feature_columns": {
            role: list(columns)
            for role, columns in ROLE_FEATURE_COLUMNS.items()
        },
        "consumer_allowlist_policy": {
            "source": "docs/feature_catalog.md",
            "feature_sets": ["baseline_v1", "extended_v1"],
            "selection_rule": "explicit_allowlist_only",
            "catalog_eligible_audit_columns": [
                "history_game_count",
                "has_history",
            ],
        },
        "request": {
            "path": str(requests_path),
            "sha256": input_hashes["requests"],
            "row_count": int(len(requests)),
            "content_fingerprint": request_fingerprint,
        },
        "inputs": {
            "player_game_batting": {
                "path": str(batting_path),
                "sha256": input_hashes["player_game_batting"],
                "required_columns": list(BATTING_REQUIRED_SOURCE_COLUMNS),
            },
            "player_game_pitching": {
                "path": str(pitching_path),
                "sha256": input_hashes["player_game_pitching"],
                "required_columns": list(PITCHING_REQUIRED_SOURCE_COLUMNS),
            },
        },
        "output": {
            "path": str(output_path),
            "row_count": int(len(features)),
            "role_row_counts": _role_row_counts(features),
            "cold_start_count": cold_start_count,
            "null_summary": _null_summary(features),
            "content_fingerprint": output_fingerprint,
            "schema": schema_manifest(features),
        },
    }


def build_player_feature_files(
    *,
    requests_path: Path,
    batting_path: Path = DEFAULT_BATTING_PATH,
    pitching_path: Path = DEFAULT_PITCHING_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    manifest_path: Path | None = None,
    rolling_windows: Sequence[int] = ROLLING_WINDOWS,
    project_root: Path = PROJECT_ROOT,
) -> tuple[pd.DataFrame, dict[str, object], Path]:
    """Request/Canonical 입력을 읽어 Feature Parquet과 Manifest를 안전하게 생성한다."""
    resolved_manifest_path = (
        manifest_path
        if manifest_path is not None
        else output_path.with_suffix(".manifest.json")
    )
    input_paths = {
        "requests": requests_path,
        "player_game_batting": batting_path,
        "player_game_pitching": pitching_path,
    }

    resolved_inputs = [path.resolve(strict=False) for path in input_paths.values()]
    if len(resolved_inputs) != len(set(resolved_inputs)):
        raise PlayerFeatureBuildError(
            "Request/Batting/Pitching 입력 경로는 서로 달라야 합니다."
        )

    validate_output_paths(
        input_paths=list(input_paths.values()),
        output_paths=[output_path, resolved_manifest_path],
        project_root=project_root,
    )
    # 이전 완료 Manifest를 먼저 제거해 실패한 재실행이 과거 완료 상태로 오인되지 않게 한다.
    invalidate_completion_marker(resolved_manifest_path)
    hashes_before = calculate_input_hashes(input_paths)

    raw_requests = read_parquet(requests_path, label="player feature requests")
    batting_source = read_parquet(batting_path, label="player_game_batting")
    pitching_source = read_parquet(pitching_path, label="player_game_pitching")
    normalized_requests = normalize_requests(raw_requests)

    features = build_player_pregame_features(
        normalized_requests,
        batting_source,
        pitching_source,
        rolling_windows=rolling_windows,
    )
    expected_fingerprint = content_fingerprint(
        features,
        sort_columns=list(REQUEST_KEY),
    )

    write_parquet_atomic(features, output_path)
    round_trip = read_parquet(output_path, label="player_pregame_features")
    round_trip = normalize_feature_output(round_trip)
    validate_feature_output(
        round_trip,
        normalized_requests,
        rolling_windows=rolling_windows,
    )
    actual_fingerprint = content_fingerprint(
        round_trip,
        sort_columns=list(REQUEST_KEY),
    )
    if actual_fingerprint != expected_fingerprint:
        raise PlayerFeatureBuildError(
            "Parquet round-trip 후 Player Feature content fingerprint가 변경되었습니다."
        )

    hashes_after = assert_input_hashes_unchanged(hashes_before, input_paths)
    manifest = _build_manifest(
        requests=normalized_requests,
        features=round_trip,
        requests_path=requests_path,
        batting_path=batting_path,
        pitching_path=pitching_path,
        output_path=output_path,
        input_hashes=hashes_after,
        rolling_windows=rolling_windows,
    )
    write_json_atomic(manifest, resolved_manifest_path)

    return round_trip, manifest, resolved_manifest_path


def log_summary(
    features: pd.DataFrame,
    manifest: Mapping[str, object],
    *,
    manifest_path: Path,
) -> None:
    """생성 결과의 Request 보존·Cold Start·Fingerprint 핵심값을 로그에 남긴다."""
    output = manifest["output"]
    LOGGER.info(
        (
            "Player Pregame Feature 생성 완료: rows=%d, key_duplicates=%d, "
            "batting=%d, pitching=%d, cold_start=%d, fingerprint=%s"
        ),
        len(features),
        int(features.duplicated(subset=list(REQUEST_KEY), keep=False).sum()),
        int(features["role"].eq("batting").sum()),
        int(features["role"].eq("pitching").sum()),
        int(output["cold_start_count"]),
        output["content_fingerprint"],
    )
    LOGGER.info("Manifest: %s", manifest_path)


def main() -> None:
    """CLI Entrypoint를 실행한다."""
    configure_logging()
    args = parse_args()
    try:
        features, manifest, manifest_path = build_player_feature_files(
            requests_path=args.requests_path,
            batting_path=args.batting_path,
            pitching_path=args.pitching_path,
            output_path=args.output_path,
            manifest_path=args.manifest_path,
        )
    except (PlayerFeatureBuildError, ProcessedContractError):
        LOGGER.exception("Player Pregame Feature 생성에 실패했습니다.")
        raise SystemExit(1)

    log_summary(
        features,
        manifest,
        manifest_path=manifest_path,
    )


if __name__ == "__main__":
    main()
