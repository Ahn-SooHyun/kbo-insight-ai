from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # 독립 실행에서도 src namespace package를 Project Root 기준으로 찾게 한다.
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.m1_dataset import M1DatasetBuildError, build_m1_dataset_files
from src.data.processed_contract import ProcessedContractError, read_json


LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "processed_dataset.json"


def parse_args() -> argparse.Namespace:
    """M1 Model-ready Dataset Builder의 CLI 인자를 파싱한다."""
    parser = argparse.ArgumentParser(
        description=(
            "Canonical games와 #23 Team Pregame Feature를 Home/Away로 결합해 "
            "M1 Model-ready Dataset을 생성합니다."
        )
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Processed Dataset 설정 JSON입니다. 기본값: {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument("--games-path", type=Path, default=None)
    parser.add_argument("--plate-appearances-path", type=Path, default=None)
    parser.add_argument("--team-features-path", type=Path, default=None)
    parser.add_argument("--team-feature-manifest-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--schema-path", type=Path, default=None)
    parser.add_argument("--quality-report-path", type=Path, default=None)
    parser.add_argument(
        "--skip-pa-audit",
        action="store_true",
        help=(
            "Terminal PA quality 재대사를 생략합니다. Canonical games 계약은 유지되지만 "
            "terminal PA warning/reconciliation 재검증은 수행하지 않습니다."
        ),
    )
    parser.add_argument(
        "--code-revision",
        type=str,
        default=None,
        help="Manifest에 기록할 Code Revision입니다. 생략하면 read-only git rev-parse를 시도합니다.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    """독립 실행 시 사용할 logging 형식을 설정한다."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def _resolve_project_path(value: str | Path) -> Path:
    """설정의 상대 경로를 Project Root 기준으로 해석한다."""
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _config_paths(config: dict[str, object]) -> dict[str, Path]:
    """processed_dataset.json의 M1 path 설정을 검증하고 Project Path로 변환한다."""
    m1 = config.get("m1")
    if not isinstance(m1, dict):
        raise M1DatasetBuildError("config.m1 object가 없습니다.")
    raw_paths = m1.get("paths")
    if not isinstance(raw_paths, dict):
        raise M1DatasetBuildError("config.m1.paths object가 없습니다.")

    required = (
        "games",
        "plate_appearances",
        "team_features",
        "team_feature_manifest",
        "output",
        "manifest",
        "schema",
        "quality_report",
    )
    missing = [key for key in required if key not in raw_paths]
    if missing:
        raise M1DatasetBuildError(f"config.m1.paths에 필수 경로가 없습니다: {missing}")
    return {key: _resolve_project_path(str(raw_paths[key])) for key in required}


def main() -> None:
    """CLI Entrypoint를 실행한다."""
    configure_logging()
    args = parse_args()
    try:
        config = read_json(args.config_path, label="processed_dataset config")
        paths = _config_paths(config)
        games_path = args.games_path or paths["games"]
        team_features_path = args.team_features_path or paths["team_features"]
        team_manifest_path = (
            args.team_feature_manifest_path or paths["team_feature_manifest"]
        )
        output_path = args.output_path or paths["output"]
        manifest_path = args.manifest_path or paths["manifest"]
        schema_path = args.schema_path or paths["schema"]
        quality_report_path = args.quality_report_path or paths["quality_report"]
        plate_appearances_path = None
        if not args.skip_pa_audit:
            plate_appearances_path = args.plate_appearances_path or paths["plate_appearances"]

        dataset, manifest = build_m1_dataset_files(
            config_path=args.config_path,
            games_path=games_path,
            team_features_path=team_features_path,
            team_feature_manifest_path=team_manifest_path,
            plate_appearances_path=plate_appearances_path,
            output_path=output_path,
            manifest_path=manifest_path,
            schema_path=schema_path,
            quality_report_path=quality_report_path,
            project_root=PROJECT_ROOT,
            code_revision=args.code_revision,
        )
    except (M1DatasetBuildError, ProcessedContractError):
        LOGGER.exception("M1 Model-ready Dataset 생성에 실패했습니다.")
        raise SystemExit(1)

    LOGGER.info(
        "M1 Dataset 생성 완료: rows=%d, duplicates=%d, excluded=%d, fingerprint=%s",
        len(dataset),
        int(dataset["game_pk"].duplicated().sum()),
        int(dataset["is_excluded"].fillna(False).sum()),
        manifest["output"]["content_fingerprint"],
    )
    LOGGER.info("Manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
