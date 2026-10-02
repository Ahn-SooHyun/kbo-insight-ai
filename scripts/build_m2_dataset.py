from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    # 독립 실행에서도 src namespace package를 Project Root 기준으로 찾게 한다.
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from src.data.m2_dataset import (
    M2DatasetBuildError,
    RAW_SEASONS,
    build_m2_dataset_files,
)
from src.data.processed_contract import (
    ProcessedContractError,
    read_json,
)


LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "processed_dataset.json"
)


def parse_args() -> argparse.Namespace:
    """M2 Model-ready Dataset Builder CLI 인자를 파싱한다."""
    parser = argparse.ArgumentParser(
        description=(
            "Canonical PA와 동일 fixed revision Raw 첫 Row에서 "
            "PA-start 선수·Context를 복원하고 #24 Player Historical Feature를 "
            "결합해 M2 Matchup Dataset을 생성합니다."
        )
    )

    parser.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=(
            "Processed Dataset 설정 JSON입니다. "
            f"기본값: {DEFAULT_CONFIG_PATH}"
        ),
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--raw-manifest-path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--raw-schema-path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--plate-appearances-path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--batting-path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--pitching-path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--schema-path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--quality-report-path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--code-revision",
        type=str,
        default=None,
        help=(
            "Manifest에 기록할 Code Revision입니다. "
            "생략하면 read-only git rev-parse를 시도합니다."
        ),
    )

    return parser.parse_args()


def configure_logging() -> None:
    """독립 실행용 logging을 설정한다."""
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
    )


def _resolve_project_path(
    value: str | Path,
) -> Path:
    """설정의 상대 경로를 Project Root 기준으로 해석한다."""
    path = Path(
        value
    )
    if path.is_absolute():
        return path

    return (
        PROJECT_ROOT
        / path
    )


def _config_paths(
    config: dict[str, object],
) -> dict[str, Path]:
    """config.m2.paths를 검증하고 Project 절대 경로로 변환한다."""
    m2 = config.get(
        "m2"
    )
    if not isinstance(
        m2,
        dict,
    ):
        raise M2DatasetBuildError(
            "config.m2 object가 없습니다."
        )

    raw_paths = m2.get(
        "paths"
    )
    if not isinstance(
        raw_paths,
        dict,
    ):
        raise M2DatasetBuildError(
            "config.m2.paths object가 없습니다."
        )

    required = (
        "raw_dir",
        "raw_manifest",
        "raw_schema",
        "plate_appearances",
        "player_game_batting",
        "player_game_pitching",
        "output",
        "manifest",
        "schema",
        "quality_report",
    )

    missing = [
        key
        for key in required
        if key not in raw_paths
    ]
    if missing:
        raise M2DatasetBuildError(
            "config.m2.paths에 필수 경로가 없습니다: "
            f"{missing}"
        )

    return {
        key: _resolve_project_path(
            str(
                raw_paths[key]
            )
        )
        for key in required
    }


def main() -> None:
    """M2 Dataset CLI Entrypoint를 실행한다."""
    configure_logging()

    args = parse_args()

    try:
        config = read_json(
            args.config_path,
            label="processed_dataset config",
        )
        paths = _config_paths(
            config
        )

        raw_dir = (
            args.raw_dir
            or paths["raw_dir"]
        )
        raw_manifest_path = (
            args.raw_manifest_path
            or paths["raw_manifest"]
        )
        raw_schema_path = (
            args.raw_schema_path
            or paths["raw_schema"]
        )
        plate_appearances_path = (
            args.plate_appearances_path
            or paths["plate_appearances"]
        )
        batting_path = (
            args.batting_path
            or paths["player_game_batting"]
        )
        pitching_path = (
            args.pitching_path
            or paths["player_game_pitching"]
        )
        output_path = (
            args.output_path
            or paths["output"]
        )
        manifest_path = (
            args.manifest_path
            or paths["manifest"]
        )
        schema_path = (
            args.schema_path
            or paths["schema"]
        )
        quality_report_path = (
            args.quality_report_path
            or paths["quality_report"]
        )

        raw_paths = {
            season: (
                raw_dir
                / f"{season}.parquet"
            )
            for season in RAW_SEASONS
        }

        dataset, manifest = build_m2_dataset_files(
            config_path=args.config_path,
            raw_manifest_path=raw_manifest_path,
            raw_schema_path=raw_schema_path,
            raw_paths=raw_paths,
            plate_appearances_path=plate_appearances_path,
            batting_path=batting_path,
            pitching_path=pitching_path,
            output_path=output_path,
            manifest_path=manifest_path,
            schema_path=schema_path,
            quality_report_path=quality_report_path,
            project_root=PROJECT_ROOT,
            code_revision=args.code_revision,
        )

    except (
        M2DatasetBuildError,
        ProcessedContractError,
    ):
        LOGGER.exception(
            "M2 Model-ready Dataset 생성에 실패했습니다."
        )
        raise SystemExit(
            1
        )

    LOGGER.info(
        (
            "M2 Dataset 생성 완료: "
            "rows=%d, supervised=%d, excluded=%d, pending=%d, "
            "fingerprint=%s"
        ),
        len(
            dataset
        ),
        int(
            dataset[
                "supervised_usable"
            ]
            .fillna(False)
            .sum()
        ),
        int(
            dataset[
                "is_excluded"
            ]
            .fillna(False)
            .sum()
        ),
        int(
            dataset[
                "is_censored"
            ]
            .fillna(False)
            .sum()
        ),
        manifest[
            "output"
        ][
            "content_fingerprint"
        ],
    )
    LOGGER.info(
        "Manifest: %s",
        manifest_path,
    )


if __name__ == "__main__":
    main()