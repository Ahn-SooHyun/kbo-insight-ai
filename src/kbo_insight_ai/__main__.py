"""도움말과 읽기 전용 환경·설정 확인 진입점을 제공한다."""

import argparse
from collections.abc import Sequence
import json
import logging
import platform

from kbo_insight_ai.settings import SettingsError, load_settings


LOG_FORMAT = "%(levelname)s %(name)s: %(message)s"
LOGGER = logging.getLogger("kbo_insight_ai")


def main(argv: Sequence[str] | None = None) -> int:
    """명령을 해석하고 정상은 0, 설정 오류는 2를 반환한다."""
    parser = argparse.ArgumentParser(
        prog="kbo-insight",
        description="KBO Insight AI의 최소 환경·설정 확인",
    )
    parser.add_argument(
        "--project-root",
        metavar="PATH",
        help="프로젝트 기준 절대 경로",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "check",
        help="설정과 경로를 읽기 전용으로 확인",
    )
    args = parser.parse_args(argv)

    try:
        settings = load_settings(args.project_root)
    except SettingsError as exc:
        logging.basicConfig(level="ERROR", format=LOG_FORMAT)
        LOGGER.error("설정 확인 실패: %s", exc)
        return 2

    logging.basicConfig(level=settings.log_level, format=LOG_FORMAT)
    LOGGER.info("환경·설정 확인 완료")

    result: dict[str, str | int] = {
        "python": platform.python_version(),
        "project_root": str(settings.project_root),
        "raw_dir": str(settings.raw_dir),
        "interim_dir": str(settings.interim_dir),
        "processed_dir": str(settings.processed_dir),
        "models_dir": str(settings.models_dir),
        "results_dir": str(settings.results_dir),
        "cache_dir": str(settings.cache_dir),
        "seed": settings.seed,
        "log_level": settings.log_level,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())