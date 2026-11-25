"""환경 확인과 명시적인 원본 수집·표준화·검사 명령을 제공한다."""

import argparse
from collections.abc import Sequence
import json
import logging
from pathlib import Path
import platform

from .settings import SettingsError, load_settings


LOG_FORMAT = "%(levelname)s %(name)s: %(message)s"
LOGGER = logging.getLogger("kbo_insight_ai")


def main(argv: Sequence[str] | None = None) -> int:
    """정상 0, 설정 오류 2, 수집 실패 3, 저장 실패 4, 검사 실패 5."""
    parser = argparse.ArgumentParser(
        prog="kbo-insight",
        description="KBO Insight AI의 환경 확인과 원본 수집·표준화",
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

    sample = commands.add_parser(
        "collect-sample",
        help="출처별 제한된 원본 표본과 이력을 보존",
    )
    sample.add_argument("--source", choices=("naver", "hf"), default="naver")
    sample.add_argument("--game-id", help="Naver 원본에서 확인한 경기 ID")
    sample.add_argument("--inning", type=int, choices=range(1, 13))

    inspection_parser = commands.add_parser(
        "inspect-sample",
        help="저장된 HF 원본의 필드·행 사례를 읽기 전용 검사",
    )
    inspection_parser.add_argument("--run-id", required=True)
    inspection_parser.add_argument("--game-id", default="20230401HHWO02023")

    for name, description in (
        ("collect-files", "승인 명세의 고정 배포 파일 수집"),
        ("update-files", "고정 배포 파일 갱신 및 판본 비교"),
    ):
        command = commands.add_parser(name, help=description)
        command.add_argument("--manifest", type=Path, required=True)

    resume = commands.add_parser(
        "resume-files",
        help="이전 실행의 미확보 파일 수집 재개",
    )
    resume.add_argument("--run-id", required=True)

    report = commands.add_parser(
        "report-files",
        help="수집 대상과 실제 원본을 읽기 전용 대조",
    )
    report.add_argument("--run-id", required=True)

    standardization = commands.add_parser(
        "standardize",
        help="저장된 불변 원본을 표준화하고 품질을 검사",
    )
    standardization.add_argument(
        "--as-of-run",
        help="지정 수집 실행까지의 판본을 채택",
    )
    standardization.add_argument(
        "--compare-run",
        help="이전 표준화 실행과 결과를 비교",
    )

    verification = commands.add_parser(
        "validate-standard",
        help="완료 표준 자료의 물리 행·집계·유효 후보 검증",
    )
    verification.add_argument(
        "--run-id",
        required=True,
    )
    verification.add_argument(
        "--compare-with",
        help="비교할 표준화 실행 ID:검증 ID",
    )

    args = parser.parse_args(argv)

    if args.command == "collect-sample":
        if args.source == "naver" and (
            args.game_id is None or args.inning is None
        ):
            parser.error("naver 수집에는 --game-id와 --inning이 필요합니다.")

        if args.source == "hf" and (
            args.game_id is not None or args.inning is not None
        ):
            parser.error(
                "hf 파일 수집에는 경기 ID·이닝 인자를 사용하지 않습니다."
            )

    try:
        settings = load_settings(args.project_root)

    except SettingsError as exc:
        logging.basicConfig(level="ERROR", format=LOG_FORMAT)
        LOGGER.error("설정 확인 실패: %s", exc)
        return 2

    logging.basicConfig(
        level=settings.log_level,
        format=LOG_FORMAT,
    )

    if args.command == "validate-standard":
        import sqlite3

        from .datasets.published_sources import SourceError
        from .validation.standard_data import StandardDataError
        from .validation.standard_quality import validate_standard

        try:
            result = validate_standard(
                settings,
                args.run_id,
                compare_with=args.compare_with,
            )

        except (
            SourceError,
            StandardDataError,
            ImportError,
            sqlite3.DatabaseError,
            KeyError,
            ValueError,
            TypeError,
        ) as exc:
            LOGGER.error("최종 표준 자료 검사 실패: %s", exc)
            return 5

        except OSError as exc:
            LOGGER.error("최종 표준 자료 입출력 실패: %s", exc)
            return 4

        except KeyboardInterrupt:
            LOGGER.warning("최종 표준 자료 검사 중단")
            return 3

        print(json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ))

        return 0

    if args.command == "standardize":
        from .datasets.published_sources import SourceError
        from .datasets.standardize import standardize
        from .validation.standard_data import StandardDataError

        try:
            result = standardize(
                settings,
                as_of_run=args.as_of_run,
                compare_run=args.compare_run,
            )

        except (SourceError, StandardDataError, ImportError) as exc:
            LOGGER.error("표준 데이터 검사 실패: %s", exc)
            return 5

        except OSError as exc:
            LOGGER.error("표준 데이터 입출력 실패: %s", exc)
            return 4

        except KeyboardInterrupt:
            LOGGER.warning("표준 데이터 구축 실행 중단")
            return 3

        print(json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ))

        return 0

    if args.command == "inspect-sample":
        from .validation.raw_sample import InspectionError, inspect_sample

        try:
            inspection_result = inspect_sample(
                settings,
                args.run_id,
                args.game_id,
            )

        except InspectionError as exc:
            LOGGER.error("원본 검사 실패: %s", exc)
            return 5

        print(json.dumps(
            inspection_result,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ))

        return 0

    if args.command == "collect-sample":
        # 명시적인 표본 수집 분기에서만 통신·쓰기 기능을 사용한다.
        from .ingestion.pilot import collect_hf_sample, collect_sample

        try:
            if args.source == "hf":
                run_dir, accepted = collect_hf_sample(settings)

            else:
                run_dir, accepted = collect_sample(
                    settings,
                    args.game_id,
                    args.inning,
                )

        except ValueError:
            LOGGER.error("수집 대상 오류: 공개 경기 ID와 이닝을 확인하세요.")
            return 2

        except OSError:
            LOGGER.error(
                "원본 또는 이력 저장 실패: 권한·공간·파일 상태를 확인하세요."
            )
            return 4

        print(json.dumps(
            {
                "run_dir": str(run_dir),
                "status": (
                    "artifact_verified"
                    if args.source == "hf"
                    else "structure_validated"
                ) if accepted else "failed",
                "semantic_validation": "not_performed",
            },
            ensure_ascii=False,
            indent=2,
        ))

        if not accepted:
            LOGGER.error("표본 수집 실패: 실행 폴더의 result.json을 확인하세요.")
            return 3

        LOGGER.info("원본 저장·재열람 및 출처별 최소 검사 완료")
        return 0

    if args.command in {
        "collect-files",
        "update-files",
        "resume-files",
        "report-files",
    }:
        from .ingestion.published import (
            ManifestError,
            collect_files,
            report_files,
        )

        try:
            if args.command == "report-files":
                result = report_files(settings, args.run_id)

            elif args.command == "resume-files":
                result = collect_files(
                    settings,
                    mode="resume",
                    run_id=args.run_id,
                )

            else:
                result = collect_files(
                    settings,
                    manifest_path=args.manifest,
                    mode=(
                        "update"
                        if args.command == "update-files"
                        else "collect"
                    ),
                )

        except ManifestError as exc:
            LOGGER.error("전체 수집 명세 오류: %s", exc)
            return 2

        except OSError:
            LOGGER.error("원본 또는 수집 이력 접근·저장 실패")
            return 4

        except KeyboardInterrupt:
            LOGGER.warning("수집 중단: 기존 실행 ID로 재개할 수 있습니다.")
            return 3

        print(json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ))

        if not result["file_collection_complete"]:
            LOGGER.warning("미확보·보류·실패 대상이 존재합니다.")
            return 3

        return 0

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