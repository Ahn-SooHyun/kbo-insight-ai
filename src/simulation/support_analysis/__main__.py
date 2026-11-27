"""예상 선발·팀 전력·일정 난이도의 통합 실행 진입점."""

import argparse
from collections.abc import Sequence
from contextlib import closing
from datetime import date, datetime, timezone
import json
import logging
from pathlib import Path
import sqlite3
from uuid import uuid4

from ...datasets.published_sources import (
    SourceError,
    as_row,
    read_json,
    sha256_file,
)
from ...settings import Settings, load_settings

from ..postseason.data import read_parent
from ..regular_season.data import (
    Scenario,
    fixture_to_dict,
    load_scenario,
    open_readonly,
    parse_cutoff,
    read_completed,
    verify_assets,
)

from . import (
    SUPPORTED_SEASON,
    VERSION,
    starter_selection,
)
from .analysis import analyze
from .data import read_starter_data
from .evaluation import evaluate_starters


LOGGER = logging.getLogger(__name__)

SOURCE_FILES = (
    "__init__.py",
    "__main__.py",
    "data.py",
    "analysis.py",
    "evaluation.py",
)


def source_hashes() -> dict[str, str]:
    """현재 보조 분석 패키지의 소스 지문을 기록한다."""

    root = Path(__file__).resolve().parent

    return {
        name: sha256_file(root / name)
        for name in SOURCE_FILES
    }


def verify_parent_run(
    settings: Settings,
    scenario: Scenario,
    parent_run: str,
) -> dict[str, object]:
    """기존 정규시즌 실행과 현재 보조 분석의 일치 여부를 검사한다."""

    _, parent = read_parent(
        settings,
        parent_run,
        scenario.rules.season,
        scenario.cutoff,
        scenario.completed,
        as_row(scenario.metadata["provenance"]),
    )

    directory = (
        settings.results_dir
        / "regular_season"
        / parent_run
    )

    scenario_path = directory / "scenario.json"

    digests = as_row(parent["files_sha256"])

    if (
        sha256_file(scenario_path)
        != digests["scenario.json"]
    ):
        raise ValueError("부모 시나리오 파일 지문 불일치")

    parent_scenario = as_row(
        read_json(scenario_path)
    )

    expected = [
        fixture_to_dict(fixture)
        for fixture in scenario.remaining
    ]

    if parent_scenario["remaining"] != expected:
        raise ValueError("부모 실행 잔여 일정 불일치")

    metadata = as_row(parent["metadata"])

    if (
        metadata["schedule"]
        != scenario.metadata["schedule"]
    ):
        raise ValueError("부모 실행 일정 근거 불일치")

    return {
        **parent,
        "result_directory": str(directory),
        "summary_path": str(
            directory / "summary.json"
        ),
        "relationship": (
            "same_assets_cutoff_completed_and_remaining"
        ),
        "simulation_recalculated": False,
    }


def run_analysis(
    settings: Settings,
    args: argparse.Namespace,
) -> dict[str, object]:
    """검증된 시즌 시나리오를 인수하여 보조 분석을 실행한다."""

    cutoff = parse_cutoff(
        args.as_of,
        args.season,
    )

    scenario = load_scenario(
        settings,
        args.run_id,
        args.asset,
        args.season,
        cutoff,
        args.schedule_snapshot,
    )

    parent = None

    if args.parent_run is not None:
        parent = verify_parent_run(
            settings,
            scenario,
            args.parent_run,
        )

    with (
        closing(
            open_readonly(scenario.standard_db)
        ) as standard,
        closing(
            open_readonly(scenario.model_db)
        ) as model,
    ):
        data = read_starter_data(
            standard,
            model,
            scenario.completed,
            args.season,
            cutoff,
        )

    result = analyze(
        scenario,
        data,
        game_id=(
            args.game_id
            if args.command == "game"
            else None
        ),
    )

    result["parent_regular_season"] = parent

    return result


def run_evaluation(
    settings: Settings,
    args: argparse.Namespace,
) -> dict[str, object]:
    """2025년 검증 범위만 사용하여 선발 순위를 비교한다."""

    standard_path, model_path, _, provenance = verify_assets(
        settings,
        args.run_id,
        args.asset,
    )

    end = date(
        SUPPORTED_SEASON + 1,
        1,
        1,
    )

    with (
        closing(
            open_readonly(standard_path)
        ) as standard,
        closing(
            open_readonly(model_path)
        ) as model,
    ):
        completed = read_completed(
            standard,
            model,
            SUPPORTED_SEASON,
            end,
        )

        data = read_starter_data(
            standard,
            model,
            completed,
            SUPPORTED_SEASON,
            end,
        )

        result = evaluate_starters(
            model,
            completed,
            data,
        )

    result["provenance"] = provenance

    return result


def execute(
    settings: Settings,
    args: argparse.Namespace,
) -> dict[str, object]:
    """실행 종류를 분리하고 공통 재현 정보를 추가한다."""

    if args.command == "evaluate":
        result = run_evaluation(
            settings,
            args,
        )

    else:
        result = run_analysis(
            settings,
            args,
        )

    return {
        **result,
        "status": "completed",
        "run_id": uuid4().hex,
        "version": VERSION,
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "source_sha256": source_hashes(),
        "starter_selection": starter_selection(),
    }


def save_result(
    settings: Settings,
    path: Path,
    result: dict[str, object],
) -> Path:
    """결과 디렉터리 안에 신규 JSON 파일만 생성한다."""

    target = path.resolve()
    root = settings.results_dir.resolve()

    if (
        not target.is_relative_to(root)
        or target.suffix.lower() != ".json"
    ):
        raise ValueError(
            "출력은 results 디렉터리 내부 JSON 파일이어야 합니다."
        )

    if target.exists():
        raise FileExistsError(
            f"기존 결과 파일을 덮어쓸 수 없습니다: {target}"
        )

    text = json.dumps(
        result,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )

    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with target.open(
        "x",
        encoding="utf-8",
        newline="\n",
    ) as stream:
        stream.write(text)
        stream.write("\n")

    return target


def build_parser() -> argparse.ArgumentParser:
    """기존 모델·시즌 CLI를 변경하지 않는 별도 진입점."""

    parser = argparse.ArgumentParser(
        description="예상 선발·팀 전력·일정 난이도 분석",
    )

    parser.add_argument(
        "--project-root",
        type=Path,
    )

    parser.add_argument(
        "--run-id",
        required=True,
    )

    parser.add_argument(
        "--asset",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="results 내부 신규 JSON 출력 경로",
    )

    commands = parser.add_subparsers(
        dest="command",
        required=True,
    )

    for name in ("game", "season"):
        command = commands.add_parser(name)

        command.add_argument(
            "--season",
            type=int,
            choices=(SUPPORTED_SEASON,),
            required=True,
        )

        command.add_argument(
            "--as-of",
            required=True,
            help="KST 날짜 YYYY-MM-DD, 해당일 자정 기준",
        )

        command.add_argument(
            "--schedule-snapshot",
            type=Path,
        )

        command.add_argument(
            "--parent-run",
            help="기존 정규시즌 전망 실행 ID",
        )

        if name == "game":
            command.add_argument(
                "--game-id",
                required=True,
            )

    commands.add_parser(
        "evaluate",
        help="2025년 검증 자료 선발 순위 비교",
    )

    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """실행 결과와 실패 상태를 구분하여 반환한다."""

    args = build_parser().parse_args(argv)

    try:
        settings = load_settings(args.project_root)

        logging.basicConfig(
            level=settings.log_level,
            format="%(levelname)s %(name)s: %(message)s",
        )

        result = execute(
            settings,
            args,
        )

        if args.output is not None:
            target = save_result(
                settings,
                args.output,
                result,
            )

            print(
                json.dumps(
                    {
                        "status": "completed",
                        "run_id": result["run_id"],
                        "scope": result["scope"],
                        "output": str(target),
                        "sha256": sha256_file(target),
                    },
                    ensure_ascii=False,
                    allow_nan=False,
                )
            )

        else:
            print(
                json.dumps(
                    result,
                    ensure_ascii=False,
                    allow_nan=False,
                )
            )

        return 0

    except (
        OSError,
        sqlite3.Error,
        ValueError,
        KeyError,
        TypeError,
        SourceError,
    ) as exc:
        LOGGER.error("보조 분석 실패: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
