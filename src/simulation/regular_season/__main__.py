"""정규시즌 전망의 실행·저장 및 개발 시나리오 진단 CLI."""

import argparse
from collections.abc import Sequence
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import platform
import sqlite3
import sys
from uuid import uuid4

from ...datasets.published_sources import (
    SourceError,
    sha256_file,
    write_json,
)
from ...settings import Settings, load_settings

from . import (
    DEFAULT_ITERATIONS,
    ITERATION_SCHEMA,
    SCENARIO_SCHEMA,
    SUMMARY_SCHEMA,
    VERSION,
)
from .data import (
    fixture_to_dict,
    load_scenario,
    parse_cutoff,
    read_diagnostic_truth,
)
from .engine import (
    Iteration,
    M1Sampler,
    build_state,
    diagnostic_metrics,
    simulate,
    summarize,
    team_result,
)
from .rules import TEAM_NAMES, resolve_standings


LOGGER = logging.getLogger(__name__)

RESULT_DIRECTORY = "regular_season"


def write_iterations(
    path: Path,
    rows: Sequence[Iteration],
) -> None:
    """반복별 공동 결과를 JSON Lines 형식으로 신규 저장한다."""

    with path.open(
        "x",
        encoding="utf-8",
        newline="\n",
    ) as stream:

        for row in rows:
            stream.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )


def run(
    settings: Settings,
    run_id: str,
    asset: Path,
    season: int,
    as_of: str,
    iterations: int,
    seed: int,
    schedule_snapshot: Path | None,
    diagnostic: bool,
) -> dict[str, object]:
    """인수 자료 검증부터 반복별 출력 저장까지 실행한다."""

    if type(iterations) is not int or iterations <= 0:
        raise ValueError("반복 횟수는 양의 정수여야 합니다.")

    cutoff = parse_cutoff(as_of, season)

    scenario = load_scenario(
        settings,
        run_id,
        asset,
        season,
        cutoff,
        schedule_snapshot,
    )

    rules = scenario.rules

    LOGGER.info(
        "정규시즌 전망 시작: season=%s cutoff=%s completed=%s "
        "remaining=%s iterations=%s seed=%s",
        season,
        as_of,
        len(scenario.completed),
        len(scenario.remaining),
        iterations,
        seed,
    )

    initial = build_state(
        rules,
        scenario.completed,
    )

    fixed = resolve_standings(initial)

    scenario_output = {
        "schema": SCENARIO_SCHEMA,
        "metadata": scenario.metadata,
        "current_teams": {
            str(team): {
                "team_id": team,
                "name": TEAM_NAMES[team],
                **team_result(
                    initial,
                    team,
                    fixed[team],
                ),
            }
            for team in rules.teams
        },
        "completed": [
            {
                **fixture_to_dict(game.fixture),
                "home_runs": game.home_runs,
                "away_runs": game.away_runs,
            }
            for game in scenario.completed
        ],
        "remaining": [
            fixture_to_dict(game)
            for game in scenario.remaining
        ],
    }

    sampler = M1Sampler(scenario.model)

    rows = simulate(
        rules,
        cutoff,
        scenario.completed,
        scenario.remaining,
        sampler,
        iterations,
        seed,
    )

    summary = summarize(rules, rows)

    diagnostic_output: dict[str, object] | None = None
    baseline_rows: list[Iteration] | None = None

    if diagnostic:
        if not scenario.remaining:
            raise ValueError(
                "개발 예측 진단에는 잔여 경기가 필요합니다."
            )

        LOGGER.info("개발 진단용 기준 계산 시작")

        baseline_rows = simulate(
            rules,
            cutoff,
            scenario.completed,
            scenario.remaining,
            M1Sampler(
                scenario.model,
                venue_only=True,
            ),
            iterations,
            seed,
        )

        baseline_summary = summarize(
            rules,
            baseline_rows,
        )

        # 실제 미래 정답은 두 시뮬레이션이 모두 끝난 후 읽는다.
        actual = read_diagnostic_truth(scenario)

        diagnostic_output = {
            "mode": "retrospective_development_diagnostic",
            "formal_backtest": False,
            "final_test_used": False,
            "selection_use": "prohibited",
            "seed": seed,
            "iterations": iterations,
            "candidate": diagnostic_metrics(
                rules,
                summary,
                actual,
            ),
            "baseline": diagnostic_metrics(
                rules,
                baseline_summary,
                actual,
            ),
            "baseline_method": (
                "venue_only_using_existing_m1_tie_weight"
            ),
            "baseline_summary": baseline_summary,
            "limitations": [
                "같은 시즌의 개발 진단이며 독립 평가가 아님",
                "기준 당시 운영 모델의 이용 가능 시점 미입증",
                "공개 일정 이력 미입증",
                "별도 기준 모델 재학습 없이 홈·원정 경험 분포만 사용",
            ],
        }

    simulation_id = uuid4().hex

    result_dir = (
        settings.results_dir
        / RESULT_DIRECTORY
        / simulation_id
    )

    result_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    scenario_path = result_dir / "scenario.json"
    summary_path = result_dir / "summary.json"
    iterations_path = result_dir / "iterations.jsonl"
    manifest_path = result_dir / "manifest.json"

    write_json(
        scenario_path,
        scenario_output,
    )

    write_json(
        summary_path,
        summary,
    )

    write_iterations(
        iterations_path,
        rows,
    )

    output_paths = {
        "scenario": scenario_path,
        "summary": summary_path,
        "iterations": iterations_path,
    }

    if diagnostic_output is not None and baseline_rows is not None:
        diagnostic_path = result_dir / "diagnostic.json"

        baseline_path = (
            result_dir / "baseline_iterations.jsonl"
        )

        write_json(
            diagnostic_path,
            diagnostic_output,
        )

        write_iterations(
            baseline_path,
            baseline_rows,
        )

        output_paths.update({
            "diagnostic": diagnostic_path,
            "baseline_iterations": baseline_path,
        })

    files = {
        name: {
            "path": str(path),
            "sha256": sha256_file(path),
        }
        for name, path in output_paths.items()
    }

    package = Path(__file__).resolve().parent

    source_hashes = {
        name: sha256_file(package / name)
        for name in (
            "__init__.py",
            "__main__.py",
            "rules.py",
            "data.py",
            "engine.py",
        )
    }

    # 완료 매니페스트는 모든 산출물 생성과 해시 계산 후에만 저장한다.
    manifest = {
        "status": "completed",
        "run_id": simulation_id,
        "version": VERSION,
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "season": season,
        "as_of_kst": scenario.metadata["as_of_kst"],
        "seed": seed,
        "iterations": iterations,
        "python": platform.python_version(),
        "sqlite": sqlite3.sqlite_version,
        "source_sha256": source_hashes,
        "schemas": {
            "scenario": SCENARIO_SCHEMA,
            "summary": SUMMARY_SCHEMA,
            "iterations": ITERATION_SCHEMA,
        },
        "metadata": scenario.metadata,
        "prediction_method": (
            "existing_m1_dynamic_daily_history_joint_score_sampling"
        ),
        "random_generator": "random.Random",
        "processing_order": "date_game_number_game_id",
        "same_day_history_policy": "exclude_all_same_day_results",
        "cache_profiles": len(sampler.cache),
        "formal_backtest_supported": False,
        "development_diagnostic": diagnostic_output is not None,
        "final_test_used": False,
        "files": files,
    }

    write_json(
        manifest_path,
        manifest,
    )

    LOGGER.info(
        "정규시즌 전망 완료: run_id=%s",
        simulation_id,
    )

    # 반복 횟수와 반복별 파일 경로는 의미와 타입이 다르다.
    # 산출물 경로 딕셔너리를 그대로 병합하지 않고 명시적으로
    # 반환하여 기존 실행 정보가 덮어써지는 문제를 방지한다.
    result: dict[str, object] = {
        "status": "completed",
        "run_id": simulation_id,
        "season": season,
        "as_of_kst": scenario.metadata["as_of_kst"],
        "iterations": iterations,
        "seed": seed,
        "completed_games": len(scenario.completed),
        "remaining_games": len(scenario.remaining),
        "missing_games": scenario.metadata["missing_games"],
        "result_dir": str(result_dir),
        "manifest": str(manifest_path),
        "scenario": str(scenario_path),
        "summary": str(summary_path),
        "iterations_path": str(iterations_path),
    }

    if "diagnostic" in output_paths:
        result["diagnostic"] = str(
            output_paths["diagnostic"]
        )
        result["baseline_iterations"] = str(
            output_paths["baseline_iterations"]
        )

    return result


def main(argv: Sequence[str] | None = None) -> int:
    """성공 0, 입력·실행 오류 2, 사용자 중단 3을 반환한다."""

    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--project-root",
        type=Path,
    )

    commands = parser.add_subparsers(
        dest="command",
        required=True,
    )

    simulation = commands.add_parser(
        "simulate",
        help="정규시즌 최종 순위 및 진출 전망 계산",
    )

    simulation.add_argument(
        "--run-id",
        required=True,
    )

    simulation.add_argument(
        "--asset",
        type=Path,
        required=True,
    )

    simulation.add_argument(
        "--season",
        type=int,
        required=True,
    )

    simulation.add_argument(
        "--as-of",
        required=True,
        help="KST 기준 날짜 YYYY-MM-DD, 당일 자정 적용",
    )

    simulation.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
    )

    simulation.add_argument(
        "--seed",
        type=int,
    )

    simulation.add_argument(
        "--schedule-snapshot",
        type=Path,
        help="자동 확보한 외부 잔여 일정 JSON 스냅샷",
    )

    simulation.add_argument(
        "--diagnostic",
        action="store_true",
        help="실제 완료 시즌을 이용한 별도 개발 진단",
    )

    args = parser.parse_args(argv)

    try:
        settings = load_settings(
            args.project_root,
        )

        logging.basicConfig(
            level=settings.log_level,
            format="%(levelname)s %(name)s: %(message)s",
        )

        seed = (
            settings.seed
            if args.seed is None
            else args.seed
        )

        result = run(
            settings=settings,
            run_id=args.run_id,
            asset=args.asset,
            season=args.season,
            as_of=args.as_of,
            iterations=args.iterations,
            seed=seed,
            schedule_snapshot=args.schedule_snapshot,
            diagnostic=args.diagnostic,
        )

        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        )

    except (
        OSError,
        SourceError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        OverflowError,
        sqlite3.DatabaseError,
    ) as exc:

        LOGGER.error(
            "정규시즌 전망 실행 실패: %s",
            exc,
        )

        return 2

    except KeyboardInterrupt:
        LOGGER.warning("정규시즌 전망 실행 중단")
        return 3

    return 0


if __name__ == "__main__":
    sys.exit(main())