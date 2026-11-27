"""포스트시즌 시리즈와 한국시리즈 우승 전망 CLI."""

import argparse
from collections.abc import Sequence
from dataclasses import asdict
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
    MAX_SERIES_GAMES,
    RESULT_DIRECTORY,
    SCENARIO_SCHEMA,
    SUMMARY_SCHEMA,
    VERSION,
)
from .data import load_inputs
from .engine import (
    FixedProfileSampler,
    IterationResult,
    simulate,
    summarize,
)
from .rules import (
    EXTRA_INNING_LIMIT,
    OPERATION_PUBLICATION,
    OPERATION_SOURCE,
    RULE_SOURCE,
    RULE_VERSION,
    get_rules,
)


LOGGER = logging.getLogger(__name__)


def write_iterations(
    path: Path,
    rows: Sequence[IterationResult],
) -> None:
    """반복별 결과를 순서가 유지된 JSON Lines로 신규 저장한다."""
    with path.open(
        "x",
        encoding="utf-8",
        newline="\n",
    ) as stream:
        for row in rows:
            stream.write(
                json.dumps(
                    row.to_dict(),
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
    parent_run: str | None,
    iterations: int | None,
    seed: int,
    archive_assumption: bool,
) -> dict[str, object]:
    """입력 검증, 시리즈 계산 및 실행 계보 저장을 수행한다."""
    inputs = load_inputs(
        settings=settings,
        run_id=run_id,
        asset=asset,
        season=season,
        as_of=as_of,
        parent_run=parent_run,
        iterations=iterations,
        archive_assumption=archive_assumption,
    )

    LOGGER.info(
        "포스트시즌 전망 시작: season=%s cutoff=%s "
        "mode=%s iterations=%s seed=%s",
        season,
        inputs.cutoff,
        inputs.metadata["mode"],
        len(inputs.draws),
        seed,
    )

    sampler = FixedProfileSampler(
        inputs.model,
        inputs.state,
        inputs.cutoff,
    )

    rows = simulate(
        season,
        inputs.draws,
        sampler,
        seed,
        inputs.archive,
    )

    summary = summarize(
        season,
        inputs.state.rules.teams,
        rows,
    )

    champion_valid = sum(
        row.champion is not None
        for row in rows
    )

    simulation_status = (
        "unsupported"
        if champion_valid == 0
        else "partial"
        if champion_valid < len(rows)
        else "resolved"
    )

    rules = {
        "version": RULE_VERSION,
        "season": season,
        "source": RULE_SOURCE,
        "operation_source": OPERATION_SOURCE,
        "operation_publication": OPERATION_PUBLICATION,
        "extra_inning_limit": EXTRA_INNING_LIMIT,
        "max_simulation_series_games": MAX_SERIES_GAMES,
        "stages": [
            asdict(rule)
            for rule in get_rules(season)
        ],
        "extra_games_policy": (
            "replay_tied_home_after_base_schedule;"
            "multiple_distinct_venues_unresolved"
        ),
        "wildcard_advantage": (
            "higher_advances_on_one_win_or_one_tie"
        ),
    }

    scenario = {
        "schema": SCENARIO_SCHEMA,
        "metadata": inputs.metadata,
        "rules": rules,
        "seed": seed,
        "iterations": len(rows),
        "fixed_matchup_probabilities": {
            f"{home}:{away}": probabilities
            for (home, away), probabilities
            in sorted(sampler.cache.items())
        },
    }

    simulation_id = uuid4().hex

    directory = (
        settings.results_dir
        / RESULT_DIRECTORY
        / simulation_id
    )

    directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    scenario_path = directory / "scenario.json"
    summary_path = directory / "summary.json"
    iterations_path = directory / "iterations.jsonl"
    manifest_path = directory / "manifest.json"

    write_json(scenario_path, scenario)
    write_json(summary_path, summary)
    write_iterations(iterations_path, rows)

    outputs = {
        "scenario": scenario_path,
        "summary": summary_path,
        "iterations": iterations_path,
    }

    files = {
        name: {
            "path": str(path),
            "sha256": sha256_file(path),
        }
        for name, path in outputs.items()
    }

    source_root = Path(__file__).resolve().parent

    source_hashes = {
        name: sha256_file(source_root / name)
        for name in (
            "__init__.py",
            "__main__.py",
            "data.py",
            "engine.py",
            "rules.py",
        )
    }

    # 모든 결과 생성과 해시 계산이 끝난 경우에만 완료
    # 매니페스트를 저장한다. 미결정 확률과 실행 성공을 구별한다.
    manifest = {
        "status": "completed",
        "simulation_status": simulation_status,
        "run_id": simulation_id,
        "version": VERSION,
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "season": season,
        "as_of_kst": inputs.metadata["as_of_kst"],
        "seed": seed,
        "iterations": len(rows),
        "seed_resolved": summary["seed_resolved"],
        "seed_unresolved": summary["seed_unresolved"],
        "champion_valid": champion_valid,
        "champion_unresolved": len(rows) - champion_valid,
        "python": platform.python_version(),
        "sqlite": sqlite3.sqlite_version,
        "schemas": {
            "scenario": SCENARIO_SCHEMA,
            "summary": SUMMARY_SCHEMA,
            "iterations": ITERATION_SCHEMA,
        },
        "metadata": inputs.metadata,
        "rules": rules,
        "source_sha256": source_hashes,
        "random_generator": "random.Random",
        "iteration_seed_policy": "seed:iteration_id",
        "cache_profiles": len(sampler.cache),
        "formal_backtest_supported": False,
        "final_test_used": False,
        "files": files,
    }

    write_json(manifest_path, manifest)

    LOGGER.info(
        "포스트시즌 계산 종료: run_id=%s "
        "valid=%s unresolved=%s status=%s",
        simulation_id,
        champion_valid,
        len(rows) - champion_valid,
        simulation_status,
    )

    return {
        "status": "completed",
        "simulation_status": simulation_status,
        "run_id": simulation_id,
        "season": season,
        "as_of_kst": inputs.metadata["as_of_kst"],
        "seed": seed,
        "iterations": len(rows),
        "seed_resolved": summary["seed_resolved"],
        "seed_unresolved": summary["seed_unresolved"],
        "champion_valid": champion_valid,
        "champion_unresolved": len(rows) - champion_valid,
        "result_dir": str(directory),
        "manifest": str(manifest_path),
        "scenario": str(scenario_path),
        "summary": str(summary_path),
        "iterations_path": str(iterations_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """성공 0, 입력·실행 실패 2, 사용자 중단 3을 반환한다."""
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
        help="포스트시즌 시리즈·우승 확률 계산",
    )

    simulation.add_argument(
        "--run-id",
        required=True,
        help="모델 자료 실행 ID",
    )

    simulation.add_argument(
        "--asset",
        required=True,
        type=Path,
        help="기존 M1 모델 자산",
    )

    simulation.add_argument(
        "--season",
        required=True,
        type=int,
    )

    simulation.add_argument(
        "--as-of",
        required=True,
        help="KST YYYY-MM-DD 자정 기준",
    )

    simulation.add_argument(
        "--parent-run",
        help="선행 정규시즌 전망 실행 ID",
    )

    simulation.add_argument(
        "--iterations",
        type=int,
        help="고정 시드 원천 판본 재생 모드의 반복 횟수",
    )

    simulation.add_argument(
        "--seed",
        type=int,
    )

    simulation.add_argument(
        "--archive-assumption",
        action="store_true",
        help="공식 기록 미검증 원천 판본 재생을 명시적으로 선택",
    )

    args = parser.parse_args(argv)

    try:
        settings = load_settings(args.project_root)

        logging.basicConfig(
            level=settings.log_level,
            format="%(levelname)s %(name)s: %(message)s",
        )

        seed = (
            settings.seed
            if args.seed is None
            else args.seed
        )

        iterations = args.iterations

        if (
            args.parent_run is None
            and iterations is None
        ):
            iterations = DEFAULT_ITERATIONS

        result = run(
            settings=settings,
            run_id=args.run_id,
            asset=args.asset,
            season=args.season,
            as_of=args.as_of,
            parent_run=args.parent_run,
            iterations=iterations,
            seed=seed,
            archive_assumption=args.archive_assumption,
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
            "포스트시즌 전망 실행 실패: %s",
            exc,
        )
        return 2

    except KeyboardInterrupt:
        LOGGER.warning("포스트시즌 전망 실행 중단")
        return 3

    return 0


if __name__ == "__main__":
    sys.exit(main())