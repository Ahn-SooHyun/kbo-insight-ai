"""정규시즌을 순서대로 검증하고 비교 결과만 저장하는 M1 실행 명령."""

import argparse
from collections.abc import Sequence
from contextlib import closing
from datetime import date
import json
from pathlib import Path
import platform
import sqlite3
import sys
from typing import TypedDict
from uuid import uuid4

from ....docs.models.game_outcome import model as m1
from ....docs.models.game_outcome import poisson
from .__main__ import BOUNDARIES, read_games, source
from ...datasets.published_sources import SourceError, sha256_file, write_json
from ...evaluation import game_outcome as evaluation
from ...settings import Settings, load_settings


BACKTEST_SEASONS = 3
BASELINE = "venue_only"
PROFILE = "profile_histogram"
POISSON = "poisson_regression"
VERSION = "m1-expanding-window-v1"


class Metrics(TypedDict):
    """한 비교 모델의 득점·결과·동점 보정 지표."""
    scores: dict[str, evaluation.ScoreMetrics]
    outcomes: evaluation.ResultMetrics
    mean_independent_tie: float
    mean_adjusted_tie: float
    omitted_joint_mass_bound: float


def season_folds(development: Sequence[m1.Game]) -> list[tuple[int, list[m1.Game], list[m1.Game]]]:
    """검증 시즌 시작보다 이른 경기만 매번 학습에 넣는다."""
    if any(game.day >= BOUNDARIES[1] for game in development):
        raise ValueError("M1 개발 백테스트에 최종 Test 경기 날짜가 포함됐습니다.")
    final_year = date.fromisoformat(BOUNDARIES[0]).year
    folds = []
    for year in range(final_year - BACKTEST_SEASONS + 1, final_year + 1):
        start = f"{year:04d}-01-01"
        end = f"{year + 1:04d}-01-01"
        train = [game for game in development if game.day < start]
        validation = [game for game in development if start <= game.day < end]
        if not train or not validation:
            raise ValueError(f"M1 {year}년 백테스트의 학습 또는 검증 경기가 비었습니다.")
        folds.append((year, train, validation))
    return folds


def venue_forecast(model: m1.Model, train: Sequence[m1.Game]) -> tuple[m1.Forecast, float]:
    """팀 변수를 사용하지 않는 홈·원정 기준에 동일한 동점 추정법을 적용한다."""
    home, away = (m1.pmf(model["site_counts"][side]) for side in m1.SIDES)
    weight = m1.fit_tie_weight([(m1.diagonal(home, away), len(train))], train)
    cap = model["cap"]
    tail = m1.TAIL_SHARE * 0.5 ** (cap + 1)
    return m1.combine(home, away, weight, 2 * tail, 4 * (cap + 2) * tail), weight


def metrics(games: Sequence[m1.Game], forecasts: Sequence[m1.Forecast]) -> Metrics:
    """기존 지표를 공통 표본과 동일한 산식으로 계산한다."""
    return {
        "scores": evaluation.score_metrics(games, [(p["home_pmf"], p["away_pmf"]) for p in forecasts]),
        "outcomes": evaluation.result_metrics(games, [p["result_probabilities"] for p in forecasts]),
        "mean_independent_tie": sum(p["joint"]["independent_tie_probability"] for p in forecasts) / len(games),
        "mean_adjusted_tie": sum(p["result_probabilities"]["tie"] for p in forecasts) / len(games),
        "omitted_joint_mass_bound": max(p["tail"]["omitted_joint_mass_bound"] for p in forecasts),
    }


def comparisons(results: dict[str, Metrics]) -> dict[str, dict[str, float]]:
    """비교값의 음수가 오류 감소를 뜻하도록 후보에서 기준 점수를 뺀다."""
    pairs = [(PROFILE, BASELINE)]
    if POISSON in results:
        pairs += [(POISSON, BASELINE), (POISSON, PROFILE)]
    deltas = {}
    for candidate, reference in pairs:
        first, second = results[candidate], results[reference]
        deltas[f"{candidate}_minus_{reference}"] = {
            "pooled_mae": first["scores"]["pooled"]["mae"] - second["scores"]["pooled"]["mae"],
            "pooled_rmse": first["scores"]["pooled"]["rmse"] - second["scores"]["pooled"]["rmse"],
            "pooled_crps": first["scores"]["pooled"]["crps"] - second["scores"]["pooled"]["crps"],
            "log_loss": first["outcomes"]["log_loss"] - second["outcomes"]["log_loss"],
            "brier": first["outcomes"]["brier"] - second["outcomes"]["brier"],
        }
    return deltas


def benchmark(development: Sequence[m1.Game], *, include_poisson: bool = False) -> dict[str, object]:
    """분할별 학습 결과와 모든 검증 경기 기준의 통합 비교를 생성한다."""
    names = [BASELINE, PROFILE, POISSON] if include_poisson else [BASELINE, PROFILE]
    combined: dict[str, list[m1.Forecast]] = {name: [] for name in names}
    all_validation: list[m1.Game] = []
    seasons = []
    for year, train, validation in season_folds(development):
        model = m1.fit(train)
        baseline, weight = venue_forecast(model, train)
        forecasts = {
            BASELINE: [baseline] * len(validation),
            PROFILE: [m1.predict(model, game.inputs) for game in validation],
        }
        fit_details: dict[str, object] = {
            BASELINE: {"tie_weight": weight},
            PROFILE: {"tie_weight": model["tie_weight"], "league_mean": model["league_mean"]},
        }
        if include_poisson:
            candidate = poisson.fit(train, model["league_mean"], model["cap"])
            forecasts[POISSON] = [poisson.predict(candidate, game.inputs) for game in validation]
            fit_details[POISSON] = {
                "tie_weight": candidate.tie_weight,
                "coefficients": dict(zip(poisson.FEATURES, candidate.coefficients, strict=True)),
                "ridge": poisson.RIDGE,
            }

        scores = {name: metrics(validation, rows) for name, rows in forecasts.items()}
        seasons.append({
            "year": year,
            "training_games": len(train),
            "training_first_day": min(game.day for game in train),
            "training_last_day": max(game.day for game in train),
            "validation_games": len(validation),
            "validation_first_day": min(game.day for game in validation),
            "validation_last_day": max(game.day for game in validation),
            "validation_ties": sum(game.outcome == "tie" for game in validation),
            "fit": fit_details,
            "metrics": scores,
            "deltas": comparisons(scores),
        })
        all_validation.extend(validation)
        for name, rows in forecasts.items():
            combined[name].extend(rows)

    pooled = {name: metrics(all_validation, rows) for name, rows in combined.items()}
    return {
        "seasons": seasons,
        "pooled_games": len(all_validation),
        "pooled_metrics": pooled,
        "pooled_deltas": comparisons(pooled),
    }


def run_backtest(settings: Settings, run_id: str, *, include_poisson: bool) -> dict[str, object]:
    """개발 DB를 읽기 전용으로 사용하고 성공한 비교 결과만 저장한다."""
    database, provenance = source(settings, run_id)
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("PRAGMA query_only = ON")
        games, ledger = read_games(db)

    result = benchmark(games["train"] + games["validation"], include_poisson=include_poisson)
    run = uuid4().hex
    result_dir = settings.results_dir / "m1_backtest" / run
    result_dir.mkdir(parents=True, exist_ok=False)
    path = result_dir / "evaluation.json"
    temporary = result_dir / "evaluation.pending.json"
    record = {
        "backtest_version": VERSION,
        "backtest_run_id": run,
        "final_test": "held_out",
        "model_asset": "unchanged",
        "include_poisson": include_poisson,
        "python": platform.python_version(),
        "sqlite": sqlite3.sqlite_version,
        "source": provenance,
        "development_index": {key: dict(counts) for key, counts in ledger.items()},
        "backtest_argv": [
            "-m", "src.models.game_outcome.backtest", "--project-root", str(settings.project_root),
            "--run-id", run_id, *(["--include-poisson"] if include_poisson else []),
        ],
        "source_sha256": {
            "backtest.py": sha256_file(Path(__file__)),
            "model.py": sha256_file(Path(m1.__file__)),
            "poisson.py": sha256_file(Path(poisson.__file__)),
            "evaluation/game_outcome.py": sha256_file(Path(evaluation.__file__)),
        },
        "result": result,
    }
    write_json(temporary, record)
    temporary.rename(path)

    return {
        "backtest_run_id": run,
        "evaluation": str(path),
        "pooled_games": result["pooled_games"],
        "pooled_deltas": result["pooled_deltas"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """프로젝트 설정에 따라 비교 실행을 시작한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--include-poisson", action="store_true")
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.project_root)
        result = run_backtest(settings, args.run_id, include_poisson=args.include_poisson)
    except (SourceError, OSError, ValueError, KeyError, TypeError, ArithmeticError, sqlite3.DatabaseError) as exc:
        print(f"M1 백테스트 실패: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 3
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())