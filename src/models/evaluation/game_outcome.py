"""경기 결과·득점의 Validation 기준 비교와 확률 보정을 계산한다."""

from collections import Counter
from collections.abc import Sequence
from math import fsum, log, sqrt
from typing import TypedDict

from ..models.game_outcome import model as m1


LOG_FLOOR = 1e-15
BINS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


class ScoreMetrics(TypedDict):
    team_rows: int
    mae: float
    rmse: float
    crps: float


class ResultMetrics(TypedDict):
    games: int
    log_loss: float
    brier: float
    calibration: dict[str, object]


def crps(pmf: m1.PMF, observed: int) -> float:
    """비음수 정수의 CDF 제곱오차와 지원 범위 밖 정답 항을 합한다."""
    total = 0.0
    cumulative = 0.0
    for runs, mass in enumerate(pmf[:-1]):
        cumulative += mass
        total += (cumulative - (runs >= observed)) ** 2
    return total + max(0, observed - len(pmf) + 1)


def score_metrics(
    games: Sequence[m1.Game], predictions: Sequence[tuple[m1.PMF, m1.PMF]],
) -> dict[str, ScoreMetrics]:
    """홈·원정·양 팀 행 통합으로 기대득점과 분포 점수를 집계한다."""
    rows: dict[str, list[tuple[float, float]]] = {side: [] for side in m1.SIDES}
    for game, pair in zip(games, predictions, strict=True):
        for side, observed, pmf in zip(m1.SIDES, game.runs, pair):
            rows[side].append((m1.mean(pmf) - observed, crps(pmf, observed)))
    rows["pooled"] = rows["home"] + rows["away"]
    return {
        side: {
            "team_rows": len(values),
            "mae": fsum(abs(error) for error, _ in values) / len(values),
            "rmse": sqrt(fsum(error**2 for error, _ in values) / len(values)),
            "crps": fsum(score for _, score in values) / len(values),
        }
        for side, values in rows.items()
    }


def result_metrics(games: Sequence[m1.Game], predictions: Sequence[dict[str, float]]) -> ResultMetrics:
    """경기별 Log loss·Brier와 범주별 신뢰도 구간을 계산한다."""
    n = len(games)
    rows = list(zip(games, predictions, strict=True))
    calibration: dict[str, object] = {}
    for outcome in m1.OUTCOMES:
        values = [(probs[outcome], game.outcome == outcome) for game, probs in rows]
        bins = []
        for low, high in zip(BINS, BINS[1:]):
            selected = [(p, y) for p, y in values if low <= p < high or p == high == 1]
            count = len(selected)
            bins.append({
                "low": low, "high": high, "games": count,
                "mean_probability": fsum(p for p, _ in selected) / count if count else None,
                "observed_frequency": sum(y for _, y in selected) / count if count else None,
            })
        calibration[outcome] = {
            "mean_probability": fsum(p for p, _ in values) / n,
            "observed_frequency": sum(y for _, y in values) / n,
            "bins": bins,
        }
    return {
        "games": n,
        "log_loss": fsum(-log(max(LOG_FLOOR, probs[game.outcome])) for game, probs in rows) / n,
        "brier": fsum((probs[label] - (game.outcome == label)) ** 2
                      for game, probs in rows for label in m1.OUTCOMES) / n,
        "calibration": calibration,
    }


def evaluate(
    model: m1.Model, train: Sequence[m1.Game], validation: Sequence[m1.Game],
) -> dict[str, object]:
    """고정 기준과 후보를 동일 Validation 경기에서 비교한다."""
    home, away = (
        tuple(n / len(train) for n in model["site_counts"][side]) for side in m1.SIDES
    )
    counts = Counter(game.outcome for game in train)
    frequency = {label: counts[label] / len(train) for label in m1.OUTCOMES}
    forecasts = [m1.predict(model, game.inputs) for game in validation]
    score_base = score_metrics(validation, [(home, away)] * len(validation))
    score_model = score_metrics(validation, [(p["home_pmf"], p["away_pmf"]) for p in forecasts])
    result_base = result_metrics(validation, [frequency] * len(validation))
    result_model = result_metrics(validation, [p["result_probabilities"] for p in forecasts])
    n = len(validation)
    residuals = [(game.runs[0] - p["home_expected_runs"], game.runs[1] - p["away_expected_runs"])
                 for game, p in zip(validation, forecasts, strict=True)]
    return {
        "baseline": {
            "train_home_mean": m1.mean(home), "train_away_mean": m1.mean(away),
            "train_outcome_frequency": frequency, "scores": score_base, "outcomes": result_base,
        },
        "candidate": {"scores": score_model, "outcomes": result_model},
        "candidate_minus_baseline": {
            "scores": {
                side: {key: score_model[side][key] - score_base[side][key]
                       for key in ("mae", "rmse", "crps")}
                for side in (*m1.SIDES, "pooled")
            },
            "outcomes": {key: result_model[key] - result_base[key]
                         for key in ("log_loss", "brier")},
        },
        "joint_checks": {
            "train_ties": counts["tie"],
            "validation_ties": sum(game.outcome == "tie" for game in validation),
            "validation_mean_independent_tie": fsum(p["joint"]["independent_tie_probability"] for p in forecasts) / n,
            "validation_mean_adjusted_tie": fsum(p["result_probabilities"]["tie"] for p in forecasts) / n,
            "validation_max_omitted_joint_mass_bound": max(p["tail"]["omitted_joint_mass_bound"] for p in forecasts),
            "validation_mean_model_covariance": fsum(p["joint"]["score_covariance"] for p in forecasts) / n,
            "validation_residual_covariance": (
                fsum(h * a for h, a in residuals) / n
                - fsum(h for h, _ in residuals) * fsum(a for _, a in residuals) / n**2
            ),
        },
    }