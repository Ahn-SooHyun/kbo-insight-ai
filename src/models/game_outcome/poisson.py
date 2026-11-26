"""정수 득점 분포를 만드는 Poisson 회귀 비교 후보."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import exp, fsum, log

from ....docs.models.game_outcome import model as m1


RIDGE = 0.01
FEATURE_LIMIT = 2.0
MIN_LOG_RATE = log(0.05)
MAX_LOG_RATE = log(40.0)
MIN_SUPPORT = 128
GRADIENT_TOLERANCE = 1e-7
MAX_ITERATIONS = 25
LINE_SEARCH_STEPS = 25
FEATURES = ("intercept", "home", "season_rf", "season_opponent_ra", "recent_rf", "recent_opponent_ra")
Row = tuple[tuple[float, ...], int]


@dataclass(frozen=True)
class Candidate:
    """분할 학습 자료에서 추정한 회귀계수와 동점 가중치."""
    coefficients: tuple[float, ...]
    league_mean: float
    cap: int
    tie_weight: float


def features(teams: Mapping[str, tuple[int, ...]], side: str, mean: float) -> tuple[float, ...]:
    """고정 순서로 홈 여부와 네 비율을 반환한다."""
    scale = max(1.0, mean)
    values = (
        max(-FEATURE_LIMIT, min(FEATURE_LIMIT, (rate - mean) / scale))
        for rate in m1.rates(teams, side, mean)
    )
    return (1.0, float(side == "home"), *values)


def objective(rows: Sequence[Row], weights: tuple[float, ...]) -> float:
    """관측 득점의 평균 음의 로그 우도에 L2 벌점을 더한다."""
    terms = []
    for x, y in rows:
        eta = sum(a * b for a, b in zip(x, weights, strict=True))
        if abs(eta) > 40:
            return float("inf")
        terms.append(exp(eta) - y * eta)
    return fsum(terms) / len(rows) + RIDGE * fsum(w * w for w in weights) / 2


def newton_step(rows: Sequence[Row], weights: tuple[float, ...]) -> tuple[float, tuple[float, ...]]:
    """기울기 크기와 Hessian을 이용한 Newton 방향을 계산한다."""
    width = len(weights)
    n = len(rows)
    gradient = [RIDGE * w for w in weights]
    hessian = [[0.0] * width for _ in weights]

    for x, y in rows:
        rate = exp(sum(a * b for a, b in zip(x, weights, strict=True)))
        residual = (rate - y) / n
        for i, xi in enumerate(x):
            gradient[i] += residual * xi
            factor = rate * xi / n
            for j in range(i + 1):
                hessian[i][j] += factor * x[j]

    norm = max(map(abs, gradient))
    if norm <= GRADIENT_TOLERANCE:
        return norm, tuple(gradient)
    for i in range(width):
        hessian[i][i] += RIDGE
        for j in range(i):
            hessian[j][i] = hessian[i][j]

    # L2 벌점으로 양의 정부호인 소규모 시스템을 가우스 소거로 푼다.
    for i in range(width):
        pivot = hessian[i][i]
        for j in range(i + 1, width):
            factor = hessian[j][i] / pivot
            for k in range(i + 1, width):
                hessian[j][k] -= factor * hessian[i][k]
            gradient[j] -= factor * gradient[i]

    step = [0.0] * width
    for i in reversed(range(width)):
        rest = sum(hessian[i][j] * step[j] for j in range(i + 1, width))
        step[i] = (gradient[i] - rest) / hessian[i][i]
    return norm, tuple(step)


def fit_coefficients(rows: Sequence[Row], mean: float) -> tuple[float, ...]:
    """학습 자료에서만 회귀계수를 추정하고 수렴 실패를 진단한다."""
    weights = (log(max(0.05, mean)), *(0.0 for _ in FEATURES[1:]))
    loss = objective(rows, weights)
    for _ in range(MAX_ITERATIONS):
        norm, direction = newton_step(rows, weights)
        if norm <= GRADIENT_TOLERANCE:
            return weights
        for trial in range(LINE_SEARCH_STEPS):
            step = 0.5**trial
            candidate = tuple(w - step * d for w, d in zip(weights, direction, strict=True))
            candidate_loss = objective(rows, candidate)
            if candidate_loss <= loss + 1e-12:
                weights, loss = candidate, candidate_loss
                break
        else:
            raise ValueError("M1 Poisson 회귀의 손실 감소 단계를 찾지 못했습니다.")
    raise ValueError("M1 Poisson 회귀가 반복 한도 내에서 수렴하지 않았습니다.")


def distribution(x: tuple[float, ...], weights: tuple[float, ...], cap: int) -> tuple[m1.PMF, float, float, float]:
    """득점 PMF, 절단 질량 상한, 절단 1차 모멘트 상한 및 평균을 반환한다."""
    eta = sum(a * b for a, b in zip(x, weights, strict=True))
    rate = exp(max(MIN_LOG_RATE, min(MAX_LOG_RATE, eta)))
    values = [exp(-rate)]
    for runs in range(1, cap + 1):
        values.append(values[-1] * rate / runs)

    # 다음 항부터 연속 항의 비율은 rate / (cap + 2) 이하로 감소한다.
    omitted = values[-1] * rate / (cap + 1) / (1 - rate / (cap + 2))
    first_moment = rate * (values[-1] + omitted)
    total = fsum(values)
    return tuple(value / total for value in values), omitted, first_moment, rate


def fit(games: Sequence[m1.Game], mean: float, histogram_cap: int) -> Candidate:
    """같은 학습 경기에서 득점 회귀와 동점 계수를 순서대로 맞춘다."""
    rows = []
    for game in games:
        teams = m1.history(game.inputs)
        for side, runs in zip(m1.SIDES, game.runs, strict=True):
            rows.append((features(teams, side, mean), runs))
    weights = fit_coefficients(rows, mean)
    cap = max(MIN_SUPPORT, histogram_cap)
    diagonals = []
    for home_row, away_row in zip(rows[::2], rows[1::2], strict=True):
        home = distribution(home_row[0], weights, cap)[0]
        away = distribution(away_row[0], weights, cap)[0]
        diagonals.append((m1.diagonal(home, away), 1))
    return Candidate(weights, mean, cap, m1.fit_tie_weight(diagonals, games))


def predict(model: Candidate, inputs: Mapping[str, object]) -> m1.Forecast:
    """두 팀 분포를 공용 결합 함수와 같은 승무패 계약으로 변환한다."""
    teams = m1.history(inputs)
    home, away = (
        distribution(features(teams, side, model.league_mean), model.coefficients, model.cap)
        for side in m1.SIDES
    )
    omitted = home[1] + away[1]
    # 양 팀 중 한 팀이라도 상한 밖인 사건에서 각 팀 기대값의 오차를 제한한다.
    mean_error = max(home[2] + home[3] * away[1], away[2] + away[3] * home[1], model.cap * omitted)
    return m1.combine(home[0], away[0], model.tie_weight, omitted, mean_error)