"""경기 전 팀 이력으로 공식 종료 득점의 결합 분포를 계산한다."""

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import floor, fsum, isfinite
from typing import TypedDict


SIDES = ("home", "away")
OUTCOMES = ("home", "tie", "away")
FIELDS = ("season_games", "season_rf", "season_ra", "recent_games", "recent_rf", "recent_ra")
INPUT_COLUMNS = ("competition", *(f"{side}_{field}" for side in SIDES for field in FIELDS))
SCHEMA = "m1-final-score-v1"
MIN_CAP = 39
MAX_CAP = 512
TAIL_SHARE = 0.001
PROFILE_PRIOR = 150
SEASON_PRIOR = 24
RECENT_PRIOR = 12
ZONE_WIDTH = 0.5
ZONE_LIMIT = 4
TOLERANCE = 1e-12

Counts = dict[str, list[int]]
PMF = tuple[float, ...]


@dataclass(frozen=True)
class Game:
    """같은 분할에 남아야 하는 경기 사전 입력과 공식 종료 정답."""
    sample_id: str
    day: str
    inputs: dict[str, object]
    runs: tuple[int, int]
    outcome: str


class Model(TypedDict):
    """한 팀 관점 득점 예측기의 학습 파라미터와 결합 가중치."""
    cap: int
    league_mean: float
    site_counts: Counts
    profile_counts: Counts
    tie_weight: float


class Joint(TypedDict):
    home_base_pmf: PMF
    away_base_pmf: PMF
    equal_score_weight: float
    different_score_weight: float
    normalizer: float
    score_covariance: float
    independent_tie_probability: float


class Forecast(TypedDict):
    schema: str
    score_scope: str
    home_pmf: PMF
    away_pmf: PMF
    home_expected_runs: float
    away_expected_runs: float
    representative_score: dict[str, int]
    result_probabilities: dict[str, float]
    joint: Joint
    tail: dict[str, int | float]


def score_outcome(home: int, away: int) -> str:
    """공식 종료 득점을 홈·무승부·원정으로 매핑한다."""
    if home == away:
        return "tie"
    return "home" if home > away else "away"


def count(value: object, field: str) -> int:
    """득점과 경기 수에서 문자열, 불리언, 음수를 거부한다."""
    if type(value) is not int or not 0 <= value <= 2**63 - 1:
        raise ValueError(f"{field}: 비음수 SQLite 정수가 필요합니다.")
    return value


def history(inputs: Mapping[str, object]) -> dict[str, tuple[int, ...]]:
    """정규시즌 입력의 허용 목록과 팀별 집계 관계를 확인한다."""
    if inputs.keys() != set(INPUT_COLUMNS) or inputs["competition"] != "regular":
        raise ValueError("정규시즌 사전 입력의 필드 목록을 확인하세요.")

    teams = {}
    for side in SIDES:
        values = []
        for window in ("season", "recent"):
            key = f"{side}_{window}"
            games = count(inputs[f"{key}_games"], f"{key}_games")
            values.append(games)
            for kind in ("rf", "ra"):
                field = f"{key}_{kind}"
                raw = inputs[field]
                value = 0 if games == 0 and raw is None else count(raw, field)
                if games == 0 and value != 0:
                    raise ValueError(f"{field}: 이력 없이 득·실점 합이 있습니다.")
                values.append(value)
        if any(recent > season for season, recent in zip(values[:3], values[3:])):
            raise ValueError(f"{side}: 최근 집계가 시즌 집계를 넘습니다.")
        teams[side] = tuple(values)
    return teams


def rates(teams: Mapping[str, tuple[int, ...]], side: str, mean: float) -> tuple[float, ...]:
    """팀 득점률과 상대 실점률을 시즌·최근 순서로 반환한다."""
    team = teams[side]
    other = teams["away" if side == "home" else "home"]
    return tuple(
        (record[offset + field] + prior * mean) / (record[offset] + prior)
        for offset, prior in ((0, SEASON_PRIOR), (3, RECENT_PRIOR))
        for record, field in ((team, 1), (other, 2))
    )


def profile(teams: Mapping[str, tuple[int, ...]], side: str, mean: float) -> str:
    """과거 득점·실점 비율을 홈·원정별 고정 구간에 배정한다."""
    values = rates(teams, side, mean)
    zone = floor((fsum(values) / len(values) - mean) / ZONE_WIDTH + 0.5)
    return f"{side}:{max(-ZONE_LIMIT, min(ZONE_LIMIT, zone))}"


def pmf(site: Sequence[int], group: Sequence[int] = ()) -> PMF:
    """표본 분포를 홈·원정 평균으로 완화하고 기하분포 꼬리를 섞는다."""
    site_total = sum(site)
    group_total = sum(group)
    if group:
        values = [
            (observed + PROFILE_PRIOR * base / site_total) / (group_total + PROFILE_PRIOR)
            for base, observed in zip(site, group, strict=True)
        ]
    else:
        values = [base / site_total for base in site]
    mixed = [
        (1 - TAIL_SHARE) * probability + TAIL_SHARE * 0.5 ** (runs + 1)
        for runs, probability in enumerate(values)
    ]
    mass = fsum(mixed)
    return tuple(probability / mass for probability in mixed)


def fit(games: Sequence[Game]) -> Model:
    """Train 정규시즌만 사용해 공용 프로필과 동점 결합 계수를 맞춘다."""
    cap = max(MIN_CAP, max(max(game.runs) for game in games))
    if cap > MAX_CAP:
        raise ValueError("M1 Train 득점이 지원 범위 0~512를 넘습니다.")
    league_mean = fsum(sum(game.runs) for game in games) / (2 * len(games))
    site_counts = {side: [0] * (cap + 1) for side in SIDES}
    profile_counts: Counts = defaultdict(lambda: [0] * (cap + 1))
    pairs: Counter[tuple[str, str]] = Counter()

    for game in games:
        teams = history(game.inputs)
        home = profile(teams, "home", league_mean)
        away = profile(teams, "away", league_mean)
        pairs[home, away] += 1
        for side, key, runs in zip(SIDES, (home, away), game.runs):
            site_counts[side][runs] += 1
            profile_counts[key][runs] += 1

    distributions = {
        key: pmf(site_counts[key.split(":")[0]], scores)
        for key, scores in profile_counts.items()
    }
    diagonals = [
        (diagonal(distributions[home], distributions[away]), n)
        for (home, away), n in pairs.items()
    ]
    return {
        "cap": cap,
        "league_mean": league_mean,
        "site_counts": site_counts,
        "profile_counts": dict(profile_counts),
        "tie_weight": fit_tie_weight(diagonals, games),
    }


def fit_tie_weight(diagonals: Sequence[tuple[float, int]], games: Sequence[Game]) -> float:
    """학습 경기의 평균 동점 확률을 평활화한 관측 빈도에 맞춘다."""
    observed = (sum(game.outcome == "tie" for game in games) + 0.5) / (len(games) + 1)
    low, high = 0.0, 1.0
    for _ in range(48):
        weight = (low + high) / 2
        average = fsum(n * tilted_tie(p, weight) for p, n in diagonals) / len(games)
        if average < observed:
            low = weight
        else:
            high = weight
    return (low + high) / 2


def diagonal(home: PMF, away: PMF) -> float:
    """독립 결합의 동점 질량을 구한다."""
    return fsum(h * a for h, a in zip(home, away, strict=True))


def tilted_tie(tie: float, weight: float) -> float:
    """동점 셀과 비동점 셀에 가중치를 주고 재정규화한다."""
    return weight * tie / (weight * tie + (1 - weight) * (1 - tie))


def greater(first: PMF, second: PMF) -> float:
    """첫 점수가 두 번째 점수보다 큰 영역을 누적합으로 더한다."""
    prior = 0.0
    terms = []
    for left, right in zip(first, second, strict=True):
        terms.append(left * prior)
        prior += right
    return fsum(terms)


def mean(probabilities: PMF) -> float:
    """명시한 정수 득점 확률질량함수의 기대득점을 구한다."""
    return fsum(n * p for n, p in enumerate(probabilities))


def check_pmf(probabilities: Sequence[float]) -> None:
    """유한성, 범위 및 합을 검사한다."""
    if (
        not all(isfinite(p) and 0 <= p <= 1 for p in probabilities)
        or abs(fsum(probabilities) - 1) > TOLERANCE
    ):
        raise ValueError("예측 확률의 범위 또는 합이 잘못됐습니다.")


def predict(model: Model, inputs: Mapping[str, object]) -> Forecast:
    """홈·원정에 같은 예측기를 적용하고 결합 분포에서 모든 출력을 계산한다."""
    teams = history(inputs)
    raw_home, raw_away = (
        pmf(model["site_counts"][side], model["profile_counts"].get(
            profile(teams, side, model["league_mean"]), []
        ))
        for side in SIDES
    )
    cap = model["cap"]
    tail = TAIL_SHARE * 0.5 ** (cap + 1)
    return combine(raw_home, raw_away, model["tie_weight"], 2 * tail, 4 * (cap + 2) * tail)


def combine(
    raw_home: PMF, raw_away: PMF, weight: float, omitted_mass: float, mean_error: float,
) -> Forecast:
    """동점 보정 전 PMF와 꼬리 상한으로 같은 경기 예측 계약을 계산한다."""
    off = 1 - weight
    tie = diagonal(raw_home, raw_away)
    normalizer = weight * tie + off * (1 - tie)
    probabilities = {
        "home": off * greater(raw_home, raw_away) / normalizer,
        "tie": weight * tie / normalizer,
        "away": off * greater(raw_away, raw_home) / normalizer,
    }
    check_pmf(tuple(probabilities.values()))

    home, away = (
        tuple(p * (off * (1 - q) + weight * q) / normalizer for p, q in zip(own, other))
        for own, other in ((raw_home, raw_away), (raw_away, raw_home))
    )
    check_pmf(home)
    check_pmf(away)
    home_mean, away_mean = mean(home), mean(away)
    moments = fsum(n * n * h * a for n, (h, a) in enumerate(zip(raw_home, raw_away)))
    cross = (off * mean(raw_home) * mean(raw_away) + (weight - off) * moments) / normalizer

    ratio = max(weight, off) / min(weight, off)
    return {
        "schema": SCHEMA,
        "score_scope": "official_final",
        "home_pmf": home,
        "away_pmf": away,
        "home_expected_runs": home_mean,
        "away_expected_runs": away_mean,
        "representative_score": {"home": floor(home_mean + 0.5), "away": floor(away_mean + 0.5)},
        "result_probabilities": probabilities,
        "joint": {
            "home_base_pmf": raw_home, "away_base_pmf": raw_away,
            "equal_score_weight": weight, "different_score_weight": off,
            "normalizer": normalizer, "score_covariance": cross - home_mean * away_mean,
            "independent_tie_probability": tie,
        },
        "tail": {
            "max_runs": len(raw_home) - 1,
            "omitted_joint_mass_bound": ratio * omitted_mass,
            "expected_runs_error_bound_per_side": ratio * mean_error,
            "normalization_tolerance": TOLERANCE,
        },
    }