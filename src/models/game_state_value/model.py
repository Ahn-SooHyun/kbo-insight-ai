"""경기 상태 입력 계약과 승리 확률 예측기를 구현한다."""

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from math import exp, isfinite, sqrt
from random import Random
import re
from typing import TypedDict


MIN_SUPPORT = 30
EPSILON = 1e-15
EPOCHS = 12
BATCH_SIZE = 512
LEARNING_RATE = 0.15

TEAM_FIELDS = (
    "season_games",
    "season_rf",
    "season_ra",
    "season_wins",
    "season_ties",
    "recent_games",
    "recent_rf",
    "recent_ra",
    "recent_wins",
    "days_since_game",
)

ALL_COLUMNS = (
    "season",
    "competition",
    "home_team_id",
    "away_team_id",
    *(
        f"{side}_{name}"
        for side in ("home", "away")
        for name in TEAM_FIELDS
    ),
    "inning",
    "half",
    "phase",
    "perspective_team_id",
    "home_score",
    "away_score",
    "outs",
    "on_first",
    "on_second",
    "on_third",
)

METHODS = {
    "frequency": (False, 0.0),
    "logistic_core": (False, 0.0),
    "logistic_history": (True, 0.0),
    "logistic_history_l2": (True, 0.001),
    "logistic_history_l2_strong": (True, 0.01),
}

StatTable = dict[str, list[int]]


def require(condition: bool, message: str) -> None:
    """자료 또는 실행 계약이 충족되지 않으면 중단한다."""
    if not condition:
        raise ValueError(message)


def count(value: object, name: str) -> int:
    """음이 아닌 정수만 허용한다. bool은 정수 입력으로 보지 않는다."""
    if type(value) is not int or value < 0:
        raise ValueError(f"{name}: 음이 아닌 정수가 필요합니다.")
    return value


def optional_count(value: object, name: str) -> int | None:
    """과거 기록의 미관측 NULL을 실제 0과 구분한다."""
    if value is None:
        return None
    return count(value, name)


@dataclass(frozen=True, slots=True)
class State:
    """관측 경계에서 사용할 수 있는 경기 상태와 과거 기록."""

    season: int
    competition: str
    home_team_id: int
    away_team_id: int
    perspective_team_id: int
    inning: int
    half: str
    phase: str
    home_score: int
    away_score: int
    outs: int
    on_first: int
    on_second: int
    on_third: int
    home_season_games: int | None
    home_season_rf: int | None
    home_season_ra: int | None
    home_season_wins: int | None
    away_season_games: int | None
    away_season_rf: int | None
    away_season_ra: int | None
    away_season_wins: int | None

    @property
    def attacking_home(self) -> bool:
        return self.half == "bot"

    @property
    def score(self) -> int:
        return self.home_score if self.attacking_home else self.away_score

    @property
    def opponent_score(self) -> int:
        return self.away_score if self.attacking_home else self.home_score

    @property
    def difference(self) -> int:
        return self.score - self.opponent_score

    @property
    def mask(self) -> int:
        """3아웃의 잔존 주자 표시는 모델 특징에서 제외한다."""
        if self.outs == 3:
            return 0

        return (
            self.on_first
            + 2 * self.on_second
            + 4 * self.on_third
        )


INPUT_COLUMNS = tuple(field.name for field in fields(State))

STATIC_COLUMNS = tuple(
    name for name in INPUT_COLUMNS
    if name not in {
        "phase",
        "home_score",
        "away_score",
        "outs",
        "on_first",
        "on_second",
        "on_third",
    }
)


def parse_state(raw: Mapping[str, object]) -> State:
    """입력 허용 목록, 상태 범위 및 공격팀 관점을 검사한다."""
    require(
        set(raw) == set(INPUT_COLUMNS),
        "M4 입력 열 누락 또는 허용되지 않은 열 존재",
    )

    for name in (
        "season", "home_team_id", "away_team_id",
        "perspective_team_id", "inning",
        "home_score", "away_score", "outs",
        "on_first", "on_second", "on_third",
    ):
        count(raw[name], name)

    require(raw["competition"] == "regular", "정규시즌만 지원합니다.")
    require(raw["half"] in ("top", "bot"), "초말 구분 오류")
    require(raw["phase"] in ("pre", "post"), "타석 경계 구분 오류")
    require(raw["season"] > 0, "시즌 오류")
    require(raw["inning"] >= 1, "이닝 범위 오류")
    require(raw["home_team_id"] > 0, "홈팀 식별자 오류")
    require(raw["away_team_id"] > 0, "원정팀 식별자 오류")
    require(raw["home_team_id"] != raw["away_team_id"], "동일 대진 오류")

    expected_team = (
        raw["home_team_id"]
        if raw["half"] == "bot"
        else raw["away_team_id"]
    )

    require(
        raw["perspective_team_id"] == expected_team,
        "현재 반이닝의 공격팀 관점 불일치",
    )

    maximum_outs = 2 if raw["phase"] == "pre" else 3
    require(raw["outs"] <= maximum_outs, "아웃 범위 오류")

    for name in ("on_first", "on_second", "on_third"):
        require(raw[name] in (0, 1), f"{name}: 주자 표시 오류")

    for side in ("home", "away"):
        names = (
            "season_games", "season_rf",
            "season_ra", "season_wins",
        )

        values = [
            optional_count(raw[f"{side}_{name}"], f"{side}_{name}")
            for name in names
        ]

        games, runs_for, runs_against, wins = values

        if games is not None and wins is not None:
            require(wins <= games, "과거 승수와 경기수 불일치")

        if games == 0:
            require(
                all(value in (None, 0) for value in (
                    runs_for, runs_against, wins
                )),
                "미출전 상태의 과거 집계 오류",
            )

    return State(**raw)


def validate_pair(pre: State, post: State) -> int:
    """같은 타석의 전후 계약을 검사하고 실제 공격팀 득점을 반환한다."""
    require(
        pre.phase == "pre" and post.phase == "post",
        "타석 전후 경계 순서 오류",
    )

    require(
        all(
            getattr(pre, name) == getattr(post, name)
            for name in STATIC_COLUMNS
        ),
        "전후 공격팀·이닝·과거 정보 불일치",
    )

    require(post.outs >= pre.outs, "타석 이후 아웃 감소")
    require(
        post.opponent_score == pre.opponent_score,
        "관측 타석 중 수비팀 점수 변경",
    )
    require(post.score >= pre.score, "공격팀 점수 감소")

    return post.score - pre.score


@dataclass(frozen=True, slots=True)
class FinalEvidence:
    """모델 특징과 분리한 관측 경기 종료 감사 근거."""

    game_id: str
    pa_number: int
    last_pa: int
    source_sha: str
    end_source_row: int
    official_status: str
    official_home_score: int
    official_away_score: int
    winner: str
    result_valid: int
    final_score_match: int
    order_valid: int
    boundary_valid: int
    terminal_state: str


def parse_final(
    raw: Mapping[str, object] | None,
) -> FinalEvidence | None:
    """종료 근거의 구조를 검사한다."""
    if raw is None:
        return None

    require(
        set(raw) == {field.name for field in fields(FinalEvidence)},
        "경기 종료 근거 열 불일치",
    )

    require(
        isinstance(raw["game_id"], str) and bool(raw["game_id"].strip()),
        "경기 종료 식별자 오류",
    )
    require(
        isinstance(raw["source_sha"], str)
        and re.fullmatch(r"[0-9a-f]{64}", raw["source_sha"]) is not None,
        "경기 종료 출처 지문 오류",
    )

    for name in (
        "pa_number", "last_pa", "end_source_row",
        "official_home_score", "official_away_score",
        "result_valid", "final_score_match",
        "order_valid", "boundary_valid",
    ):
        count(raw[name], name)

    return FinalEvidence(**raw)


def final_result(
    state: State,
    evidence: FinalEvidence | None,
) -> int | None:
    """검증된 경기 종료에서만 동일 공격팀의 승리 확정값을 반환한다."""
    if evidence is None:
        return None

    require(state.phase == "post", "타석 직전에는 종료값 적용 불가")
    require(evidence.pa_number == evidence.last_pa, "마지막 타석 불일치")

    require(
        evidence.official_status == "Final"
        and evidence.result_valid == 1
        and evidence.final_score_match == 1
        and evidence.order_valid == 1
        and evidence.boundary_valid == 1,
        "공식 경기 종료 품질 근거 부족",
    )

    require(
        state.home_score == evidence.official_home_score
        and state.away_score == evidence.official_away_score,
        "현재 득점과 공식 종료 득점 불일치",
    )

    expected_winner = (
        "home"
        if state.home_score > state.away_score
        else "away"
        if state.away_score > state.home_score
        else "tie"
    )

    require(evidence.winner == expected_winner, "공식 경기 결과 불일치")

    if evidence.terminal_state == "home_leads_partial_bottom_candidate":
        require(
            state.half == "bot"
            and state.outs < 3
            and expected_winner == "home",
            "부분 반이닝 종료 근거 불일치",
        )

    elif evidence.terminal_state == "half_ended_three_outs":
        require(
            state.outs == 3 and expected_winner != "tie",
            "3아웃 경기 종료 근거 불일치",
        )

    elif evidence.terminal_state == "final_tie":
        require(
            expected_winner == "tie",
            "무승부 종료 근거 불일치",
        )

    else:
        raise ValueError("지원되지 않는 경기 종료 근거")

    return int(state.difference > 0)


@dataclass(frozen=True, slots=True)
class Observation:
    """학습·평가용 상태와 정답. 정답은 입력 상태와 분리한다."""

    state: State
    win: int
    remaining: int
    final: int | None


class Predictor(TypedDict):
    """저장 가능한 단일 승리 확률 예측기."""

    name: str
    table: StatTable
    means: list[float]
    scales: list[float]
    weights: list[float]
    l2: float


def inning_group(inning: int) -> int:
    """초반·중반·후반·연장 상태를 구분한다."""
    if inning <= 3:
        return 1
    if inning <= 6:
        return 2
    if inning <= 9:
        return 3
    return 4


def exact_key(state: State) -> str:
    """상태 빈도와 전이 분포의 정확 상태 키."""
    return (
        f"{state.inning}|{state.half}|{state.difference}|"
        f"{state.outs}|{state.mask}"
    )


def frequency_keys(state: State) -> tuple[str, ...]:
    """세부 상태부터 전체 빈도까지 희소 후퇴 순서를 고정한다."""
    group = inning_group(state.inning)
    half = state.half
    diff = state.difference
    outs = state.outs
    mask = state.mask

    return (
        f"exact:{exact_key(state)}",
        f"group:{group}|{half}|{diff}|{outs}|{mask}",
        f"outs:{group}|{half}|{diff}|{outs}",
        f"score:{group}|{half}|{diff}",
        f"side:{half}|{diff}",
        f"diff:{diff}",
        "global",
    )


def accumulate(
    table: StatTable,
    keys: Iterable[str],
    target: int,
) -> None:
    """표본 수와 정답 합을 저장한다."""
    for key in keys:
        entry = table.setdefault(key, [0, 0])
        entry[0] += 1
        entry[1] += target


def lookup(
    table: StatTable,
    keys: Sequence[str],
) -> tuple[int, int, str]:
    """최소 지원을 충족하는 첫 집계 또는 전체 집계를 반환한다."""
    for key in keys:
        entry = table.get(key)

        if entry is None:
            continue

        n, total = entry

        if n >= MIN_SUPPORT or key == "global":
            require(n > 0, "빈 통계표")
            return n, total, key.split(":")[0]

    raise ValueError("전체 참조 통계 없음")


def fit_frequency(rows: Sequence[Observation]) -> Predictor:
    """Train 비종료 상태의 승리 빈도를 집계한다."""
    table: StatTable = {}

    for row in rows:
        if row.final is None:
            accumulate(table, frequency_keys(row.state), row.win)

    require("global" in table, "학습할 승리 상태 없음")

    return {
        "name": "frequency",
        "table": table,
        "means": [],
        "scales": [],
        "weights": [],
        "l2": 0.0,
    }


def history_features(
    games: int | None,
    runs_for: int | None,
    runs_against: int | None,
    wins: int | None,
) -> tuple[float, ...]:
    """과거 집계의 누락을 분리하고 초기 시즌의 변동성을 완화한다."""
    missing = any(
        value is None
        for value in (games, runs_for, runs_against, wins)
    )

    n = games or 0

    # 과거 5경기, 경기당 4득점·4실점, 승률 0.5의 고정 평활화.
    # 학습 자료의 실제 미관측값을 임의 관측값으로 저장하지 않는다.
    return (
        n / (n + 20.0),
        ((runs_for or 0) + 20.0) / (n + 5.0),
        ((runs_against or 0) + 20.0) / (n + 5.0),
        ((wins or 0) + 2.5) / (n + 5.0),
        float(missing),
    )


def features(state: State, history: bool) -> tuple[float, ...]:
    """정답·종료 근거·식별자를 제외한 예측 특징을 구성한다."""
    inning = float(state.inning)
    diff = float(state.difference)
    home = float(state.attacking_home)
    late = float(state.inning >= 7)
    mask = state.mask

    core = (
        inning,
        home,
        diff,
        float(state.outs),
        float(bool(mask & 1)),
        float(bool(mask & 2)),
        float(bool(mask & 4)),
        diff * inning,
        diff * home,
        abs(diff),
        late,
        diff * late,
    )

    if not history:
        return core

    home_history = history_features(
        state.home_season_games,
        state.home_season_rf,
        state.home_season_ra,
        state.home_season_wins,
    )

    away_history = history_features(
        state.away_season_games,
        state.away_season_rf,
        state.away_season_ra,
        state.away_season_wins,
    )

    attack, defense = (
        (home_history, away_history)
        if state.attacking_home
        else (away_history, home_history)
    )

    return core + attack + defense


def sigmoid(value: float) -> float:
    """큰 로짓에서도 오버플로 없이 확률을 계산한다."""
    if value >= 0:
        probability = 1.0 / (1.0 + exp(-value))
    else:
        ratio = exp(value)
        probability = ratio / (1.0 + ratio)

    return min(1.0 - EPSILON, max(EPSILON, probability))


def fit_logistic(
    rows: Sequence[Observation],
    name: str,
    seed: int,
) -> Predictor:
    """Train 전처리와 가중 미니배치 로지스틱 회귀를 적합한다."""
    require(
        name in METHODS and name != "frequency",
        "로지스틱 후보 이름 오류",
    )

    history, l2 = METHODS[name]

    # 동일 특징·정답의 중복을 집계하되 표본별 가중치를 보존한다.
    grouped = Counter(
        (features(row.state, history), row.win)
        for row in rows
        if row.final is None
    )

    require(bool(grouped), "로지스틱 학습 자료 없음")
    require(
        {label for _, label in grouped} == {0, 1},
        "이진 분류에 두 정답 범주가 모두 필요합니다.",
    )

    total = sum(grouped.values())
    dimension = len(next(iter(grouped))[0])
    sums = [0.0] * dimension
    squares = [0.0] * dimension

    for (vector, _), n in grouped.items():
        for j, value in enumerate(vector):
            sums[j] += n * value
            squares[j] += n * value * value

    means = [value / total for value in sums]

    scales = [
        max(
            sqrt(max(0.0, squares[j] / total - means[j] ** 2)),
            1e-8,
        )
        for j in range(dimension)
    ]

    # 상수 특징은 임의로 증폭하지 않는다.
    scales = [value if value > 1e-7 else 1.0 for value in scales]

    examples = [
        (
            tuple(
                (value - means[j]) / scales[j]
                for j, value in enumerate(vector)
            ),
            label,
            n,
        )
        for (vector, label), n in grouped.items()
    ]

    weights = [0.0] * (dimension + 1)
    random = Random(seed)

    for epoch in range(EPOCHS):
        random.shuffle(examples)
        rate = LEARNING_RATE / sqrt(epoch + 1.0)

        for start in range(0, len(examples), BATCH_SIZE):
            batch = examples[start:start + BATCH_SIZE]
            gradient = [0.0] * (dimension + 1)
            batch_weight = 0

            for vector, label, n in batch:
                linear = weights[0] + sum(
                    weights[j + 1] * value
                    for j, value in enumerate(vector)
                )

                error = (sigmoid(linear) - label) * n
                gradient[0] += error
                batch_weight += n

                for j, value in enumerate(vector):
                    gradient[j + 1] += error * value

            # 절편에는 L2 규제를 적용하지 않는다.
            weights[0] -= rate * gradient[0] / batch_weight

            for j in range(1, dimension + 1):
                weights[j] -= rate * (
                    gradient[j] / batch_weight + l2 * weights[j]
                )

    require(
        all(isfinite(value) for value in weights),
        "로지스틱 학습 결과 수치 오류",
    )

    return {
        "name": name,
        "table": {},
        "means": means,
        "scales": scales,
        "weights": weights,
        "l2": l2,
    }


def predict_win(
    model: Predictor,
    state: State,
    confirmed_final: int | None = None,
) -> tuple[float, str, int | None]:
    """확정 종료 또는 선택된 분류기의 무조건부 승리 확률을 반환한다."""
    if confirmed_final is not None:
        require(
            type(confirmed_final) is int
            and confirmed_final in (0, 1)
            and state.phase == "post",
            "확정 종료 확률 계약 오류",
        )
        return float(confirmed_final), "confirmed_final", None

    if model["name"] == "frequency":
        n, wins, level = lookup(
            model["table"],
            frequency_keys(state),
        )

        # 미출현 결과에 0/1 확률을 부여하지 않는 Laplace 평활화.
        return (wins + 1.0) / (n + 2.0), level, n

    history, _ = METHODS[model["name"]]
    vector = features(state, history)

    linear = model["weights"][0] + sum(
        model["weights"][j + 1]
        * (value - model["means"][j])
        / model["scales"][j]
        for j, value in enumerate(vector)
    )

    require(isfinite(linear), "승리 확률 로짓 수치 오류")

    return sigmoid(linear), model["name"], None


def validate_table(table: StatTable, binary: bool = False) -> None:
    """복원된 통계표의 표본 수와 정답 합을 검사한다."""
    require(isinstance(table, dict) and "global" in table, "통계표 오류")

    for key, entry in table.items():
        require(
            isinstance(key, str)
            and isinstance(entry, list)
            and len(entry) == 2,
            "통계표 레코드 구조 오류",
        )

        n = count(entry[0], "support")
        total = count(entry[1], "target_sum")

        require(n > 0, "통계표 지원 수 오류")

        if binary:
            require(total <= n, "이진 결과 합 오류")


def validate_predictor(model: Predictor) -> None:
    """자산 복원 시 단일 예측기의 구조와 수치 계약을 검사한다."""
    require(
        set(model) == {
            "name", "table", "means",
            "scales", "weights", "l2",
        },
        "승리 예측기 자산 구조 오류",
    )

    name = model["name"]
    require(name in METHODS, "승리 예측기 종류 오류")
    history, l2 = METHODS[name]

    require(model["l2"] == l2, "규제 설정 불일치")

    if name == "frequency":
        validate_table(model["table"], binary=True)
        require(
            not model["means"]
            and not model["scales"]
            and not model["weights"],
            "빈도 모델에 별도 분류기 존재",
        )
        return

    dimension = 22 if history else 12

    require(
        not model["table"]
        and len(model["means"]) == dimension
        and len(model["scales"]) == dimension
        and len(model["weights"]) == dimension + 1,
        "로지스틱 자산 차원 오류",
    )

    for value in (
        *model["means"], *model["scales"], *model["weights"]
    ):
        require(
            type(value) in (int, float) and isfinite(value),
            "로지스틱 자산 수치 오류",
        )

    require(
        all(value > 0 for value in model["scales"]),
        "전처리 표준편차 오류",
    )