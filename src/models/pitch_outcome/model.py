"""다음 투구의 공동 분류기와 입력·확률 계약을 구현한다."""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import logging
from math import exp, fsum, isfinite, log, sqrt
from random import Random
from sys import intern
from typing import Literal, TypedDict

from ...datasets.model_events import GAME_FEATURES, HISTORY_FEATURES
from ...validation.standard_data import PITCH_RESULT_TYPES, PITCH_TYPES


LOGGER = logging.getLogger(__name__)

TYPES = tuple(sorted(PITCH_TYPES))
RESULTS = tuple(sorted(PITCH_RESULT_TYPES))

CLASSES = tuple(
    f"{pitch_type}:{result}"
    for pitch_type in TYPES
    for result in RESULTS
)

CLASS_INDEX = {
    name: index for index, name in enumerate(CLASSES)
}

N_CLASSES = len(CLASSES)
N_RESULTS = len(RESULTS)

STATE_COLUMNS = (
    "inning",
    "half",
    "batter_id",
    "pitcher_id",
    "pitch_number",
    "balls",
    "strikes",
    "outs",
    "home_score",
    "away_score",
    "on_first",
    "on_second",
    "on_third",
    "previous_pitch_type",
    "previous_pitch_result",
)

ALL_COLUMNS = (
    *GAME_FEATURES,
    *STATE_COLUMNS,
    *HISTORY_FEATURES,
)

INPUT_COLUMNS = (
    "season",
    "competition",
    *STATE_COLUMNS,
    "batter_prior_opportunities",
    "batter_prior_so",
    "pitcher_prior_opportunities",
    "pitcher_prior_so",
)

MIN_ID_COUNT = 100
MIN_GROUP_COUNT = 50
GROUP_PRIOR = 100.0
ALPHA = 1.0
PRIOR_MIX = 0.01

EPOCHS = 3
LEARNING_RATE = 0.025

MAX_INTEGER = 2**63 - 1
TOLERANCE = 1e-9

BATTER_OTHER = "id:batter:other"
PITCHER_OTHER = "id:pitcher:other"

ModelKind = Literal[
    "frequency",
    "conditional_frequency",
    "logistic",
]

METHODS: tuple[ModelKind, ...] = (
    "frequency",
    "conditional_frequency",
    "logistic",
)


@dataclass(frozen=True, slots=True)
class Sample:
    """검증된 투구의 희소 특징과 공동 정답 인덱스."""

    keys: tuple[str, ...]
    target: int


class Model(TypedDict):
    """독립 운영 분류기 하나를 복원하는 JSON 호환 자산."""

    kind: ModelKind
    counts: list[int]
    groups: dict[str, list[int]]
    weights: dict[str, list[float]]


def require(condition: bool, message: str) -> None:
    """계약 위반 시 계산을 중단한다."""
    if not condition:
        raise ValueError(message)


def integer(
    value: object,
    name: str,
    minimum: int = 0,
    maximum: int = MAX_INTEGER,
) -> int:
    """문자열·불리언을 제외한 범위 내 정수만 허용한다."""
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name}: 정수 범위 오류")
    return value


def identifier(value: object, name: str) -> str:
    """공백이나 결측 식별자를 입력으로 사용하지 않는다."""
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
    ):
        raise ValueError(f"{name}: 식별자 오류")
    return value


def optional_integer(value: object, name: str) -> int | None:
    """미관측과 실제 0을 구분한다."""
    return None if value is None else integer(value, name)


def target_index(
    pitch_type: object,
    result: object,
    joint: object,
) -> int:
    """세 정답 열의 코드 및 공동 매핑 일치를 검사한다."""
    require(
        isinstance(pitch_type, str) and pitch_type in TYPES,
        "구종 정답 코드 오류",
    )
    require(
        isinstance(result, str) and result in RESULTS,
        "투구 결과 정답 코드 오류",
    )

    expected = f"{pitch_type}:{result}"

    require(joint == expected, "공동 정답 매핑 불일치")
    return CLASS_INDEX[expected]


def history_bucket(
    opportunities: object,
    strikeouts: object,
    role: str,
) -> str:
    """관측 노출량과 삼진율을 고정 구간의 범주로 변환한다."""
    total = optional_integer(
        opportunities, f"{role}_prior_opportunities"
    )
    so = optional_integer(strikeouts, f"{role}_prior_so")

    if total is not None and so is not None:
        require(so <= total, f"{role}: 과거 삼진 집계 오류")

    if total is None or so is None or total == 0:
        return f"history:{role}:missing"

    # 표본이 적은 기록을 충분한 과거 기록과 구분한다.
    if total < 30:
        return f"history:{role}:low_support"

    bucket = min((so * 10) // total, 9)
    return f"history:{role}:rate:{bucket}"


def encode(inputs: Mapping[str, object]) -> tuple[str, ...]:
    """현재 투구 정답이 없는 허용 입력을 희소 One-hot으로 변환한다."""
    require(
        set(inputs) == set(INPUT_COLUMNS),
        "M5 입력 열 누락 또는 비허용 입력 포함",
    )

    integer(inputs["season"], "season", 1900, 2100)
    require(
        inputs["competition"] == "regular",
        "현재 M5 운영 범위는 정규시즌입니다.",
    )

    inning = integer(inputs["inning"], "inning", 1)
    half = inputs["half"]
    require(half in ("top", "bot"), "공수 구분 오류")

    batter = identifier(inputs["batter_id"], "batter_id")
    pitcher = identifier(inputs["pitcher_id"], "pitcher_id")

    number = integer(inputs["pitch_number"], "pitch_number", 1)
    balls = integer(inputs["balls"], "balls", 0, 3)
    strikes = integer(inputs["strikes"], "strikes", 0, 2)
    outs = integer(inputs["outs"], "outs", 0, 2)

    home = integer(inputs["home_score"], "home_score")
    away = integer(inputs["away_score"], "away_score")

    first = integer(inputs["on_first"], "on_first", 0, 1)
    second = integer(inputs["on_second"], "on_second", 0, 1)
    third = integer(inputs["on_third"], "on_third", 0, 1)

    previous_type = inputs["previous_pitch_type"]
    previous_result = inputs["previous_pitch_result"]

    if number == 1:
        require(
            previous_type is None and previous_result is None,
            "첫 공에 이전 투구 코드가 존재합니다.",
        )
        previous = "first"

    elif previous_type is None and previous_result is None:
        previous = "missing"

    else:
        require(
            isinstance(previous_type, str)
            and previous_type in TYPES
            and isinstance(previous_result, str)
            and previous_result in RESULTS,
            "이전 실제 투구 코드 오류 또는 부분 결측",
        )
        previous = f"{previous_type}:{previous_result}"

    # 공격팀 관점 점수 차와 주자 상태를 고정된 구간으로 변환한다.
    score_difference = away - home if half == "top" else home - away
    score_bucket = max(-5, min(5, score_difference))
    base_mask = first + 2 * second + 4 * third

    if inning <= 3:
        inning_group = "early"
    elif inning <= 6:
        inning_group = "middle"
    elif inning <= 9:
        inning_group = "late"
    else:
        inning_group = "extra"

    keys = (
        f"count:{balls}:{strikes}",
        f"state:{outs}:{base_mask}",
        f"inning:{inning_group}:{half}",
        f"score:{score_bucket}",
        f"pitch_number:{min(number, 5)}",
        f"previous:{previous}",
        f"id:batter:value:{batter}",
        f"id:pitcher:value:{pitcher}",
        history_bucket(
            inputs["batter_prior_opportunities"],
            inputs["batter_prior_so"],
            "batter",
        ),
        history_bucket(
            inputs["pitcher_prior_opportunities"],
            inputs["pitcher_prior_so"],
            "pitcher",
        ),
    )

    # 대규모 표본에서 반복 문자열의 메모리 사용을 줄인다.
    return tuple(intern(key) for key in keys)


def training_counts(samples: Sequence[Sample]) -> list[int]:
    """학습 공동 범주의 관측 빈도를 집계한다."""
    require(bool(samples), "Train 표본이 필요합니다.")

    counts = [0] * N_CLASSES

    for sample in samples:
        index = integer(
            sample.target, "target", 0, N_CLASSES - 1
        )
        counts[index] += 1

    return counts


def global_prior(counts: Sequence[int]) -> list[float]:
    """출처 계약의 모든 조합에 Laplace 평활화를 적용한다."""
    require(len(counts) == N_CLASSES, "공동 빈도 길이 오류")

    denominator = sum(counts) + ALPHA * N_CLASSES

    return [
        (count + ALPHA) / denominator
        for count in counts
    ]


def check_probabilities(probabilities: Sequence[float]) -> None:
    """공동·주변 확률의 유한성, 범위와 정규화를 검사한다."""
    require(bool(probabilities), "확률 배열이 비어 있습니다.")
    require(
        all(
            isfinite(probability)
            and 0.0 <= probability <= 1.0
            for probability in probabilities
        ),
        "유한하지 않거나 범위를 벗어난 확률",
    )
    require(
        abs(fsum(probabilities) - 1.0) <= TOLERANCE,
        "확률 합이 1이 아닙니다.",
    )


def softmax(logits: Sequence[float]) -> list[float]:
    """최대 로짓 차감으로 지수 계산을 안정화한다."""
    require(
        bool(logits) and all(isfinite(value) for value in logits),
        "Softmax 입력 오류",
    )
    maximum = max(logits)
    values = [exp(value - maximum) for value in logits]
    total = fsum(values)

    return [value / total for value in values]


def fallback_key(key: str) -> str | None:
    """미출현 선수 식별자를 역할별 공통 범주로 후퇴시킨다."""
    if key.startswith("id:batter:"):
        return BATTER_OTHER
    if key.startswith("id:pitcher:"):
        return PITCHER_OTHER
    return None


def active_weights(
    weights: Mapping[str, list[float]],
    keys: Sequence[str],
) -> list[list[float]]:
    """학습 사전에 속한 활성 특징의 계수 참조를 반환한다."""
    active = [weights["bias"]]

    for key in keys:
        coefficients = weights.get(key)

        if coefficients is None:
            fallback = fallback_key(key)

            if fallback is not None:
                coefficients = weights.get(fallback)

        # 미출현 일반 범주는 새로운 계수를 학습하지 않는다.
        if coefficients is not None:
            active.append(coefficients)

    return active


def logits_for(active: Sequence[Sequence[float]]) -> list[float]:
    """One-hot 특징의 가중치 합으로 공동 로짓을 계산한다."""
    logits = list(active[0])

    for coefficients in active[1:]:
        logits = [
            first + second
            for first, second in zip(logits, coefficients)
        ]

    return logits


def fit_frequency(
    samples: Sequence[Sample],
    conditional: bool = False,
) -> Model:
    """Train의 전체 및 볼카운트 조건부 공동 빈도를 학습한다."""
    counts = training_counts(samples)
    groups: dict[str, list[int]] = {}

    if conditional:
        for sample in samples:
            key = sample.keys[0]
            require(key.startswith("count:"), "카운트 특징 오류")

            group = groups.setdefault(key, [0] * N_CLASSES)
            group[sample.target] += 1

    model: Model = {
        "kind": (
            "conditional_frequency"
            if conditional
            else "frequency"
        ),
        "counts": counts,
        "groups": groups,
        "weights": {},
    }

    validate_model(model)
    return model


def fit_logistic(
    samples: Sequence[Sample],
    seed: int,
) -> Model:
    """Train 전체를 사용해 단일 다항 로지스틱 분류기를 학습한다."""
    counts = training_counts(samples)

    occurrences: Counter[str] = Counter(
        key for sample in samples for key in sample.keys
    )

    # 빈도가 낮은 선수의 개별 계수를 만들지 않는다.
    vocabulary = {
        key
        for key, support in occurrences.items()
        if not key.startswith("id:") or support >= MIN_ID_COUNT
    }

    vocabulary.update((BATTER_OTHER, PITCHER_OTHER))

    weights = {
        key: [0.0] * N_CLASSES
        for key in sorted(vocabulary)
    }

    weights["bias"] = [
        log(probability) for probability in global_prior(counts)
    ]

    # 매 epoch에서 모든 Train 표본을 비복원 순회한다.
    rng = Random(seed)
    order = list(range(len(samples)))

    for epoch in range(EPOCHS):
        rng.shuffle(order)
        rate = LEARNING_RATE / sqrt(epoch + 1)
        total_loss = 0.0

        for index in order:
            sample = samples[index]
            active = active_weights(weights, sample.keys)
            probabilities = softmax(logits_for(active))

            total_loss -= log(
                max(probabilities[sample.target], 1e-15)
            )

            gradient = [
                -rate * probability
                for probability in probabilities
            ]
            gradient[sample.target] += rate

            for coefficients in active:
                coefficients[:] = [
                    weight + update
                    for weight, update in zip(
                        coefficients, gradient
                    )
                ]

        LOGGER.info(
            "M5 epoch=%d/%d train_log_loss=%.8f",
            epoch + 1,
            EPOCHS,
            total_loss / len(samples),
        )

    model: Model = {
        "kind": "logistic",
        "counts": counts,
        "groups": {},
        "weights": weights,
    }

    validate_model(model)
    return model


def validate_model(model: Model) -> None:
    """저장 자산의 분류기 종류, 빈도, 계수 차원을 검사한다."""
    require(
        set(model) == {"kind", "counts", "groups", "weights"},
        "M5 분류기 구조 오류",
    )
    require(model["kind"] in METHODS, "분류기 종류 오류")

    counts = model["counts"]

    require(
        isinstance(counts, list)
        and len(counts) == N_CLASSES
        and all(type(value) is int and value >= 0 for value in counts)
        and sum(counts) > 0,
        "학습 빈도 자산 오류",
    )

    groups = model["groups"]
    weights = model["weights"]

    require(
        isinstance(groups, dict) and isinstance(weights, dict),
        "분류기 자산 타입 오류",
    )

    if model["kind"] == "conditional_frequency":
        require(bool(groups) and not weights, "조건부 자산 오류")
        totals = [0] * N_CLASSES

        for key, group in groups.items():
            require(
                key in {
                    f"count:{balls}:{strikes}"
                    for balls in range(4)
                    for strikes in range(3)
                },
                "조건부 카운트 키 오류",
            )
            require(
                isinstance(group, list)
                and len(group) == N_CLASSES
                and all(
                    type(value) is int and value >= 0
                    for value in group
                ),
                "조건부 빈도 배열 오류",
            )

            totals = [
                first + second
                for first, second in zip(totals, group)
            ]

        require(totals == counts, "조건부·전체 학습 빈도 불일치")

    elif model["kind"] == "logistic":
        require(
            not groups
            and {"bias", BATTER_OTHER, PITCHER_OTHER}
            <= set(weights),
            "로지스틱 자산 필수 계수 누락",
        )

        for key, coefficients in weights.items():
            require(
                isinstance(key, str)
                and isinstance(coefficients, list)
                and len(coefficients) == N_CLASSES
                and all(
                    type(value) in (int, float)
                    and isfinite(value)
                    for value in coefficients
                ),
                "로지스틱 계수 오류",
            )

    else:
        require(
            not groups and not weights,
            "전체 빈도 자산에 불필요한 구성 요소 존재",
        )


class Runtime:
    """검증된 분류기의 반복 추론에서 공통 계산을 재사용한다."""

    def __init__(self, model: Model) -> None:
        validate_model(model)

        self.model = model
        self.prior = global_prior(model["counts"])
        self.conditional: dict[str, list[float]] = {}

        for key, counts in model["groups"].items():
            total = sum(counts)

            if total >= MIN_GROUP_COUNT:
                self.conditional[key] = [
                    (count + GROUP_PRIOR * prior)
                    / (total + GROUP_PRIOR)
                    for count, prior in zip(counts, self.prior)
                ]

    def probabilities(self, keys: Sequence[str]) -> list[float]:
        """입력 특징으로 고정된 공동 사전의 확률을 계산한다."""
        kind = self.model["kind"]

        if kind == "frequency":
            return self.prior.copy()

        if kind == "conditional_frequency":
            return self.conditional.get(
                keys[0], self.prior
            ).copy()

        active = active_weights(
            self.model["weights"], keys
        )
        probabilities = softmax(logits_for(active))

        # Train 전체 사전의 작은 질량을 보존한다.
        # Validation·Test 정답으로 혼합 비율을 정하지 않는다.
        return [
            (1.0 - PRIOR_MIX) * probability
            + PRIOR_MIX * prior
            for probability, prior in zip(
                probabilities, self.prior
            )
        ]


def marginalize(
    probabilities: Sequence[float],
) -> tuple[list[float], list[float]]:
    """공동 분포의 각 좌표 축을 합산해 주변 확률을 구한다."""
    require(
        len(probabilities) == N_CLASSES,
        "공동 확률 차원 오류",
    )
    check_probabilities(probabilities)

    pitch_probabilities = [
        fsum(
            probabilities[
                index * N_RESULTS:(index + 1) * N_RESULTS
            ]
        )
        for index in range(len(TYPES))
    ]

    result_probabilities = [
        fsum(probabilities[index::N_RESULTS])
        for index in range(N_RESULTS)
    ]

    check_probabilities(pitch_probabilities)
    check_probabilities(result_probabilities)

    return pitch_probabilities, result_probabilities


def predict(
    runtime: Runtime,
    inputs: Mapping[str, object],
) -> dict[str, object]:
    """현재 투구 정답 없이 공동·구종·결과 확률을 반환한다."""
    keys = encode(inputs)
    joint = runtime.probabilities(keys)
    pitch_types, results = marginalize(joint)

    return {
        "model_kind": runtime.model["kind"],
        "joint": dict(zip(CLASSES, joint)),
        "pitch_type": dict(zip(TYPES, pitch_types)),
        "pitch_result": dict(zip(RESULTS, results)),
    }