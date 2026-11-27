"""타석 결과의 공용 다중분류기와 비교용 빈도 기준을 구현한다."""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import logging
from math import exp, fsum, isfinite, log, log1p
from random import Random
from typing import Literal, TypedDict


LOGGER = logging.getLogger(__name__)

SCHEMA = "m2-plate-appearance-v1"

CLASS_NAMES = {
    "single": "단타",
    "double": "2루타",
    "triple": "3루타",
    "home_run": "홈런",
    "walk": "볼넷",
    "hit_by_pitch": "몸에 맞는 공",
    "strikeout": "삼진",
    "field_out": "일반 아웃",
    "fielders_choice": "야수선택",
    "field_error": "실책",
    "sac_bunt": "희생번트",
    "sac_fly": "희생플라이",
    "double_play": "병살",
    "triple_play": "삼중살",
    "catcher_interference": "포수 타격방해",
}

CLASSES = tuple(CLASS_NAMES)
CLASS_INDEX = {
    name: index for index, name in enumerate(CLASSES)
}
N_CLASSES = len(CLASSES)

TEAM_FIELDS = (
    "season_games",
    "season_rf",
    "season_ra",
    "recent_games",
    "recent_rf",
    "recent_ra",
)

PLAYER_FIELDS = (
    "prior_games",
    "prior_opportunities",
    "prior_hits",
    "prior_ab",
    "prior_so",
    "recent_games",
    "recent_opportunities",
    "recent_hits",
)

BASE_FIELDS = (
    "competition",
    "home_team_id",
    "away_team_id",
    "batter_id",
    "pitcher_id",
    "inning",
    "half",
    "outs",
    "on_first",
    "on_second",
    "on_third",
    "home_score",
    "away_score",
)

INPUT_COLUMNS = (
    *BASE_FIELDS,
    *(
        f"{side}_{field}"
        for side in ("home", "away")
        for field in TEAM_FIELDS
    ),
    *(
        f"{role}_{field}"
        for role in ("batter", "pitcher")
        for field in PLAYER_FIELDS
    ),
)

# 평활화와 학습 설정은 Validation 평가 전에 고정한다.
GLOBAL_ALPHA = 0.5
GROUP_PRIOR = 200
MIN_GROUP_COUNT = 30
MIN_ID_COUNT = 20
EPOCHS = 8
LEARNING_RATE = 0.025
LEARNING_RATE_DECAY = 0.35
PRIOR_MIX = 0.01
TOLERANCE = 1e-9
MAX_INTEGER = 2**63 - 1
COUNT_LOG_CAP = 10.0
SCORE_CAP = 20
INNING_CAP = 15
TEAM_RATE_CAP = 10.0

METHOD = {
    "global_alpha": GLOBAL_ALPHA,
    "group_prior": GROUP_PRIOR,
    "min_group_count": MIN_GROUP_COUNT,
    "min_id_count": MIN_ID_COUNT,
    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "learning_rate_decay": LEARNING_RATE_DECAY,
    "prior_mix": PRIOR_MIX,
    "count_log_cap": COUNT_LOG_CAP,
    "score_cap": SCORE_CAP,
    "inning_cap": INNING_CAP,
    "team_rate_cap": TEAM_RATE_CAP,
    "probability_tolerance": TOLERANCE,
    "conditional_group": "outs_and_base_mask",
    "optimizer": "shuffled_train_only_sgd",
    "preprocessing": "fixed_transforms_train_vocabulary",
}

ModelKind = Literal["global", "conditional", "logistic"]


@dataclass(frozen=True)
class Sample:
    """검증된 한 타석의 사전 입력과 확정 정답."""

    sample_id: str
    game_id: str
    day: str
    inputs: dict[str, object]
    target: str


class Model(TypedDict):
    """하나의 선택된 분류기를 재현하는 JSON 호환 파라미터."""

    kind: ModelKind
    counts: list[int]
    groups: dict[str, list[int]]
    weights: dict[str, list[float]]


def count(
    value: object,
    field: str,
    *,
    minimum: int = 0,
) -> int:
    """정수형 관측값을 확인하며 문자열과 불리언을 거부한다."""

    if (
        type(value) is not int
        or not minimum <= value <= MAX_INTEGER
    ):
        raise ValueError(
            f"{field}: 허용 범위의 정수가 필요합니다."
        )

    return value


def optional_count(
    value: object,
    field: str,
) -> int | None:
    """미관측 NULL과 실제 정수 0을 구분한다."""

    if value is None:
        return None

    return count(value, field)


def identifier(value: object, field: str) -> str:
    """시작 참가자의 식별자가 존재하는지 확인한다."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field}: 식별자가 필요합니다.")

    if value != value.strip():
        raise ValueError(f"{field}: 식별자 공백 오류")

    return value


def not_greater(
    smaller: int | None,
    greater: int | None,
    field: str,
) -> None:
    """두 관측값이 모두 있을 때만 집계 관계를 비교한다."""

    if (
        smaller is not None
        and greater is not None
        and smaller > greater
    ):
        raise ValueError(f"{field}: 과거 집계 관계 오류")


def target_class(value: object) -> str:
    """공식 확정 결과 코드만 허용한다."""

    if not isinstance(value, str) or value not in CLASS_INDEX:
        raise ValueError("미등록 또는 결측 타석 결과 코드")

    return value


def validate_inputs(inputs: Mapping[str, object]) -> None:
    """허용 입력과 시작 상태·과거 집계의 기본 조건을 검사한다."""

    if inputs.keys() != set(INPUT_COLUMNS):
        raise ValueError("M2 사전 입력 필드 목록 불일치")

    if inputs["competition"] != "regular":
        raise ValueError("현재 모델은 정규시즌 입력만 지원합니다.")

    home = count(
        inputs["home_team_id"], "home_team_id", minimum=1
    )
    away = count(
        inputs["away_team_id"], "away_team_id", minimum=1
    )

    if home == away:
        raise ValueError("홈·원정 팀 식별자가 같습니다.")

    identifier(inputs["batter_id"], "batter_id")
    identifier(inputs["pitcher_id"], "pitcher_id")

    count(inputs["inning"], "inning", minimum=1)
    count(inputs["home_score"], "home_score")
    count(inputs["away_score"], "away_score")

    if inputs["half"] not in ("top", "bot"):
        raise ValueError("이닝 초말 값 오류")

    if count(inputs["outs"], "outs") > 2:
        raise ValueError("타석 시작 아웃은 0~2여야 합니다.")

    for field in ("on_first", "on_second", "on_third"):
        if count(inputs[field], field) not in (0, 1):
            raise ValueError(f"{field}: 주자 점유 값 오류")

    for side in ("home", "away"):
        values = {
            field: optional_count(
                inputs[f"{side}_{field}"],
                f"{side}_{field}",
            )
            for field in TEAM_FIELDS
        }

        for suffix in ("games", "rf", "ra"):
            not_greater(
                values[f"recent_{suffix}"],
                values[f"season_{suffix}"],
                f"{side}_{suffix}",
            )

        for period in ("season", "recent"):
            if values[f"{period}_games"] == 0:
                if any(
                    values[f"{period}_{field}"] not in (None, 0)
                    for field in ("rf", "ra")
                ):
                    raise ValueError("경기 0건의 팀 득실점 오류")

    for role in ("batter", "pitcher"):
        values = {
            field: optional_count(
                inputs[f"{role}_{field}"],
                f"{role}_{field}",
            )
            for field in PLAYER_FIELDS
        }

        for smaller, greater in (
            ("prior_hits", "prior_ab"),
            ("prior_ab", "prior_opportunities"),
            ("prior_so", "prior_opportunities"),
            ("recent_games", "prior_games"),
            ("recent_opportunities", "prior_opportunities"),
            ("recent_hits", "prior_hits"),
            ("recent_hits", "recent_opportunities"),
        ):
            not_greater(
                values[smaller],
                values[greater],
                f"{role}_{smaller}",
            )

        for period in ("prior", "recent"):
            if values[f"{period}_games"] == 0:
                if any(
                    value not in (None, 0)
                    for key, value in values.items()
                    if key.startswith(f"{period}_")
                    and key != f"{period}_games"
                ):
                    raise ValueError("경기 0건의 선수 집계 오류")


def context_key(inputs: Mapping[str, object]) -> str:
    """아웃과 3개 베이스의 점유 상태를 조건부 기준 키로 만든다."""

    outs = count(inputs["outs"], "outs")
    mask = sum(
        count(inputs[field], field) * (2**index)
        for index, field in enumerate(
            ("on_first", "on_second", "on_third")
        )
    )

    return f"{outs}:{mask}"


def add_count(
    features: dict[str, float],
    key: str,
    value: object,
) -> None:
    """누적량을 제한된 로그 스케일로 변환한다."""

    observed = optional_count(value, key)

    if observed is None:
        features[f"{key}:missing"] = 1.0
        return

    features[key] = min(
        log1p(observed) / COUNT_LOG_CAP,
        1.0,
    )


def add_rate(
    features: dict[str, float],
    key: str,
    numerator: object,
    denominator: object,
    *,
    cap: float = 1.0,
) -> None:
    """분모 0·미관측·관측된 비율을 구분한다."""

    num = optional_count(numerator, key)
    den = optional_count(denominator, key)

    if num is None or den is None:
        features[f"{key}:missing"] = 1.0
        return

    if den == 0:
        features[f"{key}:zero_denominator"] = 1.0
        return

    features[key] = min(num / den / cap, 1.0)


def features(
    inputs: Mapping[str, object],
) -> dict[str, float]:
    """타석 시작 상태와 날짜 이전 집계를 수치·범주 변수로 변환한다."""

    validate_inputs(inputs)

    inning = count(inputs["inning"], "inning")
    home_score = count(inputs["home_score"], "home_score")
    away_score = count(inputs["away_score"], "away_score")

    batting_side = (
        "away" if inputs["half"] == "top" else "home"
    )
    fielding_side = (
        "home" if batting_side == "away" else "away"
    )

    score_diff = home_score - away_score
    if batting_side == "away":
        score_diff = -score_diff

    output = {
        "bias": 1.0,
        "inning": min(inning / INNING_CAP, 1.0),
        "home_batting": float(batting_side == "home"),
        "batting_score_diff": (
            max(-SCORE_CAP, min(SCORE_CAP, score_diff))
            / SCORE_CAP
        ),
        "total_score": min(
            (home_score + away_score) / SCORE_CAP,
            1.0,
        ),
        f"state:{context_key(inputs)}": 1.0,
        f"id:batter:{inputs['batter_id']}": 1.0,
        f"id:pitcher:{inputs['pitcher_id']}": 1.0,
        f"id:batting_team:{inputs[f'{batting_side}_team_id']}": 1.0,
        f"id:fielding_team:{inputs[f'{fielding_side}_team_id']}": 1.0,
    }

    for role, side in (
        ("batting", batting_side),
        ("fielding", fielding_side),
    ):
        for period in ("season", "recent"):
            games = inputs[f"{side}_{period}_games"]

            add_count(
                output,
                f"{role}:{period}:games",
                games,
            )

            for metric in ("rf", "ra"):
                add_rate(
                    output,
                    f"{role}:{period}:{metric}",
                    inputs[f"{side}_{period}_{metric}"],
                    games,
                    cap=TEAM_RATE_CAP,
                )

    for role in ("batter", "pitcher"):
        for metric in (
            "prior_games",
            "prior_opportunities",
            "recent_games",
            "recent_opportunities",
        ):
            add_count(
                output,
                f"{role}:{metric}",
                inputs[f"{role}_{metric}"],
            )

        add_rate(
            output,
            f"{role}:prior_hit_rate",
            inputs[f"{role}_prior_hits"],
            inputs[f"{role}_prior_ab"],
        )
        add_rate(
            output,
            f"{role}:prior_so_rate",
            inputs[f"{role}_prior_so"],
            inputs[f"{role}_prior_opportunities"],
        )
        add_rate(
            output,
            f"{role}:recent_hits_per_opportunity",
            inputs[f"{role}_recent_hits"],
            inputs[f"{role}_recent_opportunities"],
        )

    return {
        key: value
        for key, value in output.items()
        if value != 0.0
    }


def global_prior(counts: Sequence[int]) -> list[float]:
    """미출현 정상 범주도 양의 확률을 갖도록 가산 평활화한다."""

    denominator = sum(counts) + GLOBAL_ALPHA * N_CLASSES

    return [
        (value + GLOBAL_ALPHA) / denominator
        for value in counts
    ]


def fit_frequencies(
    samples: Sequence[Sample],
) -> tuple[Model, Model]:
    """Train 전체 빈도와 아웃·주자 상태별 빈도를 학습한다."""

    if not samples:
        raise ValueError("Train 타석이 필요합니다.")

    counts = [0] * N_CLASSES
    groups: dict[str, list[int]] = {}

    for sample in samples:
        validate_inputs(sample.inputs)
        target = CLASS_INDEX[target_class(sample.target)]
        key = context_key(sample.inputs)

        counts[target] += 1

        group = groups.setdefault(key, [0] * N_CLASSES)
        group[target] += 1

    return (
        {
            "kind": "global",
            "counts": counts.copy(),
            "groups": {},
            "weights": {},
        },
        {
            "kind": "conditional",
            "counts": counts.copy(),
            "groups": groups,
            "weights": {},
        },
    )


def softmax(logits: Sequence[float]) -> list[float]:
    """최대 로짓 차감으로 지수 계산의 Overflow를 방지한다."""

    if not logits or not all(isfinite(value) for value in logits):
        raise ValueError("Softmax 로짓 오류")

    maximum = max(logits)
    masses = [exp(value - maximum) for value in logits]
    total = sum(masses)

    return [value / total for value in masses]


def logits_for(
    weights: Mapping[str, list[float]],
    encoded: Mapping[str, float],
) -> list[float]:
    """학습된 변수만 사용하며 새로운 범주 ID는 후퇴 처리한다."""

    logits = [0.0] * N_CLASSES

    for name, value in encoded.items():
        coefficients = weights.get(name)

        if coefficients is None:
            continue

        for index in range(N_CLASSES):
            logits[index] += coefficients[index] * value

    return logits


def fit_logistic(
    samples: Sequence[Sample],
    seed: int,
) -> Model:
    """Train에서만 사전을 구성하고 공용 Softmax 분류기를 학습한다."""

    if not samples:
        raise ValueError("Train 타석이 필요합니다.")

    counts = [0] * N_CLASSES
    occurrences: Counter[str] = Counter()
    rows: list[tuple[dict[str, float], int]] = []

    for sample in samples:
        encoded = features(sample.inputs)
        target = CLASS_INDEX[target_class(sample.target)]

        occurrences.update(encoded.keys())
        counts[target] += 1
        rows.append((encoded, target))

    weights = {
        name: [0.0] * N_CLASSES
        for name in sorted(occurrences)
        if not name.startswith("id:")
        or occurrences[name] >= MIN_ID_COUNT
    }

    # 전체 학습 빈도를 절편 초기값으로 사용한다.
    weights["bias"] = [
        log(probability)
        for probability in global_prior(counts)
    ]

    # 반복 계산에서 사전 조회를 줄이기 위해 계수 참조를 준비한다.
    prepared: list[
        tuple[list[tuple[list[float], float]], int]
    ] = []

    for encoded, target in rows:
        prepared.append(
            (
                [
                    (weights[name], value)
                    for name, value in encoded.items()
                    if name in weights
                ],
                target,
            )
        )

    del rows

    rng = Random(seed)
    order = list(range(len(prepared)))

    for epoch in range(EPOCHS):
        rng.shuffle(order)

        learning_rate = (
            LEARNING_RATE
            / (1.0 + LEARNING_RATE_DECAY * epoch)
        )
        total_loss = 0.0

        for row_index in order:
            active, target = prepared[row_index]
            logits = [0.0] * N_CLASSES

            for coefficients, value in active:
                for index in range(N_CLASSES):
                    logits[index] += coefficients[index] * value

            probabilities = softmax(logits)

            total_loss -= log(
                max(probabilities[target], 1e-15)
            )

            gradient = [
                learning_rate
                * (float(index == target) - probability)
                for index, probability in enumerate(probabilities)
            ]

            for coefficients, value in active:
                for index in range(N_CLASSES):
                    coefficients[index] += gradient[index] * value

        LOGGER.info(
            "M2 다항 로지스틱 회귀 epoch=%d/%d train_loss=%.8f",
            epoch + 1,
            EPOCHS,
            total_loss / len(prepared),
        )

    model: Model = {
        "kind": "logistic",
        "counts": counts,
        "groups": {},
        "weights": weights,
    }

    validate_model(model)

    return model


def check_probabilities(
    probabilities: Sequence[float],
) -> None:
    """클래스 수, 유한성, 범위, 합을 확인한다."""

    if len(probabilities) != N_CLASSES:
        raise ValueError("M2 클래스 수 불일치")

    if not all(
        isfinite(value) and 0 <= value <= 1
        for value in probabilities
    ):
        raise ValueError("M2 확률 범위 또는 유한성 오류")

    if abs(fsum(probabilities) - 1.0) > TOLERANCE:
        raise ValueError("M2 확률 합이 1이 아닙니다.")


def validate_model(model: Model) -> None:
    """저장하거나 복원할 단일 분류기의 구조를 검증한다."""

    if model.keys() != {
        "kind", "counts", "groups", "weights"
    }:
        raise ValueError("M2 모델 파라미터 필드 오류")

    kind = model["kind"]
    counts = model["counts"]
    groups = model["groups"]
    weights = model["weights"]

    if kind not in ("global", "conditional", "logistic"):
        raise ValueError("지원하지 않는 M2 분류기")

    if not isinstance(counts, list) or len(counts) != N_CLASSES:
        raise ValueError("M2 학습 빈도 길이 오류")

    if sum(count(value, "class_count") for value in counts) == 0:
        raise ValueError("M2 학습 빈도가 비어 있습니다.")

    if not isinstance(groups, dict) or not isinstance(weights, dict):
        raise ValueError("M2 파라미터 구조 오류")

    if kind == "conditional":
        expected_keys = {
            f"{outs}:{mask}"
            for outs in range(3)
            for mask in range(8)
        }

        if not groups or not groups.keys() <= expected_keys:
            raise ValueError("조건부 빈도 키 오류")

        for values in groups.values():
            if not isinstance(values, list) or len(values) != N_CLASSES:
                raise ValueError("조건부 빈도 길이 오류")

            if sum(count(value, "group_count") for value in values) == 0:
                raise ValueError("조건부 빈도 표본 수 오류")

        combined = [
            sum(values[index] for values in groups.values())
            for index in range(N_CLASSES)
        ]

        if combined != counts:
            raise ValueError("조건부 빈도 전체 집계 불일치")

    elif groups:
        raise ValueError("선택 모델에 불필요한 조건부 빈도 존재")

    if kind == "logistic":
        if "bias" not in weights:
            raise ValueError("로지스틱 회귀 절편 누락")

        for name, values in weights.items():
            if not isinstance(name, str) or not name:
                raise ValueError("로지스틱 회귀 변수명 오류")

            if not isinstance(values, list) or len(values) != N_CLASSES:
                raise ValueError("로지스틱 회귀 계수 길이 오류")

            if not all(
                type(value) in (int, float) and isfinite(value)
                for value in values
            ):
                raise ValueError("로지스틱 회귀 계수 오류")

    elif weights:
        raise ValueError("선택 모델에 불필요한 회귀 계수 존재")


def predict_probabilities(
    model: Model,
    inputs: Mapping[str, object],
) -> list[float]:
    """타석 시작 입력만으로 고정 순서의 전체 확률을 산출한다."""

    kind = model["kind"]
    prior = global_prior(model["counts"])

    if kind == "logistic":
        encoded = features(inputs)
        raw = softmax(
            logits_for(model["weights"], encoded)
        )

        # 극희소 및 미출현 범주의 확률 0을 방지한다.
        probabilities = [
            (1.0 - PRIOR_MIX) * value + PRIOR_MIX * base
            for value, base in zip(raw, prior, strict=True)
        ]

    elif kind in ("global", "conditional"):
        validate_inputs(inputs)
        probabilities = prior

        if kind == "conditional":
            values = model["groups"].get(context_key(inputs))

            if values is not None and sum(values) >= MIN_GROUP_COUNT:
                denominator = sum(values) + GROUP_PRIOR

                probabilities = [
                    (value + GROUP_PRIOR * base) / denominator
                    for value, base in zip(values, prior, strict=True)
                ]

    else:
        raise ValueError("지원하지 않는 M2 예측 방식")

    check_probabilities(probabilities)

    return probabilities


def predict(
    model: Model,
    inputs: Mapping[str, object],
) -> dict[str, object]:
    """클래스 키를 명시한 타석 결과 예측 JSON을 생성한다."""

    probabilities = predict_probabilities(model, inputs)

    best = max(
        range(N_CLASSES),
        key=probabilities.__getitem__,
    )

    return {
        "schema": SCHEMA,
        "competition": "regular",
        "prediction_point": "before_plate_appearance",
        "model_kind": model["kind"],
        "class_order": list(CLASSES),
        "probabilities": dict(
            zip(CLASSES, probabilities, strict=True)
        ),
        "most_likely_event": CLASSES[best],
        "most_likely_event_name": CLASS_NAMES[CLASSES[best]],
    }