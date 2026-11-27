
"""타석 결과의 규칙 제한, 개선 학습, 확률 보정 및 운영 모델 선택을 담당한다."""

from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import date
import logging
from math import fsum, isfinite, log
from random import Random
from typing import Literal, TypedDict

from . import model as m2
from ...evaluation import plate_appearance as evaluation


LOGGER = logging.getLogger(__name__)

SCHEMA = "m2-plate-appearance-v2"
CALIBRATION_BOUNDARY = "2025-07-01"

REGULARIZED_EPOCHS = 8
REGULARIZED_LEARNING_RATE = 0.015
REGULARIZED_DECAY = 0.5
ID_SHRINKAGE = 0.90
OTHER_SHRINKAGE = 0.98
AVERAGE_EPOCHS = 4

TEMPERATURES = (
    0.75, 0.85, 1.0, 1.15, 1.30, 1.50, 2.0,
)

MIN_CALIBRATION_SUPPORT = 30
BIAS_ITERATIONS = 30
BIAS_LEARNING_RATE = 0.8
BIAS_L2 = 0.001
BIAS_LIMIT = 2.0
MAX_CONFLICT_EXAMPLES = 20

RULE_EVENTS = (
    "double_play",
    "triple_play",
    "sac_bunt",
    "sac_fly",
)

METHOD = {
    "base_method": m2.METHOD,
    "calibration_boundary": CALIBRATION_BOUNDARY,
    "rules": {
        "double_play": "outs_lt_2_and_runners_ge_1",
        "triple_play": "outs_eq_0_and_runners_ge_2",
        "sac_bunt": "outs_lt_2_and_runners_ge_1",
        "sac_fly": "outs_lt_2_and_runners_ge_1",
    },
    "rule_conflict_policy": "fail_closed",
    "regularized_epochs": REGULARIZED_EPOCHS,
    "regularized_learning_rate": REGULARIZED_LEARNING_RATE,
    "regularized_decay": REGULARIZED_DECAY,
    "id_shrinkage": ID_SHRINKAGE,
    "other_shrinkage": OTHER_SHRINKAGE,
    "average_epochs": AVERAGE_EPOCHS,
    "temperatures": list(TEMPERATURES),
    "min_calibration_support": MIN_CALIBRATION_SUPPORT,
    "bias_iterations": BIAS_ITERATIONS,
    "bias_learning_rate": BIAS_LEARNING_RATE,
    "bias_l2": BIAS_L2,
    "bias_limit": BIAS_LIMIT,
    "calibration_input": "log_of_positive_masked_probabilities",
    "calibration_selection": "log_loss_brier_classwise_ece",
}


Variant = Literal[
    "global",
    "conditional",
    "logistic",
    "regularized",
]

CalibrationKind = Literal[
    "none",
    "temperature",
    "temperature_bias",
]


class OperationalModel(TypedDict):
    """단일 분류기와 확률 보정 파라미터의 저장 계약."""

    base: m2.Model
    variant: Variant
    calibration: CalibrationKind
    temperature: float
    class_offsets: list[float]


class DetailedMetrics(evaluation.Metrics):
    """기존 평가에 클래스별 보정 오차와 규칙 위반을 추가한다."""

    classwise_ece: dict[str, float]
    macro_classwise_ece: float
    forbidden_probability_mean: float
    forbidden_nonzero_samples: int
    forbidden_class_probability_mean: dict[str, float]
    forbidden_target_count: int


def allowed_mask(
    inputs: Mapping[str, object],
) -> tuple[bool, ...]:
    """시작 상태에서 명백히 불가능한 결과만 제한한다."""

    outs = m2.count(inputs["outs"], "outs")

    if outs > 2:
        raise ValueError("타석 시작 아웃은 0~2여야 합니다.")

    bases = [
        m2.count(inputs[key], key)
        for key in ("on_first", "on_second", "on_third")
    ]

    if any(value not in (0, 1) for value in bases):
        raise ValueError("주자 점유 값은 0 또는 1이어야 합니다.")

    runners = sum(bases)

    forbidden = set()

    # 타석 도중 주자가 진루할 수 있으므로 시작 시 3루 점유를
    # 희생플라이의 필수조건으로 강제하지 않는다.
    if outs >= 2 or runners < 1:
        forbidden.update(
            ("double_play", "sac_bunt", "sac_fly")
        )

    if outs != 0 or runners < 2:
        forbidden.add("triple_play")

    return tuple(
        name not in forbidden
        for name in m2.CLASSES
    )


def audit_rules(
    samples: Sequence[m2.Sample],
) -> dict[str, object]:
    """관측 정답과 규칙의 충돌을 전수 집계한다."""

    forbidden_counts: Counter[str] = Counter()
    conflicts: Counter[str] = Counter()
    examples: list[dict[str, object]] = []

    for sample in samples:
        mask = allowed_mask(sample.inputs)
        target = m2.target_class(sample.target)

        for name, allowed in zip(
            m2.CLASSES, mask, strict=True
        ):
            if not allowed:
                forbidden_counts[name] += 1

        if not mask[m2.CLASS_INDEX[target]]:
            conflicts[target] += 1

            if len(examples) < MAX_CONFLICT_EXAMPLES:
                examples.append(
                    {
                        "sample_id": sample.sample_id,
                        "date": sample.day,
                        "target": target,
                        "context": m2.context_key(sample.inputs),
                    }
                )

    return {
        "samples": len(samples),
        "forbidden_context_counts": {
            name: forbidden_counts[name]
            for name in RULE_EVENTS
        },
        "conflict_count": sum(conflicts.values()),
        "conflicts_by_class": dict(conflicts),
        "conflict_examples": examples,
    }


def require_rule_consistency(
    samples: Sequence[m2.Sample],
    split: str,
) -> dict[str, object]:
    """충돌 표본을 임의 제거하지 않고 학습을 중단한다."""

    report = audit_rules(samples)

    if report["conflict_count"]:
        LOGGER.error(
            "M2 규칙과 실제 정답 충돌 split=%s report=%s",
            split,
            report,
        )

        raise ValueError(
            f"{split}: 규칙과 실제 정답 충돌 "
            f"{report['conflict_count']}건"
        )

    return report


def normalize_masked(
    probabilities: Sequence[float],
    mask: Sequence[bool],
) -> list[float]:
    """금지 결과의 질량을 0으로 만든 뒤 재정규화한다."""

    m2.check_probabilities(probabilities)

    if len(mask) != m2.N_CLASSES:
        raise ValueError("M2 규칙 마스크 길이 오류")

    values = [
        probability if allowed else 0.0
        for probability, allowed in zip(
            probabilities, mask, strict=True
        )
    ]

    total = fsum(values)

    if total <= 0:
        raise ValueError("허용 결과의 확률 질량이 없습니다.")

    result = [value / total for value in values]

    m2.check_probabilities(result)

    return result


def masked_softmax(
    logits: Sequence[float],
    mask: Sequence[bool],
) -> list[float]:
    """허용된 클래스 안에서만 Softmax 확률을 계산한다."""

    if len(logits) != m2.N_CLASSES or len(mask) != m2.N_CLASSES:
        raise ValueError("M2 로짓 또는 마스크 길이 오류")

    indices = [
        index for index, allowed in enumerate(mask)
        if allowed
    ]

    if not indices:
        raise ValueError("허용 클래스가 없습니다.")

    allowed_probabilities = m2.softmax(
        [logits[index] for index in indices]
    )

    result = [0.0] * m2.N_CLASSES

    for index, probability in zip(
        indices, allowed_probabilities, strict=True
    ):
        result[index] = probability

    return result


def split_validation(
    samples: Sequence[m2.Sample],
) -> tuple[list[m2.Sample], list[m2.Sample]]:
    """2025년 Validation을 보정·선정 기간으로 시간순 분리한다."""

    calibration: list[m2.Sample] = []
    selection: list[m2.Sample] = []

    game_periods: dict[str, str] = {}

    for sample in samples:
        if (
            date.fromisoformat(sample.day).isoformat() != sample.day
            or not "2025-01-01" <= sample.day < "2026-01-01"
        ):
            raise ValueError("보정·선정 자료는 2025년이어야 합니다.")

        period = (
            "calibration"
            if sample.day < CALIBRATION_BOUNDARY
            else "selection"
        )

        previous = game_periods.setdefault(
            sample.game_id, period
        )

        if previous != period:
            raise ValueError(
                "동일 경기의 보정·선정 기간이 섞였습니다."
            )

        if period == "calibration":
            calibration.append(sample)
        else:
            selection.append(sample)

    if not calibration or not selection:
        raise ValueError("보정·선정 기간 모두 유효 표본이 필요합니다.")

    return calibration, selection


def period_summary(
    samples: Sequence[m2.Sample],
) -> dict[str, object]:
    """개발 구간별 분모와 실제 날짜 범위를 기록한다."""

    if not samples:
        raise ValueError("기간 요약 대상이 비어 있습니다.")

    frequencies = Counter(
        sample.target for sample in samples
    )

    return {
        "samples": len(samples),
        "games": len({sample.game_id for sample in samples}),
        "start_date": min(sample.day for sample in samples),
        "end_date": max(sample.day for sample in samples),
        "target_frequency": {
            name: frequencies[name]
            for name in m2.CLASSES
        },
    }


def fit_regularized_logistic(
    samples: Sequence[m2.Sample],
    seed: int,
) -> m2.Model:
    """규칙 제한 Softmax, 계수 축소와 평균화를 적용한다."""

    if not samples:
        raise ValueError("Train 타석이 필요합니다.")

    counts = [0] * m2.N_CLASSES
    occurrences: Counter[str] = Counter()

    rows: list[
        tuple[dict[str, float], int, tuple[int, ...]]
    ] = []

    for sample in samples:
        encoded = m2.features(sample.inputs)
        target = m2.CLASS_INDEX[
            m2.target_class(sample.target)
        ]

        indices = tuple(
            index
            for index, allowed in enumerate(
                allowed_mask(sample.inputs)
            )
            if allowed
        )

        if target not in indices:
            raise ValueError(
                f"학습 정답의 규칙 충돌: {sample.sample_id}"
            )

        counts[target] += 1
        occurrences.update(encoded)

        rows.append((encoded, target, indices))

    # 기존 분류기와 같은 Train ID 지원 기준을 사용한다.
    weights = {
        name: [0.0] * m2.N_CLASSES
        for name in sorted(occurrences)
        if not name.startswith("id:")
        or occurrences[name] >= m2.MIN_ID_COUNT
    }

    weights["bias"] = [
        log(probability)
        for probability in m2.global_prior(counts)
    ]

    prepared: list[
        tuple[
            list[tuple[list[float], float]],
            int,
            tuple[int, ...],
        ]
    ] = []

    for encoded, target, indices in rows:
        prepared.append(
            (
                [
                    (weights[name], value)
                    for name, value in encoded.items()
                    if name in weights
                ],
                target,
                indices,
            )
        )

    del rows

    averaged = {
        name: [0.0] * m2.N_CLASSES
        for name in weights
    }

    average_count = 0

    rng = Random(seed)
    order = list(range(len(prepared)))

    for epoch in range(REGULARIZED_EPOCHS):
        rng.shuffle(order)

        learning_rate = (
            REGULARIZED_LEARNING_RATE
            / (1.0 + REGULARIZED_DECAY * epoch)
        )

        total_loss = 0.0

        for row_index in order:
            active, target, indices = prepared[row_index]
            logits = [0.0] * m2.N_CLASSES

            for coefficients, value in active:
                for index in indices:
                    logits[index] += coefficients[index] * value

            probabilities = m2.softmax(
                [logits[index] for index in indices]
            )

            target_position = indices.index(target)

            total_loss -= log(
                max(
                    probabilities[target_position],
                    evaluation.LOG_EPSILON,
                )
            )

            gradients = [
                learning_rate
                * (float(index == target) - probability)
                for index, probability in zip(
                    indices, probabilities, strict=True
                )
            ]

            # 불가능한 결과는 현재 표본의 손실과 기울기에서 제외한다.
            for coefficients, value in active:
                for index, gradient in zip(
                    indices, gradients, strict=True
                ):
                    coefficients[index] += gradient * value

        # 고정 epoch 축소는 희소 ID 계수의 과도한 확대를 완화한다.
        # 절편은 전체 결과 분포의 기준이므로 축소하지 않는다.
        for name, coefficients in weights.items():
            if name == "bias":
                continue

            factor = (
                ID_SHRINKAGE
                if name.startswith("id:")
                else OTHER_SHRINKAGE
            )

            for index in range(m2.N_CLASSES):
                coefficients[index] *= factor

        if epoch >= REGULARIZED_EPOCHS - AVERAGE_EPOCHS:
            for name, coefficients in weights.items():
                accumulator = averaged[name]

                for index in range(m2.N_CLASSES):
                    accumulator[index] += coefficients[index]

            average_count += 1

        LOGGER.info(
            "M2 개선 로지스틱 epoch=%d/%d train_loss=%.8f",
            epoch + 1,
            REGULARIZED_EPOCHS,
            total_loss / len(prepared),
        )

    if average_count != AVERAGE_EPOCHS:
        raise ValueError("M2 계수 평균화 횟수 불일치")

    final_weights = {
        name: [
            value / average_count
            for value in coefficients
        ]
        for name, coefficients in averaged.items()
    }

    model: m2.Model = {
        "kind": "logistic",
        "counts": counts,
        "groups": {},
        "weights": final_weights,
    }

    m2.validate_model(model)

    return model


def masked_probabilities(
    base: m2.Model,
    inputs: Mapping[str, object],
) -> list[float]:
    """모든 운영 후보에 동일한 규칙과 평활화 계약을 적용한다."""

    m2.validate_inputs(inputs)

    mask = allowed_mask(inputs)

    prior = normalize_masked(
        m2.global_prior(base["counts"]),
        mask,
    )

    if base["kind"] == "global":
        result = prior

    elif base["kind"] == "conditional":
        group = base["groups"].get(
            m2.context_key(inputs)
        )

        if group is None or sum(group) < m2.MIN_GROUP_COUNT:
            result = prior

        else:
            if any(
                count > 0 and not allowed
                for count, allowed in zip(
                    group, mask, strict=True
                )
            ):
                raise ValueError(
                    "조건부 빈도에 규칙 충돌 정답이 있습니다."
                )

            denominator = sum(group) + m2.GROUP_PRIOR

            result = [
                (count + m2.GROUP_PRIOR * probability)
                / denominator
                for count, probability in zip(
                    group, prior, strict=True
                )
            ]

    elif base["kind"] == "logistic":
        logits = m2.logits_for(
            base["weights"],
            m2.features(inputs),
        )

        raw = masked_softmax(logits, mask)

        # 평활화 역시 허용 클래스 안에서 수행한다.
        result = [
            (1.0 - m2.PRIOR_MIX) * probability
            + m2.PRIOR_MIX * baseline
            for probability, baseline in zip(
                raw, prior, strict=True
            )
        ]

    else:
        raise ValueError("지원하지 않는 M2 분류기")

    m2.check_probabilities(result)

    return result


def calibrate_vector(
    probabilities: Sequence[float],
    temperature: float,
    offsets: Sequence[float],
) -> list[float]:
    """양의 허용 확률만 보정하고 구조적 0은 유지한다."""

    if not isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature 값 오류")

    if len(probabilities) != m2.N_CLASSES:
        raise ValueError("M2 보정 확률 길이 오류")

    if len(offsets) != m2.N_CLASSES:
        raise ValueError("M2 보정 절편 길이 오류")

    if temperature == 1.0 and not any(offsets):
        return list(probabilities)

    indices = [
        index
        for index, value in enumerate(probabilities)
        if value > 0
    ]

    if not indices:
        raise ValueError("보정할 허용 확률이 없습니다.")

    logits = [
        log(probabilities[index]) / temperature + offsets[index]
        for index in indices
    ]

    calibrated = m2.softmax(logits)
    result = [0.0] * m2.N_CLASSES

    for index, value in zip(
        indices, calibrated, strict=True
    ):
        result[index] = value

    return result


def make_model(
    base: m2.Model,
    variant: Variant,
    calibration: CalibrationKind = "none",
    temperature: float = 1.0,
    offsets: Sequence[float] | None = None,
) -> OperationalModel:
    """후보 분류기를 단일 운영 모델 계약으로 구성한다."""

    model: OperationalModel = {
        "base": base,
        "variant": variant,
        "calibration": calibration,
        "temperature": temperature,
        "class_offsets": (
            list(offsets)
            if offsets is not None
            else [0.0] * m2.N_CLASSES
        ),
    }

    validate_model(model)

    return model


def validate_model(model: OperationalModel) -> None:
    """분류기 구조와 보정 파라미터의 유효성을 검사한다."""

    if model.keys() != {
        "base",
        "variant",
        "calibration",
        "temperature",
        "class_offsets",
    }:
        raise ValueError("M2 운영 모델 필드 불일치")

    variant = model["variant"]
    calibration = model["calibration"]

    if variant not in (
        "global", "conditional", "logistic", "regularized"
    ):
        raise ValueError("M2 운영 후보 종류 오류")

    if calibration not in (
        "none", "temperature", "temperature_bias"
    ):
        raise ValueError("M2 확률 보정 종류 오류")

    m2.validate_model(model["base"])

    expected_kind = (
        "logistic"
        if variant == "regularized"
        else variant
    )

    if model["base"]["kind"] != expected_kind:
        raise ValueError("M2 후보와 기본 분류기 종류 불일치")

    temperature = model["temperature"]
    offsets = model["class_offsets"]

    if (
        type(temperature) not in (int, float)
        or not isfinite(temperature)
        or temperature not in TEMPERATURES
    ):
        raise ValueError("M2 저장 Temperature 오류")

    if not isinstance(offsets, list) or len(offsets) != m2.N_CLASSES:
        raise ValueError("M2 저장 절편 길이 오류")

    if not all(
        type(value) in (int, float)
        and isfinite(value)
        and abs(value) <= BIAS_LIMIT
        for value in offsets
    ):
        raise ValueError("M2 저장 절편 값 오류")

    if variant in ("global", "conditional") and calibration != "none":
        raise ValueError("빈도 기준에는 보정기를 추가하지 않습니다.")

    if calibration == "none" and temperature != 1.0:
        raise ValueError("미보정 후보의 Temperature는 1이어야 합니다.")

    if calibration != "temperature_bias" and any(offsets):
        raise ValueError("허용되지 않은 클래스별 보정 절편")

    if model["base"]["kind"] == "conditional":
        for key, counts in model["base"]["groups"].items():
            outs_text, runners_text = key.split(":")
            runners = int(runners_text)

            state: dict[str, object] = {
                "outs": int(outs_text),
                "on_first": int(bool(runners & 1)),
                "on_second": int(bool(runners & 2)),
                "on_third": int(bool(runners & 4)),
            }

            mask = allowed_mask(state)

            if any(
                value > 0 and not allowed
                for value, allowed in zip(
                    counts, mask, strict=True
                )
            ):
                raise ValueError("조건부 저장 자산의 규칙 충돌")


def candidate_name(model: OperationalModel) -> str:
    """선정 후보의 비교용 고유 이름을 반환한다."""

    if model["calibration"] == "none":
        return model["variant"]

    return f"{model['variant']}_{model['calibration']}"


def predict_probabilities(
    model: OperationalModel,
    inputs: Mapping[str, object],
) -> list[float]:
    """유효한 운영 모델의 최종 공동 확률을 계산한다."""

    base = masked_probabilities(
        model["base"], inputs
    )

    result = calibrate_vector(
        base,
        model["temperature"],
        model["class_offsets"],
    )

    m2.check_probabilities(result)

    if any(
        (allowed and probability <= 0.0)
        or (not allowed and probability != 0.0)
        for probability, allowed in zip(
            result, allowed_mask(inputs), strict=True
        )
    ):
        raise ValueError("M2 운영 확률의 규칙 또는 평활화 위반")

    return result


def calibration_loss(
    probabilities: Sequence[Sequence[float]],
    targets: Sequence[int],
    temperature: float,
    offsets: Sequence[float],
) -> float:
    """보정 구간의 평균 Log loss를 계산한다."""

    if not targets:
        raise ValueError("확률 보정 표본이 필요합니다.")

    total = 0.0

    for vector, target in zip(
        probabilities, targets, strict=True
    ):
        calibrated = calibrate_vector(
            vector, temperature, offsets
        )

        total -= log(
            max(
                calibrated[target],
                evaluation.LOG_EPSILON,
            )
        )

    return total / len(targets)


def fit_calibrations(
    model: OperationalModel,
    samples: Sequence[m2.Sample],
) -> tuple[
    OperationalModel,
    OperationalModel,
    dict[str, object],
]:
    """보정 구간에서 Temperature와 절편을 순서대로 적합한다."""

    if (
        model["variant"] not in ("logistic", "regularized")
        or model["calibration"] != "none"
    ):
        raise ValueError("미보정 로지스틱 후보가 필요합니다.")

    if not samples or any(
        not "2025-01-01" <= sample.day < CALIBRATION_BOUNDARY
        for sample in samples
    ):
        raise ValueError("보정 적합에 허용되지 않은 날짜가 포함되었습니다.")

    vectors = [
        predict_probabilities(model, sample.inputs)
        for sample in samples
    ]

    targets = [
        m2.CLASS_INDEX[m2.target_class(sample.target)]
        for sample in samples
    ]

    if any(
        vector[target] <= 0.0
        for vector, target in zip(
            vectors, targets, strict=True
        )
    ):
        raise ValueError("확률 보정 정답의 규칙 충돌")

    zero_offsets = [0.0] * m2.N_CLASSES

    scores = {
        temperature: calibration_loss(
            vectors,
            targets,
            temperature,
            zero_offsets,
        )
        for temperature in TEMPERATURES
    }

    temperature = min(
        TEMPERATURES,
        key=lambda value: (
            scores[value],
            abs(value - 1.0),
            value,
        ),
    )

    temperature_model = make_model(
        model["base"],
        model["variant"],
        "temperature",
        temperature,
    )

    support = Counter(targets)

    supported_indices = [
        index
        for index in range(m2.N_CLASSES)
        if support[index] >= MIN_CALIBRATION_SUPPORT
    ]

    offsets = zero_offsets.copy()
    n = len(samples)

    # 다중분류 Log loss의 평균 기울기와 L2 규제를 이용한다.
    # 실제 정답이 부족한 클래스의 절편은 0으로 고정한다.
    for _ in range(BIAS_ITERATIONS):
        predicted_mass = [0.0] * m2.N_CLASSES

        for vector in vectors:
            calibrated = calibrate_vector(
                vector, temperature, offsets
            )

            for index in supported_indices:
                predicted_mass[index] += calibrated[index]

        for index in supported_indices:
            gradient = (
                (predicted_mass[index] - support[index]) / n
                + BIAS_L2 * offsets[index]
            )

            updated = (
                offsets[index] - BIAS_LEARNING_RATE * gradient
            )

            offsets[index] = max(
                -BIAS_LIMIT,
                min(BIAS_LIMIT, updated),
            )

    bias_model = make_model(
        model["base"],
        model["variant"],
        "temperature_bias",
        temperature,
        offsets,
    )

    report = {
        "period": period_summary(samples),
        "temperature_grid": [
            {
                "temperature": value,
                "log_loss": scores[value],
            }
            for value in TEMPERATURES
        ],
        "selected_temperature": temperature,
        "class_support": {
            name: support[index]
            for index, name in enumerate(m2.CLASSES)
        },
        "supported_bias_classes": [
            m2.CLASSES[index]
            for index in supported_indices
        ],
        "class_offsets": dict(
            zip(m2.CLASSES, offsets, strict=True)
        ),
        "unadjusted_log_loss": scores[1.0],
        "temperature_log_loss": scores[temperature],
        "temperature_bias_log_loss": calibration_loss(
            vectors,
            targets,
            temperature,
            offsets,
        ),
    }

    return temperature_model, bias_model, report


def score_predictions(
    samples: Sequence[m2.Sample],
    predictions: Sequence[Sequence[float]],
    *,
    require_rules: bool,
) -> DetailedMetrics:
    """기존 다중분류 평가와 ECE·규칙 위반을 공동 집계한다."""

    result = evaluation.metrics(samples, predictions)

    n = result["samples"]

    classwise_ece: dict[str, float] = {}

    for name, bins in result["calibration"].items():
        error = 0.0

        for interval in bins:
            count = int(str(interval["count"]))

            if not count:
                continue

            predicted = float(
                str(interval["mean_probability"])
            )
            observed = float(
                str(interval["observed_rate"])
            )

            error += count * abs(predicted - observed)

        classwise_ece[name] = error / n

    forbidden_mass = [0.0] * m2.N_CLASSES
    nonzero_samples = 0
    target_conflicts = 0

    for sample, probabilities in zip(
        samples, predictions, strict=True
    ):
        mask = allowed_mask(sample.inputs)

        target = m2.CLASS_INDEX[
            m2.target_class(sample.target)
        ]

        if not mask[target]:
            target_conflicts += 1

        has_forbidden_mass = False

        for index, allowed in enumerate(mask):
            if allowed:
                continue

            probability = probabilities[index]
            forbidden_mass[index] += probability

            if probability > 0.0:
                has_forbidden_mass = True

        nonzero_samples += int(has_forbidden_mass)

    if require_rules and (nonzero_samples or target_conflicts):
        raise ValueError("운영 후보에서 규칙 위반이 발견되었습니다.")

    return {
        **result,
        "classwise_ece": classwise_ece,
        "macro_classwise_ece": (
            fsum(classwise_ece.values()) / m2.N_CLASSES
        ),
        "forbidden_probability_mean": (
            fsum(forbidden_mass) / n
        ),
        "forbidden_nonzero_samples": nonzero_samples,
        "forbidden_class_probability_mean": {
            name: forbidden_mass[index] / n
            for index, name in enumerate(m2.CLASSES)
        },
        "forbidden_target_count": target_conflicts,
    }


def improves(
    candidate: DetailedMetrics,
    baseline: DetailedMetrics,
    fields: Sequence[str],
) -> bool:
    """같은 분모에서 지정한 손실이 악화되지 않는지 검사한다."""

    if candidate["samples"] != baseline["samples"]:
        raise ValueError("M2 비교 평가 분모 불일치")

    candidate_values = [
        float(candidate[field])
        for field in fields
    ]

    baseline_values = [
        float(baseline[field])
        for field in fields
    ]

    pairs = list(
        zip(candidate_values, baseline_values, strict=True)
    )

    return (
        all(current <= reference for current, reference in pairs)
        and any(current < reference for current, reference in pairs)
    )


def select_model(
    results: Mapping[str, DetailedMetrics],
    candidates: Mapping[str, OperationalModel],
) -> tuple[str, dict[str, object]]:
    """선정 구간에서만 최종 운영 분류기 하나를 선택한다."""

    if (
        set(results) != set(candidates)
        or not {"global", "conditional"} <= set(results)
    ):
        raise ValueError("M2 선정 후보와 평가 결과 불일치")

    if len({
        item["samples"] for item in results.values()
    }) != 1:
        raise ValueError("M2 선정 후보의 평가 분모가 다릅니다.")

    for name, candidate in candidates.items():
        if name != candidate_name(candidate):
            raise ValueError("M2 선정 후보 이름 불일치")

        if (
            results[name]["forbidden_nonzero_samples"] != 0
            or results[name]["forbidden_target_count"] != 0
        ):
            raise ValueError("규칙 위반 후보는 선정할 수 없습니다.")

    baseline = min(
        ("global", "conditional"),
        key=lambda name: (
            results[name]["log_loss"],
            results[name]["brier"],
        ),
    )

    eligible: list[str] = []
    decisions: dict[str, dict[str, object]] = {}

    for name, candidate in candidates.items():
        if candidate["variant"] in ("global", "conditional"):
            continue

        baseline_ok = improves(
            results[name],
            results[baseline],
            ("log_loss", "brier"),
        )

        calibration_ok = True

        if candidate["calibration"] != "none":
            parent = candidate["variant"]

            if parent not in results:
                raise ValueError("확률 보정 부모 후보가 없습니다.")

            calibration_ok = improves(
                results[name],
                results[parent],
                (
                    "log_loss",
                    "brier",
                    "macro_classwise_ece",
                ),
            )

        accepted = baseline_ok and calibration_ok

        decisions[name] = {
            "beats_frequency_baseline": baseline_ok,
            "calibration_gate_passed": calibration_ok,
            "eligible": accepted,
        }

        if accepted:
            eligible.append(name)

    if eligible:
        selected = min(
            eligible,
            key=lambda name: (
                results[name]["log_loss"],
                results[name]["brier"],
                results[name]["macro_classwise_ece"],
                candidates[name]["calibration"] != "none",
                candidates[name]["variant"] != "logistic",
                name,
            ),
        )

        reason = "빈도 기준 및 해당 보정 조건을 충족한 후보 중 선정"

    else:
        selected = baseline
        reason = "적격 로지스틱 후보가 없어 우수 빈도 기준 선정"

    return selected, {
        "baseline": baseline,
        "reason": reason,
        "candidate_decisions": decisions,
        "eligible_logistic_candidates": eligible,
    }


def predict(
    model: OperationalModel,
    inputs: Mapping[str, object],
) -> dict[str, object]:
    """v2 운영 추론의 클래스별 확률 JSON을 생성한다."""

    validate_model(model)

    probabilities = predict_probabilities(model, inputs)
    mask = allowed_mask(inputs)

    best = max(
        range(m2.N_CLASSES),
        key=probabilities.__getitem__,
    )

    return {
        "schema": SCHEMA,
        "competition": "regular",
        "prediction_point": "before_plate_appearance",
        "model_kind": model["base"]["kind"],
        "selected_candidate": candidate_name(model),
        "calibration_method": model["calibration"],
        "temperature": model["temperature"],
        "class_order": list(m2.CLASSES),
        "probabilities": dict(
            zip(m2.CLASSES, probabilities, strict=True)
        ),
        "forbidden_events": [
            name
            for name, allowed in zip(
                m2.CLASSES, mask, strict=True
            )
            if not allowed
        ],
        "most_likely_event": m2.CLASSES[best],
        "most_likely_event_name": (
            m2.CLASS_NAMES[m2.CLASSES[best]]
        ),
    }
