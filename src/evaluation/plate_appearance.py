"""M2 다중분류 확률의 평가와 기준 대비 비교를 담당한다."""

from collections.abc import Iterable, Sequence
from math import log
from typing import TypedDict

from ..models.plate_appearance import model as m2


LOG_EPSILON = 1e-15
CALIBRATION_BINS = 10
TOP_CONFUSIONS = 10


class ClassMetrics(TypedDict):
    """범주별 정답 수와 분류 성능."""

    support: int
    predicted_count: int
    precision: float
    recall: float
    f1: float


class Metrics(TypedDict):
    """같은 평가 표본에서 계산한 분류·확률 지표."""

    samples: int
    log_loss: float
    brier: float
    accuracy: float
    macro_f1: float
    per_class: dict[str, ClassMetrics]
    confusion_matrix: dict[str, dict[str, int]]
    major_confusions: list[dict[str, object]]
    calibration: dict[str, list[dict[str, object]]]


def safe_ratio(numerator: float, denominator: float) -> float:
    """평가 분모가 0이면 0으로 기록한다."""

    return numerator / denominator if denominator else 0.0


def metrics(
    samples: Sequence[m2.Sample],
    predictions: Iterable[Sequence[float]],
) -> Metrics:
    """Log loss, Brier, Macro F1과 클래스별 신뢰도를 집계한다."""

    if not samples:
        raise ValueError("M2 평가 표본이 필요합니다.")

    size = m2.N_CLASSES
    confusion = [[0] * size for _ in range(size)]

    bin_counts = [
        [0] * CALIBRATION_BINS
        for _ in range(size)
    ]
    bin_probability = [
        [0.0] * CALIBRATION_BINS
        for _ in range(size)
    ]
    bin_observed = [
        [0] * CALIBRATION_BINS
        for _ in range(size)
    ]

    loss = 0.0
    brier = 0.0

    for sample, probabilities in zip(
        samples, predictions, strict=True
    ):
        m2.check_probabilities(probabilities)

        target = m2.CLASS_INDEX[m2.target_class(sample.target)]

        predicted = max(
            range(size),
            key=probabilities.__getitem__,
        )

        confusion[target][predicted] += 1

        # 절단은 Log loss 계산에서만 사용한다.
        # 실제 출력 확률은 변경하지 않는다.
        loss -= log(
            max(probabilities[target], LOG_EPSILON)
        )

        for index, probability in enumerate(probabilities):
            observed = int(index == target)

            # Brier는 범주별 제곱오차의 합을 사용한다.
            brier += (probability - observed) ** 2

            bucket = min(
                int(probability * CALIBRATION_BINS),
                CALIBRATION_BINS - 1,
            )

            bin_counts[index][bucket] += 1
            bin_probability[index][bucket] += probability
            bin_observed[index][bucket] += observed

    n = len(samples)
    per_class: dict[str, ClassMetrics] = {}
    calibration: dict[str, list[dict[str, object]]] = {}
    matrix: dict[str, dict[str, int]] = {}
    mistakes: list[dict[str, object]] = []

    for index, name in enumerate(m2.CLASSES):
        true_positive = confusion[index][index]
        support = sum(confusion[index])
        predicted_count = sum(
            row[index] for row in confusion
        )

        precision = safe_ratio(
            true_positive, predicted_count
        )
        recall = safe_ratio(
            true_positive, support
        )

        per_class[name] = {
            "support": support,
            "predicted_count": predicted_count,
            "precision": precision,
            "recall": recall,
            "f1": safe_ratio(
                2.0 * precision * recall,
                precision + recall,
            ),
        }

        matrix[name] = dict(
            zip(m2.CLASSES, confusion[index], strict=True)
        )

        for other, other_name in enumerate(m2.CLASSES):
            if other != index and confusion[index][other]:
                mistakes.append(
                    {
                        "actual": name,
                        "predicted": other_name,
                        "count": confusion[index][other],
                    }
                )

        intervals: list[dict[str, object]] = []

        for bucket in range(CALIBRATION_BINS):
            count = bin_counts[index][bucket]

            intervals.append(
                {
                    "lower": bucket / CALIBRATION_BINS,
                    "upper": (bucket + 1) / CALIBRATION_BINS,
                    "upper_inclusive": (
                        bucket == CALIBRATION_BINS - 1
                    ),
                    "count": count,
                    "mean_probability": (
                        bin_probability[index][bucket] / count
                        if count else None
                    ),
                    "observed_rate": (
                        bin_observed[index][bucket] / count
                        if count else None
                    ),
                }
            )

        calibration[name] = intervals

    mistakes.sort(
        key=lambda item: -int(str(item["count"]))
    )

    return {
        "samples": n,
        "log_loss": loss / n,
        "brier": brier / n,
        "accuracy": sum(
            confusion[index][index]
            for index in range(size)
        ) / n,
        "macro_f1": sum(
            value["f1"]
            for value in per_class.values()
        ) / size,
        "per_class": per_class,
        "confusion_matrix": matrix,
        "major_confusions": mistakes[:TOP_CONFUSIONS],
        "calibration": calibration,
    }


def evaluate(
    model: m2.Model,
    samples: Sequence[m2.Sample],
) -> Metrics:
    """하나의 후보를 공동 Validation 표본에서 평가한다."""

    return metrics(
        samples,
        (
            m2.predict_probabilities(model, sample.inputs)
            for sample in samples
        ),
    )


def differences(
    candidate: Metrics,
    baseline: Metrics,
) -> dict[str, float]:
    """같은 평가 분모에서 후보와 기준의 지표 차이를 계산한다."""

    if candidate["samples"] != baseline["samples"]:
        raise ValueError("M2 비교 평가 분모 불일치")

    return {
        "log_loss": (
            candidate["log_loss"] - baseline["log_loss"]
        ),
        "brier": candidate["brier"] - baseline["brier"],
        "macro_f1": (
            candidate["macro_f1"] - baseline["macro_f1"]
        ),
        "accuracy": (
            candidate["accuracy"] - baseline["accuracy"]
        ),
    }


def select(
    results: dict[str, Metrics],
) -> tuple[str, str]:
    """평가 전에 정한 선택 규칙에 따라 분류기 하나를 고른다."""

    if set(results) != {
        "global", "conditional", "logistic"
    }:
        raise ValueError("M2 후보 평가 결과 누락")

    if len({
        item["samples"]
        for item in results.values()
    }) != 1:
        raise ValueError("M2 후보의 평가 분모가 다릅니다.")

    # 기준끼리는 Log loss 우선, Brier 차순으로 비교한다.
    # 완전 동률이면 단순한 전체 빈도 기준을 유지한다.
    baseline = min(
        ("global", "conditional"),
        key=lambda name: (
            results[name]["log_loss"],
            results[name]["brier"],
        ),
    )

    candidate = results["logistic"]
    reference = results[baseline]

    loss_ok = candidate["log_loss"] <= reference["log_loss"]
    brier_ok = candidate["brier"] <= reference["brier"]

    improved = (
        candidate["log_loss"] < reference["log_loss"]
        or candidate["brier"] < reference["brier"]
    )

    if loss_ok and brier_ok and improved:
        return (
            "logistic",
            f"{baseline} 대비 Log loss와 Brier 기준 충족",
        )

    return (
        baseline,
        "다항 로지스틱 회귀의 공동 확률 지표 개선 기준 미충족",
    )