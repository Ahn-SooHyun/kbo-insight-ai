"""M5 공동·구종·결과 확률 평가 및 후보 비교."""

from collections import Counter
from collections.abc import Mapping, Sequence
from math import fsum, log

from ..models.pitch_outcome import model as m5


CALIBRATION_BINS = 10
RARE_SUPPORT = 50
ERROR_LIMIT = 20

AXES = ("joint", "pitch_type", "pitch_result")
SUPPORT_GROUPS = ("unseen", "rare", "observed")


class Bin:
    """하나의 보정 구간에서 확률과 실제 사건 수를 누적한다."""

    __slots__ = ("count", "probability_sum", "positive")

    def __init__(self) -> None:
        self.count = 0
        self.probability_sum = 0.0
        self.positive = 0

    def add(self, probability: float, observed: bool) -> None:
        self.count += 1
        self.probability_sum += probability
        self.positive += int(observed)

    def report(self, index: int) -> dict[str, object]:
        return {
            "lower": index / CALIBRATION_BINS,
            "upper": (index + 1) / CALIBRATION_BINS,
            "count": self.count,
            "positive": self.positive,
            "mean_probability": (
                self.probability_sum / self.count
                if self.count else None
            ),
            "observed_rate": (
                self.positive / self.count
                if self.count else None
            ),
        }


def bin_index(probability: float) -> int:
    """확률 1도 마지막 보정 구간으로 배정한다."""
    return min(
        int(probability * CALIBRATION_BINS),
        CALIBRATION_BINS - 1,
    )


def expected_calibration_error(bins: Sequence[Bin]) -> float:
    """구간 분모를 가중치로 사용해 ECE를 계산한다."""
    total = sum(item.count for item in bins)

    m5.require(total > 0, "보정 평가 분모 오류")

    return fsum(
        abs(item.probability_sum - item.positive)
        for item in bins
    ) / total


class Score:
    """한 확률 좌표계의 전체·범주별 성능을 누적한다."""

    def __init__(
        self,
        labels: Sequence[str],
        train_counts: Sequence[int],
    ) -> None:
        m5.require(
            len(labels) == len(train_counts),
            "평가 범주와 Train 지원 차원 불일치",
        )

        self.labels = tuple(labels)
        self.train_counts = tuple(train_counts)

        size = len(labels)

        self.count = 0
        self.total_log_loss = 0.0
        self.total_brier = 0.0
        self.correct = 0

        self.support = [0] * size
        self.predicted = [0] * size
        self.true_positive = [0] * size
        self.actual_loss = [0.0] * size
        self.class_brier = [0.0] * size

        self.top_bins = [
            Bin() for _ in range(CALIBRATION_BINS)
        ]

        self.class_bins = [
            [Bin() for _ in range(CALIBRATION_BINS)]
            for _ in range(size)
        ]

        self.errors: Counter[tuple[int, int]] = Counter()
        self.group_count: Counter[str] = Counter()
        self.group_loss: Counter[str] = Counter()
        self.group_brier: Counter[str] = Counter()

    def add(
        self,
        target: int,
        probabilities: Sequence[float],
    ) -> None:
        """하나의 표본을 전체·지원 그룹·보정 통계에 반영한다."""
        m5.require(
            len(probabilities) == len(self.labels)
            and 0 <= target < len(self.labels),
            "평가 확률 또는 정답 차원 오류",
        )

        actual_probability = probabilities[target]

        # 모든 출처 계약 범주에 양의 평활 확률을 부여한다.
        # 0 확률을 평가 과정에서 은폐하거나 임의 치환하지 않는다.
        m5.require(actual_probability > 0.0, "정답 확률이 0입니다.")

        loss = -log(actual_probability)
        prediction = max(
            range(len(probabilities)),
            key=probabilities.__getitem__,
        )

        self.count += 1
        self.total_log_loss += loss
        self.support[target] += 1
        self.predicted[prediction] += 1
        self.actual_loss[target] += loss

        correct = prediction == target

        if correct:
            self.correct += 1
            self.true_positive[target] += 1
        else:
            self.errors[(target, prediction)] += 1

        self.top_bins[
            bin_index(probabilities[prediction])
        ].add(probabilities[prediction], correct)

        brier = 0.0

        for index, probability in enumerate(probabilities):
            observed = index == target
            squared_error = (
                probability - float(observed)
            ) ** 2

            self.class_brier[index] += squared_error
            brier += squared_error

            self.class_bins[index][
                bin_index(probability)
            ].add(probability, observed)

        self.total_brier += brier

        train_support = self.train_counts[target]

        if train_support == 0:
            group = "unseen"
        elif train_support < RARE_SUPPORT:
            group = "rare"
        else:
            group = "observed"

        self.group_count[group] += 1
        self.group_loss[group] += loss
        self.group_brier[group] += brier

    def report(self) -> dict[str, object]:
        """평가 분모와 지표, 대표 오류 및 보정 결과를 반환한다."""
        m5.require(self.count > 0, "평가 표본이 없습니다.")

        classes: dict[str, object] = {}

        for index, label in enumerate(self.labels):
            support = self.support[index]
            predicted = self.predicted[index]
            tp = self.true_positive[index]
            bins = self.class_bins[index]

            m5.require(
                sum(item.count for item in bins) == self.count,
                "클래스별 보정 분모 불일치",
            )

            classes[label] = {
                "train_support": self.train_counts[index],
                "validation_support": support,
                "predicted_count": predicted,
                "true_positive": tp,
                "precision": tp / predicted if predicted else None,
                "recall": tp / support if support else None,
                "actual_class_log_loss": (
                    self.actual_loss[index] / support
                    if support else None
                ),
                "one_vs_rest_brier": (
                    self.class_brier[index] / self.count
                ),
                "one_vs_rest_ece": expected_calibration_error(bins),
                "calibration": [
                    item.report(i)
                    for i, item in enumerate(bins)
                ],
            }

        groups = {
            group: {
                "count": self.group_count[group],
                "fraction": self.group_count[group] / self.count,
                "log_loss": (
                    self.group_loss[group] / self.group_count[group]
                    if self.group_count[group] else None
                ),
                "brier": (
                    self.group_brier[group] / self.group_count[group]
                    if self.group_count[group] else None
                ),
            }
            for group in SUPPORT_GROUPS
        }

        m5.require(
            sum(self.group_count.values()) == self.count
            and sum(self.support) == self.count
            and sum(item.count for item in self.top_bins)
            == self.count,
            "평가 지원 또는 보정 분모 불일치",
        )

        return {
            "samples": self.count,
            "log_loss": self.total_log_loss / self.count,
            "brier": self.total_brier / self.count,
            "accuracy": self.correct / self.count,
            "top_label_ece": expected_calibration_error(
                self.top_bins
            ),
            "top_label_calibration": [
                item.report(index)
                for index, item in enumerate(self.top_bins)
            ],
            "classes": classes,
            "train_support_groups": groups,
            "train_observed_support_rate": (
                1.0 - self.group_count["unseen"] / self.count
            ),
            "output_dictionary_coverage": 1.0,
            "unsupported_output_samples": 0,
            "representative_errors": [
                {
                    "actual": self.labels[actual],
                    "predicted": self.labels[predicted],
                    "count": count,
                }
                for (actual, predicted), count
                in self.errors.most_common(ERROR_LIMIT)
            ],
        }


def axis_counts(
    joint_counts: Sequence[int],
) -> dict[str, list[int]]:
    """학습 공동 빈도를 동일한 구종·결과 좌표계로 합산한다."""
    m5.require(
        len(joint_counts) == m5.N_CLASSES,
        "학습 공동 빈도 차원 오류",
    )

    return {
        "joint": list(joint_counts),
        "pitch_type": [
            sum(
                joint_counts[
                    index * m5.N_RESULTS:
                    (index + 1) * m5.N_RESULTS
                ]
            )
            for index in range(len(m5.TYPES))
        ],
        "pitch_result": [
            sum(joint_counts[index::m5.N_RESULTS])
            for index in range(m5.N_RESULTS)
        ],
    }


def evaluate(
    model: m5.Model,
    samples: Sequence[m5.Sample],
) -> dict[str, object]:
    """같은 Validation 표본에서 공동·주변 성능을 평가한다."""
    runtime = m5.Runtime(model)
    counts = axis_counts(model["counts"])

    scores = {
        "joint": Score(m5.CLASSES, counts["joint"]),
        "pitch_type": Score(m5.TYPES, counts["pitch_type"]),
        "pitch_result": Score(
            m5.RESULTS, counts["pitch_result"]
        ),
    }

    for sample in samples:
        probabilities = runtime.probabilities(sample.keys)
        pitch_type, pitch_result = m5.marginalize(probabilities)

        scores["joint"].add(sample.target, probabilities)

        scores["pitch_type"].add(
            sample.target // m5.N_RESULTS,
            pitch_type,
        )

        scores["pitch_result"].add(
            sample.target % m5.N_RESULTS,
            pitch_result,
        )

    return {
        axis: score.report()
        for axis, score in scores.items()
    }


def select(
    evaluations: Mapping[str, dict[str, object]],
) -> str:
    """고정된 공동 Log loss·Brier·단순성 순서로 하나를 선정한다."""
    m5.require(
        set(evaluations) == set(m5.METHODS),
        "후보 평가 누락",
    )

    sample_counts = {
        evaluations[name]["joint"]["samples"]
        for name in m5.METHODS
    }

    m5.require(
        len(sample_counts) == 1,
        "후보별 Validation 평가 분모 불일치",
    )

    return min(
        m5.METHODS,
        key=lambda name: (
            evaluations[name]["joint"]["log_loss"],
            evaluations[name]["joint"]["brier"],
            m5.METHODS.index(name),
        ),
    )


def compare(
    selected: str,
    evaluations: Mapping[str, dict[str, object]],
) -> dict[str, object]:
    """선정 모델과 두 기준 모델의 지표 차이를 모두 공개한다."""
    comparisons: dict[str, object] = {}

    for baseline in ("frequency", "conditional_frequency"):
        comparisons[baseline] = {
            axis: {
                metric: (
                    evaluations[selected][axis][metric]
                    - evaluations[baseline][axis][metric]
                )
                for metric in (
                    "log_loss",
                    "brier",
                    "top_label_ece",
                )
            }
            for axis in AXES
        }

    return comparisons