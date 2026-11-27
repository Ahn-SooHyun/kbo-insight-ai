"""M5 시간순 개발 평가·규제·온도 보정 및 후보 선정."""

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
import logging
from math import exp, fsum, isfinite, log, sqrt
from pathlib import Path
from random import Random
import sqlite3
from typing import TypedDict

from . import data
from . import model as m5
from ...datasets.published_sources import as_row
from ...evaluation import pitch_outcome as evaluation


LOGGER = logging.getLogger(__name__)

CALIBRATION_END = "2025-07-01"
SELECTION_END = "2025-09-01"

PERIODS = (
    "calibration",
    "selection",
    "reporting",
)

CANDIDATES = (
    "frequency",
    "conditional_frequency",
    "logistic",
    "logistic_temperature",
    "regularized",
    "regularized_temperature",
)

FREQUENCIES = (
    "frequency",
    "conditional_frequency",
)

CALIBRATED_PARENTS = {
    "logistic_temperature": "logistic",
    "regularized_temperature": "regularized",
}

TEMPERATURES = (
    0.8,
    1.0,
    1.15,
    1.3,
    1.5,
    2.0,
)

L2 = 0.00003
AVERAGE_EPOCHS = 2
SELECTION_TOLERANCE = 1e-10

PRIMARY_METRICS = (
    "log_loss",
    "brier",
)

CALIBRATION_METRICS = (
    "log_loss",
    "brier",
    "top_label_ece",
    "macro_classwise_ece",
)

AXIS_LABELS = {
    "joint": m5.CLASSES,
    "pitch_type": m5.TYPES,
    "pitch_result": m5.RESULTS,
}


@dataclass(slots=True)
class Partition:
    """시간순 개발 구간의 표본과 독립적인 분할 정보를 보존한다."""

    rows: dict[str, list[m5.Sample]]
    summary: dict[str, dict[str, object]]


class TemperatureFit(TypedDict):
    """보정 구간에서만 결정된 단일 온도와 비교 기록."""

    samples: int
    temperature: float
    losses: list[dict[str, float]]


def period_for(day: str) -> str:
    """2025년 Validation 날짜를 고정된 개발 구간에 배정한다."""
    m5.require(
        data.expected_split(day) == "validation",
        "M5 보정·선정 구간은 2025년 Validation에 한정합니다.",
    )

    if day < CALIBRATION_END:
        return "calibration"

    if day < SELECTION_END:
        return "selection"

    return "reporting"


def partition_rows(
    samples: Sequence[m5.Sample],
    metadata: Iterable[tuple[str, int, int, str]],
) -> Partition:
    """기존 SQL 정렬 순서의 표본과 감사 인덱스를 일대일 연결한다."""
    rows: dict[str, list[m5.Sample]] = {
        name: [] for name in PERIODS
    }

    games: dict[str, set[str]] = {
        name: set() for name in PERIODS
    }

    days: dict[str, set[str]] = {
        name: set() for name in PERIODS
    }

    game_dates: dict[str, str] = {}
    previous_key: tuple[str, int, int] | None = None

    # zip(strict=True)는 인덱스와 학습 표본의 길이 차이를 거부한다.
    for sample, raw in zip(samples, metadata, strict=True):
        game_id, pa_number, pitch_number, day = raw

        game_id = m5.identifier(game_id, "game_id")

        key = (
            game_id,
            m5.integer(pa_number, "pa_number", 1),
            m5.integer(pitch_number, "pitch_number", 1),
        )

        m5.require(
            previous_key is None or key > previous_key,
            "Validation 감사 인덱스 순서 또는 중복 오류",
        )

        previous_key = key

        period = period_for(day)

        if game_id in game_dates:
            m5.require(
                game_dates[game_id] == day,
                "같은 경기의 개발 구간 또는 날짜 혼합",
            )
        else:
            game_dates[game_id] = day

        rows[period].append(sample)
        games[period].add(game_id)
        days[period].add(day)

    summary: dict[str, dict[str, object]] = {}

    for period in PERIODS:
        m5.require(
            bool(rows[period]),
            f"{period}: 개발 평가 표본이 없습니다.",
        )

        summary[period] = {
            "samples": len(rows[period]),
            "games": len(games[period]),
            "first_date": min(days[period]),
            "last_date": max(days[period]),
        }

    m5.require(
        sum(len(value) for value in rows.values()) == len(samples)
        and sum(len(value) for value in games.values())
        == len(game_dates),
        "개발 구간의 표본 또는 경기 분모 오류",
    )

    return Partition(rows, summary)


def partition_from_source(
    dataset: data.Dataset,
) -> Partition:
    """검증된 인수 DB에서 Validation의 감사 인덱스만 다시 읽는다."""
    provenance = dataset.provenance

    database = Path(str(provenance["database"]))
    standard = Path(str(provenance["standard_database"]))

    with closing(sqlite3.connect(
        database.resolve().as_uri() + "?mode=ro",
        uri=True,
    )) as db:
        db.execute(
            "ATTACH DATABASE ? AS verified",
            (standard.resolve().as_uri() + "?mode=ro",),
        )

        db.execute("PRAGMA query_only=ON")

        # data.row_query와 동일한 경기 제한 및 정렬을 사용한다.
        # 정답 열과 학습 특징은 추가 조회하지 않는다.
        metadata = db.execute(
            """
            SELECT
                x.game_id,
                x.pa_number,
                x.pitch_number,
                x.sample_date
            FROM m5_index x
            JOIN verified.games g
                ON g.game_id = x.game_id
            WHERE x.split = 'validation'
              AND g.competition = 'regular'
            ORDER BY
                x.game_id,
                x.pa_number,
                x.pitch_number
            """
        )

        partition = partition_rows(
            dataset.rows["validation"],
            metadata,
        )

    for name in PERIODS:
        summary = partition.summary[name]

        LOGGER.info(
            "M5 development=%s samples=%d games=%d dates=%s~%s",
            name,
            summary["samples"],
            summary["games"],
            summary["first_date"],
            summary["last_date"],
        )

    return partition


def active_names(
    weights: Mapping[str, list[float]],
    keys: Sequence[str],
) -> list[str]:
    """기존 특징 사전과 동일한 미출현 선수 후퇴 규칙을 적용한다."""
    names = ["bias"]

    for key in keys:
        if key in weights:
            names.append(key)
            continue

        fallback = m5.fallback_key(key)

        if fallback is not None and fallback in weights:
            names.append(fallback)

    m5.require(
        len(names) == len(set(names)),
        "동일 희소 특징 중복 활성화",
    )

    return names


def materialize_decay(
    coefficients: list[float],
    decay: float,
    elapsed_steps: int,
) -> None:
    """누적된 L2 감쇠를 필요한 계수에만 적용한다."""
    if elapsed_steps == 0:
        return

    m5.require(
        elapsed_steps > 0 and 0.0 < decay <= 1.0,
        "지연 L2 감쇠 설정 오류",
    )

    factor = decay ** elapsed_steps

    coefficients[:] = [
        value * factor for value in coefficients
    ]


def fit_regularized(
    samples: Sequence[m5.Sample],
    seed: int,
) -> m5.Model:
    """Train 전용 L2 규제 SGD와 마지막 두 epoch 계수 평균을 적합한다."""
    m5.require(
        1 <= AVERAGE_EPOCHS <= m5.EPOCHS,
        "계수 평균화 epoch 설정 오류",
    )

    counts = m5.training_counts(samples)

    occurrences: Counter[str] = Counter(
        key
        for sample in samples
        for key in sample.keys
    )

    vocabulary = {
        key
        for key, support in occurrences.items()
        if not key.startswith("id:")
        or support >= m5.MIN_ID_COUNT
    }

    vocabulary.update((
        m5.BATTER_OTHER,
        m5.PITCHER_OTHER,
    ))

    weights = {
        key: [0.0] * m5.N_CLASSES
        for key in sorted(vocabulary)
    }

    weights["bias"] = [
        log(probability)
        for probability in m5.global_prior(counts)
    ]

    averaged = {
        name: [0.0] * m5.N_CLASSES
        for name in weights
    }

    rng = Random(seed)
    order = list(range(len(samples)))

    for epoch in range(m5.EPOCHS):
        rng.shuffle(order)

        rate = m5.LEARNING_RATE / sqrt(epoch + 1)
        decay = 1.0 - rate * L2

        m5.require(
            0.0 < decay <= 1.0,
            "L2 학습률 또는 규제 계수 오류",
        )

        last_steps = {
            name: 0
            for name in weights
            if name != "bias"
        }

        total_loss = 0.0

        for step, sample_index in enumerate(order):
            sample = samples[sample_index]

            names = active_names(weights, sample.keys)

            # 사용하지 않았던 특징도 매 SGD 단계의 규제를 받는다.
            # 다만 실제 계산은 다음 활성화 시점까지 지연한다.
            for name in names:
                if name == "bias":
                    continue

                materialize_decay(
                    weights[name],
                    decay,
                    step - last_steps[name],
                )

            active = [
                weights[name] for name in names
            ]

            probabilities = m5.softmax(
                m5.logits_for(active)
            )

            total_loss -= log(
                max(probabilities[sample.target], 1e-15)
            )

            gradient = [
                -rate * probability
                for probability in probabilities
            ]

            gradient[sample.target] += rate

            for name in names:
                coefficients = weights[name]

                # 절편은 규제하지 않고 특징 계수만 L2로 축소한다.
                factor = (
                    1.0 if name == "bias" else decay
                )

                coefficients[:] = [
                    factor * weight + update
                    for weight, update in zip(
                        coefficients, gradient
                    )
                ]

                if name != "bias":
                    last_steps[name] = step + 1

        # 다음 epoch와 계수 평균 전에 모든 지연 감쇠를 확정한다.
        for name, last_step in last_steps.items():
            materialize_decay(
                weights[name],
                decay,
                len(samples) - last_step,
            )

        if epoch >= m5.EPOCHS - AVERAGE_EPOCHS:
            for name, coefficients in weights.items():
                averaged[name] = [
                    total + value
                    for total, value in zip(
                        averaged[name], coefficients
                    )
                ]

        LOGGER.info(
            "M5 regularized epoch=%d/%d train_log_loss=%.8f",
            epoch + 1,
            m5.EPOCHS,
            total_loss / len(samples),
        )

    final_weights = {
        name: [
            value / AVERAGE_EPOCHS
            for value in coefficients
        ]
        for name, coefficients in averaged.items()
    }

    model: m5.Model = {
        "kind": "logistic",
        "counts": counts,
        "groups": {},
        "weights": final_weights,
    }

    m5.validate_model(model)

    return model


def scaled_model(
    model: m5.Model,
    temperature: float,
) -> m5.Model:
    """로짓을 T로 나누는 효과를 하나의 분류기 계수에 반영한다."""
    m5.validate_model(model)

    m5.require(
        model["kind"] == "logistic"
        and isfinite(temperature)
        and temperature in TEMPERATURES,
        "지원하지 않는 온도 보정 또는 모델 종류",
    )

    result: m5.Model = {
        "kind": "logistic",
        "counts": model["counts"].copy(),
        "groups": {},
        "weights": {
            name: [
                value / temperature
                for value in coefficients
            ]
            for name, coefficients in model["weights"].items()
        },
    }

    m5.validate_model(result)

    return result


def temperature_key(
    item: Mapping[str, float],
) -> tuple[float, float, float]:
    """동률이면 온도 1에 가까운 단순한 변환을 우선한다."""
    temperature = item["temperature"]

    return (
        item["log_loss"],
        abs(temperature - 1.0),
        temperature,
    )


def fit_temperature(
    model: m5.Model,
    samples: Sequence[m5.Sample],
) -> TemperatureFit:
    """보정 구간에서만 고정 온도 후보의 실제 공동 NLL을 비교한다."""
    m5.validate_model(model)

    m5.require(
        model["kind"] == "logistic" and bool(samples),
        "온도 보정에는 로지스틱 모델과 보정 표본이 필요합니다.",
    )

    prior = m5.global_prior(model["counts"])
    losses = [0.0] * len(TEMPERATURES)

    for sample in samples:
        target = m5.integer(
            sample.target,
            "target",
            0,
            m5.N_CLASSES - 1,
        )

        active = m5.active_weights(
            model["weights"], sample.keys
        )

        logits = m5.logits_for(active)

        m5.require(
            all(isfinite(value) for value in logits),
            "온도 보정 로짓 오류",
        )

        maximum = max(logits)
        shifted = [
            value - maximum for value in logits
        ]

        # 로짓 계산은 표본마다 한 번만 수행한다.
        # 모든 후보에서 기존 Train 사전 혼합 정책을 보존한다.
        for index, temperature in enumerate(TEMPERATURES):
            values = [
                exp(value / temperature)
                for value in shifted
            ]

            probability = (
                (1.0 - m5.PRIOR_MIX)
                * values[target] / fsum(values)
                + m5.PRIOR_MIX * prior[target]
            )

            m5.require(
                isfinite(probability)
                and 0.0 < probability <= 1.0,
                "온도 보정 정답 확률 오류",
            )

            losses[index] -= log(probability)

    records = [
        {
            "temperature": temperature,
            "log_loss": loss / len(samples),
        }
        for temperature, loss in zip(
            TEMPERATURES, losses
        )
    ]

    selected = min(records, key=temperature_key)

    result: TemperatureFit = {
        "samples": len(samples),
        "temperature": selected["temperature"],
        "losses": records,
    }

    validate_temperature_fit(result, len(samples))

    LOGGER.info(
        "M5 temperature samples=%d selected=%.2f log_loss=%.8f",
        len(samples),
        result["temperature"],
        selected["log_loss"],
    )

    return result


def validate_temperature_fit(
    result: Mapping[str, object],
    expected_samples: int,
) -> None:
    """저장된 보정 분모, 고정 탐색 범위와 선정 결과를 확인한다."""
    m5.require(
        result["samples"] == expected_samples
        and expected_samples > 0,
        "온도 보정 표본 수 오류",
    )

    raw_losses = result["losses"]

    m5.require(
        isinstance(raw_losses, list)
        and len(raw_losses) == len(TEMPERATURES),
        "온도 보정 탐색 결과 누락",
    )

    records: list[dict[str, float]] = []

    for raw, temperature in zip(
        raw_losses, TEMPERATURES
    ):
        item = as_row(raw)
        loss = item["log_loss"]

        m5.require(
            item["temperature"] == temperature
            and type(loss) in (int, float)
            and isfinite(loss)
            and loss >= 0.0,
            "온도 후보 또는 NLL 기록 오류",
        )

        records.append({
            "temperature": temperature,
            "log_loss": float(loss),
        })

    selected = min(records, key=temperature_key)

    m5.require(
        result["temperature"] == selected["temperature"],
        "저장된 온도와 보정 구간 선정 결과 불일치",
    )


def metric_value(
    candidate: Mapping[str, object],
    axis: str,
    metric: str,
) -> float:
    """기존 평가 결과에서 유한한 지표를 추출한다."""
    metrics = as_row(candidate[axis])

    if metric == "macro_classwise_ece":
        classes = as_row(metrics["classes"])

        m5.require(
            bool(classes),
            "클래스별 보정 자료가 없습니다.",
        )

        values = [
            as_row(value)["one_vs_rest_ece"]
            for value in classes.values()
        ]

        m5.require(
            all(
                type(value) in (int, float)
                and isfinite(value)
                for value in values
            ),
            "클래스별 ECE 오류",
        )

        return fsum(values) / len(values)

    value = metrics[metric]

    m5.require(
        type(value) in (int, float)
        and isfinite(value),
        f"평가 지표 오류: {axis}/{metric}",
    )

    return float(value)


def comparison_gate(
    candidate: Mapping[str, object],
    reference: Mapping[str, object],
    metrics: Sequence[str],
) -> bool:
    """요구 지표가 모두 악화되지 않고 하나 이상 개선되는지 검사한다."""
    differences = [
        metric_value(candidate, "joint", metric)
        - metric_value(reference, "joint", metric)
        for metric in metrics
    ]

    return (
        all(
            value <= SELECTION_TOLERANCE
            for value in differences
        )
        and any(
            value < -SELECTION_TOLERANCE
            for value in differences
        )
    )


def select_candidates(
    results: Mapping[str, dict[str, object]],
) -> dict[str, object]:
    """선정 구간의 사전 고정 기준을 통과한 운영 후보 하나를 결정한다."""
    m5.require(
        set(results) == set(CANDIDATES),
        "M5 개발 후보 평가 누락",
    )

    denominators = {
        as_row(candidate["joint"])["samples"]
        for candidate in results.values()
    }

    m5.require(
        len(denominators) == 1
        and next(iter(denominators)) > 0,
        "후보별 동일 선정 표본 분모 불일치",
    )

    def rank(name: str) -> tuple[float, float, float, float, int]:
        candidate = results[name]

        return (
            metric_value(candidate, "joint", "log_loss"),
            metric_value(candidate, "joint", "brier"),
            metric_value(candidate, "joint", "top_label_ece"),
            metric_value(
                candidate, "joint", "macro_classwise_ece"
            ),
            CANDIDATES.index(name),
        )

    baseline = min(FREQUENCIES, key=rank)

    eligible = [baseline]
    decisions: dict[str, dict[str, object]] = {}

    for name in CANDIDATES:
        if name in FREQUENCIES:
            passed = name == baseline

            decisions[name] = {
                "eligible": passed,
                "reason": (
                    "best_frequency"
                    if passed
                    else "weaker_frequency"
                ),
            }

            continue

        baseline_pass = comparison_gate(
            results[name],
            results[baseline],
            PRIMARY_METRICS,
        )

        parent = CALIBRATED_PARENTS.get(name)
        calibration_pass: bool | None = None

        if parent is not None:
            calibration_pass = comparison_gate(
                results[name],
                results[parent],
                CALIBRATION_METRICS,
            )

        passed = (
            baseline_pass
            and calibration_pass is not False
        )

        decisions[name] = {
            "eligible": passed,
            "baseline": baseline,
            "baseline_gate": baseline_pass,
            "parent": parent,
            "calibration_gate": calibration_pass,
            "reason": (
                "eligible"
                if passed
                else "selection_gate_not_met"
            ),
        }

        if passed:
            eligible.append(name)

    selected = min(eligible, key=rank)

    return {
        "selected_candidate": selected,
        "best_frequency": baseline,
        "eligible_candidates": eligible,
        "decisions": decisions,
    }


def brief_metrics(
    results: Mapping[str, dict[str, object]],
) -> dict[str, object]:
    """세 확률 좌표계의 주요 지표 및 관측 지원률을 요약한다."""
    return {
        name: {
            axis: {
                "samples": as_row(candidate[axis])["samples"],
                "log_loss": metric_value(
                    candidate, axis, "log_loss"
                ),
                "brier": metric_value(
                    candidate, axis, "brier"
                ),
                "top_label_ece": metric_value(
                    candidate, axis, "top_label_ece"
                ),
                "macro_classwise_ece": metric_value(
                    candidate, axis, "macro_classwise_ece"
                ),
                "train_observed_support_rate": as_row(
                    candidate[axis]
                )["train_observed_support_rate"],
            }
            for axis in evaluation.AXES
        }
        for name, candidate in results.items()
    }


def support_summary(
    candidate: Mapping[str, object],
) -> dict[str, object]:
    """실제 미출현·희소 조합의 지원 수와 지표를 공개한다."""
    result: dict[str, object] = {}

    for axis in evaluation.AXES:
        metrics = as_row(candidate[axis])
        classes = as_row(metrics["classes"])

        unsupported_classes: dict[str, object] = {}
        rare_classes: dict[str, object] = {}

        for label, value in classes.items():
            item = as_row(value)

            train_support = item["train_support"]
            validation_support = item["validation_support"]

            if validation_support == 0:
                continue

            summary = {
                "train_support": train_support,
                "validation_support": validation_support,
                "actual_class_log_loss": item[
                    "actual_class_log_loss"
                ],
                "one_vs_rest_brier": item[
                    "one_vs_rest_brier"
                ],
                "recall": item["recall"],
            }

            if train_support == 0:
                unsupported_classes[label] = summary

            elif train_support < evaluation.RARE_SUPPORT:
                rare_classes[label] = summary

        result[axis] = {
            "samples": metrics["samples"],
            "train_support_groups": metrics[
                "train_support_groups"
            ],
            "train_observed_support_rate": metrics[
                "train_observed_support_rate"
            ],
            "unsupported_output_samples": metrics[
                "unsupported_output_samples"
            ],
            "train_unseen_classes_with_validation_support": (
                unsupported_classes
            ),
            "rare_classes_with_validation_support": rare_classes,
        }

    return result


def validate_bins(
    records: object,
    expected_count: int,
    expected_positive: int | None = None,
) -> None:
    """보정 구간의 범위, 분모와 관측 사건 수를 검증한다."""
    m5.require(
        isinstance(records, list)
        and len(records) == evaluation.CALIBRATION_BINS,
        "보정 구간 수 오류",
    )

    total = 0
    positives = 0

    for index, raw in enumerate(records):
        item = as_row(raw)

        count = m5.integer(item["count"], "bin_count")

        positive = m5.integer(
            item["positive"],
            "bin_positive",
            0,
            count,
        )

        m5.require(
            item["lower"] == index / evaluation.CALIBRATION_BINS
            and item["upper"]
            == (index + 1) / evaluation.CALIBRATION_BINS,
            "보정 구간 범위 불일치",
        )

        mean = item["mean_probability"]
        rate = item["observed_rate"]

        if count == 0:
            m5.require(
                mean is None and rate is None,
                "빈 보정 구간의 통계 오류",
            )
        else:
            m5.require(
                type(mean) in (int, float)
                and isfinite(mean)
                and 0.0 <= mean <= 1.0
                and rate == positive / count,
                "보정 구간 평균 확률 또는 실제 비율 오류",
            )

        total += count
        positives += positive

    m5.require(
        total == expected_count,
        "보정 구간 평가 분모 불일치",
    )

    if expected_positive is not None:
        m5.require(
            positives == expected_positive,
            "보정 구간 실제 사건 수 불일치",
        )


def validate_evaluations(
    results: Mapping[str, dict[str, object]],
    train_counts: Sequence[int],
    expected_samples: int,
) -> None:
    """각 후보의 고정 범주·지원·보정 분모와 평가 계약을 검증한다."""
    m5.require(
        set(results) == set(CANDIDATES)
        and expected_samples > 0,
        "개발 평가 후보 또는 표본 수 오류",
    )

    train_axes = evaluation.axis_counts(train_counts)
    reference_support: dict[str, dict[str, int]] = {}

    for name in CANDIDATES:
        candidate = results[name]

        m5.require(
            set(candidate) == set(evaluation.AXES),
            "공동·주변 평가 결과 누락",
        )

        for axis in evaluation.AXES:
            metrics = as_row(candidate[axis])
            labels = AXIS_LABELS[axis]
            classes = as_row(metrics["classes"])

            m5.require(
                metrics["samples"] == expected_samples
                and set(classes) == set(labels)
                and metrics["unsupported_output_samples"] == 0
                and metrics["output_dictionary_coverage"] == 1.0,
                "평가 분모 또는 고정 범주 사전 불일치",
            )

            for metric in (
                "log_loss",
                "brier",
                "top_label_ece",
                "macro_classwise_ece",
            ):
                value = metric_value(
                    candidate, axis, metric
                )

                m5.require(
                    value >= 0.0,
                    "음수 평가 지표",
                )

                if metric == "brier":
                    m5.require(
                        value <= 2.0 + m5.TOLERANCE,
                        "다중분류 Brier 범위 오류",
                    )

                elif metric.endswith("ece"):
                    m5.require(
                        value <= 1.0 + m5.TOLERANCE,
                        "ECE 범위 오류",
                    )

            support: dict[str, int] = {}

            group_counts = {
                group: 0
                for group in evaluation.SUPPORT_GROUPS
            }

            correct = 0
            predicted_total = 0

            for index, label in enumerate(labels):
                item = as_row(classes[label])

                train_support = train_axes[axis][index]

                actual = m5.integer(
                    item["validation_support"],
                    "validation_support",
                    0,
                    expected_samples,
                )

                predicted = m5.integer(
                    item["predicted_count"],
                    "predicted_count",
                    0,
                    expected_samples,
                )

                tp = m5.integer(
                    item["true_positive"],
                    "true_positive",
                    0,
                    min(actual, predicted),
                )

                m5.require(
                    item["train_support"] == train_support,
                    "학습 클래스 지원 수 불일치",
                )

                m5.require(
                    item["precision"] == (
                        tp / predicted if predicted else None
                    )
                    and item["recall"] == (
                        tp / actual if actual else None
                    ),
                    "클래스별 정밀도 또는 재현율 불일치",
                )

                validate_bins(
                    item["calibration"],
                    expected_samples,
                    actual,
                )

                support[label] = actual
                predicted_total += predicted
                correct += tp

                if train_support == 0:
                    group = "unseen"
                elif train_support < evaluation.RARE_SUPPORT:
                    group = "rare"
                else:
                    group = "observed"

                group_counts[group] += actual

            m5.require(
                sum(support.values()) == expected_samples
                and predicted_total == expected_samples
                and metrics["accuracy"]
                == correct / expected_samples,
                "클래스 실제·예측 분모 또는 Accuracy 불일치",
            )

            validate_bins(
                metrics["top_label_calibration"],
                expected_samples,
                correct,
            )

            if axis not in reference_support:
                reference_support[axis] = support
            else:
                m5.require(
                    reference_support[axis] == support,
                    "후보 간 실제 클래스 분포 불일치",
                )

            raw_groups = as_row(
                metrics["train_support_groups"]
            )

            m5.require(
                set(raw_groups)
                == set(evaluation.SUPPORT_GROUPS),
                "학습 지원 그룹 누락",
            )

            for group, count in group_counts.items():
                item = as_row(raw_groups[group])

                m5.require(
                    item["count"] == count
                    and item["fraction"]
                    == count / expected_samples,
                    "희소·미출현 지원 그룹 분모 불일치",
                )

                if count == 0:
                    m5.require(
                        item["log_loss"] is None
                        and item["brier"] is None,
                        "빈 지원 그룹의 지표 오류",
                    )

            observed_rate = (
                1.0
                - group_counts["unseen"] / expected_samples
            )

            m5.require(
                metrics["train_observed_support_rate"]
                == observed_rate,
                "학습 관측 지원률 불일치",
            )

        # 공동 정답과 주변 정답의 실제 평가 빈도도 일치해야 한다.
        joint_classes = as_row(
            as_row(candidate["joint"])["classes"]
        )

        joint_support = [
            as_row(joint_classes[label])["validation_support"]
            for label in m5.CLASSES
        ]

        marginal_support = evaluation.axis_counts(
            joint_support
        )

        for axis in ("pitch_type", "pitch_result"):
            classes = as_row(
                as_row(candidate[axis])["classes"]
            )

            actual = [
                as_row(classes[label])["validation_support"]
                for label in AXIS_LABELS[axis]
            ]

            m5.require(
                actual == marginal_support[axis],
                "공동·주변 정답 지원 수 불일치",
            )