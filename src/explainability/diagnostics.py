"""고정 예측의 확률 신뢰도·실패 및 시즌 불확실성 분석."""

from bisect import insort
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import closing
from hashlib import sha256
from math import fsum, isfinite, log
from pathlib import Path

from . import (
    FAILURE_CASE_LIMIT,
    GLOBAL_SAMPLE_LIMIT,
    M2_RARE_SUPPORT,
    M5_RARE_SUPPORT,
    SEED,
    SPARSE_BIN_SUPPORT,
)
from .contributions import compare
from ..evaluation import final_scoring as scoring
from ..evaluation import final_evaluation as final
from ..evaluation import game_outcome as e1
from ..evaluation import plate_appearance as e2
from ..evaluation import pitch_outcome as e5
from ..evaluation.fixed_assets import (
    read_unique_json,
    require,
)
from ..models.game_outcome import model as m1
from ..models.plate_appearance import model as m2
from ..models.plate_appearance import probability as p2
from ..models.game_state_value import model as m4
from ..models.pitch_outcome import model as m5
from ..models.player_performance import data as d3
from ..validation import season_forecast as season
from ..validation import season_forecast_checks as checks


JsonObject = dict[str, object]


def ratio(
    numerator: float,
    denominator: int,
) -> float | None:
    """분모가 없는 경우 실제 0과 구분한다."""
    return numerator / denominator if denominator else None


class M2StreamingScore:
    """기존 M2 평가 계약을 순차 누적 방식으로 계산한다."""

    def __init__(self) -> None:
        size = m2.N_CLASSES

        self.samples = 0
        self.loss = 0.0
        self.brier = 0.0

        self.confusion = [
            [0] * size
            for _ in range(size)
        ]

        self.bins = [
            [
                [0, 0.0, 0]
                for _ in range(e2.CALIBRATION_BINS)
            ]
            for _ in range(size)
        ]

        self.forbidden_mass = [0.0] * size
        self.forbidden_nonzero_samples = 0
        self.forbidden_target_count = 0

    def add(
        self,
        target: int,
        probabilities: Sequence[float],
        allowed: Sequence[bool],
    ) -> None:
        """한 타석의 확률·분류·규칙 위반 통계를 누적한다."""
        m2.check_probabilities(probabilities)

        require(
            len(allowed) == m2.N_CLASSES
            and 0 <= target < m2.N_CLASSES,
            "M2 진단 입력 차원 오류",
        )

        predicted = max(
            range(m2.N_CLASSES),
            key=probabilities.__getitem__,
        )

        self.samples += 1
        self.confusion[target][predicted] += 1

        self.loss -= log(
            max(
                probabilities[target],
                e2.LOG_EPSILON,
            )
        )

        self.forbidden_target_count += int(
            not allowed[target]
        )

        forbidden_nonzero = False

        for index, probability in enumerate(probabilities):
            observed = int(index == target)

            self.brier += (probability - observed) ** 2

            bucket = min(
                int(probability * e2.CALIBRATION_BINS),
                e2.CALIBRATION_BINS - 1,
            )

            cell = self.bins[index][bucket]

            cell[0] += 1
            cell[1] += probability
            cell[2] += observed

            if not allowed[index]:
                self.forbidden_mass[index] += probability

                if probability > 0:
                    forbidden_nonzero = True

        self.forbidden_nonzero_samples += int(
            forbidden_nonzero
        )

    def report(self) -> JsonObject:
        """기존 평가와 동일한 구조 및 정규화로 결과를 반환한다."""
        require(
            self.samples > 0,
            "M2 진단 표본 없음",
        )

        n = self.samples

        classes = {}
        matrix = {}
        calibration = {}
        classwise_ece = {}
        mistakes = []

        for index, label in enumerate(m2.CLASSES):
            tp = self.confusion[index][index]
            support = sum(self.confusion[index])

            predicted_count = sum(
                row[index]
                for row in self.confusion
            )

            precision = e2.safe_ratio(tp, predicted_count)
            recall = e2.safe_ratio(tp, support)

            classes[label] = {
                "support": support,
                "predicted_count": predicted_count,
                "precision": precision,
                "recall": recall,
                "f1": e2.safe_ratio(
                    2.0 * precision * recall,
                    precision + recall,
                ),
            }

            matrix[label] = dict(
                zip(
                    m2.CLASSES,
                    self.confusion[index],
                    strict=True,
                )
            )

            for other, other_label in enumerate(m2.CLASSES):
                count = self.confusion[index][other]

                if index != other and count:
                    mistakes.append({
                        "actual": label,
                        "predicted": other_label,
                        "count": count,
                    })

            intervals = []
            calibration_error = 0.0

            for bucket, cell in enumerate(self.bins[index]):
                count, predicted_sum, positive = cell

                mean = ratio(predicted_sum, count)
                observed = ratio(positive, count)

                if count:
                    calibration_error += (
                        count * abs(mean - observed)
                    )

                intervals.append({
                    "lower": bucket / e2.CALIBRATION_BINS,
                    "upper": (
                        (bucket + 1) / e2.CALIBRATION_BINS
                    ),
                    "upper_inclusive": (
                        bucket == e2.CALIBRATION_BINS - 1
                    ),
                    "count": count,
                    "mean_probability": mean,
                    "observed_rate": observed,
                })

            require(
                sum(item["count"] for item in intervals) == n,
                "M2 클래스별 신뢰도 분모 오류",
            )

            calibration[label] = intervals
            classwise_ece[label] = calibration_error / n

        mistakes.sort(
            key=lambda item: -item["count"]
        )

        return {
            "samples": n,
            "log_loss": self.loss / n,
            "brier": self.brier / n,
            "accuracy": (
                sum(
                    self.confusion[index][index]
                    for index in range(m2.N_CLASSES)
                ) / n
            ),
            "macro_f1": (
                sum(
                    item["f1"]
                    for item in classes.values()
                ) / m2.N_CLASSES
            ),
            "per_class": classes,
            "confusion_matrix": matrix,
            "major_confusions": (
                mistakes[:e2.TOP_CONFUSIONS]
            ),
            "calibration": calibration,
            "classwise_ece": classwise_ece,
            "macro_classwise_ece": (
                fsum(classwise_ece.values()) / m2.N_CLASSES
            ),
            "forbidden_probability_mean": (
                fsum(self.forbidden_mass) / n
            ),
            "forbidden_nonzero_samples": (
                self.forbidden_nonzero_samples
            ),
            "forbidden_class_probability_mean": {
                name: self.forbidden_mass[index] / n
                for index, name in enumerate(m2.CLASSES)
            },
            "forbidden_target_count": (
                self.forbidden_target_count
            ),
        }


def score_m2_streaming(
    predictions: Path,
    targets: Path,
    baseline: Mapping[str, object],
) -> JsonObject:
    """대용량 타석 예측을 전량 적재하지 않고 점수화한다."""
    metrics = {
        method: M2StreamingScore()
        for method in scoring.METHODS
    }

    counts: Counter[str] = Counter()

    for row, truth in scoring.paired_rows(
        predictions,
        targets,
    ):
        label = m2.target_class(truth["target"])
        target = m2.CLASS_INDEX[label]

        counts[label] += 1

        allowed = p2.allowed_mask(row["input"])

        for method, metric in metrics.items():
            metric.add(
                target,
                row[method],
                allowed,
            )

    results = {
        name: metric.report()
        for name, metric in metrics.items()
    }

    for result in results.values():
        require(
            result["forbidden_nonzero_samples"] == 0,
            "M2 금지 확률 위반",
        )

    training = dict(
        zip(
            m2.CLASSES,
            baseline["train_counts"],
            strict=True,
        )
    )

    n = metrics["operating"].samples

    require(
        n == metrics["baseline"].samples,
        "M2 공통 평가 분모 불일치",
    )

    return {
        "population": scoring.population(n, n, n, n),
        "results": results,
        "baseline_name": baseline["name"],
        "class_support": {
            label: {
                "train": training[label],
                "test": counts[label],
            }
            for label in m2.CLASSES
        },
        "unseen_train_class_samples": sum(
            counts[label]
            for label in m2.CLASSES
            if training[label] == 0
        ),
        "rule_target_conflicts": (
            results["operating"]["forbidden_target_count"]
        ),
        "contract_valid": (
            results["operating"]["forbidden_target_count"] == 0
            and results["baseline"]["forbidden_target_count"] == 0
        ),
        "comparison": scoring.compare_metrics(
            results["operating"],
            results["baseline"],
            (
                "log_loss",
                "brier",
                "macro_f1",
                "macro_classwise_ece",
            ),
            higher=("macro_f1",),
        ),
    }


def calibration_support(
    value: object,
) -> object:
    """원래 지표를 바꾸지 않고 빈 구간과 희소 구간을 표시한다."""
    if isinstance(value, list):
        return [
            calibration_support(item)
            for item in value
        ]

    if not isinstance(value, dict):
        return value

    result = {
        key: calibration_support(item)
        for key, item in value.items()
    }

    is_interval = (
        "mean_probability" in value
        and (
            "lower" in value
            or "low" in value
        )
    )

    if is_interval:
        count = value.get(
            "count",
            value.get("games"),
        )

        require(
            type(count) is int and count >= 0,
            "신뢰도 구간 분모 오류",
        )

        result["support_status"] = (
            "empty"
            if count == 0
            else "sparse"
            if count < SPARSE_BIN_SUPPORT
            else "observed"
        )

        result["upper_inclusive"] = (
            value.get(
                "upper",
                value.get("high"),
            ) == 1.0
        )

        if count == 0:
            require(
                value["mean_probability"] is None,
                "빈 신뢰도 구간에 평균 존재",
            )

    return result


def m1_ece(
    metrics: Mapping[str, object],
) -> JsonObject:
    """기존 M1 비균등 구간에서 추가 ECE를 계산한다."""
    report = {}

    for method, result in metrics["results"].items():
        outcome = result["outcomes"]
        n = outcome["games"]

        per_class = {}

        for label, item in outcome["calibration"].items():
            bins = item["bins"]

            require(
                sum(cell["games"] for cell in bins) == n,
                "M1 신뢰도 분모 불일치",
            )

            error = fsum(
                cell["games"]
                * abs(
                    cell["mean_probability"]
                    - cell["observed_frequency"]
                )
                for cell in bins
                if cell["games"]
            )

            per_class[label] = error / n

        report[method] = {
            "samples": n,
            "bin_edges": e1.BINS,
            "classwise_ece": per_class,
            "macro_classwise_ece": (
                fsum(per_class.values()) / len(per_class)
            ),
        }

    return {
        "definition": (
            "weighted_bin_gap_then_equal_class_average"
        ),
        "original_final_evaluation_metric": False,
        "results": report,
    }


class SampleSelection:
    """고정 해시 순서로 제한된 전체 설명 표본을 선정한다."""

    def __init__(
        self,
        limit: int = GLOBAL_SAMPLE_LIMIT,
    ) -> None:
        require(limit > 0, "설명 표본 제한 오류")

        self.limit = limit
        self.rows: list[
            tuple[str, str, JsonObject]
        ] = []

    def add(self, sample: JsonObject) -> None:
        """자료 내용이나 예측 오차와 무관하게 표본을 선정한다."""
        key = sample["prediction"]["sample_id"]

        rank = sha256(
            f"{SEED}:{key}".encode("utf-8")
        ).hexdigest()

        item = (rank, key, sample)

        if (
            len(self.rows) >= self.limit
            and item[:2] >= self.rows[-1][:2]
        ):
            return

        insort(self.rows, item)

        if len(self.rows) > self.limit:
            self.rows.pop()

    def selected(self) -> dict[str, JsonObject]:
        """선정 표본을 ID에 따라 반환한다."""
        return {
            key: sample
            for _, key, sample in self.rows
        }


class FailureCases:
    """실패 사유별 전체 건수와 제한된 대표 사례를 관리한다."""

    def __init__(
        self,
        limit: int = FAILURE_CASE_LIMIT,
    ) -> None:
        require(limit > 0, "대표 사례 제한 오류")

        self.limit = limit

        self.eligible: Counter[str] = Counter()
        self.counts: Counter[str] = Counter()

        self.rows: dict[
            str,
            list[tuple[float, str, JsonObject]]
        ] = {}

    def observe(
        self,
        reason: str,
        sample: JsonObject,
        *,
        eligible: bool = True,
        score: float | None = None,
    ) -> None:
        """지원 분모와 발생 건수를 분리하고 상위 사례를 선정한다."""
        if not eligible:
            require(
                score is None,
                "분모에 없는 실패 표본",
            )
            return

        self.eligible[reason] += 1

        if score is None:
            return

        require(
            isfinite(score) and score >= 0,
            "실패 사례 순위 점수 오류",
        )

        self.counts[reason] += 1

        key = sample["prediction"]["sample_id"]

        selected = self.rows.setdefault(reason, [])
        item = (-score, key, sample)

        if (
            len(selected) >= self.limit
            and item[:2] >= selected[-1][:2]
        ):
            return

        insort(selected, item)

        if len(selected) > self.limit:
            selected.pop()

    def report(self) -> JsonObject:
        """사유별 분모·건수·대표 사례를 반환한다."""
        return {
            reason: {
                "eligible_samples": n,
                "count": self.counts[reason],
                "fraction": ratio(
                    self.counts[reason],
                    n,
                ),
                "cases": [
                    {
                        "sample_id": key,
                        "selection_score": -score,
                    }
                    for score, key, _ in self.rows.get(
                        reason, []
                    )
                ],
            }
            for reason, n in sorted(self.eligible.items())
        }

    def selected(
        self,
    ) -> tuple[
        dict[str, JsonObject],
        dict[str, list[str]],
    ]:
        """사유가 중복되어도 같은 표본은 한 번만 설명한다."""
        samples = {}
        reasons: dict[str, list[str]] = {}

        for reason, rows in sorted(self.rows.items()):
            for _, key, sample in rows:
                samples[key] = sample
                reasons.setdefault(key, []).append(reason)

        return samples, reasons


def m1_cases(
    book: FailureCases,
    sample: JsonObject,
) -> None:
    """실제 무승부 및 홈 득점 오차를 구분한다."""
    row = sample["prediction"]
    truth = sample["target"]["target"]

    forecast = row["operating"]
    probabilities = forecast["result_probabilities"]

    actual = truth["result"]

    loss = -log(
        max(
            e1.LOG_FLOOR,
            probabilities[actual],
        )
    )

    predicted = max(
        m1.OUTCOMES,
        key=probabilities.__getitem__,
    )

    home_error = (
        forecast["home_expected_runs"] - truth["home"]
    )

    book.observe(
        "observed_tie",
        sample,
        score=loss if actual == "tie" else None,
    )

    book.observe(
        "result_argmax_error",
        sample,
        score=loss if predicted != actual else None,
    )

    book.observe(
        "home_runs_overprediction",
        sample,
        score=home_error if home_error > 0 else None,
    )

    book.observe(
        "home_runs_underprediction",
        sample,
        score=-home_error if home_error < 0 else None,
    )


def classification_cases(
    book: FailureCases,
    sample: JsonObject,
    axis: str,
    labels: Sequence[str],
    target: int,
    probabilities: Sequence[float],
    train_counts: Sequence[int],
    rare_support: int,
) -> None:
    """미출현·희소 범주와 실제 분류 오류를 분리한다."""
    predicted = max(
        range(len(labels)),
        key=probabilities.__getitem__,
    )

    support = train_counts[target]

    loss = -log(
        max(
            e2.LOG_EPSILON,
            probabilities[target],
        )
    )

    book.observe(
        f"{axis}:argmax_error",
        sample,
        score=loss if predicted != target else None,
    )

    book.observe(
        f"{axis}:unseen_train_class",
        sample,
        score=loss if support == 0 else None,
    )

    book.observe(
        f"{axis}:rare_train_class",
        sample,
        score=(
            loss
            if 0 < support < rare_support
            else None
        ),
    )

    book.observe(
        f"{axis}:class:{labels[target]}:argmax_error",
        sample,
        score=loss if predicted != target else None,
    )


def m2_cases(
    book: FailureCases,
    sample: JsonObject,
    baseline: Mapping[str, object],
) -> None:
    """타석 결과별 분류 실패와 규칙 충돌을 분석한다."""
    row = sample["prediction"]
    truth = sample["target"]

    target = m2.CLASS_INDEX[
        m2.target_class(truth["target"])
    ]

    classification_cases(
        book,
        sample,
        "plate_appearance",
        m2.CLASSES,
        target,
        row["operating"],
        baseline["train_counts"],
        M2_RARE_SUPPORT,
    )

    mask = p2.allowed_mask(row["input"])

    book.observe(
        "plate_appearance:rule_target_conflict",
        sample,
        score=1.0 if not mask[target] else None,
    )


def m4_cases(
    book: FailureCases,
    sample: JsonObject,
) -> None:
    """종료 근거와 미종료 상태의 실제 확률 오차를 분리한다."""
    row = sample["prediction"]
    truth = sample["target"]

    state = m4.parse_state(row["input"])

    actual = truth["win"]

    operating = row["operating"]["win"][0]
    baseline = row["baseline"]["win"][0]

    confirmed = truth["confirmed_final"]

    book.observe(
        "observed_confirmed_final",
        sample,
        score=(
            abs(operating - actual)
            if confirmed
            else None
        ),
    )

    book.observe(
        "nonterminal:win_probability_error",
        sample,
        eligible=not confirmed,
        score=(
            abs(operating - actual)
            if not confirmed
            else None
        ),
    )

    delta = (
        (operating - actual) ** 2
        - (baseline - actual) ** 2
    )

    book.observe(
        "nonterminal:brier_worse_than_baseline",
        sample,
        eligible=not confirmed,
        score=(
            delta
            if not confirmed and delta > 0
            else None
        ),
    )

    book.observe(
        "three_outs_nonterminal_boundary",
        sample,
        eligible=not confirmed,
        score=(
            abs(operating - actual)
            if not confirmed and state.outs == 3
            else None
        ),
    )

    remaining_error = abs(
        row["operating"]["re"][0] - truth["remaining"]
    )

    book.observe(
        "nonterminal:remaining_runs_error",
        sample,
        eligible=not confirmed,
        score=(
            remaining_error
            if not confirmed and remaining_error > 0
            else None
        ),
    )


def m5_cases(
    book: FailureCases,
    sample: JsonObject,
    baseline: Mapping[str, object],
) -> None:
    """공동·구종·결과별 실제 지원과 실패를 분석한다."""
    row = sample["prediction"]

    target = sample["target"]["target"]
    joint = row["operating"]

    pitch, result = m5.marginalize(joint)

    counts = e5.axis_counts(
        baseline["train_counts"]
    )

    for axis, labels, actual, probabilities in (
        (
            "joint",
            m5.CLASSES,
            target,
            joint,
        ),
        (
            "pitch_type",
            m5.TYPES,
            target // m5.N_RESULTS,
            pitch,
        ),
        (
            "pitch_result",
            m5.RESULTS,
            target % m5.N_RESULTS,
            result,
        ),
    ):
        classification_cases(
            book,
            sample,
            axis,
            labels,
            actual,
            probabilities,
            counts[axis],
            M5_RARE_SUPPORT,
        )


def m3_cases(
    book: FailureCases,
    sample: JsonObject,
) -> None:
    """선수의 실제 정답 부족, 오차, 구간 미지원을 구분한다."""
    row = sample["prediction"]
    truth = sample["target"]

    player = d3.parse_input(row["input"])
    role = player.role

    names = d3.metrics(role)

    for horizon in d3.HORIZONS:
        target = scoring.parse_final_target(
            truth["targets"][horizon],
            player,
        )

        actual = dict(zip(names, target.counts, strict=True))

        operating = row["operating"][horizon]
        baseline = row["baseline"][horizon]

        book.observe(
            f"{horizon}:target_unusable",
            sample,
            score=1.0 if not target.usable else None,
        )

        book.observe(
            f"{horizon}:observed_zero",
            sample,
            score=(
                1.0
                if target.status == "zero_observed"
                else None
            ),
        )

        for field in names:
            key = f"{horizon}.{field}"

            observed = actual[field]

            book.observe(
                f"{key}:target_unavailable",
                sample,
                score=1.0 if observed is None else None,
            )

            if observed is None:
                continue

            prediction = operating["counts"][field]

            error = abs(prediction - observed)

            baseline_error = abs(
                baseline["counts"][field] - observed
            )

            worse = error - baseline_error

            book.observe(
                f"{key}:absolute_error",
                sample,
                score=error if error > 0 else None,
            )

            book.observe(
                f"{key}:worse_than_baseline",
                sample,
                score=worse if worse > 0 else None,
            )

            interval = operating["intervals"][field]

            lower = interval["lower"]
            upper = interval["upper"]

            supported = lower is not None

            book.observe(
                f"{key}:interval_unavailable",
                sample,
                score=1.0 if not supported else None,
            )

            if supported:
                missed = max(
                    lower - observed,
                    observed - upper,
                    0.0,
                )

                book.observe(
                    f"{key}:interval_miss",
                    sample,
                    score=missed if missed > 0 else None,
                )

        for rate, numerator, denominator in (
            ("hit_rate", "hits", "ab"),
            (
                "strikeout_rate",
                "strikeouts",
                d3.exposure(role),
            ),
        ):
            key = f"{horizon}.{rate}"

            actual_n = actual[numerator]
            actual_d = actual[denominator]

            book.observe(
                f"{key}:zero_truth_denominator",
                sample,
                score=1.0 if actual_d == 0 else None,
            )

            valid = (
                actual_n is not None
                and actual_d is not None
                and actual_d > 0
            )

            book.observe(
                f"{key}:zero_operating_denominator",
                sample,
                eligible=valid,
                score=(
                    1.0
                    if valid
                    and operating["counts"][denominator] == 0
                    else None
                ),
            )


def sample_rows(
    name: str,
    predictions: Path,
    targets: Path,
):
    """M4만 실제 타석 전후 쌍을 연결하고 나머지는 순차 반환한다."""
    pending = None

    for row, truth in scoring.paired_rows(
        predictions,
        targets,
    ):
        final.validate_day(
            row["sample_date"],
            "test",
        )

        sample = {
            "prediction": row,
            "target": truth,
        }

        if name != "m4":
            yield sample
            continue

        phase = row["input"]["phase"]

        if phase == "post":
            require(
                pending is None,
                "M4 이전 타석 쌍 미완료",
            )

            pending = sample
            continue

        require(
            phase == "pre"
            and pending is not None,
            "M4 타석 전후 상태 연결 오류",
        )

        post = pending["prediction"]

        require(
            row["sample_id"].rsplit(":", 1)[0]
            == post["sample_id"].rsplit(":", 1)[0],
            "M4 타석 쌍 식별자 불일치",
        )

        pair = {
            "pre": row,
            "post": post,
            "post_truth": pending["target"],
        }

        pending["pair"] = pair
        sample["pair"] = pair

        yield pending
        yield sample

        pending = None

    require(
        pending is None,
        "M4 마지막 타석 쌍 누락",
    )


def diagnose_model(
    name: str,
    predictions: Path,
    targets: Path,
    stored_metrics: Mapping[str, object],
    baseline: Mapping[str, object],
    expected_samples: int,
) -> tuple[
    JsonObject,
    dict[str, JsonObject],
    set[str],
    dict[str, list[str]],
]:
    """전체 모집단 진단과 설명·실패 표본을 함께 구성한다."""
    recalculated = (
        score_m2_streaming(
            predictions,
            targets,
            baseline,
        )
        if name == "m2"
        else scoring.score(
            name,
            predictions,
            targets,
            baseline,
        )
    )

    metric_error = compare(
        recalculated,
        stored_metrics,
        f"{name}: 기존 최종 평가 지표",
    )

    selection = SampleSelection()
    failures = FailureCases()

    n = 0

    for sample in sample_rows(
        name,
        predictions,
        targets,
    ):
        selection.add(sample)

        if name == "m1":
            m1_cases(failures, sample)

        elif name == "m2":
            m2_cases(failures, sample, baseline)

        elif name == "m4":
            m4_cases(failures, sample)

        elif name == "m5":
            m5_cases(failures, sample, baseline)

        else:
            require(
                name in {"m3_hitter", "m3_pitcher"},
                "미지원 실패 사례 모델",
            )

            m3_cases(failures, sample)

        n += 1

    require(
        n > 0 and n == expected_samples,
        f"{name}: 전체 진단 모집단 불일치",
    )

    global_samples = selection.selected()
    failure_samples, reasons = failures.selected()

    all_samples = {
        **global_samples,
        **failure_samples,
    }

    report = {
        "model": name,
        "population": n,
        "global_explanation_samples": len(global_samples),
        "failure_explanation_samples": len(failure_samples),
        "unique_explanation_samples": len(all_samples),
        "metrics_recalculation": {
            "status": "matched",
            "max_absolute_error": metric_error,
        },
        "metrics": calibration_support(recalculated),
        "failure_cases": failures.report(),
        "failure_case_limit": FAILURE_CASE_LIMIT,
        "sparse_calibration_bin_threshold": (
            SPARSE_BIN_SUPPORT
        ),
        "failure_counts_scope": "full_fixed_test_population",
        "global_scope": "deterministic_hash_sample",
    }

    if name == "m1":
        report["additional_ece"] = m1_ece(recalculated)

    return (
        report,
        all_samples,
        set(global_samples),
        reasons,
    )


def season_result(
    run: checks.Run,
) -> JsonObject:
    """실제 저장 시뮬레이션의 설명용 참조를 구성한다."""
    return {
        "run_id": run.manifest["run_id"],
        "directory": str(run.directory),
        "season": run.manifest["season"],
        "as_of_kst": run.manifest["as_of_kst"],
        "seed": run.manifest["seed"],
        "iterations": run.manifest["iterations"],
        "artifact_sha256": run.digests,
        "scenario_metadata": run.scenario["metadata"],
        "summary": run.summary,
    }


def check_prior_result(
    verification: Mapping[str, object],
    name: str,
    actual: object,
) -> None:
    """기존 승인 검증의 동일 계산 결과와 현재 결과를 대조한다."""
    record = verification["checks"][name]

    require(
        record["status"] == "passed",
        f"기존 시즌 검증 미통과: {name}",
    )

    compare(
        actual,
        record["result"],
        f"시즌 검증 재계산: {name}",
    )


def starter_failures(
    samples: Sequence[JsonObject],
) -> JsonObject:
    """선발 후보의 실제 미지원과 오분류 사례를 집계한다."""
    seen = set()

    counts: Counter[str] = Counter()
    examples: dict[str, list[JsonObject]] = {}

    def record(
        reason: str,
        sample: JsonObject,
    ) -> None:
        counts[reason] += 1

        selected = examples.setdefault(reason, [])

        if len(selected) < FAILURE_CASE_LIMIT:
            selected.append(sample)

    for sample in sorted(
        samples,
        key=lambda item: (
            item["date"],
            item["game_id"],
            item["team_id"],
        ),
    ):
        key = (
            sample["game_id"],
            sample["team_id"],
        )

        require(
            key not in seen,
            "선발 경기×팀 표본 중복",
        )

        seen.add(key)

        truth = sample["actual_player_id"]

        if truth is None:
            record(
                sample["truth_reason"] or "truth_unavailable",
                sample,
            )

        for method in ("baseline", "proposed"):
            candidates = sample[method]

            if not candidates:
                record(
                    f"{method}:candidate_unavailable",
                    sample,
                )

            elif truth is not None:
                if candidates[0] != truth:
                    record(
                        f"{method}:top1_miss",
                        sample,
                    )

                if truth not in candidates:
                    record(
                        f"{method}:topk_miss",
                        sample,
                    )

    return {
        "population": len(samples),
        "counts": dict(counts),
        "examples": examples,
        "selection": (
            "date_game_id_team_id_ascending"
        ),
    }


def support_reference(
    payload: Mapping[str, object],
    path: Path,
) -> JsonObject:
    """보조 분석의 실제 경기·팀 자료 참조를 제공한다."""
    games = payload["games"]

    index = {}

    for game in games:
        fixture = game["fixture"]
        game_id = fixture["game_id"]

        require(
            game_id not in index,
            "보조 분석 경기 식별자 중복",
        )

        index[game_id] = {
            "date": fixture["date"],
            "home_team_id": fixture["home_team_id"],
            "away_team_id": fixture["away_team_id"],
        }

    return {
        "path": str(path),
        "run_id": payload["run_id"],
        "scope": payload["scope"],
        "as_of_kst": payload["as_of_kst"],
        "metadata": payload["metadata"],
        "teams": payload["teams"],
        "game_count": len(index),
        "game_team_index": index,
    }


def diagnose_season(
    root: Path,
    plan: Mapping[str, object],
    verification: Mapping[str, object],
) -> JsonObject:
    """승인된 시즌·보조 분석 자료를 독립 재집계한다."""
    require(
        verification["status"] == "completed"
        and not verification["failures"],
        "선행 시즌 검증 미완료",
    )

    paths = plan["paths"]
    jobs = verification["jobs"]

    regular = {
        "reference": checks.load_run(
            root,
            Path(paths["reference_regular"]),
            "regular_season",
        ),
    }

    postseason = {
        "reference": checks.load_run(
            root,
            Path(paths["reference_postseason"]),
            "postseason",
        ),
        "ended": checks.load_run(
            root,
            Path(paths["reference_ended"]),
            "postseason",
        ),
    }

    for label, seed, iterations in season.SIMULATION_CASES:
        regular[label] = checks.load_run(
            root,
            Path(
                jobs[f"regular_{label}"]["result"]["result_dir"]
            ),
            "regular_season",
        )

        postseason[label] = checks.load_run(
            root,
            Path(
                jobs[f"post_{label}"]["result"]["result_dir"]
            ),
            "postseason",
        )

        for run in (regular[label], postseason[label]):
            compare(run.manifest["seed"], seed, "시즌 Seed")
            compare(
                run.manifest["iterations"],
                iterations,
                "시즌 반복 수",
            )

    postseason["ended_repeat"] = checks.load_run(
        root,
        Path(
            jobs["post_ended"]["result"]["result_dir"]
        ),
        "postseason",
    )

    support_paths = {
        name: Path(
            jobs[f"support_{name}"]["result"]["output"]
        )
        for name in (
            "evaluation",
            "season",
            "game",
            "early",
            "late",
        )
    }

    support = {
        name: read_unique_json(path)
        for name, path in support_paths.items()
    }

    audits = {}

    with (
        closing(
            season.open_readonly(
                Path(paths["standard_db"])
            )
        ) as standard,
        closing(
            season.open_readonly(
                Path(paths["model_db"])
            )
        ) as model,
    ):
        games, completed = season.read_regular_inputs(
            standard,
            model,
        )

        for label, run in regular.items():
            name = f"regular_{label}"

            audit = checks.audit_regular(
                run,
                games,
            )

            check_prior_result(
                verification,
                name,
                audit,
            )

            audits[name] = audit

        for label, run in postseason.items():
            name = f"postseason_{label}"

            audit = checks.audit_postseason(run)

            check_prior_result(
                verification,
                name,
                audit,
            )

            audits[name] = audit

        for label in ("reference", "repeat", "long", "seed"):
            name = f"parent_{label}"

            audit = checks.audit_parent(
                regular[label],
                postseason[label],
            )

            check_prior_result(
                verification,
                name,
                audit,
            )

            audits[name] = audit

        for label, output_name in (
            ("ended", "archive_reference"),
            ("ended_repeat", "archive_repeat"),
        ):
            archive = checks.audit_archive(
                postseason[label],
                standard,
                model,
            )

            check_prior_result(
                verification,
                output_name,
                archive,
            )

            audits[output_name] = archive

        for label in ("season", "early", "late"):
            name = f"support_{label}"

            audit = checks.audit_support_season(
                support[label],
                games,
            )

            check_prior_result(
                verification,
                name,
                audit,
            )

            audits[name] = audit

        audits["support_game"] = checks.audit_support_game(
            support["game"],
            support["season"],
        )

        check_prior_result(
            verification,
            "support_game",
            audits["support_game"],
        )

        availability = checks.season_availability(
            standard
        )

        check_prior_result(
            verification,
            "season_availability",
            availability,
        )

        official = checks.official_diagnostic(
            regular["reference"],
            postseason["reference"],
            games,
        )

        check_prior_result(
            verification,
            "official_development_diagnostic",
            official,
        )

        for label, run in regular.items():
            name = f"diagnostic_{label}"

            audit = season.audit_development_diagnostic(
                run,
                games,
                completed,
            )

            check_prior_result(
                verification,
                name,
                audit,
            )

            audits[name] = audit

    reproduction = season.compare_reproduction(
        regular["reference"],
        postseason["reference"],
        regular["repeat"],
        postseason["repeat"],
        regular["long"],
        postseason["long"],
        postseason["ended"],
        postseason["ended_repeat"],
    )

    check_prior_result(
        verification,
        "reproduction",
        reproduction,
    )

    stability = {}

    for label in ("long", "seed"):
        for package, runs in (
            ("regular", regular),
            ("postseason", postseason),
        ):
            name = f"stability_{package}_{label}"

            audit = checks.stability(
                runs["reference"].summary,
                runs[label].summary,
            )

            check_prior_result(
                verification,
                name,
                audit,
            )

            stability[name] = audit

    samples = support["evaluation"]["samples"]

    starter = checks.starter_comparison(samples)

    compare(
        starter,
        support["evaluation"]["comparison"],
        "선발 저장 비교 지표",
    )

    inherited_starter = verification[
        "checks"
    ]["starter_evaluation"]

    require(
        inherited_starter["status"] == "passed",
        "선행 선발 검증 실패",
    )

    compare(
        starter,
        inherited_starter["result"]["comparison"],
        "선발 선행 승인 지표",
    )

    truth_available = starter["baseline"]["truth_available"]

    return {
        "status": "completed",
        "regular_season": {
            label: season_result(run)
            for label, run in regular.items()
        },
        "postseason": {
            label: season_result(run)
            for label, run in postseason.items()
        },
        "audits": audits,
        "reproduction": reproduction,
        "stability": stability,
        "official_development_diagnostic": official,
        "season_availability": availability,
        "starter": {
            "path": str(support_paths["evaluation"]),
            "samples": len(samples),
            "truth_available": truth_available,
            "truth_identification_rate": ratio(
                truth_available,
                len(samples),
            ),
            "comparison": starter,
            "by_team": support["evaluation"]["by_team"],
            "failure_cases": starter_failures(samples),
            "candidate_exclusions": (
                support["evaluation"]["candidate_exclusions"]
            ),
            "candidate_reference_audit": (
                support["evaluation"]["candidate_reference_audit"]
            ),
        },
        "support_analysis": {
            label: support_reference(
                support[label],
                support_paths[label],
            )
            for label in (
                "season",
                "game",
                "early",
                "late",
            )
        },
        "representative_game": support["game"]["games"][0],
        "event_definitions": {
            "regular_season_first": (
                "first_place_under_fixed_regular_season_rules"
            ),
            "korean_series_champion": (
                "winner_of_korean_series"
            ),
            "probability": (
                "positive_events_over_valid_iterations"
            ),
            "unresolved": (
                "not_counted_as_negative_or_probability_zero"
            ),
            "conditional_matchup": (
                "ordered_higher_lower_seed_and_round"
            ),
        },
        "uncertainty": {
            "seed_iteration_stability": (
                "monte_carlo_computation_diagnostic"
            ),
            "game_uncertainty": (
                "conditional_on_fixed_model_schedule_and_rules"
            ),
            "stability_is_confidence_interval": False,
            "historical_information_availability_verified": False,
            "2026_season_forecast_supported": False,
            "2026_support_analysis_supported": False,
        },
        "limitations": verification["limitations"],
    }
