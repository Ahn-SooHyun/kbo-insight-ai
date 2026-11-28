"""운영 모델 구조에 맞는 전체·개별 예측 설명."""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import fsum, isclose, isfinite
from pathlib import Path

from . import ABS_TOLERANCE, REL_TOLERANCE
from ..evaluation import final_evaluation as final
from ..evaluation.fixed_assets import BOUNDARIES, require
from ..models.game_outcome import model as m1
from ..models.plate_appearance import model as m2
from ..models.plate_appearance import probability as p2
from ..models.game_state_value import model as m4
from ..models.game_state_value import value as v4
from ..models.pitch_outcome import model as m5
from ..models.player_performance import data as d3
from ..models.player_performance import model as m3
from ..models.player_performance import development as dev3


JsonObject = dict[str, object]

M4_FEATURE_NAMES = (
    "inning",
    "attacking_home",
    "score_difference",
    "outs",
    "on_first",
    "on_second",
    "on_third",
    "score_difference_x_inning",
    "score_difference_x_home",
    "absolute_score_difference",
    "late_inning",
    "score_difference_per_future_opportunity",
)


def compare(
    actual: object,
    expected: object,
    label: str,
) -> float:
    """구조와 수치를 비교하고 최대 절대 수치 오차를 반환한다.

    정수·문자열·불리언·결측은 정확히 비교한다.
    JSON 복원 배열과 Python 튜플의 표현 차이는 허용한다.
    """
    if isinstance(expected, Mapping):
        require(
            isinstance(actual, Mapping)
            and actual.keys() == expected.keys(),
            f"{label}: 객체 키 불일치",
        )

        return max(
            (
                compare(actual[key], value, f"{label}.{key}")
                for key, value in expected.items()
            ),
            default=0.0,
        )

    if isinstance(expected, (tuple, list)):
        require(
            isinstance(actual, (tuple, list))
            and len(actual) == len(expected),
            f"{label}: 배열 길이 불일치",
        )

        return max(
            (
                compare(left, right, f"{label}[{index}]")
                for index, (left, right) in enumerate(
                    zip(actual, expected, strict=True)
                )
            ),
            default=0.0,
        )

    if (
        type(actual) in (int, float)
        and type(expected) in (int, float)
        and (
            type(actual) is float
            or type(expected) is float
        )
    ):
        require(
            isfinite(actual)
            and isfinite(expected)
            and isclose(
                actual,
                expected,
                abs_tol=ABS_TOLERANCE,
                rel_tol=REL_TOLERANCE,
            ),
            f"{label}: 수치 불일치 {actual!r} != {expected!r}",
        )
        return abs(actual - expected)

    require(
        type(actual) is type(expected)
        and actual == expected,
        f"{label}: 값 또는 타입 불일치",
    )

    return 0.0


def subtract(
    left: Sequence[float],
    right: Sequence[float],
) -> list[float]:
    """같은 출력 순서의 차이를 계산한다."""
    return [
        first - second
        for first, second in zip(left, right, strict=True)
    ]


def add(
    left: Sequence[float],
    right: Sequence[float],
) -> list[float]:
    """같은 출력 순서의 합을 계산한다."""
    return [
        first + second
        for first, second in zip(left, right, strict=True)
    ]


def m2_vector(
    operating: Mapping[str, object],
    inputs: Mapping[str, object],
) -> dict[str, float]:
    """실제 저장 로지스틱 계수가 있는 특징만 사용한다."""
    weights = operating["base"]["weights"]

    return {
        key: value
        for key, value in m2.features(inputs).items()
        if key != "bias" and key in weights and value != 0
    }


def m5_vector(
    operating: m5.Runtime,
    inputs: Mapping[str, object],
) -> dict[str, float]:
    """운영 Runtime과 동일한 선수 ID 후퇴를 반영한다."""
    weights = operating.model["weights"]
    values: Counter[str] = Counter()

    for original in m5.encode(inputs):
        key = original

        if key not in weights:
            key = m5.fallback_key(key)

        if key is not None and key in weights:
            values[key] += 1

    return {
        key: float(value)
        for key, value in values.items()
    }


def linear_weights(
    name: str,
    operating: object,
) -> tuple[tuple[str, ...], list[float], dict[str, list[float]]]:
    """운영 선형 모델의 출력 순서, 절편, 실제 계수를 반환한다."""
    if name == "m2":
        base = operating["base"]

        require(
            base["kind"] == "logistic",
            "M2 운영 모델이 로지스틱 구조가 아닙니다.",
        )

        weights = base["weights"]

        return (
            tuple(m2.CLASSES),
            list(weights["bias"]),
            {
                key: list(value)
                for key, value in weights.items()
                if key != "bias"
            },
        )

    if name == "m5":
        model = operating.model

        require(
            model["kind"] == "logistic",
            "M5 운영 모델이 로지스틱 구조가 아닙니다.",
        )

        weights = model["weights"]

        return (
            tuple(m5.CLASSES),
            list(weights["bias"]),
            {
                key: list(value)
                for key, value in weights.items()
                if key != "bias"
            },
        )

    require(name == "m4", "지원하지 않는 선형 설명 모델")

    predictor = operating["predictor"]

    require(
        predictor["name"] == "logistic_core"
        and len(predictor["means"]) == len(M4_FEATURE_NAMES)
        and len(predictor["weights"]) == len(M4_FEATURE_NAMES) + 1,
        "M4 운영 모델 특징 또는 계수 구조 불일치",
    )

    return (
        ("win_logit",),
        [predictor["weights"][0]],
        {
            key: [value]
            for key, value in zip(
                M4_FEATURE_NAMES,
                predictor["weights"][1:],
                strict=True,
            )
        },
    )


def train_background(
    database: Path,
    name: str,
    operating: object,
) -> JsonObject:
    """정답 접근을 차단하고 고정 Train 특징의 평균을 계산한다."""
    _, _, weights = linear_weights(name, operating)

    if name == "m4":
        predictor = operating["predictor"]

        return {
            "source": "stored_weighted_train_standardization",
            "split": "train",
            "means": dict.fromkeys(weights, 0.0),
            "original_means": dict(
                zip(
                    M4_FEATURE_NAMES,
                    predictor["means"],
                    strict=True,
                )
            ),
            "original_scales": dict(
                zip(
                    M4_FEATURE_NAMES,
                    predictor["scales"],
                    strict=True,
                )
            ),
            "weighting": "training_observation_frequency",
        }

    columns = final.model_columns(name)

    query = final.input_query(name).replace(
        "x.split='test'",
        "x.split='train'",
    )

    require(
        "x.split='train'" in query
        and "x.split='test'" not in query,
        "Train 배경 조회 분할 오류",
    )

    sums: Counter[str] = Counter()
    count = 0

    vector = m2_vector if name == "m2" else m5_vector

    with final.open_database(database) as db:
        db.set_authorizer(
            final.prediction_authorizer(name, columns)
        )

        for row in db.execute(
            query,
            ("0001-01-01", BOUNDARIES[0]),
        ):
            final.validate_day(row["sample_date"], row["split"])

            require(
                row["split"] == "train",
                "설명 배경에 비학습 표본 포함",
            )

            inputs = {
                column: row[column]
                for column in columns
            }

            sums.update(vector(operating, inputs))
            count += 1

    expected_count = sum(
        operating["base"]["counts"]
        if name == "m2"
        else operating.model["counts"]
    )

    require(
        count > 0 and count == expected_count,
        f"{name}: Train 배경 분모 불일치 "
        f"({count} != {expected_count})",
    )

    return {
        "source": "fixed_model_database_encoded_inputs",
        "split": "train",
        "date_upper_exclusive": BOUNDARIES[0],
        "samples": count,
        "weighting": "one_per_eligible_training_sample",
        "means": {
            key: sums[key] / count
            for key in sorted(weights)
        },
    }


@dataclass
class LinearExplanation:
    """희소 선형 모델의 정확한 독립 개입 SHAP 설명기."""

    labels: tuple[str, ...]
    bias: list[float]
    weights: dict[str, list[float]]
    means: dict[str, float]

    def __post_init__(self) -> None:
        require(
            self.weights.keys() == self.means.keys()
            and len(self.bias) == len(self.labels)
            and all(
                len(value) == len(self.labels)
                for value in self.weights.values()
            ),
            "선형 설명 배경 및 계수 차원 오류",
        )

        self.keys = tuple(sorted(self.weights))

        self.base = [
            self.bias[index]
            + fsum(
                self.weights[key][index] * self.means[key]
                for key in self.keys
            )
            for index in range(len(self.labels))
        ]

        self.samples = 0
        self.absolute_adjustments: Counter[str] = Counter()

    def explain(
        self,
        values: Mapping[str, float],
        expected_logits: Sequence[float],
        *,
        aggregate: bool,
    ) -> JsonObject:
        """활성·비활성 기여의 합으로 실제 로짓을 검산한다."""
        require(
            values.keys() <= self.weights.keys(),
            "학습 계수가 없는 설명 특징",
        )

        active = {
            key: value
            for key, value in sorted(values.items())
            if value != 0.0
        }

        contributions = {
            key: [
                weight * (value - self.means[key])
                for weight in self.weights[key]
            ]
            for key, value in active.items()
        }

        # 비활성 특징은 x=0이므로 기여가 -w*mu이다.
        # 전체 희소 사전을 매 표본에 복제하지 않는다.
        inactive = [
            self.bias[index]
            - self.base[index]
            + fsum(
                self.weights[key][index] * self.means[key]
                for key in active
            )
            for index in range(len(self.labels))
        ]

        reconstructed = [
            self.base[index]
            + inactive[index]
            + fsum(
                value[index]
                for value in contributions.values()
            )
            for index in range(len(self.labels))
        ]

        error = compare(
            reconstructed,
            expected_logits,
            "선형 SHAP 가산성",
        )

        if aggregate:
            self.samples += 1

            # 모든 비활성 값이 0이라는 기준에서
            # 활성 값이 발생한 경우의 절대 편차만 보정한다.
            for key, value in active.items():
                mean = self.means[key]

                self.absolute_adjustments[key] += (
                    abs(value - mean) - abs(mean)
                )

        return {
            "method": "exact_interventional_linear_shap",
            "scale": "logit",
            "class_order": self.labels,
            "base_values": self.base,
            "active_features": active,
            "active_shap_values": contributions,
            "inactive_shap_sum": inactive,
            "inactive_feature_formula": "-weight * train_mean",
            "logits": reconstructed,
            "max_absolute_error": error,
            "causal_interpretation": False,
        }

    def report(self) -> JsonObject:
        """선정 모집단의 전체 평균 절대 SHAP을 계산한다."""
        if not self.samples:
            return {
                "samples": 0,
                "status": "unsupported",
                "mean_absolute_shap": None,
            }

        features = {}

        for key in self.keys:
            mean_deviation = max(
                0.0,
                abs(self.means[key])
                + self.absolute_adjustments[key] / self.samples,
            )

            values = [
                abs(weight) * mean_deviation
                for weight in self.weights[key]
            ]

            features[key] = {
                "by_output": values,
                "mean_across_outputs": (
                    fsum(values) / len(values)
                ),
            }

        order = sorted(
            features,
            key=lambda key: (
                -features[key]["mean_across_outputs"],
                key,
            ),
        )

        return {
            "samples": self.samples,
            "status": "supported",
            "method": "mean_absolute_interventional_linear_shap",
            "scale": "logit",
            "class_order": self.labels,
            "base_values": self.base,
            "features": features,
            "ranked_features": order,
            "output_weighting": "equal_per_output",
        }


def build_linear(
    name: str,
    operating: object,
    background: Mapping[str, object],
) -> LinearExplanation:
    """저장 모델과 사전 고정한 배경을 연결한다."""
    labels, bias, weights = linear_weights(name, operating)

    return LinearExplanation(
        labels,
        bias,
        weights,
        dict(background["means"]),
    )


def explain_m1(
    operating: Mapping[str, object],
    row: Mapping[str, object],
) -> JsonObject:
    """득점 프로파일과 결합 분포의 실제 계산 근거를 설명한다."""
    inputs = row["input"]
    forecast = m1.predict(operating, inputs)

    error = compare(
        forecast,
        row["operating"],
        "M1 저장 운영 예측",
    )

    teams = m1.history(inputs)
    joint = forecast["joint"]

    profiles = {}

    for side in m1.SIDES:
        key = m1.profile(
            teams,
            side,
            operating["league_mean"],
        )

        counts = operating["profile_counts"].get(key, [])
        site_counts = operating["site_counts"][side]

        independent_pmf = joint[f"{side}_base_pmf"]
        expected_runs = forecast[f"{side}_expected_runs"]

        site_pmf = [
            count / sum(site_counts)
            for count in site_counts
        ]

        site_mean = m1.mean(site_pmf)
        independent_mean = m1.mean(independent_pmf)

        reconstructed_pmf = m1.pmf(
            site_counts,
            counts,
        )

        error = max(
            error,
            compare(
                reconstructed_pmf,
                independent_pmf,
                f"M1 {side} 프로파일 PMF",
            ),
        )

        profile_change = independent_mean - site_mean
        tie_change = expected_runs - independent_mean

        error = max(
            error,
            compare(
                site_mean + profile_change + tie_change,
                expected_runs,
                f"M1 {side} 기대득점 분해",
            ),
        )

        profiles[side] = {
            "history": teams[side],
            "smoothed_rates": m1.rates(
                teams,
                side,
                operating["league_mean"],
            ),
            "profile": key,
            "profile_train_support": sum(counts),
            "site_train_support": sum(site_counts),
            "site_pmf": site_pmf,
            "profile_pmf": independent_pmf,
            "site_mean": site_mean,
            "profile_mean": independent_mean,
            "profile_change": profile_change,
            "tie_adjustment_change": tie_change,
            "final_mean": expected_runs,
        }

    home = joint["home_base_pmf"]
    away = joint["away_base_pmf"]

    independent = {
        "home": m1.greater(home, away),
        "tie": m1.diagonal(home, away),
        "away": m1.greater(away, home),
    }

    final_probability = forecast["result_probabilities"]

    stages = {}

    for side, values in profiles.items():
        stages[f"{side}.profile_change"] = [
            values["profile_change"]
        ]
        stages[f"{side}.tie_change"] = [
            values["tie_adjustment_change"]
        ]

    stages["result.tie_weight_change"] = [
        final_probability[label] - independent[label]
        for label in m1.OUTCOMES
    ]

    return {
        "method": "profile_and_joint_distribution_decomposition",
        "profiles": profiles,
        "independent_result_probability": independent,
        "adjusted_result_probability": final_probability,
        "joint": joint,
        "tail": forecast["tail"],
        "result_order": m1.OUTCOMES,
        "stages": stages,
        "max_absolute_error": error,
    }


def explain_m2(
    operating: Mapping[str, object],
    row: Mapping[str, object],
    linear: LinearExplanation,
    *,
    aggregate: bool,
) -> JsonObject:
    """로짓·규칙 마스크·평활화·보정 단계를 연결한다."""
    inputs = row["input"]
    base = operating["base"]

    logits = m2.logits_for(
        base["weights"],
        m2.features(inputs),
    )

    contribution = linear.explain(
        m2_vector(operating, inputs),
        logits,
        aggregate=aggregate,
    )

    mask = p2.allowed_mask(inputs)
    raw = p2.masked_softmax(logits, mask)

    prior = p2.normalize_masked(
        m2.global_prior(base["counts"]),
        mask,
    )

    mixed = [
        (1.0 - m2.PRIOR_MIX) * probability
        + m2.PRIOR_MIX * reference
        for probability, reference in zip(
            raw,
            prior,
            strict=True,
        )
    ]

    official_mixed = p2.masked_probabilities(
        base,
        inputs,
    )

    final_probability = p2.calibrate_vector(
        mixed,
        operating["temperature"],
        operating["class_offsets"],
    )

    error = max(
        contribution["max_absolute_error"],
        compare(mixed, official_mixed, "M2 평활화"),
        compare(
            final_probability,
            row["operating"],
            "M2 저장 운영 확률",
        ),
    )

    require(
        all(
            allowed or probability == 0.0
            for allowed, probability in zip(
                mask,
                final_probability,
                strict=True,
            )
        ),
        "M2 금지 범주 확률 위반",
    )

    return {
        "method": "linear_shap_and_probability_pipeline",
        "linear": contribution,
        "class_order": m2.CLASSES,
        "allowed_mask": mask,
        "unmasked_softmax": m2.softmax(logits),
        "masked_softmax": raw,
        "masked_train_prior": prior,
        "smoothed_probability": mixed,
        "temperature": operating["temperature"],
        "class_offsets": operating["class_offsets"],
        "final_probability": final_probability,
        "stages": {
            "rule_mask": subtract(raw, m2.softmax(logits)),
            "prior_mix": subtract(mixed, raw),
            "calibration": subtract(final_probability, mixed),
        },
        "max_absolute_error": error,
    }


def re_evidence(
    table: m4.StatTable,
    state: m4.State,
    confirmed: int | None,
    expected: tuple[float, str, int | None],
) -> JsonObject:
    """RE의 실제 선택 통계 및 후퇴 근거를 복원하고 검산한다.

    Args:
        table: 고정 Train의 잔여 실득점 통계표.
        state: 설명 대상 경기 상태.
        confirmed: 공식 종료 결과. 미확정이면 None.
        expected: 기존 운영 함수가 반환한 RE, 조회 수준, 지원 수.

    Returns:
        실제 선택 키, 표본 수, 득점 합, 조회 경로 및 검산 결과.

    Raises:
        ValueError: 통계 근거와 기존 예측 결과가 일치하지 않는 경우.
    """
    if confirmed is not None or state.outs == 3:
        error = compare(
            (0.0, "half_ended", None),
            expected,
            "M4 종료 RE",
        )

        return {
            "status": "deterministic",
            "reason": (
                "confirmed_game_final"
                if confirmed is not None
                else "three_outs_half_ended"
            ),
            "selected_key": None,
            "selected_entry": None,
            "selected_support": None,
            "selected_runs_sum": None,
            "reconstructed_re": 0.0,
            "minimum_support": m4.MIN_SUPPORT,
            "lookup_trace": [],
            "max_absolute_error": error,
        }

    trace = []

    # 운영 모델의 조회 순서와 최소 지원 조건을 그대로 사용한다.
    # 조회 수준과 실제 통계표의 복합 키를 혼동하지 않는다.
    for key in v4.re_keys(state):
        entry = table.get(key)

        if entry is None:
            trace.append({
                "key": key,
                "status": "missing",
                "support": None,
                "runs_sum": None,
            })

            continue

        support, runs_sum = entry

        require(
            type(support) is int
            and support > 0
            and type(runs_sum) is int
            and runs_sum >= 0,
            f"M4 RE 통계 항목 오류: {key}",
        )

        if key != "global" and support < m4.MIN_SUPPORT:
            trace.append({
                "key": key,
                "status": "insufficient_support",
                "support": support,
                "runs_sum": runs_sum,
            })

            continue

        level = key.split(":", 1)[0]
        reconstructed = runs_sum / support

        # 기존 운영 함수의 반환값과 실제 통계 계산을 대조한다.
        error = compare(
            (reconstructed, level, support),
            expected,
            "M4 RE 선택 통계 검산",
        )

        trace.append({
            "key": key,
            "status": "selected",
            "support": support,
            "runs_sum": runs_sum,
        })

        return {
            "status": "supported",
            "reason": (
                "exact_state_supported"
                if level == "exact"
                else "fallback_to_supported_statistics"
            ),
            "selected_key": key,
            "selected_entry": list(entry),
            "selected_support": support,
            "selected_runs_sum": runs_sum,
            "reconstructed_re": reconstructed,
            "minimum_support": m4.MIN_SUPPORT,
            "lookup_trace": trace,
            "max_absolute_error": error,
        }

    raise ValueError("M4 RE 설명에 필요한 학습 통계 없음")


def explain_m4(
    operating: Mapping[str, object],
    row: Mapping[str, object],
    truth: Mapping[str, object],
    linear: LinearExplanation,
    *,
    aggregate: bool,
    pair: Mapping[str, object] | None,
) -> JsonObject:
    """일반 로지스틱 추론과 공식 종료 경계를 분리한다."""
    state = m4.parse_state(row["input"])
    predictor = operating["predictor"]

    raw_win = m4.predict_win(predictor, state)
    raw_re = v4.predict_re(operating["re_table"], state)

    error = compare(
        {"win": raw_win, "re": raw_re},
        row["operating"],
        "M4 저장 일반 추론",
    )

    evidence = m4.parse_final(truth["final_evidence"])
    confirmed = m4.final_result(state, evidence)

    require(
        (confirmed is not None) == truth["confirmed_final"],
        "M4 공식 종료 감사 불일치",
    )

    vector = m4.features(state, False)

    standardized = [
        (value - mean) / scale
        for value, mean, scale in zip(
            vector,
            predictor["means"],
            predictor["scales"],
            strict=True,
        )
    ]

    logit = predictor["weights"][0] + sum(
        weight * value
        for weight, value in zip(
            predictor["weights"][1:],
            standardized,
            strict=True,
        )
    )

    contribution = None

    if confirmed is None:
        contribution = linear.explain(
            dict(zip(M4_FEATURE_NAMES, standardized, strict=True)),
            [logit],
            aggregate=aggregate,
        )

        error = max(
            error,
            contribution["max_absolute_error"],
            compare(
                m4.sigmoid(logit),
                raw_win[0],
                "M4 Sigmoid",
            ),
        )

    evaluated = v4.state_value(
        operating,
        state,
        evidence,
        with_importance=False,
    )

    expected_win = (
        float(confirmed)
        if confirmed is not None
        else raw_win[0]
    )

    expected_re = (
        0.0
        if confirmed is not None
        else raw_re[0]
    )

    error = max(
        error,
        compare(
            evaluated["win_probability"],
            expected_win,
            "M4 평가 승리 확률",
        ),
        compare(
            evaluated["remaining_expected_runs"],
            expected_re,
            "M4 평가 잔여 득점",
        ),
    )

    # 저장 일반 추론의 근거와 공식 종료를 반영한 근거를 분리한다.
    raw_re_evidence = re_evidence(
        operating["re_table"],
        state,
        None,
        raw_re,
    )

    evaluated_re_evidence = re_evidence(
        operating["re_table"],
        state,
        confirmed,
        (
            evaluated["remaining_expected_runs"],
            evaluated["re_method"],
            evaluated["re_support"],
        ),
    )

    error = max(
        error,
        raw_re_evidence["max_absolute_error"],
        evaluated_re_evidence["max_absolute_error"],
    )

    global_re = v4.predict_re(
        operating["re_table"],
        state,
        confirmed,
        global_only=True,
    )

    pair_result = None

    if pair is not None:
        before = m4.parse_state(pair["pre"]["input"])
        after = m4.parse_state(pair["post"]["input"])

        post_evidence = m4.parse_final(
            pair["post_truth"]["final_evidence"]
        )

        pair_result = v4.plate_value(
            operating,
            before,
            after,
            post_evidence,
            with_importance=True,
        )

        pair_result["pre_sample_id"] = pair["pre"]["sample_id"]
        pair_result["post_sample_id"] = pair["post"]["sample_id"]

    return {
        "method": (
            "observed_final_boundary"
            if confirmed is not None
            else "linear_shap_and_state_statistics"
        ),
        "linear": contribution,
        "feature_order": M4_FEATURE_NAMES,
        "features": vector,
        "raw_logit": logit,
        "raw_prediction": {
            "win": raw_win,
            "re": raw_re,
        },
        "evaluated": evaluated,
        "confirmed_final": confirmed is not None,
        "re_lookup_order": v4.re_keys(state),
        "re_selected_key": (
            evaluated_re_evidence["selected_key"]
        ),
        "re_selected_entry": (
            evaluated_re_evidence["selected_entry"]
        ),
        "raw_re_evidence": raw_re_evidence,
        "evaluated_re_evidence": evaluated_re_evidence,
        "global_re": global_re,
        "plate_value": pair_result,
        "stages": {
            "terminal_win_override": [
                expected_win - raw_win[0]
            ],
            "terminal_re_override": [
                expected_re - raw_re[0]
            ],
            "conditional_re_change": [
                expected_re - global_re[0]
            ],
        },
        "max_absolute_error": error,
    }


def explain_m5(
    operating: m5.Runtime,
    row: Mapping[str, object],
    linear: LinearExplanation,
    *,
    aggregate: bool,
) -> JsonObject:
    """실제 공동 로짓과 두 주변 확률의 관계를 설명한다."""
    inputs = row["input"]
    keys = m5.encode(inputs)

    logits = m5.logits_for(
        m5.active_weights(
            operating.model["weights"],
            keys,
        )
    )

    contribution = linear.explain(
        m5_vector(operating, inputs),
        logits,
        aggregate=aggregate,
    )

    raw = m5.softmax(logits)

    joint = [
        (1.0 - m5.PRIOR_MIX) * probability
        + m5.PRIOR_MIX * prior
        for probability, prior in zip(
            raw,
            operating.prior,
            strict=True,
        )
    ]

    pitch_type, pitch_result = m5.marginalize(joint)

    error = max(
        contribution["max_absolute_error"],
        compare(
            joint,
            operating.probabilities(keys),
            "M5 운영 Runtime 확률",
        ),
        compare(
            joint,
            row["operating"],
            "M5 저장 공동 확률",
        ),
        compare(fsum(pitch_type), 1.0, "M5 구종 확률 합"),
        compare(fsum(pitch_result), 1.0, "M5 결과 확률 합"),
    )

    return {
        "method": "linear_shap_and_joint_marginal_decomposition",
        "linear": contribution,
        "logit_contract": "stored_temperature_adjusted_weights",
        "encoded_keys": keys,
        "joint_order": m5.CLASSES,
        "pitch_type_order": m5.TYPES,
        "pitch_result_order": m5.RESULTS,
        "raw_softmax": raw,
        "train_prior": operating.prior,
        "joint_probability": joint,
        "pitch_type_probability": pitch_type,
        "pitch_result_probability": pitch_result,
        "stages": {
            "joint_prior_mix": subtract(joint, raw),
        },
        "max_absolute_error": error,
    }


def player_history(
    fitted: dev3.Predictor,
    player: d3.PlayerInput,
) -> JsonObject:
    """운영 후보의 일수당 이력 계산을 단계별로 복원한다."""
    spec = fitted.specification
    baseline = fitted.baseline

    days = max(dev3.RECENT_DAYS, m3.elapsed(player))

    rates = []
    long_rates = []
    short_rates = []
    backoffs = []

    for index, (prior, season, recent) in enumerate(
        zip(
            baseline.daily_prior,
            player.season_counts,
            player.recent_counts,
            strict=True,
        )
    ):
        long_rate = (
            prior if season is None else season / days
        )

        long_backoff = season is None

        if (
            spec.missing_history_backoff
            and index == 0
            and player.prior_rejected > 0
            and season == 0
        ):
            long_rate = prior
            long_backoff = True

        short_rate = (
            long_rate
            if recent is None
            else recent / dev3.RECENT_DAYS
        )

        short_backoff = recent is None

        if (
            spec.missing_history_backoff
            and index == 0
            and player.prior_rejected > 0
            and recent == 0
        ):
            short_rate = long_rate
            short_backoff = True

        rate = (
            spec.recent_weight * short_rate
            + (1.0 - spec.recent_weight) * long_rate
        )

        rates.append(rate)
        long_rates.append(long_rate)
        short_rates.append(short_rate)

        backoffs.append({
            "season_backoff": long_backoff,
            "recent_backoff": short_backoff,
        })

    duration = dev3.durations(
        baseline,
        player,
        use_known_end=True,
    )

    prior_raw = [
        rate * length
        for length in duration
        for rate in baseline.daily_prior
    ]

    raw = [
        rate * length
        for length in duration
        for rate in rates
    ]

    projected = m3.project(player, raw)

    expected = dev3.history_prediction(
        baseline,
        player,
        spec,
    )

    error = compare(
        projected,
        expected,
        "M3 운영 이력 기준 복원",
    )

    return {
        "record_order": d3.metrics(player.role),
        "duration": duration,
        "recent_weight": spec.recent_weight,
        "daily_prior": baseline.daily_prior,
        "season_daily_rates": long_rates,
        "recent_daily_rates": short_rates,
        "weighted_daily_rates": rates,
        "backoffs": backoffs,
        "prior_raw": prior_raw,
        "history_raw": raw,
        "history_projected": projected,
        "max_absolute_error": error,
    }


def tree_path(
    root: m3.Node,
    vector: Sequence[float],
    names: Sequence[str],
) -> tuple[list[JsonObject], m3.Node]:
    """저장된 실제 분기 조건을 따라 선택 잎을 찾는다."""
    path = []
    node = root

    while node.feature is not None:
        index = node.feature

        require(
            node.threshold is not None
            and node.left is not None
            and node.right is not None,
            "M3 트리 자산 손상",
        )

        value = vector[index]
        goes_left = value <= node.threshold

        child = node.left if goes_left else node.right

        path.append({
            "feature": names[index],
            "value": value,
            "threshold": node.threshold,
            "branch": "left" if goes_left else "right",
            "parent_values": node.values,
            "child_values": child.values,
            "child_support": child.support,
            "residual_change": subtract(
                child.values,
                node.values,
            ),
        })

        node = child

    compare(
        node.values,
        root.predict(vector),
        "M3 선택 트리 잎",
    )

    return path, node


def explain_m3(
    operating: object,
    row: Mapping[str, object],
) -> JsonObject:
    """타자 이력 또는 투수 공동 잔차 트리의 실제 계산을 설명한다."""
    fitted, intervals, _ = operating
    player = d3.parse_input(row["input"])

    require(
        isinstance(fitted, dev3.Predictor)
        and fitted.role == player.role
        and player.known_end is None,
        "M3 운영 자산 또는 입력 계약 불일치",
    )

    spec = fitted.specification

    history = player_history(fitted, player)
    base = history["history_projected"]

    stages = {
        "history_rate_change": subtract(
            history["history_raw"],
            history["prior_raw"],
        ),
        "history_projection": subtract(
            base,
            history["history_raw"],
        ),
    }

    tree = None

    if spec.mode == "history":
        require(
            fitted.tree is None,
            "M3 이력 예측기에 트리 존재",
        )
        final_values = base

    else:
        require(
            spec.mode == "residual"
            and fitted.tree is not None,
            "고정 M3 투수 잔차 트리 계약 불일치",
        )

        vector = m3.features(player)
        names = m3.feature_names(player.role)

        path, leaf = tree_path(
            fitted.tree,
            vector,
            names,
        )

        factors = dev3.scenario_factors(
            fitted.baseline,
            player,
        )

        width = len(d3.metrics(player.role))

        corrections = [
            value
            * scale
            * spec.strength
            * factors[index // width]
            for index, (value, scale) in enumerate(
                zip(
                    leaf.values,
                    fitted.baseline.scales,
                    strict=True,
                )
            )
        ]

        raw = add(base, corrections)
        final_values = m3.project(player, raw)

        stages["tree_residual_correction"] = corrections
        stages["final_projection"] = subtract(
            final_values,
            raw,
        )

        residual_reconstructed = list(fitted.tree.values)

        for step in path:
            residual_reconstructed = add(
                residual_reconstructed,
                step["residual_change"],
            )

        compare(
            residual_reconstructed,
            leaf.values,
            "M3 트리 경로 잔차",
        )

        tree = {
            "depth": fitted.depth,
            "feature_order": names,
            "features": vector,
            "path": path,
            "root_residual": fitted.tree.values,
            "leaf_residual": leaf.values,
            "leaf_support": leaf.support,
            "strength": spec.strength,
            "scales": fitted.baseline.scales,
            "scenario_factors": factors,
            "corrections": corrections,
            "before_projection": raw,
        }

    expected = fitted.predict(player)

    forecast = m3.format_prediction(
        fitted,
        player,
        intervals,
    )["periods"]

    error = max(
        history["max_absolute_error"],
        compare(final_values, expected, "M3 최종 Count"),
        compare(
            forecast,
            row["operating"],
            "M3 저장 기간·기록·구간",
        ),
    )

    reconstructed = list(history["prior_raw"])

    for changes in stages.values():
        reconstructed = add(reconstructed, changes)

    error = max(
        error,
        compare(
            reconstructed,
            final_values,
            "M3 기준값 및 단계 기여",
        ),
    )

    return {
        "method": (
            "history_calculation_decomposition"
            if tree is None
            else "history_and_residual_tree_path"
        ),
        "candidate": fitted.name,
        "role": player.role,
        "output_order": d3.output_keys(player.role),
        "anchor": player.anchor.isoformat(),
        "history": history,
        "tree": tree,
        "final_values": final_values,
        "periods": forecast,
        "stages": stages,
        "max_absolute_error": error,
    }


class ExplanationSummary:
    """설명 표본의 단계 변화 및 계산 경로를 집계한다."""

    def __init__(self) -> None:
        self.samples = 0
        self.methods: Counter[str] = Counter()
        self.routes: Counter[str] = Counter()
        self.absolute_stages: dict[str, list[float]] = {}
        self.stage_counts: Counter[str] = Counter()

    def add(self, explanation: Mapping[str, object]) -> None:
        """전체 설명 모집단의 결과만 집계한다."""
        self.samples += 1
        self.methods[explanation["method"]] += 1

        for name, changes in explanation["stages"].items():
            values = [abs(value) for value in changes]

            if name not in self.absolute_stages:
                self.absolute_stages[name] = values
            else:
                self.absolute_stages[name] = add(
                    self.absolute_stages[name],
                    values,
                )

            self.stage_counts[name] += 1

        for side, profile in explanation.get(
            "profiles", {}
        ).items():
            self.routes[
                f"{side}:{profile['profile']}"
            ] += 1

        tree = explanation.get("tree")

        if tree is not None:
            for step in tree["path"]:
                self.routes[
                    f"{step['feature']}:{step['branch']}"
                ] += 1

        evaluated = explanation.get("evaluated")

        if evaluated is not None:
            self.routes[
                f"re:{evaluated['re_method']}"
            ] += 1

            self.routes[
                f"boundary:{evaluated['boundary']}"
            ] += 1

    def report(self) -> JsonObject:
        """단계별 지원 분모를 보존한 전체 설명을 반환한다."""
        return {
            "samples": self.samples,
            "methods": dict(self.methods),
            "routes": dict(self.routes),
            "mean_absolute_stage_changes": {
                name: {
                    "samples": self.stage_counts[name],
                    "values": [
                        value / self.stage_counts[name]
                        for value in totals
                    ],
                }
                for name, totals in self.absolute_stages.items()
            },
            "interpretation": (
                "calculation_stage_changes_not_causal_effects"
            ),
        }


def explain(
    name: str,
    operating: object,
    sample: Mapping[str, object],
    linear: LinearExplanation | None,
    *,
    aggregate: bool,
) -> JsonObject:
    """고정 운영 예측기별 설명 함수를 선택한다."""
    row = sample["prediction"]
    truth = sample["target"]

    if name == "m1":
        return explain_m1(operating, row)

    if name == "m2":
        require(linear is not None, "M2 설명 배경 누락")

        return explain_m2(
            operating,
            row,
            linear,
            aggregate=aggregate,
        )

    if name == "m4":
        require(linear is not None, "M4 설명 배경 누락")

        return explain_m4(
            operating,
            row,
            truth,
            linear,
            aggregate=aggregate,
            pair=sample.get("pair"),
        )

    if name == "m5":
        require(linear is not None, "M5 설명 배경 누락")

        return explain_m5(
            operating,
            row,
            linear,
            aggregate=aggregate,
        )

    require(
        name in {"m3_hitter", "m3_pitcher"},
        "지원하지 않는 설명 모델",
    )

    return explain_m3(operating, row)
