"""M3 개선 후보의 공동 학습·공정한 평가·운영 모델 선정."""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
import math
from time import perf_counter

from . import data, evaluation, model
from ...datasets.model_data import require
from ...datasets.published_sources import as_row


BASELINE_NAME = "history_baseline"
SELECTION_TOLERANCE = 1e-12
COUNT_TOLERANCE = 1e-8
RECENT_DAYS = 28
SHORT_DAYS = 7
SCENARIO_DAYS = (1, 7, 14, 28, 35)
RESIDUAL_STRENGTHS = (0.25, 0.50, 1.00)


@dataclass(frozen=True, slots=True)
class Candidate:
    """하나의 공동 예측 계산을 정의하는 후보 설정."""

    name: str
    mode: str
    depth: int
    strength: float
    recent_weight: float
    missing_history_backoff: bool


def candidate_specs() -> tuple[Candidate, ...]:
    """개선 실험에서 비교할 후보를 고정한다."""
    history = (
        Candidate(
            BASELINE_NAME,
            "history",
            0,
            0.0,
            model.RECENT_WEIGHT,
            False,
        ),
        Candidate(
            "history_balanced_50",
            "history",
            0,
            0.0,
            0.50,
            True,
        ),
        Candidate(
            "history_recent_90",
            "history",
            0,
            0.0,
            0.90,
            True,
        ),
    )

    residual = tuple(
        Candidate(
            (
                f"joint_residual_tree_depth_{depth}"
                if strength == 1.0
                else (
                    f"joint_residual_tree_depth_{depth}"
                    f"_shrink_{int(strength * 100)}"
                )
            ),
            "residual",
            depth,
            strength,
            model.RECENT_WEIGHT,
            False,
        )
        for depth in model.CANDIDATE_DEPTHS
        for strength in RESIDUAL_STRENGTHS
    )

    direct = tuple(
        Candidate(
            f"joint_direct_tree_depth_{depth}",
            "direct",
            depth,
            1.0,
            model.RECENT_WEIGHT,
            False,
        )
        for depth in model.CANDIDATE_DEPTHS
    )

    return (*history, *residual, *direct)


CANDIDATES = candidate_specs()
SPEC_BY_NAME = {
    candidate.name: candidate
    for candidate in CANDIDATES
}
CANDIDATE_ORDER = {
    candidate.name: index
    for index, candidate in enumerate(CANDIDATES)
}


def settings() -> dict[str, object]:
    """기존 평가 결과 확인 후 새 실험 전에 고정한 설정."""
    return {
        "candidates": [
            asdict(candidate)
            for candidate in CANDIDATES
        ],
        "baseline": BASELINE_NAME,
        "selection_rule": (
            "strict_macro_mae_improvement_and_nonworse_rmse"
        ),
        "selection_tolerance": SELECTION_TOLERANCE,
        "selection_tie_break": (
            "macro_mae_then_macro_rmse_then_candidate_order"
        ),
        "fallback": BASELINE_NAME,
        "baseline_is_comparison_only": False,
        "history_missing_backoff": (
            "train_prior_or_long_rate_for_rejected_history"
        ),
        "known_end_scenario": (
            "duration_adjusted_tree_output_then_coherent_projection"
        ),
        "scenario_days": list(SCENARIO_DAYS),
        "common_ratio_policy": (
            "truth_valid_and_all_compared_prediction_denominators_positive"
        ),
        "validation_previously_observed": True,
        "final_test_used": False,
    }


def expected_remaining_days(
    baseline: model.Baseline,
    player: data.PlayerInput,
) -> float:
    """과거 Train 시즌의 종료 통계에 따른 예상 잔여 일수."""
    return max(
        float(model.MIN_REMAINING_DAYS),
        baseline.remaining_end_offset - model.elapsed(player),
    )


def durations(
    baseline: model.Baseline,
    player: data.PlayerInput,
    *,
    use_known_end: bool,
) -> model.Vector:
    """예측 시점에 허용되는 정보만으로 기간 환산 일수를 계산한다."""
    remaining = expected_remaining_days(baseline, player)

    if use_known_end and player.known_end is not None:
        remaining = float(
            (player.known_end - player.anchor).days
        )

    require(remaining > 0, "잔여 정규시즌 기간 오류")

    return (
        min(float(SHORT_DAYS), remaining),
        min(float(RECENT_DAYS), remaining),
        remaining,
    )


def history_prediction(
    baseline: model.Baseline,
    player: data.PlayerInput,
    spec: Candidate,
) -> model.Vector:
    """과거 출전량 및 기록률을 기간별 Count로 환산한다.

    기존 기준은 이전 구현과 동일하게 유지한다.
    개선 후보는 관측 거절 이력이 있는 경우 실제 기록 미관측을
    공식적인 출전 0으로 단정하지 않고 과거 통계로 후퇴한다.
    """
    if (
        spec.recent_weight == model.RECENT_WEIGHT
        and not spec.missing_history_backoff
    ):
        return baseline.predict(player)

    season_days = max(
        RECENT_DAYS,
        model.elapsed(player),
    )
    rates: list[float] = []

    for index, (prior, season, recent) in enumerate(
        zip(
            baseline.daily_prior,
            player.season_counts,
            player.recent_counts,
        )
    ):
        long_rate = (
            prior
            if season is None
            else season / season_days
        )

        if (
            spec.missing_history_backoff
            and index == 0
            and player.prior_rejected > 0
            and player.season_counts[0] == 0
        ):
            long_rate = prior

        short_rate = (
            long_rate
            if recent is None
            else recent / RECENT_DAYS
        )

        if (
            spec.missing_history_backoff
            and index == 0
            and player.prior_rejected > 0
            and player.recent_counts[0] == 0
        ):
            short_rate = long_rate

        rates.append(
            spec.recent_weight * short_rate
            + (1.0 - spec.recent_weight) * long_rate
        )

    raw = tuple(
        rate * duration
        for duration in durations(
            baseline,
            player,
            use_known_end=True,
        )
        for rate in rates
    )

    return model.project(player, raw)


def scenario_factors(
    baseline: model.Baseline,
    player: data.PlayerInput,
) -> model.Vector:
    """사전 확인 종료일이 있는 별도 시나리오의 축소 계수.

    확인되지 않은 미래 기간을 연장하거나 실제 종료일을
    학습 특징으로 입력하지 않는다. 구간 보정은 지원하지 않는다.
    """
    if player.known_end is None:
        return (1.0,) * len(data.HORIZONS)

    nominal = durations(
        baseline,
        player,
        use_known_end=False,
    )
    known = durations(
        baseline,
        player,
        use_known_end=True,
    )

    return tuple(
        min(1.0, actual / reference)
        for actual, reference in zip(known, nominal)
    )


@dataclass(frozen=True, slots=True)
class Predictor(model.Fitted):
    """기준 통계 또는 단일 공동 트리를 사용하는 운영 예측기.

    잔차 축소 계수는 한 트리의 출력에 적용하는 고정 계산이다.
    독립적으로 학습된 복수 트리를 합산하지 않는다.
    """

    variant: str

    @property
    def name(self) -> str:
        """후보 및 저장 자산의 고정 식별자."""
        return self.variant

    @property
    def specification(self) -> Candidate:
        """자산의 후보 식별자에 해당하는 고정 설정."""
        require(
            self.variant in SPEC_BY_NAME,
            "지원하지 않는 M3 후보",
        )
        return SPEC_BY_NAME[self.variant]

    def predict(self, player: data.PlayerInput) -> model.Vector:
        """모든 기간·기록을 한 번에 예측한다."""
        require(
            player.role == self.role,
            "운영 예측기와 입력 역할 불일치",
        )

        spec = self.specification
        require(
            self.depth == spec.depth,
            "후보 설정과 모델 깊이 불일치",
        )

        if spec.mode == "history":
            require(
                self.tree is None,
                "과거 기록 예측기에 불필요한 트리가 존재합니다",
            )
            return history_prediction(
                self.baseline,
                player,
                spec,
            )

        require(
            self.tree is not None,
            "공동 회귀 트리 자산 누락",
        )

        tree_values = self.tree.predict(
            model.features(player)
        )
        width = len(data.metrics(self.role))

        factors = scenario_factors(
            self.baseline,
            player,
        )
        require(
            len(tree_values)
            == len(self.baseline.scales)
            == len(data.output_keys(self.role)),
            "공동 트리 출력 차원 불일치",
        )

        corrections = tuple(
            value
            * scale
            * spec.strength
            * factors[index // width]
            for index, (value, scale) in enumerate(
                zip(tree_values, self.baseline.scales)
            )
        )

        if spec.mode == "residual":
            base = history_prediction(
                self.baseline,
                player,
                spec,
            )
            raw = tuple(
                value + correction
                for value, correction in zip(
                    base, corrections
                )
            )

        else:
            require(
                spec.mode == "direct",
                "공동 회귀 계산 방식 오류",
            )
            raw = corrections

        return model.project(player, raw)

    def to_payload(self) -> dict[str, object]:
        """한 후보의 설정·기준 통계·트리를 JSON으로 저장한다."""
        return asdict(self)


def direct_training_rows(
    samples: Sequence[data.Sample],
    baseline: model.Baseline,
    cutoff: date,
) -> list[model.TrainingRow]:
    """정규화 Count를 직접 학습하는 공동 회귀 자료를 구성한다.

    기존 TrainingRow의 residuals 필드는 트리가 처리할
    관측된 수치 벡터를 담는다. 여기서는 잔차가 아닌 Count이다.
    """
    rows = []

    for sample in samples:
        labels = data.labels_for(sample, cutoff)

        observed = tuple(
            (index, value / baseline.scales[index])
            for index, value in enumerate(labels)
            if value is not None
        )

        if not observed:
            continue

        rows.append(
            model.TrainingRow(
                model.features(sample.player),
                observed,
            )
        )

    require(bool(rows), "직접 공동 회귀 학습 정답 없음")
    return rows


def fit_family(
    role: str,
    samples: Sequence[data.Sample],
    cutoff: date,
) -> dict[str, Predictor]:
    """해당 논리적 Train 범위에서 모든 후보를 적합한다.

    동일 구조의 잔차 축소 후보는 같은 트리의 학습 결과를
    재사용한다. 각 후보를 위해 중복 학습하지 않는다.
    """
    require(
        bool(samples)
        and all(
            sample.split == "train"
            and sample.player.role == role
            and sample.player.anchor < cutoff
            and sample.player.known_end is None
            for sample in samples
        ),
        "M3 후보 학습 범위 또는 역할 오류",
    )

    baseline = model.fit_baseline(
        role, samples, cutoff
    )

    residual_rows = model.training_rows(
        samples, baseline, cutoff
    )
    direct_rows = direct_training_rows(
        samples, baseline, cutoff
    )

    dimension = len(data.output_keys(role))

    trees = {
        (mode, depth): model.fit_tree(
            rows,
            dimension,
            depth,
        )
        for mode, rows in (
            ("residual", residual_rows),
            ("direct", direct_rows),
        )
        for depth in model.CANDIDATE_DEPTHS
    }

    return {
        spec.name: Predictor(
            role=role,
            depth=spec.depth,
            baseline=baseline,
            tree=(
                None
                if spec.mode == "history"
                else trees[(spec.mode, spec.depth)]
            ),
            variant=spec.name,
        )
        for spec in CANDIDATES
    }


def restore_baseline(
    role: str,
    payload: object,
) -> model.Baseline:
    """트리가 없는 과거 기록 운영 자산의 통계를 검증한다."""
    row = as_row(payload)

    require(
        set(row) == {
            "role",
            "daily_prior",
            "remaining_end_offset",
            "scales",
        }
        and row["role"] == role,
        "기준 통계 자산 구조 오류",
    )

    daily = tuple(
        model.number(value)
        for value in row["daily_prior"]
    )
    scales = tuple(
        model.number(value)
        for value in row["scales"]
    )
    end = model.number(
        row["remaining_end_offset"]
    )

    require(
        len(daily) == len(data.metrics(role))
        and len(scales) == len(data.output_keys(role))
        and all(value >= 0 for value in daily)
        and all(value >= 1 for value in scales)
        and end > 0,
        "기준 통계 차원 또는 값 오류",
    )

    return model.Baseline(
        role=role,
        daily_prior=daily,
        remaining_end_offset=end,
        scales=scales,
    )


def restore(payload: object) -> model.Fitted:
    """개선 후보의 운영 자산을 복원한다.

    최초 구현의 트리 내부 표현은 기존 회귀 테스트 및 비교
    재현용으로 허용한다. 전체 v1 운영 자산의 호환 여부는
    CLI의 버전·스키마 검사에서 별도로 결정한다.
    """
    row = as_row(payload)

    if set(row) == {
        "role", "depth", "baseline", "tree"
    }:
        return model.restore(row)

    require(
        set(row) == {
            "role",
            "depth",
            "baseline",
            "tree",
            "variant",
        },
        "개선 모델 자산 필드 오류",
    )

    role = row["role"]
    variant = row["variant"]

    require(
        isinstance(role, str)
        and role in data.ROLES
        and isinstance(variant, str)
        and variant in SPEC_BY_NAME,
        "모델 자산 역할 또는 후보 오류",
    )

    spec = SPEC_BY_NAME[variant]

    require(
        type(row["depth"]) is int
        and row["depth"] == spec.depth,
        "후보 자산 깊이 불일치",
    )

    if spec.mode == "history":
        require(
            row["tree"] is None,
            "과거 기록 자산에 트리가 존재합니다",
        )

        baseline = restore_baseline(
            role, row["baseline"]
        )
        tree = None

    else:
        core = model.restore({
            name: row[name]
            for name in (
                "role",
                "depth",
                "baseline",
                "tree",
            )
        })

        baseline = core.baseline
        tree = core.tree

    return Predictor(
        role=role,
        depth=spec.depth,
        baseline=baseline,
        tree=tree,
        variant=variant,
    )


def point_metrics(
    errors: evaluation.Errors,
) -> dict[str, object]:
    """구간을 적합하지 않는 비율의 점 예측 지표."""
    return {
        name: value
        for name, value in errors.result().items()
        if not name.startswith("interval_")
    }


@dataclass(slots=True)
class RatioComparison:
    """동일 정답 및 모든 비교 예측 분모가 유효한 비율 집계."""

    names: tuple[str, ...]
    truth_valid: int = 0
    common_actual_denominator: float = 0.0
    truth_denominator_total: float = 0.0
    zero_denominators: Counter[str] = field(
        default_factory=Counter
    )
    common_predicted_denominators: Counter[str] = field(
        default_factory=Counter
    )
    errors: dict[str, evaluation.Errors] = field(
        init=False
    )

    def __post_init__(self) -> None:
        require(
            bool(self.names)
            and len(set(self.names)) == len(self.names),
            "비율 비교 후보 사전 오류",
        )

        self.errors = {
            name: evaluation.Errors()
            for name in self.names
        }

    def add(
        self,
        actual: float,
        actual_denominator: float,
        predictions: Mapping[
            str, tuple[float, float]
        ],
        player: data.PlayerInput,
    ) -> None:
        """한 표본을 모든 비교 후보에 동일하게 적용한다."""
        require(
            set(predictions) == set(self.names)
            and actual_denominator > 0
            and 0 <= actual <= 1,
            "공통 비율 비교 입력 오류",
        )

        self.truth_valid += 1
        self.truth_denominator_total += actual_denominator

        available = True

        for name, (numerator, denominator) in (
            predictions.items()
        ):
            require(
                0 <= numerator
                <= denominator + COUNT_TOLERANCE,
                "예측 비율 Count 구성 오류",
            )

            if denominator == 0:
                self.zero_denominators[name] += 1
                available = False

        if not available:
            return

        self.common_actual_denominator += actual_denominator

        for name, (numerator, denominator) in (
            predictions.items()
        ):
            self.common_predicted_denominators[name] += (
                denominator
            )

            self.errors[name].add(
                actual,
                numerator / denominator,
                player,
            )

    def result(
        self,
        population: int,
    ) -> dict[str, object]:
        """후보 간 공통 평가 행과 제외된 분모를 반환한다."""
        common_counts = {
            errors.n
            for errors in self.errors.values()
        }

        require(
            len(common_counts) == 1,
            "공통 비율 평가 건수 불일치",
        )

        common_n = next(iter(common_counts))

        return {
            "population": population,
            "truth_valid_n": self.truth_valid,
            "truth_unavailable_n": (
                population - self.truth_valid
            ),
            "truth_denominator_total":
                self.truth_denominator_total,
            "common_n": common_n,
            "excluded_by_prediction_denominator_n": (
                self.truth_valid - common_n
            ),
            "common_actual_denominator_total":
                self.common_actual_denominator,
            "models": {
                name: {
                    **point_metrics(self.errors[name]),
                    "zero_prediction_denominator_n":
                        self.zero_denominators[name],
                    "common_predicted_denominator_total":
                        self.common_predicted_denominators[name],
                }
                for name in self.names
            },
        }


def ratio_definitions(
    role: str,
) -> tuple[tuple[str, str, str], ...]:
    """확인된 비율의 분자·분모 사전."""
    return (
        ("hit_rate", "hits", "ab"),
        (
            "strikeout_rate",
            "strikeouts",
            data.exposure(role),
        ),
    )


def compare_ratios(
    candidates: Mapping[str, Predictor],
    samples: Sequence[data.Sample],
    role: str,
    cutoff: date,
) -> dict[str, object]:
    """전체 후보 공통 및 기준-후보 쌍별 공통 비율을 평가한다."""
    names = data.metrics(role)
    width = len(names)
    candidate_names = tuple(candidates)

    require(
        BASELINE_NAME in candidates,
        "공통 비율 평가 기준 모델 누락",
    )

    ratio_keys = tuple(
        f"{horizon}.{ratio_name}"
        for horizon in data.HORIZONS
        for ratio_name, _, _ in ratio_definitions(role)
    )

    all_common = {
        key: RatioComparison(candidate_names)
        for key in ratio_keys
    }

    pairwise = {
        name: {
            key: RatioComparison(
                (BASELINE_NAME, name)
            )
            for key in ratio_keys
        }
        for name in candidate_names
        if name != BASELINE_NAME
    }

    for sample in samples:
        labels = data.labels_for(
            sample, cutoff
        )

        if not any(value is not None for value in labels):
            continue

        predictions = {
            name: predictor.predict(sample.player)
            for name, predictor in candidates.items()
        }

        for index, horizon in enumerate(data.HORIZONS):
            start = index * width
            actual = dict(zip(
                names,
                labels[start:start + width],
            ))
            target = sample.targets[index]

            for ratio_name, numerator, denominator in (
                ratio_definitions(role)
            ):
                n = actual[numerator]
                d = actual[denominator]

                if n is None or d is None or d == 0:
                    continue

                if horizon == "7d":
                    truth = n / d
                elif ratio_name == "hit_rate":
                    truth = target.hit_rate
                else:
                    truth = target.strikeout_rate

                if truth is None:
                    continue

                key = f"{horizon}.{ratio_name}"

                predicted_counts = {
                    name: (
                        points[
                            start + names.index(numerator)
                        ],
                        points[
                            start + names.index(denominator)
                        ],
                    )
                    for name, points in predictions.items()
                }

                all_common[key].add(
                    truth,
                    float(d),
                    predicted_counts,
                    sample.player,
                )

                for name, comparisons in pairwise.items():
                    comparisons[key].add(
                        truth,
                        float(d),
                        {
                            BASELINE_NAME:
                                predicted_counts[BASELINE_NAME],
                            name:
                                predicted_counts[name],
                        },
                        sample.player,
                    )

    return {
        "truth_policy": {
            "7d": "count_derived_diagnostic",
            "28d": "stored_ratio_and_valid_denominator",
            "remaining": (
                "stored_ratio_and_valid_denominator"
            ),
        },
        "all_candidates_common": {
            key: comparison.result(len(samples))
            for key, comparison in all_common.items()
        },
        "baseline_pairwise_common": {
            name: {
                key: comparison.result(len(samples))
                for key, comparison in comparisons.items()
            }
            for name, comparisons in pairwise.items()
        },
    }


def verify_derived(
    role: str,
    counts: Mapping[str, float],
    derived: object,
) -> None:
    """비율 분자·분모와 이닝을 독립 산식으로 검증한다."""
    values = as_row(derived)

    for name, numerator, denominator in ratio_definitions(
        role
    ):
        ratio = as_row(values[name])
        n = counts[numerator]
        d = counts[denominator]

        require(
            ratio["numerator"] == n
            and ratio["denominator"] == d,
            "파생 비율 분자·분모 불일치",
        )

        if d == 0:
            require(
                ratio["value"] is None
                and ratio["reason"] == "zero_denominator",
                "분모 0 처리 오류",
            )

        else:
            require(
                ratio["reason"] is None
                and math.isclose(
                    model.number(ratio["value"]),
                    n / d,
                    rel_tol=COUNT_TOLERANCE,
                    abs_tol=COUNT_TOLERANCE,
                ),
                "파생 비율 계산 오류",
            )

    if role == "pitcher":
        require(
            math.isclose(
                model.number(values["innings"]),
                counts["outs"] / 3,
                rel_tol=COUNT_TOLERANCE,
                abs_tol=COUNT_TOLERANCE,
            ),
            "투구 이닝 환산 오류",
        )


def verify_intervals(
    counts: Mapping[str, float],
    value: object,
) -> None:
    """주변 구간의 지지 상태·폭·점 예측 포함을 검사한다."""
    intervals = as_row(value)

    require(
        set(intervals) == set(counts),
        "예측 구간 기록 사전 오류",
    )

    for name, point in counts.items():
        interval = as_row(intervals[name])

        require(
            interval["target_coverage"]
            == model.INTERVAL_LEVEL,
            "구간 목표 포함 수준 오류",
        )

        lower = interval["lower"]
        upper = interval["upper"]

        if lower is None or upper is None:
            require(
                lower is None
                and upper is None
                and bool(interval["reason"]),
                "미지원 예측 구간 처리 오류",
            )

        else:
            require(
                interval["reason"] is None
                and 0 <= model.number(lower)
                <= point
                <= model.number(upper),
                "예측 구간 순서 또는 점 예측 포함 오류",
            )


def verify_prediction(
    player: data.PlayerInput,
    value: object,
) -> tuple[
    dict[str, dict[str, float]],
    dict[str, float] | None,
]:
    """기간별 Count와 시즌 합산 결과의 전체 계약을 검사한다."""
    row = as_row(value)
    role = player.role

    require(
        row["sample_id"] == player.sample_id
        and row["player_id"] == player.player_id
        and row["role"] == role
        and row["sample_date"]
        == player.anchor.isoformat()
        and row["competition"] == "regular",
        "예측 출력 식별자 또는 경기 범위 오류",
    )

    periods = as_row(row["periods"])

    require(
        set(periods) == set(data.HORIZONS),
        "기간별 예측 출력 누락",
    )

    counts_by_horizon: dict[
        str, dict[str, float]
    ] = {}

    for horizon in data.HORIZONS:
        period = as_row(periods[horizon])

        counts = {
            name: model.number(count)
            for name, count in as_row(
                period["counts"]
            ).items()
        }

        data.check_counts(role, counts)
        verify_derived(
            role, counts, period["derived"]
        )
        verify_intervals(
            counts, period["intervals"]
        )

        if player.known_end is not None:
            require(
                all(
                    as_row(interval)["lower"] is None
                    and as_row(interval)["upper"] is None
                    for interval in as_row(
                        period["intervals"]
                    ).values()
                ),
                "보정되지 않은 종료 시나리오의 구간 출력",
            )

        counts_by_horizon[horizon] = counts

    for name in data.metrics(role):
        short = counts_by_horizon["7d"][name]
        long = counts_by_horizon["28d"][name]
        remaining = counts_by_horizon["remaining"][name]

        require(
            short <= long + COUNT_TOLERANCE
            and long <= remaining + COUNT_TOLERANCE,
            "기간별 누적 Count 관계 오류",
        )

        if player.known_end is not None:
            days = (
                player.known_end - player.anchor
            ).days

            if days <= SHORT_DAYS:
                require(
                    short == long == remaining,
                    "7일 이내 종료 시나리오 일관성 오류",
                )
            elif days <= RECENT_DAYS:
                require(
                    long == remaining,
                    "28일 이내 종료 시나리오 일관성 오류",
                )

    season = as_row(
        row["season_projection"]
    )

    if player.current_counts is None:
        require(
            season["status"] == "unavailable",
            "현재 누적 없는 시즌 전망 출력",
        )
        return counts_by_horizon, None

    expected_scope = (
        "caller_verified_cumulative_projection"
        if player.current_scope == "caller_verified"
        else "observed_pbp_provisional_projection"
    )

    require(
        season["status"] == expected_scope
        and season["current_scope"] == player.current_scope,
        "시즌 누적 확인 범위 불일치",
    )

    current = dict(zip(
        data.metrics(role),
        player.current_counts,
    ))
    require(
        as_row(season["current_counts"]) == current,
        "시즌 전망 현재 누적 불일치",
    )

    final_counts = {
        name: model.number(count)
        for name, count in as_row(
            season["counts"]
        ).items()
    }

    data.check_counts(role, final_counts)

    for name in data.metrics(role):
        expected = (
            current[name]
            + counts_by_horizon["remaining"][name]
        )

        require(
            math.isclose(
                final_counts[name],
                expected,
                rel_tol=COUNT_TOLERANCE,
                abs_tol=COUNT_TOLERANCE,
            ),
            "시즌 누적 및 잔여 Count 합산 오류",
        )

    verify_derived(
        role,
        final_counts,
        season["derived"],
    )
    verify_intervals(
        final_counts,
        season["intervals"],
    )

    remaining_intervals = as_row(
        as_row(periods["remaining"])["intervals"]
    )
    final_intervals = as_row(season["intervals"])

    for name in data.metrics(role):
        remaining = as_row(
            remaining_intervals[name]
        )
        final = as_row(final_intervals[name])

        for bound in ("lower", "upper"):
            point = remaining[bound]

            if point is None:
                require(
                    final[bound] is None,
                    "미지원 잔여 구간의 시즌 합산 오류",
                )

            else:
                require(
                    math.isclose(
                        model.number(final[bound]),
                        current[name] + model.number(point),
                        rel_tol=COUNT_TOLERANCE,
                        abs_tol=COUNT_TOLERANCE,
                    ),
                    "시즌 전망 구간 합산 오류",
                )

    return counts_by_horizon, final_counts


def audit_selected(
    fitted: Predictor,
    samples: Sequence[data.Sample],
    intervals: Mapping[str, Mapping[str, object]],
    cutoff: date,
) -> dict[str, object]:
    """선정 모델 전체 Validation 출력 및 시즌 전망을 검사한다.

    시즌 전망의 평가 정답은 관측 PBP 누적에 유효 잔여 정답을
    더한 잠정 기록이다. 공식 시즌 최종 기록으로 주장하지 않는다.
    """
    role = fitted.role
    names = data.metrics(role)
    width = len(names)

    season_errors = {
        name: evaluation.Errors()
        for name in names
    }
    season_ratios = {
        name: evaluation.RatioErrors()
        for name, _, _ in ratio_definitions(role)
    }

    projection_status: Counter[str] = Counter()
    verified_samples = 0
    season_evaluated = 0
    scenario_example = None

    for sample in samples:
        player = sample.player

        require(
            player.known_end is None,
            "개발 평가에 실제 또는 별도 종료 시나리오 혼입",
        )

        result = model.format_prediction(
            fitted,
            player,
            intervals,
        )

        _, final_counts = verify_prediction(
            player, result
        )

        verified_samples += 1

        season = as_row(
            result["season_projection"]
        )
        projection_status[str(season["status"])] += 1

        if (
            scenario_example is None
            and player.anchor + timedelta(
                days=max(SCENARIO_DAYS)
            ) < date(player.season + 1, 1, 1)
        ):
            scenario_example = sample

        if final_counts is None:
            continue

        require(
            player.current_counts is not None,
            "시즌 전망에 필요한 현재 누적 누락",
        )

        remaining = data.labels_for(
            sample, cutoff
        )[-width:]

        if any(value is not None for value in remaining):
            season_evaluated += 1

        truth: dict[str, float | None] = {}

        for index, name in enumerate(names):
            observed = remaining[index]
            current = player.current_counts[index]

            actual = (
                None
                if observed is None
                else float(current + observed)
            )
            truth[name] = actual

            if actual is not None:
                season_errors[name].add(
                    actual,
                    final_counts[name],
                    player,
                    as_row(
                        as_row(season["intervals"])[name]
                    ),
                )

        for name, numerator, denominator in (
            ratio_definitions(role)
        ):
            n = truth[numerator]
            d = truth[denominator]

            season_ratios[name].add(
                n / d
                if n is not None and d
                else None,
                d,
                final_counts[numerator],
                final_counts[denominator],
                player,
            )

    require(
        scenario_example is not None,
        "종료 경계 시나리오 검증 표본 부족",
    )

    scenarios = {}

    for days in SCENARIO_DAYS:
        request = dict(scenario_example.request)
        anchor = scenario_example.player.anchor

        request.update({
            "known_end_exclusive": (
                anchor + timedelta(days=days)
            ).isoformat(),
            "known_end_as_of": (
                anchor - timedelta(days=1)
            ).isoformat(),
            "known_end_source":
                "synthetic_contract_validation",
        })

        player = data.parse_input(request)

        result = model.format_prediction(
            fitted,
            player,
            intervals,
        )
        verify_prediction(player, result)

        scenarios[str(days)] = {
            "passed": True,
            "source": "synthetic_not_actual_season_end",
            "interval_supported": False,
        }

    return {
        "verified_validation_predictions":
            verified_samples,
        "projection_statuses":
            dict(projection_status),
        "season_evaluated_samples":
            season_evaluated,
        "season_truth_scope":
            "observed_pbp_cumulative_plus_valid_remaining",
        "official_final_record_evaluated": False,
        "season_record_metrics": {
            name: errors.result()
            for name, errors in season_errors.items()
        },
        "season_ratio_metrics": {
            name: errors.result()
            for name, errors in season_ratios.items()
        },
        "synthetic_known_end_scenarios": scenarios,
        "actual_future_end_used_as_input": False,
    }


def score(
    result: Mapping[str, object],
) -> tuple[float, float]:
    """동일 Train 척도의 Validation 주·보조 지표."""
    return (
        model.number(
            result["normalized_macro_mae"]
        ),
        model.number(
            result["normalized_macro_rmse"]
        ),
    )


def choose_candidate(
    results: Mapping[str, Mapping[str, object]],
) -> tuple[str, dict[str, dict[str, object]]]:
    """기준보다 나쁜 후보의 강제 채택을 방지한다."""
    require(
        BASELINE_NAME in results,
        "선정용 기준 모델 평가 누락",
    )

    baseline_mae, baseline_rmse = score(
        results[BASELINE_NAME]
    )

    decisions = {}
    eligible = [BASELINE_NAME]

    for name, result in results.items():
        require(
            name in SPEC_BY_NAME,
            "등록되지 않은 선정 후보",
        )

        mae, rmse = score(result)

        improves_mae = (
            mae < baseline_mae - SELECTION_TOLERANCE
        )
        nonworse_rmse = (
            rmse <= baseline_rmse + SELECTION_TOLERANCE
        )
        passes = improves_mae and nonworse_rmse

        if name != BASELINE_NAME and passes:
            eligible.append(name)

        decisions[name] = {
            "mae_difference": mae - baseline_mae,
            "rmse_difference": rmse - baseline_rmse,
            "strict_mae_improvement": improves_mae,
            "nonworse_rmse": nonworse_rmse,
            "passes_improvement_gate": passes,
            "eligible": name == BASELINE_NAME or passes,
        }

    selected = min(
        eligible,
        key=lambda name: (
            *score(results[name]),
            CANDIDATE_ORDER[name],
        ),
    )

    return selected, decisions


def develop(
    dataset: data.Dataset,
    role: str,
) -> evaluation.Development:
    """시간순 학습·구간 적합·후보 비교·선정·검증 전체를 수행한다."""
    early = [
        sample for sample in dataset.train
        if sample.player.anchor < data.CALIBRATION_START
    ]
    calibration_rows = [
        sample for sample in dataset.train
        if data.CALIBRATION_START
        <= sample.player.anchor
        < data.VALIDATION_START
    ]

    require(
        bool(early)
        and bool(calibration_rows)
        and bool(dataset.validation)
        and all(
            sample.split == "validation"
            and sample.player.role == role
            and data.VALIDATION_START
            <= sample.player.anchor
            < data.TEST_START
            and sample.player.known_end is None
            for sample in dataset.validation
        ),
        "시간순 개발 자료 또는 Validation 범위 오류",
    )

    start = perf_counter()

    preliminary = fit_family(
        role,
        early,
        data.CALIBRATION_START,
    )
    preliminary_seconds = perf_counter() - start

    start = perf_counter()

    fitted = fit_family(
        role,
        dataset.train,
        data.VALIDATION_START,
    )
    refit_seconds = perf_counter() - start

    intervals_by_candidate = {}
    results: dict[str, dict[str, object]] = {}

    for name, candidate in fitted.items():
        start = perf_counter()

        intervals = evaluation.calibrate(
            preliminary[name],
            calibration_rows,
            data.VALIDATION_START,
        )

        calibration_result = evaluation.evaluate(
            preliminary[name],
            calibration_rows,
            data.VALIDATION_START,
            intervals,
        )

        train_result = evaluation.evaluate(
            candidate,
            dataset.train,
            data.VALIDATION_START,
        )

        validation_result = evaluation.evaluate(
            candidate,
            dataset.validation,
            data.TEST_START,
            intervals,
        )

        results[name] = {
            "depth": candidate.depth,
            "candidate": asdict(
                candidate.specification
            ),
            "evaluation_seconds":
                perf_counter() - start,
            "train_in_sample": train_result,
            "calibration_in_sample_for_interval":
                calibration_result,
            "validation": validation_result,
            "calibration": intervals,
            "tree_output_support": (
                list(candidate.tree.support)
                if candidate.tree is not None else None
            ),
            "preliminary_tree_output_support": (
                list(preliminary[name].tree.support)
                if preliminary[name].tree is not None
                else None
            ),
        }

        intervals_by_candidate[name] = intervals

    baseline = results[BASELINE_NAME]["validation"]
    validation_results = {}

    for name, result in results.items():
        validation = result["validation"]

        require(
            validation["population"]
            == baseline["population"]
            and validation["active_samples"]
            == baseline["active_samples"]
            and validation["supported_output_cells"]
            == baseline["supported_output_cells"]
            and fitted[name].baseline.scales
            == fitted[BASELINE_NAME].baseline.scales,
            "후보 비교 표본 또는 정규화 척도 불일치",
        )

        result["validation_vs_baseline"] = (
            evaluation.compare_records(
                baseline["record_metrics"],
                validation["record_metrics"],
            )
        )

        validation_results[name] = validation

    selected_name, decisions = choose_candidate(
        validation_results
    )

    selected = fitted[selected_name]
    chosen = validation_results[selected_name]
    intervals = intervals_by_candidate[selected_name]

    selected_comparison = results[selected_name][
        "validation_vs_baseline"
    ]

    improved = sum(
        value["mae_difference"] is not None
        and value["mae_difference"] < 0
        for value in selected_comparison.values()
    )
    worsened = sum(
        value["mae_difference"] is not None
        and value["mae_difference"] > 0
        for value in selected_comparison.values()
    )

    common_ratios = compare_ratios(
        fitted,
        dataset.validation,
        role,
        data.TEST_START,
    )

    selected_audit = audit_selected(
        selected,
        dataset.validation,
        intervals,
        data.TEST_START,
    )

    learned_candidates_passed = [
        name
        for name, decision in decisions.items()
        if decision["passes_improvement_gate"]
        and SPEC_BY_NAME[name].mode != "history"
    ]

    selection_status = (
        "baseline_retained"
        if selected_name == BASELINE_NAME
        else "improved_joint_tree"
        if selected.tree is not None
        else "improved_history_predictor"
    )

    return evaluation.Development(
        selected=selected,
        intervals=intervals,
        report={
            "role": role,
            "selected_model": selected_name,
            "selected_candidate": asdict(
                selected.specification
            ),
            "selection_status": selection_status,
            "selected_is_tree":
                selected.tree is not None,
            "learned_candidates_passing_gate":
                learned_candidates_passed,
            "selection_rule":
                settings()["selection_rule"],
            "selection_decisions": decisions,
            "baseline_is_comparison_only": False,
            "selected_beats_baseline":
                decisions[selected_name][
                    "passes_improvement_gate"
                ],
            "selected_vs_baseline_macro_mae_difference":
                score(chosen)[0] - score(baseline)[0],
            "selected_vs_baseline_macro_rmse_difference":
                score(chosen)[1] - score(baseline)[1],
            "selected_record_mae_improved": improved,
            "selected_record_mae_worsened": worsened,
            "record_scales": dict(zip(
                data.output_keys(role),
                selected.baseline.scales,
            )),
            "candidate_count": len(results),
            "operating_predictors": 1,
            "preliminary_family_fit_seconds":
                preliminary_seconds,
            "full_train_family_fit_seconds":
                refit_seconds,
            "early_train_audit": data.audit_samples(
                early,
                role,
                data.CALIBRATION_START,
            ),
            "calibration_audit": data.audit_samples(
                calibration_rows,
                role,
                data.VALIDATION_START,
            ),
            "common_ratio_comparison": common_ratios,
            "selected_output_audit": selected_audit,
            "candidates": results,
            "validation_used_for_selection": True,
            "validation_previously_observed": True,
            "validation_is_final_test": False,
            "final_test_used": False,
            "calibration_transfer": (
                "2023_fit_2024_residuals_to_2023_2024_refit"
            ),
        },
    )