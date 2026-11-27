"""경기 상황 가치 모델의 확률·기대득점·지원 범위를 평가한다."""

from collections import Counter
from collections.abc import Sequence
from math import isfinite, log, log1p, sqrt

from ..models.game_state_value import model as m4
from ..models.game_state_value import value
from ..models.game_state_value.data import Pair


CALIBRATION_BINS = 10


class ProbabilityMetrics:
    """이진 확률 분류의 오차 및 신뢰도 구간을 누적한다."""

    def __init__(self) -> None:
        self.n = 0
        self.loss = 0.0
        self.brier = 0.0
        self.predicted_sum = 0.0
        self.observed_sum = 0
        self.bins = [
            [0, 0.0, 0]
            for _ in range(CALIBRATION_BINS)
        ]

    def add(self, probability: float, target: int) -> None:
        """한 상태의 예측과 승리 여부를 집계한다."""
        m4.require(
            isfinite(probability)
            and 0 <= probability <= 1
            and type(target) is int
            and target in (0, 1),
            "승리 확률 평가 입력 오류",
        )

        clipped = min(
            1.0 - m4.EPSILON,
            max(m4.EPSILON, probability),
        )

        # 정답과 완전히 일치하는 확정 종료 확률의 손실은 0이다.
        if probability == target:
            loss = 0.0
        else:
            loss = -log(clipped) if target == 1 else -log1p(-clipped)

        self.n += 1
        self.loss += loss
        self.brier += (probability - target) ** 2
        self.predicted_sum += probability
        self.observed_sum += target

        index = min(
            CALIBRATION_BINS - 1,
            int(probability * CALIBRATION_BINS),
        )

        self.bins[index][0] += 1
        self.bins[index][1] += probability
        self.bins[index][2] += target

    def result(self) -> dict[str, object]:
        """표본 수, Log loss, Brier 및 구간별 보정 결과를 반환한다."""
        calibration = []
        ece = 0.0

        for index, (n, predicted, observed) in enumerate(self.bins):
            mean = predicted / n if n else None
            rate = observed / n if n else None

            if n:
                ece += n * abs(mean - rate)

            calibration.append({
                "lower": index / CALIBRATION_BINS,
                "upper": (index + 1) / CALIBRATION_BINS,
                "upper_inclusive": index == CALIBRATION_BINS - 1,
                "count": n,
                "mean_probability": mean,
                "observed_win_rate": rate,
                "observed_wins": observed,
            })

        return {
            "count": self.n,
            "log_loss": self.loss / self.n if self.n else None,
            "brier": self.brier / self.n if self.n else None,
            "mean_probability": (
                self.predicted_sum / self.n if self.n else None
            ),
            "observed_win_rate": (
                self.observed_sum / self.n if self.n else None
            ),
            "ece": ece / self.n if self.n else None,
            "calibration": calibration,
        }


class NumericMetrics:
    """기대득점 오차의 MAE·RMSE를 누적한다."""

    def __init__(self) -> None:
        self.n = 0
        self.absolute_error = 0.0
        self.squared_error = 0.0
        self.predicted_sum = 0.0
        self.actual_sum = 0.0

    def add(self, prediction: float, target: int) -> None:
        """한 상태의 기대득점과 잔여 실득점을 집계한다."""
        m4.require(
            isfinite(prediction)
            and prediction >= 0
            and type(target) is int
            and target >= 0,
            "기대득점 평가 입력 오류",
        )

        error = prediction - target

        self.n += 1
        self.absolute_error += abs(error)
        self.squared_error += error * error
        self.predicted_sum += prediction
        self.actual_sum += target

    def result(self) -> dict[str, object]:
        """잔여 득점의 오차 및 평균을 반환한다."""
        return {
            "count": self.n,
            "mae": (
                self.absolute_error / self.n if self.n else None
            ),
            "rmse": (
                sqrt(self.squared_error / self.n)
                if self.n else None
            ),
            "mean_prediction": (
                self.predicted_sum / self.n if self.n else None
            ),
            "mean_actual": (
                self.actual_sum / self.n if self.n else None
            ),
        }


def evaluate_win(
    predictor: m4.Predictor,
    rows: Sequence[m4.Observation],
) -> dict[str, object]:
    """동일 상태 표본에서 승리 확률 후보를 평가한다."""
    groups = {
        name: ProbabilityMetrics()
        for name in (
            "all",
            "nonterminal",
            "confirmed_final",
            "pre",
            "post",
            "three_outs_nonterminal",
        )
    }

    methods: Counter[str] = Counter()

    for row in rows:
        probability, method, _ = m4.predict_win(
            predictor,
            row.state,
            row.final,
        )

        names = ["all", row.state.phase]

        if row.final is None:
            names.append("nonterminal")

            if row.state.outs == 3:
                names.append("three_outs_nonterminal")
        else:
            names.append("confirmed_final")

        for name in names:
            groups[name].add(probability, row.win)

        methods[method] += 1

    m4.require(
        groups["all"].n == len(rows),
        "승리 확률 평가 분모 불일치",
    )

    return {
        "method": predictor["name"],
        "groups": {
            name: metric.result()
            for name, metric in groups.items()
        },
        "prediction_methods": dict(methods),
    }


def evaluate_re(
    table: m4.StatTable,
    rows: Sequence[m4.Observation],
) -> dict[str, object]:
    """전체 평균 기준과 조건부 RE 통계를 같은 상태에서 비교한다."""
    groups = (
        "all", "nonterminal", "half_ended", "pre", "post"
    )

    results = {
        method: {
            name: NumericMetrics()
            for name in groups
        }
        for method in ("global_mean", "conditional_mean")
    }

    backoffs: Counter[str] = Counter()

    for row in rows:
        ended = row.state.outs == 3 or row.final is not None

        names = [
            "all",
            row.state.phase,
            "half_ended" if ended else "nonterminal",
        ]

        for method, global_only in (
            ("global_mean", True),
            ("conditional_mean", False),
        ):
            predicted, level, _ = value.predict_re(
                table,
                row.state,
                row.final,
                global_only=global_only,
            )

            for name in names:
                results[method][name].add(predicted, row.remaining)

            if not global_only:
                backoffs[level] += 1

    return {
        "candidates": {
            method: {
                name: metric.result()
                for name, metric in metrics.items()
            }
            for method, metrics in results.items()
        },
        "conditional_backoff_counts": dict(backoffs),
    }


def evaluate_pairs(
    asset: value.ValueModel,
    pairs: Sequence[Pair],
) -> dict[str, object]:
    """관측 타석 가치와 상황 중요도의 적용 범위를 평가한다."""
    supported = 0
    importance_sum = 0.0
    absolute_wpa_sum = 0.0
    wpa_sum = 0.0
    re_change_sum = 0.0
    run_value_sum = 0.0
    actual_runs_sum = 0

    unsupported: Counter[str] = Counter()
    importance_groups: dict[str, list[float]] = {}

    for pair in pairs:
        result = value.plate_value(
            asset,
            pair.pre.state,
            pair.post.state,
            pair.final,
            with_importance=False,
        )

        wpa = result["wpa"]
        change = result["remaining_expected_runs_change"]
        run_value = result["run_value"]

        m4.require(
            -1 <= wpa <= 1
            and isfinite(change)
            and isfinite(run_value)
            and result["actual_runs"] == pair.actual_runs,
            "타석 가치 출력 계약 오류",
        )

        m4.require(
            abs(run_value - (pair.actual_runs + change)) < 1e-12,
            "득점 포함 타석 가치 산식 오류",
        )

        wpa_sum += wpa
        absolute_wpa_sum += abs(wpa)
        re_change_sum += change
        run_value_sum += run_value
        actual_runs_sum += pair.actual_runs

        importance = value.importance(
            asset,
            pair.pre.state,
            details=False,
        )

        level = importance["value"]

        if level is None:
            unsupported[str(importance["reason"])] += 1
            continue

        m4.require(
            isfinite(level) and level >= 0,
            "상황 중요도 범위 오류",
        )

        supported += 1
        importance_sum += level

        group = (
            "below_0.5" if level < 0.5
            else "0.5_to_1" if level < 1.0
            else "1_to_2" if level < 2.0
            else "at_least_2"
        )

        entry = importance_groups.setdefault(group, [0, 0.0, 0.0])

        entry[0] += 1
        entry[1] += level
        entry[2] += abs(wpa)

    n = len(pairs)

    m4.require(
        supported + sum(unsupported.values()) == n,
        "상황 중요도 지원·미지원 건수 불일치",
    )

    return {
        "pairs": n,
        "mean_wpa": wpa_sum / n if n else None,
        "mean_absolute_wpa": (
            absolute_wpa_sum / n if n else None
        ),
        "mean_remaining_expected_runs_change": (
            re_change_sum / n if n else None
        ),
        "mean_run_value": run_value_sum / n if n else None,
        "actual_runs": actual_runs_sum,
        "importance": {
            "supported": supported,
            "unsupported": n - supported,
            "coverage": supported / n if n else None,
            "mean_supported_value": (
                importance_sum / supported if supported else None
            ),
            "unsupported_reasons": dict(unsupported),
            "groups": {
                name: {
                    "count": int(entry[0]),
                    "mean_importance": entry[1] / entry[0],
                    "mean_observed_absolute_wpa": entry[2] / entry[0],
                }
                for name, entry in importance_groups.items()
            },
        },
    }


def training_summary(
    rows: Sequence[m4.Observation],
) -> dict[str, object]:
    """최종 Test를 읽지 않고 Train의 상태·정답 분포를 요약한다."""
    states: Counter[str] = Counter()
    remaining: Counter[int] = Counter()
    wins = 0

    for row in rows:
        state = row.state

        states[
            f"{state.half}|{state.outs}|{state.mask}"
        ] += 1

        remaining[row.remaining] += 1
        wins += row.win

    n = len(rows)

    return {
        "states": n,
        "wins": wins,
        "win_rate": wins / n if n else None,
        "state_distribution": dict(states),
        "remaining_runs_distribution": {
            str(key): count
            for key, count in sorted(remaining.items())
        },
    }