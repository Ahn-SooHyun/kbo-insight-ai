"""M3 동일 표본 평가, 시간순 예측 구간 및 운영 후보 선정."""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
import math
from statistics import mean
from time import perf_counter

from . import data, model
from ...datasets.model_data import require


MIN_CALIBRATION_ROWS = 20
MIN_CALIBRATION_PLAYERS = 5
MIN_CALIBRATION_WEEKS = 3


@dataclass(slots=True)
class Errors:
    """기록별 오차 및 관측된 평가 표본의 누적 집계."""

    n: int = 0
    absolute: float = 0.0
    squared: float = 0.0
    signed: float = 0.0
    actual_total: float = 0.0
    interval_n: int = 0
    interval_hits: int = 0
    interval_width: float = 0.0
    players: set[str] = field(default_factory=set)
    weeks: set[str] = field(default_factory=set)

    def add(
        self,
        actual: float,
        predicted: float,
        player: data.PlayerInput,
        interval: Mapping[str, object] | None = None,
    ) -> None:
        """Count 또는 비율의 실측·예측 오차를 추가한다."""
        error = predicted - actual
        self.n += 1
        self.absolute += abs(error)
        self.squared += error * error
        self.signed += error
        self.actual_total += actual
        self.players.add(player.player_id)
        self.weeks.add(player.anchor.isoformat())

        if interval is None or interval["lower"] is None:
            return

        lower = model.number(interval["lower"])
        upper = model.number(interval["upper"])

        require(
            0 <= lower <= predicted <= upper,
            "구간 순서 또는 점 예측 포함 오류",
        )

        self.interval_n += 1
        self.interval_hits += int(lower <= actual <= upper)
        self.interval_width += upper - lower

    def result(self) -> dict[str, object]:
        """분모가 없는 지표는 0이 아니라 null로 반환한다."""
        return {
            "n": self.n,
            "unique_players": len(self.players),
            "anchor_weeks": len(self.weeks),
            "mae": self.absolute / self.n if self.n else None,
            "rmse": (
                math.sqrt(self.squared / self.n)
                if self.n else None
            ),
            "bias": self.signed / self.n if self.n else None,
            "actual_mean": (
                self.actual_total / self.n if self.n else None
            ),
            "interval_target_coverage": model.INTERVAL_LEVEL,
            "interval_n": self.interval_n,
            "interval_unavailable_n": self.n - self.interval_n,
            "interval_hits": self.interval_hits,
            "interval_coverage": (
                self.interval_hits / self.interval_n
                if self.interval_n else None
            ),
            "interval_mean_width": (
                self.interval_width / self.interval_n
                if self.interval_n else None
            ),
        }


@dataclass(slots=True)
class RatioErrors:
    """정답·예측의 분모를 각각 확인하는 비율 평가 집계."""

    errors: Errors = field(default_factory=Errors)
    truth_valid: int = 0
    truth_unavailable: int = 0
    predicted_zero: int = 0
    actual_denominator_total: float = 0.0
    evaluated_actual_denominator: float = 0.0
    evaluated_predicted_denominator: float = 0.0

    def add(
        self,
        actual: float | None,
        actual_denominator: float | None,
        predicted_numerator: float,
        predicted_denominator: float,
        player: data.PlayerInput,
    ) -> None:
        """정답 분모 0과 예측 분모 0을 구분한다."""
        if actual is None or not actual_denominator:
            self.truth_unavailable += 1
            return

        self.truth_valid += 1
        self.actual_denominator_total += actual_denominator

        if predicted_denominator == 0:
            self.predicted_zero += 1
            return

        self.evaluated_actual_denominator += actual_denominator
        self.evaluated_predicted_denominator += predicted_denominator

        self.errors.add(
            actual,
            predicted_numerator / predicted_denominator,
            player,
        )

    def result(self) -> dict[str, object]:
        """비율 평가 가능 행과 비교에 사용한 분모를 명시한다."""
        result = self.errors.result()
        result.update({
            "truth_valid_n": self.truth_valid,
            "truth_unavailable_n": self.truth_unavailable,
            "prediction_zero_denominator_n": self.predicted_zero,
            "actual_denominator_total": self.actual_denominator_total,
            "evaluated_actual_denominator_total":
                self.evaluated_actual_denominator,
            "evaluated_predicted_denominator_total":
                self.evaluated_predicted_denominator,
        })

        # 비율 자체의 예측 구간을 적합한 것은 아니다.
        for name in tuple(result):
            if name.startswith("interval_"):
                del result[name]

        return result


def calibrate(
    fitted: model.Fitted,
    samples: Sequence[data.Sample],
    cutoff: date,
) -> dict[str, dict[str, object]]:
    """선행 학습에 사용하지 않은 날짜의 절대 잔차를 집계한다.

    동일 선수와 중첩 기간으로 인해 독립·교환 가능성은 보장되지 않는다.
    전체 Train 재적합 모델에 전이하는 경험적 구간으로 해석한다.
    """
    keys = data.output_keys(fitted.role)
    residuals = {key: [] for key in keys}
    players = {key: set() for key in keys}
    weeks = {key: set() for key in keys}

    for sample in samples:
        require(
            data.CALIBRATION_START <= sample.player.anchor < cutoff,
            "구간 적합 표본의 날짜 오류",
        )

        labels = data.labels_for(sample, cutoff)
        predicted = fitted.predict(sample.player)

        for j, key in enumerate(keys):
            actual = labels[j]

            if actual is None:
                continue

            residuals[key].append(abs(actual - predicted[j]))
            players[key].add(sample.player.player_id)
            weeks[key].add(sample.player.anchor.isoformat())

    result = {}

    for key in keys:
        values = sorted(residuals[key])
        n = len(values)
        enough = (
            n >= MIN_CALIBRATION_ROWS
            and len(players[key]) >= MIN_CALIBRATION_PLAYERS
            and len(weeks[key]) >= MIN_CALIBRATION_WEEKS
        )

        radius = None
        rank = None

        if enough:
            rank = min(
                n,
                math.ceil((n + 1) * model.INTERVAL_LEVEL),
            )
            radius = values[rank - 1]

        result[key] = {
            "radius": radius,
            "reason": None if enough else "insufficient_support",
            "target_coverage": model.INTERVAL_LEVEL,
            "n": n,
            "unique_players": len(players[key]),
            "anchor_weeks": len(weeks[key]),
            "quantile_rank": rank,
            "calibration_start":
                data.CALIBRATION_START.isoformat(),
            "calibration_end_exclusive": cutoff.isoformat(),
            "method": "held_out_absolute_residual_quantile",
        }

    return result


def evaluate(
    fitted: model.Fitted,
    samples: Sequence[data.Sample],
    cutoff: date,
    intervals: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """같은 정답 마스크에 Count·구간·파생 비율을 평가한다."""
    role = fitted.role
    names = data.metrics(role)
    keys = data.output_keys(role)
    width = len(names)

    count_errors = {key: Errors() for key in keys}
    ratio_errors = {
        horizon: {
            "hit_rate": RatioErrors(),
            "strikeout_rate": RatioErrors(),
        }
        for horizon in data.HORIZONS
    }
    ratio_states = {
        horizon: Counter()
        for horizon in data.HORIZONS
    }
    group_errors: dict[str, Errors] = {}
    active_samples = 0
    projected_season_samples = 0
    equal_pairs = 0
    equal_gaps: list[float] = []

    for sample in samples:
        player = sample.player
        labels = data.labels_for(sample, cutoff)
        predictions = fitted.predict(player)

        if any(value is not None for value in labels):
            active_samples += 1

        remaining = sample.targets[-1]

        if (
            player.current_counts is not None
            and any(value is not None for value in labels[-width:])
        ):
            projected_season_samples += 1

        for i, horizon in enumerate(data.HORIZONS):
            target = sample.targets[i]
            start = i * width
            truth = labels[start:start + width]
            points = predictions[start:start + width]
            actual = dict(zip(names, truth))
            predicted = dict(zip(names, points))

            if not any(value is not None for value in truth):
                ratio_states[horizon]["masked"] += 1

                for stats in ratio_errors[horizon].values():
                    stats.truth_unavailable += 1

                continue

            ratio_states[horizon][target.ratio_status] += 1

            for j, name in enumerate(names):
                observed = truth[j]

                if observed is None:
                    continue

                key = f"{horizon}.{name}"
                point = points[j]
                interval = None

                if intervals is not None:
                    interval = model.prediction_interval(
                        point,
                        intervals[key],
                    )

                count_errors[key].add(
                    float(observed),
                    point,
                    player,
                    interval,
                )

                scale = fitted.baseline.scales[start + j]

                for group in (
                    f"{horizon}.season.{player.season}",
                    f"{horizon}.week.{player.anchor.isoformat()}",
                    f"{horizon}.team.{player.known_team_id}",
                ):
                    group_errors.setdefault(group, Errors()).add(
                        observed / scale,
                        point / scale,
                        player,
                    )

            for ratio_name, numerator, denominator, stored in (
                (
                    "hit_rate", "hits", "ab",
                    target.hit_rate,
                ),
                (
                    "strikeout_rate", "strikeouts",
                    data.exposure(role),
                    target.strikeout_rate,
                ),
            ):
                n = actual[numerator]
                d = actual[denominator]

                # 7일에는 기존 자료의 별도 비율 정답이 없다.
                # 유효 Count에서 만든 진단값으로만 구분해 제공한다.
                if horizon == "7d":
                    actual_rate = (
                        n / d
                        if n is not None and d
                        else None
                    )
                else:
                    actual_rate = (
                        stored
                        if n is not None and d
                        else None
                    )

                ratio_errors[horizon][ratio_name].add(
                    actual_rate,
                    float(d) if d is not None else None,
                    predicted[numerator],
                    predicted[denominator],
                    player,
                )

            # 실제 종료일은 평가 진단에만 사용한다.
            # 예측 과정의 입력 및 후처리에 전달하지 않는다.
            if (
                horizon != "remaining"
                and remaining.window_end is not None
                and target.window_end is not None
                and target.window_end >= remaining.window_end
            ):
                for j, observed in enumerate(truth):
                    remaining_actual = labels[
                        (len(data.HORIZONS) - 1) * width + j
                    ]

                    if observed is None or remaining_actual is None:
                        continue

                    require(
                        observed == remaining_actual,
                        "실제 시즌 종료 이후 누적 정답 불일치",
                    )

                    equal_pairs += 1
                    equal_gaps.append(
                        abs(points[j] - predictions[-width + j])
                    )

    records = {
        key: errors.result()
        for key, errors in count_errors.items()
    }
    normalized_mae = []
    normalized_rmse = []

    for j, key in enumerate(keys):
        metric = records[key]

        if metric["mae"] is None:
            continue

        scale = fitted.baseline.scales[j]

        normalized_mae.append(
            model.number(metric["mae"]) / scale
        )
        normalized_rmse.append(
            model.number(metric["rmse"]) / scale
        )

    require(bool(normalized_mae), "평가 가능한 Count 정답 없음")

    return {
        "model": fitted.name,
        "population": len(samples),
        "active_samples": active_samples,
        "unique_players": len({
            sample.player.player_id for sample in samples
        }),
        "anchor_weeks": len({
            sample.player.anchor for sample in samples
        }),
        "record_metrics": records,
        "normalized_macro_mae": mean(normalized_mae),
        "normalized_macro_rmse": mean(normalized_rmse),
        "supported_output_cells": len(normalized_mae),
        "ratio_metrics": {
            horizon: {
                "truth_source": (
                    "count_derived_diagnostic"
                    if horizon == "7d"
                    else "stored_ratio_and_valid_denominator"
                ),
                "statuses": dict(ratio_states[horizon]),
                **{
                    key: stats.result()
                    for key, stats in values.items()
                },
            }
            for horizon, values in ratio_errors.items()
        },
        "group_normalized_errors": {
            key: stats.result()
            for key, stats in group_errors.items()
        },
        "season_projection_available_samples":
            projected_season_samples,
        "actual_end_equality_diagnostic": {
            "record_pairs": equal_pairs,
            "prediction_mean_absolute_gap": (
                mean(equal_gaps) if equal_gaps else None
            ),
            "prediction_nonzero_gap_pairs": sum(
                gap > 1e-8 for gap in equal_gaps
            ),
            "actual_end_used_as_feature": False,
        },
        "interval_scope": "marginal_count_only",
        "overlapping_windows_are_independent": False,
        "cutoff_exclusive": cutoff.isoformat(),
    }


@dataclass(slots=True)
class Development:
    """선택된 운영 후보와 모든 비교 평가 및 구간 자료."""

    selected: model.Fitted
    intervals: dict[str, dict[str, object]]
    report: dict[str, object]


def compare_records(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
) -> dict[str, object]:
    """개선·악화를 모두 동일 방향으로 기록한다."""
    result = {}

    for key in baseline:
        old = baseline[key]
        new = candidate[key]

        require(old["n"] == new["n"], "비교 표본 수 불일치")

        changes = {}

        for metric in ("mae", "rmse"):
            a = old[metric]
            b = new[metric]

            changes[f"{metric}_difference"] = (
                None if a is None or b is None else b - a
            )

            changes[f"{metric}_relative_improvement"] = (
                None
                if a is None or b is None or a == 0
                else (a - b) / a
            )

        result[key] = changes

    return result


def develop(
    dataset: data.Dataset,
    role: str,
) -> Development:
    """2023 선행 학습, 2024 구간 적합, 2025 후보 평가를 수행한다."""
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
        bool(early) and bool(calibration_rows)
        and bool(dataset.validation),
        "시간순 개발 자료가 부족합니다",
    )

    results: dict[str, dict[str, object]] = {}
    fitted_candidates: dict[str, model.Fitted] = {}
    intervals_by_candidate = {}

    for depth in (0, *model.CANDIDATE_DEPTHS):
        start = perf_counter()

        preliminary = model.fit(
            role,
            early,
            data.CALIBRATION_START,
            depth,
        )

        calibration = calibrate(
            preliminary,
            calibration_rows,
            data.VALIDATION_START,
        )

        calibration_evaluation = evaluate(
            preliminary,
            calibration_rows,
            data.VALIDATION_START,
            calibration,
        )

        fitted = model.fit(
            role,
            dataset.train,
            data.VALIDATION_START,
            depth,
        )

        fit_seconds = perf_counter() - start
        start = perf_counter()

        train_result = evaluate(
            fitted,
            dataset.train,
            data.VALIDATION_START,
        )

        validation_result = evaluate(
            fitted,
            dataset.validation,
            data.TEST_START,
            calibration,
        )

        results[fitted.name] = {
            "depth": depth,
            "fit_and_calibration_seconds": fit_seconds,
            "evaluation_seconds": perf_counter() - start,
            "train_in_sample": train_result,
            "calibration_in_sample_for_interval":
                calibration_evaluation,
            "validation": validation_result,
            "calibration": calibration,
        }

        intervals_by_candidate[fitted.name] = calibration

        if depth:
            fitted_candidates[fitted.name] = fitted

    baseline = results["history_baseline"]["validation"]

    for name, result in results.items():
        validation = result["validation"]
        require(
            validation["population"] == baseline["population"]
            and validation["active_samples"]
            == baseline["active_samples"],
            "기준·후보 평가 모집단 불일치",
        )

        result["validation_vs_baseline"] = compare_records(
            baseline["record_metrics"],
            validation["record_metrics"],
        )

    def selection_key(name: str) -> tuple[float, float, int]:
        candidate = results[name]
        validation = candidate["validation"]

        return (
            model.number(validation["normalized_macro_mae"]),
            model.number(validation["normalized_macro_rmse"]),
            int(candidate["depth"]),
        )

    # 기준 모델은 비교 대상이다. 운영 예측기는 공동 회귀 후보 중
    # 선택한다. 기준보다 악화했으면 그 사실을 그대로 보고한다.
    selected_name = min(
        fitted_candidates,
        key=selection_key,
    )
    selected = fitted_candidates[selected_name]
    chosen = results[selected_name]["validation"]

    baseline_score = model.number(
        baseline["normalized_macro_mae"]
    )
    chosen_score = model.number(
        chosen["normalized_macro_mae"]
    )

    selected_record_comparison = results[selected_name][
        "validation_vs_baseline"
    ]

    improved = sum(
        value["mae_difference"] is not None
        and value["mae_difference"] < 0
        for value in selected_record_comparison.values()
    )
    worsened = sum(
        value["mae_difference"] is not None
        and value["mae_difference"] > 0
        for value in selected_record_comparison.values()
    )

    return Development(
        selected=selected,
        intervals=intervals_by_candidate[selected_name],
        report={
            "role": role,
            "selected_model": selected_name,
            "selection_rule":
                "joint_tree_macro_normalized_mae_rmse_depth",
            "baseline_is_comparison_only": True,
            "selected_vs_baseline_macro_mae_difference":
                chosen_score - baseline_score,
            "selected_beats_baseline": chosen_score < baseline_score,
            "selected_record_mae_improved": improved,
            "selected_record_mae_worsened": worsened,
            "record_scales": dict(zip(
                data.output_keys(role),
                selected.baseline.scales,
            )),
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
            "candidates": results,
            "validation_used_for_selection": True,
            "validation_is_final_test": False,
            "final_test_used": False,
            "calibration_transfer":
                "2023_fit_2024_residuals_to_2023_2024_refit",
        },
    )