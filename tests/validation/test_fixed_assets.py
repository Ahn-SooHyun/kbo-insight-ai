"""고정 최종 평가의 저장 예측·정답 기반 점수 계산."""

from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from datetime import timedelta
import json
from math import fsum, isfinite
from pathlib import Path

from . import game_outcome as e1
from . import game_state_value as e4
from . import pitch_outcome as e5
from .fixed_assets import require, unique_json_object
from ..datasets.published_sources import as_row
from ..models.game_outcome import model as m1
from ..models.plate_appearance import model as m2
from ..models.plate_appearance import probability as p2
from ..models.game_state_value import model as m4
from ..models.pitch_outcome import model as m5
from ..models.player_performance import data as d3
from ..models.player_performance import model as m3
from ..models.player_performance import evaluation as e3


METHODS = ("operating", "baseline")
M5_METHODS = (
    "operating",
    "frequency",
    "conditional_frequency",
)

TOLERANCE = 1e-12


def read_rows(path: Path) -> Iterator[dict[str, object]]:
    """저장 JSONL의 각 행을 중복 키 없이 읽는다."""
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            require(bool(line.strip()), "빈 JSONL 행")

            yield as_row(
                json.loads(
                    line,
                    object_pairs_hook=unique_json_object,
                )
            )


def paired_rows(
    predictions: Path,
    targets: Path,
) -> Iterator[tuple[dict[str, object], dict[str, object]]]:
    """동일한 고유 표본 키의 예측과 정답만 연결한다."""
    previous: str | None = None

    for prediction, target in zip(
        read_rows(predictions),
        read_rows(targets),
        strict=True,
    ):
        key = prediction["sample_id"]

        require(
            isinstance(key, str)
            and bool(key)
            and key == target["sample_id"],
            "예측·정답 표본 키 불일치",
        )

        require(
            previous is None or key > previous,
            "표본 키 중복 또는 저장 순서 오류",
        )

        previous = key

        yield prediction, target


def difference(
    operating: float | None,
    baseline: float | None,
    *,
    direction: str = "lower",
    target: float | None = None,
) -> dict[str, object]:
    """공통 분모 지표의 차이와 개선 방향을 구분한다."""
    require(
        direction in {"lower", "higher", "target"},
        "평가 방향 오류",
    )

    if operating is None or baseline is None:
        return {
            "operating": operating,
            "baseline": baseline,
            "difference": None,
            "direction": direction,
            "judgment": "unavailable",
        }

    delta = operating - baseline

    if direction == "lower":
        improvement = -delta
    elif direction == "higher":
        improvement = delta
    else:
        require(target is not None, "목표 지표 기준 누락")
        improvement = (
            abs(baseline - target)
            - abs(operating - target)
        )

    return {
        "operating": operating,
        "baseline": baseline,
        "difference": delta,
        "direction": direction,
        "judgment": (
            "equal"
            if abs(improvement) <= TOLERANCE
            else "improved"
            if improvement > 0
            else "worsened"
        ),
    }


def compare_metrics(
    operating: Mapping[str, object],
    baseline: Mapping[str, object],
    names: Sequence[str],
    *,
    higher: Sequence[str] = (),
) -> dict[str, object]:
    """비교할 지표를 사전 지정하고 우열을 계산한다."""
    return {
        name: difference(
            operating[name],
            baseline[name],
            direction="higher" if name in higher else "lower",
        )
        for name in names
    }


def population(
    total: int,
    operating: int,
    baseline: int,
    common: int,
) -> dict[str, int]:
    """전체·개별 유효·공통 유효·제외 분모를 계산한다."""
    require(
        0 <= common <= min(operating, baseline)
        and max(operating, baseline) <= total,
        "평가 공통 분모 오류",
    )

    return {
        "total": total,
        "operating_valid": operating,
        "baseline_valid": baseline,
        "common_valid": common,
        "excluded_from_common": total - common,
    }


def score_m1(
    predictions: Path,
    targets: Path,
) -> dict[str, object]:
    """기존 M1 득점·승무패 지표를 저장 결과에 적용한다."""
    games: list[m1.Game] = []

    distributions = {name: [] for name in METHODS}
    outcomes = {name: [] for name in METHODS}

    tail_max = 0.0
    mean_error_max = 0.0
    support = Counter()

    for row, truth in paired_rows(predictions, targets):
        actual = as_row(truth["target"])

        runs = (
            m1.count(actual["home"], "home"),
            m1.count(actual["away"], "away"),
        )

        require(
            actual["result"] == m1.score_outcome(*runs),
            "M1 득점 및 결과 정답 불일치",
        )

        games.append(
            m1.Game(
                row["sample_id"],
                row["sample_date"],
                as_row(row["input"]),
                runs,
                actual["result"],
            )
        )

        support[actual["result"]] += 1

        for name in METHODS:
            forecast = as_row(row[name])

            home = tuple(forecast["home_pmf"])
            away = tuple(forecast["away_pmf"])

            m1.check_pmf(home)
            m1.check_pmf(away)

            probabilities = as_row(
                forecast["result_probabilities"]
            )

            require(
                set(probabilities) == set(m1.OUTCOMES),
                "M1 결과 범주 불일치",
            )

            m1.check_pmf(
                tuple(probabilities[key] for key in m1.OUTCOMES)
            )

            distributions[name].append((home, away))
            outcomes[name].append(probabilities)

        tail = as_row(as_row(row["operating"])["tail"])

        tail_max = max(
            tail_max,
            tail["omitted_joint_mass_bound"],
        )

        mean_error_max = max(
            mean_error_max,
            tail["expected_runs_error_bound_per_side"],
        )

    require(bool(games), "M1 최종 평가 표본 없음")

    results = {
        name: {
            "scores": e1.score_metrics(
                games, distributions[name]
            ),
            "outcomes": e1.result_metrics(
                games, outcomes[name]
            ),
        }
        for name in METHODS
    }

    op = results["operating"]
    base = results["baseline"]
    count = len(games)

    return {
        "population": population(count, count, count, count),
        "results": results,
        "actual_class_support": dict(support),
        "tail_diagnostics": {
            "max_omitted_joint_mass_bound": tail_max,
            "max_expected_runs_error_bound_per_side": mean_error_max,
        },
        "comparison": {
            "scores": {
                side: compare_metrics(
                    op["scores"][side],
                    base["scores"][side],
                    ("mae", "rmse", "crps"),
                )
                for side in ("home", "away", "pooled")
            },
            "outcomes": compare_metrics(
                op["outcomes"],
                base["outcomes"],
                ("log_loss", "brier"),
            ),
        },
    }


def score_m2(
    predictions: Path,
    targets: Path,
    baseline_asset: Mapping[str, object],
) -> dict[str, object]:
    """M2 고정 규칙과 확률 지표를 동일 타석에서 비교한다."""
    samples: list[m2.Sample] = []

    probabilities = {
        name: [] for name in METHODS
    }

    for row, truth in paired_rows(predictions, targets):
        target = m2.target_class(truth["target"])

        inputs = as_row(row["input"])

        m2.validate_inputs(inputs)

        samples.append(
            m2.Sample(
                sample_id=row["sample_id"],
                game_id=row["game_id"],
                day=row["sample_date"],
                inputs=inputs,
                target=target,
            )
        )

        for name in METHODS:
            values = row[name]

            m2.check_probabilities(values)

            probabilities[name].append(values)

    require(bool(samples), "M2 최종 평가 표본 없음")

    results = {
        name: p2.score_predictions(
            samples,
            probabilities[name],
            require_rules=False,
        )
        for name in METHODS
    }

    counts = Counter(sample.target for sample in samples)

    training = dict(
        zip(
            m2.CLASSES,
            baseline_asset["train_counts"],
            strict=True,
        )
    )

    for name in METHODS:
        require(
            results[name]["forbidden_nonzero_samples"] == 0,
            "M2 금지 범주의 양수 예측 확률 발견",
        )

    n = len(samples)

    return {
        "population": population(n, n, n, n),
        "results": results,
        "baseline_name": baseline_asset["name"],
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
        "rule_target_conflicts": results[
            "operating"
        ]["forbidden_target_count"],
        "contract_valid": (
            results["operating"]["forbidden_target_count"] == 0
            and results["baseline"]["forbidden_target_count"] == 0
        ),
        "comparison": compare_metrics(
            results["operating"],
            results["baseline"],
            (
                "log_loss", "brier",
                "macro_f1", "macro_classwise_ece",
            ),
            higher=("macro_f1",),
        ),
    }


def score_m4(
    predictions: Path,
    targets: Path,
) -> dict[str, object]:
    """공격팀 승리·잔여 득점과 종료 상태의 분모를 구분한다."""
    groups = (
        "all",
        "pre",
        "post",
        "nonterminal",
        "confirmed_final",
        "three_outs_nonterminal",
    )

    win = {
        name: {
            group: e4.ProbabilityMetrics()
            for group in groups
        }
        for name in METHODS
    }

    remaining = {
        name: {
            group: e4.NumericMetrics()
            for group in groups
        }
        for name in METHODS
    }

    support = {
        name: Counter()
        for name in METHODS
    }

    n = 0

    for row, truth in paired_rows(predictions, targets):
        state = m4.parse_state(as_row(row["input"]))

        actual_win = truth["win"]
        actual_runs = truth["remaining"]
        confirmed_final = truth["confirmed_final"]

        require(
            type(confirmed_final) is bool,
            "M4 종료 상태 플래그 오류",
        )

        names = ["all", state.phase]

        if confirmed_final:
            require(
                state.phase == "post"
                and actual_win == int(state.difference > 0)
                and actual_runs == 0,
                "M4 확정 종료 상태 불일치",
            )

            names.append("confirmed_final")

        else:
            names.append("nonterminal")

            if state.outs == 3:
                names.append("three_outs_nonterminal")

        for method in METHODS:
            forecast = as_row(row[method])

            probability, win_method, win_support = forecast["win"]
            expected_runs, re_method, re_support = forecast["re"]

            # 종료가 확인된 post 경계는 별도 결정론적 출력이다.
            # 미종료 모델의 성능은 nonterminal에서 따로 집계한다.
            if confirmed_final:
                probability = float(state.difference > 0)
                expected_runs = 0.0
                win_method = "confirmed_final"
                re_method = "half_ended"
                win_support = None
                re_support = None

            support[method][f"win:{win_method}"] += 1
            support[method][f"re:{re_method}"] += 1

            if win_support is not None:
                support[method][
                    "win:training_support_total"
                ] += win_support

            if re_support is not None:
                support[method][
                    "re:training_support_total"
                ] += re_support

            for group in names:
                win[method][group].add(
                    probability, actual_win
                )

                remaining[method][group].add(
                    expected_runs, actual_runs
                )

        n += 1

    require(n > 0, "M4 최종 평가 표본 없음")

    results = {
        name: {
            "win": {
                group: metric.result()
                for group, metric in win[name].items()
            },
            "remaining": {
                group: metric.result()
                for group, metric in remaining[name].items()
            },
            "support": dict(support[name]),
        }
        for name in METHODS
    }

    return {
        "population": population(n, n, n, n),
        "results": results,
        "comparison": {
            output: {
                group: compare_metrics(
                    results["operating"][output][group],
                    results["baseline"][output][group],
                    metrics,
                )
                for group in groups
            }
            for output, metrics in (
                ("win", ("log_loss", "brier", "ece")),
                ("remaining", ("mae", "rmse")),
            )
        },
        "terminal_policy": (
            "observed_final_post_only; "
            "nonterminal_inference_reported_separately"
        ),
    }


def score_m5(
    predictions: Path,
    targets: Path,
    baseline_asset: Mapping[str, object],
) -> dict[str, object]:
    """M5 공동과 두 주변 확률축을 독립적으로 평가한다."""
    labels = {
        "joint": m5.CLASSES,
        "pitch_type": m5.TYPES,
        "pitch_result": m5.RESULTS,
    }

    train_counts = e5.axis_counts(
        baseline_asset["train_counts"]
    )

    scores = {
        name: {
            axis: e5.Score(
                axis_labels,
                train_counts[axis],
            )
            for axis, axis_labels in labels.items()
        }
        for name in M5_METHODS
    }

    n = 0

    for row, truth in paired_rows(predictions, targets):
        target = truth["target"]

        require(
            type(target) is int
            and 0 <= target < len(m5.CLASSES),
            "M5 공동 정답 범주 오류",
        )

        for name in M5_METHODS:
            probabilities = row[name]

            m5.check_probabilities(probabilities)

            pitch, result = m5.marginalize(probabilities)

            scores[name]["joint"].add(target, probabilities)

            scores[name]["pitch_type"].add(
                target // m5.N_RESULTS,
                pitch,
            )

            scores[name]["pitch_result"].add(
                target % m5.N_RESULTS,
                result,
            )

        n += 1

    require(n > 0, "M5 최종 평가 표본 없음")

    results: dict[str, object] = {}

    for name, axes in scores.items():
        axis_reports = {}

        for axis, metric in axes.items():
            report = metric.report()

            for values in report["classes"].values():
                values["test_support"] = values.pop(
                    "validation_support"
                )

            report["macro_classwise_ece"] = fsum(
                values["one_vs_rest_ece"]
                for values in report["classes"].values()
            ) / len(labels[axis])

            axis_reports[axis] = report

        results[name] = axis_reports

    return {
        "population": population(n, n, n, n),
        "results": results,
        "comparison": {
            baseline: {
                axis: compare_metrics(
                    results["operating"][axis],
                    results[baseline][axis],
                    (
                        "log_loss",
                        "brier",
                        "top_label_ece",
                        "macro_classwise_ece",
                    ),
                )
                for axis in labels
            }
            for baseline in (
                "frequency",
                "conditional_frequency",
            )
        },
    }


def parse_final_target(
    row: Mapping[str, object],
    player: d3.PlayerInput,
) -> d3.Target:
    """개발 로더의 Test 차단을 유지하고 최종 정답만 별도로 검사한다."""
    role = player.role
    names = d3.metrics(role)
    horizon = row["horizon"]

    require(
        horizon in d3.HORIZONS,
        "M3 기간 계약 불일치",
    )

    require(
        row["sample_id"] == player.sample_id
        and d3.day(row["window_start"]) == player.anchor
        and d3.TEST_START <= player.anchor < d3.OBSERVATION_END,
        "M3 최종 정답 키 또는 기준일 오류",
    )

    end = (
        d3.day(row["window_end"])
        if row["window_end"] is not None
        else None
    )

    ready = (
        d3.day(row["ready_date"])
        if row["ready_date"] is not None
        else None
    )

    observed_end = d3.day(
        row["observed_through_exclusive"]
    )

    if horizon != "remaining":
        require(
            end == player.anchor + timedelta(
                days=7 if horizon == "7d" else 28
            ),
            "M3 고정 기간 종료일 불일치",
        )

    require(
        player.anchor <= observed_end <= d3.OBSERVATION_END
        and (end is None or player.anchor < end)
        and (end is None or observed_end <= end)
        and (ready is None or ready == end),
        "M3 정답 관측 기간 오류",
    )

    status = row["status"]

    require(
        status in d3.ALL_STATUSES,
        "M3 정답 상태 오류",
    )

    usable = (
        status in d3.VALID_STATUSES
        and end is not None
        and ready is not None
        and ready <= d3.OBSERVATION_END
    )

    require(
        type(row["usable"]) is int
        and row["usable"] in (0, 1)
        and bool(row["usable"]) == usable,
        "M3 최종 정답 마스크 불일치",
    )

    counts = tuple(
        d3.integer(
            row[f"target_{name}"],
            f"target_{name}",
            nullable=True,
        )
        for name in names
    )

    risks = tuple(
        int(d3.integer(row[name], name))
        for name in d3.RISK_FIELDS
    )

    risk = dict(zip(d3.RISK_FIELDS, risks))

    if usable:
        require(
            any(value is not None for value in counts),
            "M3 유효 정답의 Count 누락",
        )

        d3.check_counts(role, dict(zip(names, counts)))

        require(
            not risk["missing_games"]
            and not risk["team_risky_games"]
            and not risk["player_rejected"],
            "M3 품질 위험이 있는 정답 사용",
        )

        for name, count in zip(names, counts):
            if count is not None:
                require(
                    count == row[f"observed_{name}"],
                    "M3 관측 Count와 정답 불일치",
                )

        if status == "zero_observed":
            require(
                all(value in (None, 0) for value in counts)
                and not risk["league_risky_games"]
                and not risk["league_identity_conflicts"]
                and not risk["league_rejected_rows"],
                "M3 실제 0 정답의 관측 근거 부족",
            )

    else:
        require(
            all(value is None for value in counts),
            "M3 미관측 정답에 Count가 남아 있음",
        )

    actual = dict(zip(names, counts))

    ratio_status = (
        "not_requested"
        if horizon == "7d"
        else "unavailable"
        if not usable
        else "valid"
        if actual["ab"] and actual[d3.exposure(role)]
        else "zero_denominator"
    )

    require(
        ratio_status == row["ratio_status"],
        "M3 정답 비율 상태 불일치",
    )

    rates = []

    for name, numerator, denominator in (
        ("hit_rate", "hits", "ab"),
        ("strikeout_rate", "strikeouts", d3.exposure(role)),
    ):
        stored = row[f"target_{name}"]
        n = actual[numerator]
        d = actual[denominator]

        if horizon == "7d" or not usable or not d or n is None:
            require(stored is None, "M3 비율 마스크 불일치")
            rates.append(None)
            continue

        require(
            type(stored) in (int, float)
            and isfinite(stored)
            and abs(stored - n / d) <= 1e-10,
            "M3 저장 비율과 유효 분모 불일치",
        )

        rates.append(float(stored))

    return d3.Target(
        horizon=horizon,
        window_end=end,
        observed_end=observed_end,
        ready=ready,
        status=status,
        usable=usable,
        ratio_status=ratio_status,
        counts=counts,
        hit_rate=rates[0],
        strikeout_rate=rates[1],
        risks=risks,
    )


def score_m3(
    predictions: Path,
    targets: Path,
    role: str,
    baseline_asset: Mapping[str, object],
) -> dict[str, object]:
    """역할·기간·기록별 Count와 비율 및 구간을 공통 표본에서 비교한다."""
    names = d3.metrics(role)
    keys = d3.output_keys(role)

    count_errors = {
        key: {
            name: e3.Errors()
            for name in METHODS
        }
        for key in keys
    }

    interval_common = {
        key: {
            name: e3.Errors()
            for name in METHODS
        }
        for key in keys
    }

    ratios = {
        f"{horizon}.{rate}": {
            name: e3.Errors()
            for name in METHODS
        }
        for horizon in d3.HORIZONS
        for rate in ("hit_rate", "strikeout_rate")
    }

    ratio_support = {
        key: Counter()
        for key in ratios
    }

    states = {
        horizon: Counter()
        for horizon in d3.HORIZONS
    }

    exclusions = {
        key: Counter()
        for key in keys
    }

    interval_exclusions = {
        key: {
            name: Counter()
            for name in METHODS
        }
        for key in keys
    }

    scales = as_row(baseline_asset["scales"])

    players: set[str] = set()
    weeks: set[str] = set()

    n = 0

    for row, truth in paired_rows(predictions, targets):
        player = d3.parse_input(row["input"])

        require(
            player.role == role,
            "M3 평가 역할 불일치",
        )

        players.add(player.player_id)
        weeks.add(player.anchor.isoformat())

        raw_targets = as_row(truth["targets"])

        parsed = tuple(
            parse_final_target(
                as_row(raw_targets[horizon]),
                player,
            )
            for horizon in d3.HORIZONS
        )

        d3.verify_target_relationships(parsed)

        season_end = (
            d3.day(truth["season_end_exclusive"])
            if truth["season_end_exclusive"] is not None
            else None
        )

        require(
            parsed[-1].window_end == season_end,
            "M3 공식 시즌 완료 근거 불일치",
        )

        for target in parsed:
            horizon = target.horizon

            states[horizon][target.status] += 1

            actual = dict(zip(names, target.counts))

            forecasts = {
                name: as_row(
                    as_row(row[name])[horizon]
                )
                for name in METHODS
            }

            points = {
                name: as_row(forecasts[name]["counts"])
                for name in METHODS
            }

            for name in METHODS:
                d3.check_counts(role, points[name])

                require(
                    set(points[name]) == set(names)
                    and all(
                        type(value) in (int, float)
                        and isfinite(value)
                        and value >= 0
                        for value in points[name].values()
                    ),
                    "M3 Count 예측 오류",
                )

            for field in names:
                key = f"{horizon}.{field}"
                observed = actual[field]

                if observed is None:
                    reason = (
                        target.status
                        if not target.usable
                        else "record_unavailable"
                    )
                    exclusions[key][reason] += 1
                    continue

                intervals = {
                    name: as_row(
                        as_row(forecasts[name]["intervals"])[field]
                    )
                    for name in METHODS
                }

                for name in METHODS:
                    interval = intervals[name]

                    count_errors[key][name].add(
                        observed,
                        points[name][field],
                        player,
                        interval,
                    )

                    if interval["lower"] is None:
                        interval_exclusions[key][name][
                            str(interval.get("reason"))
                        ] += 1

                if all(
                    intervals[name]["lower"] is not None
                    for name in METHODS
                ):
                    for name in METHODS:
                        interval_common[key][name].add(
                            observed,
                            points[name][field],
                            player,
                            intervals[name],
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
                ledger = ratio_support[key]

                actual_n = actual[numerator]
                actual_d = actual[denominator]

                if actual_n is None or not actual_d:
                    ledger[
                        "zero_truth_denominator"
                        if actual_d == 0
                        else "truth_unavailable"
                    ] += 1
                    continue

                ledger["truth_valid"] += 1
                ledger["truth_denominator_total"] += actual_d

                valid = {}

                for name in METHODS:
                    valid[name] = points[name][denominator] > 0

                    ledger[
                        f"{name}_valid"
                        if valid[name]
                        else f"{name}_zero_denominator"
                    ] += 1

                if not all(valid.values()):
                    ledger["excluded_from_common"] += 1
                    continue

                ledger["common"] += 1
                ledger["common_actual_denominator_total"] += actual_d

                for name in METHODS:
                    predicted_d = points[name][denominator]

                    ratios[key][name].add(
                        actual_n / actual_d,
                        points[name][numerator] / predicted_d,
                        player,
                    )

                    ledger[
                        f"{name}_common_predicted_denominator_total"
                    ] += predicted_d

        n += 1

    require(n > 0, "M3 최종 평가 모집단 없음")

    record_results = {}
    normalized = {
        name: {"mae": [], "rmse": []}
        for name in METHODS
    }

    for key, metrics in count_errors.items():
        values = {
            name: metrics[name].result()
            for name in METHODS
        }

        common_intervals = {
            name: interval_common[key][name].result()
            for name in METHODS
        }

        count = values["operating"]["n"]

        require(
            count == values["baseline"]["n"]
            and count + sum(exclusions[key].values()) == n,
            "M3 Count 공통 분모 불일치",
        )

        scale = scales[key]

        require(
            type(scale) in (int, float)
            and isfinite(scale)
            and scale > 0,
            "M3 Train 정규화 척도 오류",
        )

        if count:
            for name in METHODS:
                for metric in ("mae", "rmse"):
                    normalized[name][metric].append(
                        values[name][metric] / scale
                    )

        record_results[key] = {
            "population": population(n, count, count, count),
            "excluded_reasons": dict(exclusions[key]),
            "results": values,
            "comparison": compare_metrics(
                values["operating"],
                values["baseline"],
                ("mae", "rmse"),
            ),
            "interval_common": common_intervals,
            "interval_exclusions": {
                name: dict(interval_exclusions[key][name])
                for name in METHODS
            },
            "interval_comparison": {
                "coverage": difference(
                    common_intervals["operating"]["interval_coverage"],
                    common_intervals["baseline"]["interval_coverage"],
                    direction="target",
                    target=m3.INTERVAL_LEVEL,
                ),
                "width": difference(
                    common_intervals["operating"]["interval_mean_width"],
                    common_intervals["baseline"]["interval_mean_width"],
                ),
            },
        }

    ratio_results = {}

    for key, metrics in ratios.items():
        ledger = ratio_support[key]

        values = {
            name: metric.result()
            for name, metric in metrics.items()
        }

        ratio_results[key] = {
            "population": population(
                n,
                ledger["operating_valid"],
                ledger["baseline_valid"],
                ledger["common"],
            ),
            "support": dict(ledger),
            "truth_source": (
                "count_derived_diagnostic"
                if key.startswith("7d.")
                else "stored_valid_ratio"
            ),
            "results": values,
            "comparison": compare_metrics(
                values["operating"],
                values["baseline"],
                ("mae",),
            ),
        }

    macro = {
        name: {
            metric: (
                fsum(values) / len(values)
                if values else None
            )
            for metric, values in normalized[name].items()
        }
        for name in METHODS
    }

    return {
        "role": role,
        "population": n,
        "unique_players": len(players),
        "anchor_weeks": len(weeks),
        "target_status": {
            horizon: dict(values)
            for horizon, values in states.items()
        },
        "records": record_results,
        "ratios": ratio_results,
        "train_normalized_macro": macro,
        "macro_comparison": compare_metrics(
            macro["operating"],
            macro["baseline"],
            ("mae", "rmse"),
        ),
        "interval_target_coverage": m3.INTERVAL_LEVEL,
        "supported_output_cells": len(
            normalized["operating"]["mae"]
        ),
    }


def score(
    name: str,
    predictions: Path,
    targets: Path,
    baseline_asset: Mapping[str, object],
) -> dict[str, object]:
    """고정 모델군에 대응하는 저장 결과 평가기를 선택한다."""
    if name == "m1":
        return score_m1(predictions, targets)

    if name == "m2":
        return score_m2(
            predictions, targets, baseline_asset
        )

    if name == "m4":
        return score_m4(predictions, targets)

    if name == "m5":
        return score_m5(
            predictions, targets, baseline_asset
        )

    require(
        name in {"m3_hitter", "m3_pitcher"},
        "지원하지 않는 최종 평가 대상",
    )

    return score_m3(
        predictions,
        targets,
        name.removeprefix("m3_"),
        baseline_asset,
    )
