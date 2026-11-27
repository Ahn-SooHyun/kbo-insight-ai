"""부분 정답 공동 회귀와 선수 전망의 Count 일관성을 구현한다."""

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date, timedelta
import math
from statistics import mean

from . import data
from ...datasets.model_data import require
from ...datasets.published_sources import as_row


CANDIDATE_DEPTHS = (2, 4)
MIN_LEAF_ROWS = 40
MIN_OUTPUT_SUPPORT = 8
SHRINKAGE = 10.0
SPLIT_QUANTILES = (0.2, 0.4, 0.6, 0.8)
RECENT_WEIGHT = 0.7
REFERENCE_MONTH = 3
REFERENCE_DAY = 1
MIN_REMAINING_DAYS = 7
MIN_GAIN = 1e-12
INTERVAL_LEVEL = 0.90

Vector = tuple[float, ...]
MaskedVector = tuple[float | None, ...]


def calendar_start(season: int) -> date:
    """실제 미래 개막일을 쓰지 않는 고정 달력 기준점."""
    return date(season, REFERENCE_MONTH, REFERENCE_DAY)


def elapsed(player: data.PlayerInput) -> int:
    """고정 달력 기준의 경과일. 실제 개막일이라는 뜻은 아니다."""
    return max(1, (player.anchor - calendar_start(player.season)).days)


def number(value: object) -> float:
    """자산과 수치 계산의 유한 실수 조건."""
    require(
        type(value) in (int, float)
        and math.isfinite(value),
        "유한 실수가 필요합니다",
    )
    return float(value)


def feature_names(role: str) -> tuple[str, ...]:
    """선택된 입력의 고정 순서. 선수 ID와 팀 ID는 특징에서 제외한다."""
    names = data.metrics(role)
    return (
        "calendar_elapsed",
        "days_since_observed",
        "prior_rejected",
        "known_team_missing",
        *(f"season_daily_{name}" for name in names),
        *(f"recent28_daily_{name}" for name in names),
    )


def features(player: data.PlayerInput) -> Vector:
    """사전 기록의 달력 일수당 집계와 결측 표지를 구성한다.

    비음수 특징에서 -1은 미관측을 뜻한다. 실제 Count 0이나
    평균으로 대체하지 않는다.
    """
    days = max(28, elapsed(player))

    season = tuple(
        -1.0 if value is None else value / days
        for value in player.season_counts
    )
    recent = tuple(
        -1.0 if value is None else value / 28
        for value in player.recent_counts
    )

    return (
        float(elapsed(player)),
        float(player.days_since_observed),
        float(player.prior_rejected),
        float(player.known_team_id is None),
        *season,
        *recent,
    )


def coherent_increment(
    role: str,
    counts: dict[str, float],
) -> dict[str, float]:
    """비음수 증가량의 확인된 구성 관계를 보존한다.

    경기 수가 0인 증가량에서는 다른 기록을 양수로 만들지 않는다.
    주루 사건·무투구 타석 때문에 outs와 pitches에
    근거 없는 상대 타자 수 제약을 추가하지 않는다.
    """
    result = {
        name: max(0.0, number(value))
        for name, value in counts.items()
    }
    opportunities = data.exposure(role)

    if result["games"] == 0:
        return dict.fromkeys(result, 0.0)

    result["ab"] = min(
        result["ab"],
        result[opportunities],
    )

    free = result["walks"] + result["hbp"]
    available = max(
        0.0,
        result[opportunities] - result["ab"],
    )

    if free > available and free > 0:
        factor = available / free
        result["walks"] *= factor
        result["hbp"] *= factor

    result["hits"] = min(result["hits"], result["ab"])
    result["home_runs"] = min(
        result["home_runs"],
        result["hits"],
    )
    result["strikeouts"] = min(
        result["strikeouts"],
        max(0.0, result["ab"] - result["hits"]),
    )

    data.check_counts(role, result)
    return result


def project(
    player: data.PlayerInput,
    raw: Sequence[float],
) -> Vector:
    """누적 Count를 기간별 비음수 증가량으로 투영한다.

    7일 <= 28일 <= 잔여 정규시즌을 보존한다.
    실제 미래 종료일은 입력하지 않는다. 호출자가 사전 근거를
    제공한 종료 경계가 있을 때만 동일 기간의 Count를 일치시킨다.
    """
    role = player.role
    names = data.metrics(role)
    width = len(names)

    require(
        len(raw) == len(data.output_keys(role)),
        "공동 출력 차원 오류",
    )

    previous = dict.fromkeys(names, 0.0)
    output: list[float] = []

    for index, horizon in enumerate(data.HORIZONS):
        desired = {
            name: number(raw[index * width + j])
            for j, name in enumerate(names)
        }

        equal_previous = (
            player.known_end is not None
            and (
                (
                    horizon == "28d"
                    and player.known_end
                    <= player.anchor + timedelta(days=7)
                )
                or (
                    horizon == "remaining"
                    and player.known_end
                    <= player.anchor + timedelta(days=28)
                )
            )
        )

        if equal_previous:
            current = previous.copy()
        else:
            delta = coherent_increment(
                role,
                {
                    name: max(
                        0.0,
                        desired[name] - previous[name],
                    )
                    for name in names
                },
            )

            current = {
                name: previous[name] + delta[name]
                for name in names
            }

        data.check_counts(role, current)
        output.extend(current[name] for name in names)
        previous = current

    return tuple(output)


@dataclass(frozen=True, slots=True)
class Baseline:
    """학습 자료에서 계산한 기간 환산 기준과 출력별 정규화 척도."""

    role: str
    daily_prior: Vector
    remaining_end_offset: float
    scales: Vector

    def predict(self, player: data.PlayerInput) -> Vector:
        """실제 미래 경기 수 대신 과거 달력 일수당 기록을 환산한다."""
        require(player.role == self.role, "기준 모델 역할 불일치")

        season_days = max(28, elapsed(player))
        rates: list[float] = []

        for prior, season, recent in zip(
            self.daily_prior,
            player.season_counts,
            player.recent_counts,
        ):
            long_rate = (
                prior if season is None else season / season_days
            )
            short_rate = (
                long_rate if recent is None else recent / 28
            )

            rates.append(
                RECENT_WEIGHT * short_rate
                + (1.0 - RECENT_WEIGHT) * long_rate
            )

        remaining = max(
            float(MIN_REMAINING_DAYS),
            self.remaining_end_offset - elapsed(player),
        )

        if player.known_end is not None:
            remaining = float(
                (player.known_end - player.anchor).days
            )

        duration = (
            min(7.0, remaining),
            min(28.0, remaining),
            remaining,
        )

        raw = tuple(
            rate * days
            for days in duration
            for rate in rates
        )
        return project(player, raw)


def fit_baseline(
    role: str,
    samples: Sequence[data.Sample],
    cutoff: date,
) -> Baseline:
    """기준 통계와 정규화 척도를 해당 Train에서만 적합한다."""
    width = len(data.metrics(role))
    dimension = len(data.output_keys(role))
    values = [[] for _ in range(dimension)]
    ends: dict[int, date] = {}

    for sample in samples:
        require(
            sample.player.role == role
            and sample.player.anchor < cutoff,
            "기준 모델 학습 분할 또는 역할 오류",
        )

        labels = data.labels_for(sample, cutoff)

        for j, value in enumerate(labels):
            if value is not None:
                values[j].append(value)

        remaining = sample.targets[-1]

        if (
            remaining.usable
            and remaining.ready is not None
            and remaining.ready <= cutoff
            and remaining.window_end is not None
        ):
            previous = ends.setdefault(
                sample.player.season,
                remaining.window_end,
            )
            require(
                previous == remaining.window_end,
                "학습 시즌 종료 근거 중복 불일치",
            )

    require(
        all(values) and bool(ends),
        "기준 학습용 기간·기록별 정답 지지 부족",
    )

    return Baseline(
        role=role,
        daily_prior=tuple(
            mean(values[j]) / 7 for j in range(width)
        ),
        remaining_end_offset=mean(
            (end - calendar_start(season)).days
            for season, end in ends.items()
        ),
        scales=tuple(
            max(1.0, float(mean(column)))
            for column in values
        ),
    )


@dataclass(frozen=True, slots=True)
class Node:
    """공유된 분기와 마스크별 잔차 평균을 갖는 회귀 트리 노드."""

    values: Vector
    support: tuple[int, ...]
    feature: int | None = None
    threshold: float | None = None
    left: "Node | None" = None
    right: "Node | None" = None

    def predict(self, row: Vector) -> Vector:
        """한 입력에서 전체 기간·기록 잔차 벡터를 반환한다."""
        node = self

        while node.feature is not None:
            require(
                node.threshold is not None
                and node.left is not None
                and node.right is not None,
                "트리 분기 자산 오류",
            )

            node = (
                node.left
                if row[node.feature] <= node.threshold
                else node.right
            )

        return node.values


@dataclass(frozen=True, slots=True)
class TrainingRow:
    """입력과 관측된 출력별 정규화 잔차의 희소 표현."""

    features: Vector
    residuals: tuple[tuple[int, float], ...]


def training_rows(
    samples: Sequence[data.Sample],
    baseline: Baseline,
    cutoff: date,
) -> list[TrainingRow]:
    """마스크가 있는 기록만 학습 잔차를 생성한다."""
    rows = []

    for sample in samples:
        labels = data.labels_for(sample, cutoff)

        if all(value is None for value in labels):
            continue

        prediction = baseline.predict(sample.player)
        residuals = tuple(
            (
                j,
                (value - prediction[j]) / baseline.scales[j],
            )
            for j, value in enumerate(labels)
            if value is not None
        )

        rows.append(
            TrainingRow(
                features(sample.player),
                residuals,
            )
        )

    require(bool(rows), "공동 회귀 학습 표본 없음")
    return rows


def moments(
    rows: Sequence[TrainingRow],
    indices: Sequence[int],
    dimension: int,
) -> tuple[list[int], list[float]]:
    """마스크별 관측 건수와 잔차 합을 계산한다."""
    counts = [0] * dimension
    sums = [0.0] * dimension

    for i in indices:
        for j, value in rows[i].residuals:
            counts[j] += 1
            sums[j] += value

    return counts, sums


def thresholds(
    rows: Sequence[TrainingRow],
) -> tuple[tuple[float, ...], ...]:
    """Train 특징의 분위수만 사용해 분기 후보를 고정한다."""
    result = []

    for j in range(len(rows[0].features)):
        values = sorted(row.features[j] for row in rows)
        candidates = {
            values[int(q * (len(values) - 1))]
            for q in SPLIT_QUANTILES
        }
        result.append(
            tuple(sorted(
                value for value in candidates
                if value < values[-1]
            ))
        )

    return tuple(result)


def best_split(
    rows: Sequence[TrainingRow],
    indices: Sequence[int],
    counts: Sequence[int],
    sums: Sequence[float],
    weights: Sequence[float],
    candidates: Sequence[Sequence[float]],
) -> tuple[int, float] | None:
    """공동 출력의 마스크별 제곱 오차 감소를 최대화한다.

    전체 출력 건수의 역수로 가중하여 빈번한 7일 정답만
    분기를 지배하지 않도록 한다.
    """
    dimension = len(counts)
    best_gain = MIN_GAIN
    best = None

    for feature, cuts in enumerate(candidates):
        if not cuts:
            continue

        ordered = sorted(
            indices,
            key=lambda i: rows[i].features[feature],
        )
        left_counts = [0] * dimension
        left_sums = [0.0] * dimension
        cursor = 0

        for cut in cuts:
            while (
                cursor < len(ordered)
                and rows[ordered[cursor]].features[feature] <= cut
            ):
                for j, value in rows[ordered[cursor]].residuals:
                    left_counts[j] += 1
                    left_sums[j] += value
                cursor += 1

            if cursor < MIN_LEAF_ROWS:
                continue

            if len(ordered) - cursor < MIN_LEAF_ROWS:
                break

            gain = 0.0

            for j in range(dimension):
                nl = left_counts[j]
                nr = counts[j] - nl

                if (
                    nl < MIN_OUTPUT_SUPPORT
                    or nr < MIN_OUTPUT_SUPPORT
                ):
                    continue

                sl = left_sums[j]
                sr = sums[j] - sl

                gain += weights[j] * max(
                    0.0,
                    sl * sl / nl
                    + sr * sr / nr
                    - sums[j] * sums[j] / counts[j],
                )

            # 순서가 고정되어 있어 동률 선택도 결정적이다.
            if gain > best_gain:
                best_gain = gain
                best = (feature, cut)

    return best


def fit_tree(
    rows: Sequence[TrainingRow],
    dimension: int,
    max_depth: int,
) -> Node:
    """모든 출력이 하나의 트리 구조를 공유하도록 학습한다."""
    require(max_depth in CANDIDATE_DEPTHS, "트리 깊이 오류")

    all_indices = list(range(len(rows)))
    root_counts, _ = moments(rows, all_indices, dimension)

    require(
        all(count > 0 for count in root_counts),
        "공동 트리의 출력별 학습 정답 지지 부족",
    )

    weights = tuple(1.0 / count for count in root_counts)
    candidates = thresholds(rows)

    def build(
        indices: list[int],
        depth: int,
        parent: Vector | None,
    ) -> Node:
        counts, sums = moments(rows, indices, dimension)
        values = []

        for j, count in enumerate(counts):
            if parent is None:
                value = sums[j] / count
            elif count < MIN_OUTPUT_SUPPORT:
                # 희소 출력은 상위 노드의 값을 사용한다.
                value = parent[j]
            else:
                local_mean = sums[j] / count
                strength = count / (count + SHRINKAGE)
                value = (
                    parent[j]
                    + strength * (local_mean - parent[j])
                )

            values.append(value)

        current = tuple(values)
        support = tuple(counts)

        if (
            depth >= max_depth
            or len(indices) < 2 * MIN_LEAF_ROWS
        ):
            return Node(current, support)

        selected = best_split(
            rows,
            indices,
            counts,
            sums,
            weights,
            candidates,
        )

        if selected is None:
            return Node(current, support)

        feature, cut = selected
        left = [
            i for i in indices
            if rows[i].features[feature] <= cut
        ]
        right = [
            i for i in indices
            if rows[i].features[feature] > cut
        ]

        return Node(
            values=current,
            support=support,
            feature=feature,
            threshold=cut,
            left=build(left, depth + 1, current),
            right=build(right, depth + 1, current),
        )

    return build(all_indices, 0, None)


@dataclass(frozen=True, slots=True)
class Fitted:
    """역할별 기준 통계와 단일 공동 회귀 트리."""

    role: str
    depth: int
    baseline: Baseline
    tree: Node | None

    @property
    def name(self) -> str:
        """비교 후보 및 운영 모델 식별자."""
        if self.tree is None:
            return "history_baseline"

        return f"joint_residual_tree_depth_{self.depth}"

    def predict(self, player: data.PlayerInput) -> Vector:
        """동일 기준 입력에서 공동 Count 기대값을 계산한다."""
        require(player.role == self.role, "예측 모델 역할 불일치")
        baseline = self.baseline.predict(player)

        if self.tree is None:
            return baseline

        residuals = self.tree.predict(features(player))

        raw = tuple(
            base + residual * scale
            for base, residual, scale in zip(
                baseline,
                residuals,
                self.baseline.scales,
            )
        )

        return project(player, raw)

    def to_payload(self) -> dict[str, object]:
        """pickle을 사용하지 않는 JSON용 학습 자산."""
        return asdict(self)


def fit(
    role: str,
    samples: Sequence[data.Sample],
    cutoff: date,
    depth: int,
) -> Fitted:
    """기준 모델 또는 공유 구조 회귀 트리 후보를 적합한다."""
    require(
        depth == 0 or depth in CANDIDATE_DEPTHS,
        "지원하지 않는 후보 모델",
    )

    baseline = fit_baseline(role, samples, cutoff)

    if depth == 0:
        return Fitted(role, depth, baseline, None)

    rows = training_rows(samples, baseline, cutoff)
    tree = fit_tree(
        rows,
        len(data.output_keys(role)),
        depth,
    )
    return Fitted(role, depth, baseline, tree)


def decode_node(
    payload: object,
    dimension: int,
    feature_count: int,
    max_depth: int,
    level: int = 0,
) -> Node:
    """저장 트리 전체를 검사하고 재귀적으로 복원한다."""
    row = as_row(payload)
    require(
        set(row)
        == {
            "values", "support", "feature",
            "threshold", "left", "right",
        },
        "트리 자산 필드 오류",
    )

    values = tuple(number(value) for value in row["values"])
    support = tuple(
        int(data.integer(value, "node_support"))
        for value in row["support"]
    )

    require(
        len(values) == len(support) == dimension
        and level <= max_depth,
        "트리 차원 또는 깊이 오류",
    )

    feature = row["feature"]

    if feature is None:
        require(
            row["threshold"] is None
            and row["left"] is None
            and row["right"] is None,
            "트리 리프 구조 오류",
        )
        return Node(values, support)

    require(
        type(feature) is int
        and 0 <= feature < feature_count
        and level < max_depth,
        "트리 분기 특징 오류",
    )

    cut = number(row["threshold"])
    left = decode_node(
        row["left"], dimension, feature_count,
        max_depth, level + 1,
    )
    right = decode_node(
        row["right"], dimension, feature_count,
        max_depth, level + 1,
    )

    require(
        all(
            a + b == total
            for a, b, total in zip(
                left.support, right.support, support
            )
        ),
        "트리 출력별 관측 건수 불일치",
    )

    return Node(values, support, feature, cut, left, right)


def restore(payload: object) -> Fitted:
    """역할·버전별 고정 차원과 트리 구조를 검증한다."""
    row = as_row(payload)
    require(
        set(row) == {"role", "depth", "baseline", "tree"},
        "모델 자산 필드 오류",
    )

    role = row["role"]
    require(isinstance(role, str) and role in data.ROLES, "역할 오류")

    depth = row["depth"]
    require(
        type(depth) is int and depth in CANDIDATE_DEPTHS,
        "운영 공동 회귀 트리 깊이 오류",
    )

    base = as_row(row["baseline"])
    require(base["role"] == role, "기준 통계 역할 오류")

    daily = tuple(number(value) for value in base["daily_prior"])
    scales = tuple(number(value) for value in base["scales"])
    end = number(base["remaining_end_offset"])

    require(
        len(daily) == len(data.metrics(role))
        and len(scales) == len(data.output_keys(role))
        and all(value >= 0 for value in daily)
        and all(value >= 1 for value in scales)
        and end > 0,
        "기준 통계 자산 차원 또는 값 오류",
    )

    baseline = Baseline(role, daily, end, scales)
    tree = decode_node(
        row["tree"],
        len(scales),
        len(feature_names(role)),
        depth,
    )

    require(
        all(count > 0 for count in tree.support),
        "운영 트리의 출력별 정답 지지 부족",
    )

    return Fitted(role, depth, baseline, tree)


def ratio(
    numerator: float,
    denominator: float,
) -> dict[str, object]:
    """같은 기간의 Count 분자·분모에서 비율을 계산한다."""
    if denominator == 0:
        return {
            "value": None,
            "reason": "zero_denominator",
            "numerator": numerator,
            "denominator": denominator,
        }

    return {
        "value": numerator / denominator,
        "reason": None,
        "numerator": numerator,
        "denominator": denominator,
    }


def derived(
    role: str,
    counts: Mapping[str, float],
) -> dict[str, object]:
    """자료 계약에 근거한 파생 비율과 투구 이닝."""
    result = {
        "hit_rate": ratio(counts["hits"], counts["ab"]),
        "strikeout_rate": ratio(
            counts["strikeouts"],
            counts[data.exposure(role)],
        ),
    }

    if role == "pitcher":
        result["innings"] = counts["outs"] / 3.0

    return result


def prediction_interval(
    point: float,
    calibration: Mapping[str, object],
    *,
    known_end: bool = False,
) -> dict[str, object]:
    """주변 Count의 비음수 잔차 구간을 계산한다.

    기록·기간별 주변 구간이므로 동시 포함을 보장하지 않는다.
    """
    radius = calibration.get("radius")

    if known_end:
        return {
            "lower": None,
            "upper": None,
            "reason": "known_end_scenario_not_calibrated",
            "target_coverage": INTERVAL_LEVEL,
        }

    if radius is None:
        return {
            "lower": None,
            "upper": None,
            "reason": calibration.get("reason", "insufficient_support"),
            "target_coverage": INTERVAL_LEVEL,
        }

    radius = number(radius)
    require(radius >= 0, "예측 구간 반경 오류")

    return {
        "lower": max(0.0, point - radius),
        "upper": point + radius,
        "reason": None,
        "target_coverage": INTERVAL_LEVEL,
    }


def format_prediction(
    fitted: Fitted,
    player: data.PlayerInput,
    intervals: Mapping[str, object],
) -> dict[str, object]:
    """기간별 예측과 현재 누적을 연결한 시즌 전망을 생성한다."""
    values = fitted.predict(player)
    role = player.role
    names = data.metrics(role)
    width = len(names)
    periods: dict[str, dict[str, object]] = {}

    for i, horizon in enumerate(data.HORIZONS):
        counts = dict(zip(
            names,
            values[i * width:(i + 1) * width],
        ))
        uncertainty = {
            name: prediction_interval(
                point,
                as_row(intervals[f"{horizon}.{name}"]),
                known_end=player.known_end is not None,
            )
            for name, point in counts.items()
        }

        periods[horizon] = {
            "counts": counts,
            "derived": derived(role, counts),
            "intervals": uncertainty,
        }

    season_projection: dict[str, object] = {
        "status": "unavailable",
        "reason": "current_counts_unavailable",
    }

    if player.current_counts is not None:
        remaining = as_row(periods["remaining"]["counts"])
        remaining_intervals = as_row(
            periods["remaining"]["intervals"]
        )
        current = dict(zip(names, player.current_counts))
        final_counts = {
            name: current[name] + number(remaining[name])
            for name in names
        }
        data.check_counts(role, final_counts)

        final_intervals = {}

        for name in names:
            interval = as_row(remaining_intervals[name]).copy()

            if interval["lower"] is not None:
                interval["lower"] = (
                    current[name] + number(interval["lower"])
                )
                interval["upper"] = (
                    current[name] + number(interval["upper"])
                )

            final_intervals[name] = interval

        season_projection = {
            "status": (
                "caller_verified_cumulative_projection"
                if player.current_scope == "caller_verified"
                else "observed_pbp_provisional_projection"
            ),
            "current_scope": player.current_scope,
            "current_counts": current,
            "counts": final_counts,
            "derived": derived(role, final_counts),
            "intervals": final_intervals,
        }

    return {
        "sample_id": player.sample_id,
        "player_id": player.player_id,
        "role": role,
        "sample_date": player.anchor.isoformat(),
        "timezone": "Asia/Seoul",
        "model": fitted.name,
        "competition": "regular",
        "count_semantics": "nonnegative_expected_counts",
        "known_end_exclusive": (
            player.known_end.isoformat()
            if player.known_end is not None else None
        ),
        "known_end_evidence": player.end_evidence,
        "periods": periods,
        "season_projection": season_projection,
        "unsupported_statistics": (
            ["obp", "slg", "ops", "rbi", "runs"]
            if role == "hitter"
            else ["era", "earned_runs", "wins", "saves"]
        ),
        "interval_semantics": "marginal_count_prediction_intervals",
    }