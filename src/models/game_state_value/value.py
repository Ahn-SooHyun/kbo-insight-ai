"""잔여 기대득점과 관측 타석 가치, 상황 중요도를 계산한다."""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from math import fsum, isfinite
from typing import TypedDict

from . import model as m4
from .data import Pair


TransitionTable = dict[str, dict[str, int]]


class ImportanceReference(TypedDict):
    """지원되는 Train 타석의 기대 절대 WPA 참조 평균."""

    mean: float | None
    supported: int
    total: int


class ValueModel(TypedDict):
    """단일 승리 예측기와 비학습형 통계표."""

    predictor: m4.Predictor
    re_table: m4.StatTable
    transitions: TransitionTable
    reference: ImportanceReference


def re_keys(state: m4.State) -> tuple[str, ...]:
    """RE 통계의 세부 상태와 희소 후퇴 순서를 반환한다."""
    return (
        f"exact:{state.half}|{state.outs}|{state.mask}",
        f"base_out:{state.outs}|{state.mask}",
        f"outs:{state.outs}",
        "global",
    )


def fit_re(rows: Sequence[m4.Observation]) -> m4.StatTable:
    """Train 비종료 상태의 잔여 실득점 평균을 집계한다."""
    table: m4.StatTable = {}

    for row in rows:
        if row.final is not None or row.state.outs == 3:
            m4.require(row.remaining == 0, "종료 RE 정답 오류")
            continue

        m4.accumulate(
            table,
            re_keys(row.state),
            row.remaining,
        )

    m4.validate_table(table)
    return table


def predict_re(
    table: m4.StatTable,
    state: m4.State,
    confirmed_final: int | None = None,
    *,
    global_only: bool = False,
) -> tuple[float, str, int | None]:
    """반이닝 종료는 0, 그 외는 학습 통계의 조건부 평균을 반환한다."""
    if confirmed_final is not None or state.outs == 3:
        return 0.0, "half_ended", None

    keys = ("global",) if global_only else re_keys(state)
    n, total, level = m4.lookup(table, keys)

    return total / n, level, n


def transition_code(pair: Pair) -> str:
    """실제 득점·후속 아웃·주자·관측 종료 결과를 보존한다."""
    post = pair.post.state

    terminal = (
        -1 if pair.post.final is None
        else pair.post.final
    )

    return (
        f"{pair.actual_runs}|{post.outs}|"
        f"{post.mask}|{terminal}"
    )


def fit_transitions(pairs: Sequence[Pair]) -> TransitionTable:
    """Train의 타석 직전 상태별 경험적 다음 상태 분포를 집계한다."""
    table: TransitionTable = {}

    for pair in pairs:
        m4.require(pair.split == "train", "전이 통계에 비학습 타석 유입")

        key = m4.exact_key(pair.pre.state)
        transition = transition_code(pair)

        outcomes = table.setdefault(key, {})
        outcomes[transition] = outcomes.get(transition, 0) + 1

    m4.require(bool(table), "학습 전이 자료 없음")
    return table


def decode_transition(
    code: str,
) -> tuple[int, int, int, int | None]:
    """저장된 상태 전이 값을 복원하고 범위를 검사한다."""
    parts = code.split("|")
    m4.require(len(parts) == 4, "전이 코드 형식 오류")

    runs, outs, mask, terminal = map(int, parts)

    m4.require(
        runs >= 0
        and 0 <= outs <= 3
        and 0 <= mask <= 7
        and terminal in (-1, 0, 1)
        and (outs != 3 or mask == 0),
        "전이 상태 값 오류",
    )

    return runs, outs, mask, None if terminal == -1 else terminal


def hypothetical_post(
    pre: m4.State,
    code: str,
) -> tuple[m4.State, int | None]:
    """현재 사전 상태에 Train에서 관측한 가상 전이를 적용한다."""
    runs, outs, mask, terminal = decode_transition(code)

    post = replace(
        pre,
        phase="post",
        home_score=pre.home_score + (
            runs if pre.attacking_home else 0
        ),
        away_score=pre.away_score + (
            runs if not pre.attacking_home else 0
        ),
        outs=outs,
        on_first=int(bool(mask & 1)),
        on_second=int(bool(mask & 2)),
        on_third=int(bool(mask & 4)),
    )

    m4.validate_pair(pre, post)

    if terminal is not None:
        m4.require(
            terminal == int(post.difference > 0),
            "가상 종료 결과와 점수차 불일치",
        )

    return post, terminal


def expected_absolute_wpa(
    predictor: m4.Predictor,
    transitions: TransitionTable,
    pre: m4.State,
    *,
    details: bool = False,
) -> tuple[float | None, int, list[dict[str, object]]]:
    """실제 결과를 사용하지 않고 가능한 전이의 절대 WPA를 평균한다."""
    if pre.phase != "pre" or pre.outs == 3:
        return None, 0, []

    outcomes = transitions.get(m4.exact_key(pre), {})
    support = sum(outcomes.values())

    if support < m4.MIN_SUPPORT:
        return None, support, []

    before = m4.predict_win(predictor, pre)[0]
    weighted_values = []
    distribution = []

    for code, n in sorted(outcomes.items()):
        post, terminal = hypothetical_post(pre, code)

        probability = n / support

        # terminal은 현재 타석의 실제 종료 정보가 아니라
        # Train에서 관측된 가상 전이의 일부다.
        after = m4.predict_win(
            predictor,
            post,
            terminal,
        )[0]

        change = after - before

        weighted_values.append(probability * abs(change))

        if details:
            runs, outs, mask, _ = decode_transition(code)

            distribution.append({
                "actual_runs_if_transition": runs,
                "post_outs": outs,
                "post_base_mask": mask,
                "historical_final_win": terminal,
                "count": n,
                "probability": probability,
                "post_win_probability": after,
                "wpa": change,
                "weighted_absolute_wpa": (
                    probability * abs(change)
                ),
            })

    if details:
        m4.require(
            abs(fsum(
                row["probability"]
                for row in distribution
            ) - 1.0) < 1e-12,
            "상황 전이 확률 합 불일치",
        )

    return fsum(weighted_values), support, distribution


def fit_reference(
    predictor: m4.Predictor,
    transitions: TransitionTable,
    pairs: Sequence[Pair],
) -> ImportanceReference:
    """지원되는 Train 사전 상태의 기대 절대 WPA를 참조 평균으로 삼는다."""
    m4.require(
        all(pair.split == "train" for pair in pairs),
        "상황 중요도 참조 평균에 비학습 타석 유입",
    )

    states = Counter(pair.pre.state for pair in pairs)

    supported = 0
    weighted_values = []

    for state, n in states.items():
        expected, _, _ = expected_absolute_wpa(
            predictor,
            transitions,
            state,
        )

        if expected is None:
            continue

        supported += n
        weighted_values.append(expected * n)

    mean = fsum(weighted_values) / supported if supported else None

    return {
        "mean": mean,
        "supported": supported,
        "total": len(pairs),
    }


def importance(
    asset: ValueModel,
    pre: m4.State,
    *,
    details: bool = True,
) -> dict[str, object]:
    """상황 중요도 및 적용 가능 여부를 계산한다."""
    if pre.phase != "pre":
        return {
            "value": None,
            "reason": "pre_state_required",
            "support": 0,
            "expected_absolute_wpa": None,
            "reference_mean": asset["reference"]["mean"],
            "transitions": [],
        }

    expected, support, outcomes = expected_absolute_wpa(
        asset["predictor"],
        asset["transitions"],
        pre,
        details=details,
    )

    reference = asset["reference"]["mean"]

    reason = None

    if expected is None:
        reason = "insufficient_transition_support"

    elif reference is None or reference <= 0:
        reason = "unsupported_reference_mean"

    value = (
        None if reason is not None
        else expected / reference
    )

    return {
        "value": value,
        "reason": reason,
        "support": support,
        "expected_absolute_wpa": expected,
        "reference_mean": reference,
        "transitions": outcomes,
    }


def boundary(
    state: m4.State,
    evidence: m4.FinalEvidence | None,
) -> str:
    """경기 종료 확정과 반이닝 종료를 구분한다."""
    if evidence is not None:
        if evidence.winner == "tie":
            return "confirmed_final_tie"

        if state.outs == 3:
            return "confirmed_final_three_outs"

        return "confirmed_partial_final"

    if state.outs == 3:
        return "half_ended_game_end_unconfirmed"

    return "open_or_game_end_unconfirmed"


def state_value(
    asset: ValueModel,
    state: m4.State,
    evidence: m4.FinalEvidence | None = None,
    *,
    with_importance: bool = True,
) -> dict[str, object]:
    """관측 상태에서 승리 확률과 잔여 기대득점을 제공한다."""
    final = m4.final_result(state, evidence)

    probability, win_method, win_support = m4.predict_win(
        asset["predictor"],
        state,
        final,
    )

    remaining, re_method, re_support = predict_re(
        asset["re_table"],
        state,
        final,
    )

    result = {
        "perspective_team_id": state.perspective_team_id,
        "phase": state.phase,
        "win_probability": probability,
        "win_method": win_method,
        "win_support": win_support,
        "remaining_expected_runs": remaining,
        "re_method": re_method,
        "re_support": re_support,
        "half_ended": state.outs == 3 or final is not None,
        "game_end_confirmed": final is not None,
        "boundary": boundary(state, evidence),
    }

    if with_importance:
        result["importance"] = importance(asset, state)

    return result


def plate_value(
    asset: ValueModel,
    pre: m4.State,
    post: m4.State,
    evidence: m4.FinalEvidence | None = None,
    *,
    with_importance: bool = True,
) -> dict[str, object]:
    """동일 관점에서 관측 타석의 WPA와 기대득점 가치를 계산한다."""
    actual_runs = m4.validate_pair(pre, post)

    before = state_value(
        asset, pre, with_importance=False
    )

    after = state_value(
        asset, post, evidence, with_importance=False
    )

    wpa = after["win_probability"] - before["win_probability"]

    change = (
        after["remaining_expected_runs"]
        - before["remaining_expected_runs"]
    )

    result = {
        "perspective_team_id": pre.perspective_team_id,
        "pre": before,
        "post": after,
        "actual_runs": actual_runs,
        "wpa": wpa,
        "remaining_expected_runs_change": change,
        "run_value": actual_runs + change,
        "observed_walkoff_pattern": (
            evidence is not None
            and post.half == "bot"
            and post.inning >= 9
            and pre.difference <= 0
            and post.difference > 0
        ),
    }

    if with_importance:
        result["importance"] = importance(asset, pre)

    return result


def pair_input(pair: Pair) -> dict[str, object]:
    """정답 값을 포함하지 않은 독립 추론용 입력을 만든다."""
    return {
        "pre": asdict(pair.pre.state),
        "post": asdict(pair.post.state),
        "final": asdict(pair.final) if pair.final else None,
    }


def predict(
    asset: ValueModel,
    raw: Mapping[str, object],
) -> dict[str, object]:
    """단일 상태 또는 관측 타석 쌍 JSON 입력을 처리한다."""
    keys = set(raw)

    m4.require(
        keys in (
            {"state", "final"},
            {"pre", "post", "final"},
        ),
        "M4 추론 입력 구조 오류",
    )

    evidence_raw = raw["final"]

    m4.require(
        evidence_raw is None or isinstance(evidence_raw, Mapping),
        "종료 근거 입력 타입 오류",
    )

    evidence = m4.parse_final(evidence_raw)

    if "state" in raw:
        m4.require(isinstance(raw["state"], Mapping), "상태 입력 오류")

        return state_value(
            asset,
            m4.parse_state(raw["state"]),
            evidence,
        )

    m4.require(
        isinstance(raw["pre"], Mapping)
        and isinstance(raw["post"], Mapping),
        "전후 상태 입력 오류",
    )

    return plate_value(
        asset,
        m4.parse_state(raw["pre"]),
        m4.parse_state(raw["post"]),
        evidence,
    )


def validate_asset(asset: ValueModel) -> None:
    """복원된 분류기·RE·전이 분포·참조 평균을 확인한다."""
    m4.require(
        set(asset) == {
            "predictor", "re_table",
            "transitions", "reference",
        },
        "상황 가치 자산 구조 오류",
    )

    m4.validate_predictor(asset["predictor"])
    m4.validate_table(asset["re_table"])

    transitions = asset["transitions"]

    m4.require(
        isinstance(transitions, dict) and bool(transitions),
        "상황 전이 통계 누락",
    )

    transition_total = 0

    for key, outcomes in transitions.items():
        parts = key.split("|")

        m4.require(len(parts) == 5, "사전 상태 키 형식 오류")

        inning, half, difference, outs, mask = parts

        m4.require(
            int(inning) >= 1
            and half in ("top", "bot")
            and 0 <= int(outs) <= 2
            and 0 <= int(mask) <= 7
            and isinstance(outcomes, dict)
            and bool(outcomes),
            "사전 상태 전이 키 오류",
        )

        int(difference)

        for code, n in outcomes.items():
            _, after_outs, _, _ = decode_transition(code)

            m4.require(
                after_outs >= int(outs),
                "저장 전이에서 아웃 감소",
            )

            m4.require(m4.count(n, "transition_count") > 0, "전이 건수 오류")
            transition_total += n

    reference = asset["reference"]

    m4.require(
        set(reference) == {"mean", "supported", "total"},
        "상황 중요도 참조 구조 오류",
    )

    supported = m4.count(reference["supported"], "supported")
    total = m4.count(reference["total"], "total")
    mean = reference["mean"]

    m4.require(
        total == transition_total and total > 0
        and 0 <= supported <= total,
        "상황 중요도 참조 표본 수 오류",
    )

    expected_supported = sum(
        sum(outcomes.values())
        for outcomes in transitions.values()
        if sum(outcomes.values()) >= m4.MIN_SUPPORT
    )

    m4.require(
        supported == expected_supported,
        "상황 중요도 지원 수 불일치",
    )

    if supported == 0:
        m4.require(mean is None, "미지원 참조 평균 오류")
    else:
        m4.require(
            type(mean) in (int, float)
            and isfinite(mean)
            and 0 <= mean <= 1,
            "상황 중요도 참조 평균 범위 오류",
        )