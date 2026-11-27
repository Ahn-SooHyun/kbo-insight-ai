"""잔여 기대득점, 관측 타석 가치와 상황 중요도를 계산한다."""

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


class BoundaryInput(TypedDict):
    """예측 특징과 분리된 관측 경기·타석 식별자."""

    game_id: str
    pa_number: int
    state: dict[str, object]


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
    """반이닝 종료는 0, 그 외는 학습 통계 평균을 반환한다."""
    if confirmed_final is not None or state.outs == 3:
        return 0.0, "half_ended", None

    keys = ("global",) if global_only else re_keys(state)
    n, total, level = m4.lookup(table, keys)

    return total / n, level, n


def transition_code(pair: Pair) -> str:
    """실제 득점·후속 아웃·주자 상태만 저장한다.

    과거 경기의 공식 종료값을 전이 코드에서 제외한다.
    다른 시즌에 적용할 때는 해당 시즌의 규칙으로 종료를 판정한다.
    """
    post = pair.post.state

    return (
        f"{pair.actual_runs}|{post.outs}|{post.mask}"
    )


def fit_transitions(
    pairs: Sequence[Pair],
) -> TransitionTable:
    """Train 사전 상태별 경험적 물리 전이 분포를 집계한다."""
    table: TransitionTable = {}

    for pair in pairs:
        m4.require(
            pair.split == "train",
            "전이 통계에 비학습 타석 유입",
        )

        actual = m4.validate_pair(
            pair.pre.state,
            pair.post.state,
        )

        m4.require(
            actual == pair.actual_runs,
            "학습 전이의 실제 득점 불일치",
        )

        key = m4.exact_key(pair.pre.state)
        transition = transition_code(pair)

        outcomes = table.setdefault(key, {})
        outcomes[transition] = outcomes.get(transition, 0) + 1

    m4.require(bool(table), "학습 전이 자료 없음")

    return table


def decode_transition(code: str) -> tuple[int, int, int]:
    """득점·아웃·주자의 물리 전이 값을 복원한다."""
    parts = code.split("|")

    m4.require(len(parts) == 3, "전이 코드 형식 오류")

    runs, outs, mask = map(int, parts)

    m4.require(
        runs >= 0
        and 0 <= outs <= 3
        and 0 <= mask <= 7
        and (outs != 3 or mask == 0),
        "전이 상태 값 오류",
    )

    m4.require(
        code == f"{runs}|{outs}|{mask}",
        "전이 코드 정규 형식 오류",
    )

    return runs, outs, mask


def hypothetical_post(
    pre: m4.State,
    code: str,
) -> tuple[m4.State, int | None]:
    """물리 전이를 적용하고 현재 시즌의 정상 종료값을 계산한다."""
    runs, outs, mask = decode_transition(code)

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

    # 실제 과거 경기의 종료 결과가 아닌 현재 시즌 규칙을 사용한다.
    terminal = m4.normal_final_result(pre, post)

    return post, terminal


def expected_absolute_wpa(
    predictor: m4.Predictor,
    transitions: TransitionTable,
    pre: m4.State,
    *,
    details: bool = False,
) -> tuple[float | None, int, list[dict[str, object]]]:
    """사전 상태와 학습된 전이 확률만으로 기대 절대 WPA를 계산한다."""
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
        post, rule_final = hypothetical_post(pre, code)

        probability = n / support

        # 규칙상 확정되는 가상 종료는 해당 결과를 사용하고,
        # 계속되는 가상 상태는 운영 승리 예측기로 계산한다.
        after = m4.predict_win(
            predictor,
            post,
            rule_final,
        )[0]

        change = after - before

        weighted_values.append(probability * abs(change))

        if details:
            runs, outs, mask = decode_transition(code)

            distribution.append({
                "actual_runs_if_transition": runs,
                "post_outs": outs,
                "post_base_mask": mask,
                "normal_rule_final": rule_final is not None,
                "normal_rule_final_win": rule_final,
                "season_inning_limit": m4.inning_limit(pre.season),
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
    """지원되는 Train 사전 상태의 기대 절대 WPA 평균을 구한다."""
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

    mean = (
        fsum(weighted_values) / supported
        if supported else None
    )

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
    """상황 중요도와 지원 범위를 반환한다."""
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

    result = (
        None if reason is not None
        else expected / reference
    )

    return {
        "value": result,
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
    """검증된 내부 관측 상태에서 승리 확률과 RE를 반환한다."""
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
        "season_inning_limit": m4.inning_limit(state.season),
        "maximum_future_halves": m4.maximum_future_halves(state),
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
    """검증된 내부 타석의 동일 관점 WPA와 득점 가치를 계산한다."""
    actual_runs = m4.validate_pair(pre, post)

    before = state_value(
        asset,
        pre,
        with_importance=False,
    )

    after = state_value(
        asset,
        post,
        evidence,
        with_importance=False,
    )

    wpa = (
        after["win_probability"]
        - before["win_probability"]
    )

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
            and post.inning >= m4.REGULATION_INNINGS
            and pre.difference <= 0
            and post.difference > 0
        ),
    }

    if with_importance:
        result["importance"] = importance(asset, pre)

    return result


def validate_identity(
    game_id: object,
    pa_number: object,
) -> tuple[str, int]:
    """관측 경기 및 타석의 감사 식별자를 확인한다."""
    m4.require(
        isinstance(game_id, str)
        and bool(game_id)
        and game_id == game_id.strip(),
        "경기 식별자 오류",
    )

    number = m4.count(pa_number, "pa_number")

    m4.require(number > 0, "타석 번호는 양의 정수여야 합니다.")

    return game_id, number


def validate_evidence_identity(
    game_id: str,
    pa_number: int,
    evidence: m4.FinalEvidence | None,
) -> None:
    """종료 근거가 입력 경기와 동일 타석에 속하는지 검사한다."""
    if evidence is None:
        return

    m4.require(
        evidence.game_id == game_id
        and evidence.pa_number == pa_number
        and evidence.last_pa == pa_number,
        "입력 경기·타석과 공식 종료 근거 불일치",
    )


def boundary_input(
    game_id: str,
    pa_number: int,
    state: m4.State,
) -> BoundaryInput:
    """모델 특징과 감사 식별자를 분리한 입력을 구성한다."""
    validate_identity(game_id, pa_number)

    return {
        "game_id": game_id,
        "pa_number": pa_number,
        "state": asdict(state),
    }


def parse_boundary(
    raw: object,
) -> tuple[str, int, m4.State]:
    """독립 추론용 관측 경계의 식별자와 상태를 복원한다."""
    m4.require(
        isinstance(raw, Mapping)
        and set(raw) == {"game_id", "pa_number", "state"},
        "관측 경계 입력 구조 오류",
    )

    game_id, number = validate_identity(
        raw["game_id"],
        raw["pa_number"],
    )

    m4.require(
        isinstance(raw["state"], Mapping),
        "상태 입력 타입 오류",
    )

    return (
        game_id,
        number,
        m4.parse_state(raw["state"]),
    )


def pair_input(pair: Pair) -> dict[str, object]:
    """검증된 타석 쌍을 독립 추론 계약으로 변환한다."""
    prefix, separator, suffix = pair.pair_id.rpartition(":")

    m4.require(
        separator == ":"
        and prefix == pair.game_id
        and suffix.isascii()
        and suffix.isdecimal(),
        "자료 타석 식별자 형식 오류",
    )

    game_id, number = validate_identity(
        pair.game_id,
        int(suffix),
    )

    m4.require(
        pair.pair_id == f"{game_id}:{number}",
        "자료 타석 식별자 정규 형식 오류",
    )

    validate_evidence_identity(
        game_id,
        number,
        pair.final,
    )

    m4.validate_pair(pair.pre.state, pair.post.state)

    return {
        "pre": boundary_input(
            game_id,
            number,
            pair.pre.state,
        ),
        "post": boundary_input(
            game_id,
            number,
            pair.post.state,
        ),
        "final": asdict(pair.final) if pair.final else None,
    }


def predict(
    asset: ValueModel,
    raw: Mapping[str, object],
) -> dict[str, object]:
    """감사 식별자를 확인한 단일 상태 또는 타석 쌍을 평가한다."""
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
        game_id, number, state = parse_boundary(
            raw["state"],
        )

        validate_evidence_identity(
            game_id,
            number,
            evidence,
        )

        result = state_value(
            asset,
            state,
            evidence,
        )

        return {
            "game_id": game_id,
            "pa_number": number,
            **result,
        }

    pre_game, pre_number, pre = parse_boundary(raw["pre"])
    post_game, post_number, post = parse_boundary(raw["post"])

    m4.require(
        pre_game == post_game
        and pre_number == post_number,
        "전후 상태의 경기 또는 타석 식별자 불일치",
    )

    validate_evidence_identity(
        post_game,
        post_number,
        evidence,
    )

    result = plate_value(
        asset,
        pre,
        post,
        evidence,
    )

    return {
        "game_id": pre_game,
        "pa_number": pre_number,
        **result,
    }


def validate_asset(asset: ValueModel) -> None:
    """분류기, RE, 물리 전이 및 참조 평균을 검사한다."""
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

        m4.require(
            len(parts) == 5,
            "사전 상태 키 형식 오류",
        )

        inning, half, difference, outs, mask = parts

        m4.require(
            1 <= int(inning) <= m4.REFERENCE_INNING_LIMIT
            and half in ("top", "bot")
            and 0 <= int(outs) <= 2
            and 0 <= int(mask) <= 7
            and isinstance(outcomes, dict)
            and bool(outcomes),
            "사전 상태 전이 키 오류",
        )

        int(difference)

        for code, n in outcomes.items():
            _, after_outs, _ = decode_transition(code)

            m4.require(
                after_outs >= int(outs),
                "저장 전이에서 아웃 감소",
            )

            m4.require(
                m4.count(n, "transition_count") > 0,
                "전이 건수 오류",
            )

            transition_total += n

    reference = asset["reference"]

    m4.require(
        set(reference) == {"mean", "supported", "total"},
        "상황 중요도 참조 구조 오류",
    )

    supported = m4.count(
        reference["supported"],
        "supported",
    )

    total = m4.count(
        reference["total"],
        "total",
    )

    mean = reference["mean"]

    m4.require(
        total == transition_total
        and total > 0
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
        m4.require(
            mean is None,
            "미지원 참조 평균 오류",
        )

    else:
        m4.require(
            type(mean) in (int, float)
            and isfinite(mean)
            and 0 <= mean <= 1,
            "상황 중요도 참조 평균 범위 오류",
        )


def rule_examples(
    state: m4.State,
) -> list[dict[str, object]]:
    """동일 물리 전이에 대한 두 시즌의 종료 차이를 재현한다.

    실제 경기 평가 표본이 아니라 규칙 검증용 합성 상태다.
    """
    examples = []

    for season, expected in (
        (2024, None),
        (2025, 0),
    ):
        pre = replace(
            state,
            season=season,
            inning=11,
            half="bot",
            phase="pre",
            perspective_team_id=state.home_team_id,
            home_score=3,
            away_score=3,
            outs=2,
            on_first=0,
            on_second=0,
            on_third=0,
        )

        pre = m4.parse_state(asdict(pre))

        post, final = hypothetical_post(
            pre,
            "0|3|0",
        )

        m4.require(
            final == expected,
            f"시즌별 가상 종료 규칙 불일치: {season}",
        )

        examples.append({
            "case": "eleventh_bottom_tie_three_outs",
            "synthetic": True,
            "season": season,
            "maximum_inning": m4.inning_limit(season),
            "pre": asdict(pre),
            "post": asdict(post),
            "transition": "0|3|0",
            "normal_rule_final": final is not None,
            "normal_rule_final_win": final,
            "expected_final_win": expected,
        })

    return examples