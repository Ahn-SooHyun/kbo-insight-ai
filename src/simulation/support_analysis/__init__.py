"""예상 선발투수·팀 전력·잔여 일정 난이도 분석 계약."""

VERSION = "1.1.0"
SCHEMA = "support-analysis-v1"

STARTER_RULE_VERSION = "starter-rotation-v2"
TEAM_RULE_VERSION = "team-record-v1"
SCHEDULE_RULE_VERSION = "schedule-expected-loss-v1"

SUPPORTED_SEASON = 2025

STARTER_HISTORY_DAYS = 28
WORKLOAD_DAYS = 7
ROTATION_INTERVAL_DAYS = 5
MIN_PROJECTED_APPEARANCE_INTERVAL = 2
MAX_STARTER_FORECAST_DAYS = 7
TOP_K = 3

STARTER_SELECTED_METHOD = "baseline"

RECENT_DAYS = 28
MIN_TEAM_SAMPLE = 5

DRAW_DIFFICULTY_WEIGHT = 0.5
PROBABILITY_TOLERANCE = 1e-12


def starter_selection() -> dict[str, object]:
    """개발 평가를 근거로 선정한 고정 운영 규칙을 설명한다.

    실행할 때마다 평가 결과에 따라 규칙을 다시 선택하지 않는다.
    2026년 최종 Test는 선택 과정에서 사용하지 않는다.
    """

    return {
        "rule_version": STARTER_RULE_VERSION,
        "selected_method": STARTER_SELECTED_METHOD,
        "selection_split": "validation",
        "selection_season": SUPPORTED_SEASON,
        "comparison_methods": [
            "baseline",
            "proposed",
        ],
        "selection_reason": (
            "baseline_higher_top1_and_topk_on_common_samples"
        ),
        "selection_evaluation_run_id": (
            "5f5145d8c2734f03ac2a2334ca42319a"
        ),
        "comparison_cutoff_policy": (
            "game_day_midnight_kst"
        ),
        "independent_final_evaluation_completed": False,
        "final_test_used": False,
    }