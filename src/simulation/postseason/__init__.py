"""포스트시즌 시리즈 및 한국시리즈 우승 전망."""

VERSION = "1.0.0"

SUPPORTED_SEASON = 2025

SCENARIO_SCHEMA = "postseason-scenario-v1"
ITERATION_SCHEMA = "postseason-iteration-v1"
SUMMARY_SCHEMA = "postseason-summary-v1"

RESULT_DIRECTORY = "postseason"

DEFAULT_ITERATIONS = 1000

# 비정상 확률이나 반복 무승부에 의한 무한 실행을 방지한다.
# 한도 도달 시 임의 승자를 선정하지 않는다.
MAX_SERIES_GAMES = 64