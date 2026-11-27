# 예상 선발투수·팀 전력·일정 난이도 실행과 결과 계약

**기존 경기 예측과 시즌 전망을 변경하지 않고 관측 기록과 기존 M1을 활용하는 보조 분석.**

| 항목        | 정의                                                                 |
| ----------- | -------------------------------------------------------------------- |
| 문서 역할   | 구현 제안의 기술 계약 (적용 및 검증 전)                              |
| 검토 기준일 | 2026-11-28 KST                                                       |
| 저장소 경로 | `docs/standards/support_analysis.md`                                 |
| 소스 기준   | `ee8cfe53e72413e3461153350c4c4c70faebde9f`                           |
| 관련 작업   | [Issue #29](https://github.com/Ahn-SooHyun/kbo-insight-ai/issues/29) |
| 상태        | 구현안 제출, 로컬 적용·테스트·수용 미확인                            |

[공통 계약](project_contract.md) · [M1](game_outcome_score.md) · [정규시즌 전망](regular_season_standings.md) · [개발 흐름](../planning/development_workflow.md)

[목적](#overview) · [자료](#data) · [입력](#inputs) · [출력](#outputs) · [방법](#method) · [강의](#lecture-methods) · [평가](#evaluation) · [재현](#reproducibility) · [한계](#limitations) · [명령](#commands)

<a id="overview"></a>

## 1. 목적·지원 범위

2025년 정규시즌의 KST 날짜 경계에서 팀별 기록, 다음 경기일 예상 선발 후보, 잔여 일정 난이도를 제공한다. 예측 후보는 공식 선발 발표, 현역 등록 상태, 감독 의도, 건강 상태 또는 선발 확률이 아니다. 기존 M1 및 시즌 시뮬레이션의 입력·가중치·출력·난수 계산을 변경하지 않으며 독립 주 예측기를 추가하지 않는다. 2026년 최종 Test, 원천 재수집, 라인업·교체, 서비스 배포는 제외한다.

<a id="data"></a>

## 2. 자료·분할·품질

표준 검증판 `e020dafedc8b4fd9ac6696d0e4e4a92e:37b56c39caf5435e8fb2a17f993ec4df`, 모델 자료 `259b38decaac4ed68dd5a596cfbe4849`, M1 `0d9433b8044e48e5a4d21bbdb9b432e5`가 인수 대상이다. 기존 `verify_assets`, `read_completed`, `load_scenario`로 자료·모델 계보와 경기 키를 검증한다. 사용자 사전 점검에서 `starter_evidence` 37,810행, `starter_actual` 31,579행, `starter_history` 5,758행, `starter_pre_game` 36,153행이 확인되었다. 전체 판본 행 수이며 평가 표본 수는 아니다.

후보는 `starter_history`, `player_identity_evidence`, 표준 `player_games`·`player_admission`의 시점 이전 행에서 구성한다. 최종 admission의 이름 충돌 상태를 과거의 입력으로 사용하지 않는다. 이전 경기일 관측 자료만 사용하며, 동명이인/ID 충돌·소속 근거 부재·중복은 별도 처리한다. 과거 `starter_pre_game`은 검증 시 후보와 이전 선발일을 대조하는 자료다. `starter_actual` 대상 경기 행은 정답에만 사용한다. 이전 날짜 결과가 다음 자정에 알려졌다는 기존 가정 및 고정 검증판의 사후 정정 가능성을 보존한다. 당시 공개 시각의 독립 증거는 없다.

<a id="inputs"></a>

## 3. 입력·전처리

`--project-root`, `--run-id`, `--asset` 및 `game`/`season`의 `--season 2025`, `--as-of YYYY-MM-DD`, 선택 `--schedule-snapshot`, `--parent-run`을 사용한다. `game`은 `--game-id`가 필요하며 `evaluate`는 2025년 개발 비교 전용이다. 날짜·경기 ID·팀 ID는 기존 시즌 계약을 따른다. `--parent-run` 사용 시 선행 실행의 해시, 계보, 기준일, 완료 및 잔여 경기 일치를 검증한다.

<a id="outputs"></a>

## 4. 정답·출력 계약

`game`, `season`은 `support-analysis-v1` 결과로 동일 경기 식별자와 as-of, 기존 M1 전망, 팀 기록, 선발 후보 및 일정 난이도를 제공한다. `evaluate`는 선발 평가와 자료 감사 결과를 제공한다. 유효 비율의 분모가 0이면 `null`이다. 공식 발표를 입증할 수 있는 자료가 없어 공식 발표 상태를 부여하지 않는다. 실제 0과 관측되지 않은 값을 구분한다. 경기 전망이 없는 경우 집계에서 누락 수를 표시한다.

<a id="method"></a>

## 5. 모델·학습·선정

- 선발 후보: 이전 28일의 식별 가능한 선발 이력에서 현재 팀에 대한 최신 관측 소속과 이름 일치를 검사한다. 기준 규칙은 이전 선발 후 경과일이 긴 순서, 제안 규칙은 다음 경기일까지의 예상 간격이 5일에 가까운 순서, 관측 부담 및 선발 표본의 결정적 동률 기준을 사용한다. 규칙의 5일은 사전 제안값이며 최적값이 아니다. 기준일 이후의 등판이 없다는 조건부 간격이다. 더블헤더의 두 선발을 서로 다르게 확정하지 않는다. 다음 팀 경기일 및 7일 이내만 후보를 제공한다.
- 전력: 승률 = W/(W+L), 경기당 득점 = RF/G, 경기당 실점 = RA/G, 경기당 득실점차 = (RF-RA)/G. 시즌 및 최근 28일, 홈·원정, 상대별 집계를 분리한다. 작은 표본은 상태로 표시하며 임의 종합 점수를 만들지 않는다.
- 일정: 기존 M1 `predict` 및 정규시즌 `inputs_for`를 사용한다. 관측 상태는 기준일에 고정되며 대상일의 최근 28일 창에 들어오는 관측 이력만 사용한다. 대상일 이전에 예정된 경기의 미래 결과는 모의·입력하지 않는다. 따라서 동적 시즌 반복별 확률의 평균과 다르다. 팀 관점 난이도는 P(패)+0.5×P(무), 팀 집계는 유효 전망 경기의 동일 가중 평균이다. KBO 공식 승률 산식과 다른 난이도 정의이며 미편성 경기나 누락 전망은 분모에서 제외한다.
- 신규 학습·최적화 없음. 규칙 비교 결과를 근거로 성능을 기록하되 최종 Test로 선택하지 않는다.

<a id="lecture-methods"></a>

## 6. 강의 원리 및 프로젝트 확장

### 6.1 강의에서 확인한 원리의 실제 적용

이번 검토에서 강의 전사 원문을 직접 확인하지 않았다. 직접 강의 적용을 주장하지 않는다. 과제 가이드 4쪽의 기준 방법·동일 평가·결과 해석 요구와 별도로, 기존 M1의 학습 및 비교 원리는 해당 M1 기술 문서에서 인수한다.

### 6.2 강의 외 프로젝트 확장 원리

후보 순위 규칙, Top-k 및 Coverage의 분리, 관측 시점별 집계, 기존 확률의 선형 기대 손실 계산은 이번 프로젝트의 계산 설계다. 실제 우열 및 규칙 타당성은 검증 전에는 확정하지 않는다.

### 6.3 주요 미적용 방법과 판단

신규 분류기·회귀기·앙상블·Grid Search·Random Search·GPU 학습은 범위 외이며 실험하지 않는다. 별도 학습 성능 개선을 주장하지 않는다.

### 6.4 강의 및 이론 근거

Term Project 가이드 3~4쪽(검토·설명 및 실험 절차), 기존 M1 및 정규시즌 기술 계약. 강의 전사에 동일 세부 기법이 존재하는지 미확인.

<a id="evaluation"></a>

## 7. 평가 지표·실제 결과

2025년 검증 정규시즌 경기의 양 팀 측을 표본 단위로 사용한다. 정답 ID 유효 수, 후보 제공 수, 제공 비율, 후보 제공+정답 식별의 공통 표본 수, Top-1 및 Top-3 적중 수와 적중률, 전체 식별 가능 정답 기준 적중률을 별도로 기록한다. 후보표 차이 및 제외 사유를 기록한다. 제안 단계에서는 실측 수치가 없다.

<a id="reproducibility"></a>

## 8. 구현·산출물·재현성

`src/simulation/support_analysis/`의 신규 패키지와 `tests/simulation/test_support_analysis.py`를 제안한다. 기존 소스·설정·자료·학습 자산은 변경하지 않는다. 출력은 stdout JSON 및 선택 `--output`의 신규 JSON 파일로 제공하고, 기존 파일 덮어쓰기를 거부한다. 결과에는 입력 자산 계보, 기준일, 규칙 버전 및 실행 ID를 남긴다. 동일 입력의 수치 및 정렬은 결정적이며 실행 ID는 새로 생성된다.

<a id="limitations"></a>

## 9. 검증·한계·후속 연결

사후 일정 가정 또는 공개 진위 미인증 스냅샷, 저장 M1 생성 시점 제한으로 정식 as-of 백테스트가 아니다. 후보 ID 및 선수 소속·관측 부담은 실제 등록/총투구 수를 보장하지 않는다. 원본 판본을 다시 만들 때 사후 정정으로 기존 기록이 달라지는 문제는 별도다. 자료의 당일 행·미래 행 변경이 고정판에서 이전 계산에 영향을 주지 않는지 검사해야 한다. 전용·회귀 테스트, 실제 경기·시즌 실행, 독립 검산 및 최종 수용은 미실행이다. 실제 검증 후 README·공통 계약·개발 흐름의 상태·링크를 갱신한다.

<a id="commands"></a>

## 10. 실행 명령

프로젝트 루트 `D:/Programming/kbo-insight-ai`, 가상환경 `.venv/Scripts/python.exe`에서 실행한다. 실제 명령의 전체 순서는 Work 구현 응답의 검증 블록을 따른다.

- `python -X utf8 -B -m unittest tests.simulation.test_support_analysis`
- `python -X utf8 -B -m unittest discover -s tests`
- `python -X utf8 -B -m src.simulation.support_analysis --help`
- `python -X utf8 -B -m src.simulation.support_analysis --run-id <자료 ID> --asset <M1 경로> evaluate`
- 같은 전역 인자로 `season --season 2025 --as-of 2025-09-01` 또는 `game --season 2025 --as-of 2025-09-01 --game-id <유효 경기 ID>`
- `git diff --check`

이 문서는 제안 상태이며 수용 요청 최종 보고서가 아니다. 실제 적용·검증 결과로 갱신해야 한다.
