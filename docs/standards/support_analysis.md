# 예상 선발투수·팀 전력·일정 난이도 실행과 결과 계약

**KBO Insight AI · 기술 계약·보조 분석 실행**

| 항목           | 내용                                                                                                                           |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| 문서 역할      | 기술 계약·보조 분석 실행                                                                                                       |
| 적용 범위      | 2025년 예상 선발 후보, 관측 팀 기록, M1 기반 일정 난이도                                                                       |
| 확인 기준일    | 2026-11-29 (KST)                                                                                                               |
| 저장소 경로    | `docs/standards/support_analysis.md`                                                                                           |
| 인수 소스 기준 | [`ea4ef45`](https://github.com/Ahn-SooHyun/kbo-insight-ai/commit/ea4ef457fa35dbc37525cd97586a9f51543e31c5)                     |
| 연관 작업      | [#29](https://github.com/Ahn-SooHyun/kbo-insight-ai/issues/29), [#35](https://github.com/Ahn-SooHyun/kbo-insight-ai/issues/35) |
| 자료 이용      | 승인된 비영리 대학 과제용 로컬 처리 및 내부 평가                                                                               |

**판본 보존:** [개정 전 기준 문서 원문](https://github.com/Ahn-SooHyun/kbo-insight-ai/blob/ea4ef457fa35dbc37525cd97586a9f51543e31c5/docs/standards/support_analysis.md)은 역사적 실험·명세·검증의 원본 근거다. 개정 문서는 이전 실행 명세의 파일 지문을 대체하지 않는다.

## 찾아보기

- [1. 목적·지원 범위](#overview)
- [2. 자료·분할·품질](#data)
- [3. 입력·전처리](#inputs)
- [4. 정답·출력 계약](#outputs)
- [5. 모델·학습·선정](#method)
- [6. 강의 원리 및 프로젝트 확장](#lecture-methods)
- [7. 평가 지표·실제 결과](#evaluation)
- [8. 구현·산출물·재현성](#reproducibility)
- [9. 검증·한계·후속 연결](#limitations)
- [10. 실행 명령](#commands)

<a id="overview"></a>

## 1. 목적·지원 범위

2025년 정규시즌 KST 날짜 경계에서 다음 관측 경기의 선발 후보 순위, 팀 기록 및 일정 난이도를 제공한다. 공식 선발 발표, 실제 등록·건강 상태·감독 의도·선발 확률을 추정하지 않는다. 기존 M1 및 시즌 시뮬레이션의 학습·가중치·입력 계약을 변경하지 않는다. 신규 주 예측기는 0개다. 2026년 시즌·보조 분석, 원천 정정·재수집, 라인업·교체·서비스 배포는 범위 외다.

| 기능        | 분석 단위               | 입력                                  | 결과                               | 신규 학습 |
| ----------- | ----------------------- | ------------------------------------- | ---------------------------------- | --------- |
| 예상 선발   | 예정 경기×팀            | 이전 선발 및 관측 소속·등판·투구 부담 | 최대 3명의 결정적 순위와 지원 근거 | 없음      |
| 팀 전력     | 팀×기간×장소/상대       | 기준일 전 완료 경기 점수              | 승무패·승률·득실점·분모            | 없음      |
| 일정 난이도 | 예정 경기×팀 및 팀 집계 | 고정 관측 이력·유효 일정·저장 M1      | 경기 기대 손실 및 유효 경기 평균   | 없음      |

`game`, `season`, `evaluate`의 역할을 분리한다. 동일 시나리오의 경기별·시즌별 공통 결과는 일치해야 한다. `evaluate`는 실제 선발 ID를 정답으로 이용하지만 운영 후보의 입력으로 전달하지 않는다. 기술 수용 및 [PR #33](https://github.com/Ahn-SooHyun/kbo-insight-ai/pull/33) 병합(`53c2d41`), Issue #29 완료 종료를 확인했다. 당시 실험·형식 오류와 이후 수용을 구분한다.

<a id="data"></a>

## 2. 자료·분할·품질

### 2.1 고정 자료와 원천 역할

표준 검증판 `e020dafedc8b4fd9ac6696d0e4a92e:37b56c39caf5435e8fb2a17f993ec4df`, 모델 자료 `259b38decaac4ed68dd5a596cfbe4849`, M1 `0d9433b8044e48e5a4d21bbdb9b432e5`. [자료 계보](execution_handover.md#data)와 `verify_assets`, `read_completed`, `load_scenario`의 기존 검증을 사용한다.

| 자료                                       | 역할                          | 확인 범위·주의                                                                                                 |
| ------------------------------------------ | ----------------------------- | -------------------------------------------------------------------------------------------------------------- |
| 표준 `eligible_games`                      | 완료 경기와 팀·점수·경기 번호 | 기준일 이전 유효 정규시즌 기록                                                                                 |
| 모델 `starter_evidence`                    | 원천 후보 기록 감사           | 37,810행; 이름·ID 충돌, 실제 공식 발표 아님                                                                    |
| 모델 `starter_actual`                      | 실제 선발 정답                | 31,579행; 대상일 정답은 평가 전용                                                                              |
| 모델 `starter_history`                     | 과거 선발 이력                | 5,758행; 이전 날짜·ID·팀·증거 확인                                                                             |
| 모델 `starter_pre_game`                    | 기존 후보·이전 선발일 대조    | 36,153행; 예정 경기 후보로 복제 금지                                                                           |
| 모델 `player_identity_evidence`            | 관측 이름·소속 근거           | 동일 시즌 및 시점 이전 자료, 실제 참여 팀과 일치                                                               |
| 표준 `player_games`, `player_admission`    | 관측 투구수·등판 근거         | pitcher, preliminary_candidate=1, direct/team mismatch 및 game_state_uncertain=0; 사후 최종 identity 판정 제외 |
| 모델 `m1_index`, `m1_inputs`, `m1_targets` | 기존 경기·분할·자산 대조      | 2025 평가의 유효 경기 집합과 완료 경기 집합 대조                                                               |

위 전체 행 수는 2025년 개발 평가의 모집단 수가 아니다. 정답·후보 키는 `(game_id, team_id)` 및 `(game_id, team_id, player_id)`다. 유효 기록의 중복·날짜·소속 불일치는 실패로 처리하며 수동 이름 연결이나 정답 보충을 금지한다.

### 2.2 시간 경계 및 정답 사용

기준일 t는 KST 자정, 경기 날짜 d<t인 관측만 입력에 사용한다. 시즌 창은 해당 연도 1월 1일부터 t 직전, 최근 28일은 [t−28,t), 투구 부담은 [t−7,t)다. 같은 날짜의 먼저 끝난 경기라도 입력에서 제외한다. 실제 당시 공개/수정 시각 및 과거 모델 이용 가능성은 입증되지 않았다. `evaluate`는 2025-01-01 이상 2026-01-01 미만의 각 경기일을 cutoff로 사용한다. 2025년은 개발 선택 자료이며 2026년 최종 Test에 해당하지 않는다. 자료 감사와 성능 평가의 실제 범위를 구분한다.

### 2.3 결측·충돌·관측 상태

| 조건                                              | 처리                                           |
| ------------------------------------------------- | ---------------------------------------------- |
| 정답의 이름/ID/증거 부족                          | 평가 제외 및 이유 집계                         |
| 후보 이름 충돌·팀 근거 부재·마지막 관측 소속 충돌 | 후보 제외, `exclusions` 기록                   |
| 관측 투구 행 없음                                 | 합계 `null`, `unobserved`                      |
| 수치 결측 또는 확인된 선발의 대응 관측 행 누락    | `partial_observed`, 누락 행과 수치 결측을 분리 |
| 실제 기록 0                                       | 유효 0, 미관측과 구별                          |
| 관측된 유효 행의 합                               | `observed_rows_only`; 공식 총투구 수 아님      |
| 공개 시각·경기·선수 ID 연결 근거 없음             | 공식 발표 null 및 사유                         |

관측 부담은 모든 구원 등판을 포함하는 공식 총투구 수가 아니다. 최신 관측 소속은 현재 등록이나 선수의 건강·실제 가용성을 나타내지 않는다.

<a id="inputs"></a>

## 3. 입력·전처리

| 인자·명령             | 필요 여부 | 의미                                                  |
| --------------------- | --------- | ----------------------------------------------------- |
| `--project-root`      | 선택      | 절대 루트, 기존 설정 탐색                             |
| `--run-id`            | 필수      | 검증 모델 자료 실행                                   |
| `--asset`             | 필수      | 기존 M1 `model.json`                                  |
| `--output`            | 선택      | results 아래 신규 JSON; 덮어쓰기 거부                 |
| `game`                | 하위 명령 | `--season 2025`, `--as-of`, `--game-id`               |
| `season`              | 하위 명령 | `--season 2025`, `--as-of`                            |
| `evaluate`            | 하위 명령 | 2025년 개발 비교; 시즌/기준일 인자 없음               |
| `--schedule-snapshot` | 선택      | game/season용 기존 시즌 일정 계약                     |
| `--parent-run`        | 선택      | game/season용 완료 정규시즌 실행의 시나리오·계보 확인 |

날짜·경기·팀 키와 지원 범위를 검증하고, 부모가 있으면 산출물·지문·자료·기준일·완료 경기·잔여 일정의 일치를 확인한다. 자료 조회는 읽기 전용이며 부모 반복을 재학습하거나 재시뮬레이션하지 않는다. 부모/자산 계약 손상은 일반 미지원으로 숨기지 않는다.

<a id="outputs"></a>

## 4. 정답·출력 계약

v1.1.0의 `support-analysis-v1` 결과는 다음 계약을 유지한다.

| 주요 키                                                                       | 의미                                               |
| ----------------------------------------------------------------------------- | -------------------------------------------------- |
| `schema`, `version`, `scope`, `status`                                        | 계약·버전·game/season/starter_validation·실행 상태 |
| `season`, `as_of_kst`, `metadata` 또는 `provenance`                           | 시점·시나리오·자료/모델 계보                       |
| `starter_selection`                                                           | 운영 baseline 및 개발 선정 근거                    |
| `games`, `teams`, `missing_pairs`                                             | 경기·팀 기록·미편성 팀 쌍                          |
| `parent_regular_season`                                                       | 선택 부모 검증 정보 또는 null                      |
| `rule_versions`, `new_primary_models`                                         | 계산 규칙 및 신규 주 예측기 0                      |
| `comparison`, `by_team`, `candidate_reference_audit`, `data_audit`, `samples` | evaluate의 비교·감사                               |
| `run_id`, `created_utc`, `source_sha256`                                      | 실행 및 소스 지문                                  |
| `formal_backtest_supported`, `final_test_used`                                | false: 2025 사후 가정과 독립 Test 미사용           |

`game`의 `teams`는 해당 참가 팀들의 전체 잔여 일정 집계와 기록도 포함한다. `fixture`, `m1_forecast`, `starters`, `difficulty`는 같은 경기 ID를 사용한다. `workload_missing_known_starts`, `workload_status`는 확인된 선발의 대응 관측 행 누락과 실제 수치 결측을 구분한다.

| 지원 상태                                          | 해석                                              |
| -------------------------------------------------- | ------------------------------------------------- |
| `estimated`, `estimated_limited`                   | 후보 지원 또는 제한 지원; 공식 발표 아님          |
| `unsupported`                                      | 다음 관측 경기일 아님, 전망일 초과 또는 이력 없음 |
| `supported`, `limited_sample`, `no_observations`   | 팀 기록 5경기 이상, 1~4, 0경기                    |
| `supported_under_assumption`                       | 가정한 편성과 전망이 완전                         |
| `partial`, `schedule_missing`, `no_valid_forecast` | 부분 지원·미편성·유효 전망 없음                   |
| `no_remaining_games_under_assumption`              | 기대 잔여 0, 난이도 null                          |
| `publication_evidence_unavailable`                 | `official_announcement=null`                      |
| `win_rate_reason=no_decided_games`                 | 승률 분모 0                                       |
| `run_rate_reason=no_observations`                  | 득실점 분모 0                                     |

null은 임의 0이 아니다. 실제 지문·스키마·계산 오류는 실행 실패로 전파한다.

<a id="method"></a>

## 5. 모델·학습·선정

### 5.1 예상 선발: 후보 집합과 사전식 순위

기준일 t, 대상일 d≥t, 팀 j의 후보 집합 C_j(t)는 이전 28일 식별 선발 이력과 최신 관측 소속·이름 검사를 통과한 선수다. 최근 해당 팀 선발일 s_i, 마지막 관측 등판일 a_i, 관측 투구수 q_i, 결측 표시 m_i, 이전 28일 선발 수 n_i를 정의한다. 운영 기준 정렬은 다음과 같다.

$$K_B(i)=(-(d-s_i),\ s_i,\ id_i).$$

비교 제안 규칙은 다음 사전식 키를 사용했다.

$$K_P(i)=(\max(0,2-(d-a_i)),\ |(d-s_i)-5|,\ m_i,\ q_i,\ -n_i,\ id_i).$$

각 항목의 우선순위 비교이고 가중 선형 점수나 확률 모델이 아니다. 결측 q_i는 정렬용 0을 쓰더라도 결측 플래그가 선행하고 실제 출력은 null을 보존한다. 2일·5일은 사전 제안값이지 최적 계수 또는 학습된 파라미터가 아니다. 두 규칙은 동일 후보 집합, 최대 `min(3, len(C))`를 사용한다. v1.0.0 제안 순위를 비교했고 v1.1.0은 2025년 공통 평가에서 우세한 K_B를 선택했다. 운영 순위는 투구 부담을 가중치로 사용하지 않는다.

관측 일정상 다음 팀 경기일 및 0≤d−t≤7에서만 후보를 제공한다. 중간 등판이 없다는 조건부 날짜 차이며 실제 휴식일을 확정하지 않는다. 더블헤더 두 경기의 서로 다른 선발을 공동 배정하지 않는다.

### 5.2 팀 전력: 기술통계

완료 경기 G=W+L+T, 득점 RF, 실점 RA. 승률과 경기당 지표는 다음과 같다.

$$\mathrm{WinRate}=\frac{W}{W+L},\quad RF_G=\frac{RF}{G},\quad RA_G=\frac{RA}{G},\quad RD_G=\frac{RF-RA}{G}.$$

승률에서 무승부를 제외하지만 경기당 득실점은 모든 유효 완료 경기를 분모로 한다. 시즌·최근 28일·장소·상대별 기록을 분리하고 전체 및 분할 집계의 일관성을 검사한다. 5경기 미만 경고는 경험적 지원 기준이며 통계적 유의성 임계값이 아니다. 예를 들어 합성 5:2 승, 1:4 패, 3:3 무는 G=3, 승률 0.5, RF=RA=9, 경기당 각 3.0이다. 무승부만 있다면 승률 null이나 경기당 득실점은 계산할 수 있다. 상대 효과의 인과 추정이 아니다.

### 5.3 일정 난이도: 기대 손실

M1의 홈승 p_H, 무 p_T, 원정승 p_A는 유한 [0,1]이며 합 1(절대 허용오차 1e-12)이어야 한다. 팀 관점 승·무·패를 대응시키고 손실을 0, 0.5, 1로 두면 경기 g의 난이도는

$$D_{jg}=E[\ell(Y_{jg})]=p_L+\tfrac12p_D.$$

유효 전망 V_j개에서

$$D_j=\frac{\sum_{g\in F_j^*}D_{jg}}{V_j},\qquad V_j>0.$$

V_j=0이면 null이다. 편성 S, 미편성 U, 유효 전망 V를 구별한다. 기대 잔여 S+U, 전망 누락 S−V, 편성 제공률 V/S, 기대 잔여 기준 V/(S+U). 홈·원정 난이도 합 1은 같은 확률 분포에서 도출된다. 합성 홈승 0.6, 무 0.1, 원정승 0.3이면 홈 0.35, 원정 0.65다. 무승부 가중치 0.5·동일 경기 가중은 설계값이며 KBO 승률과 다르다.

`inputs_for`와 저장 M1을 사용하고 관측 상태는 기준일 이전 완료 경기로 고정한다. 최근 28일 창은 대상 경기일에 따라 이동한다. 그 사이 예정 경기의 미래 실제/모의 결과를 추가하지 않으므로 동적 시즌 반복의 평균과 다르다.

<a id="lecture-methods"></a>

## 6. 강의 원리 및 프로젝트 확장

### 6.1 강의 직접 적용·원리 응용

| 확인 전사·시각                                                                                                   | 구분                     | 실제 적용·코드                                     | 제한                                                       |
| ---------------------------------------------------------------------------------------------------------------- | ------------------------ | -------------------------------------------------- | ---------------------------------------------------------- |
| `ml_cv,eval-metric_빨강펜_SW중단.txt` 00:00:44.740–00:01:01.500                                                  | 공정 평가 원리 직접 적용 | `evaluation.py`의 동일 후보·정답·공통 1,429측 비교 | 2025 선택 자료, 일반화 보장 아님                           |
| 동일 전사 00:03:40.310–00:04:19.720                                                                              | 과적합/복잡도 원리 응용  | 복잡한 제안 순위의 낮은 결과를 보존하고 기준 선택  | Bias–Variance 분해·학습 곡선 미실행, 과적합 원인 입증 아님 |
| 동일 전사 00:00:54.460–00:01:01.500, 00:03:56.090–00:04:19.720                                                   | 교차검증 취지 응용       | 각 경기일 이전 후보 재구성                         | 무작위 K-fold 자체는 미실행, 공식 당시 평가 아님           |
| `LinearRegression_빨강펜-seg1.txt` 00:11:00.580–00:11:14.480                                                     | 손실 개념 응용           | 0/1 적중 오류 및 승·무·패 기대 손실                | 선형회귀/MSE 최적화 없음                                   |
| `LogisticRegression_update(1)_빨강펜.txt` 00:00:59.440–00:01:07.520                                              | 확률 개념 응용           | `team_probabilities`, `difficulty_value`           | M1 `profile_histogram` 재사용; 로지스틱 신규 fit 아님      |
| `ensemble(2)_빨강펜_updated.txt` 00:00:44.520–00:00:49.240, 00:00:53.860–00:01:01.580, 00:07:52.110–00:08:04.290 | 확인했으나 미적용        | 결정적 정렬·통계·확률 변환                         | 복원추출·Bagging·RF를 실행하지 않음                        |

이전 기술 문서의 전사 확인 시각을 인수한다. 이번 재작성에서 전 강의 원문·영상·슬라이드를 다시 전수 확인한 것은 아니다. [공통 강의 기록](../planning/development_workflow.md#lecture-methods).

### 6.2 강의 외 프로젝트 확장

| 확장 기법                   | 이론 및 적용 이유                 | 구현·실제 검증                                                                                     | 한계                               |
| --------------------------- | --------------------------------- | -------------------------------------------------------------------------------------------------- | ---------------------------------- |
| 후보 검색·사전식 정렬       | 우선순위별 비교와 ID 동률 결정    | `rank_starters`, `baseline_key`, `proposed_key`, 후보 동일성·순위·경계 검사                        | 선발 확률·로테이션 최적화 아님     |
| 시점 제한                   | 미래 입력을 제거한 관측 가능 집합 | `read_starter_data`, 당일/미래 정답·이름·부담·품질 SQLite 변이                                     | 실제 공개 시각 미입증              |
| Top-k 및 Coverage           | 후보 제공과 정답 적중 분리        | `method_metrics`, `compare_samples`의 유효·전체·공통 분모                                          | calibration 아님                   |
| 부담 관측 누락              | 알려진 선발과 관측 행의 차집합    | `workload_missing_known_starts`, `workload_status`                                                 | 공식 총투구수·건강 추정 아님       |
| 기간·장소·상대별 집계       | 동일 경기 기록의 기술통계         | `team_window`, `record_summary`, 10팀 실자료 대조                                                  | Stratified K-fold나 인과 추정 아님 |
| 기대 손실 및 동일 가중 평균 | 이산 확률 기대값의 선형성         | `difficulty_value`, `schedule_difficulty`, 두 팀 합·분모 검증                                      | 무승부 손실은 설계 선택            |
| 미편성·미지원·0 분모        | 미완전한 적격 집합의 별도 상태    | 빈 일정·누락·무승부 전용·중복 테스트                                                               | 실제 당시 공개 일정 완전성 아님    |
| 계보·SHA-256·독립 재현      | 입력·코드·결과의 결정적 비교      | `verify_assets`, `verify_parent_run`, `source_hashes`, `save_result`; 3쌍 및 개행 보완 뒤 3건 비교 | 원천 사실의 진실성 인증 아님       |

확인 강의 발췌에서 동일 상세 기법이 직접 제시된 사실을 확인하지 못해 프로젝트 확장으로 분리한다. 전 강의에서의 부재를 단정하지 않는다.

### 6.3 비교 후 미채택 및 미실험

제안 순위는 실제 실험 후 미채택했다. 선형·로지스틱 회귀·KNN·SVM·결정 트리 신규 학습, RF·Bootstrap·OOB·Bagging·Boosting·Voting·Stacking, 무작위 K-fold/Stratified K-fold, Grid/Random Search·GPU·SHAP, 강화학습·군집화는 이번 기능에서 미실험이다. 후보 및 기술통계·기존 M1 연결의 문제를 별도 학습으로 바꾸지 않는다.

과제 가이드 3쪽의 AI 활용·설명 가능성과 4쪽의 문제 정의·자료 준비·기준 비교·결론을 적용한다. 신규 학습, 별도 EDA 시각화·학습 곡선·튜닝이 완료됐다는 뜻은 아니다. 해당 요건의 프로젝트 전체 충족은 최종 제출물에서 별도 검토한다.

<a id="evaluation"></a>

## 7. 평가 지표·실제 결과

### 7.1 분모와 지표

2025년 개발 비교는 720경기, 1,440팀 측, 정답 ID 식별 1,439, 후보 제공 1,430, 공통 평가 1,429측이다. 후보 없음 10측, 정답 없음 1측. 전체 A, 식별 정답 T, 후보 제공 P_m, 유효 E_m=T∩P_m, 공통 E=T∩P_B∩P_P를 정의한다.

$$Coverage_m=|P_m|/|A|,\quad TruthCoverage=|T|/|A|.$$

$$Hit@k_m=\frac{\sum_{a\in E_m}\mathbf{1}[y_a\in Top_k(m,a)]}{|E_m|},\quad AllIdentifiedHit@k_m=\frac{\sum_{a\in E_m}\mathbf{1}[y_a\in Top_k(m,a)]}{|T|}.$$

공통 비교에서는 분자와 분모를 E로 제한한다. 분모 0이면 null.

| 지표                 |              기준 규칙 |            제안 규칙 |
| -------------------- | ---------------------: | -------------------: |
| 후보 제공            | 1,430/1,440 (99.3056%) |                 동일 |
| 유효 Top-1           |   269/1,429 (18.8244%) | 258/1,429 (18.0546%) |
| 유효 Top-3           | 1,056/1,429 (73.8978%) | 962/1,429 (67.3198%) |
| 전체 식별 정답 Top-1 |   269/1,439 (18.6935%) | 258/1,439 (17.9291%) |
| 전체 식별 정답 Top-3 | 1,056/1,439 (73.3843%) | 962/1,439 (66.8520%) |
| 제공 후보 수 합      |                  4,258 |                4,258 |

기준 대비 제안 변화는 Top-1 −0.7698%p(11건), Top-3 −6.5780%p(94건). 두 후보 집합이 같으므로 제공률은 변화하지 않았다. 18.82% Top-1도 낮아 확정 선발로 해석할 수 없다. 통계적 유의성·신뢰구간·독립 팀/시기 일반화는 확인하지 않았다. 경기 당일 자정의 과거 재구성 개발 성능이며 7일 전 전망의 동일 성능 또는 2026년 적중률이 아니다.

### 7.2 후보 감사 및 대표 사례

기존 후보표·재구성 후보 각각 8,965행, 후보 집합 1,440/1,440측 일치, 이전 선발일 8,965/8,965행 일치, 양쪽 전용 후보 0. 자료판의 검사 결과다. v1.0.0 대표 `20250902HTHH0`의 제안 순위는 한화 폰세·황준서·문동주, KIA 이의리·네일·김도현이었다. 운영 v1.1.0의 같은 경기(기준 2025-09-01, 완료 622경기) 사용자 결과는 다음과 같다.

| 팀(ID)     | 순위 | 선수   | 이전 선발일 | 최근 관측 투구수 | 부담 상태            | 알려진 선발 관측 누락 |
| ---------- | ---: | ------ | ----------- | ---------------: | -------------------- | --------------------: |
| 한화(2010) |    1 | 엄상백 | 2025-08-09  |             null | `unobserved`         |                     0 |
| 한화(2010) |    2 | 조동욱 | 2025-08-20  |               62 | `observed_rows_only` |                     0 |
| 한화(2010) |    3 | 류현진 | 2025-08-26  |             null | `unobserved`         |                     1 |
| KIA(2001)  |    1 | 김도현 | 2025-08-27  |             null | `unobserved`         |                     1 |
| KIA(2001)  |    2 | 이의리 | 2025-08-28  |               78 | `observed_rows_only` |                     0 |
| KIA(2001)  |    3 | 네일   | 2025-08-29  |             null | `unobserved`         |                     1 |

2025-09-01 시즌 전망의 후보 제공 대상 10팀 측은 경기 당일 개발 평가 1,430팀 측과 다른 범위다. 대표 사례를 실제 적중으로 주장하지 않는다. 팀·M1·일정 난이도 계산은 보완 전후 일치했다는 사용자 검산을 인수한다.

<a id="reproducibility"></a>

## 8. 구현·산출물·재현성

### 8.1 모듈 책임

| 경로                                                   | 역할                             |
| ------------------------------------------------------ | -------------------------------- |
| `src/simulation/support_analysis/__init__.py`          | 버전·규칙·선정 정보              |
| `src/simulation/support_analysis/data.py`              | 날짜 제한 읽기 전용 조회·키 검증 |
| `src/simulation/support_analysis/analysis.py`          | 후보·순위·기록·난이도·종합 결과  |
| `src/simulation/support_analysis/evaluation.py`        | 실제 정답·후보 감사·개발 비교    |
| `src/simulation/support_analysis/__main__.py`          | CLI·자산/부모 검사·새 결과 저장  |
| `tests/simulation/test_support_analysis.py`            | 초기 전용 51개                   |
| `tests/simulation/test_support_analysis_completion.py` | 보완 전용 12개                   |

### 8.2 실행 이력 및 지문

초기 결과 폴더 `results/support_analysis/4710d37c809d42e4b926c338306033bd`, 부모 `e24254420aee4d52acf2b1fb144c3d58`. 초기와 보완 사용자 로그의 ID 및 결과 SHA-256을 보존한다. 보완 폴더 `results/support_analysis/dd219be0e5b743cf948ea4bb2b80df40`; 개행 보완 뒤 `results/support_analysis/a6fd3f6a408a4a5a8316e282d52e86a2`.

| 실행           | ID                                 | SHA-256                                                            |
| -------------- | ---------------------------------- | ------------------------------------------------------------------ |
| 초기 선발 평가 | `5f5145d8c2734f03ac2a2334ca42319a` | `3164a1a027f37c1d8d5e36f40de992ecb78c957b06da7851a6702a68b3f5f65a` |
| 초기 시즌      | `1556b02ec2b34d0c8f4b7fdaa2d2201d` | `914aa520dd783eb111f0e1afe8c0da6b1842d7e6a9f27c51bbb68801d461cd3d` |
| 초기 경기      | `87439a9a1edd40a485a5847f1e30fee9` | `84cb46141feaa1f3fb6bd2bbb9e02dd1687662b1e68eaab7a0a6d33d2fc37ce7` |
| 보완 평가 1    | `9225957827f641d1b02fba21ce3c50bb` | `047959d360f6f2b83d9d8e8fb63c6a9604c5d027f32a62481e80f485b46f4311` |
| 보완 평가 2    | `f78782a710bc42dd84b1c92bec9908b3` | `22151d49141d6fa8a1658af0c69bec62f9f30e67d728b09eb4572cee03e312f9` |
| 보완 시즌 1    | `e49fcdb9bca646658df63e811ec9a3d7` | `3691eb6321541f88aa1b23feba8a31aeeec8e527d0e5c42a91549c269295f793` |
| 보완 시즌 2    | `b1f2f7fcb96e4a4e8f63f6df0940bbab` | `773e10ab0303a9241e78107816f329ebe94a5dc0e996f02be2c141ebd6209a7b` |
| 보완 경기 1    | `077bcade8d45446fa71526e6fe160115` | `eabd2a85d8df64788d259ea2cad3ad1a007234b00cab04001741fb3a846e80b9` |
| 보완 경기 2    | `3260e6c721a94b168c294370fe9f0c55` | `525a9c00922e7b43f5da4ccbaf36fe5d78a473f7c9d734048cc9aacc0933177f` |
| 개행 보완 평가 | `6000a2e5d84f4575ba1060fbd5cba7d3` | `17b82f69cb5441aceff54af264b4df2060d1d02fce6605da5898eb6d0e42b93a` |
| 개행 보완 시즌 | `acb4f0d745c6487fa900ede907d6f9cc` | `45dc5de1ed8a3015dadf3ee5328d0d4e692fee99ccaf0c755c97ac7d902f76c9` |
| 개행 보완 경기 | `17e8e08587ce4043b5af401f1df9ed67` | `7d3624a545f18ad74e3d94fce40281014bf65fcb133e3afde0a05abfb34fe04d` |

보완 3쌍은 실행 ID·시각만 제외한 계산 결과가 일치했다. EOF 보완 뒤 3건은 ID·시각·소스 지문을 제외한 결과가 일치했고 새 소스 5개 지문을 별도로 검산했다. 사용자 로그 인수 결과이며 이 문서 작업에서 실제 JSON/DB를 새로 읽어 해시를 계산한 것이 아니다. 초기 `25140a0`, 보완 `421254f`, 형식 `3959da8`, 수용 전 원격 `af2947e`, 병합 `53c2d41`의 역사 판본을 구분한다.

<a id="limitations"></a>

## 9. 검증·한계·후속 연결

| 검증                | 당시 실제 결과                                                       |
| ------------------- | -------------------------------------------------------------------- |
| 초기 전용·전체      | 51/51, 580/580                                                       |
| 보완 전용·전체      | 63/63, 592/592                                                       |
| 실자료 반복         | 평가·시즌·경기 각각 2회, 계산 결과 3쌍 일치                          |
| 누수 변이           | 대상일 실제 선발·미래 이름/등판/부담/품질 변화에 이전 입력·순위 보존 |
| 독립 검산           | 팀 기록·일정 난이도·분모·후보·M1 결과 대조                           |
| EOF·형식            | Python 7/7 AST·EOF·후행 공백 통과                                    |
| 개행 후 회귀        | 63/63(0.020초), 592/592(9.976초)                                     |
| 개행 후 실자료      | 3건 계산 일치 및 새 소스 지문 대조                                   |
| 당시 문서 링크 검사 | README 링크 누락으로 Exit 1; 나머지 검사 미도달이라는 역사 기록      |
| 현재 GitHub         | PR #33 실제 병합·Issue #29 완료 종료 확인                            |

과거 문서 검사 실패는 당시 실행의 사실로 보존한다. 현재 문서 개정의 링크·실행·Git 검증은 별개다. 2025년 개발 비교와 공식 실제 당시 공개 자료의 독립 성능을 혼동하지 않는다. 관측 부담·등록·건강·공식 발표는 제한된다. 미편성·더블헤더·중간 등판 및 원천 정정 가능성을 유지한다. 2026년 시즌 및 보조 분석은 미지원이다. [시즌 검산](season_forecast_validation.md), [실행 인수](execution_handover.md)에 연결한다.

<a id="commands"></a>

## 10. 실행 명령

Windows PowerShell 예시이며 실제 소스 및 자산 위치를 사용한다. 새 결과 경로에만 저장하고 기존 모델·DB를 재학습·수정하지 않는다. 선택 부모의 실행 ID는 같은 시나리오의 실제 완료 정규시즌 결과여야 한다.

```powershell
Set-Location -LiteralPath 'D:/Programming/kbo-insight-ai'
$root = (Get-Location).Path
$python = Join-Path $root '.venv/Scripts/python.exe'
$runId = '259b38decaac4ed68dd5a596cfbe4849'
$asset = 'models/m1/0d9433b8044e48e5a4d21bbdb9b432e5/model.json'
$parentRun = 'e24254420aee4d52acf2b1fb144c3d58'
$out = 'results/support_analysis/' + [guid]::NewGuid().ToString('N')
$common = @('-X','utf8','-B','-m','src.simulation.support_analysis','--project-root',$root,'--run-id',$runId,'--asset',$asset)
$scenario = @('--season','2025','--as-of','2025-09-01','--parent-run',$parentRun)
& $python -X utf8 -B -m unittest tests.simulation.test_support_analysis tests.simulation.test_support_analysis_completion
if ($LASTEXITCODE -ne 0) { throw '전용 테스트 실패' }
& $python @common --help
if ($LASTEXITCODE -ne 0) { throw 'CLI 확인 실패' }
& $python @common --output "$out/evaluation.json" evaluate
if ($LASTEXITCODE -ne 0) { throw '선발 평가 실패' }
& $python @common --output "$out/season.json" season @scenario
if ($LASTEXITCODE -ne 0) { throw '시즌 분석 실패' }
& $python @common --output "$out/game.json" game @scenario --game-id '20250902HTHH0'
if ($LASTEXITCODE -ne 0) { throw '경기 분석 실패' }
& $python -X utf8 -B -m unittest discover -s tests
if ($LASTEXITCODE -ne 0) { throw '전체 회귀 실패' }
git diff --check HEAD
if ($LASTEXITCODE -ne 0) { throw 'Git 형식 검사 실패' }
Write-Host "result_directory=$out"
```

새 실행의 JSON 상태·계보·입력·분모를 확인하고 동일 자료·코드·인자로 다시 실행할 경우 ID·시각과 실제 계산을 구분한다. 소스가 달라졌다면 그 지문을 별도 검증하며 기존 결과를 덮어쓰지 않는다.
