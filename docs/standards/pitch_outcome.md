# 다음 투구 구종·결과 예측 실행과 결과 계약

실제 다음 공 직전의 관측 상태와 과거 기록으로 구종·결과 공동 확률 및 두 주변 확률을 계산한다.

| 항목        | 내용                                                                                                                                                            |
| ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 문서 역할   | 모델 구현·학습·평가·운영 재현 계약                                                                                                                              |
| 적용 범위   | M5 정규시즌 실제 투구의 구종·결과, CPU 로컬 학술용                                                                                                              |
| 검토 기준일 | 2026-11-27 KST                                                                                                                                                  |
| 저장소 경로 | `docs/standards/pitch_outcome.md`                                                                                                                               |
| 선행 기준   | `f40ed33841a724d15aa9dcaaf053c3b47feb2f09` (#19 병합)                                                                                                           |
| 보완 기준   | [`33ba651f996d680a7e3b05276f1aab0e37057b5e`](https://github.com/Ahn-SooHyun/kbo-insight-ai/commit/33ba651f996d680a7e3b05276f1aab0e37057b5e)                     |
| 관련 Issue  | [#20 다음 투구 구종·결과 예측](https://github.com/Ahn-SooHyun/kbo-insight-ai/issues/20)                                                                         |
| 상태        | v1.0.0 원격 구현·사용자 실데이터 학습 및 내부 자산 검증 확인; 외부 JSON 비교 명령 실패. v1.1.0 보완 코드는 제공 단계이며 적용·실행·선정 결과·하위 수용은 미확인 |

관련 문서: [README](../../README.md) · [공통 예측·평가 계약](project_contract.md#m5) · [개발 흐름 및 강의 기준](../planning/development_workflow.md#lecture-methods) · [타석 결과 예측](plate_appearance.md)

찾아보기: [목적](#overview) · [자료](#data) · [입력](#inputs) · [출력](#outputs) · [학습](#method) · [강의 원리](#lecture-methods) · [평가](#evaluation) · [재현](#reproducibility) · [한계](#limitations) · [명령](#commands)

<a id="overview"></a>

## 1. 목적·지원 범위

해당 공을 던지기 전 시점에서 구종과 투구 결과의 공동 분포를 분류하고, 그 분포의 축별 합으로 구종 및 결과 확률을 일관되게 제공한다. 결과의 확률은 현재 공의 실제 구종을 입력한 조건부 예측이 아니다. 단일 운영 분류기만 저장한다. 구속·투구 위치·좌표, 투구 궤적, 감독 의도, 무투구 자동 판정의 투구 예측은 제외한다. 제공하는 확률은 특정 결과의 보장이 아니다. 원본 및 학습 산출물은 승인된 비영리 로컬 학술용 범위에서 처리한다.

<a id="data"></a>

## 2. 자료·분할·품질

| 항목            | 인수 기준                                                                       |
| --------------- | ------------------------------------------------------------------------------- |
| 모델 자료 실행  | `259b38decaac4ed68dd5a596cfbe4849`                                              |
| 모델 자료 버전  | `1.2.0`                                                                         |
| 모델 DB         | `data/processed/model_data/259b38decaac4ed68dd5a596cfbe4849/model_data.sqlite3` |
| 모델 DB SHA-256 | `c8383a9d3e8930830a137d1b6791cbb26e2316dbbbdf4f0f436ceda9a43890d0`              |
| 표준 실행       | `e020dafedc8b4fd9ac6696d0e4e4a92e`                                              |
| 표준 검증       | `37b56c39caf5435e8fb2a17f993ec4df`                                              |
| Train           | 2025-01-01 이전, 442,636구 / 1,440경기                                          |
| Validation      | 2025-01-01 이상 2026-01-01 미만, 216,869구 / 720경기                            |
| 최종 Test       | 2026-01-01 이상 2026-11-26 미만, 후속 독립 평가                                 |

날짜는 KST 기준이다. `m5_index`, `m5_inputs`, `m5_targets`의 `sample_id`를 일대일 대조한다. 동일 경기의 분할 및 날짜 혼합을 거부한다. 모든 인덱스의 키와 분할을 확인하되 최종 Test의 정답 값은 개발 평가에서 조회하지 않는다. 모델 DB와 표준 검증 DB는 읽기 전용으로 연결하며 매니페스트·계약·EDA·보완 영향 자료 및 지문을 대조한다.

현재 공의 표준 `pitches` 행과 `eligible_states` 경계, 참가자, 순번, 직전 상태, 코드 및 원본 식별자를 비교한다. 표준의 `non_pitch`는 실제 공의 분모에 포함하지 않는다. 실제 공의 배타적 제외 사유는 경기 제외 → 타석 상태 제외 → 투구 번호 → 순서 → 상태 → 정답 코드 순이다. `game_checks`의 원천 양의 투구 수와 표준에 저장된 투구 수의 차이를 별도 집계하며 이를 임의의 한 제외 원인으로 단정하지 않는다.

직전 공은 같은 경기·타석에서 순번이 1 작은 실제 공이며 원본 지문·행 순서가 맞고 순서 및 정답 코드가 유효한 경우에만 사용한다. 이전 공 자체가 상태 품질 때문에 모델에서 제외됐더라도 원천 조건을 충족하면 기존 자료 계약상 직전 코드는 사용할 수 있다. 첫 공, 중간 결측, 원천 순서 오류, 정답 무효 및 이전 공의 모델 제외를 구별한다. 무투구 자동 판정을 투구나 직전 기록으로 세지 않는다.

### 2.1 v1.0.0 사용자 실행에서 확인한 품질

| 항목                         |                 Train |            Validation |
| ---------------------------- | --------------------: | --------------------: |
| 실제 관측 날짜               | 2023-04-01~2024-10-01 | 2025-03-22~2025-10-04 |
| 표준 실제 투구               |               443,072 |               217,849 |
| 유효 실제 투구               |               442,636 |               216,869 |
| 유효 경기                    |                 1,440 |                   720 |
| 제외 실제 투구               |                   436 |                   980 |
| 타석 상태 제외               |                   427 |                   980 |
| 정답 코드 제외               |                     9 |                     0 |
| 별도 무투구                  |                   188 |                   122 |
| 첫 공                        |               113,508 |                55,768 |
| 전체 직전 구종 결측          |               113,516 |                55,768 |
| 중간 직전값 결측             |                     8 |                     0 |
| 관측 구종 / 결과 / 공동 조합 |            8 / 7 / 53 |           10 / 7 / 61 |

원천 검사 대상 경기의 양의 투구 수와 표준 투구 수의 차이는 두 분할 모두 0이었다. Train의 중간 결측 8구는 `previous_target_invalid`였다. 이전 구종·결과 결측을 타석 밖 기록이나 미래 값으로 보완하지 않는다. 위 집계는 사용자 실행 로그에 근거하며 새 판본에서 재대조한다.

### 2.2 v1.1.0 시간순 개발 구간

| 구간            | 날짜 경계                        | 용도                        | 실측 건수           |
| --------------- | -------------------------------- | --------------------------- | ------------------- |
| Train           | 2025-01-01 이전                  | 특징 어휘·빈도·계수 적합    | 인수 기준 442,636구 |
| 보정            | 2025-01-01 이상, 2025-07-01 미만 | Temperature 적합만 수행     | 신규 실행 대기      |
| 선정            | 2025-07-01 이상, 2025-09-01 미만 | 고정 기준에 따른 후보 선정  | 신규 실행 대기      |
| 추가 평가       | 2025-09-01 이상, 2026-01-01 미만 | 선정 완료 뒤 비교           | 신규 실행 대기      |
| 전체 Validation | 2025년                           | 전체 개발 비교 및 기술 통계 | 인수 기준 216,869구 |
| 최종 Test       | 2026-01-01 이상, 2026-11-26 미만 | 후속 종합 검증              | 미평가              |

`development.partition_from_source`는 기존 자료 로더와 동일한 정규시즌 필터 및 경기·타석·투구 순서를 사용해 감사 인덱스만 추가 조회한다. 같은 경기의 날짜와 소속이 일치하고 표본·경기 수의 합이 전체 Validation과 같아야 한다. 보정 적합, 선정 및 추가 평가를 구분하며 전체 Validation은 개발 통계이다. v1.0.0에서 이미 전체 2025년 지표를 관찰했으므로 추가 평가 구간 역시 완전히 처음 보는 독립 Test라고 주장하지 않는다. 새 실행에서 해당 구간의 표본·경기 수·날짜를 기록한다.

<a id="inputs"></a>

## 3. 입력·전처리

| 입력군       | 필드 및 처리                                                                                                                                                            |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 적용 범위    | `season`, `competition`: 타입·지원 범위 확인, 시즌 자체는 학습 특징으로 쓰지 않음                                                                                       |
| 현재 참가자  | `batter_id`, `pitcher_id`: 역할별 One-hot, Train 100구 미만 및 미출현 ID는 역할별 공통 범주                                                                             |
| 경기 상태    | `inning`, `half`, `outs`, `on_first`, `on_second`, `on_third`: 이닝 구간(1–3, 4–6, 7–9, 연장)과 초말, 아웃×주자 비트마스크                                              |
| 점수         | `home_score`, `away_score`: 공격팀 관점 점수 차, -5~5 절단                                                                                                              |
| 카운트       | `balls`, `strikes`: 12개 카운트 범주                                                                                                                                    |
| 투구 순번    | `pitch_number`: 1~4 및 5 이상                                                                                                                                           |
| 직전 실제 공 | `previous_pitch_type`, `previous_pitch_result`: 첫 공, 중간 결측, 유효 공동 코드 구분                                                                                   |
| 과거 기록    | `batter_prior_opportunities`, `batter_prior_so`, `pitcher_prior_opportunities`, `pitcher_prior_so`: 날짜 엄격 이전 기록으로 결측·노출량 30 미만·삼진율 0.1 폭 구간 구분 |

정확한 선택 입력 필드 집합만 허용한다. 데이터 생성기의 전체 허용 계약과 선택 입력을 함께 저장한다. 현재 공의 구종·결과·구속·좌표, 현재 공 이후 상태, 타석 종료 결과·참가자, `sample_id`와 원천 감사 식별자는 특징에서 제외한다. 허용된 팀 이력 등 일부 필드는 특징 수 및 계산량을 줄이기 위해 선택하지 않았다. 임의의 추론 JSON만으로 입력이 당시 이용 가능했는지 증명할 수는 없으므로 호출자가 예측 시점의 원천 상태를 제공해야 한다.

범주 구간과 규칙은 사전 고정하며 계수 사전, 빈도 및 ID 지원은 Train에서만 적합한다. 일반 미출현 입력 범주는 새로운 계수를 만들지 않는다. 미관측을 실제 0으로 바꾸지 않는다. 보정 구간의 정답으로 입력 어휘, 공동 범주 사전이나 학습 빈도를 확장하지 않는다.

<a id="outputs"></a>

## 4. 정답·출력 계약

정답은 `target_pitch_type`, `target_pitch_result`, `target_joint`가 일치하는 실제 코드다. 구종 코드는 `CH`, `CU`, `FC`, `FF`, `FO`, `KN`, `SI`, `SL`, `ST`, `SV`; 결과 코드는 `B`, `F`, `H`, `S`, `T`, `V`, `W`를 문자열 순서로 정렬한다. 의미 및 원천 코드 그룹은 `src/validation/standard_data.py`의 `PITCH_TYPES`, `PITCH_RESULT_TYPES`에 따른다. 결과 코드의 추가 세부 의미를 근거 없이 이름으로 치환하지 않는다.

두 사전의 데카르트 곱으로 70개 공동 키(`구종:결과`)를 고정한다. Train 관측 기준은 구종 8개, 결과 7개, 공동 조합 53개다. 미출현 조합은 제거하지 않고 구조적으로 불가능하다고 단정하지 않는다. 모든 출처 계약 범주에 양의 평활 확률을 제공한다. 미출현 조합의 일반화 성능을 보장하지 않으며 사전에 없는 코드 또는 매핑 불일치는 거부한다.

`predict` 출력은 `model_kind`, `joint`(70개), `pitch_type`(10개), `pitch_result`(7개)다. 각 분포의 값은 유한하고 0~1 범위이며 합은 1이다. 구종 확률은 해당 구종의 모든 결과 확률의 합이고 결과 확률은 해당 결과의 모든 구종 확률의 합이다. 반올림된 표시값이 아니라 원래 실수 확률로 검증한다. 개발 후보 이름은 `selected_candidate`, 실제 운영 분류기 종류는 `selected_model`로 구분한다.

<a id="method"></a>

## 5. 모델·학습·선정

### 5.1 기존 기준과 다항 로지스틱 회귀

Train 공동 빈도를 n_k, 전체 표본 수를 N, 공동 범주 수를 K=70으로 둔다. 전체 기준은 Laplace 평활 `(n_k + 1)/(N + K)`이다. 카운트별 조건부 기준은 지원이 50구 이상이면 `(n_gk + 100 p_k)/(n_g + 100)`을 사용하며 미달하면 전체 기준으로 후퇴한다. p_k는 Train 전체 평활 확률이다.

다항 로지스틱 후보는 절편 및 활성 One-hot 특징의 계수 합으로 로짓 z_k를 구하고, 안정적인 Softmax로 공동 분포를 계산한다. 절편을 전체 평활 확률의 로그로 초기화한다. 다중분류 NLL의 기울기 `p_k - y_k`를 사용하는 SGD를 3 epoch 적용하고 전체 Train을 매 epoch Seed 42로 비복원 순회한다. 초기 학습률 0.025, epoch 번호 e(0부터)에 대해 `0.025/sqrt(e+1)`을 사용한다. 추론 확률은 Softmax 99%와 Train 전체 평활 확률 1%의 혼합이다.

### 5.2 규제 및 계수 평균

`regularized` 후보는 기존 SGD와 동일한 특징·시드·epoch·학습률을 사용하며 L2 계수 0.00003을 고정한다. 절편은 규제하지 않는다. 특징 계수에 매 SGD 단계 `1 - learning_rate * L2`의 감쇠를 적용한다. 활성되지 않은 희소 특징의 감쇠는 다음 활성화 또는 epoch 종료 시점에 경과 단계 수만큼 곱하는 지연 계산으로 구현한다. 마지막 두 epoch 종료의 계수를 평균하여 단일 로지스틱 모델을 만든다. 별개의 독립 예측기 두 개를 운영하는 앙상블이 아니다. 규제 계수와 평균 범위는 사전 고정 후보 설정이며 최적값임이 입증된 것은 아니다.

### 5.3 온도 보정

두 로지스틱 계열 각각에서 2025년 보정 구간만 사용해 Temperature `{0.8, 1.0, 1.15, 1.3, 1.5, 2.0}`의 공동 Log loss를 비교한다. 확률은 `0.99 * softmax(z/T) + 0.01 * p_train`이다. 따라서 기존 평활 혼합 정책을 보존한다. 동률이면 1에 가까운 온도, 다시 동률이면 작은 온도를 선택한다. 온도는 로짓 계산 전 모든 계수와 절편을 T로 나누어 운영 계수에 직접 반영한다. 별도의 독립 운영 분류기나 클래스별 절편 보정기는 추가하지 않는다.

온도는 전체 공동 분포에 작용하므로 미출현 조합의 확률도 변할 수 있다. 그러나 새 구종·조합의 계수를 개별 정답으로 추가 학습하거나 학습 빈도를 갱신하지 않는다. Train 미출현 범주를 학습에서 관측한 지원으로 재분류하지 않는다.

### 5.4 비교 및 선택

| 후보                      | 설명                                |
| ------------------------- | ----------------------------------- |
| `frequency`               | 전체 Train 공동 빈도                |
| `conditional_frequency`   | 카운트 조건부 Train 공동 빈도       |
| `logistic`                | 기존 다항 로지스틱                  |
| `logistic_temperature`    | 기존 로지스틱 + 보정 구간 적합 온도 |
| `regularized`             | L2 규제와 마지막 두 epoch 계수 평균 |
| `regularized_temperature` | 규제 모델 + 보정 구간 적합 온도     |

2025년 선정 구간에서 우수 빈도 기준을 먼저 선택한다. 로지스틱 계열은 우수 빈도 기준에 대해 공동 Log loss·Brier가 모두 악화되지 않고 하나 이상 개선돼야 적격이다. 보정 후보는 추가로 미보정 부모에 비해 공동 Log loss·Brier·top-label ECE·macro classwise ECE가 모두 악화되지 않고 하나 이상 개선돼야 한다. 비교 허용 오차는 1e-10이다. 적격 후보 중 공동 Log loss, 공동 Brier, 공동 top-label ECE, 공동 macro classwise ECE, 고정 후보 순서로 하나를 결정한다. 적격 로지스틱 후보가 없으면 우수 빈도 기준을 운영 모델로 사용한다. 주변 지표의 개선 및 악화도 함께 공개하되 운영 후보 선정의 필수 게이트는 공동 지표에 한정한다. 보정 성능의 개선을 보장하지 않는다.

선정 이후에만 2025년 추가 평가 구간 및 전체 Validation 후보별 지표를 계산하며 이 지표로 후보를 다시 고르지 않는다. 최종 저장 모델은 하나다. 추가 트리 앙상블·별도 독립 구종 또는 결과 모델은 만들지 않는다.

<a id="lecture-methods"></a>

## 6. 강의 원리 및 프로젝트 확장

### 6.1 강의에서 확인한 원리의 실제 적용

선행 저장소의 [강의 방법 및 출처 정리](../planning/development_workflow.md#lecture-methods)를 인수한다. 이번 문서에서 원본 전사 및 영상 전체를 독립 재검토하지 않았다. 아래 v1.0.0 내용은 사용자 실행 및 확인된 구현에 근거하고, 신규 규제·보정은 v1.1.0 제공 코드의 구현 방식이며 신규 실제 실행은 확인 대기 중이다.

| 강의 원리                            | 구현 대응                                                                    |
| ------------------------------------ | ---------------------------------------------------------------------------- |
| 분류·확률·NLL 및 다중분류            | `model.py` 공동 Softmax, 공동 정답 매핑, `evaluation` Log loss               |
| 손실 최소화와 경사하강               | `fit_logistic` 및 신규 `fit_regularized`의 SGD와 감소 학습률                 |
| 학습·검증 분리 및 공정 비교          | Train 전용 어휘·계수, 동일 분할·평가 분모; 2025년의 순차 보정·선정·추가 평가 |
| 혼동 행렬 원리·범주별 오류           | 실제/예측 조합 빈도, Precision·Recall 및 지원 수                             |
| 과제 안내의 기준→변수 개선→비교→해석 | 빈도 기준, 로지스틱, 규제 및 온도 보정의 비교와 지표·비용·한계 공개          |

### 6.2 프로젝트 확장 원리

단일 공동 확률과 주변 합의 일치, 실제 공의 원천 순번 검증, 미출현 조합의 양의 평활 확률, 희소·미출현 선수 후퇴, 클래스별 Brier·신뢰도 구간, 자산 지문과 독립 CLI 재현은 프로젝트의 확률·자료 계약이다. L2 규제의 희소 지연 계산, 동일 계수 경로의 epoch 평균, 시간순 Temperature 보정 및 공동 지표 게이트는 v1.1.0의 확장 후보 및 절차다. 동일한 세부 알고리즘이 강의에서 직접 소개되었다고 주장하지 않는다.

### 6.3 주요 미적용 방법과 판단

무작위 K-fold, 층화 무작위 분할, 랜덤 7:3 분할은 시간 순서와 경기 묶음을 깨므로 최종 평가 방식으로 사용하지 않는다. 복원 추출·OOB·랜덤 포레스트·의사결정나무·Hard/Soft Voting·부스팅·스태킹은 이번에 실험하지 않는다. SGD의 비복원 순회와 랜덤 포레스트의 복원 추출은 다르다. 추가 독립 예측기를 운영하는 것보다 단일 공동 확률과 해석 가능한 기준 비교를 우선한다. Bias–Variance는 별도 학습곡선 실험으로 입증하지 않았으며 규제의 일반화 효과는 새 실행 지표로 판단한다. 전체 Grid/Random Search를 수행했다는 주장을 하지 않는다.

### 6.4 강의 및 이론 근거

선행 확인 출처: `LogisticRegression_update(1)_빨강펜.txt`의 확률 분류·NLL·다중분류, `LogisticRegression_update(2)_빨강펜-seg2.txt`의 MLE/NLL 및 경사하강, `LinearRegression_빨강펜-seg1.txt`의 손실 최소화, `ml_cv,eval-metric_빨강펜_SW중단.txt`의 분할 및 평가 원리. 정확한 전사 시각은 개발 흐름 문서를 참조한다. 과제 안내 `창문해_TermProject_가이드.pdf` 4쪽의 EDA·베이스라인·피처 엔지니어링·비교·최적화·결론 요구와 강의 자체를 구분한다. Temperature Scaling의 이론적 배경은 Guo et al., _On Calibration of Modern Neural Networks_ (ICML 2017)의 단일 온도 원리를 참고하는 것이며, 본 프로젝트는 로지스틱 공동 로짓과 Train 사전 혼합에 맞춰 변형한다.

<a id="evaluation"></a>

## 7. 평가 지표·실제 결과

공동·구종·결과별 Log loss, Brier, Accuracy, top-label ECE와 macro classwise ECE를 계산한다. Log loss는 `-sum(log(p_actual))/n`, Brier는 `sum_i sum_k (p_ik-y_ik)^2/n`이다. ECE는 10개 구간의 예측 평균과 실제 비율 차이 절댓값의 분모 가중 평균이다. Macro classwise ECE는 고정 사전의 클래스별 one-vs-rest ECE를 동일 가중 평균한다. 클래스별 Brier, 지원 수, Precision·Recall, 실제 정답 클래스 조건부 Log loss와 대표 오분류 쌍을 저장한다. 빈 보정 구간 및 지원 0인 일부 지표는 null로 표기한다.

각 확률 축에서 Train 지원 0, 1~49, 50 이상 정답 표본의 건수·비율·지표를 분리한다. 출처 사전 안의 미출현 범주는 전체 평가에 포함하고 학습 관측 지원과 사전 출력 지원을 구분한다. 새 판본은 후보 간 실제 클래스 지원 수, 클래스별 보정 분모 및 사건 수, 지원 그룹의 분모 및 공동·주변 정답 지원의 일치도 검증한다.

### 7.1 v1.0.0 실제 전체 Validation 결과

출처: 사용자 실행 로그(2026-11-27 KST), 학습 실행 `a8d63e0d727146cdaeb131b10f6e09d6`, 동일 216,869구. 아래는 v1.0.0의 개발 성능이며 v1.1.0 결과가 아니다.

| 대상·지표          |  전체 빈도 | 조건부 빈도 |   로지스틱 |
| ------------------ | ---------: | ----------: | ---------: |
| 공동 Log loss      | 3.24228595 |  3.18445934 | 2.92233916 |
| 공동 Brier         | 0.94066407 |  0.93075658 | 0.91747048 |
| 공동 top-label ECE | 0.00132505 |  0.00081446 | 0.04352150 |
| 구종 Log loss      | 1.70841655 |  1.72745528 | 1.44890827 |
| 구종 Brier         | 0.75152960 |  0.74223814 | 0.67594927 |
| 구종 top-label ECE | 0.00205252 |  0.00260341 | 0.01598657 |
| 결과 Log loss      | 1.54228306 |  1.46605928 | 1.46044464 |
| 결과 Brier         | 0.76304528 |  0.73720702 | 0.73862719 |
| 결과 top-label ECE | 0.00400738 |  0.00593918 | 0.04229068 |

v1.0.0은 사전 고정 공동 Log loss·Brier 기준으로 `logistic` 하나를 선정했다. 공동 Log loss와 Brier, 구종 성능은 두 기준 대비 개선됐으나 세 축의 top-label ECE는 두 기준보다 악화됐다. 결과 Brier도 조건부 빈도보다 악화됐다. 이 때문에 보완판에서 규제·온도 후보를 비교한다. 원본 `evaluation.json`의 클래스별 세부 수치는 아직 직접 검토하지 않았으며 실제 수치로 채워 넣지 않는다.

### 7.2 미출현 조합의 실제 분포와 한계

| 구분                | Train | Validation |
| ------------------- | ----: | ---------: |
| 스위퍼(ST)          |     0 |      1,564 |
| 슬러브(SV)          |     0 |          3 |
| 두 미출현 구종 합계 |     0 |      1,567 |

Validation 공동 사전 내 Train 미출현 실제 조합은 1,567구, 약 0.7226%다. 학습 관측 지원률은 공동 및 구종에서 0.99277444, 결과에서 1.0이었다. 고정 사전은 위 표본을 포함하며 실제 관측 지원과 출력 가능 지원이 다르다. 미출현 조합의 매우 작은 예측값과 실제 빈도의 차이는 분포 이동의 한계이며 새 조합을 충분히 학습했다는 증거가 아니다. v1.1.0의 희소·미출현 그룹과 클래스별 실측 지표는 신규 평가 자료로 확인한다.

### 7.3 v1.1.0 신규 개발 평가

| 항목                                              | 현재 상태      |
| ------------------------------------------------- | -------------- |
| 보정·선정·추가 평가 구간의 실제 표본 및 경기 수   | 미실행         |
| 각 로지스틱 계열의 Temperature별 보정 NLL         | 미실행         |
| 6개 후보의 선정 구간 공동·주변 지표               | 미실행         |
| 선정 미사용 구간 및 전체 Validation의 후보별 지표 | 미실행         |
| 선정 후보, 게이트 판단, 최종 운영 온도            | 미확인         |
| 희소/미출현 그룹의 실측 지표 및 대표 오류         | 미확인         |
| 학습 시간·메모리·자산 지문·회귀 및 독립 재현      | 신규 실행 대기 |

선정 적격 및 확률 보정 개선은 실제 실행 후에만 판단한다. v1.0.0의 전체 2025년 수치와 v1.1.0의 선정 구간 수치를 직접 동일 분모 비교로 해석하지 않는다. 공통 전체 Validation 비교는 별도로 보고한다.

<a id="reproducibility"></a>

## 8. 구현·산출물·재현성

| 구성                                             | 책임                                                                           |
| ------------------------------------------------ | ------------------------------------------------------------------------------ |
| `src/models/pitch_outcome/__init__.py`           | 신규 버전 `1.1.0`, 스키마 `m5-pitch-outcome-v2`                                |
| `src/models/pitch_outcome/model.py`              | 입력, 공동 사전, 빈도, 기존 SGD, 공동·주변 확률                                |
| `src/models/pitch_outcome/data.py`               | 자료 지문·키·시간 분할·원천 상태와 투구 순서                                   |
| `src/models/pitch_outcome/development.py`        | 시간순 개발 분할, 희소 지연 L2, 계수 평균, Temperature, 후보 선정 및 평가 검증 |
| `src/evaluation/pitch_outcome.py`                | 전체·클래스별·보정·지원 평가                                                   |
| `src/models/pitch_outcome/__main__.py`           | `train`, `predict`, `verify`, `compare` CLI와 자산 계보                        |
| `tests/models/test_pitch_outcome.py`             | 기존 합성 자료, 누수, 분할, 평가 및 자산 검증                                  |
| `tests/models/test_pitch_outcome_development.py` | 추가 보정·규제·선정·평가 분모·독립 비교 및 전체 저장 흐름 검사                 |

신규 학습마다 `models/m5/<training_run_id>/model.json` 및 `results/m5/<training_run_id>/evaluation.json`, `example_input.json`, `example_prediction.json`을 생성한다. `evaluation.json`에는 `calibration_fits`, `development_periods`, `selection_evaluations`, `reporting_evaluations`, `candidate_evaluations`, `selection`, `selected_support`, `resources`를 기록한다. `candidate_evaluations`는 전체 Validation이다. 모델 JSON에는 선정 모델 하나의 계수 또는 빈도표와 선택·보정 메타데이터만 저장한다. Temperature는 운영 계수에 반영하며 별도 독립 예측기를 저장하지 않는다. 저장 모델의 지문, 평가 지문, 학습 소스 지문, 실제 경계 사례와 독립 JSON 출력을 검증한다. 예전 자산은 덮어쓰지 않는다. v1.0.0 자산은 신규 버전의 엄격한 스키마 검증에 맞지 않으므로 새로 학습해야 한다.

### 8.1 기존 v1.0.0 사용자 실행 기록

| 항목                          | 값                                                                 |
| ----------------------------- | ------------------------------------------------------------------ |
| 학습 실행 ID                  | `a8d63e0d727146cdaeb131b10f6e09d6`                                 |
| 운영 모델                     | `logistic`, 1개, Seed 42, CPU, Python 3.12.15                      |
| 데이터 감사 시간              | 40.0899초                                                          |
| 로지스틱 적합 시간            | 93.0477초                                                          |
| 로지스틱 전체 Validation 평가 | 22.9170초                                                          |
| 저장 전 총 경과 시간          | 182.4081초                                                         |
| 저장 전 최대 상주 메모리      | 168,775,680 bytes (약 161 MiB)                                     |
| 모델 경로                     | `models/m5/a8d63e0d727146cdaeb131b10f6e09d6/model.json`            |
| 평가 경로                     | `results/m5/a8d63e0d727146cdaeb131b10f6e09d6/evaluation.json`      |
| 모델 SHA-256                  | `74d7a9155e1e579545493e24d1c9defca79654af95a3ac297c526ab1fd2b892f` |
| 평가 SHA-256                  | `0af670dd995d818b4fe424fc5bb5c92e28c37edbbde54ad8fff294b9d22de2ff` |

학습·verify·독립 predict는 모두 종료 코드 0이었다. `verify`가 저장 모델의 사례별 일치를 확인했다. 별도 PowerShell 인라인 Python 비교는 `NameError: name 'utf' is not defined`로 종료 코드 1이었으므로 외부 비교의 성공 증거가 아니다. v1.1.0은 `compare --expected`에 독립 `predict`의 JSON stdout을 전달해 비교한다. 사용자가 로그에서 확인한 지문 및 실행 결과와 Work가 직접 재계산한 지문을 구분한다.

<a id="limitations"></a>

## 9. 검증·한계·후속 연결

v1.0.0 사용자 실행에서 M5 테스트 40개(0.035초), 전체 회귀 326개(6.112초)가 통과했다. v1.1.0에는 신규 테스트 22개가 제안되며 기존 테스트는 유지한다. 예상 M5 합계 62개, 전체 합계 348개는 코드 기준 예상 수로서 실제 통과 수가 아니다. 합성 테스트, 실데이터 학습·평가, 저장 자산 검증 및 별도 프로세스 추론·JSON 비교가 모두 통과해야 한다. 실패 뒤 출력된 고정 성공 문자열은 성공 근거로 인정하지 않는다. 새 검증 결과와 원격 반영, 하위 수용은 별개다.

미출현 구종은 사전 평활화 또는 온도 변환으로만 처리되며 충분한 학습 관측을 확보한 것이 아니다. 최종 Test와 포스트시즌 전이 가능성은 현재 평가하지 않는다. 전체 2025년이 이미 관찰된 개발 기간인 점을 유지한다. 선택 기준과 온도 탐색 범위를 새 결과에 맞춰 사후 변경하지 않는다. 보정 효과 미개선·규제 후보 미채택도 정상적인 비교 결과일 수 있다. 실측 Bias–Variance, 다른 알고리즘 우열, 최종 일반화는 미확인이다.

기존 자료의 실제 공개 시점·당시 판본 한계를 보존한다. 모델·데이터·결과 파일은 기존 Git 제외 경로에 둔다. README와 개발 흐름의 상태·수치·상호 링크는 v1.1.0의 실제 검증 및 원격 반영 근거로 최종 정리해야 한다. 현재 문서는 사용자에게 전달할 보완판 실행 계약이며 아직 수용 보고서가 아니다.

<a id="commands"></a>

## 10. 실행 명령

Windows PowerShell의 프로젝트 루트 및 기존 Python 3.12 환경을 사용한다. 신규 외부 의존성이나 수집·ETL 재실행은 필요하지 않다. 전체 명령은 하나의 스크립트 블록으로 실행하며 각 단계의 실패는 이후 단계를 중단시킨다. 파일 생성은 정상 학습 산출물 경로에 한정된다.

```powershell
& {
    Set-Location -LiteralPath 'D:/Programming/kbo-insight-ai'
    $kboPython = 'D:/Programming/kbo-insight-ai/.venv/Scripts/python.exe'
    $dataRun = '259b38decaac4ed68dd5a596cfbe4849'

    & $kboPython --version
    if ($LASTEXITCODE -ne 0) { throw 'Python failed' }
    git branch --show-current
    git rev-parse HEAD
    git --no-optional-locks status --short --branch --untracked-files=all

    & $kboPython -X utf8 -B -m unittest discover -s tests -p 'test_pitch_outcome*.py' -v
    $m5Exit = $LASTEXITCODE
    Write-Output "m5_test_exit=$m5Exit"
    if ($m5Exit -ne 0) { throw 'M5 tests failed' }

    & $kboPython -X utf8 -B -m unittest discover -s tests
    $regressionExit = $LASTEXITCODE
    Write-Output "regression_exit=$regressionExit"
    if ($regressionExit -ne 0) { throw 'Regression tests failed' }

    $trainJson = & $kboPython -X utf8 -B -m src.models.pitch_outcome train --run-id $dataRun
    $trainExit = $LASTEXITCODE
    $trainJson
    Write-Output "train_exit=$trainExit"
    if ($trainExit -ne 0) { throw 'M5 train failed' }
    $run = ($trainJson -join "`n") | ConvertFrom-Json
    if ($run.status -ne 'completed' -or $run.model_version -ne '1.1.0' -or $run.operational_predictor_count -ne 1 -or $run.final_test_evaluated -ne $false -or $run.reporting_used_for_selection -ne $false) { throw 'M5 training contract failed' }

    $verifyJson = & $kboPython -X utf8 -B -m src.models.pitch_outcome verify --asset $run.paths.model --evaluation $run.paths.evaluation
    $verifyExit = $LASTEXITCODE
    $verifyJson
    Write-Output "verify_exit=$verifyExit"
    if ($verifyExit -ne 0) { throw 'M5 verify failed' }
    $verification = ($verifyJson -join "`n") | ConvertFrom-Json
    if ($verification.status -ne 'verified' -or $verification.reproduce_equal -ne $true -or $verification.final_test_evaluated -ne $false) { throw 'M5 asset reproduction failed' }

    $predictionJson = & $kboPython -X utf8 -B -m src.models.pitch_outcome predict --asset $run.paths.model --input $run.paths.input
    $predictExit = $LASTEXITCODE
    $predictionJson
    Write-Output "predict_exit=$predictExit"
    if ($predictExit -ne 0) { throw 'M5 predict failed' }

    $comparisonJson = ($predictionJson -join "`n") | & $kboPython -X utf8 -B -m src.models.pitch_outcome compare --expected $run.paths.prediction
    $compareExit = $LASTEXITCODE
    $comparisonJson
    Write-Output "reproduce_exit=$compareExit"
    if ($compareExit -ne 0) { throw 'M5 independent comparison failed' }
    $comparison = ($comparisonJson -join "`n") | ConvertFrom-Json
    if ($comparison.reproduce_equal -ne $true -or $comparison.joint_classes -ne 70 -or $comparison.pitch_type_classes -ne 10 -or $comparison.pitch_result_classes -ne 7) { throw 'M5 output comparison contract failed' }

    foreach ($path in @($run.paths.model, $run.paths.evaluation, $run.paths.input, $run.paths.prediction)) {
        $file = Get-Item -LiteralPath $path -ErrorAction Stop
        Write-Output "path=$($file.FullName)"
        Write-Output "bytes=$($file.Length)"
        Get-FileHash -LiteralPath $path -Algorithm SHA256 -ErrorAction Stop | Format-List Algorithm, Hash, Path
    }
    git --no-optional-locks status --short --branch --untracked-files=all
    git diff --check
    $diffExit = $LASTEXITCODE
    Write-Output "diff_check_exit=$diffExit"
    if ($diffExit -ne 0) { throw 'Git diff check failed' }

    Write-Output "training_run_id=$($run.training_run_id)"
    Write-Output "selected_candidate=$($run.selected_candidate)"
    Write-Output "selected_temperature=$($run.selected_temperature)"
    Write-Output "evaluation_path=$($run.paths.evaluation)"
    Write-Output "reporting_used_for_selection=$($run.reporting_used_for_selection)"
    Write-Output "final_test_evaluated=$($run.final_test_evaluated)"
    Write-Output 'M5_FULL_VERIFICATION_OK=True'
    Write-Output 'M5_WORK_VALIDATION_COMPLETED=True'
}
```

검증 증거: 테스트 수·종료 코드, 실제 자료와 제외·미출현 통계, 보정 구간별 후보 NLL과 선정 근거, 선정·추가 평가·전체 Validation의 세 확률 축 지표, 실행 ID·경로·지문·자원 사용, `reproduce_equal=true`, `reporting_used_for_selection=false`, `final_test_evaluated=false`. 실제 `evaluation.json`은 클래스별 보정·지원 그룹·오류 분석 검토용으로 사용자와 공유하고 Git에는 추가하지 않는다. `git diff --check`는 아직 추적되지 않은 파일의 공백 검사까지 보장하지 않으므로 Commit 단계에서 `git diff --cached --check`를 별도로 사용한다. 실제 GitHub 상태와 기술 수용은 별도 확인한다.
