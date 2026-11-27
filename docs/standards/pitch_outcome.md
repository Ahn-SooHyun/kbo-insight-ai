# 다음 투구 구종·결과 예측 실행과 결과 계약

실제 다음 공 직전의 관측 상태와 과거 기록으로 공동 확률 및 두 주변 확률을 계산한다.

| 항목        | 내용                                                                                             |
| ----------- | ------------------------------------------------------------------------------------------------ |
| 문서 역할   | 모델 구현안 및 실행·검증 계약                                                                    |
| 적용 범위   | M5 정규시즌 실제 투구의 구종·결과, CPU 로컬 학술용                                               |
| 검토 기준일 | 2026-11-27 KST                                                                                   |
| 저장소 경로 | `docs/standards/pitch_outcome.md`                                                                |
| 선행 기준   | `f40ed33841a724d15aa9dcaaf053c3b47feb2f09`                                                       |
| 관련 Issue  | [#20 다음 투구 구종·결과 예측](https://github.com/Ahn-SooHyun/kbo-insight-ai/issues/20)          |
| 상태        | 적용·실데이터 실행·성능·하위 수용 미확인; 아래 수치는 인수 자료의 기준이며 신규 실행 결과가 아님 |

관련 문서: [README](../../README.md) · [공통 예측·평가 계약](project_contract.md#m5) · [개발 흐름 및 강의 기준](../planning/development_workflow.md#lecture-methods) · [타석 결과 예측](plate_appearance.md)

찾아보기: [목적](#overview) · [자료](#data) · [입력](#inputs) · [출력](#outputs) · [학습](#method) · [강의 원리](#lecture-methods) · [평가](#evaluation) · [재현](#reproducibility) · [한계](#limitations) · [명령](#commands)

<a id="overview"></a>

## 1. 목적·지원 범위

해당 공을 던지기 전 시점에서 구종과 투구 결과의 공동 분포를 분류하고, 그 분포의 축별 합으로 구종 및 결과 확률을 일관되게 제공한다. 결과의 확률은 현재 공의 실제 구종을 입력한 조건부 예측이 아니다. 단일 운영 분류기만 저장한다. 구속·투구 위치·좌표, 투구 궤적, 감독 의도, 무투구 자동 판정의 투구 예측은 제외한다. 제공하는 확률은 특정 결과의 보장이 아니다.

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

날짜는 KST 기준이다. `m5_index`, `m5_inputs`, `m5_targets`의 `sample_id`를 일대일 대조한다. 동일 경기의 분할 및 날짜 혼합을 거부한다. 모든 인덱스의 키와 분할을 확인하되 최종 Test의 정답 값은 개발 평가에 조회하지 않는다. 모델 DB와 표준 검증 DB를 읽기 전용으로 열며 매니페스트, 계약, EDA, 보완 영향 자료와 지문을 대조한다.

현재 공의 표준 `pitches` 행 및 `eligible_states` 경계, 참가자, 순번, 상태, 코드와 원본 식별자를 비교한다. 표준의 `non_pitch`는 실제 공의 분모에 포함하지 않는다. 표준 실제 공의 배타적 제외 사유는 경기 제외 → 타석 상태 제외 → 투구 번호 → 순서 → 상태 → 정답 코드 순이다. `game_checks`에서 확인 가능한 원천 양의 투구 수와 표준에 저장된 투구 수의 차이는 별도 집계한다. 이 차이를 모두 중복이나 특정 검역 사유로 단정하지 않는다. 비정규시즌 표본은 개발 분할의 정규시즌 평가와 분리한다.

직전 공은 같은 경기·타석에서 순번이 1 작은 실제 공이며 원본 지문과 원본 행 순서가 맞고 순서·정답 코드가 유효한 경우만 사용한다. 이전 공 자체가 상태 품질로 모델 표본에서 제외되었더라도 이 원천 조건을 충족하면 기존 자료 계약상 직전 코드 사용이 가능하다. 첫 공과 중간 순번의 결측, 원천 순서 오류, 정답 무효 및 이전 공의 모델 제외를 별도로 집계한다. 인수 기준 Train 첫 공은 113,508구, 직전 구종 결측은 113,516구로 중간 결측은 8구다. 신규 실행에서 이를 재대조한다.

<a id="inputs"></a>

## 3. 입력·전처리

| 입력군       | 필드 및 처리                                                                                                                                                                           |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 적용 범위    | `season`, `competition`: 타입·지원 범위 확인, 시즌 자체는 학습 특징으로 쓰지 않음                                                                                                      |
| 현재 참가자  | `batter_id`, `pitcher_id`: 역할별 One-hot, Train 100구 미만 및 미출현 ID는 역할별 공통 범주                                                                                            |
| 경기 상태    | `inning`, `half`, `outs`, `on_first`, `on_second`, `on_third`: 이닝 구간(1–3, 4–6, 7–9, 연장)과 초말, 아웃×주자 비트마스크                                                             |
| 점수         | `home_score`, `away_score`: 공격팀 관점 점수 차, -5~5 절단                                                                                                                             |
| 카운트       | `balls`, `strikes`: 12개 카운트 범주                                                                                                                                                   |
| 투구 순번    | `pitch_number`: 1~4 및 5 이상                                                                                                                                                          |
| 직전 실제 공 | `previous_pitch_type`, `previous_pitch_result`: 첫 공, 중간 결측, 유효 공동 코드 구분                                                                                                  |
| 과거 기록    | `batter_prior_opportunities`, `batter_prior_so`, `pitcher_prior_opportunities`, `pitcher_prior_so`: 날짜 엄격 이전 기록으로 결측/노출량 30 미만/삼진율 10분위 폭 구간(0.1 간격)을 구분 |

입력은 정확한 선택 필드 집합만 허용한다. 데이터 생성기의 전체 허용 계약과 선택 입력을 함께 저장한다. 현재 공의 구종·결과·구속·좌표, 현재 공 이후 상태, 타석 종료 결과·참가자, `sample_id`와 원천 감사 식별자를 학습 특징에 넣지 않는다. 그 외 계약상 허용된 팀 이력 등은 이번 단일 후보의 특징 수 및 계산량을 줄이기 위해 선택하지 않았다. 미래 정보가 없다는 입력 자체의 진술은 임의의 추론 JSON만으로 독립적으로 증명할 수 없으므로 호출자가 예측 시점의 원천 상태를 제공해야 한다.

범주 구간과 규칙은 사전 고정하며 계수 사전, 빈도 및 ID 지원은 Train에서만 구성한다. 일반 미출현 범주는 새로운 계수를 만들지 않는다. 미관측을 실제 0으로 바꾸지 않는다.

<a id="outputs"></a>

## 4. 정답·출력 계약

정답은 `target_pitch_type`, `target_pitch_result`, `target_joint`의 일치하는 실제 코드다. 구종 코드는 `CH`, `CU`, `FC`, `FF`, `FO`, `KN`, `SI`, `SL`, `ST`, `SV`; 결과 코드는 `B`, `F`, `H`, `S`, `T`, `V`, `W`를 각각 문자열 정렬한다. 실제 의미 및 원천 코드 그룹은 `src/validation/standard_data.py`의 `PITCH_TYPES`, `PITCH_RESULT_TYPES`를 따른다. 결과 코드의 추가 세부 의미를 근거 없이 이름으로 치환하지 않는다.

두 사전의 데카르트 곱으로 70개 공동 키(`구종:결과`)를 고정한다. Train 관측 기준은 구종 8개, 결과 7개, 공동 조합 53개다. 학습 미출현 조합은 제거하지 않으며 구조적 불가능으로 단정하지 않는다. 출처 계약 범주 전체에 양의 평활 확률을 제공한다. 이는 실제로 관측된 지원이나 미출현 조합의 일반화 성능을 보장하지 않는다. 계약에 없는 코드나 매핑 불일치는 오류로 처리한다.

`predict`의 JSON 출력은 `model_kind`, `joint`(70개), `pitch_type`(10개), `pitch_result`(7개)다. 각 분포의 합은 1이며 모든 값은 유한하고 0~1 범위다. 구종 확률은 같은 구종의 모든 결과 확률 합, 결과 확률은 같은 결과의 모든 구종 확률 합이다. 반올림한 숫자가 아니라 저장된 원래 실수 확률을 검증한다.

<a id="method"></a>

## 5. 모델·학습·선정

Train 공동 빈도를 n_k, 전체 표본 수를 N, 공동 범주 수를 K=70으로 둔다. 전체 기준은 Laplace 평활 `(n_k + 1)/(N + K)`이다. 카운트별 조건부 기준은 지원이 50구 이상이면 `(n_gk + 100 p_k)/(n_g + 100)`을 사용하고 미달하면 전체 기준으로 후퇴한다. p_k는 Train 전체 평활 확률이다.

다항 로지스틱 후보는 절편 및 활성 One-hot 특징의 계수 합으로 로짓 z_k를 구한다. Softmax는 `exp(z_k - max(z))/sum_j exp(z_j - max(z))`이다. 절편은 전체 평활 확률의 로그로 초기화한다. 다중분류 NLL의 기울기 `p_k - y_k`를 사용하여 SGD로 학습하며, 3 epoch 동안 전체 Train을 epoch마다 Seed 42로 비복원 순회한다. 초기 학습률 0.025, epoch 번호 e(0부터)에 대해 `0.025/sqrt(e+1)`을 사용한다. 별도 가중치 L2 규제, 트리 앙상블, 보정기 및 GPU는 사용하지 않는다. 추론에서는 Softmax 확률 99%와 Train 전체 평활 확률 1%를 혼합한다. 이는 학습 미출현 범주에 양의 확률 질량을 유지하는 사전 고정 평활화이며 별도 독립 운영 예측기나 보정기 적합이 아니다.

후보는 전체 빈도, 조건부 빈도, 다항 로지스틱 세 가지다. 같은 Validation에서 공동 Log loss 최솟값, 공동 Brier 최솟값, 완전 동점일 경우 전체 빈도 → 조건부 빈도 → 로지스틱 순서로 하나를 선정한다. 기준 모델이 우수하면 그대로 공개하고 선정할 수 있다. 분류기 수는 저장·운영하는 독립 분류기의 실제 개수로 계산하며, 두 주변 분포를 위해 추가 모델을 학습하지 않는다. Validation은 개발 선택용이며 최종 Test 결과가 아니다.

<a id="lecture-methods"></a>

## 6. 강의 원리 및 프로젝트 확장

### 6.1 강의에서 확인한 원리의 실제 적용

선행 저장소의 [강의 방법 및 출처 정리](../planning/development_workflow.md#lecture-methods)를 인수한다. 이번 문서 작성에서 원본 전사 및 영상 전체를 독립 재검토하지 않았으므로 아래 대응은 그 선행 정리에서 확인한 원리와 제시 코드의 대응이며, 학습 실행 완료를 뜻하지 않는다.

| 강의 원리                            | 구현 대응                                                                                                    |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------ |
| 분류·확률·NLL 및 다중분류            | `model.py`의 공동 Softmax 분류 및 정답 매핑, `evaluation`의 Log loss                                         |
| 손실 최소화와 경사하강               | `fit_logistic`의 SGD 갱신 및 epoch별 손실 로그                                                               |
| 학습·평가 분리와 공정 비교           | 동일 2025년 Validation 분모, Train 전용 특징 사전·빈도·학습; 시계열 특성에 맞춘 경기 단위 날짜순 분할        |
| 혼동 행렬 원리·범주별 오류           | 평가의 실제/예측 조합 빈도, 클래스별 정밀도·재현율·지원 수                                                   |
| 과제 안내의 기준→변수 개선→비교→해석 | 빈도와 조건부 빈도 기준 및 다항 로지스틱 후보를 동일 표본에서 비교하고 한계를 공개; 실제 결과는 실행 후 기록 |

### 6.2 프로젝트 확장 원리

단일 공동 확률과 주변 합의 일치, 고정 출처 사전의 미출현 조합 평활화, 실제 공의 원천 순번 검증 및 무투구 격리, 희소·미출현 선수 후퇴, Brier 및 클래스별 신뢰도 구간, 저장 자산 지문과 독립 CLI 재현을 적용하도록 구성했다. 위 세부 규칙이 동일한 형태로 강의에서 소개됐다고 주장하지 않는다.

### 6.3 주요 미적용 방법과 판단

무작위 K-fold, 층화 무작위 분할, 7:3 랜덤 분할은 시간 순서·경기 묶음을 깨므로 현재의 최종 평가 방식으로 사용하지 않는다. 복원 추출·OOB·랜덤 포레스트·의사결정나무·Hard/Soft voting·부스팅·스태킹은 이번에 실험하지 않는다. 공동 Log loss와 보정, 비용을 먼저 확인하며 추가 후보는 별도 타당성이 있을 때만 고려한다. SGD의 비복원 epoch 순회는 랜덤 포레스트의 복원 추출과 다르다. 강의 기법을 단순히 많이 사용하는 것을 목표로 하지 않는다. Bias–Variance는 미실행 학습곡선으로 확정하지 않고, 향후 필요 시 Train 내부 시간순 실험에서 검토한다.

### 6.4 강의 및 이론 근거

선행 확인 출처: `LogisticRegression_update(1)_빨강펜.txt`의 확률 분류·NLL·다중분류, `LogisticRegression_update(2)_빨강펜-seg2.txt`의 MLE/NLL 및 경사하강, `LinearRegression_빨강펜-seg1.txt`의 손실 최소화, `ml_cv,eval-metric_빨강펜_SW중단.txt`의 분할 및 평가 원리. 정확한 전사 시각은 위 개발 흐름 문서를 참조한다. 과제 안내 `창문해_TermProject_가이드.pdf` 4쪽의 EDA·베이스라인·피처 엔지니어링·비교·최적화·결론 요구와 강의 내용 자체를 구분한다.

<a id="evaluation"></a>

## 7. 평가 지표·실제 결과

각 후보의 동일 Validation 216,869구에서 공동·구종·결과별 Log loss, Brier, Accuracy 및 top-label ECE를 계산한다. Log loss는 `-sum(log(p_actual))/n`, Brier는 `sum_i sum_k (p_ik-y_ik)^2/n`이다. ECE는 10개 확률 구간의 예측 평균과 실제 비율 차이 절댓값을 분모로 가중한 값이다. 클래스별 one-vs-rest 보정 및 Brier, 지원 수, precision, recall, 실제 정답 클래스 조건부 Log loss와 대표 오분류 쌍도 보존한다. 빈 보정 구간 및 지원 0 지표는 null로 표기한다. 세 확률 축 각각에서 Train 지원 0, 1~49, 50 이상인 정답 표본의 수·비율·지표를 분리한다. 미출현 코드 중 출처 사전 범주에 속하는 표본은 전체 평가에 포함하며, 이를 사전 출력 지원과 학습 관측 지원으로 구분한다.

| 후보          | 공동 Log loss | 공동 Brier | 구종·결과 성능 및 보정 |
| ------------- | ------------- | ---------- | ---------------------- |
| 전체 빈도     | 미실행        | 미실행     | 미실행                 |
| 조건부 빈도   | 미실행        | 미실행     | 미실행                 |
| 다항 로지스틱 | 미실행        | 미실행     | 미실행                 |

운영 선정·실제 수치·희소 및 미출현 분포·시간·메모리·자산 SHA-256은 사용자 실행 증거 확보 후 갱신한다. 성능 개선, 보정 개선 또는 완료는 아직 확정하지 않는다.

<a id="reproducibility"></a>

## 8. 구현·산출물·재현성

| 구성                                   | 책임                                               |
| -------------------------------------- | -------------------------------------------------- |
| `src/models/pitch_outcome/__init__.py` | 버전 `1.0.0`, 스키마 `m5-pitch-outcome-v1`         |
| `src/models/pitch_outcome/model.py`    | 입력, 범주, 평활, 빈도, SGD, 공동·주변 확률        |
| `src/models/pitch_outcome/data.py`     | 자료 지문·키·시간 분할·원천 상태 및 투구 순서 검사 |
| `src/evaluation/pitch_outcome.py`      | 전체/범주/보정/지원 평가 및 선택                   |
| `src/models/pitch_outcome/__main__.py` | `train`, `predict`, `verify` CLI 및 자산 연결      |
| `tests/models/test_pitch_outcome.py`   | 합성 데이터, 누수, 분할, 평가 및 자산 검증         |

학습 실행마다 설정의 `models/m5/<training_run_id>/model.json` 및 `results/m5/<training_run_id>/evaluation.json`, `example_input.json`, `example_prediction.json`을 생성한다. 실제 경계 사례와 데이터 계보, 시드, 입력·클래스 사전, 평활화·선정 정책, Python/OS, 실행 시간과 CPU 시간 및 지원 OS의 최대 상주 메모리를 기록한다. 코드·자료 지문을 대조하고 모델과 평가 지문 및 독립 예시 재현을 검증한다. 모델 JSON에는 선정된 하나의 분류기만 저장한다. 원본 및 학습 산출물은 기존 Git 제외 경로에 유지한다.

<a id="limitations"></a>

## 9. 검증·한계·후속 연결

선행 전체 회귀 테스트 기준은 286개이며 신규 테스트가 추가된다. 현재 작성한 코드는 사용자 환경에 반영되거나 실행된 사실이 아직 확인되지 않았다. 따라서 단위·합성 통합·전체 회귀·실데이터 학습 및 별도 프로세스의 `verify`, `predict` 결과가 모두 필요하다. 실패 시 작업 범위 내에서 원인을 수정한다. 희소 조합의 평활 확률은 실제 일반화 증거가 아니며, 정규시즌에서만 개발 평가한다. 자료의 공개 시점·당시 판본에 대한 선행 한계를 유지한다. 2026년 최종 Test는 후속 종합 검증에서만 평가한다. 실측 Bias–Variance, 보정 개선, 포스트시즌 전이 가능성, 다른 알고리즘 우열은 아직 미확인이다. README 및 개발 흐름의 상태·결과·상호 링크는 실제 결과가 확인된 후 갱신한다.

<a id="commands"></a>

## 10. 실행 명령

Windows PowerShell에서 프로젝트 루트 및 기존 Python 3.12 환경을 사용한다. 신규 외부 의존성이나 수집·ETL 재실행은 필요하지 않다. 오류 시 이후 단계로 진행하지 않고 종료 코드와 오류 원문을 기록한다.

```powershell
Set-Location -LiteralPath 'D:/Programming/kbo-insight-ai'
$kboPython = 'D:/Programming/kbo-insight-ai/.venv/Scripts/python.exe'
$dataRun = '259b38decaac4ed68dd5a596cfbe4849'
& $kboPython -X utf8 -B -m unittest discover -s tests -p 'test_pitch_outcome.py' -v
if ($LASTEXITCODE -ne 0) { throw 'M5 tests failed' }
& $kboPython -X utf8 -B -m unittest discover -s tests
if ($LASTEXITCODE -ne 0) { throw 'Regression tests failed' }
$trainJson = & $kboPython -X utf8 -B -m src.models.pitch_outcome train --run-id $dataRun
if ($LASTEXITCODE -ne 0) { throw 'M5 train failed' }
$run = ($trainJson -join "`n") | ConvertFrom-Json
$trainJson
& $kboPython -X utf8 -B -m src.models.pitch_outcome verify --asset $run.paths.model --evaluation $run.paths.evaluation
if ($LASTEXITCODE -ne 0) { throw 'M5 verify failed' }
$predictionJson = & $kboPython -X utf8 -B -m src.models.pitch_outcome predict --asset $run.paths.model --input $run.paths.input
if ($LASTEXITCODE -ne 0) { throw 'M5 predict failed' }
($predictionJson -join "`n") | & $kboPython -X utf8 -B -c 'import json, pathlib, sys; expected=json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")); actual=json.load(sys.stdin); equal=expected==actual; print("reproduce_equal="+str(equal)); sys.exit(0 if equal else 1)' $run.paths.prediction
if ($LASTEXITCODE -ne 0) { throw 'M5 independent reproduction failed' }
```

검증 증거: 테스트 수·종료 코드, 실제 자료와 제외·미출현 통계, 세 후보 및 세 확률 축의 지표, 실행 ID·경로·지문·자원 사용, `reproduce_equal=True`, `final_test_evaluated=false`. 성능 지표를 보고할 때 `evaluation.json`의 동일 분모와 범주별 보정 자료를 사용한다. 실제 GitHub 상태와 기술 수용은 별도 확인한다.
