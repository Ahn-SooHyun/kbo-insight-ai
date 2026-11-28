# 고정 모델 2026 최종 평가·누수 검증

| 항목           | 기준                                                                                                                           |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| 문서 역할      | 기술 계약·고정 평가 실행                                                                                                       |
| 적용 범위      | 고정 예측기 6개, 최종 Test, 누수 감사 및 지표 검산                                                                             |
| 확인 기준일    | 2026-11-29 (KST)                                                                                                               |
| 저장소 경로    | `docs/standards/final_evaluation.md`                                                                                           |
| 기준 소스      | [`ea4ef45`](https://github.com/Ahn-SooHyun/kbo-insight-ai/commit/ea4ef457fa35dbc37525cd97586a9f51543e31c5)                     |
| 연관 작업      | [#34](https://github.com/Ahn-SooHyun/kbo-insight-ai/issues/34), [#40](https://github.com/Ahn-SooHyun/kbo-insight-ai/issues/40) |
| 자료 이용 범위 | 승인된 비영리 대학 과제용 로컬 처리 및 내부 평가                                                                               |

## 요약

운영 예측기 6개를 2026년 정규시즌 고정 Test에서 평가한 기록과 누수 감사 절차다. 선정과 보정은 개발 단계에 한정하고, 평가 실행과 저장 결과의 검산을 구분한다.

[기준 소스의 이 문서](https://github.com/Ahn-SooHyun/kbo-insight-ai/blob/ea4ef457fa35dbc37525cd97586a9f51543e31c5/docs/standards/final_evaluation.md)에서 역사적 서술과 실행 이력을 확인할 수 있다. 과거 실행의 지문은 당시 기록과 일치하는 별도 판본을 따른다.

## 목차

- [1. 목적·지원 범위](#overview)
- [2. 자료·분할 및 정보 경계](#data)
- [3. 입력·전처리·정답 연결](#inputs)
- [4. 정답·출력 및 지표 계약](#outputs)
- [5. 고정 실행·검산 방법](#method)
- [6. 강의 원리 및 프로젝트 확장](#lecture-methods)
- [7. 평가 결과 및 해석](#evaluation)
- [8. 산출물·판본·재현성](#reproducibility)
- [9. 검증·한계·후속 연결](#limitations)
- [10. 실행 명령](#commands)

<a id="overview"></a>

## 1. 목적·지원 범위

기존 운영 예측기 6개(M1, M2, M4, M5, M3 타자, M3 투수)의 고정 자산을 평가한다. KST 기준 Train은 2025-01-01 이전, Validation은 2025년, 최종 Test는 2026-01-01 이상 2026-11-26 미만이다. 평가 결과를 이용한 운영 모델 학습, 선택, 전처리 변경 및 보정은 금지한다. 모델별 적용 범위는 정규시즌이며 포스트시즌은 별도 제외 집계한다.

2026-11-29 KST 기준 Issue #34의 기술 수용과 [PR #37](https://github.com/Ahn-SooHyun/kbo-insight-ai/pull/37) 병합([`11f996b`](https://github.com/Ahn-SooHyun/kbo-insight-ai/commit/11f996b6ac04c0edac706d4b16599e49cc6ebdc5)), Issue 완료 종료를 확인했다. 기존 고정 모델 평가의 결과·지표 및 사용 범위는 해당 실행 명세에서 관리한다. 본 문서의 개정으로 새로운 최종 Test 실행이나 모델의 성능 변화가 발생한 것은 아니다.

<a id="data"></a>

## 2. 자료·분할 및 정보 경계

| 자료 구간                               | 역할                           | 금지 사항                                      |
| --------------------------------------- | ------------------------------ | ---------------------------------------------- |
| 2025-01-01 이전                         | Train 및 기존 기준 통계        | 미래 관측·정답 사용 금지                       |
| 2025-01-01 이상 ~ 2026-01-01 미만       | 개발 Validation·선정·보정 근거 | 독립 최종 Test로 해석 금지                     |
| 2026-01-01 이상 ~ 2026-11-26 미만 (KST) | 고정 최종 Test                 | 재학습·새 변수 선정·재보정·Threshold 조정 금지 |

모델별 적격 표본·정답 완료 조건과 제외 규칙을 그대로 사용한다. 같은 경기에서 나온 여러 상태·타석·투구 및 동일 선수의 겹치는 주차는 독립 관측이라고 가정하지 않는다. 모델 자료와 표준 검증판은 [자산 계보](execution_handover.md#data)를 따른다.

<a id="inputs"></a>

## 3. 입력·전처리·정답 연결

평가 입력은 고정 명세의 명시적 열 목록으로 구성한다. 예측 전에 정답 또는 사후 감사 열을 읽지 못하게 하고, 저장 추론과 원천 재생성 대조 후 식별 키로 정답을 연결한다. 원천 재생성은 동일 변환 구현의 재현 검사이며 별개 알고리즘의 검증이 아니다. 예측기별 출력과 제외 사유를 원래 계약과 같은 방식으로 유지한다.

<a id="outputs"></a>

## 4. 정답·출력 및 지표 계약

| 모델 | 운영 및 비교 출력                                                       | 기준                                              |
| ---- | ----------------------------------------------------------------------- | ------------------------------------------------- |
| M1   | 홈·원정·통합 MAE, RMSE, 이산 CRPS, 승무패 Log loss·Brier·신뢰도         | Train 홈·원정 경험 분포 및 결과 빈도              |
| M2   | Log loss·Brier·Macro F1·범주별 지원·신뢰도 및 규칙 충돌                 | 기존 개발에서 확정한 전체 또는 조건부 빈도        |
| M4   | 동일 공격팀 승리 Log loss·이진 Brier·ECE, 반이닝 잔여 기대득점 MAE·RMSE | Train 상황 빈도와 전체 평균 RE                    |
| M5   | 공동 및 구종·결과 주변 Log loss·Brier·top-label 및 macro classwise ECE  | Train 전체·조건부 공동 빈도                       |
| M3   | 역할·기간·기록별 Count MAE·RMSE, 공통 유효 비율 MAE, 구간 포함률·폭     | 저장된 Train 과거 기록 기준과 기존 개발 구간 자산 |

다중분류 Brier는 클래스별 제곱오차 합의 표본 평균, M4는 이진 제곱오차의 평균이다. M4 확정 종료 post는 관측 상태에서 계산하는 결정론적 출력이며 미종료 상태와 구분한다. M3 7일 비율은 Count에서 계산한 진단이며 별도 저장 비율 정답은 28일·잔여 기간에 해당한다. 기간별 미관측, 실제 0, 분모 0, 구간 미지원 및 완료되지 않은 정답을 구분한다. Count 정규화 집계는 기존 Train 척도를 사용한다. 구간 폭만으로 구간 품질 개선을 단정하지 않는다.

M1·M2의 Log loss 수치 절단, M4 확률 경계, M5 양의 확률 요구 및 기존 희소 후퇴 규칙을 유지한다. 기준과 운영 지표는 공통 유효 표본에서 비교한다. 성능 악화 및 평가 불가도 보존한다.

<a id="method"></a>

## 5. 고정 실행·검산 방법

`src.evaluation.fixed_assets`는 자료 DB, 표준 DB, 계약, Train EDA, 보완 영향 자료 및 운영 모델과 기존 개발 평가의 SHA-256과 내부 Fingerprint를 대조한다. M3의 현재 파일 바이트 지문과 기존 기록이 다를 경우 기존 UTF-8 JSON 저장 규칙으로 재구성한 바이트의 SHA-256이 기존 기록과 정확히 일치할 때만 표현 형식 차이로 인정한다. 현재 파일, 기존 지문 및 재구성 지문을 모두 기록하며 어느 지문으로도 일치를 증명하지 못하면 차단한다. 저장 파일은 수정하지 않는다.

`prepare`는 기존 기준, 코드·환경·자산 지문, 입력과 정답의 계약, 모집단과 지표 및 제외 규칙을 고정한다. Train 통계로 기준을 복원하며 이전 개발 단계에서 확정한 비교 기준을 유지한다. 기존 개발 로더가 반환하는 Validation 자료는 기준 적합에 사용하지 않는다. 운영 모델은 재학습하거나 보정하지 않는다. 명세 생성 성공은 성능 평가 성공과 다르다.

`run`은 사용자가 지정한 명세 SHA-256을 확인한 뒤 실행한다. 입력을 명시적 열 목록으로 제한하고 읽기 전용 SQLite 연결의 Authorizer로 정답 및 사후 감사 열 접근을 차단한다. 추론 저장 후 원천 자료 재생성 대조를 수행하고 정답을 키로 연결한다. 원천 재생성은 새 임시 DB에서 기존 생성 코드를 사용하는 동일 변환의 재현 검사이며 독립적인 생성 알고리즘은 아니다. 과거 공개 시각의 완전한 복원을 주장하지 않는다. 정규시즌 종료 및 공식 정답 근거는 평가 단계에서 확인한다.

`verify`는 저장 예측·정답·기준 및 명세의 지문을 대조하고 기존 지표 구현을 새 실행에서 다시 계산한다. 재학습이나 재추론을 수행하지 않는다. 동일 경로의 최초 결과는 덮어쓰지 않는다.

<a id="lecture-methods"></a>

## 6. 강의 원리 및 프로젝트 확장

강의의 원리·시각은 [강의 전사 범위·시각·분류 기준](../planning/development_workflow.md#lecture-methods)를 기준으로 연결한다. 다음의 구현·실행 기록은 강의 설명의 직접 적용, 문제에 맞춘 응용, 제공 전사에서 동일한 상세 절차가 확인되지 않은 프로젝트 확장, 비교에만 사용한 후보를 구분해 읽는다.

| 원리                                                                         | 적용 구분      | 실제 적용                                                           | 제한                                              |
| ---------------------------------------------------------------------------- | -------------- | ------------------------------------------------------------------- | ------------------------------------------------- |
| 공정 평가·일반화 (`ml_cv,eval-metric_빨강펜_SW중단.txt`, 00:00:34–00:01:36)  | 원리 직접 적용 | 선정과 분리된 2026 Test 및 공통 유효 분모                           | 무작위 K-fold 자체는 미실행                       |
| 손실 최소화·제곱오차 (`LinearRegression_빨강펜-seg1.txt`, 00:11:09–00:14:55) | 평가 원리 응용 | MAE·RMSE·기존 Train 기준 비교                                       | 선형회귀 재학습 아님                              |
| NLL·분류 확률 (`LogisticRegression_update(1)_빨강펜.txt`, 00:14:33–00:16:06) | 원리 응용      | Log loss·확률 벡터와 정답 비교                                      | 새로운 로지스틱 회귀 적합 아님                    |
| 혼동 행렬·평가지표                                                           | 원리 응용      | Macro F1·클래스별 지원·오분류 분석                                  | 전체 분류 성능을 단일 수치로 확정하지 않음        |
| Brier·CRPS·Reliability·ECE·시간순 누수 통제                                  | 프로젝트 확장  | 확률·분포 채점, 사전 명세, 읽기 전용 SQLite Authorizer, 지표 재계산 | 강의의 확인 발췌에서 동일 세부 구현 확인되지 않음 |
| Grid/Random Search·RF·OOB·Voting·Stacking                                    | 미적용         | 이미 고정된 모델을 평가하는 단계                                    | Test에서 재선정 금지                              |

강의 시각은 [선행 확인 기록](../planning/development_workflow.md#lecture-methods)에서 인수했다. 과제 가이드 4쪽의 Baseline·성능 비교·결론 원리를 적용한다. 이 문서 개정에서 전체 강의 원문을 새로 전수 확인하지 않았으며, 강의 외 확장은 확인 범위의 구분이다.

### 6.1 평가 원리와 감사 절차의 대응

| 단계 | 적용한 원리                                                                                                 | 여기서 확보할 증거                                                                 |
| ---- | ----------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| 준비 | 보지 않은 자료에서 평가한다는 강의 원리를 운영 자산 고정으로 구체화한다.                                    | 준비 시점의 모델·자료·코드·문서 지문과 기존 Train 기준의 동일성을 기록한다.        |
| 예측 | 프로젝트의 SQLite 권한·열 허용 목록과 결과 저장 순서로 입력과 정답을 분리한다.                              | 허용 입력만 조회한 기록, 고정 예측 행, 순서와 입력의 대응 및 조회 제한을 확인한다. |
| 채점 | NLL을 확률 비교에 쓰며, 제곱오차 개념을 Brier와 RMSE에 목적별로 적용한다. 점예측·분포·확률의 단위가 다르다. | 정답 연결 후 모델마다 유효 행, 제외 수, 고정 범주와 지표 결과를 확인한다.          |
| 검산 | 고정 예측·정답·기준으로 점수를 재계산하여 집계와 재현을 분리 확인한다.                                      | 동일 지표·분모·지문의 일치와 새로운 모델 추론의 실행 여부를 독립 기록한다.         |

M1의 CRPS, M2·M5의 범주 지표, M3의 마스크별 오차, M4의 상태 구간을 함께 읽는다. 낮은 평균 Log loss와 높은 ECE는 동시에 관찰될 수 있다. 다중분류 Brier는 범주별 제곱오차의 합을 표본 평균하므로 범주별 평균을 사용하는 다른 정의와 직접 비교할 수 없다. 자세한 적용 범위는 [채점·설명·인수 원리](../planning/development_workflow.md#lecture-validation-evidence)를 따른다.

<a id="evaluation"></a>

## 7. 평가 결과 및 해석

아래 수치는 기존 고정 최종 평가 보고 및 [예측 설명 문서](prediction_explanation.md#evaluation)의 대표 결과다. 이번 문서 편집에서 새로 평가하거나 소수점 자릿수를 재계산한 것은 아니다.

| 모델    |          모집단 | 주요 지표(기준 → 운영)                                             | 악화·제한                          |
| ------- | --------------: | ------------------------------------------------------------------ | ---------------------------------- |
| M1      |         720경기 | 승무패 Log loss 0.788515 → 0.785552; 통합 RMSE 3.441284 → 3.414770 | 홈 MAE 2.504545 → 2.510265         |
| M2      |      56,517타석 | Log loss 1.703398 → 1.686291; Macro F1 0.038805 → 0.048888         | F1 절대값 낮음                     |
| M3 타자 | 7,209 선수·주차 | 정규화 MAE 0.331951 → 0.331914; RMSE 0.523049 → 0.520457           | 기간·기록별 분모 별도              |
| M3 투수 | 7,178 선수·주차 | RMSE 0.502286 → 0.494237                                           | MAE 0.323385 → 0.364164            |
| M4      |     96,528 상태 | 승리 Log loss 0.471950 → 0.463950; RE MAE 0.730868 → 0.621927      | ECE 0.003629 → 0.008507            |
| M5      |     221,065투구 | 공동 Log loss 3.671111/3.854192 → 3.369668                         | 다수 ECE 악화; 공동·구종·결과 분리 |

운영 선정은 기존 개발 구간의 근거를 유지한다. 하나의 최종 Test에서 유리한 지표만 선택해 모든 성능 개선을 주장하지 않는다. 실제 지표 전량과 분모는 고정 `evaluation/<model>/metrics.json`을 정본으로 한다.

<a id="reproducibility"></a>

## 8. 산출물·판본·재현성

인수하는 고정 실행은 `results/final_evaluation/7d125b8a685a4693a34aea70445ef294/`이다. 다음 지문은 [새 환경 재실행 검증 #40](https://github.com/Ahn-SooHyun/kbo-insight-ai/issues/40)의 확인 결과에서 인수한 기록이다.

| 파일                                                             | SHA-256                                                            |
| ---------------------------------------------------------------- | ------------------------------------------------------------------ |
| `manifest.json`                                                  | `0874e6b39e2a17d08528842230ff947358096ef0c16b743da4f2d22cded5fb69` |
| `evaluation/result.json`                                         | `7e69978252525eb9f42d8563f1177ec73f1568432bc36b72d09ce0982dcb6713` |
| `evaluation/recalculation-74d07ff588d7469d900d093290c0c8b2.json` | `ef5e61b9f347ad76417d25029cb98e2169c8415170327b8f5c215270f28a9940` |

과거 명세의 코드·문서 지문은 [`d9b3af9577a796d94ab7b98846e03273a400b365`](https://github.com/Ahn-SooHyun/kbo-insight-ai/commit/d9b3af9577a796d94ab7b98846e03273a400b365)에서 113/113개 일치했다. #40은 기존 고정 참조의 무결성 및 환경을 대조하고 저장된 예측·정답·기준의 지표를 모델 6개 모두 새로 재계산하여 Fingerprint 일치를 확인했다. 과거 판본의 별도 체크아웃에서 전체 CLI를 실행하거나 현재 판본으로 새 명세와 전체 CLI를 실행한 기록은 없으며, 최종 Test 전량 재추론도 하지 않았다. 저장 지표 재계산은 전량 재추론과 다르다.

기존 명세를 재검증하려면 그 명세와 일치한 소스·문서 판본, 환경 및 원래의 절대 참조 경로를 별도로 확보한다. 현재 문서 개정은 과거 명세의 지문을 변경할 수 있으므로 불일치를 임의로 치환하지 않는다. 현재 판본으로 새 평가를 실행할 필요가 있는 경우에는 `prepare`로 새 명세를 만들고 새 실행 경로에서 계산한다. 원래 명세와 결과는 그대로 보존하며, 최종 Test의 기존 독립 평가 지위를 훼손하는 재학습·선정·보정은 수행하지 않는다.

관련 계약: [공통 예측·평가 계약](project_contract.md), [개발 흐름](../planning/development_workflow.md), [경기 결과·득점](game_outcome_score.md), [타석 결과](plate_appearance.md), [경기 상황 가치](game_state_value.md), [다음 투구](pitch_outcome.md), [선수 미래 성적](player_performance.md), [실행 및 저장 자산 인수 안내](execution_handover.md).

<a id="limitations"></a>

## 9. 검증·한계·후속 연결

고정 모델 최종 Test의 성공과 새 환경의 전량 재추론은 별개다. #40에서는 고정 자산 및 저장 지표를 검증했으나 전량 재추론은 하지 않았다. 2026 시즌 전망·보조 분석을 이 결과로 지원한다고 해석하지 않는다. 최종 Test 악화를 개선하기 위한 개발은 새 독립 평가 설계와 별도 승인 후에만 진행한다.

<a id="commands"></a>

## 10. 실행 명령

프로젝트 루트에서 지정된 Python 3.12 환경을 사용한다. 아래는 현재 소스에서 새 평가 명세를 만들고 검산까지 진행하는 Windows PowerShell 예시다. 소스와 승인 자산을 준비하고 [8절의 판본 경계](#reproducibility)를 먼저 확인한다. 매니페스트 경로·지문과 결과 지문은 실제 JSON 출력에서 얻는다. 기존 실행의 검산만 필요할 때는 해당 역사 판본의 승인된 명세와 지문을 확인한 뒤 `verify`에 전달한다. 현재 판본의 새 전체 실행은 이번 문서 검토에서 수행하지 않았다.

```powershell
Set-Location -LiteralPath 'D:/Programming/kbo-insight-ai'
$root = (Get-Location).Path
$python = Join-Path $root '.venv/Scripts/python.exe'

& $python -X utf8 -B -m unittest discover -s tests -p 'test_final_evaluation.py' -v
if ($LASTEXITCODE -ne 0) { throw '최종 평가 전용 테스트 실패' }

$prepareOutput = & $python -X utf8 -B -m src.evaluation.final_evaluation --project-root $root prepare
if ($LASTEXITCODE -ne 0) { throw '최종 평가 명세 준비 실패' }
$prepared = ($prepareOutput -join "`n") | ConvertFrom-Json -ErrorAction Stop
if ($prepared.status -ne 'frozen' -or $prepared.asset_count -ne 6) { throw '고정 명세 또는 운영 자산 수 불일치' }

$runOutput = & $python -X utf8 -B -m src.evaluation.final_evaluation --project-root $root run --manifest $prepared.manifest --manifest-sha256 $prepared.manifest_sha256
if ($LASTEXITCODE -ne 0) { throw '최종 평가 실행 실패' }
$evaluation = ($runOutput -join "`n") | ConvertFrom-Json -ErrorAction Stop
if ($evaluation.status -ne 'completed' -or $evaluation.completed_models -ne 6) { throw '최종 평가 완료 범위 불일치' }

$verifyOutput = & $python -X utf8 -B -m src.evaluation.final_evaluation --project-root $root verify --manifest $prepared.manifest --manifest-sha256 $prepared.manifest_sha256 --result-sha256 $evaluation.result_sha256
if ($LASTEXITCODE -ne 0) { throw '저장 결과 검산 실패' }
$verification = ($verifyOutput -join "`n") | ConvertFrom-Json -ErrorAction Stop
if ($verification.status -ne 'verified' -or $verification.matched_models -ne 6) { throw '저장 결과 일치 범위 오류' }
$verification

& $python -X utf8 -B -m unittest discover -s tests
if ($LASTEXITCODE -ne 0) { throw '전체 회귀 실패' }
```

사전 검사 또는 명세 검사가 실패하면 실제 평가로 진행하지 않는다. 평가 실패는 완료로 기록하지 않으며, 최초 평가에서 정책을 변경해야 하는 경우 원래 명세·결과를 보존하고 독립 검증의 지위를 다시 판단한다.

실행 폴더는 `results/final_evaluation/<run_id>/`이다. 주요 산출물은 `manifest.json`, `baselines/`, `evaluation/started.json`, `evaluation/prediction_stage.json`, `evaluation/leakage_audit.json`, 모델별 `predictions.jsonl`, `targets.jsonl`, `metrics.json`, `evaluation/result.json`, `evaluation/recalculation-<id>.json`이다. 정답 및 입력의 식별 키와 원천 연결 근거, 지표, 제외와 지원 범위가 이 산출물에 포함된다. 대형 실자료와 예측은 로컬에 보존한다. 원천 재생성 감사에 별도의 임시 디스크 공간이 필요하며 생성한 임시 DB만 정리한다.

설명 후속 작업은 실행이 완료된 실제 매니페스트와 지표·예측·정답 파일을 인수한다. 실행 성공, 원천 재생성 대조, 저장 결과 재계산, 사용자 회귀 테스트 및 하위 수용은 별개로 확인한다.
