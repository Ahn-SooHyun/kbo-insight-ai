# 고정 모델 2026 최종 평가·누수 검증

KBO Insight AI · 기술 계약 및 실행 안내 · [Issue #34](https://github.com/Ahn-SooHyun/kbo-insight-ai/issues/34)

저장소 적용 경로: `docs/standards/final_evaluation.md`

## 목적과 범위

기존 운영 예측기 6개(M1, M2, M4, M5, M3 타자, M3 투수)의 고정 자산을 평가한다. KST 기준 Train은 2025-01-01 이전, Validation은 2025년, 최종 Test는 2026-01-01 이상 2026-11-26 미만이다. 평가 결과를 이용한 운영 모델 학습, 선택, 전처리 변경 및 보정은 금지한다. 모델별 적용 범위는 정규시즌이며 포스트시즌은 별도 제외 집계한다.

본 문서는 실행 계약과 제안된 구현을 설명한다. 실제 적용 여부, 2026년 성능, 신규 테스트 통과, 하위 수용 및 병합 여부는 실행 증거가 확보되기 전까지 미확정이다.

## 고정 실행과 무결성

`src.evaluation.fixed_assets`는 자료 DB, 표준 DB, 계약, Train EDA, 보완 영향 자료 및 운영 모델과 기존 개발 평가의 SHA-256과 내부 Fingerprint를 대조한다. M3의 현재 파일 바이트 지문과 기존 기록이 다를 경우 기존 UTF-8 JSON 저장 규칙으로 재구성한 바이트의 SHA-256이 기존 기록과 정확히 일치할 때만 표현 형식 차이로 인정한다. 현재 파일, 기존 지문 및 재구성 지문을 모두 기록하며 어느 지문으로도 일치를 증명하지 못하면 차단한다. 저장 파일은 수정하지 않는다.

`prepare`는 기존 기준, 코드·환경·자산 지문, 입력과 정답의 계약, 모집단과 지표 및 제외 규칙을 고정한다. Train 통계로 기준을 복원하며 이전 개발 단계에서 확정한 비교 기준을 유지한다. 기존 개발 로더가 반환하는 Validation 자료는 기준 적합에 사용하지 않는다. 운영 모델은 재학습하거나 보정하지 않는다. 명세 생성 성공은 성능 평가 성공과 다르다.

`run`은 사용자가 지정한 명세 SHA-256을 확인한 뒤 실행한다. 입력을 명시적 열 목록으로 제한하고 읽기 전용 SQLite 연결의 Authorizer로 정답 및 사후 감사 열 접근을 차단한다. 추론 저장 후 원천 자료 재생성 대조를 수행하고 정답을 키로 연결한다. 원천 재생성은 새 임시 DB에서 기존 생성 코드를 사용하는 동일 변환의 재현 검사이며 독립적인 생성 알고리즘은 아니다. 과거 공개 시각의 완전한 복원을 주장하지 않는다. 정규시즌 종료 및 공식 정답 근거는 평가 단계에서 확인한다.

`verify`는 저장 예측·정답·기준 및 명세의 지문을 대조하고 기존 지표 구현을 새 실행에서 다시 계산한다. 재학습이나 재추론을 수행하지 않는다. 동일 경로의 최초 결과는 덮어쓰지 않는다.

## 모델별 평가 기준

| 모델 | 운영 및 비교 출력                                                       | 기준                                              |
| ---- | ----------------------------------------------------------------------- | ------------------------------------------------- |
| M1   | 홈·원정·통합 MAE, RMSE, 이산 CRPS, 승무패 Log loss·Brier·신뢰도         | Train 홈·원정 경험 분포 및 결과 빈도              |
| M2   | Log loss·Brier·Macro F1·범주별 지원·신뢰도 및 규칙 충돌                 | 기존 개발에서 확정한 전체 또는 조건부 빈도        |
| M4   | 동일 공격팀 승리 Log loss·이진 Brier·ECE, 반이닝 잔여 기대득점 MAE·RMSE | Train 상황 빈도와 전체 평균 RE                    |
| M5   | 공동 및 구종·결과 주변 Log loss·Brier·top-label 및 macro classwise ECE  | Train 전체·조건부 공동 빈도                       |
| M3   | 역할·기간·기록별 Count MAE·RMSE, 공통 유효 비율 MAE, 구간 포함률·폭     | 저장된 Train 과거 기록 기준과 기존 개발 구간 자산 |

다중분류 Brier는 클래스별 제곱오차 합의 표본 평균, M4는 이진 제곱오차의 평균이다. M4 확정 종료 post는 관측 상태에서 계산하는 결정론적 출력이며 미종료 상태와 구분한다. M3 7일 비율은 Count에서 계산한 진단이며 별도 저장 비율 정답은 28일·잔여 기간에 해당한다. 기간별 미관측, 실제 0, 분모 0, 구간 미지원 및 완료되지 않은 정답을 구분한다. Count 정규화 집계는 기존 Train 척도를 사용한다. 구간 폭만으로 구간 품질 개선을 단정하지 않는다.

M1·M2의 Log loss 수치 절단, M4 확률 경계, M5 양의 확률 요구 및 기존 희소 후퇴 규칙을 유지한다. 기준과 운영 지표는 공통 유효 표본에서 비교한다. 성능 악화 및 평가 불가도 보존한다.

## 실행 및 산출물

프로젝트 루트에서 지정된 Python 3.12 환경을 사용한다. 다음 명령의 매니페스트 경로 및 지문은 실제 `prepare` 출력으로, 결과 지문은 실제 `run` 출력으로 대체한다.

```powershell
$python = "D:/Programming/kbo-insight-ai/.venv/Scripts/python.exe"
$root = "D:/Programming/kbo-insight-ai"
Set-Location $root
& $python -m unittest discover -s tests -p test_final_evaluation.py -v
& $python -m src.evaluation.final_evaluation --project-root $root prepare
& $python -m src.evaluation.final_evaluation --project-root $root run --manifest $manifestPath --manifest-sha256 $manifestHash
& $python -m src.evaluation.final_evaluation --project-root $root verify --manifest $manifestPath --manifest-sha256 $manifestHash --result-sha256 $resultHash
& $python -m unittest discover -s tests
```

사전 검사 또는 명세 검사가 실패하면 실제 평가로 진행하지 않는다. 평가 실패는 완료로 기록하지 않으며, 최초 평가에서 정책을 변경해야 하는 경우 원래 명세·결과를 보존하고 독립 검증의 지위를 다시 판단한다.

실행 폴더는 `results/final_evaluation/<run_id>/`이다. 주요 산출물은 `manifest.json`, `baselines/`, `evaluation/started.json`, `evaluation/prediction_stage.json`, `evaluation/leakage_audit.json`, 모델별 `predictions.jsonl`, `targets.jsonl`, `metrics.json`, `evaluation/result.json`, `evaluation/recalculation-<id>.json`이다. 정답 및 입력의 식별 키와 원천 연결 근거, 지표, 제외와 지원 범위가 이 산출물에 포함된다. 대형 실자료와 예측은 로컬에 보존한다. 원천 재생성 감사에 별도의 임시 디스크 공간이 필요하며 생성한 임시 DB만 정리한다.

설명 후속 작업은 실행이 완료된 실제 매니페스트와 지표·예측·정답 파일을 인수한다. 실행 성공, 원천 재생성 대조, 저장 결과 재계산, 사용자 회귀 테스트 및 하위 수용은 별개로 확인한다.

관련 계약: [공통 예측·평가 계약](project_contract.md), [개발 흐름](../planning/development_workflow.md), [경기 결과·득점](game_outcome_score.md), [타석 결과](plate_appearance.md), [경기 상황 가치](game_state_value.md), [다음 투구](pitch_outcome.md), [선수 미래 성적](player_performance.md).
