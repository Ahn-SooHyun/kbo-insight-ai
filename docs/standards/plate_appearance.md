# 타석 결과 예측 실행과 결과 계약

관련 작업: [Issue #18 타석 결과 예측](https://github.com/Ahn-SooHyun/kbo-insight-ai/issues/18) · [예측 계약](project_contract.md#m2) · [강의 방법의 적용 기준](../planning/development_workflow.md#lecture-methods)

## 1. 범위 및 입력 시점

M2는 정규시즌의 검증된 타석 시작 직전 정보로 종료 결과 15개의 공동 확률을 제공한다. 종료 참가자, 해당 타석의 실제 투구·교체·종료 상태, 사건 정답, 무투구·감사·제외 플래그, 출처·경기·표본 키를 예측 변수로 사용하지 않는다. 경기·표본 키는 검증과 추적에만 사용한다. 포스트시즌 적용과 2026년 최종 Test 평가는 수행하지 않는다.

팀 기록은 동일 시즌 정규시즌의 대상 날짜 엄격 이전, 선수 기록은 대상 날짜 엄격 이전에 관측된 품질 검증 후보와 식별 근거를 사용한다. 최근 기록은 28일이다. 당일 다른 경기의 종료 결과를 사용하지 않으며 과거 공개 시점의 완전한 복원을 주장하지 않는다.

## 2. 자료와 분할

- 모델 자료 실행 ID: `259b38decaac4ed68dd5a596cfbe4849`
- 표준 실행 ID: `e020dafedc8b4fd9ac6696d0e4e4a92e`
- 표준 검증 ID: `37b56c39caf5435e8fb2a17f993ec4df`
- 인수 DB SHA-256: `c8383a9d3e8930830a137d1b6791cbb26e2316dbbbdf4f0f436ceda9a43890d0`
- 입력: `data/processed/model_data/<자료 실행 ID>/model_data.sqlite3`, `manifest.json`, `feature_target_contract.json`, `train_eda.json`, `supplement-impact.json`

| 분할       | KST 기간                         | 인수 유효 타석 | 인수 유효 경기 | 인덱스 제외 |
| ---------- | -------------------------------- | -------------: | -------------: | ----------: |
| Train      | 2025-01-01 이전                  |        113,272 |          1,440 |         538 |
| Validation | 2025-01-01 이상, 2026-01-01 미만 |         55,691 |            720 |         449 |
| Test       | 2026-01-01 이상, 2026-11-26 미만 |      후속 평가 |      후속 평가 |   후속 평가 |

`m2_index`, `m2_inputs`, `m2_targets`를 `sample_id`로 일대일 연결하고 경기별 날짜·분할, 유효/제외 상태를 검사한다. DB는 읽기 전용이며 Test는 인덱스와 키 구조만 검사한다. 유효 무투구 볼넷(Train 188, Validation 122)을 보존하고 참가자 변경(122, 56), 미확정 반이닝 종료 후보(311, 143)는 별도 기록한다. 시작·종료 참가자 및 관측된 귀속 후보 일관성을 확인한다.

Validation을 2025-07-01 이전 보정 구간과 해당 날짜 이상 선정 구간으로 고정 분할한다. 같은 경기의 표본을 함께 유지하며 두 구간의 표본 수·경기 수·날짜 범위를 기록한다. 분류기와 범주 사전은 Train에서만 학습하고 보정 파라미터는 보정 구간에서만 적합한다. 선정 구간에서만 운영 후보를 선택한다. 전체 2025년 평가는 기술 통계이며 보정·선정에 사용된 자료다. 최종 Test와 혼동하지 않는다.

## 3. 입력 및 전처리

사전 입력은 `model.INPUT_COLUMNS`의 정확한 41개 키를 포함한다.

- `competition`(regular), `home_team_id`, `away_team_id`, `batter_id`, `pitcher_id`, `inning`, `half`(top/bot), `outs`(0~2), `on_first`, `on_second`, `on_third`(각 0/1), `home_score`, `away_score`.
- `home_`, `away_` 각각: `season_games`, `season_rf`, `season_ra`, `recent_games`, `recent_rf`, `recent_ra`.
- `batter_`, `pitcher_` 각각: `prior_games`, `prior_opportunities`, `prior_hits`, `prior_ab`, `prior_so`, `recent_games`, `recent_opportunities`, `recent_hits`.

기회 수는 타자 PA, 투수 BF다. 시즌 번호와 품질 제외 집계는 선택 입력에서 제외한다. 팀 기록을 공격/수비 관점으로 변환하고 이닝, 점수 차, 아웃·주자 조합을 사용한다. Count는 제한된 `log1p` 변환, 비율은 고정 상한 스케일을 적용한다. NULL·실제 0·분모 0을 별도 처리한다. ID 계수는 Train에서 20회 이상 관측된 ID에만 적합하며 신규·희소 ID는 다른 변수로 후퇴한다.

## 4. 결과 사전과 야구 규칙

| 코드                 | 결과          |
| -------------------- | ------------- |
| single               | 단타          |
| double               | 2루타         |
| triple               | 3루타         |
| home_run             | 홈런          |
| walk                 | 볼넷          |
| hit_by_pitch         | 몸에 맞는 공  |
| strikeout            | 삼진          |
| field_out            | 일반 아웃     |
| fielders_choice      | 야수선택      |
| field_error          | 실책          |
| sac_bunt             | 희생번트      |
| sac_fly              | 희생플라이    |
| double_play          | 병살          |
| triple_play          | 삼중살        |
| catcher_interference | 포수 타격방해 |

표준 `PA_EVENTS` 15개는 합치거나 삭제하지 않는다. 미등록·결측 유효 정답은 실패 처리한다. 타석 시작 아웃 수와 점유 주자 수에서 병살·삼중살·희생번트·희생플라이의 필요조건만 제한한다. 병살은 2아웃 미만 및 주자 1명 이상, 삼중살은 0아웃 및 주자 2명 이상, 희생번트와 희생플라이는 2아웃 미만 및 주자 1명 이상이다. 희생플라이에 3루 주자만을 강제하지 않는다. 타석 도중 주자 진루 가능성과 시작 상태만으로 판단할 수 없는 사건은 추가 제한하지 않는다.

Train과 Validation의 실제 정답이 제한 규칙과 충돌하는지 먼저 전수 검사한다. 충돌하면 건수와 표본 키를 로그에 기록하고 학습을 중단한다. 자료·라벨을 자동 수정하거나 제외하지 않는다. 운영 확률은 금지 결과에 정확히 0을 부여하며 나머지 클래스는 재정규화한다. 가능한 희소 결과는 Train 가산 평활화와 혼합 사전으로 양의 확률을 유지한다. 마스크는 필요조건일 뿐 완전한 야구 상태 전이 엔진은 아니다.

## 5. 후보 학습과 보정

`model.py`는 기존 분류기/변환의 저수준 비교 구현을 유지한다. `probability.py`는 운영 확률 계약을 담당한다. 외부 운영 추론은 `probability.predict` 및 `__main__.load_operational_model` 또는 CLI를 사용한다. 기존 `model.predict` 및 `__main__.load_model`은 기존 v1 자산 비교·보존용이며 새 운영 추론에 사용하지 않는다. 새 스키마는 `m2-plate-appearance-v2`다. v1 파일을 덮어쓰거나 v2로 가장하지 않는다.

Train 전체 빈도는 범주별 0.5 가산 평활화, 조건부 빈도는 아웃·주자 점유 비트마스크별 집계(최소 30건, 사전 의사 표본 200건)를 사용한다. 운영 기준에도 동일 규칙 마스크를 적용한다.

기존 로지스틱 회귀는 Train에서 8 epoch SGD(초기 학습률 0.025, 감소 계수 0.35)로 학습한다. 개선 후보는 같은 입력 및 클래스 사전을 사용하되 허용 클래스 Softmax로 학습하고, 8 epoch SGD(초기 학습률 0.015, 감소 계수 0.5), epoch 종료 시 ID 계수 0.90·기타 비절편 계수 0.98의 고정 축소 및 마지막 4 epoch 계수 평균을 적용한다. 이는 명시적인 계수 축소와 평균화이며 엄밀한 전체 L2 최적해를 주장하지 않는다. 두 로지스틱 후보 모두 운영 시 허용 Softmax 99%와 허용 Train 사전 확률 1%를 혼합한다.

두 로지스틱 후보 각각에 대해 보정 구간에서 고정 Temperature 후보(0.75, 0.85, 1.0, 1.15, 1.30, 1.50, 2.0)의 Log loss를 비교한다. 선택된 Temperature에서 클래스별 절편 보정을 추가 비교한다. 보정 구간 실제 정답이 30건 미만인 클래스는 절편을 0으로 유지한다. 나머지는 고정 30회 평균 NLL 기울기 업데이트(학습률 0.8, L2 계수 0.001, 절편 범위 ±2)로 적합한다. 보정은 양의 허용 확률의 로그에 적용하며 금지 클래스는 0을 유지한다. 별도 독립 예측기가 아니라 선택 모델의 후처리 파라미터다.

선정 구간에서 전체·조건부 빈도 기준 중 Log loss, Brier 순으로 기준을 고른다. 보정 후보는 자신의 미보정 부모 대비 Log loss·Brier·Macro Classwise ECE가 모두 악화되지 않고 하나 이상 개선될 때만 적격이다. 로지스틱 계열 후보는 우수 빈도 기준 대비 Log loss·Brier 모두 악화되지 않고 하나 이상 개선할 때만 적격이다. 적격 후보 중 Log loss, Brier, ECE 순으로 선정하고 없으면 우수 빈도 기준을 선택한다. 완전 동률은 미보정·단순 후보를 우선한다. 기존 v1의 무제한 예측은 진단 비교만 수행하며 운영 선정에 포함하지 않는다. 실제 개선되지 않으면 그 사실과 선택 근거를 기록한다. 운영 자산은 하나만 저장한다.

## 6. 평가 및 강의 방법

- Log loss는 실제 범주의 확률에 1e-15 하한을 적용한 음의 로그 평균이다. Brier는 15개 확률과 원핫 정답의 제곱오차 합을 표본 평균하며 범주 수로 다시 나누지 않는다.
- Accuracy, 고정 15개 클래스 Macro F1, 범주별 정답·최빈 예측 건수 및 Precision·Recall·F1, 혼동 행렬과 주요 혼동을 기록한다. 분모 0의 정밀도·재현율은 0이다.
- 클래스별 신뢰도는 폭 0.1의 10개 구간에서 건수, 평균 확률, 관측 비율을 집계한다. 마지막 구간은 1.0을 포함하며 빈 구간은 NULL이다. 클래스별 ECE는 각 구간의 절대 확률 오차를 표본 수로 가중하고, Macro Classwise ECE는 15개 클래스 ECE의 단순 평균이다. 구간 기반 ECE는 희소 클래스의 신뢰성을 보장하지 않는다.
- 금지 결과에 부여한 평균 확률 질량, 양의 금지 확률이 존재한 표본 수와 클래스별 질량을 기록한다. 운영 후보에서 양의 금지 확률이 발견되면 실패 처리한다.

강의의 로지스틱 회귀, 다중분류 확장, 비용함수 비교, 경사하강법 및 과적합 완화 원리를 적용한다. 시간순 검증을 무작위 K-fold·Stratified K-fold·OOB로 대체하지 않는다. 계수 평균화는 여러 독립 모델의 앙상블이 아니다. 클래스 가중치나 재표집을 강제해 실제 발생 확률을 왜곡하지 않는다. 트리·앙상블 모델의 추가 도입은 이번 범위 밖이며 후속 비교 필요성을 실제 결과로 판단한다.

## 7. 실행 및 재현

기존 Python 3.12 환경을 사용하며 추가 의존성은 없다. 아래 PowerShell 명령은 사용자 실행용이다.

```powershell
Set-Location -LiteralPath 'D:/Programming/kbo-insight-ai'
$kboPython = 'D:/Programming/kbo-insight-ai/.venv/Scripts/python.exe'
$runId = '259b38decaac4ed68dd5a596cfbe4849'
& $kboPython -X utf8 -B -m unittest tests.models.test_plate_appearance tests.models.test_plate_appearance_probability -v
if ($LASTEXITCODE -ne 0) { throw 'M2 테스트 실패' }
& $kboPython -X utf8 -B -m unittest discover -s tests
if ($LASTEXITCODE -ne 0) { throw '전체 회귀 테스트 실패' }
$oldEncoding = [Console]::OutputEncoding
try {
    [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
    $trainingJson = & $kboPython -X utf8 -B -m src.models.plate_appearance train --run-id $runId
    if ($LASTEXITCODE -ne 0) { throw 'M2 학습 실패' }
    $report = ($trainingJson -join "`n") | ConvertFrom-Json
} finally { [Console]::OutputEncoding = $oldEncoding }
$asset = $report.paths.model
$inputPath = $report.paths.input
$expected = $report.paths.prediction
& $kboPython -X utf8 -B -m src.models.plate_appearance verify --asset $asset --evaluation $report.paths.evaluation
if ($LASTEXITCODE -ne 0) { throw 'M2 자산 검증 실패' }
& $kboPython -X utf8 -B -c "import json, subprocess, sys; from pathlib import Path; asset, inputs, expected = sys.argv[1:]; actual = json.loads(subprocess.check_output([sys.executable, '-X', 'utf8', '-B', '-m', 'src.models.plate_appearance', 'predict', '--asset', asset, '--input', inputs], encoding='utf-8')); saved = json.loads(Path(expected).read_text(encoding='utf-8')); print('reproduce_equal=' + str(actual == saved)); sys.exit(0 if actual == saved else 1)" $asset $inputPath $expected
if ($LASTEXITCODE -ne 0) { throw 'M2 독립 프로세스 재현 실패' }
```

단독 추론: `python -X utf8 -B -m src.models.plate_appearance predict --asset <model.json 절대 경로> --input <사전 입력 JSON 절대 경로>`.

## 8. 산출물·검증·실패 처리

- `models/m2/<학습 실행 ID>/model.json`: 단일 선택 분류기와 보정 파라미터, 클래스·입력·방법 계약, 자료·환경·소스 해시, 평가 지문.
- `results/m2/<학습 실행 ID>/evaluation.json`: 분할·제외·정답 빈도·결측·참가자 ID 지원·조건부 후퇴, 규칙 감사, 전체 Validation·선정 구간 후보 평가, 기존 무제한 비교, 보정 적합·선정 근거.
- 같은 결과 폴더의 `example_input.json`, `example_prediction.json`: 독립 프로세스 추론 재현 예시.

학습 직후 저장 자산을 다시 읽고 동일 입력의 확률을 대조한 뒤 `model.json`을 확정한다. `verify`는 자산 JSON 지문, 평가 지문, 학습 당시 실행 소스 SHA-256 및 현재 파일, 예시 입력·저장 예측을 대조한다. 원격 Commit 일치 확인은 별도의 Git 읽기 전용 blob 비교로 수행한다. 기존 자료·모델·결과는 덮어쓰지 않는다. 미완료 실행이나 `model.pending.json`만 존재하는 실행은 완료 자산으로 보지 않는다. 실제 자료·자산·결과는 Git 추적 제외를 유지한다.

종료 코드 0은 성공, 2는 설정·자료·자산 오류, 3은 사용자 중단이다. 규칙 충돌, 해시·계약·표본 분모 불일치 시 원본을 변경하지 않고 원인을 확인한다. 실제 자료 학습·평가·검증 결과가 확보되기 전에는 성능 개선이나 Issue 완료를 확정하지 않는다.
