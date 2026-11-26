# 타석 결과 예측 실행과 결과 계약

관련 작업: [Issue #18 타석 결과 예측](https://github.com/Ahn-SooHyun/kbo-insight-ai/issues/18) · [예측 계약](project_contract.md#m2) · [강의 방법의 적용 기준](../planning/development_workflow.md#lecture-methods)

## 1. 범위 및 입력 시점

M2는 정규시즌의 검증된 타석 하나를 표본으로 삼아 타석 시작 직전의 정보로 종료 결과의 공동 확률을 제공한다. 종료 참가자, 해당 타석의 실제 투구와 교체, 종료 상태, 사건 정답, 무투구·감사·제외 플래그, 출처와 경기/표본 키를 예측 변수로 사용하지 않는다. 경기·표본 키는 검증과 추적에만 사용한다. 포스트시즌 적용과 2026년 최종 Test 평가는 이 작업에서 수행하지 않는다.

기존 모델 자료의 팀 기록은 동일 시즌 정규시즌 경기 중 대상 날짜 엄격 이전, 선수 기록은 해당 날짜 엄격 이전에 관측된 품질 검증 후보와 식별 근거를 사용한다. 최근 기록은 28일이다. 당일 다른 경기의 종료 결과를 사용하지 않으며, 과거 공개 판본·라인업·일정의 완전한 시점 복원을 주장하지 않는다.

## 2. 자료와 분할

- 모델 자료 실행 ID: `259b38decaac4ed68dd5a596cfbe4849`
- 표준 실행 ID: `e020dafedc8b4fd9ac6696d0e4e4a92e`
- 표준 검증 ID: `37b56c39caf5435e8fb2a17f993ec4df`
- 인수 DB SHA-256: `c8383a9d3e8930830a137d1b6791cbb26e2316dbbbdf4f0f436ceda9a43890d0`
- 입력: `data/processed/model_data/<자료 실행 ID>/model_data.sqlite3` 및 같은 폴더의 `manifest.json`, `feature_target_contract.json`, `train_eda.json`, `supplement-impact.json`

| 분할       | KST 경계                         | 인수 유효 타석 | 인수 유효 경기 | 인덱스 제외 |
| ---------- | -------------------------------- | -------------: | -------------: | ----------: |
| Train      | 2025-01-01 이전                  |        113,272 |          1,440 |         538 |
| Validation | 2025-01-01 이상, 2026-01-01 미만 |         55,691 |            720 |         449 |
| Test       | 2026-01-01 이상, 2026-11-26 미만 |      후속 평가 |      후속 평가 |   후속 평가 |

수량은 인수 기준이며 새 실행의 확인 결과는 `evaluation.json`에 기록한다. `m2_index`, `m2_inputs`, `m2_targets`를 `sample_id`로 일대일 연결하고 경기별 날짜·분할과 유효/제외 상태를 검사한다. DB는 읽기 전용으로 열며 Test는 인덱스와 키 구조만 점검하고 정답 내용이나 성능을 계산하지 않는다.

유효한 무투구 볼넷(Train 188, Validation 122)은 보존한다. 참가자 변경 제외(122, 56)와 미확정 반이닝 종료 후보 제외(311, 143)는 별도 기록한다. 시작 참가자와 종료 참가자 및 관측된 귀속 후보의 일관성을 검사하며, 없는 귀속 후보를 모델 입력으로 대체하지 않는다. 실제 검증 자료의 해당 조건을 대조한다.

## 3. 입력 및 전처리

사전 입력 JSON은 `model.INPUT_COLUMNS`에 선언된 41개 키를 정확히 포함해야 한다. 키 순서는 `input_columns` 메타데이터에 명시하며 객체 자체의 직렬화 순서와 혼동하지 않는다.

- `competition`(regular), `home_team_id`, `away_team_id`, `batter_id`, `pitcher_id`, `inning`, `half`(top/bot), `outs`(0~2), `on_first`, `on_second`, `on_third`(각 0/1), `home_score`, `away_score`.
- `home_`, `away_` 각각: `season_games`, `season_rf`, `season_ra`, `recent_games`, `recent_rf`, `recent_ra`.
- `batter_`, `pitcher_` 각각: `prior_games`, `prior_opportunities`, `prior_hits`, `prior_ab`, `prior_so`, `recent_games`, `recent_opportunities`, `recent_hits`.

기회 수는 타자 PA, 투수 BF다. 시즌 번호와 품질 제외 집계인 `prior_rejected`는 선택 입력에 포함하지 않는다. 팀 기록은 공격/수비 관점으로 변환하고, 시작 점수 차·이닝·주자와 아웃 조합을 사용한다. Count는 제한된 `log1p` 변환, 비율은 고정 상한 스케일을 적용한다. NULL·실제 0·분모 0을 별도 처리한다. 타자/투수와 팀 ID 범주 계수는 Train에서 20회 이상 관측된 ID에만 적합한다. 신규 또는 희소 ID의 계수는 0으로 후퇴하며 다른 관측 입력을 이용한다. Validation에서 전처리 사전을 다시 적합하지 않는다.

## 4. 결과 사전과 분류기

클래스는 표준 `PA_EVENTS`의 15개를 유지하고 원천 코드에서 동일 코드로 매핑한다.

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

Train의 전체 빈도에는 범주별 0.5 가산 평활화를 사용한다. 조건부 빈도는 아웃 수와 주자 점유 비트마스크에 따라 그룹을 정하고 표본 30 미만인 그룹은 전체 빈도로 후퇴한다. 충분한 그룹은 전체 분포를 의사 표본 200건으로 사용해 평활화한다. 희소·미출현 정상 범주를 다른 결과와 합치지 않는다. 결측이나 미등록 유효 정답이 있으면 오류 건수를 집계하고 평가를 중단한다.

다항 로지스틱 회귀는 공용 Softmax 분류기 하나로 학습한다. Train 빈도 로그를 절편 초기값으로 사용하고, 시드가 고정된 Train 내부 순서 섞기와 SGD(8 epoch, 초기 학습률 0.025, epoch별 감소 계수 0.35)를 적용한다. 시간 분할을 무작위로 바꾸지는 않는다. 반복 횟수와 학습률은 평가 전에 고정하며 Validation 튜닝은 하지 않는다. 최종 출력은 Softmax 확률 99%와 Train 평활화 전체 빈도 1%의 혼합이다. 별도의 확률 보정기 또는 범주별 독립 모델을 추가하지 않는다.

동일한 2025 Validation 표본에서 전체 빈도, 조건부 빈도, 다항 로지스틱 회귀를 비교한다. 기준끼리는 Log loss 우선, Brier 차순으로 선택하며 동률은 전체 빈도를 우선한다. 로지스틱 후보는 우수 기준 대비 Log loss·Brier 모두 악화되지 않고 하나 이상 개선될 때 채택한다. 그렇지 않으면 우수 빈도 분류기를 선택하고 개선 미충족 사실을 기록한다. 운영 자산은 선택 분류기 하나만 저장한다. 2025 결과는 모델 선택에 사용되므로 선택과 독립된 일반화 성능이 아니다. 별도의 보정 학습은 수행하지 않는다.

## 5. 평가 및 강의 방법

- Log loss: 실제 범주의 예측 확률에만 1e-15 하한을 적용한 음의 로그 평균.
- Brier score: 각 표본의 15개 범주 확률과 정답 원핫 값의 제곱오차 합을 표본 평균. 범주 수로 다시 나누지 않는다.
- Accuracy, Macro F1, 범주별 정답·예측 건수와 정밀도·재현율·F1, 혼동 행렬과 주요 혼동을 함께 기록한다. Macro F1은 고정 15개 클래스 평균이며 분모 0인 정밀도·재현율은 0이다.
- 범주별 one-vs-rest 신뢰도 구간: 폭 0.1의 10개 구간에서 건수, 평균 확률, 관측 비율을 집계한다. 마지막 구간에 1.0을 포함하며 빈 구간은 NULL이다.

강의의 로지스틱 회귀 원리를 다중분류로 확장하고 기준 모델과 동일한 평가 표본에서 비교한다. 고정 반복 횟수와 범주 지원 제한은 과적합 완화 목적의 사전 설계이며 최적화 효과는 실측 결과로 확인한다. 무작위 K-fold, Stratified K-fold, OOB는 시간순 검증의 대체로 채택하지 않는다. 트리·앙상블·추가 보정 모델은 이번 비교 범위에 포함하지 않으며, 필요성은 결과를 보고 별도 판단한다. 실제 평가 결과와 모델 선택 이유는 실행 결과에 근거해 해석한다.

## 6. 실행 및 재현

기존 Windows PowerShell의 Python 3.12 환경을 재사용한다. 새 라이브러리 설치나 Lock 파일 변경은 필요하지 않다. 기존 `inspection` 환경이 없는 경우 README의 설치 절차를 따른다. 저장소에서 아래 명령을 사용자 실행한다.

```powershell
Set-Location -LiteralPath 'D:/Programming/kbo-insight-ai'
$kboPython = 'D:/Programming/kbo-insight-ai/.venv/Scripts/python.exe'
$runId = '259b38decaac4ed68dd5a596cfbe4849'
& $kboPython -X utf8 -B -m unittest tests.models.test_plate_appearance -v
if ($LASTEXITCODE -ne 0) { throw 'M2 테스트 실패' }
& $kboPython -X utf8 -B -m unittest discover -s tests
if ($LASTEXITCODE -ne 0) { throw '전체 회귀 테스트 실패' }
$trainingJson = & $kboPython -X utf8 -B -m src.models.plate_appearance train --run-id $runId
if ($LASTEXITCODE -ne 0) { throw 'M2 학습 실패' }
$report = ($trainingJson -join "`n") | ConvertFrom-Json
$asset = $report.paths.model
$inputPath = $report.paths.input
$expected = $report.paths.prediction
& $kboPython -X utf8 -B -c "import json, subprocess, sys; from pathlib import Path; asset, inputs, expected = sys.argv[1:]; actual = json.loads(subprocess.check_output([sys.executable, '-X', 'utf8', '-B', '-m', 'src.models.plate_appearance', 'predict', '--asset', asset, '--input', inputs], encoding='utf-8')); saved = json.loads(Path(expected).read_text(encoding='utf-8')); print('reproduce_equal=' + str(actual == saved)); sys.exit(0 if actual == saved else 1)" $asset $inputPath $expected
if ($LASTEXITCODE -ne 0) { throw 'M2 독립 프로세스 재현 실패' }
```

단독 추론: `python -X utf8 -B -m src.models.plate_appearance predict --asset <model.json 절대 경로> --input <사전 입력 JSON 절대 경로>`.

## 7. 산출물과 실패 처리

- `models/m2/<학습 실행 ID>/model.json`: 단일 선택 분류기, 클래스 사전, 입력 열과 설정, 학습 파라미터, 실행·자료·환경·소스 해시, 평가 지문.
- `results/m2/<학습 실행 ID>/evaluation.json`: 분할·제외·정답 빈도·결측·신규 ID·조건부 후퇴·비교 지표·보정·혼동·선택 근거·출력 경로.
- 같은 결과 폴더의 `example_input.json`, `example_prediction.json`: 동일 사전 입력으로 저장 자산을 복원해 추론할 때 사용하는 예시.

확률은 유한하고 0~1 범위이며 합이 1이어야 한다. 모델 자산의 JSON 지문과 스키마·클래스·입력 순서를 확인한다. 학습 직후 저장 자산을 다시 읽어 동일 입력의 예측을 확인한 뒤 `model.json`을 확정한다. 새 프로세스에서 다시 대조한다. 실행 실패 또는 `model.pending.json`만 남은 실행은 완료 자산으로 취급하지 않는다. 기존 데이터·모델·결과는 덮어쓰지 않는다. 실제 자료 및 산출물은 Git 추적 제외를 유지한다.

실행 성공 코드 0, 설정/자료/자산 오류 2, 사용자 중단 3. 인수판 해시·자료 계약·표본 분모가 다르면 원본을 수정하지 말고 원인을 확인한다. 실패 시 명령, 종료 코드, 관련 오류 메시지를 보존하되 비밀값은 제거한다.

이 문서는 구현안의 실행 계약을 설명한다. 실데이터 성능, 소요 시간, 테스트 통과 및 최종 수용은 실제 실행 증거가 확보된 뒤 확정한다.
