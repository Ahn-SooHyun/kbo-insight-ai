# 포스트시즌 시리즈·한국시리즈 우승 전망 실행과 결과 계약

KBO Insight AI · 구현안 v1.0.0 · 2026-11-28 KST

- 적용 경로: `docs/standards/postseason_forecast.md`
- 관련 작업: [Issue #28](https://github.com/Ahn-SooHyun/kbo-insight-ai/issues/28)
- 시작 기준: [24a103b](https://github.com/Ahn-SooHyun/kbo-insight-ai/commit/24a103b845c547152960d52aa64add33b0bdb3aa)
- 상태: 코드 제시 후 사용자 적용·검증 대기. 실제 실행 결과와 하위 수용은 미확인이다.

[프로젝트 안내](../../README.md) · [공통 예측·평가 계약](project_contract.md) · [정규시즌 전망](regular_season_standings.md) · [경기 결과·득점](game_outcome_score.md) · [개발 흐름](../planning/development_workflow.md)

## 1. 목적·지원 범위

정규시즌 최종 순위와 포스트시즌 대진의 공동 불확실성을 기존 M1 경기 결과 확률에 연결한다. 정규시즌 1위와 한국시리즈 우승은 서로 다른 사건이다. 시리즈 참가·승리 및 한국시리즈 진출·우승을 구단별로 계산하며 특정 순서의 대진을 조건으로 한 확률을 따로 집계한다.

2025년 규칙만 지원한다. 정규시즌 전망의 부모 실행 모드와 정규시즌 종료 후 원천 판본 재생 모드를 구분한다. 두 모드는 정식 과거 시점 백테스트가 아니다. 신규 독립 주 예측기는 없으며 기존 5개 기능군 6개 주 예측기를 유지한다. 재수집, 재학습, 원본 및 기존 결과 수정, 배포, 선발·전력·일정 난이도 보조 기능은 제외한다.

### 적용 규칙

| 라운드       | 대진          | 종료 조건                 | 기본 홈 순서                             | 조항          |
| ------------ | ------------- | ------------------------- | ---------------------------------------- | ------------- |
| 와일드카드   | 4위–5위       | 4위 1승 또는 1무, 5위 2승 | 상위, 상위                               | 30–33조       |
| 준플레이오프 | 3위–WC 승자   | 3선승                     | 상위, 상위, 하위, 하위, 상위             | 34–37조       |
| 플레이오프   | 2위–준PO 승자 | 3선승                     | 상위, 상위, 하위, 하위, 상위             | 38–41조       |
| 한국시리즈   | 1위–PO 승자   | 4선승                     | 상위, 상위, 하위, 하위, 하위, 상위, 상위 | 42–44조, 46조 |

연장은 최대 15회이다. WC 외에는 무승부 경기의 승수를 추가하지 않으며 기본 일정 후 해당 홈구장에서 추가 차전을 진행한다. 승리 조건 충족 시 남은 차전은 진행하지 않는다. 복수의 무승부로 서로 다른 추가 경기 구장 순서가 명확하지 않은 경우 미결정으로 둔다. 추가 경기에서도 무승부가 발생하면 해당 구장에 추가 경기를 보존한다. 시리즈 64경기 한도는 공식 규칙이 아닌 계산 안전 한도이며 임의 승자를 만들지 않는다. 이동일·실제 개최 시각·서스펜디드의 경기 중 상태는 모의하지 않는다.

규칙 버전: `kbo-2025-postseason-v1`.

- [2025 KBO 리그 규정 제30–46조](https://6ptotvmi5753.edge.naverncp.com/KBO_FILE/ebook/pdf/2025_%EB%A6%AC%EA%B7%B8%EA%B7%9C%EC%A0%95.pdf)
- [2025 포스트시즌 운영 발표, 2025-09-28](https://www.koreabaseball.com/MediaNews/Notice/View.aspx?bdSe=11694)
- [2025년 제1차 이사회 규정 개정 발표](https://www.koreabaseball.com/MediaNews/Notice/View.aspx?bdSe=11353)

## 2. 자료·분할·품질

### 인수 자료

| 자산               | 실행·검증 식별자                                                    |
| ------------------ | ------------------------------------------------------------------- |
| 표준 검증 DB       | `e020dafedc8b4fd9ac6696d0e4e4a92e:37b56c39caf5435e8fb2a17f993ec4df` |
| 모델 자료 DB       | `259b38decaac4ed68dd5a596cfbe4849`                                  |
| M1 모델            | `0d9433b8044e48e5a4d21bbdb9b432e5`, `m1-final-score-v1`             |
| 정규시즌 대표 실행 | `e24254420aee4d52acf2b1fb144c3d58`                                  |

사용자 전달 선행 검사에서 DB와 M1 자산의 SHA-256 및 자료 계보, 부모 산출물 세 파일의 해시가 일치했다. 부모 실행은 2025-09-01 KST 자정, Seed 42, 1,000회, 시드 확정 906회 및 미결정 94회다. 기존 전체 회귀 455개가 11.619초에 통과했다. 새 구현의 실행 결과는 아직 없다.

표준판의 포스트시즌 유효 경기 결과는 193개이며 2026년 14개를 제외하면 179개가 남는다. 이를 모두 평가 가능 정답으로 확정하지 않는다. 2001~2009년과 2021~2023년 경기 결과, 2024년 KS 이전 세 라운드의 점수가 비어 있다. 2025년 원천 판본의 16개 경기에는 `final_observed=0`, 라운드별 `series_complete=1`, `season_complete=NULL`인 자료가 포함되어 있다. 일정과 완료 기록을 구분하고 시즌 완료 필드를 임의 변경하지 않는다.

### 사실 정확성 결함 및 사용 제한

표준·모델 자료의 일치와 해시 검증은 사실 정확성 검증과 별개다. 원천 판본에 실제 경기 결과와 불일치하는 사례가 발견되었다.

| 경기       | 인수 원천 판본           | 외부 확인 기록            | 영향                           |
| ---------- | ------------------------ | ------------------------- | ------------------------------ |
| WC 1차전   | NC 4:3 삼성              | NC 4:1 삼성               | 승자는 동일하나 득점 정답 불가 |
| WC 2차전   | 삼성 6:2 NC              | 삼성 3:0 NC               | 승자는 동일하나 득점 정답 불가 |
| 준PO 2차전 | 삼성 승리, 10월 10일     | SSG 승리, 10월 11일       | 진행 중 승수·시점 정확성 불가  |
| 준PO 4차전 | 삼성 7:4 SSG, 10월 13일  | 삼성 5:2 SSG, 10월 14일   | 날짜·득점 정답 불가            |
| PO 5차전   | 한화 4:1 삼성, 10월 22일 | 한화 11:2 삼성, 10월 24일 | 종료 시점·득점 정답 불가       |
| KS 1차전   | 10월 25일                | 10월 26일                 | 공식 시점 불일치               |
| KS 5차전   | LG 홈                    | 한화 홈                   | 공식 홈 편성 규칙 충돌         |

위 사례는 원본 수동 보충이나 수정의 근거로 사용하지 않는다. 원천 판본의 전수 대조는 미완료다. 별도 승인 및 자동 자료 확보로 정정·검증된 판본이 필요하다. 그 전에는 원천 판본 모드가 명시적으로 선택된 경우에만 개발 재생을 허용하고, 공식 기록 기반 성능 평가를 금지한다. 실제 불일치한 점수는 M1 예측 입력에 넣지 않는다. 공식 홈 규칙과 충돌하면 해당 시리즈 및 후속 우승 결과를 미결정으로 유지한다.

외부 확인 자료:

- [WC 1차전 전적](https://www.yna.co.kr/view/AKR20251006019300007)
- [WC 2차전 및 삼성 준PO 진출, KBO 게재 기사](https://www.koreabaseball.com/MediaNews/News/KboPhoto/View.aspx?bdSe=508895)
- [준PO 2·3차전 경과, KBO 게재 기사](https://www.koreabaseball.com/MediaNews/News/Preview/View.aspx?bdSe=60344)
- [준PO 4차전 및 삼성 3승 1패 진출, KBO 게재 기사](https://www.koreabaseball.com/MediaNews/News/KboPhoto/View.aspx?bdSe=509946)
- [PO 5차전 전적, KBO 게재 기사](https://www.koreabaseball.com/MediaNews/News/BreakingNews/View.aspx?bdSe=60367)
- [KS 일정 순연, KBO 발표](https://www.koreabaseball.com/MediaNews/Notice/View.aspx?bdSe=11730)
- [KS 5차전 개최 장소, KBO 발표](https://www.koreabaseball.com/MediaNews/Notice/View.aspx?bdSe=11744)

## 3. 입력·전처리

`--run-id`는 모델 자료 실행 ID이며 `--parent-run`은 정규시즌 전망 실행 ID이다. `--asset`은 저장 M1 모델 경로, `--season`은 2025, `--as-of`는 KST 날짜 자정이다.

부모 실행은 상태·스키마·시즌·기준일·자료/모델 계보·파일과 소스 해시·완료/잔여/누락 일정·반복 ID·순위·진출 집합·시드 및 재집계 요약을 검증한다. 반복 횟수와 ID를 그대로 사용하며 독립적으로 대진을 재추출하지 않는다. 부모 파일과 DB는 읽기 전용이다.

정규시즌 종료 후에는 720경기 완전성 및 확정 순위를 검사한다. `--archive-assumption` 옵션이 없으면 원천 판본 재생을 거부한다. 기준일 이전 일정·유효 표준 경기의 키, 차전, 홈/원정, 점수와 결과, 출처를 대조한다. 완료 시리즈의 사후 메타데이터는 이미 적용한 경기만으로 종료 상태가 결정된 경우에만 대조한다. 진행 중 시리즈에서 미래 승수·승자 정보를 조회하지 않는다. 확정되지 않은 시드나 상충 결과는 임의 보충하지 않는다.

## 4. 정답·출력 계약

| 파일               | 스키마 및 의미                                                                       |
| ------------------ | ------------------------------------------------------------------------------------ |
| `scenario.json`    | `postseason-scenario-v1`, 규칙, 시점, 자료 계보, 원천 판본 가정, 고정 대진별 M1 확률 |
| `summary.json`     | `postseason-summary-v1`, 구단별·리그 공동·조건부 확률 및 분모                        |
| `iterations.jsonl` | `postseason-iteration-v1`, 반복 ID·시드·진출·시리즈 참가·개별 경기·승자·미결정 사유  |
| `manifest.json`    | 실행 ID, 버전, 입력·소스/결과 해시, Seed, 반복 수, 계산 상태                         |

`rounds.<stage>.participants`는 상위 시드 구단, 하위 시드 구단 순서다. `winner=null`은 결과 미결정이며 탈락이 아니다. `fixed=true`는 원천 판본 재생에서 완료 근거를 통과한 시리즈라는 뜻이지 공식 기록의 사실 정확성을 보증하지 않는다. 실제 자료에서 인수한 경기에는 `game_id`, 날짜 및 점수가 포함되며 모의 경기에서는 이 필드들이 null이다.

`status=completed`는 파일 생성 성공을 의미한다. `simulation_status=resolved/partial/unsupported`는 한국시리즈 우승 공동 결과의 유효 범위를 구별한다. 실패로 중단된 디렉터리는 완료 매니페스트가 없을 수 있으므로 완료 실행으로 취급하지 않는다.

## 5. 모델·학습·선정

M1의 허용 입력 `competition=regular`를 보존하고, 기준일 이전 정규시즌의 시즌 누적 및 최근 28일 득실점·경기 수만 이용한다. 해당 프로필을 포스트시즌의 홈·원정 대진에 적용하는 것은 정규시즌 모델의 전이 가정이다. 부모 반복은 개별 모의 경기의 득실점 이력을 제공하지 않으므로 기준일 이후 실제 정규시즌 결과나 다른 반복의 이력을 섞지 않는다. 시리즈 중 모의 결과를 M1 학습·프로필에 반영하지 않는다.

결과만 필요한 이번 기능은 `result_probabilities`에서 승·무·패를 범주형 표본 추출한다. 득점 쌍은 생성하지 않으므로 `joint`를 중복 사용하지 않는다. 홈·원정 순서의 대진별 예측을 캐시하고 동일 확률을 반복 사용한다. M1의 2025년 정규시즌 평균 예측 무승부율 2.07%, 관측 3.06% 차이 및 15회 연장·라인업·선발·포스트시즌 운영 전이 한계를 기록한다.

한 경기의 승무패가 주어지면 공식 규칙에 따라 시리즈 상태를 갱신한다. 이전 라운드 승자만 다음 라운드에 진입한다. `random.Random`을 사용하고 `seed:iteration_id`로 반복별 난수 스트림을 구분한다. 확정 상태에는 난수를 사용하지 않는다. 반복 ID와 Seed가 같다면 다른 반복의 경기 수나 미결정 여부가 해당 반복의 난수열에 영향을 주지 않는다.

## 6. 강의 원리 및 프로젝트 확장

확인된 과제 가이드의 문제 정의, 데이터 출처·전처리, 학습/검증 분리, 실험 근거, 평가 및 한계 설명 요구를 따른다. 개별 강의 슬라이드 원문은 이번 작업에서 확인하지 못했으므로 아래 기법 모두를 강의에서 직접 배웠다고 단정하지 않는다.

| 기법                                         | 적용 판단                                             |
| -------------------------------------------- | ----------------------------------------------------- |
| 지도학습·검증 분리                           | 기존 M1의 학습 결과 및 시점 계약을 인수               |
| 경험적 확률분포                              | 반복별 사건의 빈도를 추정 확률로 집계                 |
| 조건부 확률                                  | 동일 반복 내에서 해당 대진이 성립한 경우만 집계       |
| 몬테카를로·상태 전이                         | 프로젝트 확장 계산; 시리즈 규칙과 M1 확률 결합        |
| Seed·반복 안정성                             | 동일 입력 재현 및 반복 증가에 따른 표본 오차 비교     |
| Brier·Log loss                               | 공식 정답과 당시 이용 가능성이 증명된 평가에서만 사용 |
| 신규 Random Forest·앙상블·Grid/Random Search | 별도 학습이 없는 시뮬레이션이므로 미적용              |
| GPU                                          | CPU 범주형 표본 추출 및 집계에 필수적이지 않아 미적용 |

## 7. 평가 지표·실제 결과

팀·사건별로 전체 반복 N, 참 T, 거짓 F, 미결정 U를 기록한다. 유효 분모 V=T+F=N-U이며 V가 0이면 확률은 null이다. 유효 반복 확률은 T/V이다. 전체 반복의 논리적 범위는 [T/N,(T+U)/N]이며 통계적 신뢰구간이 아니다. 사건별 유효 분모가 다를 수 있으므로 팀별 확률의 단순 합을 1로 강제하지 않는다.

리그 공동 시리즈 승리 확률은 해당 라운드 승자가 결정된 동일 반복을 사용하며 유효 분모가 양수일 때 합이 1이다. 대진 조건부 집계는 라운드, 상위·하위 구단 순서를 지정하고 `formed`, `formation_unresolved`, `series_valid`, `series_unresolved` 및 팀별 사건 분모를 기록한다. 발생하지 않은 대진은 확률 null이다. KS 진출과 우승 및 정규시즌 1위는 별도 사건이다.

2025년 WC에서 삼성의 시리즈 진출(1승 1패, 4위 어드밴티지)은 독립 외부 기록과 비교할 수 있다. 다만 원천 점수가 틀리므로 승자·종료 규칙 재생 수준으로 제한한다. 2025년 전체 시즌 공식 경기 재생과 확률 성능 평가는 현재 지원하지 않는다. 2026년 최종 Test는 종합 검증 단계로 이관한다.

실제 시뮬레이션 수치, 신규 실행 ID, 테스트 성공 여부, 안정성 수치는 사용자 실행 이후 작성한다. 미실행을 통과로 보고하지 않는다.

## 8. 구현·산출물·재현성

| 구성                                      | 책임                                    |
| ----------------------------------------- | --------------------------------------- |
| `src/simulation/postseason/__init__.py`   | 버전·스키마·기본 상수                   |
| `rules.py`                                | 공식 규칙·홈 배정·무승부·종료 상태      |
| `data.py`                                 | 읽기 전용 자산·부모 실행·일정·완료 근거 |
| `engine.py`                               | M1 결과 추출·반복별 대진·확률 집계      |
| `__main__.py`                             | CLI 및 별도 신규 결과 저장              |
| `tests/simulation/test_postseason.py`     | 합성 규칙·계산 검증                     |
| `tests/simulation/test_postseason_cli.py` | 자료·계보·시점·CLI 통합 검증            |

결과 경로는 `results/postseason/<run_id>/`이다. 기존 부모 실행 및 M1 자산은 변경하지 않는다. 동일 입력과 Seed의 시나리오·요약·반복 파일은 동일한 내용으로 생성되는지 해시 대조한다. UUID·생성 시각이 포함된 매니페스트 파일 자체의 해시는 비교 대상이 아니다.

## 9. 검증·한계·후속 연결

- 사용자 전달 기준: 기존 455개 테스트 통과, 인수 파일 해시 일치, 부모 1,000회 중 906개 시드 확정·94개 미결정.
- 신규 규칙, 자료, 시뮬레이션, CLI 테스트는 작성안을 제공했으나 실행 미확인이다.
- 실제 부모 1,000회 연계, Seed 42 반복 재현, Seed 43 비교, 1,000/2,000회 안정성, WC 진행 중 및 완료 재생, KS 홈 불일치 차단을 실행 검증한다.
- 원천 기록의 사실 불일치 및 당시 공개 시각 미입증은 하위 검토에 보고한다. 정정 승인·전수 검증 전 공식 경기 단위 평가 및 전체 종료 시즌 공식 재현은 보류한다.
- README·공통 계약·개발 흐름 문서는 실제 검증 결과가 확보된 뒤 현재 상태·실행 링크·지원 제한을 반영한다. 기존 문서의 완료 상태를 검증 전에 확정하지 않는다.
- GitHub Issue #28은 시작 확인 시 Open이고, 선행 PR #30 병합 및 Issue #27 종료가 확인되었다. 이번 변경의 적용·Commit·Push·PR·Merge는 미확인이다.

## 10. 실행 명령

PowerShell, 저장소 루트 기준. 아래 명령은 구현 적용 후 실행한다.

```powershell
Set-Location -LiteralPath 'D:/Programming/kbo-insight-ai'
$py = 'D:/Programming/kbo-insight-ai/.venv/Scripts/python.exe'
$run = '259b38decaac4ed68dd5a596cfbe4849'
$asset = 'D:/Programming/kbo-insight-ai/models/m1/0d9433b8044e48e5a4d21bbdb9b432e5/model.json'
$parent = 'e24254420aee4d52acf2b1fb144c3d58'
& $py -X utf8 -B -m unittest discover -s tests/simulation -p 'test_postseason*.py'
Write-Output "postseason_test_exit=$LASTEXITCODE"
& $py -X utf8 -B -m unittest discover -s tests
Write-Output "regression_test_exit=$LASTEXITCODE"
& $py -X utf8 -B -m src.simulation.postseason simulate --run-id $run --asset $asset --season 2025 --as-of 2025-09-01 --parent-run $parent --seed 42
Write-Output "parent_exit=$LASTEXITCODE"
& $py -X utf8 -B -m src.simulation.postseason simulate --run-id $run --asset $asset --season 2025 --as-of 2025-10-07 --archive-assumption --iterations 1000 --seed 42
Write-Output "progress_exit=$LASTEXITCODE"
& $py -X utf8 -B -m src.simulation.postseason simulate --run-id $run --asset $asset --season 2025 --as-of 2025-10-08 --archive-assumption --iterations 1000 --seed 42
Write-Output "completed_wc_exit=$LASTEXITCODE"
& $py -X utf8 -B -m src.simulation.postseason simulate --run-id $run --asset $asset --season 2025 --as-of 2025-11-01 --archive-assumption --iterations 1000 --seed 42
Write-Output "archive_end_exit=$LASTEXITCODE"
```

시드·확률 분모·미결정 및 홈 충돌 검산을 별도로 수행한다. 종료 코드 0은 파일 생성 성공이며 공식 자료 품질이나 성능 평가의 통과를 의미하지 않는다.

### AI 활용 내역

ChatGPT(OpenAI)를 활용하여 기존 저장소 인터페이스의 읽기 전용 검토, 시리즈 규칙 조사, 구현안·테스트·문서 작성 및 검산 계획을 작성했다. 작성한 코드 및 실제 산출물의 적용·실행·검토는 사용자에게 맡겨져 있으며 이 문서 작성 시점에는 신규 구현의 실제 검증이 완료되지 않았다.
