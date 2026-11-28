# 실행 및 저장 자산 인수 안내

| 항목           | 기준                                                                                                                           |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| 문서 역할      | 운영 실행·재현 인수                                                                                                            |
| 적용 범위      | 설치, 저장 자산, 예측, 판본, 검증 및 오류 대응                                                                                 |
| 확인 기준일    | 2026-11-29 (KST)                                                                                                               |
| 저장소 경로    | `docs/standards/execution_handover.md`                                                                                         |
| 기준 소스      | [`ea4ef45`](https://github.com/Ahn-SooHyun/kbo-insight-ai/commit/ea4ef457fa35dbc37525cd97586a9f51543e31c5)                     |
| 연관 작업      | [#40](https://github.com/Ahn-SooHyun/kbo-insight-ai/issues/40), [#41](https://github.com/Ahn-SooHyun/kbo-insight-ai/issues/41) |
| 자료 이용 범위 | 승인된 비영리 대학 과제용 로컬 처리 및 내부 평가                                                                               |

## 요약

잠금 설치 후 저장 예측기 6개와 승인된 로컬 자산의 계보를 확인한다. 먼저 저장 M1의 대표 추론을 비교하고, 목적에 따라 시즌 결과와 고정 명세를 검증한다.

기준 소스에는 이 경로의 문서가 없다. 검증 당시의 기재와 실행은 [새 환경 재실행 검증 #40](https://github.com/Ahn-SooHyun/kbo-insight-ai/issues/40) 및 7절에서 확인한다.

## 목차

- [1. 목적·실행 전제](#overview)
- [2. 자료·자산 계보](#data)
- [3. 입력·대표 추론](#inputs)
- [4. 시즌·보조 분석](#outputs)
- [5. 고정 결과와 판본](#method)
- [6. 강의 원리 및 프로젝트 확장](#lecture-methods)
- [7. 검증 증거와 완료 이력](#evaluation)
- [8. 재현·오류 대응](#reproducibility)
- [9. 한계·인수 확인](#limitations)
- [10. 실행 명령](#commands)

<a id="overview"></a>

## 1. 목적·실행 전제

1. 승인된 소스 사본과 Python 3.12.15, uv 0.13.7을 준비한다. [설치 안내](../../README.md#commands)의 잠금 설치와 의존성 확인을 진행한다. 검증된 환경은 Windows 11 AMD64, CPython 3.12.15, SQLite 3.53.1, 프로젝트 0.1.0(editable), PyArrow 26.0.1이다.
2. 프로젝트 루트의 `kbo.toml`을 확인한다. `--project-root`, 프로세스 `KBO_PROJECT_ROOT`, 편집 가능 설치의 소스 위치 순서로 루트를 선택한다. 잘못된 상위 선택을 자동으로 대체하지 않는다. 상대 자료 경로는 현재 작업 디렉터리가 아니라 선택된 루트를 기준으로 해석한다. 상세 설정은 [환경·설정·저장 책임](../../README.md#running)을 따른다.
3. 설치와 `check` 성공만으로 자료·모델의 존재나 정상 예측이 보장되지는 않는다. 다음 절에 따라 승인된 기존 자산을 확보한다. 예측에 필요한 저장 모델과 대표 입력만 준비하는 경우에는 원본 수집이나 학습을 하지 않는다.
4. 독립 추론 예제를 실행하고 기준 출력과 비교한다. 시즌 전망이나 종합 검증은 추가 자료가 필요하므로 해당 실행 계약으로 진행한다.

기본 확인 명령과 정책상 설치 EXE가 차단될 때의 조치는 [10절](#commands)에 있다.

<a id="data"></a>

## 2. 자료·자산 계보

Git 소스와 별도 승인 자산은 구분한다. 소스만 받아서는 아래 DB·모델·실행 결과가 모두 제공되지 않는다. 자산 보유자가 사용 허용 범위를 확인한 뒤 승인된 복제본을 제공해야 한다. 승인된 비영리 대학 과제의 로컬 처리·내부 평가 범위에서 사용하며 원본 행, DB, 모델 본문, 비밀값을 공개 문서나 Git에 올리지 않는다.

아래 상대 경로는 배포 `kbo.toml`의 경로 값을 기준으로 한다. 다른 설정의 경우 `check` 결과와 저장 명세의 실제 위치를 먼저 확인한다. 단독 예측에는 해당 모델 JSON과 그 계약에 맞는 입력 JSON이 필요하며, 재현 비교에는 기존 기준 출력과 모델별 추가 명세가 필요하다. 시뮬레이션과 종합 검증에는 별도의 DB·결과·고정 참조 파일도 필요하다.

| 자산             | 위치 및 식별자                                                                                                                                                 |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 불변 원본        | `data/raw/published/objects/`, `data/raw/published/runs/`, `data/raw/sources/`                                                                                 |
| 인수 수집 실행   | `4bdbd03e20634ce58135ec2311555351`                                                                                                                             |
| 기초 표준 / 검증 | `06ae4302ed7544d089fb0c5c8deb5719` / `d936d50a5320463b8dd1c1331569ff19`                                                                                        |
| 보완 표준 / 검증 | `e020dafedc8b4fd9ac6696d0e4a92e` / `37b56c39caf5435e8fb2a17f993ec4df`                                                                                          |
| 모델 자료        | `data/processed/model_data/259b38decaac4ed68dd5a596cfbe4849/model_data.sqlite3` 및 같은 폴더의 `manifest.json`, Feature/Target 계약, Train EDA, 보완 영향 기록 |

보완 표준 검증 DB는 `data/processed/standard/e020dafedc8b4fd9ac6696d0e4a92e/verified-37b56c39caf5435e8fb2a17f993ec4df.sqlite3`이며 SHA-256은 `15632b3b796ce255f4bc5038e6f680b214385beb47b9e21ebd1bb790b7962aee`이다. 모델 자료 DB SHA-256은 `c8383a9d3e8930830a137d1b6791cbb26e2316dbbbdf4f0f436ceda9a43890d0`이다.

파일 하나의 지문이 일치한다고 전체 자산이 일치하는 것은 아니다. 매니페스트, 내부 Fingerprint, 계약, 참조 파일까지 확인한다. 원본 객체와 실행 이력을 함께 보존한다. 재가공은 신규 실행 ID로 생성하며 기존 원본·표준·모델 자료를 수정하지 않는다. 원본 수집·가공·모델 자료 생성은 [README 실행 안내](../../README.md#running)를 따른다.

<a id="inputs"></a>

## 3. 입력·대표 추론

운영 예측기는 5개 기능군의 6개다. M5 위치 모델은 포함하지 않는다. 각 모델은 `models/<모델군>/<실행 ID>/model.json`, 대표 입출력은 `results/<모델군>/<실행 ID>/example_input.json` 및 `example_prediction.json`에 있다. 평가와 모델별 보조 파일도 동일 실행 ID로 연결한다. M3 타자·투수의 저장 폴더는 모두 `m3`이며 각각 별도 실행 ID를 사용한다.

| 모델군         | 운영 실행 ID                       | 상세 실행                                        |
| -------------- | ---------------------------------- | ------------------------------------------------ |
| M1 (`m1`)      | `0d9433b8044e48e5a4d21bbdb9b432e5` | [경기 결과·득점](game_outcome_score.md)          |
| M2 (`m2`)      | `158ce4d1ada94819bc3c355e281ba0e7` | [타석 결과](plate_appearance.md#commands)        |
| M3 타자 (`m3`) | `a1ac1c918d1a45ab9d9ee65ce1d34aa1` | [선수 미래 성적](player_performance.md#commands) |
| M3 투수 (`m3`) | `d6dc6240aabb4ad9853a51c44dfa5429` | [선수 미래 성적](player_performance.md#commands) |
| M4 (`m4`)      | `52df3aae7b45406790165d4c871614b7` | [경기 상황 가치](game_state_value.md#commands)   |
| M5 (`m5`)      | `c06c10d5d90c4d4ebefee2a848c92f26` | [다음 투구](pitch_outcome.md#commands)           |

대표 추론 명령은 [10절](#commands)에 있다. Exit 0과 기준 JSON의 구조 및 값 일치를 확인한다. 신규 실제 경기 예측 시에는 기존 대표 입력의 팀·날짜만 임의 치환하지 말고 M1의 예측 시점과 Feature 계약에 맞는 입력을 준비한다. M2/M4/M5의 `verify`, M3의 `inspect`, M5/M3의 공식 비교 및 M3 일반/합성 종료 입력은 각 기술 문서에서 확인한다. 모델마다 입력·출력 계약이 다르므로 M1 비교 방법을 모든 모델에 강제하지 않는다.

### 현재 저장 파일 지문과 역사 기록

아래 SHA-256은 [새 환경 재실행 검증 #40](https://github.com/Ahn-SooHyun/kbo-insight-ai/issues/40)의 확인 결과에서 인수한 현재 `model.json` 파일 바이트 지문이다. 내부 Fingerprint와 동일한 개념이 아니다.

| 모델    | 현재 SHA-256                                                       |
| ------- | ------------------------------------------------------------------ |
| M1      | `68c1ddeca8c54fed6359ec771f4026ee9802c73b49623c2d0076879da8f9da4a` |
| M2      | `7475253a1a5779186d68e0265ab102aeeb93294d5d9747ecb4a67ef086bc7026` |
| M3 타자 | `e1647d31e503e5b43e8b77f5538db6728ddcbc30c147dcb325cd243378367760` |
| M3 투수 | `b482bcebd8634e893661b985f80a72da6a965db9eef1282b8994311d85433c71` |
| M4      | `ac256bfacb9dbaa56e7fc3476fbb6541cd2b707b61e4c226e882ec4952f36a36` |
| M5      | `cdf8813d4bc1ad4dafb758687f6fe34e1b4ec15e92afc9f8cbfc91194d91b90e` |

M3 타자와 투수의 과거 기록 및 원래 직렬화로 재구성한 지문은 각각 `90c7d388133821c125797d819e7b28bc637c966b1975cd52509988257c3d355f`, `67e34294be8c7da0acaa0fa4ef93424c8f245faccbb7167eb84e900a2a6174d8`로 정확히 일치했다. 현재 파일 자체를 재직렬화하거나 수정하지 않는다.

M5의 현재 `model.json` SHA-256은 위 표의 `cdf8813d4bc1ad4dafb758687f6fe34e1b4ec15e92afc9f8cbfc91194d91b90e`다. 사용자 PowerShell `Get-FileHash` 결과와 고정 최종 평가 `results/final_evaluation/7d125b8a685a4693a34aea70445ef294/manifest.json`의 동일 실행 `model_file.sha256`이 일치했다. 내부 `model_fingerprint`는 별도 값 `0296b228ee144f2260a5aef48951d0d8736fcdd9e76ef7c9fb88450bf7bb30b7`이다.

개발 당시 [M5 자산 목록](pitch_outcome.md#reproducibility)의 파일 지문 `16b44b1e1814410e0b85f0045b8132ffa01f396130e7b61317d4c28b8ee13b66`은 당시 기록으로 보존한다. #40 보고의 별도 재구성 항목에 등장한 `50b5ef667d18d884347e61b9130e807d7f351b56ba8d875ab9bdf38301ab1e0d`는 이번 대조로 대상 파일과 재구성 절차가 확인되지 않았다. 차이의 원인도 미확인이므로 운영 자산의 확인값은 실제 파일과 평가 명세에서 일치한 값을 사용한다.

<a id="outputs"></a>

## 4. 시즌·보조 분석

[정규시즌 실행](regular_season_standings.md#commands)에서 위 모델 자료 ID와 M1 모델을 사용하고 새 실행 ID를 얻는다. [포스트시즌 실행](postseason_forecast.md#commands)의 `--parent-run`에는 해당 정규시즌의 실제 완료 실행 ID를 전달한다. 두 실행의 기준일·Seed·반복 수 및 자료 계보를 일치시킨다. 각각의 결과에 기록된 `manifest.json`, `summary.json`, `iterations.jsonl`, `scenario.json`을 확인한다. [보조 분석 실행](support_analysis.md#commands)은 별도의 새 모델 학습 없이 규칙·관측 기록·기존 M1을 재사용한다.

정규시즌 1위와 한국시리즈 우승은 별개 사건이다. 2025-09-01 KST, Seed 42, 1,000회 대표 사례는 사후 일정 가정 시나리오이며 공식 당시 정보 기반 Backtesting이 아니다. 공동 유효 분모는 정규시즌 1위 984회, 진출 922회, 순위·시드 확정 906회, 한국시리즈 우승 899회다. 팀별 분모와 공동 분모를 혼동하지 않는다. 한국시리즈 5차전 원천 기록 충돌로 종료 재생 시나리오에서는 1,000회 모두 우승 미결정이었다. 이는 공식 우승 재현 성공이 아니다. 2026년 시즌 전망·보조 분석은 현행 규칙에서 미지원이다.

<a id="method"></a>

## 5. 고정 결과와 판본

| 증거                                       | 고정 실행 경로                                                                                                                                                         | 기록 파일이 일치한 소스·문서 판본                                                                                   |
| ------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| [2026 최종 평가](final_evaluation.md)      | `results/final_evaluation/7d125b8a685a4693a34aea70445ef294/`의 `manifest.json`, `evaluation/result.json`                                                               | [`d9b3af9`](https://github.com/Ahn-SooHyun/kbo-insight-ai/commit/d9b3af9577a796d94ab7b98846e03273a400b365), 113/113 |
| [시즌 검산](season_forecast_validation.md) | `results/season_forecast_validation/015a0df87fc7416da307ad0565c0774f/`의 `plan.json`, `verification-6839d4c8050349edaf4f16c7d90d6625.json`                             | [`6a8d2b5`](https://github.com/Ahn-SooHyun/kbo-insight-ai/commit/6a8d2b5f4f138928a6daf61fd8fa66aa51672f0b), 106/106 |
| [예측 설명](prediction_explanation.md)     | `results/prediction_explanation/2803ffc86ba64ec0965a5dd698939e41/`의 `manifest.json`, 두 독립 실행 `result.json`, `verification-a3bed5f9467442c5a2278e5b1c7ea5fe.json` | [`8e914e7`](https://github.com/Ahn-SooHyun/kbo-insight-ai/commit/8e914e7f2ae2a62f4fcc5a4ab86a959e68f65e20), 126/126 |

최종 Test 기간은 2026-01-01 이상 2026-11-26 미만(KST)이다. 고정 모델의 정규시즌 평가이며 2026년 시즌 전망 지원을 의미하지 않는다. Train과 개발 Validation의 선택·보정에 최종 Test를 사용하지 않는다.

고정 명세는 당시 절대 경로, 코드·문서·환경·입력 및 결과 지문을 포함한다. 설명의 수용 이후 기술 문서 개정으로 해당 명세의 문서 지문과 병합본은 다르다. 과거 명세를 검증하려면 일치 판본과 기록된 환경·원래 경로를 별도로 확보하고 모든 참조를 검사한다. 절대 경로를 다른 사본 경로로 무단 치환하지 않는다. 과거 명세·결과·지문을 수정하거나 기존 결과 경로에 새 출력을 작성하지 않는다. 현재 판본에서 새 계산을 수행할 때에는 기술 문서의 `prepare` 절차로 새 명세와 새 결과 경로를 만들고 실제 출력 지문을 다음 단계로 전달한다. 새 명세 생성은 최종 Test를 재선정·재보정에 사용하는 권한을 부여하지 않는다.

<a id="lecture-methods"></a>

## 6. 강의 원리 및 프로젝트 확장

강의의 원리·시각은 [강의 전사 범위·시각·분류 기준](../planning/development_workflow.md#lecture-methods)를 기준으로 연결한다. 다음의 구현·실행 기록은 강의 설명의 직접 적용, 문제에 맞춘 응용, 제공 전사에서 동일한 상세 절차가 확인되지 않은 프로젝트 확장, 비교에만 사용한 후보를 구분해 읽는다.

| 원리                        | 적용 구분        | 실제 적용 및 근거                                                        | 해석 경계                                                    |
| --------------------------- | ---------------- | ------------------------------------------------------------------------ | ------------------------------------------------------------ |
| 공정 평가·일반화            | 원리 응용        | 저장 예측기와 정해진 입력을 고정하여 기준 출력을 동일 조건에서 비교한다. | 모델 성능을 신규 추정한 것이 아니다.                         |
| 기준 모델과 실험 비교       | 과제 방법론 응용 | 기존 운영 모델 및 기존 결과와 분리 사본의 결과를 대조한다.               | 재학습·새 후보 실험 없음.                                    |
| 실행 계보·SHA-256·독립 환경 | 프로젝트 확장    | 소스, 환경, DB, 모델, 결과 및 파일 변경 여부를 연결한다.                 | 지문은 바이트 동일성의 근거이지 사실 정확성의 보증이 아니다. |
| Bootstrap·OOB·Voting·튜닝   | 미적용           | 추론과 환경 재현에 불필요하다.                                           | 실제 신규 학습으로 집계하지 않는다.                          |

강의 출처 및 시각은 [개발 흐름의 강의 원리 기록](../planning/development_workflow.md#lecture-methods)을 따른다. 확인한 강의 발췌에서 같은 상세 재현 절차를 확인하지 못했으므로 프로젝트 확장으로 구분한다. 전체 강의에서의 부재는 단정하지 않는다.

### 6.1 동일한 조건으로 재현하는 원리

| 확인 항목   | 실제 시험                                                       | 동일성이 뜻하는 범위                        |
| ----------- | --------------------------------------------------------------- | ------------------------------------------- |
| 설치와 CLI  | 잠금 설치, 두 진입점, 명시 루트, 외부 작업 위치와 누락 루트     | 해당 Windows 사본에서의 설치·초기 설정 계약 |
| 입력과 판본 | 저장 모델 6개, 입력 JSON, 기준 출력 및 모델·평가 파일/내부 지문 | 인수 지정 자산과 비교 결과의 일치           |
| 계산        | 대표 추론 8건, 시즌 파일 4쌍, 저장 지표 6개, 설명 배경 3개      | 지정 입력과 지표를 독립 계산해 비교한 범위  |
| 저장 상태   | 보호 목록 458개 전후 비교 및 양쪽 기준 파일 11개 대조           | 명시한 원천·설정·DB·모델의 보존             |

강의의 공정 비교 원리는 동일 조건 설정에 활용했다. 잠금 설치, 자산의 파일 지문, 분리 사본과 불변 자산 보호는 확인한 강의 발췌에서 동일 세부 절차를 찾지 못한 프로젝트 확장이다. [공통 강의 적용 기록](../planning/development_workflow.md#lecture-validation-evidence)을 참조한다.

<a id="evaluation"></a>

## 7. 검증 증거와 완료 이력

| 작업                                                                                         | 병합·종료 근거                                                                        |
| -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| [예상 선발·팀 전력·일정 난이도 #29](https://github.com/Ahn-SooHyun/kbo-insight-ai/issues/29) | [PR #33](https://github.com/Ahn-SooHyun/kbo-insight-ai/pull/33), `53c2d41`, 완료 종료 |
| [2026 최종 평가 #34](https://github.com/Ahn-SooHyun/kbo-insight-ai/issues/34)                | [PR #37](https://github.com/Ahn-SooHyun/kbo-insight-ai/pull/37), `11f996b`, 완료 종료 |
| [시즌 전망·보조 분석 종합 검증 #35](https://github.com/Ahn-SooHyun/kbo-insight-ai/issues/35) | [PR #38](https://github.com/Ahn-SooHyun/kbo-insight-ai/pull/38), `c5ff798`, 완료 종료 |
| [예측 설명 #36](https://github.com/Ahn-SooHyun/kbo-insight-ai/issues/36)                     | [PR #39](https://github.com/Ahn-SooHyun/kbo-insight-ai/pull/39), `ea4ef45`, 완료 종료 |
| [새 환경 재실행 검증 #40](https://github.com/Ahn-SooHyun/kbo-insight-ai/issues/40)           | 저장소 변경 없음, 기술 수용, 2026-11-29 01:08:59 KST 완료 종료                        |

위 병합과 완료 종료는 링크의 원격 상태에서 확인했다. 아래 실행·회귀 수치는 #40의 사용자 실행 보고와 수용 기록에서 인수했다.

#40은 동일 Windows 장비의 분리 작업 사본과 신규 가상환경에서 다음을 확인했다.

- 잠금 설치, 두 진입점의 도움말과 `check`, 외부 작업 위치 및 명시 루트 성공; 없는 루트 Exit 2.
- 저장 원본 보완 표준 실행 `661bbee965724a23a751a55b609be594`, 검증 `a9df691c47844a2f97bbcc01147a228f`에서 기존 보완 DB와 동일 SHA-256.
- 운영 모델 6개 검사, 독립 추론 8건 및 기준 비교 8건 성공.
- 연결된 신규 정규시즌 `c23c3ec60e8b4b17b4c1dc2a1a124302`와 포스트시즌 `35fe4eb5c9e944679c1ae422eb459a30`; 기존 및 신규 계산 파일 4쌍 동일.
- 고정 참조 121개, 저장 지표 6개 및 Train 설명 배경 3개 검증·재계산. 신규 전체 회귀 734개 통과; 지정 보호 파일 458개 변경·추가·삭제 0건.

과거 명세 판본의 별도 체크아웃에서 전체 CLI 실행, 현 판본의 새 명세 및 전체 CLI, 최종 Test 전량 재추론, 원본 재수집·모델 재학습, 새 장비·Linux·다른 보안 정책·오프라인 설치는 #40에서 미실행 또는 미검증이다. 새 가공·시즌 결과는 분리 사본, 증거 로그는 저장소 밖에 보존했다. 이 기록은 사용자 실행 및 선행 수용 근거이며 다른 환경에서 동일 동작을 보장하지 않는다.

<a id="reproducibility"></a>

## 8. 재현·오류 대응

| 증상                           | 확인                                                                                                                                       |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------ |
| 설정 확인 Exit 2               | 실제 절대 루트, `KBO_PROJECT_ROOT`, UTF-8 `kbo.toml`, 경로 겹침 및 필수 설정                                                               |
| `check` 성공 후 예측 실패      | 모델·입력 JSON의 존재, 접근 권한, 해당 모델 계약, 고정 자산 지문                                                                           |
| 자료 가공 오류                 | 원본과 실행 이력, 파일 권한·디스크 공간, 수집/표준 실행 ID 및 비교 대상. 원본을 수정하지 않는다.                                           |
| uv 하드링크 경고               | 파일 복사 전환 후 설치 Exit 0 및 `uv pip check` 확인. 경고만으로 성공 또는 실패를 단정하지 않는다.                                         |
| PowerShell 버전 조회 인용 오류 | 패키지 이름을 Python `sys.argv`로 전달한다.                                                                                                |
| 테스트의 `NativeCommandError`  | stdout/stderr와 실제 프로세스 종료 코드를 구분한다. 필요 시 독립 프로세스에서 두 로그를 분리하고 `Ran ... tests`, `OK`, Exit 0을 확인한다. |
| 명세 지문 불일치               | 해당 기록 판본 및 환경과 원래 경로를 확인하거나 새 명세를 생성한다. 기존 지문은 치환하지 않는다.                                           |
| 시즌 미결정·미지원             | 유효 분모, 입력·규칙·원천 충돌 사유를 확인한다. 미결정·미지원 값을 0으로 치환하지 않는다.                                                  |

전체 회귀는 프로젝트 루트에서 `python -X utf8 -B -m unittest discover -s tests`를 같은 가상환경의 Python으로 실행한다. 상세 설치·테스트는 [README](../../README.md#commands), 결과 파일 읽는 법은 [결과 안내](../../README.md#outputs), 문제별 확인 위치는 [오류 대응](../../README.md#help)을 따른다.

<a id="limitations"></a>

## 9. 한계·인수 확인

인수자는 먼저 `check`의 루트 및 경로, Python 실행 위치, 모델 ID, 입력 계약, 저장 기준 출력, 실제 종료 코드와 결과 상태를 확인한다. 자산 미제공이면 임의 생성·재수집으로 보완하지 않는다. 새 장비·Linux·다른 보안 정책·오프라인 설치, 과거 판본의 전체 CLI 및 현 판본의 신규 전체 평가 실행은 별도 검증 대상이다. 기존 기록을 완료로 소급 변경하지 않는다.

[README](../../README.md) · [공통 계약](project_contract.md) · [개발 흐름](../planning/development_workflow.md)

<a id="commands"></a>

## 10. 실행 명령

### 10.1 기본 확인 (Windows PowerShell)

아래 예시의 루트를 실제 준비한 소스 및 자산 위치로 바꾼다. 설치가 끝난 같은 가상환경을 사용한다. 모든 정상 명령은 Exit 0이어야 한다.

```powershell
$root = 'D:/Programming/kbo-insight-ai'
Set-Location -LiteralPath $root
$py = Join-Path $root '.venv/Scripts/python.exe'
$cli = Join-Path $root '.venv/Scripts/kbo-insight.exe'
uv pip check --python $py
if ($LASTEXITCODE -ne 0) { throw '의존성 검사 실패' }
& $py -X utf8 -B -m src --help
if ($LASTEXITCODE -ne 0) { throw '모듈 도움말 실패' }
& $py -X utf8 -B -m src --project-root $root check
if ($LASTEXITCODE -ne 0) { throw '설정 확인 실패' }
& $cli --project-root $root check
if ($LASTEXITCODE -ne 0) { throw '설치 CLI 확인 실패' }
```

설치 EXE가 조직의 Application Control 정책으로 차단되는 환경에서는 허용된 동일 가상환경의 `python -m src` 진입점을 사용한다. 보안 정책을 해제하지 않는다. Python 실행 자체가 차단되면 관리자 또는 해당 정책 담당자에게 문의한다.

### 10.2 저장 M1 대표 예측

앞 절의 `$root`, `$py`를 사용한다. 이 명령은 저장 모델과 입력을 읽어 stdout으로 예측 JSON을 반환한다. 저장된 기준 출력은 읽기 전용으로 비교하며 덮어쓰지 않는다.

```powershell
$id = '0d9433b8044e48e5a4d21bbdb9b432e5'
$asset = Join-Path $root "models/m1/$id/model.json"
$inputPath = Join-Path $root "results/m1/$id/example_input.json"
$expected = Join-Path $root "results/m1/$id/example_prediction.json"
foreach ($path in @($asset, $inputPath, $expected)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "필수 파일 누락: $path" }
}
$prediction = & $py -X utf8 -B -m src.models.game_outcome --project-root $root predict --asset $asset --input $inputPath
if ($LASTEXITCODE -ne 0) { throw 'M1 예측 실패' }
($prediction -join "`n") | & $py -X utf8 -B -c 'import json, sys; from pathlib import Path; expected = json.loads(Path(sys.argv[1]).read_bytes()); sys.exit(0 if json.load(sys.stdin) == expected else 1)' $expected
if ($LASTEXITCODE -ne 0) { throw '기준 출력 비교 실패' }
($prediction -join "`n") | ConvertFrom-Json
```

대표 예측의 정상 종료값은 0이다. 비교 프로세스의 종료값 1은 기준 JSON과 구조 또는 값이 다름을 뜻한다. 운영 시험 시에는 [README의 회귀·Git 검사](../../README.md#commands) 순서를 따르고 실제 종료값을 기록한다.
