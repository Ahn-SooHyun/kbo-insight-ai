# KBO Insight AI

**KBO Insight AI · 프로젝트 진입점·사용자 실행 안내**

| 항목           | 내용                                                                                                       |
| -------------- | ---------------------------------------------------------------------------------------------------------- |
| 문서 역할      | 프로젝트 진입점·사용자 실행 안내                                                                           |
| 적용 범위      | 기능, 환경, 자료, 실행, 결과 해석 및 인수                                                                  |
| 확인 기준일    | 2026-11-29 (KST)                                                                                           |
| 저장소 경로    | `README.md`                                                                                                |
| 인수 소스 기준 | [`ea4ef45`](https://github.com/Ahn-SooHyun/kbo-insight-ai/commit/ea4ef457fa35dbc37525cd97586a9f51543e31c5) |
| 연관 작업      | [#41](https://github.com/Ahn-SooHyun/kbo-insight-ai/issues/41)                                             |
| 자료 이용      | 승인된 비영리 대학 과제용 로컬 처리 및 내부 평가                                                           |

**판본 보존:** [개정 전 기준 문서 원문](https://github.com/Ahn-SooHyun/kbo-insight-ai/blob/ea4ef457fa35dbc37525cd97586a9f51543e31c5/README.md)은 역사적 실험·명세·검증의 원본 근거다. 개정 문서는 이전 실행 명세의 파일 지문을 대체하지 않는다.

**경기에서 시즌까지, 예측 시점·정답의 품질·평가 근거와 재현성을 함께 관리하는 KBO 분석 프로젝트**

[저장소](https://github.com/Ahn-SooHyun/kbo-insight-ai) · [개발 흐름](docs/planning/development_workflow.md) · [공통 예측·평가 계약](docs/standards/project_contract.md) · [실행 및 저장 자산 인수](docs/standards/execution_handover.md)

> 현재 운영 범위: M1~M5의 저장 예측기 6개를 인수한다. 2026년 고정 모델 최종 Test는 완료 기록이 있지만, 2026년 시즌 전망·보조 분석은 지원하지 않는다. 2025년 시즌 전망은 사후 일정 가정에 따른 시나리오다. 같은 Windows 장비의 분리 사본과 신규 가상환경에서 대표 실행 및 734개 회귀를 확인했다. 새 자료와 모델은 Git 소스만으로 제공되지 않는다.

## 찾아보기

- [1. 프로젝트 개요·기능 지도](#features)
- [2. 구조·문서 역할](#documents)
- [3. 환경·설정·설치 준비](#running)
- [4. 자료·운영 모델 준비](#data)
- [5. 예측·시즌·보조 분석의 실행과 결과](#outputs)
- [6. 강의 원리 및 프로젝트 확장](#lecture-methods)
- [7. 평가·재현 검증 상태](#evaluation)
- [8. 오류 대응](#help)
- [9. 문서 유지·제한](#maintenance)
- [10. 설치·기본 확인·대표 예측·테스트 명령](#commands)

<a id="features"></a>

## 1. 프로젝트 개요·기능 지도

### 1.1 문제 정의 및 예측 단위

KBO 공개 자료를 자동 확보하고, 예측 시점 이전에 관측 가능한 자료로 경기·타석·선수·경기 상태·다음 투구의 결과를 계산한다. 개별 경기 결과를 정규시즌·포스트시즌 전망에 연결한다. 데이터와 모델의 지원 한계, 원천 자료의 누락 및 미래 정보 사용 가능성을 예측값과 분리해 관리한다.

| 기능              | 예측 단위·기준 시점                       | 결과                                                            | 실행 계약                                                                                                                                            |
| ----------------- | ----------------------------------------- | --------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| M1 경기 결과·득점 | 경기 시작 전                              | 양 팀 득점 분포, 기대득점, 대표 스코어, 홈 승·무·패             | [경기 결과·득점](docs/standards/game_outcome_score.md)                                                                                               |
| M2 타석 결과      | 타석 시작 전                              | 해당 타석의 다중 범주 확률                                      | [타석 결과](docs/standards/plate_appearance.md)                                                                                                      |
| M3 선수 미래 성적 | 월요일 시작 전                            | 7일·28일·잔여 정규시즌 Count, 파생 비율 및 구간; 타자·투수 별도 | [선수 미래 성적](docs/standards/player_performance.md)                                                                                               |
| M4 경기 상황 가치 | 동일 공격팀 관점의 타석 전·후             | 승리 확률, 반이닝 RE, WPA, 득점 가치, LI                        | [경기 상황 가치](docs/standards/game_state_value.md)                                                                                                 |
| M5 다음 투구      | 실제 다음 공 투구 전                      | 구종·결과 공동분포와 각 주변분포                                | [다음 투구](docs/standards/pitch_outcome.md)                                                                                                         |
| 정규시즌 전망     | 기준일 KST 자정                           | 최종 승수·순위, 정규시즌 1위, 진출 확률과 반복별 시드           | [정규시즌](docs/standards/regular_season_standings.md)                                                                                               |
| 한국시리즈 전망   | 같은 부모 반복의 시드 또는 종료 기록 가정 | 시리즈 참가·승리, 한국시리즈 진출·우승 및 조건부 확률           | [포스트시즌](docs/standards/postseason_forecast.md)                                                                                                  |
| 보조 분석         | 2025년 지원 기준일                        | 예상 선발 후보, 관측 팀 기록, M1 확률 기반 일정 난이도          | [예상 선발·팀 전력·일정 난이도](docs/standards/support_analysis.md)                                                                                  |
| 종합 검증·설명    | 고정 모델과 결과                          | 최종 평가, 누수 감사, 확률 진단, 모델별 설명 및 실패 사례       | [평가](docs/standards/final_evaluation.md) · [검산](docs/standards/season_forecast_validation.md) · [설명](docs/standards/prediction_explanation.md) |

### 1.2 운영 예측기 수와 계산 기능

M1·M2·M4·M5는 각 1개, M3는 타자·투수 각 1개로 총 6개 주 예측기를 운영한다. M1의 결합 득점 분포에서 승무패를 계산하므로 별도 승무패 모델을 세지 않는다. M5는 하나의 공동 확률을 구종·결과별로 주변화하며 별도 구종 및 결과 모델을 만들지 않는다. M5 위치는 제외했다. 시즌·선발·팀 기록·일정 난이도는 저장 모델·규칙·통계의 재사용이므로 신규 주 예측기 0개다.

### 1.3 비지원 및 제외 사항

예상 선발 후보는 공식 선발 발표·등록·건강 상태·감독 판단을 뜻하지 않는다. 라인업, 교체, 도루·견제사·번트 세부 기능은 자동 식별 및 별도 품질 검증을 통과한 범위만 지원한다. 숨은 감독 의도, 실제 몸 상태, 비공개 부상, 내부 교체 사유, 분모 없는 견제 확률, 수상 예측, 투구 궤적·타구 위치·군집화, LLM 결합·예측 웹/앱/API 배포는 현재 범위에서 제외한다. [지원 범위와 조건부 기능](docs/standards/project_contract.md#scope)을 따른다.

<a id="documents"></a>

## 2. 구조·문서 역할

### 2.1 소스와 로컬 생성물

| 위치                                                                                        | 책임                                           |
| ------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| `.python-version`, `pyproject.toml`, `uv.lock`                                              | Python 요청 버전, 빌드·패키지·의존성 잠금      |
| `kbo.toml`, `src/settings.py`                                                               | 경로, 설정 검증, 루트 선택 및 오류 경계        |
| `src/__main__.py`                                                                           | 환경·수집·원본 검사·표준화 CLI                 |
| `src/ingestion/`                                                                            | 제한 수집·전체 파일 수집·재개와 불변 원본 관리 |
| `src/datasets/`                                                                             | 표준화, 보완, 모델 자료 생성 및 계보           |
| `src/models/`                                                                               | 모델군별 학습·저장·추론·검증                   |
| `src/evaluation/`                                                                           | 평가, 고정 자산 검사·최종 평가                 |
| `src/simulation/`                                                                           | 정규·포스트시즌 전망 및 보조 분석              |
| `src/validation/`                                                                           | 원본·표준·시즌 등 검증                         |
| `src/explainability/`                                                                       | 예측 설명·확률 신뢰도·실패 진단                |
| `tests/`                                                                                    | 기능별 단위·통합·회귀 검사                     |
| `docs/planning/`, `docs/standards/`                                                         | 개발 절차 및 공통·기능별 기술 계약             |
| `.venv/`, `data/raw/`, `data/interim/`, `data/processed/`, `models/`, `results/`, `.cache/` | 로컬 설치·자료·모델·실행 결과; Git 소스와 분리 |

정확한 추적 파일 목록은 기준 Commit의 Git tree에서 확인한다. 예시 폴더 구조를 미래 구현이나 모든 생성 파일의 존재로 간주하지 않는다.

### 2.2 문서별 책임과 우선순위

| 문서                                                                                                                                                              | 책임                                                    |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| README                                                                                                                                                            | 설치·대표 추론·결과 찾기의 첫 진입점                    |
| [개발 흐름](docs/planning/development_workflow.md)                                                                                                                | 선행 관계, 단계별 완료·인수·강의 적용 기준              |
| [공통 예측·평가 계약](docs/standards/project_contract.md)                                                                                                         | 시점, 자료·정답, 모델 수, 지표, 시나리오와 재현 규범    |
| [실행 및 저장 자산 인수](docs/standards/execution_handover.md)                                                                                                    | 실제 운영 자산 ID·지문·대표 입력·역사 판본·검증 범위    |
| 기능별 기술 문서                                                                                                                                                  | 수식, 실제 구현, 후보, 실험 결과, 실패 사례와 실행 상세 |
| [최종 평가](docs/standards/final_evaluation.md), [시즌 검산](docs/standards/season_forecast_validation.md), [예측 설명](docs/standards/prediction_explanation.md) | 고정 결과, 독립 검증, 방법별 지원 범위와 새 실행 조건   |

공통 계약이 기능별 더 좁은 실제 지원 범위를 확대하지 않는다. 명령·결과는 코드와 해당 실행 명세를 확인한다. 현재 문서가 과거 실행 명세의 지문을 대체하지 않는다.

<a id="running"></a>

## 3. 환경·설정·설치 준비

### 3.1 확인 환경

| 항목        | 확인 값                                    |
| ----------- | ------------------------------------------ |
| 운영체제    | Windows 11 AMD64, 동일 장비 분리 작업 사본 |
| Python      | CPython 3.12.15 (`>=3.12,<3.13`)           |
| 환경 관리   | uv 0.13.7, `.venv`, 편집 가능 설치         |
| 빌드 도구   | `uv_build==0.13.7`                         |
| 패키지      | `kbo-insight-ai` 0.1.0, import `src`       |
| 기본 의존성 | 표준 라이브러리                            |
| 선택 의존성 | `inspection`: PyArrow 26.0.1               |
| SQLite      | 3.53.1 (확인한 Python 환경)                |
| 진입점      | `kbo-insight` / `python -m src`            |

다른 OS·장비·패치 버전·보안 정책·오프라인 설치는 별도 검증 대상이다. 잠금 설치는 패키지를 확보하기 위해 네트워크를 사용할 수 있다. 기존 `.venv`를 무단 덮어쓰지 않는다.

### 3.2 프로젝트 루트 및 설정

루트 선택 우선순위는 명시 `--project-root` → 프로세스 `KBO_PROJECT_ROOT` → 편집 가능 설치의 소스 위치다. 루트는 실제 존재하는 절대 디렉터리이며 설정 파일 `kbo.toml`이 있어야 한다. 상위 우선순위가 잘못되면 낮은 우선순위로 자동 대체하지 않는다. 현재 CWD는 자료 경로를 결정하지 않는다.

| `kbo.toml` 필수 설정          | 제공 값                           | 의미·제약                         |
| ----------------------------- | --------------------------------- | --------------------------------- |
| `raw_dir`                     | `data/raw`                        | 불변 원본                         |
| `interim_dir`                 | `data/interim`                    | 중간 변환                         |
| `processed_dir`               | `data/processed`                  | 표준·모델 자료                    |
| `models_dir`                  | `models`                          | 모델 자산                         |
| `results_dir`                 | `results`                         | 실행 결과                         |
| `cache_dir`                   | `.cache`                          | 로컬 캐시                         |
| `seed`                        | `42`                              | 정수, 0~4294967295, Boolean 제외  |
| `log_level`                   | `INFO`                            | DEBUG/INFO/WARNING/ERROR/CRITICAL |
| `collection_base_url`         | `https://api-gw.sports.naver.com` | 승인된 공개 원천                  |
| `collection_timeout_seconds`  | `30`                              | 1~30초                            |
| `collection_retries`          | `0`                               | 최초 요청 뒤 0~2회                |
| `collection_max_bytes`        | `1048576`                         | Naver 저장 상한 1 MiB             |
| `hf_collection_base_url`      | `https://huggingface.co`          | 고정 HF 배포 파일 원천            |
| `hf_collection_max_bytes`     | `20971520`                        | HF 상한 20 MiB                    |
| `hf_collection_max_redirects` | `3`                               | 최대 3회 전송                     |

모든 필수 키가 실제 파일에 있어야 하며 표의 값은 누락 시 대체 기본값이 아니다. `kbo.toml`은 UTF-8이다. 상대 경로는 루트에 붙이고 절대 경로는 그대로 사용한다. `~` 및 설정 문자열 안의 환경 변수를 자동 확장하지 않는다. Windows 드라이브 상대경로 또는 드라이브 없는 루트 경로를 거절한다. 경로·상위 경로가 파일인 경우, 산출물 경로 상호 중첩, 프로젝트 루트나 상위 디렉터리, `.git`·`.github`·`.venv`·`src`·`tests`·`docs`와 겹침을 거절한다.

`check`는 설정을 읽고 해석한 경로를 확인하며 디렉터리·DB·모델을 생성하지 않는다. 실제 쓰기 권한, 디스크 공간, 데이터 품질, 예측 성공을 대신 검사하지 않는다. 개별 자료 경로는 `kbo.toml`에서 지정하고 환경 변수는 루트 선택에만 사용한다.

### 3.3 Secret과 설치 진입점

API Secret, 토큰, 비밀번호, DB 접속 비밀을 소스·설정·Notebook·로그에 적지 않는다. 실제 비밀값 파일은 Git에서 제외한다. 설치된 EXE가 조직 보안 정책으로 차단되면 허용된 동일 가상환경의 Python 모듈 경로를 사용할 수 있다. 보안 정책을 해제하지 않는다.

<a id="data"></a>

## 4. 자료·운영 모델 준비

### 4.1 사용 목적별 요구 입력

| 목적                  | 반드시 확인할 입력                                          | 새 출력 또는 확인 대상                         |
| --------------------- | ----------------------------------------------------------- | ---------------------------------------------- |
| 환경 확인             | 소스, 설정, 잠금 설치                                       | `check` JSON, 정상 Exit 0                      |
| 저장 예측기 단독 예측 | 해당 `model.json`, 계약을 만족하는 입력 JSON                | 예측 JSON; 학습 없음                           |
| 대표 예측 재현        | 위 입력 + 기존 기준 출력 및 필요한 모델별 보조 명세         | 기존 기준과 구조·값 비교                       |
| 원본 가공             | 승인된 원본 객체·실행 이력·원천 파일 및 선택된 기준 ID      | 새 표준·검증 실행                              |
| 모델 자료 구축        | 검증된 표준 DB 및 검증 ID                                   | 새 모델 자료 DB, manifest, feature/target 계약 |
| 시즌·보조 분석        | 저장 M1, 모델 자료, 해당 시나리오 및 필요한 부모 실행       | 새 결과 ID, summary·반복·계보                  |
| 종합 평가·설명        | 고정 명세가 요구하는 DB, 모든 모델·기준·예측·정답·결과 파일 | 기존 증거 검증 또는 새 명세·실행               |

Git 저장소만 받아서는 승인된 DB·모델·대표 입력·기준 출력이 준비되지 않는다. 기존 보유 자산의 합법적 로컬 사용 범위를 확인해 [실행 및 저장 자산 인수](docs/standards/execution_handover.md#data)에서 상대 경로와 실행 ID·지문을 찾는다. 원본·모델 본문·비공개 자료 행 및 비밀값을 Git이나 공개 문서에 포함하지 않는다.

### 4.2 자동 수집과 원본 보존

표본 수집은 명시적인 요청에서만 수행한다. `collect-sample --game-id ... --inning ...`은 Naver 표본, `collect-sample --source hf`는 고정 HF 배포 파일에 해당한다. HF에 경기 ID·이닝 인자를 사용하지 않는다. `inspect-sample --run-id`는 이미 저장된 HF 원본을 PyArrow로 읽어 필드, 타입, null 및 원본 연결을 검사한다. 표본 구조 검사를 전체 데이터 품질 판정으로 해석하지 않는다.

전체 파일의 `collect-files --manifest`와 `update-files --manifest`는 승인 명세를 사용하며 `resume-files --run-id`는 미확보 대상을 이어 수행한다. `report-files --run-id`는 기존 수집 이력과 객체를 읽기 전용으로 대조한다. 승인 명세는 실제 경로를 확보하고 내용·대상 기간·판본·파일명·지문을 확인한다. 임의 명세나 직접 행 입력으로 원본을 만들지 않는다. 수집 성공 후에도 구조·의미·시간·키 검증을 별도로 수행한다.

원본은 `data/raw/published/objects/`, `data/raw/published/runs/`, `data/raw/sources/`에 보존한다. 채택 판본·실행 ID·수집 요청·출처·해시·실패·보류 이력을 유지한다. 보완 원본은 별도 판본으로 연결하고 이전 raw 객체를 덮어쓰지 않는다. 원본 46개, 논리 대상 48개, 판본 50개는 보완 인수판의 집계이며 새로운 원천 전체의 보장 수량이 아니다.

### 4.3 표준화·검증 및 모델 자료

`standardize`는 저장 원본을 중간 자료와 표준 DB로 변환한다. `--as-of-run`으로 이전 수집 판본, `--compare-run`으로 이전 표준 실행을 지정할 수 있다. `--base-run`과 `--base-verification`은 포스트시즌 보완에 함께 지정한다. `validate-standard --run-id`로 실제 행·키·집계·품질을 검사하며 `--compare-with <표준ID:검증ID>`로 동일 입력의 비교를 수행한다.

- 중간: `data/interim/standard/<run_id>/`.
- 표준: `data/processed/standard/<run_id>/standard.sqlite3`, `quality.json`.
- 검증판: 같은 표준 폴더의 `verified-<verification_id>.sqlite3`.
- 모델 자료: `data/processed/model_data/<run_id>/model_data.sqlite3`와 `manifest.json`, 입력·정답 계약, Train EDA 및 보완 영향 자료.

현재 인수 표준은 `e020dafedc8b4fd9ac6696d0e4a92e`, 검증 `37b56c39caf5435e8fb2a17f993ec4df`, 모델 자료 `259b38decaac4ed68dd5a596cfbe4849`이다. 수집 실행 `4bdbd03e20634ce58135ec2311555351` 및 기초 표준·검증 ID는 [자산 계보](docs/standards/execution_handover.md#data)를 따른다. #40의 새 보완 표준 DB는 기존 판본과 동일한 물리 SHA-256을 확인했다. 새 모델 자료 DB를 생성한 검증은 아니다.

모델 자료 생성기는 검증 ID를 받아 Train/Validation/Test와 입력·정답·제외를 생성한다. 생성 명령은 학습·전처리 적합·성능 평가가 아니다. 이름 식별, 당일·미래 행, 미관측과 0, M3 잔여 기간 완료 근거를 분리한다. Train에 학습형 전처리를 적합하고 이후 구간에 그대로 적용한다.

### 4.4 운영 모델·고정 결과의 찾는 법

모델 자산은 `models/<모델군>/<run_id>/model.json`, 대표 입력·기준 출력·개발 평가는 `results/<모델군>/<run_id>/`에서 찾는다. 모델군의 실제 저장 실행 ID와 바이트 SHA-256은 [운영 모델 인수](docs/standards/execution_handover.md#inputs)에 모아 두었다. `M3`는 같은 `m3` 폴더의 서로 다른 ID를 사용한다. M3 과거 지문 재구성과 M5 개발 지문 불일치는 별도 이력으로 보존한다.

<a id="outputs"></a>

## 5. 예측·시즌·보조 분석의 실행과 결과

### 5.1 결과를 읽는 순서

1. 출력의 실행 ID·모델 버전·자료판·예측 기준시각·지원 여부를 확인한다.
2. 단독 예측의 확률 합, 결과 단위 및 계약을 확인한다. 기대득점과 대표 스코어, 사건 확률의 최빈 범주는 일치할 필요가 없다.
3. 지표의 기준·운영 모델, 모집단·유효 표본·제외 사유·평가 기간을 함께 확인한다.
4. 결과가 미결정·미지원·null인 경우 입력·규칙·정답의 근거를 따라간다. 실제 0과 결측을 구분한다.
5. 실행 ID가 다른 경우 메타데이터와 실제 계산 출력의 동일성을 나누어 검사한다.

<a id="pitch-outcome-status"></a>

### 5.2 다음 투구 M5

v1.1.0 `logistic_temperature` (T=1.15) 단일 운영 모델. 공동분포의 두 축 주변합을 결과로 제공한다. 별도 위치 모델을 사용하지 않는다. 개발·최종 Test의 Log loss, Brier, 클래스별 지원 및 ECE는 [M5 실행 계약](docs/standards/pitch_outcome.md)과 [최종 평가](docs/standards/final_evaluation.md#evaluation)를 확인한다. 현재 운영 파일은 고정 최종 평가 명세와 일치하지만 개발 당시 문서의 바이트 지문과 다른 원인은 미확인이다.

<a id="player-performance-status"></a>

### 5.3 선수 미래 성적 M3

타자 `history_balanced_50`, 투수 `joint_residual_tree_depth_4_shrink_50`의 두 운영 자산. 7일·28일·잔여 기간의 Count·비율 및 지원 상태를 구별한다. 2026 Test에서 투수의 정규화 MAE는 악화됐으며 RMSE는 개선됐다. 비율은 유효 분모로만 계산하고 잔여 정답은 시즌 완료 근거가 필요하다. [M3 실행 계약](docs/standards/player_performance.md).

<a id="regular-season-status"></a>

### 5.4 정규시즌 1위·진출

[정규시즌 실행](docs/standards/regular_season_standings.md#commands)은 모델 자료와 저장 M1으로 새 시뮬레이션 결과를 생성한다. 각 날짜의 완료 경기와 잔여 일정 가정을 구분하고 같은 날짜의 예측 상태를 일괄 갱신한다. `manifest.json`, `scenario.json`, `summary.json`, `iterations.jsonl`에서 자료·규칙·반복·미결정을 확인한다. 1위와 진출의 팀별 분모 및 공동 분모를 분리한다.

<a id="postseason-status"></a>

### 5.5 한국시리즈 우승

[포스트시즌 실행](docs/standards/postseason_forecast.md#commands)은 정규시즌 같은 반복의 확정 시드를 인수한다. 별도 우승 확률을 계산하며 미결정 반복을 임의 제거하거나 모든 팀에 0으로 배분하지 않는다. 2025-09-01 KST, Seed 42, 1,000회 사후 가정의 공동 유효 분모는 정규시즌 1위 984, 진출 922, 순위·시드 906, 한국시리즈 우승 899회다. 2025-11-01 종료 재생에서 한국시리즈 5차전 원천 충돌로 우승 1,000회가 미결정이다. 이는 공식 우승팀 부재나 우승 확률 0이 아니다. 2026년 시즌 전망은 미지원이다.

<a id="support-analysis-status"></a>

### 5.6 예상 선발·팀 전력·일정 난이도

v1.1.0 기준 순위 규칙, 관측 경기 기술통계, 기존 M1을 사용한다. 2025년 경기 당일 개발 비교의 공통 1,429팀 측에서 기준 Top-1 18.82%, Top-3 73.90%가 제안 규칙보다 우세했다. 이는 당시 공개 정보가 입증된 정식 평가가 아니며 7일 전 전망의 성능을 보장하지 않는다. 2026년 보조 분석은 미지원이다. [보조 분석 실행](docs/standards/support_analysis.md#commands)과 [시즌 종합 검산](docs/standards/season_forecast_validation.md).

<a id="lecture-methods"></a>

## 6. 강의 원리 및 프로젝트 확장

| 구분                     | 실제 적용                                                                                                                  | 근거 문서                                                                    |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| 강의 직접 적용·원리 활용 | 공정 평가·기준 비교, 손실 함수, 로지스틱 분류, 제곱오차 기반 트리 분기 원리 등                                             | [공통 강의 적용 기록](docs/planning/development_workflow.md#lecture-methods) |
| 과제 방법론              | 자료 확보·문제 정의·EDA·Baseline·특성·비교·최적화·결론                                                                     | [개발 흐름](docs/planning/development_workflow.md)                           |
| 프로젝트 확장            | 시간순·경기 묶음·누수 방어, 공동 확률, 평활·보정, 다중 기간 제약, Monte Carlo, 조건부 사건, SHAP·단계별 설명, SHA-256 계보 | [공통 계약](docs/standards/project_contract.md), 기능별 기술 계약            |
| 비교 후 미채택           | M1 포아송 후보, M3 후보 트리·통계, M4 이력·강한 L2 후보, M5 L2·평균 후보, 보조 제안 규칙 등                                | 각 모델 개발 문서의 실제 실험                                                |
| 미실험                   | KNN·SVM·RF·Bootstrap·OOB·Voting·Boosting·Stacking 등에서 실제 실행 기록이 없는 항목                                        | 공통·기능별 미적용 기록                                                      |

강의 전사에서 확인된 구간을 근거로 적용·원리 응용을 구별한다. 동일 상세 방법을 확인한 강의 발췌에서 찾지 못한 항목은 프로젝트 확장으로 별도 기록하며 전체 수업에서 언급되지 않았다고 단정하지 않는다. 각 기술 문서는 실제 코드·수식·검증·한계를 연결한다. 모델 구현에 사용하지 않은 기법을 사용했다고 기록하지 않는다.

<a id="evaluation"></a>

## 7. 평가·재현 검증 상태

### 7.1 고정 모델 최종 Test와 개발 평가

Train은 2025-01-01 이전, Validation은 2025년, 최종 Test는 2026-01-01 이상 2026-11-26 미만(KST)이다. 실제 적격 표본은 모델별로 다르다. 모델 선정·보정은 개발 자료에서만 수행한다. 포스트시즌 및 시즌 전망의 공식 당시 정보 기반 성능과 동일하지 않다.

| 작업                        | 구현·검증 및 종료 상태                                                                                                                                     |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| M1~M5                       | 각각 구현·개발 검증·병합 완료 기록; 현재 주 예측기 6개                                                                                                     |
| 정규시즌 #27·포스트시즌 #28 | [PR #30](https://github.com/Ahn-SooHyun/kbo-insight-ai/pull/30), [PR #31](https://github.com/Ahn-SooHyun/kbo-insight-ai/pull/31) 병합 기록; 2025 가정 지원 |
| 예상 선발·팀·일정 #29       | [PR #33](https://github.com/Ahn-SooHyun/kbo-insight-ai/pull/33), `53c2d41`, 완료 종료                                                                      |
| 고정 최종 평가 #34          | [PR #37](https://github.com/Ahn-SooHyun/kbo-insight-ai/pull/37), `11f996b`, 완료 종료                                                                      |
| 시즌 검산 #35               | [PR #38](https://github.com/Ahn-SooHyun/kbo-insight-ai/pull/38), `c5ff798`, 완료 종료                                                                      |
| 예측 설명 #36               | [PR #39](https://github.com/Ahn-SooHyun/kbo-insight-ai/pull/39), `ea4ef45`, 완료 종료                                                                      |
| 분리 환경 검증 #40          | 저장소 변경 없음, 2026-11-29 01:08:59 KST 완료 종료                                                                                                        |

실제 지표는 [최종 평가](docs/standards/final_evaluation.md#evaluation) 및 [설명·실패 사례](docs/standards/prediction_explanation.md#evaluation)에서 확인한다. M1 홈 MAE, M3 투수 MAE, M4 ECE 및 일부 상태, M5 다수 ECE 악화는 유지한다. 문서 변경이 새 평가·성능 개선을 의미하지 않는다.

### 7.2 분리 환경의 재현 근거

#40에서 잠금 설치와 두 진입점, 외부 CWD·명시 루트, 없는 루트 Exit 2, 보완 표준 DB 지문, 저장 모델 6개·독립 추론 8건·비교 8건, 연결 시즌 계산 파일 4쌍, 고정 참조 121개, 저장 지표 6개·Train 배경 3개 재계산, 전체 회귀 734개, 보호 파일 458개 변경 없음이 확인됐다. 같은 Windows 장비의 분리 사본에 해당하며 새로운 장비·OS·보안 정책을 입증하지 않는다. 과거 판본 전체 CLI, 현재 새 명세 전체 CLI, 최종 Test 전량 재추론은 이 검증에서 수행하지 않았다. [검증 이력 및 판본](docs/standards/execution_handover.md#evaluation)을 따른다.

<a id="help"></a>

## 8. 오류 대응

| 증상                      | 확인·대응                                                              | 해석                                           |
| ------------------------- | ---------------------------------------------------------------------- | ---------------------------------------------- |
| CLI 도움말/설정 Exit 2    | 절대 루트, 환경 변수, `kbo.toml` 존재·문법·필수 키                     | 잘못된 루트를 자동 대체하지 않음               |
| 설치 실패·하드링크 경고   | 잠금 의존성 및 실제 종료 코드, `uv pip check`                          | 파일 복사 전환 성공은 이번 환경의 확인 사례    |
| 설치 EXE 차단             | 같은 가상환경의 `python -m src`                                        | 보안 정책 해제 금지                            |
| `check` 정상·예측 실패    | 모델 및 입력 존재·권한·계약·지문                                       | 환경 설정 정상과 예측 정상은 별개              |
| 수집 실패 Exit 3          | 결과·시도 이력 및 재개 가능 여부                                       | 원본 수정·무단 재수집 금지                     |
| 파일 I/O Exit 4           | 권한·공간·경로 충돌 및 파일 상태                                       | 실제 파일 존재만으로 무결성 보장 아님          |
| 검사 실패 Exit 5          | 원본/표준 자료·명세·스키마·지문·결과 상태                              | 임의 변경으로 검사 우회 금지                   |
| PowerShell 인용 오류      | Python `sys.argv`로 패키지·경로 인자 전달                              | 설치 오류와 구분                               |
| 회귀 `NativeCommandError` | stdout/stderr를 분리하고 프로세스 종료값 및 `Ran ... tests`, `OK` 확인 | stderr 출력 자체를 테스트 실패로 단정하지 않음 |
| 고정 명세 지문 불일치     | 과거 기록 판본·환경·절대 경로 확보 또는 현재 새 명세 생성              | 과거 지문 치환 금지                            |
| 시즌·선발 미지원·미결정   | 유효 분모, 원천 충돌, 정보 가용성, 규칙 적용 범위 확인                 | 0 또는 공식 사실로 해석하지 않음               |

<a id="maintenance"></a>

## 9. 문서 유지·제한

문서에는 실제 소스·코드·실행·자료·검증·Git 상태를 구분해 기록한다. 과거 파일 지문을 현재 문서에 맞춰 치환하지 않는다. 새 명세는 기존 결과를 보존한 별도 경로에서 생성한다. 최종 Test를 재선정이나 보정에 사용하지 않는다.

공통 문서의 변경은 기능별 문서의 학습·실험 이력을 삭제하거나 새로운 구현을 약속하지 않는다. [개정 전 README 전체](https://github.com/Ahn-SooHyun/kbo-insight-ai/blob/ea4ef457fa35dbc37525cd97586a9f51543e31c5/README.md)는 수집 표본의 세부 명세·모델 자료 변환 이력·개발 결과 및 초기 설치·회귀 실행 기록의 역사 원문이다. 새 실행은 현재 계약과 기록된 자산을 따른다.

과제 최종 제출물은 문제 정의, 데이터, EDA, 기준, 실험, 비교·최적화, 결과·한계 및 참고문헌을 포함하며 실제 AI 활용·사용자 검토 기록을 별도 작성한다. 현재 프로젝트 범위는 배포 및 LLM 결합을 포함하지 않는다.

<a id="commands"></a>

## 10. 설치·기본 확인·대표 예측·테스트 명령

### 10.1 최초 환경 설치

다음은 확인된 Windows PowerShell 명령이다. 경로는 사용자 환경에 맞게 변경한다. 기존 `.venv`가 있다면 새로 만들지 않고 환경·버전을 확인 후 10.2절로 이동한다. 소스 사본과 Python 3.12.15 원본 및 uv 0.13.7을 미리 준비한다.

```powershell
Set-Location -LiteralPath 'D:/Programming/kbo-insight-ai'
$root = (Get-Location).Path
$sourcePython = Join-Path $env:APPDATA 'uv/python/cpython-3.12.15-windows-x86_64-none/python.exe'
$python = Join-Path $root '.venv/Scripts/python.exe'
uv --version
if ($LASTEXITCODE -ne 0) { throw 'uv 확인 실패' }
if (-not (Test-Path -LiteralPath $sourcePython -PathType Leaf)) { throw 'Python 원본 경로를 확인하세요' }
& $sourcePython -B -c 'import sys; print(sys.version); sys.exit(0 if sys.version_info[:3] == (3,12,15) else 2)'
if ($LASTEXITCODE -ne 0) { throw 'Python 버전 확인 실패' }
if (Test-Path -LiteralPath '.venv') { throw '기존 가상환경이 있습니다. 재사용 여부 확인 필요' }
uv venv --no-python-downloads --python $sourcePython .venv
if ($LASTEXITCODE -ne 0) { throw '환경 생성 실패' }
uv sync --locked --extra inspection --python $python
if ($LASTEXITCODE -ne 0) { throw '잠금 설치 실패' }
```

### 10.2 설치·설정·진입점 확인

```powershell
Set-Location -LiteralPath 'D:/Programming/kbo-insight-ai'
$root = (Get-Location).Path
$python = Join-Path $root '.venv/Scripts/python.exe'
$cli = Join-Path $root '.venv/Scripts/kbo-insight.exe'
uv pip check --python $python
if ($LASTEXITCODE -ne 0) { throw '의존성 확인 실패' }
& $python -X utf8 -B -c 'import sys, sqlite3, platform; from importlib.metadata import version; print(sys.executable); print(sys.version); print(platform.machine()); print(sqlite3.sqlite_version); print(version(sys.argv[1])); print(version(sys.argv[2]))' 'kbo-insight-ai' 'pyarrow'
if ($LASTEXITCODE -ne 0) { throw '버전 확인 실패' }
& $python -X utf8 -B -m src --help
if ($LASTEXITCODE -ne 0) { throw '도움말 실패' }
& $python -X utf8 -B -m src --project-root $root check
if ($LASTEXITCODE -ne 0) { throw '설정 확인 실패' }
& $cli --help
if ($LASTEXITCODE -ne 0) { throw '설치 진입점 확인 실패' }
& $cli --project-root $root check
if ($LASTEXITCODE -ne 0) { throw '설치 진입점 설정 확인 실패' }
```

설치 EXE의 실행이 정책상 차단되면 모듈 경로 확인과 구분해 기록한다. `check` 성공 뒤 4절의 승인 자산을 준비한다.

### 10.3 저장 M1 대표 추론

```powershell
$id = '0d9433b8044e48e5a4d21bbdb9b432e5'
$asset = Join-Path $root "models/m1/$id/model.json"
$inputPath = Join-Path $root "results/m1/$id/example_input.json"
$expected = Join-Path $root "results/m1/$id/example_prediction.json"
foreach ($p in @($asset,$inputPath,$expected)) {
    if (-not (Test-Path -LiteralPath $p -PathType Leaf)) { throw "필수 자산 누락: $p" }
}
$prediction = & $python -X utf8 -B -m src.models.game_outcome --project-root $root predict --asset $asset --input $inputPath
if ($LASTEXITCODE -ne 0) { throw '대표 예측 실패' }
($prediction -join "`n") | & $python -X utf8 -B -c 'import json,sys; from pathlib import Path; sys.exit(0 if json.load(sys.stdin)==json.loads(Path(sys.argv[1]).read_bytes()) else 1)' $expected
if ($LASTEXITCODE -ne 0) { throw '대표 예측 비교 실패' }
($prediction -join "`n") | ConvertFrom-Json
```

이 예시는 기준 입력·출력을 읽으며 원본 재수집, 자료 재생성, 학습 및 기준 결과 덮어쓰기를 하지 않는다. 다른 모델의 대표 입력·보조 명세·비교 명령은 [인수 안내](docs/standards/execution_handover.md#inputs)와 기능별 문서를 따른다.

### 10.4 필요한 경우에만 가공·학습·검산

아래는 이미 승인된 자산이 있을 때의 가공 실행 예시다. 새 표준·모델 자료 ID는 실제 명령 출력으로 확보하며 기존 인수판 ID를 덮어쓰지 않는다. 동일 운영 모델을 단독 추론할 경우 실행할 필요가 없다.

```powershell
# 보완 표준 가공: 기존 기초판과 불변 원본을 읽고 새 실행을 생성한다.
& $python -X utf8 -B -m src --project-root $root standardize --as-of-run 4bdbd03e20634ce58135ec2311555351 --base-run 06ae4302ed7544d089fb0c5c8deb5719 --base-verification d936d50a5320463b8dd1c1331569ff19
if ($LASTEXITCODE -ne 0) { throw '표준 가공 실패' }
```

후속 표준 검사에 실제 새 run ID를 `validate-standard --run-id`로 전달한다. 모델 자료는 실제 검증된 run·verification을 사용해 `src.datasets.model_data --run-id ... --verification-id ... --validation-start 2025-01-01 --test-start 2026-01-01 --observation-end 2026-11-26`으로 생성한다. 기존 명세와 입력 지문을 확인한다. 학습 및 기능별 새 실행은 1.1절의 기술 문서를 따라가며, 최종 Test를 새로운 선택 자료로 사용하지 않는다.

### 10.5 회귀 및 읽기 전용 Git 확인

```powershell
& $python -X utf8 -B -m unittest discover -s tests
if ($LASTEXITCODE -ne 0) { throw '전체 회귀 실패' }
git diff --check
if ($LASTEXITCODE -ne 0) { throw 'Git 형식 검사 실패' }
git --no-optional-locks status --short --branch
```

회귀가 stderr 로그를 쓰는 경우 실제 프로세스 종료 코드와 테스트 요약을 확인한다. 현재 기준 문서의 링크 및 각 실행 옵션은 적용 후 다시 검증한다. [실행 및 저장 자산 인수](docs/standards/execution_handover.md)는 동일 조건의 과거 실측 결과와 제한을 관리한다.
