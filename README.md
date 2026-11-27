# KBO Insight AI

**경기에서 시즌까지, 예측 시점과 검증 근거를 함께 관리하는 KBO 분석 프로젝트**

KBO 공개 자료를 자동 확보해 경기·타석·선수·상황·투구를 예측하고, 경기 예측을 정규시즌·포스트시즌 전망으로 연결한다. 예측 정확도뿐 아니라 당시 정보 사용, 정답의 완전성, 확률의 신뢰도와 결과 재현성을 함께 관리한다.

| 항목        | 정의                                                                                                       |
| ----------- | ---------------------------------------------------------------------------------------------------------- |
| 문서 역할   | 프로젝트 진입점·운영 안내                                                                                  |
| 적용 범위   | 기능·자료·설치·실행·문제 해결                                                                              |
| 검토 기준일 | 2026-11-26 (KST)                                                                                           |
| 저장소 경로 | `README.md`                                                                                                |
| 소스 기준   | [`d5bc89f`](https://github.com/Ahn-SooHyun/kbo-insight-ai/commit/d5bc89fb96a2ce2aaabb6a58d78d39618a3a8b9d) |

[저장소](https://github.com/Ahn-SooHyun/kbo-insight-ai) · [개발 흐름](docs/planning/development_workflow.md) · [예측·평가 계약](docs/standards/project_contract.md)

> **현재 제공 범위 · 2026-11-26 (KST)**
> 보완 데이터 인수와 정규시즌 M1·M2 개발 검증을 확인했다. M2는 제출 브랜치 기준이며 병합을 완료했다고 표시하지 않는다. 최종 Test, M3~M5 모델, 시즌 전망·보조 기능의 구현 및 성능은 각 후속 증거로 확인한다. 자료 이용은 승인된 비영리 대학 과제용 로컬 처리 및 내부 평가 범위에 한정한다.

| 기능/자료                                                 | 현재 확인 범위                                                        | 다음 확인 사항                      |
| --------------------------------------------------------- | --------------------------------------------------------------------- | ----------------------------------- |
| 자료 구축 및 보완                                         | 원본 46개 / 논리 대상 48개 / 판본 50개, 보완 표준·모델 자료 재현 인수 | 새 자료·원본 판본 변경 시 재검증    |
| [M1 경기 결과·득점](docs/standards/game_outcome_score.md) | Issue #17 병합; 정규시즌 개발 평가 및 시즌별 백테스트                 | 최종 Test·포스트시즌·시즌 규칙 검증 |
| [M2 타석 결과](docs/standards/plate_appearance.md)        | Issue #18 원격 구현 및 개발 평가·로컬 회귀 테스트 통과; 병합 미확인   | PR 검토·병합·최종 Test              |
| M3·M4·M5                                                  | 자료 계약·유효 후보 인수; 모델 성능 및 완료 미확인                    | 기능별 구현·검증·운영 자산          |
| 시즌 전망·보조 기능·설명                                  | 설계 계약; 실행 및 성능 증거 미확인                                   | 정규시즌/포스트시즌 구분·후속 검증  |

## 찾아보기

| 목적                | 읽을 곳                                                                                                                              |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| 프로젝트 이해       | [기능 지도](#features) → [프로젝트 구조](#documents)                                                                                 |
| 작업 시작·인수      | [선행 관계](docs/planning/development_workflow.md#dependencies) → [개발 기반 기준](docs/planning/development_workflow.md#foundation) |
| 데이터·모델 설계    | [정보 경계](docs/standards/project_contract.md#information) → [담당 모델](docs/standards/project_contract.md#models)                 |
| 설치·기본 확인      | [환경과 설정](#running) → [실행 명령](#commands)                                                                                     |
| 결과 해석·문제 해결 | [결과 읽는 법](#outputs) → [확인 경로](#help)                                                                                        |

<a id="features"></a>

## 1. 기능 지도

### 핵심 예측

아래 표는 프로젝트의 기능 계약이다. 실행 가능 상태는 위 확인 범위 표와 해당 모델의 실제 학습·검증 문서를 함께 확인한다.

| 기능                                             | 예측 결과                                     | 기준 시점                          |
| ------------------------------------------------ | --------------------------------------------- | ---------------------------------- |
| [M1 경기](docs/standards/project_contract.md#m1) | 승무패 확률, 양 팀 득점 분포·예상 득점·스코어 | 경기 시작 전                       |
| [M2 타석](docs/standards/project_contract.md#m2) | 해당 타석 결과별 확률                         | 타석 시작 전                       |
| [M3 선수](docs/standards/project_contract.md#m3) | 출전량·Count·파생 비율·시즌 전망              | 월요일 기준 7일·28일·잔여 정규시즌 |
| [M4 상황](docs/standards/project_contract.md#m4) | 승리 확률·기대득점·전후 변화·WPA·상황 중요도  | 타석 직전·종료 직후                |
| [M5 투구](docs/standards/project_contract.md#m5) | 다음 실제 공의 구종·결과 공동 확률            | 해당 공 투구 전                    |

M1~M5는 기능군이며, **최종 주 예측기 6개**(M1·M2·M4·M5 각 1개, M3 타자·투수 각 1개)를 기본 구성으로 확정했다. M5 위치는 제외한다. 독립 모델의 실제 집계와 미검증 가정은 [모델 수 계약](docs/standards/project_contract.md#model-count)을 따른다.

### 시즌 전망과 보조 기능

- **시즌:** 최종 승수·순위, 정규시즌 우승, 포스트시즌 진출, 시리즈별 결과, 한국시리즈 진출·우승
- **라인업·선수 운용:** 예상 선발 출장·1~9번 타순·수비 위치·선발투수, 등록 선수·벤치·불펜, 투수 교체·다음 후보
- **팀·상대 분석:** 전력·최근 경기력·일정 난이도·타자/투수 상대 분석, 불확실성·예측 설명

**정규시즌 우승은 최종 1위**, **한국시리즈 우승은 포스트시즌을 거친 최종 우승**이다. 서로 다른 출력·정답으로 다룬다. [시즌 계약](docs/standards/project_contract.md#season)에 규칙·일정 가정·평가를 정의한다.

보조 기능은 규칙·통계·기존 예측을 우선 재사용한다. 대타·대주자·수비 교체, 도루·도루실패·견제사, 관측 번트는 세부 항목별 자동 식별·전체 품질 확인 후 포함/제한/제외한다. [조건부 판정](docs/standards/project_contract.md#conditional)을 확인한다.

### 범위와 한계

자료는 다운로드·API·일괄 자동 처리로 확보한다. 실제 몸 상태·비공개 부상·숨은 감독 의도·내부 교체 사유와 수동 기사/영상 라벨은 사용 범위에 넣지 않는다. 전체 시도 수 없는 견제 확률, 군집화·투구 궤적·타구 위치·수상 예측, LLM·웹/앱·예측 API 배포도 제외한다. 상세 경계는 [기능 범위](docs/standards/project_contract.md#scope)에서 관리한다.

<a id="documents"></a>

## 2. 프로젝트 구조와 문서 역할

### 현재 소스 구조

아래는 검토한 제출 브랜치의 실제 Git 추적 경로를 기준으로 한 소스 지도다. 미래 기능의 예정 폴더, 로컬 생성 데이터·자산·가상환경은 실제 추적 파일로 표시하지 않는다.

```text
kbo-insight-ai/
├── .github/
│   ├── ISSUE_TEMPLATE/
│   └── PULL_REQUEST_TEMPLATE/
├── .gitignore
├── .python-version
├── pyproject.toml
├── uv.lock
├── kbo.toml
├── README.md
├── src/
│   ├── __init__.py
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── model_data.py
│   │   ├── postseason_supplement.py
│   │   ├── model_events.py
│   │   ├── model_players.py
│   │   ├── published_sources.py
│   │   └── standardize.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── game_outcome.py
│   │   └── plate_appearance.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── pilot.py
│   │   └── published.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── game_outcome/
│   │   │   ├── __init__.py
│   │   │   ├── __main__.py
│   │   │   ├── backtest.py
│   │   │   ├── model.py
│   │   │   └── poisson.py
│   │   └── plate_appearance/
│   │       ├── __init__.py
│   │       ├── __main__.py
│   │       ├── data.py
│   │       ├── model.py
│   │       └── probability.py
│   ├── validation/
│   │   ├── __init__.py
│   │   ├── raw_sample.py
│   │   ├── standard_data.py
│   │   └── standard_quality.py
│   ├── __main__.py
│   └── settings.py
├── tests/
│   ├── foundation/
│   │   ├── __init__.py
│   │   └── test_foundation.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── test_collection.py
│   │   └── test_published.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── test_game_outcome.py
│   │   ├── test_game_outcome_backtest.py
│   │   ├── test_plate_appearance.py
│   │   └── test_plate_appearance_probability.py
│   └── validation/
│       ├── __init__.py
│       ├── test_inspection.py
│       ├── test_model_data.py
│       ├── test_postseason_supplement.py
│       ├── test_standard_data.py
│       └── test_standard_quality.py
└── docs/
    ├── planning/
    │   └── development_workflow.md
    └── standards/
        ├── game_outcome_score.md
        ├── plate_appearance.md
        └── project_contract.md
```

| 위치                                                | 책임                                                    |
| --------------------------------------------------- | ------------------------------------------------------- |
| `.github/`                                          | Issue·PR 양식                                           |
| `.python-version`                                   | 기본 Python 요청 버전                                   |
| `pyproject.toml`                                    | 패키지 정보·Python 지원 범위·빌드 도구·설치 진입점      |
| `kbo.toml`                                          | 현재 사용하는 경로·Seed·로그 설정                       |
| `.gitignore`                                        | 환경·비밀값 파일·대용량 산출물·캐시 추적 제외           |
| `src/__init__.py`                                   | 패키지 식별. 설정 읽기나 사업 기능을 자동 실행하지 않음 |
| `src/__main__.py`                                   | 도움말·설정 확인 명령과 CLI 로그                        |
| `src/settings.py`                                   | 설정 읽기·타입 및 범위 검증·경로 해석                   |
| `src/models/game_outcome/model.py`                  | 공용 득점 예측기 학습·공식 종료 득점 분포               |
| `src/models/game_outcome/__main__.py`               | 경기 결과·득점 학습·평가·예측 CLI와 자산 관리           |
| `src/models/game_outcome/backtest.py`               | 시즌별 학습 창과 정규시즌 비교 결과 저장                |
| `src/models/game_outcome/poisson.py`                | 백테스트 비교용 포아송 회귀와 득점 분포                 |
| `src/evaluation/game_outcome.py`                    | 득점·승무패 기준 비교와 확률 보정                       |
| `src/models/plate_appearance/data.py`               | 타석 자료 인수·정답·경계·누수 검사                      |
| `src/models/plate_appearance/model.py`              | 타석 클래스 사전·전처리·기존 비교 모델                  |
| `src/models/plate_appearance/probability.py`        | 규칙 제한·개선 학습·보정·운영 선정                      |
| `src/models/plate_appearance/__main__.py`           | 학습·추론·자산 검증 CLI                                 |
| `src/evaluation/plate_appearance.py`                | 타석 확률·분류 지표                                     |
| `tests/models/test_plate_appearance.py`             | 자료·모델·평가 계약                                     |
| `tests/models/test_plate_appearance_probability.py` | 규칙·보정·선정·저장 통합 검사                           |
| `src/datasets/model_data.py`                        | 모델 자료 생성·인수·논리 해시                           |
| `src/datasets/model_events.py`                      | 타석·상황·투구 입력/정답 구축                           |
| `src/datasets/model_players.py`                     | 선수 이력·정답 창 구성                                  |
| `src/datasets/postseason_supplement.py`             | 포스트시즌/시즌 완료 근거 증분 처리                     |
| `tests/validation/test_model_data.py`               | 모델 입력/정답·시간 분할·재현 검증                      |
| `tests/validation/test_postseason_supplement.py`    | 보완 데이터·시즌 완료 근거 검증                         |
| `uv.lock`                                           | 잠긴 설치 의존성                                        |
| `src/ingestion/pilot.py`                            | 제한된 이닝 원본 수집·최소 구조 검사·원본/이력 보존     |
| `src/ingestion/published.py`                        | 승인 명세 기반 배포 파일 수집·원본 버전·실행 현황       |
| `src/validation/raw_sample.py`                      | 고정 원본의 파일 동일성·행 근거를 읽기 전용으로 검사    |
| `src/datasets/published_sources.py`                 | 수집 이력·원본 무결성·판본 선택과 중간 자료 생성        |
| `src/datasets/standardize.py`                       | 경기·타석·투구·선수 기록의 연결 및 표준화               |
| `src/validation/standard_data.py`                   | 표준 스키마·품질 검사·재생성 비교                       |
| `src/validation/standard_quality.py`                | 원본 대응·집계 재계산·후속 사용 후보 최종 검증          |
| `tests/ingestion/test_collection.py`                | 합성 응답 기반 수집·실패·보존 검증                      |
| `tests/ingestion/test_published.py`                 | 합성 파일 전체 수집·갱신·재개 및 불변 저장 검증         |
| `tests/foundation/test_foundation.py`               | 설정·경로·오류·읽기 전용 동작 검증                      |
| `tests/models/test_game_outcome.py`                 | 경기 분할·결합 확률·입력·자산의 합성 검사               |
| `tests/models/test_game_outcome_backtest.py`        | 시즌별 시간 순서, 비교 표본, 포아송 후보 검사           |
| `tests/validation/test_inspection.py`               | 합성 Parquet 기반 원본 검사·실패 동작 검증              |
| `tests/validation/test_standard_data.py`            | 원본 인수·표준 상태·판본 비교의 합성 자료 검사          |
| `tests/validation/test_standard_quality.py`         | 실제 변환 함수·집계 오류·격리·판본 검증 테스트          |
| `docs/planning/`                                    | 개발 순서·선행 관계·증거·인수 기준                      |
| `docs/standards/`                                   | 예측·평가·저장·설정의 상세 계약                         |

현재 구조는 소스·테스트·문서를 폴더로 분리하고, 패키지 내부에서는 명령 진입점과 설정 책임을 파일로 분리한다. 설정과 CLI는 패키지 루트에 두고 수집과 검사는 각각 `ingestion/`과 `validation/`에 둔다.

`pyproject.toml`·`.python-version`·`.gitignore`는 프로젝트 단위 파일이며, `kbo.toml`도 현재 설정 로더가 프로젝트 기준 위치에서 읽는다. 이를 별도 설정 폴더로 옮기려면 로더와 테스트·안내도 함께 변경해야 한다.

경기 결과·득점 예측의 모델과 실행 진입점은 `src/models/game_outcome/`에 둔다. 기준 모델과의 비교·확률 지표는 `src/evaluation/`에서 계산한다. 타석 결과 예측의 운영 확률 계약과 실행 진입점은 `src/models/plate_appearance/`에 둔다. 아직 구현 근거가 없는 기능은 [개발 흐름](docs/planning/development_workflow.md#mapping)의 예정 경로를 참조한다.

### 문서의 역할

| 문서                                                 | 역할                                                  |
| ---------------------------------------------------- | ----------------------------------------------------- |
| README                                               | 프로젝트 소개·폴더 구조·설치·설정·기본 확인·결과 안내 |
| [개발 흐름](docs/planning/development_workflow.md)   | 12개 작업의 선행 관계·입력·산출물·완료 증거·인수      |
| [예측·평가 계약](docs/standards/project_contract.md) | 기능·정보 시점·정답·평가·미확정·재현성의 상세 원문    |

[경기 결과·득점 예측 실행과 결과 계약](docs/standards/game_outcome_score.md)과 [타석 결과 예측 실행과 결과 계약](docs/standards/plate_appearance.md)은 동일한 10절 구조로 입력·정답·학습·강의 원리·평가·재현·실행을 설명한다. 모델별 실제 수치와 세부 방법은 해당 실행 문서를 원문으로 유지한다.

데이터 담당은 [M3 정답](docs/standards/project_contract.md#m3-labels)과 [확정 대기](docs/standards/project_contract.md#decisions), 모델 담당은 [평가](docs/standards/project_contract.md#evaluation), 검토 담당은 [계약 검증표](docs/standards/project_contract.md#checks)와 [인수 체크](docs/planning/development_workflow.md#handover)를 함께 읽는다.

상세 규칙은 계약 한 곳에서 관리한다. README는 요약·안내, 개발 흐름은 적용·인수 절차를 담당한다. 공개 기술 문서는 실제 구현·검증·운영에 필요한 내용으로 구성하고, 개인 작업 기록·미확정 초안·중복 검토본은 별도로 관리한다.

<a id="running"></a>

## 3. 환경·설정·저장 책임

### 3.1 환경과 설치 방식

| 항목                     | 선정 내용                                                   |
| ------------------------ | ----------------------------------------------------------- |
| 확인된 실행 환경         | Windows AMD64                                               |
| Python                   | 3.12.15                                                     |
| 패키지가 허용하는 Python | 3.12 계열: `>=3.12,<3.13`                                   |
| 환경·패키지 관리         | uv 0.13.7                                                   |
| 가상환경                 | 프로젝트의 `.venv`                                          |
| 개발 설치                | 편집 가능 설치                                              |
| 빌드 도구                | `uv_build==0.13.7`                                          |
| 실행 의존성              | 기본 수집은 표준 라이브러리, inspection은 선택 PyArrow 사용 |
| 테스트 의존성            | unittest, 선택 inspection 테스트는 PyArrow 사용             |
| 설치 이름 / import 이름  | `kbo-insight-ai` / `src`                                    |
| 설치되는 CLI 이름        | `kbo-insight`                                               |

Windows AMD64의 Python 3.12.15와 uv 0.13.7에서 편집 가능 설치, 설치형 CLI, 기반·수집·검사 테스트 40개를 확인했다. 다른 OS와 다른 Python 3.12 패치 버전도 별도의 실행 검증이 필요하다.

기본 수집의 실행 의존성은 없지만 선택 inspection 의존성 pyarrow==26.0.1과 uv.lock이 현재 존재한다. Python 기본 요청과 빌드 도구는 각각 `.python-version`과 `pyproject.toml`에 고정한다. 향후 실행 의존성을 추가할 때는 재현 가능한 잠금 방식도 함께 정한다.

설치 과정에서 uv는 호환되는 내장 빌드 도구 또는 고정된 빌드 패키지를 사용한다. [uv 빌드 문서](https://docs.astral.sh/uv/concepts/build-backend/)에서 패키지 구조와 빌드 방식을 확인할 수 있다.

환경 준비·설치는 가상환경·설치 메타데이터·도구 캐시를 생성할 수 있고, 패키지 확보를 위해 네트워크를 사용할 수 있다. 아래의 읽기 전용 동작 검증과 구분한다.

### 3.2 설정 파일과 우선순위

설정은 프로젝트 기준 디렉터리의 `kbo.toml` 한 파일에서 읽는다. 설정 파일은 UTF-8 인코딩이어야 하며, 인코딩 오류와 TOML 문법 오류는 설정 오류로 처리한다. 다음 15개 항목이 모두 필수이며, 누락하거나 지원하지 않는 항목을 추가하면 오류다.

| 설정            | 타입·허용 범위                                          | 제공 파일의 값   |
| --------------- | ------------------------------------------------------- | ---------------- |
| `raw_dir`       | 비어 있지 않은 경로 문자열                              | `data/raw`       |
| `interim_dir`   | 비어 있지 않은 경로 문자열                              | `data/interim`   |
| `processed_dir` | 비어 있지 않은 경로 문자열                              | `data/processed` |
| `models_dir`    | 비어 있지 않은 경로 문자열                              | `models`         |
| `results_dir`   | 비어 있지 않은 경로 문자열                              | `results`        |
| `cache_dir`     | 비어 있지 않은 경로 문자열                              | `.cache`         |
| `seed`          | 0~4294967295의 정수. Boolean 제외                       | `42`             |
| `log_level`     | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` 중 하나 | `INFO`           |

| `collection_base_url` | 허용한 공개 원천 주소 하나만 사용 | `https://api-gw.sports.naver.com` |
| `collection_timeout_seconds` | 1~30 정수, Boolean 제외 | `30` |
| `collection_retries` | 0~2 정수, 최초 요청 뒤 추가 시도 수 | `0` |
| `hf_collection_base_url` | 허용한 공개 HF 배포 주소 | `https://huggingface.co` |
| `hf_collection_max_bytes` | 1~20971520 정수, Naver와 별도 저장 상한 | `20971520` |
| `hf_collection_max_redirects` | 0~3 정수, HF 공식 호스트 전송 상한 | `3` |
| `collection_max_bytes` | 1~1048576 정수, 저장할 응답 bytes 상한 | `1048576` |

표의 값은 제공 파일에 적힌 값이며, 항목 누락 시 적용되는 대체 기본값은 아니다. 로그 수준은 대문자만 허용한다. Seed는 후속 실행에서 재사용할 설정값으로 보관하며 현재 난수 생성이나 모델 학습을 실행하지 않는다.

프로젝트 기준 선택 순서는 다음과 같다.

1. CLI의 `--project-root` 또는 Python 함수에 명시한 기준 경로
2. 프로세스 환경 변수 `KBO_PROJECT_ROOT`
3. 편집 가능 설치에서 확인되는 `src/`의 프로젝트 위치

선택된 기준이 잘못되면 낮은 우선순위로 자동 대체하지 않고 오류를 반환한다. CLI의 기준 옵션은 `check` 앞에 둔다. 환경 변수는 기준 경로만 선택하며 Seed·로그·개별 자료 경로는 `kbo.toml`에서 읽는다.

명시하는 프로젝트 기준은 존재하는 절대 디렉터리여야 한다. 일반 설치에서 소스 위치를 찾을 수 없으면 프로젝트 기준과 그 안의 `kbo.toml`을 별도로 준비해 지정해야 한다.

### 3.3 경로 해석과 검증

상대 자료 경로는 프로젝트 기준 디렉터리에 붙이고, 절대 자료 경로는 해당 위치를 사용한다. 현재 작업 디렉터리는 자료 위치를 결정하지 않는다. 경로의 공백과 한글은 그대로 처리한다. Windows의 드라이브 상대경로(`C:raw`)와 드라이브 없는 루트 경로(`/raw`, `\raw`)는 허용하지 않는다.

경로의 `..`와 심볼릭 링크는 정규화한다. 설정 문자열 안의 환경 변수와 `~`는 자동 확장하지 않는다. 경로의 대소문자 비교는 실행 플랫폼의 `pathlib` 규칙을 따르므로, 실제 파일·폴더 이름의 대소문자도 일치시켜야 한다.

현재 검증은 다음 조건을 확인한다.

- 산출물 디렉터리가 없으면 생성하지 않고 허용한다.
- 산출물 경로나 그 상위가 파일이면 거절한다.
- 산출물끼리 같은 경로이거나 상하위로 겹치면 거절한다.
- 프로젝트 기준 자체와 그 상위 디렉터리를 산출물 경로로 쓰지 못하게 한다.
- `.git`·`.github`·`.venv`·`src`·`tests`·`docs`와 겹치는 산출물 경로를 거절한다.

이 검증은 실제 쓰기 권한·디스크 여유·자료 품질을 확인하지 않는다. 도움말·import·설정 확인을 위해 자료·모델·결과 디렉터리를 생성할 필요는 없다.

### 3.4 저장 경계

아래는 설정된 기본 저장 위치다. 현재 소스 구조에 디렉터리가 미리 생성돼 있다는 뜻은 아니다.

| 기본 경로         | 의미와 책임                                                             |
| ----------------- | ----------------------------------------------------------------------- |
| `data/raw/`       | 확보한 원본. 수정·덮어쓰지 않으며 재수집·공식 정정은 별도 버전으로 보존 |
| `data/interim/`   | 원본의 변환·수정·제외를 거친 중간 자료                                  |
| `data/processed/` | 검증·표준화·입력 구성을 거친 자료                                       |
| `models/`         | 후속 단계의 학습 모델·전처리·보정 자산                                  |
| `results/`        | 후속 단계의 예측·평가·보고 결과                                         |
| `.cache/`         | 재생성 가능한 임시 자산. 유일한 원본·재현 근거로 사용 금지              |

`models/`는 모델 소스 코드가 아닌 저장 자산의 위치다. 실행 소스는 `src/` 아래에 둔다.

원본 보존·부모 자료·버전·실행 식별자와 이력의 의미는 [계약 10절](docs/standards/project_contract.md#reproducibility)을 따른다. 설정 확인은 경로 해석·검증만 수행한다. 명시적인 수집 명령은 원본과 시도 이력을 생성한다. 공개 배포 파일 전체 수집은 3.9절을 참조하고, 표준화·변환·학습 이력 시스템은 후속 범위다.

### 3.5 로그·Secret·Git 추적

정상 확인 결과는 허용된 설정 항목과 Python 버전만 JSON으로 stdout에 출력한다. 로그는 stderr에 수준·패키지 이름·오류 맥락을 남긴다. 파일 로그는 생성하지 않는다.

설정 오류에는 잘못된 설정값 원문이나 하위 입출력 예외 원문을 넣지 않는다. 환경 변수 전체나 Secret을 출력하지 않으며, 진단용 설정 파일에도 실제 비밀값을 넣지 않는다.

현재 `.env`를 자동으로 읽지 않으며 환경 파일은 필요하지 않다. 향후 필요한 경우 허용 파일은 `.env.example`, `.env`, `.env.production`뿐이다. `.env.example`에는 비밀값을 넣지 않고, 실제 비밀값이 있는 `.env`와 `.env.production`은 Git에서 제외한다.

`.gitignore`는 기본 데이터·모델·결과·캐시·가상환경·빌드 산출물과 환경·키 파일을 제외한다. `.env.*` 제외 규칙은 실수로 만든 변형 파일의 추적도 방지하며, `.env.example`은 예외로 허용한다. 다른 환경 파일을 추가로 허용한다는 뜻은 아니다.

자료 경로를 변경하면 해당 위치의 추적 제외 규칙도 함께 검토해야 한다. 설정을 바꾼다고 `.gitignore`가 자동 변경되지는 않는다. 저장소 밖의 절대경로를 쓰면 그 위치를 관리하는 저장소의 추적 규칙도 별도로 확인한다.

### 3.6 명시적인 원본 표본 수집

`ingestion/pilot.py`는 `load_settings`가 반환한 원본 경로·원천 주소·요청 제한을 사용한다. `pilot.py`는 `collect-sample`에서만 네트워크·디렉터리 생성·파일 쓰기를 발생시킨다. import·도움말·기존 check는 수집하지 않는다. 바이트코드 쓰기를 막을 때는 Python의 `-B`를 사용한다.

대상은 공개 원본에서 확인한 경기 ID 하나와 이닝 하나다. ID 형식 검사는 실제 경기 존재 확인을 대신하지 않는다. 현재 진단 설정은 대기 제한 30초·재시도 0회로 한 번만 요청하고, 응답의 저장 크기는 1 MiB 이하로 제한한다. 이전 10초·재시도 1회 실행은 두 시도 모두 Timeout이었으며, 30초 단일 요청도 Timeout으로 실패했다. 제한 초과 확인을 위해 메모리에서 한 바이트를 더 읽을 수 있으나 초과 응답은 부분 실패로 기록한다. 자동 전송은 차단한다. timeout은 네트워크 대기 제한이며 전체 실행시간의 상한은 아니다.

HTTP 500/502/503/504와 시간 초과·일시 네트워크/불완전 응답만 설정 횟수 안에서 재시도한다. 401/403/429·주소 전송·빈/손상/구조 변경 응답은 자동 재시도하지 않는다. 차단을 우회하거나 브라우저·인증 정보를 흉내 내지 않는다.

원본과 이력은 `raw_dir/collection/{run_id}/`에 배타 생성한다.

| 파일            | 의미                                                             |
| --------------- | ---------------------------------------------------------------- |
| request.json    | 대상·공개 요청 조건·요청 제한·코드 해시·참고 revision            |
| attempt-NN.bin  | 응답 bytes. 정상/실패 여부와 완전성은 대응 JSON에서 확인         |
| attempt-NN.json | 시도별 시각·상태·실패 코드·크기·경로·SHA-256·완전성              |
| result.json     | 마지막 시도 결과. 이 파일이 없으면 완료된 수집으로 취급하지 않음 |

응답이 없거나 HTTP 오류 본문을 읽지 않은 경우 bin 파일은 없다. 부분 응답은 `body_complete=false`이며 정상 원본으로 사용하지 않는다. 원본은 저장 후 재열람해 해시를 검사한다. 반복 실행은 별도 run_id를 사용하며 기존 파일을 덮어쓰지 않는다. 쓰기·공간 부족으로 중단된 폴더도 자동 삭제하지 않는다.

`structure_validated`는 최소 JSON 구조 검사만 통과했다는 뜻이다. 사건 순서·중복·실제 공·타석 경계·교체/주루/번트·대상 응답 일치·좌표 의미·전체 포함·품질 통과를 보증하지 않는다. 추가 필드는 허용하지만 필수 구조·타입이 달라지면 실패한다. `semantic_validation=not_performed`를 유지한다.

관측 시각은 UTC로 남기고 최초 공개 시각은 확인되지 않으면 null이다. 원천 API의 명시 버전은 미확인이므로 null이며 참고 수집 코드 revision과 실제 적용 파일 hash를 별도로 남긴다. 새 수집 이력의 `code_sha256` 키는 패키지 기준 경로이며, 수집 구현은 `ingestion/pilot.py`로 기록한다. declared_base_commit은 인수 기준으로, 현재 Git HEAD 주장과 구분한다. URL·예외 원문·환경 전체·인증 헤더를 무차별 기록하지 않는다.

기본 실행은 확인된 Python 모듈 진입점이다. 설치 실행 파일 정책 제한을 우회하지 않는다. 다음 명령은 합성 응답 테스트 통과 뒤 별도로 실행하며, 실제 응답 성공은 아직 확인하지 않았다.

```powershell
& "D:/Programming/kbo-insight-ai/.venv/Scripts/python.exe" -B -m src --project-root "D:/Programming/kbo-insight-ai" collect-sample --game-id "20230401HHWO02023" --inning 1
```

정상 구조/저장은 종료 0, 인자·설정 오류 2, 수집/응답 검사 실패 3, 원본·이력 저장 실패 4다. 종료 0도 출처 이용 범위나 실제 전체 수집의 승인이 아니다. 실제 수집 전 출처 조건을 확인하고, 수집 후 원본/hash/이력과 의미 검증 근거를 인계한다. 캐시는 유일한 원본으로 사용하지 않는다.

계약의 출처·식별, 공개 시점, 교체·주루·번트, 타석·선수·투구·시즌 규칙 관련 조건은 실제 근거가 확보되지 않은 범위에서 미확정이다. 전체 기간·학습 날짜·조건부 최종 판정·M4 산식은 이번 구현에서 결정하지 않는다. 기존 테스트는 합성 응답을 실제 표본으로 보고하지 않는다.

### 3.7 고정 버전 배포 파일 표본

Naver의 두 실행은 각각 10초 두 시도, 30초 한 시도 모두 HTTP 상태·본문 없이 Timeout으로 끝났다. TCP·별도 TLS 성공만으로 수집 당시 원인을 확정하지 않으며 이 경로를 성공으로 기록하지 않는다. 동일 요청은 반복하지 않는다.

대체 원천은 공개 배포된 `slothman3878/kbo_playbyplay`의 2023년 파일 하나다. 물리 배포 단위가 시즌 파일이므로 1이닝보다 크다. 파일 전체를 원본으로 보존한 뒤 일부 경기/필드를 검토하는 표본이며, 전체 기간 수집이나 2023년 전체 품질 통과를 뜻하지 않는다.

| 항목         | 고정 값                                                          |
| ------------ | ---------------------------------------------------------------- |
| revision     | 6afc8af044e3bba5f326b688e8cb41d7ff7065ec                         |
| 파일         | v0/kbo_pbp_2023.parquet                                          |
| 게시 크기    | 17,327,447 bytes                                                 |
| 게시 SHA-256 | 818f6016655b02fe48b8118281d1b04bfe3548d376fdc70131a41ea539341edb |

[파일/해시 근거](https://huggingface.co/datasets/slothman3878/kbo_playbyplay/blob/main/v0/kbo_pbp_2023.parquet)와 [revision의 파일 크기/해시](https://huggingface.co/datasets/slothman3878/kbo_playbyplay/commit/6afc8af044e3bba5f326b688e8cb41d7ff7065ec)를 조사했다. 요청은 main 대신 위 revision으로 고정한다. 원문 commentary가 없는 파생 배포 파일이며 라이선스/원출처 조건과 사건 의미는 별도 확인한다.

명시적인 `collect-sample --source hf`만 이 파일을 요청한다. 별도 일회용 수집기 없이 기존 통신·원본·해시·시도/결과 저장을 재사용한다. Naver의 1 MiB 상한과 전송 금지 정책은 유지한다. HF는 별도 최대 20 MiB·재시도 없음·최대 3회 주소 전송으로 총 HTTP 요청 최대 4회다. HF 파일 다운로드는 [공식 문서](https://huggingface.co/docs/hub/datasets-downloading)에 명시된 저장/CDN 호스트로 전송될 수 있으므로 HTTPS의 huggingface.co 및 그 하위, \*.hf.co만 허용한다. 다른 호스트·HTTP·사용자정보·비표준 포트·fragment는 거절한다. 서명 query나 Location은 로그/manifest에 저장하지 않는다. 별도 인증 정보·브라우저 위장은 사용하지 않는다. 실제 전송/CDN 접근 성공은 아직 확인하지 않았다.

정상 파일은 `attempt-01.parquet`로 보존한다. 실패 bytes는 bin과 실패 상태로 남긴다. 게시 크기·SHA-256·Parquet 시작/끝 표식을 대조한 경우에만 `artifact_verified`를 사용한다. **이는 파일 동일성 확인이며 Parquet 필드 파싱·ID 연결·사건 검증 성공이 아니다.** `semantic_validation=not_performed`를 유지한다. 파싱 라이브러리는 아직 추가하지 않았으며, 실제 확보 후 필요한 버전/재현 방식을 확인한다. 캐시가 아닌 raw_dir의 원본·해시·이력을 인계한다.

```powershell
& "D:/Programming/kbo-insight-ai/.venv/Scripts/python.exe" -X utf8 -B -m src --project-root "D:/Programming/kbo-insight-ai" collect-sample --source hf
```

이 명령의 실제 로컬 실행 로그에서 HTTP 200·종료 0을 확인했고, 담당자가 저장된 17,327,447 bytes와 게시 해시를 대조했다. run_id는 96a971b890474a87946532f7c5ace4ea다. 실제 HTTP 요청 2회·주소 전송 1회였다. Naver 실패와 남은 핵심/조건부 자료 제한은 별도로 유지한다.

### 3.8 확보한 HF 원본의 읽기 전용 검사

`inspect-sample --run-id 실행ID`는 원본·수집 이력의 연결과 고정 파일 해시를 다시 확인한 뒤 실제 Arrow 필드/타입/null 수, 파일 메타데이터, 원본 행 번호가 붙은 표본 사례를 JSON으로 출력한다. 새 HTTP 요청·원본/이력 변경·디렉터리 생성을 하지 않는다. `quality_status=not_assessed`를 유지하며 전체 품질 검사의 성공을 반환하지 않는다. 검사 실패는 종료 코드 5다.

검사에는 선택 설치 `inspection`의 `pyarrow==26.0.1`을 사용한다. 현재 개발 환경의 설치 버전·선택 의존성 선언·uv.lock에서 같은 버전을 확인했다. 이 선언과 잠금 파일은 이미 존재한 것을 재사용하며 이번 검사 연결에서 덮어쓰지 않았다. 재현 환경은 잠금 파일의 inspection extra로 준비한다. 기본 수집·도움말·check는 PyArrow를 불러오지 않는다.

```powershell
& "D:/Programming/kbo-insight-ai/.venv/Scripts/python.exe" -X utf8 -B -m src --project-root "D:/Programming/kbo-insight-ai" inspect-sample --run-id "96a971b890474a87946532f7c5ace4ea"
```

파일의 행 수/날짜/경기 ID 수는 읽은 파일의 관측값이며 공식 일정 대비 포함률·시즌 완료·전체 품질 통과가 아니다. 같은 타석 키에서 참가자 ID가 달라지는 사례는 변화 후보로 기록하며, 원문 교체 역할·공식 결과 귀속·미완료 타석의 주루 종류는 독립 원천으로 확인해야 한다. 게시 명세만으로 실제 행을 모두 최종 참가자라고 가정하지 않는다. 원본 좌표와 null을 수정하거나 0으로 채우지 않는다.

### 3.9 공개 배포 파일 전체 수집 및 원본 버전 관리

기존 표본 수집/검사는 그대로 유지한다. 실제 전체 수집은 출처별 저장 허용과 확정 명세를 받아 별도로 실행한다. 기존 표본·정정 후보와 비교 자료의 보류 상태를 자동으로 해제하지 않는다. Naver/KBO 원천 요청 및 원천 스크래퍼는 이번 전체 수집 대상이 아니다.

`src/ingestion/published.py`는 GitHub 고정 commit의 raw 파일과 HF 고정 revision의 배포 파일을 지원한다. 기존 표본 통신·해시·시각 유틸리티를 재사용한다. 파일별 상한은 기존 `hf_collection_max_bytes` 이하이며 요청 간격, 제한된 재시도 및 Backoff를 명세로 관리한다. HTTP 403/429 발생 시 해당 출처의 남은 파일은 현재 실행에서 접근 제한 상태로 남긴다. 응답 전송·서명 주소와 오류 본문은 기록하지 않는다.

#### 대상 명세

JSON의 최상위 객체는 `{"schema_version": 1, "targets": [...]}`이다. 각 대상의 설정 필드는 아래와 같다. 현재 승인된 실제 전체 수집 명세는 제공하지 않는다. 합성 검증은 테스트에서 임시 명세를 만들고 통신을 대체한다.

| 필드                                                  | 의미                                                                                                               |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| target_id, source, repository, revision, filename     | 파일 식별자, github/hf, owner/repo, 40자 고정 commit/revision, 배포 상대 파일명                                    |
| kind, season, competition, start_date, end_date       | 자료 종류, 시즌, regular/postseason, ISO 기간                                                                      |
| publisher, upstream, evidence_url                     | 발행자, 상류, 출처/이용 근거 공개 HTTPS 링크                                                                       |
| approval_ref, permitted_uses                          | 수집 승인 기록과 허용 용도. 승인 참조 및 download/store가 모두 없으면 수집 보류. process/train/publish는 별도 판정 |
| required_fields, max_bytes, retries, interval_seconds | 최소 필드, 파일별 상한, 0~2회 추가 시도, 1~60초 요청 간격                                                          |
| expected_bytes, expected_sha256                       | 배포자가 제공한 정확한 크기·SHA-256이 있을 때 대조, 그 외 null                                                     |
| records_path                                          | JSON에서 레코드 배열까지의 키 목록, 기본 빈 목록                                                                   |
| published_at, publication_evidence                    | 근거 있는 공개 시각(시간대 포함 ISO 형식)과 근거, 없으면 null                                                      |
| season_complete, season_evidence                      | 시즌 종료 근거와 근거 참조, 미확인 null                                                                            |
| empty_evidence                                        | 정상 무자료를 허용하는 근거, 없으면 빈 레코드 실패                                                                 |
| game_id_field, expected_game_ids, coverage_evidence   | 원본 경기 ID 필드, 자동 산출한 기준 ID 목록과 근거. 없으면 경기 완전성 미평가                                      |
| game_status_field, status_map                         | 원본 상태 필드와 확인된 값→scheduled/postponed/cancelled/suspended/completed/unknown 대응. 미확인 상태는 unknown   |

`kind`는 schedule_results, play_by_play, player_game, starter_history, postseason, rules 중 하나다. CSV·JSON 레코드 배열·JSONL·Parquet의 최소 구조만 검사한다. Parquet에는 기존 inspection extra가 필요하다. 필드·상태 매핑은 실제 승인 표본에 근거해 확정한다. 예상 경기 ID 목록은 자동 확보 근거에서 생성하며 수동 경기 입력은 사용하지 않는다. 승인 참조와 공개 URL만 기록하고 Secret을 포함하지 않는다.

#### 실행 및 원본 보존

명세 파일을 확정한 뒤에만 collect-files 또는 update-files를 실행한다. 갱신은 새 고정 판본을 명세에 반영해 실행하고 원본을 덮어쓰지 않는다. 동일 내용은 SHA-256 주소의 원본 객체를 공유하되 새 실행 기록을 남긴다. 정정 후보는 이전 데이터와 내용이 다른 경우이며 공식 정정이란 단정이 아니다.

| 위치                                                                | 책임                                                                  |
| ------------------------------------------------------------------- | --------------------------------------------------------------------- |
| raw_dir/published/objects/{sha256}.raw                              | 파싱 및 지정 해시·크기 검사 통과한 원본 bytes, 내용 주소 기반 불변    |
| raw_dir/published/runs/{run_id}/manifest.json                       | 실행 시각, 사용한 대상 명세, 코드 해시와 요청 범위 및 재개 연결       |
| raw_dir/published/runs/{run_id}/{target_id}.attempt-NN.started.json | 요청 직전 시도 기록                                                   |
| raw_dir/published/runs/{run_id}/{target_id}.attempt-NN.json         | 시도별 결과, 실패와 부분 응답 크기 및 해시                            |
| raw_dir/published/runs/{run_id}/{target_id}.result.json             | 파일 결과·원본 경로·판본·관측 시각 및 최소 검사/대조                  |
| .part                                                               | 저장 중 임시 파일, 완료 기록이 아니며 중단 시 자동 삭제·채택하지 않음 |

원본 파일·이력은 같은 파일시스템에서 임시 파일 생성, flush/fsync, 배타적인 하드링크 게시 및 재열람 해시 대조로 저장한다. 파일시스템에서 하드링크나 쓰기를 허용하지 않으면 저장 실패로 중단하므로 실제 저장 매체에서 검증해야 한다. 기본 raw 경로는 Git 제외 대상이다. 캐시는 원본 대체 저장소로 사용하지 않는다.

`resume-files`는 이전 실행의 고정 명세를 재사용하고 성공 원본을 재검증한 뒤 미완료·실패 파일부터 다시 요청한다. 재개 시에도 실제 승인이 유효한지 확인한다. 부분 바이트 이어받기는 하지 않는다. 새 실행 ID와 이전 실행 연결을 보존한다. `report-files`는 수집망 접근 없이 해당 실행의 명세·결과로 읽기 전용 현황을 출력한다. 실행 중단 대상은 attempt 시작 기록의 존재에 따라 interrupted 또는 not_attempted로 표시한다.

예시 명령 형식(아래 명세 경로와 실행 ID는 확정된 실제 값으로 대체한다):

```powershell
Set-Location -LiteralPath 'D:/Programming/kbo-insight-ai'
$kboPython = Join-Path (Get-Location).Path '.venv/Scripts/python.exe'
& $kboPython -X utf8 -B -m src collect-files --manifest 'D:/approved/manifest.json'
& $kboPython -X utf8 -B -m src update-files --manifest 'D:/approved/updated-manifest.json'
& $kboPython -X utf8 -B -m src resume-files --run-id '<실행 ID>'
& $kboPython -X utf8 -B -m src report-files --run-id '<실행 ID>'
```

종료 코드 0은 해당 실행에서 선정 파일의 확보 또는 근거 있는 정상 무자료까지 확인한 것이다. 경기 완전성·시즌 종료·학습 허용·실제 프로젝트 완료와 같지 않다. 2는 명세/인자/설정 오류, 3은 파일 확보 실패 또는 보류/미완료, 4는 저장·이력 접근 실패다. 기존 inspect-sample의 5도 유지한다. 수집 결과의 경기 ID 누락·추가·미확인 상태 및 시즌 완료 근거는 별도 인수한다.

### 3.10 표준 데이터와 품질 검사

수집한 불변 원본과 수집 이력만 읽어 표준 자료를 생성한다. 재수집·네트워크 통신 없이 기존 경로 설정 및 inspection 선택 의존성(pyarrow), 표준 라이브러리 sqlite3를 재사용한다. 원본 해시·파일 크기·판본을 검사한 후 중간 자료와 표준 자료를 별도의 실행 ID 아래에 저장한다.

```powershell
# 최신 판본 적용
& .venv/Scripts/python.exe -B -m src standardize
# 이전 수집 실행까지의 판본 적용 (전체 대상이 확보된 실행 ID 사용)
& .venv/Scripts/python.exe -B -m src standardize --as-of-run ca57b43fc9604710ad5fd8bf8cf8aff6
# 이전 표준화 실행 ID와 비교 (실제 ID로 변경)
& .venv/Scripts/python.exe -B -m src standardize --compare-run <표준화_실행_ID>
```

중간 자료는 `data/interim/standard/{run_id}/`, 표준 SQLite DB(`standard.sqlite3`)와 검사 보고서(`quality.json`)는 `data/processed/standard/{run_id}/`에 생성한다. 설정에 경로가 다르게 지정된 경우 그 경로를 따른다. 중간 자료의 manifest에는 원본 SHA-256, 수집 결과와 대상 명세, 채택 판본, 코드·설정 해시와 중간 파일 해시를 기록한다. 이전 실행의 원본이나 결과를 덮어쓰지 않는다.

| 표준 테이블         | 키 또는 논리 단위                    | 내용 및 이용 제한                                             |
| ------------------- | ------------------------------------ | ------------------------------------------------------------- |
| games               | game_id                              | 정규·포스트시즌 점수/상태; 연기 일정은 완료 경기와 구분       |
| schedule_reference  | source_sha, source_row               | 월별 일정·취소·대회명 원문; 결과 및 승자를 추정하지 않음      |
| plate_appearances   | game_id, pa_number                   | 타석 전후 상태, 결과 및 시작/종료 참가자, 공식 귀속 후보      |
| pitches             | game_id, pa_number, pitch_number     | 양수 투구 번호의 관측 기록; 상태·순서·정답 후보를 별도 판정   |
| non_pitch           | source_sha, source_row               | 투구 번호 0의 관측 사건; 실제 투구 수에서 제외                |
| player_observations | player_id, season, team_id, name     | HF 내부 ID의 이름·팀 관측; 이름만으로 다른 ID를 병합하지 않음 |
| player_games        | game_id, team_id, player_id, role    | 관측 집계와 완전성 후보; 공식 기록 검증과 구분                |
| starters            | source_sha, source_row, side, method | 명시된 선발 및 PBP 최초 관측 선발 후보; 과거 공개 시점 미확인 |
| pitching_records    | source_sha, source_row               | 직접 투수 등판 기록; ID 미관측·이닝→아웃 변환 검사            |
| game_checks         | game_id                              | 경기 종료 점수 대조, 타석·투구 순서 및 경계 검사              |
| comparisons         | source_sha, source_row, metric       | 원천 간 관측 집계 일치·불일치·비교 불가                       |
| issues              | id                                   | 원인·영향·출처·행 위치와 격리·경고·제외 분류                  |

`source_row`는 0부터 시작하는 원본 대응 행 번호다. Dent 포스트시즌은 네 시리즈를 펼친 순서로 기록한다. HF의 post\_\* 값은 타석 종료 상태이며 투구 직후 상태가 아니다. 과거 공개 시각의 근거 없이 실제 선발·최신 일정 등을 경기 전 입력으로 사용하지 않는다. Window의 시즌 최종 선발 통계는 중간 자료에만 보존한다. 원본 제공 자료의 이용 조건과 2026-11-25 프로젝트 내부 사용 범위 결정(비영리 대학 과제용 로컬 표준화·학습·내부 평가; [PR #12](https://github.com/Ahn-SooHyun/kbo-insight-ai/pull/12))을 분리하며 실제 데이터 재배포/외부 서비스는 제외한다.

현재 확인한 전체 원본 46개(44개 논리 대상)를 이전 판본, 최신 판본 및 동일 입력으로 각각 처리했다. 최신판에서 정규시즌 완료 16,064경기, 연기 1,129행, 포스트시즌 실제 결과 147경기, 논리적 타석 그룹 226,819개, 실제 투구 882,451행, 무투구 386행이 생성되었다. HF 2,880경기의 최종 점수는 Dent와 일치하고 동일 입력의 12개 표준 테이블 해시는 일치했다. 사용자가 실행한 전체 테스트 72개는 통과했다. 다만 이러한 결과는 원천 필드의 의미 또는 공식 기록 귀속까지 모두 검증했다는 뜻이 아니다.

검증 보고서에서 타석 결과 미확정, 상태 경계·투구 순서 불연속, 선수 공식 기록 귀속 및 집계 불일치, 선발/등판 누락을 확인할 수 있다. `coverage_candidate`, `result_valid`, `state_valid`, `order_valid`, `target_valid`는 서로 별개의 후보 판정이다. 원천 간 비교의 불일치가 있는 후보는 후속 보완 및 인수 검토 전 최종 정답으로 쓰지 않는다. 시즌별 완료·공식 경기 규칙 및 정보 공개 시점도 미확인 부분을 남긴다. 미관측을 실제 0으로 채우지 않으며 모델별 Feature/Target 확정·시간 분할·학습은 후속 작업으로 분리한다.

표준화 명령의 정상 종료(0)는 처리가 완료됐음을 나타내며 모든 자료가 사용 가능하다는 의미는 아니다. 파일 입출력 실패(4), 검사 실패(5), 실행 중단(3)은 별개이며, 오류 시에도 원본과 이전 성공 실행은 보존한다.

### 3.11 표준 자료 최종 품질 검사 및 인수

```powershell
& .venv/Scripts/python.exe -B -m src validate-standard --run-id <표준화_실행_ID>
& .venv/Scripts/python.exe -B -m src validate-standard --run-id <표준화_실행_ID> --compare-with <이전표준화ID:검증ID>
```

`standardize`는 원본→중간→표준 자료를 생성하는 변환 단계이고, `validate-standard`는 완료된 표준 자료를 기준으로 실제 원본과 중간 행의 동일성, 논리 테이블 해시, 전체 원본 행의 변환·격리 추적과 선수 집계를 재검사하는 최종 단계다. 후속 모델별 데이터 구축에는 **최종 검증 DB와 검증 보고서**를 함께 사용한다.

완료된 표준화 폴더의 `standard.sqlite3`, `quality.json` 및 중간 자료는 변경하지 않는다. 검증마다 같은 부모 폴더에 `verified-{verification_id}.sqlite3`와 `verified-{verification_id}.json`을 생성하며, 서로 다른 입력 판본의 영향과 동일 입력 재현성을 구분해 비교한다. 실패한 검증 산출물은 완료 보고서 없이 후속 작업에 사용하지 않는다. 새 설정·설치·수집은 필요하지 않다.

| 최종 추가 테이블·뷰        | 내용 및 주의                                                 |
| -------------------------- | ------------------------------------------------------------ |
| pa_audit                   | 원본 타석 시작, 투구 순서, 전후 상태, 득점 일치, 결과 종류   |
| aggregate_checks           | 실제 투수 등판 기록과 상대 타자 측 집계 및 BB+HBP 대조       |
| player_admission           | 최초 후보, 선수·팀 불일치, ID·경기 상태 제한 및 최종 후보    |
| postseason_series          | Dent 네 단계 원본 시리즈 메타데이터, 시즌·시리즈 완료 근거   |
| postseason_schedule        | 상대 미정 일정까지 원문 보존; 실제 게임 연결은 가능한 경우만 |
| eligible_games             | 확인된 점수·결과 후보, 연기 및 미확정 일정 제외              |
| eligible_pa                | 결과·시작·상태·순서 검증 및 참가자 변경 제외                 |
| eligible_states            | 확인된 타석 전후 상태 후보; 투구 직후 상태로 사용 금지       |
| eligible_pitches           | 실제 투구의 구종·결과·순서·상태 후보                         |
| location_candidates        | 위치 값 비결측 후보, 원단위와 위치 의미 별도 확인            |
| eligible_player_games      | 선수·팀 집계 불일치 및 불확실 경계를 반영한 관측 집계 후보   |
| direct_pitching_candidates | 검증된 직접 등판 행; 선수 ID 연결 여부 별도 표시             |
| starter_candidates         | 사후 실제/최초 관측 선발 후보; 공개시점 입력 근거가 아님     |

최종 DB는 이전 DB의 내용을 복제하여 검증한다. `player_games.coverage_candidate`는 직접 및 팀 대조 불일치와 추가 감사 결과로 좁히며 최초 값은 `player_admission.preliminary_candidate`에 보존한다. 전처리·모델별 target·시간 분할 및 학습은 수행하지 않는다. 원본 관측이 어려운 공식 기록 귀속, 출전하지 않은 선수와 출전 기록이 누락된 선수의 구분, 원천별 공식 규칙 및 공개 시점은 임의로 복원하지 않는다.

#### 후속 데이터 의미 및 출처 대응

| 원본                      | 주 표준 필드 및 단위                                                                                                                                      | 제한                                                             |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| Dent 정규시즌 JSON        | gamePk→game_id, officialDate→date, teams→팀 ID/점수(정수), status.detailedState→상태                                                                      | 2001~2026 결과; 당시 공개 일정으로 소급하지 않음                 |
| HF Parquet                | game_pk→경기 ID(연도 접미부 조건 대조), at_bat_number→pa_number, pitch_number→실제 투구 순번(0은 무투구), batter/pitcher→HF 이름공간 ID, events→타석 결과 | 2023~2026; post\_\*는 타석 종료 값                               |
| HF 상태 및 투구           | balls/strikes/outs_when_up→정수 카운트, on_1b..on_3b→주자 ID/NULL, pitch_type/pitch_result/type→원천 코드, release_speed_kmh→km/h                         | 좌표의 단위·관측 시점은 원천 근거 확인 필요, 위치 누락 별도 표시 |
| Window 선발 CSV·일일 기록 | game_id, away_starter/home_starter→사후 선발 이름                                                                                                         | 이름만으로 선수 ID 병합 금지; 시즌 최종 통계는 중간 자료만 보존  |
| Window 투수 JSONL         | innings→outs(정수 1아웃 단위), tbf/pitches/ab/h/hr/bb_hbp/so/r/er→관측 집계                                                                               | 2026 일부 경기, PBP와 동일 소스 의존 가능성 및 미관측 ID를 표시  |
| Dent 포스트시즌·LOPES     | 시즌/단계/날짜/팀/실제 점수                                                                                                                               | 결과 확보 기간 한정, 2026 이전/최신 판본 구분                    |
| Dialektike 일정 CSV       | G_ID, G_DT, G_TM, HOME_ID/AWAY_ID, SR_ID/SR_NM, CANCEL_SC_ID/NM                                                                                           | 일정과 취소·대회 원문 참조, 점수/승자 생성 금지                  |

검증 보고서의 `row_accounting`은 선정 대상별 물리 행 포함 및 제외 수를, `handoff.coverage_by_season`은 기능별 분모와 후보 수를, `handoff.player_admission`·`handoff.aggregate_checks`는 제한과 비교 결과를 제공한다. `handoff.model_contract`는 후속 작업의 키와 출처·기간·정보 경계를 기록한다. M3 월요일 기준 7일/28일/잔여 정규시즌의 관측 종료는 모델 자료 생성 단계에서 판정하며, 시즌 완료 근거가 없으면 잔여 정답을 확정하지 않는다. 공식 순위·동률·진출·시리즈 규칙은 시뮬레이션 전에 별도로 검증해야 한다.

검증 대상의 DB 및 중간 파일 해시가 일치하지 않거나, 원본 행이 설명 없이 소실되거나, 선수 관측 집계의 재계산이 일치하지 않으면 실패한다. 출처 간 근본적인 불일치는 원본을 임의로 덮어쓰지 않고 최종 후보에서 제한한다. 이전 판본과 동일한 코드/설정으로 검증해야 하며 원본 변환 코드가 변경되었다면 새 표준화 실행을 생성한다.

### 모델별 입력·정답 및 시간 분할

최종 표준 검증판과 부모 DB, 품질 JSON, 중간 명세 및 원본 파일의 해시를 연결한다. 원본·표준 DB를 수정하지 않고 설정된 `processed_dir/model_data/<실행 ID>/`에 신규 산출물을 작성한다. 기존 Python 표준 라이브러리를 사용하며 추가 의존성/GPU가 필요하지 않다.

```powershell
$run = '06ae4302ed7544d089fb0c5c8deb5719'
$verification = 'd936d50a5320463b8dd1c1331569ff19'
& .venv/Scripts/python.exe -B -m src.datasets.model_data --run-id $run --verification-id $verification --validation-start 2025-01-01 --test-start 2026-01-01 --observation-end 2026-11-26
```

데이터 생성기 1.1.0은 이름 식별 시점을 분리하고 정답 품질·제외 명세를 확장한다. 변경된 코드와 이전 생성기 판본을 동일 입력으로 비교하지 않는다. 같은 수정 판본에서 새 첫 실행을 생성한 뒤 `--compare-run <첫 모델 자료 실행 ID>`를 추가한다. `manifest.json`의 `status=completed`, `comparison.reproducible=true` 및 변경 테이블 없음이 같은 입력 재생성의 검사 결과다.

| 산출물                         | 활용                                                                                                                                                                     |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `model_data.sqlite3`           | `m1`, `m2`, `m3_hitter`, `m3_pitcher`, `m4`, `m5`의 `_inputs`, `_targets`, `_index`; 선수 정답 창 경기·팀 목록, 팀·역할 품질 및 식별 최초 관측 근거; 시즌·선발 보조 원천 |
| `feature_target_contract.json` | 모델별 입력/정답 의미와 이용 시점, 결측 처리 및 학습 필터                                                                                                                |
| `train_eda.json`               | 학습 분할만의 분포·결측·클래스·비율 정답 범위                                                                                                                            |
| `manifest.json`                | 실제 분할별 수량·기간·상태, 원본/부모 버전, 파일 및 논리 해시, 설정, 제한 및 비교                                                                                        |

모델 자료의 학습 기간은 2025-01-01 전, Validation은 2025년, 최종 Test는 2026년이다. M1은 2001년부터, 타석·상태·투구·선수 기록은 2023년부터 확보되어 있다. 경기 당일 과거 집계는 동일 날짜 경기를 모두 제외한다. 예측 대상 경기 대진은 알려진 조건으로 취급하며 실제 당시 일정 스냅샷이 존재한다고 주장하지 않는다. M2·M4·M5는 사건 경계를 따른다. M2는 전체 표준 타석을 `m2_index`에 보존하며 참가자 변경·무실제 투구·경계/결과 상태·귀속 후보와 제외 사유를 별도 기록하고 유효 타석만 입력/정답으로 사용한다. M5의 현재 공 구종/결과/속도/좌표 및 타석 종료 값은 입력하지 않는다. M4의 타석 전후 상태는 같은 경기 묶음에 둔다.

M3은 선수의 과거 관측 모집단·역할·마지막 관측 팀을 기준으로 월요일 이전 기록을 집계한다. `player_identity_evidence`는 원본 행에서 이름의 최초 관측일을 추적한다. M2·M3·M5 및 선발 과거 이력에서 해당 날짜 엄격 이전에 확인된 이름 충돌만 입력의 식별 제한에 반영하며, 미래 충돌을 소급 적용하지 않는다. 경기별 품질 기본 조건은 보존하고 정답에는 최종 검증 admission을 적용한다. 과거 발표 판본의 완전 재현을 주장하지 않는다. 각 정답 창은 7일/28일/잔여 정규시즌으로 구분한다. 관측 종료·개인 격리·관측 누락·관련 팀 및 역할의 경기 상태, 집계 불일치, 식별 및 역할 행 위험을 판정하고 `usable=1`인 기간의 `target_*`만 학습 후보로 사용한다. `player_game_role_quality`는 완료 경기 양 팀의 두 역할 품질을, `m3_window_teams`는 정답 검증에 사용된 과거 및 미래 관측 팀을 기록한다. 미래 팀은 입력이 아니다. `observed_*`의 부분 집계는 정답으로 사용하지 않는다. `zero_observed`는 리그 해당 역할의 위험과 격리 기록이 없는 관측 범위의 0에 한정하며 공식 등록/출전 0을 의미하지 않는다. `valid_observed`도 관측된 팀 이외의 미확인 이동 및 공식 모집단 완전성을 보장하지 않는다. 비율은 유효 분모에서만 계산하며 NULL은 0이 아니다. 기존 검증판은 2023~2025년의 종료 근거가 없어 잔여 정답을 확정하지 않았다. 보완판은 아래 공식 정규시즌 종료 근거를 경기 집합과 대조한 후 기간별 관측 품질을 별도로 판정한다. 2026년 Test의 잔여 정답을 학습으로 바꾸지 않는다. 선수 성적은 관측 PBP 기반으로 자료 누락/팀 이동/공식 자책점의 한계가 있다.

시즌 보조 자료는 정규시즌 및 포스트시즌 결과·시리즈·일정을 별도 보존하며 최종 순위·동률·진출 규칙과 과거 일정 공개 상태를 추가 확인해야 한다. 선발의 사후 관측 후보, 근거, 기준 날짜 이전 선발/등판 이력을 분리한다. 당시 발표 선수·현재 경기 실제 선발은 M1 입력이 아니다.

이 자료 생성 명령 자체는 최종 모델 학습·전처리 fit·보정·성능 평가 및 최종 Test 정답 탐색을 하지 않는다. M1·M2의 별도 모델 학습 명령은 각 실행 문서를 따른다. 후속 학습에서 `sample_id`·인덱스·정답·감사 열을 입력 변수에 포함하지 않는다. 학습된 결측 대체/Encoder/Scaler는 학습 분할에서만 fit하고 Validation/Test에 그대로 적용한다.

### 포스트시즌 자료와 선수 장기 정답 보완

기존 검증판을 입력으로 재사용하는 증분 표준화·검증 경로다. 이전 원본·중간 자료·표준판·모델 자료는 덮어쓰지 않는다. 추가 원본은 이미 정규시즌 원본에서 동일 SHA-256으로 확보한 네 물리 파일의 포스트시즌 영역이다. #15 수집 실행 `4bdbd03e20634ce58135ec2311555351`의 성공 결과를 포함하면 물리 원본은 46개(88,503,479 B), 논리 대상은 48개, 대상별 판본은 50개다. 정규시즌 원본과 포스트시즌의 동일 SHA는 보존하며 `competition`과 원본 영역별 대응 행 번호로 변환 행을 구분한다. `standardize`는 선택한 수집 실행에 추가 포스트시즌 대상이 있는지 확인하여 기존 전체 표준화 또는 보완 표준화로 분기한다. 추가 대상이 포함된 경우 기존 완료 검증판을 기준으로 증분 처리하며, `--base-run`과 `--base-verification`으로 기준 판본을 명시할 수 있다. 생략할 경우 동일 원본 판본과 동일 논리 검증 해시를 가진 기존 완료 검증판에서 결정적인 순서로 기준을 선택하며, 기준 내용이 서로 다르면 명시적 선택을 요구한다. 과거 44개 대상 경로에서는 추가 대상 원본의 무결성을 확인하되 해당 논리 대상을 선택하지 않는다. 이전 완료 검증판을 기반으로 나머지 표준 테이블의 해시가 유지되는 조건을 검사한다.

| 시즌 | 실제 원본 점수 후보 | 범위                                                |
| ---- | ------------------: | --------------------------------------------------- |
| 2019 |                  12 | 네 라운드, 행별 종료 표시                           |
| 2020 |                  13 | 네 라운드, 행별 종료 표시                           |
| 2024 |                   5 | 한국시리즈만 점수 확보, 다른 라운드는 메타데이터만  |
| 2025 |                  16 | 네 라운드, 종료된 시리즈·점수·승자·시리즈 승수 근거 |

2024·2025년 경기별 `isFinal` 부재는 원문 그대로 보존하고, 완료된 시리즈와 실제 점수·승자·승수 대조를 별도 기록할 때만 유효 결과로 추가한다. 일정 CSV의 대회명 불일치 및 미일치 행은 참조 한계로 보고한다. 2021~2023년과 2024년 앞 세 라운드 점수는 미확보 상태다. 2001~2009년 포스트시즌 점수도 확보되지 않았다. 공식 순위·대진·시리즈 규칙을 확정하거나 미래 입력으로 사용하지 않는다.

| 시즌 | 완료일     | 공지일     | 공식 근거                                                                            |
| ---- | ---------- | ---------- | ------------------------------------------------------------------------------------ |
| 2023 | 2023-10-17 | 2023-10-18 | [KBO 결산 공지](https://www.koreabaseball.com/MediaNews/Notice/View.aspx?bdSe=9846)  |
| 2024 | 2024-10-01 | 2024-10-02 | [KBO 결산 공지](https://www.koreabaseball.com/MediaNews/Notice/View.aspx?bdSe=10231) |
| 2025 | 2025-10-04 | 2025-10-05 | [KBO 결산 공지](https://www.koreabaseball.com/MediaNews/Notice/View.aspx?bdSe=11708) |

정규시즌 720개 완료 경기와 PBP 게임 키 집합, 날짜 및 실제 경기별 점수 일치를 검토한 뒤 종료 근거를 새 표준판에 연결한다. 공식 종료 근거는 정답 관측 종료를 판단하기 위한 것이며, 과거 시점의 입력으로 소급하지 않는다. 2025년 투구 순번 누락은 원본에서도 확인돼 품질 차단을 유지한다. 경기 집합이 모두 있어도 선수별 기간 관측의 완전성을 보장하지 않는다.

기존 검증판과 모델 자료를 보존한 상태에서 실행한다. `--reproduce`는 표준 보완판의 동일 입력 재생성 및 같은 검증판에서 모델 자료의 재생성 비교를 추가한다.

```powershell
& .venv/Scripts/python.exe -B -X utf8 -m unittest discover -s tests -v
& .venv/Scripts/python.exe -B -X utf8 -m src.datasets.postseason_supplement --collection-run 4bdbd03e20634ce58135ec2311555351 --base-run 06ae4302ed7544d089fb0c5c8deb5719 --base-verification d936d50a5320463b8dd1c1331569ff19 --base-model-run 3c1a2e7dc6724b5ba4f887d55bad17d5 --reproduce
```

기존 CLI로 보완 표준화·검증만 별도로 실행할 수도 있다. `standardize`는 보완 대상에 대해 완료 표준판과 최종 검증판을 함께 생성한다. `validate-standard`는 보완판의 이미 생성된 검증 ID로 원본·중간 자료·DB·완료 근거를 읽기 전용 재검사한다. 새 검증 ID를 불필요하게 만들지 않는다. 기존 비보완 자료는 종전 표준화·검증 경로를 유지한다. 같은 입력의 새 보완판 비교에는 `standardize --compare-run <이전보완표준Run>` 또는 `validate-standard --compare-with <이전보완표준Run:검증ID>`를 사용한다.

```powershell
& .venv/Scripts/python.exe -B -X utf8 -m src standardize --as-of-run 4bdbd03e20634ce58135ec2311555351 --base-run 06ae4302ed7544d089fb0c5c8deb5719 --base-verification d936d50a5320463b8dd1c1331569ff19
& .venv/Scripts/python.exe -B -X utf8 -m src validate-standard --run-id <새표준Run>
```

이전 모델과 보완 모델의 비교는 `starter_pre_game`의 기존 행을 보존하며, 신규 포스트시즌 경기의 실제 참가 팀과 기준일 이전 선발 기록에서만 추가 행을 허용한다. 행의 중복·삭제·변경 또는 미래 시점 혼입은 실패로 처리한다.

포스트시즌 원본 및 완료 근거 테스트 18개와 원본 범위·CLI·선발 후보 추가 검증 테스트 20개는 `tests/validation/test_postseason_supplement.py`에 함께 유지한다. 이 보완판 인수 당시 기존 94개를 포함해 전체 회귀 테스트 132개가 통과했다. 이후 M1까지 143개, M2까지 189개가 로컬 환경에서 통과했다. 개수는 실행 당시의 상태다.

실행 결과에 표시된 새 standard run/verification과 model run을 사용한다. `quality.json`, 최종 검증 JSON의 `handoff`, 모델 `manifest.json` 및 첫 모델 실행 폴더의 `supplement-impact.json`에서 이전/새 자료의 변경 테이블·표본 수·기간별 유효 Count·분모·관측 제한을 확인한다. 고정 시간 분할은 유지하며 최종 Test는 무결성/분할 검사만 수행한다. 2023~2025년 잔여 정답과 2025년 28일 정답은 실제 생성 결과에서만 확정한다. 기존 모델 예측기 수·학습 범위·시뮬레이션의 후속 작업은 변경하지 않는다.

### 현재 인수 자료와 모델 실행 연결

| 구분                | 인수 식별자                                                        |
| ------------------- | ------------------------------------------------------------------ |
| 수집 실행           | `4bdbd03e20634ce58135ec2311555351`                                 |
| 표준 실행           | `e020dafedc8b4fd9ac6696d0e4e4a92e`                                 |
| 표준 검증           | `37b56c39caf5435e8fb2a17f993ec4df`                                 |
| 표준 DB SHA-256     | `15632b3b796ce255f4bc5038e6f680b214385beb47b9e21ebd1bb790b7962aee` |
| 모델 자료 실행      | `259b38decaac4ed68dd5a596cfbe4849`                                 |
| 모델 자료 생성기    | `1.2.0`                                                            |
| 모델 DB SHA-256     | `c8383a9d3e8930830a137d1b6791cbb26e2316dbbbdf4f0f436ceda9a43890d0` |
| 모델 자료 재현 비교 | `e10803a254a243fd922e12b6c2bdd806`                                 |

모델 자료의 인수 기준은 보완판이다. 기존 절의 이전 실행 ID는 역사적 자료를 재생성하거나 판본을 비교하기 위한 것이다. 새 모델 개발에 이전 자료를 무심코 전달하지 않는다.

| 실행 대상      | 실제 명령 모듈                                                 | 문서                                                       |
| -------------- | -------------------------------------------------------------- | ---------------------------------------------------------- |
| 경기 결과·득점 | `src.models.game_outcome` / `src.models.game_outcome.backtest` | [실행·재현](docs/standards/game_outcome_score.md#commands) |
| 타석 결과      | `src.models.plate_appearance` (`train`, `predict`, `verify`)   | [실행·재현](docs/standards/plate_appearance.md#commands)   |

모델별 문서의 실제 인수 실행 ID 및 저장 모델 경로를 사용한다. 최종 Test 및 미구현 기능의 CLI를 임의로 가정하지 않는다.

<a id="outputs"></a>

## 4. 결과 읽는 법

현재 `check`가 출력하는 JSON은 환경·설정 확인 결과다. 경기·선수의 예측 결과나 성능 보고가 아니다.

M1·M2의 실제 저장 예측 및 후속 예측·평가 결과에서는 **대상·기준 시점 → 기간·자료 범위 → 확정/예측/가정/미확정 상태 → 데이터·모델·실행 버전 → 평가 가능 범위** 순서로 확인한다. 실제 0과 미관측은 다르며, 높은 확률은 결과 보장이 아니다.

[산출물별 읽기 안내](docs/standards/project_contract.md#outputs)에 경기·타석·선수·상황·투구·시즌·평가 보고의 확인 항목을 정리했다. 스코어와 확률의 차이, 빈 선수 기록, 사후 WPA, 두 우승 해석은 [보충 사례](docs/standards/project_contract.md#interpretation)에서 필요한 항목만 펼쳐 읽는다.

출전량·Count·분모·이닝, 보정·Backtesting 등 용어는 [계약 용어](docs/standards/project_contract.md#terms)와 해당 모델 절을 따른다. 예시 안내는 실제 결과·성능 수치나 물리 스키마가 아니다.

<a id="help"></a>

## 5. 문제별 확인 경로

| 질문·상황                                  | 확인할 곳                                                                                                                                                                |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 패키지를 import할 수 없는가                | `.venv`의 Python으로 설치했는지와 [실행 명령](#commands)의 로딩 위치 출력                                                                                                |
| 기준 경로·설정 파일 오류가 나는가          | [설정 우선순위](#running), 기준 디렉터리의 존재, `kbo.toml`의 존재·읽기 권한                                                                                             |
| 자료 위치가 예상과 다른가                  | 프로젝트 기준, `KBO_PROJECT_ROOT`, 상대·절대경로 해석                                                                                                                    |
| Seed·로그·경로 설정 오류가 나는가          | 필수 항목·타입·허용 범위·파일 충돌·경로 겹침                                                                                                                             |
| 기록이 없는데 0으로 채워도 되는가          | [M3 정답 상태](docs/standards/project_contract.md#m3-state)                                                                                                              |
| 과거 라인업을 경기 전 입력으로 써도 되는가 | [공개 시점 근거](docs/standards/project_contract.md#lineup-schedule-publication)                                                                                         |
| 위치 예측을 현재 범위로 잘못 포함했는가    | [M5](docs/standards/project_contract.md#m5) · [좌표 평가](docs/standards/project_contract.md#metrics-location)                                                           |
| Test 결과를 보고 모델을 바꾸고 싶은가      | [분할·선택 경계](docs/standards/project_contract.md#split)                                                                                                               |
| 같은 설정인데 결과가 달라졌는가            | [자산·실행 추적](docs/standards/project_contract.md#reproducibility)                                                                                                     |
| 새 기능·모델·미확정 항목이 필요한가        | [범위](docs/standards/project_contract.md#scope) · [모델 수](docs/standards/project_contract.md#model-count) · [확정 절차](docs/standards/project_contract.md#decisions) |

정상 종료 코드는 0, 설정 오류와 잘못된 명령은 2, 수집 실패는 3, 저장 실패는 4다. `check` 성공은 설정 읽기·타입·범위·기본 경로 조건을 통과했다는 뜻이다. 자료 존재·쓰기 권한·수집 가능·학습 가능을 보장하지 않는다. 예상하지 못한 실행 오류를 설정 오류로 간주하거나 성공으로 처리하지 않는다.

테스트는 정상 설정·설정 오류·다른 실행 위치·공백과 한글 경로·상대와 절대경로·없는 경로·파일 충돌·저장 경로 겹침을 확인한다. 권한 오류 일부는 실제 권한 변경 대신 모의 예외로 검증한다.

읽기 전용 검증은 별도 Python 프로세스에 감사 후크를 설치한 이후 구간에서 네트워크·파일 쓰기·대표 모델 파일 열기를 차단하고 임시 영역의 전후 상태를 비교한다. Python 시작 단계 전체·모든 가능한 입출력·모든 모델 로더를 검증한 것으로 확대하지 않는다. 테스트는 임시 자료를 만들고 정리하며 기존 데이터는 사용하지 않는다.

Python은 일반 import 중 바이트코드 캐시를 만들 수 있다. 아래 모듈 확인은 `-B`를 사용하고, 설치된 CLI의 직접 실행은 `PYTHONDONTWRITEBYTECODE`를 임시 설정한다.

문서 링크 오류는 폴더 구조·파일명·대소문자를 확인한다. 실행 오류는 [증거·실패 관리](docs/planning/development_workflow.md#evidence)에 따라 작업 디렉터리·환경·명령·종료 코드·핵심 출력·생성 파일·보완·재검증을 남긴다. Secret이 포함된 출력은 공유하지 않는다.

자동 확보 가능성과 사용·재배포 허용은 별개다. 출처별 조건을 확인하며 이 문서가 소스 라이선스나 자료 재배포 권한을 새로 부여하지 않는다. SHAP 등의 설명도 실제 인과·감독 의도·선수 내부 상태를 입증하지 않는다.

<a id="maintenance"></a>

## 6. 문서 유지 기준

[공통 문서 작성 및 검토 규격](docs/planning/development_workflow.md#documentation-standard)을 사용한다. 모델 실행 문서는 10개 절의 공통 구조를 지키고 적용한 강의 원리·프로젝트 자체 확장·실제 검증·한계를 분리한다. 기준 변경은 계약 원문을 먼저 갱신하고 요약·인수 조건·관련 데이터/모델/평가 영향을 대조한다. 실제 자료·환경 확정은 근거·버전과 함께 기록한다. 문서 작성·파일 적용·설치·실행·검증 완료를 구분하며 수치·성공·완료 표시는 실제 증거가 있을 때만 추가한다.

소스의 명령·설정·경로·폴더를 변경하면 README와 테스트도 함께 대조한다. 현재 테스트는 수동 실행 절차를 제공하며 CI 자동 실행은 등록하지 않았다. 파일 존재나 미실행 테스트를 통과 근거로 쓰지 않는다.

<a id="commands"></a>

## 7. 설치·기본 확인·테스트 명령

아래 절차는 Windows AMD64·Python 3.12.15·uv 0.13.7 환경에서 확인한 설치·검증 동작을 재현하는 Windows PowerShell 명령이다. 패키지 버전 조회는 내부 문자열의 따옴표 전달 문제를 피하도록 패키지 이름을 별도 인자로 넘긴다. 소스·설정 파일을 적용한 뒤 실행한다.

첫 설치는 1번부터 진행한다. 기존 `.venv`가 있으면 자동 덮어쓰기를 막고 중단한다. 이 경우 기존 환경의 위치·Python 버전을 확인한 뒤 재사용 여부를 판단한다. 이미 설치된 환경의 재검증은 작업 디렉터리와 경로 변수를 준비한 뒤 4번부터 실행한다. 기존 환경에서 패키지나 CLI 진입점을 변경했다면 4번 전에 `uv sync --locked --extra inspection --reinstall-package kbo-insight-ai`로 프로젝트 설치를 갱신한다.

프로젝트 위치와 Python 원본 경로는 검증한 로컬 환경을 기준으로 한다. 다른 위치에서는 실제 확인한 절대경로로 바꾼다. 명령은 가상환경을 활성화하지 않고 해당 Python을 직접 호출한다.

5번 검증 구간에서는 기본 프로젝트 탐색을 확인하기 위해 `KBO_PROJECT_ROOT`를 잠시 비우고, 바이트코드 생성 방지와 한글 출력을 위해 `PYTHONDONTWRITEBYTECODE`·`PYTHONUTF8`을 임시 지정한다. 종료 시 기존 프로세스 값을 복원한다. 운영체제의 영구 환경 설정은 변경하지 않는다.

각 단계가 실패하면 중단하고 결과를 기록한다. 6번 추적 제외 확인은 실제 파일을 생성하지 않고 가상 경로를 조회한다. 마지막 Git 확인은 읽기 조회만 수행한다.

```powershell
# 1. 작업 위치·기준·도구 확인
Set-Location -LiteralPath 'D:/Programming/kbo-insight-ai'
$kboProjectRoot = (Get-Location).Path
$kboProjectPython = Join-Path $kboProjectRoot '.venv/Scripts/python.exe'
$kboPythonSource = Join-Path $env:APPDATA 'uv/python/cpython-3.12.15-windows-x86_64-none/python.exe'

Get-Location
git branch --show-current
git rev-parse HEAD
git --no-optional-locks status --short --branch
uv --version
if ($LASTEXITCODE -ne 0) { throw 'uv 확인 실패' }

if (-not (Test-Path -LiteralPath $kboPythonSource -PathType Leaf)) {
    throw '확인한 Python 원본 경로가 없습니다. 실제 설치 위치를 확인하세요.'
}
& $kboPythonSource -B -c 'import sys, platform; print(sys.version); print(sys.executable); print(platform.machine()); sys.exit(0 if sys.version_info[:3] == (3, 12, 15) else 2)'
if ($LASTEXITCODE -ne 0) { throw 'Python 3.12.15 확인 실패' }

# 2. 최초 가상환경 준비: 기존 환경은 덮어쓰지 않음
if (Test-Path -LiteralPath '.venv') {
    throw '기존 .venv가 있습니다. 환경을 확인한 뒤 재사용 여부를 판단하세요.'
}
uv venv --no-python-downloads --python $kboPythonSource .venv
if ($LASTEXITCODE -ne 0) { throw '가상환경 준비 실패' }

# 3. 프로젝트 설치
uv sync --locked --extra inspection --python $kboProjectPython
if ($LASTEXITCODE -ne 0) { throw '프로젝트 설치 실패' }

# 4. 설치된 버전·의존성 확인
uv pip list --python $kboProjectPython
if ($LASTEXITCODE -ne 0) { throw '설치 목록 확인 실패' }

uv pip check --python $kboProjectPython
if ($LASTEXITCODE -ne 0) { throw '설치 의존성 확인 실패' }

& $kboProjectPython -B -c 'import sys; from importlib.metadata import version; print(sys.version); print(sys.executable); print(version(sys.argv[1])); sys.exit(0 if sys.version_info[:3] == (3, 12, 15) else 2)' 'kbo-insight-ai'
if ($LASTEXITCODE -ne 0) { throw '설치 환경 확인 실패' }

# 5. import·진입점·경로·정상 및 오류 동작·테스트 확인
$kboPreviousRoot = [Environment]::GetEnvironmentVariable('KBO_PROJECT_ROOT', 'Process')
$kboPreviousBytecode = [Environment]::GetEnvironmentVariable('PYTHONDONTWRITEBYTECODE', 'Process')
$kboPreviousUtf8 = [Environment]::GetEnvironmentVariable('PYTHONUTF8', 'Process')

try {
    [Environment]::SetEnvironmentVariable('KBO_PROJECT_ROOT', $null, 'Process')
    [Environment]::SetEnvironmentVariable('PYTHONDONTWRITEBYTECODE', '1', 'Process')
    [Environment]::SetEnvironmentVariable('PYTHONUTF8', '1', 'Process')

    # 모듈 도움말과 기본 설정 확인
    & $kboProjectPython -B -m src --help
    Write-Output "help_exit=$LASTEXITCODE"
    if ($LASTEXITCODE -ne 0) { throw '도움말 실패' }

    & $kboProjectPython -B -m src check
    Write-Output "check_exit=$LASTEXITCODE"
    if ($LASTEXITCODE -ne 0) { throw '기본 확인 실패' }

    # 설치된 CLI 실행 파일은 직접 실행
    & (Join-Path $kboProjectRoot '.venv/Scripts/kbo-insight.exe') check
    Write-Output "installed_cli_exit=$LASTEXITCODE"
    if ($LASTEXITCODE -ne 0) { throw '설치된 CLI 실패' }

    # 저장소 밖에서 실제 import 위치·기본 경로·명시 기준 확인
    Push-Location -LiteralPath $env:TEMP
    try {
        & $kboProjectPython -B -c 'import src; print(src.__file__)'
        Write-Output "import_exit=$LASTEXITCODE"
        if ($LASTEXITCODE -ne 0) { throw '외부 실행 위치 import 실패' }

        & $kboProjectPython -B -m src check
        Write-Output "other_cwd_exit=$LASTEXITCODE"
        if ($LASTEXITCODE -ne 0) { throw '다른 실행 위치 확인 실패' }

        & $kboProjectPython -B -m src --project-root $kboProjectRoot check
        Write-Output "explicit_root_exit=$LASTEXITCODE"
        if ($LASTEXITCODE -ne 0) { throw '명시 기준 확인 실패' }
    }
    finally {
        Pop-Location
    }

    # 없는 기준은 오류로 종료해야 함. 실제 경로를 만들지 않음
    $kboMissingRoot = Join-Path $kboProjectRoot '.missing-root-probe'
    if (Test-Path -LiteralPath $kboMissingRoot) {
        throw '오류 확인용 경로가 이미 존재합니다. 상태를 먼저 확인하세요.'
    }
    & $kboProjectPython -B -m src --project-root $kboMissingRoot check
    Write-Output "missing_root_exit=$LASTEXITCODE"
    if ($LASTEXITCODE -ne 2) { throw '없는 기준의 오류 처리 확인 필요' }

    # 설정 타입·범위·공백·한글·경로 충돌·읽기 전용 동작 검증
    & $kboProjectPython -B -m unittest discover -s tests -v
    Write-Output "tests_exit=$LASTEXITCODE"
    if ($LASTEXITCODE -ne 0) { throw '테스트 실패' }
}
finally {
    [Environment]::SetEnvironmentVariable('KBO_PROJECT_ROOT', $kboPreviousRoot, 'Process')
    [Environment]::SetEnvironmentVariable('PYTHONDONTWRITEBYTECODE', $kboPreviousBytecode, 'Process')
    [Environment]::SetEnvironmentVariable('PYTHONUTF8', $kboPreviousUtf8, 'Process')
}

# 6. 기본 추적 제외 규칙 확인: 검사 경로의 파일은 생성하지 않음
$kboIgnoredPaths = @(
    '.venv/GITIGNORE_PROBE',
    '.env',
    '.env.production',
    'data/raw/GITIGNORE_PROBE',
    'data/interim/GITIGNORE_PROBE',
    'data/processed/GITIGNORE_PROBE',
    'models/GITIGNORE_PROBE',
    'results/GITIGNORE_PROBE',
    '.cache/GITIGNORE_PROBE'
)
foreach ($kboIgnoredPath in $kboIgnoredPaths) {
    git check-ignore --no-index -- $kboIgnoredPath
    Write-Output "ignore_exit=$LASTEXITCODE path=$kboIgnoredPath"
    if ($LASTEXITCODE -ne 0) { throw '추적 제외 확인 실패' }
}

# .env.example은 제외되지 않아야 하므로 예상 종료 코드는 1
git check-ignore --no-index -- .env.example
Write-Output "env_example_ignore_exit=$LASTEXITCODE"
if ($LASTEXITCODE -ne 1) { throw '.env.example 추적 규칙 확인 필요' }

# 7. 변경·추적된 파일의 공백 오류·최상위 생성 항목 확인
git --no-optional-locks status --short --branch
git diff --stat
git diff --check
Write-Output "diff_check_exit=$LASTEXITCODE"
if ($LASTEXITCODE -ne 0) { throw '추적된 변경의 공백 오류 확인 필요' }
Get-ChildItem -Force -Name
```
