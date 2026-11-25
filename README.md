# KBO Insight AI

**경기에서 시즌까지, 예측 시점과 검증 근거를 함께 관리하는 KBO 분석 프로젝트**

KBO 공개 자료를 자동 확보해 경기·타석·선수·상황·투구를 예측하고, 경기 예측을 정규시즌·포스트시즌 전망으로 연결한다. 예측 정확도뿐 아니라 당시 정보 사용, 정답의 완전성, 확률의 신뢰도와 결과 재현성을 함께 관리한다.

[저장소](https://github.com/Ahn-SooHyun/kbo-insight-ai) · [개발 흐름](docs/planning/development_workflow.md) · [예측·평가 계약](docs/standards/project_contract.md)

> **현재 제공 범위 · 2026년 11월 11일**  
> 도움말·읽기 전용 환경 확인과 명시적인 이닝 원본 표본 수집 명령을 제공한다. 프로젝트 기반·예측 기준 구축에서 확인한 기존 동작과 이번 수집 구현의 검증은 구분한다. HF 고정 배포 파일 하나의 원본 확보·게시 해시 대조·필드/행 표본 재독을 확인했다. Naver relay는 Timeout으로 실패했으며 전체 기간 포함·사건 의미·품질 통과·학습·예측·시뮬레이션은 아직 확인하지 않았다. 개발 흐름의 [현재 확인 범위](docs/planning/development_workflow.md#status)는 그 문서에 인수된 증거만 나타낸다.

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

아래 표는 프로젝트의 구현 목표다. 현재 실행 가능한 예측 기능을 뜻하지 않는다.

| 기능                                             | 예측 결과                                     | 기준 시점                          |
| ------------------------------------------------ | --------------------------------------------- | ---------------------------------- |
| [M1 경기](docs/standards/project_contract.md#m1) | 승무패 확률, 양 팀 득점 분포·예상 득점·스코어 | 경기 시작 전                       |
| [M2 타석](docs/standards/project_contract.md#m2) | 해당 타석 결과별 확률                         | 타석 시작 전                       |
| [M3 선수](docs/standards/project_contract.md#m3) | 출전량·Count·파생 비율·시즌 전망              | 월요일 기준 7일·28일·잔여 정규시즌 |
| [M4 상황](docs/standards/project_contract.md#m4) | 승리 확률·기대득점·전후 변화·WPA·상황 중요도  | 타석 직전·종료 직후                |
| [M5 투구](docs/standards/project_contract.md#m5) | 다음 실제 공의 구종·결과·위치                 | 해당 공 투구 전                    |

M1~M5는 기능군이며, **최종 주 예측기 7개 이내**가 검증 전 기본 목표다. 독립 모델의 실제 집계와 미검증 가정은 [모델 수 계약](docs/standards/project_contract.md#model-count)을 따른다.

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

```text
kbo-insight-ai/
├── .github/
│   ├── ISSUE_TEMPLATE/
│   └── PULL_REQUEST_TEMPLATE/
├── .gitignore
├── .python-version
├── pyproject.toml
├── kbo.toml
├── README.md
├── src/
│   ├── __init__.py
│   ├── __main__.py
│   ├── settings.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── pilot.py
│   └── validation/
│       ├── __init__.py
│       └── raw_sample.py
├── tests/
│   ├── foundation/
│   │   ├── __init__.py
│   │   └── test_foundation.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── test_collection.py
│   └── validation/
│       ├── __init__.py
│       └── test_inspection.py
└── docs/
    ├── planning/
    │   └── development_workflow.md
    └── standards/
        └── project_contract.md
```

| 위치                                  | 책임                                                    |
| ------------------------------------- | ------------------------------------------------------- |
| `.github/`                            | Issue·PR 양식                                           |
| `.python-version`                     | 기본 Python 요청 버전                                   |
| `pyproject.toml`                      | 패키지 정보·Python 지원 범위·빌드 도구·설치 진입점      |
| `kbo.toml`                            | 현재 사용하는 경로·Seed·로그 설정                       |
| `.gitignore`                          | 환경·비밀값 파일·대용량 산출물·캐시 추적 제외           |
| `src/__init__.py`                     | 패키지 식별. 설정 읽기나 사업 기능을 자동 실행하지 않음 |
| `src/__main__.py`                     | 도움말·설정 확인 명령과 CLI 로그                        |
| `src/settings.py`                     | 설정 읽기·타입 및 범위 검증·경로 해석                   |
| `src/ingestion/pilot.py`              | 제한된 이닝 원본 수집·최소 구조 검사·원본/이력 보존     |
| `src/validation/raw_sample.py`        | 고정 원본의 파일 동일성·행 근거를 읽기 전용으로 검사    |
| `tests/ingestion/test_collection.py`  | 합성 응답 기반 수집·실패·보존 검증                      |
| `tests/foundation/test_foundation.py` | 설정·경로·오류·읽기 전용 동작 검증                      |
| `tests/validation/test_inspection.py` | 합성 Parquet 기반 원본 검사·실패 동작 검증              |
| `docs/planning/`                      | 개발 순서·선행 관계·증거·인수 기준                      |
| `docs/standards/`                     | 예측·평가·저장·설정의 상세 계약                         |

현재 구조는 소스·테스트·문서를 폴더로 분리하고, 패키지 내부에서는 명령 진입점과 설정 책임을 파일로 분리한다. 설정과 CLI는 패키지 루트에 두고 수집과 검사는 각각 `ingestion/`과 `validation/`에 둔다.

`pyproject.toml`·`.python-version`·`.gitignore`는 프로젝트 단위 파일이며, `kbo.toml`도 현재 설정 로더가 프로젝트 기준 위치에서 읽는다. 이를 별도 설정 폴더로 옮기려면 로더와 테스트·안내도 함께 변경해야 한다.

수집·전처리·학습·예측은 실제 기능을 구현할 때 패키지 내부에서 책임에 맞게 분리한다. 현재는 해당 기능의 빈 폴더·모듈·클래스를 미리 만들지 않는다.

### 문서의 역할

| 문서                                                 | 역할                                                  |
| ---------------------------------------------------- | ----------------------------------------------------- |
| README                                               | 프로젝트 소개·폴더 구조·설치·설정·기본 확인·결과 안내 |
| [개발 흐름](docs/planning/development_workflow.md)   | 12개 작업의 선행 관계·입력·산출물·완료 증거·인수      |
| [예측·평가 계약](docs/standards/project_contract.md) | 기능·정보 시점·정답·평가·미확정·재현성의 상세 원문    |

데이터 담당은 [M3 정답](docs/standards/project_contract.md#m3-labels)과 [확정 대기](docs/standards/project_contract.md#decisions), 모델 담당은 [평가](docs/standards/project_contract.md#evaluation), 검토 담당은 [계약 검증표](docs/standards/project_contract.md#checks)와 [인수 체크](docs/planning/development_workflow.md#handover)를 함께 읽는다.

상세 규칙은 계약 한 곳에서 관리한다. README는 요약·안내, 개발 흐름은 적용·인수 절차를 담당한다. 프롬프트·Issue 초안·타워 운영·검토 사본·과거 참고 자료와 미래 문서용 빈 폴더는 프로젝트 문서에 추가하지 않는다.

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

원본 보존·부모 자료·버전·실행 식별자와 이력의 의미는 [계약 10절](docs/standards/project_contract.md#reproducibility)을 따른다. 설정 확인은 경로 해석·검증만 수행한다. 명시적인 collect-sample 실행은 원본과 시도 이력을 생성한다. 전체 수집·변환·학습 이력 시스템은 아직 구현하지 않는다.

### 3.5 로그·Secret·Git 추적

정상 확인 결과는 허용된 설정 항목과 Python 버전만 JSON으로 stdout에 출력한다. 로그는 stderr에 수준·패키지 이름·오류 맥락을 남긴다. 파일 로그는 생성하지 않는다.

설정 오류에는 잘못된 설정값 원문이나 하위 입출력 예외 원문을 넣지 않는다. 환경 변수 전체나 Secret을 출력하지 않으며, 진단용 설정 파일에도 실제 비밀값을 넣지 않는다.

현재 `.env`를 자동으로 읽지 않으며 환경 파일은 필요하지 않다. 향후 필요한 경우 허용 파일은 `.env.example`, `.env`, `.env.production`뿐이다. `.env.example`에는 비밀값을 넣지 않고, 실제 비밀값이 있는 `.env`와 `.env.production`은 Git에서 제외한다.

`.gitignore`는 기본 데이터·모델·결과·캐시·가상환경·빌드 산출물과 환경·키 파일을 제외한다. `.env.*` 제외 규칙은 실수로 만든 변형 파일의 추적도 방지하며, `.env.example`은 예외로 허용한다. 다른 환경 파일을 추가로 허용한다는 뜻은 아니다.

자료 경로를 변경하면 해당 위치의 추적 제외 규칙도 함께 검토해야 한다. 설정을 바꾼다고 `.gitignore`가 자동 변경되지는 않는다. 저장소 밖의 절대경로를 쓰면 그 위치를 관리하는 저장소의 추적 규칙도 별도로 확인한다.

### 3.6 명시적인 원본 표본 수집

`ingestion/pilot.py`는 `load_settings`가 반환한 원본 경로·원천 주소·요청 제한을 사용한다. `collect-sample` 분기에서만 네트워크·디렉터리 생성·파일 쓰기가 발생한다. import·도움말·기존 check는 수집하지 않는다. 바이트코드 쓰기를 막을 때는 Python의 `-B`를 사용한다.

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

정상 구조/저장은 종료 0, 인자·설정 오류 2, 수집/응답 검사 실패 3, 원본·이력 저장 실패 4다. 종료 0도 #6 완료나 #7 착수 승인이 아니다. 실제 수집 전 출처 조건을 확인하고, 수집 후 원본/hash/이력과 의미 검증 근거를 인계한다. 캐시는 유일한 원본으로 사용하지 않는다.

계약의 D01·D04~D11은 실제 근거 확인 전 미확정이다. 전체 기간·학습 날짜·조건부 최종 판정·M4 산식은 이번 구현에서 결정하지 않는다. 기존 테스트는 합성 응답을 실제 표본으로 보고하지 않는다.

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

이 명령은 사용자 실행으로 HTTP 200·종료 0을 확인했고, 담당자가 저장된 17,327,447 bytes와 게시 해시를 대조했다. run_id는 96a971b890474a87946532f7c5ace4ea다. 실제 HTTP 요청 2회·주소 전송 1회였다. Naver 실패와 남은 핵심/조건부 자료 제한은 별도로 유지한다.

### 3.8 확보한 HF 원본의 읽기 전용 검사

`inspect-sample --run-id 실행ID`는 원본·수집 이력의 연결과 고정 파일 해시를 다시 확인한 뒤 실제 Arrow 필드/타입/null 수, 파일 메타데이터, 원본 행 번호가 붙은 표본 사례를 JSON으로 출력한다. 새 HTTP 요청·원본/이력 변경·디렉터리 생성을 하지 않는다. `quality_status=not_assessed`를 유지하며 전체 품질 검사의 성공을 반환하지 않는다. 검사 실패는 종료 코드 5다.

검사에는 선택 설치 `inspection`의 `pyarrow==26.0.1`을 사용한다. 현재 개발 환경의 설치 버전·선택 의존성 선언·uv.lock에서 같은 버전을 확인했다. 이 선언과 잠금 파일은 이미 존재한 것을 재사용하며 이번 검사 연결에서 덮어쓰지 않았다. 재현 환경은 잠금 파일의 inspection extra로 준비한다. 기본 수집·도움말·check는 PyArrow를 불러오지 않는다.

```powershell
& "D:/Programming/kbo-insight-ai/.venv/Scripts/python.exe" -X utf8 -B -m src --project-root "D:/Programming/kbo-insight-ai" inspect-sample --run-id "96a971b890474a87946532f7c5ace4ea"
```

파일의 행 수/날짜/경기 ID 수는 읽은 파일의 관측값이며 공식 일정 대비 포함률·시즌 완료·전체 품질 통과가 아니다. 같은 타석 키에서 참가자 ID가 달라지는 사례는 변화 후보로 기록하며, 원문 교체 역할·공식 결과 귀속·미완료 타석의 주루 종류는 독립 원천으로 확인해야 한다. 게시 명세만으로 실제 행을 모두 최종 참가자라고 가정하지 않는다. 원본 좌표와 null을 수정하거나 0으로 채우지 않는다.

<a id="outputs"></a>

## 4. 결과 읽는 법

현재 `check`가 출력하는 JSON은 환경·설정 확인 결과다. 경기·선수의 예측 결과나 성능 보고가 아니다.

후속 예측·평가 결과에서는 **대상·기준 시점 → 기간·자료 범위 → 확정/예측/가정/미확정 상태 → 데이터·모델·실행 버전 → 평가 가능 범위** 순서로 확인한다. 실제 0과 미관측은 다르며, 높은 확률은 결과 보장이 아니다.

[산출물별 읽기 안내](docs/standards/project_contract.md#outputs)에 경기·타석·선수·상황·투구·시즌·평가 보고의 확인 항목을 정리했다. 스코어와 확률의 차이, 빈 선수 기록, 사후 WPA, 두 우승 해석은 [보충 사례](docs/standards/project_contract.md#interpretation)에서 필요한 항목만 펼쳐 읽는다.

출전량·Count·분모·이닝, 보정·Backtesting 등 용어는 [계약 용어](docs/standards/project_contract.md#terms)와 해당 모델 절을 따른다. 예시 안내는 실제 결과·성능 수치나 물리 스키마가 아니다.

<a id="help"></a>

## 5. 문제별 확인 경로

| 질문·상황                                     | 확인할 곳                                                                                                                                                                |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 패키지를 import할 수 없는가                   | `.venv`의 Python으로 설치했는지와 [실행 명령](#commands)의 로딩 위치 출력                                                                                                |
| 기준 경로·설정 파일 오류가 나는가             | [설정 우선순위](#running), 기준 디렉터리의 존재, `kbo.toml`의 존재·읽기 권한                                                                                             |
| 자료 위치가 예상과 다른가                     | 프로젝트 기준, `KBO_PROJECT_ROOT`, 상대·절대경로 해석                                                                                                                    |
| Seed·로그·경로 설정 오류가 나는가             | 필수 항목·타입·허용 범위·파일 충돌·경로 겹침                                                                                                                             |
| 기록이 없는데 0으로 채워도 되는가             | [M3 정답 상태](docs/standards/project_contract.md#m3-state)                                                                                                              |
| 과거 라인업을 경기 전 입력으로 써도 되는가    | [공개 시점 근거](docs/standards/project_contract.md#d04)                                                                                                                 |
| 좌표가 없거나 실제 구종으로 위치를 평가했는가 | [M5](docs/standards/project_contract.md#m5) · [좌표 평가](docs/standards/project_contract.md#metrics-location)                                                           |
| Test 결과를 보고 모델을 바꾸고 싶은가         | [분할·선택 경계](docs/standards/project_contract.md#split)                                                                                                               |
| 같은 설정인데 결과가 달라졌는가               | [자산·실행 추적](docs/standards/project_contract.md#reproducibility)                                                                                                     |
| 새 기능·모델·미확정 항목이 필요한가           | [범위](docs/standards/project_contract.md#scope) · [모델 수](docs/standards/project_contract.md#model-count) · [확정 절차](docs/standards/project_contract.md#decisions) |

정상 종료 코드는 0, 설정 오류와 잘못된 명령은 2, 수집 실패는 3, 저장 실패는 4다. `check` 성공은 설정 읽기·타입·범위·기본 경로 조건을 통과했다는 뜻이다. 자료 존재·쓰기 권한·수집 가능·학습 가능을 보장하지 않는다. 예상하지 못한 실행 오류를 설정 오류로 간주하거나 성공으로 처리하지 않는다.

테스트는 정상 설정·설정 오류·다른 실행 위치·공백과 한글 경로·상대와 절대경로·없는 경로·파일 충돌·저장 경로 겹침을 확인한다. 권한 오류 일부는 실제 권한 변경 대신 모의 예외로 검증한다.

읽기 전용 검증은 별도 Python 프로세스에 감사 후크를 설치한 이후 구간에서 네트워크·파일 쓰기·대표 모델 파일 열기를 차단하고 임시 영역의 전후 상태를 비교한다. Python 시작 단계 전체·모든 가능한 입출력·모든 모델 로더를 검증한 것으로 확대하지 않는다. 테스트는 임시 자료를 만들고 정리하며 기존 데이터는 사용하지 않는다.

Python은 일반 import 중 바이트코드 캐시를 만들 수 있다. 아래 모듈 확인은 `-B`를 사용하고, 설치된 CLI의 직접 실행은 `PYTHONDONTWRITEBYTECODE`를 임시 설정한다.

문서 링크 오류는 폴더 구조·파일명·대소문자를 확인한다. 실행 오류는 [증거·실패 관리](docs/planning/development_workflow.md#evidence)에 따라 작업 디렉터리·환경·명령·종료 코드·핵심 출력·생성 파일·보완·재검증을 남긴다. Secret이 포함된 출력은 공유하지 않는다.

자동 확보 가능성과 사용·재배포 허용은 별개다. 출처별 조건을 확인하며 이 문서가 소스 라이선스나 자료 재배포 권한을 새로 부여하지 않는다. SHAP 등의 설명도 실제 인과·감독 의도·선수 내부 상태를 입증하지 않는다.

<a id="maintenance"></a>

## 6. 문서 유지 기준

기준 변경은 계약 원문을 먼저 갱신하고 요약·인수 조건·관련 데이터/모델/평가 영향을 대조한다. 실제 자료·환경 확정은 근거·버전과 함께 기록한다. 문서 작성·파일 적용·설치·실행·검증 완료를 구분하며 수치·성공·완료 표시는 실제 증거가 있을 때만 추가한다.

소스의 명령·설정·경로·폴더를 변경하면 README와 테스트도 함께 대조한다. 현재 테스트는 수동 실행 절차를 제공하며 CI 자동 실행은 등록하지 않았다. 파일 존재나 미실행 테스트를 통과 근거로 쓰지 않는다.

<a id="commands"></a>

## 7. 설치·기본 확인·테스트 명령

아래 절차는 Windows AMD64·Python 3.12.15·uv 0.13.7 환경에서 확인한 설치·검증 동작을 재현하는 Windows PowerShell 명령이다. 패키지 버전 조회는 내부 문자열의 따옴표 전달 문제를 피하도록 패키지 이름을 별도 인자로 넘긴다. 소스·설정 파일을 적용한 뒤 실행한다.

첫 설치는 1번부터 진행한다. 기존 `.venv`가 있으면 자동 덮어쓰기를 막고 중단한다. 이 경우 기존 환경의 위치·Python 버전을 확인한 뒤 재사용 여부를 판단한다. 이미 설치된 환경의 재검증은 작업 디렉터리와 경로 변수를 준비한 뒤 4번부터 실행한다. 기존 환경에서 패키지나 CLI 진입점을 변경했다면 4번 전에 `uv sync --locked --extra inspection --reinstall-package kbo-insight-ai`로 프로젝트 설치를 갱신한다.

프로젝트 위치와 Python 원본 경로는 확인한 사용자 환경을 기준으로 한다. 다른 위치에서는 실제 확인한 절대경로로 바꾼다. 명령은 가상환경을 활성화하지 않고 해당 Python을 직접 호출한다.

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
    $kboMissingRoot = Join-Path $kboProjectRoot '.ct1-missing-root-probe'
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
    '.venv/CT1_IGNORE_PROBE',
    '.env',
    '.env.production',
    'data/raw/CT1_IGNORE_PROBE',
    'data/interim/CT1_IGNORE_PROBE',
    'data/processed/CT1_IGNORE_PROBE',
    'models/CT1_IGNORE_PROBE',
    'results/CT1_IGNORE_PROBE',
    '.cache/CT1_IGNORE_PROBE'
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
