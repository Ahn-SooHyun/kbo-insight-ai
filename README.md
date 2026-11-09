# KBO Insight AI

**경기에서 시즌까지, 예측 시점과 검증 근거를 함께 관리하는 KBO 분석 프로젝트**

KBO 공개 자료를 자동 확보해 경기·타석·선수·상황·투구를 예측하고, 경기 예측을 정규시즌·포스트시즌 전망으로 연결한다. 예측 정확도뿐 아니라 당시 정보 사용, 정답의 완전성, 확률의 신뢰도와 결과 재현성을 함께 관리한다.

[저장소](https://github.com/Ahn-SooHyun/kbo-insight-ai) · [개발 흐름](docs/planning/development_workflow.md) · [예측·평가 계약](docs/standards/project_contract.md)

> **현재 제공 범위 · 2026년 11월 10일**  
> 개발·예측·평가 기준 문서와 최소 Python 개발 기반을 제공한다. 현재 소스에 정의된 실행 기능은 도움말과 읽기 전용 환경·설정 확인이다. 코드·설정·테스트 파일은 확인했으나 설치·import·진입점·테스트의 실행 결과는 아직 확인하지 않았다. 실제 수집·학습·예측·시뮬레이션과 성능 결과는 포함하지 않는다. 개발 흐름의 [현재 확인 범위](docs/planning/development_workflow.md#status)는 해당 문서에 인수된 증거의 범위를 나타낸다.

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
│   └── kbo_insight_ai/
│       ├── __init__.py
│       ├── __main__.py
│       └── settings.py
├── tests/
│   └── test_foundation.py
└── docs/
    ├── planning/
    │   └── development_workflow.md
    └── standards/
        └── project_contract.md
```

| 위치                             | 책임                                                    |
| -------------------------------- | ------------------------------------------------------- |
| `.github/`                       | Issue·PR 양식                                           |
| `.python-version`                | 기본 Python 요청 버전                                   |
| `pyproject.toml`                 | 패키지 정보·Python 지원 범위·빌드 도구·설치 진입점      |
| `kbo.toml`                       | 현재 사용하는 경로·Seed·로그 설정                       |
| `.gitignore`                     | 환경·비밀값 파일·대용량 산출물·캐시 추적 제외           |
| `src/kbo_insight_ai/__init__.py` | 패키지 식별. 설정 읽기나 사업 기능을 자동 실행하지 않음 |
| `src/kbo_insight_ai/__main__.py` | 도움말·설정 확인 명령과 CLI 로그                        |
| `src/kbo_insight_ai/settings.py` | 설정 읽기·타입 및 범위 검증·경로 해석                   |
| `tests/test_foundation.py`       | 설정·경로·오류·읽기 전용 동작 검증                      |
| `docs/planning/`                 | 개발 순서·선행 관계·증거·인수 기준                      |
| `docs/standards/`                | 예측·평가·저장·설정의 상세 계약                         |

현재 구조는 소스·테스트·문서를 폴더로 분리하고, 패키지 내부에서는 명령 진입점과 설정 책임을 파일로 분리한다. 패키지의 Python 파일이 3개인 현재 단계에서는 설정이나 CLI를 별도 하위 패키지로 더 나누지 않는다.

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

| 항목                     | 선정 내용                             |
| ------------------------ | ------------------------------------- |
| 확인된 실행 환경         | Windows AMD64                         |
| Python                   | 3.12.15                               |
| 패키지가 허용하는 Python | 3.12 계열: `>=3.12,<3.13`             |
| 환경·패키지 관리         | uv 0.13.7                             |
| 가상환경                 | 프로젝트의 `.venv`                    |
| 개발 설치                | 편집 가능 설치                        |
| 빌드 도구                | `uv_build==0.13.7`                    |
| 실행 의존성              | 없음. Python 표준 라이브러리 사용     |
| 테스트 의존성            | 없음. 표준 라이브러리 `unittest` 사용 |
| 설치 이름 / import 이름  | `kbo-insight-ai` / `kbo_insight_ai`   |
| 설치되는 CLI 이름        | `kbo-insight`                         |

Python 3.12.15와 uv 0.13.7의 버전·실행은 사용자 환경에서 확인했다. 프로젝트 설치·import·기본 명령·테스트의 성공은 아직 확인하지 않았다. 다른 OS와 다른 Python 3.12 패치 버전도 별도의 실행 검증이 필요하다.

실행 의존성이 없으므로 현재 별도 의존성 목록이나 lock 파일은 두지 않는다. Python 기본 요청과 빌드 도구는 각각 `.python-version`과 `pyproject.toml`에 고정한다. 향후 실행 의존성을 추가할 때는 재현 가능한 잠금 방식도 함께 정한다.

설치 과정에서 uv는 호환되는 내장 빌드 도구 또는 고정된 빌드 패키지를 사용한다. [uv 빌드 문서](https://docs.astral.sh/uv/concepts/build-backend/)에서 패키지 구조와 빌드 방식을 확인할 수 있다.

환경 준비·설치는 가상환경·설치 메타데이터·도구 캐시를 생성할 수 있고, 패키지 확보를 위해 네트워크를 사용할 수 있다. 아래의 읽기 전용 동작 검증과 구분한다.

### 3.2 설정 파일과 우선순위

설정은 프로젝트 기준 디렉터리의 `kbo.toml` 한 파일에서 읽는다. 다음 8개 항목이 모두 필수이며, 누락하거나 지원하지 않는 항목을 추가하면 오류다.

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

표의 값은 제공 파일에 적힌 값이며, 항목 누락 시 적용되는 대체 기본값은 아니다. 로그 수준은 대문자만 허용한다. Seed는 후속 실행에서 재사용할 설정값으로 보관하며 현재 난수 생성이나 모델 학습을 실행하지 않는다.

프로젝트 기준 선택 순서는 다음과 같다.

1. CLI의 `--project-root` 또는 Python 함수에 명시한 기준 경로
2. 프로세스 환경 변수 `KBO_PROJECT_ROOT`
3. 편집 가능 설치에서 확인되는 `src/kbo_insight_ai`의 프로젝트 위치

선택된 기준이 잘못되면 낮은 우선순위로 자동 대체하지 않고 오류를 반환한다. CLI의 기준 옵션은 `check` 앞에 둔다. 환경 변수는 기준 경로만 선택하며 Seed·로그·개별 자료 경로는 `kbo.toml`에서 읽는다.

명시하는 프로젝트 기준은 존재하는 절대 디렉터리여야 한다. 일반 설치에서 소스 위치를 찾을 수 없으면 프로젝트 기준과 그 안의 `kbo.toml`을 별도로 준비해 지정해야 한다.

### 3.3 경로 해석과 검증

상대 자료 경로는 프로젝트 기준 디렉터리에 붙이고, 절대 자료 경로는 해당 위치를 사용한다. 현재 작업 디렉터리는 자료 위치를 결정하지 않는다. 경로의 공백과 한글은 그대로 처리한다.

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

`models/`는 모델 소스 코드가 아닌 저장 자산의 위치다. 실행 소스는 `src/kbo_insight_ai/` 아래에 둔다.

원본 보존·부모 자료·버전·실행 식별자와 이력의 의미는 [계약 10절](docs/standards/project_contract.md#reproducibility)을 따른다. 현재 기반은 경로를 해석하고 검증한다. 디렉터리 생성·원본 쓰기·수집 및 전체 실행 이력 시스템은 구현하지 않는다.

### 3.5 로그·Secret·Git 추적

정상 확인 결과는 허용된 설정 항목과 Python 버전만 JSON으로 stdout에 출력한다. 로그는 stderr에 수준·패키지 이름·오류 맥락을 남긴다. 파일 로그는 생성하지 않는다.

설정 오류에는 잘못된 설정값 원문이나 하위 입출력 예외 원문을 넣지 않는다. 환경 변수 전체나 Secret을 출력하지 않으며, 진단용 설정 파일에도 실제 비밀값을 넣지 않는다.

현재 `.env`를 자동으로 읽지 않으며 환경 파일은 필요하지 않다. 향후 필요한 경우 허용 파일은 `.env.example`, `.env`, `.env.production`뿐이다. `.env.example`에는 비밀값을 넣지 않고, 실제 비밀값이 있는 `.env`와 `.env.production`은 Git에서 제외한다.

`.gitignore`는 기본 데이터·모델·결과·캐시·가상환경·빌드 산출물과 환경·키 파일을 제외한다. `.env.*` 제외 규칙은 실수로 만든 변형 파일의 추적도 방지하며, `.env.example`은 예외로 허용한다. 다른 환경 파일을 추가로 허용한다는 뜻은 아니다.

자료 경로를 변경하면 해당 위치의 추적 제외 규칙도 함께 검토해야 한다. 설정을 바꾼다고 `.gitignore`가 자동 변경되지는 않는다. 저장소 밖의 절대경로를 쓰면 그 위치를 관리하는 저장소의 추적 규칙도 별도로 확인한다.

### 3.6 후속 수집 코드 연결

CT-2의 실제 수집 구현은 `src/kbo_insight_ai/` 아래에 추가하고 `settings.py`의 `load_settings`를 재사용한다. 수집 모듈은 실제 구현 시 만들며 import나 `check`에서 자동 실행하지 않는다.

API URL·Timeout·Retry 등 공통 설정이 필요해지면 `kbo.toml`과 `settings.py`의 타입·검증·테스트를 함께 확장한다. Secret은 공개 설정 파일에 넣지 않는다. 현재는 수집기·재시도 로직·API용 빈 설정을 구현하지 않는다.

수집·변환 이력과 원본의 별도 버전 저장은 데이터 구축에서, 학습·예측·평가 자산의 연결은 해당 후속 단계에서 구현한다. 실제 필드·수집 기간·조건부 기능 판정·학습 날짜는 [확정 책임](docs/standards/project_contract.md#decisions)을 따른다.

기반의 설치·import·경로·설정·부작용 검증은 [개발 기반 기준](docs/planning/development_workflow.md#task-2), 저장 원본·모델로 재실행하는 검증은 [최종 재실행 기준](docs/planning/development_workflow.md#task-12)을 따른다.

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

정상 종료 코드는 0, 설정 오류와 잘못된 명령의 종료 코드는 2다. `check` 성공은 설정 읽기·타입·범위·기본 경로 조건을 통과했다는 뜻이다. 자료 존재·쓰기 권한·수집 가능·학습 가능을 보장하지 않는다. 예상하지 못한 실행 오류를 설정 오류로 간주하거나 성공으로 처리하지 않는다.

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

아래 절차는 현재 코드에 맞춘 Windows PowerShell 명령이며 실제 성공 결과는 아직 확인하지 않았다. 소스·설정 파일을 적용한 뒤 실행한다.

첫 설치는 1번부터 진행한다. 기존 `.venv`가 있으면 자동 덮어쓰기를 막고 중단한다. 이 경우 기존 환경의 위치·Python 버전을 확인한 뒤 재사용 여부를 판단한다. 이미 설치된 환경의 재검증은 작업 디렉터리와 경로 변수를 준비한 뒤 4번부터 실행한다.

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
uv pip install --python $kboProjectPython --no-deps --editable .
if ($LASTEXITCODE -ne 0) { throw '프로젝트 설치 실패' }

# 4. 설치된 버전·의존성 확인
uv pip list --python $kboProjectPython
if ($LASTEXITCODE -ne 0) { throw '설치 목록 확인 실패' }

uv pip check --python $kboProjectPython
if ($LASTEXITCODE -ne 0) { throw '설치 의존성 확인 실패' }

& $kboProjectPython -B -c 'import sys; from importlib.metadata import version; print(sys.version); print(sys.executable); print(version("kbo-insight-ai")); sys.exit(0 if sys.version_info[:3] == (3, 12, 15) else 2)'
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
    & $kboProjectPython -B -m kbo_insight_ai --help
    Write-Output "help_exit=$LASTEXITCODE"
    if ($LASTEXITCODE -ne 0) { throw '도움말 실패' }

    & $kboProjectPython -B -m kbo_insight_ai check
    Write-Output "check_exit=$LASTEXITCODE"
    if ($LASTEXITCODE -ne 0) { throw '기본 확인 실패' }

    # 설치된 CLI 실행 파일은 직접 실행
    & (Join-Path $kboProjectRoot '.venv/Scripts/kbo-insight.exe') check
    Write-Output "installed_cli_exit=$LASTEXITCODE"
    if ($LASTEXITCODE -ne 0) { throw '설치된 CLI 실패' }

    # 저장소 밖에서 실제 import 위치·기본 경로·명시 기준 확인
    Push-Location -LiteralPath $env:TEMP
    try {
        & $kboProjectPython -B -c 'import kbo_insight_ai; print(kbo_insight_ai.__file__)'
        Write-Output "import_exit=$LASTEXITCODE"
        if ($LASTEXITCODE -ne 0) { throw '외부 실행 위치 import 실패' }

        & $kboProjectPython -B -m kbo_insight_ai check
        Write-Output "other_cwd_exit=$LASTEXITCODE"
        if ($LASTEXITCODE -ne 0) { throw '다른 실행 위치 확인 실패' }

        & $kboProjectPython -B -m kbo_insight_ai --project-root $kboProjectRoot check
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
    & $kboProjectPython -B -m kbo_insight_ai --project-root $kboMissingRoot check
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
