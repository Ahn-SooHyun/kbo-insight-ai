# KBO Insight AI

**경기에서 시즌까지, 예측 시점과 검증 근거를 함께 관리하는 KBO 분석 프로젝트**

KBO 공개 자료를 자동 확보해 경기·타석·선수·상황·투구를 예측하고, 경기 예측을 정규시즌·포스트시즌 전망으로 연결한다. 예측 정확도뿐 아니라 당시 정보 사용, 정답의 완전성, 확률의 신뢰도와 결과 재현성을 함께 관리한다.

[저장소](https://github.com/Ahn-SooHyun/kbo-insight-ai) · [개발 흐름](docs/planning/development_workflow.md) · [예측·평가 계약](docs/standards/project_contract.md)

> **현재 제공 범위 · 2026년 11월 9일**  
> 프로젝트의 개발·예측·평가 기준 문서다. 검증된 실행 명령·실제 예측·성능 결과는 이 묶음에 포함돼 있지 않다. 실제 진행 상태는 [인수된 증거 범위](docs/planning/development_workflow.md#status)로 구분한다.

| 목적                | 읽을 곳                                                                                                                           |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| 프로젝트 이해       | [기능 지도](#features) → [문서 구조](#documents)                                                                                  |
| 작업 시작·인수      | [선행 관계](docs/planning/development_workflow.md#dependencies) → [작업별 기준](docs/planning/development_workflow.md#foundation) |
| 데이터·모델 설계    | [정보 경계](docs/standards/project_contract.md#information) → [담당 모델](docs/standards/project_contract.md#models)              |
| 결과 해석·문제 해결 | [결과 읽는 법](#outputs) → [확인 경로](#help)                                                                                     |
| 실행 가능 여부 확인 | [실행 준비](#running)                                                                                                             |

<a id="features"></a>

## 1. 기능 지도

### 핵심 예측

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

## 2. 문서 구조와 역할

```text
README.md
docs/
├─ planning/
│  └─ development_workflow.md
└─ standards/
   └─ project_contract.md
```

| 문서                                                 | 역할                                               |
| ---------------------------------------------------- | -------------------------------------------------- |
| README                                               | 프로젝트 소개·문서 진입점·실행 준비·결과 안내      |
| [개발 흐름](docs/planning/development_workflow.md)   | 12개 작업의 선행 관계·입력·산출물·완료 증거·인수   |
| [예측·평가 계약](docs/standards/project_contract.md) | 기능·정보 시점·정답·평가·미확정·재현성의 상세 원문 |

데이터 담당은 [M3 정답](docs/standards/project_contract.md#m3-labels)과 [확정 대기](docs/standards/project_contract.md#decisions), 모델 담당은 [평가](docs/standards/project_contract.md#evaluation), 검토 담당은 [계약 검증표](docs/standards/project_contract.md#checks)와 [인수 체크](docs/planning/development_workflow.md#handover)를 함께 읽는다.

상세 규칙은 계약 한 곳에서 관리한다. README는 요약·안내, 개발 흐름은 적용·인수 절차를 담당한다. 프롬프트·Issue 초안·타워 운영·검토 사본·과거 참고 자료와 미래 문서용 빈 폴더는 이 묶음에 넣지 않는다.

<a id="running"></a>

## 3. 실행 준비

현재는 실제 환경·코드·데이터를 대신하는 배포물이 아니다. 설치 도구·버전·명령을 추정해 안내하지 않는다. 무엇이 확인됐고 무엇을 인수해야 하는지는 [현재 확인 범위](docs/planning/development_workflow.md#status)에 모아 관리한다.

실행 안내는 실제 근거가 갖춰지면 다음 순서로 제공한다.

**사전 조건 → 설치 → 기본 확인·테스트 → 설정 → 단계별 실행 → 결과 위치 → 오류 대응**

기반의 설치·import·경로·설정·부작용 검증은 [작업 2](docs/planning/development_workflow.md#task-2), 저장 원본·모델로 재실행하는 검증은 [작업 12](docs/planning/development_workflow.md#task-12)의 완료 기준을 따른다.

<a id="outputs"></a>

## 4. 결과 읽는 법

결과에서는 **대상·기준 시점 → 기간·자료 범위 → 확정/예측/가정/미확정 상태 → 데이터·모델·실행 버전 → 평가 가능 범위** 순서로 확인한다. 실제 0과 미관측은 다르며, 높은 확률은 결과 보장이 아니다.

[산출물별 읽기 안내](docs/standards/project_contract.md#outputs)에 경기·타석·선수·상황·투구·시즌·평가 보고의 확인 항목을 정리했다. 스코어와 확률의 차이, 빈 선수 기록, 사후 WPA, 두 우승 해석은 [보충 사례](docs/standards/project_contract.md#interpretation)에서 필요한 항목만 펼쳐 읽는다.

출전량·Count·분모·이닝, 보정·Backtesting 등 용어는 [계약 용어](docs/standards/project_contract.md#terms)와 해당 모델 절을 따른다. 예시 안내는 실제 결과·성능 수치나 물리 스키마가 아니다.

<a id="help"></a>

## 5. 문제별 확인 경로

| 질문·상황                                     | 확인할 원문                                                                                                                                                              |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 기록이 없는데 0으로 채워도 되는가             | [M3 정답 상태](docs/standards/project_contract.md#m3-state)                                                                                                              |
| 과거 라인업을 경기 전 입력으로 써도 되는가    | [공개 시점 근거](docs/standards/project_contract.md#d04)                                                                                                                 |
| 좌표가 없거나 실제 구종으로 위치를 평가했는가 | [M5](docs/standards/project_contract.md#m5) · [좌표 평가](docs/standards/project_contract.md#metrics-location)                                                           |
| Test 결과를 보고 모델을 바꾸고 싶은가         | [분할·선택 경계](docs/standards/project_contract.md#split)                                                                                                               |
| 같은 설정인데 결과가 달라졌는가               | [자산·실행 추적](docs/standards/project_contract.md#reproducibility)                                                                                                     |
| 새 기능·모델·미확정 항목이 필요한가           | [범위](docs/standards/project_contract.md#scope) · [모델 수](docs/standards/project_contract.md#model-count) · [확정 절차](docs/standards/project_contract.md#decisions) |

문서 링크 오류는 위 폴더 구조·파일명·대소문자를 확인한다. 실제 실행 오류는 [증거·실패 관리](docs/planning/development_workflow.md#evidence)에 따라 증상·원인·영향·보완·재검증을 남긴다. 확인되지 않은 명령·오류 코드·로그 경로를 만들지 않는다.

자동 확보 가능성과 사용·재배포 허용은 별개다. 출처별 조건을 확인하며 이 문서가 소스 라이선스나 자료 재배포 권한을 새로 부여하지 않는다. SHAP 등의 설명도 실제 인과·감독 의도·선수 내부 상태를 입증하지 않는다.

<a id="maintenance"></a>

## 6. 문서 유지 기준

기준 변경은 계약 원문을 먼저 갱신하고 요약·인수 조건·관련 데이터/모델/평가 영향을 대조한다. 실제 자료·환경 확정은 근거·버전과 함께 기록한다. 문서 작성·적용·실행·검증 완료를 구분하며 수치·성공·완료 표시는 실제 증거가 있을 때만 추가한다.
