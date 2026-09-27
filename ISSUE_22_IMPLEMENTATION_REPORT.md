# Issue #22 구현 보고서

## 1. 변경 요약

Issue #22 `docs(data): 예측 시점·타깃·시간 분할 계약 정의`의 문서 산출물을 완성했다.

작성한 산출물:

```text
docs/prediction_dataset_contract.md
docs/feature_catalog.md
data/README.md 삽입용 CT-2 계약 섹션
```

이번 산출물은 M1/M2/M3의 Prediction-time 계약을 고정하고, Expanded CT-2의 M4/M5가 #31/#30에서 같은 문서를 확장할 수 있도록 책임 경계를 명시한다.

GitHub Repository는 직접 수정하지 않았다.

## 2. 변경 파일

### 신규

```text
docs/prediction_dataset_contract.md
docs/feature_catalog.md
```

### 수정 대상

```text
data/README.md
```

`data/README.md`는 기존 Canonical/Derived 상세 내용을 삭제하거나 축약하지 않고 `data/processed` 설명 뒤에 CT-2 Prediction-time 계약 섹션을 추가하는 방식으로 적용한다.

## 3. 확정한 계약

### M1

- Prediction Time: Pregame
- Grain: `game_pk`
- Target:
  - `home_result ∈ {loss, tie, win}`
  - `home_runs`
  - `away_runs`
- Target 의미: Canonical observed terminal result/score
- Class Code:
  - loss=0
  - tie=1
  - win=2
- 공식 경기 종료 상태를 안다고 과장하지 않음
- Observation Quality와 Target 의미 분리

### M2

- Prediction Time: PA Start
- Grain: `(game_pk, at_bat_number)`
- Raw 실제 첫 Row에서:
  - `starting_batter`
  - `starting_pitcher`
  - `starting_stand`
  복원
- 15 source event -> 11 target class 고정
- 현재 PA Event/Post-state/Pitch Summary/종료 선수는 X 금지
- `event IS NULL`은 Target null
- unknown non-null event는 오류

### M3

- Prediction Anchor: 매주 월요일
- Historical cutoff: `source.game_date < prediction_date`
- Grain: `(player_id, role, prediction_date, horizon_id)`
- Horizon:
  - 7d
  - 28d
  - rest_of_season
- `calendar_month` 제외
- 동일 Base Key의 세 Horizon은 공통 X 사용
- Count Target 중심
- Rate는 Aggregate Count에서 계산
- 완전한 Interval + Coverage 확인 + 무출전일 때만 Count 0
- 미래 미완료/불명확은 null + censored
- `through_date`를 시즌 종료로 사용하지 않음
- 팀 remaining games와 선수 exposure 구분

### 시간 Split

```text
2023 -> train
2024 -> validation
2025 -> test
2026 -> snapshot
```

- Random split 금지
- Split 자체를 성능 튜닝 대상으로 사용하지 않음
- Test는 CT-3 최종 평가 전용
- Label interval/availability에 따른 Purge를 Split과 별도로 관리

### Feature Set

```text
baseline_v1
extended_v1
```

- `baseline_v1`: 단순 Count/Rate + Prediction Context
- `extended_v1`: 5/10/20경기 Rolling/Expanding 확장
- Same-day/Doubleheader를 임의 순서로 분할하지 않음

## 4. 기존 계약과 이번 확정 내용

기존 #22의 핵심 결정은 유지했다.

추가로 문서 수준에서 다음을 명시적으로 고정했다.

- M1 class code
- M2 11-class code
- Label Availability의 날짜 단위 보수적 의미
- Split Version
- Contract/Target/Feature Version 이름
- Cold Start
- Censoring / Purge / Exclusion 차이
- Explicit X Allowlist 원칙
- Team/Player Historical Feature semantic naming
- `baseline_v1` / `extended_v1` Allowlist
- Expanded CT-2에서 M4/M5 문서 확장 책임

## 5. 주요 설계 결정과 근거

### Strict date cutoff

정확한 same-day 이용 가능 시각이 없는 상태에서 임의 Row 순서나 `game_pk`를 이용하면 미래 결과가 과거 Feature에 섞일 수 있으므로 `source.game_date < prediction_date`를 고정했다.

### Season Snapshot 직접 사용 금지

Season Snapshot은 현재 Dataset Snapshot까지의 누적 Post-game Summary이므로 과거 Prediction Date에 직접 Join하면 미래 정보가 포함될 수 있다.

### Player Master 미래 정보 차단

`players.last_seen_*`, 전체 기간 Role은 Prediction Cutoff 이후 관측을 포함할 수 있으므로 Historical X와 M3 Population 판단에 사용하지 않는다.

### M3 Count-first

Rate를 직접 더하거나 평균하면 분모 의미가 깨질 수 있으므로 Count를 먼저 집계/예측하고 Rate는 합산 분자·분모에서 재계산한다.

### Split과 Label usability 분리

Prediction Date의 Split이 정해져 있어도 Label interval 미완료, Coverage 불명확, split boundary 침범이 있을 수 있으므로 supervised usability를 별도로 관리한다.

## 6. 확인한 누수 위험

- 현재 경기 결과를 M1 X에 포함
- Canonical credited/final 선수를 M2 PA-start 선수로 사용
- M2 현재 PA Event/Post-state/Pitch Summary 사용
- 같은 날짜 경기 결과 사용
- Season Snapshot 직접 Join
- Player Master `last_seen_*` 사용
- 미래 출장 선수로 M3 Population 생성
- 미래 PA/BF를 M3 X에 사용
- 미관측 미래를 0으로 채움
- split crossing Label 미purge
- 최신 외부 일정/라인업을 Historical 시점에 소급 적용

모두 계약/forbidden list에 반영했다.

## 7. 테스트/검증 결과

이번 작업은 문서 산출물 작성이다.

새 코드/Builder/Validator는 실행하지 않았다.

생성 문서에 대해 다음 정적 일관성 검사를 수행할 수 있도록 구성했다.

- M1/M2/M3 section 존재
- 7d/28d/rest_of_season 존재
- calendar_month 제외 명시
- same-day 차단 명시
- Train/Validation/Test/Snapshot 고정
- Censoring/Purge/Exclusion 분리
- baseline_v1/extended_v1 존재
- M4/M5 확장 책임 #31/#30 명시
- Model-ready와 모델 학습 완료 구분

저장소에 실제 반영한 뒤에는 Markdown 링크/경로와 기존 #23~#28 계약의 일치를 다시 확인해야 한다.

## 8. 미완료/주의사항

현재 생성 파일은 다운로드 가능한 산출물이며 GitHub Repository에는 직접 반영하지 않았다.

따라서 다음은 사용자가 Repository에 적용한 뒤 확인해야 한다.

1. `docs/` 디렉터리 생성 및 두 신규 문서 배치
2. `data/README.md`에 제공된 CT-2 섹션 전체 삽입
3. 실제 Git diff 확인
4. 기존 문서/Issue와 경로/용어 최종 확인
5. 필요 시 Markdown link check

M4/M5의 상세 계약은 의도적으로 #22에 미리 작성하지 않았다.

#31/#30에서 같은 두 공통 문서를 확장한다.

## 9. Issue #22 상태 판정

### 문서 설계

```text
완료
```

### Repository 실제 반영

```text
사용자 적용 전
```

따라서 Repository에 파일이 반영되고 최종 diff/문서 검증을 확인한 뒤 Issue #22를 완료 처리하는 것이 안전하다.

## 10. Commit Message 초안

```text
docs(data): 예측 데이터 계약 문서 추가

- M1~M3 예측 시점·타깃·시간 분할 계약 문서화
- Feature 가용 시점과 모델별 allowlist/forbidden 정리
- M3 multi-horizon·censoring·purge·label availability 규칙 반영
- data README에 CT-2 Prediction-time/Processed Dataset 계약 안내 추가

Resolves: #22
```

## 11. PR Gate 판정

현재 ChatGPT는 GitHub Repository를 직접 수정하지 않았다.

따라서 현재 단계에서는 PR 초안을 생성하지 않는다.

Repository 반영 후 다음을 확인해야 PR Gate를 통과할 수 있다.

- Issue #22 TODO가 문서에 전부 반영됨
- 신규 문서 경로가 실제로 존재함
- `data/README.md` 링크가 유효함
- 기존 계약과 모순 없음
- 미완료 임시 문구 없음
- 수행하지 않은 테스트를 수행했다고 기록하지 않음

PR Gate를 통과한 뒤 `pull_request_template.md` 기준으로 PR 초안을 작성한다.
