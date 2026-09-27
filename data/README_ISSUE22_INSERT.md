# data/README.md — Issue #22 삽입 섹션

> 적용 방법: 현재 `data/README.md`의 `### data/processed/` 설명 뒤, `### data/external/` 앞에 아래 **`#### Prediction-time Feature / Processed Dataset 계약`부터 끝까지**를 그대로 삽입한다.
>
> 이 파일은 기존 `data/README.md` 전체를 대체하지 않는다. 기존 Canonical/Derived 상세 설명을 삭제하지 않고 아래 섹션만 추가한다.

---

#### Prediction-time Feature / Processed Dataset 계약

Canonical Derived Layer는 관측 사실을 보존하는 Post-event/Post-game Fact Layer이며, 그 자체가 Prediction-time Feature를 의미하지 않는다.

CT-2에서는 다음 두 문서를 Prediction Dataset의 직접 Source of Truth로 사용한다.

```text
docs/prediction_dataset_contract.md
docs/feature_catalog.md
```

역할은 다음과 같다.

- `docs/prediction_dataset_contract.md`
  - Prediction Timestamp
  - Grain
  - Target
  - Population
  - Horizon
  - Label Availability
  - Censoring / Purge
  - 시간 Split
  - Schema / Manifest / Provenance
  - CT-2 / CT-3 경계
- `docs/feature_catalog.md`
  - Source Column 의미
  - Prediction-time Availability
  - Historical Feature 산식
  - `baseline_v1` / `extended_v1`
  - 모델별 X allowlist
  - 모델별 forbidden list
  - Null / Cold Start 정책

##### 공통 Historical Cutoff

M1/M2/M3의 Historical Feature는 기본적으로 다음 조건을 사용한다.

```text
source.game_date < prediction_date
```

현재 데이터에서 같은 날짜 경기의 정확한 이용 가능 순서를 보장할 timestamp가 없으므로 `game_pk`, Row 순서, 파일 순서를 이용해 Same-day/Doubleheader의 앞 경기를 과거 정보로 간주하지 않는다.

##### 시간 Split

고정 Split Version:

```text
season_split_v1
```

기본 분할:

```text
2023 -> train
2024 -> validation
2025 -> test
2026 -> snapshot
```

Random Split은 사용하지 않는다.

Split은 모델 성능을 높이기 위한 Hyperparameter가 아니다.

Test는 CT-3의 최종 평가 전용이며 Feature/모델/Hyperparameter 선택에 사용하지 않는다.

2026은 현재 Dataset Snapshot의 진행 시즌으로 취급하며 완료 시즌으로 간주하지 않는다.

##### M1

Prediction Time:

```text
pregame
```

Grain:

```text
game_pk
```

Target:

```text
home_result ∈ {loss, tie, win}
home_runs
away_runs
```

Target Source:

```text
games.final_home_score
games.final_away_score
```

현재 Final Score의 의미는 KBO 공식 최종 결과가 아니라 Canonical **observed terminal score**다.

무승부는 독립 Class로 유지한다.

현재 경기 Final Score, 승패, 경기 전체 PA/Pitch Count, 실제 출장 선수, 실제 라인업, 실제 교체 결과 등 경기 시작 후 알 수 있는 정보는 Pregame X에 포함하지 않는다.

##### M2

Prediction Time:

```text
PA start
```

Grain:

```text
(game_pk, at_bat_number)
```

PA 시작 선수는 Canonical credited/final 선수 Column을 그대로 사용하지 않고 동일 Raw snapshot의 실제 첫 Row에서 복원한다.

```text
starting_batter
starting_pitcher
starting_stand
```

Target 11-class:

```text
1B
2B
3B
HR
BB
HBP
SO
OUT
ROE
FC
CI
```

Mapping:

```text
single -> 1B
double -> 2B
triple -> 3B
home_run -> HR
walk -> BB
hit_by_pitch -> HBP
strikeout -> SO
field_out -> OUT
double_play -> OUT
triple_play -> OUT
sac_bunt -> OUT
sac_fly -> OUT
field_error -> ROE
fielders_choice -> FC
catcher_interference -> CI
```

`event IS NULL`인 미완료 PA는 Target `null`로 분리한다.

알 수 없는 non-null Event를 자동 `OTHER` 또는 `OUT`으로 병합하지 않는다.

현재 PA의 Event, Post-state, 전체 Pitch Summary, 종료 선수, 사후 교체 여부는 PA-start X에 포함하지 않는다.

##### M3

Prediction Anchor:

```text
매주 월요일 prediction_date = t
```

Historical X:

```text
source.game_date < t
```

Grain:

```text
(player_id, role, prediction_date, horizon_id)
```

Horizon:

```text
7d              = [t, t+7d)
28d             = [t, t+28d)
rest_of_season  = [t, season_end_exclusive)
```

`calendar_month` Horizon은 생성하지 않는다.

유효 Base Key `(player_id, role, prediction_date)`마다 Audit 계층에서 세 Horizon Row를 유지한다.

세 Horizon은 동일한 Historical X를 공유한다.

미래 Interval이 끝나고 전체 Coverage가 확인되었으며 해당 선수의 관측 출전이 없을 때만 Count `0`을 사용한다.

다음은 `0`이 아니다.

```text
미래 interval 미완료
coverage 불완전
coverage 불명확
season end 미검증
```

이 경우 Target은 `null`, `is_censored=true`로 유지한다.

`through_date`는 Dataset Snapshot Coverage 날짜이며 공식 시즌 종료일이 아니다.

예상 최종 시즌 Count의 의미는 다음과 같다.

```text
projected_final_count
=
season_to_date_count
+ projected_remaining_count
```

Rate는 합산 Count의 분자/분모에서 다시 계산한다.

팀의 남은 경기 수와 선수의 예상 PA/BF/출장 수를 동일시하지 않는다.

##### Feature Set

CT-2는 성능을 보고 Feature Set을 선택하지 않는다.

```text
baseline_v1
extended_v1
```

두 Version을 정의한다.

`baseline_v1`은 단순 Historical Count/Rate와 Prediction Context 중심의 최소 Feature Set이다.

`extended_v1`은 `baseline_v1`에 기본적으로 5/10/20경기 Rolling Feature를 추가한다.

같은 날짜 경기 묶음이 Window 경계에 걸리는 경우 임의 순서로 같은 날짜를 분할하지 않는다.

##### Cold Start

History가 없는 요청은 자동 삭제하지 않는다.

```text
count = 0
rate = null
has_history = false
history_game_count = 0
max_source_game_date = null
```

##### Season Snapshot 사용 금지

다음 세 파일은 현재 Dataset Snapshot까지의 누적 Post-game Summary다.

```text
player_season_batting_snapshot.parquet
player_season_pitching_snapshot.parquet
team_season_snapshot.parquet
```

과거 Prediction Date에 직접 Join하지 않는다.

Historical Feature는 Event-time Fact Table에서 Prediction Date보다 엄격히 이전인 Row를 이용해 다시 계산한다.

##### Player Metadata 사용 제한

`players.parquet`의 다음 값은 전체 Dataset Snapshot을 본 뒤 계산된 Metadata이므로 Historical Prediction X 또는 M3 Population 결정에 사용하지 않는다.

```text
is_batter
is_pitcher
first_seen_date
last_seen_date
first_seen_season
last_seen_season
```

Player ID/Name은 Join/Audit/표시 목적으로 사용할 수 있다.

M3 Role/Population은 Prediction Cutoff 이전 Event-time Player Game Source에서 다시 판단한다.

##### Label Availability / Censoring / Purge

경기 결과의 정확한 종료 timestamp가 없으면 M1/M2 Label Availability는 날짜 단위로 보수적으로 관리한다.

M3는 전체 Horizon이 종료되고 필요한 Coverage/시즌 종료 근거가 확인되어야 Label이 available하다.

다음 개념을 구분한다.

```text
Exclusion:
    key/schema/join/target domain 오류 등

Censoring:
    미래 Label interval 미완료 또는 coverage 불확실

Purge:
    유효한 Label이더라도 시간 Split 경계를 침범하거나
    해당 단계의 availability 조건을 만족하지 못함
```

Censored/Purged Row를 감사 계층에서 조용히 삭제하지 않는다.

##### Schema / Manifest / Provenance

Processed Dataset은 논리적으로 다음을 분리한다.

```text
key
audit
X
y
split
quality
label status
```

`X = all columns except y` 방식은 사용하지 않는다.

각 Dataset Run은 최소 다음을 기록한다.

```text
input revision/hash
code revision
contract version
feature catalog version
feature version
target version
split version
coverage
row/null/exclusion/censoring/purge summary
output schema version
content fingerprint
```

동일 입력·동일 계약·동일 설정에서는 Dataset Content가 결정적이어야 한다.

##### Model-ready의 의미

`Model-ready`는 다음이 준비되었다는 뜻이다.

```text
Leakage-safe X/y
고정 시간 Split
Schema
Manifest / Provenance
Feature/Target Version
Censoring/Exclusion/Quality 상태
```

다음을 뜻하지 않는다.

```text
모델 학습 완료
최적 알고리즘 선정
Hyperparameter Tuning 완료
성능 검증 완료
```

이 작업은 CT-3에서 수행한다.

##### Expanded CT-2

Issue #22는 M1/M2/M3 계약을 확정한다.

Expanded CT-2에서 다음 후속 Issue가 같은 계약 문서를 확장한다.

```text
#31 -> M4 경기 상황 가치 계약 및 Dataset
#30 -> M5 Pitch 예측 계약 및 Dataset
#29 -> 외부 보강 데이터 수집·정규화
#28 -> M1~M5 통합 검증 및 Expanded CT-2 종료 Gate
```

M4/M5는 각각 #31/#30 구현 시 `docs/prediction_dataset_contract.md`와 `docs/feature_catalog.md`에 추가한다.
