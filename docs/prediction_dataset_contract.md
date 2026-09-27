# KBO Insight AI — Prediction Dataset Contract

> 문서 역할: KBO Insight AI의 Prediction-time Feature와 Model-ready Dataset이 따라야 하는 예측 시점, Grain, Target, 모집단, Horizon, Label Availability, 시간 분할, Censoring/Purge, Leakage 차단 계약을 정의한다.
>
> 문서 버전: `prediction_dataset_contract_v1`
>
> 기준 Issue: `#22 docs(data): 예측 시점·타깃·시간 분할 계약 정의`
>
> 적용 범위:
> - Issue #22에서는 M1, M2, M3 계약을 확정한다.
> - Expanded CT-2의 M4/M5 계약은 각각 #31/#30에서 이 문서에 같은 형식으로 추가한다.
> - 이 문서는 Dataset 계약 문서이며 ML 모델 학습, 알고리즘 선택, Hyperparameter Tuning, 최종 성능 평가는 다루지 않는다.

---

## 1. 목적

Canonical Derived Layer는 관측 사실을 보존하는 Post-event/Post-game Fact Layer다.

Prediction Model에 그대로 사용하면 다음과 같은 Data Leakage가 발생할 수 있다.

- 현재 경기의 최종 점수 또는 승패가 Pregame Feature에 포함되는 문제
- 현재 PA의 결과, 종료 상태, 종료 선수가 PA-start Feature에 포함되는 문제
- 같은 날짜 경기 또는 Doubleheader의 결과를 임의 정렬로 과거 정보처럼 사용하는 문제
- 미래 선수 출장 여부, 미래 PA/BF, 미래 시즌 최종 기록을 과거 예측 Feature에 사용하는 문제
- 현재 Dataset Snapshot의 전체 시즌 집계를 과거 시점 Feature로 사용하는 문제
- 미래 Label interval이 Validation/Test 기간을 침범하는 문제

따라서 CT-2에서는 각 예측 문제에 대해 다음을 먼저 고정한다.

1. Prediction Timestamp
2. Dataset Grain
3. Population
4. X가 사용할 수 있는 정보
5. Target과 Target Availability
6. Exclusion / Censoring / Purge
7. Train / Validation / Test 시간 분할
8. Schema / Manifest / Provenance
9. 재현성과 Leakage 반례 검증

---

## 2. Source of Truth와 책임 경계

### 2.1 Canonical Source

현재 Prediction Dataset의 기본 Source of Truth는 다음 Canonical/Raw 계층이다.

```text
data/raw/hf_kbo_pbp/
├─ 2023.parquet
├─ 2024.parquet
├─ 2025.parquet
├─ 2026.parquet
├─ schema.yaml
└─ download_manifest.csv

data/interim/hf_kbo_pbp/derived/
├─ plate_appearances.parquet
├─ games.parquet
├─ team_games.parquet
├─ player_game_batting.parquet
├─ player_game_pitching.parquet
├─ players.parquet
├─ player_season_batting_snapshot.parquet
├─ player_season_pitching_snapshot.parquet
└─ team_season_snapshot.parquet
```

Raw와 Canonical Derived Artifact는 Prediction Dataset 생성 과정에서 수정하거나 덮어쓰지 않는다.

### 2.2 책임 분리

| 범위 | 책임 |
|---|---|
| #22 | M1/M2/M3 Prediction Contract, Feature Availability, Split 계약 |
| #23 | 팀 Pregame Historical Feature |
| #24 | 선수 Pregame Historical Feature |
| #25 | M1 Model-ready Dataset 및 공통 Processed utility |
| #26 | M2 Model-ready Dataset |
| #27 | M3 Model-ready Dataset |
| #29 | 외부 보강 데이터 수집·정규화 |
| #30 | M5 계약·Feature Catalog 확장 및 M5 Dataset |
| #31 | M4 계약·Feature Catalog 확장 및 M4 Dataset |
| #28 | M1~M5 Expanded CT-2 통합 검증과 종료 Gate |
| CT-3 | 모델 학습, 성능 비교, Tuning, 최종 Test 평가 |

#22에서 M4/M5의 상세 계약을 추정해서 미리 고정하지 않는다.

#30/#31이 구현될 때 이 문서와 `docs/feature_catalog.md`에 각각 M5/M4 섹션을 추가한다.

---

## 3. 공통 시간 원칙

### 3.1 Strict Historical Cutoff

Historical Feature는 기본적으로 다음 조건을 만족해야 한다.

```text
source.game_date < prediction_date
```

날짜만 존재하고 정확한 시각이 없는 데이터에서 다음 방식은 허용하지 않는다.

```text
source.game_date == prediction_date
AND
game_pk가 더 작으므로 먼저 열린 경기라고 가정
```

`game_pk`, 입력 Row 순서, 파일 정렬 순서는 이용 가능 시각의 증거가 아니다.

### 3.2 Same-day / Doubleheader

같은 날짜에 여러 경기가 있더라도 실제 시작/종료 시각 또는 당시 이용 가능 순서를 검증할 수 있는 별도 provenance가 없다면 같은 날짜의 다른 경기 결과를 Historical Feature로 사용하지 않는다.

따라서 M1/M2/M3 기본 계약에서는 같은 날짜의 모든 경기 결과를 해당 날짜 Prediction Feature에서 제외한다.

### 3.3 Timestamp Resolution

현재 핵심 Historical Fact는 날짜 단위 `game_date`를 기준으로 한다.

정확한 과거 발표/종료 시각이 없는 경우 날짜 단위보다 정밀한 availability를 추정하지 않는다.

Manifest에는 다음을 기록한다.

```text
timestamp_resolution = "date"
```

정확한 timestamp가 별도 Source로 확보되는 기능은 해당 Source의 provenance를 별도 기록한다.

### 3.4 미래 사실 사용 금지

다음 정보는 공통적으로 Historical X에 사용할 수 없다.

- 현재 예측 대상의 결과
- 예측 시점 이후 경기 결과
- 예측 시점 이후 선수 출장 여부
- 미래 PA/BF/Pitch Count
- 미래 Label availability
- 미래 censoring 결과
- 미래 split/purge 결과
- 시즌 종료 후에야 확정되는 전체 시즌 기록
- 사후 수정된 일정/라인업을 당시 이용 가능했던 값으로 가정한 정보

---

## 4. 고정 시간 분할 계약

### 4.1 Split Version

```text
split_version = "season_split_v1"
```

### 4.2 Split Assignment

CT-2의 기본 고정 시간 분할은 다음과 같다.

| Season / Prediction Date | Split | 용도 |
|---|---|---|
| 2023 | `train` | CT-3 학습 |
| 2024 | `validation` | CT-3 모델/Feature/Hyperparameter 선택 |
| 2025 | `test` | CT-3 최종 평가 전용 |
| 2026 | `snapshot` | 진행 시즌 snapshot / inference / 추후 명시적 추가 검증 후보 |

달력 경계로 표현하면 다음과 같다.

```text
train:
    2023-01-01 <= prediction_date < 2024-01-01

validation:
    2024-01-01 <= prediction_date < 2025-01-01

test:
    2025-01-01 <= prediction_date < 2026-01-01

snapshot:
    2026-01-01 <= prediction_date < 2027-01-01
```

### 4.3 Split의 의미

이 분할은 성능을 높이기 위한 Hyperparameter가 아니다.

다음은 금지한다.

- Random Split
- Stratified Random Split
- Test 성능을 확인한 뒤 연도 경계 변경
- 모델별로 유리한 연도 조합 선택
- Feature 선택을 위해 Test Target 분포 사용

### 4.4 현재 Coverage와 Split의 분리

현재 Canonical Snapshot의 관측 범위가 특정 시즌의 공식 완료 여부를 증명하는 것은 아니다.

Split Assignment와 Label Usability는 분리한다.

예:

```text
prediction_date가 2025년
→ split = test

하지만 특정 M3 rest_of_season Label의 종료/coverage 근거가 불완전
→ target = null
→ is_censored = true
→ supervised_usable = false
```

즉 split이 존재한다는 이유로 Target이 자동으로 usable이 되지 않는다.

### 4.5 2026 Snapshot

2026은 현재 Dataset Snapshot에서 진행 중인 시즌으로 취급한다.

다음은 금지한다.

```text
through_date == season_end
```

또는

```text
2026의 미관측 미래 Count = 0
```

`split="snapshot"`은 자동으로 Train/Validation/Test에 포함하지 않는다.

---

## 5. Label Availability 공통 계약

### 5.1 날짜 단위 Conservative Availability

경기 종료의 정확한 timestamp를 보존하지 않는 M1/M2 Label은 날짜 단위로 다음처럼 보수적으로 정의한다.

```text
label_available_at = game_date + 1 calendar day
availability_basis = "conservative_next_date"
```

이는 실제 경기 종료 시각을 의미하지 않는다.

### 5.2 M3 Interval Label Availability

M3는 전체 Label interval이 종료되고 해당 interval의 필요한 coverage가 확인된 뒤에만 Label이 available하다.

Fixed Horizon의 기본 날짜 단위 정의는 다음과 같다.

```text
7d:
    label_available_at >= label_end_exclusive

28d:
    label_available_at >= label_end_exclusive
```

정확한 availability가 없으면 `label_end_exclusive` 이전 시각을 추정하지 않는다.

`rest_of_season`은 시즌 종료 경계와 coverage 근거가 모두 확인되어야 한다.

근거가 없으면:

```text
label_available_at = null
```

### 5.3 Split Boundary와 Availability

다음 split 시작 날짜를 `B`라고 한다.

M3 Label interval이 다음 조건을 만족하면 경계를 직접 침범한다.

```text
label_end_exclusive > B
```

이 경우:

```text
is_purged = true
purge_reason = "label_interval_crosses_split_boundary"
```

`label_end_exclusive == B`는 half-open interval 정의상 다음 split 관측을 포함하지 않는다.

그러나 Label availability도 별도로 만족해야 한다.

날짜 단위 보수적 계약에서는 이전 split 학습에 사용하려면 기본적으로 다음을 요구한다.

```text
label_available_at < B
```

따라서 interval은 경계를 넘지 않았더라도 Label이 경계 날짜에야 available한 경우 이전 split의 supervised training view에서는 purge할 수 있다.

---

## 6. 공통 Row 상태 계약

Censoring, Purge, Exclusion은 서로 다른 의미다.

### 6.1 Exclusion

입력 또는 계약 오류로 Dataset Row 자체를 supervised 후보로 사용할 수 없는 상태다.

예:

- 필수 key 결측
- key duplicate
- join cardinality 위반
- unknown non-null Target category
- 불가능한 날짜 interval
- Source schema 위반

권장 필드:

```text
is_excluded
exclusion_reason
```

### 6.2 Censoring

미래 Label 구간이 아직 끝나지 않았거나 coverage를 확인할 수 없어 정답을 확정할 수 없는 상태다.

권장 필드:

```text
is_censored
censor_reason
coverage_status
```

Censoring은 정상적인 데이터 상태일 수 있으며 자동 오류가 아니다.

### 6.3 Purge

Target 자체는 존재할 수 있지만 시간 분할 경계를 넘어 학습에 사용할 수 없는 상태다.

권장 필드:

```text
is_purged
purge_reason
```

### 6.4 Supervised Usability

최종 supervised 사용 가능 여부는 다음 의미를 가진다.

```text
supervised_usable
=
not is_excluded
AND
not is_censored
AND
not is_purged
AND
required target is non-null
```

Target별 nullable 의미가 다른 경우 Target별 mask를 별도로 제공한다.

---

# 7. M1 — 경기 결과·득점 Prediction Contract

## 7.1 목적

경기 시작 전에 알 수 있는 정보만 사용해 다음을 예측한다.

- 홈팀 승/무/패
- 홈팀 득점
- 원정팀 득점

## 7.2 Prediction Timestamp

```text
prediction_time = pregame
prediction_date = games.game_date
```

정확한 경기 시작 시각은 현재 기본 계약에서 사용하지 않는다.

Historical Feature는 엄격히 다음 조건을 사용한다.

```text
source.game_date < prediction_date
```

## 7.3 Grain

```text
game_pk
```

유효 경기당 정확히 1행이다.

## 7.4 Key / Audit

최소 Key/Audit 후보:

```text
game_pk
game_date
season
home_team
away_team
prediction_date
split
contract_version
feature_version
target_version
split_version
```

`game_pk`, Team ID, Date는 join/audit의 기본 식별자이며 `baseline_v1` 모델 입력으로 자동 포함하지 않는다.

## 7.5 Population

Canonical `games.parquet`의 유효 `game_pk`를 기준으로 한다.

다음 기본 조건을 확인한다.

- `game_pk` non-null
- `game_pk` unique
- `game_date` non-null
- `home_team` / `away_team` non-null
- `home_team != away_team`
- Canonical Score Reconciliation 통과

Dataset 입력 수는 다음 상호 배타적 상태로 대사한다.

```text
전체 game
=
supervised usable
+ target pending/held
+ excluded
```

복수 진단 사유가 있어도 합계에 중복 계상하지 않도록 대표 사유를 둔다.

## 7.6 Target

Source:

```text
home_runs = games.final_home_score
away_runs = games.final_away_score
```

`final_*_score`의 의미는 **Canonical observed terminal score**다.

KBO 공식 경기 종료 결과라고 주장하지 않는다.

### 7.6.1 Result Mapping

문자열 Target:

```text
home_result ∈ {"loss", "tie", "win"}
```

고정 Class 순서와 Code:

| class | code |
|---|---:|
| `loss` | 0 |
| `tie` | 1 |
| `win` | 2 |

Mapping:

```text
home_runs < away_runs  -> loss
home_runs == away_runs -> tie
home_runs > away_runs  -> win
```

무승부를 패배와 합치지 않는다.

### 7.6.2 Target Version

```text
target_version = "m1_target_v1"
```

## 7.7 Observation Quality

Target 의미와 Observation Quality는 분리한다.

권장 Audit 필드:

```text
target_semantics = "canonical_observed_terminal"
observation_quality
observation_quality_reason
target_eligible
```

권장 Quality 의미:

| 상태 | 의미 |
|---|---|
| `verified_canonical` | Canonical key/score/reconciliation을 통과한 관측 결과 |
| `warning_terminal_pa_incomplete` | 마지막 관측 PA가 미완료일 수 있으나 그 사실만으로 경기 미완료를 단정하지 않음 |
| `invalid_missing_score` | Target에 필요한 observed terminal score 결측 |
| `invalid_reconciliation` | Canonical score reconciliation 위반 |

`warning_terminal_pa_incomplete`는 자동 Target 폐기 사유가 아니다.

`invalid_*`는 Target 생성 전에 명시적으로 처리한다.

## 7.8 M1 X

M1 X는 다음 계층으로 구성한다.

```text
Pregame request context
+
#23 Team Historical Feature
+
추후 계약된 보강 Pregame Feature
```

현재 #22~#25 기본 범위에서는 경기 후 사실이나 실제 출장 명단을 자동 포함하지 않는다.

## 7.9 M1 Forbidden

다음은 M1 X에서 금지한다.

- `final_home_score`
- `final_away_score`
- `home_win`
- `away_win`
- `is_tie`
- `winner_team`
- `loser_team`
- `innings_played`
- 현재 경기 `pitch_rows`
- 현재 경기 `pitch_count`
- 현재 경기 `plate_appearances`
- 현재 경기 선수 출장 결과
- 현재 경기 PA 시작/종료 상태
- `observation_quality`
- terminal PA 상태
- Target 값 및 Target code
- split / purge / censoring 정보
- 발표 timestamp를 증명할 수 없는 사후 공식 라인업

---

# 8. M2 — PA 시작 조건부 결과 Prediction Contract

## 8.1 목적

특정 Plate Appearance가 시작되는 순간에 알 수 있는 정보만 사용해 해당 PA의 최종 결과를 예측한다.

## 8.2 Prediction Timestamp

```text
prediction_time = pa_start
prediction_date = PA.game_date
```

선수 Historical Feature는 다음 조건만 사용한다.

```text
source.game_date < PA.game_date
```

같은 경기 이전 PA의 실시간 누적 결과는 M2 CT-2 v1 Feature에 포함하지 않는다.

## 8.3 Grain

```text
(game_pk, at_bat_number)
```

## 8.4 시작 선수와 기록 귀속 선수

Canonical PA의 `batter`, `pitcher`, `stand`가 반드시 PA 시작 선수 정보를 의미하지는 않는다.

따라서 #26은 동일 고정 Raw snapshot에서 정렬된 **실제 첫 Raw Row**를 이용해 다음을 복원한다.

```text
starting_batter
starting_pitcher
starting_stand
```

다음은 별도 Audit로 유지한다.

```text
credited_batter
final_pitcher
credited_stand
```

컬럼별 `groupby().first()`로 서로 다른 Raw Row의 non-null 값을 조합하지 않는다.

## 8.5 PA-start Context

허용 Context:

```text
inning
inning_topbot
outs_before
on_1b_occupied
on_2b_occupied
on_3b_occupied
score_diff_before
starting_stand
```

원시 Runner ID 자체는 기본 X에서 제외하고 점유 여부 Boolean으로 변환한다.

Team/Player ID는 기본적으로 Key/Audit이며 baseline X에 자동 포함하지 않는다.

## 8.6 Target Mapping

Source event domain:

```text
single
double
triple
home_run
walk
hit_by_pitch
strikeout
field_out
double_play
triple_play
sac_bunt
sac_fly
field_error
fielders_choice
catcher_interference
```

고정 11-class Mapping:

| source event | target class | code |
|---|---|---:|
| `single` | `1B` | 0 |
| `double` | `2B` | 1 |
| `triple` | `3B` | 2 |
| `home_run` | `HR` | 3 |
| `walk` | `BB` | 4 |
| `hit_by_pitch` | `HBP` | 5 |
| `strikeout` | `SO` | 6 |
| `field_out` | `OUT` | 7 |
| `double_play` | `OUT` | 7 |
| `triple_play` | `OUT` | 7 |
| `sac_bunt` | `OUT` | 7 |
| `sac_fly` | `OUT` | 7 |
| `field_error` | `ROE` | 8 |
| `fielders_choice` | `FC` | 9 |
| `catcher_interference` | `CI` | 10 |

고정 Class 순서:

```text
["1B", "2B", "3B", "HR", "BB", "HBP", "SO", "OUT", "ROE", "FC", "CI"]
```

`OUT`은 Event Group 이름이며 정확히 1 out이 발생했다는 뜻이 아니다.

## 8.7 Null / Unknown Event

```text
event IS NULL
```

이면 미완료 PA Target이다.

```text
target = null
supervised_usable = false
```

알 수 없는 non-null event는 자동으로 `OUT` 또는 `OTHER`에 병합하지 않는다.

```text
unknown non-null event -> contract error
```

## 8.8 선수 교체 PA

PA 도중 타자 또는 투수가 교체되어도 다음 문제 정의를 유지한다.

> PA 시작 순간의 Context와 시작 선수 이력으로 최종 PA Event를 예측한다.

미래에 교체가 발생했다는 사실을 이용해 해당 Row를 미리 제외하지 않는다.

시작 선수와 기록 귀속 선수의 차이는 Audit/Subgroup 분석용으로 보존한다.

## 8.9 M2 Forbidden

다음은 M2 X에서 금지한다.

- 현재 PA `event`
- 현재 PA `pa_completed`
- 현재 PA `runs_scored`
- `post_outs`
- `post_on_1b`
- `post_on_2b`
- `post_on_3b`
- `post_home_score`
- `post_away_score`
- 현재 PA 전체 `pitch_rows`
- 현재 PA 전체 `pitch_count`
- 현재 PA Ball/Strike/In-play Summary
- PA 종료 선수
- 미래 substitution flag
- credited/final 선수를 PA-start 선수 대신 사용한 Feature
- 현재 경기 이후 선수 누적
- target class / code
- split / quality / exclusion 결과

## 8.10 Target Version

```text
target_version = "m2_target_v1"
```

---

# 9. M3 — 선수 미래 성적 Projection Contract

## 9.1 목적

매주 월요일 Prediction Cutoff 이전의 선수 Historical Feature를 사용해 세 개의 미래 Horizon 성적을 예측한다.

## 9.2 Prediction Anchor

```text
prediction_date = t
t = Monday
```

한국 현지 날짜 기준 월요일을 사용한다.

Historical X:

```text
source.game_date < t
```

같은 월요일 경기 결과는 X에 포함하지 않는다.

## 9.3 Anchor 범위

시즌별 Anchor 후보는 현재 Dataset의 해당 시즌 관측 범위 안의 월요일로 구성한다.

기본 원칙:

- 공식 시즌 완료 여부를 `through_date`로 추정하지 않는다.
- 현재 Snapshot 이후의 미래 월요일을 과거 학습 anchor로 생성하지 않는다.
- offseason 월요일을 자동 생성하지 않는다.
- 실제 요청 anchor 범위는 config와 manifest에 기록한다.

## 9.4 Grain

```text
(player_id, role, prediction_date, horizon_id)
```

`role`:

```text
batting
pitching
```

같은 Player가 두 역할에 실제 cutoff 이전 기록을 가진 경우 두 Role Row가 존재할 수 있다.

## 9.5 Population

Population은 `t` 이전에 관측된 선수·역할로만 구성한다.

Batting:

```text
player_game_batting.game_date < t
```

에서 관측된 Batter.

Pitching:

```text
player_game_pitching.game_date < t
```

에서 관측된 Pitcher.

다음으로 모집단을 필터링하지 않는다.

- 미래 출장 여부
- 미래 최소 PA
- 미래 최소 BF
- 미래 Team
- 시즌 종료 후 전체 기간 Role
- `players.last_seen_date`
- 시즌 최종 기록 존재 여부

Trade 전후에도 동일 `player_id`의 개인 이력은 연결한다.

과거 Team Context가 필요하면 Event-time Source를 사용하며 Player Master의 단일 Team을 만들지 않는다.

## 9.6 Horizon

고정 Horizon:

| horizon_id | label_start | label_end_exclusive |
|---|---|---|
| `7d` | `t` | `t + 7 days` |
| `28d` | `t` | `t + 28 days` |
| `rest_of_season` | `t` | 검증된 `season_end_exclusive` |

고정 순서:

```text
["7d", "28d", "rest_of_season"]
```

`calendar_month`는 생성하지 않는다.

### 9.6.1 7d / 28d

Half-open interval:

```text
7d  = [t, t+7d)
28d = [t, t+28d)
```

`t` 날짜의 관측은 Label에 포함한다.

종료 경계 날짜의 관측은 포함하지 않는다.

시즌 말이라는 이유로 interval을 임의로 단축하지 않는다.

### 9.6.2 rest_of_season

```text
[t, season_end_exclusive)
```

`season_end_exclusive`는 검증 가능한 정규시즌 종료 경계 근거가 있어야 한다.

금지:

```text
season_end_exclusive
=
max(observed game_date) + 1 day
```

단순 `through_date`는 Snapshot Coverage이며 공식 종료 근거가 아니다.

## 9.7 Horizon Row 보존

유효 Base Key:

```text
(player_id, role, prediction_date)
```

마다 Filter 전 Audit Dataset에서 다음 3개 Row를 정확히 생성한다.

```text
7d
28d
rest_of_season
```

즉:

```text
valid base key count * 3
=
audit horizon row count
```

Censored/Purged Row도 Audit Output에서 삭제하지 않는다.

Eligibility에서 Base Key 자체가 제외된 경우 별도 Exclusion Report에 Key와 이유를 남긴다.

## 9.8 공통 X 재사용

같은 Base Key의 세 Horizon은 Historical X가 동일해야 한다.

다음도 동일해야 한다.

```text
feature_version
source_cutoff
source_fingerprint
prediction_date
role
```

`horizon_id`는 Target/Task 구분 Key이며 `baseline_v1` Historical X에 포함하지 않는다.

## 9.9 M3 Batting Target

권장 Count Target:

```text
target_pa
target_ab
target_h
target_single
target_double
target_triple
target_hr
target_bb
target_hbp
target_so
target_sf
target_sh
target_catcher_interference
target_tb
```

공식식:

```text
target_h
=
target_single
+ target_double
+ target_triple
+ target_hr

target_ab
=
target_pa
- target_bb
- target_hbp
- target_sh
- target_sf
- target_catcher_interference

target_tb
=
target_single
+ 2 * target_double
+ 3 * target_triple
+ 4 * target_hr
```

파생 Rate:

```text
target_avg = target_h / target_ab

target_obp
=
(target_h + target_bb + target_hbp)
/
(target_ab + target_bb + target_hbp + target_sf)

target_slg = target_tb / target_ab

target_ops = target_obp + target_slg
```

Rate는 Count Aggregate에서 계산한다.

경기별 Rate 평균을 사용하지 않는다.

0 denominator는 `null`이다.

`target_ops`는 `target_obp`, `target_slg`가 모두 정의된 경우에만 계산한다.

## 9.10 M3 Pitching Target

권장 Count Target:

```text
target_batters_faced_completed
target_pitches
target_hits_allowed
target_single_allowed
target_double_allowed
target_triple_allowed
target_hr_allowed
target_bb_allowed
target_hbp_allowed
target_so
target_sf
target_sh
target_outs_recorded
```

공식식:

```text
target_hits_allowed
=
target_single_allowed
+ target_double_allowed
+ target_triple_allowed
+ target_hr_allowed
```

현재 공개 Raw/Canonical 계약으로 공식 책임을 검증할 수 없는 다음 Target은 생성하지 않는다.

```text
target_earned_runs
target_era
target_runs_allowed
```

`target_outs_recorded`는 Source interval에 불확실 Player Game Outs가 하나라도 있으면 `null`을 전파한다.

불확실 값을 `0`으로 바꾸거나 `skipna=True` 부분 합계를 확정값으로 사용하지 않는다.

## 9.11 Count 0과 Censored Null

다음 조건을 모두 만족할 때만 Count `0`을 허용한다.

1. Label interval이 종료됨
2. interval 전체 coverage가 확인됨
3. 해당 선수·역할의 관측 출전이 interval에 없음

즉:

```text
interval complete
AND
coverage complete
AND
no observed participation
-> target count = 0
```

다음은 `0`이 아니다.

```text
interval not finished
coverage incomplete
coverage unknown
season end not verified
```

이 경우:

```text
target = null
is_censored = true
```

관측된 부분 합계를 완성된 Horizon Target으로 사용하지 않는다.

## 9.12 Target별 Null

Interval 자체는 complete하더라도 특정 Target만 계산할 수 없을 수 있다.

예:

- Rate denominator 0
- `outs_recorded` 불확실

이 경우 전체 Horizon을 자동 Censor하지 않고 Target별 mask와 사유를 둔다.

예:

```text
target_outs_recorded_usable = false
target_outs_recorded_reason = "uncertain_source_outs"
```

## 9.13 Coverage Status

권장 고정 값:

```text
complete
incomplete
unknown
```

권장 Censor Reason 예:

```text
interval_not_finished
coverage_incomplete
coverage_unknown
season_end_unverified
label_availability_unknown
```

정상 Censoring과 Schema Error를 혼동하지 않는다.

## 9.14 M3 Label Fields

모든 Horizon Row는 최소 다음 Audit/Label 상태를 가진다.

```text
label_start
label_end_exclusive
label_available_at
availability_basis
coverage_status
is_censored
censor_reason
is_purged
purge_reason
split
supervised_usable
```

## 9.15 Season-to-date와 Remaining 의미

다음 세 값을 명확히 분리한다.

```text
season_to_date_count(t)
remaining_target_count
projected_remaining_count
```

의미:

```text
season_to_date_count(t)
=
같은 시즌에서 source.game_date < t인 실제 누적 Count

remaining_target_count
=
Label interval의 사후 관측 실제 Count
학습 Target

projected_remaining_count
=
CT-3 모델이 이후 생성할 예측 Count
```

예상 시즌 최종 Count:

```text
projected_final_count
=
season_to_date_count(t)
+ projected_remaining_count
```

예상 최종 Rate도 합산 Count의 분자/분모에서 재계산한다.

현재 Rate와 미래 Rate를 더하거나 평균하지 않는다.

## 9.16 Team Remaining Games와 Player Exposure

다음은 같은 개념이 아니다.

```text
team_remaining_games
player_expected_games
player_expected_pa
player_expected_bf
```

팀 남은 경기 수를 선수 예상 출장/PA/BF로 직접 대입하지 않는다.

실제 미래 PA/BF는 `target_only`다.

예상 미래 PA/BF는 CT-3 이후 Projection 결과다.

## 9.17 M3 Forbidden

M3 X에서 다음을 금지한다.

- `t` 당일 경기 결과
- `t` 이후 경기 결과
- 미래 출장 여부
- 미래 PA/BF/Pitches
- Horizon Target
- 미래 Coverage 결과
- Censoring/Purge 결과
- `players.last_seen_date`
- `players.last_seen_season`
- 전체 Dataset 기간에서 계산한 미래 Role
- 시즌 전체 Snapshot 직접 Join
- 시즌 최종 Rate
- 미래 Team
- 사후 일정에서 복원한 당시 미검증 remaining games
- Label interval 종료 경계 자체를 자동 Feature로 사용

## 9.18 Target Version

```text
target_version = "m3_target_v1"
```

---

# 10. Feature Set Version 계약

CT-2는 성능을 보고 Feature Set을 선택하지 않는다.

의미와 availability만으로 다음 Version을 정의한다.

## 10.1 baseline_v1

단순 Count와 현재 시점 Context 중심의 최소 Feature Set이다.

목적:

- 단순하고 설명 가능한 CT-3 Baseline 제공
- Leakage-safe 최소 입력 제공
- `extended_v1`의 추가 가치 비교 기준 제공

## 10.2 extended_v1

`baseline_v1`에 rolling/expanding Historical Feature를 추가한 확장 Set이다.

대표 Window:

```text
5 games
10 games
20 games
```

동일 날짜 경기 묶음이 Window 경계에 걸리는 경우 같은 날짜를 임의 순서로 분할하지 않는다.

실제 포함 Row 수를 Audit에 기록한다.

## 10.3 Feature Version

권장 필드:

```text
feature_version ∈ {"baseline_v1", "extended_v1"}
```

모델별 정확한 allowlist는 `docs/feature_catalog.md`에서 정의한다.

---

# 11. Cold Start 계약

## 11.1 팀 History 없음

Prediction Date 이전 팀 History가 없으면 Row를 삭제하지 않는다.

```text
historical count = 0
historical rate = null
has_history = false
history_game_count = 0
max_source_game_date = null
```

## 11.2 선수 History 없음

요청된 선수·역할에 Prediction Date 이전 History가 없으면 요청 Row를 자동 삭제하지 않는다.

```text
historical count = 0
historical rate = null
has_history = false
history_game_count = 0
max_source_game_date = null
```

M3 기본 Population은 cutoff 이전 실제 관측 Role을 이용하므로 일반적으로 History가 존재하지만, 공통 Player Feature API는 History 없는 요청도 보존할 수 있어야 한다.

---

# 12. Manifest 계약

각 Processed Dataset Run은 최소 다음 정보를 기록한다.

```text
dataset_name
run_id
created_at
input_paths
input_revision
input_sha256_or_content_fingerprint
code_revision
contract_version
feature_catalog_version
feature_version
target_version
split_version
config
timestamp_resolution
prediction_range
source_coverage
row_count
usable_row_count
excluded_row_count
censored_row_count
purged_row_count
null_summary
quality_summary
output_schema_version
output_content_fingerprint
```

변동 Metadata와 결정적 Content Fingerprint를 구분한다.

예:

```text
created_at
```

은 재실행마다 달라질 수 있다.

반면 동일 입력·동일 계약·동일 설정의 Dataset Content는 결정적이어야 한다.

---

# 13. Schema 분리 계약

Dataset 파일에 다양한 Audit Column이 함께 존재하더라도 ML 입력은 명시적 allowlist로 분리한다.

논리적 Schema:

```text
key_columns
audit_columns
x_columns
y_columns
split_columns
quality_columns
label_status_columns
```

금지:

```text
X = dataframe.drop(target_column)
```

처럼 Target 하나만 제거하고 나머지 모든 Column을 자동 Feature로 사용하는 방식.

모든 모델은 `x_columns` allowlist를 명시적으로 사용한다.

---

# 14. 재현성 계약

동일한 다음 조건에서 Dataset Content는 결정적이어야 한다.

- 동일 Raw/Canonical Snapshot
- 동일 Contract Version
- 동일 Feature Version
- 동일 Target Version
- 동일 Split Version
- 동일 Config

입력 Row 순서를 섞어도 동일해야 한다.

검증 항목:

```text
row count
key set
column order
dtype
value content
stable sort order
content fingerprint
```

Parquet Byte-level SHA256 동일성은 Content Determinism의 필수 조건으로 두지 않는다.

입력 SHA256은 Builder 전후 불변성 검증에 사용한다.

---

# 15. Data Leakage 반례 계약

CT-2에서는 정상 fixture뿐 아니라 미래 정보를 변경하는 반례 검증이 필요하다.

## 15.1 M1

현재 경기 최종 점수/결과를 변경한다.

기대:

```text
y changes
pregame X unchanged
```

## 15.2 M2

PA 첫 Raw Row는 고정하고 이후 Row의 다음을 변경한다.

- event
- post-state
- 종료 선수

기대:

```text
PA-start X unchanged
target/audit may change
```

## 15.3 M3

미래 날짜 `u`의 선수 Count를 변경한다.

기대:

```text
historical X unchanged

u가 포함된 Horizon y만 변경
```

예:

```text
u = t + 3d
-> 7d / 28d / rest_of_season y 영향 가능

u = t + 10d
-> 28d / rest_of_season y 영향 가능

u = t + 35d
-> rest_of_season y만 영향 가능
```

종료 경계가 해당 날짜보다 뒤에 있다는 fixture 조건이 필요하다.

미래에만 등장하는 선수를 추가해도 과거 `t` Population이 바뀌면 안 된다.

---

# 16. 외부 데이터와 Historical Availability

#29에서 외부 보강 데이터를 도입하더라도 Historical X에는 해당 시점 당시 이용 가능한 정보만 사용할 수 있다.

필수 provenance:

```text
source
source_url_or_api
collected_at
source_effective_at
request_parameters
license_or_terms_basis
snapshot_id
raw_hash
parser_version
```

다음은 금지한다.

- 현재의 최신 일정표를 과거 경기 당시의 일정표로 간주
- 경기 후 관측 날씨를 경기 전 예보 Feature로 간주
- 발표 timestamp가 없는 실제 라인업을 Historical Pregame X로 사용
- 현재 엔트리 상태를 과거 엔트리 상태로 소급

---

# 17. 공식 라인업과 예상 라인업

Historical Backtesting에서 실제 공식 라인업을 사용하려면 해당 경기 Prediction Timestamp 이전에 발표되었다는 timestamp/provenance를 검증할 수 있어야 한다.

증거가 없으면 실제 공식 라인업은 Historical Pregame X에서 제외한다.

예상 라인업 기능은 별개다.

과거 출장/타순 Pattern을 예측 시점 이전 데이터만으로 계산하는 Predicted Lineup은 향후 별도 기능으로 사용할 수 있다.

---

# 18. 불펜 Availability 의미

공개 데이터로 계산하는 불펜 Feature는 다음 의미로 제한한다.

```text
workload / fatigue / entry based availability proxy
```

다음을 의미한다고 주장하지 않는다.

```text
actual medical condition
manager's private availability decision
guaranteed game availability
```

---

# 19. 구현하지 않는 정보

다음 항목은 현재 프로젝트에서 직접 Prediction Label로 만들지 않는다.

- 감독의 숨은 번트/런앤히트/스퀴즈 의도
- 투수 교체/대타/대주자의 내부 선택 사유
- 선수의 실제 내부 컨디션
- 감독 의사까지 포함한 actual availability
- 전체 견제 시도 분모가 없는 상태의 `P(견제 시도)`
- 전체 견제 시도 분모가 없는 상태의 `P(견제사 | 견제)`
- 발표 timestamp를 검증할 수 없는 실제 과거 공식 라인업의 Historical Pregame Feature 사용

관측 가능한 실제 행동 자체의 사후 분석은 별도 분석 계층에서 가능하다.

---

# 20. CT-2와 CT-3의 경계

## CT-2

CT-2 완료의 의미:

```text
Leakage-safe Model-ready Dataset
+
고정 시간 Split
+
Schema
+
Manifest / Provenance
+
Feature/Target Version
+
Censoring / Exclusion / Quality Summary
+
재현성 및 반례 검증
```

CT-2는 모델을 학습하지 않는다.

## CT-3

CT-3에서 수행한다.

1. Dataset EDA
2. Target/Class 분포 분석
3. Baseline 학습
4. `baseline_v1` vs `extended_v1`
5. 각 문제 내부 ML Algorithm 비교
6. Bias–Variance 분석
7. Hyperparameter Tuning
8. Validation 기반 후보 확정
9. Test 최종 평가
10. SHAP / Calibration / Backtesting 등 후속 분석

Train/Validation/Test Split 자체를 성능 향상용 Hyperparameter로 바꾸지 않는다.

Test Target은 Feature/모델/Hyperparameter 선택에 사용하지 않는다.

---

# 21. Expanded CT-2 확장 지점

#22 완료 시 이 문서는 M1/M2/M3 계약을 직접 포함한다.

이후:

```text
#31 -> M4 섹션 추가
#30 -> M5 섹션 추가
#28 -> M1~M5 전체 계약 통합 검증
```

M4/M5가 추가되더라도 기존 M1/M2/M3의 prediction-time 의미를 소급 변경하지 않는다.

공통 계약 변경이 필요한 경우 Contract Version을 올리고 변경 근거와 영향 범위를 기록한다.

---

# 22. 계약 버전 요약

```text
contract_version        = prediction_dataset_contract_v1
feature_catalog_version = feature_catalog_v1
split_version           = season_split_v1

M1 target_version       = m1_target_v1
M2 target_version       = m2_target_v1
M3 target_version       = m3_target_v1

feature_version:
- baseline_v1
- extended_v1
```

---

# 23. 완료 기준

#22 계약 문서의 완료 조건은 다음과 같다.

- M1 prediction timestamp/grain/target이 고정되어 있다.
- M2 PA-start/starting player/11-class mapping이 고정되어 있다.
- M3 Monday anchor/7d/28d/rest_of_season이 고정되어 있다.
- M3 count/rate/0/null/censoring 의미가 고정되어 있다.
- Same-day/Doubleheader 차단 규칙이 정의되어 있다.
- 시간 Split과 Test 사용 제한이 정의되어 있다.
- Label Availability와 Purge가 정의되어 있다.
- `baseline_v1`, `extended_v1`의 의미가 정의되어 있다.
- Manifest/Schema/Provenance 계약이 정의되어 있다.
- 데이터 한계로 구현하지 않는 항목이 명시되어 있다.
- `docs/feature_catalog.md`와 모순이 없다.
- `data/README.md`가 이 문서와 Feature Catalog를 Prediction-time Source of Truth로 안내한다.
- M4/M5 확장 책임이 #31/#30으로 명시되어 있다.
