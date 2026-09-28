# KBO Insight AI — Feature Catalog

> 문서 역할: Prediction-time Dataset에 사용되는 Source Column과 Derived Feature의 의미, 이용 가능 시점, 모델별 사용 가능 여부, 산식, 결측 정책, X allowlist/forbidden list를 정의한다.
>
> 문서 버전: `feature_catalog_v1`
>
> 기준 Issue: `#22 docs(data): 예측 시점·타깃·시간 분할 계약 정의`
>
> 적용 범위:
> - v1은 M1/M2/M3를 직접 정의한다.
> - M4/M5는 #31/#30에서 이 문서에 동일 형식으로 확장한다.
> - 이 문서에 명시되지 않은 Raw/Canonical Column은 모델 Feature로 자동 사용하지 않는다.

---

# 1. 기본 원칙

## 1.1 Explicit Allowlist

ML 입력은 명시적 `x_columns` allowlist만 사용한다.

금지:

```text
X = all columns except y
```

Dataset에 Key/Audit/Quality/Split/Label 상태가 함께 존재해도 자동 Feature로 사용하지 않는다.

## 1.2 Source Fact와 Prediction Feature의 구분

Post-game Fact는 과거 날짜의 Historical Aggregate를 계산하기 위한 Source가 될 수 있다.

예:

```text
2025-05-01 team_games.runs_for
```

은 2025-05-01 경기 전에 알 수 있는 Feature가 아니다.

그러나 다음처럼 더 나중의 Prediction Date에서는 Historical Source가 될 수 있다.

```text
source.game_date < prediction_date
```

## 1.3 Same-day 금지

날짜만 존재하는 경우:

```text
source.game_date == prediction_date
```

인 Source Result는 Historical Feature에서 제외한다.

`game_pk`, Row 순서, 파일 순서로 same-day availability를 추정하지 않는다.

---

# 2. Availability Category

Feature Catalog는 다음 Availability Category를 사용한다.

| Category | 의미 |
|---|---|
| `key_audit` | 식별, Join, Audit용. 기본 X에는 포함하지 않음 |
| `pregame` | 경기 시작 전 이용 가능한 Context 또는 이전 날짜 Historical Feature |
| `pa_start` | PA 시작 순간 이용 가능한 Context |
| `within_pa` | PA 진행 중 특정 시점에는 알 수 있으나 M1/M2 PA-start에는 사용 불가 |
| `post_pa` | PA 종료 이후에만 확정 |
| `post_game` | 경기 종료 이후에만 확정 |
| `target_only` | Target 생성 전용 |
| `unavailable` | 현재 계약으로 Prediction Feature 사용 금지 또는 근거 부족 |

동일 Source Column이라도 Model별 역할이 다를 수 있다.

---

# 3. 공통 금지 Column Family

다음 계열은 모델 입력에 자동 포함하지 않는다.

```text
target
target_code
label_*
is_censored
censor_reason
coverage_status
is_purged
purge_reason
supervised_usable
split
observation_quality
exclusion_reason
manifest fields
output fingerprint
```

이 값은 Audit/Training Control 정보다.

---

# 4. Raw Pitch-level Source Catalog

v1 M1/M2/M3에서 Raw Pitch-level Source는 주로 M2의 **PA 시작 선수 복원**과 Canonical 검증에 제한적으로 사용한다.

| Source Column | Availability | M1 | M2 | M3 | 비고 |
|---|---|---:|---:|---:|---|
| `game_pk` | `key_audit` | Key | Key | No | 식별자 |
| `game_date` | `key_audit` | Audit | Audit | Historical cutoff | 날짜 |
| `at_bat_number` | `key_audit` | No | Key | No | PA Grain |
| `pitch_number` | `within_pa` | No | Audit | No | 실제 첫 Raw Row 선택/순서 검증 |
| `batter` | Model-dependent | No | **첫 Raw Row만 PA-start 식별자** | No | 최종/중간 Row 자동 사용 금지 |
| `pitcher` | Model-dependent | No | **첫 Raw Row만 PA-start 식별자** | No | 최종/중간 Row 자동 사용 금지 |
| `stand` | Model-dependent | No | **첫 Raw Row만 PA-start Context** | No | 시작 타자 기준 |
| `events` | `post_pa` / `target_only` | No | Target source | No | M2 y 전용 |
| `pitch_type` | `within_pa` | No | No | No | M5에서 #30 계약 후 사용 |
| `release_speed_kmh` | `within_pa` | No | No | No | M5 또는 별도 Pitch Feature 계약 전 사용 금지 |
| `plate_x` | `within_pa` | No | No | No | M5에서 #30 계약 후 사용 |
| `plate_z` | `within_pa` | No | No | No | M5에서 #30 계약 후 사용 |
| Pitch Result / `type` | `within_pa` | No | No | No | 현재 PA 전체 summary를 M2 X에 넣지 않음 |

Raw의 나머지 Column은 별도 검토 없이 X에 추가하지 않는다.

---

# 5. Canonical `plate_appearances.parquet`

## 5.1 Key / Context

| Column | Availability | M1 | M2 | M3 | 정책 |
|---|---|---:|---:|---:|---|
| `game_pk` | `key_audit` | Key | Key | No | X 제외 |
| `game_date` | `key_audit` | Audit | prediction date | Historical source date | 직접 숫자 Feature 아님 |
| `season` | `key_audit` | Audit | Audit | Audit | split/season 식별 |
| `at_bat_number` | `key_audit` | No | Key | No | X 제외 |
| `inning` | `pa_start` | No | Yes | No | M2 Context |
| `inning_topbot` | `pa_start` | No | Yes | No | M2 Context |
| `batting_team` | `pa_start` / `key_audit` | No | Audit | No | ID 기본 X 제외 |
| `fielding_team` | `pa_start` / `key_audit` | No | Audit | No | ID 기본 X 제외 |
| `is_home_batting` | `pa_start` | No | Optional context | No | `inning_topbot`와 중복 관리 |
| `batter` | `post_pa`/audit for M2 | No | **starting_batter 대체 금지** | No | Canonical credited batter |
| `batter_name` | `key_audit` | No | No | No | 표시용 |
| `pitcher` | `post_pa`/audit for M2 | No | **starting_pitcher 대체 금지** | No | Canonical 종료 투수 기준 가능 |
| `pitcher_name` | `key_audit` | No | No | No | 표시용 |
| `stand` | Model-dependent | No | **starting_stand 대체 금지** | No | Canonical credited batter 기준 가능 |

## 5.2 PA-start State

| Column | Availability | M1 | M2 | M3 | 정책 |
|---|---|---:|---:|---:|---|
| `outs_before` | `pa_start` | No | Yes | No | 0/1/2 |
| `on_1b_before` | `pa_start` | No | Occupancy만 | No | Runner ID 자체 X 제외 |
| `on_2b_before` | `pa_start` | No | Occupancy만 | No | Runner ID 자체 X 제외 |
| `on_3b_before` | `pa_start` | No | Occupancy만 | No | Runner ID 자체 X 제외 |
| `home_score_before` | `pa_start` | No | Context source | No | 절대 점수보다 diff 우선 |
| `away_score_before` | `pa_start` | No | Context source | No | 절대 점수보다 diff 우선 |
| `batting_score_before` | `pa_start` | No | Optional | No | 중복 Feature 주의 |
| `fielding_score_before` | `pa_start` | No | Optional | No | 중복 Feature 주의 |
| `score_diff_before` | `pa_start` | No | Yes | No | 공격팀 관점 |

M2 점유 여부 Derived Column:

```text
on_1b_occupied = on_1b_before IS NOT NULL
on_2b_occupied = on_2b_before IS NOT NULL
on_3b_occupied = on_3b_before IS NOT NULL
```

Runner ID 자체는 모델 입력에 사용하지 않는다.

## 5.3 PA 종료/결과

다음은 M2 PA-start X에서 전부 금지한다.

| Column | Availability | M2 역할 |
|---|---|---|
| `event` | `target_only` | Target source |
| `pa_completed` | `post_pa` | Audit only |
| `runs_scored` | `post_pa` | Forbidden X |
| `post_outs` | `post_pa` | Forbidden X |
| `post_on_1b` | `post_pa` | Forbidden X |
| `post_on_2b` | `post_pa` | Forbidden X |
| `post_on_3b` | `post_pa` | Forbidden X |
| `post_home_score` | `post_pa` | Forbidden X |
| `post_away_score` | `post_pa` | Forbidden X |
| `pitch_rows` | `post_pa` | Forbidden X |
| `pitch_count` | `post_pa` | Forbidden X |
| `ball_pitch_count` | `post_pa` | Forbidden X |
| `strike_pitch_count` | `post_pa` | Forbidden X |
| `in_play_pitch_count` | `post_pa` | Forbidden X |

---

# 6. Canonical `games.parquet`

| Column Family | Availability | M1 | M2 | M3 | 정책 |
|---|---|---:|---:|---:|---|
| `game_pk` | `key_audit` | Key | Join | No | X 제외 |
| `game_date` | `key_audit` | prediction date | Audit | Historical cutoff | X 자동 포함 금지 |
| `season` | `key_audit` | Audit | Audit | Audit | split 식별 |
| `home_team` | `pregame`/ID | Audit/context | No | No | baseline X에 ID 자동 포함 안 함 |
| `away_team` | `pregame`/ID | Audit/context | No | No | baseline X에 ID 자동 포함 안 함 |
| `final_home_score` | `target_only` | y | No | Historical source only through team_games | 현재 경기 X 금지 |
| `final_away_score` | `target_only` | y | No | Historical source only through team_games | 현재 경기 X 금지 |
| `home_win` | `post_game` | y derivation only | No | No | 현재 경기 X 금지 |
| `away_win` | `post_game` | y derivation only | No | No | 현재 경기 X 금지 |
| `is_tie` | `post_game` | y derivation only | No | No | 현재 경기 X 금지 |
| `winner_team` | `post_game` | No | No | No | X 금지 |
| `loser_team` | `post_game` | No | No | No | X 금지 |
| `innings_played` | `post_game` | No | No | No | X 금지 |
| Game PA/Pitch Count | `post_game` | No | No | No | 현재 경기 X 금지 |
| `total_runs` | `post_game` | No | No | No | 현재 경기 X 금지 |

---

# 7. Canonical `team_games.parquet`

`team_games`는 Post-game Fact다.

현재 경기 Row를 M1 X로 직접 사용하지 않는다.

Historical Team Feature Source로 사용할 때만:

```text
source.game_date < prediction_date
```

조건을 적용한다.

Source Count:

```text
runs_for
runs_against
run_diff
win
loss
tie
plate_appearances
opponent_plate_appearances
```

이 값들은 #23에서 Prediction-time Historical Aggregate로 변환한 뒤 M1에 제공한다.

`is_home`은 Source 경기의 과거 Context이며, 현재 Prediction Game의 Home/Away는 별도 request context다.

---

# 8. Canonical `player_game_batting.parquet`

전체 Row는 Post-game Fact다.

M2/M3 Historical Feature Source로 사용할 때:

```text
source.game_date < prediction_date
```

을 적용한다.

허용 Historical Count Source:

```text
pa
ab
h
single
double
triple
hr
bb
hbp
so
sf
sh
tb
double_play
triple_play
field_error
fielders_choice
catcher_interference
```

Historical Rate Source는 경기별 Rate 평균에 사용하지 않는다.

```text
avg
obp
slg
ops
```

는 Fact/Audit로 존재할 수 있지만 Historical Aggregate Rate는 누적 Count에서 다시 계산한다.

현재 경기 Player Game Row는 해당 경기 또는 PA의 Prediction X에 사용하지 않는다.

---

# 9. Canonical `player_game_pitching.parquet`

전체 Row는 Post-game Fact다.

M2/M3 Historical Feature Source로 사용할 때:

```text
source.game_date < prediction_date
```

을 적용한다.

허용 Historical Count Source:

```text
pitch_rows
pitches
batters_faced_completed
hits_allowed
single_allowed
double_allowed
triple_allowed
hr_allowed
bb_allowed
hbp_allowed
so
sf
sh
outs_recorded
ball_pitch_count
strike_pitch_count
in_play_pitch_count
```

`outs_recorded`가 `<NA>`인 Player Game은 불확실성을 유지한다.

다음 방식은 금지한다.

```text
fillna(0)
sum(skipna=True)를 확정 Outs로 해석
```

`avg_release_speed_kmh`는 현재 Player Game에 유효 구속 관측 수 denominator가 없으므로 `feature_catalog_v1` 누적/rolling 기본 Feature에서 제외한다.

금지:

```text
mean(game_avg_release_speed_kmh)
weighted mean using total pitches
```

공식 책임 검증이 없는 다음 값은 사용하지 않는다.

```text
earned_runs
era
runs_allowed
```

---

# 10. `players.parquet`

Player Master는 Entity/Audit용이다.

| Column | 기본 역할 | Historical X |
|---|---|---:|
| `player_id` | Join Key | No |
| `display_name` | 표시/Audit | No |
| `name_variants` | Audit | No |
| `is_batter` | 전체 Snapshot Role | No |
| `is_pitcher` | 전체 Snapshot Role | No |
| `first_seen_date` | 전체 Snapshot Audit | No |
| `last_seen_date` | 전체 Snapshot Audit | No |
| `first_seen_season` | 전체 Snapshot Audit | No |
| `last_seen_season` | 전체 Snapshot Audit | No |

M3 Population/Role은 `players.parquet`의 전체 Snapshot Role로 만들지 않는다.

Prediction Cutoff 이전 `player_game_*` Source로 Role을 다시 판단한다.

`last_seen_*`를 은퇴, 부상, 미래 비출장의 정답으로 사용하지 않는다.

---

# 11. Season Snapshot 3종

```text
player_season_batting_snapshot.parquet
player_season_pitching_snapshot.parquet
team_season_snapshot.parquet
```

은 현재 Dataset Snapshot까지의 누적 요약이다.

Historical Prediction X에 직접 Join하지 않는다.

금지 예:

```text
2025-05-01 Prediction
<- 2025 Season Snapshot 전체
```

`through_date`는:

```text
max(team_games.game_date by season)
```

이며 공식 시즌 종료일이 아니다.

다음으로 사용하지 않는다.

```text
season_complete
official season_end
player last active date
```

Historical Feature는 Event-time Fact에서 cutoff 이전 Row만 사용해 계산한다.

---

# 12. 팀 Historical Feature Catalog — #23

## 12.1 공통 Key/Audit

권장 Output Grain:

```text
(game_pk, team)
```

권장 Key/Audit:

```text
game_pk
team
opponent
prediction_date
season
is_home
max_source_game_date
history_game_count
has_history
feature_version
```

`max_source_game_date`는 History가 있을 경우 항상:

```text
max_source_game_date < prediction_date
```

여야 한다.

## 12.2 Expanding Feature

고정 semantic name 권장:

```text
hist_games
hist_wins
hist_losses
hist_ties
hist_runs_for
hist_runs_against
hist_run_diff
hist_win_pct
hist_runs_for_per_game
hist_runs_against_per_game
hist_run_diff_per_game
days_since_last_observed_game
```

공식식:

```text
hist_games
=
hist_wins + hist_losses + hist_ties

hist_run_diff
=
hist_runs_for - hist_runs_against

hist_win_pct
=
hist_wins / (hist_wins + hist_losses)
```

`hist_wins + hist_losses == 0`이면:

```text
hist_win_pct = null
```

`hist_games == 0`이면 per-game Rate는 `null`이다.

`days_since_last_observed_game`는 관측된 이전 경기 날짜와 Prediction Date의 차이다.

공식 휴식일, 우천취소 횟수, 일정 공백이라고 해석하지 않는다.

## 12.3 Rolling Window

기본 Window:

```text
5 games
10 games
20 games
```

Window별 권장 naming:

```text
last_5g_*
last_10g_*
last_20g_*
```

예:

```text
last_5g_games
last_5g_wins
last_5g_losses
last_5g_ties
last_5g_runs_for
last_5g_runs_against
last_5g_run_diff
last_5g_win_pct
last_5g_runs_for_per_game
last_5g_runs_against_per_game
```

10g, 20g도 같은 의미를 사용한다.

같은 날짜 경기가 Window 경계에 걸리면 임의 경기 순서로 같은 날짜를 분할하지 않는다.

선택된 경계 날짜 묶음을 전체 포함하고 실제 포함 경기 수를 `last_*g_games`에 기록한다.

따라서 `last_5g_games`가 항상 정확히 5라는 보장은 없다.

---

# 13. 선수 Historical Feature Catalog — #24

권장 Grain:

```text
(player_id, role, prediction_date)
```

Role:

```text
batting
pitching
```

## 13.1 공통 Key/Audit

```text
player_id
role
prediction_date
max_source_game_date
history_game_count
has_history
feature_version
```

History가 있으면:

```text
max_source_game_date < prediction_date
```

## 13.2 Batting Expanding Count

권장 Feature:

```text
hist_pa
hist_ab
hist_h
hist_single
hist_double
hist_triple
hist_hr
hist_bb
hist_hbp
hist_so
hist_sf
hist_sh
hist_tb
hist_double_play
hist_triple_play
hist_field_error
hist_fielders_choice
hist_catcher_interference
```

공식식:

```text
hist_h
=
hist_single + hist_double + hist_triple + hist_hr

hist_ab
=
hist_pa
- hist_bb
- hist_hbp
- hist_sh
- hist_sf
- hist_catcher_interference

hist_tb
=
hist_single
+ 2 * hist_double
+ 3 * hist_triple
+ 4 * hist_hr
```

## 13.3 Batting Expanding Rate

```text
hist_avg
hist_obp
hist_slg
hist_ops
```

산식:

```text
hist_avg = hist_h / hist_ab

hist_obp
=
(hist_h + hist_bb + hist_hbp)
/
(hist_ab + hist_bb + hist_hbp + hist_sf)

hist_slg = hist_tb / hist_ab

hist_ops = hist_obp + hist_slg
```

0 denominator는 `null`.

Rate는 경기 Rate 평균이 아니라 Count Aggregate에서 재계산한다.

## 13.4 Pitching Expanding Count

권장 Feature:

```text
hist_pitch_rows
hist_pitches
hist_batters_faced_completed
hist_hits_allowed
hist_single_allowed
hist_double_allowed
hist_triple_allowed
hist_hr_allowed
hist_bb_allowed
hist_hbp_allowed
hist_so
hist_sf
hist_sh
hist_outs_recorded
hist_ball_pitch_count
hist_strike_pitch_count
hist_in_play_pitch_count
```

`hist_outs_recorded`는 Source History Window 안에 불확실 `outs_recorded`가 하나라도 있으면 `null`.

## 13.5 Player Rolling Window

기본 Window:

```text
5 games
10 games
20 games
```

Batting/Pitching 모두 동일한 원칙을 사용한다.

Naming:

```text
last_5g_<stat>
last_10g_<stat>
last_20g_<stat>
```

같은 날짜 경기가 Window 경계에 걸리면 동일 날짜 묶음을 임의로 분할하지 않는다.

## 13.6 Player Activity Context

허용:

```text
days_since_last_observed_appearance
```

의미:

> Prediction Date 이전 실제 관측 Player Game 중 가장 최근 `game_date`와 Prediction Date 사이 날짜 차이.

다음을 의미하지 않는다.

- 부상
- 엔트리 말소
- 감독의 비기용 의사
- actual availability

---

# 14. `baseline_v1` Allowlist

## 14.1 M1 `baseline_v1`

M1 Dataset에서 Home/Away Historical Feature를 Prefix로 분리한다.

권장 X:

```text
home_hist_games
home_hist_wins
home_hist_losses
home_hist_ties
home_hist_runs_for
home_hist_runs_against
home_hist_run_diff
home_hist_win_pct
home_hist_runs_for_per_game
home_hist_runs_against_per_game
home_hist_run_diff_per_game
home_days_since_last_observed_game
home_has_history

away_hist_games
away_hist_wins
away_hist_losses
away_hist_ties
away_hist_runs_for
away_hist_runs_against
away_hist_run_diff
away_hist_win_pct
away_hist_runs_for_per_game
away_hist_runs_against_per_game
away_hist_run_diff_per_game
away_days_since_last_observed_game
away_has_history
```

선택적인 Difference Feature는 명시적으로 Catalog에 추가한 것만 허용한다.

권장 최소 Difference:

```text
diff_hist_win_pct
diff_hist_runs_for_per_game
diff_hist_runs_against_per_game
diff_hist_run_diff_per_game
```

산식:

```text
diff_* = home_* - away_*
```

Team ID 자체는 `baseline_v1`에 포함하지 않는다.

## 14.2 M2 `baseline_v1`

PA-start Context:

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

Batter Historical:

```text
batter_hist_pa
batter_hist_ab
batter_hist_h
batter_hist_hr
batter_hist_bb
batter_hist_hbp
batter_hist_so
batter_hist_tb
batter_hist_avg
batter_hist_obp
batter_hist_slg
batter_hist_ops
batter_has_history
```

Pitcher Historical:

```text
pitcher_hist_pitches
pitcher_hist_batters_faced_completed
pitcher_hist_hits_allowed
pitcher_hist_hr_allowed
pitcher_hist_bb_allowed
pitcher_hist_hbp_allowed
pitcher_hist_so
pitcher_hist_outs_recorded
pitcher_has_history
```

`pitcher_hist_outs_recorded`는 nullable을 유지한다.

`starting_batter`, `starting_pitcher` ID는 Key/Audit이며 baseline X에 포함하지 않는다.

투수 손잡이는 현재 확인되지 않은 경우 추정해서 만들지 않는다.

## 14.3 M3 `baseline_v1`

M3는 Role별 Dataset Schema를 사용한다.

### Batting

```text
hist_pa
hist_ab
hist_h
hist_single
hist_double
hist_triple
hist_hr
hist_bb
hist_hbp
hist_so
hist_sf
hist_sh
hist_tb
hist_avg
hist_obp
hist_slg
hist_ops
history_game_count
days_since_last_observed_appearance
has_history
```

### Pitching

```text
hist_pitch_rows
hist_pitches
hist_batters_faced_completed
hist_hits_allowed
hist_single_allowed
hist_double_allowed
hist_triple_allowed
hist_hr_allowed
hist_bb_allowed
hist_hbp_allowed
hist_so
hist_sf
hist_sh
hist_outs_recorded
hist_ball_pitch_count
hist_strike_pitch_count
hist_in_play_pitch_count
history_game_count
days_since_last_observed_appearance
has_history
```

다음은 baseline X가 아니다.

```text
player_id
role
prediction_date
horizon_id
split
label_*
future target
```

---

# 15. `extended_v1` Allowlist

`extended_v1`은 `baseline_v1` 전체에 다음 Rolling Feature를 추가한다.

## 15.1 M1

Home/Away 각각:

```text
last_5g_*
last_10g_*
last_20g_*
```

허용 Statistic Family:

```text
games
wins
losses
ties
runs_for
runs_against
run_diff
win_pct
runs_for_per_game
runs_against_per_game
run_diff_per_game
```

Home/Away Difference Feature를 만들 경우 Catalog에 정확한 Column을 명시한다.

## 15.2 M2

Batter/Pitcher 각각:

```text
last_5g_*
last_10g_*
last_20g_*
```

Batting 허용 Family:

```text
pa
ab
h
hr
bb
hbp
so
tb
avg
obp
slg
ops
```

Pitching 허용 Family:

```text
pitches
batters_faced_completed
hits_allowed
hr_allowed
bb_allowed
hbp_allowed
so
outs_recorded
```

Rate는 Window Count Aggregate에서 재계산한다.

## 15.3 M3

Role별 `baseline_v1`에 5/10/20 game Rolling Count/Rate를 추가한다.

Horizon에 따라 Historical X를 다르게 계산하지 않는다.

---

# 16. Cold Start / Null Policy

## 16.1 History 없음

```text
count feature = 0
rate feature = null
has_history = false
history_game_count = 0
max_source_game_date = null
```

## 16.2 0 Denominator

```text
rate = null
```

0으로 채우지 않는다.

## 16.3 Pitching Outs 불확실성

Source Window에 불확실 Outs가 있으면 누적/rolling `outs_recorded`도 `null`.

부분 합계를 확정값으로 사용하지 않는다.

## 16.4 Missing Context

Prediction Context의 필수 Column이 결측이면 해당 모델의 계약에 따라 Exclusion/Audit한다.

임의 값으로 보정하지 않는다.

---

# 17. M1 Forbidden List

다음 Family는 M1 X에 포함할 수 없다.

```text
current game result
current game final score
current game innings played
current game PA/Pitch count
current game actual lineup
current game actual substitutions
current game strategy outcome
current game player post-game stats
observation_quality
target
target code
split
censor/purge/exclusion result
post-game state
future schedule snapshot not available at prediction time
```

공식 라인업은 당시 발표 timestamp/provenance가 검증된 경우에만 별도 Feature Contract로 추가할 수 있다.

---

# 18. M2 Forbidden List

```text
event
pa_completed
runs_scored
post_*
pitch_rows
pitch_count
ball_pitch_count
strike_pitch_count
in_play_pitch_count
credited/final player used instead of starting player
future substitution flag
same-game future/current PA outcome
target class/code
split
quality
```

---

# 19. M3 Forbidden List

```text
prediction_date 당일 결과
future game result
future participation
future PA/BF/Pitches
future Team
target
future aggregate rate
coverage result as X
censoring result as X
purge result as X
season final snapshot
players.last_seen_*
whole-period role
post-hoc remaining schedule
horizon label boundary as automatic X
```

`horizon_id`는 v1에서 Task/Key이며 Historical X가 아니다.

---

# 20. 공식 라인업 / Predicted Lineup

현재 Feature Catalog v1에서 Historical 실제 공식 라인업은 기본 X에 포함하지 않는다.

추가 조건:

```text
announcement_time <= prediction_time
```

과 Source provenance를 증명할 수 있을 때만 별도 Feature Version으로 추가한다.

Predicted Lineup은 과거 출장/타순 Pattern에서 Prediction-time 이전 정보만 사용해 생성하는 별도 기능이며 실제 사후 라인업과 구분한다.

---

# 21. Bullpen Workload / Availability Proxy

향후 Project Scope에 포함될 수 있는 불펜 Feature는 공개 관측 이력에서 계산한다.

예:

```text
pitches_last_1d
pitches_last_3d
pitches_last_7d
appearances_last_7d
consecutive_days_pitched
days_since_last_appearance
recent_batters_faced
```

의미는 workload/fatigue/availability **Proxy**다.

`actual availability`, 의료 상태, 감독의 비공개 기용 의사를 나타내지 않는다.

이 Feature가 실제 M1 X에 추가되면 Feature Version과 정확한 산식을 Catalog Version으로 갱신한다.

---

# 22. 외부 보강 Feature

#29에서 수집한 외부 데이터는 Source provenance와 Prediction-time availability를 검증한 경우에만 X에 추가한다.

예:

- 일정
- 날씨
- 공개 엔트리/IL 상태

기본 Feature Catalog v1에 외부 값을 하드코딩하지 않는다.

채택 시 다음 정보를 Catalog에 추가한다.

```text
source
availability
join key
timestamp rule
null policy
license/terms basis
feature version
```

---

# 23. M4 / M5 확장 규칙

## 23.1 M4

#31에서 다음을 추가한다.

- M4 state grain
- prediction timestamp
- current state allowlist
- post-state forbidden
- WP/RE target
- WPA/Leverage 파생값 구분

## 23.2 M5

#30에서 다음을 추가한다.

- Pitch grain
- pre-pitch context allowlist
- target pitch 정보 forbidden
- pitch sequence availability
- `pitch_type`, `plate_x`, `plate_z`, Pitch result Target
- Target별 coverage/null policy

M4/M5 추가가 기존 M1/M2/M3 Catalog 의미를 변경하면 Catalog Version을 올린다.

---

# 24. Catalog Version 요약

```text
feature_catalog_version = feature_catalog_v1

feature_version:
- baseline_v1
- extended_v1
```

---

# 25. 검증 규칙

각 Feature Builder / Dataset Builder는 다음을 기계적으로 검증한다.

1. Output Key Unique
2. Join Cardinality
3. `max_source_game_date < prediction_date`
4. Same-day 결과 미사용
5. X allowlist 외 Column 차단
6. Forbidden Column 부재
7. Null dtype 보존
8. 입력 Row 순서 변경에 대한 결정성
9. 입력 Artifact SHA256 불변성
10. Content Fingerprint 결정성
11. Manifest에 Feature Version 기록

---

# 26. 완료 기준

이 Catalog는 다음 조건을 만족해야 한다.

- M1/M2/M3 Source Column의 Prediction-time 의미가 구분되어 있다.
- Raw PA 시작 선수와 Canonical credited/final 선수의 차이를 반영한다.
- Team/Player Historical Feature 산식과 Window가 정의되어 있다.
- Same-day/Doubleheader 금지가 명시되어 있다.
- Cold Start와 0 denominator가 정의되어 있다.
- nullable Pitching Outs를 0으로 만들지 않는다.
- Season Snapshot과 Player Master 미래 정보 사용을 금지한다.
- `baseline_v1`, `extended_v1` allowlist가 정의되어 있다.
- M1/M2/M3 forbidden list가 정의되어 있다.
- 공식 라인업/불펜/외부 데이터의 현재 경계가 명시되어 있다.
- M4/M5의 확장 책임이 #31/#30으로 명시되어 있다.
- `docs/prediction_dataset_contract.md`와 모순이 없다.
