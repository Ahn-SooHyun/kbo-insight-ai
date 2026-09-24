# 데이터 관리

이 디렉터리는 KBO Insight AI 프로젝트에서 사용하는 데이터를 관리한다.

## 디렉터리 구조

### `data/raw/`

외부 출처에서 다운로드한 원본 데이터를 저장한다.

Raw 데이터는 원본 상태를 유지하며, 전처리 또는 Feature Engineering 과정에서
파일을 수정하거나 덮어쓰지 않는다.

### `data/raw/hf_kbo_pbp/`

Hugging Face에서 제공되는 KBO Play-by-Play 원본 데이터를 저장할 위치다.

원본 데이터는 `scripts/download_hf_kbo_pbp.py`를 사용하여 다운로드하며,
다운로드한 Dataset revision과 파일 metadata는 `download_manifest.csv`에 기록한다.

### `data/interim/`

Raw 데이터에서 생성된 중간 처리 결과를 저장한다.

데이터 정제, 변환, 정규화 등 중간 단계의 결과는 Raw 데이터를 수정하지 않고
이 디렉터리에 별도로 저장한다.

### `data/interim/hf_kbo_pbp/`

Hugging Face KBO Play-by-Play Raw Dataset의 검증 결과를 저장한다.

검증은 다음 명령으로 실행한다.

```bash
python scripts/validate_hf_kbo_pbp.py
```

검증 결과는 다음 두 파일로 생성된다.

- `validation_report.csv`
  - 시즌별 파일 존재 및 로드 여부
  - row/column 수
  - `game_date` 시즌 일치 여부
  - 완전 중복 및 Pitch Key 중복
  - Critical column 결측
  - 주요 Pitch Tracking 컬럼 결측률
  - unique game/batter/pitcher 수
  - PASS/WARN/FAIL 상태
- `schema_report.csv`
  - 시즌별 실제 전체 컬럼
  - dtype
  - non-null/null 수
  - null rate
  - unique 수
  - sample value

검증 과정에서는 `data/raw/hf_kbo_pbp/`의 원본 파일을 수정하지 않는다.

`validation_report.csv`와 `schema_report.csv`는 실행 시 현재 Raw snapshot을
기준으로 다시 생성하며 append하지 않는다.

#### Plate Appearance 파생 테이블

검증된 Hugging Face KBO Play-by-Play Raw Dataset에서
한 행이 하나의 Plate Appearance를 나타내는 Canonical 파생 테이블을 생성한다.

생성은 다음 명령으로 실행한다.

```bash
python scripts/build_plate_appearances.py
```

출력 파일은 다음 위치에 생성된다.

```text
data/interim/hf_kbo_pbp/derived/plate_appearances.parquet
```

Plate Appearance Grain은 다음과 같다.

```text
(game_pk, at_bat_number)
```

변환 시 2023~2026 시즌 Raw Parquet을 시즌별로 순차 처리하고,
Raw의 `(game_pk, at_bat_number, pitch_number)` 순서를 기준으로
각 PA의 실제 첫 Row와 실제 마지막 Row를 선택한다.

PA 시작 상태는 실제 첫 Row에서 생성한다.

- `outs_before`
- `on_1b_before`
- `on_2b_before`
- `on_3b_before`
- `home_score_before`
- `away_score_before`
- `batting_score_before`
- `fielding_score_before`
- `score_diff_before`

PA 결과와 종료 상태는 실제 마지막 Row에서 생성한다.

- `event`
- `runs_scored`
- `post_outs`
- `post_on_1b`
- `post_on_2b`
- `post_on_3b`
- `post_home_score`
- `post_away_score`

`event`는 Raw의 `events` Category를 그대로 유지하며
ML Target Class로 별도 축약하지 않는다.

`events`가 null인 미완료 PA도 삭제하지 않고 유지하며,
이 경우 `pa_completed`는 `False`다.

타격 팀과 수비 팀은 `inning_topbot`을 기준으로 결정한다.

- `top`
  - `batting_team = away_team`
  - `fielding_team = home_team`
  - `is_home_batting = False`
- `bot`
  - `batting_team = home_team`
  - `fielding_team = away_team`
  - `is_home_batting = True`

`batting_score_before`, `fielding_score_before`,
`score_diff_before`도 같은 공격/수비 팀 관점을 기준으로 계산한다.

Pitch Summary는 Raw Row 수와 실제 Pitch 수를 구분한다.

- `pitch_rows`
  - 해당 PA에 포함된 전체 Raw Row 수
- `pitch_count`
  - `pitch_number > 0`인 실제 Pitch 수
- `ball_pitch_count`
  - 실제 Pitch 중 `type == "B"`인 수
- `strike_pitch_count`
  - 실제 Pitch 중 `type == "S"`인 수
- `in_play_pitch_count`
  - 실제 Pitch 중 `type == "X"`인 수

Pitch-less PA는 `pitch_number = 0`인 Raw Row를 유지한다.

따라서 Pitch-less PA에서는 다음과 같은 값이 가능하다.

```text
pitch_rows = 1
pitch_count = 0
ball_pitch_count = 0
strike_pitch_count = 0
in_play_pitch_count = 0
```

PA 도중 Batter가 교체된 경우 일반적으로 PA를 끝낸 Batter에게
해당 PA를 귀속한다.

단, 2 Strike 상태에서 Batter가 교체된 뒤 Strikeout으로 PA가 종료되면
Strikeout은 교체 전 starting Batter에게 귀속한다.

이 예외에서는 다음 컬럼을 모두 starting Batter 기준으로 맞춘다.

- `batter`
- `batter_name`
- `stand`

PA 생성 과정에서는 nullable 컬럼별 마지막 non-null 값이
서로 다른 Raw Row에서 섞이지 않도록 `groupby().last()` 방식으로
PA 종료 상태를 생성하지 않는다.

각 PA의 시작 상태와 종료 상태는 정렬된 Raw에서
실제 첫 Row와 실제 마지막 Row 자체를 명시적으로 선택하여 생성한다.

생성된 PA 수는 Hugging Face Dataset Card 등 외부에서 제공되는
고정 PA Count와 비교하여 assertion하지 않는다.

Production Validation은 각 시즌 Raw의 unique
`(game_pk, at_bat_number)` 수를 기준으로 한다.

다음 조건을 검증한다.

- 파생 PA 수와 Raw unique `(game_pk, at_bat_number)` 수가 일치
- `(game_pk, at_bat_number)` 중복 0건
- `pitch_rows >= pitch_count`
- `pitch_count`가 `ball_pitch_count`, `strike_pitch_count`,
  `in_play_pitch_count`의 합과 일치
- Output의 `season` 값이 처리 중인 시즌과 일치

최종 Output은 다음 순서로 결정적으로 정렬한다.

```text
season
game_date
game_pk
at_bat_number
```

주요 Output dtype은 다음 원칙으로 정규화한다.

- `game_date`
  - timezone이 없는 pandas datetime
  - 현재 Parquet round-trip 기준 `datetime64[us]`
- 시즌, Inning, Outs, Score, Pitch Count 계열
  - pandas nullable `Int64`
- `is_home_batting`, `pa_completed`
  - pandas nullable `boolean`
- 팀, 선수 ID/Name, Base Runner ID, `event`
  - pandas `string`

후속 ML Feature Engineering에서는 예측 시점 이후의 정보를
입력 Feature로 사용하지 않아야 한다.

특히 경기 전 또는 PA 시작 전 예측에 다음과 같은 PA 진행 이후 정보가
포함되지 않도록 주의한다.

- `event`
- `pa_completed`
- `runs_scored`
- `post_outs`
- `post_on_1b`
- `post_on_2b`
- `post_on_3b`
- `post_home_score`
- `post_away_score`
- `pitch_rows`
- `pitch_count`
- `ball_pitch_count`
- `strike_pitch_count`
- `in_play_pitch_count`

파생 테이블 생성 과정에서는 `data/raw/hf_kbo_pbp/`의
원본 Parquet을 수정하거나 덮어쓰지 않는다.

#### Game 및 Team Game 파생 테이블

Canonical Plate Appearance를 기반으로 경기 단위 및 팀-경기 단위의
Post-game Fact Table을 생성한다.

생성은 다음 명령으로 실행한다.

```bash
python scripts/build_game_tables.py
```

기본 입력은 다음 Canonical Plate Appearance다.

```text
data/interim/hf_kbo_pbp/derived/plate_appearances.parquet
```

출력 파일은 다음 위치에 생성된다.

```text
data/interim/hf_kbo_pbp/derived/games.parquet
data/interim/hf_kbo_pbp/derived/team_games.parquet
```

`games.parquet`은 한 행이 하나의 경기를 나타낸다.

Grain은 다음과 같다.

```text
game_pk
```

`team_games.parquet`은 한 행이 한 경기에서 한 팀이 수행한 결과를 나타낸다.

Grain은 다음과 같다.

```text
(game_pk, team)
```

각 경기는 Home Team과 Away Team 관점의 Row를 각각 하나씩 가지므로
`team_games.parquet`에는 경기당 정확히 2행이 존재한다.

Canonical Plate Appearance에는 원본 `home_team`, `away_team`이
직접 포함되지 않으므로 다음 규칙으로 팀을 복원한다.

```text
is_home_batting == True:
    home_team = batting_team
    away_team = fielding_team

is_home_batting == False:
    home_team = fielding_team
    away_team = batting_team
```

각 경기의 `game_date`, `season`, `home_team`, `away_team`은
모든 PA에서 하나의 값으로 일관되어야 한다.

Game 내부 PA는 `at_bat_number` 기준으로 정렬한다.

Final State는 nullable 컬럼별 마지막 non-null 값을 조합하지 않고,
정렬된 실제 마지막 PA Row 자체에서 읽는다.

Final Score는 실제 마지막 PA Row의 다음 값을 기준으로 한다.

```text
final_home_score = last post_home_score
final_away_score = last post_away_score
```

미완료 PA가 경기의 마지막 PA인 경우에도 해당 실제 마지막 Row를 사용한다.
완료 PA만 남긴 뒤 Final State를 계산하지 않는다.

Final Score는 `runs_scored`를 이용해 독립적으로 다시 계산하고
Post-score와 Reconciliation한다.

```text
expected_final_home_score
= first_home_score_before
+ sum(runs_scored where is_home_batting == True)

expected_final_away_score
= first_away_score_before
+ sum(runs_scored where is_home_batting == False)
```

마지막 PA의 `post_home_score`, `post_away_score`와 위 계산 결과가
일치하지 않으면 Game Table 생성을 실패시킨다.

Score 및 `runs_scored`의 결측값을 임의로 0으로 보정하지 않는다.

경기 결과는 Final Score를 기준으로 생성한다.

```text
home_win = final_home_score > final_away_score
away_win = final_away_score > final_home_score
is_tie = final_home_score == final_away_score
```

Tie를 정상적인 경기 결과로 지원한다.

Tie인 경우 다음 값은 nullable string의 null 값이다.

```text
winner_team = <NA>
loser_team = <NA>
```

`innings_played`는 해당 경기에서 관측된 `inning`의 최댓값이다.

따라서 Home Team이 앞서 Bottom 9 없이 종료된 경기도
`innings_played = 9`이며, 10회까지 진행된 연장 경기는
`innings_played = 10`이다.

Game Summary의 주요 Count는 다음 의미를 가진다.

- `pitch_rows`
  - 경기 내 모든 Canonical PA의 `pitch_rows` 합
- `pitch_count`
  - 경기 내 모든 Canonical PA의 `pitch_count` 합
- `plate_appearances`
  - 경기 내 Canonical PA 전체 Row 수
  - 미완료 PA도 포함
- `home_plate_appearances`
  - `is_home_batting == True`인 PA 수
- `away_plate_appearances`
  - `is_home_batting == False`인 PA 수
- `total_runs`
  - `final_home_score + final_away_score`

다음 관계를 항상 검증한다.

```text
plate_appearances
= home_plate_appearances + away_plate_appearances
```

`team_games.parquet`은 검증을 통과한 `games.parquet`의 내용을
Home/Away Team 관점으로 Long Format 변환하여 생성한다.

PA를 다시 독립적으로 집계하여 Team Game 값을 만들지 않으므로
Game Fact와 Team Game Fact 사이의 동일한 경기 사실이 서로 다른
집계 경로에서 불일치하는 것을 방지한다.

2026 시즌은 진행 중인 Partial Season Snapshot으로 취급한다.

현재 데이터에서 관측된 경기만 Game 및 Team Game Fact로 생성하며
`season_complete`, `final_rank`와 같은 완료 시즌 의미를
임의로 부여하지 않는다.

`games.parquet`과 `team_games.parquet`은 경기 종료 이후의 결과를 포함하는
Canonical Post-game Fact Table이다.

따라서 Pregame Prediction의 입력 Feature로 현재 경기의 Final Score,
승패, 경기 전체 PA/Pitch Count와 같은 Post-game 정보를 직접 사용하면
Data Leakage가 발생한다.

향후 경기 전 Feature를 생성할 때는 기본적으로 다음 시간 조건을 지켜야 한다.

```text
source game_date < prediction game_date
```

특히 같은 날짜에 열린 경기나 Doubleheader를 `game_pk` 등의 임의 순서로
정렬한 뒤 앞선 Row를 이미 종료된 과거 경기로 간주하지 않는다.

같은 날짜의 경기 순서를 실제 이용 가능 시점으로 증명할 별도 정보가 없는 한
해당 날짜의 다른 경기를 Pregame Historical Source로 사용하지 않는다.

#### Player Game Batting 파생 테이블

Canonical Plate Appearance를 기반으로 선수-경기 단위의
Post-game 타격 Fact Table을 생성한다.

생성은 다음 명령으로 실행한다.

```bash
python scripts/build_player_game_batting.py
```

기본 입력은 다음 Canonical Plate Appearance다.

```text
data/interim/hf_kbo_pbp/derived/plate_appearances.parquet
```

출력 파일은 다음 위치에 생성된다.

```text
data/interim/hf_kbo_pbp/derived/player_game_batting.parquet
```

Player Game Batting Grain은 다음과 같다.

```text
(game_pk, batter)
```

공식식 타격 집계에는 Canonical Plate Appearance 중
다음 조건을 만족하는 완료 PA만 사용한다.

```text
event IS NOT NULL
```

`event`가 null인 미완료 PA는 OUT 등 다른 결과로 보정하지 않으며,
`pa`에도 포함하지 않는다.

특정 `(game_pk, batter)`에 완료 PA가 하나도 없으면
해당 Player Game Row를 별도로 생성하지 않는다.

선수 귀속은 Canonical Plate Appearance의 `batter`,
`batter_name`을 Source of Truth로 그대로 사용한다.

따라서 PA 도중 Batter 교체 및 2 Strike 상태의
Strikeout 귀속 규칙을 이 단계에서 Raw Pitch로 돌아가
다시 계산하지 않는다.

Player Game의 팀 Context는 다음과 같다.

```text
team = batting_team
opponent = fielding_team
is_home = is_home_batting
```

동일한 `(game_pk, batter)` 안에서는 다음 Context가
하나의 값으로 일관되어야 한다.

```text
game_date
season
batter_name
team
opponent
is_home
```

Context가 충돌하면 첫 값이나 마지막 값을 임의로 선택하지 않고
Player Game Batting 생성을 실패시킨다.

`batter`, `team`, `opponent`에는 null 또는 빈 문자열을 허용하지 않으며,
`team == opponent`인 Row도 허용하지 않는다.

같은 선수가 다른 경기에서 다른 Team으로 기록되는 것은
Trade 등의 상황이 있을 수 있으므로 허용한다.

완료 PA의 `event`는 다음 15개 값만 허용한다.

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

지원하지 않는 non-null Event를 OTHER나 OUT으로 합치지 않고
Player Game Batting 생성을 실패시킨다.

Output Event Count Mapping은 다음과 같다.

```text
single                -> single
double                -> double
triple                -> triple
home_run               -> hr
walk                   -> bb
hit_by_pitch           -> hbp
strikeout              -> so
sac_fly                -> sf
sac_bunt               -> sh
double_play            -> double_play
triple_play            -> triple_play
field_error            -> field_error
fielders_choice        -> fielders_choice
catcher_interference   -> catcher_interference
```

`field_out`은 별도 Output 컬럼을 생성하지 않지만
`pa`와 `ab`에는 정상적으로 포함한다.

Player Game의 `pa`는 완료 PA 수다.

Hit 수는 다음 공식으로 계산한다.

```text
h = single + double + triple + hr
```

At Bat에서 제외되는 Event는 다음과 같다.

```text
walk
hit_by_pitch
sac_bunt
sac_fly
catcher_interference
```

따라서 AB는 다음 공식으로 계산한다.

```text
ab
= pa
- bb
- hbp
- sh
- sf
- catcher_interference
```

Total Bases는 다음 공식으로 계산한다.

```text
tb
= 1 * single
+ 2 * double
+ 3 * triple
+ 4 * hr
```

Rate Stat은 다음 공식으로 계산하고
각 결과를 소수점 셋째 자리까지 반올림한다.

```text
avg = h / ab

obp
= (h + bb + hbp)
  / (ab + bb + hbp + sf)

slg = tb / ab

ops = obp + slg
```

`catcher_interference`는 AB에서는 제외하지만
OBP denominator에는 포함하지 않는다.

0 Denominator를 임의로 0으로 대체하지 않는다.

```text
AB = 0:
    AVG = <NA>
    SLG = <NA>

OBP denominator = 0:
    OBP = <NA>

OPS:
    OBP와 SLG가 모두 정의된 경우에만 계산
    하나라도 정의되지 않으면 <NA>
```

예를 들어 Walk만 존재하는 Player Game은 `AB = 0`이므로
`AVG`와 `SLG`는 정의되지 않지만,
OBP denominator가 존재하므로 `OBP`는 계산할 수 있다.

다만 `SLG`가 정의되지 않으므로 해당 Row의 `OPS`는 `<NA>`로 유지한다.

Player Game Batting에서는 다음 값을 생성하지 않는다.

```text
rbi
runs
batter_runs
```

Canonical PA의 `runs_scored`는 해당 PA 전체에서 발생한 득점 수이며,
해당 Batter의 RBI나 Batter 본인의 득점을 직접 의미하지 않는다.

따라서 `runs_scored`의 이름만 변경하여
RBI 또는 Batter Runs로 사용하지 않는다.

Player Game Batting 생성 후 다음 Grain을 검증한다.

```text
(game_pk, batter) Duplicate = 0
```

완료 PA 전체에 대해서는 다음 관계를 검증한다.

```text
sum(player_game_batting.pa)
=
Canonical PA의 event IS NOT NULL Row 수
```

가능한 경우 경기별로도 완료 PA 수의 동일성을 검증한다.

Player Game Output의 각 Event Count 합계는
Canonical PA의 Event Cross-tab과 일치해야 한다.

별도 Output 컬럼이 없는 `field_out`을 포함하여
15개 Event 전체의 합은 `pa`와 일치해야 한다.

각 Player Game Row에서는 다음 공식도 다시 검증한다.

```text
h
= single
+ double
+ triple
+ hr

ab
= pa
- bb
- hbp
- sh
- sf
- catcher_interference

tb
= single
+ 2 * double
+ 3 * triple
+ 4 * hr
```

최종 Output 컬럼은 다음과 같다.

```text
game_pk
game_date
season
batter
batter_name
team
opponent
is_home
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
avg
obp
slg
ops
```

최종 Output은 다음 순서로 결정적으로 정렬한다.

```text
season
game_date
game_pk
batter
```

Input Canonical PA의 Row 순서가 달라져도
동일한 값, Row 순서 및 dtype의 Output을 생성할 수 있어야 한다.

Byte-level Parquet hash의 동일성은 요구하지 않는다.

주요 Output dtype은 다음 원칙으로 정규화한다.

- `game_pk`
  - pandas `string`
- `game_date`
  - timezone이 없는 pandas datetime
  - Parquet round-trip 기준 `datetime64[us]`
- `season`
  - pandas nullable `Int64`
- `batter`, `batter_name`
  - pandas `string`
- `team`, `opponent`
  - pandas `string`
- `is_home`
  - pandas nullable `boolean`
- `pa`, `ab`, `h`
  - pandas nullable `Int64`
- `single`, `double`, `triple`, `hr`
  - pandas nullable `Int64`
- `bb`, `hbp`, `so`, `sf`, `sh`
  - pandas nullable `Int64`
- `tb`
  - pandas nullable `Int64`
- `double_play`, `triple_play`
  - pandas nullable `Int64`
- `field_error`, `fielders_choice`, `catcher_interference`
  - pandas nullable `Int64`
- `avg`, `obp`, `slg`, `ops`
  - pandas nullable `Float64`

Player ID인 `batter`는 숫자처럼 보이더라도
문자열 의미를 유지한다.

Player Game Batting 생성 과정에서는
입력 `plate_appearances.parquet`을 수정하거나 덮어쓰지 않는다.

CLI에서 Output 경로를 변경하는 경우에도
`data/raw/` 내부에 Derived Output을 생성하지 않는다.

또한 Output 경로를 입력
`plate_appearances.parquet`과 동일하게 지정하는 것을 허용하지 않는다.

Parquet은 임시 파일에 먼저 저장한 뒤
모든 생성 및 검증이 성공한 경우 최종 Output 경로로 교체한다.

`player_game_batting.parquet`은 해당 경기 종료 이후에 확정되는
Canonical Post-game Fact Table이다.

따라서 해당 경기의 다음과 같은 값을
그 경기의 Pregame Feature로 직접 사용하면 Data Leakage가 발생한다.

- `pa`
- `ab`
- `h`
- `single`
- `double`
- `triple`
- `hr`
- `bb`
- `hbp`
- `so`
- `sf`
- `sh`
- `tb`
- `avg`
- `obp`
- `slg`
- `ops`

향후 Historical Pregame Feature를 생성할 때는
기본적으로 다음 시간 조건을 지켜야 한다.

```text
source game_date < prediction game_date
```

특히 같은 날짜에 열린 경기나 Doubleheader를
`game_pk` 등의 임의 순서로 정렬한 뒤
앞선 경기 결과를 이미 이용 가능한 과거 정보로 간주하지 않는다.

같은 날짜의 경기 순서를 실제 이용 가능 시점으로 증명할
별도 정보가 없는 한, 해당 날짜의 다른 Player Game 결과를
현재 경기의 Pregame Historical Source로 사용하지 않는다.

### `data/processed/`

모델 학습 및 분석에 사용할 최종 가공 데이터를 저장한다.

### `data/external/`

기상 데이터 등 별도의 외부 출처에서 수집하는 보조 데이터를 저장한다.

## 데이터 관리 원칙

- Raw 데이터는 다운로드 이후 수정하지 않는다.
- Raw 데이터와 파생 데이터를 반드시 분리한다.
- 대용량 데이터 파일은 Git에 Commit하지 않는다.
- 전처리 결과는 `data/interim/` 또는 `data/processed/`에 저장한다.
- 데이터의 출처와 재현성을 관리할 수 있도록 향후 provenance 정보를 기록한다.
- provenance에는 출처, URL, 다운로드 시각, Dataset revision, 시즌, 파일 정보,
  SHA256, License 등의 정보를 포함할 수 있다.
