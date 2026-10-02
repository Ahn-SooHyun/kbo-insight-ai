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

#### Player Game Pitching 파생 테이블

KBO Pitch-level Raw Dataset과 Canonical Plate Appearance를 함께 사용하여
선수-경기 단위의 Post-game 투구 Fact Table을 생성한다.

생성은 다음 명령으로 실행한다.

```bash
python scripts/build_player_game_pitching.py
```

기본 입력은 다음 파일이다.

```text
data/raw/hf_kbo_pbp/2023.parquet
data/raw/hf_kbo_pbp/2024.parquet
data/raw/hf_kbo_pbp/2025.parquet
data/raw/hf_kbo_pbp/2026.parquet

data/interim/hf_kbo_pbp/derived/plate_appearances.parquet
```

출력 파일은 다음 위치에 생성된다.

```text
data/interim/hf_kbo_pbp/derived/player_game_pitching.parquet
```

Player Game Pitching Grain은 다음과 같다.

```text
(game_pk, pitcher)
```

Output Key 집합은 Raw Pitch-level Dataset에 실제 등장한
`(game_pk, pitcher)`를 기준으로 한다.

따라서 한 PA를 끝내지 못했더라도 실제 Raw Row에 등장한 Pitcher는
Player Game Pitching Output에 포함된다.

Player Game Pitching은 Raw Pitch Row와 Canonical PA를 서로 다른 목적으로 사용한다.

Raw Pitch Row에서는 각 Row에 실제 기록된 `pitcher`를 기준으로 다음 값을 계산한다.

```text
pitch_rows
pitches
ball_pitch_count
strike_pitch_count
in_play_pitch_count
avg_release_speed_kmh
```

Canonical Plate Appearance에서는 PA 결과와 시작/종료 Outs를 기준으로
다음 값을 계산한다.

```text
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
```

PA 도중 Pitcher가 교체될 수 있으므로 Raw Pitch Count를
Canonical PA의 최종 Pitcher에게 통째로 귀속하지 않는다.

예를 들어 한 PA에서 다음과 같이 Pitcher가 교체되었다면

```text
Pitch 1, 2 -> Pitcher A
Pitch 3, 4 -> Pitcher B
```

Raw Pitch Count는 다음처럼 분리한다.

```text
Pitcher A.pitches += 2
Pitcher B.pitches += 2
```

반면 완료 PA 결과와 `batters_faced_completed`는
Canonical Plate Appearance의 `pitcher`에게 귀속한다.

`pitch_rows`와 `pitches`는 의미가 다르다.

```text
pitch_rows
= 해당 Pitcher의 Raw Row 전체 수

pitches
= pitch_number > 0인 실제 Pitch 수
```

Pitch-less PA는 `pitch_number = 0`인 하나의 Raw Row로 존재할 수 있다.

따라서 다음과 같은 값이 정상이다.

```text
pitch_rows = 1
pitches = 0
```

Pitch-less Row는 `pitch_rows`에는 포함하지만 `pitches`에는 포함하지 않는다.

실제 Pitch의 Pitch Result Count는 `pitch_number > 0`인 Row에 대해서만
다음과 같이 계산한다.

```text
ball_pitch_count
= count(type == "B")

strike_pitch_count
= count(type == "S")

in_play_pitch_count
= count(type == "X")
```

각 Player Game에서는 다음 관계를 만족해야 한다.

```text
pitches
=
ball_pitch_count
+ strike_pitch_count
+ in_play_pitch_count
```

실제 Pitch의 `type`이 null이거나 `B`, `S`, `X` 이외의 값이면
Player Game Pitching 생성을 실패시킨다.

`pitch_number = 0`인 Pitch-less Row의 null `type`은 허용한다.

Pitch Result Count용 Boolean Mask를 만들 때는 Pitch-less Row의 null `type`을
분류용 임시 값으로만 정규화하며 원본 Raw 값을 수정하지 않는다.

평균 구속은 다음 컬럼에 저장한다.

```text
avg_release_speed_kmh
```

계산 대상은 다음 조건을 모두 만족하는 Raw Row다.

```text
pitch_number > 0
release_speed_kmh IS NOT NULL
```

`release_speed_kmh`의 단위는 km/h다.

결측 구속을 0으로 보정하지 않으며,
유효한 구속 값이 하나도 없는 Player Game은 다음 값을 유지한다.

```text
avg_release_speed_kmh = <NA>
```

평균 구속은 계산 후 소수점 셋째 자리까지 반올림한다.

Completed Batter Faced는 Canonical Plate Appearance 중 다음 조건을 만족하는 PA 수다.

```text
event IS NOT NULL
```

따라서 다음과 같이 계산한다.

```text
batters_faced_completed
= 완료 PA 수
```

`event IS NULL`인 미완료 PA는 `batters_faced_completed`에서 제외한다.

Allowed Event Mapping은 다음과 같다.

```text
single         -> single_allowed
double         -> double_allowed
triple         -> triple_allowed
home_run       -> hr_allowed
walk           -> bb_allowed
hit_by_pitch   -> hbp_allowed
strikeout      -> so
sac_fly        -> sf
sac_bunt       -> sh
```

Hit Allowed는 다음 공식으로 계산한다.

```text
hits_allowed
=
single_allowed
+ double_allowed
+ triple_allowed
+ hr_allowed
```

다음 완료 Event도 `batters_faced_completed`에는 정상적으로 포함되지만,
이번 최소 Output에서는 별도 Count 컬럼을 생성하지 않는다.

```text
field_out
double_play
triple_play
field_error
fielders_choice
catcher_interference
```

`outs_recorded`는 각 PA의 시작 Outs와 종료 후 Outs 차이를 이용한다.

```text
pa_outs_recorded
=
post_outs - outs_before
```

연속 PA의 `post_outs`를 `diff()`하여 계산하지 않는다.

각 PA 내부의 Before/Post 상태만 사용하므로 다음 Half-inning에서
`outs_before = 0`으로 Reset되는 것은 정상적으로 처리된다.

예시는 다음과 같다.

```text
outs_before = 0, post_outs = 1
-> 1 Out

outs_before = 0, post_outs = 2
-> 2 Outs

outs_before = 0, post_outs = 3
-> 3 Outs

outs_before = 2, post_outs = 3
-> 1 Out
```

허용되는 PA Out Delta는 다음과 같다.

```text
0 <= pa_outs_recorded <= 3
```

음수 또는 3보다 큰 Out Delta가 발견되면 생성을 실패시킨다.

`event IS NULL`인 미완료 PA도 주루사 등으로
`post_outs > outs_before`가 될 수 있다.

따라서 미완료 PA는 Completed BF에서는 제외하지만
`outs_recorded` 계산에서는 삭제하지 않는다.

한 PA에 여러 Pitcher가 등장하면서 Out Delta가 발생하는 경우에는
PA 전체 Out Delta를 마지막 Pitcher에게 무조건 귀속하지 않는다.

Raw Pitch Row의 실제 마지막 Pitcher와 Canonical PA의 Pitcher가 동일한지 먼저 확인한다.
두 값이 다르면 PA 결과 귀속 자체가 일치하지 않는 것이므로 생성을 실패시킨다.

마지막 Raw Pitcher와 Canonical PA Pitcher가 일치하는 경우에는
PA 종료 Event 자체로 안전하게 설명할 수 있는 최소 Out 수와
PA 전체 `pa_outs_recorded`를 비교한다.

현재 종료 Event로 안전하게 귀속할 수 있는 Out 수는 다음과 같다.

```text
strikeout       -> 1
field_out       -> 1
sac_bunt        -> 1
sac_fly         -> 1
fielders_choice -> 1
double_play     -> 2
triple_play     -> 3
```

PA 전체 Out Delta가 위 종료 Event의 Out 수와 정확히 일치하면
해당 Out Delta를 Canonical PA의 Pitcher에게 귀속한다.

반대로 다음과 같은 경우에는 공개 Raw만으로 Out 발생 시점이
투수 교체 전인지 후인지 안전하게 복원할 수 없다고 판단한다.

```text
Multi-pitcher PA
AND
pa_outs_recorded > 0
AND
(
    Event가 위 종료 Event 목록에 없음
    OR
    pa_outs_recorded != 종료 Event로 설명 가능한 Out 수
)
```

예를 들어 다음 상태는 최종 `field_out` 외에 PA 진행 중 별도 주루 Out이
포함되었을 수 있으므로 두 Out을 모두 마지막 Pitcher에게 귀속하지 않는다.

```text
event = field_out
pa_outs_recorded = 2
```

Pitch-level Raw는 `outs_when_up`을 PA 시작 상태로 유지하고
`post_outs`는 PA 종료 상태로 제공하므로, PA 중간의 별도 주루 Out이
투수 교체 전인지 후인지 최종 Parquet만으로 항상 구분할 수 없다.

이처럼 투수별 Out을 정확히 분리할 수 없는 PA가 발견되면
빌드 전체를 실패시키거나 임의의 투수에게 Out을 몰아주지 않는다.

대신 해당 PA에 실제 등장한 모든 `(game_pk, pitcher)`의
`outs_recorded`를 pandas nullable `Int64`의 `<NA>`로 유지한다.

한 Pitcher가 같은 경기에서 다른 PA의 확정 Out을 기록했더라도,
동일 Player Game 안에 위 불확실 PA가 하나라도 포함되면
그 Player Game의 최종 `outs_recorded` 전체를 `<NA>`로 둔다.
부분 합계를 정확한 경기 투구 Out처럼 오해하지 않도록 하기 위한 처리다.

다른 Pitcher의 Player Game에는 이 불확실성을 전파하지 않는다.

따라서 `outs_recorded`의 의미는 다음과 같다.

```text
정수:
    해당 Player Game의 Out 수를 현재 Raw/Canonical PA로 안전하게 확정 가능

<NA>:
    Multi-pitcher PA 내부의 별도 Out 발생 시점을 투수별로 안전하게 분리 불가
```

이 규칙은 PA 시작/종료 상태에서 관측된 Defensive Out을 보수적으로 집계하기 위한
불확실성 처리 규칙이다. Inherited Runner의 실점 책임, `runs_allowed`,
`earned_runs`, ERA를 복원하기 위한 공식 책임 투수 규칙으로 사용하지 않는다.

Pitcher Team Context는 Fielding Team 관점으로 계산한다.

Raw에서는 다음과 같다.

```text
inning_topbot = top:
    team = home_team
    opponent = away_team
    is_home = True

inning_topbot = bot:
    team = away_team
    opponent = home_team
    is_home = False
```

Canonical PA에서는 같은 의미를 다음처럼 계산한다.

```text
team = fielding_team
opponent = batting_team
is_home = NOT is_home_batting
```

Raw와 Canonical PA의 Pitcher Context는 서로 교차 검증한다.

동일한 `(game_pk, pitcher)` 안에서는 다음 값이 하나로 일관되어야 한다.

```text
game_date
season
team
opponent
is_home
```

`pitcher_name`은 null을 제외한 non-null 값끼리 충돌하지 않아야 한다.

같은 Pitcher가 다른 경기에서 다른 Team으로 기록되는 것은
Trade 등의 상황이 있을 수 있으므로 허용한다.

Player ID인 `pitcher`는 숫자처럼 보이더라도 문자열 의미를 유지한다.

Player Game Pitching 생성 후 최소 다음 Reconciliation을 검증한다.

```text
(game_pk, pitcher) Duplicate = 0

sum(output.pitch_rows)
=
Raw 전체 Row 수

sum(output.pitches)
=
Raw의 pitch_number > 0 Row 수

sum(
    ball_pitch_count
    + strike_pitch_count
    + in_play_pitch_count
)
=
sum(output.pitches)

sum(output.batters_faced_completed)
=
Canonical PA의 event IS NOT NULL Row 수

hits_allowed
=
single_allowed
+ double_allowed
+ triple_allowed
+ hr_allowed

outs_recorded가 모두 non-null인 경우:

sum(output.outs_recorded)
=
sum(Canonical PA post_outs - outs_before)

outs_recorded에 <NA>가 존재하는 경우:

- <NA>가 아닌 Player Game은 Canonical PA 기준 기대 Out과 Key별로 일치
- 불확실 Multi-pitcher PA에 참여한 Player Game만 outs_recorded = <NA>
- 전체 Out 합계를 결측값을 제외한 값으로 공식 합계처럼 비교하지 않음
```

가능한 경우 위 Count는 경기별로도 다시 교차 검증한다.

또한 Raw와 Canonical PA의 `(game_pk, at_bat_number)` 집합,
PA별 `pitch_rows`, `pitch_count`가 일치하는지 확인한다.

Canonical PA의 `(game_pk, pitcher)`가 Raw Pitcher Key에 존재하지 않으면
새로운 Output Key를 임의로 생성하지 않고 실패시킨다.

최종 Output 컬럼은 다음과 같다.

```text
game_pk
game_date
season
pitcher
pitcher_name
team
opponent
is_home
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
avg_release_speed_kmh
```

최종 Output은 다음 순서로 결정적으로 정렬한다.

```text
season
game_date
game_pk
pitcher
```

Input Raw/Canonical PA의 Row 순서가 달라져도
동일한 값, Row 순서 및 dtype의 Output을 생성할 수 있어야 한다.

Byte-level Parquet hash의 동일성은 요구하지 않는다.

주요 dtype은 다음과 같다.

```text
game_pk                    string
game_date                  datetime64[us]
season                     Int64
pitcher                    string
pitcher_name               string
team                       string
opponent                   string
is_home                    boolean
pitch_rows                 Int64
pitches                    Int64
batters_faced_completed    Int64
hits_allowed               Int64
single_allowed             Int64
double_allowed             Int64
triple_allowed             Int64
hr_allowed                 Int64
bb_allowed                 Int64
hbp_allowed                Int64
so                         Int64
sf                         Int64
sh                         Int64
outs_recorded              Int64  # nullable, 불확실 Multi-pitcher PA가 있으면 <NA>
ball_pitch_count           Int64
strike_pitch_count         Int64
in_play_pitch_count        Int64
avg_release_speed_kmh      Float64
```

이번 최소 구현에서는 다음 값을 생성하지 않는다.

```text
runs_scored_while_pitching
earned_runs
era
runs_allowed
```

Canonical PA의 `runs_scored`는 해당 PA 동안 발생한 전체 득점이며,
Inherited Runner의 책임 투수와 현재 Pitcher가 다를 수 있다.

따라서 `runs_scored`의 이름만 변경하여 공식 `runs_allowed`,
`earned_runs` 또는 ERA 계산에 사용하지 않는다.

Player Game Pitching 생성 과정에서는 Raw 시즌 Parquet과
입력 `plate_appearances.parquet`을 수정하거나 덮어쓰지 않는다.

CLI에서 Output 경로를 변경하더라도 `data/raw/` 또는 지정된 Raw 입력
디렉터리 내부에 Derived Output을 생성하지 않는다.

또한 Output 경로를 입력 `plate_appearances.parquet` 또는
시즌별 Raw Parquet과 동일하게 지정하는 것을 허용하지 않는다.

Parquet은 모든 생성 및 Validation이 성공한 뒤
임시 파일에 먼저 기록하고 최종 Output 경로로 원자적으로 교체한다.

`player_game_pitching.parquet`은 해당 경기 종료 이후에 확정되는
Canonical Post-game Fact Table이다.

따라서 현재 경기의 Pitch Count, BF, Allowed Event, Outs,
평균 구속 등의 값을 해당 경기 Pregame Feature로 직접 사용하면
Data Leakage가 발생한다.

향후 Historical Pregame Feature를 생성할 때는 기본적으로
다음 시간 조건을 적용한다.

```text
source game_date < prediction game_date
```

같은 날짜의 경기 또는 Doubleheader는 실제 경기 시작 시각 등
이용 가능 순서를 증명할 정보가 없는 한 `game_pk` 등의 임의 정렬을 이용해
앞선 경기를 이미 종료된 Historical Source로 간주하지 않는다.

#### Player Metadata 파생 테이블

Canonical Plate Appearance, Player Game Batting, Player Game Pitching에
등장하는 Batter/Pitcher ID를 하나의 Canonical Player Entity로 통합한다.

생성은 다음 명령으로 실행한다.

```bash
python scripts/build_players.py
```

기본 입력은 다음 세 파일이다.

```text
data/interim/hf_kbo_pbp/derived/plate_appearances.parquet
data/interim/hf_kbo_pbp/derived/player_game_batting.parquet
data/interim/hf_kbo_pbp/derived/player_game_pitching.parquet
```

출력 파일은 다음 위치에 생성된다.

```text
data/interim/hf_kbo_pbp/derived/players.parquet
```

Player Metadata Grain은 다음과 같다.

```text
player_id
```

`player_id`는 pandas `string`으로 유지한다.

Player ID를 정수로 변환하지 않으므로 Leading Zero가 존재하는 ID도
문자열 의미를 유지한다.

null, 빈 문자열, 공백만 있는 Player ID는 허용하지 않는다.

Player Metadata의 Key 집합은 다음 Source에서 관측된 유효 Player ID의
합집합과 정확히 일치해야 한다.

```text
Plate Appearance.batter
Plate Appearance.pitcher
Player Game Batting.batter
Player Game Pitching.pitcher
```

Plate Appearance는 미완료 PA도 Player 관측에 포함한다.

따라서 Player Metadata 생성 과정에서는 PA의 `event` 존재 여부로
Batter/Pitcher를 필터링하지 않는다.

Player Game Pitching은 Raw Pitch에 실제 등장한 Pitcher를 기반으로 생성되므로,
PA를 완료하지 못한 중간 교체 Pitcher도 Player Metadata에 포함될 수 있다.

Player Role은 전체 관측 이력의 합집합으로 계산한다.

```text
is_batter
= 어떤 Canonical Source에서든 Batter로 한 번 이상 관측

is_pitcher
= 어떤 Canonical Source에서든 Pitcher로 한 번 이상 관측
```

따라서 다음 세 형태가 모두 가능하다.

```text
is_batter=True,  is_pitcher=False
is_batter=False, is_pitcher=True
is_batter=True,  is_pitcher=True
```

두 Role이 모두 False인 Player Row는 생성하지 않는다.

이름은 Source에서 null일 수 있다.

Player ID가 유효하면 이름이 null이어도 Player Entity 자체는 유지한다.

Name 정규화에서는 leading/trailing whitespace만 제거한다.

내부 공백, 한글/영문 표기, 개명 여부 등을 추정하여 자동 수정하거나
서로 다른 Name Variant를 임의로 병합하지 않는다.

빈 문자열 또는 공백만 있는 Name은 valid Name Variant로 사용하지 않는다.

`name_variants`는 같은 `player_id`에서 관측된 모든 distinct non-null Name을
숨기지 않고 보존한다.

Parquet round-trip과 결정성을 단순하게 유지하기 위해
정렬된 UTF-8 JSON Array 문자열을 사용한다.

예:

```text
["김민수"]
["김민수", "김민수(개명전)"]
[]
```

정책은 다음과 같다.

```text
unique
deterministic lexical sort
ensure_ascii=False
valid Name이 없으면 []
dtype = string
```

동일 사실이 Plate Appearance와 Player Game 양쪽에 중복될 수 있으므로,
Source Row 수 자체를 Name 빈도 가중치로 사용하지 않는다.

최소 다음 사실 단위로 중복을 제거한 뒤 Name Metadata를 계산한다.

```text
player_id
game_pk
game_date
season
role
player_name
```

`display_name`은 다음 규칙으로 결정한다.

1. valid non-null Name Observation만 사용한다.
2. 가장 최근 `game_date`에 관측된 Name을 우선한다.
3. 가장 최근 날짜에 Variant가 하나면 해당 Name을 사용한다.
4. 같은 가장 최근 날짜에 여러 Variant가 있으면
   lexicographical ascending 첫 값을 사용한다.
5. valid Name이 없으면 `<NA>`로 유지한다.

따라서 Source 종류나 중복 Row 수가 `display_name`을 결정하지 않는다.

Player 관측 기간은 세 Canonical Input 전체에서 계산한다.

```text
first_seen_date = min(game_date)
last_seen_date  = max(game_date)
```

두 Date 컬럼은 timezone 없는 다음 dtype으로 정규화한다.

```text
datetime64[us]
```

모든 Player에서 다음 조건을 만족해야 한다.

```text
first_seen_date <= last_seen_date
```

`first_seen_season`과 `last_seen_season`은 Player 전체 Season의
단순 min/max가 아니다.

각각 `first_seen_date`, `last_seen_date`에 실제 연결된 Observation의
Season을 사용한다.

같은 Player의 같은 boundary date에 서로 다른 Season이 관측되면
Source Context Conflict로 처리하여 생성을 실패시킨다.

최종 dtype 계약은 다음과 같다.

```text
player_id             string
display_name          string
name_variants         string
is_batter             boolean
is_pitcher            boolean
first_seen_date       datetime64[us]
last_seen_date        datetime64[us]
first_seen_season     Int64
last_seen_season      Int64
```

`display_name`은 nullable string이다.

`name_variants`는 null 대신 항상 유효한 JSON Array 문자열을 가진다.

최종 Output은 `player_id` ascending 순서로 안정 정렬한다.

입력 Row 순서가 바뀌거나 동일 Snapshot에서 Builder를 다시 실행해도
값, Row 순서, dtype, `name_variants` JSON 순서는 동일해야 한다.

Byte-level Parquet Hash 동일성까지 요구하지 않는다.

Player Metadata에는 다음과 같은 단일 Team 속성을 저장하지 않는다.

```text
team
current_team
latest_team
first_team
last_team
```

선수는 시즌 중 Trade가 가능하며 Historical Team 관계가 시간에 따라
변할 수 있기 때문이다.

Player-Team 관계가 필요한 경우 다음 Event-time Fact Table에서
해당 시점의 관계를 조회한다.

```text
plate_appearances.parquet
player_game_batting.parquet
player_game_pitching.parquet
```

Player Master는 안정적인 Player ID, Name, Role, 관측 기간만 담당한다.

Player Metadata 생성 과정에서는 세 Canonical Input을 수정하거나
덮어쓰지 않는다.

Output 경로를 `data/raw/` 내부에 지정할 수 없으며,
다음 세 Input 경로와 동일하게 지정할 수도 없다.

```text
plate_appearances.parquet
player_game_batting.parquet
player_game_pitching.parquet
```

Parquet은 모든 Build/Validation을 통과한 뒤 임시 파일에 기록하고,
round-trip 검증 후 최종 경로로 원자적으로 교체한다.

#### Player / Team Season Snapshot 파생 테이블

Player Game Batting, Player Game Pitching, Team Game Canonical Fact Table을
현재 Dataset Snapshot 기준의 Season 누적 단위로 집계한다.

생성은 다음 명령으로 실행한다.

```bash
python scripts/build_season_snapshots.py
```

기본 입력은 다음 세 파일이다.

```text
data/interim/hf_kbo_pbp/derived/player_game_batting.parquet
data/interim/hf_kbo_pbp/derived/player_game_pitching.parquet
data/interim/hf_kbo_pbp/derived/team_games.parquet
```

출력은 다음 세 파일이다.

```text
data/interim/hf_kbo_pbp/derived/player_season_batting_snapshot.parquet
data/interim/hf_kbo_pbp/derived/player_season_pitching_snapshot.parquet
data/interim/hf_kbo_pbp/derived/team_season_snapshot.parquet
```

각 Snapshot의 Grain은 다음과 같다.

```text
Player Season Batting:
(season, batter)

Player Season Pitching:
(season, pitcher)

Team Season:
(season, team)
```

각 Key는 Output 안에서 반드시 Unique하다.

이 세 테이블은 공식 최종 시즌 기록이 아니라
현재 Canonical Dataset에 포함된 경기까지의 누적 Snapshot이다.

특히 2026처럼 진행 중인 시즌은 현재 Dataset에 존재하는 경기까지만 집계한다.

```text
Snapshot != Final Season Record
```

시즌 완료 여부를 추론하거나 다음과 같은 값을 생성하지 않는다.

```text
season_complete
final_record
final_rank
projected_final_record
```

세 Snapshot은 공통으로 `through_date`를 가진다.

`through_date`는 특정 Player의 마지막 출장일이 아니라
해당 Season에서 현재 Dataset Snapshot이 어디까지 관측되었는지를 나타내는
시즌 공통 Coverage 날짜다.

Source of Truth는 `team_games.parquet`이며 다음처럼 계산한다.

```text
through_date(season)
=
max(team_games.game_date for that season)
```

따라서 같은 Season의 다음 세 Snapshot에서 모든 Row는 동일한
`through_date`를 가져야 한다.

```text
player_season_batting_snapshot
player_season_pitching_snapshot
team_season_snapshot
```

`through_date` dtype은 다음과 같다.

```text
datetime64[us]
```

Player Game Batting의 모든 Season은 Team Game Coverage에 존재해야 하며,
어떤 Player Game Batting의 `game_date`도 해당 Season의
`through_date`보다 늦을 수 없다.

Player Game Pitching에도 같은 규칙을 적용한다.

Player Season Batting은 경기별 Rate를 평균하지 않는다.

다음 Count를 먼저 `(season, batter)`까지 합산한다.

```text
pa
single
double
triple
hr
bb
hbp
so
sf
sh
double_play
triple_play
field_error
fielders_choice
catcher_interference
```

그 뒤 다음 값을 Season Count에서 다시 계산한다.

```text
h
=
single
+ double
+ triple
+ hr

ab
=
pa
- bb
- hbp
- sh
- sf
- catcher_interference

tb
=
single
+ 2 * double
+ 3 * triple
+ 4 * hr
```

Rate는 Aggregate Count에서 다시 계산한다.

```text
avg = h / ab

obp
=
(h + bb + hbp)
/
(ab + bb + hbp + sf)

slg = tb / ab

ops = obp + slg
```

Player Game Batting과 동일하게 Rate는 소수점 셋째 자리까지 반올림한다.

0 denominator는 임의로 0으로 바꾸지 않고 nullable로 유지한다.

```text
ab = 0
→ avg = <NA>
→ slg = <NA>

obp denominator = 0
→ obp = <NA>

obp 또는 slg 중 하나라도 <NA>
→ ops = <NA>
```

경기별 다음 Rate 컬럼을 Season Rate의 입력으로 사용하지 않는다.

```text
avg
obp
slg
ops
```

즉 다음 방식은 허용하지 않는다.

```text
season_avg = mean(game_avg)
season_obp = mean(game_obp)
season_slg = mean(game_slg)
season_ops = mean(game_ops)
```

Player Season Pitching은 다음 Count를 `(season, pitcher)`까지 합산한다.

```text
pitch_rows
pitches
batters_faced_completed
single_allowed
double_allowed
triple_allowed
hr_allowed
bb_allowed
hbp_allowed
so
sf
sh
ball_pitch_count
strike_pitch_count
in_play_pitch_count
```

`hits_allowed`는 다음 공식에서 다시 계산한다.

```text
hits_allowed
=
single_allowed
+ double_allowed
+ triple_allowed
+ hr_allowed
```

다음 Pitch Type 공식도 유지한다.

```text
pitches
=
ball_pitch_count
+ strike_pitch_count
+ in_play_pitch_count
```

`outs_recorded`는 Player Game Pitching의 nullable 불확실성 계약을
Season에서도 보존한다.

같은 `(season, pitcher)`의 모든 Player Game `outs_recorded`가
non-null이면 정확한 정수 합계를 저장한다.

```text
모든 Player Game outs_recorded가 non-null
→ Season outs_recorded = 정확한 합계
```

한 경기라도 `outs_recorded = <NA>`이면 Season 값도 `<NA>`다.

```text
Player Game 중 하나라도 outs_recorded = <NA>
→ Season outs_recorded = <NA>
```

따라서 다음 방식은 허용하지 않는다.

```text
<NA>를 0으로 대체

pandas 기본 sum(skipna=True)의 부분 합계를
공식 Season Outs처럼 사용

불확실한 outs_recorded에서
innings_pitched / IP / ERA 생성
```

Player Season Pitching Snapshot에서는 다음 컬럼을 생성하지 않는다.

```text
innings_pitched
ip
earned_runs
era
runs_allowed
```

Player Game Pitching의 `avg_release_speed_kmh`도
Season Snapshot에서는 재집계하지 않는다.

현재 Player Game 데이터에는 경기 평균 구속은 있지만
Season 평균 구속을 정확히 재계산하기 위한
유효 구속 관측 Pitch 수 denominator가 명시되어 있지 않기 때문이다.

따라서 다음 방식은 사용하지 않는다.

```text
mean(game_avg_release_speed_kmh)

weighted_mean(
    game_avg_release_speed_kmh,
    pitches
)
```

Team Season Snapshot은 각 `(season, team)`에서 다음을 집계한다.

```text
games = Team Game Row 수
wins = sum(win)
losses = sum(loss)
ties = sum(tie)
runs_for = sum(runs_for)
runs_against = sum(runs_against)
run_diff = runs_for - runs_against
```

반드시 다음 관계를 만족해야 한다.

```text
games
=
wins
+ losses
+ ties
```

또한 시즌 전체에서 Team Game의 양 팀 대칭성을 이용해 다음을 검증한다.

```text
sum(wins) = sum(losses)

sum(runs_for) = sum(runs_against)

sum(team_snapshot.games)
=
len(team_games)

sum(team_snapshot.games)
=
2 * unique game_pk count
```

무승부 경기에서는 양 팀 Row 모두 `tie=True`이므로
시즌 전체 `sum(ties)`는 무승부 경기 수의 2배가 된다.

Player Season Batting/Pitching Snapshot에는 다음과 같은
단일 Team 속성을 저장하지 않는다.

```text
team
current_team
latest_team
first_team
last_team
```

선수는 Season 도중 Trade될 수 있으므로 단일 Team을 Player Season의
영구 속성처럼 저장하면 Historical Team 관계를 잘못 표현할 수 있다.

Player-Team 관계가 필요하면 해당 시점의 Event-time Fact Table을 사용한다.

```text
player_game_batting.parquet
player_game_pitching.parquet
team_games.parquet
```

Season Snapshot은 Post-game Fact를 Season 전체까지 누적한 요약이다.

따라서 Historical Pregame Feature로 직접 사용하면 Data Leakage가 발생한다.

예를 들어 다음 사용은 허용하지 않는다.

```text
2025-05-01 경기 예측
← 현재 Dataset의 2025 Season Snapshot 전체 사용
```

향후 Historical Pregame Feature는 Event-time Fact Table에서
기본적으로 다음 조건을 만족하는 Row만 사용해
Rolling 또는 Expanding 방식으로 생성해야 한다.

```text
source game_date < prediction game_date
```

같은 날짜의 경기 또는 Doubleheader는 실제 경기 시작·종료 순서를
증명할 수 있는 정보가 없는 한 같은 날짜의 다른 경기 결과를
과거 정보로 사용하지 않는다.

최종 Snapshot Output은 다음 순서로 안정 정렬한다.

```text
Player Season Batting:
season, batter

Player Season Pitching:
season, pitcher

Team Season:
season, team
```

Input Row 순서가 달라져도 동일한 값, Row 순서, dtype을 생성해야 한다.

Byte-level Parquet Hash 동일성까지 요구하지 않는다.

Season Snapshot Builder는 세 Canonical Input을 수정하거나 덮어쓰지 않는다.

```text
player_game_batting.parquet
player_game_pitching.parquet
team_games.parquet
```

Output을 `data/raw/` 내부에 생성할 수 없으며,
세 Snapshot Output끼리 같은 경로를 사용할 수도 없다.

세 Snapshot DataFrame의 Build와 Validation이 모두 완료된 뒤에만
Parquet 저장 단계로 진입한다.

각 Output은 다음 순서로 저장한다.

```text
temporary parquet write
→ read_parquet round-trip
→ dtype 및 내용 검증
→ final path atomic replace
```

Builder 실행 전후에는 세 Canonical Input의 SHA256을 비교해
Input 불변성을 검증한다.

#### Canonical Derived Layer 통합 검증

Canonical Derived Layer 전체 계약은 다음 명령으로 검증한다.

```bash
python scripts/validate_derived_tables.py
```

검증 Script는 기존 Raw 및 Derived Artifact를 읽기만 하며
Derived Table을 재생성하거나 Raw/Derived Parquet을 수정·덮어쓰지 않는다.

검증 대상 Canonical Derived Table은 다음 9개다.

```text
plate_appearances.parquet
games.parquet
team_games.parquet
player_game_batting.parquet
player_game_pitching.parquet
players.parquet
player_season_batting_snapshot.parquet
player_season_pitching_snapshot.parquet
team_season_snapshot.parquet
```

전체 계약 흐름은 다음과 같다.

```text
Raw Pitch
→ Plate Appearance
→ Game / Team Game
→ Player Game Batting / Pitching
→ Player Metadata
→ Player / Team Season Snapshot
```

통합 검증에서는 각 Builder의 개별 Unit Test를 넘어
Source 간 다음 계약을 교차 확인한다.

```text
File 존재 여부
Column 순서 및 dtype
Grain Key Unique

Raw PA Key ↔ Plate Appearance
Raw Game Key ↔ Games

Games ↔ Team Games Home/Away Mirror

Plate Appearance 마지막 Row ↔ Game Final Score
첫 Pre-score + runs_scored ↔ Game Final Score

Completed PA ↔ Player Game Batting
Batting Event Cross-tab
H / AB / TB 및 Rate 공식

Raw Pitch Row / 실제 Pitch / B/S/X
↔ Player Game Pitching

Completed PA ↔ Pitcher BF
Allowed Event ↔ Player Game Pitching

Multi-pitcher PA의 nullable outs_recorded 정책
Player Game Pitching ↔ Player Season Pitching Null 전파

Canonical Player ID Union ↔ Player Metadata
Batter / Pitcher Role Coverage

Team Games ↔ Team Season Snapshot

시즌 공통 through_date
2026 Partial Snapshot

Canonical Content Fingerprint
Raw / Derived SHA256 실행 전후 불변성
data/interim Git Ignore 정책
```

Raw 검증에서는 시즌별 Parquet에서 필요한 Column만 읽어
PA/Game/Pitcher 단위 Summary를 계산한다.

Production Row Count를 외부 고정값으로 사용하지 않고
항상 현재 Raw 또는 Upstream Canonical Table에서 기대값을 다시 계산한다.

특히 다음과 같은 Snapshot 진단값은 Assertion Truth로 하드코딩하지 않는다.

```text
특정 Raw Row Count
특정 Plate Appearance Count
특정 Game Count
특정 Player Count
nullable Pitching Player-Season 개수
특정 2026 through_date
```

Player Game Pitching의 `outs_recorded`는
Multi-pitcher Plate Appearance에서 공개 Raw만으로
Out 발생 시점을 안전하게 투수별 귀속할 수 없는 경우 `<NA>`를 유지한다.

따라서 Integration Validation에서도 다음 처리를 하지 않는다.

```text
<NA> → 0
skipna=True 부분 합계를 공식 Outs로 사용
모든 PA Out Delta를 마지막 Pitcher에게 무조건 귀속
```

Season Pitching Snapshot도 같은 불확실성을 전파한다.

```text
Player-Season의 모든 Player Game outs_recorded가 non-null
→ Season outs_recorded = 정확한 합계

Player Game 중 하나라도 outs_recorded = <NA>
→ Season outs_recorded = <NA>
```

`through_date`는 Player의 마지막 출장일이 아니라
해당 Season에서 Canonical Dataset이 관측된 공통 Coverage 날짜다.

```text
through_date
=
max(team_games.game_date by season)
```

따라서 2026을 완료 시즌으로 가정하거나
특정 날짜를 고정 Assertion으로 사용하지 않는다.

DataFrame Content 결정성은 각 Table을 Canonical Grain Key로
안정 정렬한 뒤 Column 순서와 dtype을 포함한 Content Fingerprint로 진단한다.

Parquet 파일의 Byte Hash는 Content Determinism의 영구 ID로 사용하지 않는다.
Parquet SHA256은 Validation 실행 전후 파일이 변경되지 않았는지 확인하는
Read-only 안전성 검증에 사용한다.

Canonical Derived Artifact는 다음 Git Ignore 정책 아래에 있다.

```text
/data/interim/**
```

Integration Validator는 어떤 Output 파일도 생성하지 않는다.

### `data/processed/`

모델 학습 및 분석에 사용할 최종 가공 데이터를 저장한다.

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

##### Team Pregame Historical Feature (#23)

`team_games.parquet`의 Post-game Fact를 경기 전 시점의 팀 Historical Feature로 변환한다.

입력:

```text
data/interim/hf_kbo_pbp/derived/games.parquet
data/interim/hf_kbo_pbp/derived/team_games.parquet
```

출력:

```text
data/processed/hf_kbo_pbp/features/team_pregame_features.parquet
data/processed/hf_kbo_pbp/features/team_pregame_features.manifest.json
```

Grain:

```text
(game_pk, team)
```

실행:

```bash
python scripts/build_team_features.py
```

Historical Source는 항상 다음 cutoff를 만족해야 한다.

```text
source.game_date < prediction_date
```

같은 날짜 경기와 Doubleheader는 정확한 이용 가능 시각 provenance가 없으므로 서로의 결과를 Historical Source로 사용하지 않는다. Season Expanding은 `(season, team)`에서 초기화하며 이전 시즌을 자동 이월하지 않는다.

최근 Window는 `5/10/20`경기를 기본값으로 사용한다. Window 경계가 같은 날짜 경기 묶음에 걸리면 해당 날짜 전체를 포함하므로 `last_5g_games`, `last_10g_games`, `last_20g_games`는 항상 정확히 5/10/20일 필요가 없다. 실제 포함 경기 수는 각 `last_*g_games`에 기록한다.

Cold Start 정책:

```text
count feature = 0
rate feature = null
has_history = false
history_game_count = 0
max_source_game_date = null
```

`hist_win_pct`와 Window별 승률은 `wins / (wins + losses)`로 계산하고 무승부를 패배에 합치지 않는다. 분모가 0이면 `null`이다. 경기당 Rate도 aggregate count에서 다시 계산한다.

주요 Audit:

```text
prediction_date
season
max_source_game_date
history_game_count
has_history
feature_version
```

`days_since_last_observed_game`는 관측된 직전 Source 경기와 Prediction Date 사이의 날짜 차이다. 공식 휴식일, 우천 취소 횟수, 실제 휴식 보장 일수로 해석하지 않는다.

Manifest에는 입력 경로/SHA256, Feature/Source Contract Version, Rolling Window 설정, Output Schema, Row Count, Key/Grain, X Allowlist, 결정적 Content Fingerprint를 기록한다. 생성 시각처럼 실행마다 달라질 수 있는 Metadata는 Content Fingerprint와 분리한다.

현재 경기의 `runs_for`, `runs_against`, `win`, `loss`, `tie`는 Output X에 직접 포함하지 않는다. Key/Audit와 모델 X는 명시적으로 분리하며 `all columns except y` 방식으로 학습 입력을 만들지 않는다.

##### Player Pregame Historical Feature (#24)

`player_game_batting.parquet`, `player_game_pitching.parquet`의 선수-경기 단위 Post-game Fact를 요청된 예측 시점 이전의 Player Historical Feature로 변환한다.

Request와 Historical Source는 분리한다. Builder가 미래 출장 여부나 `players.parquet`의 전체 기간 role을 보고 요청 Population을 자동 생성하지 않는다.

Request Grain:

```text
(player_id, role, prediction_date)
```

필수 Request 컬럼:

```text
player_id        string
role             string  # batting | pitching
prediction_date  date
```

중복 Request는 오류로 처리한다. 유효 Request는 History가 없어도 삭제하지 않으며 정확히 1행의 Output으로 보존한다.

입력 Historical Source:

```text
data/interim/hf_kbo_pbp/derived/player_game_batting.parquet
data/interim/hf_kbo_pbp/derived/player_game_pitching.parquet
```

기본 출력:

```text
data/processed/hf_kbo_pbp/features/player_pregame_features.parquet
data/processed/hf_kbo_pbp/features/player_pregame_features.manifest.json
```

실행 시 Request 파일을 반드시 명시한다.

```bash
python scripts/build_player_features.py \
  --requests-path /path/to/player_feature_requests.parquet
```

모든 Historical Source는 다음 cutoff를 만족해야 한다.

```text
source.game_date < prediction_date
```

날짜보다 세밀한 이용 가능 시각이 없으므로 `prediction_date` 당일 경기와 Same-day/Doubleheader 결과는 전부 제외한다. `game_pk` 또는 Row 순서를 경기 선후의 근거로 사용하지 않는다.

Season Expanding은 `prediction_date.year`의 해당 Season에서 초기화한다. Canonical Source의 `season`과 `game_date` 연도가 다르면 계약 오류로 실패한다.

Batting은 `docs/feature_catalog.md`의 `hist_pa`, `hist_ab`, `hist_h`, 안타 유형, BB/HBP/SO/SF/SH/TB 및 특수 Event Count를 누적한다. `hist_avg`, `hist_obp`, `hist_slg`는 경기별 Rate 평균이 아니라 누적 Count에서 다시 계산하고 소수점 셋째 자리까지 반올림한다. `hist_ops`는 이렇게 계산된 OBP와 SLG를 더한 뒤 다시 소수점 셋째 자리까지 반올림한다. 분모가 0인 Rate는 `null`이다.

Pitching은 `hist_pitch_rows`와 `hist_pitches`를 구분하고 completed BF, 피안타 유형, BB/HBP/SO/SF/SH, B/S/X Pitch Count를 누적한다. Source History에 `outs_recorded = null`인 Player Game이 하나라도 있으면 누적 `hist_outs_recorded`도 `null`이다. `history_uncertain_outs_game_count`에 불확실 Player Game 수를 Audit로 남긴다.

현재 Player Game Source에는 경기 평균 구속을 다시 정확히 가중할 유효 구속 관측 수 denominator가 없으므로 `hist_avg_release_speed_kmh`를 생성하지 않는다. ERA, earned runs, 공식 runs allowed, RBI도 생성하지 않는다.

Rolling Window는 타격/투구 모두 최근 `5/10/20` Player Game을 사용한다.

```text
last_5g_<stat>
last_10g_<stat>
last_20g_<stat>
```

Window 경계가 같은 날짜 경기 묶음에 걸리면 해당 날짜 전체를 포함한다. 따라서 실제 Window 크기는 정확히 5/10/20이 아닐 수 있으며 `last_5g_games`, `last_10g_games`, `last_20g_games`에 실제 포함 Player Game 수를 기록한다. Pitching Window의 불확실 outs 수는 `last_*g_uncertain_outs_game_count`로 별도 Audit한다.

Cold Start 정책:

```text
count feature = 0
rate feature = null
has_history = false
history_game_count = 0
max_source_game_date = null
```

`days_since_last_observed_appearance`는 Prediction Date보다 이전에 실제 관측된 해당 Role의 가장 최근 Player Game 날짜와의 차이다. 부상, 엔트리 상태, 감독 의사, 실제 availability를 의미하지 않는다.

Output의 Role별 Historical Feature와 Key/Audit 컬럼은 명시적으로 구분한다. M2/M3 소비자는 `docs/feature_catalog.md`의 `baseline_v1` 또는 `extended_v1` allowlist를 사용하며 `X = all columns except y` 방식으로 Feature를 선택하지 않는다.

Manifest에는 Request 경로/SHA256/Content Fingerprint, Batting/Pitching 입력 경로와 SHA256, 계약/Feature Version, cutoff, Season 초기화, Rolling Window, Grain, Role별 Feature 계약, Output Schema, Role별 Row 수, Cold Start 수, Null Summary, 결정적 Output Content Fingerprint를 기록한다. 실행 시각은 Content Fingerprint와 분리한다.

Builder는 입력 파일 SHA256을 실행 전후 비교하며 입력을 수정하지 않는다. Output/Manifest는 입력 경로와 충돌할 수 없고 `data/raw/`, `data/interim/` 아래에 쓸 수 없다. Parquet은 원자적 저장 후 round-trip Schema/Content Fingerprint 검증을 수행하고 완료 Manifest를 마지막에 기록한다.

##### M1 Model-ready Dataset (#25)

`games.parquet`의 경기 단위 Context/Target과 #23 `team_pregame_features.parquet`를 Home/Away 방향으로 결합해 M1 학습·평가 단계가 소비할 수 있는 Processed Dataset을 만든다.

입력:

```text
data/interim/hf_kbo_pbp/derived/games.parquet
data/interim/hf_kbo_pbp/derived/plate_appearances.parquet  # observation quality 재대사에만 사용
data/processed/hf_kbo_pbp/features/team_pregame_features.parquet
data/processed/hf_kbo_pbp/features/team_pregame_features.manifest.json
configs/processed_dataset.json
```

출력:

```text
data/processed/hf_kbo_pbp/m1/game_dataset.parquet
data/processed/hf_kbo_pbp/m1/game_dataset.manifest.json
data/processed/hf_kbo_pbp/m1/game_dataset.schema.json
data/processed/hf_kbo_pbp/m1/game_dataset.quality.json
```

Grain:

```text
game_pk당 1행
```

실행:

```bash
python scripts/build_m1_dataset.py
```

기본 실행은 Canonical `plate_appearances.parquet`의 마지막 관측 PA를 제한적으로 다시 확인해 observed terminal score reconciliation과 terminal PA warning을 Quality Audit에 남긴다. 이 재대사를 생략해야 할 때만 `--skip-pa-audit`를 사용한다. 공식 경기 status는 현재 Source에 없으므로 `official_status_verified=false`로 기록하며, 그 사실만으로 Target을 폐기하지 않는다.

Target은 current-main #22 계약을 그대로 사용한다.

```text
home_runs = games.final_home_score
away_runs = games.final_away_score

loss = 0
tie  = 1
win  = 2
```

여기서 `final_*_score`는 KBO 공식 종료 결과가 아니라 Canonical **observed terminal score**다. 필수 Score 결측이나 실제 Score/Result reconciliation 오류는 `is_excluded=true`와 primary `exclusion_reason`으로 추적하고 학습 Target은 null 처리한다. 마지막 관측 PA가 미완료라는 사실만 있는 경우에는 `warning_terminal_pa_incomplete`로 남기되 자동 제외하지 않는다.

#23 Team Feature는 다시 계산하지 않는다. Manifest의 `(game_pk, team)` Grain, Audit Column, `x_columns`, `feature_version`, `source_contract_version`, Row Count를 검증한 뒤 각 경기에 대해 Home Feature와 Away Feature가 정확히 하나씩 존재하는지 확인한다. `team`, `opponent`, `prediction_date`, `season`, `is_home` Context가 `games`와 다르면 실패하며 Join으로 Row가 증식하면 실패한다.

M1 Feature Set은 current-main Feature Catalog의 `extended_v1`을 사용한다. #23 Manifest의 Team Historical `x_columns`를 다음 prefix로 분리하고, Catalog가 baseline X로 허용한 `home_has_history`/`away_has_history`를 포함한다.

```text
home_<team_feature>
away_<team_feature>
home_has_history
away_has_history
```

또한 Catalog에 명시된 최소 Difference만 생성한다.

```text
diff_hist_win_pct
diff_hist_runs_for_per_game
diff_hist_runs_against_per_game
diff_hist_run_diff_per_game
```

각 Difference는 `home_* - away_*`이며 nullable 의미를 그대로 보존한다. 임의의 Difference나 Team ID는 X에 자동 추가하지 않는다.

Key/Audit/Target/Split/Quality/Control/Label Status는 X와 논리적으로 분리한다. 현재 경기 Final Score·승패·이닝·PA/Pitch Count·실제 출장 정보·terminal 상태·observation quality·split·exclusion/censor/purge/label 상태는 X에 포함할 수 없다. Dataset 소비자는 `X = all columns except y` 방식 대신 Schema/Manifest의 명시적 `x_columns`를 사용해야 한다.

시간 Split은 고정 `season_split_v1`을 사용한다.

```text
2023 -> train
2024 -> validation
2025 -> test
2026 -> snapshot
```

Random Split이나 Test 결과를 본 뒤 경계를 바꾸는 방식은 사용하지 않는다. M1 Label availability는 날짜 단위 보수적 계약에 따라 `label_available_at = game_date + 1 calendar day`로 기록하며 실제 경기 종료 시각을 의미하지 않는다.

Schema Artifact는 다음 역할을 명시적으로 구분한다.

```text
key_columns
audit_columns
x_columns
y_columns
split_columns
quality_columns
control_columns
label_status_columns
```

Manifest는 입력 경로와 SHA256, 가능한 입력 revision, code revision, 계약/Feature/Target/Split/Schema version, 설정, Prediction 범위, Source coverage, Row/Null/Quality/Exclusion 요약, Output Content Fingerprint를 기록한다. `created_at`과 `run_id` 같은 실행 metadata는 결정적 `deterministic_identity.fingerprint`와 분리한다. Dataset/Schema/Quality Report의 round-trip 및 입력 hash 불변성 검증이 끝난 뒤 `run_status=complete` Manifest를 마지막에 기록한다.

`Model-ready`는 X/y/split/schema 계약이 고정되어 후속 ML 단계가 소비할 수 있다는 뜻이다. 학습형 Imputation, Scaling, Encoding, Feature Selection, 모델 학습, 성능 비교, Calibration, Backtesting은 #25 범위에 포함하지 않는다.

##### M2 Model-ready Dataset (#26)

M2는 특정 Plate Appearance가 시작되는 순간 이용 가능한 Context와
시작 타자·투수의 전일까지 Historical Feature를 사용해
해당 PA의 최종 Event Class를 예측하기 위한 Processed Dataset이다.

입력:

```text
data/raw/hf_kbo_pbp/2023.parquet
data/raw/hf_kbo_pbp/2024.parquet
data/raw/hf_kbo_pbp/2025.parquet
data/raw/hf_kbo_pbp/2026.parquet
data/raw/hf_kbo_pbp/download_manifest.csv
data/raw/hf_kbo_pbp/schema.yaml
data/interim/hf_kbo_pbp/derived/plate_appearances.parquet
data/interim/hf_kbo_pbp/derived/player_game_batting.parquet
data/interim/hf_kbo_pbp/derived/player_game_pitching.parquet
configs/processed_dataset.json
```

출력:

```text
data/processed/hf_kbo_pbp/m2/matchup_dataset.parquet
data/processed/hf_kbo_pbp/m2/matchup_dataset.manifest.json
data/processed/hf_kbo_pbp/m2/matchup_dataset.schema.json
data/processed/hf_kbo_pbp/m2/matchup_dataset.quality.json
```

Grain:

```text
(game_pk, at_bat_number)
```

실행:

```bash
python scripts/build_m2_dataset.py
```

M2 Builder는 Canonical `plate_appearances.parquet`를
PA Grain과 최종 Event의 기준으로 사용하지만,
Canonical `batter`, `pitcher`, `stand`를 PA 시작 선수라고 가정하지 않는다.

동일한 고정 Raw snapshot을
`(game_pk, at_bat_number, pitch_number)` 순서로 안정 정렬한 뒤
PA별 실제 첫 Raw Row 한 행 자체에서 다음 값을 복원한다.

```text
starting_batter
starting_pitcher
starting_stand
```

컬럼별 `groupby().first()`를 사용해 첫 Row의 null을
뒤 Row의 non-null 값으로 채운 가상의 시작 Context를 만들지 않는다.

Canonical 귀속 값은 별도 Audit으로 유지한다.

```text
credited_batter
final_pitcher
credited_stand
```

`starting_batter`, `starting_pitcher`는 Join/Audit Key이며
기본 모델 X에 포함하지 않는다.

PA-start Context X:

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

원시 Runner ID는 X에 포함하지 않고 Base 점유 여부만 사용한다.
투수 손잡이는 Raw에서 별도로 확인된 정보가 없으므로 추정하지 않는다.

시작 타자·투수 Historical Feature는
#24 `player_pregame_v1` Builder를 재사용한다.

Prediction Date:

```text
prediction_date = PA.game_date
```

Historical Cutoff:

```text
source.game_date < prediction_date
```

따라서 Prediction Date 당일 Player Game과
Same-day/Doubleheader 결과는 Historical Source에 포함하지 않는다.

Player Feature 요청은 다음 Grain으로 중복 제거한 뒤 #24 Builder에 전달한다.

```text
(player_id, role, prediction_date)
```

History가 없는 시작 선수도 PA에서 삭제하지 않는다.

Cold Start:

```text
count feature = 0
rate feature = null
has_history = false
history_game_count = 0
max_source_game_date = null
```

Pitching Historical Feature의 `hist_outs_recorded`는
#24의 nullable 불확실성 계약을 그대로 유지한다.

Source History에 불확실 `outs_recorded`가 하나라도 존재하면
0으로 채우거나 결측을 제외한 부분 합계를 정확한 Outs처럼 사용하지 않는다.

M2 Feature Set은 `feature_catalog_v1`의 `extended_v1`을 사용한다.

`baseline_v1`의 PA-start Context와 Batter/Pitcher Expanding Feature에
최근 5/10/20 Player Game Rolling Feature를 추가한다.

현재 PA의 다음 정보는 X에 포함하지 않는다.

```text
event
pa_completed
runs_scored
post_outs
post_on_1b
post_on_2b
post_on_3b
post_home_score
post_away_score
pitch_rows
pitch_count
ball_pitch_count
strike_pitch_count
in_play_pitch_count
credited_batter
final_pitcher
credited_stand
미래 substitution flag
target_class
target_code
split
quality/exclusion/censor/purge/label 상태
```

Dataset 소비자는 `X = all columns except y` 방식으로 Feature를 만들지 않는다.
Schema/Manifest의 명시적 `x_columns`만 사용한다.

Target Source:

```text
plate_appearances.event
```

고정 15→11 Mapping:

```text
single               -> 1B
double               -> 2B
triple               -> 3B
home_run             -> HR
walk                 -> BB
hit_by_pitch         -> HBP
strikeout            -> SO
field_out            -> OUT
double_play           -> OUT
triple_play           -> OUT
sac_bunt              -> OUT
sac_fly               -> OUT
field_error           -> ROE
fielders_choice       -> FC
catcher_interference  -> CI
```

고정 Class 순서와 Code:

```text
1B   = 0
2B   = 1
3B   = 2
HR   = 3
BB   = 4
HBP  = 5
SO   = 6
OUT  = 7
ROE  = 8
FC   = 9
CI   = 10
```

`OUT`은 위 Event Group의 분류명이며
정확히 하나의 out이 발생했다는 의미가 아니다.

`ROE`, `FC`, `CI`를 임의로 `OUT` 또는 `OTHER`로 병합하지 않는다.

원본 `source_event`는 Audit으로 그대로 보존한다.

`event IS NULL`인 미완료 PA는 다음 상태로 유지한다.

```text
target_class = null
target_code = null
is_censored = true
supervised_usable = false
label_status = target_pending
```

알 수 없는 non-null Event는 자동 보정하지 않고
계약 오류로 Builder를 중단한다.

Pitch-less PA는 자동 제외하지 않는다.

Raw 첫 관측 Row의 `pitch_number=0`이고 실제 Pitch가 하나도 없는 경우:

```text
start_context_quality = verified_pitchless_start
```

Raw 첫 관측 Row가 `pitch_number=1`이면:

```text
start_context_quality = verified_first_pitch_start
```

Raw 첫 관측 Row가 `pitch_number>1`이면
초기 Pitch Row가 누락되었을 가능성이 있으므로
PA 시작 Context를 완전히 관측했다고 주장하지 않는다.

```text
start_context_quality = unverified_missing_initial_pitch_rows
is_excluded = true
exclusion_reason = start_context_not_fully_observed
```

이 Row는 Audit Dataset에는 유지하지만
supervised 학습 대상으로 사용하지 않는다.

PA 도중 타자 또는 투수가 교체되어도
미래 교체 사실을 이용해 Row를 미리 제거하지 않는다.

Prediction 의미는 다음과 같다.

```text
PA 시작 Context
+
PA 시작 타자 Historical Feature
+
PA 시작 투수 Historical Feature

-> 최종 PA Event
```

시작 선수와 기록 귀속 선수의 차이는 Audit 및 subgroup 진단용이며,
PA 최종 결과를 시작 투수 개인의 책임으로 해석한다는 뜻이 아니다.

시간 Split은 `season_split_v1`을 사용한다.

```text
2023 -> train
2024 -> validation
2025 -> test
2026 -> snapshot
```

Random Split은 사용하지 않는다.

동일 `game_pk`의 PA는 반드시 하나의 `prediction_date`와
하나의 Split만 가져야 한다.

Label Availability는 날짜보다 세밀한 종료 timestamp가 없으므로
완료 PA에 대해 보수적으로 다음과 같이 기록한다.

```text
label_available_at = game_date + 1 calendar day
availability_basis = conservative_next_date
```

미완료 PA의 `label_available_at`은 null이다.

Schema Artifact는 다음 역할을 명시적으로 분리한다.

```text
key_columns
audit_columns
x_columns
y_columns
split_columns
quality_columns
control_columns
label_status_columns
```

Manifest/Quality Artifact는 최소 다음 provenance와 대사를 기록한다.

```text
Raw fixed revision
Raw/Canonical/Player Game 입력 경로
입력 SHA256
Code Revision
Prediction Contract Version
Feature Catalog Version
M2 Feature Version
Player Feature Version
Player Feature Request Count
Player Feature Content Fingerprint
Target Version
Split Version
Output Schema Version
Prediction Range
Raw/Canonical/Player Game Source Coverage
Row / Null / Split Count
Source Event Count
Target Class Count
Source Event별 input -> target -> status Reconciliation
시작 선수와 Canonical 귀속 선수 차이 진단
Start Context 품질 수량
Exclusion / Censor / Purge 수량
Dataset Content Fingerprint
Deterministic Identity
```

Player Feature provenance는 실제 M2에 Join된
중복 제거 Batter/Pitcher 요청 결과를 대상으로 결정적으로 계산한다.

Builder 실행 전후 모든 입력 파일 SHA256을 비교해
입력이 변경되지 않았음을 확인한다.

Output/Manifest/Schema/Quality Report는 입력 파일과 같은 경로를 사용할 수 없으며
`data/raw/`, `data/interim/` 아래에 쓸 수 없다.

Dataset Parquet은 원자적으로 저장한 뒤
dtype, Column 순서, PA Grain, Content Fingerprint를 다시 확인한다.

Schema와 Quality Report 검증이 끝난 뒤
`run_status=complete` Manifest를 마지막에 기록한다.

현재 고정 Raw snapshot 검증에서 사용한 revision:

```text
6afc8af044e3bba5f326b688e8cb41d7ff7065ec
```

해당 검증 Run에서 관측된 수량:

```text
input PA            = 207133
supervised usable   = 206370
target pending      = 550
explicit excluded   = 213

207133
= 206370
+ 550
+ 213
```

동일 검증 Run에서 다음 진단값을 확인했다.

```text
pitch-less PA                   = 364
starting_stand null             = 1764
starting batter != credited     = 44
starting pitcher != final       = 183
```

위 수량은 해당 고정 Raw revision에서 확인한 Snapshot 진단값이며
향후 Dataset의 영구 고정 Row Count나 하드코딩 Assertion이 아니다.

동일 입력·동일 계약·동일 설정으로 다시 생성하면
Dataset Content Fingerprint는 결정적이어야 한다.

`Model-ready`는 PA-start 시점의 Leakage-safe X, Target,
고정 시간 Split, Schema, Manifest, Quality/Exclusion 상태가
후속 ML 단계에서 소비 가능한 형태로 고정됐다는 의미다.

다음을 의미하지 않는다.

```text
Matchup 승률 모델 학습 완료
최적 알고리즘 선정 완료
Hyperparameter Tuning 완료
성능 평가 완료
Pitch-by-pitch M5 구현 완료
당일 실시간 Player History 구현 완료
```

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

M3 v1의 `7d`, `28d`, `rest_of_season` Target은 **KBO 정규시즌 성적만** 대상으로 하며,
포스트시즌 성적은 포함하지 않는다.

검증된 정규시즌 종료일과 해당 시즌의 정규시즌 전체 coverage 근거가 모두 확인된
완료 시즌에서는 `7d` 또는 `28d` label interval이 정규시즌 종료일 이후까지
이어지더라도 원래의 half-open interval을 임의로 단축하지 않는다.

정규시즌 종료 이후 부분은 추가 정규시즌 경기가 존재하지 않는 것으로 검증된
무경기 구간이므로, 전체 정규시즌 coverage가 확인된 경우 해당 interval의
coverage를 `complete`로 판정할 수 있다.

반대로 정규시즌 종료일 또는 전체 coverage를 검증할 수 없는 진행 시즌·partial
snapshot에서는 `through_date`나 단순 `max(observed game_date)`만으로 종료 이후
무경기 구간을 `complete`로 간주하지 않는다.

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