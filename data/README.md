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
