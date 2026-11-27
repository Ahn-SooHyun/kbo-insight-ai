# 시즌 전망·보조 분석 종합 검증

KBO Insight AI · 기술 계약 및 실행 안내 · [Issue #35](https://github.com/Ahn-SooHyun/kbo-insight-ai/issues/35)

저장소 적용 경로: `docs/standards/season_forecast_validation.md`

## 범위와 실행 경계

기준 `main`은 `11f996b6ac04c0edac706d4b16599e49cc6ebdc5`이다. 중간 구현 커밋은 `b2e5d2665984cc72dee6cc97e384b68c31976b38`이다. 고정 표준 DB, 모델 자료 DB, M1, 기존 정규시즌·포스트시즌·보조 분석과 고정 최종 평가 결과를 변경하지 않는다. 검증 전용 실행이며 학습·선정·보정 또는 새 예측기를 추가하지 않는다.

첫 사용자 실행은 신규 테스트 29개, 시뮬레이션 테스트 176개, 전체 회귀 657개 및 실자료 실행 12건을 통과했다. 최종 독립 검산은 `read_completed` import 누락으로 실패했다. 아래 보완 계약은 적용 및 실행 후 검증할 사항이며 통과를 선확정하지 않는다.

## 사전 고정 및 실행 재사용

`prepare`는 신규 검증 디렉터리에 `plan.json`을 작성하고 SHA-256을 출력한다. 계획에는 현재 HEAD, Git 상태, Python·SQLite 버전, 고정 자산 및 기존 산출물, 관련 코드·테스트·설정·본 문서의 지문과 실행 조합이 포함된다. `run`과 `verify`에는 계획 경로와 SHA-256을 전달한다. 코드나 자료의 변동은 차단한다.

정규시즌 대표 기준은 2025-09-01 KST 자정, Seed 42, 1,000회이며 개발 진단을 포함한다. 동일 조건 재실행, Seed 42/2,000회 및 Seed 43/1,000회를 비교한다. 각각의 포스트시즌은 해당 부모 실행 ID를 인수한다. 2025-11-01 종료 재생은 Seed 42/1,000회이며 원천 판본 가정을 명시한다. 보조 분석은 대표 기준일의 시즌/경기, 2025년 선발 개발 평가, 최초 지원일과 종료 이후 경계일을 검증한다.

보완 이후 `prepare --reuse-manifest <기존 계획> --reuse-manifest-sha256 <기존 지문>`은 기존 계획 및 성공 기록을 읽기 전용으로 인수한다. 기존 계획의 고정 자료·운영 코드·기존 결과가 새 계획과 같아야 한다. 검증 실행 모듈과 문서의 이전 지문은 중간 구현 커밋의 Git 객체 바이트(또는 정확한 LF/CRLF 표현)와 비교한다. 새 테스트는 이전 계산의 증거가 아니므로 다시 실행한다. 기존 작업 기록의 명령, 종료 코드, 결과 파일 지문 및 입력 계보를 확인한다. 인수 작업은 새 계획에 이전 계획 및 작업 기록 지문을 포함하며 기존 파일을 복사하거나 덮어쓰지 않는다. 신규 검증 기록에는 `reused_from`이 남는다. 이 방식은 최초 실행의 실제 계산과 최신 검증 코드의 재검산을 구분한다. 지문 또는 실행 조건이 다르면 인수를 거부한다. 기존 계획을 지정하지 않은 `prepare`는 모든 계산을 새로 실행한다.

## 독립 검산과 판정 기준

- 정규시즌: 표준 DB 경기 식별자·날짜·홈/원정·점수 및 기준일 절단을 대조한다. 반복별 팀 기록과 순위·진출·시드를 검사하고 승수·순위·1위·진출 분포, 분모 및 미결정 사유를 재집계한다. 정규시즌 반복 출력은 팀별 집계이므로 반복 내부의 완료 경기별 점수 행까지 저장되지는 않는다. 실제 완료 경기 시나리오 연결과 반복 집계 하한·합계 검증을 구별한다.
- 포스트시즌: 부모·자식 반복의 식별자, 확정 시드, 진출 상태, 순위, 1위 상태 및 미결정 사유를 전수 대조한다. 시리즈 연결과 실제 기록의 접두 구간을 검사하고 기존 규칙으로 차전·홈·종료 상태를 검증한다. 팀별 참가·승리 사건과 라운드 공동 분모, 대진 조건부 분모를 독립 재집계한다. 규칙 재생은 기존 규칙 재사용이며 별개 규칙 구현이 아니다.
- 재현: 동일 Seed·입력·반복 수에서 계산 출력은 정확히 같아야 한다. 정규시즌과 포스트시즌의 2,000회 실행에서 앞 1,000개 반복 행도 일치해야 한다. 실행 ID·시각 등 비계산 메타데이터는 명시한 범위에서만 제외한다.
- 변동: 사건별 유효 분모가 30 이상일 때 확률 변화가 `max(0.02, 5 * sqrt(0.25/n1 + 0.25/n2))`를 초과하면 조사 대상으로 표시한다. 기대 승수·순위는 각각 가능한 범위(144, 9)를 곱한 동일한 보수적 척도와 0.5의 최소 변화 기준을 사용한다. 이는 탐색적 경고이며 공식 신뢰구간 또는 합격 성능 기준이 아니다. 중첩 반복과 미결정 분모의 의존성을 고려한다.
- 선발: `starter_actual`의 경기×팀 키를 확인하고 전체·정답 식별·후보 제공·공통 분모를 구분하여 두 규칙의 Top-1/Top-3을 재계산한다. 전체 이력과 기준일 이전 자료에서 기존 후보 규칙을 재실행하여 미래 입력의 영향을 비교한다. 별도 모델 또는 독립 순위 알고리즘은 아니다.
- 팀·일정: 실제 완료 경기 행에서 시즌/최근 28일·홈/원정·상대별 기록을 재계산한다. 일정의 팀별 경기 키, 유효 확률, 미편성·누락 수를 검사한다. 난이도는 `P(loss)+0.5*P(tie)`이며 두 팀 값의 합은 1이다. 팀 평균은 유효 전망 경기의 동일 가중 평균이며 분모 0은 null이다. 실제 누락 사례와 합성 테스트를 구별한다.

## 평가 자격과 공식 사실

2025년은 개발·사후 일정 가정이다. 당시 공개 일정 판본과 운영 모델의 생성·선정·이용 가능 시점이 입증되지 않아 정식 Backtesting으로 인정하지 않는다. 실제 정답과의 차이는 개발 사실 대조이며 독립 성능이 아니다. 2026년은 고정 시즌 전망 및 보조 분석 구현이 지원하지 않는다. 표준 DB에 자료가 존재하더라도 공식 완료와 규칙 지원, 당시 정보 가용성을 자동 인정하지 않는다.

2025년 확인 사실: 정규시즌 상위 5팀은 LG·한화·SSG·삼성·NC, 와일드카드 및 준플레이오프 통과 삼성, 플레이오프 통과 한화, 한국시리즈 LG 우승(4승 1패)이다. 한국시리즈 5차전은 2025-10-31 대전, 한화 홈·LG 원정, LG 4–1 승리였다. 공식 일정 및 경기 보도는 확인한 사건의 근거이며 전체 경기기록 전수 인증은 아니다.

- [KBO 포스트시즌 개막 및 규칙 공지](https://www.koreabaseball.com/MediaNews/Notice/View.aspx?bdSe=11694)
- [KBO 한국시리즈 예매 및 장소 공지](https://www.koreabaseball.com/MediaNews/Notice/View.aspx?bdSe=11735)
- [KBO 홈페이지 게재 플레이오프 결과](https://www.koreabaseball.com/MediaNews/News/BreakingNews/View.aspx?bdSe=60367)
- [KBO 홈페이지 게재 한국시리즈 결과](https://koreabaseball.com/MediaNews/News/BreakingNews/View.aspx?bdSe=60387)
- [연합뉴스 한국시리즈 우승 보도](https://www.yna.co.kr/view/AKR20251031176652007)
- [2025년 정규시즌 최종 순위 보조 자료](https://kbo-dashboard.co.kr/standings/2025/)

기존 종료 재생의 홈 표기 충돌은 표준 DB의 경기 및 원천 연결 키, 일정 근거, 완료 시리즈 및 공식 사실을 대조해 보존한다. 원본 바이트 자체의 독립 확인과 공식 정정 자료 구축은 별도 범위다. 우승 1,000회 미결정은 오류 차단의 성공이지 공식 우승 재현의 성공이 아니다.

## 실행

기존 성공 실행 인수 시 프로젝트 루트에서 아래 명령을 사용한다. 기존 계획이 없거나 지문이 다르면 인수를 중단하며 임의로 결과를 정상 처리하지 않는다.

```powershell
& {
    Set-Location 'D:/Programming/kbo-insight-ai'
    $py = 'D:/Programming/kbo-insight-ai/.venv/Scripts/python.exe'
    $old = 'results/season_forecast_validation/bf6c5cd8bce54cc8a2b821a3eee1c79b/plan.json'
    $oldHash = '5abc1fc0cf732ef1ea39cc49869ba866a2ccac2f45473b871d8351a2011480bc'
    $raw = & $py -X utf8 -B -m src.validation.season_forecast prepare --reuse-manifest $old --reuse-manifest-sha256 $oldHash
    if ($LASTEXITCODE -ne 0) { throw '검증 계획 생성 실패' }
    $plan = ($raw -join "`n") | ConvertFrom-Json
    $plan | ConvertTo-Json -Depth 5
    & $py -X utf8 -B -m src.validation.season_forecast run --manifest $plan.manifest --manifest-sha256 $plan.sha256
    if ($LASTEXITCODE -ne 0) { throw '검증 실행 실패' }
    & $py -X utf8 -B -m src.validation.season_forecast verify --manifest $plan.manifest --manifest-sha256 $plan.sha256
    if ($LASTEXITCODE -ne 0) { throw '독립 검산 실패' }
    git diff --check HEAD
    if ($LASTEXITCODE -ne 0) { throw 'Git 형식 검사 실패' }
    git status --short --branch
}
```

새 계획에서 실패 후에는 같은 계획 경로 및 지문으로 `run`을 재호출한다. 코드나 입력을 변경하는 경우 이전 계획과 결과를 보존하고 새 계획을 작성한다. 종료 코드 0, 모든 검산 항목 통과 및 최신 테스트 결과를 확인한다. Git 변경은 사용자가 수행한다.

## 산출물과 후속 인계

`results/season_forecast_validation/<run_id>/plan.json`, `jobs/*.json`, `verification-<id>.json` 및 각 운영 CLI의 독립 결과를 보존한다. 경로와 지문으로 정규시즌·포스트시즌 `iterations.jsonl`, `summary.json`, `scenario.json`, `manifest.json` 및 보조 분석 JSON에 연결한다. 반복 연결 키는 `iteration_id`, 선발·일정은 `game_id`와 `team_id`, 대진은 라운드와 순서 있는 상위/하위 시드 구단 ID다. 미결정·가정·미지원은 실패 또는 실제 0과 구별한다.

실제 완료 여부·검산 수치·최신 테스트·종료 코드·적용 HEAD·하위 판정은 사용자 실행 증거 확보 후 수용 요청 보고서에 기록한다.
