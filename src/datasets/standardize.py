"""경기, 타석, 투구 및 선수 기록의 표준 데이터를 구축한다."""

from collections import Counter, defaultdict
from collections.abc import Iterator
import hashlib
from itertools import groupby
import json
import logging
from pathlib import Path
from uuid import uuid4

from ..settings import Settings
from ..validation.standard_data import (
    HIT_EVENTS,
    NON_AB_EVENTS,
    PA_EVENTS,
    POSTSEASON_NAMES,
    TEAM_CODES,
    TRANSFORM_VERSION,
    StandardDataError,
    StandardStore,
    after_state,
    before_state,
    boundary_valid,
    compare_summaries,
    hf_player_id,
    innings_to_outs,
    normalized_date,
    optional_hf_player_id,
    optional_number,
    optional_text,
    pa_state_valid,
    pitch_order_valid,
    regular_game_key,
    required_integer,
    required_text,
    team_id,
    valid_pbp_row,
    valid_pitch_target,
)
from .published_sources import (
    RUN_PATTERN,
    USE_APPROVAL,
    Row,
    SourceVersion,
    as_row,
    intake_sources,
    materialize_source,
    read_interim_rows,
    read_json,
    sha256_file,
    write_json,
)


LOGGER = logging.getLogger(__name__)

ROLE_HITTER = "hitter"
ROLE_PITCHER = "pitcher"


def _source_index(row: Row) -> int:
    return required_integer(row["_source_row"])


def _flag(value: object) -> int | None:
    if value is None:
        return None

    if type(value) is not bool:
        raise StandardDataError("논리형 값 오류")

    return int(value)


def _winner(
    away_score: int,
    home_score: int,
) -> str:
    if away_score == home_score:
        return "tie"

    return "away" if away_score > home_score else "home"


def _get_game(
    store: StandardStore,
    game_id: str,
) -> Row | None:
    row = store.connection.execute(
        """
        SELECT *
        FROM games
        WHERE game_id = ?
          AND competition = 'regular'
          AND status = 'Final'
          AND result_valid = 1
        """,
        (game_id,),
    ).fetchone()

    return dict(row) if row is not None else None


def _quarantine(
    store: StandardStore,
    source: SourceVersion,
    row: Row,
    entity: str,
    reason: str,
    impact: str,
) -> None:
    store.issue(
        reason,
        entity,
        impact,
        source_sha=source.sha256,
        source_row=_source_index(row),
        severity="quarantine",
    )


def convert_regular_games(
    store: StandardStore,
    source: SourceVersion,
    rows: Iterator[Row],
) -> None:
    """Dent 정규시즌의 완료 경기와 연기 일정을 분리한다."""
    records = list(rows)
    duplicate_counts = Counter(
        str(row["gamePk"]) for row in records
    )

    for row in records:
        index = _source_index(row)

        try:
            key = required_text(row["gamePk"])
            day = normalized_date(row["officialDate"])
            season = int(day[:4])

            if season != source.target["season"]:
                raise StandardDataError("명세 시즌과 경기 날짜 불일치")

            if row.get("isPostseason") is True:
                raise StandardDataError("정규시즌 파일의 포스트시즌 행")

            teams = as_row(row["teams"])
            away = as_row(teams["away"])
            home = as_row(teams["home"])

            away_id = required_integer(as_row(away["team"])["id"])
            home_id = required_integer(as_row(home["team"])["id"])

            if (
                away_id == home_id
                or away_id not in TEAM_CODES.values()
                or home_id not in TEAM_CODES.values()
            ):
                raise StandardDataError("경기 팀 ID 오류")

            status = required_text(
                as_row(row["status"])["detailedState"]
            )

            start_time = optional_text(row.get("gameDate"))
            away_score = None
            home_score = None
            winner = None
            result_valid = 0
            game_number = None
            tie = None

            if status == "Final":
                regular_game_key(
                    key,
                    day,
                    key[8:10],
                    key[10:12],
                )

                if (
                    TEAM_CODES.get(key[8:10]) != away_id
                    or TEAM_CODES.get(key[10:12]) != home_id
                ):
                    raise StandardDataError("경기 ID와 팀 ID 불일치")

                away_score = required_integer(away["score"])
                home_score = required_integer(home["score"])
                winner = _winner(away_score, home_score)
                tie = int(winner == "tie")

                if (
                    row.get("isTie") is not bool(tie)
                    or away.get("isWinner") is not (winner == "away")
                    or home.get("isWinner") is not (winner == "home")
                ):
                    raise StandardDataError("경기 점수와 승리 팀 불일치")

                if duplicate_counts[key] != 1:
                    raise StandardDataError("완료 경기 ID 중복")

                game_id = key
                game_number = int(key[-1])
                result_valid = 1

            else:
                # 연기된 같은 ID는 시간이나 별도 일정의 가능성이
                # 있으므로 원본 행 단위로 보존한다.
                game_id = f"schedule:{key}:{index}"

                if status != "Postponed":
                    store.issue(
                        "unrecognized_game_status",
                        "game",
                        "M1 및 시즌 전망 정답 제외",
                        entity_key=game_id,
                        source_sha=source.sha256,
                        source_row=index,
                    )

                if duplicate_counts[key] > 1:
                    store.issue(
                        "duplicate_schedule_id_preserved",
                        "schedule",
                        "일정 행 보존; 동일 완료 경기로 병합하지 않음",
                        entity_key=key,
                        source_sha=source.sha256,
                        source_row=index,
                    )

            store.put("games", {
                "game_id": game_id,
                "source_key": key,
                "season": season,
                "competition": "regular",
                "stage": None,
                "date": day,
                "start_time": start_time,
                "game_number": game_number,
                "away_team_id": away_id,
                "home_team_id": home_id,
                "away_score": away_score,
                "home_score": home_score,
                "status": status,
                "winner": winner,
                "is_tie": tie,
                "result_valid": result_valid,
                "season_complete": _flag(
                    source.target.get("season_complete")
                ),
                "source_sha": source.sha256,
                "source_row": index,
            })

        except (StandardDataError, KeyError, TypeError) as exc:
            _quarantine(
                store,
                source,
                row,
                "game",
                str(exc),
                "M1 및 시즌 전망 정답 제외",
            )


def convert_postseason(
    store: StandardStore,
    source: SourceVersion,
    rows: Iterator[Row],
) -> None:
    """실제 점수가 있는 포스트시즌과 미완료 일정을 구분한다."""
    observed_dent = source.target_id.startswith("dent-")

    for row in rows:
        index = _source_index(row)

        try:
            if observed_dent:
                day = normalized_date(row["date"])
                stage = required_text(row["_stage"])
                away_id = team_id(row["away"])
                home_id = team_id(row["home"])
                number = required_integer(row["game"])
                final = row.get("isFinal") is True
                completed = _flag(row.get("_season_complete"))
                time = optional_text(row.get("time"))
                source_key = f"{day[:4]}:{stage}:{number}"
                game_id = f"postseason:dent:{source_key}"

                away_score = (
                    required_integer(row["awayScore"])
                    if final
                    else None
                )
                home_score = (
                    required_integer(row["homeScore"])
                    if final
                    else None
                )

                if final:
                    winning = _winner(away_score, home_score)
                    expected_winner = {
                        "away": away_id,
                        "home": home_id,
                        "tie": None,
                    }[winning]

                    if row.get("winnerId") != expected_winner:
                        raise StandardDataError("포스트시즌 승자 불일치")

            else:
                stage = POSTSEASON_NAMES.get(row.get("비고"))

                if stage is None:
                    store.issue(
                        "outside_selected_postseason_scope",
                        "source_row",
                        "중간 자료만 보존; 포스트시즌 결과에서 제외",
                        source_sha=source.sha256,
                        source_row=index,
                        severity="excluded",
                    )
                    continue

                raw_date = required_text(row["Date"])
                day = normalized_date(raw_date[:10])
                time = raw_date[11:] or None
                away_id = team_id(row["원정팀"])
                home_id = team_id(row["홈팀"])
                away_score = required_integer(int(row["원정팀점수"]))
                home_score = required_integer(int(row["홈팀점수"]))
                number = None
                completed = None
                final = True
                source_key = f"{day}:{stage}:{away_id}:{home_id}"
                game_id = f"postseason:lopes:{source_key}:{index}"

                winning = _winner(away_score, home_score)

                expected = {
                    "away": ("승", "패"),
                    "home": ("패", "승"),
                    "tie": ("무", "무"),
                }[winning]

                if (
                    row["원정팀결과"],
                    row["홈팀결과"],
                ) != expected:
                    raise StandardDataError(
                        "LOPES 점수와 승패 표기 불일치"
                    )

            if away_id == home_id:
                raise StandardDataError("포스트시즌 홈·원정 팀 중복")

            store.put("games", {
                "game_id": game_id,
                "source_key": source_key,
                "season": int(day[:4]),
                "competition": "postseason",
                "stage": stage,
                "date": day,
                "start_time": time,
                "game_number": number,
                "away_team_id": away_id,
                "home_team_id": home_id,
                "away_score": away_score,
                "home_score": home_score,
                "status": "Final" if final else "Unconfirmed",
                "winner": (
                    _winner(away_score, home_score)
                    if final
                    else None
                ),
                "is_tie": (
                    int(away_score == home_score)
                    if final
                    else None
                ),
                "result_valid": int(final),
                "season_complete": completed,
                "source_sha": source.sha256,
                "source_row": index,
            })

        except (StandardDataError, KeyError, TypeError, ValueError) as exc:
            _quarantine(
                store,
                source,
                row,
                "postseason",
                str(exc),
                "포스트시즌 실제 결과에서 제외",
            )


def convert_schedule_reference(
    store: StandardStore,
    source: SourceVersion,
    rows: Iterator[Row],
) -> None:
    """일정 참조 자료의 원래 대회명과 불확실성을 보존한다."""
    records = list(rows)
    counts = Counter(
        row.get("G_ID")
        for row in records
        if row.get("G_ID")
    )

    for row in records:
        index = _source_index(row)

        try:
            day = normalized_date(row["G_DT"])
            key = optional_text(row["G_ID"])
            away = optional_text(row.get("AWAY_ID"))
            home = optional_text(row.get("HOME_ID"))
            series_name = optional_text(row.get("SR_NM"))

            if key is None:
                classification = "no_game"

            else:
                regular_game_key(key, day, away, home)

                classification = (
                    "regular_reference"
                    if row.get("SR_ID") == "0"
                    and series_name == "정규시즌"
                    else "competition_unverified"
                )

            store.put("schedule_reference", {
                "source_sha": source.sha256,
                "source_row": index,
                "source_key": key,
                "date": day,
                "start_time": optional_text(row.get("G_TM")),
                "away_code": away,
                "home_code": home,
                "series_code": optional_text(row.get("SR_ID")),
                "series_name": series_name,
                "cancellation_code": optional_text(
                    row.get("CANCEL_SC_ID")
                ),
                "cancellation_name": optional_text(
                    row.get("CANCEL_SC_NM")
                ),
                "classification": classification,
            })

            if key is not None and counts[key] > 1:
                store.issue(
                    "duplicate_schedule_reference",
                    "schedule",
                    "중단·재개 여부 미확인; 두 일정 모두 보존",
                    entity_key=key,
                    source_sha=source.sha256,
                    source_row=index,
                )

            if classification == "competition_unverified":
                store.issue(
                    "competition_classification_unverified",
                    "schedule",
                    "원본 대회명 보존; 결과·승자·시리즈 정답 미생성",
                    entity_key=key,
                    source_sha=source.sha256,
                    source_row=index,
                )

        except (StandardDataError, KeyError, TypeError) as exc:
            _quarantine(
                store,
                source,
                row,
                "schedule",
                str(exc),
                "일정 표준 연결 제외",
            )


def _add_player_observation(
    store: StandardStore,
    source: SourceVersion,
    row: Row,
    role: str,
    season: int,
    team: int,
) -> str:
    field = "batter" if role == ROLE_HITTER else "pitcher"
    player = hf_player_id(row[field])

    store.put(
        "player_observations",
        {
            "player_id": player,
            "season": season,
            "team_id": team,
            "name": required_text(row[f"{field}_name"]),
            "source_sha": source.sha256,
            "source_row": _source_index(row),
        },
        ignore_existing=True,
    )

    return player


def _state_record(
    state: tuple[object, ...],
    prefix: str,
) -> Row:
    return {
        f"{prefix}_home_score": state[0],
        f"{prefix}_away_score": state[1],
        f"{prefix}_outs": state[2],
        f"{prefix}_on_1b": optional_hf_player_id(state[3]),
        f"{prefix}_on_2b": optional_hf_player_id(state[4]),
        f"{prefix}_on_3b": optional_hf_player_id(state[5]),
    }


def _store_starter(
    store: StandardStore,
    source: SourceVersion,
    row: Row,
    game: Row,
    side: str,
    name: str,
    player: str | None,
    method: str,
    valid: bool,
) -> None:
    store.put("starters", {
        "source_sha": source.sha256,
        "source_row": _source_index(row),
        "game_id": game["game_id"],
        "side": side,
        "team_id": game[f"{side}_team_id"],
        "name": name,
        "player_id": player,
        "method": method,
        "valid": int(valid),
    })


def _terminal_state(last: Row, game: Row) -> str:
    """경기 종료 원인을 단정하지 않고 확인된 형태를 분류한다."""
    if game["winner"] == "tie":
        return "final_tie"

    if (
        last["inning_topbot"] == "bot"
        and last["post_outs"] < 3
        and game["winner"] == "home"
    ):
        return "home_leads_partial_bottom_candidate"

    if last["post_outs"] == 3:
        return "half_ended_three_outs"

    return "terminal_reason_unresolved"


def _convert_pbp_game(
    store: StandardStore,
    source: SourceVersion,
    rows: list[Row],
) -> None:
    """한 경기의 원본 순서를 보존하면서 논리적 타석으로 집계한다."""
    first = rows[0]
    last = rows[-1]
    raw_game_id = str(first["game_pk"])

    try:
        game_id = regular_game_key(
            raw_game_id,
            first["game_date"],
            first["away_team"],
            first["home_team"],
            hf=True,
        )

        game = _get_game(store, game_id)

        if game is None:
            raise StandardDataError("완료 경기와 PBP 연결 실패")

        if (
            team_id(first["away_team"]) != game["away_team_id"]
            or team_id(first["home_team"]) != game["home_team_id"]
            or first["game_date"] != game["date"]
        ):
            raise StandardDataError("경기 날짜·팀 교차 대조 불일치")

        if not all(
            valid_pbp_row(row)
            and row["game_pk"] == raw_game_id
            and row["game_date"] == first["game_date"]
            and row["away_team"] == first["away_team"]
            and row["home_team"] == first["home_team"]
            for row in rows
        ):
            raise StandardDataError("PBP 핵심 스키마 또는 경기 키 오류")

    except (StandardDataError, KeyError, TypeError) as exc:
        for row in rows:
            _quarantine(
                store,
                source,
                row,
                "pbp",
                str(exc),
                "M2·M3·M4·M5 경기 단위 격리",
            )
        return

    appearances = [
        list(group)
        for _, group in groupby(
            rows,
            key=lambda row: row["at_bat_number"],
        )
    ]

    numbers = [
        int(group[0]["at_bat_number"])
        for group in appearances
    ]

    if len(set(numbers)) != len(numbers):
        for row in rows:
            _quarantine(
                store,
                source,
                row,
                "pbp",
                "noncontiguous_duplicate_pa",
                "논리적 타석 키 충돌로 경기 단위 격리",
            )
        return

    order_complete = numbers == list(
        range(1, len(appearances) + 1)
    )

    score_match = (
        last["post_home_score"] == game["home_score"]
        and last["post_away_score"] == game["away_score"]
    )

    if not order_complete:
        store.issue(
            "pa_sequence_gap",
            "game",
            "M3 전체 관측 집계 및 M4 연속 상태 제한",
            entity_key=game_id,
            source_sha=source.sha256,
            source_row=_source_index(first),
        )

    if not score_match:
        store.issue(
            "final_score_mismatch",
            "game",
            "M3·M4 전체 경기 정답 후보 제한",
            entity_key=game_id,
            source_sha=source.sha256,
            source_row=_source_index(last),
            severity="quarantine",
        )

    stats: dict[
        tuple[int, str, str],
        Counter[str],
    ] = defaultdict(Counter)

    all_states_valid = True
    all_pitches_ordered = True
    starter_sides: set[str] = set()

    for position, pa in enumerate(appearances):
        start = pa[0]
        end = pa[-1]
        number = int(start["at_bat_number"])
        half = str(start["inning_topbot"])

        batting_side = "away" if half == "top" else "home"
        pitching_side = "home" if half == "top" else "away"

        batting_team = int(game[f"{batting_side}_team_id"])
        pitching_team = int(game[f"{pitching_side}_team_id"])

        batter_ids = []
        pitcher_ids = []

        for row in pa:
            batter_ids.append(_add_player_observation(
                store,
                source,
                row,
                ROLE_HITTER,
                int(game["season"]),
                batting_team,
            ))
            pitcher_ids.append(_add_player_observation(
                store,
                source,
                row,
                ROLE_PITCHER,
                int(game["season"]),
                pitching_team,
            ))

        batter_changed = len(set(batter_ids)) > 1
        pitcher_changed = len(set(pitcher_ids)) > 1
        participant_changed = batter_changed or pitcher_changed

        events = {row["events"] for row in pa}

        event = (
            next(iter(events))
            if len(events) == 1
            else None
        )

        label_valid = (
            len(events) == 1
            and event in PA_EVENTS
            and score_match
        )

        pitch_order = pitch_order_valid(pa)
        state_valid = pa_state_valid(pa) and score_match

        if position > 0:
            previous = appearances[position - 1][-1]
            previous_number = int(previous["at_bat_number"])

            if (
                number != previous_number + 1
                or not boundary_valid(previous, start)
            ):
                state_valid = False

        if position + 1 < len(appearances):
            following = appearances[position + 1][0]

            if (
                int(following["at_bat_number"]) != number + 1
                or not boundary_valid(end, following)
            ):
                state_valid = False

        if not pitch_order:
            all_pitches_ordered = False

            store.issue(
                "pitch_sequence_gap_or_duplicate",
                "pa",
                "M5 연속 투구 순서 후보 제한",
                entity_key=f"{game_id}:{number}",
                source_sha=source.sha256,
                source_row=_source_index(start),
                severity="quarantine",
            )

        if not state_valid:
            all_states_valid = False

            store.issue(
                "pa_state_or_boundary_unverified",
                "pa",
                "M4 상태 전이와 투수 아웃 귀속 제한",
                entity_key=f"{game_id}:{number}",
                source_sha=source.sha256,
                source_row=_source_index(start),
                severity="quarantine",
            )

        if not label_valid:
            store.issue(
                (
                    "unlabeled_half_end"
                    if event is None and end["post_outs"] == 3
                    else "pa_result_unverified"
                ),
                "pa",
                "공식 타석·타수·상대 타자 집계에서 제외",
                entity_key=f"{game_id}:{number}",
                source_sha=source.sha256,
                source_row=_source_index(start),
                severity="quarantine",
            )

        if participant_changed:
            store.issue(
                "participant_change_official_credit_unverified",
                "pa",
                "시작·종료 참가자 보존; 변경 역할의 공식 귀속 미확정",
                entity_key=f"{game_id}:{number}",
                source_sha=source.sha256,
                source_row=_source_index(start),
            )

        credited_batter = (
            batter_ids[-1]
            if label_valid and not batter_changed
            else None
        )

        credited_pitcher = (
            pitcher_ids[-1]
            if label_valid and not pitcher_changed
            else None
        )

        positive_pitches = sum(
            int(row["pitch_number"]) > 0 for row in pa
        )

        store.put("plate_appearances", {
            "game_id": game_id,
            "pa_number": number,
            "inning": start["inning"],
            "half": half,
            "start_batter_id": batter_ids[0],
            "end_batter_id": batter_ids[-1],
            "start_pitcher_id": pitcher_ids[0],
            "end_pitcher_id": pitcher_ids[-1],
            "credited_batter_candidate": credited_batter,
            "credited_pitcher_candidate": credited_pitcher,
            "participant_changed": int(participant_changed),
            "event": event,
            "result_valid": int(label_valid),
            "state_valid": int(state_valid),
            "physical_pitch_rows": positive_pitches,
            "pitch_order_valid": int(pitch_order),
            **_state_record(before_state(start), "pre"),
            **_state_record(after_state(end), "post"),
            "source_sha": source.sha256,
            "start_source_row": _source_index(start),
            "end_source_row": _source_index(end),
        })

        if (
            pitching_side not in starter_sides
            and start["inning"] == 1
            and start["outs_when_up"] == 0
        ):
            _store_starter(
                store,
                source,
                start,
                game,
                pitching_side,
                str(start["pitcher_name"]),
                pitcher_ids[0],
                "hf_first_observed",
                pitch_order and score_match,
            )

            starter_sides.add(pitching_side)

        for batter in set(batter_ids):
            counter = stats[
                (batting_team, batter, ROLE_HITTER)
            ]

            if batter != credited_batter:
                counter["ambiguous_pa"] += 1

        for pitcher in set(pitcher_ids):
            counter = stats[
                (pitching_team, pitcher, ROLE_PITCHER)
            ]

            if pitcher != credited_pitcher:
                counter["ambiguous_pa"] += 1

            if pitcher_changed or not state_valid:
                counter["ambiguous_outs"] += 1

        if credited_batter is not None:
            counter = stats[
                (batting_team, credited_batter, ROLE_HITTER)
            ]

            counter["pa"] += 1
            counter["ab"] += int(event not in NON_AB_EVENTS)
            counter["hits"] += int(event in HIT_EVENTS)
            counter["home_runs"] += int(event == "home_run")
            counter["walks"] += int(event == "walk")
            counter["hbp"] += int(event == "hit_by_pitch")
            counter["strikeouts"] += int(event == "strikeout")

        if credited_pitcher is not None:
            counter = stats[
                (pitching_team, credited_pitcher, ROLE_PITCHER)
            ]

            counter["bf"] += 1
            counter["ab"] += int(event not in NON_AB_EVENTS)
            counter["hits"] += int(event in HIT_EVENTS)
            counter["home_runs"] += int(event == "home_run")
            counter["walks"] += int(event == "walk")
            counter["hbp"] += int(event == "hit_by_pitch")
            counter["strikeouts"] += int(event == "strikeout")

        if state_valid and not pitcher_changed:
            counter = stats[
                (pitching_team, pitcher_ids[-1], ROLE_PITCHER)
            ]

            counter["outs"] += (
                int(end["post_outs"])
                - int(start["outs_when_up"])
            )

        seen_pitch_numbers: set[int] = set()

        for row, batter, pitcher in zip(
            pa,
            batter_ids,
            pitcher_ids,
        ):
            index = _source_index(row)
            pitch_number = int(row["pitch_number"])

            if pitch_number == 0:
                store.put("non_pitch", {
                    "source_sha": source.sha256,
                    "source_row": index,
                    "game_id": game_id,
                    "pa_number": number,
                    "event": event,
                })
                continue

            if pitch_number in seen_pitch_numbers:
                _quarantine(
                    store,
                    source,
                    row,
                    "pitch",
                    "duplicate_pitch_number",
                    "실제 투구 키 중복",
                )
                continue

            seen_pitch_numbers.add(pitch_number)

            target_valid = valid_pitch_target(row)

            if not target_valid:
                store.issue(
                    "pitch_target_missing_or_unknown",
                    "pitch",
                    "M5 구종·결과 정답 후보 제외",
                    entity_key=(
                        f"{game_id}:{number}:{pitch_number}"
                    ),
                    source_sha=source.sha256,
                    source_row=index,
                    severity="quarantine",
                )

            counter = stats[
                (pitching_team, pitcher, ROLE_PITCHER)
            ]
            counter["pitches"] += 1

            measurements = {}

            for field in (
                "release_speed_kmh",
                "plate_x",
                "plate_z",
            ):
                try:
                    measurements[field] = optional_number(
                        row.get(field)
                    )
                except StandardDataError:
                    measurements[field] = None

                    store.issue(
                        f"invalid_measurement:{field}",
                        "pitch",
                        "측정값 NULL 처리; 구종·결과와 별도 판정",
                        source_sha=source.sha256,
                        source_row=index,
                    )

            store.put("pitches", {
                "game_id": game_id,
                "pa_number": number,
                "pitch_number": pitch_number,
                "batter_id": batter,
                "pitcher_id": pitcher,
                "pitch_type": row.get("pitch_type"),
                "pitch_result": row.get("pitch_result"),
                "pitch_result_type": row.get("type"),
                **measurements,
                "balls": row["balls"],
                "strikes": row["strikes"],
                "outs_when_up": row["outs_when_up"],
                "home_score": row["home_score"],
                "away_score": row["away_score"],
                "on_1b": optional_hf_player_id(row["on_1b"]),
                "on_2b": optional_hf_player_id(row["on_2b"]),
                "on_3b": optional_hf_player_id(row["on_3b"]),
                "state_valid": int(state_valid),
                "order_valid": int(pitch_order),
                "target_valid": int(target_valid),
                "source_sha": source.sha256,
                "source_row": index,
            })

    complete = (
        score_match
        and order_complete
        and all_states_valid
        and all_pitches_ordered
    )

    for (team, player, role), counter in sorted(stats.items()):
        hitter = role == ROLE_HITTER

        store.put("player_games", {
            "game_id": game_id,
            "team_id": team,
            "player_id": player,
            "role": role,
            "observed_pa": counter["pa"] if hitter else None,
            "observed_ab": counter["ab"],
            "observed_hits": counter["hits"],
            "observed_home_runs": counter["home_runs"],
            "observed_walks": counter["walks"],
            "observed_hbp": counter["hbp"],
            "observed_strikeouts": counter["strikeouts"],
            "observed_bf": None if hitter else counter["bf"],
            "observed_outs": None if hitter else counter["outs"],
            "observed_pitches": (
                None if hitter else counter["pitches"]
            ),
            "ambiguous_pa": counter["ambiguous_pa"],
            "ambiguous_outs": counter["ambiguous_outs"],
            "coverage_candidate": int(
                complete
                and counter["ambiguous_pa"] == 0
                and counter["ambiguous_outs"] == 0
            ),
            # PBP의 점수 변화만으로 자책점을 확정하지 않는다.
            "earned_runs": None,
            "source_sha": source.sha256,
        })

    store.put("game_checks", {
        "game_id": game_id,
        "source_sha": source.sha256,
        "physical_rows": len(rows),
        "pa_count": len(appearances),
        "positive_pitch_rows": sum(
            int(row["pitch_number"]) > 0 for row in rows
        ),
        "non_pitch_rows": sum(
            row["pitch_number"] == 0 for row in rows
        ),
        "final_score_match": int(score_match),
        "order_valid": int(
            order_complete and all_pitches_ordered
        ),
        "boundary_valid": int(all_states_valid),
        "terminal_state": _terminal_state(last, game),
    })


def convert_pbp(
    store: StandardStore,
    source: SourceVersion,
    rows: Iterator[Row],
) -> None:
    """시즌 전체 PBP를 원래 경기 순서대로 처리한다."""
    seen_games: set[str] = set()

    for key, group in groupby(
        rows,
        key=lambda row: row["game_pk"],
    ):
        game_id = str(key)

        if game_id in seen_games:
            raise StandardDataError(
                f"원본에서 경기 그룹이 비연속으로 повтор됩니다: {game_id}"
            )

        seen_games.add(game_id)

        _convert_pbp_game(
            store,
            source,
            list(group),
        )


def _window_game(
    store: StandardStore,
    row: Row,
) -> Row:
    key = required_text(row["game_id"])
    game = _get_game(store, key)

    if game is None:
        raise StandardDataError("Window 완료 경기 연결 실패")

    if "date" in row and normalized_date(row["date"]) != game["date"]:
        raise StandardDataError("Window 경기 날짜 불일치")

    return game


def _compare_starter(
    store: StandardStore,
    source: SourceVersion,
    row: Row,
    game: Row,
    side: str,
    name: str,
    method: str,
) -> None:
    candidate = store.connection.execute(
        """
        SELECT name
        FROM starters
        WHERE game_id = ?
          AND side = ?
          AND method = 'hf_first_observed'
          AND valid = 1
        """,
        (game["game_id"], side),
    ).fetchone()

    store.comparison(
        source.sha256,
        _source_index(row),
        f"{method}:{side}:starter_name",
        str(game["game_id"]),
        name,
        candidate[0] if candidate is not None else None,
    )


def convert_window_starters(
    store: StandardStore,
    source: SourceVersion,
    rows: Iterator[Row],
    *,
    daily: bool = False,
) -> None:
    """실제 선발과 일일 경기 결과를 경기 키로 연결한다."""
    method = "window_daily" if daily else "window_history"

    for row in rows:
        try:
            game = _window_game(store, row)

            if daily:
                if row.get("status") != "final":
                    raise StandardDataError("완료 경기 상태 불일치")

                for side in ("away", "home"):
                    if team_id(row[f"{side}_team"]) != game[
                        f"{side}_team_id"
                    ]:
                        raise StandardDataError(
                            "Window 일일 경기 팀 불일치"
                        )

                    store.comparison(
                        source.sha256,
                        _source_index(row),
                        f"{side}:score",
                        str(game["game_id"]),
                        row[f"{side}_score"],
                        game[f"{side}_score"],
                    )

            for side in ("away", "home"):
                name = required_text(row[f"{side}_starter"])

                _store_starter(
                    store,
                    source,
                    row,
                    game,
                    side,
                    name,
                    None,
                    method,
                    True,
                )

                _compare_starter(
                    store,
                    source,
                    row,
                    game,
                    side,
                    name,
                    method,
                )

            # 시즌 최종 ERA/FIP/WAR 등은 경기 전 입력으로 복원하지 않는다.
            # 해당 원본 필드는 interim에만 보존한다.

        except (StandardDataError, KeyError, TypeError) as exc:
            _quarantine(
                store,
                source,
                row,
                "starter",
                str(exc),
                "실제 선발 연결 제외",
            )


def _hf_pitcher_candidate(
    store: StandardStore,
    game: Row,
    team: int,
    raw_player: str | None,
    name: str,
) -> str | None:
    """동일 경기·팀·원본 ID·이름이 일치할 때만 연결 후보를 반환한다."""
    if raw_player is None:
        return None

    candidate = hf_player_id(raw_player)

    matches = store.connection.execute(
        """
        SELECT DISTINCT p.player_id
        FROM player_games AS p
        JOIN player_observations AS o
          ON o.player_id = p.player_id
         AND o.team_id = p.team_id
         AND o.season = ?
        WHERE p.game_id = ?
          AND p.team_id = ?
          AND p.role = 'pitcher'
          AND p.player_id = ?
          AND o.name = ?
        """,
        (
            game["season"],
            game["game_id"],
            team,
            candidate,
            name,
        ),
    ).fetchall()

    return candidate if len(matches) == 1 else None


def convert_window_pitching(
    store: StandardStore,
    source: SourceVersion,
    rows: Iterator[Row],
) -> None:
    """직접 관측된 투수 등판 기록과 PBP 집계 후보를 비교한다."""
    records = list(rows)

    name_contexts = Counter(
        (row.get("game_id"), row.get("side"), row.get("name"))
        for row in records
    )

    groups: dict[
        tuple[str, str],
        list[Row],
    ] = defaultdict(list)

    for row in records:
        index = _source_index(row)

        try:
            game = _window_game(store, row)
            side = required_text(row["side"])

            if side not in {"away", "home"}:
                raise StandardDataError("투수 소속 경기 방향 오류")

            team = team_id(row["team"])

            if team != game[f"{side}_team_id"]:
                raise StandardDataError("투수 소속 팀 불일치")

            name = required_text(row["name"])
            raw_player = optional_text(row.get("player_id"))
            candidate = _hf_pitcher_candidate(
                store,
                game,
                team,
                raw_player,
                name,
            )

            outs = required_integer(row["outs"])
            converted_outs = innings_to_outs(row["innings"])
            valid = outs == converted_outs

            values = {
                "batters_faced": required_integer(row["tbf"]),
                "pitches": required_integer(row["pitches"]),
                "at_bats": required_integer(row["ab"]),
                "hits": required_integer(row["h"]),
                "home_runs": required_integer(row["hr"]),
                "walks_hbp": required_integer(row["bb_hbp"]),
                "strikeouts": required_integer(row["so"]),
                "runs": required_integer(row["r"]),
                "earned_runs": required_integer(row["er"]),
            }

            if (
                values["hits"] > values["at_bats"]
                or values["home_runs"] > values["hits"]
                or values["earned_runs"] > values["runs"]
                or values["at_bats"] > values["batters_faced"]
            ):
                valid = False

            starter = row.get("is_starter")

            if type(starter) is not bool:
                raise StandardDataError("선발 여부 타입 오류")

            store.put("pitching_records", {
                "source_sha": source.sha256,
                "source_row": index,
                "game_id": game["game_id"],
                "side": side,
                "team_id": team,
                "name": name,
                "raw_player_id": raw_player,
                "hf_candidate_id": candidate,
                "is_starter": int(starter),
                "outs": outs,
                **values,
                "valid": int(valid),
            })

            store.comparison(
                source.sha256,
                index,
                "innings_to_outs",
                str(game["game_id"]),
                outs,
                converted_outs,
            )

            if not valid:
                store.issue(
                    "direct_pitching_stat_inconsistent",
                    "player_game",
                    "M3 직접 투수 기록 정답 후보 제외",
                    source_sha=source.sha256,
                    source_row=index,
                    severity="quarantine",
                )

            if candidate is None:
                reason = (
                    "player_id_missing"
                    if raw_player is None
                    else "player_identity_unmatched"
                )

                store.issue(
                    reason,
                    "player_game",
                    "선수 ID 자동 병합 금지; 직접 기록은 별도 보존",
                    source_sha=source.sha256,
                    source_row=index,
                )

            if name_contexts[
                (row["game_id"], side, name)
            ] > 1:
                store.issue(
                    "same_game_team_player_name_collision",
                    "player_game",
                    "동명이인 가능성; 원본 행별 등판 기록 유지",
                    source_sha=source.sha256,
                    source_row=index,
                )

            if starter:
                _store_starter(
                    store,
                    source,
                    row,
                    game,
                    side,
                    name,
                    candidate,
                    "window_pitching",
                    valid,
                )

                _compare_starter(
                    store,
                    source,
                    row,
                    game,
                    side,
                    name,
                    "window_pitching",
                )

            if candidate is not None:
                derived = store.connection.execute(
                    """
                    SELECT *
                    FROM player_games
                    WHERE game_id = ?
                      AND team_id = ?
                      AND player_id = ?
                      AND role = 'pitcher'
                    """,
                    (
                        game["game_id"],
                        team,
                        candidate,
                    ),
                ).fetchone()

                mappings = {
                    "outs": "observed_outs",
                    "batters_faced": "observed_bf",
                    "pitches": "observed_pitches",
                    "at_bats": "observed_ab",
                    "hits": "observed_hits",
                    "home_runs": "observed_home_runs",
                    "strikeouts": "observed_strikeouts",
                }

                direct = {"outs": outs, **values}

                for metric, field in mappings.items():
                    value = (
                        derived[field]
                        if derived is not None and valid
                        else None
                    )

                    store.comparison(
                        source.sha256,
                        index,
                        f"player:{metric}",
                        str(game["game_id"]),
                        direct[metric],
                        value,
                    )

            groups[(str(game["game_id"]), side)].append({
                "_source_row": index,
                "team_id": team,
                "valid": int(valid),
                "outs": outs,
                **values,
            })

        except (StandardDataError, KeyError, TypeError) as exc:
            _quarantine(
                store,
                source,
                row,
                "player_game",
                str(exc),
                "직접 투수 기록 연결 제외",
            )

    for (game_id, side), group in sorted(groups.items()):
        index = _source_index(group[0])
        team = group[0]["team_id"]

        derived = store.connection.execute(
            """
            SELECT
                SUM(observed_outs) AS outs,
                SUM(observed_bf) AS batters_faced,
                SUM(observed_pitches) AS pitches,
                SUM(observed_ab) AS at_bats,
                SUM(observed_hits) AS hits,
                SUM(observed_home_runs) AS home_runs,
                SUM(observed_strikeouts) AS strikeouts
            FROM player_games
            WHERE game_id = ?
              AND team_id = ?
              AND role = 'pitcher'
            """,
            (game_id, team),
        ).fetchone()

        for metric in (
            "outs",
            "batters_faced",
            "pitches",
            "at_bats",
            "hits",
            "home_runs",
            "strikeouts",
        ):
            direct = (
                sum(int(row[metric]) for row in group)
                if all(row["valid"] for row in group)
                else None
            )

            store.comparison(
                source.sha256,
                index,
                f"team:{side}:{metric}",
                game_id,
                direct,
                derived[metric],
            )


def inspect_player_identity(
    store: StandardStore,
) -> None:
    """HF 내부 ID·이름·팀 변경을 검사하되 자동 병합하지 않는다."""
    for row in store.connection.execute(
        """
        SELECT player_id, COUNT(DISTINCT name) AS names
        FROM player_observations
        GROUP BY player_id
        HAVING COUNT(DISTINCT name) > 1
        ORDER BY player_id
        """
    ):
        store.issue(
            "player_id_multiple_names",
            "player",
            "이름 변경·표기 차이·ID 충돌 여부 검토",
            entity_key=row["player_id"],
        )

    for row in store.connection.execute(
        """
        SELECT name, COUNT(DISTINCT player_id) AS players
        FROM player_observations
        GROUP BY name
        HAVING COUNT(DISTINCT player_id) > 1
        ORDER BY name
        """
    ):
        store.issue(
            "player_name_multiple_ids",
            "player",
            "동명이인 또는 별도 ID 유지",
            entity_key=row["name"],
        )

    for row in store.connection.execute(
        """
        SELECT player_id, COUNT(DISTINCT team_id) AS teams
        FROM player_observations
        GROUP BY player_id
        HAVING COUNT(DISTINCT team_id) > 1
        ORDER BY player_id
        """
    ):
        store.issue(
            "player_multiple_team_observations",
            "player",
            "시즌·팀별 관측 유지; 소속 이동 자동 확정 금지",
            entity_key=row["player_id"],
            severity="information",
        )


def coverage_report(store: StandardStore) -> Row:
    """분모를 포함한 시즌·기능별 유효 범위 및 비교 결과를 반환한다."""
    connection = store.connection

    seasons = [
        dict(row)
        for row in connection.execute(
            """
            SELECT
                season,
                competition,
                COUNT(*) AS schedule_rows,
                SUM(result_valid) AS valid_results,
                SUM(status = 'Postponed') AS postponed_rows
            FROM games
            GROUP BY season, competition
            ORDER BY season, competition
            """
        )
    ]

    pbp = [
        dict(row)
        for row in connection.execute(
            """
            SELECT
                g.season,
                COUNT(*) AS games,
                SUM(c.physical_rows) AS physical_rows,
                SUM(c.pa_count) AS pa_count,
                SUM(c.positive_pitch_rows) AS positive_pitch_rows,
                SUM(c.non_pitch_rows) AS non_pitch_rows,
                SUM(c.final_score_match) AS score_matches,
                SUM(c.order_valid) AS ordered_games,
                SUM(c.boundary_valid) AS valid_boundary_games
            FROM game_checks AS c
            JOIN games AS g ON g.game_id = c.game_id
            GROUP BY g.season
            ORDER BY g.season
            """
        )
    ]

    pa = dict(connection.execute(
        """
        SELECT COUNT(*) AS total,
               SUM(result_valid) AS result_candidates,
               SUM(state_valid) AS valid_states,
               SUM(participant_changed) AS participant_changes,
               SUM(pitch_order_valid) AS ordered_pa
        FROM plate_appearances
        """
    ).fetchone())

    pitches = dict(connection.execute(
        """
        SELECT COUNT(*) AS total,
               SUM(target_valid) AS target_candidates,
               SUM(order_valid) AS ordered_pitches,
               SUM(state_valid) AS state_candidates,
               SUM(plate_x IS NULL OR plate_z IS NULL)
                   AS missing_location
        FROM pitches
        """
    ).fetchone())

    player_games = [
        dict(row)
        for row in connection.execute(
            """
            SELECT role,
                   COUNT(*) AS observed_player_games,
                   SUM(coverage_candidate) AS coverage_candidates,
                   SUM(ambiguous_pa) AS ambiguous_pa,
                   SUM(ambiguous_outs) AS ambiguous_outs
            FROM player_games
            GROUP BY role
            ORDER BY role
            """
        )
    ]

    historical_denominator = connection.execute(
        """
        SELECT COUNT(*)
        FROM games
        WHERE competition = 'regular'
          AND result_valid = 1
          AND season BETWEEN 2001 AND 2025
        """
    ).fetchone()[0]

    historical_starters = connection.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT game_id
            FROM starters
            WHERE method = 'window_history' AND valid = 1
            GROUP BY game_id
            HAVING COUNT(DISTINCT side) = 2
        )
        """
    ).fetchone()[0]

    denominator_2026 = connection.execute(
        """
        SELECT COUNT(*)
        FROM games
        WHERE competition = 'regular'
          AND result_valid = 1
          AND season = 2026
        """
    ).fetchone()[0]

    direct_pitching_games = connection.execute(
        """
        SELECT COUNT(DISTINCT game_id)
        FROM pitching_records
        """
    ).fetchone()[0]

    comparisons = [
        dict(row)
        for row in connection.execute(
            """
            SELECT metric, status, COUNT(*) AS count
            FROM comparisons
            GROUP BY metric, status
            ORDER BY metric, status
            """
        )
    ]

    postseason = [
        dict(row)
        for row in connection.execute(
            """
            SELECT season, stage,
                   COUNT(*) AS observed_rows,
                   SUM(result_valid) AS valid_results,
                   MAX(season_complete) AS season_complete
            FROM games
            WHERE competition = 'postseason'
            GROUP BY season, stage
            ORDER BY season, stage
            """
        )
    ]

    return {
        "games_by_season": seasons,
        "pbp_by_season": pbp,
        "plate_appearances": pa,
        "pitches": pitches,
        "player_games": player_games,
        "historical_starters": {
            "denominator": historical_denominator,
            "observed": historical_starters,
            "missing": historical_denominator - historical_starters,
        },
        "direct_pitching_2026": {
            "denominator": denominator_2026,
            "observed": direct_pitching_games,
            "missing": denominator_2026 - direct_pitching_games,
        },
        "postseason": postseason,
        "comparisons": comparisons,
        "limitations": {
            "M1": [
                "실제 경기 결과와 당시 공개된 입력은 별개",
                "실제 선발 기록의 과거 공개 시점 미확인",
            ],
            "M2": [
                "타석 결과 중복 제거",
                "참가자 변경 타석의 공식 귀속 제한",
            ],
            "M3": [
                "관측 집계는 공식 선수 기록과 구분",
                "미관측 선수 경기 기록을 0으로 확정하지 않음",
                "PBP 자책점 미확정",
                "월요일 기준 관측 종료 및 정답 분할은 후속 작업",
            ],
            "M4": [
                "타석 종료 상태와 투구 직후 상태를 혼용하지 않음",
                "복원 불확실 경계는 상태 후보에서 제외",
            ],
            "M5": [
                "무투구 행의 실제 투구 집계 제외",
                "구종 누락 및 순서 불일치 별도 표시",
                "좌표 단위와 위치 모델의 적용 범위는 별도 검토",
            ],
            "season": [
                "공식 순위·타이브레이커·시리즈 규칙 미검증",
                "일정 참조만으로 점수·승자 생성 금지",
            ],
        },
    }


def _input_fingerprint(
    settings: Settings,
    selected: dict[str, SourceVersion],
) -> tuple[str, Row]:
    """채택 판본·변환 코드·설정의 재현 식별자를 생성한다."""
    project = settings.project_root

    paths = (
        "src/datasets/published_sources.py",
        "src/datasets/standardize.py",
        "src/validation/standard_data.py",
        "kbo.toml",
    )

    inputs = {
        "transform_version": TRANSFORM_VERSION,
        "code_and_settings_sha256": {
            path: sha256_file(project / path)
            for path in paths
        },
        "selected_sources": {
            key: {
                "sha256": source.sha256,
                "revision": source.revision,
            }
            for key, source in sorted(selected.items())
        },
        "approval": USE_APPROVAL,
    }

    payload = json.dumps(
        inputs,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    digest = hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()

    return digest, inputs


def standardize(
    settings: Settings,
    *,
    as_of_run: str | None = None,
    compare_run: str | None = None,
) -> Row:
    """모든 불변 원본을 검사하고 선택된 판본의 표준 자료를 생성한다."""
    intake = intake_sources(
        settings.raw_dir / "published",
        as_of_run=as_of_run,
    )

    fingerprint, inputs = _input_fingerprint(
        settings,
        intake.selected,
    )

    previous = None

    if compare_run is not None:
        if not RUN_PATTERN.fullmatch(compare_run):
            raise StandardDataError("비교 실행 ID 형식 오류")

        previous = as_row(read_json(
            settings.processed_dir
            / "standard"
            / compare_run
            / "quality.json"
        ))

        if previous.get("status") != "completed":
            raise StandardDataError("비교 대상 실행이 완료되지 않았습니다.")

    run_id = uuid4().hex

    interim = settings.interim_dir / "standard" / run_id
    processed = settings.processed_dir / "standard" / run_id

    interim.mkdir(parents=True, exist_ok=False)
    processed.mkdir(parents=True, exist_ok=False)

    version_records: list[Row] = []
    interim_paths: dict[str, Path] = {}

    store = None

    try:
        for source in intake.versions:
            LOGGER.info(
                "원본 대응 자료 생성: %s %s",
                source.target_id,
                source.sha256[:12],
            )

            evidence = materialize_source(source, interim)
            selected = intake.selected[source.target_id]

            evidence["selected"] = (
                source.sha256 == selected.sha256
            )

            version_records.append(evidence)

            interim_paths[source.version_key] = (
                interim / str(evidence["interim_file"])
            )

        write_json(interim / "manifest.json", {
            "run_id": run_id,
            "as_of_run": as_of_run,
            "input_fingerprint": fingerprint,
            "inputs": inputs,
            "object_count": intake.object_count,
            "raw_bytes": intake.raw_bytes,
            "logical_target_count": len(intake.selected),
            "versions": version_records,
            "collection_history": intake.history,
        })

        store = StandardStore(processed / "standard.sqlite3")

        sources = [
            source
            for _, source in sorted(intake.selected.items())
        ]

        for source in sources:
            if source.target_id.startswith("dent-results-"):
                convert_regular_games(
                    store,
                    source,
                    read_interim_rows(
                        interim_paths[source.version_key]
                    ),
                )

        for source in sources:
            if source.target_id.startswith("hf-pbp-"):
                LOGGER.info("PBP 표준화: %s", source.target_id)

                convert_pbp(
                    store,
                    source,
                    read_interim_rows(
                        interim_paths[source.version_key]
                    ),
                )

        for source in sources:
            key = source.target_id
            rows = read_interim_rows(
                interim_paths[source.version_key]
            )

            if key.startswith(("dent-results-", "hf-pbp-")):
                continue

            if key.startswith(("dent-postseason-", "lopes-")):
                convert_postseason(store, source, rows)

            elif key.startswith("dialektike-"):
                convert_schedule_reference(store, source, rows)

            elif key == "window-starters-2023":
                convert_window_starters(store, source, rows)

            elif key == "window-daily-results-2026":
                convert_window_starters(
                    store,
                    source,
                    rows,
                    daily=True,
                )

            elif key == "window-pitching-2023":
                convert_window_pitching(store, source, rows)

            else:
                raise StandardDataError(
                    f"표준화 처리기가 없는 대상: {key}"
                )

        inspect_player_identity(store)

        coverage = coverage_report(store)
        summary = store.summary()

        quality = {
            "status": "completed",
            "run_id": run_id,
            "as_of_run": as_of_run,
            "input_fingerprint": fingerprint,
            "transform_version": TRANSFORM_VERSION,
            "source_objects": intake.object_count,
            "source_bytes": intake.raw_bytes,
            "logical_targets": len(intake.selected),
            "source_versions": len(intake.versions),
            "interim_manifest_sha256": sha256_file(
                interim / "manifest.json"
            ),
            "coverage": coverage,
            **summary,
        }

        if previous is not None:
            comparison = compare_summaries(previous, quality)
            quality["comparison"] = comparison

            if comparison["reproducible"] is False:
                quality["status"] = "reproduction_failed"

        store.connection.commit()
        store.close()
        store = None

        write_json(processed / "quality.json", quality)

        if quality["status"] != "completed":
            raise StandardDataError(
                "같은 입력의 표준 자료 재생성 결과가 다릅니다."
            )

        return {
            "run_id": run_id,
            "status": quality["status"],
            "interim_dir": str(interim),
            "processed_dir": str(processed),
            "quality_path": str(processed / "quality.json"),
            "input_fingerprint": fingerprint,
            "source_objects": intake.object_count,
            "logical_targets": len(intake.selected),
            "comparison": quality.get("comparison"),
        }

    except BaseException:
        if store is not None:
            store.connection.rollback()
            store.close()

        LOGGER.error(
            "표준화 실행 미완료: %s",
            run_id,
        )

        # 실패한 실행의 폴더도 조사할 수 있도록 보존한다.
        # 불변 원본 및 이전 성공 실행은 삭제하거나 덮어쓰지 않는다.
        raise