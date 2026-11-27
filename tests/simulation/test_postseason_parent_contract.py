"""정규시즌 공동 순위 인수와 포스트시즌 연결 회귀 테스트."""

from collections import defaultdict
from contextlib import closing
from datetime import date
from pathlib import Path
from random import Random
import sqlite3
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from src.simulation.postseason.data import (
    audit_archive_bracket,
    load_inputs,
    seed_input,
    verify_completion,
)
from src.simulation.postseason.engine import (
    ArchiveState,
    regular_first_event,
    simulate,
    simulate_round,
    summarize,
    team_event,
)
from src.simulation.postseason.rules import (
    Game,
    SeriesState,
    get_rules as get_postseason_rules,
)
from src.simulation.regular_season.engine import (
    Iteration,
    make_iteration,
)
from src.simulation.regular_season.rules import (
    SeasonState,
    Standing,
    get_rules as get_regular_rules,
    range_standing,
    resolve_tied_group,
)


SEASON = 2025

REGULAR_RULES = get_regular_rules(SEASON)
TEAMS = REGULAR_RULES.teams
SLOTS = REGULAR_RULES.playoff_slots

WC, SEMI, PO, KS = get_postseason_rules(SEASON)

SOURCE = "a" * 64


def fixed_standings() -> dict[int, Standing]:
    """구단별 확정 순위를 갖는 합성 상태를 구성한다."""
    return {
        team: range_standing(rank, rank, SLOTS)
        for rank, team in enumerate(
            TEAMS,
            start=1,
        )
    }


def make_parent_row(
    iteration_id: int = 1,
    tied_start: int | None = None,
    lower_unresolved: bool = False,
) -> Iteration:
    """기존 정규시즌 규칙에서 생성한 반복을 반환한다."""
    state = SeasonState(REGULAR_RULES)
    standings = fixed_standings()

    if tied_start is not None:
        group = list(
            TEAMS[tied_start - 1:tied_start + 1]
        )

        standings.update(
            resolve_tied_group(
                state,
                group,
                tied_start,
            )
        )

    if lower_unresolved:
        for team in TEAMS[5:7]:
            standings[team] = range_standing(
                6,
                7,
                SLOTS,
                "synthetic_lower_rank_unresolved",
            )

    return make_iteration(
        iteration_id,
        state,
        standings,
    )


def observed_game(
    stage: str,
    number: int,
    day: str,
    home: int,
    away: int,
) -> Game:
    """공식 홈 배정 및 날짜 검증에 사용할 합성 완료 경기."""
    return Game(
        number=number,
        home=home,
        away=away,
        outcome="home",
        game_id=f"test:{stage}:{number}",
        day=day,
        home_score=1,
        away_score=0,
    )


class PostseasonParentContractTests(unittest.TestCase):
    """부모 반복의 확정 순위와 시드 의미를 검증한다."""

    def test_lower_joint_rank_is_valid(self) -> None:
        parent = make_parent_row(
            tied_start=6,
        )

        first = str(TEAMS[5])
        second = str(TEAMS[6])

        self.assertEqual(
            parent["teams"][first]["rank"],
            6,
        )

        self.assertEqual(
            parent["teams"][second]["rank"],
            6,
        )

        self.assertTrue(parent["rank_resolved"])

        draw = seed_input(
            parent,
            TEAMS,
        )

        self.assertEqual(
            draw.seeds,
            TEAMS[:5],
        )

        self.assertEqual(
            draw.ranks[TEAMS[5]],
            6,
        )

        self.assertEqual(
            draw.ranks[TEAMS[6]],
            6,
        )

    def test_lower_rank_unknown_does_not_block_seeds(self) -> None:
        parent = make_parent_row(
            lower_unresolved=True,
        )

        self.assertFalse(
            parent["rank_resolved"],
        )

        self.assertTrue(
            parent["qualification_resolved"],
        )

        draw = seed_input(
            parent,
            TEAMS,
        )

        self.assertEqual(
            draw.seeds,
            TEAMS[:5],
        )

        self.assertIsNone(
            draw.ranks[TEAMS[5]],
        )

        self.assertFalse(
            draw.qualification[TEAMS[5]],
        )

    def test_duplicate_playoff_seed_is_rejected(self) -> None:
        parent = make_parent_row()

        parent["teams"][str(TEAMS[2])]["rank"] = 2

        with self.assertRaisesRegex(
            ValueError,
            "포스트시즌 확정 시드 중복",
        ):
            seed_input(
                parent,
                TEAMS,
            )

    def test_incorrect_rank_resolved_flag_is_rejected(self) -> None:
        parent = make_parent_row(
            tied_start=6,
        )

        parent["rank_resolved"] = False

        with self.assertRaisesRegex(
            ValueError,
            "순위 확정 표시 불일치",
        ):
            seed_input(
                parent,
                TEAMS,
            )

    def test_missing_confirmed_seed_is_rejected(self) -> None:
        parent = make_parent_row(
            tied_start=6,
        )

        parent["postseason_seeds"] = None

        with self.assertRaisesRegex(
            ValueError,
            "시드와 확정 순위 불일치",
        ):
            seed_input(
                parent,
                TEAMS,
            )

    def test_qualification_boundary_remains_unresolved(self) -> None:
        parent = make_parent_row(
            tied_start=5,
        )

        draw = seed_input(
            parent,
            TEAMS,
        )

        self.assertIsNone(draw.seeds)
        self.assertIsNone(draw.qualifiers)

        self.assertEqual(
            draw.regular_first,
            TEAMS[0],
        )

        self.assertIsNone(
            draw.qualification[TEAMS[4]],
        )

        self.assertIsNone(
            draw.qualification[TEAMS[5]],
        )

        self.assertIn(
            "separate_tiebreak_game_required",
            draw.reasons,
        )

    def test_first_place_tie_preserves_known_negative(self) -> None:
        parent = make_parent_row(
            tied_start=1,
        )

        draw = seed_input(
            parent,
            TEAMS,
        )

        self.assertIsNone(draw.regular_first)
        self.assertIsNone(draw.seeds)

        self.assertIsNone(
            regular_first_event(
                draw,
                TEAMS[0],
            )
        )

        self.assertFalse(
            regular_first_event(
                draw,
                TEAMS[2],
            )
        )

        self.assertFalse(
            regular_first_event(
                draw,
                TEAMS[9],
            )
        )

    def test_unresolved_seed_preserves_direct_entry(self) -> None:
        parent = make_parent_row(
            tied_start=5,
        )

        sampler = Mock(return_value="home")

        row = simulate(
            SEASON,
            [seed_input(parent, TEAMS)],
            sampler,
            42,
        )[0]

        sampler.assert_not_called()

        self.assertIsNone(row.champion)

        self.assertTrue(
            team_event(
                row,
                TEAMS[0],
                KS,
                "reach",
            )
        )

        self.assertTrue(
            team_event(
                row,
                TEAMS[3],
                WC,
                "reach",
            )
        )

        self.assertFalse(
            team_event(
                row,
                TEAMS[1],
                WC,
                "reach",
            )
        )

        self.assertIsNone(
            team_event(
                row,
                TEAMS[4],
                WC,
                "reach",
            )
        )

    def test_first_place_tie_preserves_wildcard_entry(self) -> None:
        parent = make_parent_row(
            tied_start=1,
        )

        sampler = Mock(return_value="home")

        row = simulate(
            SEASON,
            [seed_input(parent, TEAMS)],
            sampler,
            42,
        )[0]

        sampler.assert_not_called()

        self.assertTrue(
            team_event(
                row,
                TEAMS[3],
                WC,
                "reach",
            )
        )

        self.assertTrue(
            team_event(
                row,
                TEAMS[4],
                WC,
                "reach",
            )
        )

        self.assertIsNone(
            team_event(
                row,
                TEAMS[3],
                WC,
                "win",
            )
        )

    def test_mixed_rows_keep_iteration_identity(self) -> None:
        parents = [
            make_parent_row(
                iteration_id=1,
                tied_start=6,
            ),
            make_parent_row(
                iteration_id=2,
                tied_start=5,
            ),
            make_parent_row(
                iteration_id=3,
                lower_unresolved=True,
            ),
        ]

        draws = [
            seed_input(parent, TEAMS)
            for parent in parents
        ]

        sampler = Mock(return_value="home")

        rows = simulate(
            SEASON,
            draws,
            sampler,
            42,
        )

        self.assertEqual(
            [
                row.draw.iteration_id
                for row in rows
            ],
            [1, 2, 3],
        )

        self.assertEqual(
            rows[0].champion,
            TEAMS[0],
        )

        self.assertIsNone(
            rows[1].champion,
        )

        self.assertEqual(
            rows[2].champion,
            TEAMS[0],
        )

        result = summarize(
            SEASON,
            TEAMS,
            rows,
        )

        self.assertEqual(result["seed_resolved"], 2)
        self.assertEqual(result["seed_unresolved"], 1)

        first_team = result["teams"][str(TEAMS[0])]

        self.assertEqual(
            first_team["korean_series_entry"]["positive"],
            3,
        )

        self.assertEqual(
            first_team["korean_series_champion"]["valid"],
            2,
        )

        self.assertEqual(
            first_team["korean_series_champion"]["unresolved"],
            1,
        )

    def test_first_event_denominator_matches_parent(self) -> None:
        parents = [
            make_parent_row(
                iteration_id=1,
                tied_start=1,
            ),
            make_parent_row(
                iteration_id=2,
                tied_start=6,
            ),
        ]

        rows = simulate(
            SEASON,
            [
                seed_input(parent, TEAMS)
                for parent in parents
            ],
            Mock(return_value="home"),
            42,
        )

        summary = summarize(
            SEASON,
            TEAMS,
            rows,
        )

        for team in TEAMS:
            key = str(team)

            expected = [
                parent["teams"][key]["first"]
                for parent in parents
            ]

            result = summary["teams"][key]["regular_first"]

            self.assertEqual(
                result["valid"],
                sum(
                    value is not None
                    for value in expected
                ),
            )

            self.assertEqual(
                result["positive"],
                sum(
                    value is True
                    for value in expected
                ),
            )

    def test_iteration_output_preserves_individual_status(self) -> None:
        parent = make_parent_row(
            tied_start=5,
        )

        row = simulate(
            SEASON,
            [seed_input(parent, TEAMS)],
            Mock(return_value="home"),
            42,
        )[0].to_dict()

        for team in TEAMS:
            self.assertEqual(
                row["regular_ranks"][team],
                parent["teams"][str(team)]["rank"],
            )

            self.assertEqual(
                row["regular_first_status"][team],
                parent["teams"][str(team)]["first"],
            )


class PostseasonArchiveBoundaryTests(unittest.TestCase):
    """완료 라운드의 시간 순서와 상충 근거를 검사한다."""

    def test_cross_round_date_conflict_is_blocked(self) -> None:
        seeds = TEAMS[:5]

        archive = ArchiveState(
            games={
                "wildCard": (
                    observed_game(
                        "wildCard",
                        1,
                        "2025-10-07",
                        seeds[3],
                        seeds[4],
                    ),
                ),
                "semiPlayoff": (
                    observed_game(
                        "semiPlayoff",
                        1,
                        "2025-10-06",
                        seeds[2],
                        seeds[3],
                    ),
                ),
            },
        )

        sources = defaultdict(
            set,
            {
                "wildCard": {SOURCE},
                "semiPlayoff": {SOURCE},
            },
        )

        with (
            closing(sqlite3.connect(":memory:")) as standard,
            closing(sqlite3.connect(":memory:")) as model,
            patch(
                "src.simulation.postseason.data.verify_completion",
                return_value={
                    "winner": seeds[3],
                },
            ) as mocked,
        ):
            audit_archive_bracket(
                standard,
                model,
                SEASON,
                seeds,
                archive,
                sources,
            )

        self.assertEqual(mocked.call_count, 1)

        self.assertIn(
            "wildCard",
            archive.completion_evidence,
        )

        self.assertIn(
            "날짜 충돌",
            archive.blocked["semiPlayoff"],
        )

    def test_original_archive_block_reason_is_preserved(self) -> None:
        reason = (
            "archive_replay_conflict: "
            "이전 라운드 종료일과 후속 경기 날짜 충돌"
        )

        archive = ArchiveState(
            games={
                "semiPlayoff": (
                    observed_game(
                        "semiPlayoff",
                        1,
                        "2025-10-06",
                        TEAMS[3],
                        TEAMS[2],
                    ),
                ),
            },
            blocked={
                "semiPlayoff": reason,
            },
        )

        sampler = Mock(return_value="home")

        result = simulate_round(
            SEMI,
            TEAMS[2],
            TEAMS[3],
            sampler,
            Random(42),
            archive,
        )

        self.assertEqual(
            result.reason,
            reason,
        )

        self.assertIsNone(result.winner)
        sampler.assert_not_called()

    def test_conflicting_champion_metadata_is_rejected(self) -> None:
        seeds = TEAMS[:5]

        state = SeriesState(
            WC,
            seeds[3],
            seeds[4],
        )

        state.apply(
            observed_game(
                "wildCard",
                1,
                "2025-10-06",
                seeds[3],
                seeds[4],
            )
        )

        import json

        metadata = {
            "isComplete": True,
            "_score_rows": 1,
            "winnerId": seeds[3],
            "championId": seeds[4],
            "seeds": {
                str(rank): {"id": team}
                for rank, team in enumerate(
                    seeds,
                    start=1,
                )
            },
            "team4": {
                "id": seeds[3],
                "wins": 1,
                "isWinner": True,
            },
            "team5": {
                "id": seeds[4],
                "wins": 0,
                "isWinner": False,
            },
        }

        with (
            closing(sqlite3.connect(":memory:")) as standard,
            closing(sqlite3.connect(":memory:")) as model,
        ):
            columns = (
                "source_sha TEXT, season INTEGER, stage TEXT, "
                "series_complete INTEGER, metadata_json TEXT"
            )

            row = (
                SOURCE,
                SEASON,
                "wildCard",
                1,
                json.dumps(metadata),
            )

            for db, table in (
                (standard, "postseason_series"),
                (model, "season_series_evidence"),
            ):
                db.execute(
                    f"CREATE TABLE {table} ({columns})"
                )

                db.execute(
                    f"INSERT INTO {table} VALUES (?,?,?,?,?)",
                    row,
                )

            with self.assertRaisesRegex(
                ValueError,
                "시리즈 재생 승자와 종료 증거 불일치",
            ):
                verify_completion(
                    standard,
                    model,
                    SEASON,
                    state,
                    seeds,
                    {SOURCE},
                )

    def check_parent_archive_conflict(
        self,
        completed_game: bool,
    ) -> None:
        """부모 전망에서 기존 포스트시즌 기록을 무시하지 않게 한다."""
        standard = sqlite3.connect(":memory:")
        model = sqlite3.connect(":memory:")

        self.addCleanup(standard.close)
        self.addCleanup(model.close)

        standard.execute(
            """
            CREATE TABLE eligible_games (
                season INTEGER,
                competition TEXT,
                date TEXT
            )
            """
        )

        standard.execute(
            """
            CREATE TABLE postseason_schedule (
                source_sha TEXT,
                source_row INTEGER,
                stage TEXT,
                series_game_number INTEGER,
                date TEXT,
                away_name TEXT,
                home_name TEXT,
                final_observed INTEGER,
                linked_game_id TEXT
            )
            """
        )

        if completed_game:
            standard.execute(
                """
                INSERT INTO eligible_games
                VALUES (?, ?, ?)
                """,
                (
                    SEASON,
                    "postseason",
                    "2025-10-06",
                ),
            )

        else:
            standard.execute(
                """
                INSERT INTO postseason_schedule
                VALUES (?,?,?,?,?,?,?,?,?)
                """,
                (
                    SOURCE,
                    0,
                    "wildCard",
                    1,
                    "2025-10-06",
                    "NC Dinos",
                    "Samsung Lions",
                    0,
                    None,
                ),
            )

        draw = seed_input(
            make_parent_row(),
            TEAMS,
        )

        settings = SimpleNamespace()

        with (
            patch(
                "src.simulation.postseason.data.verify_assets",
                return_value=(
                    Path("standard.sqlite3"),
                    Path("model.sqlite3"),
                    {},
                    {},
                ),
            ),
            patch(
                "src.simulation.postseason.data.open_readonly",
                side_effect=[
                    standard,
                    model,
                ],
            ),
            patch(
                "src.simulation.postseason.data.read_completed",
                return_value=(),
            ),
            patch(
                "src.simulation.postseason.data.read_parent",
                return_value=(
                    (draw,),
                    {},
                ),
            ),
        ):
            with self.assertRaisesRegex(
                ValueError,
                "기준일 이전 포스트시즌",
            ):
                load_inputs(
                    settings=settings,
                    run_id="a" * 32,
                    asset=Path("model.json"),
                    season=SEASON,
                    as_of="2025-10-07",
                    parent_run="b" * 32,
                    iterations=None,
                    archive_assumption=False,
                )

    def test_parent_rejects_prior_completed_postseason(self) -> None:
        self.check_parent_archive_conflict(
            completed_game=True,
        )

    def test_parent_rejects_prior_postseason_schedule(self) -> None:
        self.check_parent_archive_conflict(
            completed_game=False,
        )


if __name__ == "__main__":
    unittest.main()