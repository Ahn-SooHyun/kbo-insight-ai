from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_player_game_pitching import (  # noqa: E402
    OUTPUT_COLUMNS,
    PlayerGamePitchingBuildError,
    build_player_game_pitching,
    build_player_game_pitching_file,
    ensure_output_path_safe,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_row(
    *,
    game_pk: str = "20230401AABB02023",
    game_date: str = "2023-04-01",
    home_team: str = "BB",
    away_team: str = "AA",
    inning_topbot: str = "top",
    at_bat_number: int = 1,
    pitch_number: int = 1,
    pitcher: str = "001",
    pitcher_name: str | None = "투수1",
    pitch_type: str | None = "S",
    release_speed_kmh: float | None = 145.0,
) -> dict[str, object]:
    return {
        "game_pk": game_pk,
        "game_date": game_date,
        "home_team": home_team,
        "away_team": away_team,
        "inning_topbot": inning_topbot,
        "at_bat_number": at_bat_number,
        "pitch_number": pitch_number,
        "pitcher": pitcher,
        "pitcher_name": pitcher_name,
        "type": pitch_type,
        "release_speed_kmh": release_speed_kmh,
    }


def pa_row(
    *,
    game_pk: str = "20230401AABB02023",
    game_date: str = "2023-04-01",
    season: int = 2023,
    at_bat_number: int = 1,
    batting_team: str = "AA",
    fielding_team: str = "BB",
    is_home_batting: bool = False,
    pitcher: str = "001",
    pitcher_name: str | None = "투수1",
    event: str | None = "field_out",
    outs_before: int = 0,
    post_outs: int = 1,
    pitch_rows: int = 1,
    pitch_count: int = 1,
) -> dict[str, object]:
    return {
        "game_pk": game_pk,
        "game_date": game_date,
        "season": season,
        "at_bat_number": at_bat_number,
        "batting_team": batting_team,
        "fielding_team": fielding_team,
        "is_home_batting": is_home_batting,
        "pitcher": pitcher,
        "pitcher_name": pitcher_name,
        "event": event,
        "outs_before": outs_before,
        "post_outs": post_outs,
        "pitch_rows": pitch_rows,
        "pitch_count": pitch_count,
    }


def frames(
    raw_rows: list[dict[str, object]],
    pa_rows: list[dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.DataFrame(raw_rows), pd.DataFrame(pa_rows)


class PlayerGamePitchingAggregationTest(unittest.TestCase):
    def test_01_builds_one_pitcher_game_row(self) -> None:
        raw, pa = frames([raw_row()], [pa_row()])
        result = build_player_game_pitching(raw, pa, season=2023)
        self.assertEqual(len(result), 1)
        self.assertEqual(tuple(result.columns), OUTPUT_COLUMNS)
        self.assertEqual(result.loc[0, "pitcher"], "001")

    def test_02_aggregates_multiple_pitches_and_pas(self) -> None:
        raw, pa = frames(
            [
                raw_row(at_bat_number=1, pitch_number=1, pitch_type="B"),
                raw_row(at_bat_number=1, pitch_number=2, pitch_type="X"),
                raw_row(at_bat_number=2, pitch_number=1, pitch_type="S"),
            ],
            [
                pa_row(at_bat_number=1, event="single", post_outs=0, pitch_rows=2, pitch_count=2),
                pa_row(at_bat_number=2, event="strikeout", outs_before=0, post_outs=1),
            ],
        )
        result = build_player_game_pitching(raw, pa, season=2023)
        row = result.iloc[0]
        self.assertEqual(int(row["pitch_rows"]), 3)
        self.assertEqual(int(row["pitches"]), 3)
        self.assertEqual(int(row["batters_faced_completed"]), 2)

    def test_03_separates_multiple_pitchers_in_game(self) -> None:
        raw, pa = frames(
            [
                raw_row(at_bat_number=1, pitcher="001", pitcher_name="투수1"),
                raw_row(at_bat_number=2, pitcher="002", pitcher_name="투수2"),
            ],
            [
                pa_row(at_bat_number=1, pitcher="001", pitcher_name="투수1"),
                pa_row(at_bat_number=2, pitcher="002", pitcher_name="투수2"),
            ],
        )
        result = build_player_game_pitching(raw, pa, season=2023)
        self.assertEqual(set(result["pitcher"].tolist()), {"001", "002"})

    def test_04_team_opponent_and_is_home(self) -> None:
        raw, pa = frames([raw_row(inning_topbot="top")], [pa_row()])
        row = build_player_game_pitching(raw, pa, season=2023).iloc[0]
        self.assertEqual(row["team"], "BB")
        self.assertEqual(row["opponent"], "AA")
        self.assertTrue(bool(row["is_home"]))

    def test_05_same_pitcher_can_change_team_between_games(self) -> None:
        raw, pa = frames(
            [
                raw_row(game_pk="20230401AABB02023", at_bat_number=1, pitcher="001"),
                raw_row(
                    game_pk="20230402CCDD02023",
                    game_date="2023-04-02",
                    home_team="DD",
                    away_team="CC",
                    at_bat_number=1,
                    pitcher="001",
                ),
            ],
            [
                pa_row(game_pk="20230401AABB02023", at_bat_number=1, pitcher="001"),
                pa_row(
                    game_pk="20230402CCDD02023",
                    game_date="2023-04-02",
                    batting_team="CC",
                    fielding_team="DD",
                    at_bat_number=1,
                    pitcher="001",
                ),
            ],
        )
        result = build_player_game_pitching(raw, pa, season=2023)
        self.assertEqual(len(result), 2)
        self.assertEqual(set(result["team"]), {"BB", "DD"})

    def test_06_counts_raw_pitch_rows(self) -> None:
        raw, pa = frames(
            [
                raw_row(pitch_number=1),
                raw_row(pitch_number=2),
                raw_row(pitch_number=3),
            ],
            [pa_row(pitch_rows=3, pitch_count=3)],
        )
        result = build_player_game_pitching(raw, pa, season=2023)
        self.assertEqual(int(result.loc[0, "pitch_rows"]), 3)

    def test_07_pitches_count_only_pitch_number_greater_than_zero(self) -> None:
        raw, pa = frames(
            [raw_row(pitch_number=0, pitch_type=None, release_speed_kmh=None)],
            [pa_row(event="walk", post_outs=0, pitch_rows=1, pitch_count=0)],
        )
        result = build_player_game_pitching(raw, pa, season=2023)
        self.assertEqual(int(result.loc[0, "pitches"]), 0)

    def test_08_pitchless_row_counts_as_row_not_pitch(self) -> None:
        raw, pa = frames(
            [raw_row(pitch_number=0, pitch_type=None, release_speed_kmh=None)],
            [pa_row(event="walk", post_outs=0, pitch_rows=1, pitch_count=0)],
        )
        row = build_player_game_pitching(raw, pa, season=2023).iloc[0]
        self.assertEqual(int(row["pitch_rows"]), 1)
        self.assertEqual(int(row["pitches"]), 0)

    def test_09_counts_ball_strike_and_in_play_pitches(self) -> None:
        raw, pa = frames(
            [
                raw_row(pitch_number=1, pitch_type="B"),
                raw_row(pitch_number=2, pitch_type="S"),
                raw_row(pitch_number=3, pitch_type="X"),
            ],
            [pa_row(pitch_rows=3, pitch_count=3)],
        )
        row = build_player_game_pitching(raw, pa, season=2023).iloc[0]
        self.assertEqual(int(row["ball_pitch_count"]), 1)
        self.assertEqual(int(row["strike_pitch_count"]), 1)
        self.assertEqual(int(row["in_play_pitch_count"]), 1)

    def test_10_rejects_null_or_unknown_actual_pitch_type(self) -> None:
        for bad_type in (None, "Q"):
            with self.subTest(bad_type=bad_type):
                raw, pa = frames(
                    [raw_row(pitch_type=bad_type)],
                    [pa_row()],
                )
                with self.assertRaises(PlayerGamePitchingBuildError):
                    build_player_game_pitching(raw, pa, season=2023)

    def test_11_calculates_average_release_speed(self) -> None:
        raw, pa = frames(
            [
                raw_row(pitch_number=1, release_speed_kmh=140.0),
                raw_row(pitch_number=2, release_speed_kmh=141.0),
                raw_row(pitch_number=3, release_speed_kmh=141.0),
            ],
            [pa_row(pitch_rows=3, pitch_count=3)],
        )
        row = build_player_game_pitching(raw, pa, season=2023).iloc[0]
        self.assertEqual(float(row["avg_release_speed_kmh"]), 140.667)

    def test_12_average_release_speed_excludes_null(self) -> None:
        raw, pa = frames(
            [
                raw_row(pitch_number=1, release_speed_kmh=140.0),
                raw_row(pitch_number=2, release_speed_kmh=None),
            ],
            [pa_row(pitch_rows=2, pitch_count=2)],
        )
        row = build_player_game_pitching(raw, pa, season=2023).iloc[0]
        self.assertEqual(float(row["avg_release_speed_kmh"]), 140.0)

    def test_13_average_release_speed_is_null_without_valid_speed(self) -> None:
        raw, pa = frames(
            [raw_row(release_speed_kmh=None)],
            [pa_row()],
        )
        row = build_player_game_pitching(raw, pa, season=2023).iloc[0]
        self.assertTrue(pd.isna(row["avg_release_speed_kmh"]))

    def test_14_completed_pa_counts_as_batter_faced(self) -> None:
        raw, pa = frames([raw_row()], [pa_row(event="field_out")])
        row = build_player_game_pitching(raw, pa, season=2023).iloc[0]
        self.assertEqual(int(row["batters_faced_completed"]), 1)

    def test_15_incomplete_pa_is_excluded_from_batter_faced(self) -> None:
        raw, pa = frames([raw_row()], [pa_row(event=None, post_outs=0)])
        row = build_player_game_pitching(raw, pa, season=2023).iloc[0]
        self.assertEqual(int(row["batters_faced_completed"]), 0)

    def test_16_hits_allowed_formula(self) -> None:
        events = ["single", "double", "triple", "home_run"]
        raw_rows = [raw_row(at_bat_number=i) for i in range(1, 5)]
        pa_rows = [
            pa_row(at_bat_number=i, event=event, post_outs=0)
            for i, event in enumerate(events, start=1)
        ]
        row = build_player_game_pitching(*frames(raw_rows, pa_rows), season=2023).iloc[0]
        self.assertEqual(int(row["hits_allowed"]), 4)
        self.assertEqual(int(row["single_allowed"]), 1)
        self.assertEqual(int(row["double_allowed"]), 1)
        self.assertEqual(int(row["triple_allowed"]), 1)
        self.assertEqual(int(row["hr_allowed"]), 1)

    def test_17_counts_bb_hbp_so_sf_and_sh(self) -> None:
        events = ["walk", "hit_by_pitch", "strikeout", "sac_fly", "sac_bunt"]
        raw_rows = [raw_row(at_bat_number=i) for i in range(1, 6)]
        pa_rows = [
            pa_row(
                at_bat_number=i,
                event=event,
                post_outs=0 if event in {"walk", "hit_by_pitch"} else 1,
            )
            for i, event in enumerate(events, start=1)
        ]
        row = build_player_game_pitching(*frames(raw_rows, pa_rows), season=2023).iloc[0]
        for column in ("bb_allowed", "hbp_allowed", "so", "sf", "sh"):
            self.assertEqual(int(row[column]), 1)

    def test_18_pitcher_change_splits_raw_pitch_count(self) -> None:
        raw, pa = frames(
            [
                raw_row(pitch_number=1, pitcher="A", pitcher_name="A"),
                raw_row(pitch_number=2, pitcher="B", pitcher_name="B"),
            ],
            [
                pa_row(
                    pitcher="B",
                    pitcher_name="B",
                    event="single",
                    post_outs=0,
                    pitch_rows=2,
                    pitch_count=2,
                )
            ],
        )
        result = build_player_game_pitching(raw, pa, season=2023).set_index("pitcher")
        self.assertEqual(int(result.loc["A", "pitches"]), 1)
        self.assertEqual(int(result.loc["B", "pitches"]), 1)

    def test_19_pa_result_is_credited_to_canonical_pa_pitcher(self) -> None:
        raw, pa = frames(
            [
                raw_row(pitch_number=1, pitcher="A", pitcher_name="A"),
                raw_row(pitch_number=2, pitcher="B", pitcher_name="B", pitch_type="X"),
            ],
            [
                pa_row(
                    pitcher="B",
                    pitcher_name="B",
                    event="single",
                    post_outs=0,
                    pitch_rows=2,
                    pitch_count=2,
                )
            ],
        )
        result = build_player_game_pitching(raw, pa, season=2023).set_index("pitcher")
        self.assertEqual(int(result.loc["A", "single_allowed"]), 0)
        self.assertEqual(int(result.loc["B", "single_allowed"]), 1)

    def test_20_records_one_out(self) -> None:
        row = build_player_game_pitching(
            *frames([raw_row()], [pa_row(outs_before=0, post_outs=1)]),
            season=2023,
        ).iloc[0]
        self.assertEqual(int(row["outs_recorded"]), 1)

    def test_21_records_double_play_two_outs(self) -> None:
        row = build_player_game_pitching(
            *frames([raw_row(pitch_type="X")], [pa_row(event="double_play", outs_before=0, post_outs=2)]),
            season=2023,
        ).iloc[0]
        self.assertEqual(int(row["outs_recorded"]), 2)

    def test_22_records_triple_play_three_outs(self) -> None:
        row = build_player_game_pitching(
            *frames([raw_row(pitch_type="X")], [pa_row(event="triple_play", outs_before=0, post_outs=3)]),
            season=2023,
        ).iloc[0]
        self.assertEqual(int(row["outs_recorded"]), 3)

    def test_23_records_two_to_three_as_one_out(self) -> None:
        row = build_player_game_pitching(
            *frames([raw_row()], [pa_row(outs_before=2, post_outs=3)]),
            season=2023,
        ).iloc[0]
        self.assertEqual(int(row["outs_recorded"]), 1)

    def test_24_half_inning_outs_reset_is_not_diffed_across_pas(self) -> None:
        raw, pa = frames(
            [raw_row(at_bat_number=1), raw_row(at_bat_number=2)],
            [
                pa_row(at_bat_number=1, outs_before=2, post_outs=3),
                pa_row(at_bat_number=2, outs_before=0, post_outs=1),
            ],
        )
        row = build_player_game_pitching(raw, pa, season=2023).iloc[0]
        self.assertEqual(int(row["outs_recorded"]), 2)

    def test_25_incomplete_pa_runner_out_is_counted_in_outs(self) -> None:
        row = build_player_game_pitching(
            *frames([raw_row()], [pa_row(event=None, outs_before=1, post_outs=2)]),
            season=2023,
        ).iloc[0]
        self.assertEqual(int(row["batters_faced_completed"]), 0)
        self.assertEqual(int(row["outs_recorded"]), 1)

    def test_26_rejects_negative_out_delta(self) -> None:
        raw, pa = frames([raw_row()], [pa_row(outs_before=2, post_outs=1)])
        with self.assertRaises(PlayerGamePitchingBuildError):
            build_player_game_pitching(raw, pa, season=2023)

    def test_27_rejects_out_delta_greater_than_three(self) -> None:
        raw, pa = frames([raw_row()], [pa_row(outs_before=0, post_outs=4)])
        with self.assertRaises(PlayerGamePitchingBuildError):
            build_player_game_pitching(raw, pa, season=2023)

    def test_28_allows_multi_pitcher_terminal_out(self) -> None:
        raw, pa = frames(
            [
                raw_row(pitch_number=1, pitcher="A", pitcher_name="A", pitch_type="B"),
                raw_row(pitch_number=2, pitcher="B", pitcher_name="B", pitch_type="X"),
            ],
            [
                pa_row(
                    pitcher="B",
                    pitcher_name="B",
                    event="field_out",
                    outs_before=0,
                    post_outs=1,
                    pitch_rows=2,
                    pitch_count=2,
                )
            ],
        )
        result = build_player_game_pitching(raw, pa, season=2023).set_index("pitcher")
        self.assertEqual(int(result.loc["A", "outs_recorded"]), 0)
        self.assertEqual(int(result.loc["B", "outs_recorded"]), 1)

    def test_29_output_grain_is_unique(self) -> None:
        raw, pa = frames(
            [raw_row(at_bat_number=1), raw_row(at_bat_number=2)],
            [pa_row(at_bat_number=1), pa_row(at_bat_number=2)],
        )
        result = build_player_game_pitching(raw, pa, season=2023)
        self.assertFalse(result.duplicated(["game_pk", "pitcher"]).any())

    def test_30_raw_row_reconciliation(self) -> None:
        raw, pa = frames(
            [raw_row(pitch_number=1), raw_row(pitch_number=2)],
            [pa_row(pitch_rows=2, pitch_count=2)],
        )
        result = build_player_game_pitching(raw, pa, season=2023)
        self.assertEqual(int(result["pitch_rows"].sum()), len(raw))

    def test_31_actual_pitch_reconciliation(self) -> None:
        raw, pa = frames(
            [raw_row(pitch_number=0, pitch_type=None, release_speed_kmh=None)],
            [pa_row(event="walk", post_outs=0, pitch_rows=1, pitch_count=0)],
        )
        result = build_player_game_pitching(raw, pa, season=2023)
        self.assertEqual(int(result["pitches"].sum()), 0)

    def test_32_completed_bf_reconciliation(self) -> None:
        raw, pa = frames(
            [raw_row(at_bat_number=1), raw_row(at_bat_number=2)],
            [pa_row(at_bat_number=1), pa_row(at_bat_number=2, event=None, post_outs=0)],
        )
        result = build_player_game_pitching(raw, pa, season=2023)
        self.assertEqual(int(result["batters_faced_completed"].sum()), 1)

    def test_33_allowed_event_reconciliation(self) -> None:
        raw, pa = frames([raw_row()], [pa_row(event="home_run", post_outs=0)])
        result = build_player_game_pitching(raw, pa, season=2023)
        self.assertEqual(int(result["hr_allowed"].sum()), 1)
        self.assertEqual(int(result["hits_allowed"].sum()), 1)

    def test_34_outs_reconciliation(self) -> None:
        raw, pa = frames(
            [raw_row(at_bat_number=1), raw_row(at_bat_number=2)],
            [
                pa_row(at_bat_number=1, outs_before=0, post_outs=1),
                pa_row(at_bat_number=2, event="double_play", outs_before=1, post_outs=3),
            ],
        )
        result = build_player_game_pitching(raw, pa, season=2023)
        self.assertEqual(int(result["outs_recorded"].sum()), 3)

    def test_35_input_shuffle_is_deterministic(self) -> None:
        raw, pa = frames(
            [
                raw_row(at_bat_number=1, pitch_number=1),
                raw_row(at_bat_number=1, pitch_number=2),
                raw_row(at_bat_number=2, pitch_number=1),
            ],
            [
                pa_row(at_bat_number=1, pitch_rows=2, pitch_count=2),
                pa_row(at_bat_number=2),
            ],
        )
        first = build_player_game_pitching(raw, pa, season=2023)
        second = build_player_game_pitching(
            raw.sample(frac=1, random_state=1).reset_index(drop=True),
            pa.sample(frac=1, random_state=2).reset_index(drop=True),
            season=2023,
        )
        pd.testing.assert_frame_equal(first, second, check_dtype=True)

    def test_37_pitcher_id_preserves_string_meaning(self) -> None:
        raw, pa = frames(
            [raw_row(pitcher="00123")],
            [pa_row(pitcher="00123")],
        )
        result = build_player_game_pitching(raw, pa, season=2023)
        self.assertEqual(result.loc[0, "pitcher"], "00123")

    def test_39_rerun_produces_same_dataframe(self) -> None:
        raw, pa = frames([raw_row()], [pa_row()])
        first = build_player_game_pitching(raw, pa, season=2023)
        second = build_player_game_pitching(raw, pa, season=2023)
        pd.testing.assert_frame_equal(first, second, check_dtype=True)

    def test_45_prohibited_run_responsibility_columns_are_absent(self) -> None:
        raw, pa = frames([raw_row()], [pa_row()])
        result = build_player_game_pitching(raw, pa, season=2023)
        for column in ("earned_runs", "era", "runs_allowed"):
            self.assertNotIn(column, result.columns)

    def test_46_allows_multi_pitcher_fielders_choice_out(self) -> None:
        raw, pa = frames(
            [
                raw_row(pitch_number=1, pitcher="A", pitcher_name="A", pitch_type="B"),
                raw_row(pitch_number=2, pitcher="B", pitcher_name="B", pitch_type="X"),
            ],
            [
                pa_row(
                    pitcher="B",
                    pitcher_name="B",
                    event="fielders_choice",
                    outs_before=0,
                    post_outs=1,
                    pitch_rows=2,
                    pitch_count=2,
                )
            ],
        )
        result = build_player_game_pitching(raw, pa, season=2023).set_index("pitcher")
        self.assertEqual(int(result.loc["B", "outs_recorded"]), 1)

    def test_47_allows_multi_pitcher_double_play_outs(self) -> None:
        raw, pa = frames(
            [
                raw_row(pitch_number=1, pitcher="A", pitcher_name="A", pitch_type="B"),
                raw_row(pitch_number=2, pitcher="B", pitcher_name="B", pitch_type="X"),
            ],
            [
                pa_row(
                    pitcher="B",
                    pitcher_name="B",
                    event="double_play",
                    outs_before=0,
                    post_outs=2,
                    pitch_rows=2,
                    pitch_count=2,
                )
            ],
        )
        result = build_player_game_pitching(raw, pa, season=2023).set_index("pitcher")
        self.assertEqual(int(result.loc["B", "outs_recorded"]), 2)

    def test_48_marks_unsupported_multi_pitcher_out_delta_unknown(self) -> None:
        raw, pa = frames(
            [
                raw_row(pitch_number=1, pitcher="A", pitcher_name="A", pitch_type="B"),
                raw_row(pitch_number=2, pitcher="B", pitcher_name="B", pitch_type="X"),
            ],
            [
                pa_row(
                    pitcher="B",
                    pitcher_name="B",
                    event="single",
                    outs_before=0,
                    post_outs=1,
                    pitch_rows=2,
                    pitch_count=2,
                )
            ],
        )
        result = build_player_game_pitching(raw, pa, season=2023).set_index("pitcher")
        self.assertTrue(pd.isna(result.loc["A", "outs_recorded"]))
        self.assertTrue(pd.isna(result.loc["B", "outs_recorded"]))
        self.assertEqual(str(result["outs_recorded"].dtype), "Int64")

    def test_49_marks_multi_pitcher_event_out_delta_mismatch_unknown(self) -> None:
        raw, pa = frames(
            [
                raw_row(pitch_number=1, pitcher="A", pitcher_name="A", pitch_type="B"),
                raw_row(pitch_number=2, pitcher="B", pitcher_name="B", pitch_type="X"),
            ],
            [
                pa_row(
                    pitcher="B",
                    pitcher_name="B",
                    event="field_out",
                    outs_before=0,
                    post_outs=2,
                    pitch_rows=2,
                    pitch_count=2,
                )
            ],
        )
        result = build_player_game_pitching(raw, pa, season=2023).set_index("pitcher")
        self.assertTrue(pd.isna(result.loc["A", "outs_recorded"]))
        self.assertTrue(pd.isna(result.loc["B", "outs_recorded"]))

    def test_50_allows_multi_pitcher_strikeout_out(self) -> None:
        raw, pa = frames(
            [
                raw_row(pitch_number=1, pitcher="A", pitcher_name="A", pitch_type="B"),
                raw_row(pitch_number=2, pitcher="B", pitcher_name="B", pitch_type="S"),
            ],
            [
                pa_row(
                    pitcher="B",
                    pitcher_name="B",
                    event="strikeout",
                    outs_before=1,
                    post_outs=2,
                    pitch_rows=2,
                    pitch_count=2,
                )
            ],
        )
        result = build_player_game_pitching(raw, pa, season=2023).set_index("pitcher")
        self.assertEqual(int(result.loc["B", "outs_recorded"]), 1)

    def test_51_allows_multi_pitcher_sac_bunt_out(self) -> None:
        raw, pa = frames(
            [
                raw_row(pitch_number=1, pitcher="A", pitcher_name="A", pitch_type="B"),
                raw_row(pitch_number=2, pitcher="B", pitcher_name="B", pitch_type="X"),
            ],
            [
                pa_row(
                    pitcher="B",
                    pitcher_name="B",
                    event="sac_bunt",
                    outs_before=0,
                    post_outs=1,
                    pitch_rows=2,
                    pitch_count=2,
                )
            ],
        )
        result = build_player_game_pitching(raw, pa, season=2023).set_index("pitcher")
        self.assertEqual(int(result.loc["B", "outs_recorded"]), 1)

    def test_52_rejects_multi_pitcher_last_raw_pitcher_mismatch(self) -> None:
        raw, pa = frames(
            [
                raw_row(pitch_number=1, pitcher="B", pitcher_name="B", pitch_type="B"),
                raw_row(pitch_number=2, pitcher="A", pitcher_name="A", pitch_type="X"),
            ],
            [
                pa_row(
                    pitcher="B",
                    pitcher_name="B",
                    event="field_out",
                    outs_before=0,
                    post_outs=1,
                    pitch_rows=2,
                    pitch_count=2,
                )
            ],
        )
        with self.assertRaises(PlayerGamePitchingBuildError):
            build_player_game_pitching(raw, pa, season=2023)

    def test_53_unknown_out_masks_entire_affected_player_game(self) -> None:
        raw, pa = frames(
            [
                raw_row(
                    at_bat_number=1,
                    pitch_number=1,
                    pitcher="A",
                    pitcher_name="A",
                    pitch_type="X",
                ),
                raw_row(
                    at_bat_number=2,
                    pitch_number=1,
                    pitcher="A",
                    pitcher_name="A",
                    pitch_type="B",
                ),
                raw_row(
                    at_bat_number=2,
                    pitch_number=2,
                    pitcher="B",
                    pitcher_name="B",
                    pitch_type="X",
                ),
            ],
            [
                pa_row(
                    at_bat_number=1,
                    pitcher="A",
                    pitcher_name="A",
                    event="field_out",
                    outs_before=0,
                    post_outs=1,
                ),
                pa_row(
                    at_bat_number=2,
                    pitcher="B",
                    pitcher_name="B",
                    event="field_out",
                    outs_before=1,
                    post_outs=3,
                    pitch_rows=2,
                    pitch_count=2,
                ),
            ],
        )
        result = build_player_game_pitching(raw, pa, season=2023).set_index("pitcher")
        self.assertTrue(pd.isna(result.loc["A", "outs_recorded"]))
        self.assertTrue(pd.isna(result.loc["B", "outs_recorded"]))

    def test_54_unknown_out_does_not_mask_unrelated_pitcher(self) -> None:
        raw, pa = frames(
            [
                raw_row(
                    at_bat_number=1,
                    pitch_number=1,
                    pitcher="A",
                    pitcher_name="A",
                    pitch_type="B",
                ),
                raw_row(
                    at_bat_number=1,
                    pitch_number=2,
                    pitcher="B",
                    pitcher_name="B",
                    pitch_type="X",
                ),
                raw_row(
                    at_bat_number=2,
                    pitch_number=1,
                    pitcher="C",
                    pitcher_name="C",
                    pitch_type="X",
                ),
            ],
            [
                pa_row(
                    at_bat_number=1,
                    pitcher="B",
                    pitcher_name="B",
                    event="field_out",
                    outs_before=0,
                    post_outs=2,
                    pitch_rows=2,
                    pitch_count=2,
                ),
                pa_row(
                    at_bat_number=2,
                    pitcher="C",
                    pitcher_name="C",
                    event="field_out",
                    outs_before=0,
                    post_outs=1,
                ),
            ],
        )
        result = build_player_game_pitching(raw, pa, season=2023).set_index("pitcher")
        self.assertTrue(pd.isna(result.loc["A", "outs_recorded"]))
        self.assertTrue(pd.isna(result.loc["B", "outs_recorded"]))
        self.assertEqual(int(result.loc["C", "outs_recorded"]), 1)


class PlayerGamePitchingParquetIOTest(unittest.TestCase):
    def _write_fixture(
        self,
        root: Path,
        *,
        seasons: tuple[int, ...] = (2023,),
    ) -> tuple[Path, Path, list[pd.DataFrame]]:
        raw_dir = root / "raw"
        raw_dir.mkdir(parents=True)
        pa_frames: list[pd.DataFrame] = []
        raw_frames: list[pd.DataFrame] = []

        for season in seasons:
            date = f"{season}-04-01"
            game_pk = f"{season}0401AABB02{season}"
            raw_frame = pd.DataFrame(
                [
                    raw_row(
                        game_pk=game_pk,
                        game_date=date,
                        at_bat_number=1,
                        pitcher=f"{season}01",
                    )
                ]
            )
            pa_frame = pd.DataFrame(
                [
                    pa_row(
                        game_pk=game_pk,
                        game_date=date,
                        season=season,
                        at_bat_number=1,
                        pitcher=f"{season}01",
                    )
                ]
            )
            raw_frame.to_parquet(raw_dir / f"{season}.parquet", index=False, engine="pyarrow")
            raw_frames.append(raw_frame)
            pa_frames.append(pa_frame)

        pa_path = root / "plate_appearances.parquet"
        pd.concat(pa_frames, ignore_index=True).to_parquet(pa_path, index=False, engine="pyarrow")
        return raw_dir, pa_path, raw_frames

    def test_36_multi_season_and_game_sorting(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir, pa_path, _ = self._write_fixture(root, seasons=(2024, 2023))
            output_path = root / "derived" / "player_game_pitching.parquet"
            result = build_player_game_pitching_file(
                raw_dir=raw_dir,
                pa_path=pa_path,
                output_path=output_path,
                seasons=(2024, 2023),
            )
            self.assertEqual(result["season"].tolist(), [2023, 2024])

    def test_38_parquet_round_trip_preserves_dtypes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir, pa_path, _ = self._write_fixture(root)
            output_path = root / "derived" / "player_game_pitching.parquet"
            expected = build_player_game_pitching_file(
                raw_dir=raw_dir,
                pa_path=pa_path,
                output_path=output_path,
                seasons=(2023,),
            )
            actual = pd.read_parquet(output_path, engine="pyarrow")
            self.assertEqual(str(actual["game_pk"].dtype), "string")
            self.assertEqual(str(actual["pitcher"].dtype), "string")
            self.assertEqual(str(actual["season"].dtype), "Int64")
            self.assertEqual(str(actual["is_home"].dtype), "boolean")
            self.assertEqual(str(actual["avg_release_speed_kmh"].dtype), "Float64")
            self.assertEqual(str(actual["game_date"].dtype), "datetime64[us]")
            pd.testing.assert_frame_equal(expected, actual, check_dtype=True)

    def test_40_file_build_preserves_raw_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir, pa_path, _ = self._write_fixture(root)
            raw_path = raw_dir / "2023.parquet"
            before = sha256(raw_path)
            build_player_game_pitching_file(
                raw_dir=raw_dir,
                pa_path=pa_path,
                output_path=root / "derived" / "out.parquet",
                seasons=(2023,),
            )
            self.assertEqual(before, sha256(raw_path))

    def test_41_file_build_preserves_pa_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir, pa_path, _ = self._write_fixture(root)
            before = sha256(pa_path)
            build_player_game_pitching_file(
                raw_dir=raw_dir,
                pa_path=pa_path,
                output_path=root / "derived" / "out.parquet",
                seasons=(2023,),
            )
            self.assertEqual(before, sha256(pa_path))

    def test_42_data_raw_output_is_rejected(self) -> None:
        with self.assertRaises(PlayerGamePitchingBuildError):
            ensure_output_path_safe(
                raw_dir=PROJECT_ROOT / "data" / "raw" / "hf_kbo_pbp",
                pa_path=PROJECT_ROOT / "data" / "interim" / "pa.parquet",
                output_path=PROJECT_ROOT / "data" / "raw" / "out.parquet",
                seasons=(2023,),
            )

    def test_43_pa_overwrite_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pa_path = root / "pa.parquet"
            with self.assertRaises(PlayerGamePitchingBuildError):
                ensure_output_path_safe(
                    raw_dir=root / "raw",
                    pa_path=pa_path,
                    output_path=pa_path,
                    seasons=(2023,),
                )

    def test_44_raw_season_file_overwrite_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir = root / "raw"
            with self.assertRaises(PlayerGamePitchingBuildError):
                ensure_output_path_safe(
                    raw_dir=raw_dir,
                    pa_path=root / "pa.parquet",
                    output_path=raw_dir / "2023.parquet",
                    seasons=(2023,),
                )


if __name__ == "__main__":
    unittest.main()
