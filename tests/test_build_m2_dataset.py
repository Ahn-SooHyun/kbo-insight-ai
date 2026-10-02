from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from src.data.m2_dataset import (
    EVENT_TO_CLASS,
    M2_X_COLUMNS,
    M2DatasetBuildError,
    TARGET_CODES,
    build_m2_dataset,
)
from src.features.player import (
    BATTING_SOURCE_COUNTS,
    PITCHING_SOURCE_COUNTS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "processed_dataset.json"
)


def load_config() -> dict[str, object]:
    """current-main Processed Dataset 설정을 읽는다."""
    return json.loads(
        CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )


def make_pa(
    *,
    game_pk: str = "G1",
    game_date: str = "2023-04-10",
    at_bat_number: int = 1,
    batter: str = "B1",
    pitcher: str = "P1",
    stand: str | None = "R",
    event: str | None = "single",
    pitch_count: int = 1,
) -> dict[str, object]:
    """M2 테스트용 최소 Canonical PA를 만든다."""
    return {
        "game_pk": game_pk,
        "game_date": game_date,
        "season": 2023,
        "at_bat_number": at_bat_number,
        "inning": 1,
        "inning_topbot": "top",
        "batting_team": "A",
        "fielding_team": "B",
        "is_home_batting": False,
        "batter": batter,
        "pitcher": pitcher,
        "stand": stand,
        "outs_before": 0,
        "on_1b_before": pd.NA,
        "on_2b_before": pd.NA,
        "on_3b_before": pd.NA,
        "home_score_before": 0,
        "away_score_before": 0,
        "score_diff_before": 0,
        "event": event,
        "pa_completed": event is not None,
        "pitch_count": pitch_count,
    }


def make_raw_row(
    *,
    game_pk: str = "G1",
    game_date: str = "2023-04-10",
    at_bat_number: int = 1,
    pitch_number: int = 1,
    batter: str = "B1",
    pitcher: str = "P1",
    stand: str | None = "R",
) -> dict[str, object]:
    """M2 테스트용 Raw Pitch Row를 만든다."""
    return {
        "game_pk": game_pk,
        "game_date": game_date,
        "inning": 1,
        "inning_topbot": "top",
        "at_bat_number": at_bat_number,
        "pitch_number": pitch_number,
        "batter": batter,
        "pitcher": pitcher,
        "stand": stand,
        "outs_when_up": 0,
        "on_1b": pd.NA,
        "on_2b": pd.NA,
        "on_3b": pd.NA,
        "home_score": 0,
        "away_score": 0,
    }


def make_batting_source(
    *,
    player_id: str = "B1",
    game_date: str = "2023-04-01",
    game_pk: str = "H1",
    pa: int = 1,
    h: int = 1,
) -> dict[str, object]:
    """#24 계약을 만족하는 최소 Player Game Batting Row를 만든다."""
    row: dict[str, object] = {
        "game_pk": game_pk,
        "game_date": game_date,
        "season": 2023,
        "batter": player_id,
    }

    row.update(
        {
            column: 0
            for column in BATTING_SOURCE_COUNTS
        }
    )

    row["pa"] = pa
    row["ab"] = pa
    row["h"] = h
    row["single"] = h
    row["tb"] = h

    return row


def make_pitching_source(
    *,
    player_id: str = "P1",
    game_date: str = "2023-04-01",
    game_pk: str = "H1",
    pitches: int = 1,
) -> dict[str, object]:
    """#24 계약을 만족하는 최소 Player Game Pitching Row를 만든다."""
    row: dict[str, object] = {
        "game_pk": game_pk,
        "game_date": game_date,
        "season": 2023,
        "pitcher": player_id,
    }

    row.update(
        {
            column: 0
            for column in PITCHING_SOURCE_COUNTS
        }
    )

    row["pitch_rows"] = pitches
    row["pitches"] = pitches
    row["strike_pitch_count"] = pitches
    row["batters_faced_completed"] = 1
    row["outs_recorded"] = 1

    return row


def empty_batting_source() -> pd.DataFrame:
    """필수 batting schema를 가진 빈 DataFrame을 만든다."""
    columns = [
        "game_pk",
        "game_date",
        "season",
        "batter",
        *BATTING_SOURCE_COUNTS,
    ]

    return pd.DataFrame(
        {
            column: pd.Series(
                dtype="object"
            )
            for column in columns
        }
    )


def empty_pitching_source() -> pd.DataFrame:
    """필수 pitching schema를 가진 빈 DataFrame을 만든다."""
    columns = [
        "game_pk",
        "game_date",
        "season",
        "pitcher",
        *PITCHING_SOURCE_COUNTS,
    ]

    return pd.DataFrame(
        {
            column: pd.Series(
                dtype="object"
            )
            for column in columns
        }
    )


class M2TargetMappingTest(
    unittest.TestCase
):
    """고정 15→11 Target Mapping을 검증한다."""

    def setUp(
        self,
    ) -> None:
        self.config = load_config()

    def test_all_known_events_map_to_fixed_classes(
        self,
    ) -> None:
        pa_rows = []
        raw_rows = []

        for index, event in enumerate(
            EVENT_TO_CLASS,
            start=1,
        ):
            game_pk = f"G{index}"

            pa_rows.append(
                make_pa(
                    game_pk=game_pk,
                    at_bat_number=1,
                    event=event,
                )
            )

            raw_rows.append(
                make_raw_row(
                    game_pk=game_pk,
                    at_bat_number=1,
                )
            )

        dataset, _, _ = build_m2_dataset(
            pd.DataFrame(
                pa_rows
            ),
            pd.DataFrame(
                raw_rows
            ),
            empty_batting_source(),
            empty_pitching_source(),
            config=self.config,
        )

        actual = {
            row.source_event: (
                row.target_class,
                int(
                    row.target_code
                ),
            )
            for row in dataset.itertuples()
        }

        for (
            event,
            target_class,
        ) in EVENT_TO_CLASS.items():
            self.assertEqual(
                actual[event],
                (
                    target_class,
                    TARGET_CODES[
                        target_class
                    ],
                ),
            )

    def test_unknown_non_null_event_fails(
        self,
    ) -> None:
        with self.assertRaises(
            M2DatasetBuildError
        ):
            build_m2_dataset(
                pd.DataFrame(
                    [
                        make_pa(
                            event="unknown_event"
                        )
                    ]
                ),
                pd.DataFrame(
                    [
                        make_raw_row()
                    ]
                ),
                empty_batting_source(),
                empty_pitching_source(),
                config=self.config,
            )

    def test_null_event_is_pending_not_supervised(
        self,
    ) -> None:
        dataset, _, summary = build_m2_dataset(
            pd.DataFrame(
                [
                    make_pa(
                        event=None,
                    )
                ]
            ),
            pd.DataFrame(
                [
                    make_raw_row()
                ]
            ),
            empty_batting_source(),
            empty_pitching_source(),
            config=self.config,
        )

        row = dataset.iloc[
            0
        ]

        self.assertTrue(
            pd.isna(
                row[
                    "target_class"
                ]
            )
        )
        self.assertTrue(
            bool(
                row[
                    "is_censored"
                ]
            )
        )
        self.assertFalse(
            bool(
                row[
                    "supervised_usable"
                ]
            )
        )
        self.assertEqual(
            row[
                "label_status"
            ],
            "target_pending",
        )
        self.assertEqual(
            summary[
                "target_pending_or_held"
            ],
            1,
        )


class M2StartIdentityTest(
    unittest.TestCase
):
    """Raw 시작 선수와 Canonical 귀속 선수를 분리하는지 검증한다."""

    def setUp(
        self,
    ) -> None:
        self.config = load_config()

    def test_starting_batter_feature_not_credited_batter(
        self,
    ) -> None:
        pa = pd.DataFrame(
            [
                make_pa(
                    batter="B2",
                    event="strikeout",
                    pitch_count=2,
                )
            ]
        )

        raw = pd.DataFrame(
            [
                make_raw_row(
                    pitch_number=1,
                    batter="B1",
                ),
                make_raw_row(
                    pitch_number=2,
                    batter="B2",
                ),
            ]
        )

        batting = pd.DataFrame(
            [
                make_batting_source(
                    player_id="B1",
                    game_pk="H1",
                    h=1,
                ),
                make_batting_source(
                    player_id="B2",
                    game_pk="H2",
                    h=0,
                ),
            ]
        )

        dataset, roles, _ = build_m2_dataset(
            pa,
            raw,
            batting,
            empty_pitching_source(),
            config=self.config,
        )

        row = dataset.iloc[
            0
        ]

        self.assertEqual(
            row[
                "starting_batter"
            ],
            "B1",
        )
        self.assertEqual(
            row[
                "credited_batter"
            ],
            "B2",
        )
        self.assertTrue(
            bool(
                row[
                    "batter_identity_diff"
                ]
            )
        )
        self.assertEqual(
            int(
                row[
                    "batter_hist_h"
                ]
            ),
            1,
        )
        self.assertNotIn(
            "credited_batter",
            roles[
                "x_columns"
            ],
        )

    def test_starting_pitcher_feature_not_final_pitcher(
        self,
    ) -> None:
        pa = pd.DataFrame(
            [
                make_pa(
                    pitcher="P2",
                    pitch_count=2,
                )
            ]
        )

        raw = pd.DataFrame(
            [
                make_raw_row(
                    pitch_number=1,
                    pitcher="P1",
                ),
                make_raw_row(
                    pitch_number=2,
                    pitcher="P2",
                ),
            ]
        )

        pitching = pd.DataFrame(
            [
                make_pitching_source(
                    player_id="P1",
                    game_pk="H1",
                    pitches=5,
                ),
                make_pitching_source(
                    player_id="P2",
                    game_pk="H2",
                    pitches=9,
                ),
            ]
        )

        dataset, _, _ = build_m2_dataset(
            pa,
            raw,
            empty_batting_source(),
            pitching,
            config=self.config,
        )

        row = dataset.iloc[
            0
        ]

        self.assertEqual(
            row[
                "starting_pitcher"
            ],
            "P1",
        )
        self.assertEqual(
            row[
                "final_pitcher"
            ],
            "P2",
        )
        self.assertTrue(
            bool(
                row[
                    "pitcher_identity_diff"
                ]
            )
        )
        self.assertEqual(
            int(
                row[
                    "pitcher_hist_pitches"
                ]
            ),
            5,
        )

    def test_null_starting_stand_is_preserved(
        self,
    ) -> None:
        dataset, roles, _ = build_m2_dataset(
            pd.DataFrame(
                [
                    make_pa(
                        stand="R",
                    )
                ]
            ),
            pd.DataFrame(
                [
                    make_raw_row(
                        stand=None,
                    )
                ]
            ),
            empty_batting_source(),
            empty_pitching_source(),
            config=self.config,
        )

        row = dataset.iloc[
            0
        ]

        self.assertTrue(
            pd.isna(
                row[
                    "starting_stand"
                ]
            )
        )
        self.assertTrue(
            bool(
                row[
                    "starting_stand_missing"
                ]
            )
        )
        self.assertIn(
            "starting_stand",
            roles[
                "x_columns"
            ],
        )


class M2RawStartQualityTest(
    unittest.TestCase
):
    """Pitch-less 및 초기 Pitch 미관측 PA의 정책을 검증한다."""

    def setUp(
        self,
    ) -> None:
        self.config = load_config()

    def test_pitchless_pa_is_kept(
        self,
    ) -> None:
        dataset, _, _ = build_m2_dataset(
            pd.DataFrame(
                [
                    make_pa(
                        event="walk",
                        pitch_count=0,
                    )
                ]
            ),
            pd.DataFrame(
                [
                    make_raw_row(
                        pitch_number=0,
                    )
                ]
            ),
            empty_batting_source(),
            empty_pitching_source(),
            config=self.config,
        )

        row = dataset.iloc[
            0
        ]

        self.assertTrue(
            bool(
                row[
                    "pitchless_pa"
                ]
            )
        )
        self.assertFalse(
            bool(
                row[
                    "is_excluded"
                ]
            )
        )
        self.assertTrue(
            bool(
                row[
                    "supervised_usable"
                ]
            )
        )

    def test_first_observed_pitch_gt_one_is_explicitly_excluded(
        self,
    ) -> None:
        dataset, _, summary = build_m2_dataset(
            pd.DataFrame(
                [
                    make_pa(
                        pitch_count=1,
                    )
                ]
            ),
            pd.DataFrame(
                [
                    make_raw_row(
                        pitch_number=2,
                    )
                ]
            ),
            empty_batting_source(),
            empty_pitching_source(),
            config=self.config,
        )

        row = dataset.iloc[
            0
        ]

        self.assertTrue(
            bool(
                row[
                    "is_excluded"
                ]
            )
        )
        self.assertEqual(
            row[
                "exclusion_reason"
            ],
            "start_context_not_fully_observed",
        )
        self.assertFalse(
            bool(
                row[
                    "supervised_usable"
                ]
            )
        )
        self.assertEqual(
            summary[
                "excluded"
            ],
            1,
        )


class M2LeakageTest(
    unittest.TestCase
):
    """현재/미래 정보 변경이 M2 시작 X를 바꾸지 않는지 검증한다."""

    def setUp(
        self,
    ) -> None:
        self.config = load_config()

    def _build(
        self,
        pa: pd.DataFrame,
        raw: pd.DataFrame,
        batting: pd.DataFrame,
        pitching: pd.DataFrame,
    ) -> pd.DataFrame:
        dataset, roles, _ = build_m2_dataset(
            pa,
            raw,
            batting,
            pitching,
            config=self.config,
        )

        return (
            dataset.loc[
                :,
                roles[
                    "x_columns"
                ],
            ]
            .copy()
        )

    def test_event_and_finishing_player_change_do_not_change_x(
        self,
    ) -> None:
        raw = pd.DataFrame(
            [
                make_raw_row(
                    batter="B1",
                    pitcher="P1",
                )
            ]
        )

        original_pa = pd.DataFrame(
            [
                make_pa(
                    batter="B1",
                    pitcher="P1",
                    event="single",
                )
            ]
        )

        changed_pa = pd.DataFrame(
            [
                make_pa(
                    batter="B9",
                    pitcher="P9",
                    stand="L",
                    event="home_run",
                )
            ]
        )

        batting = pd.DataFrame(
            [
                make_batting_source(
                    player_id="B1"
                )
            ]
        )

        pitching = pd.DataFrame(
            [
                make_pitching_source(
                    player_id="P1"
                )
            ]
        )

        original_x = self._build(
            original_pa,
            raw,
            batting,
            pitching,
        )

        changed_x = self._build(
            changed_pa,
            raw,
            batting,
            pitching,
        )

        assert_frame_equal(
            original_x,
            changed_x,
        )

    def test_same_day_and_future_history_do_not_change_x(
        self,
    ) -> None:
        pa = pd.DataFrame(
            [
                make_pa()
            ]
        )

        raw = pd.DataFrame(
            [
                make_raw_row()
            ]
        )

        base_batting = pd.DataFrame(
            [
                make_batting_source(
                    player_id="B1",
                    game_date="2023-04-01",
                    game_pk="H1",
                )
            ]
        )

        changed_batting = pd.DataFrame(
            [
                make_batting_source(
                    player_id="B1",
                    game_date="2023-04-01",
                    game_pk="H1",
                ),
                make_batting_source(
                    player_id="B1",
                    game_date="2023-04-10",
                    game_pk="SAME_DAY",
                    h=1,
                ),
                make_batting_source(
                    player_id="B1",
                    game_date="2023-04-11",
                    game_pk="FUTURE",
                    h=1,
                ),
            ]
        )

        pitching = pd.DataFrame(
            [
                make_pitching_source(
                    player_id="P1"
                )
            ]
        )

        original_x = self._build(
            pa,
            raw,
            base_batting,
            pitching,
        )

        changed_x = self._build(
            pa,
            raw,
            changed_batting,
            pitching,
        )

        assert_frame_equal(
            original_x,
            changed_x,
        )

    def test_input_reordering_is_deterministic(
        self,
    ) -> None:
        pa = pd.DataFrame(
            [
                make_pa(
                    game_pk="G1",
                    at_bat_number=1,
                ),
                make_pa(
                    game_pk="G2",
                    at_bat_number=1,
                ),
            ]
        )

        raw = pd.DataFrame(
            [
                make_raw_row(
                    game_pk="G1",
                ),
                make_raw_row(
                    game_pk="G2",
                ),
            ]
        )

        batting = pd.DataFrame(
            [
                make_batting_source(
                    player_id="B1"
                )
            ]
        )

        pitching = pd.DataFrame(
            [
                make_pitching_source(
                    player_id="P1"
                )
            ]
        )

        first, _, _ = build_m2_dataset(
            pa,
            raw,
            batting,
            pitching,
            config=self.config,
        )

        second, _, _ = build_m2_dataset(
            pa.sample(
                frac=1,
                random_state=7,
            ),
            raw.sample(
                frac=1,
                random_state=11,
            ),
            batting.sample(
                frac=1,
                random_state=13,
            ),
            pitching.sample(
                frac=1,
                random_state=17,
            ),
            config=self.config,
        )

        assert_frame_equal(
            first,
            second,
        )


class M2RoleAndSplitTest(
    unittest.TestCase
):
    """X 역할 분리와 고정 시간 Split을 검증한다."""

    def setUp(
        self,
    ) -> None:
        self.config = load_config()

    def test_x_allowlist_does_not_contain_result_or_identity_audit(
        self,
    ) -> None:
        forbidden = {
            "source_event",
            "target_class",
            "target_code",
            "credited_batter",
            "final_pitcher",
            "pa_completed",
            "pitch_count",
            "split",
            "is_excluded",
        }

        self.assertTrue(
            forbidden.isdisjoint(
                M2_X_COLUMNS
            )
        )

    def test_2023_row_is_train_split(
        self,
    ) -> None:
        dataset, _, _ = build_m2_dataset(
            pd.DataFrame(
                [
                    make_pa(
                        game_date="2023-04-10"
                    )
                ]
            ),
            pd.DataFrame(
                [
                    make_raw_row(
                        game_date="2023-04-10"
                    )
                ]
            ),
            empty_batting_source(),
            empty_pitching_source(),
            config=self.config,
        )

        self.assertEqual(
            dataset.iloc[
                0
            ][
                "split"
            ],
            "train",
        )


if __name__ == "__main__":
    unittest.main()