from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from pandas.testing import assert_frame_equal

from src.data.m2_dataset import (
    EVENT_TO_CLASS,
    M2_FEATURE_VERSION,
    M2_X_COLUMNS,
    M2DatasetBuildError,
    OUTPUT_SCHEMA_VERSION,
    PA_KEY,
    TARGET_CLASSES,
    TARGET_CODES,
    TARGET_VERSION,
    _coerce_round_trip_dtypes,
    build_deterministic_identity,
    build_m2_dataset,
    build_quality_report,
    build_schema_payload,
)
from src.data.processed_contract import (
    FEATURE_CATALOG_VERSION,
    PREDICTION_CONTRACT_VERSION,
    SPLIT_VERSION,
    ProcessedContractError,
    assert_input_hashes_unchanged,
    calculate_input_hashes,
    content_fingerprint,
    validate_output_paths,
)
from src.features.player import (
    BATTING_SOURCE_COUNTS,
    FEATURE_VERSION as PLAYER_FEATURE_VERSION,
    PITCHING_SOURCE_COUNTS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "processed_dataset.json"
)
PYARROW_AVAILABLE = (
    importlib.util.find_spec("pyarrow")
    is not None
)


def load_config() -> dict[str, object]:
    """현재 Processed Dataset 설정을 읽는다."""
    return json.loads(
        CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )


def make_pa(
    *,
    game_pk: str = "G1",
    game_date: str = "2023-04-10",
    season: int | None = None,
    at_bat_number: int = 1,
    inning: int = 1,
    inning_topbot: str = "top",
    batting_team: str = "A",
    fielding_team: str = "B",
    is_home_batting: bool = False,
    batter: str = "B1",
    pitcher: str = "P1",
    stand: str | None = "R",
    outs_before: int = 0,
    on_1b_before: object = pd.NA,
    on_2b_before: object = pd.NA,
    on_3b_before: object = pd.NA,
    home_score_before: int = 0,
    away_score_before: int = 0,
    score_diff_before: int = 0,
    event: str | None = "single",
    pitch_count: int = 1,
) -> dict[str, object]:
    """M2 테스트용 Canonical PA 한 행을 만든다."""
    resolved_season = (
        int(
            pd.Timestamp(
                game_date
            ).year
        )
        if season is None
        else season
    )

    return {
        "game_pk": game_pk,
        "game_date": game_date,
        "season": resolved_season,
        "at_bat_number": at_bat_number,
        "inning": inning,
        "inning_topbot": inning_topbot,
        "batting_team": batting_team,
        "fielding_team": fielding_team,
        "is_home_batting": is_home_batting,
        "batter": batter,
        "pitcher": pitcher,
        "stand": stand,
        "outs_before": outs_before,
        "on_1b_before": on_1b_before,
        "on_2b_before": on_2b_before,
        "on_3b_before": on_3b_before,
        "home_score_before": home_score_before,
        "away_score_before": away_score_before,
        "score_diff_before": score_diff_before,
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
    inning: int = 1,
    inning_topbot: str = "top",
    batter: str = "B1",
    pitcher: str = "P1",
    stand: str | None = "R",
    outs_when_up: int = 0,
    on_1b: object = pd.NA,
    on_2b: object = pd.NA,
    on_3b: object = pd.NA,
    home_score: int = 0,
    away_score: int = 0,
) -> dict[str, object]:
    """M2 테스트용 Raw Pitch Row를 만든다."""
    return {
        "game_pk": game_pk,
        "game_date": game_date,
        "inning": inning,
        "inning_topbot": inning_topbot,
        "at_bat_number": at_bat_number,
        "pitch_number": pitch_number,
        "batter": batter,
        "pitcher": pitcher,
        "stand": stand,
        "outs_when_up": outs_when_up,
        "on_1b": on_1b,
        "on_2b": on_2b,
        "on_3b": on_3b,
        "home_score": home_score,
        "away_score": away_score,
    }


def make_batting_source(
    *,
    player_id: str = "B1",
    game_date: str = "2023-04-01",
    game_pk: str = "H1",
    season: int | None = None,
    pa: int = 1,
    h: int = 1,
) -> dict[str, object]:
    """#24 계약을 만족하는 최소 Player Game Batting Row를 만든다."""
    resolved_season = (
        int(
            pd.Timestamp(
                game_date
            ).year
        )
        if season is None
        else season
    )

    row: dict[str, object] = {
        "game_pk": game_pk,
        "game_date": game_date,
        "season": resolved_season,
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
    season: int | None = None,
    pitches: int = 1,
    outs_recorded: int | None = 1,
) -> dict[str, object]:
    """#24 계약을 만족하는 최소 Player Game Pitching Row를 만든다."""
    resolved_season = (
        int(
            pd.Timestamp(
                game_date
            ).year
        )
        if season is None
        else season
    )

    row: dict[str, object] = {
        "game_pk": game_pk,
        "game_date": game_date,
        "season": resolved_season,
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
    row["outs_recorded"] = (
        pd.NA
        if outs_recorded is None
        else outs_recorded
    )

    return row


def empty_batting_source() -> pd.DataFrame:
    """필수 Batting Source Schema를 가진 빈 DataFrame을 만든다."""
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
    """필수 Pitching Source Schema를 가진 빈 DataFrame을 만든다."""
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


def build_single_fixture(
    *,
    pa: dict[str, object] | None = None,
    raw: dict[str, object] | None = None,
    batting_source: pd.DataFrame | None = None,
    pitching_source: pd.DataFrame | None = None,
) -> tuple[
    pd.DataFrame,
    dict[str, list[str]],
    dict[str, object],
]:
    """단일 PA 기본 Fixture를 M2 Dataset으로 조립한다."""
    return build_m2_dataset(
        pd.DataFrame(
            [
                pa
                if pa is not None
                else make_pa()
            ]
        ),
        pd.DataFrame(
            [
                raw
                if raw is not None
                else make_raw_row()
            ]
        ),
        (
            batting_source
            if batting_source is not None
            else empty_batting_source()
        ),
        (
            pitching_source
            if pitching_source is not None
            else empty_pitching_source()
        ),
        config=load_config(),
    )


class M2TargetMappingTest(
    unittest.TestCase
):
    """15개 Source Event와 고정 11-class Target 계약을 검증한다."""

    def test_all_known_events_map_to_fixed_classes(
        self,
    ) -> None:
        pa_rows: list[
            dict[str, object]
        ] = []
        raw_rows: list[
            dict[str, object]
        ] = []

        for index, event in enumerate(
            EVENT_TO_CLASS,
            start=1,
        ):
            game_pk = f"G{index}"

            pa_rows.append(
                make_pa(
                    game_pk=game_pk,
                    event=event,
                )
            )
            raw_rows.append(
                make_raw_row(
                    game_pk=game_pk,
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
            config=load_config(),
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
                actual[
                    event
                ],
                (
                    target_class,
                    TARGET_CODES[
                        target_class
                    ],
                ),
            )

    def test_out_event_family_uses_same_out_code(
        self,
    ) -> None:
        events = (
            "field_out",
            "double_play",
            "triple_play",
            "sac_bunt",
            "sac_fly",
        )

        pa_rows = [
            make_pa(
                game_pk=f"G{index}",
                event=event,
            )
            for index, event in enumerate(
                events,
                start=1,
            )
        ]
        raw_rows = [
            make_raw_row(
                game_pk=f"G{index}",
            )
            for index in range(
                1,
                len(events) + 1,
            )
        ]

        dataset, _, _ = build_m2_dataset(
            pd.DataFrame(
                pa_rows
            ),
            pd.DataFrame(
                raw_rows
            ),
            empty_batting_source(),
            empty_pitching_source(),
            config=load_config(),
        )

        self.assertTrue(
            dataset[
                "target_class"
            ]
            .eq(
                "OUT"
            )
            .all()
        )
        self.assertTrue(
            dataset[
                "target_code"
            ]
            .eq(
                TARGET_CODES[
                    "OUT"
                ]
            )
            .all()
        )

    def test_target_class_order_matches_config(
        self,
    ) -> None:
        expected = [
            {
                "class": name,
                "code": TARGET_CODES[
                    name
                ],
            }
            for name in TARGET_CLASSES
        ]

        self.assertEqual(
            load_config()[
                "m2"
            ][
                "target_classes"
            ],
            expected,
        )

    def test_unknown_non_null_event_fails(
        self,
    ) -> None:
        with self.assertRaises(
            M2DatasetBuildError
        ):
            build_single_fixture(
                pa=make_pa(
                    event="unknown_event"
                )
            )

    def test_null_event_is_pending_not_supervised(
        self,
    ) -> None:
        dataset, _, summary = (
            build_single_fixture(
                pa=make_pa(
                    event=None
                )
            )
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
            pd.isna(
                row[
                    "target_code"
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

    def test_source_event_is_preserved_for_audit(
        self,
    ) -> None:
        dataset, _, _ = (
            build_single_fixture(
                pa=make_pa(
                    event="field_error"
                )
            )
        )

        row = dataset.iloc[
            0
        ]

        self.assertEqual(
            row[
                "source_event"
            ],
            "field_error",
        )
        self.assertEqual(
            row[
                "target_class"
            ],
            "ROE",
        )


class M2StartIdentityTest(
    unittest.TestCase
):
    """Raw 첫 Row의 시작 선수와 Canonical 귀속 선수를 분리해 검증한다."""

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
            config=load_config(),
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
            config=load_config(),
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

    def test_first_row_null_stand_is_not_backfilled_from_later_row(
        self,
    ) -> None:
        pa = pd.DataFrame(
            [
                make_pa(
                    stand="L",
                    pitch_count=2,
                )
            ]
        )

        raw = pd.DataFrame(
            [
                make_raw_row(
                    pitch_number=1,
                    stand=None,
                ),
                make_raw_row(
                    pitch_number=2,
                    stand="L",
                ),
            ]
        )

        dataset, roles, _ = build_m2_dataset(
            pa,
            raw,
            empty_batting_source(),
            empty_pitching_source(),
            config=load_config(),
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
        self.assertTrue(
            bool(
                row[
                    "stand_identity_diff"
                ]
            )
        )
        self.assertIn(
            "starting_stand",
            roles[
                "x_columns"
            ],
        )

    def test_first_row_identity_is_not_mixed_with_later_values(
        self,
    ) -> None:
        pa = pd.DataFrame(
            [
                make_pa(
                    batter="B2",
                    pitcher="P2",
                    stand="L",
                    pitch_count=2,
                )
            ]
        )

        raw = pd.DataFrame(
            [
                make_raw_row(
                    pitch_number=1,
                    batter="B1",
                    pitcher="P1",
                    stand=None,
                ),
                make_raw_row(
                    pitch_number=2,
                    batter="B2",
                    pitcher="P2",
                    stand="L",
                ),
            ]
        )

        dataset, _, _ = build_m2_dataset(
            pa,
            raw,
            empty_batting_source(),
            empty_pitching_source(),
            config=load_config(),
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
                "starting_pitcher"
            ],
            "P1",
        )
        self.assertTrue(
            pd.isna(
                row[
                    "starting_stand"
                ]
            )
        )

    def test_two_strike_substitution_strikeout_keeps_pa_start_batter(
        self,
    ) -> None:
        pa = pd.DataFrame(
            [
                make_pa(
                    batter="B1",
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

        dataset, _, _ = build_m2_dataset(
            pa,
            raw,
            empty_batting_source(),
            empty_pitching_source(),
            config=load_config(),
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
            "B1",
        )
        self.assertEqual(
            row[
                "target_class"
            ],
            "SO",
        )
        self.assertEqual(
            int(
                row[
                    "target_code"
                ]
            ),
            TARGET_CODES[
                "SO"
            ],
        )


class M2RawStartQualityTest(
    unittest.TestCase
):
    """Raw ordering, PA key, pitch-less 및 시작 Context 품질 계약을 검증한다."""

    def test_pitchless_pa_is_kept(
        self,
    ) -> None:
        dataset, _, _ = (
            build_single_fixture(
                pa=make_pa(
                    event="walk",
                    pitch_count=0,
                ),
                raw=make_raw_row(
                    pitch_number=0,
                ),
            )
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
        dataset, _, summary = (
            build_single_fixture(
                pa=make_pa(
                    pitch_count=1,
                ),
                raw=make_raw_row(
                    pitch_number=2,
                ),
            )
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

    def test_pitch_zero_followed_by_actual_pitch_fails(
        self,
    ) -> None:
        with self.assertRaises(
            M2DatasetBuildError
        ):
            build_m2_dataset(
                pd.DataFrame(
                    [
                        make_pa(
                            pitch_count=1
                        )
                    ]
                ),
                pd.DataFrame(
                    [
                        make_raw_row(
                            pitch_number=0,
                        ),
                        make_raw_row(
                            pitch_number=1,
                        ),
                    ]
                ),
                empty_batting_source(),
                empty_pitching_source(),
                config=load_config(),
            )

    def test_duplicate_raw_pitch_key_fails(
        self,
    ) -> None:
        duplicate_row = (
            make_raw_row()
        )

        with self.assertRaises(
            M2DatasetBuildError
        ):
            build_m2_dataset(
                pd.DataFrame(
                    [
                        make_pa()
                    ]
                ),
                pd.DataFrame(
                    [
                        duplicate_row,
                        dict(
                            duplicate_row
                        ),
                    ]
                ),
                empty_batting_source(),
                empty_pitching_source(),
                config=load_config(),
            )

    def test_missing_raw_pa_key_fails(
        self,
    ) -> None:
        pa = pd.DataFrame(
            [
                make_pa(
                    game_pk="G1"
                ),
                make_pa(
                    game_pk="G2"
                ),
            ]
        )

        raw = pd.DataFrame(
            [
                make_raw_row(
                    game_pk="G1"
                )
            ]
        )

        with self.assertRaises(
            M2DatasetBuildError
        ):
            build_m2_dataset(
                pa,
                raw,
                empty_batting_source(),
                empty_pitching_source(),
                config=load_config(),
            )

    def test_orphan_raw_pa_key_fails(
        self,
    ) -> None:
        pa = pd.DataFrame(
            [
                make_pa(
                    game_pk="G1"
                )
            ]
        )

        raw = pd.DataFrame(
            [
                make_raw_row(
                    game_pk="G1"
                ),
                make_raw_row(
                    game_pk="G2"
                ),
            ]
        )

        with self.assertRaises(
            M2DatasetBuildError
        ):
            build_m2_dataset(
                pa,
                raw,
                empty_batting_source(),
                empty_pitching_source(),
                config=load_config(),
            )

    def test_raw_canonical_game_date_mismatch_fails(
        self,
    ) -> None:
        with self.assertRaises(
            M2DatasetBuildError
        ):
            build_single_fixture(
                pa=make_pa(
                    game_date="2023-04-10"
                ),
                raw=make_raw_row(
                    game_date="2023-04-11"
                ),
            )

    def test_raw_canonical_start_context_mismatch_fails(
        self,
    ) -> None:
        with self.assertRaises(
            M2DatasetBuildError
        ):
            build_single_fixture(
                pa=make_pa(
                    outs_before=0
                ),
                raw=make_raw_row(
                    outs_when_up=1
                ),
            )

    def test_raw_actual_pitch_count_mismatch_fails(
        self,
    ) -> None:
        with self.assertRaises(
            M2DatasetBuildError
        ):
            build_single_fixture(
                pa=make_pa(
                    pitch_count=2
                ),
                raw=make_raw_row(
                    pitch_number=1
                ),
            )


class M2PlayerFeatureJoinTest(
    unittest.TestCase
):
    """#24 Player Feature 요청·Cutoff·Join 계약을 검증한다."""

    def test_duplicate_player_requests_are_deduplicated(
        self,
    ) -> None:
        pa = pd.DataFrame(
            [
                make_pa(
                    at_bat_number=1
                ),
                make_pa(
                    at_bat_number=2
                ),
            ]
        )

        raw = pd.DataFrame(
            [
                make_raw_row(
                    at_bat_number=1
                ),
                make_raw_row(
                    at_bat_number=2
                ),
            ]
        )

        dataset, _, summary = build_m2_dataset(
            pa,
            raw,
            empty_batting_source(),
            empty_pitching_source(),
            config=load_config(),
        )

        provenance = summary[
            "player_feature_provenance"
        ]

        self.assertEqual(
            len(
                dataset
            ),
            2,
        )
        self.assertEqual(
            provenance[
                "batting_request_count"
            ],
            1,
        )
        self.assertEqual(
            provenance[
                "pitching_request_count"
            ],
            1,
        )
        self.assertEqual(
            provenance[
                "request_count"
            ],
            2,
        )

    def test_cold_start_player_requests_are_preserved(
        self,
    ) -> None:
        dataset, _, _ = (
            build_single_fixture()
        )

        row = dataset.iloc[
            0
        ]

        self.assertFalse(
            bool(
                row[
                    "batter_has_history"
                ]
            )
        )
        self.assertFalse(
            bool(
                row[
                    "pitcher_has_history"
                ]
            )
        )
        self.assertEqual(
            int(
                row[
                    "batter_hist_pa"
                ]
            ),
            0,
        )
        self.assertEqual(
            int(
                row[
                    "pitcher_hist_pitches"
                ]
            ),
            0,
        )
        self.assertTrue(
            pd.isna(
                row[
                    "batter_hist_avg"
                ]
            )
        )

    def test_nullable_pitcher_outs_is_preserved(
        self,
    ) -> None:
        pitching = pd.DataFrame(
            [
                make_pitching_source(
                    outs_recorded=None
                )
            ]
        )

        dataset, _, _ = (
            build_single_fixture(
                pitching_source=pitching,
            )
        )

        row = dataset.iloc[
            0
        ]

        self.assertTrue(
            bool(
                row[
                    "pitcher_has_history"
                ]
            )
        )
        self.assertTrue(
            pd.isna(
                row[
                    "pitcher_hist_outs_recorded"
                ]
            )
        )

    def test_historical_feature_uses_previous_date_row(
        self,
    ) -> None:
        batting = pd.DataFrame(
            [
                make_batting_source(
                    game_date="2023-04-01",
                    h=1,
                )
            ]
        )

        dataset, _, _ = (
            build_single_fixture(
                batting_source=batting,
            )
        )

        row = dataset.iloc[
            0
        ]

        self.assertTrue(
            bool(
                row[
                    "batter_has_history"
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
        self.assertLess(
            row[
                "batter_max_source_game_date"
            ],
            row[
                "prediction_date"
            ],
        )

    def test_player_feature_join_preserves_pa_cardinality(
        self,
    ) -> None:
        pa_rows = [
            make_pa(
                game_pk="G1",
                at_bat_number=1,
            ),
            make_pa(
                game_pk="G1",
                at_bat_number=2,
            ),
            make_pa(
                game_pk="G2",
                at_bat_number=1,
            ),
        ]

        raw_rows = [
            make_raw_row(
                game_pk="G1",
                at_bat_number=1,
            ),
            make_raw_row(
                game_pk="G1",
                at_bat_number=2,
            ),
            make_raw_row(
                game_pk="G2",
                at_bat_number=1,
            ),
        ]

        dataset, _, _ = build_m2_dataset(
            pd.DataFrame(
                pa_rows
            ),
            pd.DataFrame(
                raw_rows
            ),
            empty_batting_source(),
            empty_pitching_source(),
            config=load_config(),
        )

        self.assertEqual(
            len(
                dataset
            ),
            len(
                pa_rows
            ),
        )
        self.assertEqual(
            int(
                dataset.duplicated(
                    list(
                        PA_KEY
                    ),
                    keep=False,
                ).sum()
            ),
            0,
        )


class M2LeakageTest(
    unittest.TestCase
):
    """현재·사후·Same-day·미래 정보가 M2 시작 X에 유입되지 않는지 검증한다."""

    def _build_x(
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
            config=load_config(),
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

        original_x = self._build_x(
            original_pa,
            raw,
            batting,
            pitching,
        )
        changed_x = self._build_x(
            changed_pa,
            raw,
            batting,
            pitching,
        )

        assert_frame_equal(
            original_x,
            changed_x,
        )

    def test_post_state_change_does_not_change_x(
        self,
    ) -> None:
        raw = pd.DataFrame(
            [
                make_raw_row()
            ]
        )

        original_pa = pd.DataFrame(
            [
                {
                    **make_pa(),
                    "post_outs": 1,
                    "post_on_1b": "R1",
                    "post_on_2b": pd.NA,
                    "post_on_3b": pd.NA,
                    "post_home_score": 0,
                    "post_away_score": 0,
                    "runs_scored": 0,
                }
            ]
        )

        changed_pa = original_pa.copy()
        changed_pa.loc[
            0,
            "post_outs",
        ] = 3
        changed_pa.loc[
            0,
            "post_on_1b",
        ] = pd.NA
        changed_pa.loc[
            0,
            "post_on_2b",
        ] = "R2"
        changed_pa.loc[
            0,
            "post_home_score",
        ] = 99
        changed_pa.loc[
            0,
            "post_away_score",
        ] = 88
        changed_pa.loc[
            0,
            "runs_scored",
        ] = 7

        original_x = self._build_x(
            original_pa,
            raw,
            empty_batting_source(),
            empty_pitching_source(),
        )
        changed_x = self._build_x(
            changed_pa,
            raw,
            empty_batting_source(),
            empty_pitching_source(),
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

        original_x = self._build_x(
            pa,
            raw,
            base_batting,
            pitching,
        )
        changed_x = self._build_x(
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
                ),
                make_pa(
                    game_pk="G2",
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

        first, _, first_summary = build_m2_dataset(
            pa,
            raw,
            batting,
            pitching,
            config=load_config(),
        )
        second, _, second_summary = build_m2_dataset(
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
            config=load_config(),
        )

        assert_frame_equal(
            first,
            second,
        )
        self.assertEqual(
            first_summary[
                "player_feature_provenance"
            ][
                "content_fingerprint"
            ],
            second_summary[
                "player_feature_provenance"
            ][
                "content_fingerprint"
            ],
        )


class M2RoleAndSplitTest(
    unittest.TestCase
):
    """X 역할 분리와 고정 시간 Split 계약을 검증한다."""

    def test_x_allowlist_does_not_contain_result_or_identity_audit(
        self,
    ) -> None:
        forbidden = {
            "source_event",
            "target_class",
            "target_code",
            "credited_batter",
            "final_pitcher",
            "credited_stand",
            "pa_completed",
            "pitch_count",
            "split",
            "is_excluded",
            "exclusion_reason",
            "is_censored",
            "censor_reason",
            "post_outs",
            "post_on_1b",
            "post_on_2b",
            "post_on_3b",
            "post_home_score",
            "post_away_score",
            "runs_scored",
        }

        self.assertTrue(
            forbidden.isdisjoint(
                M2_X_COLUMNS
            )
        )

    def test_player_ids_are_audit_not_x(
        self,
    ) -> None:
        self.assertNotIn(
            "starting_batter",
            M2_X_COLUMNS,
        )
        self.assertNotIn(
            "starting_pitcher",
            M2_X_COLUMNS,
        )

    def test_all_fixed_split_boundaries(
        self,
    ) -> None:
        fixtures = [
            (
                "G2023",
                "2023-04-01",
                "train",
            ),
            (
                "G2024",
                "2024-04-01",
                "validation",
            ),
            (
                "G2025",
                "2025-04-01",
                "test",
            ),
            (
                "G2026",
                "2026-04-01",
                "snapshot",
            ),
        ]

        pa = pd.DataFrame(
            [
                make_pa(
                    game_pk=game_pk,
                    game_date=game_date,
                )
                for (
                    game_pk,
                    game_date,
                    _
                ) in fixtures
            ]
        )
        raw = pd.DataFrame(
            [
                make_raw_row(
                    game_pk=game_pk,
                    game_date=game_date,
                )
                for (
                    game_pk,
                    game_date,
                    _
                ) in fixtures
            ]
        )

        dataset, _, _ = build_m2_dataset(
            pa,
            raw,
            empty_batting_source(),
            empty_pitching_source(),
            config=load_config(),
        )

        actual = dict(
            zip(
                dataset[
                    "game_pk"
                ].astype(str),
                dataset[
                    "split"
                ].astype(str),
            )
        )

        for (
            game_pk,
            _,
            expected_split,
        ) in fixtures:
            self.assertEqual(
                actual[
                    game_pk
                ],
                expected_split,
            )

    def test_same_game_cannot_have_multiple_prediction_dates(
        self,
    ) -> None:
        pa = pd.DataFrame(
            [
                make_pa(
                    game_pk="G1",
                    game_date="2023-12-31",
                    at_bat_number=1,
                ),
                make_pa(
                    game_pk="G1",
                    game_date="2024-01-01",
                    at_bat_number=2,
                ),
            ]
        )
        raw = pd.DataFrame(
            [
                make_raw_row(
                    game_pk="G1",
                    game_date="2023-12-31",
                    at_bat_number=1,
                ),
                make_raw_row(
                    game_pk="G1",
                    game_date="2024-01-01",
                    at_bat_number=2,
                ),
            ]
        )

        with self.assertRaises(
            M2DatasetBuildError
        ):
            build_m2_dataset(
                pa,
                raw,
                empty_batting_source(),
                empty_pitching_source(),
                config=load_config(),
            )

    def test_schema_roles_are_disjoint(
        self,
    ) -> None:
        _, roles, _ = (
            build_single_fixture()
        )

        all_columns: list[str] = []

        for columns in roles.values():
            all_columns.extend(
                columns
            )

        self.assertEqual(
            len(
                all_columns
            ),
            len(
                set(
                    all_columns
                )
            ),
        )


class M2ArtifactContractTest(
    unittest.TestCase
):
    """Quality·Schema·Fingerprint·Parquet·Hash·Path 계약을 검증한다."""

    def test_quantity_conservation(
        self,
    ) -> None:
        pa = pd.DataFrame(
            [
                make_pa(
                    game_pk="USABLE",
                    event="single",
                ),
                make_pa(
                    game_pk="PENDING",
                    event=None,
                ),
                make_pa(
                    game_pk="EXCLUDED",
                    event="walk",
                    pitch_count=1,
                ),
            ]
        )
        raw = pd.DataFrame(
            [
                make_raw_row(
                    game_pk="USABLE",
                    pitch_number=1,
                ),
                make_raw_row(
                    game_pk="PENDING",
                    pitch_number=1,
                ),
                make_raw_row(
                    game_pk="EXCLUDED",
                    pitch_number=2,
                ),
            ]
        )

        dataset, _, summary = build_m2_dataset(
            pa,
            raw,
            empty_batting_source(),
            empty_pitching_source(),
            config=load_config(),
        )

        self.assertEqual(
            len(
                dataset
            ),
            3,
        )
        self.assertEqual(
            summary[
                "supervised_usable"
            ],
            1,
        )
        self.assertEqual(
            summary[
                "target_pending_or_held"
            ],
            1,
        )
        self.assertEqual(
            summary[
                "excluded"
            ],
            1,
        )
        self.assertEqual(
            summary[
                "reconciled_total"
            ],
            3,
        )

    def test_event_mapping_reconciliation_matches_input_rows(
        self,
    ) -> None:
        pa_rows = []
        raw_rows = []

        for index, event in enumerate(
            EVENT_TO_CLASS,
            start=1,
        ):
            pa_rows.append(
                make_pa(
                    game_pk=f"G{index}",
                    event=event,
                )
            )
            raw_rows.append(
                make_raw_row(
                    game_pk=f"G{index}",
                )
            )

        pa_rows.append(
            make_pa(
                game_pk="NULL_EVENT",
                event=None,
            )
        )
        raw_rows.append(
            make_raw_row(
                game_pk="NULL_EVENT",
            )
        )

        dataset, _, summary = build_m2_dataset(
            pd.DataFrame(
                pa_rows
            ),
            pd.DataFrame(
                raw_rows
            ),
            empty_batting_source(),
            empty_pitching_source(),
            config=load_config(),
        )

        reconciliation = summary[
            "event_mapping_reconciliation"
        ]

        reconciled_total = sum(
            int(
                payload[
                    "input_rows"
                ]
            )
            for payload in reconciliation.values()
        )

        self.assertEqual(
            reconciled_total,
            len(
                dataset
            ),
        )

        for (
            event,
            expected_target,
        ) in EVENT_TO_CLASS.items():
            self.assertEqual(
                reconciliation[
                    event
                ][
                    "target_class"
                ],
                expected_target,
            )

        self.assertIsNone(
            reconciliation[
                "<NULL>"
            ][
                "target_class"
            ]
        )

    def test_schema_payload_contains_all_contract_versions(
        self,
    ) -> None:
        dataset, roles, _ = (
            build_single_fixture()
        )

        schema = build_schema_payload(
            dataset,
            roles,
        )

        self.assertEqual(
            schema[
                "schema_version"
            ],
            OUTPUT_SCHEMA_VERSION,
        )
        self.assertEqual(
            schema[
                "prediction_contract_version"
            ],
            PREDICTION_CONTRACT_VERSION,
        )
        self.assertEqual(
            schema[
                "feature_catalog_version"
            ],
            FEATURE_CATALOG_VERSION,
        )
        self.assertEqual(
            schema[
                "feature_version"
            ],
            M2_FEATURE_VERSION,
        )
        self.assertEqual(
            schema[
                "player_feature_version"
            ],
            PLAYER_FEATURE_VERSION,
        )
        self.assertEqual(
            schema[
                "target_version"
            ],
            TARGET_VERSION,
        )
        self.assertEqual(
            schema[
                "split_version"
            ],
            SPLIT_VERSION,
        )

    def test_player_feature_fingerprint_is_recorded(
        self,
    ) -> None:
        _, _, summary = (
            build_single_fixture()
        )

        provenance = summary[
            "player_feature_provenance"
        ]

        self.assertEqual(
            provenance[
                "feature_version"
            ],
            PLAYER_FEATURE_VERSION,
        )
        self.assertEqual(
            len(
                provenance[
                    "content_fingerprint"
                ]
            ),
            64,
        )
        self.assertEqual(
            len(
                provenance[
                    "batting_content_fingerprint"
                ]
            ),
            64,
        )
        self.assertEqual(
            len(
                provenance[
                    "pitching_content_fingerprint"
                ]
            ),
            64,
        )

    def test_output_dtypes_use_nullable_contract(
        self,
    ) -> None:
        dataset, _, _ = (
            build_single_fixture()
        )

        self.assertEqual(
            str(
                dataset[
                    "target_code"
                ].dtype
            ),
            "Int64",
        )
        self.assertEqual(
            str(
                dataset[
                    "supervised_usable"
                ].dtype
            ),
            "boolean",
        )
        self.assertTrue(
            str(
                dataset[
                    "target_class"
                ].dtype
            ).startswith(
                "string"
            )
        )
        self.assertEqual(
            str(
                dataset[
                    "prediction_date"
                ].dtype
            ),
            "datetime64[us]",
        )

    def test_content_fingerprint_is_stable_after_row_reordering(
        self,
    ) -> None:
        pa = pd.DataFrame(
            [
                make_pa(
                    game_pk="G1"
                ),
                make_pa(
                    game_pk="G2"
                ),
            ]
        )
        raw = pd.DataFrame(
            [
                make_raw_row(
                    game_pk="G1"
                ),
                make_raw_row(
                    game_pk="G2"
                ),
            ]
        )

        first, _, _ = build_m2_dataset(
            pa,
            raw,
            empty_batting_source(),
            empty_pitching_source(),
            config=load_config(),
        )
        second, _, _ = build_m2_dataset(
            pa.iloc[
                ::-1
            ].reset_index(
                drop=True
            ),
            raw.iloc[
                ::-1
            ].reset_index(
                drop=True
            ),
            empty_batting_source(),
            empty_pitching_source(),
            config=load_config(),
        )

        first_fingerprint = content_fingerprint(
            first,
            sort_columns=list(
                PA_KEY
            ),
        )
        second_fingerprint = content_fingerprint(
            second,
            sort_columns=list(
                PA_KEY
            ),
        )

        self.assertEqual(
            first_fingerprint,
            second_fingerprint,
        )

    @unittest.skipUnless(
        PYARROW_AVAILABLE,
        "pyarrow가 설치된 환경에서만 Parquet round-trip을 검증합니다.",
    )
    def test_parquet_round_trip_preserves_dataset(
        self,
    ) -> None:
        dataset, _, _ = (
            build_single_fixture()
        )

        with TemporaryDirectory() as directory:
            path = (
                Path(
                    directory
                )
                / "m2.parquet"
            )

            dataset.to_parquet(
                path,
                engine="pyarrow",
                index=False,
            )
            raw_round_trip = pd.read_parquet(
                path,
                engine="pyarrow",
            )
            round_trip = _coerce_round_trip_dtypes(
                raw_round_trip,
                dataset,
            )

        assert_frame_equal(
            dataset,
            round_trip,
        )

    def test_input_hash_guard_detects_mutation(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            path = (
                Path(
                    directory
                )
                / "input.txt"
            )

            path.write_text(
                "before",
                encoding="utf-8",
            )

            paths = {
                "input": path
            }

            before = calculate_input_hashes(
                paths
            )

            path.write_text(
                "after",
                encoding="utf-8",
            )

            with self.assertRaises(
                ProcessedContractError
            ):
                assert_input_hashes_unchanged(
                    before,
                    paths,
                )

    def test_output_path_collision_with_input_fails(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            path = (
                Path(
                    directory
                )
                / "artifact.parquet"
            )

            with self.assertRaises(
                ProcessedContractError
            ):
                validate_output_paths(
                    input_paths=[
                        path
                    ],
                    output_paths=[
                        path
                    ],
                    project_root=PROJECT_ROOT,
                )

    def test_deterministic_identity_is_stable(
        self,
    ) -> None:
        dataset, roles, summary = (
            build_single_fixture()
        )

        dataset_fingerprint = (
            content_fingerprint(
                dataset,
                sort_columns=list(
                    PA_KEY
                ),
            )
        )

        schema = build_schema_payload(
            dataset,
            roles,
        )
        quality = build_quality_report(
            dataset,
            summary,
        )

        first = build_deterministic_identity(
            raw_revision="fixture-revision",
            input_hashes={
                "fixture": "0" * 64
            },
            dataset_content_fingerprint=dataset_fingerprint,
            schema_payload=schema,
            quality_payload=quality,
        )
        second = build_deterministic_identity(
            raw_revision="fixture-revision",
            input_hashes={
                "fixture": "0" * 64
            },
            dataset_content_fingerprint=dataset_fingerprint,
            schema_payload=schema,
            quality_payload=quality,
        )

        self.assertEqual(
            first,
            second,
        )

    def test_player_feature_fingerprint_changes_deterministic_identity(
        self,
    ) -> None:
        dataset, roles, summary = (
            build_single_fixture()
        )

        dataset_fingerprint = (
            content_fingerprint(
                dataset,
                sort_columns=list(
                    PA_KEY
                ),
            )
        )

        schema = build_schema_payload(
            dataset,
            roles,
        )
        first_quality = build_quality_report(
            dataset,
            summary,
        )

        changed_summary = dict(
            summary
        )
        changed_provenance = dict(
            summary[
                "player_feature_provenance"
            ]
        )
        changed_provenance[
            "content_fingerprint"
        ] = "f" * 64
        changed_summary[
            "player_feature_provenance"
        ] = changed_provenance

        second_quality = build_quality_report(
            dataset,
            changed_summary,
        )

        first = build_deterministic_identity(
            raw_revision="fixture-revision",
            input_hashes={
                "fixture": "0" * 64
            },
            dataset_content_fingerprint=dataset_fingerprint,
            schema_payload=schema,
            quality_payload=first_quality,
        )
        second = build_deterministic_identity(
            raw_revision="fixture-revision",
            input_hashes={
                "fixture": "0" * 64
            },
            dataset_content_fingerprint=dataset_fingerprint,
            schema_payload=schema,
            quality_payload=second_quality,
        )

        self.assertNotEqual(
            first[
                "fingerprint"
            ],
            second[
                "fingerprint"
            ],
        )


if __name__ == "__main__":
    unittest.main()
