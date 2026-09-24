from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from scripts.build_players import (  # noqa: E402
    OUTPUT_COLUMNS,
    PlayersBuildError,
    build_players,
    build_players_file,
    ensure_output_path_safe,
)


PA_COLUMNS = (
    "game_pk",
    "game_date",
    "season",
    "batter",
    "batter_name",
    "pitcher",
    "pitcher_name",
)

BATTING_COLUMNS = (
    "game_pk",
    "game_date",
    "season",
    "batter",
    "batter_name",
)

PITCHING_COLUMNS = (
    "game_pk",
    "game_date",
    "season",
    "pitcher",
    "pitcher_name",
)


def sha256(
    path: Path,
) -> str:
    """테스트용 파일 SHA256을 계산한다."""
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:
        for chunk in iter(
            lambda: file.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


def empty_pa() -> pd.DataFrame:
    """필수 Schema만 가진 빈 PA DataFrame을 만든다."""
    return pd.DataFrame(
        columns=PA_COLUMNS
    )


def empty_batting() -> pd.DataFrame:
    """필수 Schema만 가진 빈 Batting DataFrame을 만든다."""
    return pd.DataFrame(
        columns=BATTING_COLUMNS
    )


def empty_pitching() -> pd.DataFrame:
    """필수 Schema만 가진 빈 Pitching DataFrame을 만든다."""
    return pd.DataFrame(
        columns=PITCHING_COLUMNS
    )


def pa_row(
    *,
    game_pk: str = "20230401AABB02023",
    game_date: str = "2023-04-01",
    season: int = 2023,
    batter: object = "001",
    batter_name: object = "타자1",
    pitcher: object = "101",
    pitcher_name: object = "투수1",
) -> dict[str, object]:
    """테스트용 최소 Canonical PA Row를 만든다."""
    return {
        "game_pk": game_pk,
        "game_date": game_date,
        "season": season,
        "batter": batter,
        "batter_name": batter_name,
        "pitcher": pitcher,
        "pitcher_name": pitcher_name,
    }


def batting_row(
    *,
    game_pk: str = "20230401AABB02023",
    game_date: str = "2023-04-01",
    season: int = 2023,
    batter: object = "001",
    batter_name: object = "타자1",
) -> dict[str, object]:
    """테스트용 최소 Player Game Batting Row를 만든다."""
    return {
        "game_pk": game_pk,
        "game_date": game_date,
        "season": season,
        "batter": batter,
        "batter_name": batter_name,
    }


def pitching_row(
    *,
    game_pk: str = "20230401AABB02023",
    game_date: str = "2023-04-01",
    season: int = 2023,
    pitcher: object = "101",
    pitcher_name: object = "투수1",
) -> dict[str, object]:
    """테스트용 최소 Player Game Pitching Row를 만든다."""
    return {
        "game_pk": game_pk,
        "game_date": game_date,
        "season": season,
        "pitcher": pitcher,
        "pitcher_name": pitcher_name,
    }


def build(
    *,
    pa_rows: list[dict[str, object]] | None = None,
    batting_rows: list[dict[str, object]] | None = None,
    pitching_rows: list[dict[str, object]] | None = None,
) -> pd.DataFrame:
    """Synthetic Row 목록으로 Player Metadata를 생성한다."""
    pa = (
        pd.DataFrame(
            pa_rows
        )
        if pa_rows
        else empty_pa()
    )
    batting = (
        pd.DataFrame(
            batting_rows
        )
        if batting_rows
        else empty_batting()
    )
    pitching = (
        pd.DataFrame(
            pitching_rows
        )
        if pitching_rows
        else empty_pitching()
    )

    return build_players(
        pa,
        batting,
        pitching,
    )


class PlayerMetadataBuildTest(
    unittest.TestCase
):
    """Issue #12 Player Metadata 계약을 검증한다."""

    def _write_fixture(
        self,
        root: Path,
    ) -> tuple[
        Path,
        Path,
        Path,
        Path,
    ]:
        """File I/O 테스트용 Canonical Input 세 개를 생성한다."""
        pa_path = (
            root
            / "plate_appearances.parquet"
        )
        batting_path = (
            root
            / "player_game_batting.parquet"
        )
        pitching_path = (
            root
            / "player_game_pitching.parquet"
        )
        output_path = (
            root
            / "players.parquet"
        )

        pd.DataFrame(
            [
                pa_row()
            ]
        ).to_parquet(
            pa_path,
            index=False,
            engine="pyarrow",
        )

        pd.DataFrame(
            [
                batting_row()
            ]
        ).to_parquet(
            batting_path,
            index=False,
            engine="pyarrow",
        )

        pd.DataFrame(
            [
                pitching_row()
            ]
        ).to_parquet(
            pitching_path,
            index=False,
            engine="pyarrow",
        )

        return (
            pa_path,
            batting_path,
            pitching_path,
            output_path,
        )

    def test_01_batter_only_player(
        self,
    ) -> None:
        result = build(
            batting_rows=[
                batting_row(
                    batter="001"
                )
            ]
        )

        row = result.iloc[0]

        self.assertTrue(
            bool(
                row["is_batter"]
            )
        )
        self.assertFalse(
            bool(
                row["is_pitcher"]
            )
        )

    def test_02_pitcher_only_player(
        self,
    ) -> None:
        result = build(
            pitching_rows=[
                pitching_row(
                    pitcher="101"
                )
            ]
        )

        row = result.iloc[0]

        self.assertFalse(
            bool(
                row["is_batter"]
            )
        )
        self.assertTrue(
            bool(
                row["is_pitcher"]
            )
        )

    def test_03_batter_and_pitcher_player(
        self,
    ) -> None:
        result = build(
            batting_rows=[
                batting_row(
                    batter="001"
                )
            ],
            pitching_rows=[
                pitching_row(
                    pitcher="001",
                    pitcher_name="타자1",
                )
            ],
        )

        row = result.iloc[0]

        self.assertTrue(
            bool(
                row["is_batter"]
            )
        )
        self.assertTrue(
            bool(
                row["is_pitcher"]
            )
        )

    def test_04_same_player_across_pa_and_player_game_is_one_row(
        self,
    ) -> None:
        result = build(
            pa_rows=[
                pa_row(
                    batter="001",
                    batter_name="타자1",
                )
            ],
            batting_rows=[
                batting_row(
                    batter="001",
                    batter_name="타자1",
                )
            ],
        )

        matches = result.loc[
            result["player_id"].eq(
                "001"
            )
        ]

        self.assertEqual(
            len(matches),
            1,
        )

    def test_05_player_id_preserves_string_and_leading_zero(
        self,
    ) -> None:
        result = build(
            batting_rows=[
                batting_row(
                    batter="001"
                )
            ]
        )

        self.assertEqual(
            result.loc[
                0,
                "player_id",
            ],
            "001",
        )
        self.assertEqual(
            str(
                result["player_id"]
                .dtype
            ),
            "string",
        )

    def test_06_rejects_null_player_id(
        self,
    ) -> None:
        with self.assertRaises(
            PlayersBuildError
        ):
            build(
                batting_rows=[
                    batting_row(
                        batter=None
                    )
                ]
            )

    def test_07_rejects_empty_player_id(
        self,
    ) -> None:
        with self.assertRaises(
            PlayersBuildError
        ):
            build(
                batting_rows=[
                    batting_row(
                        batter=""
                    )
                ]
            )

    def test_08_rejects_whitespace_only_player_id(
        self,
    ) -> None:
        with self.assertRaises(
            PlayersBuildError
        ):
            build(
                batting_rows=[
                    batting_row(
                        batter="   "
                    )
                ]
            )

    def test_09_null_name_keeps_player(
        self,
    ) -> None:
        result = build(
            batting_rows=[
                batting_row(
                    batter="001",
                    batter_name=None,
                )
            ]
        )

        self.assertEqual(
            len(result),
            1,
        )
        self.assertTrue(
            pd.isna(
                result.loc[
                    0,
                    "display_name",
                ]
            )
        )

    def test_10_empty_name_is_excluded_from_variants(
        self,
    ) -> None:
        result = build(
            batting_rows=[
                batting_row(
                    batter_name="   "
                )
            ]
        )

        self.assertEqual(
            json.loads(
                result.loc[
                    0,
                    "name_variants",
                ]
            ),
            [],
        )

    def test_11_single_name_variant(
        self,
    ) -> None:
        result = build(
            batting_rows=[
                batting_row(
                    batter_name="김민수"
                )
            ]
        )

        self.assertEqual(
            json.loads(
                result.loc[
                    0,
                    "name_variants",
                ]
            ),
            ["김민수"],
        )

    def test_12_multiple_name_variants_are_preserved(
        self,
    ) -> None:
        result = build(
            batting_rows=[
                batting_row(
                    game_pk="G1",
                    game_date="2023-04-01",
                    batter_name="김민수",
                ),
                batting_row(
                    game_pk="G2",
                    game_date="2023-05-01",
                    batter_name="김민수(개명전)",
                ),
            ]
        )

        self.assertEqual(
            set(
                json.loads(
                    result.loc[
                        0,
                        "name_variants",
                    ]
                )
            ),
            {
                "김민수",
                "김민수(개명전)",
            },
        )

    def test_13_name_variants_are_deterministically_sorted(
        self,
    ) -> None:
        result = build(
            batting_rows=[
                batting_row(
                    game_pk="G2",
                    game_date="2023-05-01",
                    batter_name="나",
                ),
                batting_row(
                    game_pk="G1",
                    game_date="2023-04-01",
                    batter_name="가",
                ),
            ]
        )

        self.assertEqual(
            json.loads(
                result.loc[
                    0,
                    "name_variants",
                ]
            ),
            [
                "가",
                "나",
            ],
        )

    def test_14_name_variants_is_valid_utf8_json_array(
        self,
    ) -> None:
        result = build(
            batting_rows=[
                batting_row(
                    batter_name="홍길동"
                )
            ]
        )

        value = result.loc[
            0,
            "name_variants",
        ]

        encoded = value.encode(
            "utf-8"
        )
        decoded = encoded.decode(
            "utf-8"
        )

        parsed = json.loads(
            decoded
        )

        self.assertIsInstance(
            parsed,
            list,
        )
        self.assertEqual(
            parsed,
            ["홍길동"],
        )

    def test_15_name_trims_leading_and_trailing_whitespace(
        self,
    ) -> None:
        result = build(
            batting_rows=[
                batting_row(
                    batter_name="  김민수  "
                )
            ]
        )

        self.assertEqual(
            result.loc[
                0,
                "display_name",
            ],
            "김민수",
        )

    def test_16_name_internal_whitespace_is_not_corrected(
        self,
    ) -> None:
        result = build(
            batting_rows=[
                batting_row(
                    batter_name="김  민수"
                )
            ]
        )

        self.assertEqual(
            result.loc[
                0,
                "display_name",
            ],
            "김  민수",
        )

    def test_17_display_name_uses_most_recent_valid_name(
        self,
    ) -> None:
        result = build(
            batting_rows=[
                batting_row(
                    game_pk="G1",
                    game_date="2023-04-01",
                    batter_name="이전이름",
                ),
                batting_row(
                    game_pk="G2",
                    game_date="2023-05-01",
                    batter_name="최근이름",
                ),
            ]
        )

        self.assertEqual(
            result.loc[
                0,
                "display_name",
            ],
            "최근이름",
        )

    def test_18_display_name_uses_lexical_tie_break_on_latest_date(
        self,
    ) -> None:
        result = build(
            batting_rows=[
                batting_row(
                    game_pk="G1",
                    game_date="2023-05-01",
                    batter_name="나",
                ),
                batting_row(
                    game_pk="G2",
                    game_date="2023-05-01",
                    batter_name="가",
                ),
            ]
        )

        self.assertEqual(
            result.loc[
                0,
                "display_name",
            ],
            "가",
        )

    def test_19_no_valid_name_means_null_display_and_empty_variants(
        self,
    ) -> None:
        result = build(
            batting_rows=[
                batting_row(
                    game_pk="G1",
                    batter_name=None,
                ),
                batting_row(
                    game_pk="G2",
                    game_date="2023-04-02",
                    batter_name=" ",
                ),
            ]
        )

        self.assertTrue(
            pd.isna(
                result.loc[
                    0,
                    "display_name",
                ]
            )
        )
        self.assertEqual(
            result.loc[
                0,
                "name_variants",
            ],
            "[]",
        )

    def test_20_first_seen_date(
        self,
    ) -> None:
        result = build(
            batting_rows=[
                batting_row(
                    game_pk="G2",
                    game_date="2023-05-01",
                ),
                batting_row(
                    game_pk="G1",
                    game_date="2023-04-01",
                ),
            ]
        )

        self.assertEqual(
            result.loc[
                0,
                "first_seen_date",
            ],
            pd.Timestamp(
                "2023-04-01"
            ),
        )

    def test_21_last_seen_date(
        self,
    ) -> None:
        result = build(
            batting_rows=[
                batting_row(
                    game_pk="G1",
                    game_date="2023-04-01",
                ),
                batting_row(
                    game_pk="G2",
                    game_date="2023-05-01",
                ),
            ]
        )

        self.assertEqual(
            result.loc[
                0,
                "last_seen_date",
            ],
            pd.Timestamp(
                "2023-05-01"
            ),
        )

    def test_22_first_seen_season_is_linked_to_first_date(
        self,
    ) -> None:
        result = build(
            batting_rows=[
                batting_row(
                    game_pk="G24",
                    game_date="2024-04-01",
                    season=2024,
                ),
                batting_row(
                    game_pk="G23",
                    game_date="2023-04-01",
                    season=2023,
                ),
            ]
        )

        self.assertEqual(
            int(
                result.loc[
                    0,
                    "first_seen_season",
                ]
            ),
            2023,
        )

    def test_23_last_seen_season_is_linked_to_last_date(
        self,
    ) -> None:
        result = build(
            batting_rows=[
                batting_row(
                    game_pk="G23",
                    game_date="2023-04-01",
                    season=2023,
                ),
                batting_row(
                    game_pk="G24",
                    game_date="2024-04-01",
                    season=2024,
                ),
            ]
        )

        self.assertEqual(
            int(
                result.loc[
                    0,
                    "last_seen_season",
                ]
            ),
            2024,
        )

    def test_24_rejects_boundary_date_season_conflict(
        self,
    ) -> None:
        with self.assertRaises(
            PlayersBuildError
        ):
            build(
                batting_rows=[
                    batting_row(
                        game_pk="G1",
                        game_date="2023-12-31",
                        season=2023,
                    ),
                    batting_row(
                        game_pk="G2",
                        game_date="2023-12-31",
                        season=2024,
                    ),
                ]
            )

    def test_25_rejects_source_game_pk_context_conflict(
        self,
    ) -> None:
        with self.assertRaises(
            PlayersBuildError
        ):
            build(
                batting_rows=[
                    batting_row(
                        game_pk="SAME",
                        game_date="2023-04-01",
                        batter="001",
                    ),
                    batting_row(
                        game_pk="SAME",
                        game_date="2023-04-02",
                        batter="002",
                    ),
                ]
            )

    def test_26_rejects_cross_source_game_context_conflict(
        self,
    ) -> None:
        with self.assertRaises(
            PlayersBuildError
        ):
            build(
                batting_rows=[
                    batting_row(
                        game_pk="SAME",
                        game_date="2023-04-01",
                        season=2023,
                    )
                ],
                pitching_rows=[
                    pitching_row(
                        game_pk="SAME",
                        game_date="2024-04-01",
                        season=2024,
                    )
                ],
            )

    def test_27_all_batting_players_exist_in_players(
        self,
    ) -> None:
        batting = [
            batting_row(
                game_pk="G1",
                batter="001",
            ),
            batting_row(
                game_pk="G2",
                game_date="2023-04-02",
                batter="002",
            ),
        ]

        result = build(
            batting_rows=batting
        )

        self.assertTrue(
            {
                "001",
                "002",
            }.issubset(
                set(
                    result["player_id"]
                )
            )
        )

    def test_28_all_pitching_players_exist_in_players(
        self,
    ) -> None:
        pitching = [
            pitching_row(
                game_pk="G1",
                pitcher="101",
            ),
            pitching_row(
                game_pk="G2",
                game_date="2023-04-02",
                pitcher="102",
            ),
        ]

        result = build(
            pitching_rows=pitching
        )

        self.assertTrue(
            {
                "101",
                "102",
            }.issubset(
                set(
                    result["player_id"]
                )
            )
        )

    def test_29_pa_only_player_exists(
        self,
    ) -> None:
        result = build(
            pa_rows=[
                pa_row(
                    batter="PA_ONLY",
                    batter_name="PA전용",
                    pitcher="P1",
                )
            ]
        )

        self.assertIn(
            "PA_ONLY",
            set(
                result["player_id"]
            ),
        )

    def test_30_pitching_only_source_player_exists(
        self,
    ) -> None:
        result = build(
            pitching_rows=[
                pitching_row(
                    pitcher="MID_RELIEF",
                    pitcher_name="중간투수",
                )
            ]
        )

        self.assertIn(
            "MID_RELIEF",
            set(
                result["player_id"]
            ),
        )

    def test_31_player_id_is_unique(
        self,
    ) -> None:
        result = build(
            pa_rows=[
                pa_row(
                    batter="001",
                    pitcher="001",
                    pitcher_name="타자1",
                )
            ],
            batting_rows=[
                batting_row(
                    batter="001"
                )
            ],
            pitching_rows=[
                pitching_row(
                    pitcher="001",
                    pitcher_name="타자1",
                )
            ],
        )

        self.assertEqual(
            int(
                result.duplicated(
                    subset=["player_id"]
                ).sum()
            ),
            0,
        )

    def test_32_team_columns_are_absent(
        self,
    ) -> None:
        result = build(
            batting_rows=[
                batting_row()
            ]
        )

        prohibited = {
            "team",
            "current_team",
            "latest_team",
            "first_team",
            "last_team",
        }

        self.assertTrue(
            prohibited.isdisjoint(
                result.columns
            )
        )

    def test_33_input_row_shuffle_is_deterministic(
        self,
    ) -> None:
        batting = pd.DataFrame(
            [
                batting_row(
                    game_pk="G3",
                    game_date="2023-06-01",
                    batter="003",
                    batter_name="다",
                ),
                batting_row(
                    game_pk="G1",
                    game_date="2023-04-01",
                    batter="001",
                    batter_name="가",
                ),
                batting_row(
                    game_pk="G2",
                    game_date="2023-05-01",
                    batter="002",
                    batter_name="나",
                ),
            ]
        )

        result_a = build_players(
            empty_pa(),
            batting,
            empty_pitching(),
        )

        result_b = build_players(
            empty_pa(),
            batting.sample(
                frac=1,
                random_state=42,
            ).reset_index(
                drop=True
            ),
            empty_pitching(),
        )

        pd.testing.assert_frame_equal(
            result_a,
            result_b,
            check_dtype=True,
        )

    def test_34_multi_season_determinism(
        self,
    ) -> None:
        rows = [
            batting_row(
                game_pk="G24",
                game_date="2024-04-01",
                season=2024,
                batter_name="최근",
            ),
            batting_row(
                game_pk="G23",
                game_date="2023-04-01",
                season=2023,
                batter_name="이전",
            ),
        ]

        result_a = build(
            batting_rows=rows
        )
        result_b = build(
            batting_rows=list(
                reversed(
                    rows
                )
            )
        )

        pd.testing.assert_frame_equal(
            result_a,
            result_b,
            check_dtype=True,
        )

    def test_35_output_is_sorted_by_player_id(
        self,
    ) -> None:
        result = build(
            batting_rows=[
                batting_row(
                    game_pk="G3",
                    batter="003",
                ),
                batting_row(
                    game_pk="G1",
                    batter="001",
                ),
                batting_row(
                    game_pk="G2",
                    batter="002",
                ),
            ]
        )

        self.assertEqual(
            result["player_id"]
            .tolist(),
            [
                "001",
                "002",
                "003",
            ],
        )

    def test_36_parquet_round_trip_preserves_dtypes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(
                temp_dir
            )

            (
                pa_path,
                batting_path,
                pitching_path,
                output_path,
            ) = self._write_fixture(
                root
            )

            build_players_file(
                pa_path=pa_path,
                batting_path=batting_path,
                pitching_path=pitching_path,
                output_path=output_path,
            )

            round_trip = pd.read_parquet(
                output_path,
                engine="pyarrow",
            )

            expected = {
                "player_id": "string",
                "display_name": "string",
                "name_variants": "string",
                "is_batter": "boolean",
                "is_pitcher": "boolean",
                "first_seen_date": "datetime64[us]",
                "last_seen_date": "datetime64[us]",
                "first_seen_season": "Int64",
                "last_seen_season": "Int64",
            }

            for column, dtype in expected.items():
                self.assertEqual(
                    str(
                        round_trip[
                            column
                        ].dtype
                    ),
                    dtype,
                )

    def test_37_json_round_trip_is_parseable(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(
                temp_dir
            )

            (
                pa_path,
                batting_path,
                pitching_path,
                output_path,
            ) = self._write_fixture(
                root
            )

            build_players_file(
                pa_path=pa_path,
                batting_path=batting_path,
                pitching_path=pitching_path,
                output_path=output_path,
            )

            round_trip = pd.read_parquet(
                output_path,
                engine="pyarrow",
            )

            for value in round_trip[
                "name_variants"
            ]:
                parsed = json.loads(
                    value
                )

                self.assertIsInstance(
                    parsed,
                    list,
                )

    def test_38_repeated_build_is_dataframe_deterministic(
        self,
    ) -> None:
        pa = pd.DataFrame(
            [
                pa_row(
                    batter="002",
                    pitcher="101",
                ),
                pa_row(
                    game_pk="G2",
                    game_date="2023-04-02",
                    batter="001",
                    pitcher="102",
                ),
            ]
        )

        batting = pd.DataFrame(
            [
                batting_row(
                    batter="002"
                )
            ]
        )

        pitching = pd.DataFrame(
            [
                pitching_row(
                    pitcher="101"
                )
            ]
        )

        first = build_players(
            pa,
            batting,
            pitching,
        )
        second = build_players(
            pa,
            batting,
            pitching,
        )

        pd.testing.assert_frame_equal(
            first,
            second,
            check_dtype=True,
        )

    def test_39_pa_input_is_immutable(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(
                temp_dir
            )

            (
                pa_path,
                batting_path,
                pitching_path,
                output_path,
            ) = self._write_fixture(
                root
            )

            before = sha256(
                pa_path
            )

            build_players_file(
                pa_path=pa_path,
                batting_path=batting_path,
                pitching_path=pitching_path,
                output_path=output_path,
            )

            after = sha256(
                pa_path
            )

            self.assertEqual(
                before,
                after,
            )

    def test_40_batting_input_is_immutable(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(
                temp_dir
            )

            (
                pa_path,
                batting_path,
                pitching_path,
                output_path,
            ) = self._write_fixture(
                root
            )

            before = sha256(
                batting_path
            )

            build_players_file(
                pa_path=pa_path,
                batting_path=batting_path,
                pitching_path=pitching_path,
                output_path=output_path,
            )

            after = sha256(
                batting_path
            )

            self.assertEqual(
                before,
                after,
            )

    def test_41_pitching_input_is_immutable(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(
                temp_dir
            )

            (
                pa_path,
                batting_path,
                pitching_path,
                output_path,
            ) = self._write_fixture(
                root
            )

            before = sha256(
                pitching_path
            )

            build_players_file(
                pa_path=pa_path,
                batting_path=batting_path,
                pitching_path=pitching_path,
                output_path=output_path,
            )

            after = sha256(
                pitching_path
            )

            self.assertEqual(
                before,
                after,
            )

    def test_42_rejects_output_under_data_raw(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(
                temp_dir
            )

            pa_path = (
                root
                / "pa.parquet"
            )
            batting_path = (
                root
                / "batting.parquet"
            )
            pitching_path = (
                root
                / "pitching.parquet"
            )

            raw_output = (
                PROJECT_ROOT
                / "data"
                / "raw"
                / "players.parquet"
            )

            with self.assertRaises(
                PlayersBuildError
            ):
                ensure_output_path_safe(
                    pa_path=pa_path,
                    batting_path=batting_path,
                    pitching_path=pitching_path,
                    output_path=raw_output,
                )

    def test_43_rejects_pa_input_overwrite(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(
                temp_dir
            )

            pa_path = (
                root
                / "plate_appearances.parquet"
            )

            with self.assertRaises(
                PlayersBuildError
            ):
                ensure_output_path_safe(
                    pa_path=pa_path,
                    batting_path=root
                    / "batting.parquet",
                    pitching_path=root
                    / "pitching.parquet",
                    output_path=pa_path,
                )

    def test_44_rejects_batting_input_overwrite(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(
                temp_dir
            )

            batting_path = (
                root
                / "player_game_batting.parquet"
            )

            with self.assertRaises(
                PlayersBuildError
            ):
                ensure_output_path_safe(
                    pa_path=root
                    / "pa.parquet",
                    batting_path=batting_path,
                    pitching_path=root
                    / "pitching.parquet",
                    output_path=batting_path,
                )

    def test_45_rejects_pitching_input_overwrite(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(
                temp_dir
            )

            pitching_path = (
                root
                / "player_game_pitching.parquet"
            )

            with self.assertRaises(
                PlayersBuildError
            ):
                ensure_output_path_safe(
                    pa_path=root
                    / "pa.parquet",
                    batting_path=root
                    / "batting.parquet",
                    pitching_path=pitching_path,
                    output_path=pitching_path,
                )


if __name__ == "__main__":
    unittest.main()