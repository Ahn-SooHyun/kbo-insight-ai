from __future__ import annotations

import hashlib
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


from scripts.build_season_snapshots import (  # noqa: E402
    BATTING_OUTPUT_COLUMNS,
    BATTING_SNAPSHOT_FILENAME,
    PITCHING_OUTPUT_COLUMNS,
    PITCHING_SNAPSHOT_FILENAME,
    RAW_DATA_DIR,
    TEAM_OUTPUT_COLUMNS,
    TEAM_SNAPSHOT_FILENAME,
    SeasonSnapshotsBuildError,
    build_season_snapshots,
    build_season_snapshots_file,
    ensure_output_paths_safe,
)


def sha256(
    path: Path,
) -> str:
    """테스트용 SHA256을 계산한다."""
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


def batting_row(
    *,
    game_pk: str = "20230401AABB02023",
    game_date: str = "2023-04-01",
    season: int = 2023,
    batter: str = "001",
    team: str = "AA",
    **changes: object,
) -> dict[str, object]:
    """테스트용 Player Game Batting Row를 만든다."""
    row: dict[str, object] = {
        "game_pk": game_pk,
        "game_date": game_date,
        "season": season,
        "batter": batter,
        "team": team,
        "pa": 1,
        "ab": 1,
        "h": 1,
        "single": 1,
        "double": 0,
        "triple": 0,
        "hr": 0,
        "bb": 0,
        "hbp": 0,
        "so": 0,
        "sf": 0,
        "sh": 0,
        "tb": 1,
        "double_play": 0,
        "triple_play": 0,
        "field_error": 0,
        "fielders_choice": 0,
        "catcher_interference": 0,
        # Season Builder는 경기 Rate를 입력으로 사용하지 않아야 한다.
        "avg": 1.0,
        "obp": 1.0,
        "slg": 1.0,
        "ops": 2.0,
    }

    row.update(
        changes
    )

    return row


def pitching_row(
    *,
    game_pk: str = "20230401AABB02023",
    game_date: str = "2023-04-01",
    season: int = 2023,
    pitcher: str = "101",
    team: str = "BB",
    **changes: object,
) -> dict[str, object]:
    """테스트용 Player Game Pitching Row를 만든다."""
    row: dict[str, object] = {
        "game_pk": game_pk,
        "game_date": game_date,
        "season": season,
        "pitcher": pitcher,
        "team": team,
        "pitch_rows": 3,
        "pitches": 3,
        "batters_faced_completed": 1,
        "hits_allowed": 0,
        "single_allowed": 0,
        "double_allowed": 0,
        "triple_allowed": 0,
        "hr_allowed": 0,
        "bb_allowed": 0,
        "hbp_allowed": 0,
        "so": 1,
        "sf": 0,
        "sh": 0,
        "outs_recorded": 1,
        "ball_pitch_count": 1,
        "strike_pitch_count": 2,
        "in_play_pitch_count": 0,
        "avg_release_speed_kmh": 145.0,
    }

    row.update(
        changes
    )

    return row


def team_game_rows(
    *,
    game_pk: str = "20230401AABB02023",
    game_date: str = "2023-04-01",
    season: int = 2023,
    away_team: str = "AA",
    home_team: str = "BB",
    away_runs: int = 1,
    home_runs: int = 2,
) -> list[dict[str, object]]:
    """한 경기를 나타내는 Team Game 2행을 만든다."""
    if home_runs > away_runs:
        home_win = True
        away_win = False
        tie = False
    elif away_runs > home_runs:
        home_win = False
        away_win = True
        tie = False
    else:
        home_win = False
        away_win = False
        tie = True

    return [
        {
            "game_pk": game_pk,
            "game_date": game_date,
            "season": season,
            "team": away_team,
            "opponent": home_team,
            "is_home": False,
            "runs_for": away_runs,
            "runs_against": home_runs,
            "run_diff": (
                away_runs
                - home_runs
            ),
            "win": away_win,
            "loss": (
                not away_win
                and not tie
            ),
            "tie": tie,
        },
        {
            "game_pk": game_pk,
            "game_date": game_date,
            "season": season,
            "team": home_team,
            "opponent": away_team,
            "is_home": True,
            "runs_for": home_runs,
            "runs_against": away_runs,
            "run_diff": (
                home_runs
                - away_runs
            ),
            "win": home_win,
            "loss": (
                not home_win
                and not tie
            ),
            "tie": tie,
        },
    ]


def build_frames(
    *,
    batting_rows: list[dict[str, object]] | None = None,
    pitching_rows: list[dict[str, object]] | None = None,
    team_rows: list[dict[str, object]] | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Synthetic Row를 세 Canonical DataFrame으로 변환한다."""
    batting = pd.DataFrame(
        batting_rows
        if batting_rows is not None
        else [
            batting_row()
        ]
    )

    pitching = pd.DataFrame(
        pitching_rows
        if pitching_rows is not None
        else [
            pitching_row()
        ]
    )

    teams = pd.DataFrame(
        team_rows
        if team_rows is not None
        else team_game_rows()
    )

    return (
        batting,
        pitching,
        teams,
    )


def build_default() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """기본 Synthetic Fixture로 세 Snapshot을 생성한다."""
    batting, pitching, teams = (
        build_frames()
    )

    return build_season_snapshots(
        batting,
        pitching,
        teams,
    )


class SeasonSnapshotBuildTest(
    unittest.TestCase
):
    """Issue #13 Season Snapshot 계약을 검증한다."""

    def _write_file_fixture(
        self,
        root: Path,
    ) -> tuple[
        Path,
        Path,
        Path,
        Path,
    ]:
        """File I/O 검증용 Canonical Input을 생성한다."""
        input_dir = (
            root
            / "input"
        )
        output_dir = (
            root
            / "output"
        )

        input_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        batting_path = (
            input_dir
            / "player_game_batting.parquet"
        )
        pitching_path = (
            input_dir
            / "player_game_pitching.parquet"
        )
        team_path = (
            input_dir
            / "team_games.parquet"
        )

        batting, pitching, teams = (
            build_frames()
        )

        batting.to_parquet(
            batting_path,
            engine="pyarrow",
            index=False,
        )

        pitching.to_parquet(
            pitching_path,
            engine="pyarrow",
            index=False,
        )

        teams.to_parquet(
            team_path,
            engine="pyarrow",
            index=False,
        )

        return (
            batting_path,
            pitching_path,
            team_path,
            output_dir,
        )

    # ------------------------------------------------------------------
    # Common / through_date
    # ------------------------------------------------------------------

    def test_01_through_date_is_team_game_season_max_date(
        self,
    ) -> None:
        team_rows = (
            team_game_rows(
                game_pk="20230401AABB02023",
                game_date="2023-04-01",
            )
            + team_game_rows(
                game_pk="20230920CCDD02023",
                game_date="2023-09-20",
                away_team="CC",
                home_team="DD",
            )
        )

        batting_rows = [
            batting_row()
        ]

        pitching_rows = [
            pitching_row()
        ]

        batting, pitching, team = (
            build_frames(
                batting_rows=batting_rows,
                pitching_rows=pitching_rows,
                team_rows=team_rows,
            )
        )

        (
            batting_snapshot,
            _,
            _,
        ) = build_season_snapshots(
            batting,
            pitching,
            team,
        )

        self.assertEqual(
            batting_snapshot.loc[
                0,
                "through_date",
            ],
            pd.Timestamp(
                "2023-09-20"
            ),
        )

    def test_02_same_season_uses_same_through_date_across_all_snapshots(
        self,
    ) -> None:
        (
            batting_snapshot,
            pitching_snapshot,
            team_snapshot,
        ) = build_default()

        expected = pd.Timestamp(
            "2023-04-01"
        )

        self.assertTrue(
            batting_snapshot[
                "through_date"
            ]
            .eq(expected)
            .all()
        )

        self.assertTrue(
            pitching_snapshot[
                "through_date"
            ]
            .eq(expected)
            .all()
        )

        self.assertTrue(
            team_snapshot[
                "through_date"
            ]
            .eq(expected)
            .all()
        )

    def test_03_rejects_player_game_later_than_through_date(
        self,
    ) -> None:
        batting, pitching, teams = (
            build_frames(
                batting_rows=[
                    batting_row(
                        game_date="2023-04-02",
                    )
                ],
            )
        )

        with self.assertRaises(
            SeasonSnapshotsBuildError
        ):
            build_season_snapshots(
                batting,
                pitching,
                teams,
            )

    def test_04_rejects_player_season_missing_from_team_games(
        self,
    ) -> None:
        batting, pitching, teams = (
            build_frames(
                batting_rows=[
                    batting_row(
                        season=2024,
                    )
                ],
            )
        )

        with self.assertRaises(
            SeasonSnapshotsBuildError
        ):
            build_season_snapshots(
                batting,
                pitching,
                teams,
            )

    def test_05_rejects_input_grain_duplicates(
        self,
    ) -> None:
        with self.subTest(
            source="batting"
        ):
            batting, pitching, teams = (
                build_frames(
                    batting_rows=[
                        batting_row(),
                        batting_row(),
                    ]
                )
            )

            with self.assertRaises(
                SeasonSnapshotsBuildError
            ):
                build_season_snapshots(
                    batting,
                    pitching,
                    teams,
                )

        with self.subTest(
            source="pitching"
        ):
            batting, pitching, teams = (
                build_frames(
                    pitching_rows=[
                        pitching_row(),
                        pitching_row(),
                    ]
                )
            )

            with self.assertRaises(
                SeasonSnapshotsBuildError
            ):
                build_season_snapshots(
                    batting,
                    pitching,
                    teams,
                )

        with self.subTest(
            source="team"
        ):
            duplicate_team_rows = (
                team_game_rows()
            )

            duplicate_team_rows.append(
                duplicate_team_rows[
                    0
                ].copy()
            )

            batting, pitching, teams = (
                build_frames(
                    team_rows=duplicate_team_rows,
                )
            )

            with self.assertRaises(
                SeasonSnapshotsBuildError
            ):
                build_season_snapshots(
                    batting,
                    pitching,
                    teams,
                )

    def test_06_through_date_dtype_is_datetime64_us(
        self,
    ) -> None:
        (
            batting_snapshot,
            pitching_snapshot,
            team_snapshot,
        ) = build_default()

        for frame in (
            batting_snapshot,
            pitching_snapshot,
            team_snapshot,
        ):
            self.assertEqual(
                str(
                    frame[
                        "through_date"
                    ].dtype
                ),
                "datetime64[us]",
            )

    # ------------------------------------------------------------------
    # Batting
    # ------------------------------------------------------------------

    def test_07_builds_normal_batter_season(
        self,
    ) -> None:
        batting_snapshot, _, _ = (
            build_default()
        )

        self.assertEqual(
            len(
                batting_snapshot
            ),
            1,
        )
        self.assertEqual(
            tuple(
                batting_snapshot.columns
            ),
            BATTING_OUTPUT_COLUMNS,
        )

    def test_08_sums_batting_counts_across_games(
        self,
    ) -> None:
        second_game = (
            "20230402CCDD02023"
        )

        team_rows = (
            team_game_rows()
            + team_game_rows(
                game_pk=second_game,
                game_date="2023-04-02",
                away_team="CC",
                home_team="DD",
            )
        )

        batting_rows = [
            batting_row(),
            batting_row(
                game_pk=second_game,
                game_date="2023-04-02",
                team="CC",
            ),
        ]

        batting, pitching, teams = (
            build_frames(
                batting_rows=batting_rows,
                team_rows=team_rows,
            )
        )

        batting_snapshot, _, _ = (
            build_season_snapshots(
                batting,
                pitching,
                teams,
            )
        )

        row = batting_snapshot.iloc[
            0
        ]

        self.assertEqual(
            int(
                row["pa"]
            ),
            2,
        )
        self.assertEqual(
            int(
                row["h"]
            ),
            2,
        )

    def test_09_separates_batting_seasons(
        self,
    ) -> None:
        game_2024 = (
            "20240401AABB02024"
        )

        team_rows = (
            team_game_rows()
            + team_game_rows(
                game_pk=game_2024,
                game_date="2024-04-01",
                season=2024,
            )
        )

        batting_rows = [
            batting_row(),
            batting_row(
                game_pk=game_2024,
                game_date="2024-04-01",
                season=2024,
            ),
        ]

        pitching_rows = [
            pitching_row(),
        ]

        batting, pitching, teams = (
            build_frames(
                batting_rows=batting_rows,
                pitching_rows=pitching_rows,
                team_rows=team_rows,
            )
        )

        batting_snapshot, _, _ = (
            build_season_snapshots(
                batting,
                pitching,
                teams,
            )
        )

        self.assertEqual(
            set(
                batting_snapshot[
                    "season"
                ].astype(int)
            ),
            {
                2023,
                2024,
            },
        )

    def test_10_separates_multiple_batters_same_season(
        self,
    ) -> None:
        batting, pitching, teams = (
            build_frames(
                batting_rows=[
                    batting_row(
                        batter="001"
                    ),
                    batting_row(
                        batter="002"
                    ),
                ]
            )
        )

        # 같은 game_pk에 다른 batter는 Source Grain상 허용된다.
        batting_snapshot, _, _ = (
            build_season_snapshots(
                batting,
                pitching,
                teams,
            )
        )

        self.assertEqual(
            set(
                batting_snapshot[
                    "batter"
                ]
            ),
            {
                "001",
                "002",
            },
        )

    def test_11_trade_keeps_single_batter_season_without_team_column(
        self,
    ) -> None:
        second_game = (
            "20230402CCDD02023"
        )

        team_rows = (
            team_game_rows()
            + team_game_rows(
                game_pk=second_game,
                game_date="2023-04-02",
                away_team="CC",
                home_team="DD",
            )
        )

        batting_rows = [
            batting_row(
                team="AA"
            ),
            batting_row(
                game_pk=second_game,
                game_date="2023-04-02",
                team="CC",
            ),
        ]

        batting, pitching, teams = (
            build_frames(
                batting_rows=batting_rows,
                team_rows=team_rows,
            )
        )

        batting_snapshot, _, _ = (
            build_season_snapshots(
                batting,
                pitching,
                teams,
            )
        )

        self.assertEqual(
            len(
                batting_snapshot
            ),
            1,
        )
        self.assertNotIn(
            "team",
            batting_snapshot.columns,
        )

    def test_12_recalculates_h(
        self,
    ) -> None:
        batting, pitching, teams = (
            build_frames(
                batting_rows=[
                    batting_row(
                        h=4,
                        single=1,
                        double=1,
                        triple=1,
                        hr=1,
                        pa=4,
                        ab=4,
                        tb=10,
                    )
                ]
            )
        )

        batting_snapshot, _, _ = (
            build_season_snapshots(
                batting,
                pitching,
                teams,
            )
        )

        self.assertEqual(
            int(
                batting_snapshot.loc[
                    0,
                    "h",
                ]
            ),
            4,
        )

    def test_13_recalculates_ab(
        self,
    ) -> None:
        batting, pitching, teams = (
            build_frames(
                batting_rows=[
                    batting_row(
                        pa=6,
                        bb=1,
                        hbp=1,
                        sh=1,
                        sf=1,
                        catcher_interference=1,
                        ab=1,
                        h=1,
                        single=1,
                        tb=1,
                    )
                ]
            )
        )

        batting_snapshot, _, _ = (
            build_season_snapshots(
                batting,
                pitching,
                teams,
            )
        )

        self.assertEqual(
            int(
                batting_snapshot.loc[
                    0,
                    "ab",
                ]
            ),
            1,
        )

    def test_14_recalculates_tb(
        self,
    ) -> None:
        batting, pitching, teams = (
            build_frames(
                batting_rows=[
                    batting_row(
                        pa=4,
                        ab=4,
                        h=4,
                        single=1,
                        double=1,
                        triple=1,
                        hr=1,
                        tb=10,
                    )
                ]
            )
        )

        batting_snapshot, _, _ = (
            build_season_snapshots(
                batting,
                pitching,
                teams,
            )
        )

        self.assertEqual(
            int(
                batting_snapshot.loc[
                    0,
                    "tb",
                ]
            ),
            10,
        )

    def test_15_calculates_avg_from_aggregate_counts(
        self,
    ) -> None:
        batting, pitching, teams = (
            build_frames(
                batting_rows=[
                    batting_row(
                        pa=4,
                        ab=4,
                        h=1,
                        single=1,
                        so=3,
                        tb=1,
                    )
                ]
            )
        )

        batting_snapshot, _, _ = (
            build_season_snapshots(
                batting,
                pitching,
                teams,
            )
        )

        self.assertEqual(
            float(
                batting_snapshot.loc[
                    0,
                    "avg",
                ]
            ),
            0.25,
        )

    def test_16_calculates_obp(
        self,
    ) -> None:
        batting, pitching, teams = (
            build_frames(
                batting_rows=[
                    batting_row(
                        pa=4,
                        ab=2,
                        h=1,
                        single=1,
                        bb=1,
                        hbp=1,
                        tb=1,
                    )
                ]
            )
        )

        batting_snapshot, _, _ = (
            build_season_snapshots(
                batting,
                pitching,
                teams,
            )
        )

        self.assertEqual(
            float(
                batting_snapshot.loc[
                    0,
                    "obp",
                ]
            ),
            0.75,
        )

    def test_17_calculates_slg(
        self,
    ) -> None:
        batting, pitching, teams = (
            build_frames(
                batting_rows=[
                    batting_row(
                        pa=4,
                        ab=4,
                        h=2,
                        single=1,
                        double=1,
                        tb=3,
                    )
                ]
            )
        )

        batting_snapshot, _, _ = (
            build_season_snapshots(
                batting,
                pitching,
                teams,
            )
        )

        self.assertEqual(
            float(
                batting_snapshot.loc[
                    0,
                    "slg",
                ]
            ),
            0.75,
        )

    def test_18_calculates_ops(
        self,
    ) -> None:
        batting, pitching, teams = (
            build_frames(
                batting_rows=[
                    batting_row(
                        pa=4,
                        ab=4,
                        h=2,
                        single=1,
                        double=1,
                        tb=3,
                    )
                ]
            )
        )

        batting_snapshot, _, _ = (
            build_season_snapshots(
                batting,
                pitching,
                teams,
            )
        )

        self.assertEqual(
            float(
                batting_snapshot.loc[
                    0,
                    "ops",
                ]
            ),
            1.25,
        )

    def test_19_does_not_average_game_rates(
        self,
    ) -> None:
        second_game = (
            "20230402CCDD02023"
        )

        team_rows = (
            team_game_rows()
            + team_game_rows(
                game_pk=second_game,
                game_date="2023-04-02",
                away_team="CC",
                home_team="DD",
            )
        )

        batting_rows = [
            batting_row(
                avg=1.0,
            ),
            batting_row(
                game_pk=second_game,
                game_date="2023-04-02",
                team="CC",
                pa=3,
                ab=3,
                h=0,
                single=0,
                so=3,
                tb=0,
                avg=0.0,
                obp=0.0,
                slg=0.0,
                ops=0.0,
            ),
        ]

        batting, pitching, teams = (
            build_frames(
                batting_rows=batting_rows,
                team_rows=team_rows,
            )
        )

        batting_snapshot, _, _ = (
            build_season_snapshots(
                batting,
                pitching,
                teams,
            )
        )

        self.assertEqual(
            float(
                batting_snapshot.loc[
                    0,
                    "avg",
                ]
            ),
            0.25,
        )

        self.assertNotEqual(
            float(
                batting_snapshot.loc[
                    0,
                    "avg",
                ]
            ),
            0.5,
        )

    def test_20_zero_denominator_rates_are_nullable(
        self,
    ) -> None:
        batting, pitching, teams = (
            build_frames(
                batting_rows=[
                    batting_row(
                        pa=1,
                        ab=0,
                        h=0,
                        single=0,
                        bb=0,
                        hbp=0,
                        sf=0,
                        sh=1,
                        tb=0,
                        avg=pd.NA,
                        obp=pd.NA,
                        slg=pd.NA,
                        ops=pd.NA,
                    )
                ]
            )
        )

        batting_snapshot, _, _ = (
            build_season_snapshots(
                batting,
                pitching,
                teams,
            )
        )

        row = batting_snapshot.iloc[
            0
        ]

        self.assertTrue(
            pd.isna(
                row["avg"]
            )
        )
        self.assertTrue(
            pd.isna(
                row["obp"]
            )
        )
        self.assertTrue(
            pd.isna(
                row["slg"]
            )
        )
        self.assertTrue(
            pd.isna(
                row["ops"]
            )
        )

    def test_21_rates_round_to_three_decimals(
        self,
    ) -> None:
        batting, pitching, teams = (
            build_frames(
                batting_rows=[
                    batting_row(
                        pa=3,
                        ab=3,
                        h=1,
                        single=1,
                        so=2,
                        tb=1,
                    )
                ]
            )
        )

        batting_snapshot, _, _ = (
            build_season_snapshots(
                batting,
                pitching,
                teams,
            )
        )

        self.assertEqual(
            float(
                batting_snapshot.loc[
                    0,
                    "avg",
                ]
            ),
            0.333,
        )

    def test_22_batting_source_snapshot_count_reconciliation(
        self,
    ) -> None:
        batting, pitching, teams = (
            build_frames()
        )

        batting_snapshot, _, _ = (
            build_season_snapshots(
                batting,
                pitching,
                teams,
            )
        )

        for column in (
            "pa",
            "h",
            "ab",
            "tb",
        ):
            self.assertEqual(
                int(
                    batting[
                        column
                    ].sum()
                ),
                int(
                    batting_snapshot[
                        column
                    ].sum()
                ),
            )

    def test_23_batter_season_key_is_unique(
        self,
    ) -> None:
        batting_snapshot, _, _ = (
            build_default()
        )

        self.assertEqual(
            int(
                batting_snapshot.duplicated(
                    subset=[
                        "season",
                        "batter",
                    ]
                ).sum()
            ),
            0,
        )

    # ------------------------------------------------------------------
    # Pitching
    # ------------------------------------------------------------------

    def test_24_builds_normal_pitcher_season(
        self,
    ) -> None:
        _, pitching_snapshot, _ = (
            build_default()
        )

        self.assertEqual(
            len(
                pitching_snapshot
            ),
            1,
        )

        self.assertEqual(
            tuple(
                pitching_snapshot.columns
            ),
            PITCHING_OUTPUT_COLUMNS,
        )

    def test_25_sums_pitching_counts_across_games(
        self,
    ) -> None:
        second_game = (
            "20230402CCDD02023"
        )

        team_rows = (
            team_game_rows()
            + team_game_rows(
                game_pk=second_game,
                game_date="2023-04-02",
                away_team="CC",
                home_team="DD",
            )
        )

        pitching_rows = [
            pitching_row(),
            pitching_row(
                game_pk=second_game,
                game_date="2023-04-02",
                team="DD",
            ),
        ]

        batting, pitching, teams = (
            build_frames(
                pitching_rows=pitching_rows,
                team_rows=team_rows,
            )
        )

        _, pitching_snapshot, _ = (
            build_season_snapshots(
                batting,
                pitching,
                teams,
            )
        )

        self.assertEqual(
            int(
                pitching_snapshot.loc[
                    0,
                    "pitch_rows",
                ]
            ),
            6,
        )

    def test_26_separates_pitching_seasons(
        self,
    ) -> None:
        game_2024 = (
            "20240401AABB02024"
        )

        team_rows = (
            team_game_rows()
            + team_game_rows(
                game_pk=game_2024,
                game_date="2024-04-01",
                season=2024,
            )
        )

        pitching_rows = [
            pitching_row(),
            pitching_row(
                game_pk=game_2024,
                game_date="2024-04-01",
                season=2024,
            ),
        ]

        batting, pitching, teams = (
            build_frames(
                pitching_rows=pitching_rows,
                team_rows=team_rows,
            )
        )

        _, pitching_snapshot, _ = (
            build_season_snapshots(
                batting,
                pitching,
                teams,
            )
        )

        self.assertEqual(
            set(
                pitching_snapshot[
                    "season"
                ].astype(int)
            ),
            {
                2023,
                2024,
            },
        )

    def test_27_separates_multiple_pitchers_same_season(
        self,
    ) -> None:
        batting, pitching, teams = (
            build_frames(
                pitching_rows=[
                    pitching_row(
                        pitcher="101"
                    ),
                    pitching_row(
                        pitcher="102"
                    ),
                ]
            )
        )

        _, pitching_snapshot, _ = (
            build_season_snapshots(
                batting,
                pitching,
                teams,
            )
        )

        self.assertEqual(
            set(
                pitching_snapshot[
                    "pitcher"
                ]
            ),
            {
                "101",
                "102",
            },
        )

    def test_28_trade_keeps_single_pitcher_season_without_team_column(
        self,
    ) -> None:
        second_game = (
            "20230402CCDD02023"
        )

        team_rows = (
            team_game_rows()
            + team_game_rows(
                game_pk=second_game,
                game_date="2023-04-02",
                away_team="CC",
                home_team="DD",
            )
        )

        pitching_rows = [
            pitching_row(
                team="BB"
            ),
            pitching_row(
                game_pk=second_game,
                game_date="2023-04-02",
                team="DD",
            ),
        ]

        batting, pitching, teams = (
            build_frames(
                pitching_rows=pitching_rows,
                team_rows=team_rows,
            )
        )

        _, pitching_snapshot, _ = (
            build_season_snapshots(
                batting,
                pitching,
                teams,
            )
        )

        self.assertEqual(
            len(
                pitching_snapshot
            ),
            1,
        )

        self.assertNotIn(
            "team",
            pitching_snapshot.columns,
        )

    def test_29_recalculates_hits_allowed(
        self,
    ) -> None:
        batting, pitching, teams = (
            build_frames(
                pitching_rows=[
                    pitching_row(
                        hits_allowed=4,
                        single_allowed=1,
                        double_allowed=1,
                        triple_allowed=1,
                        hr_allowed=1,
                    )
                ]
            )
        )

        _, pitching_snapshot, _ = (
            build_season_snapshots(
                batting,
                pitching,
                teams,
            )
        )

        self.assertEqual(
            int(
                pitching_snapshot.loc[
                    0,
                    "hits_allowed",
                ]
            ),
            4,
        )

    def test_30_validates_pitch_type_count_formula(
        self,
    ) -> None:
        batting, pitching, teams = (
            build_frames(
                pitching_rows=[
                    pitching_row(
                        pitches=4,
                        ball_pitch_count=1,
                        strike_pitch_count=2,
                        in_play_pitch_count=0,
                    )
                ]
            )
        )

        with self.assertRaises(
            SeasonSnapshotsBuildError
        ):
            build_season_snapshots(
                batting,
                pitching,
                teams,
            )

    def test_31_sums_outs_when_all_source_values_are_non_null(
        self,
    ) -> None:
        second_game = (
            "20230402CCDD02023"
        )

        team_rows = (
            team_game_rows()
            + team_game_rows(
                game_pk=second_game,
                game_date="2023-04-02",
                away_team="CC",
                home_team="DD",
            )
        )

        pitching_rows = [
            pitching_row(
                outs_recorded=1,
            ),
            pitching_row(
                game_pk=second_game,
                game_date="2023-04-02",
                team="DD",
                outs_recorded=2,
            ),
        ]

        batting, pitching, teams = (
            build_frames(
                pitching_rows=pitching_rows,
                team_rows=team_rows,
            )
        )

        _, pitching_snapshot, _ = (
            build_season_snapshots(
                batting,
                pitching,
                teams,
            )
        )

        self.assertEqual(
            int(
                pitching_snapshot.loc[
                    0,
                    "outs_recorded",
                ]
            ),
            3,
        )

    def test_32_propagates_outs_null_to_season(
        self,
    ) -> None:
        second_game = (
            "20230402CCDD02023"
        )

        team_rows = (
            team_game_rows()
            + team_game_rows(
                game_pk=second_game,
                game_date="2023-04-02",
                away_team="CC",
                home_team="DD",
            )
        )

        pitching_rows = [
            pitching_row(
                outs_recorded=1,
            ),
            pitching_row(
                game_pk=second_game,
                game_date="2023-04-02",
                team="DD",
                outs_recorded=pd.NA,
            ),
        ]

        batting, pitching, teams = (
            build_frames(
                pitching_rows=pitching_rows,
                team_rows=team_rows,
            )
        )

        _, pitching_snapshot, _ = (
            build_season_snapshots(
                batting,
                pitching,
                teams,
            )
        )

        self.assertTrue(
            pd.isna(
                pitching_snapshot.loc[
                    0,
                    "outs_recorded",
                ]
            )
        )

    def test_33_does_not_replace_outs_null_with_zero(
        self,
    ) -> None:
        batting, pitching, teams = (
            build_frames(
                pitching_rows=[
                    pitching_row(
                        outs_recorded=pd.NA,
                    )
                ]
            )
        )

        _, pitching_snapshot, _ = (
            build_season_snapshots(
                batting,
                pitching,
                teams,
            )
        )

        self.assertTrue(
            pd.isna(
                pitching_snapshot.loc[
                    0,
                    "outs_recorded",
                ]
            )
        )

    def test_34_does_not_use_skipna_partial_out_sum(
        self,
    ) -> None:
        second_game = (
            "20230402CCDD02023"
        )

        team_rows = (
            team_game_rows()
            + team_game_rows(
                game_pk=second_game,
                game_date="2023-04-02",
                away_team="CC",
                home_team="DD",
            )
        )

        pitching_rows = [
            pitching_row(
                outs_recorded=9,
            ),
            pitching_row(
                game_pk=second_game,
                game_date="2023-04-02",
                team="DD",
                outs_recorded=pd.NA,
            ),
        ]

        batting, pitching, teams = (
            build_frames(
                pitching_rows=pitching_rows,
                team_rows=team_rows,
            )
        )

        _, pitching_snapshot, _ = (
            build_season_snapshots(
                batting,
                pitching,
                teams,
            )
        )

        self.assertTrue(
            pd.isna(
                pitching_snapshot.loc[
                    0,
                    "outs_recorded",
                ]
            )
        )

    def test_35_pitching_snapshot_excludes_era_and_run_columns(
        self,
    ) -> None:
        _, pitching_snapshot, _ = (
            build_default()
        )

        for column in (
            "earned_runs",
            "era",
            "runs_allowed",
        ):
            self.assertNotIn(
                column,
                pitching_snapshot.columns,
            )

    def test_36_pitching_snapshot_excludes_release_speed_average(
        self,
    ) -> None:
        _, pitching_snapshot, _ = (
            build_default()
        )

        self.assertNotIn(
            "avg_release_speed_kmh",
            pitching_snapshot.columns,
        )

    def test_37_pitcher_season_key_is_unique(
        self,
    ) -> None:
        _, pitching_snapshot, _ = (
            build_default()
        )

        self.assertEqual(
            int(
                pitching_snapshot.duplicated(
                    subset=[
                        "season",
                        "pitcher",
                    ]
                ).sum()
            ),
            0,
        )

    # ------------------------------------------------------------------
    # Team
    # ------------------------------------------------------------------

    def test_38_builds_normal_team_season(
        self,
    ) -> None:
        _, _, team_snapshot = (
            build_default()
        )

        self.assertEqual(
            len(
                team_snapshot
            ),
            2,
        )

        self.assertEqual(
            tuple(
                team_snapshot.columns
            ),
            TEAM_OUTPUT_COLUMNS,
        )

    def test_39_calculates_team_games(
        self,
    ) -> None:
        _, _, team_snapshot = (
            build_default()
        )

        self.assertTrue(
            team_snapshot[
                "games"
            ]
            .eq(1)
            .all()
        )

    def test_40_calculates_wins_losses_ties(
        self,
    ) -> None:
        _, _, team_snapshot = (
            build_default()
        )

        aa = (
            team_snapshot
            .set_index("team")
            .loc["AA"]
        )

        bb = (
            team_snapshot
            .set_index("team")
            .loc["BB"]
        )

        self.assertEqual(
            int(
                aa["losses"]
            ),
            1,
        )

        self.assertEqual(
            int(
                bb["wins"]
            ),
            1,
        )

    def test_41_calculates_runs_for_and_against(
        self,
    ) -> None:
        _, _, team_snapshot = (
            build_default()
        )

        bb = (
            team_snapshot
            .set_index("team")
            .loc["BB"]
        )

        self.assertEqual(
            int(
                bb["runs_for"]
            ),
            2,
        )

        self.assertEqual(
            int(
                bb["runs_against"]
            ),
            1,
        )

    def test_42_calculates_run_diff(
        self,
    ) -> None:
        _, _, team_snapshot = (
            build_default()
        )

        bb = (
            team_snapshot
            .set_index("team")
            .loc["BB"]
        )

        self.assertEqual(
            int(
                bb["run_diff"]
            ),
            1,
        )

    def test_43_games_equals_wins_losses_ties(
        self,
    ) -> None:
        _, _, team_snapshot = (
            build_default()
        )

        expected = (
            team_snapshot["wins"]
            + team_snapshot["losses"]
            + team_snapshot["ties"]
        )

        self.assertTrue(
            team_snapshot[
                "games"
            ]
            .eq(expected)
            .all()
        )

    def test_44_season_wins_equal_losses(
        self,
    ) -> None:
        _, _, team_snapshot = (
            build_default()
        )

        self.assertEqual(
            int(
                team_snapshot[
                    "wins"
                ].sum()
            ),
            int(
                team_snapshot[
                    "losses"
                ].sum()
            ),
        )

    def test_45_season_runs_for_equal_runs_against(
        self,
    ) -> None:
        _, _, team_snapshot = (
            build_default()
        )

        self.assertEqual(
            int(
                team_snapshot[
                    "runs_for"
                ].sum()
            ),
            int(
                team_snapshot[
                    "runs_against"
                ].sum()
            ),
        )

    def test_46_team_game_rows_equal_snapshot_games_sum(
        self,
    ) -> None:
        batting, pitching, teams = (
            build_frames()
        )

        _, _, team_snapshot = (
            build_season_snapshots(
                batting,
                pitching,
                teams,
            )
        )

        self.assertEqual(
            len(
                teams
            ),
            int(
                team_snapshot[
                    "games"
                ].sum()
            ),
        )

    def test_47_team_season_key_is_unique(
        self,
    ) -> None:
        _, _, team_snapshot = (
            build_default()
        )

        self.assertEqual(
            int(
                team_snapshot.duplicated(
                    subset=[
                        "season",
                        "team",
                    ]
                ).sum()
            ),
            0,
        )

    # ------------------------------------------------------------------
    # Safety / Determinism / I/O
    # ------------------------------------------------------------------

    def test_48_input_row_shuffle_is_deterministic(
        self,
    ) -> None:
        second_game = (
            "20230402CCDD02023"
        )

        batting_rows = [
            batting_row(),
            batting_row(
                game_pk=second_game,
                game_date="2023-04-02",
                team="CC",
            ),
        ]

        pitching_rows = [
            pitching_row(),
            pitching_row(
                game_pk=second_game,
                game_date="2023-04-02",
                team="DD",
            ),
        ]

        team_rows = (
            team_game_rows()
            + team_game_rows(
                game_pk=second_game,
                game_date="2023-04-02",
                away_team="CC",
                home_team="DD",
            )
        )

        batting, pitching, teams = (
            build_frames(
                batting_rows=batting_rows,
                pitching_rows=pitching_rows,
                team_rows=team_rows,
            )
        )

        expected = (
            build_season_snapshots(
                batting,
                pitching,
                teams,
            )
        )

        actual = (
            build_season_snapshots(
                batting.iloc[
                    ::-1
                ].reset_index(
                    drop=True
                ),
                pitching.iloc[
                    ::-1
                ].reset_index(
                    drop=True
                ),
                teams.iloc[
                    ::-1
                ].reset_index(
                    drop=True
                ),
            )
        )

        for expected_frame, actual_frame in zip(
            expected,
            actual,
        ):
            pd.testing.assert_frame_equal(
                expected_frame,
                actual_frame,
                check_dtype=True,
                check_like=False,
            )

    def test_49_outputs_use_deterministic_sort(
        self,
    ) -> None:
        batting_rows = [
            batting_row(
                batter="002"
            ),
            batting_row(
                batter="001"
            ),
        ]

        pitching_rows = [
            pitching_row(
                pitcher="102"
            ),
            pitching_row(
                pitcher="101"
            ),
        ]

        batting, pitching, teams = (
            build_frames(
                batting_rows=batting_rows,
                pitching_rows=pitching_rows,
            )
        )

        (
            batting_snapshot,
            pitching_snapshot,
            team_snapshot,
        ) = build_season_snapshots(
            batting,
            pitching,
            teams,
        )

        self.assertEqual(
            batting_snapshot[
                "batter"
            ].tolist(),
            [
                "001",
                "002",
            ],
        )

        self.assertEqual(
            pitching_snapshot[
                "pitcher"
            ].tolist(),
            [
                "101",
                "102",
            ],
        )

        self.assertEqual(
            team_snapshot[
                "team"
            ].tolist(),
            sorted(
                team_snapshot[
                    "team"
                ].tolist()
            ),
        )

    def test_50_parquet_round_trip_preserves_dtypes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(
                temp_dir
            )

            (
                batting_path,
                pitching_path,
                team_path,
                output_dir,
            ) = self._write_file_fixture(
                root
            )

            build_season_snapshots_file(
                batting_path=batting_path,
                pitching_path=pitching_path,
                team_games_path=team_path,
                output_dir=output_dir,
            )

            batting = pd.read_parquet(
                output_dir
                / BATTING_SNAPSHOT_FILENAME,
                engine="pyarrow",
            )

            pitching = pd.read_parquet(
                output_dir
                / PITCHING_SNAPSHOT_FILENAME,
                engine="pyarrow",
            )

            team = pd.read_parquet(
                output_dir
                / TEAM_SNAPSHOT_FILENAME,
                engine="pyarrow",
            )

            self.assertEqual(
                str(
                    batting[
                        "through_date"
                    ].dtype
                ),
                "datetime64[us]",
            )

            self.assertEqual(
                str(
                    batting[
                        "season"
                    ].dtype
                ),
                "Int64",
            )

            self.assertEqual(
                str(
                    batting[
                        "avg"
                    ].dtype
                ),
                "Float64",
            )

            self.assertEqual(
                str(
                    pitching[
                        "outs_recorded"
                    ].dtype
                ),
                "Int64",
            )

            self.assertEqual(
                str(
                    team[
                        "games"
                    ].dtype
                ),
                "Int64",
            )

    def test_51_reexecution_is_dataframe_deterministic(
        self,
    ) -> None:
        batting, pitching, teams = (
            build_frames()
        )

        first = (
            build_season_snapshots(
                batting,
                pitching,
                teams,
            )
        )

        second = (
            build_season_snapshots(
                batting,
                pitching,
                teams,
            )
        )

        for first_frame, second_frame in zip(
            first,
            second,
        ):
            pd.testing.assert_frame_equal(
                first_frame,
                second_frame,
                check_dtype=True,
                check_like=False,
            )

    def test_52_rejects_output_under_data_raw(
        self,
    ) -> None:
        output_dir = (
            RAW_DATA_DIR
            / "issue13-test"
        )

        with self.assertRaises(
            SeasonSnapshotsBuildError
        ):
            ensure_output_paths_safe(
                batting_path=Path(
                    "/tmp/player_game_batting.parquet"
                ),
                pitching_path=Path(
                    "/tmp/player_game_pitching.parquet"
                ),
                team_games_path=Path(
                    "/tmp/team_games.parquet"
                ),
                batting_output_path=(
                    output_dir
                    / BATTING_SNAPSHOT_FILENAME
                ),
                pitching_output_path=(
                    output_dir
                    / PITCHING_SNAPSHOT_FILENAME
                ),
                team_output_path=(
                    output_dir
                    / TEAM_SNAPSHOT_FILENAME
                ),
            )

    def test_53_rejects_batting_input_overwrite(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(
                temp_dir
            )

            batting_input = (
                root
                / BATTING_SNAPSHOT_FILENAME
            )

            with self.assertRaises(
                SeasonSnapshotsBuildError
            ):
                ensure_output_paths_safe(
                    batting_path=batting_input,
                    pitching_path=(
                        root
                        / "player_game_pitching.parquet"
                    ),
                    team_games_path=(
                        root
                        / "team_games.parquet"
                    ),
                    batting_output_path=batting_input,
                    pitching_output_path=(
                        root
                        / PITCHING_SNAPSHOT_FILENAME
                    ),
                    team_output_path=(
                        root
                        / TEAM_SNAPSHOT_FILENAME
                    ),
                )

    def test_54_rejects_pitching_input_overwrite(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(
                temp_dir
            )

            pitching_input = (
                root
                / PITCHING_SNAPSHOT_FILENAME
            )

            with self.assertRaises(
                SeasonSnapshotsBuildError
            ):
                ensure_output_paths_safe(
                    batting_path=(
                        root
                        / "player_game_batting.parquet"
                    ),
                    pitching_path=pitching_input,
                    team_games_path=(
                        root
                        / "team_games.parquet"
                    ),
                    batting_output_path=(
                        root
                        / BATTING_SNAPSHOT_FILENAME
                    ),
                    pitching_output_path=pitching_input,
                    team_output_path=(
                        root
                        / TEAM_SNAPSHOT_FILENAME
                    ),
                )

    def test_55_rejects_team_game_input_overwrite(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(
                temp_dir
            )

            team_input = (
                root
                / TEAM_SNAPSHOT_FILENAME
            )

            with self.assertRaises(
                SeasonSnapshotsBuildError
            ):
                ensure_output_paths_safe(
                    batting_path=(
                        root
                        / "player_game_batting.parquet"
                    ),
                    pitching_path=(
                        root
                        / "player_game_pitching.parquet"
                    ),
                    team_games_path=team_input,
                    batting_output_path=(
                        root
                        / BATTING_SNAPSHOT_FILENAME
                    ),
                    pitching_output_path=(
                        root
                        / PITCHING_SNAPSHOT_FILENAME
                    ),
                    team_output_path=team_input,
                )

    def test_56_rejects_duplicate_snapshot_output_paths(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(
                temp_dir
            )

            duplicate = (
                root
                / "duplicate.parquet"
            )

            with self.assertRaises(
                SeasonSnapshotsBuildError
            ):
                ensure_output_paths_safe(
                    batting_path=(
                        root
                        / "player_game_batting.parquet"
                    ),
                    pitching_path=(
                        root
                        / "player_game_pitching.parquet"
                    ),
                    team_games_path=(
                        root
                        / "team_games.parquet"
                    ),
                    batting_output_path=duplicate,
                    pitching_output_path=duplicate,
                    team_output_path=(
                        root
                        / TEAM_SNAPSHOT_FILENAME
                    ),
                )

    def test_57_build_file_preserves_three_input_sha256_values(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(
                temp_dir
            )

            (
                batting_path,
                pitching_path,
                team_path,
                output_dir,
            ) = self._write_file_fixture(
                root
            )

            before = {
                batting_path: sha256(
                    batting_path
                ),
                pitching_path: sha256(
                    pitching_path
                ),
                team_path: sha256(
                    team_path
                ),
            }

            build_season_snapshots_file(
                batting_path=batting_path,
                pitching_path=pitching_path,
                team_games_path=team_path,
                output_dir=output_dir,
            )

            after = {
                batting_path: sha256(
                    batting_path
                ),
                pitching_path: sha256(
                    pitching_path
                ),
                team_path: sha256(
                    team_path
                ),
            }

            self.assertEqual(
                before,
                after,
            )

    def test_58_2026_is_partial_snapshot_without_completion_special_case(
        self,
    ) -> None:
        game_pk = (
            "20260510AABB02026"
        )

        batting_rows = [
            batting_row(
                game_pk=game_pk,
                game_date="2026-05-10",
                season=2026,
            )
        ]

        pitching_rows = [
            pitching_row(
                game_pk=game_pk,
                game_date="2026-05-10",
                season=2026,
            )
        ]

        team_rows = team_game_rows(
            game_pk=game_pk,
            game_date="2026-05-10",
            season=2026,
        )

        batting, pitching, teams = (
            build_frames(
                batting_rows=batting_rows,
                pitching_rows=pitching_rows,
                team_rows=team_rows,
            )
        )

        (
            batting_snapshot,
            pitching_snapshot,
            team_snapshot,
        ) = build_season_snapshots(
            batting,
            pitching,
            teams,
        )

        for frame in (
            batting_snapshot,
            pitching_snapshot,
            team_snapshot,
        ):
            self.assertTrue(
                frame[
                    "through_date"
                ]
                .eq(
                    pd.Timestamp(
                        "2026-05-10"
                    )
                )
                .all()
            )

            self.assertNotIn(
                "season_complete",
                frame.columns,
            )

            self.assertNotIn(
                "final_record",
                frame.columns,
            )


if __name__ == "__main__":
    unittest.main()