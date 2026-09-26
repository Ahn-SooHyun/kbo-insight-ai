from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from scripts.build_game_tables import (  # noqa: E402
    build_game_tables,
)
from scripts.build_plate_appearances import (  # noqa: E402
    build_plate_appearances,
)
from scripts.build_player_game_batting import (  # noqa: E402
    build_player_game_batting,
)
from scripts.build_player_game_pitching import (  # noqa: E402
    build_player_game_pitching,
)
from scripts.build_players import (  # noqa: E402
    build_players,
)
from scripts.build_season_snapshots import (  # noqa: E402
    build_season_snapshots,
)
from scripts.validate_derived_tables import (  # noqa: E402
    TABLE_CONTRACTS,
    DerivedValidationError,
    build_expected_season_batting,
    calculate_rate,
    content_fingerprint,
    validate_derived_path_policy,
    validate_derived_tables,
    validate_score_chain,
)


@dataclass(frozen=True)
class Fixture:
    """통합 Validator 테스트용 임시 프로젝트 정보를 보관한다."""

    project_root: Path
    raw_dir: Path
    derived_dir: Path
    season: int
    frames: dict[str, pd.DataFrame]


def raw_row(
    *,
    game_pk: str,
    game_date: str,
    home_team: str,
    away_team: str,
    inning: int = 1,
    inning_topbot: str = "top",
    at_bat_number: int = 1,
    pitch_number: int = 1,
    batter: str = "B1",
    pitcher: str = "P1",
    batter_name: str | None = "타자1",
    pitcher_name: str | None = "투수1",
    strikes: int = 0,
    outs_when_up: int = 0,
    on_1b: str | None = None,
    on_2b: str | None = None,
    on_3b: str | None = None,
    home_score: int = 0,
    away_score: int = 0,
    pitch_type: str | None = "X",
    stand: str | None = "R",
    events: str | None = "field_out",
    post_home_score: int = 0,
    post_away_score: int = 0,
    post_outs: int = 0,
    runs_scored: int = 0,
    post_on_1b: str | None = None,
    post_on_2b: str | None = None,
    post_on_3b: str | None = None,
    release_speed_kmh: float | None = 145.0,
) -> dict[str, object]:
    """Plate/Pitching Builder 모두가 읽을 수 있는 Synthetic Raw Row를 만든다."""
    return {
        "game_pk": game_pk,
        "game_date": game_date,
        "home_team": home_team,
        "away_team": away_team,
        "inning": inning,
        "inning_topbot": inning_topbot,
        "at_bat_number": at_bat_number,
        "pitch_number": pitch_number,
        "batter": batter,
        "pitcher": pitcher,
        "batter_name": batter_name,
        "pitcher_name": pitcher_name,
        "strikes": strikes,
        "outs_when_up": outs_when_up,
        "on_1b": on_1b,
        "on_2b": on_2b,
        "on_3b": on_3b,
        "home_score": home_score,
        "away_score": away_score,
        "type": pitch_type,
        "stand": stand,
        "events": events,
        "post_home_score": post_home_score,
        "post_away_score": post_away_score,
        "post_outs": post_outs,
        "runs_scored": runs_scored,
        "post_on_1b": post_on_1b,
        "post_on_2b": post_on_2b,
        "post_on_3b": post_on_3b,
        "release_speed_kmh": release_speed_kmh,
    }


def make_raw(
    *,
    season: int,
    first_date: str,
    second_date: str,
) -> pd.DataFrame:
    """
    두 경기의 Synthetic Raw를 만든다.

    첫 경기는 Pitch-less PA를 포함하고,
    두 번째 경기는 Multi-pitcher + ambiguous Out을 포함한다.
    """
    game_one = (
        f"{season}0401AAHH"
    )

    game_two = (
        f"{season}0402A2H2"
    )

    rows = [
        # G1 PA1: Away Single
        raw_row(
            game_pk=game_one,
            game_date=first_date,
            home_team="HH",
            away_team="AA",
            at_bat_number=1,
            pitch_number=1,
            batter="B1",
            pitcher="PH",
            batter_name="타자1",
            pitcher_name="홈투수",
            pitch_type="X",
            events="single",
            outs_when_up=0,
            post_outs=0,
            post_on_1b="B1",
        ),

        # G1 PA2: Pitch-less Walk
        raw_row(
            game_pk=game_one,
            game_date=first_date,
            home_team="HH",
            away_team="AA",
            at_bat_number=2,
            pitch_number=0,
            batter="B2",
            pitcher="PH",
            batter_name="타자2",
            pitcher_name="홈투수",
            pitch_type=None,
            release_speed_kmh=None,
            events="walk",
            outs_when_up=0,
            on_1b="B1",
            post_outs=0,
            post_on_1b="B2",
            post_on_2b="B1",
        ),

        # G1 PA3: Away 3-run HR
        raw_row(
            game_pk=game_one,
            game_date=first_date,
            home_team="HH",
            away_team="AA",
            at_bat_number=3,
            pitch_number=1,
            batter="B3",
            pitcher="PH",
            batter_name="타자3",
            pitcher_name="홈투수",
            pitch_type="X",
            events="home_run",
            outs_when_up=0,
            on_1b="B2",
            on_2b="B1",
            home_score=0,
            away_score=0,
            post_home_score=0,
            post_away_score=3,
            post_outs=0,
            runs_scored=3,
        ),

        # G1 PA4: Home Field Out
        raw_row(
            game_pk=game_one,
            game_date=first_date,
            home_team="HH",
            away_team="AA",
            inning_topbot="bot",
            at_bat_number=4,
            pitch_number=1,
            batter="H1",
            pitcher="PA",
            batter_name="홈타자",
            pitcher_name="원정투수",
            pitch_type="X",
            events="field_out",
            outs_when_up=0,
            home_score=0,
            away_score=3,
            post_home_score=0,
            post_away_score=3,
            post_outs=1,
        ),

        # G2 PA1 pitch 1:
        # 투수 교체 전 첫 Pitch. PA 종료 Event는 아직 없다.
        raw_row(
            game_pk=game_two,
            game_date=second_date,
            home_team="H2",
            away_team="A2",
            at_bat_number=1,
            pitch_number=1,
            batter="B4",
            pitcher="P2A",
            batter_name="타자4",
            pitcher_name="교체전투수",
            pitch_type="B",
            events=None,
            outs_when_up=0,
            post_outs=0,
        ),

        # G2 PA1 pitch 2:
        # field_out인데 PA Out Delta가 2이므로
        # 공개 Raw만으로 두 Out 발생 시점을 투수별 분리할 수 없다.
        raw_row(
            game_pk=game_two,
            game_date=second_date,
            home_team="H2",
            away_team="A2",
            at_bat_number=1,
            pitch_number=2,
            batter="B4",
            pitcher="P2B",
            batter_name="타자4",
            pitcher_name="교체후투수",
            pitch_type="X",
            events="field_out",
            outs_when_up=0,
            post_outs=2,
        ),
    ]

    return pd.DataFrame(
        rows
    )


def build_fixture(
    root: Path,
    *,
    season: int = 2023,
    first_date: str = "2023-04-01",
    second_date: str = "2023-04-02",
) -> Fixture:
    """기존 Builder로 정상 Synthetic Canonical Layer 9종을 만든다."""
    project_root = (
        root
        / "project"
    )

    raw_dir = (
        project_root
        / "data"
        / "raw"
        / "hf_kbo_pbp"
    )

    derived_dir = (
        project_root
        / "data"
        / "interim"
        / "hf_kbo_pbp"
        / "derived"
    )

    raw_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    derived_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        project_root
        / ".gitignore"
    ).write_text(
        "/data/interim/**\n",
        encoding="utf-8",
    )

    raw = make_raw(
        season=season,
        first_date=first_date,
        second_date=second_date,
    )

    raw.to_parquet(
        raw_dir
        / f"{season}.parquet",
        index=False,
    )

    plate_appearances = (
        build_plate_appearances(
            raw,
            season,
        )
    )

    games, team_games = (
        build_game_tables(
            plate_appearances
        )
    )

    player_game_batting = (
        build_player_game_batting(
            plate_appearances
        )
    )

    player_game_pitching = (
        build_player_game_pitching(
            raw,
            plate_appearances,
            season=season,
        )
    )

    players = build_players(
        plate_appearances,
        player_game_batting,
        player_game_pitching,
    )

    (
        player_season_batting,
        player_season_pitching,
        team_season,
    ) = build_season_snapshots(
        player_game_batting,
        player_game_pitching,
        team_games,
    )

    frames = {
        "plate_appearances": (
            plate_appearances
        ),
        "games": games,
        "team_games": team_games,
        "player_game_batting": (
            player_game_batting
        ),
        "player_game_pitching": (
            player_game_pitching
        ),
        "players": players,
        "player_season_batting_snapshot": (
            player_season_batting
        ),
        "player_season_pitching_snapshot": (
            player_season_pitching
        ),
        "team_season_snapshot": (
            team_season
        ),
    }

    filename_by_name = {
        contract.name: (
            contract.filename
        )
        for contract
        in TABLE_CONTRACTS
    }

    for name, frame in frames.items():
        frame.to_parquet(
            derived_dir
            / filename_by_name[
                name
            ],
            index=False,
        )

    return Fixture(
        project_root=project_root,
        raw_dir=raw_dir,
        derived_dir=derived_dir,
        season=season,
        frames=frames,
    )


def run_fixture(
    fixture: Fixture,
):
    """Fixture를 Production Validator와 같은 경로로 검증한다."""
    return validate_derived_tables(
        raw_dir=(
            fixture.raw_dir
        ),
        derived_dir=(
            fixture.derived_dir
        ),
        seasons=(
            fixture.season,
        ),
        project_root=(
            fixture.project_root
        ),
    )


def overwrite_table(
    fixture: Fixture,
    table_name: str,
    frame: pd.DataFrame,
) -> None:
    """특정 Derived Table만 오류 Fixture로 교체한다."""
    contract = next(
        contract
        for contract
        in TABLE_CONTRACTS
        if contract.name
        == table_name
    )

    frame.to_parquet(
        fixture.derived_dir
        / contract.filename,
        index=False,
    )


class DerivedValidationHappyPathTest(
    unittest.TestCase
):
    """정상 통합 검증과 Read-only/결정성 계약을 검증한다."""

    def test_valid_canonical_layer_passes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = build_fixture(
                Path(
                    directory
                )
            )

            summary = run_fixture(
                fixture
            )

            self.assertEqual(
                summary.raw_rows,
                6,
            )

            self.assertEqual(
                summary.raw_actual_pitches,
                5,
            )

            self.assertEqual(
                summary.raw_unique_pa,
                5,
            )

            self.assertEqual(
                summary.raw_unique_games,
                2,
            )

            self.assertEqual(
                len(
                    summary
                    .content_fingerprints
                ),
                9,
            )

            self.assertTrue(
                all(
                    count == 0
                    for count
                    in summary
                    .duplicate_key_counts
                    .values()
                )
            )

            self.assertEqual(
                summary.raw_sha256_before,
                summary.raw_sha256_after,
            )

            self.assertEqual(
                summary.derived_sha256_before,
                summary.derived_sha256_after,
            )

    def test_ambiguous_multi_pitcher_outs_are_nullable(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = build_fixture(
                Path(
                    directory
                )
            )

            pitching = (
                fixture.frames[
                    "player_game_pitching"
                ]
            )

            ambiguous = (
                pitching.loc[
                    pitching[
                        "pitcher"
                    ].isin(
                        [
                            "P2A",
                            "P2B",
                        ]
                    )
                ]
            )

            self.assertEqual(
                len(
                    ambiguous
                ),
                2,
            )

            self.assertTrue(
                ambiguous[
                    "outs_recorded"
                ]
                .isna()
                .all()
            )

            run_fixture(
                fixture
            )

    def test_validator_rerun_summary_is_deterministic(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = build_fixture(
                Path(
                    directory
                )
            )

            first = run_fixture(
                fixture
            )

            second = run_fixture(
                fixture
            )

            self.assertEqual(
                first,
                second,
            )

    def test_validator_does_not_create_or_modify_files(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = build_fixture(
                Path(
                    directory
                )
            )

            before = {
                path.relative_to(
                    fixture.project_root
                ): (
                    path.read_bytes()
                )
                for path
                in fixture.project_root.rglob(
                    "*"
                )
                if path.is_file()
            }

            run_fixture(
                fixture
            )

            after = {
                path.relative_to(
                    fixture.project_root
                ): (
                    path.read_bytes()
                )
                for path
                in fixture.project_root.rglob(
                    "*"
                )
                if path.is_file()
            }

            self.assertEqual(
                before,
                after,
            )


class FileSchemaGrainContractTest(
    unittest.TestCase
):
    """File / Schema / Grain 계약 오류를 검증한다."""

    def test_missing_derived_file_fails(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = build_fixture(
                Path(
                    directory
                )
            )

            (
                fixture.derived_dir
                / "games.parquet"
            ).unlink()

            with self.assertRaises(
                DerivedValidationError
            ):
                run_fixture(
                    fixture
                )

    def test_missing_required_column_fails(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = build_fixture(
                Path(
                    directory
                )
            )

            games = (
                fixture.frames[
                    "games"
                ]
                .drop(
                    columns=[
                        "total_runs"
                    ]
                )
            )

            overwrite_table(
                fixture,
                "games",
                games,
            )

            with self.assertRaises(
                DerivedValidationError
            ):
                run_fixture(
                    fixture
                )

    def test_dtype_mismatch_fails(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = build_fixture(
                Path(
                    directory
                )
            )

            games = (
                fixture.frames[
                    "games"
                ]
                .copy()
            )

            games[
                "season"
            ] = (
                games[
                    "season"
                ]
                .astype("string")
            )

            overwrite_table(
                fixture,
                "games",
                games,
            )

            with self.assertRaises(
                DerivedValidationError
            ):
                run_fixture(
                    fixture
                )

    def test_plate_appearance_persisted_game_date_dtype_is_accepted(
        self,
    ) -> None:
        """PA Parquet round-trip의 game_date datetime64[us] 계약을 검증한다."""
        with tempfile.TemporaryDirectory() as directory:
            fixture = build_fixture(
                Path(
                    directory
                )
            )

            plate_appearances = pd.read_parquet(
                fixture.derived_dir
                / "plate_appearances.parquet"
            )

            self.assertEqual(
                str(
                    plate_appearances[
                        "game_date"
                    ].dtype
                ),
                "datetime64[us]",
            )

            run_fixture(
                fixture
            )

    def test_all_nine_grain_duplicates_fail(
        self,
    ) -> None:
        for contract in TABLE_CONTRACTS:
            with self.subTest(
                table=(
                    contract.name
                )
            ):
                with tempfile.TemporaryDirectory() as directory:
                    fixture = build_fixture(
                        Path(
                            directory
                        )
                    )

                    frame = (
                        fixture.frames[
                            contract.name
                        ]
                    )

                    duplicated = pd.concat(
                        [
                            frame,
                            frame.iloc[
                                [0]
                            ],
                        ],
                        ignore_index=True,
                    )

                    overwrite_table(
                        fixture,
                        contract.name,
                        duplicated,
                    )

                    with self.assertRaises(
                        DerivedValidationError
                    ):
                        run_fixture(
                            fixture
                        )


class RawPaGameContractTest(
    unittest.TestCase
):
    """Raw ↔ PA/Game 및 Score Chain 계약을 검증한다."""

    def test_pitchless_pa_is_required_in_key_reconciliation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = build_fixture(
                Path(
                    directory
                )
            )

            plate_appearances = (
                fixture.frames[
                    "plate_appearances"
                ]
                .copy()
            )

            pitchless = (
                plate_appearances[
                    "pitch_count"
                ]
                .eq(0)
            )

            self.assertEqual(
                int(
                    pitchless.sum()
                ),
                1,
            )

            plate_appearances = (
                plate_appearances.loc[
                    ~pitchless
                ]
                .reset_index(
                    drop=True
                )
            )

            overwrite_table(
                fixture,
                "plate_appearances",
                plate_appearances,
            )

            with self.assertRaises(
                DerivedValidationError
            ):
                run_fixture(
                    fixture
                )

    def test_raw_game_key_mismatch_fails(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = build_fixture(
                Path(
                    directory
                )
            )

            games = (
                fixture.frames[
                    "games"
                ]
                .iloc[
                    :-1
                ]
                .reset_index(
                    drop=True
                )
            )

            overwrite_table(
                fixture,
                "games",
                games,
            )

            with self.assertRaises(
                DerivedValidationError
            ):
                run_fixture(
                    fixture
                )

    def test_team_game_mirror_mismatch_fails(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = build_fixture(
                Path(
                    directory
                )
            )

            team_games = (
                fixture.frames[
                    "team_games"
                ]
                .copy()
            )

            team_games.loc[
                0,
                "runs_for",
            ] = (
                int(
                    team_games.loc[
                        0,
                        "runs_for",
                    ]
                )
                + 1
            )

            overwrite_table(
                fixture,
                "team_games",
                team_games,
            )

            with self.assertRaises(
                DerivedValidationError
            ):
                run_fixture(
                    fixture
                )

    def test_final_score_must_match_actual_last_pa_row(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = build_fixture(
                Path(
                    directory
                )
            )

            games = (
                fixture.frames[
                    "games"
                ]
                .copy()
            )

            games.loc[
                0,
                "final_home_score",
            ] = (
                int(
                    games.loc[
                        0,
                        "final_home_score",
                    ]
                )
                + 1
            )

            with self.assertRaises(
                DerivedValidationError
            ):
                validate_score_chain(
                    fixture.frames[
                        "plate_appearances"
                    ],
                    games,
                )

    def test_final_score_must_match_pre_score_plus_runs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = build_fixture(
                Path(
                    directory
                )
            )

            plate_appearances = (
                fixture.frames[
                    "plate_appearances"
                ]
                .copy()
            )

            plate_appearances.loc[
                plate_appearances[
                    "event"
                ].eq(
                    "home_run"
                ),
                "runs_scored",
            ] = 2

            with self.assertRaises(
                DerivedValidationError
            ):
                validate_score_chain(
                    plate_appearances,
                    fixture.frames[
                        "games"
                    ],
                )

    def test_pa_row_shuffle_does_not_change_last_pa_selection(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = build_fixture(
                Path(
                    directory
                )
            )

            shuffled = (
                fixture.frames[
                    "plate_appearances"
                ]
                .sample(
                    frac=1.0,
                    random_state=17,
                )
                .reset_index(
                    drop=True
                )
            )

            overwrite_table(
                fixture,
                "plate_appearances",
                shuffled,
            )

            run_fixture(
                fixture
            )


class BattingContractTest(
    unittest.TestCase
):
    """Batting Event/Formula/Season 계약을 검증한다."""

    def test_player_game_batting_event_mismatch_fails(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = build_fixture(
                Path(
                    directory
                )
            )

            batting = (
                fixture.frames[
                    "player_game_batting"
                ]
                .copy()
            )

            index = (
                batting[
                    "hr"
                ]
                .gt(0)
                .idxmax()
            )

            batting.loc[
                index,
                "hr",
            ] = 0

            overwrite_table(
                fixture,
                "player_game_batting",
                batting,
            )

            with self.assertRaises(
                DerivedValidationError
            ):
                run_fixture(
                    fixture
                )

    def test_zero_denominator_rate_is_nullable(
        self,
    ) -> None:
        numerator = pd.Series(
            [
                0,
                1,
            ],
            dtype="Int64",
        )

        denominator = pd.Series(
            [
                0,
                2,
            ],
            dtype="Int64",
        )

        result = calculate_rate(
            numerator,
            denominator,
        )

        self.assertTrue(
            pd.isna(
                result.iloc[
                    0
                ]
            )
        )

        self.assertEqual(
            float(
                result.iloc[
                    1
                ]
            ),
            0.5,
        )

    def test_season_batting_mismatch_fails(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = build_fixture(
                Path(
                    directory
                )
            )

            snapshot = (
                fixture.frames[
                    "player_season_batting_snapshot"
                ]
                .copy()
            )

            snapshot.loc[
                0,
                "pa",
            ] = (
                int(
                    snapshot.loc[
                        0,
                        "pa",
                    ]
                )
                + 1
            )

            overwrite_table(
                fixture,
                "player_season_batting_snapshot",
                snapshot,
            )

            with self.assertRaises(
                DerivedValidationError
            ):
                run_fixture(
                    fixture
                )

    def test_season_rate_does_not_use_game_rate_average(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = build_fixture(
                Path(
                    directory
                )
            )

            source = (
                fixture.frames[
                    "player_game_batting"
                ]
                .copy()
            )

            expected_before = (
                build_expected_season_batting(
                    source,
                    fixture.frames[
                        "team_games"
                    ],
                )
            )

            source[
                "avg"
            ] = 999.0

            source[
                "obp"
            ] = 999.0

            source[
                "slg"
            ] = 999.0

            source[
                "ops"
            ] = 999.0

            expected_after = (
                build_expected_season_batting(
                    source,
                    fixture.frames[
                        "team_games"
                    ],
                )
            )

            pd.testing.assert_frame_equal(
                expected_before,
                expected_after,
            )


class PitchingContractTest(
    unittest.TestCase
):
    """Raw Pitch/BF/Allowed Event/nullable Outs 계약을 검증한다."""

    def test_raw_pitch_count_mismatch_fails(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = build_fixture(
                Path(
                    directory
                )
            )

            pitching = (
                fixture.frames[
                    "player_game_pitching"
                ]
                .copy()
            )

            pitching.loc[
                0,
                "pitches",
            ] = (
                int(
                    pitching.loc[
                        0,
                        "pitches",
                    ]
                )
                + 1
            )

            overwrite_table(
                fixture,
                "player_game_pitching",
                pitching,
            )

            with self.assertRaises(
                DerivedValidationError
            ):
                run_fixture(
                    fixture
                )

    def test_bf_mismatch_fails(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = build_fixture(
                Path(
                    directory
                )
            )

            pitching = (
                fixture.frames[
                    "player_game_pitching"
                ]
                .copy()
            )

            pitching.loc[
                0,
                "batters_faced_completed",
            ] = (
                int(
                    pitching.loc[
                        0,
                        "batters_faced_completed",
                    ]
                )
                + 1
            )

            overwrite_table(
                fixture,
                "player_game_pitching",
                pitching,
            )

            with self.assertRaises(
                DerivedValidationError
            ):
                run_fixture(
                    fixture
                )

    def test_ambiguous_outs_must_not_be_replaced_with_zero(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = build_fixture(
                Path(
                    directory
                )
            )

            pitching = (
                fixture.frames[
                    "player_game_pitching"
                ]
                .copy()
            )

            ambiguous_index = (
                pitching[
                    "pitcher"
                ]
                .eq(
                    "P2A"
                )
            )

            self.assertTrue(
                pitching.loc[
                    ambiguous_index,
                    "outs_recorded",
                ]
                .isna()
                .all()
            )

            pitching.loc[
                ambiguous_index,
                "outs_recorded",
            ] = 0

            pitching[
                "outs_recorded"
            ] = (
                pitching[
                    "outs_recorded"
                ]
                .astype("Int64")
            )

            overwrite_table(
                fixture,
                "player_game_pitching",
                pitching,
            )

            with self.assertRaises(
                DerivedValidationError
            ):
                run_fixture(
                    fixture
                )

    def test_season_outs_null_propagation_must_be_preserved(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = build_fixture(
                Path(
                    directory
                )
            )

            snapshot = (
                fixture.frames[
                    "player_season_pitching_snapshot"
                ]
                .copy()
            )

            ambiguous_index = (
                snapshot[
                    "pitcher"
                ]
                .eq(
                    "P2A"
                )
            )

            self.assertTrue(
                snapshot.loc[
                    ambiguous_index,
                    "outs_recorded",
                ]
                .isna()
                .all()
            )

            snapshot.loc[
                ambiguous_index,
                "outs_recorded",
            ] = 0

            snapshot[
                "outs_recorded"
            ] = (
                snapshot[
                    "outs_recorded"
                ]
                .astype("Int64")
            )

            overwrite_table(
                fixture,
                "player_season_pitching_snapshot",
                snapshot,
            )

            with self.assertRaises(
                DerivedValidationError
            ):
                run_fixture(
                    fixture
                )


class PlayerAndSeasonContractTest(
    unittest.TestCase
):
    """Player Coverage, Team Season, through_date 계약을 검증한다."""

    def test_player_union_missing_player_fails(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = build_fixture(
                Path(
                    directory
                )
            )

            players = (
                fixture.frames[
                    "players"
                ]
                .iloc[
                    1:
                ]
                .reset_index(
                    drop=True
                )
            )

            overwrite_table(
                fixture,
                "players",
                players,
            )

            with self.assertRaises(
                DerivedValidationError
            ):
                run_fixture(
                    fixture
                )

    def test_player_role_mismatch_fails(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = build_fixture(
                Path(
                    directory
                )
            )

            players = (
                fixture.frames[
                    "players"
                ]
                .copy()
            )

            players.loc[
                0,
                "is_batter",
            ] = (
                not bool(
                    players.loc[
                        0,
                        "is_batter",
                    ]
                )
            )

            overwrite_table(
                fixture,
                "players",
                players,
            )

            with self.assertRaises(
                DerivedValidationError
            ):
                run_fixture(
                    fixture
                )

    def test_forbidden_player_team_column_fails_schema(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = build_fixture(
                Path(
                    directory
                )
            )

            players = (
                fixture.frames[
                    "players"
                ]
                .copy()
            )

            players[
                "team"
            ] = "INVALID"

            overwrite_table(
                fixture,
                "players",
                players,
            )

            with self.assertRaises(
                DerivedValidationError
            ):
                run_fixture(
                    fixture
                )

    def test_team_season_mismatch_fails(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = build_fixture(
                Path(
                    directory
                )
            )

            snapshot = (
                fixture.frames[
                    "team_season_snapshot"
                ]
                .copy()
            )

            snapshot.loc[
                0,
                "games",
            ] = (
                int(
                    snapshot.loc[
                        0,
                        "games",
                    ]
                )
                + 1
            )

            overwrite_table(
                fixture,
                "team_season_snapshot",
                snapshot,
            )

            with self.assertRaises(
                DerivedValidationError
            ):
                run_fixture(
                    fixture
                )

    def test_through_date_mismatch_fails(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = build_fixture(
                Path(
                    directory
                )
            )

            snapshot = (
                fixture.frames[
                    "team_season_snapshot"
                ]
                .copy()
            )

            snapshot.loc[
                :,
                "through_date",
            ] = (
                pd.Timestamp(
                    "2023-12-31"
                )
            )

            snapshot[
                "through_date"
            ] = (
                snapshot[
                    "through_date"
                ]
                .astype(
                    "datetime64[us]"
                )
            )

            overwrite_table(
                fixture,
                "team_season_snapshot",
                snapshot,
            )

            with self.assertRaises(
                DerivedValidationError
            ):
                run_fixture(
                    fixture
                )

    def test_2026_partial_snapshot_uses_source_max_date(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = build_fixture(
                Path(
                    directory
                ),
                season=2026,
                first_date="2026-04-03",
                second_date="2026-09-17",
            )

            summary = run_fixture(
                fixture
            )

            self.assertEqual(
                summary.through_dates[
                    2026
                ],
                "2026-09-17",
            )


class FingerprintAndSafetyTest(
    unittest.TestCase
):
    """Content Fingerprint와 Path/Git Ignore 정책을 검증한다."""

    def test_same_dataframe_has_same_fingerprint(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = build_fixture(
                Path(
                    directory
                )
            )

            games = (
                fixture.frames[
                    "games"
                ]
            )

            first = content_fingerprint(
                games,
                sort_columns=(
                    "game_pk",
                ),
            )

            second = content_fingerprint(
                games.copy(),
                sort_columns=(
                    "game_pk",
                ),
            )

            self.assertEqual(
                first,
                second,
            )

    def test_row_shuffle_has_same_fingerprint_after_canonical_sort(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = build_fixture(
                Path(
                    directory
                )
            )

            games = (
                fixture.frames[
                    "games"
                ]
            )

            shuffled = (
                games.sample(
                    frac=1.0,
                    random_state=31,
                )
                .reset_index(
                    drop=True
                )
            )

            self.assertEqual(
                content_fingerprint(
                    games,
                    sort_columns=(
                        "game_pk",
                    ),
                ),
                content_fingerprint(
                    shuffled,
                    sort_columns=(
                        "game_pk",
                    ),
                ),
            )

    def test_value_change_changes_fingerprint(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = build_fixture(
                Path(
                    directory
                )
            )

            games = (
                fixture.frames[
                    "games"
                ]
            )

            changed = (
                games.copy()
            )

            changed.loc[
                0,
                "total_runs",
            ] = (
                int(
                    changed.loc[
                        0,
                        "total_runs",
                    ]
                )
                + 1
            )

            self.assertNotEqual(
                content_fingerprint(
                    games,
                    sort_columns=(
                        "game_pk",
                    ),
                ),
                content_fingerprint(
                    changed,
                    sort_columns=(
                        "game_pk",
                    ),
                ),
            )

    def test_missing_gitignore_policy_fails(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = build_fixture(
                Path(
                    directory
                )
            )

            (
                fixture.project_root
                / ".gitignore"
            ).write_text(
                "*.pyc\n",
                encoding="utf-8",
            )

            with self.assertRaises(
                DerivedValidationError
            ):
                validate_derived_path_policy(
                    fixture.derived_dir,
                    project_root=(
                        fixture.project_root
                    ),
                )

    def test_derived_path_outside_interim_fails(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = (
                Path(
                    directory
                )
                / "project"
            )

            root.mkdir(
                parents=True,
                exist_ok=True,
            )

            (
                root
                / ".gitignore"
            ).write_text(
                "/data/interim/**\n",
                encoding="utf-8",
            )

            outside = (
                root
                / "derived"
            )

            with self.assertRaises(
                DerivedValidationError
            ):
                validate_derived_path_policy(
                    outside,
                    project_root=root,
                )


if __name__ == "__main__":
    unittest.main()