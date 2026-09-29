from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from pandas.testing import assert_frame_equal

PYARROW_AVAILABLE = importlib.util.find_spec("pyarrow") is not None


from scripts.build_team_features import (
    AUDIT_COLUMNS,
    EXPANDING_FEATURE_COLUMNS,
    FEATURE_VERSION,
    KEY_COLUMNS,
    ROLLING_WINDOWS,
    TeamFeatureBuildError,
    build_team_feature_files,
    build_team_pregame_features,
    calculate_sha256,
    content_fingerprint,
)


def make_game(
    *,
    game_pk: str,
    game_date: str,
    season: int = 2023,
    home_team: str = "A",
    away_team: str = "B",
    home_score: int = 1,
    away_score: int = 0,
    home_pa: int = 30,
    away_pa: int = 30,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """테스트용 games 1행과 mirror team_games 2행을 만든다."""
    home_win = home_score > away_score
    away_win = away_score > home_score
    tie = home_score == away_score

    game = {
        "game_pk": game_pk,
        "game_date": game_date,
        "season": season,
        "home_team": home_team,
        "away_team": away_team,
        "final_home_score": home_score,
        "final_away_score": away_score,
        "home_win": home_win,
        "away_win": away_win,
        "is_tie": tie,
    }
    team_rows = [
        {
            "game_pk": game_pk,
            "game_date": game_date,
            "season": season,
            "team": home_team,
            "opponent": away_team,
            "is_home": True,
            "runs_for": home_score,
            "runs_against": away_score,
            "run_diff": home_score - away_score,
            "win": home_win,
            "loss": away_win,
            "tie": tie,
            "plate_appearances": home_pa,
            "opponent_plate_appearances": away_pa,
        },
        {
            "game_pk": game_pk,
            "game_date": game_date,
            "season": season,
            "team": away_team,
            "opponent": home_team,
            "is_home": False,
            "runs_for": away_score,
            "runs_against": home_score,
            "run_diff": away_score - home_score,
            "win": away_win,
            "loss": home_win,
            "tie": tie,
            "plate_appearances": away_pa,
            "opponent_plate_appearances": home_pa,
        },
    ]
    return game, team_rows


def make_dataset(
    specs: list[dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """경기 사양 목록을 Canonical 최소 games/team_games fixture로 변환한다."""
    games: list[dict[str, object]] = []
    team_rows: list[dict[str, object]] = []
    for spec in specs:
        game, rows = make_game(**spec)
        games.append(game)
        team_rows.extend(rows)
    return pd.DataFrame(games), pd.DataFrame(team_rows)


def get_feature(
    frame: pd.DataFrame,
    *,
    game_pk: str,
    team: str,
) -> pd.Series:
    """특정 `(game_pk, team)` Feature Row를 유일하게 조회한다."""
    selected = frame.loc[
        frame["game_pk"].eq(game_pk)
        & frame["team"].eq(team)
    ]
    if len(selected) != 1:
        raise AssertionError(
            f"Feature Row가 유일하지 않습니다: game_pk={game_pk}, team={team}"
        )
    return selected.iloc[0]


def historical_columns(frame: pd.DataFrame) -> list[str]:
    """Leakage 불변성 비교에 사용할 Historical X Column 목록을 반환한다."""
    excluded = set(KEY_COLUMNS + AUDIT_COLUMNS)
    return [
        column
        for column in frame.columns
        if column not in excluded
    ]


class TeamFeatureCalculationTest(unittest.TestCase):
    """Team Pregame Feature의 핵심 산식과 시점 계약을 검증한다."""

    def test_hand_calculated_expanding_features(self) -> None:
        """수기 계산 가능한 W/L/T와 득실 집계가 예상값과 일치해야 한다."""
        games, team_games = make_dataset(
            [
                {
                    "game_pk": "G001",
                    "game_date": "2023-04-01",
                    "home_team": "A",
                    "away_team": "B",
                    "home_score": 3,
                    "away_score": 1,
                },
                {
                    "game_pk": "G002",
                    "game_date": "2023-04-02",
                    "home_team": "C",
                    "away_team": "A",
                    "home_score": 4,
                    "away_score": 2,
                },
                {
                    "game_pk": "G003",
                    "game_date": "2023-04-03",
                    "home_team": "A",
                    "away_team": "D",
                    "home_score": 2,
                    "away_score": 2,
                },
                {
                    "game_pk": "G004",
                    "game_date": "2023-04-05",
                    "home_team": "E",
                    "away_team": "A",
                    "home_score": 0,
                    "away_score": 1,
                },
            ]
        )

        features = build_team_pregame_features(games, team_games)
        row = get_feature(features, game_pk="G004", team="A")

        self.assertEqual(row["hist_games"], 3)
        self.assertEqual(row["hist_wins"], 1)
        self.assertEqual(row["hist_losses"], 1)
        self.assertEqual(row["hist_ties"], 1)
        self.assertEqual(row["hist_runs_for"], 7)
        self.assertEqual(row["hist_runs_against"], 7)
        self.assertEqual(row["hist_run_diff"], 0)
        self.assertAlmostEqual(float(row["hist_win_pct"]), 0.5)
        self.assertAlmostEqual(float(row["hist_runs_for_per_game"]), 7 / 3)
        self.assertAlmostEqual(float(row["hist_runs_against_per_game"]), 7 / 3)
        self.assertAlmostEqual(float(row["hist_run_diff_per_game"]), 0.0)
        self.assertEqual(row["days_since_last_observed_game"], 2)
        self.assertEqual(
            row["max_source_game_date"],
            pd.Timestamp("2023-04-03"),
        )
        self.assertLess(
            row["max_source_game_date"],
            row["prediction_date"],
        )

    def test_first_game_is_cold_start_without_dropping_row(self) -> None:
        """시즌 첫 경기는 count=0, rate=null, history flag=false로 보존되어야 한다."""
        games, team_games = make_dataset(
            [
                {
                    "game_pk": "G001",
                    "game_date": "2023-04-01",
                    "home_team": "A",
                    "away_team": "B",
                    "home_score": 0,
                    "away_score": 0,
                }
            ]
        )

        features = build_team_pregame_features(games, team_games)
        self.assertEqual(len(features), 2)
        for _, row in features.iterrows():
            self.assertEqual(row["hist_games"], 0)
            self.assertEqual(row["history_game_count"], 0)
            self.assertFalse(bool(row["has_history"]))
            self.assertTrue(pd.isna(row["max_source_game_date"]))
            self.assertTrue(pd.isna(row["hist_win_pct"]))
            self.assertTrue(pd.isna(row["hist_runs_for_per_game"]))
            self.assertTrue(pd.isna(row["days_since_last_observed_game"]))
            for window in ROLLING_WINDOWS:
                self.assertEqual(row[f"last_{window}g_games"], 0)
                self.assertTrue(pd.isna(row[f"last_{window}g_win_pct"]))

    def test_tie_only_history_keeps_zero_denominator_rate_null(self) -> None:
        """무승부만 있는 이력에서 승률 분모를 0으로 유지하고 null을 반환해야 한다."""
        games, team_games = make_dataset(
            [
                {
                    "game_pk": "G001",
                    "game_date": "2023-04-01",
                    "home_team": "A",
                    "away_team": "B",
                    "home_score": 2,
                    "away_score": 2,
                },
                {
                    "game_pk": "G002",
                    "game_date": "2023-04-02",
                    "home_team": "A",
                    "away_team": "C",
                    "home_score": 1,
                    "away_score": 0,
                },
            ]
        )

        features = build_team_pregame_features(games, team_games)
        row = get_feature(features, game_pk="G002", team="A")
        self.assertEqual(row["hist_games"], 1)
        self.assertEqual(row["hist_ties"], 1)
        self.assertEqual(row["hist_wins"], 0)
        self.assertEqual(row["hist_losses"], 0)
        self.assertTrue(pd.isna(row["hist_win_pct"]))

    def test_season_history_is_reset(self) -> None:
        """새 시즌 첫 경기에 이전 시즌 누적을 자동 이월하지 않아야 한다."""
        games, team_games = make_dataset(
            [
                {
                    "game_pk": "S23",
                    "game_date": "2023-10-01",
                    "season": 2023,
                    "home_team": "A",
                    "away_team": "B",
                    "home_score": 5,
                    "away_score": 1,
                },
                {
                    "game_pk": "S24",
                    "game_date": "2024-03-23",
                    "season": 2024,
                    "home_team": "A",
                    "away_team": "C",
                    "home_score": 0,
                    "away_score": 1,
                },
            ]
        )

        features = build_team_pregame_features(games, team_games)
        row = get_feature(features, game_pk="S24", team="A")
        self.assertEqual(row["hist_games"], 0)
        self.assertFalse(bool(row["has_history"]))

    def test_same_date_games_do_not_use_each_other(self) -> None:
        """같은 날짜 두 경기는 game_pk 순서와 무관하게 서로의 결과를 Source로 쓰지 않아야 한다."""
        games, team_games = make_dataset(
            [
                {
                    "game_pk": "PREV",
                    "game_date": "2023-04-01",
                    "home_team": "A",
                    "away_team": "B",
                    "home_score": 3,
                    "away_score": 0,
                },
                {
                    "game_pk": "DH1",
                    "game_date": "2023-04-02",
                    "home_team": "A",
                    "away_team": "C",
                    "home_score": 10,
                    "away_score": 0,
                },
                {
                    "game_pk": "DH2",
                    "game_date": "2023-04-02",
                    "home_team": "D",
                    "away_team": "A",
                    "home_score": 0,
                    "away_score": 8,
                },
            ]
        )

        features = build_team_pregame_features(games, team_games)
        first = get_feature(features, game_pk="DH1", team="A")
        second = get_feature(features, game_pk="DH2", team="A")
        columns = historical_columns(features)
        self.assertEqual(first["hist_games"], 1)
        self.assertEqual(second["hist_games"], 1)
        for column in columns:
            left = first[column]
            right = second[column]
            if pd.isna(left) and pd.isna(right):
                continue
            self.assertEqual(left, right, column)

    def test_rolling_5_10_20_exact_boundaries(self) -> None:
        """동일 날짜가 없는 충분한 이력에서는 5/10/20경기 window가 정확히 N행이어야 한다."""
        specs: list[dict[str, object]] = []
        base = pd.Timestamp("2023-04-01")
        for index in range(21):
            specs.append(
                {
                    "game_pk": f"H{index:02d}",
                    "game_date": (base + pd.Timedelta(days=index)).strftime("%Y-%m-%d"),
                    "home_team": "A" if index % 2 == 0 else f"O{index:02d}",
                    "away_team": f"O{index:02d}" if index % 2 == 0 else "A",
                    "home_score": 2 if index % 3 == 0 else 1,
                    "away_score": 1 if index % 3 == 0 else 2,
                }
            )
        specs.append(
            {
                "game_pk": "TARGET",
                "game_date": (base + pd.Timedelta(days=21)).strftime("%Y-%m-%d"),
                "home_team": "A",
                "away_team": "Z",
                "home_score": 1,
                "away_score": 0,
            }
        )
        games, team_games = make_dataset(specs)

        features = build_team_pregame_features(games, team_games)
        row = get_feature(features, game_pk="TARGET", team="A")
        self.assertEqual(row["hist_games"], 21)
        self.assertEqual(row["last_5g_games"], 5)
        self.assertEqual(row["last_10g_games"], 10)
        self.assertEqual(row["last_20g_games"], 20)

    def test_rolling_boundary_includes_entire_same_date_group(self) -> None:
        """5경기 경계 날짜에 더블헤더가 걸리면 같은 날짜 묶음을 쪼개지 않아야 한다."""
        specs = [
            {
                "game_pk": "DHA",
                "game_date": "2023-04-01",
                "home_team": "A",
                "away_team": "B",
                "home_score": 1,
                "away_score": 0,
            },
            {
                "game_pk": "DHB",
                "game_date": "2023-04-01",
                "home_team": "C",
                "away_team": "A",
                "home_score": 0,
                "away_score": 2,
            },
        ]
        for index, date in enumerate(
            ["2023-04-02", "2023-04-03", "2023-04-04", "2023-04-05"],
            start=2,
        ):
            specs.append(
                {
                    "game_pk": f"G{index}",
                    "game_date": date,
                    "home_team": "A",
                    "away_team": f"O{index}",
                    "home_score": 1,
                    "away_score": 0,
                }
            )
        specs.append(
            {
                "game_pk": "TARGET",
                "game_date": "2023-04-06",
                "home_team": "A",
                "away_team": "Z",
                "home_score": 0,
                "away_score": 1,
            }
        )
        games, team_games = make_dataset(specs)

        features = build_team_pregame_features(games, team_games)
        row = get_feature(features, game_pk="TARGET", team="A")
        self.assertEqual(row["hist_games"], 6)
        self.assertEqual(row["last_5g_games"], 6)
        self.assertEqual(row["last_10g_games"], 6)
        self.assertEqual(row["last_20g_games"], 6)


class TeamFeatureLeakageTest(unittest.TestCase):
    """현재·같은 날짜·미래 결과가 과거 Pregame X로 유입되지 않는지 검증한다."""

    def test_current_game_result_change_does_not_change_current_pregame_feature(self) -> None:
        """현재 경기 결과를 바꿔도 해당 경기의 Pregame X는 동일해야 한다."""
        base_specs = [
            {
                "game_pk": "PREV",
                "game_date": "2023-04-01",
                "home_team": "A",
                "away_team": "B",
                "home_score": 3,
                "away_score": 1,
            },
            {
                "game_pk": "TARGET",
                "game_date": "2023-04-02",
                "home_team": "A",
                "away_team": "C",
                "home_score": 1,
                "away_score": 0,
            },
        ]
        changed_specs = [dict(spec) for spec in base_specs]
        changed_specs[1]["home_score"] = 0
        changed_specs[1]["away_score"] = 9
        games_a, team_games_a = make_dataset(base_specs)
        games_b, team_games_b = make_dataset(changed_specs)

        feature_a = build_team_pregame_features(games_a, team_games_a)
        feature_b = build_team_pregame_features(games_b, team_games_b)
        row_a = get_feature(feature_a, game_pk="TARGET", team="A")
        row_b = get_feature(feature_b, game_pk="TARGET", team="A")
        for column in historical_columns(feature_a):
            left = row_a[column]
            right = row_b[column]
            if pd.isna(left) and pd.isna(right):
                continue
            self.assertEqual(left, right, column)

    def test_same_date_other_game_result_change_does_not_change_that_dates_features(self) -> None:
        """같은 날짜 다른 경기 결과 변조가 그 날짜의 Pregame X를 바꾸지 않아야 한다."""
        specs = [
            {
                "game_pk": "PREV",
                "game_date": "2023-04-01",
                "home_team": "A",
                "away_team": "B",
                "home_score": 2,
                "away_score": 0,
            },
            {
                "game_pk": "DH1",
                "game_date": "2023-04-02",
                "home_team": "A",
                "away_team": "C",
                "home_score": 1,
                "away_score": 0,
            },
            {
                "game_pk": "DH2",
                "game_date": "2023-04-02",
                "home_team": "D",
                "away_team": "A",
                "home_score": 0,
                "away_score": 1,
            },
        ]
        changed = [dict(spec) for spec in specs]
        changed[1]["home_score"] = 0
        changed[1]["away_score"] = 12
        games_a, team_games_a = make_dataset(specs)
        games_b, team_games_b = make_dataset(changed)

        feature_a = build_team_pregame_features(games_a, team_games_a)
        feature_b = build_team_pregame_features(games_b, team_games_b)
        for game_pk in ("DH1", "DH2"):
            row_a = get_feature(feature_a, game_pk=game_pk, team="A")
            row_b = get_feature(feature_b, game_pk=game_pk, team="A")
            for column in historical_columns(feature_a):
                left = row_a[column]
                right = row_b[column]
                if pd.isna(left) and pd.isna(right):
                    continue
                self.assertEqual(left, right, f"{game_pk}:{column}")

    def test_future_game_change_or_addition_does_not_change_past_feature(self) -> None:
        """미래 경기 추가·변조가 이전 cutoff 시점의 Feature를 바꾸지 않아야 한다."""
        past_specs = [
            {
                "game_pk": "PREV",
                "game_date": "2023-04-01",
                "home_team": "A",
                "away_team": "B",
                "home_score": 4,
                "away_score": 1,
            },
            {
                "game_pk": "TARGET",
                "game_date": "2023-04-02",
                "home_team": "A",
                "away_team": "C",
                "home_score": 0,
                "away_score": 1,
            },
        ]
        future_specs = past_specs + [
            {
                "game_pk": "FUTURE",
                "game_date": "2023-04-10",
                "home_team": "D",
                "away_team": "A",
                "home_score": 99,
                "away_score": 0,
            }
        ]
        games_a, team_games_a = make_dataset(past_specs)
        games_b, team_games_b = make_dataset(future_specs)

        feature_a = build_team_pregame_features(games_a, team_games_a)
        feature_b = build_team_pregame_features(games_b, team_games_b)
        row_a = get_feature(feature_a, game_pk="TARGET", team="A")
        row_b = get_feature(feature_b, game_pk="TARGET", team="A")
        for column in historical_columns(feature_a):
            left = row_a[column]
            right = row_b[column]
            if pd.isna(left) and pd.isna(right):
                continue
            self.assertEqual(left, right, column)


class TeamFeatureValidationTest(unittest.TestCase):
    """입력 경계, 결정성, dtype, 파일 I/O 안전성을 검증한다."""

    def setUp(self) -> None:
        """각 검증 테스트에서 사용할 소규모 정상 fixture를 준비한다."""
        self.games, self.team_games = make_dataset(
            [
                {
                    "game_pk": "G001",
                    "game_date": "2023-04-01",
                    "home_team": "A",
                    "away_team": "B",
                    "home_score": 3,
                    "away_score": 1,
                },
                {
                    "game_pk": "G002",
                    "game_date": "2023-04-02",
                    "home_team": "C",
                    "away_team": "A",
                    "home_score": 0,
                    "away_score": 2,
                },
            ]
        )

    def test_duplicate_team_game_key_fails(self) -> None:
        """`(game_pk, team)` 중복 입력은 실패해야 한다."""
        duplicated = pd.concat(
            [self.team_games, self.team_games.iloc[[0]]],
            ignore_index=True,
        )
        with self.assertRaises(TeamFeatureBuildError):
            build_team_pregame_features(self.games, duplicated)

    def test_empty_input_fails(self) -> None:
        """빈 입력을 정상 Feature로 해석하지 않아야 한다."""
        with self.assertRaises(TeamFeatureBuildError):
            build_team_pregame_features(
                self.games.iloc[0:0].copy(),
                self.team_games.iloc[0:0].copy(),
            )

    def test_invalid_game_team_relationship_fails(self) -> None:
        """opponent 등 Game/Team Game mirror 관계가 깨지면 실패해야 한다."""
        invalid = self.team_games.copy()
        invalid.loc[0, "opponent"] = "WRONG"
        with self.assertRaises(TeamFeatureBuildError):
            build_team_pregame_features(self.games, invalid)

    def test_invalid_date_fails(self) -> None:
        """해석할 수 없는 game_date를 허용하지 않아야 한다."""
        invalid_games = self.games.copy()
        invalid_games.loc[0, "game_date"] = "not-a-date"
        with self.assertRaises(TeamFeatureBuildError):
            build_team_pregame_features(invalid_games, self.team_games)

    def test_shuffled_input_is_deterministic(self) -> None:
        """입력 Row 순서를 섞어도 값·정렬·dtype·fingerprint가 동일해야 한다."""
        ordered = build_team_pregame_features(
            self.games,
            self.team_games,
        )
        shuffled = build_team_pregame_features(
            self.games.sample(frac=1, random_state=17).reset_index(drop=True),
            self.team_games.sample(frac=1, random_state=29).reset_index(drop=True),
        )
        assert_frame_equal(ordered, shuffled)
        self.assertEqual(
            content_fingerprint(ordered),
            content_fingerprint(shuffled),
        )

    def test_output_dtypes_are_normalized_in_memory(self) -> None:
        """Parquet 엔진과 무관하게 계산 결과 dtype 계약을 검증한다."""
        features = build_team_pregame_features(self.games, self.team_games)
        self.assertEqual(str(features["game_pk"].dtype), "string")
        self.assertEqual(str(features["season"].dtype), "Int64")
        self.assertEqual(str(features["hist_win_pct"].dtype), "Float64")
        self.assertEqual(str(features["has_history"].dtype), "boolean")
        self.assertEqual(str(features["prediction_date"].dtype), "datetime64[us]")
        self.assertEqual(
            str(features["max_source_game_date"].dtype),
            "datetime64[us]",
        )

    @unittest.skipUnless(
        PYARROW_AVAILABLE,
        "실행 환경에 pyarrow가 없어 Parquet round-trip을 검증할 수 없습니다.",
    )
    def test_output_dtype_and_parquet_round_trip(self) -> None:
        """파일 Builder가 dtype, hash 불변성, manifest와 round-trip을 보존해야 한다."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "inputs"
            output_dir = root / "data" / "processed" / "features"
            input_dir.mkdir(parents=True)
            games_path = input_dir / "games.parquet"
            team_games_path = input_dir / "team_games.parquet"
            output_path = output_dir / "team_pregame_features.parquet"
            self.games.to_parquet(games_path, engine="pyarrow", index=False)
            self.team_games.to_parquet(team_games_path, engine="pyarrow", index=False)
            games_hash_before = calculate_sha256(games_path)
            team_games_hash_before = calculate_sha256(team_games_path)

            features, manifest, manifest_path = build_team_feature_files(
                games_path=games_path,
                team_games_path=team_games_path,
                output_path=output_path,
                project_root=root,
            )

            self.assertTrue(output_path.is_file())
            self.assertTrue(manifest_path.is_file())
            self.assertEqual(str(features["game_pk"].dtype), "string")
            self.assertEqual(str(features["season"].dtype), "Int64")
            self.assertEqual(str(features["hist_win_pct"].dtype), "Float64")
            self.assertEqual(str(features["has_history"].dtype), "boolean")
            self.assertEqual(str(features["prediction_date"].dtype), "datetime64[us]")
            self.assertEqual(
                str(features["max_source_game_date"].dtype),
                "datetime64[us]",
            )
            self.assertEqual(
                games_hash_before,
                calculate_sha256(games_path),
            )
            self.assertEqual(
                team_games_hash_before,
                calculate_sha256(team_games_path),
            )
            self.assertEqual(
                manifest["output"]["content_fingerprint"],
                content_fingerprint(features),
            )
            self.assertEqual(manifest["feature_version"], FEATURE_VERSION)
            self.assertEqual(manifest["rolling_windows"], [5, 10, 20])
            self.assertEqual(
                manifest["historical_cutoff"],
                "source.game_date < prediction_date",
            )
            with manifest_path.open("r", encoding="utf-8") as file:
                persisted = json.load(file)
            self.assertEqual(persisted, manifest)

    def test_output_path_collision_with_input_fails(self) -> None:
        """Output이 Canonical 입력 파일을 덮어쓰려 하면 쓰기 전에 실패해야 한다."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "inputs"
            input_dir.mkdir(parents=True)
            games_path = input_dir / "games.parquet"
            team_games_path = input_dir / "team_games.parquet"
            # 경로 충돌 검증은 Parquet 파싱 전에 실행되어야 하므로
            # 실제 Parquet 엔진 없이도 일반 파일로 안전하게 검증할 수 있다.
            games_path.write_bytes(b"games-input")
            team_games_path.write_bytes(b"team-games-input")
            original_hash = calculate_sha256(team_games_path)

            with self.assertRaises(TeamFeatureBuildError):
                build_team_feature_files(
                    games_path=games_path,
                    team_games_path=team_games_path,
                    output_path=team_games_path,
                    manifest_path=root / "manifest.json",
                    project_root=root,
                )

            self.assertEqual(
                original_hash,
                calculate_sha256(team_games_path),
            )

    def test_output_schema_separates_key_audit_and_historical_features(self) -> None:
        """Output에는 Key/Audit와 명시적 Historical Feature가 계약 순서로 존재해야 한다."""
        features = build_team_pregame_features(self.games, self.team_games)
        expected_prefix = list(KEY_COLUMNS + AUDIT_COLUMNS + EXPANDING_FEATURE_COLUMNS)
        self.assertEqual(
            list(features.columns[: len(expected_prefix)]),
            expected_prefix,
        )
        self.assertNotIn("runs_for", features.columns)
        self.assertNotIn("runs_against", features.columns)
        self.assertNotIn("win", features.columns)
        self.assertNotIn("loss", features.columns)
        self.assertNotIn("tie", features.columns)


if __name__ == "__main__":
    unittest.main()
