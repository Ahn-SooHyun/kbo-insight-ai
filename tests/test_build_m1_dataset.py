from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from pandas.testing import assert_frame_equal

from src.data.m1_dataset import (
    FORBIDDEN_X_COLUMNS,
    FORBIDDEN_X_PREFIXES,
    M1_DIFFERENCE_COLUMNS,
    M1DatasetBuildError,
    TEAM_SOURCE_X_COLUMNS,
    build_deterministic_identity,
    build_m1_dataset,
    build_quality_report,
    build_schema_payload,
)
from src.data.processed_contract import (
    ProcessedContractError,
    assert_input_hashes_unchanged,
    calculate_input_hashes,
    calculate_sha256,
    content_fingerprint,
    invalidate_completion_marker,
    stable_json_fingerprint,
    validate_label_interval,
    validate_output_paths,
    validate_x_allowlist,
)


PYARROW_AVAILABLE = importlib.util.find_spec("pyarrow") is not None
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "processed_dataset.json"

TEAM_X_COLUMNS = TEAM_SOURCE_X_COLUMNS


def load_config() -> dict[str, object]:
    """테스트용 current-main M1 계약 설정을 읽는다."""
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def make_game(
    *,
    game_pk: str,
    game_date: str,
    season: int,
    home_team: str,
    away_team: str,
    home_score: int | None,
    away_score: int | None,
) -> dict[str, object]:
    """Canonical games 최소 fixture 1행을 만든다."""
    score_known = home_score is not None and away_score is not None
    return {
        "game_pk": game_pk,
        "game_date": game_date,
        "season": season,
        "home_team": home_team,
        "away_team": away_team,
        "final_home_score": home_score,
        "final_away_score": away_score,
        "home_win": (home_score > away_score) if score_known else pd.NA,
        "away_win": (away_score > home_score) if score_known else pd.NA,
        "is_tie": (home_score == away_score) if score_known else pd.NA,
    }


def make_team_x_values(
    *,
    hist_games: int,
    hist_win_pct: float | None,
) -> dict[str, object]:
    """current-main #23 X allowlist 전체에 대한 synthetic nullable 값을 만든다."""
    values: dict[str, object] = {}
    for column in TEAM_X_COLUMNS:
        if column.endswith(("win_pct", "per_game")):
            values[column] = hist_win_pct
        elif column == "days_since_last_observed_game":
            values[column] = 1 if hist_games > 0 else pd.NA
        elif column.endswith("games") or column == "hist_games":
            values[column] = hist_games
        else:
            values[column] = hist_games
    return values


def make_team_rows(
    game: dict[str, object],
    *,
    home_hist_games: int = 3,
    away_hist_games: int = 4,
    home_hist_win_pct: float | None = 0.5,
    away_hist_win_pct: float | None = 0.75,
) -> list[dict[str, object]]:
    """#23 Team Pregame Feature의 Home/Away mirror fixture를 만든다."""
    game_date = pd.Timestamp(str(game["game_date"]))
    source_date = game_date - pd.Timedelta(days=1)
    common_home = make_team_x_values(
        hist_games=home_hist_games, hist_win_pct=home_hist_win_pct
    )
    common_away = make_team_x_values(
        hist_games=away_hist_games, hist_win_pct=away_hist_win_pct
    )
    return [
        {
            "game_pk": game["game_pk"],
            "team": game["home_team"],
            "opponent": game["away_team"],
            "prediction_date": game["game_date"],
            "season": game["season"],
            "is_home": True,
            "max_source_game_date": source_date,
            "history_game_count": home_hist_games,
            "has_history": home_hist_games > 0,
            "feature_version": "team_pregame_v1",
            **common_home,
        },
        {
            "game_pk": game["game_pk"],
            "team": game["away_team"],
            "opponent": game["home_team"],
            "prediction_date": game["game_date"],
            "season": game["season"],
            "is_home": False,
            "max_source_game_date": source_date,
            "history_game_count": away_hist_games,
            "has_history": away_hist_games > 0,
            "feature_version": "team_pregame_v1",
            **common_away,
        },
    ]


def normalize_team_feature_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    """#23 Output과 같은 nullable dtype으로 synthetic Team Feature를 정규화한다."""
    result = frame.copy()
    for column in ("game_pk", "team", "opponent", "feature_version"):
        result[column] = result[column].astype("string")
    result["prediction_date"] = pd.to_datetime(result["prediction_date"]).astype("datetime64[us]")
    result["max_source_game_date"] = pd.to_datetime(result["max_source_game_date"]).astype("datetime64[us]")
    for column in ("season", "history_game_count"):
        result[column] = result[column].astype("Int64")
    for column in TEAM_X_COLUMNS:
        if column.endswith(("win_pct", "per_game")):
            result[column] = result[column].astype("Float64")
        else:
            result[column] = result[column].astype("Int64")
    for column in ("is_home", "has_history"):
        result[column] = result[column].astype("boolean")
    return result


def make_manifest(row_count: int) -> dict[str, object]:
    """#23 Builder가 기록하는 역할 계약을 최소 형태로 만든다."""
    return {
        "artifact": "team_pregame_features",
        "feature_version": "team_pregame_v1",
        "source_contract_version": "feature_catalog_v1",
        "historical_cutoff": "source.game_date < prediction_date",
        "same_day_policy": "exclude_all_same_date_source_results",
        "season_reset": True,
        "rolling_windows": [5, 10, 20],
        "grain": ["game_pk", "team"],
        "key_columns": ["game_pk", "team"],
        "audit_columns": [
            "opponent",
            "prediction_date",
            "season",
            "is_home",
            "max_source_game_date",
            "history_game_count",
            "has_history",
            "feature_version",
        ],
        "x_columns": list(TEAM_X_COLUMNS),
        "output": {
            "row_count": row_count,
            "content_fingerprint": "fixture",
        },
    }


def make_pa(game: dict[str, object], *, pa_completed: bool = True) -> dict[str, object]:
    """Terminal score audit에 필요한 최소 PA 1행을 만든다."""
    return {
        "game_pk": game["game_pk"],
        "at_bat_number": 1,
        "post_home_score": game["final_home_score"],
        "post_away_score": game["final_away_score"],
        "pa_completed": pa_completed,
    }


def build_fixture(
    games: list[dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], pd.DataFrame]:
    """games 목록에서 M1 Builder synthetic 입력 묶음을 만든다."""
    team_rows: list[dict[str, object]] = []
    pa_rows: list[dict[str, object]] = []
    for index, game in enumerate(games):
        team_rows.extend(
            make_team_rows(
                game,
                home_hist_games=2 + index,
                away_hist_games=5 + index,
            )
        )
        pa_rows.append(make_pa(game))
    team_features = normalize_team_feature_dtypes(pd.DataFrame(team_rows))
    return (
        pd.DataFrame(games),
        team_features,
        make_manifest(len(team_features)),
        pd.DataFrame(pa_rows),
    )


class M1DatasetJoinAndTargetTest(unittest.TestCase):
    """Home/Away Join, Target, Quality 계약을 검증한다."""

    def setUp(self) -> None:
        self.config = load_config()

    def test_normal_home_away_join_preserves_direction(self) -> None:
        game = make_game(
            game_pk="G1",
            game_date="2023-04-01",
            season=2023,
            home_team="A",
            away_team="B",
            home_score=3,
            away_score=1,
        )
        games, features, manifest, pa = build_fixture([game])
        dataset, roles, _ = build_m1_dataset(
            games,
            features,
            team_feature_manifest=manifest,
            config=self.config,
            plate_appearances=pa,
        )
        row = dataset.iloc[0]
        self.assertEqual(row["home_hist_games"], 2)
        self.assertEqual(row["away_hist_games"], 5)
        self.assertIn("home_hist_games", roles["x_columns"])
        self.assertIn("away_hist_games", roles["x_columns"])
        self.assertIn("home_has_history", roles["x_columns"])
        self.assertIn("away_has_history", roles["x_columns"])
        for column in M1_DIFFERENCE_COLUMNS:
            self.assertIn(column, roles["x_columns"])
        self.assertEqual(len(roles["x_columns"]), 96)
        self.assertAlmostEqual(float(row["diff_hist_win_pct"]), -0.25)
        self.assertTrue(bool(row["home_has_history"]))
        self.assertTrue(bool(row["away_has_history"]))
        self.assertEqual(len(dataset), 1)

    def test_wrong_home_away_direction_fails(self) -> None:
        game = make_game(
            game_pk="G1",
            game_date="2023-04-01",
            season=2023,
            home_team="A",
            away_team="B",
            home_score=3,
            away_score=1,
        )
        games, features, manifest, pa = build_fixture([game])
        features.loc[features["team"].eq("A"), "is_home"] = False
        with self.assertRaises(M1DatasetBuildError):
            build_m1_dataset(
                games,
                features,
                team_feature_manifest=manifest,
                config=self.config,
                plate_appearances=pa,
            )

    def test_duplicate_team_feature_fails(self) -> None:
        game = make_game(
            game_pk="G1",
            game_date="2023-04-01",
            season=2023,
            home_team="A",
            away_team="B",
            home_score=3,
            away_score=1,
        )
        games, features, manifest, pa = build_fixture([game])
        duplicated = pd.concat([features, features.iloc[[0]]], ignore_index=True)
        with self.assertRaises(M1DatasetBuildError):
            build_m1_dataset(
                games,
                duplicated,
                team_feature_manifest=manifest,
                config=self.config,
                plate_appearances=pa,
            )

    def test_missing_team_feature_fails(self) -> None:
        game = make_game(
            game_pk="G1",
            game_date="2023-04-01",
            season=2023,
            home_team="A",
            away_team="B",
            home_score=3,
            away_score=1,
        )
        games, features, manifest, pa = build_fixture([game])
        missing = features.loc[~features["team"].eq("B")].copy()
        with self.assertRaises(M1DatasetBuildError):
            build_m1_dataset(
                games,
                missing,
                team_feature_manifest=manifest,
                config=self.config,
                plate_appearances=pa,
            )

    def test_orphan_team_feature_fails(self) -> None:
        game = make_game(
            game_pk="G1",
            game_date="2023-04-01",
            season=2023,
            home_team="A",
            away_team="B",
            home_score=3,
            away_score=1,
        )
        games, features, manifest, pa = build_fixture([game])
        orphan = features.iloc[[0]].copy()
        orphan["game_pk"] = "ORPHAN"
        orphan["team"] = "X"
        orphan["opponent"] = "Y"
        features = pd.concat([features, orphan], ignore_index=True)
        with self.assertRaises(M1DatasetBuildError):
            build_m1_dataset(
                games,
                features,
                team_feature_manifest=manifest,
                config=self.config,
                plate_appearances=pa,
            )

    def test_wrong_opponent_relation_fails(self) -> None:
        game = make_game(
            game_pk="G1",
            game_date="2023-04-01",
            season=2023,
            home_team="A",
            away_team="B",
            home_score=3,
            away_score=1,
        )
        games, features, manifest, pa = build_fixture([game])
        features.loc[features["team"].eq("A"), "opponent"] = "WRONG"
        with self.assertRaises(M1DatasetBuildError):
            build_m1_dataset(
                games,
                features,
                team_feature_manifest=manifest,
                config=self.config,
                plate_appearances=pa,
            )

    def test_home_win_target(self) -> None:
        game = make_game(
            game_pk="W",
            game_date="2023-04-01",
            season=2023,
            home_team="A",
            away_team="B",
            home_score=5,
            away_score=2,
        )
        games, features, manifest, pa = build_fixture([game])
        dataset, _, _ = build_m1_dataset(
            games, features, team_feature_manifest=manifest, config=self.config, plate_appearances=pa
        )
        self.assertEqual(dataset.loc[0, "home_result"], "win")
        self.assertEqual(dataset.loc[0, "home_result_code"], 2)

    def test_away_win_target(self) -> None:
        game = make_game(
            game_pk="L",
            game_date="2023-04-01",
            season=2023,
            home_team="A",
            away_team="B",
            home_score=1,
            away_score=4,
        )
        games, features, manifest, pa = build_fixture([game])
        dataset, _, _ = build_m1_dataset(
            games, features, team_feature_manifest=manifest, config=self.config, plate_appearances=pa
        )
        self.assertEqual(dataset.loc[0, "home_result"], "loss")
        self.assertEqual(dataset.loc[0, "home_result_code"], 0)

    def test_tie_target(self) -> None:
        game = make_game(
            game_pk="T",
            game_date="2023-04-01",
            season=2023,
            home_team="A",
            away_team="B",
            home_score=2,
            away_score=2,
        )
        games, features, manifest, pa = build_fixture([game])
        dataset, _, _ = build_m1_dataset(
            games, features, team_feature_manifest=manifest, config=self.config, plate_appearances=pa
        )
        self.assertEqual(dataset.loc[0, "home_result"], "tie")
        self.assertEqual(dataset.loc[0, "home_result_code"], 1)

    def test_terminal_score_reconciliation_excludes_mismatch(self) -> None:
        game = make_game(
            game_pk="G1",
            game_date="2023-04-01",
            season=2023,
            home_team="A",
            away_team="B",
            home_score=3,
            away_score=1,
        )
        games, features, manifest, pa = build_fixture([game])
        pa.loc[0, "post_home_score"] = 2
        dataset, _, summary = build_m1_dataset(
            games, features, team_feature_manifest=manifest, config=self.config, plate_appearances=pa
        )
        self.assertTrue(bool(dataset.loc[0, "is_excluded"]))
        self.assertEqual(dataset.loc[0, "observation_quality"], "invalid_reconciliation")
        self.assertEqual(dataset.loc[0, "exclusion_reason"], "terminal_pa_score_reconciliation")
        self.assertTrue(pd.isna(dataset.loc[0, "home_result"]))
        self.assertTrue(pd.isna(dataset.loc[0, "home_result_code"]))
        self.assertTrue(pd.isna(dataset.loc[0, "home_runs"]))
        self.assertTrue(pd.isna(dataset.loc[0, "away_runs"]))
        self.assertEqual(summary["excluded"], 1)

    def test_missing_terminal_score_is_excluded_and_target_null(self) -> None:
        """필수 observed terminal score 결측은 제외하고 y를 nullable null로 유지해야 한다."""
        game = make_game(
            game_pk="G1",
            game_date="2023-04-01",
            season=2023,
            home_team="A",
            away_team="B",
            home_score=None,
            away_score=1,
        )
        games, features, manifest, _ = build_fixture([game])
        dataset, _, summary = build_m1_dataset(
            games,
            features,
            team_feature_manifest=manifest,
            config=self.config,
            plate_appearances=None,
        )
        self.assertTrue(bool(dataset.loc[0, "is_excluded"]))
        self.assertEqual(dataset.loc[0, "observation_quality"], "invalid_missing_score")
        self.assertTrue(pd.isna(dataset.loc[0, "home_result"]))
        self.assertTrue(pd.isna(dataset.loc[0, "home_runs"]))
        self.assertEqual(summary["excluded"], 1)

    def test_official_status_is_explicitly_unverified(self) -> None:
        """Source에 공식 경기 status가 없다는 한계를 Quality Audit으로 명시해야 한다."""
        game = make_game(
            game_pk="G1",
            game_date="2023-04-01",
            season=2023,
            home_team="A",
            away_team="B",
            home_score=3,
            away_score=1,
        )
        games, features, manifest, pa = build_fixture([game])
        dataset, _, _ = build_m1_dataset(
            games,
            features,
            team_feature_manifest=manifest,
            config=self.config,
            plate_appearances=pa,
        )
        self.assertFalse(bool(dataset.loc[0, "official_status_verified"]))
        self.assertEqual(
            dataset.loc[0, "target_semantics"],
            "canonical_observed_terminal",
        )

    def test_terminal_pa_incomplete_is_warning_not_exclusion(self) -> None:
        game = make_game(
            game_pk="G1",
            game_date="2023-04-01",
            season=2023,
            home_team="A",
            away_team="B",
            home_score=3,
            away_score=1,
        )
        games, features, manifest, pa = build_fixture([game])
        pa.loc[0, "pa_completed"] = False
        dataset, _, _ = build_m1_dataset(
            games, features, team_feature_manifest=manifest, config=self.config, plate_appearances=pa
        )
        self.assertFalse(bool(dataset.loc[0, "is_excluded"]))
        self.assertEqual(
            dataset.loc[0, "observation_quality"],
            "warning_terminal_pa_incomplete",
        )
        self.assertTrue(bool(dataset.loc[0, "target_eligible"]))


class M1DatasetSplitLeakageAndDeterminismTest(unittest.TestCase):
    """Split, Leakage Counterexample, 결정성 계약을 검증한다."""

    def setUp(self) -> None:
        self.config = load_config()

    def test_split_boundaries(self) -> None:
        specs = [
            ("TR", "2023-12-31", 2023),
            ("VA", "2024-01-01", 2024),
            ("TE", "2025-01-01", 2025),
            ("SN", "2026-01-01", 2026),
        ]
        games = [
            make_game(
                game_pk=game_pk,
                game_date=date,
                season=season,
                home_team=f"H{index}",
                away_team=f"A{index}",
                home_score=1,
                away_score=0,
            )
            for index, (game_pk, date, season) in enumerate(specs)
        ]
        games_df, features, manifest, pa = build_fixture(games)
        dataset, _, _ = build_m1_dataset(
            games_df, features, team_feature_manifest=manifest, config=self.config, plate_appearances=pa
        )
        actual = dict(zip(dataset["game_pk"], dataset["split"]))
        self.assertEqual(actual, {"TR": "train", "VA": "validation", "TE": "test", "SN": "snapshot"})

    def test_nullable_dtypes_are_preserved(self) -> None:
        game = make_game(
            game_pk="G1",
            game_date="2023-04-01",
            season=2023,
            home_team="A",
            away_team="B",
            home_score=0,
            away_score=0,
        )
        games, features, manifest, pa = build_fixture([game])
        features.loc[features["team"].eq("A"), "hist_win_pct"] = pd.NA
        dataset, _, _ = build_m1_dataset(
            games, features, team_feature_manifest=manifest, config=self.config, plate_appearances=pa
        )
        self.assertEqual(str(dataset["home_hist_win_pct"].dtype), "Float64")
        self.assertEqual(str(dataset["home_result_code"].dtype), "Int64")
        self.assertEqual(str(dataset["official_status_verified"].dtype), "boolean")
        self.assertTrue(pd.isna(dataset.loc[0, "home_hist_win_pct"]))

    def test_score_mutation_changes_y_but_not_x(self) -> None:
        game = make_game(
            game_pk="G1",
            game_date="2023-04-01",
            season=2023,
            home_team="A",
            away_team="B",
            home_score=3,
            away_score=1,
        )
        games, features, manifest, _ = build_fixture([game])
        first, roles, _ = build_m1_dataset(
            games, features, team_feature_manifest=manifest, config=self.config, plate_appearances=None
        )

        changed = games.copy()
        changed.loc[0, "final_home_score"] = 0
        changed.loc[0, "final_away_score"] = 5
        changed.loc[0, "home_win"] = False
        changed.loc[0, "away_win"] = True
        changed.loc[0, "is_tie"] = False
        second, _, _ = build_m1_dataset(
            changed, features, team_feature_manifest=manifest, config=self.config, plate_appearances=None
        )
        assert_frame_equal(
            first.loc[:, roles["x_columns"]],
            second.loc[:, roles["x_columns"]],
            check_dtype=True,
        )
        self.assertNotEqual(first.loc[0, "home_result"], second.loc[0, "home_result"])

    def test_future_game_and_features_do_not_change_past_x(self) -> None:
        past = make_game(
            game_pk="PAST",
            game_date="2025-06-01",
            season=2025,
            home_team="A",
            away_team="B",
            home_score=2,
            away_score=1,
        )
        games, features, manifest, pa = build_fixture([past])
        base, roles, _ = build_m1_dataset(
            games, features, team_feature_manifest=manifest, config=self.config, plate_appearances=pa
        )

        future = make_game(
            game_pk="FUTURE",
            game_date="2026-06-01",
            season=2026,
            home_team="C",
            away_team="D",
            home_score=5,
            away_score=4,
        )
        future_games, future_features, _, future_pa = build_fixture([future])
        all_games = pd.concat([games, future_games], ignore_index=True)
        all_features = pd.concat([features, future_features], ignore_index=True)
        all_pa = pd.concat([pa, future_pa], ignore_index=True)
        all_manifest = make_manifest(len(all_features))
        extended, _, _ = build_m1_dataset(
            all_games,
            all_features,
            team_feature_manifest=all_manifest,
            config=self.config,
            plate_appearances=all_pa,
        )
        past_extended = extended.loc[extended["game_pk"].eq("PAST")].reset_index(drop=True)
        assert_frame_equal(
            base.loc[:, roles["x_columns"]].reset_index(drop=True),
            past_extended.loc[:, roles["x_columns"]],
            check_dtype=True,
        )

    def test_input_shuffle_keeps_dataset_and_fingerprint(self) -> None:
        games = [
            make_game(
                game_pk=f"G{index}",
                game_date=f"2023-04-0{index + 1}",
                season=2023,
                home_team=f"H{index}",
                away_team=f"A{index}",
                home_score=index + 1,
                away_score=index,
            )
            for index in range(3)
        ]
        games_df, features, manifest, pa = build_fixture(games)
        first, _, _ = build_m1_dataset(
            games_df, features, team_feature_manifest=manifest, config=self.config, plate_appearances=pa
        )
        second, _, _ = build_m1_dataset(
            games_df.sample(frac=1, random_state=7).reset_index(drop=True),
            features.sample(frac=1, random_state=11).reset_index(drop=True),
            team_feature_manifest=manifest,
            config=self.config,
            plate_appearances=pa.sample(frac=1, random_state=13).reset_index(drop=True),
        )
        assert_frame_equal(first, second, check_dtype=True)
        self.assertEqual(
            content_fingerprint(first, sort_columns=["game_pk"]),
            content_fingerprint(second, sort_columns=["game_pk"]),
        )

    def test_x_allowlist_rejects_target_quality_split_and_control(self) -> None:
        available = [
            "home_hist_games",
            "home_runs",
            "observation_quality",
            "split",
            "is_excluded",
        ]
        for forbidden in ("home_runs", "observation_quality", "split", "is_excluded"):
            with self.subTest(forbidden=forbidden), self.assertRaises(ProcessedContractError):
                validate_x_allowlist(
                    available_columns=available,
                    x_columns=["home_hist_games", forbidden],
                    forbidden_columns=FORBIDDEN_X_COLUMNS,
                    forbidden_prefixes=FORBIDDEN_X_PREFIXES,
                )

    def test_team_manifest_forbidden_semantics_fail_before_prefixing(self) -> None:
        """#23 x_columns에 quality/terminal/split/target 의미가 섞이면 Home/Away prefix 전 실패해야 한다."""
        game = make_game(
            game_pk="G1",
            game_date="2023-04-01",
            season=2023,
            home_team="A",
            away_team="B",
            home_score=1,
            away_score=0,
        )
        games, features, manifest, pa = build_fixture([game])
        for forbidden in (
            "observation_quality",
            "terminal_pa_completed",
            "split",
            "home_result",
        ):
            with self.subTest(forbidden=forbidden):
                changed_features = features.copy()
                changed_features[forbidden] = 0
                changed_manifest = dict(manifest)
                changed_manifest["x_columns"] = [*TEAM_X_COLUMNS, forbidden]
                with self.assertRaises(M1DatasetBuildError):
                    build_m1_dataset(
                        games,
                        changed_features,
                        team_feature_manifest=changed_manifest,
                        config=self.config,
                        plate_appearances=pa,
                    )

    def test_schema_quality_and_content_fingerprints_are_deterministic(self) -> None:
        game = make_game(
            game_pk="G1",
            game_date="2023-04-01",
            season=2023,
            home_team="A",
            away_team="B",
            home_score=2,
            away_score=1,
        )
        games, features, manifest, pa = build_fixture([game])
        dataset, roles, summary = build_m1_dataset(
            games, features, team_feature_manifest=manifest, config=self.config, plate_appearances=pa
        )
        schema = build_schema_payload(dataset, roles)
        quality = build_quality_report(dataset, summary)
        self.assertEqual(stable_json_fingerprint(schema), stable_json_fingerprint(schema))
        self.assertEqual(stable_json_fingerprint(quality), stable_json_fingerprint(quality))
        self.assertEqual(
            content_fingerprint(dataset, sort_columns=["game_pk"]),
            content_fingerprint(dataset.copy(), sort_columns=["game_pk"]),
        )


class ProcessedContractUtilityTest(unittest.TestCase):
    """#26/#27 재사용 범위를 포함한 공통 Processed utility를 검증한다."""

    def test_manifest_deterministic_identity_excludes_runtime_metadata(self) -> None:
        """Manifest의 결정적 Identity는 실행 시각과 독립적으로 같은 입력에서 같아야 한다."""
        game = make_game(
            game_pk="G1",
            game_date="2024-04-01",
            season=2024,
            home_team="A",
            away_team="B",
            home_score=2,
            away_score=1,
        )
        games, features, manifest, pa = build_fixture([game])
        dataset, roles, summary = build_m1_dataset(
            games,
            features,
            team_feature_manifest=manifest,
            config=load_config(),
            plate_appearances=pa,
        )
        schema_payload = build_schema_payload(dataset, roles)
        quality_payload = build_quality_report(dataset, summary)
        dataset_fp = content_fingerprint(dataset, sort_columns=["game_pk"])
        kwargs = {
            "input_hashes": {"games": "a", "team_features": "b"},
            "team_feature_content_fingerprint": "team-fp",
            "dataset_content_fingerprint": dataset_fp,
            "schema_payload": schema_payload,
            "quality_payload": quality_payload,
        }
        first = build_deterministic_identity(**kwargs)
        second = build_deterministic_identity(**kwargs)
        self.assertEqual(first, second)
        self.assertEqual(first["fingerprint"], second["fingerprint"])

        changed = build_deterministic_identity(
            **{**kwargs, "dataset_content_fingerprint": "changed"}
        )
        self.assertNotEqual(first["fingerprint"], changed["fingerprint"])

    def test_completion_marker_is_invalidated_before_rerun(self) -> None:
        """이전 complete Manifest는 새 Run 시작 전에 제거되어 stale 완료 상태가 남지 않아야 한다."""
        with TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "game_dataset.manifest.json"
            manifest_path.write_text('{"run_status":"complete"}', encoding="utf-8")
            invalidate_completion_marker(manifest_path)
            self.assertFalse(manifest_path.exists())
            invalidate_completion_marker(manifest_path)

    def test_path_collision_fails(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "data" / "interim" / "games.parquet"
            input_path.parent.mkdir(parents=True)
            input_path.write_bytes(b"fixture")
            with self.assertRaises(ProcessedContractError):
                validate_output_paths(
                    input_paths=[input_path],
                    output_paths=[input_path],
                    project_root=root,
                )

    def test_raw_or_interim_output_fails(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.bin"
            input_path.write_bytes(b"fixture")
            with self.assertRaises(ProcessedContractError):
                validate_output_paths(
                    input_paths=[input_path],
                    output_paths=[root / "data" / "interim" / "m1.parquet"],
                    project_root=root,
                )

    def test_input_sha256_is_unchanged_by_read_only_hashing(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "input.bin"
            path.write_bytes(b"canonical-input")
            before = calculate_sha256(path)
            _ = path.read_bytes()
            after = calculate_sha256(path)
            self.assertEqual(before, after)

    def test_input_hash_guard_detects_mutation(self) -> None:
        """Builder 전후 입력이 바뀌면 공통 hash guard가 실패해야 한다."""
        with TemporaryDirectory() as directory:
            path = Path(directory) / "input.bin"
            path.write_bytes(b"canonical-input")
            paths = {"games": path}
            before = calculate_input_hashes(paths)
            path.write_bytes(b"mutated-input")
            with self.assertRaises(ProcessedContractError):
                assert_input_hashes_unchanged(before, paths)

    def test_label_interval_common_primitive(self) -> None:
        """공통 Label interval primitive는 half-open 순서와 availability를 검증해야 한다."""
        validate_label_interval(
            label_start="2025-01-01",
            label_end_exclusive="2025-01-08",
            label_available_at="2025-01-08",
        )
        with self.assertRaises(ProcessedContractError):
            validate_label_interval(
                label_start="2025-01-08",
                label_end_exclusive="2025-01-08",
            )
        with self.assertRaises(ProcessedContractError):
            validate_label_interval(
                label_start="2025-01-01",
                label_end_exclusive="2025-01-08",
                label_available_at="2025-01-07",
            )

    def test_team_feature_manifest_source_contract_mismatch_fails(self) -> None:
        """#23 Manifest가 다른 Feature Catalog 계약이면 M1 조립을 중단해야 한다."""
        game = make_game(
            game_pk="G1",
            game_date="2023-04-01",
            season=2023,
            home_team="A",
            away_team="B",
            home_score=1,
            away_score=0,
        )
        games, features, manifest, pa = build_fixture([game])
        manifest["source_contract_version"] = "unexpected_catalog"
        with self.assertRaises(M1DatasetBuildError):
            build_m1_dataset(
                games,
                features,
                team_feature_manifest=manifest,
                config=load_config(),
                plate_appearances=pa,
            )

    def test_team_feature_manifest_row_count_mismatch_fails(self) -> None:
        """#23 Manifest의 row_count와 실제 Team Feature Row 수가 다르면 실패해야 한다."""
        game = make_game(
            game_pk="G1",
            game_date="2023-04-01",
            season=2023,
            home_team="A",
            away_team="B",
            home_score=1,
            away_score=0,
        )
        games, features, manifest, pa = build_fixture([game])
        manifest["output"]["row_count"] = len(features) + 1
        with self.assertRaises(M1DatasetBuildError):
            build_m1_dataset(
                games,
                features,
                team_feature_manifest=manifest,
                config=load_config(),
                plate_appearances=pa,
            )

    def test_split_gap_config_fails(self) -> None:
        """고정 Split 사이에 누락 날짜가 생기는 설정은 실패해야 한다."""
        config = load_config()
        config["m1"]["split_ranges"][1]["start"] = "2024-01-02"
        game = make_game(
            game_pk="G1",
            game_date="2023-04-01",
            season=2023,
            home_team="A",
            away_team="B",
            home_score=1,
            away_score=0,
        )
        games, features, manifest, pa = build_fixture([game])
        with self.assertRaises(M1DatasetBuildError):
            build_m1_dataset(
                games, features, team_feature_manifest=manifest,
                config=config, plate_appearances=pa
            )

    def test_split_overlap_config_fails(self) -> None:
        """고정 Split 날짜 범위가 중복되는 설정은 실패해야 한다."""
        config = load_config()
        config["m1"]["split_ranges"][1]["start"] = "2023-12-31"
        game = make_game(
            game_pk="G1",
            game_date="2023-04-01",
            season=2023,
            home_team="A",
            away_team="B",
            home_score=1,
            away_score=0,
        )
        games, features, manifest, pa = build_fixture([game])
        with self.assertRaises(M1DatasetBuildError):
            build_m1_dataset(
                games, features, team_feature_manifest=manifest,
                config=config, plate_appearances=pa
            )

    def test_bad_class_order_config_fails(self) -> None:
        config = load_config()
        config["m1"]["home_result_classes"] = [
            {"class": "win", "code": 0},
            {"class": "tie", "code": 1},
            {"class": "loss", "code": 2},
        ]
        game = make_game(
            game_pk="G1",
            game_date="2023-04-01",
            season=2023,
            home_team="A",
            away_team="B",
            home_score=1,
            away_score=0,
        )
        games, features, manifest, pa = build_fixture([game])
        with self.assertRaises(M1DatasetBuildError):
            build_m1_dataset(
                games,
                features,
                team_feature_manifest=manifest,
                config=config,
                plate_appearances=pa,
            )

    @unittest.skipUnless(PYARROW_AVAILABLE, "pyarrow가 없어 실제 Parquet round-trip을 실행할 수 없습니다.")
    def test_parquet_round_trip_contract(self) -> None:
        """pyarrow가 있는 환경에서는 nullable dtype과 Column 순서가 실제 Parquet 왕복 후 유지되어야 한다."""
        game = make_game(
            game_pk="G1",
            game_date="2023-04-01",
            season=2023,
            home_team="A",
            away_team="B",
            home_score=1,
            away_score=0,
        )
        games, features, manifest, pa = build_fixture([game])
        dataset, _, _ = build_m1_dataset(
            games, features, team_feature_manifest=manifest, config=load_config(), plate_appearances=pa
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "m1.parquet"
            dataset.to_parquet(path, engine="pyarrow", index=False)
            round_trip = pd.read_parquet(path, engine="pyarrow")
            self.assertEqual(list(round_trip.columns), list(dataset.columns))
            self.assertEqual(len(round_trip), len(dataset))
            self.assertEqual(round_trip["game_pk"].astype("string").tolist(), ["G1"])


if __name__ == "__main__":
    unittest.main()
