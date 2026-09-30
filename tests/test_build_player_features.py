from __future__ import annotations

import importlib.util
import inspect
import json
import sys
import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from pandas.testing import assert_frame_equal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PYARROW_AVAILABLE = importlib.util.find_spec("pyarrow") is not None

from src.features.player import (
    BATTING_REQUIRED_SOURCE_COLUMNS,
    FEATURE_VERSION,
    OUTPUT_COLUMNS,
    PITCHING_REQUIRED_SOURCE_COLUMNS,
    REQUEST_KEY,
    ROLLING_WINDOWS,
    PlayerFeatureBuildError,
    build_player_pregame_features,
)



def make_requests(*rows: tuple[str, str, str]) -> pd.DataFrame:
    """테스트용 Player Feature Request DataFrame을 만든다."""
    return pd.DataFrame(
        [
            {
                "player_id": player_id,
                "role": role,
                "prediction_date": prediction_date,
            }
            for player_id, role, prediction_date in rows
        ]
    )


def empty_batting() -> pd.DataFrame:
    """필수 Schema만 가진 빈 Batting Source를 만든다."""
    return pd.DataFrame(columns=BATTING_REQUIRED_SOURCE_COLUMNS)


def empty_pitching() -> pd.DataFrame:
    """필수 Schema만 가진 빈 Pitching Source를 만든다."""
    return pd.DataFrame(columns=PITCHING_REQUIRED_SOURCE_COLUMNS)


def make_batting_row(
    *,
    game_pk: str,
    game_date: str,
    player_id: str = "B001",
    pa: int = 4,
    single: int = 1,
    double: int = 0,
    triple: int = 0,
    hr: int = 0,
    bb: int = 0,
    hbp: int = 0,
    so: int = 0,
    sf: int = 0,
    sh: int = 0,
    double_play: int = 0,
    triple_play: int = 0,
    field_error: int = 0,
    fielders_choice: int = 0,
    catcher_interference: int = 0,
    season: int | None = None,
    **extra: object,
) -> dict[str, object]:
    """Catalog 산식을 만족하는 Player Game Batting Row를 만든다."""
    h = single + double + triple + hr
    ab = pa - bb - hbp - sh - sf - catcher_interference
    tb = single + 2 * double + 3 * triple + 4 * hr
    parsed_date = pd.Timestamp(game_date)
    row: dict[str, object] = {
        "game_pk": game_pk,
        "game_date": game_date,
        "season": season if season is not None else parsed_date.year,
        "batter": player_id,
        "pa": pa,
        "ab": ab,
        "h": h,
        "single": single,
        "double": double,
        "triple": triple,
        "hr": hr,
        "bb": bb,
        "hbp": hbp,
        "so": so,
        "sf": sf,
        "sh": sh,
        "tb": tb,
        "double_play": double_play,
        "triple_play": triple_play,
        "field_error": field_error,
        "fielders_choice": fielders_choice,
        "catcher_interference": catcher_interference,
    }
    row.update(extra)
    return row


def make_pitching_row(
    *,
    game_pk: str,
    game_date: str,
    player_id: str = "P001",
    pitch_rows: int = 12,
    balls: int = 4,
    strikes: int = 5,
    in_play: int = 3,
    batters_faced_completed: int = 3,
    single_allowed: int = 1,
    double_allowed: int = 0,
    triple_allowed: int = 0,
    hr_allowed: int = 0,
    bb_allowed: int = 0,
    hbp_allowed: int = 0,
    so: int = 1,
    sf: int = 0,
    sh: int = 0,
    outs_recorded: int | None = 3,
    season: int | None = None,
    **extra: object,
) -> dict[str, object]:
    """Count Reconciliation을 만족하는 Player Game Pitching Row를 만든다."""
    pitches = balls + strikes + in_play
    hits_allowed = single_allowed + double_allowed + triple_allowed + hr_allowed
    parsed_date = pd.Timestamp(game_date)
    row: dict[str, object] = {
        "game_pk": game_pk,
        "game_date": game_date,
        "season": season if season is not None else parsed_date.year,
        "pitcher": player_id,
        "pitch_rows": pitch_rows,
        "pitches": pitches,
        "batters_faced_completed": batters_faced_completed,
        "hits_allowed": hits_allowed,
        "single_allowed": single_allowed,
        "double_allowed": double_allowed,
        "triple_allowed": triple_allowed,
        "hr_allowed": hr_allowed,
        "bb_allowed": bb_allowed,
        "hbp_allowed": hbp_allowed,
        "so": so,
        "sf": sf,
        "sh": sh,
        "outs_recorded": outs_recorded,
        "ball_pitch_count": balls,
        "strike_pitch_count": strikes,
        "in_play_pitch_count": in_play,
    }
    row.update(extra)
    return row


def get_row(
    features: pd.DataFrame,
    *,
    player_id: str,
    role: str,
    prediction_date: str,
) -> pd.Series:
    """Request Key로 Feature Row 하나를 조회한다."""
    selected = features.loc[
        features["player_id"].eq(player_id)
        & features["role"].eq(role)
        & features["prediction_date"].eq(pd.Timestamp(prediction_date))
    ]
    if len(selected) != 1:
        raise AssertionError(
            f"Feature Row가 유일하지 않습니다: {player_id}, {role}, {prediction_date}"
        )
    return selected.iloc[0]


class PlayerFeatureCalculationTest(unittest.TestCase):
    """Player Pregame Historical Feature의 핵심 산식과 leakage 계약을 검증한다."""

    def test_batting_hand_calculated_counts_and_rates(self) -> None:
        """경기별 Rate 평균이 아니라 합산 Count로 AVG/OBP/SLG/OPS를 계산해야 한다."""
        batting = pd.DataFrame(
            [
                make_batting_row(
                    game_pk="G1",
                    game_date="2025-04-01",
                    pa=5,
                    single=1,
                    double=1,
                    bb=1,
                    so=1,
                ),
                make_batting_row(
                    game_pk="G2",
                    game_date="2025-04-03",
                    pa=4,
                    single=0,
                    hr=1,
                    hbp=1,
                    sf=1,
                    so=1,
                ),
            ]
        )
        features = build_player_pregame_features(
            make_requests(("B001", "batting", "2025-04-05")),
            batting,
            empty_pitching(),
        )
        row = features.iloc[0]

        self.assertEqual(row["hist_pa"], 9)
        self.assertEqual(row["hist_ab"], 6)
        self.assertEqual(row["hist_h"], 3)
        self.assertEqual(row["hist_single"], 1)
        self.assertEqual(row["hist_double"], 1)
        self.assertEqual(row["hist_hr"], 1)
        self.assertEqual(row["hist_bb"], 1)
        self.assertEqual(row["hist_hbp"], 1)
        self.assertEqual(row["hist_sf"], 1)
        self.assertEqual(row["hist_tb"], 7)
        self.assertEqual(float(row["hist_avg"]), 0.5)
        self.assertEqual(float(row["hist_obp"]), 0.556)
        self.assertEqual(float(row["hist_slg"]), 1.167)
        self.assertEqual(float(row["hist_ops"]), 1.723)

    def test_batting_rates_are_rounded_to_three_decimals(self) -> None:
        """Expanding과 Rolling 타격 Rate를 합산 Count 기준 소수점 3자리로 고정한다."""
        batting = pd.DataFrame(
            [
                make_batting_row(
                    game_pk="G1",
                    game_date="2025-04-01",
                    pa=3,
                    single=2,
                ),
                make_batting_row(
                    game_pk="G2",
                    game_date="2025-04-02",
                    pa=3,
                    single=2,
                ),
                make_batting_row(
                    game_pk="G3",
                    game_date="2025-04-03",
                    pa=3,
                    single=2,
                ),
            ]
        )
        row = build_player_pregame_features(
            make_requests(("B001", "batting", "2025-04-04")),
            batting,
            empty_pitching(),
        ).iloc[0]

        expected = {
            "avg": 0.667,
            "obp": 0.667,
            "slg": 0.667,
            "ops": 1.334,
        }
        for name, value in expected.items():
            self.assertEqual(float(row[f"hist_{name}"]), value)
            self.assertEqual(float(row[f"last_5g_{name}"]), value)
            self.assertEqual(float(row[f"last_10g_{name}"]), value)
            self.assertEqual(float(row[f"last_20g_{name}"]), value)

    def test_batting_zero_denominator_rate_is_null(self) -> None:
        """AB와 OBP denominator가 0이면 Rate를 0으로 채우지 않아야 한다."""
        batting = pd.DataFrame(
            [
                make_batting_row(
                    game_pk="G1",
                    game_date="2025-04-01",
                    pa=1,
                    catcher_interference=1,
                )
            ]
        )
        row = build_player_pregame_features(
            make_requests(("B001", "batting", "2025-04-02")),
            batting,
            empty_pitching(),
        ).iloc[0]
        self.assertEqual(row["hist_ab"], 0)
        self.assertTrue(pd.isna(row["hist_avg"]))
        self.assertTrue(pd.isna(row["hist_obp"]))
        self.assertTrue(pd.isna(row["hist_slg"]))
        self.assertTrue(pd.isna(row["hist_ops"]))

    def test_pitching_counts_keep_pitch_rows_and_pitches_distinct(self) -> None:
        """Raw Row 수와 actual pitch 수를 서로 다른 누적 Count로 보존해야 한다."""
        pitching = pd.DataFrame(
            [
                make_pitching_row(
                    game_pk="G1",
                    game_date="2025-04-01",
                    pitch_rows=15,
                    balls=4,
                    strikes=4,
                    in_play=2,
                    single_allowed=1,
                    double_allowed=1,
                    so=2,
                    outs_recorded=4,
                ),
                make_pitching_row(
                    game_pk="G2",
                    game_date="2025-04-02",
                    pitch_rows=9,
                    balls=3,
                    strikes=3,
                    in_play=1,
                    hr_allowed=1,
                    single_allowed=0,
                    so=1,
                    outs_recorded=2,
                ),
            ]
        )
        row = build_player_pregame_features(
            make_requests(("P001", "pitching", "2025-04-03")),
            empty_batting(),
            pitching,
        ).iloc[0]
        self.assertEqual(row["hist_pitch_rows"], 24)
        self.assertEqual(row["hist_pitches"], 17)
        self.assertEqual(row["hist_hits_allowed"], 3)
        self.assertEqual(row["hist_so"], 3)
        self.assertEqual(row["hist_outs_recorded"], 6)

    def test_pitchless_player_game_keeps_row_count_without_actual_pitch(self) -> None:
        """Pitch-less 기록은 pitch_rows에는 포함하고 pitches/B/S/X에는 포함하지 않아야 한다."""
        pitching = pd.DataFrame(
            [
                make_pitching_row(
                    game_pk="PL",
                    game_date="2025-04-01",
                    pitch_rows=1,
                    balls=0,
                    strikes=0,
                    in_play=0,
                    batters_faced_completed=1,
                    single_allowed=0,
                    outs_recorded=0,
                )
            ]
        )
        row = build_player_pregame_features(
            make_requests(("P001", "pitching", "2025-04-02")),
            empty_batting(),
            pitching,
        ).iloc[0]
        self.assertEqual(row["hist_pitch_rows"], 1)
        self.assertEqual(row["hist_pitches"], 0)
        self.assertEqual(row["hist_ball_pitch_count"], 0)
        self.assertEqual(row["hist_strike_pitch_count"], 0)
        self.assertEqual(row["hist_in_play_pitch_count"], 0)

    def test_pitching_outs_null_propagates_to_expanding_and_rolling(self) -> None:
        """History Window에 불확실 outs가 하나라도 있으면 aggregate outs도 null이어야 한다."""
        pitching = pd.DataFrame(
            [
                make_pitching_row(
                    game_pk=f"G{i}",
                    game_date=f"2025-04-{i:02d}",
                    outs_recorded=None if i == 3 else 3,
                )
                for i in range(1, 7)
            ]
        )
        row = build_player_pregame_features(
            make_requests(("P001", "pitching", "2025-04-07")),
            empty_batting(),
            pitching,
        ).iloc[0]
        self.assertTrue(pd.isna(row["hist_outs_recorded"]))
        self.assertEqual(row["history_uncertain_outs_game_count"], 1)
        self.assertEqual(row["last_5g_uncertain_outs_game_count"], 1)
        self.assertTrue(pd.isna(row["last_5g_outs_recorded"]))
        self.assertTrue(pd.isna(row["last_10g_outs_recorded"]))
        self.assertTrue(pd.isna(row["last_20g_outs_recorded"]))

    def test_history_absent_preserves_batting_and_pitching_requests(self) -> None:
        """History가 없어도 유효 Request를 삭제하지 않고 role별 cold start를 반환해야 한다."""
        requests = make_requests(
            ("NEW", "batting", "2025-04-01"),
            ("NEW", "pitching", "2025-04-01"),
        )
        features = build_player_pregame_features(
            requests,
            empty_batting(),
            empty_pitching(),
        )
        self.assertEqual(len(features), 2)
        for _, row in features.iterrows():
            self.assertEqual(row["history_game_count"], 0)
            self.assertFalse(bool(row["has_history"]))
            self.assertTrue(pd.isna(row["max_source_game_date"]))
            self.assertTrue(pd.isna(row["days_since_last_observed_appearance"]))
            for window in ROLLING_WINDOWS:
                self.assertEqual(row[f"last_{window}g_games"], 0)
        batting_row = get_row(
            features,
            player_id="NEW",
            role="batting",
            prediction_date="2025-04-01",
        )
        pitching_row = get_row(
            features,
            player_id="NEW",
            role="pitching",
            prediction_date="2025-04-01",
        )
        self.assertEqual(batting_row["hist_pa"], 0)
        self.assertTrue(pd.isna(batting_row["hist_avg"]))
        self.assertEqual(pitching_row["hist_pitches"], 0)
        self.assertEqual(pitching_row["hist_outs_recorded"], 0)

    def test_same_day_doubleheader_is_not_history(self) -> None:
        """같은 날짜 두 경기 모두 그 날짜 Prediction Feature의 Source가 아니어야 한다."""
        batting = pd.DataFrame(
            [
                make_batting_row(game_pk="PREV", game_date="2025-04-01", single=1),
                make_batting_row(game_pk="DH1", game_date="2025-04-02", hr=1, single=0),
                make_batting_row(game_pk="DH2", game_date="2025-04-02", double=1, single=0),
            ]
        )
        row = build_player_pregame_features(
            make_requests(("B001", "batting", "2025-04-02")),
            batting,
            empty_pitching(),
        ).iloc[0]
        self.assertEqual(row["history_game_count"], 1)
        self.assertEqual(row["hist_h"], 1)
        self.assertEqual(row["max_source_game_date"], pd.Timestamp("2025-04-01"))

    def test_same_day_other_game_mutation_does_not_change_feature(self) -> None:
        """같은 날짜 다른 경기 결과를 바꿔도 그 날짜 Prediction Feature는 동일해야 한다."""
        base = pd.DataFrame(
            [
                make_batting_row(game_pk="PREV", game_date="2025-04-01", single=1),
                make_batting_row(game_pk="DH1", game_date="2025-04-02", single=1),
                make_batting_row(game_pk="DH2", game_date="2025-04-02", double=1, single=0),
            ]
        )
        changed = base.copy()
        replacement = make_batting_row(
            game_pk="DH2",
            game_date="2025-04-02",
            pa=4,
            hr=4,
            single=0,
        )
        for column, value in replacement.items():
            if column in changed.columns:
                changed.loc[changed["game_pk"].eq("DH2"), column] = value
        request = make_requests(("B001", "batting", "2025-04-02"))
        feature_a = build_player_pregame_features(request, base, empty_pitching())
        feature_b = build_player_pregame_features(request, changed, empty_pitching())
        assert_frame_equal(feature_a, feature_b)

    def test_current_day_mutation_does_not_change_historical_x(self) -> None:
        """prediction_date 당일 Player Game 결과 변조가 Historical X를 바꾸면 안 된다."""
        base = [
            make_batting_row(game_pk="PREV", game_date="2025-04-01", single=1),
            make_batting_row(game_pk="TODAY", game_date="2025-04-02", single=1),
        ]
        changed = [dict(row) for row in base]
        changed[1] = make_batting_row(
            game_pk="TODAY",
            game_date="2025-04-02",
            pa=4,
            hr=4,
            single=0,
        )
        request = make_requests(("B001", "batting", "2025-04-02"))
        feature_a = build_player_pregame_features(
            request,
            pd.DataFrame(base),
            empty_pitching(),
        )
        feature_b = build_player_pregame_features(
            request,
            pd.DataFrame(changed),
            empty_pitching(),
        )
        assert_frame_equal(feature_a, feature_b)

    def test_future_addition_and_mutation_do_not_change_past_request(self) -> None:
        """미래 Player Game 추가·변조가 과거 Request Feature에 영향을 주면 안 된다."""
        base = pd.DataFrame(
            [make_batting_row(game_pk="PREV", game_date="2025-04-01", single=1)]
        )
        future = pd.concat(
            [
                base,
                pd.DataFrame(
                    [
                        make_batting_row(
                            game_pk="FUTURE",
                            game_date="2025-05-01",
                            pa=4,
                            hr=4,
                            single=0,
                        )
                    ]
                ),
            ],
            ignore_index=True,
        )
        request = make_requests(("B001", "batting", "2025-04-02"))
        feature_a = build_player_pregame_features(request, base, empty_pitching())
        feature_b = build_player_pregame_features(request, future, empty_pitching())
        assert_frame_equal(feature_a, feature_b)

    def test_rolling_5_10_20_games(self) -> None:
        """기본 Rolling Window의 실제 포함 경기 수와 Count 합이 일치해야 한다."""
        batting = pd.DataFrame(
            [
                make_batting_row(
                    game_pk=f"G{i:02d}",
                    game_date=(pd.Timestamp("2025-04-01") + pd.Timedelta(days=i - 1)).strftime("%Y-%m-%d"),
                    single=1,
                )
                for i in range(1, 22)
            ]
        )
        row = build_player_pregame_features(
            make_requests(("B001", "batting", "2025-04-22")),
            batting,
            empty_pitching(),
        ).iloc[0]
        self.assertEqual(row["history_game_count"], 21)
        self.assertEqual(row["last_5g_games"], 5)
        self.assertEqual(row["last_10g_games"], 10)
        self.assertEqual(row["last_20g_games"], 20)
        self.assertEqual(row["last_5g_h"], 5)
        self.assertEqual(row["last_10g_h"], 10)
        self.assertEqual(row["last_20g_h"], 20)

    def test_rolling_boundary_includes_entire_same_date_group(self) -> None:
        """N경기 경계 날짜에 Doubleheader가 걸리면 해당 날짜 묶음을 쪼개지 않아야 한다."""
        rows = [
            make_batting_row(game_pk="B1", game_date="2025-04-01"),
            make_batting_row(game_pk="B2", game_date="2025-04-01"),
            make_batting_row(game_pk="G3", game_date="2025-04-02"),
            make_batting_row(game_pk="G4", game_date="2025-04-03"),
            make_batting_row(game_pk="G5", game_date="2025-04-04"),
            make_batting_row(game_pk="G6", game_date="2025-04-05"),
        ]
        row = build_player_pregame_features(
            make_requests(("B001", "batting", "2025-04-06")),
            pd.DataFrame(rows),
            empty_pitching(),
        ).iloc[0]
        self.assertEqual(row["history_game_count"], 6)
        self.assertEqual(row["last_5g_games"], 6)

    def test_season_reset_ignores_previous_season(self) -> None:
        """Player Historical Feature는 prediction_date 연도 기준 Season마다 초기화해야 한다."""
        batting = pd.DataFrame(
            [
                make_batting_row(game_pk="S23", game_date="2023-10-01", single=1),
                make_batting_row(game_pk="S24", game_date="2024-04-01", double=1, single=0),
            ]
        )
        row = build_player_pregame_features(
            make_requests(("B001", "batting", "2024-04-02")),
            batting,
            empty_pitching(),
        ).iloc[0]
        self.assertEqual(row["history_game_count"], 1)
        self.assertEqual(row["hist_single"], 0)
        self.assertEqual(row["hist_double"], 1)

    def test_trade_does_not_break_same_player_history(self) -> None:
        """Team Context가 달라도 동일 player_id의 개인 기록은 같은 Season 이력으로 연결해야 한다."""
        batting = pd.DataFrame(
            [
                make_batting_row(game_pk="T1", game_date="2025-04-01", team="A"),
                make_batting_row(game_pk="T2", game_date="2025-04-02", team="B"),
            ]
        )
        row = build_player_pregame_features(
            make_requests(("B001", "batting", "2025-04-03")),
            batting,
            empty_pitching(),
        ).iloc[0]
        self.assertEqual(row["history_game_count"], 2)
        self.assertEqual(row["hist_pa"], 8)

    def test_same_name_different_player_ids_are_not_merged(self) -> None:
        """표시 이름이 같아도 서로 다른 player_id의 이력을 합치지 않아야 한다."""
        batting = pd.DataFrame(
            [
                make_batting_row(
                    game_pk="A1",
                    game_date="2025-04-01",
                    player_id="ID-A",
                    batter_name="동명이인",
                    single=1,
                ),
                make_batting_row(
                    game_pk="B1",
                    game_date="2025-04-01",
                    player_id="ID-B",
                    batter_name="동명이인",
                    hr=1,
                    single=0,
                ),
            ]
        )
        features = build_player_pregame_features(
            make_requests(
                ("ID-A", "batting", "2025-04-02"),
                ("ID-B", "batting", "2025-04-02"),
            ),
            batting,
            empty_pitching(),
        )
        row_a = get_row(
            features,
            player_id="ID-A",
            role="batting",
            prediction_date="2025-04-02",
        )
        row_b = get_row(
            features,
            player_id="ID-B",
            role="batting",
            prediction_date="2025-04-02",
        )
        self.assertEqual(row_a["hist_single"], 1)
        self.assertEqual(row_a["hist_hr"], 0)
        self.assertEqual(row_b["hist_single"], 0)
        self.assertEqual(row_b["hist_hr"], 1)

    def test_dual_role_player_has_independent_feature_rows(self) -> None:
        """같은 player_id의 batting/pitching 요청을 별도 Role Feature로 계산해야 한다."""
        features = build_player_pregame_features(
            make_requests(
                ("DUAL", "batting", "2025-04-02"),
                ("DUAL", "pitching", "2025-04-02"),
            ),
            pd.DataFrame(
                [make_batting_row(game_pk="B1", game_date="2025-04-01", player_id="DUAL")]
            ),
            pd.DataFrame(
                [make_pitching_row(game_pk="P1", game_date="2025-04-01", player_id="DUAL")]
            ),
        )
        batter = get_row(
            features,
            player_id="DUAL",
            role="batting",
            prediction_date="2025-04-02",
        )
        pitcher = get_row(
            features,
            player_id="DUAL",
            role="pitching",
            prediction_date="2025-04-02",
        )
        self.assertEqual(batter["hist_pa"], 4)
        self.assertTrue(pd.isna(batter["hist_pitches"]))
        self.assertEqual(pitcher["hist_pitches"], 12)
        self.assertTrue(pd.isna(pitcher["hist_pa"]))

    def test_batting_request_does_not_use_pitching_history(self) -> None:
        """Batting Request는 같은 ID의 Pitching History만 존재해도 cold start여야 한다."""
        pitching = pd.DataFrame(
            [make_pitching_row(game_pk="P1", game_date="2025-04-01", player_id="X")]
        )
        row = build_player_pregame_features(
            make_requests(("X", "batting", "2025-04-02")),
            empty_batting(),
            pitching,
        ).iloc[0]
        self.assertFalse(bool(row["has_history"]))
        self.assertEqual(row["hist_pa"], 0)

    def test_pitching_request_does_not_use_batting_history(self) -> None:
        """Pitching Request는 같은 ID의 Batting History만 존재해도 cold start여야 한다."""
        batting = pd.DataFrame(
            [make_batting_row(game_pk="B1", game_date="2025-04-01", player_id="X")]
        )
        row = build_player_pregame_features(
            make_requests(("X", "pitching", "2025-04-02")),
            batting,
            empty_pitching(),
        ).iloc[0]
        self.assertFalse(bool(row["has_history"]))
        self.assertEqual(row["hist_pitches"], 0)

    def test_request_row_is_preserved_one_to_one(self) -> None:
        """유효 Request 1개는 History 유무와 관계없이 Output 1행이어야 한다."""
        features = build_player_pregame_features(
            make_requests(("ONE", "batting", "2025-04-01")),
            empty_batting(),
            empty_pitching(),
        )
        self.assertEqual(len(features), 1)
        self.assertEqual(tuple(features.loc[0, list(REQUEST_KEY)]), ("ONE", "batting", pd.Timestamp("2025-04-01")))

    def test_duplicate_request_fails_explicitly(self) -> None:
        """중복 Request를 조용히 deduplicate하지 않고 명시적으로 실패해야 한다."""
        requests = make_requests(
            ("B001", "batting", "2025-04-01"),
            ("B001", "batting", "2025-04-01"),
        )
        with self.assertRaises(PlayerFeatureBuildError):
            build_player_pregame_features(requests, empty_batting(), empty_pitching())

    def test_duplicate_batting_source_key_fails(self) -> None:
        """Canonical Player Game Source key 중복은 Historical 집계 전에 실패해야 한다."""
        row = make_batting_row(game_pk="G1", game_date="2025-04-01")
        with self.assertRaises(PlayerFeatureBuildError):
            build_player_pregame_features(
                make_requests(("B001", "batting", "2025-04-02")),
                pd.DataFrame([row, row]),
                empty_pitching(),
            )

    def test_duplicate_pitching_source_key_fails(self) -> None:
        """Pitching Source의 (game_pk, pitcher) 중복도 실패해야 한다."""
        row = make_pitching_row(game_pk="G1", game_date="2025-04-01")
        with self.assertRaises(PlayerFeatureBuildError):
            build_player_pregame_features(
                make_requests(("P001", "pitching", "2025-04-02")),
                empty_batting(),
                pd.DataFrame([row, row]),
            )

    def test_empty_request_returns_typed_empty_output(self) -> None:
        """빈 Request는 오류 대신 고정 Schema의 0행 Artifact로 처리해야 한다."""
        requests = pd.DataFrame(columns=REQUEST_KEY)
        features = build_player_pregame_features(
            requests,
            empty_batting(),
            empty_pitching(),
        )
        self.assertTrue(features.empty)
        self.assertEqual(tuple(features.columns), OUTPUT_COLUMNS)
        self.assertEqual(str(features["player_id"].dtype), "string")
        self.assertEqual(str(features["history_game_count"].dtype), "Int64")
        self.assertEqual(str(features["hist_avg"].dtype), "Float64")

    def test_invalid_role_fails(self) -> None:
        """계약되지 않은 role을 자동 추론하거나 다른 역할로 매핑하면 안 된다."""
        with self.assertRaises(PlayerFeatureBuildError):
            build_player_pregame_features(
                make_requests(("X", "fielder", "2025-04-01")),
                empty_batting(),
                empty_pitching(),
            )

    def test_player_id_must_keep_string_semantics(self) -> None:
        """Request ID가 숫자 dtype이면 문자열 의미 손실 위험 때문에 명시적으로 실패해야 한다."""
        requests = pd.DataFrame(
            [{"player_id": 123, "role": "batting", "prediction_date": "2025-04-01"}]
        )
        with self.assertRaises(PlayerFeatureBuildError):
            build_player_pregame_features(requests, empty_batting(), empty_pitching())

    def test_output_dtypes_are_deterministic_nullable_types(self) -> None:
        """주요 Output dtype이 string/Int64/Float64/boolean/datetime64[us] 계약을 따라야 한다."""
        features = build_player_pregame_features(
            make_requests(("B001", "batting", "2025-04-02")),
            pd.DataFrame([make_batting_row(game_pk="G1", game_date="2025-04-01")]),
            empty_pitching(),
        )
        self.assertEqual(str(features["player_id"].dtype), "string")
        self.assertEqual(str(features["role"].dtype), "string")
        self.assertEqual(str(features["history_game_count"].dtype), "Int64")
        self.assertEqual(str(features["hist_pa"].dtype), "Int64")
        self.assertEqual(str(features["hist_avg"].dtype), "Float64")
        self.assertEqual(str(features["has_history"].dtype), "boolean")
        self.assertEqual(str(features["prediction_date"].dtype), "datetime64[us]")

    def test_input_reordering_does_not_change_values_order_or_dtypes(self) -> None:
        """Request/Source Row 순서를 섞어도 결정적 Output이 동일해야 한다."""
        requests = make_requests(
            ("A", "batting", "2025-04-04"),
            ("B", "batting", "2025-04-04"),
        )
        batting = pd.DataFrame(
            [
                make_batting_row(game_pk="A1", game_date="2025-04-01", player_id="A"),
                make_batting_row(game_pk="A2", game_date="2025-04-02", player_id="A"),
                make_batting_row(game_pk="B1", game_date="2025-04-01", player_id="B", hr=1, single=0),
            ]
        )
        feature_a = build_player_pregame_features(requests, batting, empty_pitching())
        feature_b = build_player_pregame_features(
            requests.sample(frac=1.0, random_state=7).reset_index(drop=True),
            batting.sample(frac=1.0, random_state=11).reset_index(drop=True),
            empty_pitching(),
        )
        assert_frame_equal(feature_a, feature_b)

    def test_average_release_speed_and_prohibited_stats_are_not_features(self) -> None:
        """가중 denominator가 없는 평균 구속과 근거 부족 공식 통계를 Feature로 승격하지 않는다."""
        pitching = pd.DataFrame(
            [
                make_pitching_row(
                    game_pk="G1",
                    game_date="2025-04-01",
                    avg_release_speed_kmh=145.2,
                    era=1.23,
                    earned_runs=1,
                    runs_allowed=2,
                )
            ]
        )
        features = build_player_pregame_features(
            make_requests(("P001", "pitching", "2025-04-02")),
            empty_batting(),
            pitching,
        )
        forbidden = {
            "hist_avg_release_speed_kmh",
            "hist_era",
            "hist_earned_runs",
            "hist_runs_allowed",
            "hist_rbi",
        }
        self.assertTrue(forbidden.isdisjoint(features.columns))

    def test_player_master_is_not_a_feature_dependency(self) -> None:
        """Player Master의 whole-period role/last_seen이 계산 API 의존성이 아니어야 한다."""
        parameters = inspect.signature(build_player_pregame_features).parameters
        self.assertNotIn("players", parameters)
        self.assertNotIn("player_master", parameters)

    def test_invalid_batting_reconciliation_fails(self) -> None:
        """Canonical Batting Count 산식이 깨진 Source를 조용히 집계하지 않아야 한다."""
        row = make_batting_row(game_pk="G1", game_date="2025-04-01")
        row["h"] = 99
        with self.assertRaises(PlayerFeatureBuildError):
            build_player_pregame_features(
                make_requests(("B001", "batting", "2025-04-02")),
                pd.DataFrame([row]),
                empty_pitching(),
            )

    def test_invalid_pitching_reconciliation_fails(self) -> None:
        """pitches와 B/S/X 합이 불일치하는 Source를 실패시켜야 한다."""
        row = make_pitching_row(game_pk="G1", game_date="2025-04-01")
        row["pitches"] = 999
        with self.assertRaises(PlayerFeatureBuildError):
            build_player_pregame_features(
                make_requests(("P001", "pitching", "2025-04-02")),
                empty_batting(),
                pd.DataFrame([row]),
            )

    def test_source_season_must_match_game_date_year(self) -> None:
        """시즌 초기화 기준을 모호하게 만들 Source season/date 불일치를 허용하지 않는다."""
        batting = pd.DataFrame(
            [make_batting_row(game_pk="G1", game_date="2025-04-01", season=2024)]
        )
        with self.assertRaises(PlayerFeatureBuildError):
            build_player_pregame_features(
                make_requests(("B001", "batting", "2025-04-02")),
                batting,
                empty_pitching(),
            )


class PlayerFeatureFileContractTest(unittest.TestCase):
    """Parquet round-trip, path safety, input hash, Manifest 계약을 검증한다."""

    def test_file_builder_preserves_input_hashes_with_mocked_parquet_io(self) -> None:
        """Parquet 엔진과 무관하게 Builder가 세 Input byte를 변경하지 않는 코드 경로를 검증한다."""
        from scripts import build_player_features as module
        from src.data.processed_contract import calculate_input_hashes

        with TemporaryDirectory() as directory:
            root = Path(directory)
            requests_path = root / "requests.parquet"
            batting_path = root / "batting.parquet"
            pitching_path = root / "pitching.parquet"
            output_path = root / "data" / "processed" / "player.parquet"
            manifest_path = root / "data" / "processed" / "player.manifest.json"
            for path, payload in (
                (requests_path, b"requests-source"),
                (batting_path, b"batting-source"),
                (pitching_path, b"pitching-source"),
            ):
                path.write_bytes(payload)

            requests = make_requests(("B001", "batting", "2025-04-02"))
            batting = pd.DataFrame(
                [make_batting_row(game_pk="B1", game_date="2025-04-01")]
            )
            pitching = empty_pitching()
            written: dict[str, pd.DataFrame] = {}

            def fake_read(path: Path, *, label: str) -> pd.DataFrame:
                if path == requests_path:
                    return requests.copy()
                if path == batting_path:
                    return batting.copy()
                if path == pitching_path:
                    return pitching.copy()
                if path == output_path:
                    return written["output"].copy()
                raise AssertionError(f"예상하지 못한 read_parquet 경로: {path}, {label}")

            def fake_write_parquet(frame: pd.DataFrame, path: Path) -> None:
                self.assertEqual(path, output_path)
                written["output"] = frame.copy()

            persisted_manifest: dict[str, object] = {}

            def fake_write_json(payload: dict[str, object], path: Path) -> None:
                self.assertEqual(path, manifest_path)
                persisted_manifest.update(payload)

            paths = {
                "requests": requests_path,
                "player_game_batting": batting_path,
                "player_game_pitching": pitching_path,
            }
            hashes_before = calculate_input_hashes(paths)
            with (
                patch.object(module, "read_parquet", side_effect=fake_read),
                patch.object(module, "write_parquet_atomic", side_effect=fake_write_parquet),
                patch.object(module, "write_json_atomic", side_effect=fake_write_json),
            ):
                features, manifest, returned_manifest_path = module.build_player_feature_files(
                    requests_path=requests_path,
                    batting_path=batting_path,
                    pitching_path=pitching_path,
                    output_path=output_path,
                    manifest_path=manifest_path,
                    project_root=root,
                )
            hashes_after = calculate_input_hashes(paths)

            self.assertEqual(hashes_before, hashes_after)
            self.assertEqual(returned_manifest_path, manifest_path)
            self.assertEqual(len(features), 1)
            self.assertEqual(manifest["run_status"], "complete")
            self.assertEqual(manifest["request"]["sha256"], hashes_before["requests"])
            self.assertEqual(persisted_manifest["output"]["row_count"], 1)

    @unittest.skipUnless(PYARROW_AVAILABLE, "pyarrow가 설치된 환경에서만 파일 계약을 검증합니다.")
    def test_parquet_round_trip_manifest_and_input_hash_preservation(self) -> None:
        """파일 Builder가 입력을 변경하지 않고 결정적 Output/Manifest를 생성해야 한다."""
        from scripts.build_player_features import build_player_feature_files
        from src.data.processed_contract import calculate_input_hashes, content_fingerprint

        with TemporaryDirectory() as directory:
            root = Path(directory)
            requests_path = root / "requests.parquet"
            batting_path = root / "batting.parquet"
            pitching_path = root / "pitching.parquet"
            output_path = root / "data" / "processed" / "player.parquet"
            manifest_path = root / "data" / "processed" / "player.manifest.json"

            requests = make_requests(
                ("B001", "batting", "2025-04-02"),
                ("P001", "pitching", "2025-04-02"),
            )
            batting = pd.DataFrame(
                [make_batting_row(game_pk="B1", game_date="2025-04-01")]
            )
            pitching = pd.DataFrame(
                [make_pitching_row(game_pk="P1", game_date="2025-04-01")]
            )
            requests.to_parquet(requests_path, engine="pyarrow", index=False)
            batting.to_parquet(batting_path, engine="pyarrow", index=False)
            pitching.to_parquet(pitching_path, engine="pyarrow", index=False)

            paths = {
                "requests": requests_path,
                "player_game_batting": batting_path,
                "player_game_pitching": pitching_path,
            }
            hashes_before = calculate_input_hashes(paths)
            features, manifest, returned_manifest_path = build_player_feature_files(
                requests_path=requests_path,
                batting_path=batting_path,
                pitching_path=pitching_path,
                output_path=output_path,
                manifest_path=manifest_path,
                project_root=root,
            )
            hashes_after = calculate_input_hashes(paths)

            self.assertEqual(hashes_before, hashes_after)
            self.assertEqual(returned_manifest_path, manifest_path)
            self.assertTrue(output_path.is_file())
            self.assertTrue(manifest_path.is_file())
            self.assertEqual(len(features), 2)
            self.assertEqual(manifest["artifact"], "player_pregame_features")
            self.assertEqual(manifest["feature_version"], FEATURE_VERSION)
            self.assertEqual(manifest["request"]["row_count"], 2)
            self.assertEqual(manifest["output"]["row_count"], 2)
            self.assertEqual(manifest["output"]["role_row_counts"], {"batting": 1, "pitching": 1})
            self.assertEqual(
                manifest["output"]["content_fingerprint"],
                content_fingerprint(features, sort_columns=list(REQUEST_KEY)),
            )

            with manifest_path.open("r", encoding="utf-8") as file:
                persisted = json.load(file)
            self.assertEqual(
                persisted["output"]["content_fingerprint"],
                manifest["output"]["content_fingerprint"],
            )

    def test_path_collision_fails_before_reading_parquet(self) -> None:
        """Output이 입력과 겹치면 Parquet 내용과 무관하게 읽기 전에 실패해야 한다."""
        from scripts.build_player_features import build_player_feature_files
        from src.data.processed_contract import ProcessedContractError

        with TemporaryDirectory() as directory:
            root = Path(directory)
            requests_path = root / "requests.parquet"
            batting_path = root / "batting.parquet"
            pitching_path = root / "pitching.parquet"
            for path in (requests_path, batting_path, pitching_path):
                path.write_bytes(b"not-a-parquet")

            with self.assertRaises(ProcessedContractError):
                build_player_feature_files(
                    requests_path=requests_path,
                    batting_path=batting_path,
                    pitching_path=pitching_path,
                    output_path=requests_path,
                    manifest_path=root / "manifest.json",
                    project_root=root,
                )

    def test_content_fingerprint_is_stable_after_input_reordering(self) -> None:
        """입력 Row 재정렬이 Output 값·dtype뿐 아니라 content fingerprint도 바꾸면 안 된다."""
        from src.data.processed_contract import content_fingerprint

        requests = make_requests(
            ("A", "batting", "2025-04-03"),
            ("B", "batting", "2025-04-03"),
        )
        batting = pd.DataFrame(
            [
                make_batting_row(game_pk="A1", game_date="2025-04-01", player_id="A"),
                make_batting_row(game_pk="B1", game_date="2025-04-01", player_id="B"),
            ]
        )
        feature_a = build_player_pregame_features(requests, batting, empty_pitching())
        feature_b = build_player_pregame_features(
            requests.iloc[::-1].reset_index(drop=True),
            batting.iloc[::-1].reset_index(drop=True),
            empty_pitching(),
        )
        self.assertEqual(
            content_fingerprint(feature_a, sort_columns=list(REQUEST_KEY)),
            content_fingerprint(feature_b, sort_columns=list(REQUEST_KEY)),
        )


if __name__ == "__main__":
    unittest.main()
