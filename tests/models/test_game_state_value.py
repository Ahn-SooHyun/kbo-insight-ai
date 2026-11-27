"""경기 상황 가치의 상태 계약·통계·종료·재현 테스트."""

from dataclasses import asdict, replace
from pathlib import Path
import json
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from src.datasets.model_data import fingerprint
from src.evaluation import game_state_value as evaluation
from src.models.game_state_value import SCHEMA, VERSION
from src.models.game_state_value import data
from src.models.game_state_value import model as m4
from src.models.game_state_value import value
from src.models.game_state_value.__main__ import (
    load_model,
    method_settings,
    save_json,
    select_predictor,
)


SOURCE_SHA = "a" * 64


def state(**changes: object) -> m4.State:
    """정상적인 정규시즌 9회말 동점 사전 상태를 생성한다."""
    raw = {
        "season": 2024,
        "competition": "regular",
        "home_team_id": 2001,
        "away_team_id": 2002,
        "perspective_team_id": 2001,
        "inning": 9,
        "half": "bot",
        "phase": "pre",
        "home_score": 3,
        "away_score": 3,
        "outs": 0,
        "on_first": 0,
        "on_second": 0,
        "on_third": 0,
        "home_season_games": 100,
        "home_season_rf": 500,
        "home_season_ra": 450,
        "home_season_wins": 55,
        "away_season_games": 100,
        "away_season_rf": 450,
        "away_season_ra": 500,
        "away_season_wins": 45,
    }

    raw.update(changes)

    return m4.parse_state(raw)


def evidence(
    post: m4.State,
    **changes: object,
) -> m4.FinalEvidence:
    """합성 경기의 공식 종료 근거를 생성한다."""
    winner = (
        "home" if post.home_score > post.away_score
        else "away" if post.away_score > post.home_score
        else "tie"
    )

    terminal = (
        "final_tie" if winner == "tie"
        else "half_ended_three_outs" if post.outs == 3
        else "home_leads_partial_bottom_candidate"
    )

    raw = {
        "game_id": "synthetic-game",
        "pa_number": 70,
        "last_pa": 70,
        "source_sha": SOURCE_SHA,
        "end_source_row": 100,
        "official_status": "Final",
        "official_home_score": post.home_score,
        "official_away_score": post.away_score,
        "winner": winner,
        "result_valid": 1,
        "final_score_match": 1,
        "order_valid": 1,
        "boundary_valid": 1,
        "terminal_state": terminal,
    }

    raw.update(changes)

    result = m4.parse_final(raw)
    assert result is not None

    return result


def pair(
    *,
    terminal: bool = False,
    index: int = 0,
    split: str = "train",
) -> data.Pair:
    """동일 사전 상태에서 두 종류의 다음 상태를 구성한다."""
    pre = state()

    if terminal:
        post = replace(
            pre,
            phase="post",
            home_score=4,
        )
        final = evidence(post)
        runs = 1
        win = 1

    else:
        post = replace(
            pre,
            phase="post",
            outs=1,
        )
        final = None
        runs = 0
        win = 0

    return data.Pair(
        game_id=f"synthetic-{index}",
        pair_id=f"synthetic-{index}:1",
        sample_date="2024-05-01" if split == "train" else "2025-05-01",
        split=split,
        pre=m4.Observation(pre, win, runs, None),
        post=m4.Observation(
            post,
            win,
            0,
            m4.final_result(post, final),
        ),
        final=final,
        actual_runs=runs,
    )


def training_pairs() -> list[data.Pair]:
    """최소 지원을 충족하고 두 이진 정답을 포함한 합성 자료."""
    return [
        pair(terminal=index % 2 == 0, index=index)
        for index in range(40)
    ]


def trained_asset() -> value.ValueModel:
    """테스트용 단일 예측기와 통계표를 학습한다."""
    pairs = training_pairs()
    rows = data.observations(pairs)

    predictor = m4.fit_frequency(rows)
    transitions = value.fit_transitions(pairs)

    return {
        "predictor": predictor,
        "re_table": value.fit_re(rows),
        "transitions": transitions,
        "reference": value.fit_reference(
            predictor,
            transitions,
            pairs,
        ),
    }


class StateContractTests(unittest.TestCase):
    """사전 상태의 입력 허용 목록과 관점 계약."""

    def test_full_and_selected_input_contract(self) -> None:
        self.assertEqual(len(m4.ALL_COLUMNS), 34)
        self.assertEqual(len(m4.INPUT_COLUMNS), 22)
        self.assertLessEqual(
            set(m4.INPUT_COLUMNS),
            set(m4.ALL_COLUMNS),
        )

    def test_attack_team_perspective(self) -> None:
        away = state(
            half="top",
            perspective_team_id=2002,
        )
        self.assertFalse(away.attacking_home)

        with self.assertRaises(ValueError):
            state(half="top", perspective_team_id=2001)

    def test_invalid_outs_and_bases(self) -> None:
        for changes in (
            {"outs": 3},
            {"outs": -1},
            {"on_first": 2},
            {"on_second": True},
        ):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    state(**changes)

    def test_invalid_score(self) -> None:
        for invalid in (None, -1, 1.5, True, "3"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    state(home_score=invalid)

    def test_invalid_competition(self) -> None:
        with self.assertRaises(ValueError):
            state(competition="postseason")

    def test_missing_and_extra_fields(self) -> None:
        raw = asdict(state())
        raw.pop("outs")

        with self.assertRaises(ValueError):
            m4.parse_state(raw)

        raw = asdict(state())
        raw["target_result"] = "win"

        with self.assertRaises(ValueError):
            m4.parse_state(raw)

    def test_future_and_audit_columns_rejected(self) -> None:
        for name in (
            "game_id", "sample_id", "pair_id",
            "target_remaining_runs", "terminal_evidence",
            "post_home_score", "final_score_match",
        ):
            raw = asdict(state())
            raw[name] = 1

            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    m4.parse_state(raw)

    def test_history_missing_supported(self) -> None:
        missing = state(
            home_season_games=0,
            home_season_rf=None,
            home_season_ra=None,
            home_season_wins=None,
        )

        vector = m4.features(missing, True)

        self.assertEqual(len(vector), 22)
        self.assertEqual(vector[16], 1.0)

    def test_invalid_history_counts(self) -> None:
        with self.assertRaises(ValueError):
            state(home_season_games=1, home_season_wins=2)

        with self.assertRaises(ValueError):
            state(home_season_games=0, home_season_rf=1)

    def test_phase_not_model_feature(self) -> None:
        pre = state()
        post = replace(pre, phase="post")

        self.assertEqual(
            m4.features(pre, True),
            m4.features(post, True),
        )

        self.assertEqual(
            m4.frequency_keys(pre),
            m4.frequency_keys(post),
        )

    def test_three_outs_base_normalization(self) -> None:
        post = state(
            phase="post",
            outs=3,
            on_first=1,
            on_third=1,
        )

        self.assertEqual(post.mask, 0)


class PairAndEndingTests(unittest.TestCase):
    """관측 전후 상태와 공식 종료 근거의 적용 범위."""

    def test_actual_runs(self) -> None:
        pre = state()
        post = replace(pre, phase="post", home_score=5)

        self.assertEqual(m4.validate_pair(pre, post), 2)

    def test_pair_different_inning(self) -> None:
        pre = state()
        post = replace(pre, phase="post", inning=10)

        with self.assertRaises(ValueError):
            m4.validate_pair(pre, post)

    def test_pair_history_change(self) -> None:
        pre = state()
        post = replace(
            pre,
            phase="post",
            home_season_wins=56,
        )

        with self.assertRaises(ValueError):
            m4.validate_pair(pre, post)

    def test_pair_opponent_runs(self) -> None:
        pre = state()
        post = replace(pre, phase="post", away_score=4)

        with self.assertRaises(ValueError):
            m4.validate_pair(pre, post)

    def test_pair_outs_decrease(self) -> None:
        pre = state(outs=2)
        post = replace(pre, phase="post", outs=1)

        with self.assertRaises(ValueError):
            m4.validate_pair(pre, post)

    def test_pair_score_decrease(self) -> None:
        pre = state()
        post = replace(pre, phase="post", home_score=2)

        with self.assertRaises(ValueError):
            m4.validate_pair(pre, post)

    def test_confirmed_final_win(self) -> None:
        post = state(phase="post", home_score=4)

        self.assertEqual(
            m4.final_result(post, evidence(post)),
            1,
        )

    def test_confirmed_final_loss(self) -> None:
        post = state(
            phase="post",
            away_score=4,
            outs=3,
        )

        self.assertEqual(
            m4.final_result(post, evidence(post)),
            0,
        )

    def test_confirmed_tie_is_not_win(self) -> None:
        post = state(
            phase="post",
            inning=12,
            outs=3,
        )

        self.assertEqual(
            m4.final_result(post, evidence(post)),
            0,
        )

    def test_three_outs_not_automatically_final(self) -> None:
        post = state(phase="post", inning=5, outs=3)

        self.assertIsNone(m4.final_result(post, None))

        asset = trained_asset()
        result = value.state_value(asset, post)

        self.assertEqual(result["remaining_expected_runs"], 0.0)
        self.assertFalse(result["game_end_confirmed"])
        self.assertGreater(result["win_probability"], 0.0)
        self.assertLess(result["win_probability"], 1.0)

    def test_home_lead_not_automatically_final(self) -> None:
        post = state(phase="post", home_score=5)

        self.assertIsNone(m4.final_result(post, None))

        asset = trained_asset()
        result = value.state_value(asset, post)

        self.assertFalse(result["game_end_confirmed"])
        self.assertLess(result["win_probability"], 1.0)

    def test_repeated_half_evidence_not_final(self) -> None:
        post = state(phase="post", home_score=4)

        invalid = evidence(
            post,
            pa_number=68,
            last_pa=70,
        )

        with self.assertRaises(ValueError):
            m4.final_result(post, invalid)

    def test_final_score_mismatch(self) -> None:
        post = state(phase="post", home_score=4)

        invalid = evidence(post, official_home_score=5)

        with self.assertRaises(ValueError):
            m4.final_result(post, invalid)

    def test_final_quality_failure(self) -> None:
        post = state(phase="post", home_score=4)

        for name in (
            "result_valid", "final_score_match",
            "order_valid", "boundary_valid",
        ):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    m4.final_result(
                        post,
                        evidence(post, **{name: 0}),
                    )

    def test_final_requires_post_boundary(self) -> None:
        pre = state(home_score=4)
        post = replace(pre, phase="post")

        with self.assertRaises(ValueError):
            m4.final_result(pre, evidence(post))

    def test_partial_final_reason_not_walkoff_by_itself(self) -> None:
        asset = trained_asset()

        pre = state(home_score=4)
        post = replace(pre, phase="post", home_score=5)

        result = value.plate_value(
            asset,
            pre,
            post,
            evidence(post),
        )

        self.assertFalse(result["observed_walkoff_pattern"])
        self.assertTrue(result["post"]["game_end_confirmed"])


class PredictorTests(unittest.TestCase):
    """기준 확률, 희소 후퇴 및 로지스틱 회귀 재현."""

    def test_frequency_smoothing(self) -> None:
        rows = [
            m4.Observation(state(), 1, 1, None)
            for _ in range(40)
        ]

        predictor = m4.fit_frequency(rows)

        probability, method, support = m4.predict_win(
            predictor, state()
        )

        self.assertAlmostEqual(probability, 41 / 42)
        self.assertEqual(method, "exact")
        self.assertEqual(support, 40)

    def test_frequency_sparse_backoff(self) -> None:
        predictor = trained_asset()["predictor"]

        probability, method, support = m4.predict_win(
            predictor,
            state(inning=1, home_score=20),
        )

        self.assertEqual(method, "global")
        self.assertGreater(support, 0)
        self.assertGreater(probability, 0)
        self.assertLess(probability, 1)

    def test_confirmed_final_bypasses_predictor(self) -> None:
        predictor = trained_asset()["predictor"]
        post = state(phase="post", home_score=4)

        result = m4.predict_win(predictor, post, 1)

        self.assertEqual(result, (1.0, "confirmed_final", None))

    def test_logistic_reproducible(self) -> None:
        rows = data.observations(training_pairs())

        for name in (
            "logistic_core",
            "logistic_history",
            "logistic_history_l2",
        ):
            with self.subTest(name=name):
                first = m4.fit_logistic(rows, name, 42)
                second = m4.fit_logistic(rows, name, 42)

                self.assertEqual(first, second)
                m4.validate_predictor(first)

                probability = m4.predict_win(first, state())[0]

                self.assertGreater(probability, 0)
                self.assertLess(probability, 1)

    def test_logistic_single_class_error(self) -> None:
        rows = [
            m4.Observation(state(), 1, 0, None)
            for _ in range(40)
        ]

        with self.assertRaises(ValueError):
            m4.fit_logistic(rows, "logistic_core", 42)

    def test_invalid_logistic_asset(self) -> None:
        rows = data.observations(training_pairs())

        predictor = m4.fit_logistic(
            rows, "logistic_core", 42
        )

        predictor["scales"][0] = 0.0

        with self.assertRaises(ValueError):
            m4.validate_predictor(predictor)

    def test_probability_numerical_stability(self) -> None:
        self.assertGreater(m4.sigmoid(-1000.0), 0.0)
        self.assertLess(m4.sigmoid(1000.0), 1.0)

    def test_candidate_selection_same_denominator(self) -> None:
        rows = data.observations(training_pairs())

        selected, metrics = select_predictor(
            rows,
            rows,
            42,
        )

        self.assertEqual(set(metrics), set(m4.METHODS))
        self.assertIn(selected["name"], m4.METHODS)

        for report in metrics.values():
            self.assertEqual(
                report["groups"]["all"]["count"],
                len(rows),
            )


class ValueTests(unittest.TestCase):
    """기대득점·WPA·상황 중요도와 미지원 처리."""

    def test_re_mean_and_backoff(self) -> None:
        rows = [
            m4.Observation(state(), 1, 2, None)
            for _ in range(40)
        ]

        table = value.fit_re(rows)

        self.assertEqual(
            value.predict_re(table, state()),
            (2.0, "exact", 40),
        )

        self.assertEqual(
            value.predict_re(table, state(outs=2))[1],
            "global",
        )

    def test_re_three_outs_zero(self) -> None:
        asset = trained_asset()

        post = state(phase="post", outs=3)

        self.assertEqual(
            value.predict_re(asset["re_table"], post)[0],
            0.0,
        )

    def test_re_partial_final_zero(self) -> None:
        asset = trained_asset()
        post = state(phase="post", home_score=4)

        self.assertEqual(
            value.predict_re(asset["re_table"], post, 1)[0],
            0.0,
        )

    def test_re_inconsistent_ending_target(self) -> None:
        rows = [
            m4.Observation(
                state(phase="post", outs=3),
                1,
                2,
                None,
            )
        ]

        with self.assertRaises(ValueError):
            value.fit_re(rows)

    def test_wpa_and_run_value(self) -> None:
        asset = trained_asset()
        observed = pair(terminal=True)

        result = value.plate_value(
            asset,
            observed.pre.state,
            observed.post.state,
            observed.final,
        )

        before = result["pre"]
        after = result["post"]

        self.assertEqual(result["actual_runs"], 1)
        self.assertEqual(after["win_probability"], 1.0)
        self.assertEqual(after["remaining_expected_runs"], 0.0)

        self.assertAlmostEqual(
            result["wpa"],
            after["win_probability"] - before["win_probability"],
        )

        self.assertAlmostEqual(
            result["run_value"],
            1 + result["remaining_expected_runs_change"],
        )

    def test_importance_distribution(self) -> None:
        asset = trained_asset()
        result = value.importance(asset, state())

        self.assertEqual(result["support"], 40)
        self.assertEqual(len(result["transitions"]), 2)

        self.assertAlmostEqual(
            sum(
                row["probability"]
                for row in result["transitions"]
            ),
            1.0,
        )

        self.assertAlmostEqual(result["value"], 1.0)

    def test_importance_independent_of_observed_post(self) -> None:
        asset = trained_asset()
        win = pair(terminal=True)
        ordinary = pair(terminal=False)

        first = value.predict(asset, value.pair_input(win))
        second = value.predict(asset, value.pair_input(ordinary))

        self.assertEqual(
            first["importance"],
            second["importance"],
        )

        self.assertNotEqual(first["actual_runs"], second["actual_runs"])

    def test_importance_sparse_state(self) -> None:
        asset = trained_asset()
        result = value.importance(asset, state(inning=1))

        self.assertIsNone(result["value"])

        self.assertEqual(
            result["reason"],
            "insufficient_transition_support",
        )

    def test_importance_zero_reference(self) -> None:
        asset = trained_asset()
        asset["reference"]["mean"] = 0.0

        result = value.importance(asset, state())

        self.assertIsNone(result["value"])

        self.assertEqual(
            result["reason"],
            "unsupported_reference_mean",
        )

    def test_importance_post_not_supported(self) -> None:
        asset = trained_asset()

        result = value.importance(
            asset,
            state(phase="post"),
        )

        self.assertIsNone(result["value"])
        self.assertEqual(result["reason"], "pre_state_required")

    def test_validation_not_allowed_in_transition_fit(self) -> None:
        with self.assertRaises(ValueError):
            value.fit_transitions([
                pair(split="validation")
            ])

    def test_validation_not_allowed_in_reference_fit(self) -> None:
        asset = trained_asset()

        with self.assertRaises(ValueError):
            value.fit_reference(
                asset["predictor"],
                asset["transitions"],
                [pair(split="validation")],
            )

    def test_invalid_transition_rejected(self) -> None:
        for code in (
            "0|4|0|-1",
            "-1|1|0|-1",
            "0|3|1|-1",
            "0|1|0|2",
        ):
            with self.subTest(code=code):
                with self.assertRaises(ValueError):
                    value.decode_transition(code)

    def test_asset_transition_support_validation(self) -> None:
        asset = trained_asset()

        value.validate_asset(asset)
        asset["reference"]["supported"] -= 1

        with self.assertRaises(ValueError):
            value.validate_asset(asset)

    def test_state_only_prediction(self) -> None:
        asset = trained_asset()

        result = value.predict(asset, {
            "state": asdict(state()),
            "final": None,
        })

        self.assertIn("win_probability", result)
        self.assertIn("remaining_expected_runs", result)

    def test_prediction_rejects_target(self) -> None:
        asset = trained_asset()

        with self.assertRaises(ValueError):
            value.predict(asset, {
                "state": asdict(state()),
                "final": None,
                "target_result": "win",
            })


class EvaluationAndAssetTests(unittest.TestCase):
    """평가 산식과 운영 자산 복원 검사."""

    def test_probability_metrics(self) -> None:
        metric = evaluation.ProbabilityMetrics()

        metric.add(0.8, 1)
        metric.add(0.2, 0)

        result = metric.result()

        self.assertAlmostEqual(result["brier"], 0.04)
        self.assertAlmostEqual(result["ece"], 0.2)
        self.assertEqual(result["count"], 2)

        self.assertEqual(
            sum(
                row["count"]
                for row in result["calibration"]
            ),
            2,
        )

    def test_perfect_final_probability_metrics(self) -> None:
        metric = evaluation.ProbabilityMetrics()

        metric.add(1.0, 1)
        metric.add(0.0, 0)

        self.assertEqual(metric.result()["log_loss"], 0.0)
        self.assertEqual(metric.result()["brier"], 0.0)

    def test_numeric_metrics(self) -> None:
        metric = evaluation.NumericMetrics()

        metric.add(1.0, 0)
        metric.add(3.0, 1)

        result = metric.result()

        self.assertAlmostEqual(result["mae"], 1.5)
        self.assertAlmostEqual(result["rmse"], (2.5) ** 0.5)

    def test_same_sample_re_evaluation(self) -> None:
        asset = trained_asset()
        rows = data.observations(training_pairs())

        report = evaluation.evaluate_re(
            asset["re_table"],
            rows,
        )

        for candidate in report["candidates"].values():
            self.assertEqual(
                candidate["all"]["count"],
                len(rows),
            )

    def test_pair_evaluation_support_count(self) -> None:
        asset = trained_asset()

        pairs = [
            pair(terminal=True, split="validation"),
            pair(terminal=False, split="validation"),
        ]

        report = evaluation.evaluate_pairs(asset, pairs)

        self.assertEqual(report["pairs"], 2)
        self.assertEqual(report["importance"]["supported"], 2)
        self.assertEqual(report["importance"]["coverage"], 1.0)

    def test_asset_roundtrip(self) -> None:
        asset = trained_asset()
        inputs = value.pair_input(pair(terminal=True))

        payload = {
            "schema": SCHEMA,
            "version": VERSION,
            "training_run_id": "b" * 32,
            "configured_seed": 42,
            "selected_model": asset["predictor"]["name"],
            "operational_predictor_count": 1,
            "input_columns": list(m4.INPUT_COLUMNS),
            "allowed_input_columns": list(m4.ALL_COLUMNS),
            "method_settings": method_settings(),
            "model": asset,
        }

        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"

            save_json(path, {
                "payload": payload,
                "sha256": fingerprint(payload),
            })

            restored = load_model(path)

            self.assertEqual(
                value.predict(asset, inputs),
                value.predict(restored, inputs),
            )

    def test_asset_fingerprint_tampering(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"

            save_json(path, {
                "payload": {"schema": SCHEMA},
                "sha256": "0" * 64,
            })

            with self.assertRaises(ValueError):
                load_model(path)

    def test_json_no_overwrite(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "asset.json"

            save_json(path, {"value": 1})

            with self.assertRaises(FileExistsError):
                save_json(path, {"value": 2})

            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"value": 1},
            )


class DataIntegrationTests(unittest.TestCase):
    """합성 SQLite에서 실제 자료 조회 SQL과 종료 연결을 검증한다."""

    def setUp(self) -> None:
        self.db = sqlite3.connect(":memory:")
        self.db.execute(
            "ATTACH DATABASE ':memory:' AS verified"
        )

        self.db.executescript(f"""
            CREATE TABLE m4_index (
                sample_id, game_id, sample_date,
                split, status, pair_id, pa_number,
                phase, terminal_evidence, source_sha,
                start_source_row, end_source_row
            );

            CREATE TABLE m4_inputs (
                sample_id,
                {", ".join(m4.ALL_COLUMNS)}
            );

            CREATE TABLE m4_targets (
                sample_id, target_result, target_remaining_runs
            );

            CREATE TABLE verified.games (
                game_id, season, competition, date,
                home_team_id, away_team_id,
                status, home_score, away_score,
                winner, result_valid
            );

            CREATE TABLE verified.game_checks (
                game_id, final_score_match,
                order_valid, boundary_valid, terminal_state
            );

            CREATE TABLE verified.plate_appearances (
                game_id, pa_number, inning, half,
                pre_home_score, pre_away_score, pre_outs,
                pre_on_1b, pre_on_2b, pre_on_3b,
                post_home_score, post_away_score, post_outs,
                post_on_1b, post_on_2b, post_on_3b,
                source_sha, start_source_row, end_source_row
            );

            CREATE VIEW verified.eligible_states AS
            SELECT * FROM plate_appearances;
        """)

        self.db.execute("""
            INSERT INTO verified.games
            VALUES (
                'g', 2025, 'regular', '2025-05-01',
                2001, 2002, 'Final', 4, 3, 'home', 1
            )
        """)

        self.db.execute("""
            INSERT INTO verified.game_checks
            VALUES (
                'g', 1, 1, 1,
                'home_leads_partial_bottom_candidate'
            )
        """)

        first_pre = state(season=2025)

        first_post = replace(
            first_pre,
            phase="post",
            outs=1,
        )

        second_pre = replace(first_post, phase="pre")

        second_post = replace(
            second_pre,
            phase="post",
            home_score=4,
        )

        self.insert_pair(1, first_pre, first_post, 1, 1)
        self.insert_pair(2, second_pre, second_post, 1, 0)

    def tearDown(self) -> None:
        self.db.close()

    def insert_row(
        self,
        table: str,
        values: dict[str, object],
    ) -> None:
        columns = ", ".join(values)
        placeholders = ", ".join("?" for _ in values)

        self.db.execute(
            f"INSERT INTO {table} ({columns}) VALUES ({placeholders})",
            tuple(values.values()),
        )

    def insert_pair(
        self,
        number: int,
        pre: m4.State,
        post: m4.State,
        before_remaining: int,
        after_remaining: int,
    ) -> None:
        pair_id = f"g:{number}"

        raw = {
            "game_id": "g",
            "pa_number": number,
            "inning": pre.inning,
            "half": pre.half,
            "source_sha": SOURCE_SHA,
            "start_source_row": number * 10,
            "end_source_row": number * 10 + 1,
        }

        for phase, current in (
            ("pre", pre),
            ("post", post),
        ):
            raw[f"{phase}_home_score"] = current.home_score
            raw[f"{phase}_away_score"] = current.away_score
            raw[f"{phase}_outs"] = current.outs

            for base, name in (
                ("1b", "on_first"),
                ("2b", "on_second"),
                ("3b", "on_third"),
            ):
                raw[f"{phase}_on_{base}"] = (
                    "runner" if getattr(current, name) else None
                )

        self.insert_row("verified.plate_appearances", raw)

        for current, remaining in (
            (pre, before_remaining),
            (post, after_remaining),
        ):
            sample_id = f"{pair_id}:{current.phase}"

            self.insert_row("m4_index", {
                "sample_id": sample_id,
                "game_id": "g",
                "sample_date": "2025-05-01",
                "split": "validation",
                "status": "eligible",
                "pair_id": pair_id,
                "pa_number": number,
                "phase": current.phase,
                "terminal_evidence": "home_leads_partial_bottom_candidate",
                "source_sha": SOURCE_SHA,
                "start_source_row": number * 10,
                "end_source_row": number * 10 + 1,
            })

            inputs = {
                name: None
                for name in m4.ALL_COLUMNS
            }

            inputs.update(asdict(current))
            inputs["sample_id"] = sample_id

            self.insert_row("m4_inputs", inputs)

            self.insert_row("m4_targets", {
                "sample_id": sample_id,
                "target_result": "win",
                "target_remaining_runs": remaining,
            })

    def test_repeated_half_evidence_only_last_pa_is_final(self) -> None:
        data.verify_keys(self.db)

        pairs = data.read_pairs(self.db)["validation"]

        self.assertEqual(len(pairs), 2)

        self.assertIsNone(pairs[0].pre.final)
        self.assertIsNone(pairs[0].post.final)
        self.assertIsNone(pairs[0].final)

        self.assertIsNone(pairs[1].pre.final)
        self.assertEqual(pairs[1].post.final, 1)
        self.assertIsNotNone(pairs[1].final)
        self.assertEqual(pairs[1].actual_runs, 1)

    def test_source_state_mismatch(self) -> None:
        self.db.execute("""
            UPDATE m4_inputs
            SET home_score = 10
            WHERE sample_id = 'g:1:pre'
        """)

        with self.assertRaises(ValueError):
            data.read_pairs(self.db)

    def test_wrong_target_remaining_runs(self) -> None:
        self.db.execute("""
            UPDATE m4_targets
            SET target_remaining_runs = 99
            WHERE sample_id = 'g:1:pre'
        """)

        with self.assertRaises(ValueError):
            data.read_pairs(self.db)

    def test_duplicate_input_key(self) -> None:
        self.db.execute("""
            INSERT INTO m4_inputs
            SELECT *
            FROM m4_inputs
            WHERE sample_id = 'g:1:pre'
        """)

        with self.assertRaises(ValueError):
            data.verify_keys(self.db)

    def test_missing_target_key(self) -> None:
        self.db.execute("""
            DELETE FROM m4_targets
            WHERE sample_id = 'g:1:post'
        """)

        with self.assertRaises(ValueError):
            data.verify_keys(self.db)

    def test_wrong_split(self) -> None:
        self.db.execute("""
            UPDATE m4_index
            SET split = 'train'
            WHERE sample_id = 'g:1:pre'
        """)

        with self.assertRaises(ValueError):
            data.verify_keys(self.db)

    def test_final_score_mismatch(self) -> None:
        self.db.execute("""
            UPDATE verified.games
            SET home_score = 5
            WHERE game_id = 'g'
        """)

        with self.assertRaises(ValueError):
            data.read_pairs(self.db)

    def test_test_target_values_not_read(self) -> None:
        for phase in ("pre", "post"):
            sample_id = f"test-game:1:{phase}"

            self.insert_row("m4_index", {
                "sample_id": sample_id,
                "game_id": "test-game",
                "sample_date": "2026-04-01",
                "split": "test",
                "status": "eligible",
                "pair_id": "test-game:1",
                "pa_number": 1,
                "phase": phase,
                "terminal_evidence": "three_outs",
                "source_sha": SOURCE_SHA,
                "start_source_row": 1,
                "end_source_row": 2,
            })

            self.insert_row("m4_inputs", {
                "sample_id": sample_id,
            })

            self.insert_row("m4_targets", {
                "sample_id": sample_id,
                "target_result": "invalid_test_target",
                "target_remaining_runs": -999,
            })

        # 키·분할 검사에는 성공하되 Test의 비정상 정답 값은
        # 개발 표본을 조회하는 과정에서 참조하지 않는다.
        data.verify_keys(self.db)

        pairs = data.read_pairs(self.db)

        self.assertEqual(len(pairs["validation"]), 2)
        self.assertEqual(len(pairs["train"]), 0)


if __name__ == "__main__":
    unittest.main()