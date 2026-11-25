"""표준화, 경기 연결, 상태 검사 및 원본 판본 관리 테스트."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.datasets.published_sources import (
    SourceError,
    intake_sources,
    materialize_source,
    read_interim_rows,
)
from src.validation.standard_data import (
    StandardDataError,
    StandardStore,
    boundary_valid,
    compare_summaries,
    innings_to_outs,
    pa_state_valid,
    pitch_order_valid,
    regular_game_key,
    valid_pbp_row,
    valid_pitch_target,
)


def make_pbp_row(**changes: object) -> dict[str, object]:
    """실제 확인된 HF 구조와 동일한 최소 테스트 행."""
    return {
        "game_pk": "20230401HHWO02023",
        "game_date": "2023-04-01",
        "away_team": "HH",
        "home_team": "WO",
        "inning": 1,
        "inning_topbot": "top",
        "at_bat_number": 1,
        "pitch_number": 1,
        "batter": "63704",
        "batter_name": "타자",
        "pitcher": "68341",
        "pitcher_name": "투수",
        "balls": 0,
        "strikes": 0,
        "outs_when_up": 0,
        "on_1b": None,
        "on_2b": None,
        "on_3b": None,
        "home_score": 0,
        "away_score": 0,
        "pitch_type": "FF",
        "pitch_name": "4-Seam Fastball",
        "pitch_result": "T",
        "type": "S",
        "events": "strikeout",
        "post_home_score": 0,
        "post_away_score": 0,
        "post_outs": 1,
        "post_on_1b": None,
        "post_on_2b": None,
        "post_on_3b": None,
        "runs_scored": 0,
        **changes,
    }


class StandardValidationTests(unittest.TestCase):

    def test_regular_game_identifier(self) -> None:
        self.assertEqual(
            regular_game_key(
                "20230401HHWO02023",
                "2023-04-01",
                "HH",
                "WO",
                hf=True,
            ),
            "20230401HHWO0",
        )

    def test_wrong_hf_year_rejected(self) -> None:
        with self.assertRaises(StandardDataError):
            regular_game_key(
                "20230401HHWO02024",
                "2023-04-01",
                "HH",
                "WO",
                hf=True,
            )

    def test_home_and_away_mismatch_rejected(self) -> None:
        with self.assertRaises(StandardDataError):
            regular_game_key(
                "20230401HHWO0",
                "2023-04-01",
                "WO",
                "HH",
            )

    def test_innings_conversion(self) -> None:
        cases = {
            "0": 0,
            "1/3": 1,
            "2/3": 2,
            "1": 3,
            "5": 15,
            "5 2/3": 17,
            "9": 27,
        }

        for innings, outs in cases.items():
            with self.subTest(innings=innings):
                self.assertEqual(innings_to_outs(innings), outs)

    def test_decimal_innings_rejected(self) -> None:
        with self.assertRaises(StandardDataError):
            innings_to_outs("5.2")

    def test_invalid_state_range_rejected(self) -> None:
        self.assertFalse(
            valid_pbp_row(make_pbp_row(outs_when_up=3))
        )

        self.assertFalse(
            valid_pbp_row(make_pbp_row(balls=4))
        )

        self.assertFalse(
            valid_pbp_row(make_pbp_row(pitch_number=-1))
        )

    def test_zero_pitch_event_is_not_actual_pitch(self) -> None:
        row = make_pbp_row(
            pitch_number=0,
            pitch_type=None,
            pitch_name=None,
            pitch_result=None,
            type=None,
            events="walk",
        )

        self.assertTrue(pitch_order_valid([row]))
        self.assertFalse(valid_pitch_target(row))

    def test_missing_pitch_number(self) -> None:
        rows = [
            make_pbp_row(pitch_number=1),
            make_pbp_row(pitch_number=3),
        ]

        self.assertFalse(pitch_order_valid(rows))

    def test_repeated_pa_result_is_consistent(self) -> None:
        rows = [
            make_pbp_row(pitch_number=1),
            make_pbp_row(pitch_number=2, strikes=1),
            make_pbp_row(pitch_number=3, strikes=2),
        ]

        self.assertTrue(pa_state_valid(rows))
        self.assertTrue(pitch_order_valid(rows))

    def test_conflicting_after_state(self) -> None:
        rows = [
            make_pbp_row(pitch_number=1),
            make_pbp_row(
                pitch_number=2,
                post_outs=2,
            ),
        ]

        self.assertFalse(pa_state_valid(rows))

    def test_pitch_type_and_result_mapping(self) -> None:
        self.assertTrue(valid_pitch_target(make_pbp_row()))

        self.assertFalse(valid_pitch_target(
            make_pbp_row(pitch_type=None)
        ))

        self.assertFalse(valid_pitch_target(
            make_pbp_row(type="X")
        ))

    def test_bunt_result_codes(self) -> None:
        self.assertTrue(valid_pitch_target(
            make_pbp_row(pitch_result="W")
        ))

        self.assertTrue(valid_pitch_target(
            make_pbp_row(pitch_result="V")
        ))

    def test_same_half_boundary(self) -> None:
        previous = make_pbp_row(post_outs=1)

        following = make_pbp_row(
            at_bat_number=2,
            outs_when_up=1,
        )

        self.assertTrue(boundary_valid(previous, following))

    def test_half_inning_transition(self) -> None:
        previous = make_pbp_row(
            outs_when_up=2,
            post_outs=3,
        )

        following = make_pbp_row(
            inning_topbot="bot",
            at_bat_number=2,
            outs_when_up=0,
        )

        self.assertTrue(boundary_valid(previous, following))

    def test_unresolved_half_transition(self) -> None:
        previous = make_pbp_row(post_outs=2)

        following = make_pbp_row(
            inning_topbot="bot",
            at_bat_number=2,
            outs_when_up=0,
        )

        self.assertFalse(boundary_valid(previous, following))


class SourceIntakeTests(unittest.TestCase):

    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.base = Path(self.temp.name)

        self.published = self.base / "published"
        self.objects = self.published / "objects"
        self.runs = self.published / "runs"

        self.objects.mkdir(parents=True)
        self.runs.mkdir()

        self.expected = frozenset({"sample"})

        self.add_version(
            "a" * 32,
            "2026-11-25T00:00:00+00:00",
            "1" * 40,
            [{"game_id": "first"}],
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def add_version(
        self,
        run_id: str,
        created_at: str,
        revision: str,
        records: list[dict[str, object]],
    ) -> str:
        payload = json.dumps(records).encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()

        object_path = self.objects / f"{digest}.raw"

        if not object_path.exists():
            object_path.write_bytes(payload)

        run_dir = self.runs / run_id
        run_dir.mkdir()

        target = {
            "target_id": "sample",
            "source": "github",
            "repository": "example/sample",
            "filename": "sample.json",
            "kind": "schedule_results",
            "season": 2026,
            "competition": "regular",
            "revision": revision,
            "records_path": [],
            "required_fields": ["game_id"],
            "expected_sha256": None,
            "expected_bytes": None,
        }

        result = {
            "run_id": run_id,
            "target_id": "sample",
            "source": "github",
            "revision": revision,
            "identity": [
                "github",
                "example/sample",
                "sample.json",
                "schedule_results",
                2026,
                "regular",
            ],
            "status": "included",
            "body_complete": True,
            "sha256": digest,
            "bytes": len(payload),
            "object_path": f"objects/{digest}.raw",
            "inspection": {
                "rows": len(records),
            },
        }

        (run_dir / "manifest.json").write_text(
            json.dumps({
                "run_id": run_id,
                "created_at": created_at,
                "targets": [target],
            }),
            encoding="utf-8",
        )

        (run_dir / "sample.result.json").write_text(
            json.dumps(result),
            encoding="utf-8",
        )

        return digest

    def test_original_hash_verified(self) -> None:
        intake = intake_sources(
            self.published,
            expected_targets=self.expected,
        )

        self.assertEqual(intake.object_count, 1)
        self.assertEqual(len(intake.selected), 1)

    def test_modified_raw_rejected(self) -> None:
        path = next(self.objects.glob("*.raw"))
        path.write_bytes(b"corrupted")

        with self.assertRaises(SourceError):
            intake_sources(
                self.published,
                expected_targets=self.expected,
            )

    def test_missing_raw_rejected(self) -> None:
        next(self.objects.glob("*.raw")).unlink()

        with self.assertRaises(SourceError):
            intake_sources(
                self.published,
                expected_targets=self.expected,
            )

    def test_correction_and_previous_version(self) -> None:
        newest = self.add_version(
            "b" * 32,
            "2026-11-25T01:00:00+00:00",
            "2" * 40,
            [{"game_id": "corrected"}],
        )

        latest = intake_sources(
            self.published,
            expected_targets=self.expected,
        )

        historical = intake_sources(
            self.published,
            as_of_run="a" * 32,
            expected_targets=self.expected,
        )

        self.assertEqual(latest.object_count, 2)
        self.assertEqual(latest.selected["sample"].sha256, newest)

        self.assertNotEqual(
            latest.selected["sample"].sha256,
            historical.selected["sample"].sha256,
        )

    def test_same_content_is_not_new_object(self) -> None:
        self.add_version(
            "c" * 32,
            "2026-11-25T02:00:00+00:00",
            "1" * 40,
            [{"game_id": "first"}],
        )

        intake = intake_sources(
            self.published,
            expected_targets=self.expected,
        )

        self.assertEqual(intake.object_count, 1)
        self.assertEqual(len(intake.versions), 1)
        self.assertEqual(len(intake.history), 2)

    def test_interim_preserves_source_row_index(self) -> None:
        intake = intake_sources(
            self.published,
            expected_targets=self.expected,
        )

        output = self.base / "interim"
        output.mkdir()

        evidence = materialize_source(
            intake.selected["sample"],
            output,
        )

        rows = list(read_interim_rows(
            output / str(evidence["interim_file"])
        ))

        self.assertEqual(
            rows,
            [{"game_id": "first", "_source_row": 0}],
        )


class StandardStoreTests(unittest.TestCase):

    def test_same_data_reproduction(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)

            first = StandardStore(root / "first.sqlite3")
            second = StandardStore(root / "second.sqlite3")

            try:
                for store in (first, second):
                    store.issue(
                        "sample",
                        "test",
                        "test impact",
                    )

                previous = {
                    **first.summary(),
                    "input_fingerprint": "same-input",
                }

                current = {
                    **second.summary(),
                    "input_fingerprint": "same-input",
                }

                comparison = compare_summaries(
                    previous,
                    current,
                )

                self.assertTrue(comparison["reproducible"])

            finally:
                first.close()
                second.close()

    def test_correction_comparison(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)

            first = StandardStore(root / "first.sqlite3")
            second = StandardStore(root / "second.sqlite3")

            try:
                second.issue(
                    "corrected",
                    "test",
                    "changed content",
                )

                previous = {
                    **first.summary(),
                    "input_fingerprint": "old",
                }

                current = {
                    **second.summary(),
                    "input_fingerprint": "new",
                }

                comparison = compare_summaries(
                    previous,
                    current,
                )

                self.assertFalse(comparison["same_input"])
                self.assertIsNone(comparison["reproducible"])
                self.assertTrue(
                    comparison["changes"]["issues"]["content_changed"]
                )

            finally:
                first.close()
                second.close()


if __name__ == "__main__":
    unittest.main()