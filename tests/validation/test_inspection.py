"""실제 원본과 구분한 합성 Parquet으로 읽기 전용 검사를 검증한다."""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = None
    pq = None

from kbo_insight_ai.validation import raw_sample as inspection
from kbo_insight_ai.__main__ import main
from kbo_insight_ai.ingestion.pilot import HF_FILENAME, HF_REVISION, sha256
from kbo_insight_ai.settings import load_settings


SOURCE_ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "a" * 32
GAME_ID = "20230401HHWO02023"


@unittest.skipIf(pa is None, "inspection 선택 설치가 필요한 합성 Parquet 검사")
class InspectionTests(unittest.TestCase):
    """파일 동일성 전제와 행 근거·원본 비변경·실패 상태를 확인한다."""

    def setUp(self) -> None:
        """실제 경기와 무관한 다섯 행의 임시 Parquet과 이력을 만든다."""
        temporary = TemporaryDirectory(prefix="kbo-inspection-")
        self.addCleanup(temporary.cleanup)
        self.settings = replace(
            load_settings(SOURCE_ROOT, environ={}),
            raw_dir=Path(temporary.name) / "원본",
        )
        self.run_dir = self.settings.raw_dir / "collection" / RUN_ID
        self.run_dir.mkdir(parents=True)
        row = {
            "game_pk": GAME_ID, "game_date": "2023-04-01",
            "inning": 1, "inning_topbot": "top", "at_bat_number": 1,
            "pitch_number": 1, "batter": "A", "pitcher": "P1",
            "balls": 0, "strikes": 0, "outs_when_up": 0,
            "on_1b": None, "on_2b": None, "on_3b": None,
            "events": "walk", "pitch_type": "FF", "pitch_result": "B", "type": "B",
            "home_score": 0, "away_score": 0,
            "post_home_score": 0, "post_away_score": 0, "post_outs": 0,
            "post_on_1b": "B", "post_on_2b": None, "post_on_3b": None,
            "release_speed_kmh": 140.0, "plate_x": 0.1, "plate_z": 2.0,
        }
        rows = [
            {**row, "game_pk": "20230401HTSK02023"},
            row,
            {**row, "pitch_number": 2, "batter": "B", "pitcher": "P2",
             "pitch_type": "SL", "pitch_result": "F", "type": "S"},
            {**row, "at_bat_number": 2, "pitch_number": 0, "batter": "C",
             "pitch_type": None, "pitch_result": None, "type": None,
             "release_speed_kmh": None, "plate_x": None, "plate_z": None},
            {**row, "at_bat_number": 3, "inning_topbot": "bot",
             "batter": "D", "pitch_result": "W", "type": "S"},
        ]
        self.table = pa.Table.from_pylist(rows)
        self.path = self.run_dir / "attempt-01.parquet"
        pq.write_table(self.table, self.path)
        self._set_expected_file()

    def _set_expected_file(self) -> None:
        """합성 파일 해시를 검사 상수와 이력에만 주입한다."""
        digest = sha256(self.path.read_bytes())
        size = self.path.stat().st_size
        size_patch = patch.object(inspection, "HF_SAMPLE_BYTES", size)
        hash_patch = patch.object(inspection, "HF_SAMPLE_SHA256", digest)
        size_patch.start()
        hash_patch.start()
        self.addCleanup(size_patch.stop)
        self.addCleanup(hash_patch.stop)
        request = {
            "run_id": RUN_ID, "source": "huggingface_parquet",
            "source_version": HF_REVISION,
            "request": {
                "filename": HF_FILENAME, "expected_bytes": size,
                "expected_sha256": digest,
            },
        }
        self.record = {
            "run_id": RUN_ID, "status": "artifact_verified",
            "body_complete": True, "bytes": size, "sha256": digest,
            "body_path": "attempt-01.parquet",
            "semantic_validation": "not_performed",
        }
        self._write("request.json", request)
        self._write("result.json", self.record)

    def _write(self, name: str, value: object) -> None:
        """검사 대상 임시 이력만 기록한다."""
        (self.run_dir / name).write_text(json.dumps(value), encoding="utf-8")

    def test_rows_and_metadata_without_mutating_raw(self) -> None:
        """투구 사례를 원본 행에 연결하고 원본과 이력을 변경하지 않는다."""
        before = {p.name: p.read_bytes() for p in self.run_dir.iterdir()}
        result = inspection.inspect_sample(self.settings, RUN_ID)
        self.assertEqual(result["physical_file_rows"], 5)
        self.assertEqual(result["selected_game_rows"], 4)
        cases = result["case_rows"]
        pitch_fields = (
            "_source_row_index", "pitch_number", "pitch_type", "pitch_result", "type",
        )
        self.assertEqual(
            [tuple(row[field] for field in pitch_fields) for row in cases["first_pa"]],
            [(1, 1, "FF", "B", "B"), (2, 2, "SL", "F", "S")],
        )
        zero_pitch = cases["pitch_zero"][0]
        self.assertEqual(
            tuple(zero_pitch[field] for field in pitch_fields),
            (3, 0, None, None, None),
        )
        changes = result["participant_changes"]
        self.assertEqual(changes["batter_count_distinct"]["pa_key_count"], 1)
        self.assertEqual(changes["pitcher_count_distinct"]["pa_key_count"], 1)
        self.assertEqual(result["quality_status"], "not_assessed")
        self.assertEqual(result["observed_counts"]["zero_pitch_rows"], 1)
        self.assertEqual(result["observed_counts"]["positive_pitch_rows"], 4)
        self.assertEqual(cases["selected_game_last_pa"][0]["_source_row_index"], 4)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.run_dir.iterdir()})

    def test_tampered_bytes_are_rejected(self) -> None:
        """기존 이력이 남아 있어도 원본 bytes가 변하면 읽기를 거절한다."""
        original = self.path.read_bytes()
        self.path.write_bytes(b"X" + original[1:])
        with self.assertRaisesRegex(inspection.InspectionError, "해시"):
            inspection.inspect_sample(self.settings, RUN_ID)

    def test_invalid_run_and_manifest_are_rejected(self) -> None:
        """다른 실행·불완전·경로 주입·미검증 파일을 검사하지 않는다."""
        for changes in (
            {"run_id": "b" * 32},
            {"body_complete": False},
            {"status": "failed"},
            {"body_path": "../outside.parquet"},
        ):
            with self.subTest(changes=changes):
                self._write("result.json", {**self.record, **changes})
                with self.assertRaises(inspection.InspectionError):
                    inspection.inspect_sample(self.settings, RUN_ID)
        with self.assertRaises(inspection.InspectionError):
            inspection.inspect_sample(self.settings, "../outside")

    def test_missing_fields_are_not_reported_as_success(self) -> None:
        """해시가 맞는 유효 Parquet이라도 필수 필드가 없으면 실패한다."""
        pq.write_table(pa.table({"unexpected": [1]}), self.path)
        self._set_expected_file()
        with self.assertRaisesRegex(inspection.InspectionError, "필드"):
            inspection.inspect_sample(self.settings, RUN_ID)

    def test_cli_inspection_error_has_separate_exit_code(self) -> None:
        """원본 검사 실패를 수집 성공이나 설정 성공으로 반환하지 않는다."""
        with (
            patch("kbo_insight_ai.__main__.load_settings", return_value=self.settings),
            patch.object(inspection, "inspect_sample",
                         side_effect=inspection.InspectionError("검사 오류 대역")),
            redirect_stdout(StringIO()),
            redirect_stderr(StringIO()),
        ):
            self.assertEqual(main(["inspect-sample", "--run-id", RUN_ID]), 5)

    def test_missing_selected_game_is_rejected(self) -> None:
        """없는 경기 표본을 빈 성공 결과로 바꾸지 않는다."""
        with self.assertRaisesRegex(inspection.InspectionError, "선택 경기"):
            inspection.inspect_sample(self.settings, RUN_ID, "not-a-present-game")


if __name__ == "__main__":
    unittest.main()