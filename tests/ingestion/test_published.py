"""합성 배포 파일로 전체 수집·버전·재개 및 승인 경계를 검증한다."""

from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.ingestion import published
from src.ingestion.pilot import Attempt, sha256
from src.settings import load_settings


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIRST_BODY = b"game_id,state,score\nG1,final,3\n"
CORRECTED_BODY = b"game_id,state,score\nG1,final,4\n"


class PublishedCollectionTests(unittest.TestCase):
    """실제 경기 자료와 외부 연결을 사용하지 않는 검사."""

    def setUp(self) -> None:
        """각 검사에 독립적인 원본 저장 위치를 사용한다."""
        temporary = TemporaryDirectory(prefix="kbo-published-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

        self.settings = replace(
            load_settings(PROJECT_ROOT, environ={}),
            raw_dir=self.root / "원본",
        )

        self.target = published.Target(
            target_id="synthetic-games",
            source="github",
            repository="example/synthetic",
            revision="a" * 40,
            filename="games.csv",
            kind="schedule_results",
            season=2023,
            competition="regular",
            start_date="2023-04-01",
            end_date="2023-10-31",
            publisher="synthetic-test",
            upstream="synthetic-test",
            evidence_url="https://example.com/synthetic",
            approval_ref="synthetic-test-only",
            permitted_uses=["download", "store"],
            required_fields=["game_id", "state", "score"],
            max_bytes=4096,
            game_id_field="game_id",
            expected_game_ids=["G1"],
            coverage_evidence="synthetic-test-only",
            game_status_field="state",
            status_map={"final": "completed"},
        )
        self.manifest = self.root / "manifest.json"

        sleeping = patch.object(published.time, "sleep")
        sleeping.start()
        self.addCleanup(sleeping.stop)

    def _manifest(self, *targets: published.Target) -> Path:
        """검사용 합성 명세를 생성한다."""
        self.manifest.write_text(
            json.dumps({
                "schema_version": 1,
                "targets": [
                    published.asdict(target)
                    for target in targets
                ],
            }),
            encoding="utf-8",
        )
        return self.manifest

    def _collect(self, mode: str = "collect") -> dict[str, object]:
        """지정한 모드로 합성 수집을 실행한다."""
        return published.collect_files(
            self.settings,
            manifest_path=self.manifest,
            mode=mode,
        )

    @staticmethod
    def _success(body: bytes) -> Attempt:
        """완전히 받은 합성 파일 응답을 생성한다."""
        return Attempt(body, 200, None, False, True)

    def test_preserves_original_and_deduplicates_content(self) -> None:
        """동일 내용은 재사용하고 실행 이력은 분리한다."""
        self._manifest(self.target)

        with patch.object(
            published,
            "request_once",
            return_value=self._success(FIRST_BODY),
        ):
            first = self._collect()
            second = self._collect()

        self.assertTrue(first["file_collection_complete"])
        self.assertTrue(second["file_collection_complete"])
        self.assertNotEqual(first["run_id"], second["run_id"])

        first_record = first["results"][0]
        second_record = second["results"][0]

        self.assertTrue(first_record["object_created"])
        self.assertFalse(second_record["object_created"])
        self.assertEqual(
            second_record["version_relationship"],
            "same_content",
        )
        self.assertEqual(first_record["sha256"], second_record["sha256"])

        objects = self.settings.raw_dir / "published" / "objects"
        self.assertEqual(len(list(objects.glob("*.raw"))), 1)
        self.assertEqual(
            (objects / f"{sha256(FIRST_BODY)}.raw").read_bytes(),
            FIRST_BODY,
        )

    def test_correction_preserves_both_versions(self) -> None:
        """내용이 다른 판본은 새 원본으로 보존한다."""
        self._manifest(self.target)

        with patch.object(
            published,
            "request_once",
            return_value=self._success(FIRST_BODY),
        ):
            first = self._collect()

        corrected = replace(self.target, revision="b" * 40)
        self._manifest(corrected)

        with patch.object(
            published,
            "request_once",
            return_value=self._success(CORRECTED_BODY),
        ):
            second = self._collect("update")

        record = second["results"][0]
        self.assertEqual(
            record["version_relationship"],
            "correction_candidate",
        )
        self.assertEqual(
            record["previous_version"]["run_id"],
            first["run_id"],
        )

        objects = self.settings.raw_dir / "published" / "objects"
        self.assertEqual(len(list(objects.glob("*.raw"))), 2)
        self.assertEqual(
            (objects / f"{sha256(FIRST_BODY)}.raw").read_bytes(),
            FIRST_BODY,
        )
        self.assertEqual(
            (objects / f"{sha256(CORRECTED_BODY)}.raw").read_bytes(),
            CORRECTED_BODY,
        )

    def test_resume_skips_verified_original(self) -> None:
        """중단된 실행에서 성공 파일을 재사용하고 남은 파일만 요청한다."""
        other = replace(
            self.target,
            target_id="synthetic-other",
            filename="other.csv",
        )
        self._manifest(self.target, other)

        with patch.object(
            published,
            "request_once",
            side_effect=[
                self._success(FIRST_BODY),
                KeyboardInterrupt(),
            ],
        ):
            with self.assertRaises(KeyboardInterrupt):
                self._collect()

        runs = self.settings.raw_dir / "published" / "runs"
        interrupted = next(runs.iterdir())

        before = published.report_files(
            self.settings,
            interrupted.name,
        )
        self.assertEqual(before["status_counts"]["included"], 1)
        self.assertEqual(before["status_counts"]["interrupted"], 1)

        with patch.object(
            published,
            "request_once",
            return_value=self._success(FIRST_BODY),
        ) as fetch:
            resumed = published.collect_files(
                self.settings,
                mode="resume",
                run_id=interrupted.name,
            )

        self.assertEqual(fetch.call_count, 1)
        self.assertTrue(resumed["file_collection_complete"])
        self.assertEqual(
            resumed["results"][0]["reused_from_run"],
            interrupted.name,
        )

    def test_unapproved_target_never_requests_data(self) -> None:
        """다운로드·저장 미승인 대상은 통신 없이 보류한다."""
        self._manifest(replace(
            self.target,
            approval_ref=None,
            permitted_uses=[],
        ))

        with patch.object(published, "request_once") as fetch:
            result = self._collect()

        fetch.assert_not_called()
        self.assertFalse(result["file_collection_complete"])
        self.assertEqual(result["status_counts"]["blocked"], 1)

    def test_invalid_and_partial_responses_are_not_originals(self) -> None:
        """빈 응답·오류 응답·부분 응답·해시 불일치를 성공으로 기록하지 않는다."""
        responses = (
            self._success(b""),
            self._success(b"<html>error</html>"),
            Attempt(
                FIRST_BODY,
                200,
                "incomplete_response",
                False,
                False,
            ),
        )

        for response in responses:
            with self.subTest(response=response.failure):
                self._manifest(self.target)
                with patch.object(
                    published,
                    "request_once",
                    return_value=response,
                ):
                    result = self._collect()

                self.assertEqual(result["status_counts"]["failed"], 1)
                self.assertFalse(result["file_collection_complete"])

        wrong_hash = replace(
            self.target,
            expected_sha256="0" * 64,
        )
        self._manifest(wrong_hash)

        with patch.object(
            published,
            "request_once",
            return_value=self._success(FIRST_BODY),
        ):
            result = self._collect()

        self.assertEqual(
            result["results"][0]["failure"],
            "artifact_hash_mismatch",
        )

        objects = self.settings.raw_dir / "published" / "objects"
        self.assertEqual(list(objects.glob("*.raw")), [])

    def test_bounded_retry(self) -> None:
        """일시적인 통신 실패만 설정 횟수 이내에서 재시도한다."""
        self._manifest(replace(self.target, retries=1))

        with patch.object(
            published,
            "request_once",
            side_effect=[
                Attempt(b"", None, "timeout", True, False),
                self._success(FIRST_BODY),
            ],
        ) as fetch:
            result = self._collect()

        self.assertEqual(fetch.call_count, 2)
        self.assertTrue(result["file_collection_complete"])

    def test_game_coverage_and_season_completion_are_distinct(self) -> None:
        """원본 수집·경기 누락·시즌 완료를 서로 구분한다."""
        target = replace(
            self.target,
            expected_game_ids=["G1", "G2"],
        )
        self._manifest(target)

        with patch.object(
            published,
            "request_once",
            return_value=self._success(FIRST_BODY),
        ):
            result = self._collect()

        self.assertTrue(result["file_collection_complete"])
        inspection = result["results"][0]["inspection"]

        self.assertEqual(inspection["missing_game_ids"], ["G2"])
        self.assertEqual(inspection["coverage_status"], "incomplete")
        self.assertEqual(inspection["game_state_counts"], {"completed": 1})
        self.assertIsNone(inspection["season_complete"])

    def test_normal_no_data_requires_evidence(self) -> None:
        """근거 없는 빈 레코드와 정상 무자료를 구분한다."""
        empty_body = b"game_id,state,score\n"

        with self.assertRaises(published.ArtifactError):
            published.inspect_body(self.target, empty_body)

        target = replace(
            self.target,
            expected_game_ids=[],
            empty_evidence="synthetic-no-data-evidence",
        )
        self._manifest(target)

        with patch.object(
            published,
            "request_once",
            return_value=self._success(empty_body),
        ):
            result = self._collect()

        self.assertEqual(result["status_counts"]["no_data"], 1)

    def test_report_detects_damaged_object(self) -> None:
        """저장 이후 원본 손상이 발견되면 정상 확보로 취급하지 않는다."""
        self._manifest(self.target)

        with patch.object(
            published,
            "request_once",
            return_value=self._success(FIRST_BODY),
        ):
            result = self._collect()

        objects = self.settings.raw_dir / "published" / "objects"
        path = objects / f"{sha256(FIRST_BODY)}.raw"
        path.write_bytes(b"damaged-synthetic-object")

        report = published.report_files(
            self.settings,
            str(result["run_id"]),
        )

        self.assertFalse(report["file_collection_complete"])
        self.assertEqual(
            report["status_counts"]["damaged_or_missing"],
            1,
        )
        self.assertEqual(path.read_bytes(), b"damaged-synthetic-object")


if __name__ == "__main__":
    unittest.main()