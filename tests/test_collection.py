"""합성 응답으로 수집·실패·원본 보존 경계를 검사한다.

이 파일의 JSON과 사건은 테스트용이며 실제 경기 확보 근거가 아니다.
"""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import http.client
from io import BytesIO, StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

from kbo_insight_ai.ingestion import pilot as collection
from kbo_insight_ai.__main__ import main
from kbo_insight_ai.settings import load_settings


GAME_ID = "20230401HHWO02023"
SOURCE_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_BODY = json.dumps(
    {
        "result": {
            "textRelayData": {
                "textRelays": [
                    {"textOptions": [{"type": 1, "seqno": 1}]}
                ]
            }
        }
    }
).encode("utf-8")


class CollectionTests(unittest.TestCase):
    """실제 외부 통신 없이 원본·이력·명령의 실패 처리를 확인한다."""

    def setUp(self) -> None:
        """각 검사를 별도 임시 원본 경로에 격리한다."""
        temporary = TemporaryDirectory(prefix="kbo-ct2-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.settings = replace(
            load_settings(SOURCE_ROOT, environ={}),
            raw_dir=self.root / "한글 원본",
            collection_retries=1,
        )

    def _record(self, run_dir: Path) -> dict[str, object]:
        """완료 이력을 다시 읽는다."""
        return json.loads((run_dir / "result.json").read_text(encoding="utf-8"))

    def _response(
        self, chunks: list[object], headers: dict[str, str] | None = None
    ) -> MagicMock:
        """한 번의 HTTP 응답을 대신할 합성 스트림을 만든다."""
        response = MagicMock()
        response.__enter__.return_value = response
        response.getcode.return_value = 200
        response.headers = {} if headers is None else headers
        response.read.side_effect = chunks
        return response

    def test_preserves_bytes_hash_and_distinct_runs(self) -> None:
        """원본·해시·코드 버전을 연결하고 재실행 때 덮어쓰지 않는다."""
        good = collection.Attempt(SYNTHETIC_BODY, 200, None, False, True)
        with patch.object(collection, "request_once", return_value=good):
            first, accepted = collection.collect_sample(self.settings, GAME_ID, 1)
            before = {p.name: p.read_bytes() for p in first.iterdir()}
            second, accepted_again = collection.collect_sample(
                self.settings, GAME_ID, 1
            )

        self.assertTrue(accepted and accepted_again)
        self.assertNotEqual(first, second)
        self.assertEqual(before, {p.name: p.read_bytes() for p in first.iterdir()})
        record = self._record(first)
        self.assertEqual((first / str(record["body_path"])).read_bytes(), SYNTHETIC_BODY)
        self.assertEqual(record["sha256"], collection.sha256(SYNTHETIC_BODY))
        self.assertTrue(record["body_complete"])
        self.assertEqual(record["semantic_validation"], "not_performed")
        self.assertIsNone(record["source_published_at"])
        request = json.loads((first / "request.json").read_text(encoding="utf-8"))
        self.assertEqual(request["run_id"], record["run_id"])
        self.assertEqual(
            set(request["code_sha256"]),
            {"ingestion/pilot.py", "settings.py", "__main__.py"},
        )
        for name, digest in request["code_sha256"].items():
            source = SOURCE_ROOT / "src" / "kbo_insight_ai" / name
            self.assertEqual(digest, collection.sha256(source.read_bytes()))

    def test_rejects_bad_responses_without_retry(self) -> None:
        """빈 값·손상·오류·필드 결손·타입 변경을 정상 원본으로 인정하지 않는다."""
        cases = [
            (b"", "empty_body"),
            (b"<html>error</html>", "invalid_json"),
            (b"[]", "invalid_envelope"),
            (b'{"error":{"code":1}}', "api_error"),
            (b"{}", "missing_result"),
            (b'{"result":{}}', "missing_relay"),
            (b'{"result":{"textRelayData":{"textRelays":[]}}}', "empty_or_invalid_blocks"),
            (SYNTHETIC_BODY.replace(b'"seqno": 1', b'"seqno": true'), "invalid_event_identity"),
            (b'{"result": NaN}', "invalid_json"),
        ]
        for body, failure in cases:
            with self.subTest(failure=failure):
                result = collection.Attempt(body, 200, None, False, True)
                with patch.object(collection, "request_once", return_value=result) as fetch:
                    run_dir, accepted = collection.collect_sample(
                        self.settings, GAME_ID, 1
                    )
                self.assertFalse(accepted)
                self.assertEqual(fetch.call_count, 1)
                self.assertEqual(self._record(run_dir)["failure"], failure)

    def test_timeout_retry_is_bounded_and_attempts_survive(self) -> None:
        """시간 초과 후 한 번만 재시도하고 각 실패를 남긴다."""
        failure = collection.Attempt(b"", None, "timeout", True, False)
        with (
            patch.object(collection, "request_once", return_value=failure) as fetch,
            patch.object(collection.time, "sleep") as pause,
        ):
            run_dir, accepted = collection.collect_sample(self.settings, GAME_ID, 1)
        self.assertFalse(accepted)
        self.assertEqual(fetch.call_count, 2)
        pause.assert_called_once_with(1)
        self.assertTrue((run_dir / "attempt-01.json").exists())
        self.assertTrue((run_dir / "attempt-02.json").exists())
        self.assertEqual(self._record(run_dir)["attempt"], 2)

    def test_retry_can_recover_without_erasing_failed_prefix(self) -> None:
        """부분 실패 뒤 성공해도 첫 부분 응답과 실패 이력을 보존한다."""
        results = [
            collection.Attempt(b"partial", 200, "incomplete_response", True, False),
            collection.Attempt(SYNTHETIC_BODY, 200, None, False, True),
        ]
        with (
            patch.object(collection, "request_once", side_effect=results),
            patch.object(collection.time, "sleep"),
        ):
            run_dir, accepted = collection.collect_sample(self.settings, GAME_ID, 1)
        self.assertTrue(accepted)
        self.assertEqual((run_dir / "attempt-01.bin").read_bytes(), b"partial")
        first = json.loads((run_dir / "attempt-01.json").read_text(encoding="utf-8"))
        self.assertEqual(first["status"], "failed")
        self.assertFalse(first["body_complete"])

    def test_invalid_target_creates_nothing(self) -> None:
        """주소 주입·허용 밖 이닝은 통신이나 원본 경로 생성 전에 거절한다."""
        for game_id, inning in [(GAME_ID + "?token=x", 1), (GAME_ID, 0), (GAME_ID, True)]:
            with self.subTest(game_id=game_id, inning=inning):
                with patch.object(collection, "request_once") as fetch:
                    with self.assertRaises(ValueError):
                        collection.collect_sample(self.settings, game_id, inning)
                    fetch.assert_not_called()
        self.assertFalse(self.settings.raw_dir.exists())

    def test_existing_json_is_not_overwritten(self) -> None:
        """배타 생성으로 기존 이력을 보호한다."""
        path = self.root / "record.json"
        collection.write_json(path, {"first": True})
        before = path.read_bytes()
        with self.assertRaises(FileExistsError):
            collection.write_json(path, {"first": False})
        self.assertEqual(path.read_bytes(), before)

    def test_network_timeout_does_not_disclose_exception(self) -> None:
        """민감한 예외 원문을 실패 코드로 대체한다."""
        marker = "비밀값_테스트_대역"
        with patch.object(collection, "build_opener") as builder:
            builder.return_value.open.side_effect = URLError(TimeoutError(marker))
            attempt = collection.request_once("https://example.invalid", self.settings)
        self.assertEqual(attempt.failure, "timeout")
        self.assertNotIn(marker, repr(attempt))

    def test_http_error_retry_and_redirect_policy(self) -> None:
        """차단·전송·요청 제한은 재시도하지 않고 일시 서버 오류만 구분한다."""
        for code, retryable in [(302, False), (403, False), (429, False), (503, True)]:
            with self.subTest(code=code):
                error = HTTPError(
                    "https://example.invalid/?token=대역", code, "대역", {}, BytesIO()
                )
                with patch.object(collection, "build_opener") as builder:
                    builder.return_value.open.side_effect = error
                    attempt = collection.request_once(
                        "https://example.invalid", self.settings
                    )
                self.assertEqual(attempt.failure, f"http_{code}")
                self.assertEqual(attempt.retryable, retryable)
        self.assertIsNone(
            collection.NoRedirect().redirect_request(
                None, None, 302, "", None, "https://example.invalid"
            )
        )

    def test_reads_complete_response_and_passes_timeout(self) -> None:
        """응답 bytes를 바꾸지 않으며 설정한 대기 제한을 사용한다."""
        response = self._response(
            [SYNTHETIC_BODY, b""],
            {"Content-Length": str(len(SYNTHETIC_BODY))},
        )
        with patch.object(collection, "build_opener") as builder:
            builder.return_value.open.return_value = response
            attempt = collection.request_once("https://example.invalid", self.settings)
            self.assertEqual(
                builder.return_value.open.call_args.kwargs["timeout"],
                self.settings.collection_timeout_seconds,
            )
        self.assertIsNone(attempt.failure)
        self.assertTrue(attempt.body_complete)
        self.assertEqual(attempt.body, SYNTHETIC_BODY)

    def test_enforces_byte_limit_and_detects_truncation(self) -> None:
        """길이 없는 초과·미완성 응답도 성공으로 오인하지 않는다."""
        settings = replace(self.settings, collection_max_bytes=8)
        cases = [
            (self._response([b"123456789"]), "body_too_large", b"12345678"),
            (self._response([b"123", b""], {"Content-Length": "8"}), "incomplete_response", b"123"),
            (self._response([], {"Content-Length": "9"}), "body_too_large", b""),
            (self._response([], {"Content-Length": "wrong"}), "invalid_content_length", b""),
            (self._response([], {"Content-Encoding": "gzip"}), "unsupported_encoding", b""),
            (self._response([http.client.IncompleteRead(b"123", 8)]), "incomplete_response", b"123"),
        ]
        for response, failure, body in cases:
            with self.subTest(failure=failure, body=body):
                with patch.object(collection, "build_opener") as builder:
                    builder.return_value.open.return_value = response
                    attempt = collection.request_once("https://example.invalid", settings)
                self.assertEqual(attempt.failure, failure)
                self.assertEqual(attempt.body, body)
                self.assertFalse(attempt.body_complete)

    def test_partial_valid_json_is_still_failure(self) -> None:
        """부분 응답이 JSON 구조와 일치해도 성공으로 인정하지 않는다."""
        partial = collection.Attempt(SYNTHETIC_BODY, 200, None, False, False)
        with patch.object(collection, "request_once", return_value=partial):
            run_dir, accepted = collection.collect_sample(self.settings, GAME_ID, 1)
        self.assertFalse(accepted)
        self.assertEqual(self._record(run_dir)["failure"], "incomplete_response")

    def test_storage_failure_has_no_success_record(self) -> None:
        """이력을 쓰지 못하면 성공 완료 파일을 남기지 않는다."""
        good = collection.Attempt(SYNTHETIC_BODY, 200, None, False, True)
        original = collection.write_json

        def fail_attempt(path: Path, value: object) -> None:
            """시도 이력 저장만 권한 오류로 모의한다."""
            if path.name.startswith("attempt-"):
                raise PermissionError("저장 오류 대역")
            original(path, value)

        with (
            patch.object(collection, "request_once", return_value=good),
            patch.object(collection, "write_json", side_effect=fail_attempt),
        ):
            with self.assertRaises(PermissionError):
                collection.collect_sample(self.settings, GAME_ID, 1)
        self.assertFalse(list(self.settings.raw_dir.rglob("result.json")))

    def test_cli_failure_and_storage_exit_codes(self) -> None:
        """명령이 통신 실패와 저장 실패를 서로 다른 비정상 코드로 반환한다."""
        with (
            patch("kbo_insight_ai.__main__.load_settings", return_value=self.settings),
            patch.object(collection, "collect_sample") as collect,
            redirect_stdout(StringIO()),
            redirect_stderr(StringIO()),
        ):
            collect.return_value = (self.root, False)
            self.assertEqual(main(["collect-sample", "--game-id", GAME_ID, "--inning", "1"]), 3)
            collect.side_effect = PermissionError("비밀값_예외_대역")
            self.assertEqual(main(["collect-sample", "--game-id", GAME_ID, "--inning", "1"]), 4)


    def test_hf_file_checks_declared_size_hash_and_envelope(self) -> None:
        """합성 bytes로 파일 검사를 확인하며 실제 Parquet 파싱으로 간주하지 않는다."""
        body = b"PAR1synthetic-onlyPAR1"
        with (
            patch.object(collection, "HF_SAMPLE_BYTES", len(body)),
            patch.object(collection, "HF_SAMPLE_SHA256", collection.sha256(body)),
        ):
            self.assertIsNone(collection.validate_hf_file(body))
            self.assertEqual(collection.validate_hf_file(body[:-1]), "artifact_size_mismatch")
            altered = body[:-5] + b"X" + body[-4:]
            self.assertEqual(collection.validate_hf_file(altered), "artifact_hash_mismatch")
        bad = b"not-a-parquet-envelope"
        with (
            patch.object(collection, "HF_SAMPLE_BYTES", len(bad)),
            patch.object(collection, "HF_SAMPLE_SHA256", collection.sha256(bad)),
        ):
            self.assertEqual(collection.validate_hf_file(bad), "invalid_parquet_envelope")

    def test_hf_reuses_preservation_and_records_pinned_asset(self) -> None:
        """공통 저장 경로가 고정 revision·예상 해시·별도 제한을 연결한다."""
        body = b"PAR1synthetic-onlyPAR1"
        good = collection.Attempt(body, 200, None, False, True, 2, 3)
        with (
            patch.object(collection, "HF_SAMPLE_BYTES", len(body)),
            patch.object(collection, "HF_SAMPLE_SHA256", collection.sha256(body)),
            patch.object(collection, "request_once", return_value=good) as fetch,
        ):
            run_dir, accepted = collection.collect_hf_sample(self.settings)
        self.assertTrue(accepted)
        self.assertIn(collection.HF_REVISION, fetch.call_args.args[0])
        self.assertTrue(fetch.call_args.kwargs["hf"])
        record = self._record(run_dir)
        self.assertEqual(record["status"], "artifact_verified")
        self.assertEqual(record["http_requests"], 3)
        self.assertEqual(record["redirects_followed"], 2)
        self.assertEqual((run_dir / str(record["body_path"])).suffix, ".parquet")
        self.assertEqual((run_dir / str(record["body_path"])).read_bytes(), body)
        request = json.loads((run_dir / "request.json").read_text(encoding="utf-8"))
        self.assertEqual(request["source_version"], collection.HF_REVISION)
        self.assertEqual(request["limits"]["max_http_requests"], 4)
        self.assertEqual(request["limits"]["max_bytes"], self.settings.hf_collection_max_bytes)
        self.assertEqual(request["limits"]["retries"], 0)
        self.assertEqual(request["request"]["expected_sha256"], collection.sha256(body))
        self.assertEqual(record["semantic_validation"], "not_performed")

    def test_hf_disables_retries_even_if_naver_retry_is_enabled(self) -> None:
        """별도 파일 전송 예산은 Naver 재시도 설정으로 늘어나지 않는다."""
        settings = replace(self.settings, collection_retries=2)
        failure = collection.Attempt(b"", None, "timeout", True, False)
        with patch.object(collection, "request_once", return_value=failure) as fetch:
            run_dir, accepted = collection.collect_hf_sample(settings)
        self.assertFalse(accepted)
        fetch.assert_called_once()
        self.assertEqual(self._record(run_dir)["failure"], "timeout")

    def test_hf_redirect_allowlist_and_budget(self) -> None:
        """공식 HTTPS 호스트 전송만 허용하고 무한 전송·주소 주입을 막는다."""
        from email.message import Message
        from urllib.request import Request

        allowed = [
            "https://huggingface.co/file",
            "https://cdn-lfs.huggingface.co/file",
            "https://us.aws.cdn.hf.co/file?Signature=synthetic-secret",
            "https://cas-bridge.xethub.hf.co/file",
        ]
        rejected = [
            "http://huggingface.co/file",
            "https://huggingface.co.evil.invalid/file",
            "https://evil.invalid/file",
            "https://user:password@huggingface.co/file",
            "https://huggingface.co:444/file",
            "https://huggingface.co/file#fragment",
            "https://huggingface.co:invalid/file",
        ]
        for url in allowed:
            self.assertTrue(collection.allowed_hf_url(url))
        for url in rejected:
            self.assertFalse(collection.allowed_hf_url(url))

        handler = collection.HFRedirect(1)
        request = Request("https://huggingface.co/file")
        headers = Message()
        self.assertIsNotNone(handler.redirect_request(
            request, BytesIO(), 302, "", headers, allowed[2]
        ))
        self.assertEqual(handler.followed, 1)
        with self.assertRaisesRegex(collection.RedirectRejected, "^redirect_limit$"):
            handler.redirect_request(request, BytesIO(), 302, "", headers, allowed[0])
        with self.assertRaisesRegex(collection.RedirectRejected, "^redirect_host_rejected$"):
            collection.HFRedirect(3).redirect_request(
                request, BytesIO(), 302, "", headers, rejected[2]
            )

    def test_hf_redirect_failure_is_sanitized_and_not_retried(self) -> None:
        """서명 주소 대신 고정 실패 코드만 통신 결과에 남긴다."""
        with patch.object(collection, "build_opener") as builder:
            builder.return_value.open.side_effect = collection.RedirectRejected("redirect_limit")
            attempt = collection.request_once(
                "https://huggingface.co/file", self.settings, hf=True
            )
        self.assertEqual(attempt.failure, "redirect_limit")
        self.assertFalse(attempt.retryable)

    def test_hf_http_reader_uses_its_own_size_limit(self) -> None:
        """Naver의 작은 상한을 바꾸지 않고 HF 파일 상한만 적용한다."""
        settings = replace(
            self.settings, collection_max_bytes=2, hf_collection_max_bytes=16
        )
        body = b"PAR1testPAR1"
        response = self._response([body, b""], {"Content-Length": str(len(body))})
        with patch.object(collection, "build_opener") as builder:
            builder.return_value.open.return_value = response
            attempt = collection.request_once(
                "https://huggingface.co/file", settings, hf=True
            )
        self.assertIsNone(attempt.failure)
        self.assertEqual(attempt.body, body)
        self.assertEqual(settings.collection_max_bytes, 2)

    def test_hf_cli_and_invalid_cross_source_arguments(self) -> None:
        """출처 선택은 올바른 수집 함수를 호출하며 잘못된 인자는 쓰기 전에 거절한다."""
        with (
            patch("kbo_insight_ai.__main__.load_settings", return_value=self.settings),
            patch.object(collection, "collect_hf_sample", return_value=(self.root, True)) as fetch,
            redirect_stdout(StringIO()) as output,
            redirect_stderr(StringIO()),
        ):
            self.assertEqual(main(["collect-sample", "--source", "hf"]), 0)
            fetch.assert_called_once_with(self.settings)
            self.assertEqual(json.loads(output.getvalue())["status"], "artifact_verified")
        for args in (
            ["collect-sample", "--source", "hf", "--game-id", GAME_ID],
            ["collect-sample", "--source", "naver"],
        ):
            with self.subTest(args=args):
                with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as error:
                    main(args)
                self.assertEqual(error.exception.code, 2)



    def test_naver_zero_retries_stops_after_first_failure(self) -> None:
        """실제 진단 설정처럼 재시도 0회이면 첫 실패 뒤 멈춘다."""
        settings = replace(self.settings, collection_retries=0)
        failure = collection.Attempt(b"", None, "timeout", True, False)
        with patch.object(collection, "request_once", return_value=failure) as fetch:
            run_dir, accepted = collection.collect_sample(settings, GAME_ID, 1)
        self.assertFalse(accepted)
        fetch.assert_called_once()
        self.assertEqual(self._record(run_dir)["attempt"], 1)
        self.assertFalse((run_dir / "attempt-02.json").exists())


    def test_hf_redirect_body_is_closed_without_reading(self) -> None:
        """전송 본문은 크기와 무관하게 읽지 않고 닫으며 요청 예산을 지킨다."""
        from email.message import Message
        from urllib.request import Request

        for maximum in (0, 1):
            with self.subTest(maximum=maximum):
                handler = collection.HFRedirect(maximum)
                parent = MagicMock()
                handler.add_parent(parent)
                request = Request("https://huggingface.co/file")
                request.timeout = 30
                body = MagicMock()
                body.read.side_effect = AssertionError("전송 본문을 읽으면 안 됩니다.")
                headers = Message()
                headers["Location"] = "https://us.aws.cdn.hf.co/file?Signature=synthetic"
                if maximum == 0:
                    with self.assertRaisesRegex(collection.RedirectRejected, "^redirect_limit$"):
                        handler.http_error_302(request, body, 302, "", headers)
                    parent.open.assert_not_called()
                else:
                    result = handler.http_error_302(request, body, 302, "", headers)
                    self.assertEqual(result, parent.open.return_value)
                    self.assertEqual(parent.open.call_args.kwargs["timeout"], 30)
                    self.assertEqual(handler.followed, 1)
                body.read.assert_not_called()
                body.close.assert_called_once()

    def test_hf_hash_failure_preserves_failed_bytes_only(self) -> None:
        """HTTP 성공이어도 게시 해시가 다르면 실패 bin으로만 남긴다."""
        body = b"PAR1synthetic-onlyPAR1"
        good_http = collection.Attempt(body, 200, None, False, True)
        with (
            patch.object(collection, "HF_SAMPLE_BYTES", len(body)),
            patch.object(collection, "HF_SAMPLE_SHA256", "0" * 64),
            patch.object(collection, "request_once", return_value=good_http) as fetch,
        ):
            run_dir, accepted = collection.collect_hf_sample(self.settings)
        self.assertFalse(accepted)
        fetch.assert_called_once()
        record = self._record(run_dir)
        self.assertEqual(record["failure"], "artifact_hash_mismatch")
        self.assertEqual((run_dir / str(record["body_path"])).suffix, ".bin")
        self.assertFalse(list(run_dir.glob("*.parquet")))

if __name__ == "__main__":
    unittest.main()