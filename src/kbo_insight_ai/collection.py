"""명시적으로 요청한 이닝 원본과 비밀값 없는 수집 이력을 보존한다."""

from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import Message
import hashlib
import http.client
import json
from pathlib import Path
import re
import time
from typing import BinaryIO
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from uuid import uuid4

from kbo_insight_ai.settings import Settings


IMPLEMENTATION_VERSION = "ct2-01-p3"
DECLARED_BASE_COMMIT = "9660b1d315ddbb6672e7855ad7d5a47d9c31742a"
REFERENCE_CODE_REVISION = "95dced9c843eebe76d09da1712426b96eaef936a"
GAME_ID_PATTERN = re.compile(r"[0-9]{8}[A-Z]{4}[0-9]{5}")
RETRYABLE_HTTP = frozenset({500, 502, 503, 504})
HF_REPOSITORY = "slothman3878/kbo_playbyplay"
HF_REVISION = "6afc8af044e3bba5f326b688e8cb41d7ff7065ec"
HF_FILENAME = "v0/kbo_pbp_2023.parquet"
HF_SAMPLE_BYTES = 17327447
HF_SAMPLE_SHA256 = "818f6016655b02fe48b8118281d1b04bfe3548d376fdc70131a41ea539341edb"



@dataclass(frozen=True)
class Attempt:
    """통신 결과를 담되 예외 원문이나 인증 정보를 보관하지 않는다."""

    body: bytes
    http_status: int | None
    failure: str | None
    retryable: bool
    body_complete: bool
    redirects: int = 0
    http_requests: int = 1


class NoRedirect(HTTPRedirectHandler):
    """미확인 주소로 추가 요청이 발생하지 않도록 자동 전송을 막는다."""

    def redirect_request(
        self,
        req: Request,
        fp: BinaryIO,
        code: int,
        msg: str,
        headers: Message,
        newurl: str,
    ) -> Request | None:
        """전송 주소에 접근하지 않고 HTTP 오류로 돌려보낸다."""
        return None


class RedirectRejected(ValueError):
    """허용하지 않은 파일 전송이나 요청 예산 초과를 나타낸다."""


def allowed_hf_url(url: str) -> bool:
    """HF 공개 저장소와 공식 저장 호스트의 HTTPS 주소만 허용한다."""
    try:
        parts = urlsplit(url)
        host = parts.hostname or ""
        return (
            parts.scheme == "https"
            and parts.port in (None, 443)
            and parts.username is None
            and parts.password is None
            and not parts.fragment
            and (
                host == "huggingface.co"
                or host.endswith(".huggingface.co")
                or host.endswith(".hf.co")
            )
        )
    except ValueError:
        return False


class HFRedirect(HTTPRedirectHandler):
    """HF 파일 배포의 공식 호스트 간 전송만 제한 횟수 안에서 따른다."""

    def __init__(self, maximum: int) -> None:
        """한 시도의 전송 횟수를 별도로 센다."""
        super().__init__()
        self.maximum = maximum
        self.followed = 0

    def redirect_request(
        self,
        req: Request,
        fp: BinaryIO,
        code: int,
        msg: str,
        headers: Message,
        newurl: str,
    ) -> Request | None:
        """서명 주소를 기록하지 않고 호스트와 횟수만 확인한다."""
        if not allowed_hf_url(newurl):
            raise RedirectRejected("redirect_host_rejected")
        if self.followed >= self.maximum:
            raise RedirectRejected("redirect_limit")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None:
            self.followed += 1
        return redirected


    def http_error_302(
        self,
        req: Request,
        fp: BinaryIO,
        code: int,
        msg: str,
        headers: Message,
    ) -> BinaryIO:
        """전송 응답 본문은 읽지 않고 닫아 별도 대용량 읽기를 막는다."""
        try:
            location = headers.get("Location")
            if not location:
                raise RedirectRejected("redirect_missing_location")
            redirected = self.redirect_request(
                req, fp, code, msg, headers, urljoin(req.full_url, location)
            )
        finally:
            fp.close()
        if redirected is None:
            raise RedirectRejected("redirect_rejected")
        return self.parent.open(redirected, timeout=req.timeout)

    http_error_301 = http_error_302
    http_error_303 = http_error_302
    http_error_307 = http_error_302
    http_error_308 = http_error_302



def utc_now() -> str:
    """시간대가 명시된 UTC 관측 시각을 반환한다."""
    return datetime.now(timezone.utc).isoformat()


def sha256(data: bytes) -> str:
    """원본이나 코드의 내용을 식별하는 SHA-256을 반환한다."""
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: object) -> None:
    """기존 파일을 덮어쓰지 않고 JSON을 배타 생성한다."""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def _reject_constant(value: str) -> None:
    """JSON 규격 밖의 NaN과 무한대 상수를 거절한다."""
    raise ValueError("지원하지 않는 JSON 상수입니다.")


def validate_relay(body: bytes) -> str | None:
    """응답 최소 구조만 검사하며 사건 의미나 전체 포함은 보증하지 않는다."""
    if not body:
        return "empty_body"
    try:
        payload = json.loads(body, parse_constant=_reject_constant)
    except (ValueError, UnicodeError, RecursionError):
        return "invalid_json"

    if not isinstance(payload, dict):
        return "invalid_envelope"
    if payload.get("error"):
        return "api_error"

    result = payload.get("result")
    if not isinstance(result, dict):
        return "missing_result"
    relay = result.get("textRelayData")
    if not isinstance(relay, dict):
        return "missing_relay"
    blocks = relay.get("textRelays")
    if not isinstance(blocks, list) or not blocks:
        return "empty_or_invalid_blocks"

    event_count = 0
    for block in blocks:
        if not isinstance(block, dict):
            return "invalid_block"
        events = block.get("textOptions")
        if not isinstance(events, list):
            return "invalid_events"
        for event in events:
            if not isinstance(event, dict):
                return "invalid_event"
            if (
                type(event.get("type")) is not int
                or type(event.get("seqno")) is not int
            ):
                return "invalid_event_identity"
            event_count += 1

    return None if event_count else "empty_events"


def validate_hf_file(body: bytes) -> str | None:
    """게시된 크기·내용 해시를 대조한다. Parquet 필드 파싱은 하지 않는다."""
    if len(body) != HF_SAMPLE_BYTES:
        return "artifact_size_mismatch"
    if sha256(body) != HF_SAMPLE_SHA256:
        return "artifact_hash_mismatch"
    if not body.startswith(b"PAR1") or not body.endswith(b"PAR1"):
        return "invalid_parquet_envelope"
    return None


def request_once(
    url: str,
    settings: Settings,
    *,
    hf: bool = False,
) -> Attempt:
    """한 번의 논리 시도에서 응답 크기와 허용된 전송 횟수를 제한한다."""
    request = Request(
        url,
        headers={
            "Accept": "*/*" if hf else "application/json",
            "Accept-Encoding": "identity",
            "User-Agent": "KBO-Insight-AI/0.1 source-pilot",
        },
    )
    body = bytearray()
    status: int | None = None
    limit = settings.hf_collection_max_bytes if hf else settings.collection_max_bytes
    redirect_handler = HFRedirect(settings.hf_collection_max_redirects) if hf else None

    def outcome(
        content: bytes,
        code: int | None,
        failure: str | None,
        retryable: bool,
        complete: bool,
    ) -> Attempt:
        """서명 주소 없이 실제로 따라간 전송 횟수만 결과에 담는다."""
        redirects = redirect_handler.followed if redirect_handler is not None else 0
        return Attempt(content, code, failure, retryable, complete, redirects, 1 + redirects)

    try:
        opener = build_opener(redirect_handler if redirect_handler is not None else NoRedirect())
        with opener.open(
            request,
            timeout=settings.collection_timeout_seconds,
        ) as response:
            status = response.getcode()
            if status != 200:
                return outcome(b"", status, "unexpected_http_status", False, False)
            if response.headers.get("Content-Encoding", "identity") != "identity":
                return outcome(b"", status, "unsupported_encoding", False, False)

            declared_length = response.headers.get("Content-Length")
            expected: int | None = None
            if declared_length is not None:
                if not declared_length.isascii() or not declared_length.isdigit():
                    return outcome(b"", status, "invalid_content_length", False, False)
                # 비정상적으로 긴 정수 문자열을 변환하기 전에 거절한다.
                if len(declared_length) > 10:
                    return outcome(b"", status, "body_too_large", False, False)
                expected = int(declared_length)
                if expected > limit:
                    return outcome(b"", status, "body_too_large", False, False)

            while len(body) <= limit:
                remaining = limit + 1 - len(body)
                chunk = response.read(min(65536, remaining))
                if not chunk:
                    break
                body.extend(chunk)

            if len(body) > limit:
                # 초과 여부 확인용 한 바이트는 버리고 부분 응답임을 명시한다.
                return outcome(bytes(body[:limit]), status, "body_too_large", False, False)
            if expected is not None and len(body) != expected:
                return outcome(bytes(body), status, "incomplete_response", True, False)

    except RedirectRejected as exc:
        return outcome(b"", None, str(exc), False, False)
    except HTTPError as exc:
        # 오류 본문·인증 헤더·주소·예외 원문을 이력에 복사하지 않는다.
        code = exc.code
        exc.close()
        return outcome(b"", code, f"http_{code}", code in RETRYABLE_HTTP, False)
    except http.client.IncompleteRead as exc:
        body.extend(exc.partial[:max(0, limit - len(body))])
        return outcome(bytes(body), status, "incomplete_response", True, False)
    except TimeoutError:
        return outcome(bytes(body), status, "timeout", True, False)
    except URLError as exc:
        failure = "timeout" if isinstance(exc.reason, TimeoutError) else "network"
        return outcome(bytes(body), status, failure, True, False)
    except (OSError, http.client.HTTPException):
        return outcome(bytes(body), status, "network", True, False)

    return outcome(bytes(body), status, None, False, True)


def collect_sample(
    settings: Settings,
    game_id: str,
    inning: int,
) -> tuple[Path, bool]:
    """경기 하나의 이닝 하나를 보존하고 경로와 구조 검사 결과를 반환한다."""
    if not GAME_ID_PATTERN.fullmatch(game_id):
        raise ValueError("game_id: 공개 원본에서 확인한 경기 ID가 필요합니다.")
    if type(inning) is not int or not 1 <= inning <= 12:
        raise ValueError("inning: 1 이상 12 이하의 정수가 필요합니다.")

    url = (
        f"{settings.collection_base_url}/schedule/games/"
        f"{game_id}/relay?inning={inning}"
    )
    return _collect(settings, {"game_id": game_id, "inning": inning}, url)


def collect_hf_sample(settings: Settings) -> tuple[Path, bool]:
    """고정 revision의 배포 파일 하나만 동일한 원본 보존 경로로 확보한다."""
    url = (
        f"{settings.hf_collection_base_url}/datasets/{HF_REPOSITORY}/"
        f"resolve/{HF_REVISION}/{HF_FILENAME}"
    )
    target: dict[str, str | int] = {
        "repository": HF_REPOSITORY,
        "revision": HF_REVISION,
        "filename": HF_FILENAME,
        "expected_bytes": HF_SAMPLE_BYTES,
        "expected_sha256": HF_SAMPLE_SHA256,
    }
    return _collect(settings, target, url, hf=True)


def _collect(
    settings: Settings,
    target: dict[str, str | int],
    url: str,
    *,
    hf: bool = False,
) -> tuple[Path, bool]:
    """두 출처에 공통인 배타 저장·해시 대조·실패 이력을 처리한다."""
    retries = 0 if hf else settings.collection_retries
    maximum_bytes = settings.hf_collection_max_bytes if hf else settings.collection_max_bytes
    maximum_redirects = settings.hf_collection_max_redirects if hf else 0
    accepted_status = "artifact_verified" if hf else "structure_validated"
    run_id = uuid4().hex
    run_dir = settings.raw_dir / "collection" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    module_dir = Path(__file__).resolve().parent
    code_hashes = {
        name: sha256((module_dir / name).read_bytes())
        for name in ("collection.py", "settings.py", "__main__.py")
    }
    write_json(
        run_dir / "request.json",
        {
            "run_id": run_id,
            "created_at": utc_now(),
            "source": "huggingface_parquet" if hf else "naver_sports_relay",
            "source_version": HF_REVISION if hf else None,
            "reference_code_revision": REFERENCE_CODE_REVISION,
            "implementation_version": IMPLEMENTATION_VERSION,
            "declared_base_commit": DECLARED_BASE_COMMIT,
            "code_sha256": code_hashes,
            "request": {
                "method": "GET",
                "base_url": settings.hf_collection_base_url if hf else settings.collection_base_url,
                **target,
            },
            "limits": {
                "timeout_seconds": settings.collection_timeout_seconds,
                "retries": retries,
                "max_bytes": maximum_bytes,
                "max_redirects": maximum_redirects,
                "max_http_requests": (retries + 1) * (maximum_redirects + 1),
            },
            "source_published_at": None,
        },
    )

    for number in range(1, retries + 2):
        started_at = utc_now()
        attempt = request_once(url, settings, hf=True) if hf else request_once(url, settings)
        observed_at = utc_now()
        failure = attempt.failure
        if failure is None and not attempt.body_complete:
            failure = "incomplete_response"
        if failure is None:
            failure = validate_hf_file(attempt.body) if hf else validate_relay(attempt.body)

        body_path: Path | None = None
        digest: str | None = None
        if attempt.body:
            suffix = "parquet" if hf and failure is None else "bin"
            body_path = run_dir / f"attempt-{number:02d}.{suffix}"
            with body_path.open("xb") as stream:
                stream.write(attempt.body)
            digest = sha256(attempt.body)
            if sha256(body_path.read_bytes()) != digest:
                raise OSError("저장 원본의 해시가 일치하지 않습니다.")

        accepted = failure is None
        record = {
            "run_id": run_id,
            "attempt": number,
            "request_started_at": started_at,
            "response_observed_at": observed_at,
            "stored_at": utc_now(),
            "http_status": attempt.http_status,
            "http_requests": attempt.http_requests,
            "redirects_followed": attempt.redirects,
            "bytes": len(attempt.body),
            "body_complete": attempt.body_complete,
            "body_path": body_path.name if body_path else None,
            "sha256": digest,
            "status": accepted_status if accepted else "failed",
            "failure": failure,
            "semantic_validation": "not_performed",
            "source_published_at": None,
        }
        write_json(run_dir / f"attempt-{number:02d}.json", record)

        if (
            not accepted
            and attempt.retryable
            and number <= retries
        ):
            time.sleep(number)
            continue

        write_json(run_dir / "result.json", record)
        return run_dir, accepted

    raise RuntimeError("수집의 종료 상태에 도달하지 못했습니다.")
