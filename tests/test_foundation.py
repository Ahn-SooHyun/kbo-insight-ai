"""최소 개발 기반의 설정·경로·읽기 전용 동작을 검증한다."""

from contextlib import chdir
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import kbo_insight_ai
from kbo_insight_ai.settings import SettingsError, load_settings


BASE_CONFIG: dict[str, object] = {
    "raw_dir": "data/raw",
    "interim_dir": "data/interim",
    "processed_dir": "data/processed",
    "models_dir": "models",
    "results_dir": "results",
    "cache_dir": ".cache",
    "seed": 42,
    "log_level": "INFO",
}

READONLY_BOOTSTRAP = r'''
import os
from pathlib import Path
import runpy
import sys

def guard(event: str, args: tuple[object, ...]) -> None:
    """감시 구간의 네트워크·쓰기·대표 모델 파일 열기를 차단한다."""
    if event.startswith("socket.") or event in {
        "subprocess.Popen", "os.system", "os.mkdir", "os.remove",
        "os.rename", "os.rmdir", "os.link", "os.symlink",
    }:
        raise RuntimeError("읽기 전용 경계를 벗어났습니다.")

    if event == "open":
        path, _, flags = args
        write_flags = (
            os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
        )
        if isinstance(flags, int) and flags & write_flags:
            raise RuntimeError("파일 쓰기를 시도했습니다.")
        if isinstance(path, (str, bytes, os.PathLike)):
            suffix = Path(os.fsdecode(path)).suffix.lower()
            if suffix in {
                ".pkl", ".pickle", ".joblib", ".onnx",
                ".pt", ".pth", ".safetensors",
            }:
                raise RuntimeError("모델 파일을 열려고 했습니다.")

sys.addaudithook(guard)
arguments = sys.argv[1:]
if arguments == ["__import__"]:
    import kbo_insight_ai
    print(kbo_insight_ai.__file__)
else:
    sys.argv = ["kbo-insight", *arguments]
    runpy.run_module("kbo_insight_ai", run_name="__main__")
'''


class FoundationTests(unittest.TestCase):
    """설치된 패키지의 핵심 정상·오류 계약을 확인한다."""

    def setUp(self) -> None:
        """공백·한글이 포함된 독립 테스트 영역을 준비한다."""
        temporary = TemporaryDirectory(prefix="kbo-ct1-")
        self.addCleanup(temporary.cleanup)
        self.sandbox = Path(temporary.name)
        self.root = self.sandbox / "한글 공백 프로젝트"
        self.other = self.sandbox / "다른 실행 위치"
        self.root.mkdir()
        self.other.mkdir()
        self._write_config()

    def _write_config(self, **changes: object) -> None:
        """테스트용 설정만 임시 영역에 기록한다."""
        values = {**BASE_CONFIG, **changes}
        content = "\n".join(
            f"{key} = {json.dumps(value, ensure_ascii=False)}"
            for key, value in values.items()
        )
        (self.root / "kbo.toml").write_text(content, encoding="utf-8")

    def _snapshot(self) -> dict[str, bytes | None]:
        """임시 영역의 파일 내용과 디렉터리 목록을 반환한다."""
        return {
            str(path.relative_to(self.sandbox)): (
                path.read_bytes() if path.is_file() else None
            )
            for path in self.sandbox.rglob("*")
        }

    def _run(
        self,
        *arguments: str,
        root: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """설치된 패키지를 다른 실행 위치에서 감시하며 실행한다."""
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        environment["PYTHONUTF8"] = "1"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["KBO_PROJECT_ROOT"] = str(
            self.root if root is None else root
        )
        environment["CT1_TEST_SECRET"] = "검증용_비밀값_대역"

        return subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                READONLY_BOOTSTRAP,
                *arguments,
            ],
            cwd=self.other,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=20,
            check=False,
        )

    def test_settings_do_not_depend_on_cwd(self) -> None:
        """상대 자료 경로와 기본 프로젝트 탐색이 실행 위치에 독립적이다."""
        with chdir(self.other):
            settings = load_settings(self.root, environ={})
            default_settings = load_settings(environ={})

        self.assertEqual(settings.raw_dir, self.root / "data" / "raw")
        self.assertFalse(settings.raw_dir.exists())
        expected = Path(kbo_insight_ai.__file__).resolve().parents[2]
        self.assertEqual(default_settings.project_root, expected)

    def test_root_priority_and_absolute_paths(self) -> None:
        """명시 기준이 환경 변수보다 우선하며 절대 자료 경로는 유지된다."""
        absolute = self.sandbox / "별도 원본"
        self._write_config(raw_dir=str(absolute))
        settings = load_settings(
            self.root,
            environ={"KBO_PROJECT_ROOT": str(self.sandbox / "없는 기준")},
        )
        self.assertEqual(settings.raw_dir, absolute)

        settings = load_settings(
            environ={"KBO_PROJECT_ROOT": str(self.root)}
        )
        self.assertEqual(settings.project_root, self.root)

    def test_invalid_types_and_ranges(self) -> None:
        """Seed·로그·경로의 잘못된 타입과 범위를 거절한다."""
        invalid: tuple[dict[str, object], ...] = (
            {"seed": True},
            {"seed": 1.5},
            {"seed": -1},
            {"seed": 2**32},
            {"seed": "42"},
            {"log_level": "info"},
            {"log_level": 20},
            {"raw_dir": ""},
            {"raw_dir": 1},
        )
        for changes in invalid:
            with self.subTest(changes=changes):
                self._write_config(**changes)
                with self.assertRaises(SettingsError):
                    load_settings(self.root, environ={})

    def test_config_schema_and_read_errors(self) -> None:
        """누락·미지원 설정·문법·읽기 실패를 명시적 오류로 처리한다."""
        config_file = self.root / "kbo.toml"
        for content in ('seed = 42\n', 'seed = [\n'):
            with self.subTest(content=content):
                config_file.write_text(content, encoding="utf-8")
                with self.assertRaises(SettingsError):
                    load_settings(self.root, environ={})

        self._write_config(unsupported=True)
        with self.assertRaises(SettingsError):
            load_settings(self.root, environ={})

        config_file.unlink()
        with self.assertRaises(SettingsError):
            load_settings(self.root, environ={})

        self._write_config()
        with patch.object(Path, "open", side_effect=PermissionError):
            with self.assertRaises(SettingsError):
                load_settings(self.root, environ={})

    def test_missing_or_invalid_root(self) -> None:
        """없는 기준·상대 기준·파일 기준을 거절한다."""
        blocker = self.sandbox / "기준 파일"
        blocker.write_text("테스트", encoding="utf-8")
        for root in (self.sandbox / "없는 기준", Path("."), blocker):
            with self.subTest(root=root):
                with self.assertRaises(SettingsError):
                    load_settings(root, environ={})

    def test_file_and_parent_conflicts(self) -> None:
        """산출물 경로나 그 상위가 파일이면 거절한다."""
        blocker = self.root / "파일"
        blocker.write_text("테스트", encoding="utf-8")
        for value in ("파일", "파일/하위"):
            with self.subTest(value=value):
                self._write_config(raw_dir=value)
                with self.assertRaises(SettingsError):
                    load_settings(self.root, environ={})

    def test_overlapping_or_protected_paths(self) -> None:
        """원본과 다른 산출물·소스·프로젝트 기준의 겹침을 거절한다."""
        for value in ("data/interim", "data", "src/자료", ".", ".."):
            with self.subTest(value=value):
                self._write_config(raw_dir=value)
                with self.assertRaises(SettingsError):
                    load_settings(self.root, environ={})

    def test_readonly_entry_points(self) -> None:
        """import·도움말·기본 확인이 감시 구간에서 읽기만 수행한다."""
        before = self._snapshot()
        imported = self._run("__import__")
        help_result = self._run(
            "--help",
            root=self.sandbox / "없는 기준",
        )
        checked = self._run("check")

        for result in (imported, help_result, checked):
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("검증용_비밀값_대역", result.stdout)
            self.assertNotIn("검증용_비밀값_대역", result.stderr)

        self.assertIn("check", help_result.stdout)
        payload = json.loads(checked.stdout)
        self.assertEqual(
            Path(payload["raw_dir"]),
            self.root / "data" / "raw",
        )
        self.assertEqual(before, self._snapshot())

    def test_cli_rejects_invalid_values_without_echo(self) -> None:
        """설정 오류를 성공 처리하거나 입력값을 로그에 노출하지 않는다."""
        marker = "검증용_잘못된_설정값"
        self._write_config(seed=marker)
        result = self._run("check")

        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertIn("ERROR kbo_insight_ai", result.stderr)
        self.assertIn("seed", result.stderr)
        self.assertNotIn(marker, result.stderr)
        self.assertNotIn("검증용_비밀값_대역", result.stderr)

    def test_non_utf8_config_error(self) -> None:
        """UTF-8이 아닌 설정을 오류 로그와 종료 코드 2로 처리한다."""
        (self.root / "kbo.toml").write_bytes(b"\xff")

        with self.assertRaises(SettingsError):
            load_settings(self.root, environ={})

        result = self._run("check")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertIn("UTF-8", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    @unittest.skipUnless(os.name == "nt", "Windows 경로 규칙 검증")
    def test_windows_ambiguous_paths(self) -> None:
        """프로젝트 기준을 벗어날 수 있는 모호한 Windows 경로를 거절한다."""
        for value in ("C:raw", "D:raw", "/raw", "\\raw"):
            with self.subTest(value=value):
                self._write_config(raw_dir=value)
                with self.assertRaises(SettingsError):
                    load_settings(self.root, environ={})


if __name__ == "__main__":
    unittest.main()