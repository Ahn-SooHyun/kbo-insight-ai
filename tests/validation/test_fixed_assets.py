"""고정 최종 평가 자산의 사전 검사 계약을 검증한다."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.datasets.published_sources import sha256_file
from src.evaluation.fixed_assets import (
    ASSETS,
    BOUNDARIES,
    TIMEZONE,
    describe_file,
    verify_boundaries,
    verify_source_hashes,
)


class FixedAssetsTests(unittest.TestCase):
    """실제 최종 Test를 읽지 않는 합성 검증."""

    def test_asset_inventory(self) -> None:
        """운영 예측기 6개와 역할 구분을 유지한다."""
        self.assertEqual(len(ASSETS), 6)

        self.assertEqual(
            {asset.name for asset in ASSETS},
            {
                "m1",
                "m2",
                "m4",
                "m5",
                "m3_hitter",
                "m3_pitcher",
            },
        )

        self.assertEqual(
            len({asset.run_id for asset in ASSETS}),
            6,
        )

        self.assertEqual(
            {
                asset.role
                for asset in ASSETS
                if asset.family == "m3"
            },
            {"hitter", "pitcher"},
        )

    def test_valid_boundaries(self) -> None:
        """합의된 시간대와 날짜 경계를 허용한다."""
        verify_boundaries(
            {
                "timezone": TIMEZONE,
                "validation_start": BOUNDARIES[0],
                "test_start": BOUNDARIES[1],
                "observation_end": BOUNDARIES[2],
            }
        )

    def test_invalid_timezone(self) -> None:
        """다른 시간대는 최종 평가 전에 차단한다."""
        with self.assertRaises(ValueError):
            verify_boundaries(
                {
                    "timezone": "UTC",
                    "validation_start": BOUNDARIES[0],
                    "test_start": BOUNDARIES[1],
                    "observation_end": BOUNDARIES[2],
                }
            )

    def test_invalid_test_boundary(self) -> None:
        """최종 Test 종료 경계 변경을 거부한다."""
        with self.assertRaises(ValueError):
            verify_boundaries(
                {
                    "timezone": TIMEZONE,
                    "validation_start": BOUNDARIES[0],
                    "test_start": BOUNDARIES[1],
                    "observation_end": "2026-11-27",
                }
            )

    def test_file_description(self) -> None:
        """지문 검사로 파일 내용이 변경되지 않는다."""
        with TemporaryDirectory() as directory:
            path = Path(directory) / "asset.json"

            original = b'{"value": 1}'
            path.write_bytes(original)

            result = describe_file(path)

            self.assertEqual(
                result["sha256"],
                sha256_file(path),
            )

            self.assertEqual(
                result["bytes"],
                len(original),
            )

            self.assertEqual(
                path.read_bytes(),
                original,
            )

    def test_missing_file(self) -> None:
        """없는 파일은 정상 자산으로 간주하지 않는다."""
        with TemporaryDirectory() as directory:
            path = Path(directory) / "missing.json"

            with self.assertRaises(FileNotFoundError):
                describe_file(path)

    def test_source_hash_match(self) -> None:
        """같은 소스의 저장 지문을 허용한다."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "model.py"

            path.write_text(
                "VALUE = 1\n",
                encoding="utf-8",
            )

            expected = {
                "model.py": sha256_file(path),
            }

            self.assertEqual(
                verify_source_hashes(root, expected),
                expected,
            )

    def test_modified_source_rejected(self) -> None:
        """저장 모델 학습 이후의 소스 변경을 탐지한다."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "model.py"

            path.write_text(
                "VALUE = 1\n",
                encoding="utf-8",
            )

            expected = {
                "model.py": sha256_file(path),
            }

            path.write_text(
                "VALUE = 2\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                verify_source_hashes(root, expected)

    def test_external_source_rejected(self) -> None:
        """저장소 외부 경로를 학습 소스로 허용하지 않는다."""
        with TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "repository"
            root.mkdir()

            external = parent / "external.py"

            external.write_text(
                "VALUE = 1\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                verify_source_hashes(
                    root,
                    {
                        "../external.py": sha256_file(external),
                    },
                )

    def test_empty_source_hashes_rejected(self) -> None:
        """학습 소스 지문이 없는 자산을 거부한다."""
        with TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                verify_source_hashes(
                    Path(directory),
                    {},
                )


if __name__ == "__main__":
    unittest.main()
