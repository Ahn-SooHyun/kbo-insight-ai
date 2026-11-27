"""최종 평가에 사용할 고정 자료 및 운영 모델의 무결성 검사."""

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
import platform
import sqlite3
from typing import Callable

from ..datasets.model_data import fingerprint
from ..datasets.published_sources import (
    as_row,
    read_json,
    sha256_file,
)
from ..models.game_outcome import __main__ as m1_cli
from ..models.plate_appearance import __main__ as m2_cli
from ..models.game_state_value import __main__ as m4_cli
from ..models.pitch_outcome import __main__ as m5_cli
from ..models.player_performance import __main__ as m3_cli
from ..settings import Settings, load_settings


LOGGER = logging.getLogger(__name__)

BASE_COMMIT = "53c2d41f541b1599a253ebc6f863ea1f8f8cfa4a"

DATA_RUN = "259b38decaac4ed68dd5a596cfbe4849"

DATA_SHA256 = (
    "c8383a9d3e8930830a137d1b6791cbb26"
    "e2316dbbbdf4f0f436ceda9a43890d0"
)

STANDARD_RUN = "e020dafedc8b4fd9ac6696d0e4e4a92e"
STANDARD_VERIFICATION = "37b56c39caf5435e8fb2a17f993ec4df"

TIMEZONE = "Asia/Seoul"

BOUNDARIES = (
    "2025-01-01",
    "2026-01-01",
    "2026-11-26",
)


@dataclass(frozen=True)
class AssetSpec:
    """최종 평가 대상 운영 모델의 고정 식별 계약."""

    name: str
    family: str
    run_id: str
    loader: Callable[[Path], object]
    role: str | None = None


def load_pitch_runtime(path: Path) -> object:
    """실제 학습 메타데이터가 있는 M5 모델만 복원한다."""
    return m5_cli.load_runtime(path, training_asset=True)


ASSETS = (
    AssetSpec(
        "m1", "m1",
        "0d9433b8044e48e5a4d21bbdb9b432e5",
        m1_cli.load_model,
    ),
    AssetSpec(
        "m2", "m2",
        "158ce4d1ada94819bc3c355e281ba0e7",
        m2_cli.load_operational_model,
    ),
    AssetSpec(
        "m4", "m4",
        "52df3aae7b45406790165d4c871614b7",
        m4_cli.load_model,
    ),
    AssetSpec(
        "m5", "m5",
        "c06c10d5d90c4d4ebefee2a848c92f26",
        load_pitch_runtime,
    ),
    AssetSpec(
        "m3_hitter", "m3",
        "a1ac1c918d1a45ab9d9ee65ce1d34aa1",
        m3_cli.load_asset, "hitter",
    ),
    AssetSpec(
        "m3_pitcher", "m3",
        "d6dc6240aabb4ad9853a51c44dfa5429",
        m3_cli.load_asset, "pitcher",
    ),
)


def require(condition: bool, message: str) -> None:
    """계약 불일치 시 후속 평가를 차단한다."""
    if not condition:
        raise ValueError(message)


def describe_file(path: Path) -> dict[str, object]:
    """파일을 변경하지 않고 실제 바이트 지문을 계산한다."""
    path = path.resolve(strict=True)

    require(path.is_file(), f"일반 파일이 아닙니다: {path}")

    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def unique_json_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    """동일 JSON 키가 중복되어 의미가 달라지는 경우를 거부한다."""
    result: dict[str, object] = {}

    for key, value in pairs:
        require(key not in result, f"JSON 중복 키: {key}")
        result[key] = value

    return result


def read_unique_json(path: Path) -> dict[str, object]:
    """중복 키가 없는 JSON 객체만 읽는다."""
    return as_row(
        json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique_json_object,
        )
    )


def original_json_bytes(value: object) -> bytes:
    """기존 published_sources.write_json의 저장 형식을 재현한다."""
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
            sort_keys=True,
        ) + "\n"
    ).encode("utf-8")


def verify_historical_asset(
    path: Path,
    expected: str,
) -> dict[str, object]:
    """현재 파일 또는 기존 저장 형식의 정확한 SHA-256을 검사한다.

    내부 Fingerprint 일치만으로 실제 바이트 불일치를 허용하지 않는다.
    기존 저장 형식의 재현 결과도 일치하지 않으면 평가를 차단한다.
    원본 파일과 개발 평가 기록은 변경하지 않는다.
    """
    actual = sha256_file(path)
    envelope = read_unique_json(path)

    reconstructed = hashlib.sha256(
        original_json_bytes(envelope)
    ).hexdigest()

    require(
        actual == expected or reconstructed == expected,
        (
            f"저장 모델의 원래 바이트 지문을 증명할 수 없습니다: "
            f"{path.name}; recorded={expected}; "
            f"actual={actual}; reconstructed={reconstructed}"
        ),
    )

    return {
        "recorded_sha256": expected,
        "actual_sha256": actual,
        "reconstructed_sha256": reconstructed,
        "mode": (
            "exact_bytes"
            if actual == expected
            else "original_serialization_reconstructed"
        ),
    }


def verify_boundaries(config: Mapping[str, object]) -> None:
    """KST 시간대와 고정 분할 경계를 검사한다."""
    limits = tuple(
        config.get(name)
        for name in (
            "validation_start",
            "test_start",
            "observation_end",
        )
    )

    require(
        config.get("timezone") == TIMEZONE,
        "자료 시간대 불일치",
    )

    require(
        limits == BOUNDARIES,
        "자료 분할 경계 불일치",
    )


def verify_source_hashes(
    root: Path,
    expected: Mapping[str, object],
) -> dict[str, str]:
    """저장 자산에 기록된 학습 코드 및 설정 지문을 대조한다."""
    root = root.resolve(strict=True)

    require(bool(expected), "학습 소스 지문 누락")

    verified: dict[str, str] = {}

    for name, digest in sorted(expected.items()):
        relative = Path(name)

        require(
            not relative.is_absolute(),
            f"상대 소스 경로가 필요합니다: {name}",
        )

        path = (root / relative).resolve(strict=True)

        require(
            path.is_relative_to(root) and path.is_file(),
            f"저장소 외부 또는 잘못된 소스 경로: {name}",
        )

        actual = sha256_file(path)

        require(
            actual == digest,
            f"학습 당시 소스 지문 불일치: {name}",
        )

        verified[name] = actual

    return verified


def inspect_data(settings: Settings) -> dict[str, object]:
    """고정 모델 DB 및 선행 자료 계약을 검사한다."""
    directory = settings.processed_dir / "model_data" / DATA_RUN

    paths = {
        "database": directory / "model_data.sqlite3",
        "manifest": directory / "manifest.json",
        "contract": directory / "feature_target_contract.json",
        "eda": directory / "train_eda.json",
        "supplement": directory / "supplement-impact.json",
        "standard": (
            settings.processed_dir / "standard" / STANDARD_RUN
            / f"verified-{STANDARD_VERIFICATION}.sqlite3"
        ),
    }

    files = {
        name: describe_file(path)
        for name, path in paths.items()
    }

    manifest = as_row(read_json(paths["manifest"]))
    config = as_row(manifest["config"])
    source = as_row(manifest["source"])
    contract = as_row(read_json(paths["contract"]))
    eda = read_json(paths["eda"])

    as_row(read_json(paths["supplement"]))

    require(
        manifest["status"] == "completed"
        and manifest["run_id"] == DATA_RUN,
        "완료 모델 자료 실행 계약 불일치",
    )

    verify_boundaries(config)

    require(
        files["database"]["sha256"]
        == manifest["database_sha256"]
        == DATA_SHA256,
        "모델 DB 지문 불일치",
    )

    require(
        fingerprint(contract) == manifest["contract_sha256"],
        "입력·정답 계약 지문 불일치",
    )

    require(
        fingerprint(eda) == manifest["eda_sha256"],
        "Train EDA 지문 불일치",
    )

    require(
        source["run_id"] == STANDARD_RUN
        and source["verification_id"] == STANDARD_VERIFICATION
        and source["database_sha256"] == files["standard"]["sha256"],
        "표준 자료 인수 계약 불일치",
    )

    require(
        all(spec.name in contract for spec in ASSETS),
        "최종 평가 대상 모델 입력 계약 누락",
    )

    return {
        "run_id": DATA_RUN,
        "standard_run_id": STANDARD_RUN,
        "standard_verification_id": STANDARD_VERIFICATION,
        "timezone": TIMEZONE,
        "boundaries": list(BOUNDARIES),
        "contract_fingerprint": fingerprint(contract),
        "eda_fingerprint": fingerprint(eda),
        "files": files,
    }


def verify_report(
    spec: AssetSpec,
    payload: Mapping[str, object],
    report: Mapping[str, object],
    model_sha256: str,
) -> None:
    """저장 모델과 기존 개발 평가 기록의 대응 관계를 검사한다."""
    if spec.family == "m3":
        checks = {
            "run_id": (
                payload["run_id"]
                == report["run_id"]
                == spec.run_id
            ),
            "role": (
                payload["role"]
                == report["role"]
                == spec.role
            ),
            "selected_model": (
                payload["selected_model"] == report["selected_model"]
            ),
            "final_test_used": (
                payload["final_test_used"] is False
                and report["final_test_used"] is False
            ),
            "asset_sha256": report["asset_sha256"] == model_sha256,
            "asset_fingerprint": (
                report["asset_fingerprint"] == fingerprint(payload)
            ),
            "source_hashes": (
                payload["source_hashes"] == report["source_hashes"]
            ),
            "data": (
                payload["data"] == {
                    key: value
                    for key, value in as_row(report["data"]).items()
                    if key != "verified_season_ends"
                }
            ),
        }

        for name, valid in checks.items():
            require(valid, f"{spec.name}: 개발 평가 {name} 불일치")

        return

    require(
        payload["training_run_id"]
        == report["training_run_id"]
        == spec.run_id,
        f"{spec.name}: 학습 실행 ID 불일치",
    )

    if spec.family == "m1":
        require(
            report["model_schema"] == payload["schema"]
            and report["splits"] == payload["splits"]
            and report["final_test"] == "held_out",
            "M1 개발 평가 계약 불일치",
        )
        return

    require(
        payload["evaluation_fingerprint"] == fingerprint(report),
        f"{spec.name}: 개발 평가 지문 불일치",
    )

    require(
        payload["selected_model"] == report["selected_model"]
        and payload["data"] == report["data"],
        f"{spec.name}: 모델·자료 계약 불일치",
    )

    if spec.family == "m2":
        require(
            report["final_test"] == "held_out_targets_not_evaluated",
            "M2 최종 Test 미사용 기록 불일치",
        )
        return

    require(
        report["final_test_evaluated"] is False,
        f"{spec.name}: 최종 Test 미사용 기록 불일치",
    )


def inspect_asset(
    settings: Settings,
    spec: AssetSpec,
    data_info: Mapping[str, object],
) -> dict[str, object]:
    """기존 운영 로더와 저장 당시 무결성 계약을 함께 검사한다."""
    model_path = (
        settings.models_dir / spec.family / spec.run_id / "model.json"
    )

    result_dir = settings.results_dir / spec.family / spec.run_id
    evaluation_path = result_dir / "evaluation.json"

    model_file = describe_file(model_path)
    evaluation_file = describe_file(evaluation_path)

    envelope = read_unique_json(model_path)
    payload = as_row(envelope["payload"])
    report = read_unique_json(evaluation_path)
    provenance = as_row(payload["data"])

    digest_key = "fingerprint" if spec.family == "m3" else "sha256"

    require(
        envelope[digest_key] == fingerprint(payload),
        f"{spec.name}: 모델 내부 지문 불일치",
    )

    spec.loader(model_path)

    extra_files: dict[str, object] = {}
    historical: dict[str, object] | None = None

    verified_model_hash = str(model_file["sha256"])

    if spec.family == "m3":
        manifest_path = result_dir / "manifest.json"
        manifest = read_unique_json(manifest_path)

        require(
            manifest["status"] == "completed"
            and manifest["run_id"] == spec.run_id
            and manifest["evaluation_sha256"]
            == evaluation_file["sha256"]
            and manifest["asset_fingerprint"]
            == envelope["fingerprint"],
            f"{spec.name}: 완료 실행 명세 불일치",
        )

        require(
            manifest["asset_sha256"] == report["asset_sha256"],
            f"{spec.name}: 기존 두 기록의 모델 SHA-256 불일치",
        )

        verified_model_hash = str(manifest["asset_sha256"])

        historical = verify_historical_asset(
            model_path,
            verified_model_hash,
        )

        extra_files["manifest"] = describe_file(manifest_path)

    verify_report(spec, payload, report, verified_model_hash)

    require(
        provenance["model_data_run_id"] == DATA_RUN
        and provenance["timezone"] == TIMEZONE
        and provenance["contract_sha256"]
        == data_info["contract_fingerprint"],
        f"{spec.name}: 학습 자료 계약 불일치",
    )

    database_digest_key = (
        "database_sha256_from_manifest"
        if spec.family == "m1"
        else "database_sha256"
    )

    require(
        provenance[database_digest_key] == DATA_SHA256,
        f"{spec.name}: 학습 DB 지문 불일치",
    )

    hashes_key = (
        "source_hashes"
        if spec.family == "m3"
        else "source_sha256"
    )

    source_hashes = verify_source_hashes(
        settings.project_root,
        as_row(payload[hashes_key]),
    )

    return {
        "name": spec.name,
        "family": spec.family,
        "role": spec.role,
        "training_run_id": spec.run_id,
        "schema": payload["schema"],
        "selected_model": payload.get(
            "selected_model", "profile_histogram"
        ),
        "model_file": model_file,
        "evaluation_file": evaluation_file,
        "model_fingerprint": fingerprint(payload),
        "historical_integrity": historical,
        "source_hashes": source_hashes,
        "extra_files": extra_files,
        "status": "verified",
    }


def inspect_assets(settings: Settings) -> dict[str, object]:
    """최종 정답을 조회하지 않고 운영 자산 전체를 검사한다."""
    data_info = inspect_data(settings)

    assets = [
        inspect_asset(settings, spec, data_info)
        for spec in ASSETS
    ]

    return {
        "status": "verified",
        "expected_base_commit": BASE_COMMIT,
        "python": platform.python_version(),
        "sqlite": sqlite3.sqlite_version,
        "configured_seed": settings.seed,
        "data": data_info,
        "asset_count": len(assets),
        "assets": assets,
        "final_test_scored": False,
        "final_evaluation_manifest": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """성공 0, 검사 실패 2, 사용자 중단 3을 반환한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)

    args = parser.parse_args(argv)

    try:
        settings = load_settings(args.project_root)

        logging.basicConfig(
            level=settings.log_level,
            format="%(levelname)s %(name)s: %(message)s",
        )

        result = inspect_assets(settings)

        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        )

        return 0

    except KeyboardInterrupt:
        LOGGER.warning("고정 자산 검사 중단")
        return 3

    except (
        OSError, ValueError, TypeError, KeyError,
        AttributeError, sqlite3.DatabaseError,
    ) as exc:
        LOGGER.error("고정 자산 검사 실패: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())