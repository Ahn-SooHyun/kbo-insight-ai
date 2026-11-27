"""최종 평가에 사용할 고정 자료 및 운영 모델을 사전 검사한다.

이 모듈에서는 최종 Test 정답을 조회하거나 성능을 계산하지 않는다.
학습, 보정, 모델 선정 및 원본 자산 변경도 수행하지 않는다.
"""

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
    """최종 평가 대상 저장 자산의 식별 계약."""

    name: str
    family: str
    run_id: str
    loader: Callable[[Path], object]
    role: str | None = None


def load_pitch_runtime(path: Path) -> object:
    """실제 학습 메타데이터가 있는 M5 자산만 복원한다."""
    return m5_cli.load_runtime(
        path,
        training_asset=True,
    )


ASSETS = (
    AssetSpec(
        "m1",
        "m1",
        "0d9433b8044e48e5a4d21bbdb9b432e5",
        m1_cli.load_model,
    ),
    AssetSpec(
        "m2",
        "m2",
        "158ce4d1ada94819bc3c355e281ba0e7",
        m2_cli.load_operational_model,
    ),
    AssetSpec(
        "m4",
        "m4",
        "52df3aae7b45406790165d4c871614b7",
        m4_cli.load_model,
    ),
    AssetSpec(
        "m5",
        "m5",
        "c06c10d5d90c4d4ebefee2a848c92f26",
        load_pitch_runtime,
    ),
    AssetSpec(
        "m3_hitter",
        "m3",
        "a1ac1c918d1a45ab9d9ee65ce1d34aa1",
        m3_cli.load_asset,
        "hitter",
    ),
    AssetSpec(
        "m3_pitcher",
        "m3",
        "d6dc6240aabb4ad9853a51c44dfa5429",
        m3_cli.load_asset,
        "pitcher",
    ),
)


def require(condition: bool, message: str) -> None:
    """자산 계약 위반을 발견하면 이후 평가를 차단한다."""
    if not condition:
        raise ValueError(message)


def describe_file(path: Path) -> dict[str, object]:
    """파일을 변경하지 않고 경로, 크기, 지문을 확인한다."""
    path = path.resolve(strict=True)

    require(
        path.is_file(),
        f"일반 파일이 아닙니다: {path}",
    )

    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def verify_boundaries(config: Mapping[str, object]) -> None:
    """사전에 합의한 시간대와 분할 경계를 검사한다."""
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
    """저장 자산에 기록된 실행 소스 지문을 비교한다."""
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
    """모델 DB 및 선행 자료 계약을 검사한다."""
    directory = (
        settings.processed_dir
        / "model_data"
        / DATA_RUN
    )

    paths = {
        "database": directory / "model_data.sqlite3",
        "manifest": directory / "manifest.json",
        "contract": directory / "feature_target_contract.json",
        "eda": directory / "train_eda.json",
        "supplement": directory / "supplement-impact.json",
        "standard": (
            settings.processed_dir
            / "standard"
            / STANDARD_RUN
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

    # 보완 자료의 JSON 파싱 가능 여부도 확인한다.
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
        "최종 평가 대상 모델의 입력 계약 누락",
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
    """개발 평가 기록과 저장 자산의 대응 관계를 확인한다."""
    if spec.family == "m3":
        require(
            payload["run_id"] == report["run_id"] == spec.run_id
            and payload["role"] == report["role"] == spec.role
            and payload["selected_model"] == report["selected_model"]
            and payload["final_test_used"] is False
            and report["final_test_used"] is False
            and report["asset_sha256"] == model_sha256
            and report["asset_fingerprint"] == fingerprint(payload),
            f"{spec.name}: 자산 및 개발 평가 계약 불일치",
        )
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
        payload["evaluation_fingerprint"] == fingerprint(report)
        and payload["selected_model"] == report["selected_model"]
        and payload["data"] == report["data"],
        f"{spec.name}: 개발 평가 지문 또는 모델 계약 불일치",
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
    """기존 로더를 재사용해 운영 모델을 복원하고 지문을 확인한다."""
    model_path = (
        settings.models_dir
        / spec.family
        / spec.run_id
        / "model.json"
    )

    result_dir = (
        settings.results_dir
        / spec.family
        / spec.run_id
    )

    evaluation_path = result_dir / "evaluation.json"

    model_file = describe_file(model_path)
    evaluation_file = describe_file(evaluation_path)

    envelope = as_row(read_json(model_path))
    payload = as_row(envelope["payload"])
    report = as_row(read_json(evaluation_path))
    provenance = as_row(payload["data"])

    digest_key = (
        "fingerprint"
        if spec.family == "m3"
        else "sha256"
    )

    require(
        envelope[digest_key] == fingerprint(payload),
        f"{spec.name}: 모델 내부 지문 불일치",
    )

    # 모델별 기존 운영 로더의 검증을 우회하지 않는다.
    spec.loader(model_path)

    verify_report(
        spec,
        payload,
        report,
        str(model_file["sha256"]),
    )

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

    extra_files: dict[str, object] = {}

    if spec.family == "m3":
        manifest_path = result_dir / "manifest.json"
        manifest = as_row(read_json(manifest_path))

        require(
            manifest["status"] == "completed"
            and manifest["run_id"] == spec.run_id
            and manifest["asset_sha256"] == model_file["sha256"]
            and manifest["evaluation_sha256"]
            == evaluation_file["sha256"],
            f"{spec.name}: 완료 실행 명세 불일치",
        )

        extra_files["manifest"] = describe_file(manifest_path)

    return {
        "name": spec.name,
        "family": spec.family,
        "role": spec.role,
        "training_run_id": spec.run_id,
        "schema": payload["schema"],
        "selected_model": payload.get(
            "selected_model",
            "profile_histogram",
        ),
        "model_file": model_file,
        "evaluation_file": evaluation_file,
        "model_fingerprint": fingerprint(payload),
        "source_hashes": source_hashes,
        "extra_files": extra_files,
        "status": "verified",
    }


def inspect_assets(settings: Settings) -> dict[str, object]:
    """최종 성능 평가 없이 고정 운영 자산 전체를 검사한다."""
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

    parser.add_argument(
        "--project-root",
        required=True,
        type=Path,
    )

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

    except Exception as exc:
        LOGGER.error("고정 자산 검사 실패: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
