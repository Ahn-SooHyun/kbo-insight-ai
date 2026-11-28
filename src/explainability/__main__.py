"""고정 예측 설명의 사전 계획·실행·독립 재현 검증 CLI."""

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
from uuid import uuid4

from . import (
    ABS_TOLERANCE,
    BASE_COMMIT,
    FAILURE_CASE_LIMIT,
    FINAL_EVALUATION_RUN,
    FINAL_MANIFEST_HASH,
    FINAL_RESULT_HASH,
    GLOBAL_SAMPLE_LIMIT,
    M2_RARE_SUPPORT,
    M5_RARE_SUPPORT,
    REL_TOLERANCE,
    RESULT_DIRECTORY,
    SCHEMA,
    SEASON_PLAN_HASH,
    SEASON_RESULT_HASH,
    SEASON_VALIDATION_RUN,
    SEASON_VERIFICATION_ID,
    SEED,
    SPARSE_BIN_SUPPORT,
    VERSION,
)
from . import contributions as explanation
from . import diagnostics
from ..datasets.model_data import fingerprint
from ..datasets.published_sources import sha256_file
from ..evaluation import final_evaluation as final
from ..evaluation import fixed_assets as assets
from ..evaluation import game_outcome as e1
from ..models.player_performance import model as m3
from ..settings import Settings, load_settings


LOGGER = logging.getLogger(__name__)

JsonObject = dict[str, object]

MODEL_NAMES = tuple(
    spec.name for spec in assets.ASSETS
)

LINEAR_MODELS = ("m2", "m4", "m5")


def utc_now() -> str:
    """증거 생성 시각을 UTC로 반환한다."""
    return datetime.now(timezone.utc).isoformat()


def write_json(
    path: Path,
    data: Mapping[str, object],
) -> None:
    """기존 파일을 덮어쓰지 않고 JSON 증거를 저장한다."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "x",
        encoding="utf-8",
        newline="\n",
    ) as stream:
        json.dump(
            data,
            stream,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )

        stream.write("\n")


def git_state(root: Path) -> JsonObject:
    """기준 커밋 계보와 실제 작업 트리를 읽기 전용으로 확인한다."""
    head = final.git_read(
        root,
        "rev-parse",
        "HEAD",
    )

    final.git_read(
        root,
        "merge-base",
        "--is-ancestor",
        BASE_COMMIT,
        head,
    )

    return {
        "base_commit": BASE_COMMIT,
        "head": head,
        "branch": final.git_read(
            root,
            "branch",
            "--show-current",
        ),
        "worktree_status": final.git_read(
            root,
            "status",
            "--short",
        ),
    }


def policy() -> JsonObject:
    """Test 설명 전에 고정하는 분석 방법과 지원 범위."""
    return {
        "models": {
            "m1": {
                "method": "profile_and_joint_distribution",
                "outputs": [
                    "home_expected_runs",
                    "away_expected_runs",
                    "home_tie_away_probability",
                ],
                "background": (
                    "stored_train_site_and_profile_counts"
                ),
            },
            "m2": {
                "method": "exact_interventional_linear_shap",
                "outputs": "all_raw_class_logits",
                "background": "all_eligible_train_encoded_inputs",
                "postprocessing": [
                    "rule_mask",
                    "train_prior_mix",
                    "stored_temperature",
                    "stored_class_offsets",
                ],
            },
            "m4": {
                "method": "exact_interventional_linear_shap",
                "outputs": [
                    "nonterminal_win_logit",
                    "remaining_expected_runs",
                    "observed_final_boundary",
                    "observed_plate_value",
                ],
                "background": "stored_train_weighted_means",
            },
            "m5": {
                "method": "exact_interventional_linear_shap",
                "outputs": [
                    "all_joint_logits",
                    "joint_probability",
                    "pitch_type_marginal",
                    "pitch_result_marginal",
                ],
                "background": "all_eligible_train_encoded_inputs",
            },
            "m3_hitter": {
                "method": "history_calculation_decomposition",
                "outputs": "all_supported_period_counts",
                "background": "stored_train_daily_prior",
            },
            "m3_pitcher": {
                "method": "history_and_residual_tree_path",
                "outputs": "all_supported_period_counts",
                "background": (
                    "stored_train_daily_prior_and_tree_nodes"
                ),
            },
        },
        "train_boundary_exclusive": assets.BOUNDARIES[0],
        "test_start": assets.BOUNDARIES[1],
        "test_end_exclusive": assets.BOUNDARIES[2],
        "seed": SEED,
        "global_sample_limit": GLOBAL_SAMPLE_LIMIT,
        "global_sample_selection": (
            "smallest_sha256_of_seed_colon_sample_id"
        ),
        "global_sample_weighting": "equal_per_selected_sample",
        "failure_case_limit_per_reason": FAILURE_CASE_LIMIT,
        "failure_case_selection": (
            "error_descending_then_sample_id_ascending"
        ),
        "failure_case_counts": "full_fixed_test_population",
        "failure_case_global_weighting": "excluded",
        "calibration": {
            "m1_bin_edges": e1.BINS,
            "other_probability_bin_edges": [
                index / 10
                for index in range(11)
            ],
            "last_bin_includes_one": True,
            "empty_bin_mean": None,
            "sparse_bin_support_exclusive": (
                SPARSE_BIN_SUPPORT
            ),
            "m2_rare_train_support_exclusive": (
                M2_RARE_SUPPORT
            ),
            "m5_rare_train_support_exclusive": (
                M5_RARE_SUPPORT
            ),
            "unseen_train_support": 0,
        },
        "player_interval_target_coverage": (
            m3.INTERVAL_LEVEL
        ),
        "absolute_tolerance": ABS_TOLERANCE,
        "relative_tolerance": REL_TOLERANCE,
        "test_used_for_training": False,
        "test_used_for_selection": False,
        "test_used_for_calibration": False,
        "causal_interpretation": False,
        "new_operating_models": 0,
    }


def fixed_paths(
    settings: Settings,
) -> dict[str, Path]:
    """인수한 평가 및 시즌 검증의 고정 파일 경로."""
    final_dir = (
        settings.results_dir
        / "final_evaluation"
        / FINAL_EVALUATION_RUN
    )

    season_dir = (
        settings.results_dir
        / "season_forecast_validation"
        / SEASON_VALIDATION_RUN
    )

    return {
        "final_manifest": (
            final_dir / "manifest.json"
        ),
        "final_result": (
            final_dir / "evaluation" / "result.json"
        ),
        "season_plan": (
            season_dir / "plan.json"
        ),
        "season_verification": (
            season_dir
            / f"verification-{SEASON_VERIFICATION_ID}.json"
        ),
    }


def verify_digest(
    root: Path,
    path: Path,
    expected: str,
) -> Path:
    """저장소 내부의 일반 파일 및 실제 SHA-256을 확인한다."""
    resolved = path.resolve(strict=True)

    assets.require(
        resolved.is_relative_to(root.resolve())
        and resolved.is_file(),
        f"고정 파일 경로 오류: {path}",
    )

    assets.require(
        isinstance(expected, str)
        and len(expected) == 64
        and sha256_file(resolved) == expected.lower(),
        f"고정 파일 SHA-256 불일치: {resolved}",
    )

    return resolved


def remember_reference(
    root: Path,
    inventory: dict[str, str],
    path: Path,
    digest: str,
) -> Path:
    """동일 참조가 서로 다른 지문을 요구하면 중단한다."""
    resolved = path.resolve(strict=True)

    assets.require(
        resolved.is_relative_to(root.resolve())
        and resolved.is_file(),
        f"입력 참조 경로 오류: {path}",
    )

    key = str(resolved)
    digest = digest.lower()

    if key in inventory:
        assets.require(
            inventory[key] == digest,
            f"동일 파일의 지문 충돌: {resolved}",
        )
    else:
        verify_digest(root, resolved, digest)
        inventory[key] = digest

    return resolved


def remember_info(
    root: Path,
    inventory: dict[str, str],
    information: Mapping[str, object],
    *,
    parent: Path | None = None,
) -> Path:
    """기존 파일 참조의 경로·크기·지문을 검증한다."""
    path = remember_reference(
        root,
        inventory,
        Path(information["path"]),
        information["sha256"],
    )

    if parent is not None:
        assets.require(
            path.is_relative_to(parent.resolve()),
            f"실행 폴더 외부 파일 참조: {path}",
        )

    if "bytes" in information:
        assets.require(
            path.stat().st_size == information["bytes"],
            f"파일 크기 불일치: {path}",
        )

    return path


def read_fixed(
    settings: Settings,
) -> tuple[
    JsonObject,
    JsonObject,
    JsonObject,
    JsonObject,
]:
    """선행 명세·결과의 고정 지문과 식별자를 검사한다."""
    paths = fixed_paths(settings)

    for name, digest in (
        ("final_manifest", FINAL_MANIFEST_HASH),
        ("final_result", FINAL_RESULT_HASH),
        ("season_plan", SEASON_PLAN_HASH),
        ("season_verification", SEASON_RESULT_HASH),
    ):
        verify_digest(
            settings.project_root,
            paths[name],
            digest,
        )

    manifest = assets.read_unique_json(
        paths["final_manifest"]
    )

    result = assets.read_unique_json(
        paths["final_result"]
    )

    plan = assets.read_unique_json(
        paths["season_plan"]
    )

    verification = assets.read_unique_json(
        paths["season_verification"]
    )

    assets.require(
        manifest["schema"] == "fixed-final-evaluation"
        and manifest["status"] == "frozen"
        and manifest["run_id"] == FINAL_EVALUATION_RUN
        and manifest["final_test_scored"] is False,
        "선행 최종 평가 명세 오류",
    )

    assets.require(
        result["status"] == "completed"
        and result["run_id"] == FINAL_EVALUATION_RUN
        and result["manifest_sha256"] == FINAL_MANIFEST_HASH
        and result["completed_models"] == len(MODEL_NAMES)
        and set(result["models"]) == set(MODEL_NAMES)
        and result["final_test_used_for_selection"] is False
        and result["final_test_used_for_refit"] is False
        and result["final_test_used_for_calibration"] is False,
        "선행 최종 평가 완료 또는 누수 계약 오류",
    )

    assets.require(
        plan["status"] == "prepared"
        and plan["run_id"] == SEASON_VALIDATION_RUN
        and verification["status"] == "completed"
        and verification["run_id"] == SEASON_VALIDATION_RUN
        and verification["plan_sha256"] == SEASON_PLAN_HASH
        and not verification["failures"],
        "선행 시즌 검증 명세 또는 완료 상태 오류",
    )

    return manifest, result, plan, verification


def input_inventory(
    settings: Settings,
    fixed: tuple[
        JsonObject,
        JsonObject,
        JsonObject,
        JsonObject,
    ],
    inspection: Mapping[str, object],
) -> dict[str, str]:
    """고정 설명에 실제 사용하는 선행 파일의 지문을 연결한다."""
    manifest, result, plan, verification = fixed

    root = settings.project_root
    paths = fixed_paths(settings)

    inventory: dict[str, str] = {}

    for name, digest in (
        ("final_manifest", FINAL_MANIFEST_HASH),
        ("final_result", FINAL_RESULT_HASH),
        ("season_plan", SEASON_PLAN_HASH),
        ("season_verification", SEASON_RESULT_HASH),
    ):
        remember_reference(
            root,
            inventory,
            paths[name],
            digest,
        )

    final_dir = paths["final_manifest"].parent
    evaluation_dir = paths["final_result"].parent

    for information in inspection["data"]["files"].values():
        remember_info(
            root,
            inventory,
            information,
        )

    for record in inspection["assets"]:
        for information in (
            record["model_file"],
            record["evaluation_file"],
            *record["extra_files"].values(),
        ):
            remember_info(
                root,
                inventory,
                information,
            )

    for name in MODEL_NAMES:
        remember_info(
            root,
            inventory,
            manifest["baselines"][name],
            parent=final_dir,
        )

        record = result["models"][name]

        assets.require(
            record["status"] == "completed",
            f"{name}: 선행 평가 미완료",
        )

        for filename in (
            "predictions",
            "targets",
            "metrics",
        ):
            remember_info(
                root,
                inventory,
                record[filename],
                parent=evaluation_dir / name,
            )

        assets.require(
            record["predictions"]["rows"]
            == record["targets"]["rows"]
            == record["prediction_samples"],
            f"{name}: 선행 저장 행 분모 오류",
        )

    for key in ("audit", "prediction_stage"):
        remember_info(
            root,
            inventory,
            result[key],
            parent=evaluation_dir,
        )

    # 이전 검증은 당시 코드 전체의 지문을 보존한다.
    # 현재 설명 코드가 추가되는 것은 허용하되
    # 선행 데이터·모델·실행 산출물의 변경은 허용하지 않는다.
    immutable_roots = (
        settings.processed_dir.resolve(),
        settings.models_dir.resolve(),
        settings.results_dir.resolve(),
        settings.raw_dir.resolve(),
        settings.interim_dir.resolve(),
    )

    for raw_path, digest in plan["input_sha256"].items():
        path = Path(raw_path).resolve()

        if any(
            path.is_relative_to(directory)
            for directory in immutable_roots
        ):
            remember_reference(
                root,
                inventory,
                path,
                digest,
            )

    jobs = verification["jobs"]

    for name, job in jobs.items():
        assets.require(
            job["returncode"] == 0,
            f"선행 검증 작업 실패: {name}",
        )

        record = remember_reference(
            root,
            inventory,
            Path(job["record"]),
            job["record_sha256"],
        )

        record_payload = assets.read_unique_json(record)

        assets.require(
            record_payload["name"] == name
            and record_payload["status"] == "completed"
            and record_payload["returncode"] == 0
            and record_payload["plan_sha256"]
            == SEASON_PLAN_HASH,
            f"선행 작업 기록 식별 오류: {name}",
        )

        explanation.compare(
            record_payload["artifact_sha256"],
            job["artifact_sha256"],
            f"{name}: 작업 산출물 참조",
        )

        explanation.compare(
            record_payload["result"],
            job["result"],
            f"{name}: 작업 결과 참조",
        )

        for raw_path, digest in job["artifact_sha256"].items():
            remember_reference(
                root,
                inventory,
                Path(raw_path),
                digest,
            )

    return dict(sorted(inventory.items()))


def verify_inventory(
    root: Path,
    inventory: Mapping[str, str],
) -> None:
    """사전에 고정한 입력 파일 전체의 변경 여부를 검사한다."""
    assets.require(
        bool(inventory),
        "고정 설명 입력 지문 목록 누락",
    )

    for raw_path, digest in inventory.items():
        verify_digest(
            root,
            Path(raw_path),
            digest,
        )


def load_operating(
    settings: Settings,
    spec: assets.AssetSpec,
):
    """기존 운영 로더를 사용해 고정 모델을 복원한다."""
    path = (
        settings.models_dir
        / spec.family
        / spec.run_id
        / "model.json"
    )

    return spec.loader(path)


def prepare(
    settings: Settings,
) -> JsonObject:
    """실제 Test 설명 전에 배경·방법·입력 지문을 고정한다."""
    LOGGER.info("고정 평가 및 시즌 검증 자료 확인")

    root = settings.project_root
    state = git_state(root)
    runtime = final.environment(settings)

    fixed = read_fixed(settings)

    manifest = fixed[0]

    inspection = assets.inspect_assets(settings)

    explanation.compare(
        inspection["data"],
        manifest["assets"]["data"],
        "선행 모델 자료 계보",
    )

    for spec in assets.ASSETS:
        current = final.get_asset(
            inspection,
            spec.name,
        )

        previous = final.get_asset(
            manifest["assets"],
            spec.name,
        )

        for key in (
            "training_run_id",
            "model_file",
            "model_fingerprint",
            "historical_integrity",
        ):
            explanation.compare(
                current[key],
                previous[key],
                f"{spec.name}: 고정 모델 {key}",
            )

    inventory = input_inventory(
        settings,
        fixed,
        inspection,
    )

    code_hashes = final.code_inventory(root)

    database, _, _ = final.data_paths(inspection)

    directory = (
        settings.results_dir
        / RESULT_DIRECTORY
        / uuid4().hex
    )

    directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    try:
        backgrounds = {}

        for spec in assets.ASSETS:
            if spec.name not in LINEAR_MODELS:
                continue

            LOGGER.info(
                "Train 설명 배경 계산: %s",
                spec.name,
            )

            operating = load_operating(
                settings,
                spec,
            )

            backgrounds[spec.name] = (
                explanation.train_background(
                    database,
                    spec.name,
                    operating,
                )
            )

        background_path = directory / "background.json"

        write_json(
            background_path,
            backgrounds,
        )

        assets.require(
            code_hashes == final.code_inventory(root),
            "설명 준비 중 실행 코드 변경",
        )

        verify_inventory(
            root,
            inventory,
        )

        plan = {
            "schema": SCHEMA,
            "version": VERSION,
            "status": "prepared",
            "plan_id": directory.name,
            "created_utc": utc_now(),
            "git": state,
            "runtime": runtime,
            "prepare_command": [
                sys.executable,
                *sys.argv,
            ],
            "code_sha256": code_hashes,
            "input_sha256": inventory,
            "fixed_assets": {
                "final_evaluation_run": (
                    FINAL_EVALUATION_RUN
                ),
                "final_manifest_sha256": FINAL_MANIFEST_HASH,
                "final_result_sha256": FINAL_RESULT_HASH,
                "season_validation_run": (
                    SEASON_VALIDATION_RUN
                ),
                "season_plan_sha256": SEASON_PLAN_HASH,
                "season_result_sha256": SEASON_RESULT_HASH,
                "model_data_run": assets.DATA_RUN,
                "model_data_sha256": assets.DATA_SHA256,
                "models": {
                    spec.name: spec.run_id
                    for spec in assets.ASSETS
                },
            },
            "background": assets.describe_file(
                background_path
            ),
            "policy": policy(),
            "test_explanation_started": False,
        }

        plan_path = directory / "manifest.json"

        write_json(
            plan_path,
            plan,
        )

        return {
            "status": "prepared",
            "plan_id": directory.name,
            "plan": str(plan_path),
            "sha256": sha256_file(plan_path),
            "input_files": len(inventory),
            "background_models": len(backgrounds),
        }

    except Exception as exc:
        write_json(
            directory / "prepare_failure.json",
            {
                "status": "failed",
                "plan_id": directory.name,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "created_utc": utc_now(),
            },
        )

        raise


def load_plan(
    settings: Settings,
    path: Path,
    expected_sha256: str,
) -> tuple[
    Path,
    JsonObject,
    JsonObject,
]:
    """현재 실행이 사전 고정한 설명 명세와 일치하는지 검사한다."""
    root = settings.project_root

    path = verify_digest(
        root,
        path,
        expected_sha256,
    )

    output_root = (
        settings.results_dir / RESULT_DIRECTORY
    ).resolve()

    assets.require(
        path.name == "manifest.json"
        and path.parent.parent == output_root,
        "설명 명세 경로 오류",
    )

    plan = assets.read_unique_json(path)

    assets.require(
        plan["schema"] == SCHEMA
        and plan["version"] == VERSION
        and plan["status"] == "prepared"
        and plan["plan_id"] == path.parent.name
        and plan["test_explanation_started"] is False,
        "설명 명세 식별 또는 상태 오류",
    )

    explanation.compare(
        plan["policy"],
        policy(),
        "고정 설명 정책",
    )

    explanation.compare(
        plan["runtime"],
        final.environment(settings),
        "설명 실행 환경",
    )

    explanation.compare(
        plan["code_sha256"],
        final.code_inventory(root),
        "설명 실행 코드 지문",
    )

    git_state(root)

    verify_inventory(
        root,
        plan["input_sha256"],
    )

    background_path = final.checked_file(
        plan["background"],
        path.parent,
    )

    backgrounds = assets.read_unique_json(
        background_path
    )

    assets.require(
        set(backgrounds) == set(LINEAR_MODELS),
        "선형 설명 배경 모델 목록 불일치",
    )

    return path, plan, backgrounds


def execute_model(
    settings: Settings,
    spec: assets.AssetSpec,
    manifest: Mapping[str, object],
    record: Mapping[str, object],
    backgrounds: Mapping[str, object],
    directory: Path,
) -> JsonObject:
    """한 모델의 전량 진단과 선정 표본 설명을 저장한다."""
    name = spec.name

    evaluation_dir = (
        fixed_paths(settings)["final_result"].parent
    )

    model_dir = evaluation_dir / name

    predictions = final.checked_file(
        record["predictions"],
        model_dir,
    )

    targets = final.checked_file(
        record["targets"],
        model_dir,
    )

    metrics_path = final.checked_file(
        record["metrics"],
        model_dir,
    )

    stored_metrics = assets.read_unique_json(metrics_path)

    assets.require(
        fingerprint(stored_metrics)
        == record["metrics_fingerprint"],
        f"{name}: 기존 평가 내부 지문 불일치",
    )

    baseline = final.load_baseline(
        manifest,
        evaluation_dir.parent,
        name,
    )

    operating = load_operating(
        settings,
        spec,
    )

    predict = final.make_predictor(
        spec,
        operating,
        baseline,
    )

    linear = (
        explanation.build_linear(
            name,
            operating,
            backgrounds[name],
        )
        if name in LINEAR_MODELS
        else None
    )

    report, samples, global_keys, reasons = (
        diagnostics.diagnose_model(
            name,
            predictions,
            targets,
            stored_metrics,
            baseline,
            record["prediction_samples"],
        )
    )

    summary = explanation.ExplanationSummary()

    max_error = 0.0
    explained = 0

    def rows():
        """선정 표본마다 저장 예측과 설명 기여를 검산한다."""
        nonlocal max_error, explained

        for key, sample in sorted(samples.items()):
            prediction = sample["prediction"]
            truth = sample["target"]

            reproduced = predict(
                prediction["input"]
            )

            for method, forecast in reproduced.items():
                error = explanation.compare(
                    forecast,
                    prediction[method],
                    f"{name}:{key}:{method}",
                )

                max_error = max(
                    max_error,
                    error,
                )

            aggregate = key in global_keys

            result = explanation.explain(
                name,
                operating,
                sample,
                linear,
                aggregate=aggregate,
            )

            max_error = max(
                max_error,
                result["max_absolute_error"],
            )

            if aggregate:
                summary.add(result)

            explained += 1

            yield {
                "sample_id": key,
                "model": name,
                "model_run_id": spec.run_id,
                "final_evaluation_run": (
                    FINAL_EVALUATION_RUN
                ),
                "global_explanation_sample": aggregate,
                "failure_case_reasons": reasons.get(key, []),
                "stored_prediction": prediction,
                "stored_target": truth,
                "explanation": result,
                "verification_status": "matched",
            }

    local_path = directory / f"{name}-local.jsonl"

    local = final.write_rows(
        local_path,
        rows(),
    )

    assets.require(
        explained
        == local["rows"]
        == len(samples)
        and summary.samples == len(global_keys),
        f"{name}: 설명 결과 분모 불일치",
    )

    report["sources"] = {
        "training_run_id": spec.run_id,
        "final_evaluation_run": FINAL_EVALUATION_RUN,
        "predictions": record["predictions"],
        "targets": record["targets"],
        "metrics": record["metrics"],
        "population": record["population"],
        "background_reference": (
            name
            if name in LINEAR_MODELS
            else "stored_model_statistics"
        ),
    }

    report["global_explanation"] = summary.report()

    report["linear_global_explanation"] = (
        linear.report()
        if linear is not None
        else None
    )

    report["verification"] = {
        "status": "matched",
        "explained_samples": explained,
        "global_samples": len(global_keys),
        "failure_count": 0,
        "max_absolute_error": max_error,
        "absolute_tolerance": ABS_TOLERANCE,
        "relative_tolerance": REL_TOLERANCE,
    }

    summary_path = directory / f"{name}-summary.json"

    write_json(
        summary_path,
        report,
    )

    return {
        "status": "completed",
        "population": report["population"],
        "explained_samples": explained,
        "global_samples": len(global_keys),
        "max_absolute_error": max_error,
        "metrics_max_absolute_error": (
            report["metrics_recalculation"][
                "max_absolute_error"
            ]
        ),
        "files": {
            "local": local,
            "summary": assets.describe_file(summary_path),
        },
    }


def checked_result(
    path: Path,
    plan_path: Path,
    plan_hash: str,
) -> JsonObject:
    """설명 결과 및 참조한 모든 성공 산출물의 지문을 검사한다."""
    path = path.resolve(strict=True)

    assets.require(
        path.name == "result.json"
        and path.parent.parent == plan_path.parent,
        "설명 결과 실행 경로 오류",
    )

    result = assets.read_unique_json(path)

    assets.require(
        result["schema"] == SCHEMA
        and result["version"] == VERSION
        and result["plan_sha256"] == plan_hash
        and result["plan_id"] == plan_path.parent.name
        and result["run_id"] == path.parent.name
        and result["status"] in {
            "completed",
            "failed_or_partial",
        }
        and set(result["models"]) == set(MODEL_NAMES),
        "설명 실행 결과 식별 오류",
    )

    records = {
        **result["models"],
        "season": result["season"],
    }

    for name, record in records.items():
        if record["status"] != "completed":
            continue

        files = record["files"]

        assets.require(
            set(files) == (
                {"summary"}
                if name == "season"
                else {"local", "summary"}
            ),
            f"{name}: 설명 산출물 목록 오류",
        )

        for kind, information in files.items():
            output = final.checked_file(
                information,
                plan_path.parent,
            )

            expected_name = (
                "season.json"
                if name == "season"
                else (
                    f"{name}-local.jsonl"
                    if kind == "local"
                    else f"{name}-summary.json"
                )
            )

            assets.require(
                output.name == expected_name,
                f"{name}: 설명 산출물 파일명 오류",
            )

    return result


def run(
    settings: Settings,
    plan_path: Path,
    plan_hash: str,
    reuse_result: Path | None = None,
) -> JsonObject:
    """고정 설명 명세로 새 실행 결과를 생성한다."""
    plan_hash = plan_hash.lower()

    plan_path, plan, backgrounds = load_plan(
        settings,
        plan_path,
        plan_hash,
    )

    manifest, result, season_plan, season_verification = (
        read_fixed(settings)
    )

    reused = (
        checked_result(
            reuse_result,
            plan_path,
            plan_hash,
        )
        if reuse_result is not None
        else None
    )

    if reused is not None:
        assets.require(
            reused.get("integrity_error") is None,
            "무결성 실패 실행의 결과 재사용 금지",
        )

    directory = plan_path.parent / uuid4().hex

    directory.mkdir(exist_ok=False)

    write_json(
        directory / "started.json",
        {
            "status": "started",
            "plan_id": plan["plan_id"],
            "plan_sha256": plan_hash,
            "run_id": directory.name,
            "created_utc": utc_now(),
        },
    )

    records = {}

    for spec in assets.ASSETS:
        name = spec.name

        previous = (
            reused["models"][name]
            if reused is not None
            else None
        )

        if (
            previous is not None
            and previous["status"] == "completed"
        ):
            records[name] = {
                **previous,
                "reused_from": str(
                    reuse_result.resolve()
                ),
            }

            LOGGER.info(
                "%s: 기존 성공 산출물 재사용",
                name,
            )

            continue

        LOGGER.info(
            "%s: 설명 및 진단 시작",
            name,
        )

        try:
            records[name] = execute_model(
                settings,
                spec,
                manifest,
                result["models"][name],
                backgrounds,
                directory,
            )

            LOGGER.info(
                "%s: 설명 및 진단 완료",
                name,
            )

        except Exception as exc:
            LOGGER.exception(
                "%s: 설명 및 진단 실패",
                name,
            )

            records[name] = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

    previous_season = (
        reused["season"]
        if reused is not None
        else None
    )

    if (
        previous_season is not None
        and previous_season["status"] == "completed"
    ):
        season_record = {
            **previous_season,
            "reused_from": str(
                reuse_result.resolve()
            ),
        }

    else:
        try:
            LOGGER.info("시즌 및 보조 분석 진단 시작")

            season_report = diagnostics.diagnose_season(
                settings.project_root,
                season_plan,
                season_verification,
            )

            path = directory / "season.json"

            write_json(
                path,
                season_report,
            )

            season_record = {
                "status": "completed",
                "files": {
                    "summary": assets.describe_file(path),
                },
            }

        except Exception as exc:
            LOGGER.exception(
                "시즌 및 보조 분석 진단 실패"
            )

            season_record = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

    integrity_error = None

    try:
        load_plan(
            settings,
            plan_path,
            plan_hash,
        )

    except Exception as exc:
        LOGGER.exception(
            "설명 종료 무결성 재검사 실패"
        )

        integrity_error = str(exc)

    completed = sum(
        record["status"] == "completed"
        for record in records.values()
    )

    status = (
        "completed"
        if (
            completed == len(MODEL_NAMES)
            and season_record["status"] == "completed"
            and integrity_error is None
        )
        else "failed_or_partial"
    )

    output = {
        "schema": SCHEMA,
        "version": VERSION,
        "status": status,
        "plan_id": plan["plan_id"],
        "plan_sha256": plan_hash,
        "run_id": directory.name,
        "finished_utc": utc_now(),
        "git": git_state(settings.project_root),
        "runtime": final.environment(settings),
        "command": [
            sys.executable,
            *sys.argv,
        ],
        "completed_models": completed,
        "expected_models": len(MODEL_NAMES),
        "models": records,
        "season": season_record,
        "integrity_error": integrity_error,
        "new_operating_models": 0,
        "test_used_for_training": False,
        "test_used_for_selection": False,
        "test_used_for_calibration": False,
    }

    result_path = directory / "result.json"

    write_json(
        result_path,
        output,
    )

    return {
        "status": status,
        "plan_id": plan["plan_id"],
        "run_id": directory.name,
        "result": str(result_path),
        "sha256": sha256_file(result_path),
        "completed_models": completed,
        "expected_models": len(MODEL_NAMES),
        "season_status": season_record["status"],
        "models": {
            name: {
                "status": record["status"],
                "population": record.get("population"),
                "explained_samples": record.get(
                    "explained_samples"
                ),
                "error": record.get("error"),
            }
            for name, record in records.items()
        },
        "integrity_error": integrity_error,
    }


def verify(
    settings: Settings,
    plan_path: Path,
    plan_hash: str,
    first_path: Path,
    second_path: Path,
) -> JsonObject:
    """독립 실행 결과와 Train 설명 배경의 재현 여부를 검증한다."""
    plan_hash = plan_hash.lower()

    plan_path, plan, backgrounds = load_plan(
        settings,
        plan_path,
        plan_hash,
    )

    first_path = first_path.resolve(strict=True)
    second_path = second_path.resolve(strict=True)

    assets.require(
        first_path != second_path,
        "동일 실행 결과를 재현 검증에 사용할 수 없습니다.",
    )

    first = checked_result(
        first_path,
        plan_path,
        plan_hash,
    )

    second = checked_result(
        second_path,
        plan_path,
        plan_hash,
    )

    assets.require(
        first["status"] == second["status"] == "completed"
        and first["integrity_error"] is None
        and second["integrity_error"] is None,
        "미완료 설명 실행은 최종 재현 검증 불가",
    )

    first_records = {
        **first["models"],
        "season": first["season"],
    }

    second_records = {
        **second["models"],
        "season": second["season"],
    }

    matched = {}

    for name, record in first_records.items():
        other = second_records[name]

        assets.require(
            record["status"]
            == other["status"]
            == "completed",
            f"{name}: 독립 설명 실행 미완료",
        )

        comparison = {}

        for kind, left in record["files"].items():
            right = other["files"][kind]

            assets.require(
                Path(left["path"]).resolve()
                != Path(right["path"]).resolve(),
                f"{name}: 동일 산출물 재사용으로 재현 검증 불가",
            )

            explanation.compare(
                left["sha256"],
                right["sha256"],
                f"{name}:{kind}: 핵심 산출물 지문",
            )

            if "rows" in left:
                explanation.compare(
                    left["rows"],
                    right["rows"],
                    f"{name}:{kind}: 설명 행 수",
                )

            comparison[kind] = {
                "sha256": left["sha256"],
                "first": left["path"],
                "second": right["path"],
            }

        matched[name] = comparison

    # 배경은 고정 명세를 단순 복사하지 않고
    # 실제 Train 자료에서 한 번 더 계산해 검증한다.
    manifest = read_fixed(settings)[0]

    database, _, _ = final.data_paths(
        manifest["assets"]
    )

    background_checks = {}

    for spec in assets.ASSETS:
        if spec.name not in LINEAR_MODELS:
            continue

        LOGGER.info(
            "Train 설명 배경 독립 재검산: %s",
            spec.name,
        )

        operating = load_operating(
            settings,
            spec,
        )

        actual = explanation.train_background(
            database,
            spec.name,
            operating,
        )

        error = explanation.compare(
            actual,
            backgrounds[spec.name],
            f"{spec.name}: Train 배경 재현",
        )

        background_checks[spec.name] = {
            "status": "matched",
            "max_absolute_error": error,
        }

    load_plan(
        settings,
        plan_path,
        plan_hash,
    )

    verification = {
        "schema": SCHEMA,
        "version": VERSION,
        "status": "completed",
        "created_utc": utc_now(),
        "plan_id": plan["plan_id"],
        "plan_sha256": plan_hash,
        "first": assets.describe_file(first_path),
        "second": assets.describe_file(second_path),
        "matched_models": len(MODEL_NAMES),
        "season_matched": True,
        "matched_files": matched,
        "background_checks": background_checks,
        "verification_command": [
            sys.executable,
            *sys.argv,
        ],
        "git": git_state(settings.project_root),
        "integrity_verified": True,
        "reproduction_failure_count": 0,
    }

    output = (
        plan_path.parent
        / f"verification-{uuid4().hex}.json"
    )

    write_json(
        output,
        verification,
    )

    return {
        "status": "completed",
        "plan_id": plan["plan_id"],
        "verification": str(output),
        "sha256": sha256_file(output),
        "matched_models": len(MODEL_NAMES),
        "season_matched": True,
        "backgrounds_matched": len(background_checks),
        "reproduction_failure_count": 0,
    }


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """준비·설명 실행·독립 재현 검증 진입점."""
    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--project-root",
        type=Path,
    )

    commands = parser.add_subparsers(
        dest="command",
        required=True,
    )

    commands.add_parser("prepare")

    execute = commands.add_parser("run")

    execute.add_argument(
        "--plan",
        type=Path,
        required=True,
    )

    execute.add_argument(
        "--sha256",
        required=True,
    )

    execute.add_argument(
        "--reuse-result",
        type=Path,
    )

    verification = commands.add_parser("verify")

    verification.add_argument(
        "--plan",
        type=Path,
        required=True,
    )

    verification.add_argument(
        "--sha256",
        required=True,
    )

    verification.add_argument(
        "--first",
        type=Path,
        required=True,
    )

    verification.add_argument(
        "--second",
        type=Path,
        required=True,
    )

    args = parser.parse_args(argv)

    try:
        settings = load_settings(
            args.project_root
        )

        logging.basicConfig(
            level=settings.log_level,
            format="%(levelname)s %(name)s: %(message)s",
        )

        if args.command == "prepare":
            result = prepare(settings)

        elif args.command == "run":
            result = run(
                settings,
                args.plan,
                args.sha256,
                args.reuse_result,
            )

        else:
            result = verify(
                settings,
                args.plan,
                args.sha256,
                args.first,
                args.second,
            )

        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        )

        return (
            0
            if result["status"] in {
                "prepared",
                "completed",
            }
            else 5
        )

    except KeyboardInterrupt:
        LOGGER.warning("사용자가 설명 실행을 중단했습니다.")
        return 3

    except OSError:
        LOGGER.exception("설명 자료 입출력 실패")
        return 4

    except Exception:
        LOGGER.exception("설명 실행 또는 검증 계약 실패")
        return 5


if __name__ == "__main__":
    raise SystemExit(main())
