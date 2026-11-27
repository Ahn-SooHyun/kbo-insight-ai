"""M5 공동 분류기의 학습·저장·검증·추론 CLI."""

import argparse
from collections.abc import Sequence
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import platform
import sqlite3
import sys
from time import perf_counter, process_time
from typing import cast
from uuid import uuid4

from . import SCHEMA, VERSION
from . import data
from . import model as m5
from ...datasets.model_data import fingerprint
from ...datasets.published_sources import (
    RUN_PATTERN,
    SourceError,
    as_row,
    read_json,
    sha256_file,
)
from ...evaluation import pitch_outcome as evaluation
from ...settings import Settings, load_settings


LOGGER = logging.getLogger(__name__)


def method_settings() -> dict[str, object]:
    """검증 성능 확인 전에 고정한 학습·평가 계약을 반환한다."""
    return {
        "pitch_types": list(m5.TYPES),
        "pitch_results": list(m5.RESULTS),
        "joint_classes": list(m5.CLASSES),
        "joint_dictionary": "verified_code_cartesian_product",
        "unseen_joint_policy": "positive_train_smoothed_probability",
        "structural_impossibility_assumed": False,
        "minimum_id_count": m5.MIN_ID_COUNT,
        "minimum_group_count": m5.MIN_GROUP_COUNT,
        "group_prior": m5.GROUP_PRIOR,
        "laplace_alpha": m5.ALPHA,
        "prior_mix": m5.PRIOR_MIX,
        "epochs": m5.EPOCHS,
        "learning_rate": m5.LEARNING_RATE,
        "learning_rate_schedule": "initial / sqrt(epoch + 1)",
        "optimizer": "online_sgd",
        "sample_policy": "all_train_rows_without_replacement_per_epoch",
        "selection": "joint_log_loss_then_brier_then_simplicity",
        "candidates": list(m5.METHODS),
        "calibration_bins": evaluation.CALIBRATION_BINS,
        "rare_support": evaluation.RARE_SUPPORT,
        "calibration_fit": False,
        "brier": "mean_sum_squared_class_errors",
        "unknown_player_policy": "role_specific_other",
        "unknown_non_id_policy": "no_new_coefficient",
        "first_pitch_previous_policy": "both_null",
        "missing_previous_policy": "preserve_null",
        "preprocessing_fit": "train_only",
        "final_test_used": False,
    }


def source_hashes() -> dict[str, str]:
    """실행에 관여하는 주요 코드와 의존성 정의의 지문을 기록한다."""
    root = Path(__file__).resolve().parents[3]

    files = (
        "src/models/pitch_outcome/__init__.py",
        "src/models/pitch_outcome/model.py",
        "src/models/pitch_outcome/data.py",
        "src/models/pitch_outcome/__main__.py",
        "src/evaluation/pitch_outcome.py",
        "src/datasets/model_data.py",
        "src/datasets/model_events.py",
        "src/datasets/published_sources.py",
        "src/validation/standard_data.py",
        "src/settings.py",
        "pyproject.toml",
        "uv.lock",
    )

    return {
        name: sha256_file(root / name)
        for name in files
    }


def peak_memory_bytes() -> int | None:
    """지원되는 OS에서 프로세스 최대 상주 메모리를 조회한다."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)

            kernel.GetCurrentProcess.argtypes = []
            kernel.GetCurrentProcess.restype = wintypes.HANDLE

            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(ProcessMemoryCounters),
                wintypes.DWORD,
            ]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)

            success = psapi.GetProcessMemoryInfo(
                kernel.GetCurrentProcess(),
                ctypes.byref(counters),
                counters.cb,
            )

            if success:
                return int(counters.PeakWorkingSetSize)

            LOGGER.warning("M5 Windows 최대 메모리 조회 실패")
            return None

        if sys.platform.startswith("linux") or sys.platform == "darwin":
            import resource

            peak = resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss

            return int(
                peak if sys.platform == "darwin" else peak * 1024
            )

    except (OSError, AttributeError, ImportError, ValueError):
        LOGGER.warning("M5 최대 메모리 조회를 지원하지 않습니다.")

    return None


def save_json(path: Path, content: object) -> None:
    """기존 자산을 덮어쓰지 않고 UTF-8 JSON으로 저장한다."""
    with path.open(
        "x", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(
            content,
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")


def load_payload(path: Path) -> dict[str, object]:
    """저장 모델의 지문과 버전·입력 계약·공동 사전을 검사한다."""
    envelope = as_row(read_json(path))

    m5.require(
        set(envelope) == {"payload", "sha256"},
        "모델 외부 자산 구조 오류",
    )

    payload = as_row(envelope["payload"])

    m5.require(
        envelope["sha256"] == fingerprint(payload)
        and payload["schema"] == SCHEMA
        and payload["version"] == VERSION
        and payload["input_columns"] == list(m5.INPUT_COLUMNS)
        and payload["allowed_input_columns"] == list(m5.ALL_COLUMNS)
        and payload["method_settings"] == method_settings()
        and payload["operational_predictor_count"] == 1,
        "모델 자산 지문·버전·계약 불일치",
    )

    run_id = payload["training_run_id"]

    m5.require(
        isinstance(run_id, str)
        and RUN_PATTERN.fullmatch(run_id) is not None,
        "학습 실행 ID 오류",
    )

    m5.integer(
        payload["configured_seed"],
        "configured_seed",
        0,
        2**32 - 1,
    )

    model = cast(m5.Model, as_row(payload["model"]))

    m5.validate_model(model)

    m5.require(
        model["kind"] == payload["selected_model"],
        "선정 모델과 운영 모델 불일치",
    )

    return payload


def load_runtime(path: Path) -> m5.Runtime:
    """저장 자산을 검사하고 운영 추론기를 복원한다."""
    payload = load_payload(path)
    model = cast(m5.Model, as_row(payload["model"]))

    return m5.Runtime(model)


def build_examples(
    model: m5.Model,
    examples: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    """실제 경계 사례에서 현재 정답 없는 독립 추론을 기록한다."""
    runtime = m5.Runtime(model)

    result = {
        key: {
            **example,
            "prediction": m5.predict(
                runtime, as_row(example["input"])
            ),
        }
        for key, example in examples.items()
    }

    m5.require(
        "train:first_pitch" in result
        and "validation:first_pitch" in result,
        "실제 첫 공 재현 사례 누락",
    )

    m5.require(
        "validation:previous_available" in result,
        "실제 직전 투구 재현 사례 누락",
    )

    return result


def verify_examples(
    runtime: m5.Runtime,
    examples: dict[str, object],
) -> None:
    """모든 저장 사례의 공동·주변 확률을 다시 계산한다."""
    m5.require(bool(examples), "재현 사례 누락")

    for key, value in examples.items():
        example = as_row(value)

        actual = m5.predict(
            runtime, as_row(example["input"])
        )

        m5.require(
            actual == example["prediction"],
            f"저장 전후 예측 불일치: {key}",
        )


def brief_metrics(
    evaluations: dict[str, dict[str, object]],
) -> dict[str, object]:
    """터미널 출력에 필요한 후보별 주요 지표만 반환한다."""
    result: dict[str, object] = {}

    for name, candidate in evaluations.items():
        result[name] = {
            axis: {
                metric: as_row(candidate[axis])[metric]
                for metric in (
                    "samples",
                    "log_loss",
                    "brier",
                    "top_label_ece",
                    "train_observed_support_rate",
                )
            }
            for axis in evaluation.AXES
        }

    return result


def train(
    settings: Settings,
    run_id: str,
) -> dict[str, object]:
    """인수 자료를 감사하고 비교 후보 중 운영 분류기 하나를 저장한다."""
    started = perf_counter()
    cpu_started = process_time()

    initial_hashes = source_hashes()
    configured_method = method_settings()

    dataset = data.load(settings, run_id)
    data_seconds = perf_counter() - started

    train_rows = dataset.rows["train"]
    validation_rows = dataset.rows["validation"]

    models: dict[str, m5.Model] = {}
    evaluations: dict[str, dict[str, object]] = {}
    timings: dict[str, dict[str, float]] = {}

    for kind in m5.METHODS:
        fit_started = perf_counter()

        if kind == "frequency":
            candidate = m5.fit_frequency(train_rows)

        elif kind == "conditional_frequency":
            candidate = m5.fit_frequency(
                train_rows, conditional=True
            )

        else:
            candidate = m5.fit_logistic(
                train_rows, settings.seed
            )

        fit_seconds = perf_counter() - fit_started
        evaluation_started = perf_counter()

        score = evaluation.evaluate(
            candidate, validation_rows
        )

        models[kind] = candidate
        evaluations[kind] = score

        timings[kind] = {
            "fit_seconds": fit_seconds,
            "evaluation_seconds": (
                perf_counter() - evaluation_started
            ),
        }

        LOGGER.info(
            "M5 candidate=%s joint_log_loss=%.8f joint_brier=%.8f",
            kind,
            as_row(score["joint"])["log_loss"],
            as_row(score["joint"])["brier"],
        )

    selected = evaluation.select(evaluations)
    model = models[selected]

    examples = build_examples(model, dataset.examples)

    primary_case = "validation:previous_available"
    example = examples[primary_case]

    training_run = uuid4().hex
    created_at = datetime.now(timezone.utc).isoformat()

    model_directory = settings.models_dir / "m5" / training_run
    result_directory = settings.results_dir / "m5" / training_run

    paths = {
        "model": str(model_directory / "model.json"),
        "evaluation": str(result_directory / "evaluation.json"),
        "input": str(result_directory / "example_input.json"),
        "prediction": str(
            result_directory / "example_prediction.json"
        ),
    }

    m5.require(
        source_hashes() == initial_hashes
        and method_settings() == configured_method,
        "학습 중 소스 또는 고정 실험 설정 변경",
    )

    provenance = dataset.provenance

    m5.require(
        sha256_file(Path(provenance["database"]))
        == provenance["database_sha256"]
        and sha256_file(Path(provenance["standard_database"]))
        == provenance["standard_database_sha256"],
        "학습 중 인수 데이터 변경",
    )

    resources = {
        "device": "cpu",
        "data_audit_seconds": data_seconds,
        "candidate_seconds": timings,
        "wall_seconds_before_save": perf_counter() - started,
        "process_cpu_seconds_before_save": (
            process_time() - cpu_started
        ),
        "peak_resident_memory_bytes_before_save": peak_memory_bytes(),
    }

    report = {
        "status": "completed",
        "training_run_id": training_run,
        "created_at_utc": created_at,
        "model_schema": SCHEMA,
        "model_version": VERSION,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "configured_seed": settings.seed,
        "selected_model": selected,
        "operational_predictor_count": 1,
        "method_settings": configured_method,
        "data": provenance,
        "training_joint_counts": dict(
            zip(m5.CLASSES, model["counts"])
        ),
        "candidate_evaluations": evaluations,
        "baseline_comparison": evaluation.compare(
            selected, evaluations
        ),
        "examples": examples,
        "primary_example": primary_case,
        "source_sha256": initial_hashes,
        "paths": paths,
        "resources": resources,
        "final_test_evaluated": False,
        "validation_used_for_selection": True,
    }

    payload = {
        "schema": SCHEMA,
        "version": VERSION,
        "training_run_id": training_run,
        "created_at_utc": created_at,
        "python_version": platform.python_version(),
        "configured_seed": settings.seed,
        "selected_model": selected,
        "operational_predictor_count": 1,
        "input_columns": list(m5.INPUT_COLUMNS),
        "allowed_input_columns": list(m5.ALL_COLUMNS),
        "method_settings": configured_method,
        "model": model,
        "data": provenance,
        "source_sha256": initial_hashes,
        "evaluation_fingerprint": fingerprint(report),
    }

    model_directory.mkdir(parents=True, exist_ok=False)
    result_directory.mkdir(parents=True, exist_ok=False)

    pending = model_directory / "model.pending.json"

    save_json(
        pending,
        {
            "payload": payload,
            "sha256": fingerprint(payload),
        },
    )

    restored = load_runtime(pending)

    # 저장 자산을 다시 읽어 모든 경계 사례의 확률을 비교한다.
    verify_examples(restored, examples)

    save_json(Path(paths["input"]), example["input"])
    save_json(Path(paths["prediction"]), example["prediction"])
    save_json(Path(paths["evaluation"]), report)

    # 검증된 자산에만 최종 운영 파일명을 부여한다.
    pending.rename(Path(paths["model"]))

    return {
        "status": "completed",
        "training_run_id": training_run,
        "model_version": VERSION,
        "selected_model": selected,
        "operational_predictor_count": 1,
        "configured_seed": settings.seed,
        "counts": provenance["counts"],
        "source_exclusions": provenance["source_exclusions"],
        "non_regular_excluded": provenance["non_regular_excluded"],
        "metrics": brief_metrics(evaluations),
        "baseline_comparison": report["baseline_comparison"],
        "resources": resources,
        "example_cases": sorted(examples),
        "paths": paths,
        "model_sha256": sha256_file(Path(paths["model"])),
        "evaluation_sha256": sha256_file(
            Path(paths["evaluation"])
        ),
        "saved_prediction_match": True,
        "final_test_evaluated": False,
    }


def verify_run(
    asset: Path,
    evaluation_path: Path,
) -> dict[str, object]:
    """원천 DB 없이 모델·평가·예시의 연결과 재현을 검사한다."""
    m5.require(asset.name == "model.json", "확정 모델이 필요합니다.")

    payload = load_payload(asset)
    report = as_row(read_json(evaluation_path))

    m5.require(
        fingerprint(report) == payload["evaluation_fingerprint"]
        and report["status"] == "completed"
        and report["training_run_id"] == payload["training_run_id"]
        and report["model_schema"] == SCHEMA
        and report["model_version"] == VERSION
        and report["selected_model"] == payload["selected_model"]
        and report["configured_seed"] == payload["configured_seed"]
        and report["operational_predictor_count"] == 1
        and report["method_settings"] == payload["method_settings"]
        and report["data"] == payload["data"]
        and report["final_test_evaluated"] is False
        and report["validation_used_for_selection"] is True,
        "모델·평가 자료 연결 계약 불일치",
    )

    m5.require(
        report["source_sha256"] == payload["source_sha256"]
        and payload["source_sha256"] == source_hashes(),
        "현재 소스와 학습 소스의 지문 불일치",
    )

    provenance = as_row(report["data"])

    m5.require(
        provenance["test_target_values_read"] is False,
        "최종 Test 격리 위반",
    )

    counts = as_row(provenance["counts"])
    train_count = as_row(counts["train"])["samples"]
    validation_count = as_row(counts["validation"])["samples"]

    model = cast(m5.Model, as_row(payload["model"]))

    m5.require(
        sum(model["counts"]) == train_count
        and report["training_joint_counts"]
        == dict(zip(m5.CLASSES, model["counts"])),
        "저장 모델의 Train 공동 빈도 불일치",
    )

    candidates = {
        name: as_row(value)
        for name, value in as_row(
            report["candidate_evaluations"]
        ).items()
    }

    m5.require(
        evaluation.select(candidates) == payload["selected_model"],
        "후보 선정 기준 불일치",
    )

    for candidate in candidates.values():
        for axis in evaluation.AXES:
            metrics = as_row(candidate[axis])
            classes = as_row(metrics["classes"])
            support_groups = as_row(
                metrics["train_support_groups"]
            )

            m5.require(
                metrics["samples"] == validation_count
                and sum(
                    item["count"]
                    for item in metrics["top_label_calibration"]
                ) == validation_count
                and sum(
                    as_row(item)["validation_support"]
                    for item in classes.values()
                ) == validation_count
                and sum(
                    as_row(item)["count"]
                    for item in support_groups.values()
                ) == validation_count,
                "후보·범주·보정 평가 분모 불일치",
            )

            for class_metrics in classes.values():
                m5.require(
                    sum(
                        item["count"]
                        for item in as_row(
                            class_metrics
                        )["calibration"]
                    ) == validation_count,
                    "클래스별 보정 분모 불일치",
                )

    examples = as_row(report["examples"])
    runtime = m5.Runtime(model)

    verify_examples(runtime, examples)

    primary = as_row(examples[report["primary_example"]])

    stored_input = read_json(
        evaluation_path.parent / "example_input.json"
    )
    stored_prediction = read_json(
        evaluation_path.parent / "example_prediction.json"
    )

    m5.require(
        stored_input == primary["input"]
        and stored_prediction == primary["prediction"]
        and m5.predict(
            runtime, as_row(stored_input)
        ) == stored_prediction,
        "독립 추론 파일과 평가 사례 불일치",
    )

    return {
        "status": "verified",
        "training_run_id": payload["training_run_id"],
        "selected_model": payload["selected_model"],
        "operational_predictor_count": 1,
        "model_sha256": sha256_file(asset),
        "evaluation_sha256": sha256_file(evaluation_path),
        "verified_example_cases": sorted(examples),
        "validation_samples": validation_count,
        "reproduce_equal": True,
        "final_test_evaluated": False,
    }


def parser() -> argparse.ArgumentParser:
    """사용자가 호출하는 학습·추론·검증 진입점을 구성한다."""
    result = argparse.ArgumentParser(
        description="M5 다음 투구 구종·결과 공동 분류"
    )

    result.add_argument("--project-root", type=Path)

    commands = result.add_subparsers(
        dest="command", required=True
    )

    train_command = commands.add_parser("train")
    train_command.add_argument("--run-id", required=True)

    predict_command = commands.add_parser("predict")
    predict_command.add_argument(
        "--asset", type=Path, required=True
    )
    predict_command.add_argument(
        "--input", type=Path, required=True
    )

    verify_command = commands.add_parser("verify")
    verify_command.add_argument(
        "--asset", type=Path, required=True
    )
    verify_command.add_argument(
        "--evaluation", type=Path, required=True
    )

    return result


def main(argv: Sequence[str] | None = None) -> int:
    """정상 결과는 JSON, 운영 로그와 오류는 stderr로 출력한다."""
    args = parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        if args.command == "train":
            settings = load_settings(args.project_root)
            logging.getLogger().setLevel(settings.log_level)
            result = train(settings, args.run_id)

        elif args.command == "predict":
            result = m5.predict(
                load_runtime(args.asset),
                as_row(read_json(args.input)),
            )

        else:
            result = verify_run(args.asset, args.evaluation)

        print(json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ))

        return 0

    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        OverflowError,
        sqlite3.Error,
        SourceError,
    ) as error:
        LOGGER.error("M5 실행 실패: %s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())