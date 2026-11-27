"""M5 학습·시간순 개발 평가·자산 검증 및 독립 JSON 비교 CLI."""

import argparse
from collections.abc import Mapping, Sequence
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
from . import data, development
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
    """개발 결과를 확인하기 전에 고정한 학습·평가 설정."""
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
        "regularized_l2": development.L2,
        "regularized_bias_penalty": False,
        "regularized_lazy_decay": True,
        "regularized_average_epochs": development.AVERAGE_EPOCHS,
        "temperature_candidates": list(development.TEMPERATURES),
        "temperature_method": "raw_logits_divided_before_prior_mix",
        "temperature_asset_policy": "fold_into_logistic_coefficients",
        "temperature_fit_period": "calibration",
        "temperature_fit_metric": "joint_log_loss",
        "temperature_tie_break": "distance_to_one_then_temperature",
        "calibration_end_exclusive": development.CALIBRATION_END,
        "selection_end_exclusive": development.SELECTION_END,
        "selection_period": "selection",
        "reporting_period": "reporting",
        "selection": "joint_log_loss_brier_top_ece_macro_ece_then_order",
        "selection_baseline_gate": list(
            development.PRIMARY_METRICS
        ),
        "selection_calibration_gate": list(
            development.CALIBRATION_METRICS
        ),
        "selection_tolerance": development.SELECTION_TOLERANCE,
        "candidates": list(development.CANDIDATES),
        "calibration_bins": evaluation.CALIBRATION_BINS,
        "rare_support": evaluation.RARE_SUPPORT,
        "calibration_fit": True,
        "brier": "mean_sum_squared_class_errors",
        "unknown_player_policy": "role_specific_other",
        "unknown_non_id_policy": "no_new_coefficient",
        "first_pitch_previous_policy": "both_null",
        "missing_previous_policy": "preserve_null",
        "preprocessing_fit": "train_only",
        "full_validation_role": "development_summary_not_final_test",
        "reporting_previously_observed_in_v1": True,
        "final_test_used": False,
    }


def source_hashes() -> dict[str, str]:
    """실행에 사용한 모델·평가·자료 계약 코드의 지문."""
    root = Path(__file__).resolve().parents[3]

    files = (
        "src/models/pitch_outcome/__init__.py",
        "src/models/pitch_outcome/model.py",
        "src/models/pitch_outcome/data.py",
        "src/models/pitch_outcome/development.py",
        "src/models/pitch_outcome/__main__.py",
        "src/evaluation/pitch_outcome.py",
        "src/datasets/model_data.py",
        "src/datasets/model_events.py",
        "src/datasets/model_players.py",
        "src/datasets/published_sources.py",
        "src/validation/standard_data.py",
        "src/validation/standard_quality.py",
        "src/settings.py",
        "kbo.toml",
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

            kernel = ctypes.WinDLL(
                "kernel32", use_last_error=True
            )

            psapi = ctypes.WinDLL(
                "psapi", use_last_error=True
            )

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
        LOGGER.warning(
            "M5 최대 메모리 조회를 지원하지 않습니다."
        )

    return None


def save_json(path: Path, content: object) -> None:
    """기존 자산을 덮어쓰지 않고 UTF-8 JSON을 저장한다."""
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


def candidate_kind(candidate: str) -> m5.ModelKind:
    """개발 후보 이름과 실제 운영 분류기 종류를 구분한다."""
    m5.require(
        candidate in development.CANDIDATES,
        "지원하지 않는 M5 후보",
    )

    if candidate == "frequency":
        return "frequency"

    if candidate == "conditional_frequency":
        return "conditional_frequency"

    return "logistic"


def selected_temperature(
    candidate: str,
    calibrations: Mapping[str, object],
) -> float | None:
    """선정 후보의 실제 적용 온도를 반환한다."""
    parent = development.CALIBRATED_PARENTS.get(candidate)

    if parent is not None:
        return float(
            as_row(calibrations[parent])["temperature"]
        )

    if candidate_kind(candidate) == "logistic":
        return 1.0

    return None


def load_payload(
    path: Path,
    *,
    training_asset: bool = False,
) -> dict[str, object]:
    """자산의 지문·버전·분류기 및 운영 메타데이터를 검증한다."""
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

    model = cast(
        m5.Model,
        as_row(payload["model"]),
    )

    m5.validate_model(model)

    m5.require(
        model["kind"] == payload["selected_model"],
        "저장된 분류기 종류 불일치",
    )

    # 저수준 합성 자산 검사는 기존 테스트 계약을 유지한다.
    # 실제 CLI 추론 및 verify에서는 학습 메타데이터를 필수로 요구한다.
    if training_asset:
        required = {
            "selected_candidate",
            "selected_temperature",
            "python_version",
            "data",
            "source_sha256",
            "evaluation_fingerprint",
            "development_periods",
            "calibration_fits",
        }

        m5.require(
            required <= set(payload),
            "실제 학습 자산의 메타데이터 누락",
        )

    if "selected_candidate" in payload:
        candidate = payload["selected_candidate"]

        m5.require(
            isinstance(candidate, str)
            and candidate_kind(candidate) == model["kind"],
            "선정 후보와 운영 분류기 종류 불일치",
        )

        calibrations = as_row(payload["calibration_fits"])

        m5.require(
            payload["selected_temperature"]
            == selected_temperature(candidate, calibrations),
            "저장된 선정 후보의 적용 온도 불일치",
        )

    return payload


def load_runtime(
    path: Path,
    *,
    training_asset: bool = False,
) -> m5.Runtime:
    """검증된 JSON 모델에서 기존 단일 운영 추론기를 복원한다."""
    payload = load_payload(
        path,
        training_asset=training_asset,
    )

    model = cast(
        m5.Model,
        as_row(payload["model"]),
    )

    return m5.Runtime(model)


def verify_prediction(
    raw: object,
) -> dict[str, object]:
    """독립 JSON 출력의 공동 사전과 주변 합을 검증한다."""
    prediction = as_row(raw)

    m5.require(
        set(prediction) == {
            "model_kind",
            "joint",
            "pitch_type",
            "pitch_result",
        }
        and prediction["model_kind"] in m5.METHODS,
        "M5 예측 출력 구조 오류",
    )

    vectors: dict[str, list[float]] = {}

    for axis, labels in development.AXIS_LABELS.items():
        values = as_row(prediction[axis])

        m5.require(
            set(values) == set(labels),
            f"{axis}: 예측 범주 사전 불일치",
        )

        probabilities = [
            values[label] for label in labels
        ]

        m5.require(
            all(
                type(value) in (int, float)
                for value in probabilities
            ),
            f"{axis}: 예측 확률 타입 오류",
        )

        m5.check_probabilities(probabilities)
        vectors[axis] = probabilities

    pitch_types, results = m5.marginalize(
        vectors["joint"]
    )

    m5.require(
        vectors["pitch_type"] == pitch_types
        and vectors["pitch_result"] == results,
        "공동 확률과 주변 확률 합산 불일치",
    )

    return prediction


def compare_predictions(
    actual: object,
    expected: object,
) -> dict[str, object]:
    """두 독립 JSON 결과를 구조·확률 및 실제 값까지 비교한다."""
    actual_prediction = verify_prediction(actual)
    expected_prediction = verify_prediction(expected)

    m5.require(
        actual_prediction == expected_prediction,
        "독립 추론 JSON과 저장 예측값 불일치",
    )

    return {
        "status": "verified",
        "reproduce_equal": True,
        "joint_classes": m5.N_CLASSES,
        "pitch_type_classes": len(m5.TYPES),
        "pitch_result_classes": len(m5.RESULTS),
    }


def build_examples(
    model: m5.Model,
    examples: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    """실제 첫 공·이전 공·중간 결측의 정답 없는 추론 사례."""
    runtime = m5.Runtime(model)

    result = {
        key: {
            **example,
            "prediction": m5.predict(
                runtime,
                as_row(example["input"]),
            ),
        }
        for key, example in examples.items()
    }

    required = {
        "train:first_pitch",
        "train:previous_target_invalid",
        "validation:first_pitch",
        "validation:previous_available",
    }

    m5.require(
        required <= set(result),
        "인수 자료의 필수 투구 경계 사례 누락",
    )

    return result


def verify_examples(
    runtime: m5.Runtime,
    examples: Mapping[str, object],
) -> None:
    """저장된 모든 실제 투구 사례의 동일 확률을 재현한다."""
    m5.require(bool(examples), "재현 사례가 없습니다.")

    for key, raw in examples.items():
        example = as_row(raw)

        actual = m5.predict(
            runtime,
            as_row(example["input"]),
        )

        try:
            compare_predictions(
                actual,
                example["prediction"],
            )

        except ValueError as error:
            raise ValueError(
                f"저장 예시 재현 실패: {key}"
            ) from error


def validate_periods(
    periods: Mapping[str, object],
    validation_summary: Mapping[str, object],
) -> dict[str, int]:
    """분할 구간·전체 검증 표본·경기 수와 날짜 경계를 대조한다."""
    m5.require(
        set(periods) == set(development.PERIODS),
        "시간순 개발 구간 누락",
    )

    counts: dict[str, int] = {}
    games = 0
    first_dates: list[str] = []
    last_dates: list[str] = []

    for name in development.PERIODS:
        item = as_row(periods[name])

        counts[name] = m5.integer(
            item["samples"], "period_samples", 1
        )

        games += m5.integer(
            item["games"], "period_games", 1
        )

        first = item["first_date"]
        last = item["last_date"]

        m5.require(
            isinstance(first, str)
            and isinstance(last, str)
            and first <= last
            and development.period_for(first) == name
            and development.period_for(last) == name,
            "개발 구간의 날짜 경계 불일치",
        )

        first_dates.append(first)
        last_dates.append(last)

    m5.require(
        sum(counts.values()) == validation_summary["samples"]
        and games == validation_summary["games"]
        and min(first_dates) == validation_summary["first_date"]
        and max(last_dates) == validation_summary["last_date"],
        "시간순 개발 구간과 전체 Validation 분모 불일치",
    )

    return counts


def fit_candidates(
    train_rows: Sequence[m5.Sample],
    calibration_rows: Sequence[m5.Sample],
    seed: int,
) -> tuple[
    dict[str, m5.Model],
    dict[str, object],
    dict[str, float],
]:
    """Train의 분류기 적합과 별도 구간의 온도 보정을 수행한다."""
    models: dict[str, m5.Model] = {}
    calibrations: dict[str, object] = {}
    timings: dict[str, float] = {}

    for name in development.FREQUENCIES:
        started = perf_counter()

        models[name] = m5.fit_frequency(
            train_rows,
            conditional=name == "conditional_frequency",
        )

        timings[name] = perf_counter() - started

    for name in ("logistic", "regularized"):
        started = perf_counter()

        if name == "logistic":
            model = m5.fit_logistic(train_rows, seed)
        else:
            model = development.fit_regularized(
                train_rows, seed
            )

        models[name] = model
        timings[name] = perf_counter() - started

        started = perf_counter()

        fitted = development.fit_temperature(
            model,
            calibration_rows,
        )

        calibrations[name] = fitted

        calibrated_name = f"{name}_temperature"

        models[calibrated_name] = development.scaled_model(
            model,
            fitted["temperature"],
        )

        timings[calibrated_name] = (
            perf_counter() - started
        )

    m5.require(
        set(models) == set(development.CANDIDATES)
        and set(calibrations) == {"logistic", "regularized"},
        "후보 또는 보정 자료 누락",
    )

    return models, calibrations, timings


def evaluate_candidates(
    models: Mapping[str, m5.Model],
    rows: Sequence[m5.Sample],
    period: str,
) -> tuple[
    dict[str, dict[str, object]],
    dict[str, float],
]:
    """동일 기간·표본에서 기존의 공동 및 주변 평가를 수행한다."""
    results: dict[str, dict[str, object]] = {}
    timings: dict[str, float] = {}

    reference = models["frequency"]["counts"]

    for name in development.CANDIDATES:
        started = perf_counter()
        model = models[name]

        m5.require(
            model["counts"] == reference,
            "후보마다 서로 다른 Train 공동 빈도를 사용했습니다.",
        )

        score = evaluation.evaluate(model, rows)

        results[name] = score
        timings[name] = perf_counter() - started

        LOGGER.info(
            "M5 period=%s candidate=%s joint_log_loss=%.8f "
            "joint_brier=%.8f joint_top_ece=%.8f",
            period,
            name,
            development.metric_value(
                score, "joint", "log_loss"
            ),
            development.metric_value(
                score, "joint", "brier"
            ),
            development.metric_value(
                score, "joint", "top_label_ece"
            ),
        )

    development.validate_evaluations(
        results,
        reference,
        len(rows),
    )

    return results, timings


def train(
    settings: Settings,
    run_id: str,
) -> dict[str, object]:
    """데이터 감사부터 선택·추가 평가·저장까지 일괄 수행한다."""
    started = perf_counter()
    cpu_started = process_time()

    initial_hashes = source_hashes()
    configured_method = method_settings()

    dataset = data.load(settings, run_id)
    partition = development.partition_from_source(dataset)

    data_seconds = perf_counter() - started

    train_rows = dataset.rows["train"]
    validation_rows = dataset.rows["validation"]

    provenance = dataset.provenance
    counts = as_row(provenance["counts"])

    validate_periods(
        partition.summary,
        as_row(counts["validation"]),
    )

    models, calibrations, fit_times = fit_candidates(
        train_rows,
        partition.rows["calibration"],
        settings.seed,
    )

    # 선정 구간 외의 성능은 운영 후보를 고르기 전에 계산하지 않는다.
    selection_results, selection_times = evaluate_candidates(
        models,
        partition.rows["selection"],
        "selection",
    )

    selection = development.select_candidates(
        selection_results
    )

    selected = str(selection["selected_candidate"])
    model = models[selected]

    temperature = selected_temperature(
        selected,
        calibrations,
    )

    LOGGER.info(
        "M5 selected_candidate=%s temperature=%s",
        selected,
        temperature,
    )

    # 운영 후보 확정 후에만 선정 미사용 구간과 전체 개발 지표를 계산한다.
    reporting_results, reporting_times = evaluate_candidates(
        models,
        partition.rows["reporting"],
        "reporting",
    )

    full_results, full_times = evaluate_candidates(
        models,
        validation_rows,
        "full_validation",
    )

    evaluations = {
        "selection": selection_results,
        "reporting": reporting_results,
        "full_validation": full_results,
    }

    comparisons = {
        name: evaluation.compare(selected, results)
        for name, results in evaluations.items()
    }

    support = {
        name: development.support_summary(results[selected])
        for name, results in evaluations.items()
    }

    examples = build_examples(
        model,
        dataset.examples,
    )

    primary_case = "validation:previous_available"
    example = examples[primary_case]

    training_run = uuid4().hex
    created_at = datetime.now(timezone.utc).isoformat()

    model_directory = (
        settings.models_dir / "m5" / training_run
    )

    result_directory = (
        settings.results_dir / "m5" / training_run
    )

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

    m5.require(
        sha256_file(Path(str(provenance["database"])))
        == provenance["database_sha256"]
        and sha256_file(Path(str(provenance["standard_database"])))
        == provenance["standard_database_sha256"],
        "학습 중 인수 자료 변경",
    )

    resources = {
        "device": "cpu",
        "data_audit_and_partition_seconds": data_seconds,
        "fit_seconds": fit_times,
        "selection_evaluation_seconds": selection_times,
        "reporting_evaluation_seconds": reporting_times,
        "full_validation_evaluation_seconds": full_times,
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
        "selected_model": model["kind"],
        "selected_candidate": selected,
        "selected_temperature": temperature,
        "operational_predictor_count": 1,
        "method_settings": configured_method,
        "data": provenance,
        "development_periods": partition.summary,
        "calibration_fits": calibrations,
        "selection": selection,
        "training_joint_counts": dict(
            zip(m5.CLASSES, model["counts"])
        ),
        "selection_evaluations": selection_results,
        "reporting_evaluations": reporting_results,
        "candidate_evaluations": full_results,
        "baseline_comparison": comparisons,
        "selected_support": support,
        "examples": examples,
        "primary_example": primary_case,
        "source_sha256": initial_hashes,
        "paths": paths,
        "resources": resources,
        "validation_used_for_selection": True,
        "reporting_used_for_selection": False,
        "reporting_previously_observed_in_v1": True,
        "final_test_evaluated": False,
    }

    payload = {
        "schema": SCHEMA,
        "version": VERSION,
        "training_run_id": training_run,
        "created_at_utc": created_at,
        "python_version": platform.python_version(),
        "configured_seed": settings.seed,
        "selected_model": model["kind"],
        "selected_candidate": selected,
        "selected_temperature": temperature,
        "operational_predictor_count": 1,
        "input_columns": list(m5.INPUT_COLUMNS),
        "allowed_input_columns": list(m5.ALL_COLUMNS),
        "method_settings": configured_method,
        "model": model,
        "data": provenance,
        "development_periods": partition.summary,
        "calibration_fits": calibrations,
        "source_sha256": initial_hashes,
        "evaluation_fingerprint": fingerprint(report),
    }

    model_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    result_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    pending = model_directory / "model.pending.json"

    save_json(
        pending,
        {
            "payload": payload,
            "sha256": fingerprint(payload),
        },
    )

    restored = load_runtime(
        pending,
        training_asset=True,
    )

    verify_examples(restored, examples)

    save_json(
        Path(paths["input"]),
        example["input"],
    )

    save_json(
        Path(paths["prediction"]),
        example["prediction"],
    )

    save_json(
        Path(paths["evaluation"]),
        report,
    )

    # 저장 및 복원을 통과한 경우에만 최종 운영 파일명을 부여한다.
    pending.rename(Path(paths["model"]))

    return {
        "status": "completed",
        "training_run_id": training_run,
        "model_schema": SCHEMA,
        "model_version": VERSION,
        "selected_model": model["kind"],
        "selected_candidate": selected,
        "selected_temperature": temperature,
        "operational_predictor_count": 1,
        "configured_seed": settings.seed,
        "counts": provenance["counts"],
        "source_exclusions": provenance["source_exclusions"],
        "non_regular_excluded": provenance["non_regular_excluded"],
        "development_periods": partition.summary,
        "calibration_fits": calibrations,
        "selection": selection,
        "metrics": development.brief_metrics(full_results),
        "selection_metrics": development.brief_metrics(
            selection_results
        ),
        "reporting_metrics": development.brief_metrics(
            reporting_results
        ),
        "baseline_comparison": comparisons,
        "selected_support": support,
        "resources": resources,
        "example_cases": sorted(examples),
        "paths": paths,
        "model_sha256": sha256_file(Path(paths["model"])),
        "evaluation_sha256": sha256_file(
            Path(paths["evaluation"])
        ),
        "saved_prediction_match": True,
        "reporting_used_for_selection": False,
        "final_test_evaluated": False,
    }


def read_evaluations(
    raw: object,
) -> dict[str, dict[str, object]]:
    """평가 JSON의 후보별 객체를 복원한다."""
    return {
        name: as_row(value)
        for name, value in as_row(raw).items()
    }


def verify_run(
    asset: Path,
    evaluation_path: Path,
) -> dict[str, object]:
    """저장 모델·보정·평가·선정·예측 결과의 연결을 검증한다."""
    m5.require(
        asset.name == "model.json",
        "확정된 운영 모델이 필요합니다.",
    )

    payload = load_payload(
        asset,
        training_asset=True,
    )

    report = as_row(read_json(evaluation_path))

    m5.require(
        fingerprint(report) == payload["evaluation_fingerprint"]
        and report["status"] == "completed"
        and report["training_run_id"] == payload["training_run_id"]
        and report["model_schema"] == SCHEMA
        and report["model_version"] == VERSION
        and report["operational_predictor_count"] == 1
        and report["final_test_evaluated"] is False
        and report["validation_used_for_selection"] is True
        and report["reporting_used_for_selection"] is False
        and report["reporting_previously_observed_in_v1"] is True,
        "모델·평가 연결 또는 시간순 평가 계약 불일치",
    )

    for name in (
        "python_version",
        "configured_seed",
        "selected_model",
        "selected_candidate",
        "selected_temperature",
        "method_settings",
        "data",
        "development_periods",
        "calibration_fits",
        "source_sha256",
    ):
        m5.require(
            report[name] == payload[name],
            f"모델·평가 메타데이터 불일치: {name}",
        )

    m5.require(
        payload["source_sha256"] == source_hashes(),
        "현재 소스와 학습 소스의 SHA-256 불일치",
    )

    provenance = as_row(report["data"])

    m5.require(
        provenance["test_target_values_read"] is False,
        "최종 Test 정답 격리 위반",
    )

    counts = as_row(provenance["counts"])

    train_summary = as_row(counts["train"])
    validation_summary = as_row(counts["validation"])

    train_count = m5.integer(
        train_summary["samples"],
        "train_samples",
        1,
    )

    validation_count = m5.integer(
        validation_summary["samples"],
        "validation_samples",
        1,
    )

    periods = as_row(report["development_periods"])

    period_counts = validate_periods(
        periods,
        validation_summary,
    )

    model = cast(
        m5.Model,
        as_row(payload["model"]),
    )

    m5.require(
        sum(model["counts"]) == train_count
        and report["training_joint_counts"]
        == dict(zip(m5.CLASSES, model["counts"])),
        "저장 모델의 Train 공동 빈도 불일치",
    )

    calibrations = as_row(report["calibration_fits"])

    m5.require(
        set(calibrations) == {"logistic", "regularized"},
        "저장된 보정 후보 누락",
    )

    for raw in calibrations.values():
        development.validate_temperature_fit(
            as_row(raw),
            period_counts["calibration"],
        )

    evaluations = {
        "selection": read_evaluations(
            report["selection_evaluations"]
        ),
        "reporting": read_evaluations(
            report["reporting_evaluations"]
        ),
        "full_validation": read_evaluations(
            report["candidate_evaluations"]
        ),
    }

    expected_counts = {
        "selection": period_counts["selection"],
        "reporting": period_counts["reporting"],
        "full_validation": validation_count,
    }

    for name, results in evaluations.items():
        development.validate_evaluations(
            results,
            model["counts"],
            expected_counts[name],
        )

    # 전체 Validation의 클래스 지원 수가 하위 개발 구간의
    # 합보다 작아지는 잘못된 보고 구조를 허용하지 않는다.
    for name in development.CANDIDATES:
        for axis in evaluation.AXES:
            full_classes = as_row(
                as_row(
                    evaluations["full_validation"][name][axis]
                )["classes"]
            )

            selection_classes = as_row(
                as_row(
                    evaluations["selection"][name][axis]
                )["classes"]
            )

            reporting_classes = as_row(
                as_row(
                    evaluations["reporting"][name][axis]
                )["classes"]
            )

            calibration_total = 0

            for label in development.AXIS_LABELS[axis]:
                full = as_row(
                    full_classes[label]
                )["validation_support"]

                selected_count = as_row(
                    selection_classes[label]
                )["validation_support"]

                reporting_count = as_row(
                    reporting_classes[label]
                )["validation_support"]

                remainder = (
                    full - selected_count - reporting_count
                )

                m5.require(
                    remainder >= 0,
                    "전체 및 하위 구간의 클래스 지원 수 불일치",
                )

                calibration_total += remainder

            m5.require(
                calibration_total
                == period_counts["calibration"],
                "보정 구간 클래스 지원 수 합계 오류",
            )

    selection = development.select_candidates(
        evaluations["selection"]
    )

    selected = str(selection["selected_candidate"])

    m5.require(
        selection == report["selection"]
        and selected == payload["selected_candidate"]
        and candidate_kind(selected) == model["kind"]
        and selected_temperature(selected, calibrations)
        == payload["selected_temperature"],
        "저장된 운영 후보 선정 또는 보정 계약 불일치",
    )

    expected_comparisons = {
        name: evaluation.compare(selected, results)
        for name, results in evaluations.items()
    }

    expected_support = {
        name: development.support_summary(results[selected])
        for name, results in evaluations.items()
    }

    m5.require(
        report["baseline_comparison"] == expected_comparisons
        and report["selected_support"] == expected_support,
        "저장된 비교 지표 또는 희소·미출현 자료 불일치",
    )

    runtime = m5.Runtime(model)
    examples = as_row(report["examples"])

    verify_examples(runtime, examples)

    primary = as_row(
        examples[report["primary_example"]]
    )

    stored_input = read_json(
        evaluation_path.parent / "example_input.json"
    )

    stored_prediction = read_json(
        evaluation_path.parent / "example_prediction.json"
    )

    m5.require(
        stored_input == primary["input"]
        and stored_prediction == primary["prediction"],
        "독립 추론 파일과 평가 사례 불일치",
    )

    compare_predictions(
        m5.predict(
            runtime,
            as_row(stored_input),
        ),
        stored_prediction,
    )

    return {
        "status": "verified",
        "training_run_id": payload["training_run_id"],
        "model_schema": SCHEMA,
        "model_version": VERSION,
        "selected_model": model["kind"],
        "selected_candidate": selected,
        "selected_temperature": payload["selected_temperature"],
        "operational_predictor_count": 1,
        "model_sha256": sha256_file(asset),
        "evaluation_sha256": sha256_file(evaluation_path),
        "verified_example_cases": sorted(examples),
        "validation_samples": validation_count,
        "development_periods": periods,
        "reproduce_equal": True,
        "reporting_used_for_selection": False,
        "final_test_evaluated": False,
    }


def parser() -> argparse.ArgumentParser:
    """학습·추론·검증 및 독립 출력 비교 명령을 구성한다."""
    result = argparse.ArgumentParser(
        description="M5 다음 투구 구종·결과 공동 분류"
    )

    result.add_argument(
        "--project-root",
        type=Path,
    )

    commands = result.add_subparsers(
        dest="command",
        required=True,
    )

    train_command = commands.add_parser("train")

    train_command.add_argument(
        "--run-id",
        required=True,
    )

    predict_command = commands.add_parser("predict")

    predict_command.add_argument(
        "--asset",
        type=Path,
        required=True,
    )

    predict_command.add_argument(
        "--input",
        type=Path,
        required=True,
    )

    verify_command = commands.add_parser("verify")

    verify_command.add_argument(
        "--asset",
        type=Path,
        required=True,
    )

    verify_command.add_argument(
        "--evaluation",
        type=Path,
        required=True,
    )

    compare_command = commands.add_parser("compare")

    compare_command.add_argument(
        "--expected",
        type=Path,
        required=True,
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

            logging.getLogger().setLevel(
                settings.log_level
            )

            result = train(
                settings,
                args.run_id,
            )

        elif args.command == "predict":
            runtime = load_runtime(
                args.asset,
                training_asset=True,
            )

            result = m5.predict(
                runtime,
                as_row(read_json(args.input)),
            )

            verify_prediction(result)

        elif args.command == "verify":
            result = verify_run(
                args.asset,
                args.evaluation,
            )

        else:
            # 독립 predict 프로세스의 JSON stdout을 표준 입력에서 읽는다.
            # PowerShell 인라인 Python 인용부호에 의존하지 않는다.
            actual = json.load(sys.stdin)
            expected = read_json(args.expected)

            result = compare_predictions(
                actual,
                expected,
            )

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
        LOGGER.error(
            "M5 실행 실패: %s",
            error,
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(main())