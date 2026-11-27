
"""M2 정규시즌 타석 결과 학습·보정·평가·저장 및 추론 CLI."""

import argparse
from collections.abc import Callable, Mapping, Sequence
from contextlib import closing
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import platform
import sqlite3
import sys
from typing import cast
from uuid import uuid4

from . import data
from . import model as m2
from . import probability as operational
from ...datasets.model_data import fingerprint
from ...datasets.published_sources import (
    RUN_PATTERN,
    SourceError,
    as_row,
    read_json,
    sha256_file,
    write_json,
)
from ...evaluation import plate_appearance as evaluation
from ...settings import Settings, load_settings


LOGGER = logging.getLogger(__name__)


def load_model(path: Path) -> m2.Model:
    """기존 v1 자산의 보존·비교용 분류기를 읽는다."""

    envelope = as_row(read_json(path))
    payload = as_row(envelope["payload"])

    if (
        envelope["sha256"] != fingerprint(payload)
        or payload["schema"] != m2.SCHEMA
        or payload["class_order"] != list(m2.CLASSES)
        or payload["class_names"] != m2.CLASS_NAMES
        or payload["input_columns"] != list(m2.INPUT_COLUMNS)
        or payload["method"] != m2.METHOD
    ):
        raise ValueError("M2 기존 자산 무결성 또는 계약 불일치")

    run_id = payload["training_run_id"]

    if not isinstance(run_id, str) or not RUN_PATTERN.fullmatch(run_id):
        raise ValueError("M2 학습 실행 ID 오류")

    seed = m2.count(payload["configured_seed"], "configured_seed")

    if seed > 2**32 - 1:
        raise ValueError("M2 저장 시드 범위 오류")

    model = cast(
        m2.Model,
        as_row(payload["model"]),
    )

    m2.validate_model(model)

    return model


def operational_payload(path: Path) -> dict[str, object]:
    """새 운영 자산의 지문, 스키마와 모델 파라미터를 확인한다."""

    envelope = as_row(read_json(path))
    payload = as_row(envelope["payload"])

    if payload["schema"] != operational.SCHEMA:
        raise ValueError(
            "M2 운영 모델은 v2 자산이 필요합니다."
            " 기존 v1 자산은 비교용으로만 보존합니다."
        )

    if (
        envelope["sha256"] != fingerprint(payload)
        or payload["class_order"] != list(m2.CLASSES)
        or payload["class_names"] != m2.CLASS_NAMES
        or payload["input_columns"] != list(m2.INPUT_COLUMNS)
        or payload["method"] != operational.METHOD
    ):
        raise ValueError("M2 운영 자산 무결성 또는 계약 불일치")

    run_id = payload["training_run_id"]

    if not isinstance(run_id, str) or not RUN_PATTERN.fullmatch(run_id):
        raise ValueError("M2 운영 학습 실행 ID 오류")

    seed = m2.count(
        payload["configured_seed"],
        "configured_seed",
    )

    if seed > 2**32 - 1:
        raise ValueError("M2 저장 시드 범위 오류")

    model = cast(
        operational.OperationalModel,
        as_row(payload["model"]),
    )

    operational.validate_model(model)

    if payload["selected_model"] != operational.candidate_name(model):
        raise ValueError("M2 저장 후보 이름 불일치")

    return payload


def load_operational_model(
    path: Path,
) -> operational.OperationalModel:
    """규칙 제한과 보정이 포함된 v2 단일 분류기를 복원한다."""

    payload = operational_payload(path)

    return cast(
        operational.OperationalModel,
        as_row(payload["model"]),
    )


def source_hashes() -> dict[str, str]:
    """실제 학습·평가·추론에 사용되는 소스 파일 지문을 기록한다."""

    files = {
        "src/models/plate_appearance/model.py": Path(
            m2.__file__
        ),
        "src/models/plate_appearance/data.py": Path(
            data.__file__
        ),
        "src/models/plate_appearance/probability.py": Path(
            operational.__file__
        ),
        "src/models/plate_appearance/__main__.py": Path(
            __file__
        ),
        "src/evaluation/plate_appearance.py": Path(
            evaluation.__file__
        ),
    }

    return {
        name: sha256_file(path)
        for name, path in files.items()
    }


def evaluate_periods(
    calibration: Sequence[m2.Sample],
    selection: Sequence[m2.Sample],
    predictor: Callable[
        [Mapping[str, object]], list[float]
    ],
    *,
    require_rules: bool,
) -> dict[str, operational.DetailedMetrics]:
    """동일 예측 결과로 전체·보정·선정 구간을 각각 평가한다."""

    samples = [*calibration, *selection]

    # 후보별 확률을 한 번만 계산하여 비교 분모를 고정한다.
    predictions = [
        predictor(sample.inputs)
        for sample in samples
    ]

    boundary = len(calibration)

    return {
        "validation": operational.score_predictions(
            samples,
            predictions,
            require_rules=require_rules,
        ),
        "calibration": operational.score_predictions(
            calibration,
            predictions[:boundary],
            require_rules=require_rules,
        ),
        "selection": operational.score_predictions(
            selection,
            predictions[boundary:],
            require_rules=require_rules,
        ),
    }


def metric_difference(
    candidate: operational.DetailedMetrics,
    baseline: operational.DetailedMetrics,
) -> dict[str, float]:
    """기존 확률 지표와 추가 보정·규칙 지표의 차이를 계산한다."""

    return {
        **evaluation.differences(candidate, baseline),
        "macro_classwise_ece": (
            candidate["macro_classwise_ece"]
            - baseline["macro_classwise_ece"]
        ),
        "forbidden_probability_mean": (
            candidate["forbidden_probability_mean"]
            - baseline["forbidden_probability_mean"]
        ),
    }


def participant_coverage(
    train: Sequence[m2.Sample],
    validation: Sequence[m2.Sample],
    logistic: m2.Model,
    regularized: m2.Model,
) -> dict[str, object]:
    """미출현 선수와 ID 계수 후퇴 건수를 후보별 기록한다."""

    train_batters = {
        sample.inputs["batter_id"]
        for sample in train
    }

    train_pitchers = {
        sample.inputs["pitcher_id"]
        for sample in train
    }

    coefficient_coverage = {}

    for name, model in (
        ("logistic", logistic),
        ("regularized", regularized),
    ):
        weights = model["weights"]

        coefficient_coverage[name] = {
            "no_batter_id_coefficient_samples": sum(
                f"id:batter:{sample.inputs['batter_id']}"
                not in weights
                for sample in validation
            ),
            "no_pitcher_id_coefficient_samples": sum(
                f"id:pitcher:{sample.inputs['pitcher_id']}"
                not in weights
                for sample in validation
            ),
        }

    return {
        "new_batter_validation_samples": sum(
            sample.inputs["batter_id"] not in train_batters
            for sample in validation
        ),
        "new_pitcher_validation_samples": sum(
            sample.inputs["pitcher_id"] not in train_pitchers
            for sample in validation
        ),
        "coefficient_coverage": coefficient_coverage,
    }


def train_run(
    settings: Settings,
    run_id: str,
) -> dict[str, object]:
    """자료를 검증하고 단일 운영 모델을 학습·선정·저장한다."""

    initial_hashes = source_hashes()

    database, provenance = data.source(settings, run_id)

    LOGGER.info("M2 모델 자료 무결성 및 입력 계약 확인 완료")

    with closing(
        sqlite3.connect(
            database.as_uri() + "?mode=ro",
            uri=True,
        )
    ) as db:
        db.execute("PRAGMA query_only = ON")
        samples, ledger = data.read_samples(db)

    data.verify_intake(samples, ledger)

    train = samples["train"]
    validation = samples["validation"]

    summaries = data.split_summary(samples, ledger)

    LOGGER.info(
        "M2 유효 타석 train=%d validation=%d",
        len(train),
        len(validation),
    )

    rule_audit = {
        "train": operational.require_rule_consistency(
            train, "train"
        ),
        "validation": operational.require_rule_consistency(
            validation, "validation"
        ),
    }

    LOGGER.info(
        "M2 실제 관측 정답의 규칙 충돌 검사 완료"
    )

    calibration, selection = operational.split_validation(
        validation
    )

    LOGGER.info(
        "M2 개발 구간 calibration=%d selection=%d",
        len(calibration),
        len(selection),
    )

    global_base, conditional_base = m2.fit_frequencies(train)

    # 기존 모델을 같은 Train에서 학습하여 변경 전후를 비교한다.
    logistic_base = m2.fit_logistic(train, settings.seed)

    regularized_base = operational.fit_regularized_logistic(
        train,
        settings.seed,
    )

    legacy_models: dict[str, m2.Model] = {
        "global": global_base,
        "conditional": conditional_base,
        "logistic": logistic_base,
    }

    candidates: dict[str, operational.OperationalModel] = {
        "global": operational.make_model(
            global_base, "global"
        ),
        "conditional": operational.make_model(
            conditional_base, "conditional"
        ),
        "logistic": operational.make_model(
            logistic_base, "logistic"
        ),
        "regularized": operational.make_model(
            regularized_base, "regularized"
        ),
    }

    calibration_reports: dict[str, dict[str, object]] = {}

    for name in ("logistic", "regularized"):
        temperature, bias, fit_report = (
            operational.fit_calibrations(
                candidates[name],
                calibration,
            )
        )

        candidates[operational.candidate_name(temperature)] = (
            temperature
        )

        candidates[operational.candidate_name(bias)] = bias

        calibration_reports[name] = fit_report

        LOGGER.info(
            "M2 확률 보정 parent=%s temperature=%s",
            name,
            fit_report["selected_temperature"],
        )

    full_results: dict[str, operational.DetailedMetrics] = {}
    calibration_results: dict[str, operational.DetailedMetrics] = {}
    selection_results: dict[str, operational.DetailedMetrics] = {}

    for name, candidate in candidates.items():
        operational.validate_model(candidate)

        period_results = evaluate_periods(
            calibration,
            selection,
            lambda inputs: operational.predict_probabilities(
                candidate, inputs
            ),
            require_rules=True,
        )

        full_results[name] = period_results["validation"]
        calibration_results[name] = period_results["calibration"]
        selection_results[name] = period_results["selection"]

        result = selection_results[name]

        LOGGER.info(
            "M2 선정 평가 candidate=%s "
            "log_loss=%.8f brier=%.8f "
            "macro_ece=%.8f macro_f1=%.8f",
            name,
            result["log_loss"],
            result["brier"],
            result["macro_classwise_ece"],
            result["macro_f1"],
        )

    selected, selection_report = operational.select_model(
        selection_results,
        candidates,
    )

    model = candidates[selected]

    legacy_full: dict[str, operational.DetailedMetrics] = {}
    legacy_selection: dict[str, operational.DetailedMetrics] = {}

    # 기존 무제한 확률은 진단 비교 전용으로만 계산한다.
    # 운영 후보 선정에는 절대 사용하지 않는다.
    for name, base in legacy_models.items():
        m2.validate_model(base)

        result = evaluate_periods(
            calibration,
            selection,
            lambda inputs: m2.predict_probabilities(
                base, inputs
            ),
            require_rules=False,
        )

        legacy_full[name] = result["validation"]
        legacy_selection[name] = result["selection"]

    baseline_name = str(selection_report["baseline"])

    training_run = uuid4().hex

    model_dir = settings.models_dir / "m2" / training_run
    result_dir = settings.results_dir / "m2" / training_run

    paths = {
        "model": str(model_dir / "model.json"),
        "evaluation": str(result_dir / "evaluation.json"),
        "input": str(result_dir / "example_input.json"),
        "prediction": str(result_dir / "example_prediction.json"),
    }

    example = selection[0]

    prediction = operational.predict(
        model,
        example.inputs,
    )

    unseen = [
        name
        for name, count in zip(
            m2.CLASSES,
            global_base["counts"],
            strict=True,
        )
        if count == 0
    ]

    fallback_count = sum(
        sum(
            conditional_base["groups"].get(
                m2.context_key(sample.inputs),
                [],
            )
        ) < m2.MIN_GROUP_COUNT
        for sample in validation
    )

    # 실행 도중 코드가 변경되었다면 혼합 버전 자산을 발행하지 않는다.
    if source_hashes() != initial_hashes:
        raise ValueError("M2 학습 도중 소스 파일이 변경되었습니다.")

    report: dict[str, object] = {
        "training_run_id": training_run,
        "model_schema": operational.SCHEMA,
        "configured_seed": settings.seed,
        "class_order": list(m2.CLASSES),
        "class_names": m2.CLASS_NAMES,
        "class_mapping": {
            name: name for name in m2.CLASSES
        },
        "data": provenance,
        "paths": paths,
        "source_sha256": initial_hashes,
        "splits": summaries,
        "development_periods": {
            "calibration_boundary": (
                operational.CALIBRATION_BOUNDARY
            ),
            "calibration": operational.period_summary(
                calibration
            ),
            "selection": operational.period_summary(
                selection
            ),
        },
        "selected_model": selected,
        "selected_base_kind": model["base"]["kind"],
        "selected_calibration": model["calibration"],
        "selected_temperature": model["temperature"],
        "selected_class_offsets": dict(
            zip(
                m2.CLASSES,
                model["class_offsets"],
                strict=True,
            )
        ),
        "selection_reason": selection_report["reason"],
        "operational_predictor_count": 1,
        "experimental_candidate_count": len(candidates),
        "selection": {
            **selection_report,
            "fit_period": "train",
            "calibration_fit_period": (
                "2025_before_07_01"
            ),
            "selection_period": "2025_from_07_01",
            "full_validation_independent_of_selection": False,
            "validation_refit_classifier": False,
            "selected_probability_calibrator": (
                model["calibration"] != "none"
            ),
        },
        "rule_audit": rule_audit,
        "calibration_fit": calibration_reports,
        "metric_definitions": {
            "log_loss_epsilon": evaluation.LOG_EPSILON,
            "brier": "mean_sum_of_class_squared_errors",
            "macro_f1": "fixed_15_classes_zero_division_zero",
            "calibration_bins": evaluation.CALIBRATION_BINS,
            "macro_classwise_ece": (
                "mean_of_15_sample_weighted_classwise_bin_errors"
            ),
            "forbidden_probability_mean": (
                "mean_sum_of_forbidden_class_probabilities"
            ),
        },
        "rare_and_unseen": {
            "unseen_train_classes": unseen,
            "unseen_class_validation_samples": sum(
                sample.target in unseen
                for sample in validation
            ),
            "global_alpha": m2.GLOBAL_ALPHA,
            "logistic_prior_mix": m2.PRIOR_MIX,
            "minimum_calibration_support": (
                operational.MIN_CALIBRATION_SUPPORT
            ),
            "unknown_target_policy": "fail_closed",
            "missing_target_policy": "fail_closed",
            "class_merging": False,
            "forbidden_class_probability": 0.0,
        },
        "participant_coverage": participant_coverage(
            train,
            validation,
            logistic_base,
            regularized_base,
        ),
        "conditional_frequency": {
            "group_sample_counts": {
                key: sum(values)
                for key, values in conditional_base["groups"].items()
            },
            "validation_global_fallback_samples": fallback_count,
            "minimum_group_count": m2.MIN_GROUP_COUNT,
            "global_prior_weight": m2.GROUP_PRIOR,
        },
        "evaluation": full_results,
        "calibration_evaluation": calibration_results,
        "selection_evaluation": selection_results,
        "legacy_unmasked_evaluation": legacy_full,
        "legacy_unmasked_selection_evaluation": legacy_selection,
        "differences": {
            "selected_minus_frequency_full_validation": (
                metric_difference(
                    full_results[selected],
                    full_results[baseline_name],
                )
            ),
            "selected_minus_frequency_selection": (
                metric_difference(
                    selection_results[selected],
                    selection_results[baseline_name],
                )
            ),
            "selected_minus_legacy_logistic_full_validation": (
                metric_difference(
                    full_results[selected],
                    legacy_full["logistic"],
                )
            ),
            "selected_minus_legacy_logistic_selection": (
                metric_difference(
                    selection_results[selected],
                    legacy_selection["logistic"],
                )
            ),
        },
        "example_sample_id": example.sample_id,
        "example_date": example.day,
        "final_test": "held_out_targets_not_evaluated",
        "limitations": [
            "정규시즌 개발 자료에 한정",
            "전체 2025 Validation은 보정과 후보 선정에 사용됨",
            "2026 최종 Test 일반화 성능은 후속 평가",
            "과거 공개 시점의 완전한 복원을 주장하지 않음",
            "참가자 ID 미출현은 과거 기록과 공통 변수로 후퇴",
            "희소 클래스별 신뢰도는 실제 표본 수에 제한됨",
            "규칙 마스크는 완전한 야구 상태 전이 엔진이 아님",
            "최빈 결과의 정확도와 공동 확률 품질은 별개임",
        ],
    }

    payload = {
        "schema": operational.SCHEMA,
        "input_columns": list(m2.INPUT_COLUMNS),
        "class_order": list(m2.CLASSES),
        "class_names": m2.CLASS_NAMES,
        "method": operational.METHOD,
        "model": model,
        "training_run_id": training_run,
        "configured_seed": settings.seed,
        "uses_randomness_for_fit": (
            model["base"]["kind"] == "logistic"
        ),
        "data": provenance,
        "outputs": paths,
        "splits": summaries,
        "selected_model": selected,
        "selection_reason": selection_report["reason"],
        "evaluation_fingerprint": fingerprint(report),
        "python": platform.python_version(),
        "sqlite": sqlite3.sqlite_version,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": initial_hashes,
        "reproduce_argv": [
            "-m",
            "src.models.plate_appearance",
            "--project-root",
            str(settings.project_root),
            "train",
            "--run-id",
            run_id,
        ],
    }

    model_dir.mkdir(parents=True, exist_ok=False)
    result_dir.mkdir(parents=True, exist_ok=False)

    temporary = model_dir / "model.pending.json"

    write_json(
        temporary,
        {
            "payload": payload,
            "sha256": fingerprint(payload),
        },
    )

    restored = load_operational_model(temporary)

    if operational.predict(restored, example.inputs) != prediction:
        raise ValueError("M2 저장 전후 예측 불일치")

    write_json(Path(paths["input"]), example.inputs)
    write_json(Path(paths["prediction"]), prediction)
    write_json(Path(paths["evaluation"]), report)

    # 정상적으로 저장·복원된 자산 하나만 운영 파일로 확정한다.
    temporary.rename(Path(paths["model"]))

    LOGGER.info(
        "M2 학습·보정·선정·저장 완료 run=%s selected=%s",
        training_run,
        selected,
    )

    return report


def verify_run(
    asset: Path,
    evaluation_path: Path,
) -> dict[str, object]:
    """저장 자산, 평가 지문, 실행 소스와 예시 추론을 재검사한다."""

    if asset.name != "model.json":
        raise ValueError("확정된 model.json 자산이 필요합니다.")

    payload = operational_payload(asset)
    report = as_row(read_json(evaluation_path))

    if fingerprint(report) != payload["evaluation_fingerprint"]:
        raise ValueError("M2 저장 평가 지문 불일치")

    if (
        report["training_run_id"] != payload["training_run_id"]
        or report["model_schema"] != operational.SCHEMA
        or report["selected_model"] != payload["selected_model"]
        or report["data"] != payload["data"]
        or report["configured_seed"] != payload["configured_seed"]
        or report["class_order"] != list(m2.CLASSES)
        or report["operational_predictor_count"] != 1
    ):
        raise ValueError("M2 자산과 평가 자료의 계약 불일치")

    saved_hashes = as_row(payload["source_sha256"])
    current_hashes = source_hashes()

    if (
        saved_hashes != current_hashes
        or report["source_sha256"] != saved_hashes
    ):
        raise ValueError("M2 학습 당시 소스와 현재 소스 SHA-256 불일치")

    model = cast(
        operational.OperationalModel,
        as_row(payload["model"]),
    )

    inputs = as_row(
        read_json(
            evaluation_path.with_name("example_input.json")
        )
    )

    expected = as_row(
        read_json(
            evaluation_path.with_name("example_prediction.json")
        )
    )

    actual = operational.predict(model, inputs)

    if actual != expected:
        raise ValueError("M2 예시 저장 추론 결과 불일치")

    selected = operational.candidate_name(model)

    selected_metrics = as_row(
        as_row(report["evaluation"])[selected]
    )

    selection_metrics = as_row(
        as_row(report["selection_evaluation"])[selected]
    )

    for metrics in (selected_metrics, selection_metrics):
        if (
            metrics["forbidden_nonzero_samples"] != 0
            or metrics["forbidden_target_count"] != 0
            or metrics["forbidden_probability_mean"] != 0.0
        ):
            raise ValueError("M2 선택 모델 평가에 규칙 위반 존재")

    return {
        "training_run_id": payload["training_run_id"],
        "schema": operational.SCHEMA,
        "selected_model": selected,
        "asset_sha256": sha256_file(asset),
        "evaluation_fingerprint": fingerprint(report),
        "source_sha256": current_hashes,
        "source_hash_match": True,
        "evaluation_hash_match": True,
        "example_prediction_match": True,
        "forbidden_probability_verified": True,
        "operational_predictor_count": 1,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """성공 0, 설정·자료 오류 2, 사용자 중단 3을 반환한다."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path)

    commands = parser.add_subparsers(
        dest="command",
        required=True,
    )

    training = commands.add_parser("train")
    training.add_argument("--run-id", required=True)

    inference = commands.add_parser("predict")
    inference.add_argument("--asset", required=True, type=Path)
    inference.add_argument("--input", required=True, type=Path)

    verification = commands.add_parser("verify")
    verification.add_argument("--asset", required=True, type=Path)
    verification.add_argument(
        "--evaluation",
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

        if args.command == "train":
            result = train_run(settings, args.run_id)

        elif args.command == "verify":
            result = verify_run(
                args.asset,
                args.evaluation,
            )

        else:
            if args.asset.name != "model.json":
                raise ValueError(
                    "확정된 model.json 자산이 필요합니다."
                )

            model = load_operational_model(args.asset)
            inputs = as_row(read_json(args.input))

            result = operational.predict(model, inputs)

        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        )

    except (
        OSError,
        SourceError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        OverflowError,
        sqlite3.DatabaseError,
    ) as exc:
        LOGGER.error("M2 실행 실패: %s", exc)
        return 2

    except KeyboardInterrupt:
        LOGGER.warning("M2 실행 중단")
        return 3

    return 0


if __name__ == "__main__":
    sys.exit(main())
