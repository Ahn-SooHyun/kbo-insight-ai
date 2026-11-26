"""M2 정규시즌 타석 결과 학습·평가·저장 및 추론 CLI."""

import argparse
from collections.abc import Sequence
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
    """JSON 자산의 무결성·버전·클래스·입력 계약을 확인한다."""

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
        raise ValueError("M2 저장 자산 무결성 또는 계약 불일치")

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


def train_run(
    settings: Settings,
    run_id: str,
) -> dict[str, object]:
    """고정 자료에서 세 방식을 비교하고 선정된 분류기 하나를 저장한다."""

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

    global_model, conditional_model = m2.fit_frequencies(train)
    logistic_model = m2.fit_logistic(train, settings.seed)

    candidates: dict[str, m2.Model] = {
        "global": global_model,
        "conditional": conditional_model,
        "logistic": logistic_model,
    }

    results: dict[str, evaluation.Metrics] = {}

    for name, candidate in candidates.items():
        m2.validate_model(candidate)
        result = evaluation.evaluate(candidate, validation)
        results[name] = result

        LOGGER.info(
            "M2 평가 %s log_loss=%.8f brier=%.8f macro_f1=%.8f",
            name,
            result["log_loss"],
            result["brier"],
            result["macro_f1"],
        )

    selected, reason = evaluation.select(results)
    model = candidates[selected]

    training_run = uuid4().hex

    model_dir = settings.models_dir / "m2" / training_run
    result_dir = settings.results_dir / "m2" / training_run

    paths = {
        "model": str(model_dir / "model.json"),
        "evaluation": str(result_dir / "evaluation.json"),
        "input": str(result_dir / "example_input.json"),
        "prediction": str(result_dir / "example_prediction.json"),
    }

    example = validation[0]
    prediction = m2.predict(model, example.inputs)

    unseen = [
        name
        for name, count in zip(
            m2.CLASSES,
            global_model["counts"],
            strict=True,
        )
        if count == 0
    ]

    train_batters = {
        sample.inputs["batter_id"] for sample in train
    }
    train_pitchers = {
        sample.inputs["pitcher_id"] for sample in train
    }

    logistic_weights = logistic_model["weights"]

    fallback_count = sum(
        sum(
            conditional_model["groups"].get(
                m2.context_key(sample.inputs), []
            )
        ) < m2.MIN_GROUP_COUNT
        for sample in validation
    )

    report = {
        "training_run_id": training_run,
        "model_schema": m2.SCHEMA,
        "configured_seed": settings.seed,
        "class_order": list(m2.CLASSES),
        "class_names": m2.CLASS_NAMES,
        "class_mapping": {
            name: name for name in m2.CLASSES
        },
        "data": provenance,
        "paths": paths,
        "splits": summaries,
        "selected_model": selected,
        "selection_reason": reason,
        "operational_predictor_count": 1,
        "selection": {
            "fit_period": "train",
            "selection_period": "2025_validation",
            "independent_of_selection": False,
            "validation_refit": False,
            "validation_hyperparameter_tuning": False,
            "fitted_probability_calibrator": False,
            "rule": (
                "frequency baseline by log loss then brier; "
                "logistic must improve at least one and worsen neither"
            ),
        },
        "metric_definitions": {
            "log_loss_epsilon": evaluation.LOG_EPSILON,
            "brier": "mean_sum_of_class_squared_errors",
            "macro_f1": "fixed_15_classes_zero_division_zero",
            "calibration_bins": evaluation.CALIBRATION_BINS,
        },
        "rare_and_unseen": {
            "unseen_train_classes": unseen,
            "unseen_class_validation_samples": sum(
                sample.target in unseen for sample in validation
            ),
            "global_alpha": m2.GLOBAL_ALPHA,
            "logistic_prior_mix": m2.PRIOR_MIX,
            "unknown_target_policy": "fail_closed",
            "missing_target_policy": "fail_closed",
            "class_merging": False,
        },
        "participant_coverage": {
            "new_batter_validation_samples": sum(
                sample.inputs["batter_id"] not in train_batters
                for sample in validation
            ),
            "new_pitcher_validation_samples": sum(
                sample.inputs["pitcher_id"] not in train_pitchers
                for sample in validation
            ),
            "no_batter_id_coefficient_samples": sum(
                f"id:batter:{sample.inputs['batter_id']}"
                not in logistic_weights
                for sample in validation
            ),
            "no_pitcher_id_coefficient_samples": sum(
                f"id:pitcher:{sample.inputs['pitcher_id']}"
                not in logistic_weights
                for sample in validation
            ),
        },
        "conditional_frequency": {
            "group_sample_counts": {
                key: sum(values)
                for key, values in conditional_model["groups"].items()
            },
            "validation_global_fallback_samples": fallback_count,
            "minimum_group_count": m2.MIN_GROUP_COUNT,
            "global_prior_weight": m2.GROUP_PRIOR,
        },
        "evaluation": results,
        "differences": {
            "conditional_minus_global": evaluation.differences(
                results["conditional"], results["global"]
            ),
            "logistic_minus_global": evaluation.differences(
                results["logistic"], results["global"]
            ),
            "logistic_minus_conditional": evaluation.differences(
                results["logistic"], results["conditional"]
            ),
        },
        "example_sample_id": example.sample_id,
        "example_date": example.day,
        "final_test": "held_out_targets_not_evaluated",
        "limitations": [
            "정규시즌 개발 자료에 한정",
            "2025 Validation은 후보 선택에 사용됨",
            "과거 공개 시점의 완전한 복원을 주장하지 않음",
            "참가자 ID 미출현은 과거 기록과 공통 변수로 후퇴",
            "희소 클래스별 확률의 신뢰도는 표본 수에 제한됨",
        ],
    }

    payload = {
        "schema": m2.SCHEMA,
        "input_columns": list(m2.INPUT_COLUMNS),
        "class_order": list(m2.CLASSES),
        "class_names": m2.CLASS_NAMES,
        "method": m2.METHOD,
        "model": model,
        "training_run_id": training_run,
        "configured_seed": settings.seed,
        "uses_randomness_for_fit": selected == "logistic",
        "data": provenance,
        "outputs": paths,
        "splits": summaries,
        "selected_model": selected,
        "selection_reason": reason,
        "evaluation_fingerprint": fingerprint(report),
        "python": platform.python_version(),
        "sqlite": sqlite3.sqlite_version,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": {
            "src/models/plate_appearance/model.py": sha256_file(
                Path(m2.__file__)
            ),
            "src/models/plate_appearance/data.py": sha256_file(
                Path(data.__file__)
            ),
            "src/models/plate_appearance/__main__.py": sha256_file(
                Path(__file__)
            ),
            "src/evaluation/plate_appearance.py": sha256_file(
                Path(evaluation.__file__)
            ),
        },
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

    restored = load_model(temporary)

    if m2.predict(restored, example.inputs) != prediction:
        raise ValueError("M2 저장 전후 예측 불일치")

    write_json(Path(paths["input"]), example.inputs)
    write_json(Path(paths["prediction"]), prediction)
    write_json(Path(paths["evaluation"]), report)

    # 저장 검증을 통과한 단일 자산만 최종 파일명으로 공개한다.
    temporary.rename(Path(paths["model"]))

    LOGGER.info(
        "M2 학습·평가·저장 완료 run=%s selected=%s",
        training_run,
        selected,
    )

    return report


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

    args = parser.parse_args(argv)

    try:
        settings = load_settings(args.project_root)

        logging.basicConfig(
            level=settings.log_level,
            format="%(levelname)s %(name)s: %(message)s",
        )

        if args.command == "train":
            result = train_run(settings, args.run_id)

        else:
            model = load_model(args.asset)
            inputs = as_row(read_json(args.input))
            result = m2.predict(model, inputs)

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