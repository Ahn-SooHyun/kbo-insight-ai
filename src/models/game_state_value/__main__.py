"""경기 상황 가치 학습·평가·자산 검증 및 추론 CLI."""

import argparse
from collections.abc import Sequence
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import platform
import sqlite3
import sys
from typing import cast
from uuid import uuid4

from . import SCHEMA, VERSION
from . import data
from . import model as m4
from . import value
from ...datasets.model_data import fingerprint
from ...datasets.published_sources import (
    RUN_PATTERN,
    SourceError,
    as_row,
    read_json,
    sha256_file,
)
from ...evaluation import game_state_value as evaluation
from ...settings import Settings, load_settings


LOGGER = logging.getLogger(__name__)


def source_hashes() -> dict[str, str]:
    """학습·평가·추론에 사용한 소스의 SHA-256을 기록한다."""
    root = Path(__file__).resolve().parents[3]

    files = (
        "src/models/game_state_value/__init__.py",
        "src/models/game_state_value/data.py",
        "src/models/game_state_value/model.py",
        "src/models/game_state_value/value.py",
        "src/models/game_state_value/__main__.py",
        "src/evaluation/game_state_value.py",
        "src/datasets/model_data.py",
        "src/datasets/published_sources.py",
        "src/settings.py",
    )

    return {
        name: sha256_file(root / name)
        for name in files
    }


def method_settings() -> dict[str, object]:
    """실험 전에 고정된 후보와 학습 설정을 기록한다."""
    return {
        "minimum_support": m4.MIN_SUPPORT,
        "epochs": m4.EPOCHS,
        "batch_size": m4.BATCH_SIZE,
        "learning_rate": m4.LEARNING_RATE,
        "learning_rate_schedule": "initial / sqrt(epoch + 1)",
        "frequency_smoothing": "laplace_alpha_1",
        "candidate_methods": {
            name: {
                "history_features": history,
                "l2": l2,
            }
            for name, (history, l2) in m4.METHODS.items()
        },
        "selection": "validation_all_log_loss_then_brier",
        "win_target": "win=1; tie=0; loss=0",
        "calibration_bins": evaluation.CALIBRATION_BINS,
        "calibration_fit": False,
        "calibration_diagnostics": True,
        "re_backoff": [
            "half_outs_base_mask",
            "outs_base_mask",
            "outs",
            "global",
        ],
        "importance": (
            "expected_absolute_wpa / supported_train_reference_mean"
        ),
        "importance_transition_key": (
            "inning|half|score_difference|outs|base_mask"
        ),
        "importance_minimum_support": m4.MIN_SUPPORT,
        "importance_unsupported": None,
    }


def save_json(path: Path, content: object) -> None:
    """기존 자산을 덮어쓰지 않고 JSON을 저장한다."""
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(
            content,
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")


def load_payload(path: Path) -> dict[str, object]:
    """모델 자산의 버전·지문·입력 계약과 구성 요소를 검증한다."""
    envelope = as_row(read_json(path))

    m4.require(
        set(envelope) == {"payload", "sha256"},
        "M4 자산 외부 구조 오류",
    )

    payload = as_row(envelope["payload"])

    m4.require(
        envelope["sha256"] == fingerprint(payload)
        and payload["schema"] == SCHEMA
        and payload["version"] == VERSION
        and payload["input_columns"] == list(m4.INPUT_COLUMNS)
        and payload["allowed_input_columns"] == list(m4.ALL_COLUMNS)
        and payload["operational_predictor_count"] == 1
        and payload["method_settings"] == method_settings(),
        "M4 자산 버전 또는 지문 불일치",
    )

    run_id = payload["training_run_id"]

    m4.require(
        isinstance(run_id, str)
        and RUN_PATTERN.fullmatch(run_id) is not None,
        "M4 학습 실행 ID 오류",
    )

    seed = m4.count(payload["configured_seed"], "configured_seed")

    m4.require(seed < 2**32, "저장 Seed 범위 오류")

    asset = cast(
        value.ValueModel,
        as_row(payload["model"]),
    )

    value.validate_asset(asset)

    m4.require(
        asset["predictor"]["name"] == payload["selected_model"],
        "저장된 운영 후보 불일치",
    )

    return payload


def load_model(path: Path) -> value.ValueModel:
    """검증된 단일 운영 예측기와 통계표를 복원한다."""
    return cast(
        value.ValueModel,
        load_payload(path)["model"],
    )


def candidate_key(
    metrics: dict[str, object],
) -> tuple[float, float]:
    """동일 Validation 표본의 사전 고정 선정 기준을 반환한다."""
    groups = as_row(metrics["groups"])
    overall = as_row(groups["all"])

    loss = overall["log_loss"]
    brier = overall["brier"]

    m4.require(
        type(loss) in (int, float)
        and type(brier) in (int, float),
        "운영 후보 선정 지표 오류",
    )

    return float(loss), float(brier)


def select_predictor(
    train: list[m4.Observation],
    validation: list[m4.Observation],
    seed: int,
) -> tuple[m4.Predictor, dict[str, dict[str, object]]]:
    """기준과 로지스틱 후보를 비교하고 운영 예측기 하나를 선택한다."""
    results: dict[str, dict[str, object]] = {}

    selected = None
    selected_key = None

    for name in m4.METHODS:
        LOGGER.info("승리 확률 후보 학습: %s", name)

        predictor = (
            m4.fit_frequency(train)
            if name == "frequency"
            else m4.fit_logistic(train, name, seed)
        )

        m4.validate_predictor(predictor)

        metrics = evaluation.evaluate_win(
            predictor,
            validation,
        )

        results[name] = metrics
        key = candidate_key(metrics)

        LOGGER.info(
            "후보 평가: %s log_loss=%.8f brier=%.8f",
            name,
            key[0],
            key[1],
        )

        # 정확히 동률이면 앞서 평가한 더 단순한 후보를 유지한다.
        if selected_key is None or key < selected_key:
            selected = predictor
            selected_key = key

    m4.require(selected is not None, "운영 예측기 선정 실패")

    return selected, results


def example_cases(
    asset: value.ValueModel,
    pairs: Sequence[data.Pair],
) -> list[dict[str, object]]:
    """실제 Validation에서 종료·계속·희소 경계 예시를 보존한다."""
    examples = []
    found: set[str] = set()

    for pair in pairs:
        pre = pair.pre.state
        post = pair.post.state

        names = []

        if pair.final is None and post.outs < 3:
            names.append("ordinary_plate_appearance")

        if pair.final is None and post.outs == 3:
            names.append("three_outs_game_continues")

        if pair.final is not None:
            if pair.final.winner == "tie":
                names.append("confirmed_final_tie")
            elif pair.post.final == 1:
                names.append("confirmed_final_win")
            else:
                names.append("confirmed_final_loss")

            if post.outs < 3:
                names.append("confirmed_partial_final")

            if (
                post.half == "bot"
                and post.inning >= 9
                and pre.difference <= 0
                and post.difference > 0
            ):
                names.append("observed_walkoff_pattern")

        support = sum(
            asset["transitions"].get(
                m4.exact_key(pre), {}
            ).values()
        )

        if support < m4.MIN_SUPPORT:
            names.append("unsupported_importance")

        new_names = set(names) - found

        if not new_names:
            continue

        inputs = value.pair_input(pair)
        prediction = value.predict(asset, inputs)

        examples.append({
            "cases": sorted(new_names),
            "pair_id": pair.pair_id,
            "sample_date": pair.sample_date,
            "input": inputs,
            "prediction": prediction,
        })

        found.update(new_names)

        if len(found) == 8:
            break

    m4.require(bool(examples), "독립 추론 검증 예시 없음")

    required = {
        "ordinary_plate_appearance",
        "three_outs_game_continues",
        "confirmed_final_tie",
        "confirmed_final_win",
        "confirmed_final_loss",
        "confirmed_partial_final",
    }

    m4.require(
        required <= found,
        "인수 Validation의 필수 종료·계속 사례 누락",
    )

    return examples


def compare_baseline(
    selected: str,
    candidates: dict[str, dict[str, object]],
) -> dict[str, object]:
    """운영 후보와 기준 모델의 지표 변화량을 기록한다."""
    baseline = as_row(
        as_row(candidates["frequency"]["groups"])["all"]
    )

    current = as_row(
        as_row(candidates[selected]["groups"])["all"]
    )

    return {
        "baseline": "frequency",
        "selected": selected,
        "selected_minus_baseline": {
            name: current[name] - baseline[name]
            for name in ("log_loss", "brier", "ece")
        },
        "lower_is_better": ["log_loss", "brier", "ece"],
    }


def train_run(
    settings: Settings,
    run_id: str,
) -> dict[str, object]:
    """자료 검사부터 개발 모델 선택, 자산 저장까지 수행한다."""
    initial_hashes = source_hashes()

    LOGGER.info("M4 자료 인수 검사 시작: %s", run_id)

    pairs, provenance = data.load(settings, run_id)

    train_pairs = pairs["train"]
    validation_pairs = pairs["validation"]

    train_rows = data.observations(train_pairs)
    validation_rows = data.observations(validation_pairs)

    LOGGER.info(
        "M4 자료 확인: train=%s validation=%s",
        len(train_rows),
        len(validation_rows),
    )

    predictor, candidates = select_predictor(
        train_rows,
        validation_rows,
        settings.seed,
    )

    selected = predictor["name"]

    LOGGER.info("M4 운영 승리 예측기 선정: %s", selected)
    LOGGER.info("Train 잔여 기대득점 및 상태 전이 집계")

    re_table = value.fit_re(train_rows)
    transitions = value.fit_transitions(train_pairs)

    LOGGER.info("상황 중요도 Train 참조 평균 계산")

    reference = value.fit_reference(
        predictor,
        transitions,
        train_pairs,
    )

    asset: value.ValueModel = {
        "predictor": predictor,
        "re_table": re_table,
        "transitions": transitions,
        "reference": reference,
    }

    value.validate_asset(asset)

    LOGGER.info("Validation 기대득점·타석 가치·상황 중요도 평가")

    re_metrics = evaluation.evaluate_re(
        re_table,
        validation_rows,
    )

    pair_metrics = evaluation.evaluate_pairs(
        asset,
        validation_pairs,
    )

    examples = example_cases(
        asset,
        validation_pairs,
    )

    example = examples[0]

    training_run = uuid4().hex
    created_at = datetime.now(timezone.utc).isoformat()

    model_directory = (
        settings.models_dir / "m4" / training_run
    )

    result_directory = (
        settings.results_dir / "m4" / training_run
    )

    paths = {
        "model": str(model_directory / "model.json"),
        "evaluation": str(result_directory / "evaluation.json"),
        "input": str(result_directory / "example_input.json"),
        "prediction": str(result_directory / "example_prediction.json"),
    }

    m4.require(
        source_hashes() == initial_hashes,
        "학습 실행 중 소스 변경 확인",
    )

    # 자료는 읽기 전용으로 접근하며 실행 전후 지문을 대조한다.
    m4.require(
        sha256_file(Path(provenance["database"]))
        == provenance["database_sha256"]
        and sha256_file(Path(provenance["standard_database"]))
        == provenance["standard_database_sha256"],
        "학습 실행 중 인수 자료 변경 확인",
    )

    report = {
        "status": "completed",
        "training_run_id": training_run,
        "created_at_utc": created_at,
        "model_schema": SCHEMA,
        "model_version": VERSION,
        "configured_seed": settings.seed,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "device": "cpu",
        "selected_model": selected,
        "operational_predictor_count": 1,
        "data": provenance,
        "method_settings": method_settings(),
        "train_summary": evaluation.training_summary(train_rows),
        "candidate_evaluations": candidates,
        "baseline_comparison": compare_baseline(selected, candidates),
        "re_evaluation": re_metrics,
        "pair_evaluation": pair_metrics,
        "importance_reference": reference,
        "transition_state_count": len(transitions),
        "re_table_count": len(re_table),
        "examples": examples,
        "source_sha256": initial_hashes,
        "paths": paths,
        "final_test_evaluated": False,
        "validation_used_for_selection": True,
        "reproduce_argv": [
            "-m",
            "src.models.game_state_value",
            "--project-root",
            str(settings.project_root),
            "train",
            "--run-id",
            run_id,
        ],
    }

    payload = {
        "schema": SCHEMA,
        "version": VERSION,
        "training_run_id": training_run,
        "created_at_utc": created_at,
        "configured_seed": settings.seed,
        "selected_model": selected,
        "operational_predictor_count": 1,
        "input_columns": list(m4.INPUT_COLUMNS),
        "allowed_input_columns": list(m4.ALL_COLUMNS),
        "method_settings": method_settings(),
        "model": asset,
        "data": provenance,
        "source_sha256": initial_hashes,
        "evaluation_fingerprint": fingerprint(report),
    }

    model_directory.mkdir(parents=True, exist_ok=False)
    result_directory.mkdir(parents=True, exist_ok=False)

    pending = model_directory / "model.pending.json"

    save_json(pending, {
        "payload": payload,
        "sha256": fingerprint(payload),
    })

    restored = load_model(pending)

    for case in examples:
        m4.require(
            value.predict(restored, case["input"]) == case["prediction"],
            "저장 전후 관측 타석 가치 재현 불일치",
        )

    save_json(Path(paths["input"]), example["input"])
    save_json(Path(paths["prediction"]), example["prediction"])
    save_json(Path(paths["evaluation"]), report)

    # 모든 자료 저장 및 복원 검사를 통과한 뒤 운영 파일을 확정한다.
    pending.rename(Path(paths["model"]))

    LOGGER.info(
        "M4 학습 및 저장 완료: run=%s selected=%s",
        training_run,
        selected,
    )

    return {
        "status": "completed",
        "training_run_id": training_run,
        "selected_model": selected,
        "model_schema": SCHEMA,
        "operational_predictor_count": 1,
        "configured_seed": settings.seed,
        "paths": paths,
        "counts": provenance["counts"],
        "ending_counts": provenance["ending_counts"],
        "baseline_comparison": report["baseline_comparison"],
        "importance_reference": reference,
        "importance_validation": pair_metrics["importance"],
        "evaluation_fingerprint": fingerprint(report),
        "evaluation_sha256": sha256_file(Path(paths["evaluation"])),
        "model_sha256": sha256_file(Path(paths["model"])),
        "saved_prediction_match": True,
        "final_test_evaluated": False,
    }


def verify_run(
    model_path: Path,
    evaluation_path: Path,
) -> dict[str, object]:
    """원본 DB 없이 운영 자산·평가·독립 예시를 다시 검증한다."""
    m4.require(
        model_path.name == "model.json",
        "확정된 model.json 자산이 필요합니다.",
    )

    payload = load_payload(model_path)
    report = as_row(read_json(evaluation_path))

    m4.require(
        fingerprint(report) == payload["evaluation_fingerprint"]
        and report["status"] == "completed"
        and report["training_run_id"] == payload["training_run_id"]
        and report["model_schema"] == SCHEMA
        and report["model_version"] == VERSION
        and report["selected_model"] == payload["selected_model"]
        and report["configured_seed"] == payload["configured_seed"]
        and report["operational_predictor_count"] == 1
        and report["data"] == payload["data"]
        and report["method_settings"] == payload["method_settings"]
        and report["final_test_evaluated"] is False,
        "M4 자산·평가 실행 계약 불일치",
    )

    m4.require(
        payload["source_sha256"] == source_hashes()
        and report["source_sha256"] == payload["source_sha256"],
        "학습 당시 소스와 현재 소스 SHA-256 불일치",
    )

    asset = cast(value.ValueModel, payload["model"])

    candidates = as_row(report["candidate_evaluations"])

    m4.require(
        set(candidates) == set(m4.METHODS),
        "후보 평가 자료 누락",
    )

    counts = as_row(as_row(report["data"])["counts"])

    validation_count = as_row(
        as_row(counts["validation"])["eligible"]
    )["states"]

    for candidate in candidates.values():
        overall = as_row(
            as_row(as_row(candidate)["groups"])["all"]
        )

        m4.require(
            overall["count"] == validation_count,
            "후보별 동일 Validation 분모 오류",
        )

        m4.require(
            sum(
                row["count"]
                for row in overall["calibration"]
            ) == validation_count,
            "보정 구간 평가 분모 오류",
        )

    selected = min(
        m4.METHODS,
        key=lambda name: candidate_key(candidates[name]),
    )

    m4.require(
        selected == payload["selected_model"],
        "저장 운영 후보의 선정 기준 불일치",
    )

    examples = report["examples"]

    m4.require(
        isinstance(examples, list) and bool(examples),
        "저장 독립 추론 예시 오류",
    )

    verified_cases: set[str] = set()

    for example in examples:
        example = as_row(example)

        m4.require(
            value.predict(asset, example["input"]) == example["prediction"],
            "독립 프로세스 추론 예시 불일치",
        )

        verified_cases.update(example["cases"])

    input_path = evaluation_path.with_name("example_input.json")
    prediction_path = evaluation_path.with_name("example_prediction.json")

    inputs = as_row(read_json(input_path))
    expected = as_row(read_json(prediction_path))

    m4.require(
        inputs == examples[0]["input"]
        and expected == examples[0]["prediction"]
        and value.predict(asset, inputs) == expected,
        "별도 예시 입력·출력 재현 불일치",
    )

    return {
        "status": "verified",
        "training_run_id": payload["training_run_id"],
        "selected_model": selected,
        "model_schema": SCHEMA,
        "operational_predictor_count": 1,
        "source_hashes_match": True,
        "evaluation_fingerprint_match": True,
        "example_prediction_match": True,
        "verified_examples": len(examples),
        "verified_cases": sorted(verified_cases),
        "model_sha256": sha256_file(model_path),
        "evaluation_sha256": sha256_file(evaluation_path),
        "final_test_evaluated": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """성공 0, 자료·설정·저장 오류 2, 사용자 중단 3을 반환한다."""
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
            m4.require(
                args.asset.name == "model.json",
                "확정된 model.json 자산이 필요합니다.",
            )

            asset = load_model(args.asset)
            raw = as_row(read_json(args.input))

            result = value.predict(asset, raw)

        print(json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ))

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
        LOGGER.error("M4 실행 실패: %s", exc)
        return 2

    except KeyboardInterrupt:
        LOGGER.warning("M4 실행 중단")
        return 3

    return 0


if __name__ == "__main__":
    sys.exit(main())