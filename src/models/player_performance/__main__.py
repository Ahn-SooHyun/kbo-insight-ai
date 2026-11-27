"""M3 학습·검증·자산 검사·독립 추론 실행 진입점."""

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import platform
import sqlite3
from time import perf_counter
from uuid import uuid4

from . import SCHEMA, VERSION, data, evaluation, model
from ...datasets.model_data import fingerprint, require
from ...datasets.published_sources import (
    RUN_PATTERN,
    SourceError,
    as_row,
    read_json,
    sha256_file,
    write_json,
)
from ...settings import Settings, load_settings


LOGGER = logging.getLogger(__name__)


def method_settings() -> dict[str, object]:
    """후보 비교 전에 고정한 전체 학습·평가 설정."""
    return {
        "model_type": "masked_multi_output_residual_regression_tree",
        "candidate_depths": list(model.CANDIDATE_DEPTHS),
        "min_leaf_rows": model.MIN_LEAF_ROWS,
        "min_output_support": model.MIN_OUTPUT_SUPPORT,
        "shrinkage": model.SHRINKAGE,
        "split_quantiles": list(model.SPLIT_QUANTILES),
        "min_gain": model.MIN_GAIN,
        "recent_weight": model.RECENT_WEIGHT,
        "calendar_reference": [
            model.REFERENCE_MONTH,
            model.REFERENCE_DAY,
        ],
        "min_remaining_days": model.MIN_REMAINING_DAYS,
        "loss": "masked_normalized_squared_error",
        "split_weight": "inverse_train_output_support",
        "record_scale": "max_one_train_target_mean",
        "missing_feature_value": -1.0,
        "missing_target": "excluded_from_corresponding_loss",
        "sparse_leaf_policy": "parent_mean_with_shrinkage",
        "player_id_feature": False,
        "team_id_feature": False,
        "bootstrap": False,
        "random_split": False,
        "stochastic_training": False,
        "projection": "nonnegative_coherent_period_increments",
        "interval_level": model.INTERVAL_LEVEL,
        "minimum_calibration_rows":
            evaluation.MIN_CALIBRATION_ROWS,
        "minimum_calibration_players":
            evaluation.MIN_CALIBRATION_PLAYERS,
        "minimum_calibration_weeks":
            evaluation.MIN_CALIBRATION_WEEKS,
        "calibration_start":
            data.CALIBRATION_START.isoformat(),
        "validation_start":
            data.VALIDATION_START.isoformat(),
        "test_start": data.TEST_START.isoformat(),
        "observation_end": data.OBSERVATION_END.isoformat(),
        "calibration_method":
            "held_out_absolute_residual_quantile",
        "selection":
            "joint_tree_macro_normalized_mae_rmse_depth",
        "baseline_is_comparison_only": True,
        "final_test_used": False,
    }


def source_hashes(root: Path) -> dict[str, str]:
    """실행에 필요한 코드·환경·설정의 지문을 계산한다."""
    files = (
        "src/models/player_performance/__init__.py",
        "src/models/player_performance/data.py",
        "src/models/player_performance/model.py",
        "src/models/player_performance/evaluation.py",
        "src/models/player_performance/__main__.py",
        "src/datasets/model_data.py",
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


def runtime() -> dict[str, object]:
    """실제 실행 런타임과 장치 사용 여부."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "implementation": platform.python_implementation(),
        "device": "cpu",
        "external_runtime_dependencies": [],
    }


def validate_intervals(
    role: str,
    value: object,
) -> dict[str, dict[str, object]]:
    """출력별 구간 지지 및 반경 자산을 검증한다."""
    rows = as_row(value)

    require(
        set(rows) == set(data.output_keys(role)),
        "구간 출력 사전 불일치",
    )

    result = {}

    for key, value in rows.items():
        row = as_row(value)
        n = data.integer(row["n"], "calibration_n")
        players = data.integer(
            row["unique_players"], "calibration_players"
        )
        weeks = data.integer(
            row["anchor_weeks"], "calibration_weeks"
        )

        require(
            players <= n
            and weeks <= n
            and row["target_coverage"] == model.INTERVAL_LEVEL
            and row["calibration_start"]
            == data.CALIBRATION_START.isoformat()
            and row["calibration_end_exclusive"]
            == data.VALIDATION_START.isoformat(),
            "구간 지지 또는 시간 계약 오류",
        )

        supported = (
            n >= evaluation.MIN_CALIBRATION_ROWS
            and players >= evaluation.MIN_CALIBRATION_PLAYERS
            and weeks >= evaluation.MIN_CALIBRATION_WEEKS
        )

        if supported:
            require(
                row["radius"] is not None
                and model.number(row["radius"]) >= 0
                and row["reason"] is None
                and type(row["quantile_rank"]) is int
                and 1 <= row["quantile_rank"] <= n,
                "지원되는 예측 구간의 자산 오류",
            )
        else:
            require(
                row["radius"] is None
                and row["quantile_rank"] is None
                and row["reason"] == "insufficient_support",
                "구간 지지 부족 처리 오류",
            )

        result[key] = row

    return result


def load_asset(
    path: Path,
) -> tuple[
    model.Fitted,
    dict[str, dict[str, object]],
    dict[str, object],
]:
    """JSON 자산의 지문·코드·런타임·출력 구조를 검증한다.

    모델 DB와 표준 DB는 열지 않는다.
    """
    envelope = as_row(read_json(path))

    require(
        set(envelope) == {"payload", "fingerprint"},
        "모델 자산 외부 구조 오류",
    )

    payload = as_row(envelope["payload"])

    require(
        fingerprint(payload) == envelope["fingerprint"],
        "모델 자산 지문 불일치",
    )

    require(
        payload["schema"] == SCHEMA
        and payload["version"] == VERSION
        and payload["method"] == method_settings()
        and payload["final_test_used"] is False
        and isinstance(payload["run_id"], str)
        and bool(RUN_PATTERN.fullmatch(payload["run_id"])),
        "모델 자산 버전 또는 실행 계약 불일치",
    )

    role = payload["role"]
    require(
        isinstance(role, str) and role in data.ROLES,
        "모델 자산 역할 오류",
    )

    stored_runtime = as_row(payload["runtime"])

    require(
        stored_runtime["device"] == "cpu"
        and stored_runtime["python"].split(".")[:2] == ["3", "12"]
        and platform.python_version_tuple()[:2] == ("3", "12"),
        "지원하지 않는 모델 런타임",
    )

    require(
        payload["features"] == list(model.feature_names(role))
        and payload["outputs"] == list(data.output_keys(role)),
        "저장 입력 또는 출력 사전 불일치",
    )

    root = Path(__file__).resolve().parents[3]

    require(
        payload["source_hashes"] == source_hashes(root),
        "저장 모델과 현재 실행 코드·설정의 지문 불일치",
    )

    fitted = model.restore(payload["model"])
    require(
        fitted.role == role
        and fitted.name == payload["selected_model"],
        "운영 모델 역할 또는 식별자 불일치",
    )

    intervals = validate_intervals(role, payload["intervals"])

    return fitted, intervals, {
        "run_id": payload["run_id"],
        "role": role,
        "schema": SCHEMA,
        "version": VERSION,
        "model": fitted.name,
        "fingerprint": envelope["fingerprint"],
        "asset_sha256": sha256_file(path),
        "runtime": stored_runtime,
        "final_test_used": False,
        "operating_predictors": 1,
    }


def train(
    settings: Settings,
    run_id: str,
    role: str,
) -> dict[str, object]:
    """승인 자료로 실제 Train 학습과 Validation 평가를 실행한다."""
    require(
        platform.python_version_tuple()[:2] == ("3", "12"),
        "Python 3.12 환경이 필요합니다",
    )

    start = perf_counter()
    hashes = source_hashes(settings.project_root)

    LOGGER.info("M3 %s 인수 자료 검증 시작", role)
    dataset = data.load_dataset(settings, run_id, role)

    intake_seconds = perf_counter() - start
    LOGGER.info(
        "M3 %s Train=%d Validation=%d",
        role,
        len(dataset.train),
        len(dataset.validation),
    )

    development = evaluation.develop(dataset, role)
    require(
        development.selected.tree is not None,
        "공동 회귀 운영 트리가 필요합니다",
    )

    require(
        hashes == source_hashes(settings.project_root),
        "학습 중 실행 코드 또는 설정이 변경되었습니다",
    )

    identifier = uuid4().hex

    asset_dir = settings.models_dir / "m3" / identifier
    result_dir = settings.results_dir / "m3" / identifier

    asset_dir.mkdir(parents=True, exist_ok=False)
    result_dir.mkdir(parents=True, exist_ok=False)

    asset_path = asset_dir / "model.json"
    evaluation_path = result_dir / "evaluation.json"
    input_path = result_dir / "example_input.json"
    expected_path = result_dir / "example_prediction.json"
    manifest_path = result_dir / "manifest.json"

    provenance = {
        name: value
        for name, value in dataset.provenance.items()
        if name != "verified_season_ends"
    }

    payload = {
        "schema": SCHEMA,
        "version": VERSION,
        "run_id": identifier,
        "role": role,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": settings.seed,
        "runtime": runtime(),
        "source_hashes": hashes,
        "method": method_settings(),
        "data": provenance,
        "features": list(model.feature_names(role)),
        "outputs": list(data.output_keys(role)),
        "selected_model": development.selected.name,
        "model": development.selected.to_payload(),
        "intervals": development.intervals,
        "final_test_used": False,
    }

    write_json(
        asset_path,
        {
            "payload": payload,
            "fingerprint": fingerprint(payload),
        },
    )

    example = max(
        dataset.validation,
        key=lambda sample: (
            sample.player.current_counts is not None,
            sample.targets[-1].usable,
            sum(
                value is not None
                for value in data.labels_for(
                    sample, data.TEST_START
                )
            ),
        ),
    )

    expected = model.format_prediction(
        development.selected,
        example.player,
        development.intervals,
    )

    restored, restored_intervals, inspected = load_asset(asset_path)

    reloaded = model.format_prediction(
        restored,
        data.parse_input(example.request),
        restored_intervals,
    )

    require(
        fingerprint(expected) == fingerprint(reloaded),
        "저장 전후 모델 예측 불일치",
    )

    write_json(input_path, example.request)
    write_json(expected_path, expected)

    report = {
        "schema": SCHEMA,
        "version": VERSION,
        "run_id": identifier,
        "role": role,
        "runtime": runtime(),
        "seed": settings.seed,
        "source_hashes": hashes,
        "data": dataset.provenance,
        "data_audit": dataset.audit,
        "method": method_settings(),
        "development": development.report,
        "selected_model": development.selected.name,
        "operating_predictors": 1,
        "intake_seconds": intake_seconds,
        "total_seconds": perf_counter() - start,
        "asset": str(asset_path),
        "asset_sha256": inspected["asset_sha256"],
        "asset_fingerprint": inspected["fingerprint"],
        "example_input_fingerprint": fingerprint(example.request),
        "example_prediction_fingerprint": fingerprint(expected),
        "same_process_reload_equal": True,
        "independent_process_reload":
            "requires_separate_predict_and_compare",
        "final_test_used": False,
    }

    write_json(evaluation_path, report)

    summary = {
        "status": "completed",
        "run_id": identifier,
        "role": role,
        "model": development.selected.name,
        "operating_predictors": 1,
        "asset": str(asset_path),
        "asset_sha256": inspected["asset_sha256"],
        "asset_fingerprint": inspected["fingerprint"],
        "evaluation": str(evaluation_path),
        "evaluation_sha256": sha256_file(evaluation_path),
        "input": str(input_path),
        "expected": str(expected_path),
        "manifest": str(manifest_path),
        "selected_beats_baseline":
            development.report["selected_beats_baseline"],
        "same_process_reload_equal": True,
        "final_test_used": False,
    }

    # 완료 매니페스트를 마지막에 작성한다. 중단된 실행은 완료로
    # 간주하지 않으며 기존 실행을 덮어쓰지 않는다.
    write_json(manifest_path, summary)

    return summary


def predict(
    asset_path: Path,
    input_path: Path,
    output_path: Path | None,
) -> dict[str, object]:
    """학습 DB 없이 저장된 예측기로 사전 입력을 추론한다."""
    fitted, intervals, _ = load_asset(asset_path)
    request = data.parse_input(read_json(input_path))

    result = model.format_prediction(
        fitted,
        request,
        intervals,
    )

    if output_path is not None:
        write_json(output_path, result)

    return result


def compare(
    expected_path: Path,
    actual_path: Path,
) -> dict[str, object]:
    """별도 프로세스 결과의 전체 JSON 지문을 비교한다."""
    expected = fingerprint(read_json(expected_path))
    actual = fingerprint(read_json(actual_path))

    require(
        expected == actual,
        "독립 프로세스 예측 결과 불일치",
    )

    return {
        "equal": True,
        "expected_fingerprint": expected,
        "actual_fingerprint": actual,
    }


def parser() -> argparse.ArgumentParser:
    """M3 전용 실행 인자."""
    result = argparse.ArgumentParser(
        description="선수 미래 성적 예측",
    )

    subparsers = result.add_subparsers(
        dest="command",
        required=True,
    )

    training = subparsers.add_parser("train")
    training.add_argument("--project-root", type=Path)
    training.add_argument(
        "--data-run",
        default=data.INTAKE_RUN,
    )
    training.add_argument(
        "--role",
        required=True,
        choices=tuple(data.ROLES),
    )

    inference = subparsers.add_parser("predict")
    inference.add_argument("--asset", type=Path, required=True)
    inference.add_argument("--input", type=Path, required=True)
    inference.add_argument("--output", type=Path)

    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("--asset", type=Path, required=True)

    comparison = subparsers.add_parser("compare")
    comparison.add_argument("--expected", type=Path, required=True)
    comparison.add_argument("--actual", type=Path, required=True)

    return result


def main(argv: list[str] | None = None) -> int:
    """CLI 결과는 stdout, 운영 진단과 오류는 stderr에 기록한다."""
    args = parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        if args.command == "train":
            settings = load_settings(args.project_root)
            LOGGER.setLevel(settings.log_level)
            result = train(
                settings,
                args.data_run,
                args.role,
            )

        elif args.command == "predict":
            result = predict(
                args.asset,
                args.input,
                args.output,
            )

        elif args.command == "inspect":
            _, _, result = load_asset(args.asset)

        else:
            result = compare(
                args.expected,
                args.actual,
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
        sqlite3.Error,
        SourceError,
    ) as exc:
        LOGGER.error("M3 실행 실패: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())