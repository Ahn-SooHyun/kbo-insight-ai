"""고정 운영 예측기 6개의 사전 명세·최종 평가·재계산 CLI."""

import argparse
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import closing, contextmanager
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from itertools import groupby
import json
import logging
from pathlib import Path
import platform
import sqlite3
import subprocess
from tempfile import TemporaryDirectory
from uuid import uuid4

from . import final_scoring as scoring
from . import fixed_assets as assets
from ..datasets import model_events, model_players
from ..datasets.model_data import fingerprint
from ..datasets.published_sources import (
    as_row,
    read_json,
    sha256_file,
    write_json,
)
from ..models.game_outcome import __main__ as c1
from ..models.game_outcome import model as m1
from ..models.plate_appearance import model as m2
from ..models.plate_appearance import probability as p2
from ..models.plate_appearance import data as d2
from ..models.game_state_value import model as m4
from ..models.game_state_value import value as v4
from ..models.game_state_value import data as d4
from ..models.game_state_value import __main__ as c4
from ..models.pitch_outcome import model as m5
from ..models.pitch_outcome import data as d5
from ..models.pitch_outcome import __main__ as c5
from ..models.player_performance import model as m3
from ..models.player_performance import data as d3
from ..models.player_performance import __main__ as c3
from ..models.player_performance import development as dev3
from ..settings import Settings, load_settings


LOGGER = logging.getLogger(__name__)

VERSION = "1.0.0"

WORK_START_COMMIT = (
    "e82f0866c1243f01006eba21409693d8843ee189"
)

TEST_START = assets.BOUNDARIES[1]
OBSERVATION_END = assets.BOUNDARIES[2]

EVENT_MODELS = ("m1", "m2", "m4", "m5")

MODEL_NAMES = tuple(spec.name for spec in assets.ASSETS)

OUTPUT_DIRECTORY = "final_evaluation"

require = assets.require


def utc_now() -> str:
    """실행 감사에 사용하는 UTC 시각."""
    return datetime.now(timezone.utc).isoformat()


def git_read(root: Path, *args: str) -> str:
    """Git 상태를 읽기 전용 명령으로 조회한다."""
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    require(
        result.returncode == 0,
        f"Git 읽기 전용 검사 실패: {' '.join(args)}",
    )

    return result.stdout.strip()


def git_state(root: Path) -> dict[str, object]:
    """기준 커밋과 실제 평가 코드의 적용 HEAD를 기록한다."""
    head = git_read(root, "rev-parse", "HEAD")

    branch = git_read(root, "branch", "--show-current")

    require(
        branch == "test/34",
        f"최종 평가 작업 브랜치 불일치: {branch}",
    )

    git_read(
        root,
        "merge-base",
        "--is-ancestor",
        assets.BASE_COMMIT,
        head,
    )

    git_read(
        root,
        "merge-base",
        "--is-ancestor",
        WORK_START_COMMIT,
        head,
    )

    return {
        "base_commit": assets.BASE_COMMIT,
        "work_start_commit": WORK_START_COMMIT,
        "head": head,
        "branch": branch,
        "worktree_status": git_read(root, "status", "--short"),
    }


def code_inventory(root: Path) -> dict[str, str]:
    """실제 평가에 적용된 소스·설정·테스트·계약의 지문."""
    files = {
        *root.joinpath("src").rglob("*.py"),
        *root.joinpath("tests").rglob("*.py"),
        *root.joinpath("docs", "standards").glob("*.md"),
        root / "docs/planning/development_workflow.md",
        root / "kbo.toml",
        root / "pyproject.toml",
        root / "uv.lock",
    }

    require(bool(files), "실행 소스가 없습니다.")

    result = {}

    for path in sorted(files):
        resolved = path.resolve(strict=True)

        require(
            resolved.is_relative_to(root.resolve())
            and resolved.is_file(),
            f"실행 소스 경로 오류: {path}",
        )

        relative = path.relative_to(root).as_posix()
        result[relative] = sha256_file(resolved)

    return result


def environment(settings: Settings) -> dict[str, object]:
    """최종 실행 환경 및 설정을 고정한다."""
    require(
        platform.python_version_tuple()[:2] == ("3", "12"),
        "Python 3.12 환경이 필요합니다.",
    )

    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "sqlite": sqlite3.sqlite_version,
        "configured_seed": settings.seed,
        "device": "cpu",
        "new_operating_model_training": False,
        "test_calibration": False,
    }


def model_columns(name: str) -> tuple[str, ...]:
    """기존 운영 모델의 명시적 입력 열만 반환한다."""
    if name == "m1":
        return tuple(m1.INPUT_COLUMNS)

    if name == "m2":
        return tuple(m2.INPUT_COLUMNS)

    if name == "m4":
        return tuple(m4.INPUT_COLUMNS)

    if name == "m5":
        return tuple(m5.INPUT_COLUMNS)

    require(
        name in ("m3_hitter", "m3_pitcher"),
        "미지원 평가 모델",
    )

    return d3.input_columns(
        name.removeprefix("m3_")
    )


def get_asset(
    inspection: Mapping[str, object],
    name: str,
) -> dict[str, object]:
    """고정 자산 인수 결과에서 모델을 식별한다."""
    matches = [
        as_row(value)
        for value in inspection["assets"]
        if value["name"] == name
    ]

    require(
        len(matches) == 1,
        f"자산 식별 오류: {name}",
    )

    return matches[0]


def data_paths(
    inspection: Mapping[str, object],
) -> tuple[Path, Path, Path]:
    """고정 모델 DB·표준 DB·계약의 경로를 반환한다."""
    files = inspection["data"]["files"]

    return (
        Path(files["database"]["path"]),
        Path(files["standard"]["path"]),
        Path(files["contract"]["path"]),
    )


@contextmanager
def open_database(
    database: Path,
    standard: Path | None = None,
) -> Iterator[sqlite3.Connection]:
    """고정 원본을 읽기 전용 URI와 query_only로 연결한다."""
    with closing(
        sqlite3.connect(
            database.resolve(strict=True).as_uri() + "?mode=ro",
            uri=True,
        )
    ) as db:
        db.row_factory = sqlite3.Row

        if standard is not None:
            db.execute(
                "ATTACH DATABASE ? AS verified",
                (
                    standard.resolve(strict=True).as_uri()
                    + "?mode=ro",
                ),
            )

        db.execute("PRAGMA query_only=ON")

        require(
            db.execute("PRAGMA query_only").fetchone()[0] == 1,
            "읽기 전용 연결 설정 실패",
        )

        yield db


def prediction_authorizer(
    name: str,
    columns: Sequence[str],
):
    """추론 연결에서 명시적인 입력·식별 열만 읽도록 강제한다."""
    index_columns = {
        "sample_id",
        "game_id",
        "sample_date",
        "split",
        "status",
        "role",
        "last_observed_date",
        "player_id",
        "season",
    }

    allowed = {
        f"{name}_index": index_columns,
        f"{name}_inputs": {"sample_id", *columns},
    }

    def authorize(
        action: int,
        table: str | None,
        column: str | None,
        database: str | None,
        trigger: str | None,
    ) -> int:
        """정답·감사 열 및 쓰기 동작을 추론에서 차단한다."""
        if action == sqlite3.SQLITE_SELECT:
            return sqlite3.SQLITE_OK

        if (
            action == sqlite3.SQLITE_READ
            and database == "main"
            and table in allowed
            and column in allowed[table]
            and trigger is None
        ):
            return sqlite3.SQLITE_OK

        return sqlite3.SQLITE_DENY

    return authorize


def validate_day(day: str, split: str) -> None:
    """KST 날짜와 고정 Train·Validation·Test 소속을 확인한다."""
    parsed = date.fromisoformat(day)

    require(
        parsed.isoformat() == day
        and day < OBSERVATION_END,
        "평가 자료 날짜 형식 또는 종료 경계 오류",
    )

    expected = (
        "train"
        if day < assets.BOUNDARIES[0]
        else "validation"
        if day < TEST_START
        else "test"
    )

    require(
        split == expected,
        "평가 자료 날짜와 분할 불일치",
    )


def verify_keys(
    db: sqlite3.Connection,
    name: str,
    contract: Mapping[str, object],
) -> None:
    """정답 값이 아닌 키 및 스키마로 구조와 분할을 확인한다."""
    definition = as_row(contract[name])
    allowed = as_row(definition["inputs"])

    columns = model_columns(name)

    require(
        set(columns) <= set(allowed),
        f"{name}: 운영 입력 허용 계약 오류",
    )

    actual = {
        row[1]
        for row in db.execute(
            f"PRAGMA main.table_info({name}_inputs)"
        )
    }

    require(
        actual == {"sample_id", *allowed},
        f"{name}: 입력 테이블 전체 계약 불일치",
    )

    require(
        not any(
            column.startswith("target_")
            for column in actual
        ),
        f"{name}: 입력에 정답 열 존재",
    )

    if name == "m2":
        d2.verify_keys(db)

    elif name == "m4":
        d4.verify_keys(db)

    elif name == "m5":
        d5.verify_keys(db)

    elif name.startswith("m3_"):
        d3.verify_keys(
            db,
            name.removeprefix("m3_"),
            definition,
        )

    for table in (
        f"{name}_index",
        f"{name}_inputs",
    ):
        duplicate = db.execute(
            f"""
            SELECT 1
            FROM {table}
            GROUP BY sample_id
            HAVING sample_id IS NULL OR COUNT(*) <> 1
            LIMIT 1
            """
        ).fetchone()

        require(
            duplicate is None,
            f"{table}: 중복 또는 NULL 키",
        )

    if name in EVENT_MODELS:
        duplicate = db.execute(
            f"""
            SELECT 1
            FROM {name}_targets
            GROUP BY sample_id
            HAVING sample_id IS NULL OR COUNT(*) <> 1
            LIMIT 1
            """
        ).fetchone()

        require(
            duplicate is None,
            f"{name}: 중복 또는 NULL 정답 키",
        )

        for table in (
            f"{name}_inputs",
            f"{name}_targets",
        ):
            missing = db.execute(
                f"""
                SELECT 1
                FROM {name}_index x
                LEFT JOIN {table} t USING(sample_id)
                WHERE x.status='eligible'
                    AND t.sample_id IS NULL
                LIMIT 1
                """
            ).fetchone()

            orphan = db.execute(
                f"""
                SELECT 1
                FROM {table} t
                LEFT JOIN {name}_index x USING(sample_id)
                WHERE x.sample_id IS NULL
                    OR x.status<>'eligible'
                LIMIT 1
                """
            ).fetchone()

            require(
                missing is None and orphan is None,
                f"{name}: 입력·정답 키의 누락 또는 고아 행",
            )

        mixed = db.execute(
            f"""
            SELECT 1
            FROM {name}_index
            GROUP BY game_id
            HAVING COUNT(DISTINCT split) <> 1
                OR COUNT(DISTINCT sample_date) <> 1
            LIMIT 1
            """
        ).fetchone()

        require(
            mixed is None,
            f"{name}: 동일 경기 분할 교차",
        )

    for day, split in db.execute(
        f"""
        SELECT DISTINCT sample_date, split
        FROM {name}_index
        """
    ):
        validate_day(day, split)


def evaluation_policy(
    contract: Mapping[str, object],
) -> dict[str, object]:
    """최종 정답 점수화 전에 모든 평가 규칙을 고정한다."""
    return {
        "version": VERSION,
        "timezone": assets.TIMEZONE,
        "train_end_exclusive": assets.BOUNDARIES[0],
        "validation_start": assets.BOUNDARIES[0],
        "validation_end_exclusive": TEST_START,
        "test_start_inclusive": TEST_START,
        "test_end_exclusive": OBSERVATION_END,
        "competition": "regular",
        "postseason": "excluded_and_counted",
        "population": (
            "existing_model_index; upstream exclusions "
            "and unsupported scopes reported separately"
        ),
        "input_access": (
            "sqlite_authorizer; exact_model_columns; "
            "targets_and_post_event_audit_columns_denied"
        ),
        "prediction_before_targets": True,
        "target_join": (
            "unique sample_id; M3 sample_id plus horizon; "
            "strict sorted key equality"
        ),
        "source_audit": (
            "regenerate_existing_model_data_builder_in_isolated_db; "
            "compare_test_inputs_indexes_targets_and_M3_windows; "
            "not_an_independent_builder_implementation"
        ),
        "retrospective_availability_limit": (
            "original_publication_time_not_fully_reconstructed"
        ),
        "probability_metrics": {
            "multiclass_brier": (
                "mean_of_sum_of_squared_class_errors"
            ),
            "binary_brier": "mean_of_squared_probability_error",
            "log_loss": "mean_negative_natural_log_actual_probability",
            "m1_log_floor": 1e-15,
            "m2_log_floor": 1e-15,
            "m4_probability_clip": m4.EPSILON,
            "m5_zero_actual_probability": "fail_closed",
            "calibration_bins": {
                "m1": list(scoring.e1.BINS),
                "m2": 10,
                "m4": 10,
                "m5": 10,
            },
            "last_bin_includes_one": True,
            "top_label_ece": "sample_weighted_confidence_bin_error",
            "macro_classwise_ece": (
                "equal_class_weight_mean_of_one_vs_rest_ece"
            ),
            "m2_macro_f1": "fixed_class_dictionary_zero_division_zero",
        },
        "numeric_metrics": {
            "mae": "mean_absolute_error",
            "rmse": "sqrt_mean_squared_error",
            "m1_crps": (
                "sum_of_discrete_cdf_squared_errors;"
                "include_observed_score_outside_support"
            ),
            "m3_normalization": (
                "existing_train_record_scales;"
                "equal_weight_supported_record_mae_and_rmse"
            ),
        },
        "common_denominator": (
            "same_valid_target_and_predictions;"
            "ratio_requires_positive_denominator_for_both;"
            "interval_comparison_requires_both_intervals"
        ),
        "excluded_reason_accounting": (
            "exclusive_index_reasons;"
            "record_and_interval_diagnostics_have_separate_populations"
        ),
        "comparison": {
            "error_metrics": "lower_is_better",
            "macro_f1": "higher_is_better",
            "interval_coverage": (
                "closer_to_existing_nominal_coverage"
            ),
            "interval_width": (
                "smaller_width_diagnostic_not_standalone_quality_claim"
            ),
            "equal_tolerance": scoring.TOLERANCE,
        },
        "models": {
            name: {
                "input_columns": list(model_columns(name)),
                "input_and_target_contract": contract[name],
            }
            for name in MODEL_NAMES
        },
        "m1": {
            "method": c1.METHOD,
            "class_order": list(m1.OUTCOMES),
            "baseline": "train_site_empirical_pmf_and_result_frequency",
            "score_metrics": ["mae", "rmse", "crps"],
            "result_metrics": ["log_loss", "brier"],
            "joint_policy": "existing_diagonal_weighted_product",
            "tie_samples": "included",
        },
        "m2": {
            "method": p2.METHOD,
            "class_order": list(m2.CLASSES),
            "baseline": (
                "original_development_selected_global_or_conditional;"
                "restore_frequencies_from_train_only"
            ),
            "unknown_target": "fail_closed",
            "rule_target_conflict": "report_and_fail_contract",
        },
        "m4": {
            "method": c4.method_settings(),
            "baseline": "train_frequency_win_and_global_mean_re",
            "win_target": "same_attacking_team_win",
            "loss_or_tie_target": 0,
            "known_final": (
                "observed_verified_post_boundary;"
                "deterministic_state_result_not_future_model_input"
            ),
            "three_outs_re": 0,
            "nonterminal_metrics": "reported_separately",
        },
        "m5": {
            "method": c5.method_settings(),
            "joint_classes": list(m5.CLASSES),
            "pitch_type_classes": list(m5.TYPES),
            "pitch_result_classes": list(m5.RESULTS),
            "baselines": [
                "frequency",
                "conditional_frequency",
            ],
            "rare_support_threshold": scoring.e5.RARE_SUPPORT,
        },
        "m3": {
            "method": c3.method_settings(),
            "horizons": list(d3.HORIZONS),
            "baseline": dev3.BASELINE_NAME,
            "input_cutoff": "strictly_before_monday",
            "historical_end_date_in_input": False,
            "target_cutoff": OBSERVATION_END,
            "valid_statuses": sorted(d3.VALID_STATUSES),
            "record_missing": "mask_not_zero",
            "ratio_zero_denominator": "unavailable",
            "seven_day_ratio": "count_derived_diagnostic",
            "remaining_season": (
                "requires_verified_regular_season_completion"
            ),
            "interval_target_coverage": m3.INTERVAL_LEVEL,
            "interval_policy": "restore_existing_calibration",
        },
    }


def restore_baseline(
    settings: Settings,
    inspection: Mapping[str, object],
    spec: assets.AssetSpec,
) -> dict[str, object]:
    """기존 기준을 복원하거나 Train으로만 통계 기준을 계산한다."""
    info = get_asset(inspection, spec.name)

    model_path = Path(info["model_file"]["path"])

    report = as_row(
        read_json(
            Path(info["evaluation_file"]["path"])
        )
    )

    database, standard, _ = data_paths(inspection)

    operating = spec.loader(model_path)

    if spec.name == "m1":
        with open_database(database) as db:
            games, _ = c1.read_games(db)

        train = games["train"]

        n = len(train)

        require(n > 0, "M1 Train 경기 누락")

        site = operating["site_counts"]
        cap = operating["cap"]

        for index, side in enumerate(m1.SIDES):
            counts = Counter(
                game.runs[index]
                for game in train
            )

            require(
                site[side] == [
                    counts[value]
                    for value in range(cap + 1)
                ],
                "M1 저장 Train 득점 빈도 불일치",
            )

        counts = Counter(game.outcome for game in train)

        result_frequency = {
            label: counts[label] / n
            for label in m1.OUTCOMES
        }

        require(
            result_frequency
            == report["evaluation"]["baseline"][
                "train_outcome_frequency"
            ],
            "M1 기존 승무패 기준 불일치",
        )

        return {
            "name": "train_empirical_frequency",
            "train_games": n,
            "home_pmf": [
                count / n
                for count in site["home"]
            ],
            "away_pmf": [
                count / n
                for count in site["away"]
            ],
            "result_probabilities": result_frequency,
        }

    if spec.name == "m2":
        with open_database(database) as db:
            samples, ledger = d2.read_samples(db)

        d2.verify_intake(samples, ledger)

        train = samples["train"]

        global_base, conditional_base = m2.fit_frequencies(
            train
        )

        name = report["selection"]["baseline"]

        require(
            name in ("global", "conditional"),
            "M2 기존 기준 선택 기록 오류",
        )

        raw = (
            global_base
            if name == "global"
            else conditional_base
        )

        baseline = p2.make_model(raw, name)

        require(
            raw["counts"] == operating["base"]["counts"],
            "M2 운영 자산의 Train 범주 빈도 불일치",
        )

        return {
            "name": name,
            "train_samples": len(train),
            "train_counts": raw["counts"],
            "model": baseline,
        }

    if spec.name == "m4":
        with open_database(database, standard) as db:
            d4.verify_keys(db)
            d4.split_counts(db)
            pairs = d4.read_pairs(db)

        train = d4.observations(pairs["train"])

        frequency = m4.fit_frequency(train)
        re_table = v4.fit_re(train)

        require(
            re_table == operating["re_table"],
            "M4 저장 RE 통계와 Train 복원 결과 불일치",
        )

        return {
            "name": "frequency_win_and_global_re",
            "train_states": len(train),
            "predictor": frequency,
            "re_table": re_table,
        }

    if spec.name == "m5":
        with open_database(database, standard) as db:
            d5.verify_keys(db)
            train, summary, _ = d5.read_split(db, "train")

        frequency = m5.fit_frequency(train)

        conditional = m5.fit_frequency(
            train,
            conditional=True,
        )

        require(
            frequency["counts"]
            == conditional["counts"]
            == operating.model["counts"],
            "M5 운영 자산의 Train 공동 범주 빈도 불일치",
        )

        return {
            "name": "train_frequency_baselines",
            "train_samples": len(train),
            "train_summary": summary,
            "train_counts": frequency["counts"],
            "frequency": frequency,
            "conditional_frequency": conditional,
        }

    require(
        spec.family == "m3"
        and spec.role is not None,
        "M3 기준 자산 역할 오류",
    )

    fitted, _, _ = operating

    scales = dict(
        zip(
            d3.output_keys(spec.role),
            fitted.baseline.scales,
            strict=True,
        )
    )

    development = as_row(report["development"])

    require(
        scales == development["record_scales"],
        "M3 저장 Train 정규화 척도 불일치",
    )

    baseline_report = as_row(
        development["candidates"][dev3.BASELINE_NAME]
    )

    intervals = c3.validate_intervals(
        spec.role,
        baseline_report["calibration"],
    )

    return {
        "name": dev3.BASELINE_NAME,
        "role": spec.role,
        "baseline": asdict(fitted.baseline),
        "intervals": intervals,
        "scales": scales,
        "source": "existing_saved_train_baseline_and_development_intervals",
    }


def checked_file(
    info: Mapping[str, object],
    root: Path,
) -> Path:
    """실행 폴더 내 고정 파일의 경로·바이트 지문을 확인한다."""
    path = Path(info["path"]).resolve(strict=True)

    require(
        path.is_relative_to(root.resolve())
        and path.is_file()
        and path.stat().st_size == info["bytes"]
        and sha256_file(path) == info["sha256"],
        f"고정 실행 파일 불일치: {path.name}",
    )

    return path


def prepare(settings: Settings) -> dict[str, object]:
    """Test 정답을 점수화하기 전에 실행 명세를 신규 고정한다."""
    inspection = assets.inspect_assets(settings)

    state = git_state(settings.project_root)
    hashes = code_inventory(settings.project_root)
    runtime = environment(settings)

    _, _, contract_path = data_paths(inspection)

    contract = as_row(read_json(contract_path))

    run_id = uuid4().hex

    directory = (
        settings.results_dir
        / OUTPUT_DIRECTORY
        / run_id
    )

    directory.mkdir(parents=True, exist_ok=False)

    baseline_dir = directory / "baselines"
    baseline_dir.mkdir(exist_ok=False)

    baseline_files = {}

    try:
        for spec in assets.ASSETS:
            LOGGER.info(
                "Train 기준 복원: %s",
                spec.name,
            )

            baseline = restore_baseline(
                settings,
                inspection,
                spec,
            )

            path = baseline_dir / f"{spec.name}.json"

            write_json(path, baseline)

            baseline_files[spec.name] = (
                assets.describe_file(path)
            )

        require(
            code_inventory(settings.project_root) == hashes,
            "기준 복원 중 실행 소스 변경",
        )

        require(
            fingerprint(assets.inspect_assets(settings))
            == fingerprint(inspection),
            "기준 복원 중 고정 자산 변경",
        )

        manifest = {
            "schema": "fixed-final-evaluation",
            "version": VERSION,
            "status": "frozen",
            "run_id": run_id,
            "created_at_utc": utc_now(),
            "git": state,
            "runtime": runtime,
            "code_sha256": hashes,
            "assets": inspection,
            "policy": evaluation_policy(contract),
            "baselines": baseline_files,
            "final_test_scored": False,
        }

        manifest_path = directory / "manifest.json"

        write_json(manifest_path, manifest)

    except Exception as exc:
        write_json(
            directory / "prepare_failure.json",
            {
                "status": "failed",
                "run_id": run_id,
                "error": str(exc),
                "completed_baselines": baseline_files,
                "final_test_scored": False,
            },
        )

        raise

    return {
        "status": "frozen",
        "run_id": run_id,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "asset_count": len(assets.ASSETS),
        "baseline_count": len(baseline_files),
        "final_test_scored": False,
    }


def load_manifest(
    settings: Settings,
    path: Path,
    expected_sha256: str,
    *,
    verify_assets: bool,
) -> dict[str, object]:
    """외부에서 확정한 명세 지문과 현재 실행 계약을 검사한다."""
    path = path.resolve(strict=True)

    output_root = (
        settings.results_dir / OUTPUT_DIRECTORY
    ).resolve()

    require(
        path.name == "manifest.json"
        and path.is_relative_to(output_root)
        and sha256_file(path) == expected_sha256,
        "최종 평가 명세 SHA-256 또는 경로 불일치",
    )

    manifest = assets.read_unique_json(path)

    require(
        manifest["schema"] == "fixed-final-evaluation"
        and manifest["version"] == VERSION
        and manifest["status"] == "frozen"
        and manifest["run_id"] == path.parent.name
        and manifest["final_test_scored"] is False,
        "최종 평가 명세 상태 오류",
    )

    require(
        manifest["code_sha256"]
        == code_inventory(settings.project_root),
        "사전 고정 이후 평가 소스 변경",
    )

    require(
        manifest["runtime"] == environment(settings),
        "사전 고정 이후 평가 환경 변경",
    )

    require(
        set(manifest["baselines"]) == set(MODEL_NAMES),
        "사전 고정 기준 모델 누락",
    )

    for info in manifest["baselines"].values():
        checked_file(info, path.parent)

    if verify_assets:
        state = git_state(settings.project_root)

        require(
            state["head"] == manifest["git"]["head"]
            and state["branch"] == manifest["git"]["branch"],
            "사전 고정 이후 적용 HEAD 또는 브랜치 변경",
        )

        current_assets = assets.inspect_assets(settings)

        require(
            fingerprint(current_assets)
            == fingerprint(manifest["assets"]),
            "사전 고정 이후 데이터 또는 모델 자산 변경",
        )

        _, _, contract_path = data_paths(current_assets)

        require(
            fingerprint(
                evaluation_policy(
                    as_row(read_json(contract_path))
                )
            )
            == fingerprint(manifest["policy"]),
            "사전 고정 이후 평가 정책 변경",
        )

    return manifest


def load_baseline(
    manifest: Mapping[str, object],
    directory: Path,
    name: str,
) -> dict[str, object]:
    """명세에 고정된 기준 모델만 읽는다."""
    info = manifest["baselines"][name]

    path = checked_file(info, directory)

    return assets.read_unique_json(path)


def input_query(name: str) -> str:
    """최종 추론에 필요한 사전 입력과 식별자만 조회한다."""
    columns = ", ".join(
        f'i."{column}" AS "{column}"'
        for column in model_columns(name)
    )

    if name.startswith("m3_"):
        return f"""
            SELECT
                x.sample_id,
                x.role,
                x.sample_date,
                x.last_observed_date,
                x.split,
                {columns}
            FROM {name}_index x
            JOIN {name}_inputs i USING(sample_id)
            WHERE x.split='test'
                AND x.status='candidate'
                AND x.sample_date>=?
                AND x.sample_date<?
            ORDER BY x.sample_id
        """

    return f"""
        SELECT
            x.sample_id,
            x.game_id,
            x.sample_date,
            x.split,
            {columns}
        FROM {name}_index x
        JOIN {name}_inputs i USING(sample_id)
        WHERE x.split='test'
            AND x.status='eligible'
            AND i.competition='regular'
            AND x.sample_date>=?
            AND x.sample_date<?
        ORDER BY x.sample_id
    """


def make_predictor(
    spec: assets.AssetSpec,
    operating: object,
    baseline: Mapping[str, object],
):
    """저장 자산에서 모델별 반복 추론기를 구성한다."""
    name = spec.name

    if name == "m1":
        def predict(inputs: Mapping[str, object]) -> dict[str, object]:
            return {
                "operating": m1.predict(operating, inputs),
                "baseline": {
                    "home_pmf": baseline["home_pmf"],
                    "away_pmf": baseline["away_pmf"],
                    "result_probabilities": baseline[
                        "result_probabilities"
                    ],
                },
            }

        return predict

    if name == "m2":
        p2.validate_model(baseline["model"])

        def predict(inputs: Mapping[str, object]) -> dict[str, object]:
            return {
                "operating": p2.predict_probabilities(
                    operating, inputs
                ),
                "baseline": p2.predict_probabilities(
                    baseline["model"], inputs
                ),
            }

        return predict

    if name == "m4":
        def predict(inputs: Mapping[str, object]) -> dict[str, object]:
            state = m4.parse_state(inputs)

            return {
                "operating": {
                    "win": m4.predict_win(
                        operating["predictor"], state
                    ),
                    "re": v4.predict_re(
                        operating["re_table"], state
                    ),
                },
                "baseline": {
                    "win": m4.predict_win(
                        baseline["predictor"], state
                    ),
                    "re": v4.predict_re(
                        baseline["re_table"],
                        state,
                        global_only=True,
                    ),
                },
            }

        return predict

    if name == "m5":
        frequency = m5.Runtime(baseline["frequency"])

        conditional = m5.Runtime(
            baseline["conditional_frequency"]
        )

        def predict(inputs: Mapping[str, object]) -> dict[str, object]:
            keys = m5.encode(inputs)

            return {
                "operating": operating.probabilities(keys),
                "frequency": frequency.probabilities(keys),
                "conditional_frequency": (
                    conditional.probabilities(keys)
                ),
            }

        return predict

    fitted, intervals, _ = operating

    require(
        fingerprint(asdict(fitted.baseline))
        == fingerprint(baseline["baseline"]),
        "M3 사전 고정 기준 통계 변경",
    )

    reference = m3.Fitted(
        fitted.role,
        0,
        fitted.baseline,
        None,
    )

    def predict(inputs: Mapping[str, object]) -> dict[str, object]:
        player = d3.parse_input(inputs)

        require(
            player.role == fitted.role
            and player.known_end is None,
            "M3 역할 또는 사후 종료일 입력 오류",
        )

        return {
            "operating": m3.format_prediction(
                fitted, player, intervals
            )["periods"],
            "baseline": m3.format_prediction(
                reference,
                player,
                baseline["intervals"],
            )["periods"],
        }

    return predict


def write_rows(
    path: Path,
    rows: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    """고유 순서의 JSONL을 신규 작성하고 행 수와 지문을 보존한다."""
    count = 0
    previous: str | None = None

    with path.open(
        "x",
        encoding="utf-8",
        newline="\n",
    ) as stream:
        for row in rows:
            key = row["sample_id"]

            require(
                isinstance(key, str)
                and bool(key)
                and (previous is None or key > previous),
                "저장 표본 키 순서 또는 중복 오류",
            )

            previous = key

            stream.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )

            stream.write("\n")
            count += 1

    return {
        **assets.describe_file(path),
        "rows": count,
    }


def prediction_rows(
    database: Path,
    spec: assets.AssetSpec,
    operating: object,
    baseline: Mapping[str, object],
) -> Iterator[dict[str, object]]:
    """정답 접근이 차단된 연결에서 실제 최종 추론을 수행한다."""
    name = spec.name
    columns = model_columns(name)

    predict = make_predictor(
        spec,
        operating,
        baseline,
    )

    with open_database(database) as db:
        db.set_authorizer(
            prediction_authorizer(name, columns)
        )

        for row in db.execute(
            input_query(name),
            (TEST_START, OBSERVATION_END),
        ):
            validate_day(
                row["sample_date"],
                row["split"],
            )

            require(
                row["split"] == "test",
                "최종 추론에 다른 분할 포함",
            )

            inputs = {
                column: row[column]
                for column in columns
            }

            record = {
                "sample_id": row["sample_id"],
                "sample_date": row["sample_date"],
            }

            if spec.family == "m3":
                inputs.update({
                    "sample_id": row["sample_id"],
                    "role": row["role"],
                    "sample_date": row["sample_date"],
                    "last_observed_date": row[
                        "last_observed_date"
                    ],
                    "season_active": True,
                })

                player = d3.parse_input(inputs)

                require(
                    player.last_observed < player.anchor
                    and player.anchor.weekday() == 0,
                    "M3 기준 월요일 이전 입력 조건 위반",
                )

            else:
                record["game_id"] = row["game_id"]

                require(
                    inputs["competition"] == "regular",
                    "정규시즌 외 표본 유입",
                )

            record["input"] = inputs
            record.update(predict(inputs))

            yield record


def compare_sql_rows(
    db: sqlite3.Connection,
    left: str,
    right: str,
) -> dict[str, object]:
    """같은 키와 스키마의 자료를 양방향으로 대조한다."""
    left_count = db.execute(
        f"SELECT COUNT(*) FROM ({left})"
    ).fetchone()[0]

    right_count = db.execute(
        f"SELECT COUNT(*) FROM ({right})"
    ).fetchone()[0]

    left_only = db.execute(
        f"""
        SELECT COUNT(*)
        FROM ({left} EXCEPT {right})
        """
    ).fetchone()[0]

    right_only = db.execute(
        f"""
        SELECT COUNT(*)
        FROM ({right} EXCEPT {left})
        """
    ).fetchone()[0]

    return {
        "rebuilt_rows": left_count,
        "locked_rows": right_count,
        "rebuilt_only": left_only,
        "locked_only": right_only,
        "equal": (
            left_count == right_count
            and left_only == 0
            and right_only == 0
        ),
    }


def final_table_query(
    schema: str,
    name: str,
    part: str,
) -> str:
    """원천 재생성 감사의 Test 범위를 명시한다."""
    require(
        schema in ("main", "locked")
        and name in MODEL_NAMES
        and part in ("index", "inputs", "targets"),
        "재생성 감사 테이블 식별 오류",
    )

    if part == "index":
        return (
            f"SELECT * FROM {schema}.{name}_index "
            "WHERE split='test'"
        )

    return f"""
        SELECT t.*
        FROM {schema}.{name}_{part} t
        JOIN {schema}.{name}_index i USING(sample_id)
        WHERE i.split='test'
    """


def audit_source_rebuild(
    database: Path,
    standard: Path,
    run_directory: Path,
) -> dict[str, object]:
    """기존 원천 변환 계약을 별도 임시 DB에서 재생성·대조한다.

    고정 DB에는 쓰지 않는다. 이 함수는 모든 사전 추론을
    완료한 뒤에만 호출된다.
    """
    reports = {}

    with TemporaryDirectory(
        prefix="source_audit_",
        dir=run_directory,
    ) as directory:
        scratch = Path(directory) / "rebuilt.sqlite3"

        with closing(
            sqlite3.connect(scratch, uri=True)
        ) as db:
            db.row_factory = sqlite3.Row

            db.execute(
                "ATTACH DATABASE ? AS source",
                (
                    standard.resolve().as_uri()
                    + "?mode=ro",
                ),
            )

            db.execute(
                "ATTACH DATABASE ? AS locked",
                (
                    database.resolve().as_uri()
                    + "?mode=ro",
                ),
            )

            db.execute(
                """
                CREATE TEMP TABLE config(
                    validation_start TEXT,
                    test_start TEXT,
                    observation_end TEXT
                )
                """
            )

            db.execute(
                "INSERT INTO config VALUES (?, ?, ?)",
                assets.BOUNDARIES,
            )

            require(
                db.execute(
                    "SELECT MAX(date) FROM source.eligible_games"
                ).fetchone()[0] < OBSERVATION_END,
                "표준 자료가 고정 관측 종료일을 초과",
            )

            model_events.build_events(db)

            model_players.build_players(
                db,
                OBSERVATION_END,
            )

            for name in MODEL_NAMES:
                for part in (
                    "index", "inputs", "targets",
                ):
                    table = f"{name}_{part}"

                    db.execute(
                        f"""
                        CREATE INDEX audit_{table}_sample
                        ON {table}(sample_id)
                        """
                    )

                db.execute(
                    f"""
                    CREATE INDEX audit_{name}_split
                    ON {name}_index(split)
                    """
                )

            for name in MODEL_NAMES:
                LOGGER.info("원천 재생성 대조: %s", name)

                tables = {
                    part: compare_sql_rows(
                        db,
                        final_table_query(
                            "main", name, part
                        ),
                        final_table_query(
                            "locked", name, part
                        ),
                    )
                    for part in (
                        "index", "inputs", "targets"
                    )
                }

                if name.startswith("m3_"):
                    role = name.removeprefix("m3_")

                    tables["window_games"] = (
                        compare_sql_rows(
                            db,
                            """
                            SELECT *
                            FROM main.m3_window_games
                            WHERE season=2026
                            """,
                            """
                            SELECT *
                            FROM locked.m3_window_games
                            WHERE season=2026
                            """,
                        )
                    )

                    left = f"""
                        SELECT t.*
                        FROM main.m3_window_teams t
                        JOIN main.{name}_index i USING(sample_id)
                        WHERE i.split='test'
                            AND t.role='{role}'
                    """

                    right = f"""
                        SELECT t.*
                        FROM locked.m3_window_teams t
                        JOIN locked.{name}_index i USING(sample_id)
                        WHERE i.split='test'
                            AND t.role='{role}'
                    """

                    tables["window_teams"] = (
                        compare_sql_rows(db, left, right)
                    )

                reports[name] = {
                    "tables": tables,
                    "equal": all(
                        value["equal"]
                        for value in tables.values()
                    ),
                }

    return {
        "method": "same_builder_read_only_source_reconstruction",
        "independent_builder": False,
        "original_database_modified": False,
        "models": reports,
    }


def m4_final_query() -> str:
    """M4 개발용 로더와 분리된 최종 Test 원천 감사 질의."""
    selected = ", ".join(
        f"x.{name} AS {name}"
        for name in m4.INPUT_COLUMNS
    )

    raw_state = ", ".join(
        f"""
        CASE WHEN i.phase='pre'
            THEN p.pre_{name}
            ELSE p.post_{name}
        END AS raw_{name}
        """
        for name in ("home_score", "away_score", "outs")
    )

    raw_bases = ", ".join(
        f"""
        CASE WHEN i.phase='pre'
            THEN p.pre_on_{base} IS NOT NULL
            ELSE p.post_on_{base} IS NOT NULL
        END AS raw_{name}
        """
        for base, name in (
            ("1b", "on_first"),
            ("2b", "on_second"),
            ("3b", "on_third"),
        )
    )

    return f"""
        WITH final_games AS (
            SELECT DISTINCT game_id
            FROM m4_index
            WHERE split='test'
                AND status='eligible'
        ),
        last_game AS (
            SELECT
                p.game_id,
                MAX(p.pa_number) AS last_pa
            FROM verified.plate_appearances p
            JOIN final_games d USING(game_id)
            GROUP BY p.game_id
        ),
        last_half AS (
            SELECT
                p.game_id,
                p.inning,
                p.half,
                MAX(p.pa_number) AS last_half_pa
            FROM verified.plate_appearances p
            JOIN final_games d USING(game_id)
            GROUP BY p.game_id, p.inning, p.half
        )
        SELECT
            i.sample_id AS index_sample_id,
            i.game_id,
            i.pair_id,
            i.pa_number,
            i.sample_date,
            i.split,
            i.phase AS index_phase,
            i.terminal_evidence,
            i.source_sha AS index_source_sha,
            i.start_source_row AS index_start_row,
            i.end_source_row AS index_end_row,
            {selected},
            t.target_result,
            t.target_remaining_runs,
            p.inning AS raw_inning,
            p.half AS raw_half,
            p.source_sha AS raw_source_sha,
            p.start_source_row AS raw_start_row,
            p.end_source_row AS raw_end_row,
            {raw_state},
            {raw_bases},
            g.season AS official_season,
            g.competition AS official_competition,
            g.date AS official_date,
            g.home_team_id AS official_home_team,
            g.away_team_id AS official_away_team,
            g.status AS official_status,
            g.home_score AS official_home_score,
            g.away_score AS official_away_score,
            g.winner,
            g.result_valid,
            c.final_score_match,
            c.order_valid,
            c.boundary_valid,
            c.terminal_state,
            lg.last_pa,
            lh.last_half_pa,
            h.post_outs AS half_end_outs,
            h.post_home_score AS half_end_home_score,
            h.post_away_score AS half_end_away_score
        FROM m4_index i
        JOIN m4_inputs x USING(sample_id)
        JOIN m4_targets t USING(sample_id)
        JOIN verified.eligible_states p
            ON p.game_id=i.game_id
            AND p.pa_number=i.pa_number
        JOIN verified.games g
            ON g.game_id=i.game_id
        JOIN verified.game_checks c
            ON c.game_id=i.game_id
        JOIN last_game lg
            ON lg.game_id=i.game_id
        JOIN last_half lh
            ON lh.game_id=p.game_id
            AND lh.inning=p.inning
            AND lh.half=p.half
        JOIN verified.plate_appearances h
            ON h.game_id=lh.game_id
            AND h.pa_number=lh.last_half_pa
        WHERE i.split='test'
            AND i.status='eligible'
            AND g.competition='regular'
            AND i.sample_date>=?
            AND i.sample_date<?
        ORDER BY i.sample_id
    """


def match_target(
    prediction: Mapping[str, object],
    key: str,
) -> None:
    """정답 연결 단계에서 동일 예측 키를 확인한다."""
    require(
        prediction["sample_id"] == key,
        "저장 예측과 실제 정답의 키 불일치",
    )


def event_truth_rows(
    db: sqlite3.Connection,
    name: str,
    predictions: Path,
) -> Iterator[dict[str, object]]:
    """M1·M2의 공식 경기 또는 타석 정답을 사후 연결한다."""
    fields = (
        "t.target_home_runs, t.target_away_runs, t.target_result"
        if name == "m1"
        else "t.target_event"
    )

    rows = db.execute(
        f"""
        SELECT x.*, {fields}
        FROM {name}_index x
        JOIN {name}_inputs i USING(sample_id)
        JOIN {name}_targets t USING(sample_id)
        WHERE x.split='test'
            AND x.status='eligible'
            AND i.competition='regular'
            AND x.sample_date>=?
            AND x.sample_date<?
        ORDER BY x.sample_id
        """,
        (TEST_START, OBSERVATION_END),
    )

    for prediction, row in zip(
        scoring.read_rows(predictions),
        rows,
        strict=True,
    ):
        key = row["sample_id"]

        match_target(prediction, key)

        validate_day(row["sample_date"], row["split"])

        require(
            prediction["sample_date"] == row["sample_date"]
            and prediction["game_id"] == row["game_id"],
            "경기 또는 타석 날짜·키 연결 오류",
        )

        if name == "m1":
            require(
                key == row["game_id"],
                "M1 경기 키 불일치",
            )

            yield {
                "sample_id": key,
                "target": {
                    "home": row["target_home_runs"],
                    "away": row["target_away_runs"],
                    "result": row["target_result"],
                },
                "source_sha": row["source_sha"],
                "source_row": row["source_row"],
            }

            continue

        require(
            key == f"{row['game_id']}:{row['pa_number']}"
            and d2.exclusion_reason(row) is None
            and row["exclusion_reason"] is None
            and row["target_event"] == row["event"],
            "M2 타석 품질·정답 계약 오류",
        )

        inputs = as_row(prediction["input"])

        d2.verify_participants(row, inputs)

        require(
            row["no_actual_pitch"]
            == int(row["physical_pitch_rows"] == 0),
            "M2 무투구 감사 불일치",
        )

        if row["no_actual_pitch"]:
            require(
                row["target_event"] == "walk",
                "M2 유효 무투구 결과 불일치",
            )

        yield {
            "sample_id": key,
            "target": m2.target_class(row["target_event"]),
            "source_sha": row["source_sha"],
            "start_source_row": row["start_source_row"],
            "end_source_row": row["end_source_row"],
            "no_actual_pitch": bool(row["no_actual_pitch"]),
        }


def state_truth_rows(
    db: sqlite3.Connection,
    predictions: Path,
) -> Iterator[dict[str, object]]:
    """M4의 상태·공식 종료·반이닝 실득점을 대조한다."""
    rows = db.execute(
        m4_final_query(),
        (TEST_START, OBSERVATION_END),
    )

    pending: tuple[str, m4.Observation] | None = None

    for prediction, row in zip(
        scoring.read_rows(predictions),
        rows,
        strict=True,
    ):
        key = row["index_sample_id"]

        match_target(prediction, key)

        validate_day(row["sample_date"], row["split"])

        require(
            prediction["sample_date"] == row["sample_date"]
            and prediction["game_id"] == row["game_id"]
            and as_row(prediction["input"]) == {
                column: row[column]
                for column in m4.INPUT_COLUMNS
            },
            "M4 저장 추론 입력의 원천 대조 실패",
        )

        observation, evidence = d4.read_observation(row)

        if observation.state.phase == "post":
            require(
                pending is None,
                "M4 타석 pre/post 연결 누락",
            )

            pending = (
                row["pair_id"],
                observation,
            )

        else:
            require(
                pending is not None
                and pending[0] == row["pair_id"]
                and evidence is None,
                "M4 타석 경계 또는 종료 근거 오류",
            )

            post = pending[1]

            actual_runs = m4.validate_pair(
                observation.state,
                post.state,
            )

            require(
                observation.win == post.win
                and observation.remaining - post.remaining
                == actual_runs,
                "M4 타석 전후 실득점·승리 정답 불일치",
            )

            pending = None

        yield {
            "sample_id": key,
            "win": observation.win,
            "remaining": observation.remaining,
            "confirmed_final": observation.final is not None,
            "final_evidence": (
                asdict(evidence)
                if evidence is not None
                else None
            ),
            "source_sha": row["raw_source_sha"],
            "start_source_row": row["raw_start_row"],
            "end_source_row": row["raw_end_row"],
        }

    require(
        pending is None,
        "M4 마지막 타석 경계 누락",
    )


def pitch_truth_rows(
    db: sqlite3.Connection,
    predictions: Path,
) -> Iterator[dict[str, object]]:
    """M5 실제 공의 원천·순번·직전 공·현재 정답을 대조한다."""
    query = d5.row_query()

    rows = db.execute(
        f"""
        SELECT *
        FROM ({query})
        ORDER BY sample_id
        """,
        ("test",),
    )

    for prediction, raw in zip(
        scoring.read_rows(predictions),
        rows,
        strict=True,
    ):
        row = dict(raw)

        key = row["sample_id"]

        match_target(prediction, key)

        validate_day(row["sample_date"], "test")

        require(
            prediction["sample_date"] == row["sample_date"]
            and prediction["game_id"] == row["game_id"]
            and as_row(prediction["input"]) == {
                column: row[column]
                for column in m5.INPUT_COLUMNS
            },
            "M5 저장 추론 입력의 원천 대조 실패",
        )

        d5.audit_current(row)

        previous_type, previous_result, reason, gap = (
            d5.audit_previous(row)
        )

        require(
            row["previous_pitch_type"] == previous_type
            and row["previous_pitch_result"] == previous_result,
            "M5 직전 실제 공 정보 불일치",
        )

        target = m5.target_index(
            row["target_pitch_type"],
            row["target_pitch_result"],
            row["target_joint"],
        )

        yield {
            "sample_id": key,
            "target": target,
            "joint_label": m5.CLASSES[target],
            "source_sha": row["source_sha"],
            "source_row": row["source_row"],
            "previous_pitch_reason": reason,
            "previous_pitch_gap": bool(gap),
        }


def player_truth_rows(
    db: sqlite3.Connection,
    name: str,
    predictions: Path,
) -> Iterator[dict[str, object]]:
    """M3 기준 월요일 및 기간별 정답을 연결한다."""
    season = db.execute(
        """
        SELECT MAX(date)
        FROM verified.eligible_games
        WHERE season=2026
            AND competition='regular'
        GROUP BY season
        HAVING MIN(COALESCE(season_complete, 0))=1
        """
    ).fetchone()

    end = (
        (
            date.fromisoformat(season[0])
            + timedelta(days=1)
        ).isoformat()
        if season is not None
        else None
    )

    rows = db.execute(
        f"""
        SELECT t.*
        FROM {name}_targets t
        JOIN {name}_index x USING(sample_id)
        WHERE x.split='test'
            AND x.sample_date>=?
            AND x.sample_date<?
        ORDER BY t.sample_id, t.horizon
        """,
        (TEST_START, OBSERVATION_END),
    )

    grouped = groupby(
        rows,
        key=lambda row: row["sample_id"],
    )

    for prediction, group in zip(
        scoring.read_rows(predictions),
        grouped,
        strict=True,
    ):
        key, raw = group

        match_target(prediction, key)

        records = [dict(row) for row in raw]

        by_horizon = {
            row["horizon"]: row
            for row in records
        }

        require(
            len(records) == len(d3.HORIZONS)
            and set(by_horizon) == set(d3.HORIZONS),
            "M3 기간별 정답 키 중복 또는 누락",
        )

        player = d3.parse_input(prediction["input"])

        targets = tuple(
            scoring.parse_final_target(
                by_horizon[horizon],
                player,
            )
            for horizon in d3.HORIZONS
        )

        d3.verify_target_relationships(targets)

        require(
            (
                targets[-1].window_end.isoformat()
                if targets[-1].window_end is not None
                else None
            ) == end,
            "M3 공식 정규시즌 종료 근거 불일치",
        )

        yield {
            "sample_id": key,
            "season_end_exclusive": end,
            "targets": by_horizon,
        }


def truth_rows(
    database: Path,
    standard: Path,
    name: str,
    predictions: Path,
) -> Iterator[dict[str, object]]:
    """저장 추론 이후의 정답 연결 전용 경로."""
    with open_database(database, standard) as db:
        if name in ("m1", "m2"):
            yield from event_truth_rows(
                db, name, predictions
            )

        elif name == "m4":
            yield from state_truth_rows(
                db, predictions
            )

        elif name == "m5":
            yield from pitch_truth_rows(
                db, predictions
            )

        else:
            yield from player_truth_rows(
                db, name, predictions
            )


def pitch_exclusions(
    db: sqlite3.Connection,
) -> dict[str, object]:
    """M5 표준 실제 공과 무투구·제외 사유를 구분한다."""
    reasons = dict(
        db.execute(
            """
            SELECT
                CASE
                    WHEN g.result_valid<>1
                        OR g.status<>'Final'
                        THEN 'game_excluded'
                    WHEN a.game_id IS NULL
                        THEN 'pa_state_excluded'
                    WHEN p.pitch_number<1
                        THEN 'invalid_pitch_number'
                    WHEN p.order_valid<>1
                        THEN 'invalid_order'
                    WHEN p.state_valid<>1
                        THEN 'invalid_state'
                    WHEN p.target_valid<>1
                        THEN 'invalid_target'
                    ELSE 'eligible'
                END AS reason,
                COUNT(*)
            FROM verified.pitches p
            JOIN verified.games g USING(game_id)
            LEFT JOIN verified.eligible_states a
                USING(game_id, pa_number)
            WHERE g.competition='regular'
                AND g.date>=?
                AND g.date<?
            GROUP BY reason
            """,
            (TEST_START, OBSERVATION_END),
        )
    )

    non_pitch = db.execute(
        """
        SELECT COUNT(*)
        FROM verified.non_pitch p
        JOIN verified.games g USING(game_id)
        WHERE g.competition='regular'
            AND g.date>=?
            AND g.date<?
        """,
        (TEST_START, OBSERVATION_END),
    ).fetchone()[0]

    raw = db.execute(
        """
        SELECT
            COUNT(*),
            COALESCE(SUM(c.positive_pitch_rows), 0),
            COALESCE(SUM(c.non_pitch_rows), 0)
        FROM verified.game_checks c
        JOIN verified.games g USING(game_id)
        WHERE g.competition='regular'
            AND g.date>=?
            AND g.date<?
        """,
        (TEST_START, OBSERVATION_END),
    ).fetchone()

    checked_stored = db.execute(
        """
        SELECT COUNT(*)
        FROM verified.pitches p
        JOIN verified.game_checks c USING(game_id)
        JOIN verified.games g USING(game_id)
        WHERE g.competition='regular'
            AND g.date>=?
            AND g.date<?
        """,
        (TEST_START, OBSERVATION_END),
    ).fetchone()[0]

    return {
        "standard_pitch_rows": sum(reasons.values()),
        "eligible": reasons.get("eligible", 0),
        "excluded": (
            sum(reasons.values())
            - reasons.get("eligible", 0)
        ),
        "exclusive_reasons": reasons,
        "non_pitch_rows": non_pitch,
        "raw_checked_games": raw[0],
        "raw_positive_pitch_rows": raw[1],
        "raw_non_pitch_rows": raw[2],
        "checked_standard_pitch_rows": checked_stored,
        "raw_minus_standard_pitch_rows": (
            raw[1] - checked_stored
        ),
    }


def population_report(
    database: Path,
    standard: Path,
    name: str,
) -> dict[str, object]:
    """최종 평가 인덱스의 모집단 및 배타적 제외 사유를 집계한다."""
    with open_database(database, standard) as db:
        if name.startswith("m3_"):
            n = db.execute(
                f"""
                SELECT COUNT(*)
                FROM {name}_index
                WHERE split='test'
                """
            ).fetchone()[0]

            return {
                "index_total": n,
                "selected": n,
                "index_excluded": 0,
                "scope": "regular_observed_player_role_week",
                "record_exclusions": "see_period_record_metrics",
            }

        reason = (
            "x.exclusion_reason"
            if name == "m2"
            else "x.status"
        )

        rows = db.execute(
            f"""
            SELECT
                g.competition,
                x.status,
                {reason} AS reason,
                COUNT(*) AS n
            FROM {name}_index x
            LEFT JOIN verified.eligible_games g
                ON x.game_id=g.game_id
            WHERE x.split='test'
            GROUP BY g.competition, x.status, reason
            """
        )

        total = 0
        selected = 0
        exclusions = Counter()
        competitions = Counter()

        for competition, status, reason, n in rows:
            require(
                competition in ("regular", "postseason"),
                "평가 인덱스와 경기 대회 구분 연결 실패",
            )

            total += n
            competitions[competition] += n

            if competition != "regular":
                exclusions["postseason"] += n

            elif status != "eligible":
                exclusions[str(reason or status)] += n

            else:
                selected += n

        require(
            total == selected + sum(exclusions.values()),
            "인덱스 제외 사유 분모 불일치",
        )

        result = {
            "index_total": total,
            "selected": selected,
            "index_excluded": total - selected,
            "competition_counts": dict(competitions),
            "exclusive_exclusion_reasons": dict(exclusions),
        }

        if name == "m5":
            source = pitch_exclusions(db)

            require(
                source["eligible"] == selected,
                "M5 표준 실제 공과 모델 모집단 불일치",
            )

            result["source_pitch_population"] = source

        return result


def run_evaluation(
    settings: Settings,
    manifest_path: Path,
    manifest_sha256: str,
) -> dict[str, object]:
    """명세를 재검증하고 6개 모델의 추론 및 실제 점수화를 수행한다."""
    manifest_path = manifest_path.resolve(strict=True)

    manifest = load_manifest(
        settings,
        manifest_path,
        manifest_sha256,
        verify_assets=True,
    )

    directory = manifest_path.parent

    run_directory = directory / "evaluation"

    # 최초 실행을 덮어쓰거나 같은 경로에서 재시도하지 않는다.
    run_directory.mkdir(exist_ok=False)

    database, standard, contract_path = data_paths(
        manifest["assets"]
    )

    contract = as_row(read_json(contract_path))

    write_json(
        run_directory / "started.json",
        {
            "run_id": manifest["run_id"],
            "started_at_utc": utc_now(),
            "manifest_sha256": manifest_sha256,
            "final_test_scored": False,
        },
    )

    reports = {}

    # 첫 번째 단계: 모든 운영 예측기의 실제 추론만 수행한다.
    # 이 단계에서는 Test 정답 열을 추론 연결에서 조회할 수 없다.
    for spec in assets.ASSETS:
        name = spec.name

        model_directory = run_directory / name

        model_directory.mkdir(exist_ok=False)

        LOGGER.info("최종 추론 시작: %s", name)

        try:
            with open_database(database) as db:
                verify_keys(db, name, contract)

            info = get_asset(manifest["assets"], name)

            operating = spec.loader(
                Path(info["model_file"]["path"])
            )

            baseline = load_baseline(
                manifest,
                directory,
                name,
            )

            predictions = write_rows(
                model_directory / "predictions.jsonl",
                prediction_rows(
                    database,
                    spec,
                    operating,
                    baseline,
                ),
            )

            require(
                predictions["rows"] > 0,
                f"{name}: 최종 추론 모집단 없음",
            )

            reports[name] = {
                "status": "predicted",
                "predictions": predictions,
                "prediction_samples": predictions["rows"],
            }

            LOGGER.info(
                "최종 추론 완료: %s, rows=%d",
                name,
                predictions["rows"],
            )

        except Exception as exc:
            LOGGER.error(
                "%s 최종 추론 실패: %s",
                name,
                exc,
            )

            reports[name] = {
                "status": "prediction_failed",
                "error": str(exc),
            }

    write_json(
        run_directory / "prediction_stage.json",
        {
            "models": reports,
            "target_values_used_for_prediction": False,
            "final_test_scored": False,
        },
    )

    # 두 번째 단계: 모든 추론 시도 종료 후 원천 자료를 대조한다.
    try:
        audit = audit_source_rebuild(
            database,
            standard,
            run_directory,
        )

    except Exception as exc:
        LOGGER.error(
            "원천 재생성 감사 실패: %s",
            exc,
        )

        audit = {
            "status": "failed",
            "error": str(exc),
            "models": {},
        }

    audit_path = run_directory / "leakage_audit.json"

    write_json(audit_path, audit)

    # 세 번째 단계: 실제 정답을 키로 연결하고 저장 자료로 평가한다.
    for spec in assets.ASSETS:
        name = spec.name

        record = reports[name]

        if record["status"] != "predicted":
            continue

        model_directory = run_directory / name

        LOGGER.info("최종 점수화 시작: %s", name)

        try:
            require(
                audit["models"].get(name, {}).get("equal")
                is True,
                f"{name}: 원천 입력·정답 재생성 대조 실패",
            )

            predictions_path = checked_file(
                record["predictions"],
                model_directory,
            )

            targets = write_rows(
                model_directory / "targets.jsonl",
                truth_rows(
                    database,
                    standard,
                    name,
                    predictions_path,
                ),
            )

            record["targets"] = targets

            population = population_report(
                database,
                standard,
                name,
            )

            require(
                targets["rows"]
                == record["prediction_samples"]
                == population["selected"],
                f"{name}: 실제 추론·정답·모집단 분모 불일치",
            )

            baseline = load_baseline(
                manifest,
                directory,
                name,
            )

            metrics = scoring.score(
                name,
                predictions_path,
                Path(targets["path"]),
                baseline,
            )

            metrics_path = (
                model_directory / "metrics.json"
            )

            write_json(metrics_path, metrics)

            record["metrics"] = assets.describe_file(
                metrics_path
            )

            record["metrics_fingerprint"] = fingerprint(
                metrics
            )

            record["population"] = population

            record["status"] = (
                "completed"
                if metrics.get("contract_valid", True)
                else "contract_failed"
            )

            LOGGER.info(
                "최종 점수화 종료: %s, status=%s",
                name,
                record["status"],
            )

        except Exception as exc:
            LOGGER.error(
                "%s 최종 점수화 실패: %s",
                name,
                exc,
            )

            record["status"] = "evaluation_failed"
            record["error"] = str(exc)

    integrity_error = None

    try:
        load_manifest(
            settings,
            manifest_path,
            manifest_sha256,
            verify_assets=True,
        )

    except Exception as exc:
        integrity_error = str(exc)

        LOGGER.error(
            "최종 평가 종료 지문 재검사 실패: %s",
            integrity_error,
        )

    completed = sum(
        report["status"] == "completed"
        for report in reports.values()
    )

    status = (
        "completed"
        if completed == len(MODEL_NAMES)
        and integrity_error is None
        else "failed_or_partial"
    )

    result = {
        "status": status,
        "run_id": manifest["run_id"],
        "finished_at_utc": utc_now(),
        "manifest_sha256": manifest_sha256,
        "model_count": len(MODEL_NAMES),
        "completed_models": completed,
        "models": reports,
        "audit": assets.describe_file(audit_path),
        "prediction_stage": assets.describe_file(
            run_directory / "prediction_stage.json"
        ),
        "post_run_integrity_error": integrity_error,
        "new_operating_models": 0,
        "final_test_used_for_selection": False,
        "final_test_used_for_refit": False,
        "final_test_used_for_calibration": False,
        "recalculation_required": True,
    }

    result_path = run_directory / "result.json"

    write_json(result_path, result)

    return {
        "status": status,
        "run_id": manifest["run_id"],
        "result": str(result_path),
        "result_sha256": sha256_file(result_path),
        "completed_models": completed,
        "models": {
            name: {
                "status": record["status"],
                "prediction_samples": record.get(
                    "prediction_samples"
                ),
                "error": record.get("error"),
                "metrics": record.get("metrics"),
            }
            for name, record in reports.items()
        },
        "post_run_integrity_error": integrity_error,
    }


def verify_evaluation(
    settings: Settings,
    manifest_path: Path,
    manifest_sha256: str,
    result_sha256: str,
) -> dict[str, object]:
    """저장한 예측·정답·마스크만으로 전체 지표를 재계산한다."""
    manifest_path = manifest_path.resolve(strict=True)

    manifest = load_manifest(
        settings,
        manifest_path,
        manifest_sha256,
        verify_assets=False,
    )

    directory = manifest_path.parent

    run_directory = directory / "evaluation"

    result_path = run_directory / "result.json"

    require(
        sha256_file(result_path) == result_sha256,
        "저장 평가 결과 지문 불일치",
    )

    result = assets.read_unique_json(result_path)

    require(
        result["run_id"] == manifest["run_id"]
        and result["manifest_sha256"] == manifest_sha256
        and set(result["models"]) == set(MODEL_NAMES),
        "저장 평가와 명세 식별 계약 불일치",
    )

    checked_file(
        result["audit"],
        run_directory,
    )

    checked_file(
        result["prediction_stage"],
        run_directory,
    )

    reports = {}

    for name in MODEL_NAMES:
        record = result["models"][name]

        if record["status"] not in (
            "completed",
            "contract_failed",
        ):
            reports[name] = {
                "status": "not_recalculable",
                "evaluation_status": record["status"],
            }

            continue

        model_directory = run_directory / name

        try:
            predictions = checked_file(
                record["predictions"],
                model_directory,
            )

            targets = checked_file(
                record["targets"],
                model_directory,
            )

            metrics_path = checked_file(
                record["metrics"],
                model_directory,
            )

            baseline = load_baseline(
                manifest,
                directory,
                name,
            )

            old = as_row(read_json(metrics_path))

            recalculated = scoring.score(
                name,
                predictions,
                targets,
                baseline,
            )

            expected = record["metrics_fingerprint"]
            actual = fingerprint(recalculated)

            require(
                fingerprint(old) == expected == actual,
                f"{name}: 저장 평가 지표 재계산 불일치",
            )

            reports[name] = {
                "status": "matched",
                "evaluation_status": record["status"],
                "metrics_fingerprint": actual,
                "prediction_samples": record[
                    "prediction_samples"
                ],
            }

        except Exception as exc:
            reports[name] = {
                "status": "failed",
                "error": str(exc),
            }

    matched = sum(
        report["status"] == "matched"
        for report in reports.values()
    )

    status = (
        "verified"
        if matched == len(MODEL_NAMES)
        and result["status"] == "completed"
        else "failed_or_partial"
    )

    verification = {
        "status": status,
        "run_id": manifest["run_id"],
        "verified_at_utc": utc_now(),
        "manifest_sha256": manifest_sha256,
        "result_sha256": result_sha256,
        "matched_models": matched,
        "models": reports,
        "model_refit": False,
        "model_inference_repeated": False,
        "source": "saved_predictions_targets_and_fixed_metric_code",
    }

    output = (
        run_directory
        / f"recalculation-{uuid4().hex}.json"
    )

    write_json(output, verification)

    return {
        **verification,
        "verification": str(output),
        "verification_sha256": sha256_file(output),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """사전 명세, 최종 평가 및 저장 결과 재검사 CLI."""
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--project-root",
        required=True,
        type=Path,
    )

    commands = parser.add_subparsers(
        dest="command",
        required=True,
    )

    commands.add_parser("prepare")

    for name in ("run", "verify"):
        command = commands.add_parser(name)

        command.add_argument(
            "--manifest",
            required=True,
            type=Path,
        )

        command.add_argument(
            "--manifest-sha256",
            required=True,
        )

        if name == "verify":
            command.add_argument(
                "--result-sha256",
                required=True,
            )

    args = parser.parse_args(argv)

    try:
        settings = load_settings(args.project_root)

        logging.basicConfig(
            level=settings.log_level,
            format="%(levelname)s %(name)s: %(message)s",
        )

        if args.command == "prepare":
            result = prepare(settings)
            expected = "frozen"

        elif args.command == "run":
            result = run_evaluation(
                settings,
                args.manifest,
                args.manifest_sha256,
            )

            expected = "completed"

        else:
            result = verify_evaluation(
                settings,
                args.manifest,
                args.manifest_sha256,
                args.result_sha256,
            )

            expected = "verified"

        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        )

        return 0 if result["status"] == expected else 2

    except KeyboardInterrupt:
        LOGGER.warning("최종 평가 사용자 중단")
        return 3

    except Exception as exc:
        LOGGER.error("최종 평가 실패: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
