"""고정 시즌 전망·보조 분석의 종합 검증 실행."""

import argparse
from collections.abc import Callable, Sequence
from contextlib import closing
from datetime import date, datetime, timezone
from hashlib import sha256
import json
import logging
from math import fsum, log
from pathlib import Path
import platform
import re
import sqlite3
import subprocess
import sys
from uuid import uuid4

from src.settings import Settings, load_settings
from src.simulation.regular_season.data import (
    open_readonly,
    read_completed,
    verify_assets,
)
from src.simulation.regular_season.engine import (
    build_state,
    team_result,
)
from src.simulation.regular_season.rules import (
    KBO_2025,
    resolve_standings,
)
from src.validation import season_forecast_checks as checks


LOGGER = logging.getLogger(__name__)

SCHEMA = "season-forecast-validation-v2"
LEGACY_SCHEMA = "season-forecast-validation-v1"

BASE_HEAD = "11f996b6ac04c0edac706d4b16599e49cc6ebdc5"
INITIAL_HEAD = "b2e5d2665984cc72dee6cc97e384b68c31976b38"

MODEL_DATA_RUN = "259b38decaac4ed68dd5a596cfbe4849"
STANDARD_RUN = "e020dafedc8b4fd9ac6696d0e4e4a92e"
STANDARD_VERIFICATION = "37b56c39caf5435e8fb2a17f993ec4df"
MODEL_RUN = "0d9433b8044e48e5a4d21bbdb9b432e5"

STANDARD_HASH = (
    "15632b3b796ce255f4bc5038e6f680b214385beb47b9e21ebd1bb790b7962aee"
)
MODEL_DATA_HASH = (
    "c8383a9d3e8930830a137d1b6791cbb26e2316dbbbdf4f0f436ceda9a43890d0"
)
MODEL_HASH = (
    "68c1ddeca8c54fed6359ec771f4026ee9802c73b49623c2d0076879da8f9da4a"
)

REFERENCE_REGULAR = "e24254420aee4d52acf2b1fb144c3d58"
REFERENCE_POSTSEASON = "8cd2a67f518047c8a77f765134345b21"
REFERENCE_ENDED = "2958768011784b3ba54e152e8cc6dc44"
REFERENCE_SUPPORT = "a6fd3f6a408a4a5a8316e282d52e86a2"
FINAL_EVALUATION = "7d125b8a685a4693a34aea70445ef294"

RESULT_DIRECTORY = "season_forecast_validation"

CUTOFF = "2025-09-01"
END_CUTOFF = "2025-11-01"
EARLY_CUTOFF = "2025-03-22"
REPRESENTATIVE_GAME = "20250902HTHH0"

EXPECTED_GAMES = 720
MIN_REGRESSION_TESTS = 657
LOG_LOSS_FLOOR = 1e-15

SIMULATION_CASES = (
    ("repeat", 42, 1000),
    ("long", 42, 2000),
    ("seed", 43, 1000),
)

RUNNER_PATH = "src/validation/season_forecast.py"
CHECKS_PATH = "src/validation/season_forecast_checks.py"
TEST_PATH = "tests/validation/test_season_forecast.py"
RESUME_TEST_PATH = "tests/validation/test_season_forecast_resume.py"
DOC_PATH = "docs/standards/season_forecast_validation.md"

REVISED_PATHS = frozenset({
    RUNNER_PATH,
    DOC_PATH,
})

ALLOWED_CHANGES = frozenset({
    RUNNER_PATH,
    CHECKS_PATH,
    TEST_PATH,
    RESUME_TEST_PATH,
    DOC_PATH,
})


def utc_now() -> str:
    """검증 증거의 생성 시각을 UTC로 기록한다."""
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: checks.JsonObject) -> None:
    """기존 증거를 덮어쓰지 않고 신규 JSON을 작성한다."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(
            data,
            stream,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        stream.write("\n")


def git_read(root: Path, *args: str) -> str:
    """Git 정보를 읽기 전용으로 조회한다."""
    process = subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )

    return process.stdout.strip()


def git_state(root: Path) -> checks.JsonObject:
    """실제 적용 기준과 Git 상태를 기록한다."""
    return {
        "head": git_read(root, "rev-parse", "HEAD"),
        "branch": git_read(root, "branch", "--show-current"),
        "status": git_read(root, "status", "--short", "--branch"),
        "origin_main": git_read(root, "rev-parse", "origin/main"),
    }


def verify_git_scope(root: Path) -> checks.JsonObject:
    """기준 커밋 계보 및 허용된 변경 범위를 검사한다."""
    state = git_state(root)

    git_read(
        root,
        "merge-base",
        "--is-ancestor",
        BASE_HEAD,
        "HEAD",
    )

    git_read(
        root,
        "merge-base",
        "--is-ancestor",
        INITIAL_HEAD,
        "HEAD",
    )

    changed = set(
        git_read(
            root,
            "diff",
            "--name-only",
            BASE_HEAD,
            "--",
        ).splitlines()
    )

    untracked = set(
        git_read(
            root,
            "ls-files",
            "--others",
            "--exclude-standard",
        ).splitlines()
    )

    changed.update(untracked)

    checks.require(
        changed <= ALLOWED_CHANGES,
        f"허용되지 않은 변경 파일: {sorted(changed - ALLOWED_CHANGES)}",
    )

    for name in ALLOWED_CHANGES:
        checks.require(
            (root / name).is_file(),
            f"필수 구현 파일 누락: {name}",
        )

    state["changed_files"] = sorted(changed)

    return state


def historical_hashes(root: Path, relative: str) -> set[str]:
    """중간 커밋의 원본 및 정확한 LF/CRLF 표현 지문을 반환한다."""
    checks.require(
        relative in REVISED_PATHS,
        f"이전 판본 대조 대상 오류: {relative}",
    )

    process = subprocess.run(
        ["git", "show", f"{INITIAL_HEAD}:{relative}"],
        cwd=root,
        capture_output=True,
        check=True,
    )

    original = process.stdout
    normalized = original.replace(b"\r\n", b"\n")
    windows = normalized.replace(b"\n", b"\r\n")

    return {
        sha256(original).hexdigest(),
        sha256(normalized).hexdigest(),
        sha256(windows).hexdigest(),
    }


def fixed_paths(settings: Settings) -> dict[str, Path]:
    """인수한 고정 자료와 기존 결과의 경로를 반환한다."""
    return {
        "standard_db": (
            settings.processed_dir
            / "standard"
            / STANDARD_RUN
            / f"verified-{STANDARD_VERIFICATION}.sqlite3"
        ),
        "model_db": (
            settings.processed_dir
            / "model_data"
            / MODEL_DATA_RUN
            / "model_data.sqlite3"
        ),
        "model_asset": (
            settings.models_dir
            / "m1"
            / MODEL_RUN
            / "model.json"
        ),
        "reference_regular": (
            settings.results_dir
            / "regular_season"
            / REFERENCE_REGULAR
        ),
        "reference_postseason": (
            settings.results_dir
            / "postseason"
            / REFERENCE_POSTSEASON
        ),
        "reference_ended": (
            settings.results_dir
            / "postseason"
            / REFERENCE_ENDED
        ),
        "reference_support": (
            settings.results_dir
            / "support_analysis"
            / REFERENCE_SUPPORT
        ),
        "final_evaluation": (
            settings.results_dir
            / "final_evaluation"
            / FINAL_EVALUATION
        ),
    }


def snapshot_paths(
    settings: Settings,
    paths: dict[str, Path],
) -> dict[str, str]:
    """코드·설정·고정 자료·기존 결과의 지문을 계산한다."""
    root = settings.project_root

    files = [
        paths["standard_db"],
        paths["model_db"],
        paths["model_asset"],
        paths["model_db"].parent / "manifest.json",
        root / "kbo.toml",
        root / "pyproject.toml",
        root / DOC_PATH,
        paths["final_evaluation"] / "manifest.json",
        paths["final_evaluation"] / "evaluation" / "result.json",
        *sorted((root / "src").rglob("*.py")),
        *sorted((root / "tests").rglob("*.py")),
    ]

    for name in (
        "reference_regular",
        "reference_postseason",
        "reference_ended",
        "reference_support",
    ):
        files.extend(
            path
            for path in paths[name].rglob("*")
            if path.is_file()
        )

    return {
        str(path.resolve()): checks.file_hash(path)
        for path in sorted(set(files))
    }


def verify_fixed_assets(
    settings: Settings,
    paths: dict[str, Path],
) -> checks.JsonObject:
    """고정 지문 및 기존 M1 자산 계보를 대조한다."""
    for key, expected in (
        ("standard_db", STANDARD_HASH),
        ("model_db", MODEL_DATA_HASH),
        ("model_asset", MODEL_HASH),
    ):
        checks.equal(
            checks.file_hash(paths[key]),
            expected,
            f"고정 자산 {key}",
        )

    standard, model, _, provenance = verify_assets(
        settings,
        MODEL_DATA_RUN,
        paths["model_asset"],
    )

    checks.equal(
        standard.resolve(),
        paths["standard_db"].resolve(),
        "표준 자료 계보 경로",
    )

    checks.equal(
        model.resolve(),
        paths["model_db"].resolve(),
        "모델 자료 계보 경로",
    )

    return provenance


def criteria() -> checks.JsonObject:
    """결과 확인 전에 고정하는 비교 기준."""
    return {
        "same_seed": "exact_calculation_output_equality",
        "expanded_iterations": "exact_iteration_prefix",
        "probability_tolerance": checks.TOLERANCE,
        "stability_min_valid": checks.STABILITY_MIN_VALID,
        "stability_multiplier": checks.STABILITY_MULTIPLIER,
        "stability_probability_floor": checks.STABILITY_PROBABILITY_FLOOR,
        "stability_mean_floor": checks.STABILITY_MEAN_FLOOR,
        "stability_warning_only": True,
        "formal_backtest_supported": False,
        "final_test_used_for_selection": False,
    }


def build_jobs(asset: Path) -> list[checks.JsonObject]:
    """실자료 실행과 최신 테스트의 조건을 고정한다."""
    jobs = []

    common = [
        "--run-id", MODEL_DATA_RUN,
        "--asset", str(asset),
    ]

    for label, seed, iterations in SIMULATION_CASES:
        regular = f"regular_{label}"

        jobs.append({
            "name": regular,
            "kind": "regular_season",
            "args": [
                "-m", "src.simulation.regular_season",
                "simulate",
                *common,
                "--season", "2025",
                "--as-of", CUTOFF,
                "--seed", str(seed),
                "--iterations", str(iterations),
                "--diagnostic",
            ],
        })

        jobs.append({
            "name": f"post_{label}",
            "kind": "postseason",
            "args": [
                "-m", "src.simulation.postseason",
                "simulate",
                *common,
                "--season", "2025",
                "--as-of", CUTOFF,
                "--seed", str(seed),
                "--parent-run", f"@parent:{regular}",
            ],
        })

    jobs.append({
        "name": "post_ended",
        "kind": "postseason",
        "args": [
            "-m", "src.simulation.postseason",
            "simulate",
            *common,
            "--season", "2025",
            "--as-of", END_CUTOFF,
            "--seed", "42",
            "--iterations", "1000",
            "--archive-assumption",
        ],
    })

    support = [
        "-m", "src.simulation.support_analysis",
        *common,
        "--output", "@output",
    ]

    scenario = [
        "--season", "2025",
        "--as-of", CUTOFF,
        "--parent-run", REFERENCE_REGULAR,
    ]

    for name, arguments in (
        ("support_evaluation", ["evaluate"]),
        ("support_season", ["season", *scenario]),
        (
            "support_game",
            [
                "game",
                *scenario,
                "--game-id",
                REPRESENTATIVE_GAME,
            ],
        ),
        (
            "support_early",
            [
                "season",
                "--season", "2025",
                "--as-of", EARLY_CUTOFF,
            ],
        ),
        (
            "support_late",
            [
                "season",
                "--season", "2025",
                "--as-of", END_CUTOFF,
            ],
        ),
    ):
        jobs.append({
            "name": name,
            "kind": "support",
            "args": [*support, *arguments],
        })

    jobs.extend([
        {
            "name": "validation_tests",
            "kind": "test",
            "args": [
                "-m", "unittest",
                "tests.validation.test_season_forecast",
                "tests.validation.test_season_forecast_resume",
                "-v",
            ],
        },
        {
            "name": "simulation_tests",
            "kind": "test",
            "args": [
                "-m", "unittest", "discover",
                "-s", "tests/simulation",
                "-v",
            ],
        },
        {
            "name": "regression",
            "kind": "test",
            "args": [
                "-m", "unittest", "discover",
                "-s", "tests",
            ],
        },
    ])

    return jobs


def verify_old_inputs(
    root: Path,
    previous: dict[str, str],
    current: dict[str, str],
) -> None:
    """수정 전 실행의 자료와 운영 코드 동일성을 확인한다."""
    added = str((root / RESUME_TEST_PATH).resolve())

    checks.equal(
        set(current) - set(previous),
        {added},
        "이전 계획 대비 신규 파일",
    )

    checks.require(
        set(previous) <= set(current),
        "이전 계획의 입력 파일 누락",
    )

    revised = {
        str((root / name).resolve()): name
        for name in REVISED_PATHS
    }

    for path, digest in previous.items():
        if path in revised:
            variants = historical_hashes(root, revised[path])

            checks.require(
                digest in variants,
                f"중간 커밋과 이전 검증 코드 지문 불일치: {path}",
            )

        else:
            checks.equal(
                current[path],
                digest,
                f"기존 실행 입력 변경: {path}",
            )


def record_path(directory: Path, name: str) -> Path:
    """작업별 성공 기록 경로."""
    return directory / "jobs" / f"{name}.json"


def read_plan(
    settings: Settings,
    manifest: Path,
    expected_hash: str,
    schema: str,
) -> checks.JsonObject:
    """계획의 경로·파일 지문·식별자를 검증한다."""
    manifest = manifest.resolve()
    allowed = (settings.results_dir / RESULT_DIRECTORY).resolve()

    checks.require(
        manifest.is_relative_to(allowed)
        and manifest.name == "plan.json"
        and manifest.parent.parent == allowed,
        "검증 계획 경로 오류",
    )

    checks.equal(
        checks.file_hash(manifest),
        expected_hash,
        "검증 계획 SHA-256",
    )

    plan = checks.read_json(manifest)

    checks.equal(plan["schema"], schema, "검증 계획 스키마")
    checks.equal(plan["status"], "prepared", "검증 계획 상태")
    checks.equal(plan["run_id"], manifest.parent.name, "계획 ID")

    return plan


def load_job(
    directory: Path,
    name: str,
    plan_hash: str,
) -> checks.JsonObject:
    """성공 기록과 산출물의 실제 파일 지문을 확인한다."""
    record = checks.read_json(record_path(directory, name))

    checks.check_fields(
        record,
        {
            "name": name,
            "plan_sha256": plan_hash,
            "returncode": 0,
            "status": "completed",
        },
        "작업 완료 기록",
    )

    artifacts = record["artifact_sha256"]

    checks.require(
        isinstance(artifacts, dict),
        "작업 산출물 지문 형식 오류",
    )

    if record["kind"] != "test":
        checks.require(bool(artifacts), "실자료 산출물 누락")

    for path, digest in artifacts.items():
        checks.equal(
            checks.file_hash(Path(path)),
            digest,
            f"작업 산출물 변경: {path}",
        )

    return record


def expected_command(
    job: checks.JsonObject,
    previous: dict[str, checks.JsonObject],
    output: Path | None = None,
) -> list[str]:
    """부모 실행 ID와 보조 분석 출력 경로를 해석한다."""
    arguments = []

    for value in job["args"]:
        if value.startswith("@parent:"):
            name = value.split(":", 1)[1]

            checks.require(
                name in previous,
                f"선행 작업 누락: {name}",
            )

            arguments.append(
                previous[name]["result"]["run_id"]
            )

        elif value == "@output":
            checks.require(output is not None, "보조 분석 출력 경로 누락")
            arguments.append(str(output))

        else:
            arguments.append(value)

    return [
        sys.executable,
        "-X", "utf8",
        "-B",
        *arguments,
    ]


def validate_test_result(
    name: str,
    stdout: str,
    stderr: str,
) -> checks.JsonObject:
    """실행된 테스트 수와 성공 상태를 확인한다."""
    output = stdout + "\n" + stderr
    matched = re.search(r"Ran (\d+) tests?", output)

    checks.require(matched is not None, "테스트 실행 수 확인 실패")

    count = int(matched.group(1))

    checks.require(
        count > 0,
        "실행된 테스트 없음",
    )

    checks.require(
        re.search(r"(?m)^OK\s*$", output) is not None,
        "테스트 성공 상태 또는 건너뛴 테스트 확인 필요",
    )

    if name == "regression":
        checks.require(
            count >= MIN_REGRESSION_TESTS,
            "기존 회귀 테스트 수 감소",
        )

    return {"tests": count}


def artifact_hashes(
    settings: Settings,
    directory: Path,
    kind: str,
    result: checks.JsonObject,
    provenance: checks.JsonObject,
) -> dict[str, str]:
    """실제 산출물 위치·계보·스키마·지문을 확인한다."""
    checks.equal(result["status"], "completed", "CLI 완료 상태")

    if kind == "support":
        output = Path(result["output"])

        checks.require(
            output.resolve().parent == (directory / "support").resolve(),
            "보조 분석 출력 경로 오류",
        )

        payload = checks.read_json(output)

        checks.check_fields(
            payload,
            {
                "status": "completed",
                "run_id": result["run_id"],
                "scope": result["scope"],
                "final_test_used": False,
            },
            "보조 분석 실행 결과",
        )

        checks.check_sources(
            settings.project_root,
            "support_analysis",
            payload["source_sha256"],
        )

        observed = (
            payload["provenance"]
            if payload["scope"] == "starter_validation"
            else payload["metadata"]["provenance"]
        )

        checks.equal(observed, provenance, "보조 분석 자료 계보")

        digest = checks.file_hash(output)
        checks.equal(digest, result["sha256"], "보조 분석 출력 지문")

        return {str(output): digest}

    checks.require(
        kind in ("regular_season", "postseason"),
        "지원하지 않는 작업 종류",
    )

    output = Path(result["result_dir"])
    expected_parent = (settings.results_dir / kind).resolve()

    checks.require(
        output.resolve().parent == expected_parent
        and output.name == result["run_id"],
        "시뮬레이션 결과 경로 오류",
    )

    run = checks.load_run(
        settings.project_root,
        output,
        kind,
    )

    for key in ("run_id", "seed", "iterations", "season", "as_of_kst"):
        checks.equal(
            result[key],
            run.manifest[key],
            f"실행 결과 및 매니페스트: {key}",
        )

    checks.equal(
        run.scenario["metadata"]["provenance"],
        provenance,
        "시뮬레이션 자료 계보",
    )

    artifacts = {
        str(output / filename): digest
        for filename, digest in run.digests.items()
    }

    manifest = output / "manifest.json"
    artifacts[str(manifest)] = checks.file_hash(manifest)

    return artifacts


def validate_job_record(
    settings: Settings,
    directory: Path,
    job: checks.JsonObject,
    record: checks.JsonObject,
    previous: dict[str, checks.JsonObject],
    provenance: checks.JsonObject,
) -> None:
    """기록의 실행 명령과 실제 결과 계약을 다시 대조한다."""
    checks.check_fields(
        record,
        {
            "name": job["name"],
            "kind": job["kind"],
            "status": "completed",
            "returncode": 0,
        },
        "작업 상태",
    )

    result = record["result"]
    output = (
        Path(result["output"])
        if job["kind"] == "support"
        else None
    )

    checks.equal(
        record["command"],
        expected_command(job, previous, output),
        "실제 실행 명령",
    )

    if job["kind"] == "test":
        expected = validate_test_result(
            job["name"],
            record["stdout"],
            record["stderr"],
        )

        checks.equal(result, expected, "테스트 완료 기록")
        checks.equal(record["artifact_sha256"], {}, "테스트 파일 계약")

        return

    checks.equal(
        json.loads(record["stdout"]),
        result,
        "CLI stdout 기록",
    )

    checks.equal(
        record["artifact_sha256"],
        artifact_hashes(
            settings,
            directory,
            job["kind"],
            result,
            provenance,
        ),
        "실제 산출물 지문 집합",
    )


def prepare_reuse(
    settings: Settings,
    manifest: Path,
    expected_hash: str,
    paths: dict[str, Path],
    snapshot: dict[str, str],
    provenance: checks.JsonObject,
    jobs: list[checks.JsonObject],
) -> checks.JsonObject:
    """이전 성공 계산의 무결성과 새 검증 계획의 호환성을 확인한다."""
    manifest = manifest.resolve()

    previous_plan = read_plan(
        settings,
        manifest,
        expected_hash,
        LEGACY_SCHEMA,
    )

    checks.equal(
        previous_plan["git"]["head"],
        BASE_HEAD,
        "이전 계획 기준 HEAD",
    )

    checks.equal(
        previous_plan["python"],
        platform.python_version(),
        "이전 Python 버전",
    )

    checks.equal(
        previous_plan["sqlite"],
        sqlite3.sqlite_version,
        "이전 SQLite 버전",
    )

    checks.equal(
        Path(previous_plan["python_executable"]).resolve(),
        Path(sys.executable).resolve(),
        "이전 Python 실행 파일",
    )

    checks.equal(
        previous_plan["paths"],
        {
            name: str(path.resolve())
            for name, path in paths.items()
        },
        "이전 계획 자료 경로",
    )

    checks.equal(
        previous_plan["provenance"],
        provenance,
        "이전 계획 자료 계보",
    )

    checks.equal(
        previous_plan["criteria"],
        criteria(),
        "이전 계획 판정 기준",
    )

    verify_old_inputs(
        settings.project_root,
        previous_plan["input_sha256"],
        snapshot,
    )

    old_jobs = [
        job
        for job in previous_plan["jobs"]
        if job["kind"] != "test"
    ]

    new_jobs = [
        job
        for job in jobs
        if job["kind"] != "test"
    ]

    checks.equal(
        old_jobs,
        new_jobs,
        "이전 및 신규 실자료 실행 조건",
    )

    records = {}
    references = {}

    for job in old_jobs:
        name = job["name"]

        record = load_job(
            manifest.parent,
            name,
            expected_hash,
        )

        validate_job_record(
            settings,
            manifest.parent,
            job,
            record,
            records,
            provenance,
        )

        records[name] = record

        path = record_path(manifest.parent, name)

        references[name] = {
            "record": str(path),
            "record_sha256": checks.file_hash(path),
        }

    return {
        "manifest": str(manifest),
        "manifest_sha256": expected_hash,
        "jobs": references,
    }


def prepare(
    settings: Settings,
    reuse_manifest: Path | None = None,
    reuse_hash: str | None = None,
) -> checks.JsonObject:
    """새 검증 명세를 생성하고 필요한 기존 증거를 인수한다."""
    state = verify_git_scope(settings.project_root)
    paths = fixed_paths(settings)
    provenance = verify_fixed_assets(settings, paths)
    snapshot = snapshot_paths(settings, paths)
    jobs = build_jobs(paths["model_asset"])

    checks.require(
        (reuse_manifest is None) == (reuse_hash is None),
        "이전 계획 경로와 지문을 함께 지정해야 합니다.",
    )

    reuse = None

    if reuse_manifest is not None and reuse_hash is not None:
        reuse = prepare_reuse(
            settings,
            reuse_manifest,
            reuse_hash,
            paths,
            snapshot,
            provenance,
            jobs,
        )

    directory = settings.results_dir / RESULT_DIRECTORY / uuid4().hex
    directory.mkdir(parents=True, exist_ok=False)

    manifest = directory / "plan.json"

    plan = {
        "schema": SCHEMA,
        "status": "prepared",
        "created_utc": utc_now(),
        "run_id": directory.name,
        "git": state,
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "sqlite": sqlite3.sqlite_version,
        "prepare_command": [sys.executable, *sys.argv],
        "paths": {
            name: str(path.resolve())
            for name, path in paths.items()
        },
        "provenance": provenance,
        "input_sha256": snapshot,
        "jobs": jobs,
        "criteria": criteria(),
        "official_references": checks.OFFICIAL_REFERENCES,
        "reuse": reuse,
    }

    write_json(manifest, plan)

    return {
        "status": "prepared",
        "run_id": directory.name,
        "manifest": str(manifest),
        "sha256": checks.file_hash(manifest),
        "reusable_jobs": len(reuse["jobs"]) if reuse else 0,
    }


def load_plan(
    settings: Settings,
    manifest: Path,
    plan_hash: str,
) -> checks.JsonObject:
    """신규 계획과 모든 고정 입력의 현재 상태를 확인한다."""
    plan = read_plan(
        settings,
        manifest,
        plan_hash,
        SCHEMA,
    )

    state = verify_git_scope(settings.project_root)

    checks.equal(state["head"], plan["git"]["head"], "검증 중 HEAD")
    checks.equal(state["branch"], plan["git"]["branch"], "검증 중 브랜치")

    checks.equal(
        platform.python_version(),
        plan["python"],
        "검증 중 Python 버전",
    )

    checks.equal(
        sqlite3.sqlite_version,
        plan["sqlite"],
        "검증 중 SQLite 버전",
    )

    checks.equal(
        Path(sys.executable).resolve(),
        Path(plan["python_executable"]).resolve(),
        "검증 중 Python 실행 파일",
    )

    paths = fixed_paths(settings)

    checks.equal(
        plan["paths"],
        {
            name: str(path.resolve())
            for name, path in paths.items()
        },
        "검증 중 자료 경로",
    )

    checks.equal(
        snapshot_paths(settings, paths),
        plan["input_sha256"],
        "검증 중 고정 입력",
    )

    checks.equal(
        plan["jobs"],
        build_jobs(paths["model_asset"]),
        "검증 중 실행 조건",
    )

    checks.equal(plan["criteria"], criteria(), "검증 중 판정 기준")

    reuse = plan["reuse"]

    if reuse is not None:
        checks.equal(
            checks.file_hash(Path(reuse["manifest"])),
            reuse["manifest_sha256"],
            "이전 계획 무결성",
        )

        for name, reference in reuse["jobs"].items():
            checks.equal(
                checks.file_hash(Path(reference["record"])),
                reference["record_sha256"],
                f"이전 실행 기록 무결성: {name}",
            )

    return plan


def reuse_reference(
    plan: checks.JsonObject,
    name: str,
) -> checks.JsonObject | None:
    """새 계획이 승인한 이전 작업 기록의 연결 정보를 반환한다."""
    reuse = plan["reuse"]

    if reuse is None or name not in reuse["jobs"]:
        return None

    return {
        "manifest": reuse["manifest"],
        "manifest_sha256": reuse["manifest_sha256"],
        **reuse["jobs"][name],
    }


def validate_saved_job(
    settings: Settings,
    directory: Path,
    job: checks.JsonObject,
    record: checks.JsonObject,
    previous: dict[str, checks.JsonObject],
    plan: checks.JsonObject,
) -> None:
    """신규 실행과 인수 실행을 구분하여 결과를 검증한다."""
    reference = reuse_reference(plan, job["name"])
    recorded = record.get("reused_from")

    checks.equal(
        recorded,
        reference,
        "이전 실행 인수 계보",
    )

    source_directory = (
        Path(reference["manifest"]).parent
        if reference is not None
        else directory
    )

    validate_job_record(
        settings,
        source_directory,
        job,
        record,
        previous,
        plan["provenance"],
    )


def adopt_job(
    settings: Settings,
    directory: Path,
    job: checks.JsonObject,
    plan: checks.JsonObject,
    plan_hash: str,
    previous: dict[str, checks.JsonObject],
    reference: checks.JsonObject,
) -> checks.JsonObject:
    """이전 성공 계산을 덮어쓰지 않고 새 계획에 연결한다."""
    source_directory = Path(reference["manifest"]).parent
    source_record = Path(reference["record"])

    checks.equal(
        checks.file_hash(source_record),
        reference["record_sha256"],
        "인수 작업 기록 SHA-256",
    )

    old_record = load_job(
        source_directory,
        job["name"],
        reference["manifest_sha256"],
    )

    validate_job_record(
        settings,
        source_directory,
        job,
        old_record,
        previous,
        plan["provenance"],
    )

    adopted = {
        **old_record,
        "plan_sha256": plan_hash,
        "reused_from": reference,
        "adopted_utc": utc_now(),
    }

    write_json(
        record_path(directory, job["name"]),
        adopted,
    )

    LOGGER.info("이전 성공 실행 인수: %s", job["name"])

    return adopted


def execute_job(
    settings: Settings,
    directory: Path,
    job: checks.JsonObject,
    plan: checks.JsonObject,
    plan_hash: str,
    previous: dict[str, checks.JsonObject],
) -> checks.JsonObject:
    """기존 결과를 인수하거나 신규 작업을 실행한다."""
    name = job["name"]
    target = record_path(directory, name)

    if target.exists():
        record = load_job(directory, name, plan_hash)

        validate_saved_job(
            settings,
            directory,
            job,
            record,
            previous,
            plan,
        )

        LOGGER.info("현재 계획의 성공 기록 재사용: %s", name)

        return record

    reference = reuse_reference(plan, name)

    if reference is not None:
        checks.require(
            job["kind"] != "test",
            "수정 전 테스트는 재사용할 수 없습니다.",
        )

        return adopt_job(
            settings,
            directory,
            job,
            plan,
            plan_hash,
            previous,
            reference,
        )

    output = (
        directory / "support" / f"{name}-{uuid4().hex}.json"
        if job["kind"] == "support"
        else None
    )

    command = expected_command(job, previous, output)

    LOGGER.info("신규 검증 실행 시작: %s", name)
    started = utc_now()

    process = subprocess.run(
        command,
        cwd=settings.project_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    record = {
        "name": name,
        "kind": job["kind"],
        "plan_sha256": plan_hash,
        "command": command,
        "started_utc": started,
        "finished_utc": utc_now(),
        "returncode": process.returncode,
        "stdout": process.stdout,
        "stderr": process.stderr,
        "status": "failed",
        "result": None,
        "artifact_sha256": {},
        "reused_from": None,
    }

    try:
        checks.require(
            process.returncode == 0,
            f"{name}: 종료 코드 {process.returncode}",
        )

        if job["kind"] == "test":
            record["result"] = validate_test_result(
                name,
                process.stdout,
                process.stderr,
            )

        else:
            result = json.loads(process.stdout)
            record["result"] = result

            record["artifact_sha256"] = artifact_hashes(
                settings,
                directory,
                job["kind"],
                result,
                plan["provenance"],
            )

        record["status"] = "completed"

        validate_saved_job(
            settings,
            directory,
            job,
            record,
            previous,
            plan,
        )

        write_json(target, record)

        LOGGER.info("신규 검증 실행 완료: %s", name)

        return record

    except Exception as exc:
        record["status"] = "failed"
        record["error"] = str(exc)

        write_json(
            directory / "jobs" / f"{name}-failed-{uuid4().hex}.json",
            record,
        )

        raise


def run_jobs(
    settings: Settings,
    manifest: Path,
    plan_hash: str,
) -> checks.JsonObject:
    """실자료 인수 및 최신 테스트 실행을 순서대로 수행한다."""
    plan = load_plan(settings, manifest, plan_hash)
    directory = manifest.resolve().parent
    records = {}

    for job in plan["jobs"]:
        records[job["name"]] = execute_job(
            settings,
            directory,
            job,
            plan,
            plan_hash,
            records,
        )

    return {
        "status": "completed",
        "run_id": plan["run_id"],
        "reused_jobs": sum(
            record.get("reused_from") is not None
            for record in records.values()
        ),
        "jobs": {
            name: {
                "returncode": record["returncode"],
                "reused": record.get("reused_from") is not None,
                "result": record["result"],
            }
            for name, record in records.items()
        },
    }


def load_records(
    settings: Settings,
    manifest: Path,
    plan_hash: str,
    plan: checks.JsonObject,
) -> dict[str, checks.JsonObject]:
    """모든 성공 기록을 새 검산 전에 다시 검증한다."""
    directory = manifest.resolve().parent
    records = {}

    for job in plan["jobs"]:
        name = job["name"]
        record = load_job(directory, name, plan_hash)

        validate_saved_job(
            settings,
            directory,
            job,
            record,
            records,
            plan,
        )

        records[name] = record

    return records


def read_regular_inputs(
    standard: sqlite3.Connection,
    model: sqlite3.Connection,
) -> tuple[list[checks.JsonObject], tuple]:
    """표준 경기와 M1의 실제 완료 경기 계약을 대조한다."""
    games = checks.regular_games(standard)

    completed = read_completed(
        standard,
        model,
        checks.SEASON,
        date(checks.SEASON + 1, 1, 1),
    )

    checks.equal(
        len(completed),
        len(games),
        "표준·M1 전체 완료 경기 수",
    )

    checks.equal(
        len(games),
        EXPECTED_GAMES,
        "2025년 완료 정규시즌 경기 수",
    )

    return games, completed


def compare_reproduction(
    regular: checks.Run,
    post: checks.Run,
    repeat_regular: checks.Run,
    repeat_post: checks.Run,
    long_regular: checks.Run,
    long_post: checks.Run,
    ended: checks.Run,
    repeat_ended: checks.Run,
) -> checks.JsonObject:
    """동일 Seed 출력과 반복 확장 접두 구간을 검증한다."""
    for name, reference, repeat in (
        ("regular", regular, repeat_regular),
        ("postseason", post, repeat_post),
        ("ended", ended, repeat_ended),
    ):
        checks.require(
            reference.rows == repeat.rows,
            f"{name}: 동일 Seed 반복 행 불일치",
        )

        checks.require(
            reference.summary == repeat.summary,
            f"{name}: 동일 Seed 요약 불일치",
        )

        if name == "postseason":
            left = json.loads(json.dumps(reference.scenario))
            right = json.loads(json.dumps(repeat.scenario))

            for scenario in (left, right):
                parent = scenario["metadata"]["parent"]
                parent.pop("run_id")
                parent.pop("manifest_sha256")

            checks.require(
                left == right,
                "포스트시즌 계산 시나리오 불일치",
            )

        else:
            checks.require(
                reference.scenario == repeat.scenario,
                f"{name}: 계산 시나리오 불일치",
            )

    checks.require(
        regular.rows == long_regular.rows[:len(regular.rows)],
        "정규시즌 반복 확장 접두 구간 불일치",
    )

    checks.require(
        post.rows == long_post.rows[:len(post.rows)],
        "포스트시즌 반복 확장 접두 구간 불일치",
    )

    for filename in ("diagnostic.json", "baseline_iterations.jsonl"):
        checks.equal(
            regular.digests[filename],
            repeat_regular.digests[filename],
            f"정규시즌 개발 진단 재현: {filename}",
        )

    return {
        "same_seed_reproduction": True,
        "regular_prefix_matched": len(regular.rows),
        "postseason_prefix_matched": len(post.rows),
        "ended_reproduction": True,
    }


def metric(values: Sequence[float]) -> checks.JsonObject:
    """실제 평가에 사용된 팀 수와 평균을 함께 보존한다."""
    return {
        "team_rows": len(values),
        "value": fsum(values) / len(values) if values else None,
    }


def recalculate_development_metrics(
    summary: checks.JsonObject,
    actual: dict[str, checks.JsonObject],
) -> checks.JsonObject:
    """기존 개발 진단의 승수·순위·사건 오차를 독립 계산한다."""
    errors = {
        "wins_mae": [],
        "rank_mae": [],
        "first_brier": [],
        "first_binary_log_loss": [],
        "qualification_brier": [],
        "qualification_binary_log_loss": [],
    }

    for team, observed in actual.items():
        forecast = summary["teams"][team]

        for field, key, output in (
            ("final_wins", "final_wins", "wins_mae"),
            ("rank", "rank", "rank_mae"),
        ):
            predicted = forecast[field]["expected"]
            target = observed[key]

            if predicted is not None and target is not None:
                errors[output].append(abs(predicted - target))

        for field, key, prefix in (
            ("regular_season_first", "first", "first"),
            (
                "postseason_qualification",
                "qualifies",
                "qualification",
            ),
        ):
            predicted = forecast[field]["expected"]
            target = observed[key]

            if predicted is None or target is None:
                continue

            errors[f"{prefix}_brier"].append(
                (predicted - int(target)) ** 2
            )

            mass = predicted if target else 1 - predicted

            errors[f"{prefix}_binary_log_loss"].append(
                -log(max(mass, LOG_LOSS_FLOOR))
            )

    return {
        key: metric(values)
        for key, values in errors.items()
    }


def read_baseline_rows(run: checks.Run) -> list[checks.JsonObject]:
    """개발 비교 기준의 저장 반복 행을 확인한다."""
    rows = []

    with (run.directory / "baseline_iterations.jsonl").open(
        encoding="utf-8",
    ) as stream:
        for index, line in enumerate(stream, start=1):
            row = json.loads(line)

            checks.equal(
                row["iteration_id"],
                index,
                "기준 반복 ID",
            )

            checks.equal(
                row["schema"],
                run.manifest["schemas"]["iterations"],
                "기준 반복 스키마",
            )

            rows.append(row)

    checks.equal(
        len(rows),
        len(run.rows),
        "운영·기준 반복 수",
    )

    return rows


def audit_development_diagnostic(
    run: checks.Run,
    games: Sequence[checks.JsonObject],
    completed: Sequence,
) -> checks.JsonObject:
    """운영·기준 개발 진단과 실제 종료 시즌의 차이를 대조한다."""
    diagnostic = checks.read_json(
        run.directory / "diagnostic.json"
    )

    checks.check_fields(
        diagnostic,
        {
            "mode": "retrospective_development_diagnostic",
            "formal_backtest": False,
            "final_test_used": False,
            "selection_use": "prohibited",
            "seed": run.manifest["seed"],
            "iterations": len(run.rows),
        },
        "개발 진단 계약",
    )

    state = build_state(KBO_2025, completed)
    standings = resolve_standings(state)

    actual = {
        str(team): team_result(
            state,
            team,
            standings[team],
        )
        for team in KBO_2025.teams
    }

    actual_seeds = [
        team
        for team in sorted(
            KBO_2025.teams,
            key=lambda team: standings[team].rank or len(KBO_2025.teams) + 1,
        )
        if standings[team].rank is not None
        and standings[team].rank <= KBO_2025.playoff_slots
    ]

    checks.equal(
        actual_seeds,
        checks.OFFICIAL_SEEDS,
        "종료 시즌 상위 시드 사실 대조",
    )

    baseline_summary = diagnostic["baseline_summary"]
    baseline_rows = read_baseline_rows(run)

    baseline_run = checks.Run(
        directory=run.directory,
        manifest=run.manifest,
        scenario=run.scenario,
        summary=baseline_summary,
        rows=baseline_rows,
        digests=run.digests,
    )

    baseline_audit = checks.audit_regular(
        baseline_run,
        games,
    )

    results = {}

    for name, summary in (
        ("candidate", run.summary),
        ("baseline", baseline_summary),
    ):
        stored = diagnostic[name]

        checks.check_fields(
            stored,
            {
                "actual_teams": actual,
                "independent_seasons": 1,
                "formal_backtest": False,
                "final_test_used": False,
            },
            f"{name}: 실제 종료 시즌",
        )

        metrics = recalculate_development_metrics(
            summary,
            actual,
        )

        checks.check_fields(
            stored,
            metrics,
            f"{name}: 실제 시즌 대비 오차",
        )

        results[name] = metrics

    return {
        "scope": "retrospective_development_diagnostic",
        "independent_seasons": 1,
        "formal_backtest_supported": False,
        "baseline_audit": baseline_audit,
        "actual_seeds": actual_seeds,
        "metrics": results,
    }


def check_reference_counts(
    regular: checks.Run,
    postseason: checks.Run,
    ended: checks.Run,
    starter: checks.JsonObject,
) -> checks.JsonObject:
    """하위 타워가 인수한 대표 검산값을 재확인한다."""
    checks.check_fields(
        regular.summary["league"],
        {
            "first_joint_valid": 984,
            "qualification_joint_valid": 922,
            "postseason_seed_resolved_iterations": 906,
        },
        "정규시즌 인수 수치",
    )

    checks.check_fields(
        postseason.summary["league"]["koreanSeries"],
        {
            "matchup_formed": 902,
            "valid": 899,
            "unresolved": 101,
        },
        "한국시리즈 인수 수치",
    )

    checks.equal(
        ended.manifest["champion_unresolved"],
        1000,
        "종료 재생 우승 미결정",
    )

    comparison = starter["comparison"]

    checks.equal(
        comparison["common_samples"],
        1429,
        "선발 공통 분모",
    )

    checks.check_fields(
        comparison["baseline"],
        {
            "team_game_samples": 1440,
            "truth_available": 1439,
            "candidate_provided": 1430,
            "evaluable_samples": 1429,
            "top1_hits": 269,
            "top_k_hits": 1056,
        },
        "운영 선발 인수 수치",
    )

    return {"matched": True}


def verify(
    settings: Settings,
    manifest: Path,
    plan_hash: str,
) -> checks.JsonObject:
    """고정 계획 및 실자료 결과를 종합 독립 검산한다."""
    plan = load_plan(settings, manifest, plan_hash)

    directory = manifest.resolve().parent
    paths = {
        key: Path(value)
        for key, value in plan["paths"].items()
    }

    records = load_records(
        settings,
        manifest,
        plan_hash,
        plan,
    )

    root = settings.project_root

    regular = {
        "reference": checks.load_run(
            root,
            paths["reference_regular"],
            "regular_season",
        ),
    }

    postseason = {
        "reference": checks.load_run(
            root,
            paths["reference_postseason"],
            "postseason",
        ),
        "ended": checks.load_run(
            root,
            paths["reference_ended"],
            "postseason",
        ),
    }

    for label, seed, iterations in SIMULATION_CASES:
        regular[label] = checks.load_run(
            root,
            Path(records[f"regular_{label}"]["result"]["result_dir"]),
            "regular_season",
        )

        postseason[label] = checks.load_run(
            root,
            Path(records[f"post_{label}"]["result"]["result_dir"]),
            "postseason",
        )

        for run in (regular[label], postseason[label]):
            checks.equal(
                run.manifest["seed"],
                seed,
                f"{label}: Seed",
            )

            checks.equal(
                run.manifest["iterations"],
                iterations,
                f"{label}: 반복 수",
            )

    postseason["ended_repeat"] = checks.load_run(
        root,
        Path(records["post_ended"]["result"]["result_dir"]),
        "postseason",
    )

    support = {}

    for name in (
        "evaluation",
        "season",
        "game",
        "early",
        "late",
    ):
        result = checks.read_json(
            Path(records[f"support_{name}"]["result"]["output"])
        )

        checks.check_sources(
            root,
            "support_analysis",
            result["source_sha256"],
        )

        support[name] = result

    results = {}
    failures = {}

    def audit(name: str, action: Callable[[], object]) -> None:
        """독립 검사 실패를 기록하고 가능한 나머지 검사를 계속한다."""
        try:
            results[name] = {
                "status": "passed",
                "result": action(),
            }

            LOGGER.info("검산 통과: %s", name)

        except Exception as exc:
            failures[name] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }

            LOGGER.exception("검산 실패: %s", name)

    with (
        closing(open_readonly(paths["standard_db"])) as standard,
        closing(open_readonly(paths["model_db"])) as model,
    ):
        games, completed = read_regular_inputs(
            standard,
            model,
        )

        results["regular_source_crosscheck"] = {
            "status": "passed",
            "result": {
                "standard_games": len(games),
                "m1_games": len(completed),
            },
        }

        for label, run in regular.items():
            audit(
                f"regular_{label}",
                lambda run=run: checks.audit_regular(
                    run,
                    games,
                ),
            )

            audit(
                f"diagnostic_{label}",
                lambda run=run: audit_development_diagnostic(
                    run,
                    games,
                    completed,
                ),
            )

        for label, run in postseason.items():
            audit(
                f"postseason_{label}",
                lambda run=run: checks.audit_postseason(run),
            )

        for label in ("reference", "repeat", "long", "seed"):
            audit(
                f"parent_{label}",
                lambda label=label: checks.audit_parent(
                    regular[label],
                    postseason[label],
                ),
            )

        audit(
            "archive_reference",
            lambda: checks.audit_archive(
                postseason["ended"],
                standard,
                model,
            ),
        )

        audit(
            "archive_repeat",
            lambda: checks.audit_archive(
                postseason["ended_repeat"],
                standard,
                model,
            ),
        )

        audit(
            "reproduction",
            lambda: compare_reproduction(
                regular["reference"],
                postseason["reference"],
                regular["repeat"],
                postseason["repeat"],
                regular["long"],
                postseason["long"],
                postseason["ended"],
                postseason["ended_repeat"],
            ),
        )

        for label in ("long", "seed"):
            for package, runs in (
                ("regular", regular),
                ("postseason", postseason),
            ):
                audit(
                    f"stability_{package}_{label}",
                    lambda runs=runs, label=label: checks.stability(
                        runs["reference"].summary,
                        runs[label].summary,
                    ),
                )

        audit(
            "starter_evaluation",
            lambda: checks.audit_starters(
                support["evaluation"],
                standard,
                model,
            ),
        )

        for label in ("season", "early", "late"):
            audit(
                f"support_{label}",
                lambda label=label: checks.audit_support_season(
                    support[label],
                    games,
                ),
            )

        audit(
            "support_game",
            lambda: checks.audit_support_game(
                support["game"],
                support["season"],
            ),
        )

        def support_reproduction() -> checks.JsonObject:
            """이전 보조 분석과 인수한 신규 계산을 대조한다."""
            for name in ("evaluation", "game", "season"):
                reference = checks.read_json(
                    paths["reference_support"] / f"{name}.json"
                )

                checks.check_sources(
                    root,
                    "support_analysis",
                    reference["source_sha256"],
                )

                checks.require(
                    checks.normalize_support(reference)
                    == checks.normalize_support(support[name]),
                    f"보조 분석 재현 불일치: {name}",
                )

            return {
                "matched": [
                    "evaluation",
                    "game",
                    "season",
                ],
            }

        audit(
            "support_reproduction",
            support_reproduction,
        )

        audit(
            "official_development_diagnostic",
            lambda: checks.official_diagnostic(
                regular["reference"],
                postseason["reference"],
                games,
            ),
        )

        audit(
            "season_availability",
            lambda: checks.season_availability(standard),
        )

        audit(
            "reference_counts",
            lambda: check_reference_counts(
                regular["reference"],
                postseason["reference"],
                postseason["ended"],
                support["evaluation"],
            ),
        )

    # 이전 자산과 코드가 검증 과정에서 변경되지 않았는지 재확인한다.
    load_plan(settings, manifest, plan_hash)

    output = directory / f"verification-{uuid4().hex}.json"

    report = {
        "schema": SCHEMA,
        "status": "failed" if failures else "completed",
        "created_utc": utc_now(),
        "run_id": plan["run_id"],
        "plan": str(manifest.resolve()),
        "plan_sha256": plan_hash,
        "verify_command": [sys.executable, *sys.argv],
        "git": git_state(root),
        "provenance": plan["provenance"],
        "criteria": plan["criteria"],
        "reuse": plan["reuse"],
        "jobs": {
            name: {
                "record": str(record_path(directory, name)),
                "record_sha256": checks.file_hash(
                    record_path(directory, name)
                ),
                "command": record["command"],
                "returncode": record["returncode"],
                "result": record["result"],
                "artifact_sha256": record["artifact_sha256"],
                "reused_from": record.get("reused_from"),
            }
            for name, record in records.items()
        },
        "checks": results,
        "failures": failures,
        "limitations": [
            "2025 retrospective development assumption",
            "2026 season simulation unsupported",
            "no historical publication proof",
            "no independent official full-game record certification",
            "raw source bytes not independently reopened",
            "regular iteration rows contain team aggregates",
            "stability thresholds are diagnostic warnings",
            "existing starter ranking and series rules reused",
            "historical executions and current verification separated",
        ],
    }

    write_json(output, report)

    return {
        "status": report["status"],
        "run_id": plan["run_id"],
        "output": str(output),
        "sha256": checks.file_hash(output),
        "passed_checks": len(results),
        "failed_checks": failures,
        "reused_jobs": sum(
            record.get("reused_from") is not None
            for record in records.values()
        ),
        "tests": {
            name: records[name]["result"]
            for name in (
                "validation_tests",
                "simulation_tests",
                "regression",
            )
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    """계획 작성·실행 인수·독립 검산의 CLI 진입점."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path)

    commands = parser.add_subparsers(
        dest="command",
        required=True,
    )

    preparation = commands.add_parser("prepare")

    preparation.add_argument(
        "--reuse-manifest",
        type=Path,
    )

    preparation.add_argument(
        "--reuse-manifest-sha256",
    )

    for name in ("run", "verify"):
        command = commands.add_parser(name)

        command.add_argument(
            "--manifest",
            type=Path,
            required=True,
        )

        command.add_argument(
            "--manifest-sha256",
            required=True,
        )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        settings = load_settings(args.project_root)

        if args.command == "prepare":
            result = prepare(
                settings,
                args.reuse_manifest,
                args.reuse_manifest_sha256,
            )

        elif args.command == "run":
            result = run_jobs(
                settings,
                args.manifest,
                args.manifest_sha256,
            )

        else:
            result = verify(
                settings,
                args.manifest,
                args.manifest_sha256,
            )

        print(json.dumps(
            result,
            ensure_ascii=False,
            allow_nan=False,
        ))

        return 2 if result["status"] == "failed" else 0

    except KeyboardInterrupt:
        LOGGER.warning("사용자가 검증을 중단했습니다.")
        return 3

    except Exception:
        LOGGER.exception("시즌 전망 종합 검증 실패")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
