"""고정 시즌 전망·보조 분석의 종합 검증 실행."""

import argparse
from collections.abc import Callable, Sequence
from contextlib import closing
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import platform
import re
import sqlite3
import subprocess
import sys
from uuid import uuid4

from src.settings import Settings, load_settings
from src.simulation.regular_season.data import open_readonly, verify_assets
from src.validation import season_forecast_checks as checks


LOGGER = logging.getLogger(__name__)

SCHEMA = "season-forecast-validation-v1"
BASE_HEAD = "11f996b6ac04c0edac706d4b16599e49cc6ebdc5"

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

SIMULATION_CASES = (
    ("repeat", 42, 1000),
    ("long", 42, 2000),
    ("seed", 43, 1000),
)


def utc_now() -> str:
    """검증 증거 생성 시각을 UTC로 기록한다."""
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
    """Git 저장소 상태만 읽고 작업 트리는 변경하지 않는다."""
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )

    return result.stdout.strip()


def git_state(root: Path) -> checks.JsonObject:
    """실제 적용 기준 HEAD와 작업 트리를 기록한다."""
    return {
        "head": git_read(root, "rev-parse", "HEAD"),
        "branch": git_read(root, "branch", "--show-current"),
        "status": git_read(root, "status", "--short", "--branch"),
        "origin_main": git_read(root, "rev-parse", "origin/main"),
    }


def fixed_paths(settings: Settings) -> dict[str, Path]:
    """하위 타워에서 인수한 고정 자료와 기존 결과의 경로."""
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
            settings.models_dir / "m1" / MODEL_RUN / "model.json"
        ),
        "reference_regular": (
            settings.results_dir / "regular_season" / REFERENCE_REGULAR
        ),
        "reference_postseason": (
            settings.results_dir / "postseason" / REFERENCE_POSTSEASON
        ),
        "reference_ended": (
            settings.results_dir / "postseason" / REFERENCE_ENDED
        ),
        "reference_support": (
            settings.results_dir / "support_analysis" / REFERENCE_SUPPORT
        ),
        "final_evaluation": (
            settings.results_dir / "final_evaluation" / FINAL_EVALUATION
        ),
    }


def snapshot_paths(
    settings: Settings,
    paths: dict[str, Path],
) -> dict[str, str]:
    """코드·설정·자료·기존 결과의 실제 지문을 고정한다."""
    root = settings.project_root

    files = [
        paths["standard_db"],
        paths["model_db"],
        paths["model_asset"],
        paths["model_db"].parent / "manifest.json",
        root / "kbo.toml",
        root / "pyproject.toml",
        root / "docs" / "standards" / "season_forecast_validation.md",
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
    """인수 지문과 기존 M1 자료 계보를 함께 확인한다."""
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


def build_jobs(asset: Path) -> list[checks.JsonObject]:
    """결과를 확인하기 전에 모든 재현 실행 조건을 고정한다."""
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
            ["game", *scenario, "--game-id", REPRESENTATIVE_GAME],
        ),
        (
            "support_early",
            ["season", "--season", "2025", "--as-of", EARLY_CUTOFF],
        ),
        (
            "support_late",
            ["season", "--season", "2025", "--as-of", END_CUTOFF],
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
                "-m", "unittest", "discover", "-s", "tests",
            ],
        },
    ])

    return jobs


def prepare(settings: Settings) -> checks.JsonObject:
    """검증 대상과 판정 기준을 고정한 신규 명세를 저장한다."""
    state = git_state(settings.project_root)

    checks.equal(state["head"], BASE_HEAD, "기준 HEAD")

    paths = fixed_paths(settings)
    provenance = verify_fixed_assets(settings, paths)

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
        "input_sha256": snapshot_paths(settings, paths),
        "jobs": build_jobs(paths["model_asset"]),
        "criteria": {
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
        },
        "official_references": checks.OFFICIAL_REFERENCES,
    }

    write_json(manifest, plan)

    return {
        "status": "prepared",
        "run_id": directory.name,
        "manifest": str(manifest),
        "sha256": checks.file_hash(manifest),
    }


def load_plan(
    settings: Settings,
    manifest: Path,
    expected_hash: str,
) -> checks.JsonObject:
    """계획 및 입력 파일이 고정된 상태인지 확인한다."""
    manifest = manifest.resolve()
    allowed = (settings.results_dir / RESULT_DIRECTORY).resolve()

    checks.require(
        manifest.is_relative_to(allowed)
        and manifest.name == "plan.json",
        "검증 계획 경로 오류",
    )

    checks.equal(
        checks.file_hash(manifest),
        expected_hash,
        "검증 계획 SHA-256",
    )

    plan = checks.read_json(manifest)

    checks.equal(plan["schema"], SCHEMA, "검증 계획 스키마")
    checks.equal(plan["status"], "prepared", "검증 계획 상태")

    checks.equal(
        git_read(settings.project_root, "rev-parse", "HEAD"),
        plan["git"]["head"],
        "검증 중 HEAD 변경",
    )
    checks.equal(
        platform.python_version(),
        plan["python"],
        "검증 중 Python 버전 변경",
    )
    checks.equal(
        sqlite3.sqlite_version,
        plan["sqlite"],
        "검증 중 SQLite 버전 변경",
    )

    for path, digest in plan["input_sha256"].items():
        checks.equal(
            checks.file_hash(Path(path)),
            digest,
            f"고정 입력 변경 {path}",
        )

    return plan


def record_path(directory: Path, name: str) -> Path:
    """실행별 성공 증거 경로를 반환한다."""
    return directory / "jobs" / f"{name}.json"


def load_job(
    directory: Path,
    name: str,
    plan_hash: str,
) -> checks.JsonObject:
    """완료된 실행 증거와 산출물 지문을 검증한다."""
    record = checks.read_json(record_path(directory, name))

    checks.equal(record["name"], name, "작업명")
    checks.equal(record["plan_sha256"], plan_hash, "작업 계획 지문")
    checks.equal(record["returncode"], 0, "작업 종료 코드")
    checks.equal(record["status"], "completed", "작업 상태")

    for path, digest in record["artifact_sha256"].items():
        checks.equal(
            checks.file_hash(Path(path)),
            digest,
            f"작업 산출물 변경 {path}",
        )

    return record


def resolve_arguments(
    directory: Path,
    job: checks.JsonObject,
    plan_hash: str,
) -> list[str]:
    """고정 계획의 부모 ID 및 신규 출력 경로를 해석한다."""
    result = []

    for value in job["args"]:
        if value.startswith("@parent:"):
            name = value.split(":", 1)[1]
            parent = load_job(directory, name, plan_hash)
            result.append(parent["result"]["run_id"])

        elif value == "@output":
            result.append(str(
                directory / "support" / f"{job['name']}-{uuid4().hex}.json"
            ))

        else:
            result.append(value)

    return result


def execute_job(
    settings: Settings,
    directory: Path,
    job: checks.JsonObject,
    plan_hash: str,
) -> checks.JsonObject:
    """기존 운영 CLI 또는 테스트를 실행하고 종료 증거를 보존한다."""
    name = job["name"]
    target = record_path(directory, name)

    if target.exists():
        record = load_job(directory, name, plan_hash)
        LOGGER.info("기존 성공 실행 재사용: %s", name)
        return record

    command = [
        sys.executable,
        "-X", "utf8",
        "-B",
        *resolve_arguments(directory, job, plan_hash),
    ]

    LOGGER.info("검증 실행 시작: %s", name)
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
    }

    try:
        checks.require(
            process.returncode == 0,
            f"{name}: 종료 코드 {process.returncode}",
        )

        if job["kind"] == "test":
            output = process.stdout + "\n" + process.stderr
            matched = re.search(r"Ran (\d+) tests?", output)

            checks.require(matched is not None, "테스트 수 확인 실패")

            count = int(matched.group(1))
            checks.require(count > 0, "실행된 테스트 없음")

            if name == "regression":
                checks.require(count >= 628, "기존 회귀 테스트 수 감소")

            record["result"] = {
                "tests": count,
            }

        else:
            result = json.loads(process.stdout)

            checks.equal(result["status"], "completed", "CLI 완료 상태")
            record["result"] = result

            if job["kind"] == "support":
                output = Path(result["output"])
                digest = checks.file_hash(output)

                checks.equal(digest, result["sha256"], "보조 분석 출력 지문")

                record["artifact_sha256"] = {
                    str(output): digest,
                }

            else:
                output = Path(result["result_dir"])

                run = checks.load_run(
                    settings.project_root,
                    output,
                    job["kind"],
                )

                record["artifact_sha256"] = {
                    str(output / filename): digest
                    for filename, digest in run.digests.items()
                }

                manifest = output / "manifest.json"
                record["artifact_sha256"][str(manifest)] = checks.file_hash(
                    manifest
                )

        record["status"] = "completed"
        write_json(target, record)

        LOGGER.info("검증 실행 완료: %s", name)

        return record

    except (OSError, ValueError, KeyError, TypeError) as exc:
        record["error"] = str(exc)

        write_json(
            directory / "jobs" / f"{name}-failed-{uuid4().hex}.json",
            record,
        )

        raise ValueError(f"{name}: {exc}") from exc


def run_jobs(
    settings: Settings,
    manifest: Path,
    plan_hash: str,
) -> checks.JsonObject:
    """미완료 작업만 실행하고 기존 성공 실행은 재사용한다."""
    plan = load_plan(settings, manifest, plan_hash)
    directory = manifest.resolve().parent
    completed = {}

    for job in plan["jobs"]:
        record = execute_job(settings, directory, job, plan_hash)

        completed[job["name"]] = {
            "returncode": record["returncode"],
            "result": record["result"],
        }

    return {
        "status": "completed",
        "run_id": plan["run_id"],
        "jobs": completed,
    }


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
    """동일 Seed 재현 및 반복 확장 접두 구간을 확인한다."""
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

            checks.require(left == right, "포스트시즌 계산 시나리오 불일치")

        else:
            checks.require(
                reference.scenario == repeat.scenario,
                f"{name}: 시나리오 불일치",
            )

    checks.require(
        regular.rows == long_regular.rows[:len(regular.rows)],
        "정규시즌 반복 확장 접두 구간 불일치",
    )
    checks.require(
        post.rows == long_post.rows[:len(post.rows)],
        "포스트시즌 반복 확장 접두 구간 불일치",
    )

    if "diagnostic.json" in regular.digests:
        for filename in ("diagnostic.json", "baseline_iterations.jsonl"):
            checks.equal(
                regular.digests[filename],
                repeat_regular.digests[filename],
                f"정규시즌 개발 진단 재현 {filename}",
            )

    return {
        "same_seed_reproduction": True,
        "regular_prefix_matched": len(regular.rows),
        "postseason_prefix_matched": len(post.rows),
        "ended_reproduction": True,
    }


def verify(
    settings: Settings,
    manifest: Path,
    plan_hash: str,
) -> checks.JsonObject:
    """고정 명세 및 신규 실행의 실제 결과를 독립 검산한다."""
    plan = load_plan(settings, manifest, plan_hash)
    directory = manifest.resolve().parent
    paths = {
        key: Path(value)
        for key, value in plan["paths"].items()
    }

    records = {
        job["name"]: load_job(directory, job["name"], plan_hash)
        for job in plan["jobs"]
    }

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
            checks.equal(run.manifest["seed"], seed, f"{label}: Seed")
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
        """한 검산의 실패가 독립된 다른 검산 증거를 소실시키지 않게 한다."""
        try:
            results[name] = {
                "status": "passed",
                "result": action(),
            }

            LOGGER.info("검산 통과: %s", name)

        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            sqlite3.Error,
        ) as exc:
            failures[name] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }

            LOGGER.error("검산 실패: %s: %s", name, exc)

    with (
        closing(open_readonly(paths["standard_db"])) as standard,
        closing(open_readonly(paths["model_db"])) as model,
    ):
        games = checks.regular_games(standard)

        # 개별 경기 원천과 M1 표준 계약의 교차 검증도 재사용한다.
        completed = read_completed(
            standard,
            model,
            checks.SEASON,
            checks.date(2026, 1, 1),
        )

        checks.equal(
            len(completed),
            len(games),
            "표준·M1 전체 완료 경기 수",
        )
        checks.equal(len(games), 720, "인수 정규시즌 경기 수")

        for label, run in regular.items():
            audit(
                f"regular_{label}",
                lambda run=run: checks.audit_regular(run, games),
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
                postseason["ended"], standard, model
            ),
        )
        audit(
            "archive_repeat",
            lambda: checks.audit_archive(
                postseason["ended_repeat"], standard, model
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
                    support[label], games
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
            """기존 보조 분석 계산과 신규 계산을 대조한다."""
            for name in ("evaluation", "game", "season"):
                reference = checks.read_json(
                    paths["reference_support"] / f"{name}.json"
                )

                checks.require(
                    checks.normalize_support(reference)
                    == checks.normalize_support(support[name]),
                    f"보조 분석 재현 불일치: {name}",
                )

            return {"matched": ["evaluation", "game", "season"]}

        audit("support_reproduction", support_reproduction)

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

        def reference_counts() -> checks.JsonObject:
            """하위 타워가 인수한 대표 검산값을 대조한다."""
            checks.check_fields(
                regular["reference"].summary["league"],
                {
                    "first_joint_valid": 984,
                    "qualification_joint_valid": 922,
                    "postseason_seed_resolved_iterations": 906,
                },
                "정규시즌 인수 수치",
            )

            checks.check_fields(
                postseason["reference"].summary["league"]["koreanSeries"],
                {
                    "matchup_formed": 902,
                    "valid": 899,
                    "unresolved": 101,
                },
                "한국시리즈 인수 수치",
            )

            checks.equal(
                postseason["ended"].manifest["champion_unresolved"],
                1000,
                "종료 재생 인수 수치",
            )

            comparison = support["evaluation"]["comparison"]

            checks.equal(comparison["common_samples"], 1429, "선발 공통 분모")

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

        audit("reference_counts", reference_counts)

    # 신규 실행 및 검산 이후에도 이전 자산이 유지되는지 확인한다.
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
        "tests": {
            name: records[name]["result"]
            for name in ("validation_tests", "simulation_tests", "regression")
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    """계획 작성·신규 실행·독립 검산의 진입점을 제공한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path)

    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")

    for name in ("run", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--manifest-sha256", required=True)

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        settings = load_settings(args.project_root)

        if args.command == "prepare":
            result = prepare(settings)
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

    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        sqlite3.Error,
        subprocess.SubprocessError,
    ) as exc:
        LOGGER.error("종합 검증 실패: %s", exc)
        return 2

    except KeyboardInterrupt:
        LOGGER.warning("사용자가 검증을 중단했습니다.")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
