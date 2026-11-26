"""M1 정규시즌 개발 학습·평가와 JSON 사전 입력 예측의 CLI."""

import argparse
from collections import Counter
from collections.abc import Sequence
from contextlib import closing
from datetime import date, datetime, timezone
import json
import logging
from math import isfinite
from pathlib import Path
import platform
import sqlite3
import sys
from typing import cast
from uuid import uuid4

from . import model as m1
from ...datasets.model_data import fingerprint, inside
from ...datasets.published_sources import RUN_PATTERN, SourceError, as_row, read_json, sha256_file, write_json
from ...evaluation import game_outcome as evaluation
from ...settings import Settings, load_settings


LOGGER = logging.getLogger(__name__)
BOUNDARIES = ("2025-01-01", "2026-01-01", "2026-11-26")
SPLITS = ("train", "validation")
METHOD = {
    "profile_prior": m1.PROFILE_PRIOR,
    "season_prior": m1.SEASON_PRIOR,
    "recent_prior": m1.RECENT_PRIOR,
    "zone_width": m1.ZONE_WIDTH,
    "zone_limit": m1.ZONE_LIMIT,
    "tail_share": m1.TAIL_SHARE,
    "min_cap": m1.MIN_CAP,
    "max_cap": m1.MAX_CAP,
    "probability_tolerance": m1.TOLERANCE,
    "joint": "diagonal_weighted_product",
    "representative_score": "round_each_marginal_mean_half_up",
}


def read_games(db: sqlite3.Connection) -> tuple[dict[str, list[m1.Game]], dict[str, Counter[str]]]:
    """개발 경기만 일대일 연결하고 정답·분할·제외를 점검한다."""
    features = ", ".join(f"i.{column}" for column in m1.INPUT_COLUMNS)
    rows = db.execute(
        f"""SELECT x.sample_id, x.game_id, x.sample_date, x.split, x.status,
                   i.sample_id, t.sample_id, t.target_home_runs, t.target_away_runs,
                   t.target_result, {features}
            FROM m1_index x
            LEFT JOIN m1_inputs i ON x.sample_id = i.sample_id
            LEFT JOIN m1_targets t ON x.sample_id = t.sample_id
              AND x.status = 'eligible' AND i.competition = 'regular'
              AND x.sample_date < ?
            WHERE x.split IN ('train', 'validation')
            ORDER BY x.sample_date, x.sample_id""",
        (BOUNDARIES[1],),
    )
    games: dict[str, list[m1.Game]] = {split: [] for split in SPLITS}
    ledger: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    seen: set[str] = set()

    for sample, game, day, split, status, input_id, target_id, home, away, result, *raw in rows:
        if not isinstance(sample, str) or not sample or sample != game or sample in seen:
            raise ValueError(f"M1 경기 키 누락·중복: {sample}")
        seen.add(sample)
        if not isinstance(day, str) or date.fromisoformat(day).isoformat() != day:
            raise ValueError(f"M1 경기 날짜 오류: {sample}")
        expected = "train" if day < BOUNDARIES[0] else "validation"
        if split != expected or day >= BOUNDARIES[1]:
            raise ValueError(f"M1 개발 분할 경계 오류: {sample}")
        ledger[split]["index_rows"] += 1
        if status != "eligible":
            ledger[split][f"status:{status}"] += 1
            continue
        if input_id != sample:
            raise ValueError(f"M1 사전 입력 누락: {sample}")
        inputs = dict(zip(m1.INPUT_COLUMNS, raw, strict=True))
        if inputs["competition"] == "postseason":
            ledger[split]["postseason"] += 1
            continue
        if target_id != sample:
            raise ValueError(f"M1 정규시즌 정답 누락 또는 대회 구분 오류: {sample}")
        runs = (m1.count(home, "target_home_runs"), m1.count(away, "target_away_runs"))
        if result != m1.score_outcome(*runs):
            raise ValueError(f"M1 공식 득점과 결과 불일치: {sample}")
        try:
            m1.history(inputs)
        except ValueError as exc:
            raise ValueError(f"M1 사전 입력 오류 {sample}: {exc}") from exc
        games[split].append(m1.Game(sample, day, inputs, runs, result))

    if any(not games[split] for split in SPLITS):
        raise ValueError("Train과 Validation에 정규시즌 경기 자료가 필요합니다.")
    return games, ledger


def source(settings: Settings, run_id: str) -> tuple[Path, dict[str, object]]:
    """완료된 자료와 입력 계약을 확인하고 매니페스트 출처를 보존한다."""
    if not RUN_PATTERN.fullmatch(run_id):
        raise ValueError("모델 자료 실행 ID는 소문자 16진수 32자리입니다.")
    directory = settings.processed_dir / "model_data" / run_id
    manifest = as_row(read_json(inside(directory, "manifest.json")))
    contract_path = inside(directory, "feature_target_contract.json")
    definitions = as_row(read_json(contract_path))
    contract = as_row(definitions["m1"])
    allowed = as_row(contract["inputs"])
    config = as_row(manifest["config"])
    limits = tuple(config[key] for key in ("validation_start", "test_start", "observation_end"))
    contract_hash = fingerprint(definitions)
    if (
        manifest["status"] != "completed"
        or manifest["run_id"] != run_id
        or config["timezone"] != "Asia/Seoul"
        or limits != BOUNDARIES
        or manifest["contract_sha256"] != contract_hash
        or not set(m1.INPUT_COLUMNS) <= allowed.keys()
        or contract["index"] != "m1_index"
        or contract["target_columns"] != ["sample_id", "target_home_runs", "target_away_runs", "target_result"]
    ):
        raise ValueError("인수 매니페스트, M1 허용 목록 또는 시간 계약이 일치하지 않습니다.")
    database = inside(directory, "model_data.sqlite3")
    provenance = {
        "model_data_run_id": run_id,
        "model_data_version": config["version"],
        "database": str(database),
        "database_sha256_from_manifest": manifest["database_sha256"],
        "contract_sha256": contract_hash,
        "manifest_sha256": sha256_file(directory / "manifest.json"),
        "timezone": config["timezone"],
        "validation_start": limits[0],
        "test_start": limits[1],
        "observation_end_exclusive": limits[2],
    }
    return database, provenance


def load_model(path: Path) -> m1.Model:
    """저장 자산의 버전·계약·무결성과 파라미터를 검사한다."""
    envelope = as_row(read_json(path))
    payload = as_row(envelope["payload"])
    if (
        envelope["sha256"] != fingerprint(payload)
        or payload["schema"] != m1.SCHEMA
        or payload["input_columns"] != list(m1.INPUT_COLUMNS)
        or payload["method"] != METHOD
    ):
        raise ValueError("M1 자산 무결성, 버전 또는 스키마 불일치")

    model = cast(m1.Model, as_row(payload["model"]))
    cap = m1.count(model["cap"], "cap")
    if not m1.MIN_CAP <= cap <= m1.MAX_CAP or not (
        isfinite(model["league_mean"]) and 0 <= model["league_mean"] <= cap
        and 0 < model["tie_weight"] < 1
    ):
        raise ValueError("M1 자산의 득점 범위 또는 학습 계수 오류")

    site = model["site_counts"]
    groups = model["profile_counts"]
    keys = {f"{side}:{zone}" for side in m1.SIDES for zone in range(-m1.ZONE_LIMIT, m1.ZONE_LIMIT + 1)}
    if site.keys() != set(m1.SIDES) or not groups or not groups.keys() <= keys:
        raise ValueError("M1 자산의 홈·원정 또는 프로필 키 오류")

    for scores in (*site.values(), *groups.values()):
        if not isinstance(scores, list) or len(scores) != cap + 1:
            raise ValueError("M1 자산의 히스토그램 길이 오류")
        if sum(m1.count(value, "histogram") for value in scores) == 0:
            raise ValueError("M1 자산의 빈 히스토그램")
    for side in m1.SIDES:
        parts = [scores for key, scores in groups.items() if key.startswith(f"{side}:")]
        if [sum(column) for column in zip(*parts, strict=True)] != site[side]:
            raise ValueError("M1 자산의 프로필 집계 불일치")
    return model


def train_run(settings: Settings, run_id: str) -> dict[str, object]:
    """개발 자료를 읽기 전용으로 열고 고유 경로에 평가·재현 예제를 저장한다."""
    database, provenance = source(settings, run_id)
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("PRAGMA query_only = ON")
        games, ledger = read_games(db)
    train, validation = (games[key] for key in SPLITS)
    model = m1.fit(train)
    run = uuid4().hex
    result_dir = settings.results_dir / "m1" / run
    model_dir = settings.models_dir / "m1" / run
    paths = {
        "model": str(model_dir / "model.json"),
        "evaluation": str(result_dir / "evaluation.json"),
        "input": str(result_dir / "example_input.json"),
        "prediction": str(result_dir / "example_prediction.json"),
    }
    example = validation[0]
    payload = {
        "schema": m1.SCHEMA, "input_columns": list(m1.INPUT_COLUMNS), "method": METHOD,
        "model": model, "training_run_id": run, "data": provenance, "outputs": paths,
        "python": platform.python_version(), "sqlite": sqlite3.sqlite_version,
        "source_sha256": {
            "src/models/game_outcome/model.py": sha256_file(Path(m1.__file__)),
            "src/models/game_outcome/__main__.py": sha256_file(Path(__file__)),
            "src/evaluation/game_outcome.py": sha256_file(Path(evaluation.__file__)),
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "configured_seed": settings.seed, "uses_randomness_for_fit": False,
        "splits": {
            key: {
                "games": len(games[key]), "team_rows": 2 * len(games[key]),
                "first_date": games[key][0].day, "last_date": games[key][-1].day,
                "index_and_exclusions": dict(ledger[key]),
            }
            for key in SPLITS
        },
        "reproduce_argv": [
            "-m", "src.models.game_outcome", "--project-root", str(settings.project_root),
            "train", "--run-id", run_id,
        ],
    }
    report = {
        "training_run_id": run, "model_schema": m1.SCHEMA, "paths": paths,
        "splits": payload["splits"], "example_sample_id": example.sample_id, "example_date": example.day,
        "selection": "single_predeclared_candidate; no validation tuning or refit",
        "final_test": "held_out", "evaluation": evaluation.evaluate(model, train, validation),
    }
    result_dir.mkdir(parents=True, exist_ok=False)
    model_dir.mkdir(parents=True, exist_ok=False)
    write_json(Path(paths["evaluation"]), report)
    write_json(Path(paths["input"]), example.inputs)
    forecast = m1.predict(model, example.inputs)
    write_json(Path(paths["prediction"]), forecast)
    temporary = model_dir / "model.pending.json"
    write_json(temporary, {"payload": payload, "sha256": fingerprint(payload)})
    if m1.predict(load_model(temporary), example.inputs) != forecast:
        raise ValueError("M1 저장 전후 예측 불일치")
    temporary.rename(paths["model"])
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """성공 0, 설정·입력·자산 오류 2, 사용자 중단 3을 반환한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("train").add_argument("--run-id", required=True)
    prediction = commands.add_parser("predict")
    prediction.add_argument("--asset", required=True, type=Path)
    prediction.add_argument("--input", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.project_root)
        logging.basicConfig(level=settings.log_level, format="%(levelname)s %(name)s: %(message)s")
        if args.command == "train":
            result = train_run(settings, args.run_id)
        else:
            result = m1.predict(load_model(args.asset), as_row(read_json(args.input)))
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (OSError, SourceError, ValueError, TypeError, KeyError, AttributeError, OverflowError,
            sqlite3.DatabaseError) as exc:
        LOGGER.error("M1 실행 실패: %s", exc)
        return 2
    except KeyboardInterrupt:
        LOGGER.warning("M1 실행 중단")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())