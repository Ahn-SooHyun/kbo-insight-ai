"""검증된 표준 DB에서 모델 자료를 별도 생성·검사하는 실행 진입점."""

import argparse
from collections.abc import Sequence
from contextlib import closing
from datetime import date
import hashlib
import json
import logging
from pathlib import Path
import re
import sqlite3
from uuid import uuid4

from . import model_events, model_players
from .published_sources import RUN_PATTERN, as_row, read_json, sha256_file, write_json
from ..settings import Settings, load_settings
from ..validation.standard_data import TABLE_NAMES
from ..validation.standard_quality import AUDIT_SCHEMA, AUDIT_TABLES, _digest_tables


LOGGER = logging.getLogger(__name__)
VERSION = "1.2.0"
EVENT_MODELS = ("m1", "m2", "m4", "m5")
MODELS = (*EVENT_MODELS, *model_players.ROLES.values())
SPLITS = ("train", "validation", "test")
CANDIDATES = {
    "m1": "eligible_games",
    "m2": "plate_appearances",
    "m4": "eligible_states",
    "m5": "eligible_pitches",
}
CATEGORICAL = {
    "competition", "half", "phase", "player_id", "batter_id",
    "pitcher_id", "previous_pitch_type", "previous_pitch_result",
    "known_team_id", "home_team_id", "away_team_id",
    "perspective_team_id",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def inside(directory: Path, name: str) -> Path:
    path = (directory / name).resolve()
    require(
        path.is_relative_to(directory.resolve()) and path.is_file(),
        f"자료 경로 또는 파일 오류: {name}",
    )
    return path


def fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def verify_version_inventory(
    records: list[dict[str, object]],
    object_count: int,
    logical_targets: int,
) -> None:
    """물리 원본, 논리 수집 대상 및 대상별 판본을 별도로 검증한다."""
    versions = {
        (record["target_id"], record["source_sha"])
        for record in records
    }

    objects = {
        record["source_sha"]
        for record in records
    }

    selected = [
        record["target_id"]
        for record in records
        if record["selected"] is True
    ]

    require(
        len(versions) == len(records),
        "대상별 원본 판본 중복",
    )

    require(
        len(objects) == object_count,
        "물리 원본 고유 해시 수 불일치",
    )

    require(
        len(selected) == len(set(selected)) == logical_targets,
        "논리 대상 채택 판본 수 불일치",
    )

    require(
        set(selected) == {
            record["target_id"]
            for record in records
        },
        "논리 대상의 채택 판본 누락",
    )


def verify_source(
    settings: Settings,
    run_id: str,
    verification_id: str,
) -> tuple[Path, dict[str, object]]:
    """선행 산출물과 원본·중간 판본을 읽기 전용으로 연결한다."""
    require(
        bool(
            RUN_PATTERN.fullmatch(run_id)
            and RUN_PATTERN.fullmatch(verification_id)
        ),
        "실행/검증 ID 오류",
    )

    standard = settings.processed_dir / "standard" / run_id
    interim = settings.interim_dir / "standard" / run_id

    db_path = inside(
        standard,
        f"verified-{verification_id}.sqlite3",
    )

    report_path = inside(
        standard,
        f"verified-{verification_id}.json",
    )

    parent_path = inside(standard, "standard.sqlite3")
    quality_path = inside(standard, "quality.json")
    manifest_path = inside(interim, "manifest.json")

    report = as_row(read_json(report_path))
    parent = as_row(read_json(quality_path))
    manifest = as_row(read_json(manifest_path))

    require(
        report["status"] == parent["status"] == "completed"
        and report["run_id"] == parent["run_id"]
        == manifest["run_id"] == run_id
        and report["verification_id"] == verification_id
        and report["verification_key"]
        == f"{run_id}:{verification_id}",
        "선행 완료 보고서 불일치",
    )

    require(
        fingerprint(report["inputs"])
        == report["input_fingerprint"]
        and fingerprint(manifest["inputs"])
        == manifest["input_fingerprint"]
        and parent["input_fingerprint"]
        == manifest["input_fingerprint"]
        == report["inputs"]["parent_input_fingerprint"]
        and sha256_file(manifest_path)
        == parent["interim_manifest_sha256"],
        "선행 입력 지문 또는 중간 명세 오류",
    )

    require(
        report["source_objects"] == manifest["object_count"]
        and report["logical_targets"]
        == manifest["logical_target_count"],
        "원본 물리 수량 또는 논리 대상 수 불일치",
    )

    verify_version_inventory(
        manifest["versions"],
        manifest["object_count"],
        manifest["logical_target_count"],
    )

    versions = []
    actual_objects = {}

    for record in manifest["versions"]:
        sha = record["source_sha"]
        result = record["collection_result"]

        raw_path = inside(
            settings.raw_dir / "published",
            result["object_path"],
        )

        require(
            result["sha256"] == sha
            and result["target_id"] == record["target_id"]
            and result["revision"] == record["revision"]
            and raw_path.stat().st_size == result["bytes"],
            "수집 결과와 원본 판본 불일치",
        )

        if sha not in actual_objects:
            require(
                sha256_file(raw_path) == sha,
                "원본 파일 해시 오류",
            )

            actual_objects[sha] = raw_path.stat().st_size

        require(
            sha256_file(
                inside(interim, record["interim_file"])
            ) == record["interim_sha256"],
            "중간 파일 해시 오류",
        )

        versions.append({
            "target_id": record["target_id"],
            "sha256": sha,
            "selected": record["selected"],
            "interim_sha256": record["interim_sha256"],
            "revision": result.get("revision"),
            "collected_at": result.get("observed_at"),
            "permitted_uses": result.get("permitted_uses"),
            "approval_ref": result.get("approval_ref"),
        })

    require(
        sum(actual_objects.values()) == manifest["raw_bytes"],
        "물리 원본 고유 바이트 수 불일치",
    )

    if "regular_completion" in manifest["inputs"]:
        require(
            manifest["inputs"]["regular_completion"]
            == report["handoff"]["regular_completion"]
            == report["parent_coverage"]["regular_completion"],
            "정규시즌 완료 근거 연결 오류",
        )

    for name, digest in report["source_specs"].items():
        require(
            sha256_file(
                inside(settings.raw_dir / "sources", name)
            ) == digest,
            "원본 요청 명세 불일치",
        )

    with closing(
        sqlite3.connect(
            parent_path.as_uri() + "?mode=ro",
            uri=True,
        )
    ) as db:
        require(
            _digest_tables(db, TABLE_NAMES)
            == report["inputs"]["parent_tables"],
            "부모 DB 논리 해시 불일치",
        )

    with closing(
        sqlite3.connect(
            db_path.as_uri() + "?mode=ro",
            uri=True,
        )
    ) as db:
        require(
            db.execute(
                "PRAGMA integrity_check"
            ).fetchone()[0] == "ok"
            and db.execute(
                "PRAGMA foreign_key_check"
            ).fetchone() is None,
            "최종 DB 무결성 오류",
        )

        require(
            _digest_tables(
                db,
                TABLE_NAMES + AUDIT_TABLES,
            ) == report["tables"],
            "검증 DB 논리 해시 불일치",
        )

        recorded = report["handoff"]["candidate_counts"]

        for name, expected in recorded.items():
            declared = re.search(
                rf"CREATE VIEW {re.escape(name)} AS\b.*?;",
                AUDIT_SCHEMA,
                re.DOTALL,
            )

            actual = db.execute(
                "SELECT sql FROM sqlite_master WHERE type='view' AND name=?",
                (name,),
            ).fetchone()

            require(
                declared is not None
                and actual is not None
                and declared.group().rstrip(";") == actual[0],
                f"뷰 정의 오류: {name}",
            )

            require(
                db.execute(
                    f"SELECT COUNT(*) FROM {name}"
                ).fetchone()[0] == expected,
                f"후보 수 불일치: {name}",
            )

    return db_path, {
        "run_id": run_id,
        "verification_id": verification_id,
        "verification_fingerprint": report["input_fingerprint"],
        "database_sha256": sha256_file(db_path),
        "report_sha256": sha256_file(report_path),
        "parent_db_sha256": sha256_file(parent_path),
        "parent_report_sha256": sha256_file(quality_path),
        "interim_manifest_sha256": sha256_file(manifest_path),
        "source_specs": report["source_specs"],
        "source_objects": manifest["object_count"],
        "logical_targets": manifest["logical_target_count"],
        "source_version_count": len(versions),
        "versions": versions,
        "row_accounting": report["row_accounting"],
        "parent_coverage": report["parent_coverage"],
        "handoff": report["handoff"],
        "usage": (
            "상위 승인: 비영리 대학 과제 로컬 처리 및 내부 평가; "
            "공개 재배포 제외"
        ),
    }


def columns(db: sqlite3.Connection, table: str) -> list[str]:
    return [
        row[1]
        for row in db.execute(f"PRAGMA table_info({table})")
    ]


def scalar(db: sqlite3.Connection, sql: str) -> int:
    return int(db.execute(sql).fetchone()[0])


def validate(db: sqlite3.Connection) -> dict[str, object]:
    """전체 자료의 키·분할·정답 범위·모델 입력 차단을 검사한다."""
    result = {}

    for model in MODELS:
        index, inputs, targets = (
            f"{model}_{part}"
            for part in ("index", "inputs", "targets")
        )

        event = model in EVENT_MODELS
        allowed = "status='eligible'" if event else "status='candidate'"

        n = scalar(
            db,
            f"SELECT COUNT(*) FROM {index} WHERE {allowed}",
        )

        require(
            n == scalar(db, f"SELECT COUNT(*) FROM {inputs}")
            and n == scalar(
                db,
                f"SELECT COUNT(DISTINCT sample_id) FROM {inputs}",
            ),
            f"입력 키 오류: {model}",
        )

        require(
            scalar(db, f"SELECT COUNT(*) FROM {index}")
            == scalar(
                db,
                f"SELECT COUNT(DISTINCT sample_id) FROM {index}",
            ),
            f"인덱스 키 오류: {model}",
        )

        require(
            not scalar(
                db,
                f"""
                SELECT COUNT(*)
                FROM {inputs} x LEFT JOIN {index} i USING(sample_id)
                WHERE i.sample_id IS NULL OR NOT(i.{allowed})
                """,
            ),
            f"입력 인덱스 불일치: {model}",
        )

        require(
            scalar(db, f"SELECT COUNT(*) FROM {targets}")
            == n * (1 if event else 3),
            f"정답 수 오류: {model}",
        )

        require(
            not scalar(
                db,
                f"""
                SELECT COUNT(*)
                FROM {targets} t LEFT JOIN {inputs} x USING(sample_id)
                WHERE x.sample_id IS NULL
                """,
            ),
            f"고아 정답: {model}",
        )

        require(
            not scalar(
                db,
                f"""
                SELECT COUNT(*)
                FROM {index} i CROSS JOIN config c
                WHERE i.split<>CASE
                    WHEN i.sample_date<c.validation_start THEN 'train'
                    WHEN i.sample_date<c.test_start THEN 'validation'
                    ELSE 'test'
                END
                """,
            ),
            f"시간 분할 오류: {model}",
        )

        require(
            not any(
                name.startswith("target_")
                or name.startswith("post_")
                or name in {
                    "winner", "event", "pitch_type", "pitch_result",
                    "release_speed_kmh", "plate_x", "plate_z",
                }
                for name in columns(db, inputs)
            ),
            f"사후 정보 입력: {model}",
        )

        if event:
            source = CANDIDATES[model]

            require(
                n == scalar(
                    db,
                    f"SELECT COUNT(DISTINCT sample_id) FROM {targets}",
                ),
                f"정답 키 오류: {model}",
            )

            require(
                scalar(db, f"SELECT COUNT(*) FROM {index}")
                == scalar(
                    db,
                    f"SELECT COUNT(*) FROM source.{source}",
                ) * (2 if model == "m4" else 1),
                f"후보 분모 불일치: {model}",
            )

            require(
                not scalar(
                    db,
                    f"""
                    SELECT COUNT(*) FROM (
                        SELECT game_id FROM {index}
                        GROUP BY game_id
                        HAVING COUNT(DISTINCT split)>1
                    )
                    """,
                ),
                f"경기 분할 오류: {model}",
            )

            require(
                not scalar(
                    db,
                    f"""
                    SELECT COUNT(*)
                    FROM {index} i JOIN game_context g USING(game_id)
                    WHERE i.sample_date<>g.date OR i.split<>g.split
                    """,
                ),
                f"경기 날짜/분할 불일치: {model}",
            )

            if model == "m2":
                require(
                    n == scalar(
                        db,
                        "SELECT COUNT(*) FROM source.eligible_pa",
                    ),
                    "M2 유효 타석 분모 불일치",
                )

                require(
                    not scalar(
                        db,
                        """
                        SELECT COUNT(*)
                        FROM m2_index i
                        LEFT JOIN source.eligible_pa a
                            USING(game_id,pa_number)
                        WHERE (i.status='eligible')<>(a.game_id IS NOT NULL)
                           OR (
                               i.status='excluded'
                               AND i.exclusion_reason IS NULL
                           )
                           OR (
                               i.status='eligible'
                               AND i.exclusion_reason IS NOT NULL
                           )
                        """,
                    ),
                    "M2 상태·제외 사유 불일치",
                )

                require(
                    not scalar(
                        db,
                        """
                        SELECT COUNT(*)
                        FROM m2_index i
                        JOIN m2_inputs x USING(sample_id)
                        JOIN m2_targets t USING(sample_id)
                        JOIN source.eligible_pa a
                            USING(game_id,pa_number)
                        WHERE x.batter_id IS NOT a.start_batter_id
                           OR x.pitcher_id IS NOT a.start_pitcher_id
                           OR t.target_event IS NOT a.event
                           OR i.credited_batter_candidate
                                IS NOT a.credited_batter_candidate
                           OR i.credited_pitcher_candidate
                                IS NOT a.credited_pitcher_candidate
                        """,
                    ),
                    "M2 참가자/정답 귀속 불일치",
                )

        else:
            require(
                not scalar(
                    db,
                    f"""
                    SELECT COUNT(*) FROM {index}
                    WHERE strftime('%w',sample_date)<>'1'
                       OR last_observed_date>=sample_date
                    """,
                ),
                f"월요일/모집단 시점 오류: {model}",
            )

            require(
                not scalar(
                    db,
                    f"""
                    SELECT COUNT(*) FROM {index} i
                    WHERE last_observed_date IS NOT (
                        SELECT MAX(h.date)
                        FROM player_history_base h
                        WHERE h.player_id=i.player_id
                          AND h.role=i.role
                          AND h.season=i.season
                          AND h.date<i.sample_date
                          AND (
                              h.conflict_date IS NULL
                              OR h.conflict_date>=i.sample_date
                          )
                    )
                    """,
                ),
                f"시점별 선수 식별 모집단 오류: {model}",
            )

            role = model.removeprefix("m3_")

            require(
                not scalar(
                    db,
                    f"""
                    WITH dates AS (
                        SELECT DISTINCT season,sample_date FROM {index}
                    ), expected AS (
                        SELECT DISTINCT h.season,d.sample_date,h.player_id
                        FROM dates d JOIN player_history_base h
                            ON h.season=d.season
                           AND h.role='{role}'
                           AND h.date<d.sample_date
                           AND (
                               h.conflict_date IS NULL
                               OR h.conflict_date>=d.sample_date
                           )
                    )
                    SELECT COUNT(*)
                    FROM expected e
                    LEFT JOIN {index} i USING(season,sample_date,player_id)
                    WHERE i.sample_id IS NULL
                    """,
                ),
                f"시점별 모집단 누락: {model}",
            )

            require(
                not scalar(
                    db,
                    f"""
                    SELECT COUNT(*)
                    FROM {targets} t JOIN {index} i USING(sample_id)
                    CROSS JOIN config c
                    WHERE t.window_start<>i.sample_date
                       OR (
                           t.usable=1 AND (
                               t.ready_date IS NULL
                               OR t.ready_date>CASE i.split
                                   WHEN 'train' THEN c.validation_start
                                   WHEN 'validation' THEN c.test_start
                                   ELSE c.observation_end
                               END
                               OR t.status NOT IN (
                                   'valid_observed','zero_observed'
                               )
                           )
                       )
                    """,
                ),
                f"정답 완료/절단 오류: {model}",
            )

            require(
                not scalar(
                    db,
                    f"""
                    SELECT COUNT(*) FROM {targets}
                    WHERE (
                        horizon='7d'
                        AND window_end<>date(window_start,'+7 days')
                    )
                    OR (
                        horizon='28d'
                        AND window_end<>date(window_start,'+28 days')
                    )
                    OR (usable=0 AND target_games IS NOT NULL)
                    OR (status='zero_observed' AND observed_games<>0)
                    """,
                ),
                f"기간/관측 0 오류: {model}",
            )

            require(
                not scalar(
                    db,
                    f"""
                    SELECT COUNT(*)
                    FROM (
                        SELECT DISTINCT i.season,i.sample_date,i.split,
                               t.horizon,t.window_start,t.window_end
                        FROM {targets} t
                        JOIN {index} i USING(sample_id)
                        WHERE t.usable=1
                    ) a
                    JOIN m3_window_games w
                      ON w.season=a.season
                     AND w.anchor_date=a.sample_date
                     AND w.horizon=a.horizon
                    JOIN game_context g USING(game_id)
                    WHERE g.date<a.window_start
                       OR g.date>=a.window_end
                       OR g.split<>a.split
                    """,
                ),
                f"기간 정답 경기 혼입: {model}",
            )

            # 생성된 위험 값과 실제 경기·팀·역할 감사 결과를 대조한다.
            require(
                not scalar(
                    db,
                    f"""
                    SELECT COUNT(*) FROM {targets}
                    WHERE usable=1 AND (
                        missing_games>0
                        OR team_risky_games>0
                        OR player_rejected>0
                        OR (
                            status='zero_observed'
                            AND (
                                league_risky_games>0
                                OR league_identity_conflicts>0
                                OR league_rejected_rows>0
                            )
                        )
                    )
                    """,
                ),
                f"불완전 정답 사용: {model}",
            )

            require(
                not scalar(
                    db,
                    f"""
                    WITH relevant AS (
                        SELECT DISTINCT i.season,i.sample_date,
                               t.horizon,r.team_id
                        FROM {targets} t
                        JOIN {index} i USING(sample_id)
                        JOIN m3_window_teams r
                          ON r.sample_id=t.sample_id
                         AND r.horizon=t.horizon
                         AND r.role=i.role
                        WHERE t.usable=1
                    )
                    SELECT COUNT(*)
                    FROM relevant a
                    JOIN m3_window_games w
                      ON w.season=a.season
                     AND w.anchor_date=a.sample_date
                     AND w.horizon=a.horizon
                    JOIN player_game_role_quality q
                      ON q.game_id=w.game_id
                     AND q.team_id=a.team_id
                     AND q.role='{role}'
                    WHERE q.risky=1
                    """,
                ),
                f"팀·역할 품질 위험 정답 사용: {model}",
            )

            require(
                not scalar(
                    db,
                    f"""
                    WITH zero_windows AS (
                        SELECT DISTINCT i.season,i.sample_date,t.horizon
                        FROM {targets} t
                        JOIN {index} i USING(sample_id)
                        WHERE t.usable=1 AND t.status='zero_observed'
                    )
                    SELECT COUNT(*)
                    FROM zero_windows a JOIN m3_window_games w
                      ON w.season=a.season
                     AND w.anchor_date=a.sample_date
                     AND w.horizon=a.horizon
                    JOIN player_game_role_quality q
                      ON q.game_id=w.game_id
                     AND q.role='{role}'
                    WHERE q.risky=1 OR q.rejected_rows>0
                    """,
                ),
                f"미확인 0 사용: {model}",
            )

            require(
                not scalar(
                    db,
                    f"""
                    SELECT COUNT(*)
                    FROM {targets} t JOIN {inputs} x USING(sample_id)
                    WHERE x.known_team_id IS NOT NULL
                      AND NOT EXISTS (
                        SELECT 1 FROM m3_window_teams r
                        WHERE r.sample_id=t.sample_id
                          AND r.role='{role}'
                          AND r.horizon=t.horizon
                          AND r.team_id=x.known_team_id
                    )
                    """,
                ),
                f"정답 창 과거 팀 추적 누락: {model}",
            )

            require(
                not scalar(
                    db,
                    f"""
                    SELECT COUNT(*)
                    FROM {targets} t JOIN {index} i USING(sample_id)
                    JOIN player_history_base h
                      ON h.player_id=i.player_id
                     AND h.role=i.role
                     AND h.season=i.season
                     AND h.date>=t.window_start
                     AND h.date<t.observed_through_exclusive
                    WHERE NOT EXISTS (
                        SELECT 1 FROM m3_window_teams r
                        WHERE r.sample_id=t.sample_id
                          AND r.role=i.role
                          AND r.horizon=t.horizon
                          AND r.team_id=h.team_id
                    )
                    """,
                ),
                f"정답 창 실제 관측 팀 추적 누락: {model}",
            )

            blocked = " OR ".join(
                f"{name} IS NOT NULL"
                for name in columns(db, targets)
                if name.startswith("target_")
            )

            require(
                not scalar(
                    db,
                    f"""
                    SELECT COUNT(*) FROM {targets}
                    WHERE usable=0 AND ({blocked})
                    """,
                ),
                f"사용 불가 정답 값 잔존: {model}",
            )

        result[model] = [
            dict(row)
            for row in db.execute(
                f"""
                SELECT split,status,COUNT(*) rows,
                       MIN(sample_date) first_date,
                       MAX(sample_date) last_date
                FROM {index}
                GROUP BY split,status
                ORDER BY split,status
                """
            )
        ]

        if not event:
            result[model + "_target_status"] = [
                dict(row)
                for row in db.execute(
                    f"""
                    SELECT i.split,t.horizon,t.status,t.usable,
                           t.ratio_status,COUNT(*) rows
                    FROM {targets} t JOIN {index} i USING(sample_id)
                    GROUP BY 1,2,3,4,5
                    ORDER BY 1,2,3,4,5
                    """
                )
            ]

    require(
        not scalar(
            db,
            """
            SELECT COUNT(*) FROM m1_targets
            WHERE target_home_runs<0 OR target_away_runs<0
               OR target_result<>CASE
                   WHEN target_home_runs>target_away_runs THEN 'home'
                   WHEN target_home_runs<target_away_runs THEN 'away'
                   ELSE 'tie'
               END
            """,
        ),
        "M1 정답 오류",
    )

    require(
        not scalar(
            db,
            """
            SELECT COUNT(*) FROM m4_targets
            WHERE target_remaining_runs<0
               OR target_result NOT IN ('win','loss','tie')
            """,
        ),
        "M4 정답 오류",
    )

    require(
        not scalar(
            db,
            """
            SELECT COUNT(*) FROM (
                SELECT pair_id,COUNT(*) n,
                       COUNT(DISTINCT split) s,
                       COUNT(DISTINCT status) st
                FROM m4_index GROUP BY pair_id
                HAVING n<>2 OR s<>1 OR st<>1
            )
            """,
        ),
        "M4 상태 쌍 오류",
    )

    require(
        not scalar(
            db,
            """
            SELECT COUNT(*) FROM m5_targets
            WHERE target_joint<>target_pitch_type||':'||target_pitch_result
            """,
        ),
        "M5 공동 범주 오류",
    )

    require(
        not scalar(
            db,
            """
            SELECT COUNT(*) FROM (
                SELECT game_id FROM player_game_role_quality
                GROUP BY game_id HAVING COUNT(*)<>4
            )
            """,
        ),
        "경기별 양 팀·두 역할 품질 행 누락",
    )

    require(
        not scalar(
            db,
            """
            SELECT COUNT(*) FROM starter_pre_game
            WHERE previous_start_date>=date
               OR days_since_start<=0
               OR previous_appearance_date>=date
               OR days_since_appearance<=0
            """,
        ),
        "선발 과거 이력 누수",
    )

    return result


def train_eda(db: sqlite3.Connection) -> dict[str, object]:
    """정답 및 변수의 분포·결측은 학습 표본에만 접근한다."""
    eda = {}

    for model in MODELS:
        index, inputs, targets = (
            f"{model}_{part}"
            for part in ("index", "inputs", "targets")
        )

        feature_summary = {}

        where = (
            f"FROM {inputs} x JOIN {index} i USING(sample_id) "
            "WHERE i.split='train'"
        )

        names = columns(db, inputs)[1:]

        # 큰 투구 테이블은 수치 요약을 한 번에 계산한다.
        numeric = [
            name for name in names
            if name not in CATEGORICAL
        ]

        expressions = []

        for name in numeric:
            expressions.extend((
                f"COUNT(*)-COUNT(x.{name})",
                f"MIN(x.{name})",
                f"MAX(x.{name})",
                f"AVG(x.{name})",
            ))

        values = (
            tuple(
                db.execute(
                    f"SELECT {','.join(expressions)} {where}"
                ).fetchone()
            )
            if expressions
            else ()
        )

        for pos, name in enumerate(numeric):
            feature_summary[name] = dict(
                zip(
                    ("missing", "min", "max", "mean"),
                    values[pos * 4:pos * 4 + 4],
                )
            )

        for name in (
            name for name in names
            if name in CATEGORICAL
        ):
            field = f"x.{name}"

            row = db.execute(
                f"""
                SELECT COUNT(*)-COUNT({field}) missing,
                       COUNT(DISTINCT {field}) distinct_values
                {where}
                """
            ).fetchone()

            feature_summary[name] = dict(row)

        labels = {}

        target_where = (
            f"FROM {targets} t JOIN {index} i USING(sample_id) "
            "WHERE i.split='train'"
        )

        if model in EVENT_MODELS:
            for name in columns(db, targets)[1:]:
                field = f"t.{name}"

                if name in {
                    "target_result", "target_event", "target_pitch_type",
                    "target_pitch_result", "target_joint",
                }:
                    labels[name] = [
                        dict(row)
                        for row in db.execute(
                            f"""
                            SELECT {field} value,COUNT(*) rows
                            {target_where}
                            GROUP BY {field}
                            ORDER BY rows DESC,value
                            """
                        )
                    ]

                else:
                    labels[name] = dict(
                        db.execute(
                            f"""
                            SELECT MIN({field}) min,
                                   MAX({field}) max,
                                   AVG({field}) mean
                            {target_where}
                            """
                        ).fetchone()
                    )

        else:
            for name in (
                name
                for name in columns(db, targets)
                if name.startswith("target_")
            ):
                field = f"t.{name}"

                labels[name] = [
                    dict(row)
                    for row in db.execute(
                        f"""
                        SELECT horizon,COUNT({field}) nonnull,
                               COUNT(*)-COUNT({field}) missing,
                               MIN({field}) min,
                               MAX({field}) max,
                               AVG({field}) mean
                        {target_where} AND usable=1
                        GROUP BY horizon
                        ORDER BY horizon
                        """
                    )
                ]

        eda[model] = {
            "features": feature_summary,
            "targets": labels,
        }

    return eda


def contract(db: sqlite3.Connection) -> dict[str, object]:
    """모델별 입력 열과 원천·정보 시점·정답을 기록한다."""
    result = {}

    for model in MODELS:
        features = {}

        for name in columns(db, f"{model}_inputs")[1:]:
            if name.startswith((
                "home_season_", "away_season_",
                "home_recent_", "away_recent_",
            )) or name.endswith("days_since_game"):
                origin = (
                    "eligible_games; 해당 팀·같은 시즌 정규경기; "
                    "날짜 엄격 이전, recent=28일; rf/ra=득점/실점 합, "
                    "wins/ties=승/무; 미관측 집계 NULL"
                )

            elif name.startswith((
                "batter_prior_", "pitcher_prior_",
                "batter_recent_", "pitcher_recent_",
            )):
                origin = (
                    "player_games+player_admission 경기별 품질 및 "
                    "player_identity_evidence; 역할별 날짜 엄격 이전, "
                    "식별 충돌 최초 관측일 이후만 차단; recent=28일, "
                    "rejected=제외 행수; 기회=PA/BF"
                )

            elif model.startswith("m3_"):
                origin = (
                    "player_games+player_admission 경기별 품질 및 "
                    "player_identity_evidence; 기준 월요일 엄격 이전 "
                    "같은 시즌의 당시 식별 참가자·집계, 미래 이름 충돌 "
                    "소급 금지; recent28=28일, last known team=당시 "
                    "마지막 관측 팀; 미관측 통계 NULL"
                )

            elif name.startswith("previous_pitch_"):
                origin = (
                    "pitches; 같은 타석 바로 이전 실제 공만, "
                    "연속 번호 및 앞선 source_row, 유효 코드 없으면 NULL"
                )

            elif name in {
                "season", "competition",
                "home_team_id", "away_team_id",
            }:
                origin = (
                    "eligible_games; 예측 대상 경기의 알려진 대진/시즌 "
                    "조건; 과거 일정 공개 시점 복원 아님"
                )

            else:
                origin = {
                    "m2": (
                        "eligible_pa; 타석 시작 상태 및 참가자"
                    ),
                    "m4": (
                        "eligible_states; 해당 pre/post 경계까지 "
                        "관측 상태, 공수 팀 관점"
                    ),
                    "m5": (
                        "eligible_pitches; 현재 공 직전 상태·참가자와 "
                        "해당 타석 이닝/공수"
                    ),
                }[model]

            features[name] = {
                "origin_and_cutoff": origin,
                "null_policy": (
                    "NULL 보존; 학습 자료만으로 후속 대체/인코딩 fit"
                ),
                "key": "sample_id",
            }

        target_definition = {
            "m1": "eligible_games: 종료 득점·승무패",
            "m2": (
                "eligible_pa: 관측 확정 사건 코드; "
                "전체 plate_appearances의 상태·제외 근거와 "
                "시작/종료/귀속 후보는 인덱스에 보존"
            ),
            "m4": (
                "eligible_states+eligible_games: 타석 전후 "
                "동일 공격팀 관점 결과와 반이닝 종료 득점-현재 득점"
            ),
            "m5": (
                "eligible_pitches: 현재 실제 공 구종/결과 및 "
                "공동 코드; 위치 제외"
            ),
        }.get(
            model,
            (
                "player_games+최종 player_admission: 월요일부터 "
                "7/28일/잔여 정규시즌 관측 합; "
                "player_game_role_quality와 m3_window_teams로 "
                "관련 팀 위험 검사, 0은 리그 역할 위험·격리 없는 "
                "관측에 한정; target_*는 usable만, observed_*는 "
                "부분도 보존; hit_rate=hits/AB, "
                "strikeout_rate=SO/(PA 또는 BF), 분모 0은 NULL"
            ),
        )

        result[model] = {
            "inputs": features,
            "target_columns": columns(db, f"{model}_targets"),
            "target_definition": target_definition,
            "index": f"{model}_index",
            "training_filter": (
                "index.split='train'"
                + (
                    " AND targets.usable=1 (기간·기록별 NULL mask)"
                    if model.startswith("m3_")
                    else " AND index.status='eligible'"
                )
            ),
            "preprocessing": (
                "학습 전처리는 이번에 fit하지 않음; "
                "sample_id/인덱스/정답/감사 열은 입력 제외"
            ),
        }

    return result


def build_model_data(
    settings: Settings,
    run_id: str,
    verification_id: str,
    validation_start: str,
    test_start: str,
    observation_end: str,
    *,
    compare_run: str | None = None,
) -> dict[str, object]:
    """새 실행 폴더에 모델 자료·학습 EDA·계약·재현 근거를 저장한다."""
    for value in (
        validation_start,
        test_start,
        observation_end,
    ):
        require(
            date.fromisoformat(value).isoformat() == value,
            "날짜는 YYYY-MM-DD 필요",
        )

    require(
        validation_start < test_start < observation_end,
        "시간 분할 경계 오류",
    )

    source, provenance = verify_source(
        settings,
        run_id,
        verification_id,
    )

    config = {
        "version": VERSION,
        "validation_start": validation_start,
        "test_start": test_start,
        "observation_end": observation_end,
        "timezone": "Asia/Seoul",
        "seed": settings.seed,
        "code_sha256": {
            Path(module.__file__).name:
                sha256_file(Path(module.__file__))
            for module in (model_events, model_players)
        },
        "builder_sha256": sha256_file(Path(__file__)),
        "history_cutoff": (
            "strict prior date; identity conflict effective "
            "after first observed conflicting date; "
            "retrospective game-quality records"
        ),
    }

    key = fingerprint({
        "source": provenance,
        "config": config,
    })

    output_root = settings.processed_dir / "model_data"
    previous = None

    if compare_run:
        require(
            bool(RUN_PATTERN.fullmatch(compare_run)),
            "비교 실행 ID 오류",
        )

        previous = as_row(
            read_json(
                inside(
                    output_root / compare_run,
                    "manifest.json",
                )
            )
        )

        require(
            previous["status"] == "completed"
            and previous["input_fingerprint"] == key,
            "동일 입력·설정의 완료 실행만 비교 가능",
        )

    output_id = uuid4().hex
    directory = output_root / output_id
    directory.mkdir(parents=True, exist_ok=False)

    database = directory / "model_data.sqlite3"

    LOGGER.info("모델 자료 생성 시작: %s", output_id)

    with closing(
        sqlite3.connect(database, uri=True)
    ) as db:
        db.row_factory = sqlite3.Row

        db.execute(
            "ATTACH DATABASE ? AS source",
            (source.as_uri() + "?mode=ro",),
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
            "INSERT INTO config VALUES (?,?,?)",
            (
                validation_start,
                test_start,
                observation_end,
            ),
        )

        require(
            db.execute(
                "SELECT MAX(date) FROM source.eligible_games"
            ).fetchone()[0] < observation_end,
            "관측 종료일이 입력 자료보다 앞섭니다.",
        )

        model_events.build_events(db)

        LOGGER.info("선수별 월요일 및 정답 기간 생성")

        model_players.build_players(
            db,
            observation_end,
        )

        model_events.build_support(db)

        for model in MODELS:
            for part in ("index", "inputs", "targets"):
                table = f"{model}_{part}"

                db.execute(
                    f"CREATE INDEX {table}_sample "
                    f"ON {table}(sample_id)"
                )

            db.execute(
                f"CREATE INDEX {model}_split "
                f"ON {model}_index(split,status)"
            )

        db.commit()

        LOGGER.info("모델별 전체 무결성 및 학습 자료 EDA 검사")

        counts = validate(db)

        require(
            db.execute(
                "PRAGMA main.integrity_check"
            ).fetchone()[0] == "ok",
            "파생 DB 무결성 오류",
        )

        eda = train_eda(db)
        definitions = contract(db)

        tables = tuple(
            row[0]
            for row in db.execute(
                """
                SELECT name FROM main.sqlite_master
                WHERE type='table' ORDER BY name
                """
            )
        )

        hashes = _digest_tables(db, tables)

    require(
        sha256_file(source) == provenance["database_sha256"],
        "처리 중 표준 DB 변경 감지",
    )

    comparison = None

    if previous is not None:
        comparison = {
            "same_input": True,
            "changed_tables": [
                name
                for name in hashes
                if hashes[name] != previous["tables"].get(name)
            ],
        }

        comparison["reproducible"] = (
            not comparison["changed_tables"]
            and previous["eda_sha256"] == fingerprint(eda)
            and previous["contract_sha256"]
            == fingerprint(definitions)
        )

    write_json(directory / "train_eda.json", eda)

    write_json(
        directory / "feature_target_contract.json",
        definitions,
    )

    manifest = {
        "status": (
            "completed"
            if comparison is None or comparison["reproducible"]
            else "reproduction_failed"
        ),
        "run_id": output_id,
        "database_path": str(database),
        "input_fingerprint": key,
        "source": provenance,
        "config": config,
        "tables": hashes,
        "database_sha256": sha256_file(database),
        "splits": counts,
        "eda_sha256": fingerprint(eda),
        "contract_sha256": fingerprint(definitions),
        "comparison": comparison,
        "limitations": [
            (
                "과거 실제 공개 판본·라인업·일정 스냅샷 미확인; "
                "당시 공개 재현이 아닌 소급 관측 자료"
            ),
            (
                "선수 이력은 이름 충돌의 최초 관측일 이전에 소급 "
                "제외하지 않으며 정답에는 최종 품질 판정을 유지"
            ),
            (
                "M3 관련 팀·역할의 경기 상태·집계·식별·역할 행 "
                "위험 차단; 관측 0은 리그 해당 역할의 위험 및 격리 "
                "없는 경우만; 관련 팀 밖 미확인 이동/공식 등록은 "
                "보장하지 않음"
            ),
            (
                "선수 집계는 관측 후보; 실제 팀 이동/등록·공식 출전 "
                "및 자책점 미확정; 잔여 시즌 완료 근거 없는 시즌 "
                "정답 금지"
            ),
            (
                "장기 정답의 교차 기간 중첩 존재, cutoff 기준 "
                "마스크 적용; 0 및 분모 0 구분"
            ),
            (
                "정규시즌/포스트시즌 공식 순위·동률·대진 규칙 "
                "후속 확인; 실제 포스트시즌 결과 확보 범위는 "
                "source.handoff의 시즌별 범위를 따름"
            ),
            (
                "선발 자료는 사후 관측 후보와 과거 이력만; "
                "당시 예정 선발 입력 없음"
            ),
            (
                "M5 위치 정답 미사용; 독립 예측기 6개, "
                "모델 학습·최종 Test 평가 미수행"
            ),
        ],
    }

    write_json(directory / "manifest.json", manifest)

    require(
        manifest["status"] == "completed",
        "동일 입력 재생성 불일치",
    )

    return {
        "run_id": output_id,
        "status": manifest["status"],
        "directory": str(directory),
        "splits": counts,
        "comparison": comparison,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="검증된 표준 자료에서 모델별 입력·정답 생성"
    )

    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--verification-id", required=True)
    parser.add_argument("--validation-start", required=True)
    parser.add_argument("--test-start", required=True)
    parser.add_argument("--observation-end", required=True)
    parser.add_argument("--compare-run")

    args = parser.parse_args(argv)

    try:
        settings = load_settings(args.project_root)

        logging.basicConfig(
            level=settings.log_level,
            format="%(levelname)s %(name)s: %(message)s",
        )

        result = build_model_data(
            settings,
            args.run_id,
            args.verification_id,
            args.validation_start,
            args.test_start,
            args.observation_end,
            compare_run=args.compare_run,
        )

    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        sqlite3.DatabaseError,
    ) as exc:
        LOGGER.error("모델 자료 생성·검증 실패: %s", exc)
        return 2

    except KeyboardInterrupt:
        LOGGER.warning("모델 자료 생성 중단")
        return 3

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())