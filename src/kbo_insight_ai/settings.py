"""프로젝트 설정을 읽고 저장 경로의 기본 조건을 검증한다."""

from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path
import stat
import tomllib


PATH_NAMES = (
    "raw_dir",
    "interim_dir",
    "processed_dir",
    "models_dir",
    "results_dir",
    "cache_dir",
)
CONFIG_KEYS = frozenset((*PATH_NAMES, "seed", "log_level"))
LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})
MAX_SEED = 2**32 - 1
PROTECTED_NAMES = (".git", ".github", ".venv", "src", "tests", "docs")


class SettingsError(ValueError):
    """설정이나 경로가 개발 기반의 계약을 만족하지 않을 때 발생한다."""


@dataclass(frozen=True)
class Settings:
    """검증이 끝난 프로젝트 경로와 실행 공통 설정."""

    project_root: Path
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    models_dir: Path
    results_dir: Path
    cache_dir: Path
    seed: int
    log_level: str


def _directory(path: Path, name: str, *, required: bool = False) -> Path:
    """경로를 정규화하고 기존 파일과의 충돌을 확인한다.

    없는 산출물 디렉터리는 허용하지만 생성하지 않는다.
    프로젝트 기준 디렉터리는 실제로 존재해야 한다.
    """
    try:
        resolved = path.resolve()
        for current in (resolved, *resolved.parents):
            try:
                mode = current.stat().st_mode
            except FileNotFoundError:
                if required and current == resolved:
                    raise SettingsError(
                        f"{name}: 존재하는 디렉터리가 필요합니다."
                    ) from None
                continue

            if not stat.S_ISDIR(mode):
                raise SettingsError(
                    f"{name}: 경로나 상위 경로가 파일입니다."
                )

        return resolved
    except (OSError, RuntimeError, ValueError) as exc:
        if isinstance(exc, SettingsError):
            raise
        raise SettingsError(
            f"{name}: 경로를 확인할 수 없습니다."
        ) from None


def _project_root(
    explicit: str | Path | None,
    environ: Mapping[str, str],
) -> Path:
    """명시 인자, 환경 변수, 편집 가능 설치 위치 순서로 기준을 선택한다."""
    selected = explicit
    if selected is None:
        selected = environ.get("KBO_PROJECT_ROOT")

    if selected is None:
        module_path = Path(__file__).resolve()
        if module_path.parents[1].name != "src":
            raise SettingsError(
                "project_root: --project-root 또는 KBO_PROJECT_ROOT가 필요합니다."
            )
        selected = module_path.parents[2]

    try:
        root = Path(selected)
        if not root.is_absolute():
            raise SettingsError(
                "project_root: 절대 경로를 지정해야 합니다."
            )
    except (TypeError, ValueError):
        raise SettingsError(
            "project_root: 올바른 절대 경로가 필요합니다."
        ) from None

    return _directory(root, "project_root", required=True)


def _overlaps(first: Path, second: Path) -> bool:
    """같은 경로이거나 한쪽이 다른 쪽의 상위 경로인지 반환한다."""
    return (
        first == second
        or first in second.parents
        or second in first.parents
    )


def load_settings(
    project_root: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> Settings:
    """설정을 읽고 타입·범위·경로를 검증해 반환한다.

    설정값 자체나 하위 입출력 예외의 원문은 오류에 포함하지 않는다.
    이 함수는 설정을 읽기만 하며 파일과 디렉터리를 생성하지 않는다.
    """
    environment = os.environ if environ is None else environ
    root = _project_root(project_root, environment)

    try:
        with (root / "kbo.toml").open("rb") as stream:
            config: dict[str, object] = tomllib.load(stream)
    except (UnicodeDecodeError, tomllib.TOMLDecodeError):
        raise SettingsError(
            "kbo.toml: UTF-8 인코딩과 TOML 문법을 확인하세요."
        ) from None
    except OSError:
        raise SettingsError(
            "kbo.toml: 설정 파일의 존재와 읽기 권한을 확인하세요."
        ) from None

    if set(config) != CONFIG_KEYS:
        raise SettingsError(
            "kbo.toml: 필수 설정의 누락 또는 지원하지 않는 설정이 있습니다."
        )

    seed = config["seed"]
    if type(seed) is not int or not 0 <= seed <= MAX_SEED:
        raise SettingsError(
            f"seed: 0 이상 {MAX_SEED} 이하의 정수가 필요합니다."
        )

    log_level = config["log_level"]
    if not isinstance(log_level, str) or log_level not in LOG_LEVELS:
        raise SettingsError(
            "log_level: DEBUG, INFO, WARNING, ERROR, CRITICAL 중 하나가 필요합니다."
        )

    paths: dict[str, Path] = {}
    for name in PATH_NAMES:
        value = config[name]
        if not isinstance(value, str) or not value.strip():
            raise SettingsError(
                f"{name}: 비어 있지 않은 경로 문자열이 필요합니다."
            )

        candidate = Path(value)
        if not candidate.is_absolute():
            if candidate.drive or candidate.root:
                raise SettingsError(
                    f"{name}: 드라이브 상대경로나 루트 상대경로는 허용하지 않습니다."
                )
            candidate = root / candidate
        resolved = _directory(candidate, name)

        if resolved == root or resolved in root.parents:
            raise SettingsError(
                f"{name}: 프로젝트 기준 또는 그 상위 경로는 사용할 수 없습니다."
            )

        if any(
            _overlaps(resolved, root / protected)
            for protected in PROTECTED_NAMES
        ):
            raise SettingsError(
                f"{name}: 소스·문서·환경·Git 관리 경로와 겹칠 수 없습니다."
            )

        if any(_overlaps(resolved, other) for other in paths.values()):
            raise SettingsError(
                f"{name}: 다른 산출물 경로와 겹칠 수 없습니다."
            )

        paths[name] = resolved

    return Settings(
        project_root=root,
        raw_dir=paths["raw_dir"],
        interim_dir=paths["interim_dir"],
        processed_dir=paths["processed_dir"],
        models_dir=paths["models_dir"],
        results_dir=paths["results_dir"],
        cache_dir=paths["cache_dir"],
        seed=seed,
        log_level=log_level,
    )