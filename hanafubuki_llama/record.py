"""What an instance holds: the record the install script writes in ``core``, and its models."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RECORD = ".hanafubuki-llama.json"
STAGING = ".staging"


@dataclass(frozen=True)
class Model:
    """A model file in the models folder."""

    name: str
    size: int
    modified: float


def read_record(core: Path) -> dict[str, Any] | None:
    """The installed build, as the install script recorded it.

    Args:
        core: The instance's ``core`` directory.

    Returns:
        The record (``tag``, ``variant``, ``asset``, ...), or ``None`` when
        there is none or it does not read.
    """
    try:
        value = json.loads((core / RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def write_record(core: Path, record: dict[str, Any]) -> None:
    """Writes the record beside its old self, then moves it into place.

    Args:
        core: The instance's ``core`` directory.
        record: What to record.
    """
    target = core / RECORD
    partial = target.with_name(target.name + ".partial")
    partial.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, target)


def list_models(models: Path) -> list[Model]:
    """The GGUF files in the models folder and its subfolders.

    Args:
        models: The instance's ``models`` directory.

    Returns:
        The models by name; none when the folder does not exist.
    """
    found = []
    for path in sorted(models.rglob("*.gguf")) if models.is_dir() else []:
        try:
            stat = path.stat()
        except OSError:
            continue
        if path.is_file():
            found.append(Model(path.relative_to(models).as_posix(), stat.st_size, stat.st_mtime))
    return found
