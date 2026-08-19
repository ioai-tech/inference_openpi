"""Discover Orbax checkpoints without assuming a fixed directory layout."""

from __future__ import annotations

from pathlib import Path


def is_orbax_checkpoint(path: Path) -> bool:
    return (path / "_CHECKPOINT_METADATA").is_file() and (path / "params").exists()


def _step_hint(path: Path) -> int:
    try:
        return int(path.name)
    except ValueError:
        return -1


def discover_checkpoints(root: Path) -> list[Path]:
    root = root.expanduser().resolve()
    if not root.exists():
        return []
    found: list[Path] = []
    if is_orbax_checkpoint(root):
        found.append(root)
    for metadata in root.rglob("_CHECKPOINT_METADATA"):
        checkpoint = metadata.parent
        if checkpoint not in found and (checkpoint / "params").exists():
            found.append(checkpoint)
    found.sort(key=lambda path: (_step_hint(path), path.stat().st_mtime), reverse=True)
    return found


def resolve_checkpoint(path: str | Path | None, *, search_root: Path) -> Path:
    if path:
        checkpoint = Path(path).expanduser().resolve()
        if not is_orbax_checkpoint(checkpoint):
            raise FileNotFoundError(
                f"Not an Orbax checkpoint (need _CHECKPOINT_METADATA and params/): {checkpoint}"
            )
        return checkpoint

    matches = discover_checkpoints(search_root)
    if not matches:
        raise FileNotFoundError(
            f"No Orbax checkpoint under {search_root}. "
            "Place weights so the directory contains _CHECKPOINT_METADATA and params/."
        )
    return matches[0]
