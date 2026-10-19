from pathlib import Path


def is_valid_ckpt(p: Path) -> bool:
    return p.is_file() and p.suffix == ".ckpt" and not p.name.startswith("._")
