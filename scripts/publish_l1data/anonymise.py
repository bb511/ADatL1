# Scrubbing rules for the published dataset, in one place so both exporters agree.
#
# The data itself is clean: parquet footers carry only the writer library version and
# awkward's structural metadata. What leaks is the packaging -- tar member ownership,
# resolved Hydra paths, and commit SHAs -- so that is what this module handles.
import argparse
import re
import sys
import tarfile
from pathlib import Path

# Everything the deposition must never contain. Matched case-insensitively.
BLOCKLIST = [
    "deodagiu",
    "podagiu",
    "olqti",
    "clariden",
    "cscs",
    "ethz",
    "bb511",
    "cdfpzmvpvg",
    "cernbox",
    "a0166",
    "odagiu",
    "/Users/",
    "/data/",
    "/users/",
    "/iopsstor/",
]

# Bare git SHAs resolve to a repo and its author through GitHub's commit search.
SHA_RE = re.compile(rb"(?<![0-9a-zA-Z])[0-9a-f]{40}(?![0-9a-zA-Z])")

STRAY_GLOBS = [".DS_Store", "._*", ".AppleDouble", "Thumbs.db"]

# Fixed timestamp so repeated runs produce byte-identical tars, which is what makes
# the published sha256sums verifiable by a third party.
SOURCE_DATE = "UTC 2026-01-01"


def tar_cmd(export_root: Path, member: str, out_path: Path) -> list[str]:
    """GNU tar invocation that records no owner, no timestamp and no absolute path.

    :param export_root: Directory to run tar from; member is relative to it.
    """
    return [
        "tar",
        "--sort=name",
        "--owner=0",
        "--group=0",
        "--numeric-owner",
        f"--mtime={SOURCE_DATE}",
        "--format=pax",
        "--pax-option=delete=atime,delete=ctime",
        "--no-acls",
        "--no-xattrs",
        "--no-selinux",
        "-C",
        str(export_root),
        "-cf",
        str(out_path),
        member,
    ]


def prune_stray(root: Path) -> list[Path]:
    """Delete macOS and Windows turds that record directory listings or download URLs."""
    removed = []
    for pattern in STRAY_GLOBS:
        for path in root.rglob(pattern):
            path.unlink()
            removed.append(path)

    return removed


def scrub_resolved_config(cfg: dict, raw_data_dir: str) -> dict:
    """Strip machine-specific paths out of a resolved Hydra config.

    Drops the `paths` and `hydra` nodes, rewrites every `<raw_data_dir>/X` to a bare `X`
    so dataset entries stay readable, and replaces any other absolute path with
    `<local>`. The dump is meant to record how the data was produced -- features,
    filters, split fractions -- not where the cache directories happened to live.
    """
    prefix = raw_data_dir.rstrip("/") + "/"

    def walk(node):
        if isinstance(node, dict):
            return {k: walk(v) for k, v in node.items() if k not in ("paths", "hydra")}
        if isinstance(node, list):
            return [walk(v) for v in node]
        if isinstance(node, str):
            stripped = node.replace(prefix, "").replace(raw_data_dir.rstrip("/"), "")
            return "<local>" if stripped.startswith("/") else stripped
        return node

    return walk(cfg)


def _hits(blob: bytes, where: str) -> list[str]:
    lowered = blob.lower()
    found = [f"{where}: {term!r}" for term in BLOCKLIST if term.lower().encode() in lowered]
    found += [f"{where}: git sha {m.group().decode()}" for m in SHA_RE.finditer(blob)]

    return found


def _scan_parquet_footer(path: Path) -> bytes:
    """Return just the metadata a parquet reader exposes, not the 500 MB of columns."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    parts = [str(pf.metadata.created_by).encode()]
    parts += [k + b":" + v for k, v in (pf.schema_arrow.metadata or {}).items()]

    return b"\n".join(parts)


def scan(root: Path, deep: bool = False) -> list[str]:
    """Walk the export tree and every finished tar, looking for anything identifying.

    :param deep: Also scan parquet column data, not just the footer metadata.
    """
    findings = []
    for pattern in STRAY_GLOBS:
        findings += [
            f"{p.relative_to(root)}: stray file, prune before packaging"
            for p in root.rglob(pattern)
        ]

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        findings += _hits(str(rel).encode(), f"{rel} (name)")

        if path.suffix == ".parquet" and not deep:
            findings += _hits(_scan_parquet_footer(path), f"{rel} (footer)")
        elif path.suffix == ".tar":
            findings += _scan_tar(path, rel)
        else:
            findings += _hits(path.read_bytes(), str(rel))

    return findings


def _scan_tar(path: Path, rel: Path) -> list[str]:
    """Check member ownership and names inside a finished archive."""
    findings = []
    with tarfile.open(path) as tar:
        for member in tar:
            if member.uid or member.gid or member.uname or member.gname:
                findings.append(
                    f"{rel}: member {member.name} owned by "
                    f"{member.uname or member.uid}/{member.gname or member.gid}"
                )
            findings += _hits(member.name.encode(), f"{rel}!{member.name} (name)")

    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="export root to scan")
    parser.add_argument("--deep", action="store_true", help="scan parquet payloads too")
    args = parser.parse_args()

    findings = scan(args.root, deep=args.deep)
    if not findings:
        print(f"clean: nothing identifying under {args.root}")
        return 0

    print(f"{len(findings)} leak(s) under {args.root}:")
    for finding in findings:
        print(f"  {finding}")

    return 1


if __name__ == "__main__":
    sys.exit(main())
