"""Build deterministic AMC source and reproducibility archives.

The ZIP members are sorted, carry fixed timestamps and permissions, and include
an internal SHA-256 manifest. Running this script twice on unchanged inputs
therefore produces byte-identical archives.

Run from any directory::

    python paper-amc/scripts/build_submission_archives.py
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AMC = ROOT / "paper-amc"
OUT = AMC / "submission-artifacts"
FIXED_TIME = (1980, 1, 1, 0, 0, 0)

SOURCE_FILES = (
    "main.tex",
    "main.bbl",
    "supplement.tex",
    "supplement.bbl",
    "refs.bib",
    "cas-sc.cls",
    "cas-common.sty",
    "elsarticle-num-names.bst",
    "tab_race.tex",
    "tab_cost.tex",
    "tab_verification.tex",
    "tab_decomposition.tex",
    "tab_regimes.tex",
    "tab_floorcap.tex",
    "tab_kernelshape.tex",
)

SOURCE_GLOBS = (
    "figures/*.pdf",
    "thumbnails/*.jpeg",
)

REPRO_FILES = (
    "CITATION.cff",
    "LICENSE",
    "README.md",
    "pyproject.toml",
    "paper-amc/README.md",
    "paper-amc/main.tex",
    "paper-amc/main.bbl",
    "paper-amc/supplement.tex",
    "paper-amc/supplement.bbl",
    "paper-amc/refs.bib",
    "paper-amc/cas-sc.cls",
    "paper-amc/cas-common.sty",
    "paper-amc/elsarticle-num-names.bst",
    "paper-amc/numerics_generated.tex",
    "paper-amc/numerics_key_numbers.json",
    "paper-amc/highlights.txt",
)

REPRO_GLOBS = (
    "src/dcascade/*.py",
    "scripts/*.py",
    "tests/*.py",
    "results/key_numbers.json",
    "results/robustness_summary.json",
    "results/grid.npz",
    "results/tables/*",
    "results/figures/*",
    "paper/*.tex",
    "paper/figures/*.pdf",
    "paper-amc/scripts/*.py",
    "paper-amc/tab_*.tex",
    "paper-amc/figures/*.pdf",
    "paper-amc/thumbnails/*.jpeg",
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _collect(base: Path, names: tuple[str, ...], globs: tuple[str, ...]) -> list[Path]:
    paths = {base / name for name in names}
    unmatched = []
    for pattern in globs:
        matches = [path for path in base.glob(pattern) if path.is_file()]
        if not matches:
            unmatched.append(pattern)
        paths.update(matches)
    missing = sorted(str(path.relative_to(base)) for path in paths if not path.is_file())
    if missing or unmatched:
        details = missing + [f"unmatched pattern {pattern}" for pattern in unmatched]
        raise SystemExit("missing archive inputs: " + ", ".join(details))
    return sorted(paths, key=lambda path: path.relative_to(base).as_posix())


def _git_state() -> tuple[str | None, bool | None]:
    """Return the HEAD used as the archive base and whether it has local edits."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return commit, bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        return None, None


def _environment(commit: str | None, worktree_dirty: bool | None) -> bytes:
    direct = ("numpy", "scipy", "matplotlib", "pandas", "egttools", "mpmath", "pytest")
    versions = {}
    for package in direct:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    payload = {
        "base_git_commit": commit,
        "git_worktree_dirty": worktree_dirty,
        "packages": versions,
        "python": sys.version.split()[0],
        "schema": 2,
    }
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_zip(
    destination: Path,
    members: dict[str, bytes],
    virtual_members: dict[str, bytes] | None = None,
) -> None:
    payload = dict(members)
    payload.update(virtual_members or {})
    manifest = "".join(
        f"{_sha256(data)}  {name}\n" for name, data in sorted(payload.items())
    ).encode("utf-8")
    payload["MANIFEST.sha256"] = manifest

    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        destination,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for name, data in sorted(payload.items()):
            info = zipfile.ZipInfo(name, date_time=FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def _members(base: Path, paths: list[Path]) -> dict[str, bytes]:
    return {path.relative_to(base).as_posix(): path.read_bytes() for path in paths}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-clean",
        action="store_true",
        help="refuse to build unless HEAD exists and the Git worktree is clean",
    )
    args = parser.parse_args()
    commit, worktree_dirty = _git_state()
    if args.require_clean and (commit is None or worktree_dirty is not False):
        raise SystemExit(
            "refusing final archives: commit the exact tested state and make "
            "the Git worktree clean before using --require-clean"
        )

    source_paths = _collect(AMC, SOURCE_FILES, SOURCE_GLOBS)
    source_zip = OUT / "delegation-cascade-amc-source.zip"
    _write_zip(source_zip, _members(AMC, source_paths))

    repro_paths = _collect(ROOT, REPRO_FILES, REPRO_GLOBS)
    repro_zip = OUT / "delegation-cascade-reproducibility.zip"
    _write_zip(
        repro_zip,
        _members(ROOT, repro_paths),
        {"ENVIRONMENT.json": _environment(commit, worktree_dirty)},
    )

    archives = (source_zip, repro_zip)
    checksum_text = "".join(
        f"{_sha256(path.read_bytes())}  {path.name}\n" for path in archives
    )
    (OUT / "submission_artifacts.sha256").write_text(checksum_text, encoding="ascii")
    summary = {
        "archives": [
            {
                "bytes": path.stat().st_size,
                "file": path.name,
                "sha256": _sha256(path.read_bytes()),
            }
            for path in archives
        ],
        "schema": 1,
    }
    (OUT / "submission_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for item in summary["archives"]:
        print(item["file"], item["bytes"], item["sha256"])
    if worktree_dirty is not False:
        print(
            "WARNING: QA archives record git_worktree_dirty="
            f"{json.dumps(worktree_dirty)}; use --require-clean for the final release"
        )


if __name__ == "__main__":
    main()
