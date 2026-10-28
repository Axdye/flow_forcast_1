"""Download upstream raw datasets and verify their sha256.

The repository intentionally does not vendor datasets.  This script fetches the
*original* files (the ones an open-source mirror publishes) into a scratch
directory, verifies size and sha256 against ``provenance/raw_sources.json``, and
never leaves a partially downloaded file behind.  Conversion into the layout
``prepare-data`` expects is a separate step: ``scripts/import_raw.py``.

Usage::

    python scripts/fetch_data.py --all --dest raw
    python scripts/fetch_data.py --dataset PEMS08 METRLA --dest raw
    python scripts/fetch_data.py --all --dest raw --verify-only

A manifest entry looks like::

    {
      "name": "PEMS08.npz",
      "role": "series",
      "source": "ASTGNN",
      "license": "research use, no re-licence claimed",
      "urls": ["https://raw.githubusercontent.com/.../PEMS08.npz",
               "archive:https://codeload.github.com/.../tar.gz/refs/heads/main#ASTGNN-main/data/PEMS08/PEMS08.npz"],
      "sha256": "...",
      "size": 18525419
    }

``urls`` is tried in order.  An ``archive:`` URL downloads a tarball once and
extracts a single member; this is used when a direct raw file is unreliable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "provenance" / "raw_sources.json"
DEFAULT_DEST = REPO_ROOT / "raw"
CHUNK = 1 << 20
USER_AGENT = "sbona-fetch/1.0 (+dataset provenance verification)"


class FetchError(RuntimeError):
    """Raised when a file cannot be downloaded or fails verification."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FetchError(f"manifest not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if document.get("schema_version") != 1:
        raise FetchError(f"unsupported manifest schema_version: {document.get('schema_version')!r}")
    if not isinstance(document.get("datasets"), dict):
        raise FetchError("manifest has no 'datasets' object")
    return document


def verify_file(path: Path, entry: dict[str, Any]) -> None:
    """Raise ``FetchError`` unless ``path`` matches the recorded size and hash."""
    if not path.is_file():
        raise FetchError(f"missing file: {path}")
    expected_size = entry.get("size")
    if expected_size is not None and path.stat().st_size != int(expected_size):
        raise FetchError(
            f"size mismatch for {path.name}: got {path.stat().st_size}, expected {expected_size}"
        )
    actual = sha256_file(path)
    expected = str(entry["sha256"]).lower()
    if actual != expected:
        raise FetchError(f"sha256 mismatch for {path.name}: got {actual}, expected {expected}")


def _download_direct(url: str, part: Path, timeout: float, retries: int) -> None:
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response, part.open("wb") as out:
                shutil.copyfileobj(response, out, CHUNK)
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            last = exc
            part.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(min(2 * attempt, 10))
    raise FetchError(f"download failed after {retries} attempts: {url} ({last})")


def _extract_member(
    archive_url: str,
    member: str,
    dest: Path,
    timeout: float,
    retries: int,
    cache_dir: Path | None = None,
) -> None:
    cache_dir = cache_dir or Path(tempfile.gettempdir()) / "sbona-raw-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(archive_url.encode("utf-8")).hexdigest()[:16]
    archive = cache_dir / f"{key}.tar.gz"
    if not archive.is_file():
        _download_direct(archive_url, archive, timeout, retries)
    try:
        with tarfile.open(archive, "r:gz") as tar:
            extracted = tar.extractfile(member)
            if extracted is None:
                raise FetchError(f"member {member!r} not found in {archive_url}")
            with dest.open("wb") as out:
                shutil.copyfileobj(extracted, out, CHUNK)
    except tarfile.TarError as exc:
        archive.unlink(missing_ok=True)
        raise FetchError(f"cannot read archive {archive_url}: {exc}") from exc


def _fetch_one(
    url: str,
    dest: Path,
    timeout: float,
    retries: int,
    cache_dir: Path | None = None,
) -> None:
    if url.startswith("archive:"):
        rest = url[len("archive:") :]
        if "#" not in rest:
            raise FetchError(f"archive URL needs '#member': {url}")
        archive_url, member = rest.rsplit("#", 1)
        _extract_member(archive_url, member, dest, timeout, retries, cache_dir)
    else:
        _download_direct(url, dest, timeout, retries)


def fetch_entry(
    entry: dict[str, Any],
    dest_dir: Path,
    *,
    force: bool,
    timeout: float,
    retries: int,
    cache_dir: Path | None = None,
) -> str:
    """Ensure ``dest_dir/entry['name']`` exists and verifies.  Returns 'cached' | 'downloaded'."""
    name = entry["name"]
    dest = dest_dir / name
    if dest.is_file() and not force:
        verify_file(dest, entry)
        return "cached"

    dest_dir.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    urls = entry.get("urls") or []
    if not urls:
        raise FetchError(f"{name}: manifest entry has no 'urls'")
    errors: list[str] = []
    for url in urls:
        try:
            _fetch_one(url, part, timeout, retries, cache_dir)
            verify_file(part, entry)
            part.replace(dest)
            return "downloaded"
        except FetchError as exc:
            errors.append(f"{url}: {exc}")
            part.unlink(missing_ok=True)
    raise FetchError(f"{name}: all sources failed\n  " + "\n  ".join(errors))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="scratch directory for raw files")
    parser.add_argument("--dataset", nargs="+", action="extend", default=None, metavar="NAME")
    parser.add_argument("--all", action="store_true", help="fetch every dataset in the manifest")
    parser.add_argument("--verify-only", action="store_true", help="do not download; verify existing files")
    parser.add_argument("--force", action="store_true", help="re-download even when a verified file exists")
    parser.add_argument("--timeout", type=float, default=60.0, help="per-request timeout in seconds")
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args(argv)

    try:
        document = load_manifest(args.manifest)
        datasets: dict[str, Any] = document["datasets"]
        selected = list(datasets) if args.all else (args.dataset or [])
        if not selected:
            parser.error("pass --all or --dataset NAME (available: " + ", ".join(sorted(datasets)) + ")")
        unknown = [name for name in selected if name not in datasets]
        if unknown:
            raise FetchError(f"unknown dataset(s): {unknown}; available: {sorted(datasets)}")

        failures = 0
        for dataset in selected:
            info = datasets[dataset]
            dest_dir = args.dest / dataset
            print(f"[{dataset}] -> {dest_dir}")
            for entry in info["files"]:
                label = f"  {entry['name']:<24}"
                try:
                    if args.verify_only:
                        verify_file(dest_dir / entry["name"], entry)
                        print(f"{label} ok")
                    else:
                        status = fetch_entry(
                            entry,
                            dest_dir,
                            force=args.force,
                            timeout=args.timeout,
                            retries=args.retries,
                            cache_dir=args.dest / ".cache",
                        )
                        print(f"{label} {status} sha256={str(entry['sha256'])[:16]}")
                except FetchError as exc:
                    failures += 1
                    print(f"{label} FAILED: {exc}", file=sys.stderr)
        if failures:
            print(f"{failures} file(s) failed", file=sys.stderr)
            return 1
        print(f"all files verified under {args.dest}")
        return 0
    except FetchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
