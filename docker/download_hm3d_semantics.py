#!/usr/bin/env python3
"""Download official HM3D v0.2 validation semantics with in-memory credentials."""

from __future__ import annotations

import getpass
import concurrent.futures
import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "data/scene_datasets/hm3d_v0.2"
CACHE = ROOT / "data/.downloads"
PACKAGES = (
    "hm3d-val-semantic-configs-v0.2.tar",
    "hm3d-val-semantic-annots-v0.2.tar",
)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download_chunks(name: str, url: str, configuration: bytes) -> Path:
    probe = subprocess.run([
        "curl", "--config", "-", "--http2", "--location", "--fail", "--silent",
        "--retry", "3", "--retry-all-errors", "--connect-timeout", "15",
        "--max-time", "45", "--range", "0-0", "--dump-header", "-",
        "--output", "/dev/null", url,
    ], input=configuration, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ranges = re.findall(rb"content-range:\s*bytes\s+\d+-\d+/(\d+)", probe.stdout, re.I)
    if probe.returncode or not ranges:
        raise RuntimeError(f"Cannot determine official archive size (curl={probe.returncode})")
    total = int(ranges[-1])
    chunk_size = 2 * 1024 * 1024
    chunk_directory = CACHE / (name + ".chunks-2m")
    chunk_directory.mkdir(exist_ok=True)
    count = (total + chunk_size - 1) // chunk_size
    # Reuse complete prefixes of the previous larger chunks, including partial
    # transfers, so switching transfer sizes does not discard received data.
    old_directory = CACHE / (name + ".chunks")
    sources = [(p, int(p.stem) * 8 * 1024 * 1024)
               for p in old_directory.glob("*") if p.stem.isdecimal()]
    legacy_partial = CACHE / (name + ".part")
    if legacy_partial.is_file():
        sources.append((legacy_partial, 0))
    for source_path, base in sources:
        with source_path.open("rb") as source:
            for offset in range(0, source_path.stat().st_size - chunk_size + 1, chunk_size):
                index = (base + offset) // chunk_size
                chunk = chunk_directory / f"{index:04d}.bin"
                if chunk.is_file():
                    continue
                source.seek(offset)
                chunk.write_bytes(source.read(chunk_size))
    print(f"[CHUNKED] {name} total={total / 1e6:.1f} MB chunks={count} connections=16", flush=True)

    def get_chunk(index: int) -> Path:
        start, end = index * chunk_size, min((index + 1) * chunk_size, total) - 1
        expected = end - start + 1
        chunk = chunk_directory / f"{index:04d}.bin"
        if chunk.is_file() and chunk.stat().st_size == expected:
            return chunk
        temporary = chunk.with_suffix(".part")
        for attempt in range(12):
            received = temporary.stat().st_size if temporary.exists() else 0
            if received == expected:
                os.replace(temporary, chunk)
                return chunk
            if received > expected:
                raise RuntimeError("Invalid cached chunk size")
            tail = chunk.with_suffix(".tail")
            result = subprocess.run([
                "curl", "--config", "-", "--http2", "--location", "--fail",
                "--silent", "--connect-timeout", "15", "--max-time", "60",
                "--range", f"{start + received}-{end}", "--output", str(tail),
                "--write-out", "%{http_code}", url,
            ], input=configuration, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.stdout.strip() == b"206" and tail.exists():
                if tail.stat().st_size > expected - received:
                    raise RuntimeError("Unexpected range response length")
                with temporary.open("ab") as output, tail.open("rb") as source:
                    shutil.copyfileobj(source, output)
                tail.unlink()
            if temporary.is_file() and temporary.stat().st_size == expected:
                os.replace(temporary, chunk)
                return chunk
            time.sleep(1 + attempt)
        raise RuntimeError(f"Archive range {index} failed; completed and partial chunks retained")

    last_report = time.monotonic()
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(get_chunk, index) for index in range(count)]
        for future in concurrent.futures.as_completed(futures):
            completed += future.result().stat().st_size
            if time.monotonic() - last_report >= 15 or completed == total:
                print(f"[PROGRESS] {name} completed={completed / 1e6:.1f}/{total / 1e6:.1f} MB", flush=True)
                last_report = time.monotonic()
    assembled = CACHE / (name + ".assembling")
    with assembled.open("wb") as output:
        for index in range(count):
            with (chunk_directory / f"{index:04d}.bin").open("rb") as source:
                shutil.copyfileobj(source, output, length=4 * 1024 * 1024)
    if assembled.stat().st_size != total or not tarfile.is_tarfile(assembled):
        raise RuntimeError("Assembled TAR archive failed validation")
    target = CACHE / name
    os.replace(assembled, target)
    print(f"[DOWNLOADED] {name} bytes={total}", flush=True)
    return target


def download(name: str, credentials: tuple[str, str]) -> Path:
    target = CACHE / name
    partial = target.with_suffix(target.suffix + ".part")
    if target.is_file() and tarfile.is_tarfile(target):
        print(f"[CACHED] {name} bytes={target.stat().st_size}", flush=True)
        return target
    url = f"https://api.matterport.com/resources/habitat/{name}"
    if any(character in "".join(credentials) for character in "\n\r"):
        raise RuntimeError("Token fields must each be a single line")
    # Curl negotiates HTTP/2 with the official endpoint. Passing its configuration
    # through a pipe keeps the token out of command arguments and disk files.
    credential = ":".join(credentials).replace("\\", "\\\\").replace('"', '\\"')
    configuration = f'user = "{credential}"\n'.encode()
    if "-annots-" in name:
        return download_chunks(name, url, configuration)
    print(f"[DOWNLOADING] {name}", flush=True)
    process = subprocess.Popen([
        "curl", "--config", "-", "--http2", "--location", "--fail",
        "--silent", "--show-error", "--retry", "8", "--retry-all-errors", "--retry-delay", "3",
        "--connect-timeout", "20", "--speed-limit", "1024", "--speed-time", "120",
        "--continue-at", "-", "--output", str(partial),
        "--write-out", "%{http_code}", url,
    ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process.stdin.write(configuration)
    process.stdin.close()
    last_report = time.monotonic()
    while process.poll() is None:
        if time.monotonic() - last_report >= 15:
            size = partial.stat().st_size if partial.exists() else 0
            print(f"[PROGRESS] {name} {size / 1e6:.1f} MB", flush=True)
            last_report = time.monotonic()
        time.sleep(1)
    status = process.stdout.read().decode().strip()
    process.stderr.read()  # Never log potentially signed redirect URLs.
    if process.returncode:
        raise RuntimeError(f"Official download failed: curl={process.returncode}, HTTP={status}")
    if not tarfile.is_tarfile(partial):
        raise RuntimeError("Server response is not a valid TAR archive")
    partial.rename(target)
    print(f"[DOWNLOADED] {name} bytes={target.stat().st_size}", flush=True)
    return target


def extract(archive_path: Path) -> dict:
    scene_directories = {
        scene.name.removesuffix(".basis.glb"): scene.parent
        for scene in (DESTINATION / "val").glob("*/*.basis.glb")
    }
    installed, unchanged, skipped = 0, 0, 0
    with tarfile.open(archive_path, "r:*") as archive:
        for member in archive:
            parts = PurePosixPath(member.name)
            if parts.is_absolute() or ".." in parts.parts:
                raise RuntimeError("Unsafe archive member path")
            if member.isdir():
                continue
            if not member.isfile():
                raise RuntimeError("Unexpected non-regular archive member")
            name = parts.name
            if name.endswith((".semantic.glb", ".semantic.txt")):
                scene_id = name.split(".semantic.", 1)[0]
                if scene_id not in scene_directories:
                    skipped += 1
                    continue
                target = scene_directories[scene_id] / name
            elif name == "hm3d_annotated_basis.scene_dataset_config.json":
                target = DESTINATION / name
            elif name.endswith(".scene_dataset_config.json"):
                target = DESTINATION / "val" / name
            else:
                skipped += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".download-part")
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError("Cannot read archive member")
            try:
                with source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output, length=4 * 1024 * 1024)
                if temporary.stat().st_size != member.size:
                    raise RuntimeError("Archive member size mismatch")
                if target.exists():
                    if digest(target) != digest(temporary):
                        raise RuntimeError(f"Existing file differs; refusing to overwrite {target.name}")
                    temporary.unlink()
                    unchanged += 1
                else:
                    os.replace(temporary, target)
                    installed += 1
            finally:
                temporary.unlink(missing_ok=True)
    result = {"package": archive_path.name, "bytes": archive_path.stat().st_size,
              "sha256": digest(archive_path), "installed": installed,
              "unchanged": unchanged, "skipped": skipped}
    print(f"[EXTRACTED] {archive_path.name} installed={installed} unchanged={unchanged} skipped={skipped}", flush=True)
    return result


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    if not list((DESTINATION / "val").glob("*/*.basis.glb")):
        raise RuntimeError("Validation geometry must be installed first")
    token_id = getpass.getpass("Matterport token ID (hidden): ")
    token_secret = getpass.getpass("Matterport token secret (hidden): ")
    credentials = (token_id, token_secret)
    results = []
    for name in PACKAGES:
        results.append(extract(download(name, credentials)))
    (CACHE / "hm3d-val-semantics-manifest.json").write_text(
        json.dumps({"packages": results}, indent=2) + "\n"
    )
    print("HM3D_SEMANTICS_DOWNLOAD_OK", flush=True)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(f"ERROR: {error}", flush=True)
        raise SystemExit(1)
