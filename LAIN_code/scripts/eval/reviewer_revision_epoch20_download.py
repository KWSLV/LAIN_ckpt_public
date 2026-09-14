#!/usr/bin/env python3
"""Resume and verify reviewer-revision epoch-20 GitHub release assets."""

from __future__ import annotations

import concurrent.futures
import hashlib
import os
from pathlib import Path
import time

import requests


RUN_ROOT = Path("/root/autodl-tmp/Lain/reviewer_revision")
CHUNK_SIZE = 8 * 1024 * 1024
MAX_WORKERS = 24
PROXY_PREFIX = "https://ghproxy.net/"
ASSETS = (
    (
        "06_uo_text_rank8_seed66",
        706_468_912,
        "bcecc780735f6acf791d734fd41bbf84e5b82bd4ce9330737700058ddff9b41bc",
    ),
    (
        "08_uo_paper_gate_rank64_seed66",
        735_433_348,
        "30c03ae79aa241e4d273861f588d1f6a25408ae7c664d9f11a62a51e5b4d640c7",
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def range_has_data(fd: int, start: int, end: int) -> bool:
    try:
        data_start = os.lseek(fd, start, os.SEEK_DATA)
        hole_start = os.lseek(fd, start, os.SEEK_HOLE)
    except OSError:
        return False
    return data_start <= start and hole_start >= end


def download_asset(tag: str, size: int, expected_sha256: str) -> None:
    target = RUN_ROOT / tag / "checkpoints" / "ckpt_77900_20.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size == size:
        actual = sha256(target)
        if actual == expected_sha256:
            print(f"ALREADY_COMPLETE {tag} {size} {actual}", flush=True)
            return

    url = (
        f"{PROXY_PREFIX}https://github.com/KWSLV/LAIN_ckpt_public/"
        f"releases/download/{tag}/ckpt_77900_20.pt"
    )
    temporary = target.with_suffix(target.suffix + ".download")

    for full_retry in range(2):
        if not temporary.exists() or temporary.stat().st_size != size:
            temporary.unlink(missing_ok=True)
            with temporary.open("wb") as stream:
                stream.truncate(size)

        fd = os.open(temporary, os.O_RDWR)
        ranges = [
            (start, min(size, start + CHUNK_SIZE))
            for start in range(0, size, CHUNK_SIZE)
        ]
        complete = [pair for pair in ranges if range_has_data(fd, *pair)]
        missing = [pair for pair in ranges if pair not in complete]
        completed_bytes = sum(end - start for start, end in complete)
        began = time.time()
        print(
            f"RESUME {tag} {completed_bytes}/{size} "
            f"{completed_bytes / size:.1%} missing_chunks={len(missing)}",
            flush=True,
        )

        def fetch(pair: tuple[int, int]) -> int:
            start, end = pair
            expected_length = end - start
            for attempt in range(1, 21):
                try:
                    response = requests.get(
                        url,
                        headers={"Range": f"bytes={start}-{end - 1}"},
                        timeout=(20, 240),
                    )
                    if response.status_code != 206:
                        raise RuntimeError(f"HTTP {response.status_code}")
                    if len(response.content) != expected_length:
                        raise RuntimeError(
                            f"received {len(response.content)}, expected {expected_length}"
                        )
                    os.pwrite(fd, response.content, start)
                    return expected_length
                except Exception as error:
                    if attempt == 20:
                        raise RuntimeError(
                            f"range {start}-{end - 1} failed: {error}"
                        ) from error
                    time.sleep(min(60, attempt * 3))
            raise AssertionError("unreachable")

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = [pool.submit(fetch, pair) for pair in missing]
            for future in concurrent.futures.as_completed(futures):
                completed_bytes += future.result()
                elapsed = max(time.time() - began, 0.1)
                print(
                    f"DOWNLOAD {tag} {completed_bytes}/{size} "
                    f"{completed_bytes / size:.1%} "
                    f"new_speed={(completed_bytes - sum(e - s for s, e in complete)) / elapsed / 1048576:.2f}MiB/s",
                    flush=True,
                )
        os.close(fd)

        actual = sha256(temporary)
        print(f"SHA256 {tag} {actual}", flush=True)
        if actual == expected_sha256:
            os.replace(temporary, target)
            print(f"COMPLETE {target} {size}", flush=True)
            return

        print(f"CHECKSUM_MISMATCH {tag}; restarting cleanly", flush=True)
        temporary.unlink(missing_ok=True)

    raise RuntimeError(f"checksum verification failed twice for {tag}")


def main() -> None:
    for asset in ASSETS:
        download_asset(*asset)
    print("ALL_DOWNLOADS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
