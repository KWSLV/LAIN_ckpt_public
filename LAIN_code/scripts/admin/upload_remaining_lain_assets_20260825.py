#!/root/miniconda3/bin/python
"""Upload remaining LAIN checkpoints and per-run training records to Releases.

The script is restart-safe: it compares SHA-256 digests against every existing
asset in the repository, skips already uploaded content, and never deletes any
source file. Only temporary deterministic training-record archives are removed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request


OWNER = "KWSLV"
REPO = "LAIN_ckpt_public"
RUN_ROOT = Path("/root/autodl-tmp/Lain")
WORK_ROOT = Path("/root/autodl-tmp/lain_public_release_upload_20260825")
STATUS_PATH = Path("/root/lain_public_release_upload_20260825_status.json")
MANIFEST_PATH = Path("/root/lain_public_release_upload_20260825_manifest.json")
HASH_CACHE_PATH = Path("/root/lain_public_release_upload_20260825_hash_cache.json")
MAX_ASSET_BYTES = 2_000_000_000

LEGACY_RELEASE_TAGS = {
    "UO_a4_ep9_objinherit_objectonly_r4_seed66":
        "UO_a4_ep9_objinherit_objectonly_r4",
    "UO_a4_ep9_objreset_objectonly_r4_seed66":
        "UO-a4-ep9-objectonly-r4",
}


def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def load_json(path: Path, default: object) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError, TypeError):
        return default


def load_token(required: bool) -> str:
    for key in ("GH_TOKEN", "GITHUB_TOKEN"):
        value = os.environ.get(key, "").strip()
        if value:
            return value
    hosts = Path("/root/.config/gh/hosts.yml")
    if hosts.is_file():
        match = re.search(
            r"^\s*oauth_token:\s*([^\s]+)\s*$",
            hosts.read_text(encoding="utf-8", errors="ignore"),
            re.MULTILINE,
        )
        if match:
            return match.group(1)
    if required:
        raise RuntimeError(
            "GitHub CLI authentication is missing; run `gh auth login -h github.com -w`."
        )
    return ""


class GitHub:
    def __init__(self, token: str):
        self.token = token
        self.releases: dict[str, dict] = {}
        self.digest_locations: dict[str, list[dict]] = {}

    def api(self, method: str, path: str, payload: object | None = None) -> object:
        url = "https://api.github.com" + path
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "LAIN-public-release-uploader-20260825",
        }
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        if body is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            url, data=body, method=method, headers=headers
        )
        last_error: Exception | None = None
        for attempt in range(6):
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    raw = response.read()
                    return json.loads(raw.decode("utf-8")) if raw else {}
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[-1500:]
                if exc.code in {502, 503, 504} and attempt < 5:
                    last_error = RuntimeError(f"HTTP {exc.code}: {detail}")
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(
                    f"GitHub API {method} {path} failed HTTP {exc.code}: {detail}"
                ) from exc
            except (OSError, TimeoutError) as exc:
                last_error = exc
                if attempt == 5:
                    break
                time.sleep(2 ** attempt)
        raise RuntimeError(f"GitHub API failed after retries: {last_error}")

    def refresh(self) -> None:
        releases: list[dict] = []
        page = 1
        while True:
            batch = self.api(
                "GET",
                f"/repos/{OWNER}/{REPO}/releases?per_page=100&page={page}",
            )
            assert isinstance(batch, list)
            releases.extend(batch)
            if len(batch) < 100:
                break
            page += 1

        self.releases = {}
        self.digest_locations = {}
        for release in releases:
            tag = release["tag_name"]
            assets: list[dict] = []
            page = 1
            while True:
                batch = self.api(
                    "GET",
                    f"/repos/{OWNER}/{REPO}/releases/{release['id']}/assets"
                    f"?per_page=100&page={page}",
                )
                assert isinstance(batch, list)
                assets.extend(batch)
                if len(batch) < 100:
                    break
                page += 1
            release["assets_full"] = assets
            self.releases[tag] = release
            for asset in assets:
                digest = (asset.get("digest") or "").removeprefix("sha256:")
                if digest:
                    self.digest_locations.setdefault(digest, []).append({
                        "tag": tag,
                        "name": asset["name"],
                        "size": int(asset.get("size", -1)),
                        "url": asset.get("browser_download_url"),
                    })

    def get_or_create_release(self, tag: str, run_name: str) -> dict:
        if tag in self.releases:
            return self.releases[tag]
        payload = {
            "tag_name": tag,
            "name": tag,
            "body": (
                f"LAIN server experiment `{run_name}`. Checkpoints and a "
                "training-record archive are uploaded without deleting the "
                "server originals."
            ),
            "draft": False,
            "prerelease": False,
        }
        release = self.api(
            "POST", f"/repos/{OWNER}/{REPO}/releases", payload
        )
        assert isinstance(release, dict)
        release["assets_full"] = []
        self.releases[tag] = release
        return release

    def upload(self, release: dict, path: Path, requested_name: str) -> dict:
        size = path.stat().st_size
        if size > MAX_ASSET_BYTES:
            raise RuntimeError(
                f"Asset exceeds GitHub's 2GB limit: {path} ({size} bytes)"
            )
        digest = sha256_file(path)
        for asset in release.get("assets_full", []):
            remote_digest = (asset.get("digest") or "").removeprefix("sha256:")
            if (
                asset.get("name") == requested_name
                and int(asset.get("size", -1)) == size
                and (not remote_digest or remote_digest == digest)
            ):
                return {"state": "already_in_target_release", "asset": asset}

        used_names = {asset["name"] for asset in release.get("assets_full", [])}
        upload_name = requested_name
        if upload_name in used_names:
            stem, suffix = os.path.splitext(requested_name)
            upload_name = f"{stem}__sha256_{digest[:12]}{suffix}"

        query_name = urllib.parse.quote(upload_name, safe="")
        url = (
            f"https://uploads.github.com/repos/{OWNER}/{REPO}/releases/"
            f"{release['id']}/assets?name={query_name}"
        )
        with tempfile.NamedTemporaryFile(
            prefix="lain-gh-upload-", suffix=".json", delete=False
        ) as handle:
            response_path = Path(handle.name)
        config = (
            f'header = "Authorization: Bearer {self.token}"\n'
            'header = "Accept: application/vnd.github+json"\n'
            'header = "X-GitHub-Api-Version: 2022-11-28"\n'
            'header = "Content-Type: application/octet-stream"\n'
            'request = "POST"\n'
        )
        try:
            command = [
                "curl", "--silent", "--show-error", "--fail-with-body",
                "--retry", "8", "--retry-all-errors",
                "--connect-timeout", "30", "--max-time", "0",
                "--config", "-", "--output", str(response_path),
                "--data-binary", "@" + str(path), url,
            ]
            process = subprocess.run(
                command,
                input=config.encode("utf-8"),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=False,
            )
            response_raw = response_path.read_bytes()
            if process.returncode:
                raise RuntimeError(
                    f"Upload failed for {upload_name}: "
                    f"{process.stderr.decode('utf-8', 'replace')[-1500:]} "
                    f"{response_raw.decode('utf-8', 'replace')[-1500:]}"
                )
            asset = json.loads(response_raw.decode("utf-8"))
        finally:
            response_path.unlink(missing_ok=True)

        remote_digest = (asset.get("digest") or "").removeprefix("sha256:")
        if int(asset.get("size", -1)) != size:
            raise RuntimeError(f"Uploaded size mismatch for {upload_name}")
        if remote_digest and remote_digest != digest:
            raise RuntimeError(f"Uploaded digest mismatch for {upload_name}")

        release.setdefault("assets_full", []).append(asset)
        self.digest_locations.setdefault(digest, []).append({
            "tag": release["tag_name"],
            "name": asset["name"],
            "size": size,
            "url": asset.get("browser_download_url"),
        })
        return {"state": "uploaded", "asset": asset}


hash_cache = load_json(HASH_CACHE_PATH, {})
if not isinstance(hash_cache, dict):
    hash_cache = {}


def sha256_file(path: Path) -> str:
    stat = path.stat()
    key = str(path)
    cached = hash_cache.get(key)
    if (
        isinstance(cached, dict)
        and cached.get("size") == stat.st_size
        and cached.get("mtime_ns") == stat.st_mtime_ns
        and isinstance(cached.get("sha256"), str)
    ):
        return cached["sha256"]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    result = digest.hexdigest()
    hash_cache[key] = {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": result,
    }
    atomic_json(HASH_CACHE_PATH, hash_cache)
    return result


def pt_asset_name(run_dir: Path, path: Path) -> str:
    relative = path.relative_to(run_dir)
    if relative.parent == Path("checkpoints"):
        return relative.name
    return "__".join(relative.parts)


def iter_record_files(run_dir: Path):
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(run_dir)
        if ".git" in relative.parts:
            continue
        if path.suffix.lower() == ".pt":
            continue
        yield path


def build_records_archive(run_dir: Path) -> Path:
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    destination = WORK_ROOT / f"{run_dir.name}__training_records.tar.gz"
    destination.unlink(missing_ok=True)
    with destination.open("wb") as output:
        tar = subprocess.Popen(
            [
                "tar", "--sort=name", "--mtime=@0", "--owner=0", "--group=0",
                "--numeric-owner", "--exclude=.git", "--exclude=*.pt",
                "-cf", "-", "-C", str(run_dir.parent), run_dir.name,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert tar.stdout is not None
        gzip_process = subprocess.Popen(
            ["gzip", "-n", "-6"],
            stdin=tar.stdout,
            stdout=output,
            stderr=subprocess.PIPE,
        )
        tar.stdout.close()
        gzip_stderr = gzip_process.communicate()[1]
        tar_stderr = tar.communicate()[1]
        if tar.returncode or gzip_process.returncode:
            destination.unlink(missing_ok=True)
            raise RuntimeError(
                f"Archive failed for {run_dir.name}: "
                f"{tar_stderr.decode('utf-8', 'replace')[-1000:]} "
                f"{gzip_stderr.decode('utf-8', 'replace')[-1000:]}"
            )
    if destination.stat().st_size > MAX_ASSET_BYTES:
        raise RuntimeError(
            f"Training-record archive exceeds 2GB: {destination}"
        )
    return destination


def build_manifest(github: GitHub) -> dict:
    runs = []
    total_local_pt_bytes = 0
    remaining_pt_bytes = 0
    for run_dir in sorted(
        (path for path in RUN_ROOT.iterdir() if path.is_dir()),
        key=lambda path: path.name.lower(),
    ):
        tag = LEGACY_RELEASE_TAGS.get(run_dir.name, run_dir.name)
        checkpoints = []
        for path in sorted(run_dir.rglob("*.pt")):
            if ".git" in path.relative_to(run_dir).parts:
                continue
            size = path.stat().st_size
            digest = sha256_file(path)
            locations = github.digest_locations.get(digest, [])
            state = "already_uploaded" if locations else "pending_upload"
            total_local_pt_bytes += size
            if not locations:
                remaining_pt_bytes += size
            checkpoints.append({
                "path": str(path),
                "relative_path": str(path.relative_to(run_dir)),
                "asset_name": pt_asset_name(run_dir, path),
                "size": size,
                "sha256": digest,
                "state": state,
                "remote_locations": locations,
            })
        record_files = list(iter_record_files(run_dir))
        runs.append({
            "run_name": run_dir.name,
            "run_directory": str(run_dir),
            "release_tag": tag,
            "release_url": (
                github.releases.get(tag, {}).get("html_url")
                if tag in github.releases else None
            ),
            "checkpoints": checkpoints,
            "record_file_count": len(record_files),
            "record_source_bytes": sum(path.stat().st_size for path in record_files),
            "records_asset_name": f"{run_dir.name}__training_records.tar.gz",
        })
    return {
        "generated_at": now(),
        "repository": f"https://github.com/{OWNER}/{REPO}",
        "source_root": str(RUN_ROOT),
        "server_sources_are_never_deleted": True,
        "run_count": len(runs),
        "total_local_checkpoint_bytes": total_local_pt_bytes,
        "remaining_checkpoint_upload_bytes": remaining_pt_bytes,
        "runs": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    arguments = parser.parse_args()

    token = load_token(required=not arguments.prepare_only)
    github = GitHub(token)
    github.refresh()
    manifest = build_manifest(github)
    atomic_json(MANIFEST_PATH, manifest)
    print(
        "MANIFEST",
        f"runs={manifest['run_count']}",
        f"remaining_pt_bytes={manifest['remaining_checkpoint_upload_bytes']}",
        flush=True,
    )
    if arguments.prepare_only:
        return 0

    status = {
        "state": "running",
        "started_at": now(),
        "manifest": str(MANIFEST_PATH),
        "server_sources_are_never_deleted": True,
        "runs": [],
    }
    atomic_json(STATUS_PATH, status)
    try:
        for run_index, run in enumerate(manifest["runs"], 1):
            run_entry = {
                "run_name": run["run_name"],
                "release_tag": run["release_tag"],
                "state": "processing",
                "started_at": now(),
                "checkpoints": [],
            }
            status["runs"].append(run_entry)
            status["current_run"] = run["run_name"]
            status["current_run_index"] = run_index
            status["run_count"] = len(manifest["runs"])
            atomic_json(STATUS_PATH, status)

            release = github.get_or_create_release(
                run["release_tag"], run["run_name"]
            )
            run_entry["release_url"] = release["html_url"]
            print(
                f"RUN {run_index}/{len(manifest['runs'])} {run['run_name']}",
                flush=True,
            )

            for checkpoint in run["checkpoints"]:
                digest = checkpoint["sha256"]
                locations = github.digest_locations.get(digest, [])
                if locations:
                    result = {
                        "path": checkpoint["path"],
                        "asset_name": checkpoint["asset_name"],
                        "size": checkpoint["size"],
                        "sha256": digest,
                        "state": "already_uploaded_by_digest",
                        "remote_locations": locations,
                    }
                    print(
                        f"SKIP_PT {run['run_name']} {checkpoint['relative_path']}",
                        flush=True,
                    )
                else:
                    print(
                        f"UPLOAD_PT {run['run_name']} {checkpoint['relative_path']} "
                        f"bytes={checkpoint['size']}",
                        flush=True,
                    )
                    uploaded = github.upload(
                        release,
                        Path(checkpoint["path"]),
                        checkpoint["asset_name"],
                    )
                    asset = uploaded["asset"]
                    result = {
                        "path": checkpoint["path"],
                        "asset_name": asset["name"],
                        "size": checkpoint["size"],
                        "sha256": digest,
                        "state": uploaded["state"],
                        "url": asset.get("browser_download_url"),
                    }
                run_entry["checkpoints"].append(result)
                atomic_json(STATUS_PATH, status)

            archive = build_records_archive(Path(run["run_directory"]))
            try:
                archive_digest = sha256_file(archive)
                archive_locations = github.digest_locations.get(archive_digest, [])
                if archive_locations:
                    run_entry["training_records"] = {
                        "state": "already_uploaded_by_digest",
                        "asset_name": archive.name,
                        "size": archive.stat().st_size,
                        "sha256": archive_digest,
                        "remote_locations": archive_locations,
                    }
                    print(f"SKIP_RECORDS {run['run_name']}", flush=True)
                else:
                    print(
                        f"UPLOAD_RECORDS {run['run_name']} bytes={archive.stat().st_size}",
                        flush=True,
                    )
                    uploaded = github.upload(release, archive, archive.name)
                    asset = uploaded["asset"]
                    run_entry["training_records"] = {
                        "state": uploaded["state"],
                        "asset_name": asset["name"],
                        "size": archive.stat().st_size,
                        "sha256": archive_digest,
                        "url": asset.get("browser_download_url"),
                    }
            finally:
                # Generated archive only; original server training files remain.
                archive.unlink(missing_ok=True)

            run_entry["state"] = "completed"
            run_entry["completed_at"] = now()
            atomic_json(STATUS_PATH, status)
            print(f"DONE_RUN {run['run_name']}", flush=True)

        status["state"] = "completed"
        status["completed_at"] = now()
        status.pop("current_run", None)
        atomic_json(STATUS_PATH, status)
        print("ALL_COMPLETED", flush=True)
        return 0
    except Exception as exc:
        status["state"] = "failed"
        status["failed_at"] = now()
        status["error"] = str(exc)
        atomic_json(STATUS_PATH, status)
        print(f"FAILED {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
