"""Fail-closed reads of shared lists on the Modal volume (held-out lists, merged judge rejects).

A shared list is read before it is rewritten. `modal volume ls` says whether the file exists; when it exists, a failed
`modal volume get` is an error (never "no list yet"), so a transient failure can never make a writer start from an
empty list and force-put it over the real one."""
import re
import subprocess
from pathlib import Path

PIPE = Path(__file__).resolve().parents[2] / "pipeline"
VOLUME = "parakeet-spoken-form"


class VolumeReadError(RuntimeError):
    pass


def vlist(remote_dir: str) -> set:
    """Base names of the entries of a volume directory; raises VolumeReadError when the listing fails."""
    r = subprocess.run(["modal", "volume", "ls", VOLUME, remote_dir], cwd=PIPE, capture_output=True, text=True)
    if r.returncode != 0:
        raise VolumeReadError(f"modal volume ls {remote_dir} failed: {(r.stderr or r.stdout).strip()[:300]}")
    return set(re.findall(r"[\w.\-]+\.(?:jsonl|json|txt)\b", r.stdout))


def vget(remote: str, local) -> bool:
    """`modal volume get --force`; True when it succeeded."""
    return subprocess.run(["modal", "volume", "get", "--force", VOLUME, remote, str(local)], cwd=PIPE,
                          capture_output=True).returncode == 0


def read_shared(remote: str, local) -> bool:
    """Fetch a shared list to `local`. False when the volume has no such file (the caller may start a first list);
    True when fetched. Raises VolumeReadError when the listing fails or the file exists but cannot be fetched."""
    d, _, name = remote.rpartition("/")
    if name not in vlist(d):
        return False
    Path(local).unlink(missing_ok=True)  # a stale local copy must not stand in for a failed fetch
    if not vget(remote, local) or not Path(local).exists():
        raise VolumeReadError(f"{remote} exists on the volume but could not be fetched: refusing to rewrite it")
    return True


def put(local, remote: str):
    subprocess.run(["modal", "volume", "put", "--force", VOLUME, str(local), remote], cwd=PIPE, check=True,
                   capture_output=True)
