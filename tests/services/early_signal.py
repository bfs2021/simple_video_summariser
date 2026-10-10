"""Helpers for the early-signal tests of the three long-lived entrypoints (issue #108).

Not a test module. Each entrypoint test file calls ``run_window`` and
``race_once`` with its own module name.

The child never needs Postgres: ``DATABASE_URL`` points at a closed local
port. Every wait has a timeout and a child that overruns is killed before
the test fails.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CLOSED_PORT_DSN = "postgresql://u:p@127.0.0.1:1/x"
WAIT_SEC = 20.0

# The child: a meta-path finder that, the first time ``trigger`` is imported,
# records the SIGTERM disposition, sends the signal to itself and then declines
# the import, so normal import carries on. Then it runs the entrypoint as
# ``__main__``. Exit 98 means the trigger never fired (already imported), so
# the test fails instead of letting the service run on.
BOOTSTRAP = r"""
import importlib.abc, os, runpy, signal, sys

module, trigger, signame = sys.argv[1], sys.argv[2], sys.argv[3]
sys.argv = [module] + sys.argv[4:]
fired = []

class Trigger(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == trigger and not fired:
            fired.append(True)
            handler = signal.getsignal(signal.SIGTERM)
            print("SIGTERM-HANDLER-AT-IMPORT:" + ("DEFAULT" if handler == signal.SIG_DFL else "SET"),
                  flush=True)
            os.kill(os.getpid(), getattr(signal, signame))
        return None

sys.meta_path.insert(0, Trigger())
try:
    runpy.run_module(module, run_name="__main__", alter_sys=True)
finally:
    if not fired:
        print("TRIGGER-NEVER-FIRED", flush=True)
        os._exit(98)
"""


def clean_env(**values: str) -> dict[str, str]:
    """A from-scratch environment: nothing of the developer's leaks into the child."""
    return {"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1", **values}


def service_env(tmp_dir: Path) -> dict[str, str]:
    return clean_env(DATABASE_URL=CLOSED_PORT_DSN, AUDIO_DIR=str(tmp_dir / "audio"))


def _finish(proc: subprocess.Popen[str]) -> tuple[int, str]:
    try:
        out, _ = proc.communicate(timeout=WAIT_SEC)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, _ = proc.communicate()
        raise AssertionError(f"child did not exit within {WAIT_SEC}s:\n{out}") from None
    return proc.returncode, out


def run_window(
    module: str, trigger: str, tmp_dir: Path, *, sig: str = "SIGTERM", args: tuple[str, ...] = ()
) -> tuple[int, str]:
    """Run ``module`` as __main__ and deliver ``sig`` when ``trigger`` is first imported."""
    proc = subprocess.Popen(
        [sys.executable, "-c", BOOTSTRAP, module, trigger, sig, *args],
        cwd=REPO_ROOT,
        env=service_env(tmp_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return _finish(proc)


_SIGTERM_BIT = 1 << (signal.SIGTERM - 1)


def _sigcgt(pid: int) -> int | None:
    try:
        with open(f"/proc/{pid}/status") as status:
            for line in status:
                if line.startswith("SigCgt:"):
                    return int(line.split()[1], 16)
    except OSError:
        return None
    return None


def race_once(module: str, tmp_dir: Path) -> tuple[int, str]:
    """Start the service and SIGTERM it the instant its SIGTERM handler exists."""
    proc = subprocess.Popen(
        [sys.executable, "-m", module],
        cwd=REPO_ROOT,
        env=service_env(tmp_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        # Busy-poll (no sleep) until the child has a Python SIGTERM handler.
        while proc.poll() is None:
            mask = _sigcgt(proc.pid)
            if mask is not None and mask & _SIGTERM_BIT:
                proc.send_signal(signal.SIGTERM)
                break
    except BaseException:
        proc.kill()
        raise
    return _finish(proc)


def proc_status_available() -> bool:
    return os.path.exists(f"/proc/{os.getpid()}/status")


def scratch() -> Path:
    return Path(tempfile.mkdtemp(prefix="i108-"))
