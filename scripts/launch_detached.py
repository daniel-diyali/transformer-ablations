"""Start a long-running command that outlives the shell that launched it.

A 27-run sweep is hours of work, and it kept dying partway through. The cause
was not the sweep: `nohup cmd &` leaves the child in the launching shell's
process group, so when that session is torn down the child is reaped with it.
macOS has no `setsid(1)` to escape that, which is the usual fix on Linux.

This calls `setsid` via `start_new_session=True`, making the child its own
session leader. The launcher exits immediately and the child reparents to
pid 1, where nothing but an explicit kill or a reboot will stop it.

    python scripts/launch_detached.py runs/sweep.log -- \
        .venv/bin/python -u -m minigpt.experiments run --thermal cool
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def launch(log_path: Path, command: list[str]) -> int:
    """Start `command` detached, appending output to `log_path`. Returns its pid."""
    if not command:
        raise ValueError("no command given")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Line-buffered append so a reader can follow progress live, and so an
    # abrupt kill still leaves every line written up to that moment.
    log = log_path.open("a", buffering=1)
    try:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    finally:
        log.close()
    return process.pid


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="file to append stdout and stderr to")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- then the command")
    args = parser.parse_args(argv)

    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("give the command after --")

    pid = launch(args.log, command)
    print(f"launched pid {pid}, logging to {args.log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
