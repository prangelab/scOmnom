#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import datetime


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a command and kill it if the combined RSS of the full process tree "
            "exceeds a user-defined threshold."
        )
    )
    parser.add_argument(
        "--rss-limit-gb",
        type=float,
        required=True,
        help="Kill the command when combined RSS across the process tree exceeds this many GB.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=5.0,
        help="Polling interval in seconds. Default: 5.",
    )
    parser.add_argument(
        "--grace-seconds",
        type=float,
        default=10.0,
        help="Seconds to wait after SIGTERM before sending SIGKILL. Default: 10.",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to run. Prefix with -- to separate wrapper flags from the command.",
    )
    args = parser.parse_args()
    if args.rss_limit_gb <= 0:
        parser.error("--rss-limit-gb must be > 0")
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be > 0")
    if args.grace_seconds < 0:
        parser.error("--grace-seconds must be >= 0")
    if not args.command:
        parser.error("missing command; use -- <command> [args...]")
    if args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("missing command after --")
    return args


def _ps_snapshot() -> dict[int, tuple[int, int]]:
    proc = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,rss="],
        check=True,
        capture_output=True,
        text=True,
    )
    snapshot: dict[int, tuple[int, int]] = {}
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        pid, ppid, rss_kb = map(int, parts)
        snapshot[pid] = (ppid, rss_kb)
    return snapshot


def _tree_rss_kb(root_pid: int) -> tuple[int, list[int]]:
    snapshot = _ps_snapshot()
    children: dict[int, list[int]] = {}
    for pid, (ppid, _rss_kb) in snapshot.items():
        children.setdefault(ppid, []).append(pid)

    stack = [root_pid]
    seen: set[int] = set()
    rss_kb = 0
    ordered_pids: list[int] = []

    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        ordered_pids.append(pid)
        if pid in snapshot:
            _ppid, pid_rss_kb = snapshot[pid]
            rss_kb += pid_rss_kb
        stack.extend(children.get(pid, ()))

    return rss_kb, ordered_pids


def _fmt_gb(rss_kb: int) -> str:
    return f"{rss_kb / (1024 * 1024):.2f} GB"


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _terminate_process_group(pid: int, grace_seconds: float) -> None:
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        try:
            os.killpg(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.2)

    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def main() -> int:
    args = _parse_args()
    limit_kb = int(args.rss_limit_gb * 1024 * 1024)

    child = subprocess.Popen(
        args.command,
        start_new_session=True,
    )

    print(
        f"[{_timestamp()}] started pid={child.pid} limit={args.rss_limit_gb:.2f} GB "
        f"command={' '.join(args.command)}",
        file=sys.stderr,
        flush=True,
    )

    limit_exceeded = False
    peak_rss_kb = 0

    try:
        while True:
            exit_code = child.poll()
            try:
                rss_kb, pids = _tree_rss_kb(child.pid)
            except subprocess.CalledProcessError as exc:
                print(
                    f"[{_timestamp()}] warning: failed to sample memory via ps: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                rss_kb, pids = 0, [child.pid]

            peak_rss_kb = max(peak_rss_kb, rss_kb)
            print(
                f"[{_timestamp()}] rss={_fmt_gb(rss_kb)} pids={','.join(map(str, pids))}",
                file=sys.stderr,
                flush=True,
            )

            if exit_code is not None:
                print(
                    f"[{_timestamp()}] peak_rss={_fmt_gb(peak_rss_kb)} exit_code={exit_code}",
                    file=sys.stderr,
                    flush=True,
                )
                return exit_code

            if rss_kb > limit_kb:
                limit_exceeded = True
                print(
                    f"[{_timestamp()}] combined RSS exceeded limit: "
                    f"{_fmt_gb(rss_kb)} > {args.rss_limit_gb:.2f} GB; terminating",
                    file=sys.stderr,
                    flush=True,
                )
                _terminate_process_group(child.pid, args.grace_seconds)
                break

            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        print(
            f"[{_timestamp()}] interrupted; peak_rss={_fmt_gb(peak_rss_kb)}; terminating child process group",
            file=sys.stderr,
            flush=True,
        )
        _terminate_process_group(child.pid, args.grace_seconds)
        return 130

    try:
        exit_code = child.wait(timeout=max(args.grace_seconds, 1.0))
        print(
            f"[{_timestamp()}] peak_rss={_fmt_gb(peak_rss_kb)} exit_code={exit_code}",
            file=sys.stderr,
            flush=True,
        )
        return exit_code
    except subprocess.TimeoutExpired:
        print(
            f"[{_timestamp()}] peak_rss={_fmt_gb(peak_rss_kb)} exit_code={'137' if limit_exceeded else '1'}",
            file=sys.stderr,
            flush=True,
        )
        return 137 if limit_exceeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
