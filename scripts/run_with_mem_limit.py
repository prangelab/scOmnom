#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import platform
import re
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
        "--compressed-limit-gb",
        type=float,
        default=None,
        help="On macOS, terminate when system compressed memory exceeds this many GB.",
    )
    parser.add_argument(
        "--swap-used-limit-gb",
        type=float,
        default=None,
        help="On macOS, terminate when system swap used exceeds this many GB.",
    )
    parser.add_argument(
        "--compressed-delta-limit-gb",
        type=float,
        default=None,
        help="On macOS, terminate when system compressed memory rises by more than this many GB above baseline.",
    )
    parser.add_argument(
        "--swap-used-delta-limit-gb",
        type=float,
        default=None,
        help="On macOS, terminate when system swap used rises by more than this many GB above baseline.",
    )
    parser.add_argument(
        "--pressure-consecutive-breaches",
        type=int,
        default=2,
        help="On macOS, require this many consecutive pressure breaches before terminating. Default: 2.",
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
    if args.compressed_limit_gb is not None and args.compressed_limit_gb <= 0:
        parser.error("--compressed-limit-gb must be > 0")
    if args.swap_used_limit_gb is not None and args.swap_used_limit_gb <= 0:
        parser.error("--swap-used-limit-gb must be > 0")
    if args.compressed_delta_limit_gb is not None and args.compressed_delta_limit_gb <= 0:
        parser.error("--compressed-delta-limit-gb must be > 0")
    if args.swap_used_delta_limit_gb is not None and args.swap_used_delta_limit_gb <= 0:
        parser.error("--swap-used-delta-limit-gb must be > 0")
    if args.pressure_consecutive_breaches <= 0:
        parser.error("--pressure-consecutive-breaches must be > 0")
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


def _fmt_bytes_as_gb(value_bytes: int) -> str:
    return f"{value_bytes / (1024 ** 3):.2f} GB"


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _terminate_process_group(pid: int, grace_seconds: float) -> None:
    try:
        os.killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return

    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        try:
            os.killpg(pid, 0)
        except (ProcessLookupError, PermissionError):
            return
        time.sleep(0.2)

    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        return


def _parse_vm_stat_bytes() -> dict[str, int]:
    proc = subprocess.run(
        ["vm_stat"],
        check=True,
        capture_output=True,
        text=True,
    )
    page_size = 4096
    lines = proc.stdout.splitlines()
    header = lines[0] if lines else ""
    m = re.search(r"page size of (\d+) bytes", header)
    if m:
        page_size = int(m.group(1))

    stats: dict[str, int] = {}
    for line in lines[1:]:
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        value_txt = raw_value.strip().rstrip(".").replace(".", "")
        value_txt = value_txt.replace(",", "")
        if not value_txt.isdigit():
            continue
        stats[key.strip()] = int(value_txt)

    out: dict[str, int] = {}
    if "Pages occupied by compressor" in stats:
        out["compressed_bytes"] = stats["Pages occupied by compressor"] * page_size
    if "Pages free" in stats:
        out["free_bytes"] = stats["Pages free"] * page_size
    return out


def _parse_swapusage_bytes() -> dict[str, int]:
    proc = subprocess.run(
        ["sysctl", "vm.swapusage"],
        check=True,
        capture_output=True,
        text=True,
    )
    text = proc.stdout.strip()
    out: dict[str, int] = {}
    for key in ("total", "used", "free"):
        m = re.search(rf"{key}\s*=\s*([0-9.]+)([MG])", text)
        if not m:
            continue
        value = float(m.group(1))
        unit = m.group(2)
        multiplier = 1024 ** 2 if unit == "M" else 1024 ** 3
        out[f"swap_{key}_bytes"] = int(value * multiplier)
    return out


def _sample_macos_pressure() -> dict[str, int]:
    sample: dict[str, int] = {}
    sample.update(_parse_vm_stat_bytes())
    sample.update(_parse_swapusage_bytes())
    return sample


def _macos_pressure_breach_reason(
    sample: dict[str, int],
    *,
    baseline: dict[str, int],
    compressed_limit_bytes: int | None,
    swap_limit_bytes: int | None,
    compressed_delta_limit_bytes: int | None,
    swap_delta_limit_bytes: int | None,
) -> str | None:
    compressed_bytes = sample.get("compressed_bytes", 0)
    swap_used_bytes = sample.get("swap_used_bytes", 0)
    compressed_delta = compressed_bytes - baseline.get("compressed_bytes", compressed_bytes)
    swap_delta = swap_used_bytes - baseline.get("swap_used_bytes", swap_used_bytes)

    compressed_abs_breach = (
        compressed_limit_bytes is not None and compressed_bytes > compressed_limit_bytes
    )
    compressed_delta_breach = (
        compressed_delta_limit_bytes is not None and compressed_delta > compressed_delta_limit_bytes
    )
    swap_abs_breach = swap_limit_bytes is not None and swap_used_bytes > swap_limit_bytes
    swap_delta_breach = swap_delta_limit_bytes is not None and swap_delta > swap_delta_limit_bytes

    if swap_abs_breach and swap_delta_breach:
        return (
            "swap used exceeded absolute and delta limits: "
            f"{_fmt_bytes_as_gb(swap_used_bytes)} (delta {_fmt_bytes_as_gb(swap_delta)})"
        )
    if swap_abs_breach:
        return f"swap used exceeded limit: {_fmt_bytes_as_gb(swap_used_bytes)}"
    if swap_delta_breach:
        return f"swap used delta exceeded limit: {_fmt_bytes_as_gb(swap_delta)}"

    if compressed_abs_breach and compressed_delta_breach and swap_used_bytes > 0:
        return (
            "compressed memory exceeded absolute and delta limits with swap in use: "
            f"{_fmt_bytes_as_gb(compressed_bytes)} (delta {_fmt_bytes_as_gb(compressed_delta)}), "
            f"swap_used={_fmt_bytes_as_gb(swap_used_bytes)}"
        )
    if compressed_abs_breach and swap_used_bytes > 0:
        return (
            "compressed memory exceeded limit with swap in use: "
            f"{_fmt_bytes_as_gb(compressed_bytes)}, swap_used={_fmt_bytes_as_gb(swap_used_bytes)}"
        )
    if compressed_delta_breach and swap_used_bytes > 0:
        return (
            "compressed memory delta exceeded limit with swap in use: "
            f"{_fmt_bytes_as_gb(compressed_delta)}, swap_used={_fmt_bytes_as_gb(swap_used_bytes)}"
        )

    return None


def main() -> int:
    args = _parse_args()
    limit_kb = int(args.rss_limit_gb * 1024 * 1024)
    is_macos = platform.system() == "Darwin"
    compressed_limit_bytes = (
        int(args.compressed_limit_gb * (1024 ** 3)) if args.compressed_limit_gb is not None else None
    )
    swap_limit_bytes = (
        int(args.swap_used_limit_gb * (1024 ** 3)) if args.swap_used_limit_gb is not None else None
    )
    compressed_delta_limit_bytes = (
        int(args.compressed_delta_limit_gb * (1024 ** 3))
        if args.compressed_delta_limit_gb is not None
        else None
    )
    swap_delta_limit_bytes = (
        int(args.swap_used_delta_limit_gb * (1024 ** 3))
        if args.swap_used_delta_limit_gb is not None
        else None
    )

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
    peak_compressed_bytes = 0
    peak_swap_used_bytes = 0
    pressure_baseline: dict[str, int] = {}
    pressure_breach_count = 0

    if is_macos:
        try:
            pressure_baseline = _sample_macos_pressure()
        except subprocess.CalledProcessError as exc:
            print(
                f"[{_timestamp()}] warning: failed to sample macOS pressure baseline: {exc}",
                file=sys.stderr,
                flush=True,
            )
            pressure_baseline = {}
        else:
            baseline_parts = [f"[{_timestamp()}] baseline"]
            baseline_parts.append(
                f"compressed={_fmt_bytes_as_gb(pressure_baseline.get('compressed_bytes', 0))}"
            )
            baseline_parts.append(
                f"swap_used={_fmt_bytes_as_gb(pressure_baseline.get('swap_used_bytes', 0))}"
            )
            print(" ".join(baseline_parts), file=sys.stderr, flush=True)

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
            log_parts = [f"[{_timestamp()}] rss={_fmt_gb(rss_kb)} pids={','.join(map(str, pids))}"]
            macos_sample: dict[str, int] = {}
            if is_macos:
                try:
                    macos_sample = _sample_macos_pressure()
                except subprocess.CalledProcessError as exc:
                    print(
                        f"[{_timestamp()}] warning: failed to sample macOS pressure stats: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                else:
                    compressed_bytes = macos_sample.get("compressed_bytes", 0)
                    swap_used_bytes = macos_sample.get("swap_used_bytes", 0)
                    compressed_delta = compressed_bytes - pressure_baseline.get(
                        "compressed_bytes", compressed_bytes
                    )
                    swap_delta = swap_used_bytes - pressure_baseline.get(
                        "swap_used_bytes", swap_used_bytes
                    )
                    peak_compressed_bytes = max(peak_compressed_bytes, compressed_bytes)
                    peak_swap_used_bytes = max(peak_swap_used_bytes, swap_used_bytes)
                    log_parts.append(f"compressed={_fmt_bytes_as_gb(compressed_bytes)}")
                    log_parts.append(f"swap_used={_fmt_bytes_as_gb(swap_used_bytes)}")
                    log_parts.append(f"compressed_delta={_fmt_bytes_as_gb(max(0, compressed_delta))}")
                    log_parts.append(f"swap_delta={_fmt_bytes_as_gb(max(0, swap_delta))}")
            print(" ".join(log_parts), file=sys.stderr, flush=True)

            if exit_code is not None:
                peak_parts = [f"[{_timestamp()}] peak_rss={_fmt_gb(peak_rss_kb)}"]
                if is_macos:
                    peak_parts.append(f"peak_compressed={_fmt_bytes_as_gb(peak_compressed_bytes)}")
                    peak_parts.append(f"peak_swap_used={_fmt_bytes_as_gb(peak_swap_used_bytes)}")
                peak_parts.append(f"exit_code={exit_code}")
                print(
                    " ".join(peak_parts),
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
            if is_macos and (
                compressed_limit_bytes is not None
                or swap_limit_bytes is not None
                or compressed_delta_limit_bytes is not None
                or swap_delta_limit_bytes is not None
            ):
                reason = _macos_pressure_breach_reason(
                    macos_sample,
                    baseline=pressure_baseline,
                    compressed_limit_bytes=compressed_limit_bytes,
                    swap_limit_bytes=swap_limit_bytes,
                    compressed_delta_limit_bytes=compressed_delta_limit_bytes,
                    swap_delta_limit_bytes=swap_delta_limit_bytes,
                )
                if reason is not None:
                    pressure_breach_count += 1
                    print(
                        f"[{_timestamp()}] pressure breach {pressure_breach_count}/{args.pressure_consecutive_breaches}: {reason}",
                        file=sys.stderr,
                        flush=True,
                    )
                    if pressure_breach_count >= args.pressure_consecutive_breaches:
                        limit_exceeded = True
                        print(
                            f"[{_timestamp()}] macOS pressure guard triggered after {pressure_breach_count} consecutive breaches; terminating",
                            file=sys.stderr,
                            flush=True,
                        )
                        _terminate_process_group(child.pid, args.grace_seconds)
                        break
                else:
                    pressure_breach_count = 0

            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        peak_parts = [f"[{_timestamp()}] interrupted; peak_rss={_fmt_gb(peak_rss_kb)}"]
        if is_macos:
            peak_parts.append(f"peak_compressed={_fmt_bytes_as_gb(peak_compressed_bytes)}")
            peak_parts.append(f"peak_swap_used={_fmt_bytes_as_gb(peak_swap_used_bytes)}")
        peak_parts.append("terminating child process group")
        print(
            " ".join(peak_parts),
            file=sys.stderr,
            flush=True,
        )
        _terminate_process_group(child.pid, args.grace_seconds)
        return 130

    try:
        exit_code = child.wait(timeout=max(args.grace_seconds, 1.0))
        peak_parts = [f"[{_timestamp()}] peak_rss={_fmt_gb(peak_rss_kb)}"]
        if is_macos:
            peak_parts.append(f"peak_compressed={_fmt_bytes_as_gb(peak_compressed_bytes)}")
            peak_parts.append(f"peak_swap_used={_fmt_bytes_as_gb(peak_swap_used_bytes)}")
        peak_parts.append(f"exit_code={exit_code}")
        print(
            " ".join(peak_parts),
            file=sys.stderr,
            flush=True,
        )
        return exit_code
    except subprocess.TimeoutExpired:
        peak_parts = [f"[{_timestamp()}] peak_rss={_fmt_gb(peak_rss_kb)}"]
        if is_macos:
            peak_parts.append(f"peak_compressed={_fmt_bytes_as_gb(peak_compressed_bytes)}")
            peak_parts.append(f"peak_swap_used={_fmt_bytes_as_gb(peak_swap_used_bytes)}")
        peak_parts.append(f"exit_code={'137' if limit_exceeded else '1'}")
        print(
            " ".join(peak_parts),
            file=sys.stderr,
            flush=True,
        )
        return 137 if limit_exceeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
