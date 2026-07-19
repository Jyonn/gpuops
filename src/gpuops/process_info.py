from __future__ import annotations

import os
import pwd
import time
from typing import Iterable, Optional

from .models import GPUProcess


def _boot_time() -> Optional[float]:
    try:
        with open("/proc/stat", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("btime "):
                    return float(line.split()[1])
    except OSError:
        return None
    return None


def _clock_ticks() -> int:
    try:
        return os.sysconf(os.sysconf_names["SC_CLK_TCK"])
    except (KeyError, ValueError, OSError):
        return 100


def enrich_process(proc: GPUProcess, now: Optional[float] = None) -> GPUProcess:
    now = time.time() if now is None else now
    proc_dir = f"/proc/{proc.pid}"

    try:
        stat_info = os.stat(proc_dir)
        proc.user = pwd.getpwuid(stat_info.st_uid).pw_name
    except (OSError, KeyError):
        pass

    try:
        with open(f"{proc_dir}/cmdline", "rb") as handle:
            raw = handle.read().replace(b"\x00", b" ").strip()
        if raw:
            proc.command = raw.decode("utf-8", errors="replace")
    except OSError:
        pass

    try:
        proc.cwd = os.readlink(f"{proc_dir}/cwd")
    except OSError:
        pass

    boot = _boot_time()
    if boot is not None:
        try:
            with open(f"{proc_dir}/stat", "r", encoding="utf-8") as handle:
                stat = handle.read()
            parts = stat.rsplit(") ", 1)[1].split()
            start_ticks = int(parts[19])
            proc.started_at = boot + (start_ticks / _clock_ticks())
            proc.running_seconds = max(0.0, now - proc.started_at)
        except (OSError, ValueError, IndexError):
            pass

    if not proc.command:
        proc.command = proc.process_name
    return proc


def enrich_processes(processes: Iterable[GPUProcess]) -> None:
    now = time.time()
    for proc in processes:
        enrich_process(proc, now=now)

