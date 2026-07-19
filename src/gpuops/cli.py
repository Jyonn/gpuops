from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

from .actions import filter_processes, kill_processes, signal_for
from .alerts import evaluate_snapshot
from .formatting import gpu_rows, json_dump, process_rows, table
from .history import DEFAULT_HISTORY_PATH, read_snapshots, record_snapshot, since_days, summarize_user_usage
from .nvidia import NvidiaSmiError, collect_snapshot
from .process_info import enrich_processes
from .models import Snapshot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gpuops", description="GPUOps shared GPU administration CLI")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser("status", help="show real-time GPU status")
    status.add_argument("--json", action="store_true", help="print machine-readable JSON")
    status.add_argument("--sort", choices=["index", "free", "memory", "util", "temp"], default="index")
    status.add_argument("--record", action="store_true", help="append this snapshot to the history file")

    ps = subparsers.add_parser("ps", help="show GPU processes")
    ps.add_argument("--json", action="store_true", help="print machine-readable JSON")
    ps.add_argument("--gpu", type=int)
    ps.add_argument("--user")

    users = subparsers.add_parser("users", help="summarize current usage by user")
    users.add_argument("--json", action="store_true", help="print machine-readable JSON")
    users.add_argument("--sort", choices=["memory", "processes", "runtime"], default="memory")

    top = subparsers.add_parser("top", help="rank GPU processes")
    top.add_argument("--json", action="store_true", help="print machine-readable JSON")
    top.add_argument("--by", choices=["memory", "runtime"], default="memory")
    top.add_argument("--limit", type=int, default=20)

    doctor = subparsers.add_parser("doctor", help="show suspicious GPU states")
    doctor.add_argument("--json", action="store_true", help="print machine-readable JSON")
    doctor.add_argument("--idle-util", type=int, default=5)
    doctor.add_argument("--occupied-mb", type=int, default=1024)

    kill = subparsers.add_parser("kill", help="kill selected GPU processes; dry-run unless --yes is passed")
    kill.add_argument("--json", action="store_true", help="print machine-readable JSON")
    kill.add_argument("--gpu", type=int)
    kill.add_argument("--user")
    kill.add_argument("--force", action="store_true", help="use SIGKILL instead of SIGTERM")
    kill.add_argument("--yes", action="store_true", help="actually send the signal")

    history = subparsers.add_parser("history", help="summarize recorded GPU usage")
    history.add_argument("--json", action="store_true", help="print machine-readable JSON")
    history.add_argument("--days", type=int, default=30)
    history.add_argument("--path", default=str(DEFAULT_HISTORY_PATH))

    return parser


def _snapshot() -> Snapshot:
    snap = collect_snapshot()
    enrich_processes(snap.processes)
    return snap


def _sort_gpus(gpus, key: str):
    if key == "free":
        return sorted(gpus, key=lambda gpu: gpu.memory_free_mb, reverse=True)
    if key == "memory":
        return sorted(gpus, key=lambda gpu: gpu.memory_used_mb, reverse=True)
    if key == "util":
        return sorted(gpus, key=lambda gpu: gpu.utilization_gpu_percent, reverse=True)
    if key == "temp":
        return sorted(gpus, key=lambda gpu: gpu.temperature_c or -1, reverse=True)
    return sorted(gpus, key=lambda gpu: gpu.index)


def cmd_status(args) -> int:
    snap = _snapshot()
    if args.record:
        record_snapshot(snap)
    alerts = evaluate_snapshot(snap)
    gpus = _sort_gpus(snap.gpus, args.sort)
    if args.json:
        print(json_dump({"snapshot": snap.to_dict(), "alerts": [alert.__dict__ for alert in alerts]}))
        return 0
    print(table(["GPU", "Name", "Memory", "Free", "Util", "Temp", "Power", "Alerts"], gpu_rows(gpus, alerts)))
    return 0


def cmd_ps(args) -> int:
    snap = _snapshot()
    processes = filter_processes(snap.processes, gpu=args.gpu, user=args.user)
    processes.sort(key=lambda proc: (proc.gpu_index if proc.gpu_index is not None else 9999, -proc.used_memory_mb))
    if args.json:
        print(json_dump([proc.__dict__ for proc in processes]))
        return 0
    print(table(["GPU", "PID", "User", "Memory", "Running", "Command", "CWD"], process_rows(processes)))
    return 0


def cmd_users(args) -> int:
    snap = _snapshot()
    buckets = defaultdict(lambda: {"memory": 0, "processes": 0, "runtime": 0.0, "gpus": set()})
    for proc in snap.processes:
        user = proc.user or "unknown"
        buckets[user]["memory"] += proc.used_memory_mb
        buckets[user]["processes"] += 1
        buckets[user]["runtime"] += proc.running_seconds or 0.0
        if proc.gpu_index is not None:
            buckets[user]["gpus"].add(proc.gpu_index)
    summaries = []
    for user, values in buckets.items():
        summaries.append(
            {
                "user": user,
                "gpus": ",".join(str(gpu) for gpu in sorted(values["gpus"])) or "-",
                "processes": values["processes"],
                "memory_mb": values["memory"],
                "runtime_seconds": values["runtime"],
            }
        )
    if args.sort == "memory":
        summaries.sort(key=lambda item: item["memory_mb"], reverse=True)
    elif args.sort == "processes":
        summaries.sort(key=lambda item: item["processes"], reverse=True)
    else:
        summaries.sort(key=lambda item: item["runtime_seconds"], reverse=True)
    if args.json:
        print(json_dump(summaries))
        return 0
    rows = [
        [item["user"], item["gpus"], item["processes"], f"{item['memory_mb']} MB", _duration(item["runtime_seconds"])]
        for item in summaries
    ]
    print(table(["User", "GPUs", "Processes", "Memory", "Total Running"], rows))
    return 0


def _duration(seconds: float) -> str:
    from .formatting import human_duration

    return human_duration(seconds)


def cmd_top(args) -> int:
    snap = _snapshot()
    if args.by == "runtime":
        processes = sorted(snap.processes, key=lambda proc: proc.running_seconds or 0, reverse=True)
    else:
        processes = sorted(snap.processes, key=lambda proc: proc.used_memory_mb, reverse=True)
    processes = processes[: max(0, args.limit)]
    if args.json:
        print(json_dump([proc.__dict__ for proc in processes]))
        return 0
    print(table(["GPU", "PID", "User", "Memory", "Running", "Command", "CWD"], process_rows(processes)))
    return 0


def cmd_doctor(args) -> int:
    snap = _snapshot()
    alerts = evaluate_snapshot(snap, idle_util_percent=args.idle_util, occupied_memory_mb=args.occupied_mb)
    if args.json:
        print(json_dump([alert.__dict__ for alert in alerts]))
        return 0
    if not alerts:
        print("No suspicious GPU state found.")
        return 0
    rows = [[alert.level, alert.gpu_index, alert.message] for alert in alerts]
    print(table(["Level", "GPU", "Message"], rows))
    return 1 if any(alert.level == "crit" for alert in alerts) else 0


def cmd_kill(args) -> int:
    if args.gpu is None and args.user is None:
        print("Refusing to select every GPU process. Pass --gpu, --user, or both.", file=sys.stderr)
        return 2
    snap = _snapshot()
    selected = filter_processes(snap.processes, gpu=args.gpu, user=args.user)
    sig = signal_for(args.force)
    results = kill_processes(selected, sig=sig, dry_run=not args.yes)
    rows = [[r.gpu_index if r.gpu_index is not None else "-", r.pid, r.user or "-", r.signal_name, r.message] for r in results]
    if args.json:
        print(json_dump([result.__dict__ for result in results]))
        return 0
    if not rows:
        print("No matching GPU processes.")
        return 0
    print(table(["GPU", "PID", "User", "Signal", "Result"], rows))
    if not args.yes:
        print("\nDry-run only. Re-run with --yes to send the signal.")
    return 0


def cmd_history(args) -> int:
    samples = read_snapshots(path=Path(args.path).expanduser(), since=since_days(args.days))
    usage = summarize_user_usage(samples)
    if args.json:
        print(json_dump([item.__dict__ for item in usage]))
        return 0
    if not usage:
        print("No history samples found. Run `gpuops status --record` periodically first.")
        return 0
    rows = [[item.user, f"{item.gpu_hours:.2f}", f"{item.memory_gb_hours:.2f}", item.samples] for item in usage]
    print(table(["User", "GPU Hours", "GB Hours", "Samples"], rows))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return {
            "status": cmd_status,
            "ps": cmd_ps,
            "users": cmd_users,
            "top": cmd_top,
            "doctor": cmd_doctor,
            "kill": cmd_kill,
            "history": cmd_history,
        }[args.command](args)
    except NvidiaSmiError as exc:
        print(str(exc), file=sys.stderr)
        return 127
