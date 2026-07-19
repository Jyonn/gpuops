from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

from .actions import filter_processes, kill_processes, signal_for
from .alerts import evaluate_snapshot
from .formatting import json_dump
from .history import DEFAULT_HISTORY_PATH, read_snapshots, record_snapshot, since_days, summarize_user_usage
from .models import Snapshot
from .nvidia import NvidiaSmiError, collect_snapshot
from .process_info import enrich_processes
from .ui import (
    confirm,
    loading,
    print_alerts,
    print_empty,
    print_error,
    print_gpu_status,
    print_history,
    print_kill_results,
    print_processes,
    print_users,
    progress,
)


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
    kill.add_argument("--interactive", "-i", action="store_true", help="preview and ask before sending signals")

    history = subparsers.add_parser("history", help="summarize recorded GPU usage")
    history.add_argument("--json", action="store_true", help="print machine-readable JSON")
    history.add_argument("--days", type=int, default=30)
    history.add_argument("--path", default=str(DEFAULT_HISTORY_PATH))

    return parser


def _snapshot(show_loading: bool = True) -> Snapshot:
    with loading("Collecting GPU telemetry", enabled=show_loading):
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
    snap = _snapshot(show_loading=not args.json)
    if args.record:
        with loading("Recording history sample", enabled=not args.json):
            record_snapshot(snap)
    alerts = evaluate_snapshot(snap)
    gpus = _sort_gpus(snap.gpus, args.sort)
    if args.json:
        print(json_dump({"snapshot": snap.to_dict(), "alerts": [alert.__dict__ for alert in alerts]}))
        return 0
    print_gpu_status(gpus, alerts)
    return 0


def cmd_ps(args) -> int:
    snap = _snapshot(show_loading=not args.json)
    processes = filter_processes(snap.processes, gpu=args.gpu, user=args.user)
    processes.sort(key=lambda proc: (proc.gpu_index if proc.gpu_index is not None else 9999, -proc.used_memory_mb))
    if args.json:
        print(json_dump([proc.__dict__ for proc in processes]))
        return 0
    if not processes:
        print_empty("No matching GPU processes.")
        return 0
    print_processes(processes)
    return 0


def cmd_users(args) -> int:
    snap = _snapshot(show_loading=not args.json)
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
    if not summaries:
        print_empty("No active GPU users.")
        return 0
    print_users(summaries)
    return 0


def cmd_top(args) -> int:
    snap = _snapshot(show_loading=not args.json)
    if args.by == "runtime":
        processes = sorted(snap.processes, key=lambda proc: proc.running_seconds or 0, reverse=True)
    else:
        processes = sorted(snap.processes, key=lambda proc: proc.used_memory_mb, reverse=True)
    processes = processes[: max(0, args.limit)]
    if args.json:
        print(json_dump([proc.__dict__ for proc in processes]))
        return 0
    if not processes:
        print_empty("No GPU processes to rank.")
        return 0
    print_processes(processes, title=f"GPU Process Top by {args.by.title()}")
    return 0


def cmd_doctor(args) -> int:
    snap = _snapshot(show_loading=not args.json)
    alerts = evaluate_snapshot(snap, idle_util_percent=args.idle_util, occupied_memory_mb=args.occupied_mb)
    if args.json:
        print(json_dump([alert.__dict__ for alert in alerts]))
        return 0
    print_alerts(alerts)
    return 1 if any(alert.level == "crit" for alert in alerts) else 0


def cmd_kill(args) -> int:
    if args.gpu is None and args.user is None:
        print_error("Refusing to select every GPU process. Pass --gpu, --user, or both.")
        return 2
    snap = _snapshot(show_loading=not args.json)
    selected = filter_processes(snap.processes, gpu=args.gpu, user=args.user)
    sig = signal_for(args.force)
    dry_run = not args.yes
    if args.interactive and selected and not args.yes and not args.json:
        preview = kill_processes(selected, sig=sig, dry_run=True)
        print_kill_results(preview)
        dry_run = not confirm(f"Send {sig.name} to {len(selected)} selected process(es)?", default=False)

    results = []
    if selected and not dry_run and not args.json:
        with progress("Sending signals", total=len(selected)) as bar:
            task_id = bar.add_task("Sending signals", total=len(selected))
            for proc in selected:
                results.extend(kill_processes([proc], sig=sig, dry_run=False))
                bar.advance(task_id)
    else:
        results = kill_processes(selected, sig=sig, dry_run=dry_run)
    if args.json:
        print(json_dump([result.__dict__ for result in results]))
        return 0
    if not results:
        print_empty("No matching GPU processes.")
        return 0
    print_kill_results(results)
    if dry_run:
        print_empty("Dry-run only. Re-run with --yes, or use --interactive to confirm in-place.")
    return 0


def cmd_history(args) -> int:
    with loading("Reading usage history", enabled=not args.json):
        samples = read_snapshots(path=Path(args.path).expanduser(), since=since_days(args.days))
    with loading("Summarizing GPU hours", enabled=not args.json):
        usage = summarize_user_usage(samples)
    if args.json:
        print(json_dump([item.__dict__ for item in usage]))
        return 0
    if not usage:
        print_empty("No history samples found. Run `gpuops status --record` periodically first.")
        return 0
    print_history(usage, args.days)
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
        print_error(str(exc))
        return 127
