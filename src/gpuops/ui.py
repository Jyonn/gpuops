from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, Iterator, Sequence

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.prompt import Confirm
from rich.table import Table
from rich.text import Text

from .actions import KillResult
from .alerts import Alert
from .formatting import human_duration
from .history import UserUsage
from .models import GPU, GPUProcess


console = Console()
error_console = Console(stderr=True)


@contextmanager
def loading(message: str, enabled: bool = True) -> Iterator[None]:
    if not enabled:
        yield
        return
    with console.status(f"[bold cyan]{message}[/]", spinner="dots"):
        yield


def memory_bar(used_mb: int, total_mb: int, width: int = 18) -> Text:
    ratio = 0 if total_mb <= 0 else min(1.0, used_mb / total_mb)
    filled = round(ratio * width)
    color = "green"
    if ratio >= 0.9:
        color = "red"
    elif ratio >= 0.75:
        color = "yellow"
    text = Text()
    text.append("█" * filled, style=color)
    text.append("░" * (width - filled), style="dim")
    text.append(f" {ratio * 100:4.0f}%", style=color)
    return text


def util_text(value: int) -> Text:
    style = "green"
    if value >= 90:
        style = "red"
    elif value >= 60:
        style = "yellow"
    elif value <= 5:
        style = "dim"
    return Text(f"{value}%", style=style)


def alert_badge(alerts: Sequence[str]) -> Text:
    if not alerts:
        return Text("healthy", style="green")
    text = Text()
    for index, alert in enumerate(alerts):
        if index:
            text.append("\n")
        text.append(alert, style="bold yellow")
    return text


def print_gpu_status(gpus: Sequence[GPU], alerts: Sequence[Alert]) -> None:
    alert_by_gpu: dict[int, list[str]] = {}
    for alert in alerts:
        alert_by_gpu.setdefault(alert.gpu_index, []).append(alert.message)

    table = Table(title="GPUOps Status", box=box.ROUNDED, header_style="bold cyan", show_lines=False)
    table.add_column("GPU", justify="right", style="bold", no_wrap=True)
    table.add_column("Model", overflow="fold")
    table.add_column("Memory", min_width=22)
    table.add_column("Util", justify="right", no_wrap=True)
    table.add_column("Temp", justify="right", no_wrap=True)
    table.add_column("Power", justify="right", no_wrap=True)
    table.add_column("State")
    for gpu in gpus:
        memory = Text()
        memory.append(f"{gpu.memory_used_mb}/{gpu.memory_total_mb} MB\n", style="bold")
        memory.append(memory_bar(gpu.memory_used_mb, gpu.memory_total_mb, width=10))
        table.add_row(
            str(gpu.index),
            gpu.name,
            memory,
            util_text(gpu.utilization_gpu_percent),
            "-" if gpu.temperature_c is None else f"{gpu.temperature_c}C",
            "-" if gpu.power_w is None else f"{gpu.power_w:.0f}W",
            alert_badge(alert_by_gpu.get(gpu.index, [])),
        )
    console.print(table)


def print_processes(processes: Sequence[GPUProcess], title: str = "GPU Processes") -> None:
    table = Table(title=title, box=box.ROUNDED, header_style="bold cyan")
    table.add_column("GPU", justify="right")
    table.add_column("PID", justify="right", style="bold")
    table.add_column("User")
    table.add_column("Memory", justify="right")
    table.add_column("Running", justify="right")
    table.add_column("Command", overflow="fold")
    table.add_column("CWD", overflow="fold", style="dim")
    for proc in processes:
        table.add_row(
            "-" if proc.gpu_index is None else str(proc.gpu_index),
            str(proc.pid),
            proc.user or "-",
            f"{proc.used_memory_mb} MB",
            human_duration(proc.running_seconds),
            proc.command or proc.process_name,
            proc.cwd or "-",
        )
    console.print(table)


def print_users(summaries: Sequence[dict[str, object]]) -> None:
    table = Table(title="GPU Usage by User", box=box.ROUNDED, header_style="bold cyan")
    table.add_column("User", style="bold")
    table.add_column("GPUs")
    table.add_column("Processes", justify="right")
    table.add_column("Memory", justify="right")
    table.add_column("Total Running", justify="right")
    for item in summaries:
        table.add_row(
            str(item["user"]),
            str(item["gpus"]),
            str(item["processes"]),
            f"{item['memory_mb']} MB",
            human_duration(float(item["runtime_seconds"])),
        )
    console.print(table)


def print_alerts(alerts: Sequence[Alert]) -> None:
    if not alerts:
        console.print(Panel("No suspicious GPU state found.", title="GPUOps Doctor", border_style="green"))
        return
    table = Table(title="GPUOps Doctor", box=box.ROUNDED, header_style="bold cyan")
    table.add_column("Level")
    table.add_column("GPU", justify="right")
    table.add_column("Message")
    for alert in alerts:
        style = "bold red" if alert.level == "crit" else "yellow"
        table.add_row(Text(alert.level, style=style), str(alert.gpu_index), Text(alert.message, style=style))
    console.print(table)


def print_history(usage: Sequence[UserUsage], days: int) -> None:
    table = Table(title=f"GPU Usage History ({days}d)", box=box.ROUNDED, header_style="bold cyan")
    table.add_column("User", style="bold")
    table.add_column("GPU Hours", justify="right")
    table.add_column("GB Hours", justify="right")
    table.add_column("Samples", justify="right")
    for item in usage:
        table.add_row(item.user, f"{item.gpu_hours:.2f}", f"{item.memory_gb_hours:.2f}", str(item.samples))
    console.print(table)


def print_kill_results(results: Sequence[KillResult]) -> None:
    is_plan = all(result.message == "dry-run" for result in results)
    table = Table(title="Kill Plan" if is_plan else "Kill Results", box=box.ROUNDED, header_style="bold cyan")
    table.add_column("GPU", justify="right")
    table.add_column("PID", justify="right", style="bold")
    table.add_column("User")
    table.add_column("Signal")
    table.add_column("Result")
    for result in results:
        style = "green" if result.killed else "yellow" if result.message == "dry-run" else "red"
        table.add_row(
            "-" if result.gpu_index is None else str(result.gpu_index),
            str(result.pid),
            result.user or "-",
            result.signal_name,
            Text(result.message, style=style),
        )
    console.print(table)


def confirm(message: str, default: bool = False) -> bool:
    return Confirm.ask(message, default=default, console=console)


def progress(description: str, total: int) -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )


def print_empty(message: str) -> None:
    console.print(Panel(message, border_style="dim"))


def print_error(message: str) -> None:
    error_console.print(f"[bold red]error:[/] {message}")
