"""Rich renderables for the model console and attached monitors."""
from datetime import datetime
from urllib.parse import urlsplit

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .pageproof import LABELS


MODELS = {
    "paddle": ("PaddleOCR-VL", "基线", "bright_cyan"),
    "oxr": ("OXR", "教师", "bright_blue"),
    "ovis": ("OvisOCR3", "教师", "bright_magenta"),
    "jev": ("Jev", "评分", "bright_yellow"),
}


def elapsed(ms):
    if ms is None:
        return "—"
    return f"{ms} ms" if ms < 1000 else f"{ms / 1000:.1f} s"


def model_card(alias, index, spec, selected, requests, frame, now, spacious):
    name, role, accent = MODELS[alias]
    name = spec.get("display_name") or name
    label, status_color = LABELS[spec["status"]]
    state = Text(f"{frame if spec['status'] in {'starting', 'stopping'} else '●'} {label}",
                 style=status_color, no_wrap=True, overflow="ellipsis")
    state.append(f"  {role}", style="dim")
    address = urlsplit(spec["url"])
    hardware = Text(f"GPU {spec['gpu']}  ·  上限 {spec['gpu_memory']:.0%}", no_wrap=True, overflow="ellipsis")
    current = [r for r in requests if r["source"].split("/")[0] == alias]
    activity = Text(f":{address.port}", style=accent, no_wrap=True, overflow="ellipsis")
    if current:
        pending = sum(r["status"] is None for r in current)
        activity.append(f"  {frame} POST ×{pending}" if pending else f"  POST {elapsed(current[0]['elapsed_ms'])}")
    title = Text(f"{'▸ ' if selected else ''}{index} {name}", style=f"bold {accent}")
    lines = [state, hardware, activity]
    if spacious:
        bins = [0] * 12
        for event in current:
            age = int((now - event["started"]) // 5)
            if 0 <= age < len(bins):
                bins[-1-age] += 1
        graph = "".join("▁▂▃▄▅▆▇█"[min(7, count)] for count in bins)
        lines.append(Text(f"{graph}  60s", style=accent if any(bins) else "dim",
                          no_wrap=True, overflow="ellipsis"))
    return Panel(Group(*lines), title=title, title_align="left",
                 border_style=accent if selected else "grey50", box=box.HEAVY if selected else box.ROUNDED,
                 padding=(0, 1))


def traffic_panel(requests, limit, now, frame):
    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column(width=8)
    table.add_column(width=10)
    table.add_column(ratio=1, no_wrap=True, overflow="ellipsis")
    table.add_column(width=7, justify="right")
    table.add_column(width=9, justify="right")
    for row in requests[:limit]:
        code = row["status"]
        color = "cyan" if code is None else "green" if 200 <= code < 400 else "red"
        status = f"{frame} LIVE" if code is None else str(code or "中断")
        duration = max(0, round((now - row["started"]) * 1000)) if code is None else row["elapsed_ms"]
        table.add_row(datetime.fromtimestamp(row["started"]).strftime("%H:%M:%S"), Text(row["source"]),
                      Text("POST " + row["path"]), Text(status, style=color), Text(elapsed(duration)))
    if not requests:
        table.add_row("—", "", "等待 POST 请求", "", "")
    return Panel(table, title="实时 POST", title_align="left", border_style="cyan", box=box.ROUNDED,
                 height=limit + 2)


def log_panel(entries, rows):
    lines = []
    for entry in entries[-rows:]:
        stamp = datetime.fromtimestamp(entry["time"]).strftime("%H:%M:%S")
        line = Text(f"{stamp}  ", style="dim", no_wrap=True, overflow="ellipsis")
        message = "".join(char if char.isprintable() else " " for char in entry["message"])
        line.append(message, style="red" if entry["error"] else "")
        lines.append(line)
    if not lines:
        lines.append(Text("等待模型或任务活动", style="dim", no_wrap=True, overflow="ellipsis"))
    return Panel(Group(*lines), title="运行日志", title_align="left", border_style="cyan", box=box.ROUNDED,
                 height=rows + 2)


def console_view(snap, selected, message, expanded, width, height, now, frame):
    requests = snap.get("requests", [])
    logs = snap.get("logs", [])
    if height <= 5:
        return log_panel(logs, max(1, height - 2))
    columns = 4 if width >= 100 else 2
    spacious = height >= 30
    grid = Table.grid(expand=True, padding=(0, 1))
    for _ in range(columns):
        grid.add_column(ratio=1)
    cards = [model_card(alias, i, spec, alias == selected, requests, frame, now, spacious)
             for i, (alias, spec) in enumerate(snap["models"].items(), 1)]
    for i in range(0, len(cards), columns):
        grid.add_row(*cards[i:i + columns])
    active = [j for j in snap["jobs"] if j["status"] in {"running", "queued"}]
    finished = [j for j in reversed(snap["jobs"]) if j["status"] not in {"running", "queued"}]
    header = Text(no_wrap=True, overflow="ellipsis")
    header.append(" GLYPHSTACK ", style="bold black on bright_cyan")
    header.append(f"   {snap['mode']}   127.0.0.1:{snap['port']}", style="bold")
    header.append(f"   {len(active)} 项任务", style="cyan")
    controls = Text(no_wrap=True, overflow="ellipsis")
    for key, action in (("1–4", "选卡"), ("S", "启动"), ("X", "停止"), ("T", "↻ 重启"),
                        ("R", "运行"), ("P", "请求视图"), ("^C", "退出")):
        controls.append(f" {key} ", style="bold black on bright_cyan" if key == "T" else "bold cyan")
        controls.append(action + " ")
    jobs = Table.grid(expand=True, padding=(0, 1))
    jobs.add_column(width=8)
    jobs.add_column(width=9)
    jobs.add_column(ratio=1, overflow="ellipsis", no_wrap=True)
    jobs.add_column(ratio=1, overflow="ellipsis", no_wrap=True)
    recent_jobs = (active + finished)[:3 if spacious else 1]
    for j in recent_jobs:
        label, color = LABELS[j["status"]]
        jobs.add_row(Text(label, style=color), j["action"], Text(j["value"]), Text(j["detail"] or j["stage"]))
    if not (active or finished):
        jobs.add_row("空闲", "", Text(snap["input_path"]), "")
    visible = [header]
    remaining = height - 3  # Header, controls and footer.
    card_height = ((len(cards) + columns - 1) // columns) * (5 + int(spacious))
    if not expanded and remaining >= card_height + 3:
        visible.append(grid)
        remaining -= card_height
    visible.append(controls)
    job_height = max(1, len(recent_jobs)) + 2
    if not expanded and remaining >= job_height + 6:
        visible.append(Panel(jobs, title="任务队列", title_align="left", box=box.ROUNDED))
        remaining -= job_height
    log_height = min(8 if spacious else 5, max(3, remaining - 3))
    if remaining - log_height >= 3:
        visible.append(traffic_panel(requests, remaining - log_height - 2, now, frame))
    else:
        log_height = remaining
    footer = message or snap["models"][selected]["detail"] or snap.get("config_path") or snap["input_path"]
    visible.append(Text(footer, style="yellow" if message else "dim", no_wrap=True, overflow="ellipsis"))
    visible.append(log_panel(logs, log_height - 2))
    return Group(*visible)
