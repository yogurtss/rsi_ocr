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
    address = urlsplit(spec["url"])
    state.append(f"  {role} :{address.port}", style="dim")
    hardware = Text(f"GPU {spec['gpu']}  ·  上限 {spec['gpu_memory']:.0%}", no_wrap=True, overflow="ellipsis")
    current = [r for r in requests if r["source"].split("/")[0] == alias]
    load = spec.get("load", {})
    limit = spec.get("max_sequences", "—")
    running = load.get("running", "—")
    waiting = load.get("waiting", "—")
    if running != "—" or waiting != "—":
        activity = f"运行 {running}/{limit}  等 {waiting}"
    elif spec["status"] == "healthy":
        source = alias + "/engine" if alias == "jev" else alias
        pending = sum(r["status"] is None and r["source"] == source for r in current)
        activity = f"POST {pending}  并发 {limit}"
    else:
        activity = f"并发 {limit}"
    title = Text(f"{'▸ ' if selected else ''}{index} {name}", style=f"bold {accent}")
    lines = [state, hardware, Text(activity, style="yellow" if isinstance(waiting, int) and waiting else accent,
                                  no_wrap=True, overflow="ellipsis")]
    if spacious:
        kv = load.get("kv_usage")
        bins = [0] * (8 if kv is not None else 12)
        for event in current:
            age = int((now - event["started"]) * len(bins) // 60)
            if 0 <= age < len(bins):
                bins[-1-age] += 1
        graph = "".join("▁▂▃▄▅▆▇█"[min(7, count)] for count in bins)
        detail = f"KV {kv:.0%}  {graph}" if kv is not None else f"{graph}  60s"
        lines.append(Text(detail, style=accent if kv is not None or any(bins) else "dim",
                          no_wrap=True, overflow="ellipsis"))
    return Panel(Group(*lines), title=title, title_align="left",
                 border_style=accent if selected else "grey50", box=box.HEAVY if selected else box.ROUNDED,
                 padding=(0, 1))


def layout_card(layout, height):
    device, _, index = layout["device"].partition(":")
    hardware = Text(f"GPU {index}" if device == "gpu" else "CPU", style="bright_green",
                    no_wrap=True, overflow="ellipsis")
    weights = Text(layout["path"], style="dim", no_wrap=True, overflow="ellipsis")
    if height == 3:
        hardware.append("  ·  ")
        hardware.append_text(weights)
        content = hardware
    else:
        content = Group(hardware, weights)
    return Panel(content, title="版面检测", title_align="left", border_style="grey50", box=box.ROUNDED,
                 padding=(0, 1), height=height)


def traffic_panel(requests, limit, now, frame):
    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column(width=8)
    table.add_column(width=10)
    table.add_column(ratio=1, no_wrap=True, overflow="ellipsis")
    table.add_column(width=8, justify="right", no_wrap=True)
    table.add_column(width=9, justify="right")
    for row in requests[:limit]:
        code = row["status"]
        if code is None:
            status, color = f"{frame} LIVE", "cyan"
        elif not code:
            status, color = "中断", "red"
        elif 200 <= code < 400:
            status, color = f"DONE {code}", "green"
        else:
            status, color = f"ERR {code}", "red"
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
    layout = snap.get("layout")
    layout_in_grid = bool(layout and width >= 125)
    columns = 5 if layout_in_grid else 4 if width >= 100 else 2
    spacious = height >= 30
    card_height = 5 + int(spacious)
    grid = Table.grid(expand=True, padding=(0, 1))
    for _ in range(columns):
        grid.add_column(ratio=1)
    cards = [model_card(alias, i, spec, alias == selected, requests, frame, now, spacious)
             for i, (alias, spec) in enumerate(snap["models"].items(), 1)]
    if layout_in_grid:
        cards.append(layout_card(layout, card_height))
    for i in range(0, len(cards), columns):
        grid.add_row(*cards[i:i + columns])
    active = [j for j in snap["jobs"] if j["status"] in {"running", "queued"}]
    finished = [j for j in reversed(snap["jobs"]) if j["status"] not in {"running", "queued"}]
    header = Text(no_wrap=True, overflow="ellipsis")
    header.append(" GLYPHSTACK ", style="bold black on bright_cyan")
    header.append(f"   {snap['mode']}   127.0.0.1:{snap['port']}", style="bold")
    header.append(f"   {len(active)} 项任务", style="cyan")
    controls = Text(no_wrap=True, overflow="ellipsis")
    for key, action in (("1–4", "选卡"), ("C", "并发"), ("S", "启动"), ("X", "停止"), ("A", "全停"), ("T", "↻ 重启"),
                        ("R", "运行"), ("P", "请求视图"), ("^C", "退出")):
        controls.append(f" {key} ", style="bold black on bright_cyan" if key == "T" else "bold cyan")
        controls.append(action + " ")
    jobs = Table.grid(expand=True, padding=(0, 1))
    jobs.add_column(width=8)
    jobs.add_column(width=9, no_wrap=True, overflow="ellipsis")
    jobs.add_column(ratio=1, overflow="ellipsis", no_wrap=True)
    jobs.add_column(ratio=1, overflow="ellipsis", no_wrap=True)
    recent_jobs = (active + finished)[:3 if spacious else 1]
    for j in recent_jobs:
        label, color = LABELS[j["status"]]
        action = "设置并发" if j["action"] == "concurrency" else j["action"]
        jobs.add_row(Text(label, style=color), action, Text(j["value"]), Text(j["detail"] or j["stage"]))
    if not (active or finished):
        jobs.add_row("空闲", "", Text(snap["input_path"]), "")
    visible = [header]
    remaining = height - 3  # Header, controls and footer.
    grid_height = ((len(cards) + columns - 1) // columns) * card_height
    if not expanded and remaining >= grid_height + 3:
        visible.append(grid)
        remaining -= grid_height
        if layout and not layout_in_grid and remaining >= 6:
            visible.append(layout_card(layout, 3))
            remaining -= 3
    visible.append(controls)
    job_height = max(1, len(recent_jobs)) + 2
    if not expanded:
        if remaining >= job_height + 9:
            visible.append(Panel(jobs, title="任务队列", title_align="left", box=box.ROUNDED))
            remaining -= job_height
        elif recent_jobs and remaining >= 9:
            job = recent_jobs[0]
            label, color = LABELS[job["status"]]
            summary = f"任务 · {label} · {job['detail'] or job['stage']}"
            visible.append(Text(summary, style=color, no_wrap=True, overflow="ellipsis"))
            remaining -= 1
    log_target = (8 if spacious else 5) if expanded else max(10, (remaining * 2 + 2) // 3)
    log_height = min(log_target, max(3, remaining - 3))
    if remaining - log_height >= 3:
        visible.append(traffic_panel(requests, remaining - log_height - 2, now, frame))
    else:
        log_height = remaining
    footer = message or snap["models"][selected]["detail"] or snap.get("config_path") or snap["input_path"]
    visible.append(Text(footer, style="yellow" if message else "dim", no_wrap=True, overflow="ellipsis"))
    visible.append(log_panel(logs, log_height - 2))
    return Group(*visible)
