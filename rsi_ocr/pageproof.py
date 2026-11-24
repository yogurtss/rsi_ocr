"""PageProof model console, inference client and review launcher."""
import argparse
import json
import os
import select
import sys
import termios
import time
import tty
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import load_config

DEFAULT_PORT = 8520
DEFAULT_SETTINGS = Path("configs/import.yaml")


LABELS = {"offline": ("离线", "dim"), "starting": ("加载中", "yellow"), "healthy": ("健康", "green"),
          "disabled": ("已关闭", "dim"),
          "unhealthy": ("探测失败", "red"), "failed": ("失败", "red"), "stopping": ("释放中", "yellow"),
          "running": ("执行中", "cyan"), "queued": ("排队", "yellow"), "done": ("完成", "green")}


def endpoint(port, path, payload=None, timeout=5):
    data = json.dumps(payload).encode() if payload is not None else None
    req = Request(f"http://127.0.0.1:{port}{path}", data=data,
                  headers={"Content-Type": "application/json", "X-PageProof-Client": "cli"})
    try:
        with urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as error:
        raise ValueError(json.load(error).get("detail", str(error))) from error
    except (URLError, TimeoutError) as error:
        raise ValueError(f"无法连接 GlyphStack 后台 127.0.0.1:{port}，请检查 console 是否运行") from error


def dashboard(snapshot, enqueue, closing=lambda: False):
    from rich.console import Console
    from rich.live import Live
    from .console_view import console_view

    fd = sys.stdin.fileno()
    original = termios.tcgetattr(fd)
    initial = snapshot()
    aliases = list(initial["models"])
    selected = aliases[0]
    snap = initial
    last_refresh = 0.0
    console = Console()
    expanded = False
    last_message = ""
    concurrency_input = None
    try:
        tty.setcbreak(fd)
        with Live(console=console, screen=True, auto_refresh=False, redirect_stdout=True) as live:
            while not closing():
                if time.monotonic() - last_refresh >= 0.8:
                    snap = snapshot()
                    last_refresh = time.monotonic()
                frame = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"[int(time.monotonic() * 7) % 10]
                message = last_message
                if concurrency_input is not None:
                    limit = snap["models"][selected].get("max_sequences", "—")
                    message = f"{selected} 并发 {limit} → {concurrency_input or '_'}   Enter 提交 · Esc 取消"
                live.update(console_view(snap, selected, message, expanded, console.width, console.height,
                                         time.time(), frame), refresh=True)
                if select.select([fd], [], [], 0.18)[0]:
                    key = os.read(fd, 1).decode(errors="ignore").lower()
                    if key == "\x1b" and select.select([fd], [], [], 0.02)[0]:
                        os.read(fd, 32)
                        continue
                    if concurrency_input is not None:
                        if key in ("\r", "\n") and concurrency_input:
                            try:
                                job_id = enqueue("concurrency", selected, max_sequences=int(concurrency_input))
                                last_message = "已提交并发设置 " + job_id[:8]
                            except ValueError as error:
                                last_message = str(error)
                            concurrency_input = None
                        elif key == "\x1b":
                            concurrency_input = None
                        elif key in ("\x08", "\x7f"):
                            concurrency_input = concurrency_input[:-1]
                        elif key.isascii() and key.isdigit():
                            concurrency_input += key
                    elif key in ("1", "2", "3", "4"):
                        selected = aliases[int(key) - 1]
                    elif key == "c":
                        concurrency_input = ""
                    elif key == "p":
                        expanded = not expanded
                    elif key in ("s", "x", "a", "t", "r"):
                        action = {"s": "start", "x": "stop", "a": "stop", "t": "restart", "r": "run"}[key]
                        value = {"r": None, "a": "all"}.get(key, selected)
                        try:
                            job_id = enqueue(action, value)
                            last_message = "已提交 " + job_id[:8]
                        except ValueError as error:
                            last_message = str(error)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, original)


def main(argv=None):
    cli = argparse.ArgumentParser(prog="glyphstack", description="GlyphStack · Model control and document review")
    sub = cli.add_subparsers(dest="command", required=True)
    server = sub.add_parser("console", help="启动模型后台与动态终端面板，常驻模式自动并行加载")
    server.add_argument("--config", type=Path, default=DEFAULT_SETTINGS)
    server.add_argument("--port", type=int)
    server.add_argument("--headless", action="store_true", help="关闭交互面板，适用于服务进程")
    for name, help_text in (("run", "提交并跟踪 PDF 或图片任务"), ("studio", "启动前端质控平台"),
                            ("monitor", "连接已运行的后台并打开交互面板"), ("status", "查看模型与任务状态"), ("start", "加载模型"), ("stop", "卸载模型"), ("restart", "重读模型配置并重启")):
        client = sub.add_parser(name, help=help_text)
        client.add_argument("--port", type=int, default=DEFAULT_PORT, help="模型后台端口")
        if name == "run":
            client.add_argument("input", type=Path, nargs="?", help="省略时使用后台配置的 input_path")
        if name in ("start", "stop", "restart"):
            client.add_argument("alias", choices=["paddle", "oxr", "ovis", "jev", "all"])
    args = cli.parse_args(argv)
    try:
        if args.command == "console":
            import uvicorn
            from .manager import Manager, manager_app

            settings = load_config(args.config)
            port = args.port if args.port is not None else settings.get("manager_port", DEFAULT_PORT)
            defaults = Path(__file__).parents[1] / "configs/default.yaml"
            if not defaults.exists():
                import sysconfig
                defaults = Path(sysconfig.get_path("data")) / "share/pageproof/default.yaml"
            interactive = sys.stdin.isatty() and sys.stdout.isatty() and not args.headless
            manager = Manager(settings, defaults, port, args.config.resolve(),
                              log_sink=None if interactive else lambda message: print(message, flush=True))
            uvicorn.run(manager_app(manager, (lambda m: dashboard(m.snapshot, m.enqueue, m.closed.is_set)) if interactive else None), host="127.0.0.1",
                        port=port, log_level="warning" if interactive else "info", access_log=False)
        elif args.command == "monitor":
            if not (sys.stdin.isatty() and sys.stdout.isatty()):
                raise ValueError("交互面板需要终端；可用 glyphstack status 查看状态")
            dashboard(lambda: endpoint(args.port, "/status"),
                      lambda action, value, **options: endpoint(
                          args.port, "/jobs", {"action": action, "value": value, **options}
                      )["id"])
        elif args.command in {"studio", "status"}:
            snap = endpoint(args.port, "/status")
            if snap.get("service") != "pageproof":
                raise ValueError("指定端口未运行 GlyphStack 后台")
            if args.command == "status":
                print(json.dumps(snap, ensure_ascii=False, indent=2))
            else:
                import uvicorn
                from .review_app import create_app

                uvicorn.run(create_app(library=Path(snap["output_root"]), manager_port=args.port), host="127.0.0.1", port=snap["studio_port"])
        else:
            if args.command == "run":
                value = str(args.input.resolve()) if args.input else None
            else:
                value = args.alias
            job = endpoint(args.port, "/jobs", {"action": args.command, "value": value})
            print("任务 " + job["id"], flush=True)
            previous = None
            while True:
                info = endpoint(args.port, "/jobs/" + job["id"])
                current = (info["status"], info["stage"])
                if current != previous:
                    print(f"{LABELS[current[0]][0]} · {info['detail'] or current[1]}", flush=True)
                    previous = current
                if current[0] == "done":
                    if info.get("output"):
                        print("批次目录 " + info["output"])
                    break
                if current[0] == "failed":
                    return 1
                time.sleep(0.6)
    except KeyboardInterrupt:
        return 130
    except (ValueError, FileNotFoundError) as error:
        cli.exit(1, f"{error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
