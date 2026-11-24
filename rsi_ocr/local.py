"""Configure local OCR services and reuse their cached responses."""
import json
import os
import signal
import subprocess
import time
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path

from rsi_ocr.capture import capture_pages
from rsi_ocr.config import load_config
from rsi_ocr.mining import mine, warm
from rsi_ocr.service import Service, ServiceError, command
from rsi_ocr.store import json_text


def check_checkpoint(spec):
    if "local_weights" not in spec:
        return
    root = Path(spec["local_weights"])
    metadata = json.loads((root / "config.json").read_bytes())
    hidden_size = metadata.get("text_config", metadata).get("hidden_size")
    with (root / "model.safetensors").open("rb") as file:
        header_length = int.from_bytes(file.read(8), "little")
        if not 2 <= header_length <= 64 * 1024 * 1024:
            raise ValueError(f"invalid safetensors header: {root}")
        header = json.loads(file.read(header_length))
    widths = {value["shape"][-1] for name, value in header.items() if name.endswith("embed_tokens.weight")}
    if hidden_size and widths and widths != {hidden_size}:
        raise ValueError(f"{root} config expects hidden size {hidden_size}, checkpoint has {sorted(widths)}")


@contextmanager
def running(config, alias, log, sidecar=None):
    if sidecar is None:
        check_checkpoint(config["models"][alias])
    args, env, cwd = command(config, alias, sidecar)
    if journal := config.get("request_journal"):
        env["GLYPHSTACK_JOURNAL"] = journal
        env["GLYPHSTACK_SOURCE"] = "jev/engine" if alias == "jev" and not sidecar else alias
        if sidecar == "render":
            # Launch the SDK's ASGI app with the same renderer environment as start_render_score.sh.
            sdk = cwd / "sdk"
            port, engine = args[-2:]
            env["JEV_BACKEND_URL"] = engine
            env.setdefault("PYTHONPYCACHEPREFIX", str(Path(env.get("TMPDIR", "/tmp")) / "jev-pycache"))
            args = ["python3", "-m", "uvicorn", "rsi_ocr.traffic:render_app", "--factory", "--app-dir", str(sdk),
                    "--host", env.get("HOST", "127.0.0.1"), "--port", port, "--no-access-log"]
            cwd = sdk
        else:
            args.extend(["--middleware", "rsi_ocr.traffic.RequestMeter"])
    with log.open("ab") as stream:
        process = subprocess.Popen(args, env=env, cwd=cwd, stdout=stream, stderr=stream, start_new_session=True)
        try:
            yield process
        finally:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def healthy(spec, path):
    try:
        response = Service({**spec, "timeout": 2}).request(path)
        if path == "/healthz":
            return response.get("status") == "ok"
        return spec["model"] in {row["id"] for row in response.get("data", [])}
    except (ServiceError, KeyError, ValueError):
        return False


def endpoints(spec):
    if spec["kind"] == "jev":
        engine_url = spec.get("engine_url") or spec["sidecars"]["render"]["argv"][-1]
        return [({**spec, "base_url": engine_url}, "/v1/models"), (spec, "/healthz")]
    return [(spec, "/models")]


@contextmanager
def model_session(config, alias, output, cancelled=lambda: False):
    import socket
    from urllib.parse import urlsplit

    spec = config["models"][alias]
    probes = endpoints(spec)
    with ExitStack() as stack:
        processes = []
        for index, (endpoint, path) in enumerate(probes):
            address = urlsplit(endpoint["base_url"])
            with socket.socket() as sock:
                if sock.connect_ex((address.hostname, address.port)) == 0:
                    raise RuntimeError(f"端口已被占用：{address.port}")
            log = output / (f"{alias}-render.log" if index else f"{alias}.log")
            process = stack.enter_context(running(config, alias, log, "render" if index else None))
            deadline = time.monotonic() + 900
            while process.poll() is None and not cancelled() and time.monotonic() < deadline:
                if healthy(endpoint, path):
                    break
                time.sleep(1)
            else:
                raise RuntimeError(f"{alias} 启动中断或未就绪，日志 {log}")
            processes.append(process)
        yield processes


def mining_models(policy):
    aliases = [policy["target"], *policy["teachers"]]
    if judge := policy.get("judge"):
        aliases.append(judge)
    return aliases


def apply_model_selection(config, policy):
    def enabled(alias):
        return config["models"][alias].get("enabled", True)

    judge = policy.get("judge")
    config["mining"] = {**policy, "teachers": [alias for alias in policy["teachers"] if enabled(alias)],
                        "judge": judge if judge and enabled(judge) else None}


def run_local(config, output, jobs, pages=None, stage="all", session=None, progress=None):
    output.mkdir(parents=True, exist_ok=True)
    policy = config["mining"]
    target = policy["target"]
    if config["models"][target].get("local_weights"):
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "1")

    def use_model(alias):
        if progress is None:
            print(f"Starting {alias}. Log: {output / (alias + '.log')}", flush=True)
        return session(alias, output) if session else model_session(config, alias, output)

    for alias in [target, *policy["teachers"]]:
        if stage not in ("all", alias):
            continue
        with use_model(alias):
            if alias == target and pages:
                capture_pages(config, pages.resolve(), output / "capture", progress=progress)
            for manifest, run in jobs:
                counts = warm(config, manifest, run, alias)
                if progress:
                    progress(f"{alias} 转录完成 · {sum(counts.values())} 个区域 · "
                             f"{counts.get('service_error', 0)} 个服务错误")
                else:
                    print(f"{alias} {run.name} {json_text(counts)}", flush=True)
                if counts.get("service_error"):
                    raise RuntimeError(f"{alias} returned service errors for {run.name}; see {output}")
    judge = policy.get("judge")
    if stage in ("all", judge or "mine"):
        with (use_model(judge) if judge else nullcontext()):
            for manifest, run in jobs:
                summary = mine(config, manifest, run, cached_readers=True)
                if progress:
                    progress(f"筛选完成 · 扫描 {summary['scanned']} 个区域 · {summary['review']} 个待复核")
                else:
                    print(f"{run.name} {json_text(summary)}", flush=True)


def local_config(settings, base):
    config = load_config(base)
    deployment = settings.get("deployment", {})
    root = deployment.get("model_root", settings.get("model_root", "/mnt/g/models"))
    minicpm = deployment.get("minicpm_repo", settings.get("minicpm_repo", "runs/MiniCPM-OCR"))
    configure_models(config, Path(root).expanduser().resolve(), Path(minicpm).expanduser(), deployment)
    config["mining"].update(settings["mining"])
    return config


def configure_models(config, root, minicpm, deployment=None):
    deployment = deployment or {}
    models = config["models"]
    models["paddle"]["revision"] = "PaddleOCR-VL-1.6-local"
    models["oxr"]["revision"] = "OXR-1.0-c708407f"
    models["ovis"]["revision"] = "OvisOCR3-5b252386"
    models["jev"]["revision"] = "MiniCPM-OCR-Jev-a0ed278e"

    for alias, directory, length in [
        ("paddle", "paddleocr-vl-16", 12288), ("oxr", "OXR-1.0", 16384),
        ("ovis", "OvisOCR3", 16384), ("jev", "MiniCPM-OCR-Jev", 16384),
    ]:
        launch = models[alias]["launch"]
        options = deployment.get("models", {}).get(alias, {})
        enabled = options.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError(f"{alias}.enabled 需为 true 或 false")
        if alias == "paddle" and not enabled:
            raise ValueError("PaddleOCR-VL 必须启用")
        models[alias]["enabled"] = enabled
        models[alias]["display_name"] = options.get("display_name")
        path = options.get("path", directory)
        if not isinstance(path, str) or not path.strip():
            raise ValueError(f"{alias} 的权重路径不能为空")
        model_dir = root / Path(path).expanduser()
        models[alias]["configured_path"] = path
        models[alias]["configured_gpu"] = str(options["gpu"]) if "gpu" in options else ""
        length = options.get("context", length)
        if "revision" in options:
            models[alias]["revision"] = options["revision"]
        if alias == "ovis":
            models[alias]["tasks"]["text"]["prompt_file"] = str(model_dir / "prompts/plain_ocr.txt")
        models[alias]["local_weights"] = str(model_dir)
        if alias == "oxr":
            launch.clear()
            launch["env"] = {"VLLM_PLUGINS": "oxr"}
            launch["argv"] = ["vllm", "serve", str(model_dir), "--port", "8080", "--trust-remote-code",
                              "--served-model-name", "oxr", "--chat-template", str(model_dir / "chat_template.jinja"),
                              "--dtype", "bfloat16", "--attention-config", '{"backend":"TRITON_ATTN"}']
        else:
            argv = launch["argv"]
            argv[2] = str(model_dir)
            host_index = argv.index("--host")
            del argv[host_index:host_index+2]
            if "--max-model-len" in argv:
                limit_index = argv.index("--max-model-len")
                del argv[limit_index:limit_index+2]
            launch["env"] = {"VLLM_PLUGINS": ""}
        port_index = launch["argv"].index("--port") + 1
        port = int(options.get("port", launch["argv"][port_index]))
        launch["argv"][port_index] = str(port)
        endpoint = f"http://127.0.0.1:{port}"
        if alias == "jev":
            models[alias]["engine_url"] = endpoint
        else:
            models[alias]["base_url"] = endpoint + "/v1"
        resource = {**deployment, **options}
        if "gpu" in resource:
            launch["env"]["CUDA_VISIBLE_DEVICES"] = str(resource["gpu"])
        fraction = float(resource.get("gpu_memory", 0.65))
        sequences = int(resource.get("max_sequences", 4))
        parallel = int(resource.get("tensor_parallel", 1))
        if not 0 < fraction <= 1 or min(sequences, parallel, int(length)) < 1:
            raise ValueError(f"{alias} 的显存比例须在 (0,1] 内，并发、张量并行和上下文长度须为正整数")
        launch["argv"] += ["--host", "127.0.0.1", "--gpu-memory-utilization", str(fraction),
                           "--max-num-seqs", str(sequences), "--enforce-eager", "--max-model-len", str(length),
                           "--tensor-parallel-size", str(parallel)]
        models[alias]["gpu"] = launch["env"].get("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "auto"))
        models[alias]["gpu_memory"] = fraction
        launch["env"]["VLLM_USE_V2_MODEL_RUNNER"] = "0"
        models[alias]["concurrency"] = 3
        models[alias]["timeout"] = 300

    jev_dir = minicpm.resolve() / "scripts/minicpm-ocr-jev"
    sidecar = models["jev"]["sidecars"]["render"]
    render_port = deployment.get("models", {}).get("jev", {}).get("render_port", 18500)
    sidecar["argv"][-2:] = [str(render_port), models["jev"]["engine_url"]]
    models["jev"]["base_url"] = f"http://127.0.0.1:{render_port}"
    sidecar["cwd"] = str(jev_dir)
    sidecar["env"] = {"HOST": "127.0.0.1", "JEV_WEIGHTS": models["jev"]["local_weights"],
                      "FONTCONFIG_FILE": str(Path("configs/jev-fonts.conf").resolve()),
                      "PLAYWRIGHT_BROWSERS_PATH": str(Path("runs/playwright").resolve())}
    libraries = Path("runs/chromium-libs/usr/lib/x86_64-linux-gnu").resolve()
    if libraries.is_dir():
        sidecar["env"]["LD_LIBRARY_PATH"] = os.pathsep.join(
            entry for entry in (str(libraries), os.environ.get("LD_LIBRARY_PATH")) if entry
        )
    config["capture"]["pipeline"].update(
        pipeline_version="v1.6", device="cpu", layout_detection_model_dir=str(root / deployment.get("layout_path", "PP-DocLayoutV3")),
    )
    config["mining"]["workers"] = 4
    config["mining"]["splits"].append("benchmark")
