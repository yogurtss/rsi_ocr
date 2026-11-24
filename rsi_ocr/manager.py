"""Local model service ownership and inference queue for PageProof."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, asynccontextmanager, contextmanager, nullcontext
from copy import deepcopy
from pathlib import Path
from queue import Queue
from tempfile import TemporaryDirectory
from threading import Event, RLock, Thread
from time import time
from typing import Literal
from urllib.parse import urlsplit
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, StrictBool, StrictInt
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .traffic import RequestMeter, model_load, open_journal, recent, record
from .config import load_config, resolve_config
from .importer import render_pages, resume_batch
from .local import (
    apply_model_selection, check_checkpoint, endpoints, healthy, local_config, mining_models,
    model_session, run_local, set_concurrency,
)
from .store import atomic_output


class Manager:
    def __init__(self, settings, base, port, config_path=None, log_sink=None):
        self.settings = settings
        self.config_path = config_path
        self.mode = settings.get("deployment", {}).get("mode", "on_demand")
        if self.mode not in {"on_demand", "managed"}:
            raise ValueError("deployment.mode 需为 on_demand 或 managed")
        self.port = port
        self.root = Path(settings["output_root"]).resolve()
        self.default_input = Path(settings.get("input_path", "data/inbox")).resolve()
        self.config = local_config(settings, base)
        self.base = base
        self.policy = self.config["mining"]
        apply_model_selection(self.config, self.policy)
        self.aliases = mining_models(self.policy)
        self.validate_ports(self.config["models"])
        self.models = {alias: {**self.card(self.config["models"][alias]), "detail": ""} for alias in self.aliases}
        self.journal_dir = None
        self.journal_path = None
        self.lock = RLock()
        self.logs = deque(maxlen=128)
        self.log_sink = log_sink
        self.closing = Event()
        self.closed = Event()
        self.queue = Queue()
        self.jobs = {}
        self.sessions = {}
        self.worker = Thread(target=self.work, name="pageproof-worker")
        self.monitor = Thread(target=self.monitor_health, name="pageproof-health")

    @staticmethod
    def card(spec):
        return {"model": spec["model"], "display_name": spec.get("display_name"), "url": spec["base_url"], "gpu": spec["gpu"],
                "gpu_memory": spec["gpu_memory"], "max_sequences": spec["max_sequences"], "load": {},
                "path": spec["local_weights"], "enabled": spec["enabled"],
                "configured_path": spec["configured_path"], "configured_gpu": spec["configured_gpu"],
                "status": "offline" if spec["enabled"] else "disabled"}

    def validate_ports(self, models):
        ports = [self.port, self.settings["port"]]
        for alias in self.aliases:
            if models[alias]["enabled"]:
                ports.extend(urlsplit(s["base_url"]).port for s, _ in endpoints(models[alias]))
        if len(ports) != len(set(ports)) or not all(isinstance(p, int) and 0 < p < 65536 for p in ports):
            raise ValueError("后台、前端及模型的端口必须有效且互不重复")

    def prepare_restart(self, aliases):
        if not self.config_path:
            for alias in aliases:
                self.unload(alias)
            return
        settings = load_config(self.config_path)
        config = local_config(settings, self.base)
        replacements = {a: config["models"][a] for a in aliases}
        self.validate_ports({**self.config["models"], **replacements})
        for spec in replacements.values():
            if spec["enabled"]:
                check_checkpoint(spec)
        for alias in aliases:
            self.unload(alias)
        with self.lock:
            for alias, spec in replacements.items():
                self.config["models"][alias] = spec
                self.models[alias].update(self.card(spec), detail="")
            if "paddle" in replacements:
                self.config["capture"] = config["capture"]
                self.config["layout"] = config["layout"]
            apply_model_selection(self.config, self.policy)

    def start(self):
        self.journal_dir = TemporaryDirectory(prefix="glyphstack-")
        self.journal_path = str(Path(self.journal_dir.name) / "traffic.sqlite")
        open_journal(self.journal_path)
        self.config["request_journal"] = self.journal_path
        if self.mode == "managed":
            self.enqueue("start", "all")
        self.worker.start()
        self.monitor.start()

    def close(self):
        with self.lock:
            self.closing.set()
            self.log("后台正在关闭 · 正在卸载模型并中止任务")
            self.queue.put(None)
        try:
            self.unload_all()
        finally:
            self.worker.join()
            self.monitor.join()
            with self.lock:
                self.journal_path = None
            try:
                self.journal_dir.cleanup()
            finally:
                self.closed.set()

    def check_closing(self):
        if self.closing.is_set():
            raise RuntimeError("后台正在关闭")

    def report_progress(self, message):
        self.check_closing()
        self.log(message)

    def snapshot(self):
        with self.lock:
            requests = recent(self.journal_path)
            active = [j for j in self.jobs.values() if j["status"] in {"running", "queued"}]
            finished = [j for j in self.jobs.values() if j["status"] not in {"running", "queued"}]
            return {"requests": requests, "logs": [entry.copy() for entry in self.logs],
                    "config_path": str(self.config_path or ""),
                    "service": "pageproof", "mode": self.mode, "port": self.port, "studio_port": self.settings["port"],
                    "layout": self.config["layout"].copy(), "layout_device": self.config["layout"]["device"],
                    "output_root": str(self.root), "input_path": str(self.default_input),
                    "models": {k: v.copy() for k, v in self.models.items()},
                    "jobs": [v.copy() for v in finished[-25:] + active]}

    def enqueue(self, action, value, max_sequences=None):
        if max_sequences is not None and action != "concurrency":
            raise ValueError("只有并发设置操作接受 max_sequences")
        if action == "run":
            value = str(Path(value or self.default_input).resolve())
            source = Path(value)
            if not source.exists():
                raise ValueError(f"输入路径不存在：{source}")
            if source.is_relative_to(self.root):
                raise ValueError("请将输入文件与结果目录分开")
        elif action == "concurrency":
            if value not in self.models:
                raise ValueError("请选择一个模型调整并发")
            if type(max_sequences) is not int or max_sequences < 1:
                raise ValueError("并发须为正整数")
        elif (value not in self.models and value != "all") or action not in {"start", "stop", "restart"}:
            raise ValueError("未知模型或操作")
        job = {"id": uuid4().hex, "action": action, "value": value, "status": "queued", "stage": "等待", "detail": ""}
        if action == "concurrency":
            job.update(max_sequences=max_sequences, stage=f"等待设置为 {max_sequences}")
        with self.lock:
            if self.closing.is_set():
                raise ValueError("后台正在关闭")
            if action == "start" and value == "all" and self.mode != "managed":
                raise ValueError("start all 需要后台并行常驻模式")
            if action == "start" and value != "all" and not self.models[value]["enabled"]:
                raise ValueError(f"{value} 已在模型设置中关闭")
            self.jobs[job["id"]] = job
            self.queue.put(job)
        return job["id"]

    def log(self, message, *, error=False):
        with self.lock:
            self.logs.append({"time": time(), "message": message, "error": error})
        if self.log_sink:
            self.log_sink(message)

    def update_job(self, job, **values):
        with self.lock:
            job.update(values)
            status = values.get("status")
            if status == "failed":
                self.log(f"任务失败 · {job['detail']}", error=True)
            elif status == "running":
                self.log(f"任务开始 · {job['action']} · {job['value']}")
            elif "stage" in values:
                self.log("任务 · " + job["stage"])

    def set_mode(self, mode):
        if mode not in {"on_demand", "managed"}:
            raise ValueError("未知部署模式")
        with self.lock:
            if self.mode == mode:
                return
            self.validate_settings_change()
            if self.config_path:
                yaml, config = self.read_settings()
                config.setdefault("deployment", {})["mode"] = mode
                self.write_settings(yaml, config)
            self.apply_mode(mode)

    def read_settings(self):
        from ruamel.yaml import YAML

        yaml = YAML()
        yaml.preserve_quotes = True
        return yaml, yaml.load(self.config_path)

    def write_settings(self, yaml, settings):
        with atomic_output(self.config_path) as tmp:
            yaml.dump(settings, tmp)
            tmp.chmod(self.config_path.stat().st_mode)

    def update_concurrency(self, alias, sequences):
        with self.lock:
            self.check_closing()
            spec = self.config["models"][alias]
            if spec["max_sequences"] == sequences:
                return
            if self.config_path:
                yaml, settings = self.read_settings()
                models = settings.setdefault("deployment", {}).setdefault("models", {})
                models.setdefault(alias, {})["max_sequences"] = sequences
                self.write_settings(yaml, settings)
            restart = alias in self.sessions
            set_concurrency(spec, sequences)
            self.models[alias]["max_sequences"] = sequences
            self.log(f"并发已设置 · {alias} · {sequences}")
        if restart:
            self.unload(alias)
            directory = self.root / ".manager"
            directory.mkdir(parents=True, exist_ok=True)
            self.load(alias, directory)

    def validate_settings_change(self):
        if self.closing.is_set():
            raise ValueError("后台正在关闭")
        if any(j["status"] in {"queued", "running"} for j in self.jobs.values()):
            raise ValueError("请等待当前任务和排队操作完成后保存模型设置")

    def apply_mode(self, mode):
        previous = self.mode
        self.mode = mode
        if self.worker.is_alive():
            if mode == "managed":
                self.enqueue("start", "all")
            elif previous == "managed":
                self.enqueue("stop", "all")

    def set_deployment(self, mode, changes, layout_device=None):
        with self.lock:
            if not self.config_path:
                raise ValueError("后台缺少 YAML 配置路径")
            self.validate_settings_change()
            if changes.keys() - self.models.keys() - {"layout"}:
                raise ValueError("未知模型")
            yaml, settings = self.read_settings()
            deployment = settings.setdefault("deployment", {})
            deployment["mode"] = mode
            options = deployment.setdefault("models", {})
            if layout_device is not None and "layout" not in changes:
                options.setdefault("layout", {})["device"] = layout_device
                deployment.pop("layout_device", None)
            for alias, change in changes.items():
                if alias == "layout":
                    if not isinstance(change, LayoutSettings):
                        raise ValueError("版面检测设置需要权重路径和设备")
                    options.setdefault("layout", {}).update(path=change.path.strip(), device=change.device)
                    deployment.pop("layout_path", None)
                    deployment.pop("layout_device", None)
                    continue
                if not isinstance(change, ModelSettings):
                    raise ValueError(f"{alias} 设置需要启用状态、权重路径和 GPU")
                model = options.setdefault(alias, {})
                model.update(enabled=change.enabled, path=change.path.strip())
                if gpu := change.gpu.strip():
                    model["gpu"] = gpu
                else:
                    model.pop("gpu", None)
            config = local_config(settings, self.base)
            self.validate_ports(config["models"])
            changed = [alias for alias in self.aliases if config["models"][alias] != self.config["models"][alias]]
            if config["capture"] != self.config["capture"] and "paddle" not in changed:
                changed.append("paddle")
            self.write_settings(yaml, settings)
            for alias in changed:
                self.unload(alias)
                spec = config["models"][alias]
                self.config["models"][alias] = spec
                self.models[alias].update(self.card(spec), detail="")
            self.config["capture"] = config["capture"]
            self.config["layout"] = config["layout"]
            apply_model_selection(self.config, self.policy)
            self.apply_mode(mode)

    def unload(self, alias):
        with self.lock:
            session = self.sessions.pop(alias, None)
            if session is None:
                return
            self.models[alias].update(status="stopping", load={})
            self.log(f"正在卸载模型 · {alias}")
        session[0].close()
        if self.journal_path:
            record(self.journal_path, "UPDATE requests SET status=0 WHERE status IS NULL AND source IN (?,?)",
                   (alias, alias + "/engine"))
        with self.lock:
            self.models[alias].update(status="offline", detail="")
            self.log(f"模型已卸载 · {alias}")

    def unload_all(self):
        with self.lock:
            aliases = list(self.sessions)
        if len(aliases) > 1:
            with ThreadPoolExecutor(max_workers=len(aliases)) as pool:
                list(pool.map(self.unload, aliases))
        elif aliases:
            self.unload(aliases[0])

    def is_ready(self, alias):
        session = self.sessions.get(alias)
        return session and all(p.poll() is None for p in session[1]) and all(
            healthy(s, path) for s, path in endpoints(self.config["models"][alias]))

    def load(self, alias, output):
        self.check_closing()
        if not self.models[alias]["enabled"]:
            raise ValueError(f"{alias} 已在模型设置中关闭")
        if self.is_ready(alias):
            self.log(f"模型已就绪 · 复用 {alias}")
            return
        if self.mode == "on_demand":
            self.unload_all()
        else:
            self.unload(alias)
        with self.lock:
            self.check_closing()
            self.models[alias].update(status="starting", detail=str(output / (alias + ".log")))
            self.log(f"正在加载模型 · {alias} · GPU {self.models[alias]['gpu']}")
        try:
            with ExitStack() as stack:
                processes = stack.enter_context(model_session(self.config, alias, output, self.closing.is_set))
                with self.lock:
                    self.check_closing()
                    self.sessions[alias] = (stack.pop_all(), processes)
                    self.models[alias]["status"] = "healthy"
                    self.log(f"模型已就绪 · {alias}")
        except Exception as error:
            with self.lock:
                self.models[alias].update(status="failed", detail=str(error))
                self.log(f"模型加载失败 · {alias} · {error}", error=True)
            raise

    def monitor_health(self):
        while not self.closing.wait(2):
            with self.lock:
                sessions = list(self.sessions.items())
            for alias, session in sessions:
                if self.closing.is_set():
                    return
                ok = self.is_ready(alias)
                spec = self.config["models"][alias]
                load = model_load(spec.get("engine_url") or spec["base_url"]) if ok else {}
                with self.lock:
                    if self.sessions.get(alias) is session:
                        self.models[alias].update(status="healthy" if ok else "unhealthy", load=load)

    def infer(self, job):
        if self.mode == "managed":
            missing = [alias for alias in mining_models(self.config["mining"]) if not self.is_ready(alias)]
            if missing:
                raise ValueError("模型未就绪，详情见工作台或 glyphstack status：" + ", ".join(missing))
        source = Path(job["value"])
        self.update_job(job, stage="检查输入与续跑进度")
        with resume_batch(self.root, source, self.settings, self.check_closing) as (output, resumed, signatures):
            if resumed:
                self.log(f"续跑批次 · {output}")
            self.update_job(job, output=str(output), stage="导入页面")
            pages = render_pages(source, output / "input", self.settings, progress=self.report_progress,
                                 reuse_existing=resumed, signatures=signatures)
            self.infer_pages(job, output, pages)

    def infer_pages(self, job, output, pages):
        config = deepcopy(self.config)
        resolve_config(config, self.base.parent)

        @contextmanager
        def use_model(alias, directory):
            self.check_closing()
            self.update_job(job, stage=f"加载 {alias}")
            self.load(alias, directory)
            self.update_job(job, stage={"paddle": "版面与基线识别", "oxr": "OXR 转录", "ovis": "Ovis 转录",
                                        "jev": "Jev 评分与难例筛选"}[alias])
            yield

        jobs = [(output / "capture/regions.jsonl", output / "production")]
        if self.mode == "managed":
            policy = config["mining"]

            def stage(alias):
                run_local(config, output, jobs, pages, stage=alias, session=lambda *_: nullcontext(),
                          progress=self.report_progress, check_cancelled=self.check_closing)

            self.update_job(job, stage="版面与基线识别")
            stage(policy["target"])
            if policy["teachers"]:
                self.update_job(job, stage="教师模型转录")
                with ThreadPoolExecutor(max_workers=len(policy["teachers"])) as pool:
                    list(pool.map(stage, policy["teachers"]))
            self.update_job(job, stage="Jev 评分与难例筛选" if policy["judge"] else "难例筛选")
            stage(policy["judge"] or "mine")
            return
        try:
            run_local(config, output, jobs, pages,
                      session=use_model, progress=self.report_progress, check_cancelled=self.check_closing)
        finally:
            self.unload_all()

    def start_models(self, aliases, directory, job):
        aliases = [alias for alias in aliases if self.models[alias]["enabled"]]
        if self.mode == "managed" and len(aliases) > 1:
            self.update_job(job, stage="并行加载模型")
            failures = []
            with ThreadPoolExecutor(max_workers=len(aliases)) as pool:
                starts = {alias: pool.submit(self.load, alias, directory) for alias in aliases}
                for alias, future in starts.items():
                    try:
                        future.result()
                    except Exception as error:
                        failures.append(f"{alias}：{error}")
            if failures:
                raise RuntimeError("；".join(failures))
            return
        for alias in aliases:
            self.update_job(job, stage=f"加载 {alias}")
            self.load(alias, directory)

    def work(self):
        try:
            while (job := self.queue.get()) is not None:
                if self.closing.is_set():
                    self.update_job(job, status="failed", detail="后台已关闭")
                    continue
                self.update_job(job, status="running", stage=job["action"])
                try:
                    if job["action"] == "run":
                        self.infer(job)
                    elif job["action"] == "concurrency":
                        self.update_job(job, stage=f"设置 {job['value']} 并发为 {job['max_sequences']}")
                        self.update_concurrency(job["value"], job["max_sequences"])
                    elif job["action"] in {"start", "restart"}:
                        directory = self.root / ".manager"
                        directory.mkdir(parents=True, exist_ok=True)
                        aliases = mining_models(self.config["mining"]) if job["value"] == "all" else [job["value"]]
                        if job["action"] == "restart":
                            if job["value"] == "all":
                                aliases = list(self.sessions)
                            self.prepare_restart(aliases)
                        self.start_models(aliases, directory, job)
                    elif job["value"] == "all":
                        self.update_job(job, stage="卸载全部模型")
                        self.unload_all()
                    else:
                        self.unload(job["value"])
                    self.update_job(job, status="done", stage="完成")
                except Exception as error:
                    self.update_job(job, status="failed", detail=str(error))
        finally:
            self.unload_all()


class Command(BaseModel):
    action: str
    value: str | None = None
    max_sequences: StrictInt | None = None


class ModelSettings(BaseModel):
    enabled: StrictBool
    path: str
    gpu: str


class LayoutSettings(BaseModel):
    path: str
    device: str


class DeploymentSettings(BaseModel):
    mode: Literal["on_demand", "managed"]
    models: dict[str, ModelSettings | LayoutSettings]
    layout_device: str | None = None


def manager_app(manager, terminal=None):
    @asynccontextmanager
    async def lifespan(app):
        manager.start()
        screen = Thread(target=terminal, args=(manager,), daemon=True) if terminal else None
        if screen:
            screen.start()
        try:
            yield
        finally:
            manager.close()
            if screen:
                screen.join(timeout=2)

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])

    @app.middleware("http")
    async def local_client(request: Request, call_next):
        if request.method == "POST" and request.headers.get("X-PageProof-Client") != "cli":
            return JSONResponse({"detail": "请使用本机 GlyphStack 客户端"}, status_code=403)
        return await call_next(request)

    app.add_middleware(RequestMeter, source="manager", journal=lambda: manager.journal_path)

    @app.get("/status")
    def status():
        return manager.snapshot()

    @app.get("/jobs/{job_id}")
    def job_status(job_id: str):
        with manager.lock:
            if job_id not in manager.jobs:
                raise HTTPException(404, "任务不存在")
            return manager.jobs[job_id].copy()

    @app.post("/deployment/{mode}")
    def deployment(mode: str):
        try:
            manager.set_mode(mode)
        except (ValueError, OSError) as error:
            raise HTTPException(409, str(error)) from error
        return {"mode": manager.mode}

    @app.post("/deployment")
    def save_deployment(settings: DeploymentSettings):
        try:
            manager.set_deployment(settings.mode, settings.models, layout_device=settings.layout_device)
        except (ValueError, OSError) as error:
            raise HTTPException(409, str(error)) from error
        return {"mode": manager.mode}

    @app.post("/jobs", status_code=202)
    def submit(command: Command):
        try:
            return {"id": manager.enqueue(command.action, command.value, command.max_sequences)}
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    return app
