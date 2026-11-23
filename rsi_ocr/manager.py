"""Local model service ownership and inference queue for PageProof."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, asynccontextmanager, contextmanager, nullcontext
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue
from tempfile import TemporaryDirectory
from threading import Event, RLock, Thread
from typing import Literal
from urllib.parse import urlsplit
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, StrictBool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .traffic import RequestMeter, open_journal, recent, record
from .config import load_config, resolve_config
from .importer import render_pages
from .local import apply_model_selection, check_checkpoint, endpoints, healthy, local_config, mining_models, model_session, run_local
from .store import atomic_output


class Manager:
    def __init__(self, settings, base, port, config_path=None):
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
        self.closing = Event()
        self.queue = Queue()
        self.jobs = {}
        self.sessions = {}
        self.worker = Thread(target=self.work, name="pageproof-worker")
        self.monitor = Thread(target=self.monitor_health, name="pageproof-health")

    @staticmethod
    def card(spec):
        return {"model": spec["model"], "display_name": spec.get("display_name"), "url": spec["base_url"], "gpu": spec["gpu"],
                "gpu_memory": spec["gpu_memory"], "path": spec["local_weights"], "enabled": spec["enabled"],
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
            apply_model_selection(self.config, self.policy)

    def start(self):
        self.journal_dir = TemporaryDirectory(prefix="glyphstack-")
        self.journal_path = str(Path(self.journal_dir.name) / "traffic.sqlite")
        open_journal(self.journal_path)
        self.config["request_journal"] = self.journal_path
        self.worker.start()
        self.monitor.start()

    def close(self):
        with self.lock:
            self.closing.set()
            self.queue.put(None)
        self.worker.join()
        self.monitor.join()
        self.journal_path = None
        self.journal_dir.cleanup()

    def snapshot(self):
        requests = recent(self.journal_path)
        with self.lock:
            active = [j for j in self.jobs.values() if j["status"] in {"running", "queued"}]
            finished = [j for j in self.jobs.values() if j["status"] not in {"running", "queued"}]
            return {"requests": requests, "config_path": str(self.config_path or ""),
                    "service": "pageproof", "mode": self.mode, "port": self.port, "studio_port": self.settings["port"],
                    "output_root": str(self.root), "input_path": str(self.default_input),
                    "models": {k: v.copy() for k, v in self.models.items()},
                    "jobs": [v.copy() for v in finished[-25:] + active]}

    def enqueue(self, action, value):
        if action == "run":
            value = str(Path(value or self.default_input).resolve())
            source = Path(value)
            if not source.exists():
                raise ValueError(f"输入路径不存在：{source}")
            if source.is_relative_to(self.root):
                raise ValueError("请将输入文件与结果目录分开")
        elif (value not in self.models and value != "all") or action not in {"start", "stop", "restart"}:
            raise ValueError("未知模型或操作")
        job = {"id": uuid4().hex, "action": action, "value": value, "status": "queued", "stage": "等待", "detail": ""}
        with self.lock:
            if self.closing.is_set():
                raise ValueError("后台正在关闭")
            if action == "start" and value == "all" and self.mode != "managed":
                raise ValueError("start all 需要 CLI 常驻模式")
            if action == "start" and value != "all" and not self.models[value]["enabled"]:
                raise ValueError(f"{value} 已在模型设置中关闭")
            self.jobs[job["id"]] = job
            self.queue.put(job)
        return job["id"]

    def update(self, row, **values):
        with self.lock:
            row.update(values)

    def set_mode(self, mode):
        if mode not in {"on_demand", "managed"}:
            raise ValueError("未知部署模式")
        with self.lock:
            if self.mode == mode:
                return
            self.validate_settings_change(mode)
            if self.config_path:
                yaml, config = self.read_settings()
                config.setdefault("deployment", {})["mode"] = mode
                self.write_settings(yaml, config)
            self.mode = mode

    def read_settings(self):
        from ruamel.yaml import YAML

        yaml = YAML()
        yaml.preserve_quotes = True
        return yaml, yaml.load(self.config_path)

    def write_settings(self, yaml, settings):
        with atomic_output(self.config_path) as tmp:
            yaml.dump(settings, tmp)
            tmp.chmod(self.config_path.stat().st_mode)

    def validate_settings_change(self, mode, unloading=()):
        if any(j["status"] in {"queued", "running"} for j in self.jobs.values()):
            raise ValueError("请等待当前任务和排队操作完成后保存模型设置")
        if mode == "on_demand" and len(self.sessions.keys() - set(unloading)) > 1:
            raise ValueError("请先用 CLI 停止多余模型，仅保留一个模型后切换")

    def set_deployment(self, mode, changes):
        with self.lock:
            if not self.config_path:
                raise ValueError("后台缺少 YAML 配置路径")
            if changes.keys() - self.models.keys():
                raise ValueError("未知模型")
            yaml, settings = self.read_settings()
            deployment = settings.setdefault("deployment", {})
            deployment["mode"] = mode
            options = deployment.setdefault("models", {})
            for alias, change in changes.items():
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
            self.validate_settings_change(mode, changed)
            self.write_settings(yaml, settings)
            for alias in changed:
                self.unload(alias)
                spec = config["models"][alias]
                self.config["models"][alias] = spec
                self.models[alias].update(self.card(spec), detail="")
            self.config["capture"] = config["capture"]
            apply_model_selection(self.config, self.policy)
            self.mode = mode

    def unload(self, alias):
        with self.lock:
            session = self.sessions.pop(alias, None)
            if session is None:
                return
            self.models[alias]["status"] = "stopping"
        session[0].close()
        if self.journal_path:
            record(self.journal_path, "UPDATE requests SET status=0 WHERE status IS NULL AND source IN (?,?)",
                   (alias, alias + "/engine"))
        self.update(self.models[alias], status="offline", detail="")

    def unload_all(self):
        for alias in list(self.sessions):
            self.unload(alias)

    def is_ready(self, alias):
        session = self.sessions.get(alias)
        return session and all(p.poll() is None for p in session[1]) and all(
            healthy(s, path) for s, path in endpoints(self.config["models"][alias]))

    def load(self, alias, output):
        if not self.models[alias]["enabled"]:
            raise ValueError(f"{alias} 已在模型设置中关闭")
        if self.is_ready(alias):
            return
        if self.mode == "on_demand":
            self.unload_all()
        else:
            self.unload(alias)
        self.update(self.models[alias], status="starting", detail=str(output / (alias + ".log")))
        stack = ExitStack()
        try:
            processes = stack.enter_context(model_session(self.config, alias, output, self.closing.is_set))
        except Exception as error:
            self.update(self.models[alias], status="failed", detail=str(error))
            raise
        with self.lock:
            self.sessions[alias] = (stack, processes)
            self.models[alias]["status"] = "healthy"

    def monitor_health(self):
        while not self.closing.wait(2):
            with self.lock:
                sessions = list(self.sessions.items())
            for alias, session in sessions:
                ok = self.is_ready(alias)
                with self.lock:
                    if self.sessions.get(alias) is session:
                        self.models[alias]["status"] = "healthy" if ok else "unhealthy"

    def infer(self, job):
        if self.mode == "managed":
            missing = [alias for alias in mining_models(self.config["mining"]) if not self.is_ready(alias)]
            if missing:
                raise ValueError("CLI 常驻模式需要先启动这些模型：" + ", ".join(missing))
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        output = self.root / stamp
        output.mkdir(parents=True)
        self.update(job, output=str(output), stage="导入页面")
        pages = render_pages(Path(job["value"]), output / "input", self.settings)
        config = deepcopy(self.config)
        resolve_config(config, self.base.parent)

        @contextmanager
        def use_model(alias, directory):
            if self.closing.is_set():
                raise RuntimeError("后台正在关闭")
            self.update(job, stage=f"加载 {alias}")
            self.load(alias, directory)
            self.update(job, stage={"paddle": "版面与基线识别", "oxr": "OXR 转录", "ovis": "Ovis 转录",
                                    "jev": "Jev 评分与难例筛选"}[alias])
            yield

        jobs = [(output / "capture/regions.jsonl", output / "production")]
        if self.mode == "managed":
            policy = config["mining"]

            def stage(alias):
                if self.closing.is_set():
                    raise RuntimeError("后台正在关闭")
                run_local(config, output, jobs, pages, stage=alias, session=lambda *_: nullcontext())

            self.update(job, stage="版面与基线识别")
            stage(policy["target"])
            if policy["teachers"]:
                self.update(job, stage="教师模型转录")
                with ThreadPoolExecutor(max_workers=len(policy["teachers"])) as pool:
                    list(pool.map(stage, policy["teachers"]))
            self.update(job, stage="Jev 评分与难例筛选" if policy["judge"] else "难例筛选")
            stage(policy["judge"] or "mine")
            return
        try:
            run_local(config, output, jobs, pages,
                      session=use_model)
        finally:
            self.unload_all()

    def work(self):
        try:
            while (job := self.queue.get()) is not None:
                if self.closing.is_set():
                    self.update(job, status="failed", detail="后台已关闭")
                    continue
                self.update(job, status="running", stage=job["action"])
                try:
                    if job["action"] == "run":
                        self.infer(job)
                    elif job["action"] in {"start", "restart"}:
                        directory = self.root / ".manager"
                        directory.mkdir(parents=True, exist_ok=True)
                        aliases = mining_models(self.config["mining"]) if job["value"] == "all" else [job["value"]]
                        if job["action"] == "restart":
                            if job["value"] == "all":
                                aliases = list(self.sessions)
                            self.prepare_restart(aliases)
                        for alias in aliases:
                            if not self.models[alias]["enabled"]:
                                continue
                            self.update(job, stage=f"加载 {alias}")
                            self.load(alias, directory)
                    elif job["value"] == "all":
                        self.unload_all()
                    else:
                        self.unload(job["value"])
                    self.update(job, status="done", stage="完成")
                except Exception as error:
                    self.update(job, status="failed", detail=str(error))
        finally:
            self.unload_all()


class Command(BaseModel):
    action: str
    value: str | None = None


class ModelSettings(BaseModel):
    enabled: StrictBool
    path: str
    gpu: str


class DeploymentSettings(BaseModel):
    mode: Literal["on_demand", "managed"]
    models: dict[str, ModelSettings]


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
            manager.set_deployment(settings.mode, settings.models)
        except (ValueError, OSError) as error:
            raise HTTPException(409, str(error)) from error
        return {"mode": manager.mode}

    @app.post("/jobs", status_code=202)
    def submit(command: Command):
        try:
            return {"id": manager.enqueue(command.action, command.value)}
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    return app
