import base64
import json
import os
import traceback
from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from time import monotonic
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import ExitStack, closing, contextmanager, nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from multiprocessing import get_context
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock, Thread

from .config import expand
from .importer import InputChangedError, RenderPlan
from .service import Service, ServiceError, chat_result
from .store import atomic_output, digest, json_text, previous_batch, read_rows, write_rows


def locate_crops(image, boxes, root, rows):
    """Keep only unique pixel matches to a rectangle on the original page."""
    from PIL import Image

    def pixels(rgb):
        return rgb.size, digest(rgb.tobytes())

    positions = defaultdict(set)
    with Image.open(image) as source:
        page = source.convert("RGB")
    for item in boxes:
        if "coordinate" not in item or len(item["coordinate"]) != 4:
            continue
        left, top, right, bottom = (int(value) for value in item["coordinate"])
        box = max(0, left), max(0, top), min(page.width, right), min(page.height, bottom)
        if box[0] < box[2] and box[1] < box[3]:
            positions[pixels(page.crop(box))].add(box)
    for row in rows:
        with Image.open(root / row["image"]) as crop:
            matches = positions.get(pixels(crop.convert("RGB")), ())
        if len(matches) == 1:
            row["bbox"], = matches
            row["bbox_source"] = "exact_paddle_layout_crop"


class Recorder:
    def __init__(self, target, root, service=None):
        self.service = service if service is not None else Service(target)
        self.root = root
        self.by_prompt = {spec["prompt"]: (task, spec["format"]) for task, spec in target["tasks"].items()}
        self.lock = Lock()
        self.rows = []
        self.page = None
        self.request_times = []

    def reserve(self, payload):
        message = next(item for item in payload["messages"] if item["role"] == "user")
        content = message["content"]
        prompt = next(part["text"] for part in content if part["type"] == "text").strip()
        if prompt not in self.by_prompt:
            return None
        url = next(part["image_url"]["url"] for part in content if part["type"] == "image_url")
        header, encoded = url.split(",", 1)
        extension = {"data:image/png;base64": ".png", "data:image/jpeg;base64": ".jpg"}[header]
        image = base64.b64decode(encoded)
        image_hash = digest(image)
        path = self.root / "crops" / (image_hash + extension)
        task, output_format = self.by_prompt[prompt]
        with self.lock:
            if not path.exists():
                with atomic_output(path) as temporary:
                    temporary.write_bytes(image)
            row = {
                **self.page, "region_id": str(len(self.rows)), "task": task,
                "image": path.relative_to(self.root).as_posix(), "image_sha256": image_hash,
                "origin": "paddle_vllm_request", "target": {
                    "prompt": prompt, "format": output_format, "model": payload["model"],
                    "revision": self.service.config["revision"],
                    "parameters": {key: value for key, value in payload.items() if key not in {"model", "messages"}},
                },
            }
            self.rows.append(row)
        return row

    def forward(self, resource, payload):
        row = self.reserve(payload) if resource == "/chat/completions" else None
        started = monotonic()
        try:
            result = self.service.request(resource, payload)
        except ServiceError as error:
            if row is not None:
                row["target"].update(status="service_error", detail=str(error))
            return 502, {"error": {"message": str(error)}}
        finally:
            if resource == "/chat/completions":
                with self.lock:
                    self.request_times.append((started, monotonic()))
        if row is not None:
            row["target"].update(chat_result(result))
        return 200, result

    def request_seconds(self):
        # Count overlapping client requests once.
        total = end = 0.0
        for start, stop in sorted(self.request_times):
            total += max(0.0, stop - max(start, end))
            end = max(end, stop)
        return total

    @contextmanager
    def listen(self):
        recorder = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.forward()

            def do_POST(self):
                self.forward()

            def forward(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"]))) if self.command == "POST" else None
                status, result = recorder.forward(self.path.removeprefix("/v1"), payload)
                body = json_text(result).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_):
                pass

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                yield f"http://127.0.0.1:{server.server_port}/v1"
            finally:
                server.shutdown()
                thread.join()


@contextmanager
def predictor_devices():
    """Observe the native device selected when Paddle creates each predictor."""
    from paddle import inference

    create_predictor = inference.create_predictor
    devices = {}

    def track_device(config):
        predictor = create_predictor(config)
        try:
            if config.use_gpu():
                devices[id(predictor)] = f"gpu:{config.gpu_device_id()}"
            elif not any(getattr(config, name, lambda: False)() for name in ("use_xpu", "use_custom_device")):
                devices[id(predictor)] = "cpu"
        except (AttributeError, RuntimeError, TypeError):
            pass
        return predictor

    inference.create_predictor = track_device
    try:
        yield devices
    finally:
        inference.create_predictor = create_predictor


@dataclass
class CapturePage:
    page: dict
    image: Path
    shard: Path | None
    cached: bool


def restore_page(source, destination, image):
    """Bring a completed page into the new batch, keeping it self-contained."""
    rows = list(read_rows(source))
    for row in rows:
        relative = Path(row["image"])
        if relative.parent != Path("crops") or relative.stem != row.get("image_sha256"):
            return False
        old_crop = source.parent.parent / relative
        new_crop = destination.parent.parent / relative
        if not new_crop.exists():
            if not old_crop.is_file():
                return False
            data = old_crop.read_bytes()
            if digest(data) != row["image_sha256"]:
                return False
            with atomic_output(new_crop) as temporary:
                temporary.write_bytes(data)
        row["page_image"] = str(image)
    write_rows(destination, rows)
    return True


def capture_plan(config, manifest, root, check_cancelled=None, *, records=None):
    render = manifest if isinstance(manifest, RenderPlan) else None
    if records is None:
        records = render.rows if render else read_rows(manifest)
    if render:
        manifest = render.manifest
    target = config["models"][config["mining"]["target"]]
    options = config["capture"]
    identity = [expand(target["base_url"]), target["model"], target["revision"], target["tasks"], "layout-bbox-v1"]
    previous = previous_batch(root.parent)

    def shard_paths(directory, page, page_image, image_hash):
        parts = [page, str(page_image), image_hash, options, *identity]
        new_key = digest(json_text([*parts, target.get("local_weights")]).encode())
        old_key = digest(json_text(parts).encode())
        cache = directory / "capture_cache"
        return cache / (new_key + ".jsonl"), cache / (old_key + ".jsonl")

    plan = []
    for page in records:
        if check_cancelled:
            check_cancelled()
        image = (manifest.parent / page["image"]).resolve()
        if image.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            raise ValueError(f"capture requires a single page as PNG or JPEG: {image}")
        existing = render.existing_image(page) if render else image
        if existing is None:
            plan.append(CapturePage(page, image, None, False))
            continue
        image_hash = digest(existing.read_bytes())
        shard, legacy = shard_paths(root, page, image, image_hash)
        if legacy.is_file() and not shard.is_file():
            shard = legacy
        cached = shard.is_file()
        if not cached and previous and image.is_relative_to(root.parent):
            old_image = previous / image.relative_to(root.parent)
            for old in shard_paths(previous / "capture", page, old_image, image_hash):
                if old.is_file() and restore_page(old, shard, image):
                    cached = True
                    break
        plan.append(CapturePage(page, image, shard, cached))
    return plan


@lru_cache(maxsize=1)
def paddlex_version():
    try:
        return version("paddlex")
    except PackageNotFoundError:
        return None


@contextmanager
def layout_checkpoint(detector, shard, report, activity=None):
    """Save single-page PaddleX layout boxes before the downstream VLM call."""
    process = getattr(detector, "process", None)
    if process is None:
        yield
        return
    path = shard.parent.parent / "layout_cache" / (shard.stem + ".json")

    def run_detector(batch, *args, **kwargs):
        if activity:
            activity(1)
        try:
            return process(batch, *args, **kwargs)
        finally:
            if activity:
                activity(0)

    def checkpointed(batch, *args, **kwargs):
        # The single-image pipeline supplies the original BGR array here.
        if len(batch.instances) != 1 or not hasattr(batch.instances[0], "shape"):
            return run_detector(batch, *args, **kwargs)
        if path.is_file():
            saved = json.loads(path.read_bytes())
            if saved["paddlex"] == paddlex_version():
                report(True, 0.0)
                return {"boxes": saved["boxes"], "input_img": batch.instances,
                        "input_path": batch.input_paths, "page_index": batch.page_indexes}
        start = monotonic()
        result = run_detector(batch, *args, **kwargs)
        seconds = monotonic() - start
        if set(result) == {"boxes", "input_img", "input_path", "page_index"}:
            with atomic_output(path) as temporary:
                temporary.write_text(json.dumps({"paddlex": paddlex_version(), "boxes": result["boxes"]},
                                                default=lambda value: value.tolist()), encoding="utf-8")
            report(False, seconds)
        return result

    detector.process = checkpointed
    try:
        yield
    finally:
        detector.process = process


@contextmanager
def rendered_entries(config, source, root, plan, progress, check_cancelled, report_work):
    if not isinstance(source, RenderPlan):
        yield iter(plan)
        return

    def render():
        with closing(source.images(progress, check_cancelled)) as images:
            for entry in plan:
                row = next(images)
                if report_work:
                    report_work("render", row["doc_id"])
                if entry.shard is None:
                    entry, = capture_plan(config, source.manifest, root, records=[row])
                yield entry
            # Exhaust the renderer to publish the complete page manifest.
            next(images, None)
            if report_work:
                report_work("render", finished=True)

    # Keep PDFium on one thread, including releasing its document on early exit.
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="pdf-render") as pool:
        entries = render()
        ahead = pool.submit(next, entries, None)

        def consume():
            nonlocal ahead
            while (entry := ahead.result()) is not None:
                ahead = pool.submit(next, entries, None)
                yield entry

        try:
            yield consume()
        finally:
            ahead.cancel()
            pool.submit(entries.close).result()


def _capture_gpu_worker(config, manifest, root, messages, plan=None):
    with messages, (root / "layout.log").open("a") as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        os.environ.setdefault("FLAGS_allocator_strategy", "auto_growth")
        lock = Lock()

        def send(kind, value):
            with lock:
                messages.send((kind, value))

        try:
            capture_pages(config, manifest, root, progress=lambda text: send("progress", text),
                          report_device=lambda device: send("device", device), plan=plan,
                          report_page=lambda state: send("page", state),
                          report_work=lambda *args, **kwargs: send("work", (args, kwargs)))
        except InputChangedError as error:
            send("input_changed", str(error))
        except Exception as error:
            traceback.print_exc(file=log)
            messages.send(("error", str(error)))


def capture_gpu_pages(config, manifest, root, progress=None, check_cancelled=None, report_device=None,
                      plan=None, report_page=None, report_work=None):
    """Release Paddle's CUDA context before handing the GPU to the next model."""
    root.mkdir(parents=True, exist_ok=True)
    context = get_context("spawn")
    reader, writer = context.Pipe(duplex=False)
    with reader, writer:
        worker = context.Process(target=_capture_gpu_worker, args=(config, manifest, root, writer, plan))
        worker.start()
        writer.close()
        try:
            while True:
                if check_cancelled:
                    check_cancelled()
                if not reader.poll(0.2):
                    continue
                try:
                    kind, message = reader.recv()
                except EOFError:
                    break
                if kind == "input_changed":
                    raise InputChangedError(message)
                if kind == "error":
                    raise RuntimeError(f"{message} · 日志 {root / 'layout.log'}")
                if kind == "device":
                    if report_device:
                        report_device(message)
                elif kind == "page":
                    if report_page:
                        report_page(message)
                elif kind == "work":
                    if report_work:
                        args, kwargs = message
                        report_work(*args, **kwargs)
                elif progress:
                    progress(message)
            while worker.is_alive():
                if check_cancelled:
                    check_cancelled()
                worker.join(timeout=0.2)
            if worker.exitcode != 0:
                raise RuntimeError(f"GPU 版面检测退出（{worker.exitcode}）· 日志 {root / 'layout.log'}")
        finally:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=2)
                if worker.is_alive():
                    worker.kill()
            worker.join()
            worker.close()


@contextmanager
def prepared_layout(inner, source, result):
    """Replay one native detection only for the same unmodified page array."""
    original = getattr(inner, "layout_det_model", None)

    def detect(images, **kwargs):
        if len(images) != 1 or images[0] is not source:
            raise RuntimeError("预取的版面与识别页不一致")
        return result

    inner.layout_det_model = detect
    try:
        yield
    finally:
        inner.layout_det_model = original


def capture_pages(config, manifest, root, progress=None, report_device=None, plan=None, report_page=None,
                  check_cancelled=None, report_work=None):
    root.mkdir(parents=True, exist_ok=True)
    target = config["models"][config["mining"]["target"]]
    upstream = Service(target)
    options = config["capture"]

    def create_client(resources, *, recognition_only=False):
        recorder = Recorder(target, root, upstream)
        url = resources.enter_context(recorder.listen())
        pipeline_options = {"vl_rec_max_concurrency": target.get("concurrency", 4), **options.get("pipeline", {})}
        if recognition_only:
            pipeline_options.update(use_layout_detection=False, use_doc_orientation_classify=False,
                                    use_doc_unwarping=False, device="cpu")
        device = pipeline_options.get("device", "cpu")
        if str(device).partition(":")[0] == "gpu":
            import paddle

            if not paddle.is_compiled_with_cuda():
                raise ValueError("GPU 版面检测需要支持 CUDA 的 paddlepaddle-gpu，请检查 Console 的 Python 环境")

        from paddleocr import PaddleOCRVL

        pipeline = PaddleOCRVL(
            **pipeline_options, vl_rec_backend="vllm-server", vl_rec_server_url=url,
            vl_rec_api_model_name=target["model"], vl_rec_api_key="capture-local",
        )
        if callable(getattr(pipeline, "close", None)):
            resources.callback(pipeline.close)
        px = getattr(pipeline, "paddlex_pipeline", None)
        return pipeline, getattr(px, "_pipeline", px), recorder

    if plan is None:
        plan = capture_plan(config, manifest, root, check_cancelled)
        if progress:
            progress(f"版面与 OCR · 复用 {sum(entry.cached for entry in plan)}/{len(plan)} 页")

    reused = sum(entry.cached for entry in plan)
    done = 0
    state_lock = Lock()
    active = {}
    recognizing = set()
    mode = "逐页"

    def update(entry, phase, **values):
        nonlocal done, reused
        with state_lock:
            current = f"{entry.page['doc_id']} · 第 {entry.page['page_id']} 页"
            if "layout_running" in values:
                active["layout_running"] = values.pop("layout_running")
            if phase == "复用整页":
                reused += 1
            elif phase == "准备版面":
                active["layout_page"] = current
            elif phase == "区域识别":
                recognizing.add(current)
                if active.get("layout_page") == current:
                    active.pop("layout_page")
            elif phase == "整页已保存":
                done += 1
            if phase in ("整页已保存", "区域服务错误"):
                recognizing.discard(current)
            if report_page:
                report_page({"current": current, "phase": phase, "done": done, "total": len(plan),
                             "reused": reused, "mode": mode, "ocr_pages": len(recognizing),
                             "ocr_page": next(iter(recognizing), None), **active, **values})

    def checkpoint(entry, detector, times):
        def layout_ready(cached, seconds):
            times.update(layout_reused=cached, layout_seconds=round(seconds, 2))
            if progress:
                label = "复用本页版面断点" if cached else f"版面断点已保存 · 检测 {seconds:.2f} 秒"
                progress(f"{label} · {entry.page['doc_id']} 第 {entry.page['page_id']} 页")

        def activity(running):
            update(entry, "版面检测" if running else "版面检测结束", layout_running=running)

        return layout_checkpoint(detector, entry.shard, layout_ready, activity if report_page else None)

    def recognize(client, entry, source, detector, prepared, times, started, scratch):
        pipeline, inner, recorder = client
        page, shard = entry.page, entry.shard
        recorder.page = {k: page[k] for k in ("doc_id", "page_id", "eval_partition") if k in page}
        recorder.page.update(split=page.get("split", "mine"), page_image=str(entry.image))
        recorder.rows = []
        recorder.request_times = []
        update(entry, "区域识别")

        predict = dict(options.get("predict", {}))
        if prepared is None:
            context = checkpoint(entry, detector, times)
        else:
            context = prepared_layout(inner, source, prepared)
            predict["use_layout_detection"] = True
            predict["use_queues"] = False

        received_page = False
        with context:
            for result in pipeline.predict_iter(source, **predict):
                if "error" in result:
                    raise RuntimeError(str(result["error"]))
                received_page = True
                boxes = result.get("layout_det_res", {}).get("boxes", [])
                if boxes:
                    locate_crops(entry.image, boxes, root, recorder.rows)
        if not received_page:
            raise RuntimeError(f"PaddleOCR 未返回页面结果 · {entry.image}")
        failed = any(row["target"]["status"] == "service_error" for row in recorder.rows)
        if not failed:
            write_rows(shard, recorder.rows)
            seconds = monotonic() - started
            request_seconds = recorder.request_seconds()
            update(entry, "整页已保存", seconds=round(seconds, 2), request_seconds=round(request_seconds, 2), **times)
            if progress:
                progress(f"整页断点已保存 · {page['doc_id']} 第 {page['page_id']} 页 · {len(recorder.rows)} 个区域 · "
                         f"单页 {seconds:.1f} 秒 · vLLM 请求 {request_seconds:.1f} 秒")
        else:
            # Include failures in this run's manifest without marking the page resumable.
            shard = scratch / shard.name
            write_rows(shard, recorder.rows)
            update(entry, "区域服务错误")
            if progress:
                progress(f"区域服务错误 · {entry.image} · 下次重试 OCR")
        if report_work:
            report_work("paddle", page["doc_id"], failed=failed)
        return shard

    def rows():
        nonlocal mode
        page_counts = Counter(entry.page["doc_id"] for entry in plan)
        file_numbers = {doc_id: index for index, doc_id in enumerate(page_counts, 1)}
        main_client = None
        pending = {}
        free_clients = []
        shards = []
        predict = options.get("predict", {})

        def collect(finished):
            for future in finished:
                index, client = pending.pop(future)
                shards[index] = future.result()
                free_clients.append(client)

        with ExitStack() as stack:
            scratch = Path(stack.enter_context(TemporaryDirectory(dir=root, prefix=".capture-")))
            resources = stack.enter_context(ExitStack())
            entries = stack.enter_context(rendered_entries(config, manifest, root, plan, progress, check_cancelled, report_work))
            for index, entry in enumerate(entries):
                if check_cancelled:
                    check_cancelled()
                collect([future for future in pending if future.done()])
                shards.append(entry.shard)
                if entry.cached:
                    if not plan[index].cached:
                        update(entry, "复用整页")
                        if report_work:
                            report_work("paddle", entry.page["doc_id"])
                    continue
                if main_client is None:
                    if progress:
                        progress("初始化 PaddleOCR 版面识别")
                    observe = report_device and predict.get("use_layout_detection", True)
                    with (predictor_devices() if observe else nullcontext()) as devices:
                        main_client = create_client(resources)
                    pipeline, inner, _ = main_client
                    px = getattr(pipeline, "paddlex_pipeline", None)
                    detector = getattr(inner, "layout_det_model", None)
                    if observe:
                        runner = getattr(detector, "runner", None)
                        report_device(devices.get(id(getattr(runner, "predictor", None))))
                    # Precomputed layout is valid only for unmodified page pixels.
                    fast = (callable(detector) and callable(getattr(inner, "img_reader", None))
                            and not getattr(px, "multi_device_inference", False)
                            and predict.get("use_layout_detection", True)
                            and not getattr(inner, "use_doc_preprocessor", False)
                            and not predict.get("use_doc_orientation_classify") and not predict.get("use_doc_unwarping"))
                    if fast:
                        page_window = min(target.get("page_window", 4), target.get("concurrency", 4))
                        if getattr(inner, "vl_rec_model", None) is None:
                            page_window = 1
                        mode = f"跨页 OCR 队列 · 最多 {page_window} 页" if page_window > 1 else "双页流水线"
                        worker = stack.enter_context(ThreadPoolExecutor(max_workers=page_window, thread_name_prefix="paddle-ocr"))
                        free_clients.append(main_client)
                    if progress:
                        progress(f"Paddle · {mode}")
                page = entry.page
                if progress:
                    doc = page["doc_id"]
                    progress(f"识别 · 第 {page['page_id']}/{page_counts[doc]} 页 · "
                             f"文件 {file_numbers[doc]}/{len(file_numbers)} · {doc}")
                started = monotonic()
                times = {}
                if not fast:
                    shards[index] = recognize(main_client, entry, str(entry.image), detector, None, times, started, scratch)
                    continue
                update(entry, "准备版面")
                image, = inner.img_reader([str(entry.image)])

                with checkpoint(entry, detector, times):
                    layout = list(detector([image], threshold=predict.get("layout_threshold"),
                                           layout_nms=predict.get("layout_nms"),
                                           layout_unclip_ratio=predict.get("layout_unclip_ratio"),
                                           layout_merge_bboxes_mode=predict.get("layout_merge_bboxes_mode"),
                                           layout_shape_mode=predict.get("layout_shape_mode", "auto"),
                                           filter_overlap_boxes=False))
                if len(layout) != 1:
                    raise RuntimeError(f"版面检测未返回单页结果 · {entry.image}")
                if len(pending) == page_window:
                    collect(wait(pending, return_when=FIRST_COMPLETED).done)
                else:
                    collect([future for future in pending if future.done()])
                client = free_clients.pop() if free_clients else create_client(resources, recognition_only=True)
                future = worker.submit(recognize, client, entry, image, detector, layout, times, started, scratch)
                pending[future] = index, client
            collect(list(pending))
            for shard in shards:
                if check_cancelled:
                    check_cancelled()
                yield from read_rows(shard)

    output = root / "regions.jsonl"
    write_rows(output, rows())
    return output
