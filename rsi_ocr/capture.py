import base64
import json
from collections import defaultdict
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread

from .config import expand
from .service import Service, ServiceError, chat_result
from .store import atomic_output, digest, json_text, read_rows, write_rows


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
    def __init__(self, target, root):
        self.service = Service(target)
        self.root = root
        self.by_prompt = {spec["prompt"]: (task, spec["format"]) for task, spec in target["tasks"].items()}
        self.lock = Lock()
        self.rows = []
        self.page = None

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
        try:
            result = self.service.request(resource, payload)
        except ServiceError as error:
            if row is not None:
                row["target"].update(status="service_error", detail=str(error))
            return 502, {"error": {"message": str(error)}}
        if row is not None:
            row["target"].update(chat_result(result))
        return 200, result

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


def capture_pages(config, manifest, root):
    from paddleocr import PaddleOCRVL

    root.mkdir(parents=True, exist_ok=True)
    target = config["models"][config["mining"]["target"]]
    recorder = Recorder(target, root)
    options = config["capture"]

    def capture_key(page, image):
        return digest(json_text([page, str(image), digest(image.read_bytes()), options,
                                 expand(target["base_url"]), target["model"], target["revision"],
                                 target["tasks"], "layout-bbox-v1"]).encode())

    def rows():
        with recorder.listen() as url:
            pipeline = PaddleOCRVL(
                **options.get("pipeline", {}), vl_rec_backend="vllm-server", vl_rec_server_url=url,
                vl_rec_api_model_name=target["model"], vl_rec_api_key="capture-local",
            )
            for page in read_rows(manifest):
                image = (manifest.parent / page["image"]).resolve()
                if image.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                    raise ValueError(f"capture requires a single page as PNG or JPEG: {image}")
                shard = root / "capture_cache" / (capture_key(page, image) + ".jsonl")
                if shard.exists():
                    yield from read_rows(shard)
                    continue
                recorder.page = {
                    "doc_id": page["doc_id"], "page_id": page["page_id"],
                    "split": page.get("split", "mine"), "page_image": str(image),
                }
                if "eval_partition" in page:
                    recorder.page["eval_partition"] = page["eval_partition"]
                recorder.rows = []
                for result in pipeline.predict_iter(str(image), **options.get("predict", {})):
                    boxes = result.get("layout_det_res", {}).get("boxes", [])
                    if boxes:
                        locate_crops(image, boxes, root, recorder.rows)
                if all(row["target"]["status"] != "service_error" for row in recorder.rows):
                    write_rows(shard, recorder.rows)
                yield from recorder.rows

    output = root / "regions.jsonl"
    write_rows(output, rows())
    return output
