import base64
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import BoundedSemaphore
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import expand
from .store import digest, json_text


def image_data(image):
    mime = "image/png" if image.startswith(b"\x89PNG\r\n\x1a\n") else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(image).decode('ascii')}"


class ServiceError(Exception):
    pass


class Service:
    def __init__(self, config, cache=None):
        self.config = config
        self.cache = cache
        self.slots = BoundedSemaphore(config.get("concurrency", 4))

    def request(self, path, payload=None):
        url = expand(self.config["base_url"]).rstrip("/") + path
        headers = {"Content-Type": "application/json"}
        if env := self.config.get("api_key_env"):
            headers["Authorization"] = f"Bearer {os.environ[env]}"
        data = json_text(payload).encode() if payload is not None else None
        request = Request(url, data, headers=headers)
        try:
            with self.slots, urlopen(request, timeout=self.config.get("timeout", 120)) as response:
                return json.load(response)
        except HTTPError as error:
            raise ServiceError(f"HTTP {error.code} from {url}") from error
        except (URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as error:
            raise ServiceError(f"{url}: {error}") from error

    def cached(self, path, payload, convert):
        key = digest(json_text([
            expand(self.config["base_url"]), self.config["model"],
            self.config["revision"], path, payload,
        ]).encode())
        hit = self.cache.get(key) if self.cache else None
        if hit is not None:
            return hit
        start = time.perf_counter()
        try:
            result = convert(self.request(path, payload))
        except (ServiceError, KeyError, IndexError, TypeError, ValueError) as error:
            return {"status": "service_error", "detail": str(error)}
        result["latency_s"] = round(time.perf_counter() - start, 4)
        transient = {"service_error", "forward_error", "render_error", "encode_error"}
        if self.cache and result["status"] not in transient:
            self.cache.put(key, result)
        return result

    def read(self, task, image):
        spec = self.config["tasks"][task]
        payload = {
            "model": self.config["model"],
            "temperature": 0,
            "max_tokens": spec.get("max_tokens", 4096),
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_data(image)}},
                {"type": "text", "text": spec["prompt"]},
            ]}],
            **self.config.get("request", {}),
            **spec.get("request", {}),
        }
        result = self.cached("/chat/completions", payload, chat_result)
        return {**result, "prompt": spec["prompt"], "format": spec["format"]}

    def score(self, task, image, candidates):
        payload = {
            "input_form": "ternary",
            "items": [
                {"case_id": str(index), "image": base64.b64encode(image).decode("ascii"),
                 "category": task, "ocr_text": text}
                for index, text in enumerate(candidates)
            ],
        }

        def convert(response):
            results = response["results"]
            if len(results) != len(candidates):
                raise ValueError("Jev returned a different number of items")
            entries = []
            for row in results:
                verdict = row["verdict"]
                probability = float(row["prob_yes"]) if verdict == "scored" else None
                if probability is not None and not 0 <= probability <= 1:
                    raise ValueError("Jev prob_yes outside [0, 1]")
                entries.append({"status": verdict, "prob_yes": probability, "detail": row.get("detail")})
            failures = {"forward_error", "render_error", "encode_error"}
            status = next((row["status"] for row in entries if row["status"] in failures), "ok")
            return {"status": status, "entries": entries}

        result = self.cached("/v1/rm/score", payload, convert)
        if "entries" not in result:
            return [result.copy() for _ in candidates]
        return result["entries"]


def chat_result(response):
    choice = response["choices"][0]
    content = choice["message"].get("content")
    finish = choice["finish_reason"]
    status = "ok" if finish == "stop" and isinstance(content, str) else "incomplete"
    return {"status": status, "raw": content, "finish_reason": finish, "usage": response.get("usage")}


def command(config, alias, service_name=None):
    spec = config["models"][alias]
    service = spec["launch"] if service_name is None else spec["sidecars"][service_name]
    env = {**os.environ, **{key: expand(val) for key, val in service.get("env", {}).items()}}
    args = [expand(arg) for arg in service["argv"]]
    cwd = Path(expand(service["cwd"])) if "cwd" in service else None
    return args, env, cwd


def serve(config, alias, sidecar, show):
    args, env, cwd = command(config, alias, sidecar)
    if show:
        print(json_text({"argv": args, "cwd": str(cwd) if cwd else None}))
        return 0
    return subprocess.call(args, env=env, cwd=cwd)


def probe(models, aliases):
    def check(alias):
        spec = models[alias]
        jev = spec["kind"] == "jev"
        endpoint = "/healthz" if jev else "/models"
        try:
            result = Service(spec).request(endpoint)
            ready = result.get("status") == "ok" if jev else spec["model"] in {
                row["id"] for row in result["data"]
            }
            return {"alias": alias, "ready": ready, "response": result}
        except (ServiceError, KeyError) as error:
            return {"alias": alias, "ready": False, "detail": str(error)}

    with ThreadPoolExecutor() as pool:
        results = list(pool.map(check, aliases))
    for result in results:
        print(json_text(result))
    return 0 if all(result["ready"] for result in results) else 1
