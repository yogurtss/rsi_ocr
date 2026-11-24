"""Observe request traffic and vLLM engine load for managed services."""
import asyncio
import logging
import os
import re
import sqlite3
import time
from contextlib import closing
from http.client import HTTPException
from math import isfinite
from urllib.parse import urlsplit
from urllib.request import urlopen
from uuid import uuid4


def model_load(base_url):
    url = urlsplit(base_url)._replace(path="/metrics", query="", fragment="").geturl()
    try:
        with urlopen(url, timeout=1) as response:
            metrics = response.read().decode("utf-8", errors="replace")
    except (OSError, HTTPException):
        return {}

    values = {}
    names = ("num_requests_running", "num_requests_waiting", "kv_cache_usage_perc", "gpu_cache_usage_perc")
    pattern = rf"^vllm:({'|'.join(names)})(?:\{{.*\}})?[ \t]+(\S+)"
    for name, value in re.findall(pattern, metrics, re.MULTILINE):
        try:
            number = float(value)
        except ValueError:
            continue
        if isfinite(number) and number >= 0:
            values.setdefault(name, []).append(number)
    load = {key: sum(int(number) for number in values[f"num_requests_{key}"])
            for key in ("running", "waiting") if f"num_requests_{key}" in values}
    cache = values.get("kv_cache_usage_perc", values.get("gpu_cache_usage_perc", []))
    if cache and max(cache) <= 1:
        load["kv_usage"] = max(cache)
    return load


def open_journal(path):
    with closing(sqlite3.connect(path)) as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("""CREATE TABLE requests (
            id TEXT PRIMARY KEY, source TEXT, path TEXT, started REAL,
            status INTEGER, elapsed_ms INTEGER)""")


def record(path, query, args):
    try:
        with closing(sqlite3.connect(path, timeout=1)) as db, db:
            db.execute(query, args)
            db.execute("""DELETE FROM requests WHERE rowid IN (
                SELECT rowid FROM requests WHERE status IS NOT NULL ORDER BY rowid DESC LIMIT -1 OFFSET 128)""")
    except sqlite3.Error:
        logging.getLogger(__name__).warning("Request journal unavailable", exc_info=True)


def recent(path):
    if not path:
        return []
    with closing(sqlite3.connect(path)) as db:
        db.row_factory = sqlite3.Row
        return [dict(row) for row in db.execute(
            "SELECT source,path,started,status,elapsed_ms FROM requests ORDER BY rowid DESC LIMIT 128")]


class RequestMeter:
    def __init__(self, app, source=None, journal=None):
        self.app = app
        self.source = source or os.environ.get("GLYPHSTACK_SOURCE")
        self.journal = journal or (lambda: os.environ.get("GLYPHSTACK_JOURNAL"))

    async def __call__(self, scope, receive, send):
        path = self.journal()
        if scope["type"] != "http" or scope["method"] != "POST" or not path:
            return await self.app(scope, receive, send)
        key = uuid4().hex
        start = time.monotonic()
        await asyncio.to_thread(record, path, "INSERT INTO requests VALUES (?,?,?,?,NULL,NULL)",
                                (key, self.source, scope["path"][:180], time.time()))
        status = 500

        async def observe(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, observe)
        finally:
            elapsed = round((time.monotonic() - start) * 1000)
            await asyncio.to_thread(record, path, "UPDATE requests SET status=?, elapsed_ms=? WHERE id=?",
                                    (status, elapsed, key))


def render_app():
    # Uvicorn's --app-dir points at the official Jev SDK.
    from app import app
    return RequestMeter(app)
