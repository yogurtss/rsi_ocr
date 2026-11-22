"""Observe POST metadata at the ASGI boundary of each managed service."""
import asyncio
import logging
import os
import sqlite3
import time
from contextlib import closing
from uuid import uuid4


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
