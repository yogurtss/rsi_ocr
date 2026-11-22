import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rsi_ocr.traffic import RequestMeter, open_journal, recent


class TrafficTests(unittest.IsolatedAsyncioTestCase):
    async def test_live_and_completed_post_without_body_or_query(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def target(scope, receive, send):
            entered.set()
            await release.wait()
            await send({"type": "http.response.start", "status": 422})
            await send({"type": "http.response.body", "body": b"invalid"})

        async def discard(message):
            pass

        async def inline_write(fn, *args):
            return fn(*args)

        with tempfile.TemporaryDirectory() as directory, patch("rsi_ocr.traffic.asyncio.to_thread", inline_write):
            path = Path(directory) / "traffic.sqlite"
            open_journal(path)
            meter = RequestMeter(target, "paddle", lambda: path)
            scope = {"type": "http", "method": "POST", "path": "/v1/chat/completions", "query_string": b"unused"}
            task = asyncio.create_task(meter(scope, None, discard))
            await asyncio.wait_for(entered.wait(), 2)
            self.assertIsNone(recent(path)[0]["status"])
            release.set()
            await task
            row, = recent(path)
            self.assertEqual((row["status"], row["source"], row["path"]), (422, "paddle", scope["path"]))
            self.assertGreaterEqual(row["elapsed_ms"], 0)
            await meter({**scope, "method": "GET"}, None, discard)
            self.assertEqual(recent(path), [row])
