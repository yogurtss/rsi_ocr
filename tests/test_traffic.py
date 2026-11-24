import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from rsi_ocr.traffic import RequestMeter, model_load, open_journal, recent


class TrafficTests(unittest.IsolatedAsyncioTestCase):
    def test_model_load_parses_engine_counts_and_kv_usage(self):
        metrics = b'''
vllm:num_requests_running{engine="0",model_name="paddle"} 2.0
vllm:num_requests_running{engine="1",model_name="paddle"} 1.0
vllm:num_requests_waiting{engine="0"} 3.0 12345
vllm:num_requests_waiting_by_reason{reason="capacity"} 3
vllm:kv_cache_usage_perc{engine="0"} 0.42
vllm:kv_cache_usage_perc{engine="1"} 0.27
'''
        response = MagicMock()
        response.__enter__.return_value.read.return_value = metrics
        with patch("rsi_ocr.traffic.urlopen", return_value=response) as read:
            self.assertEqual(model_load("http://127.0.0.1:8100/v1"), {"running": 3, "waiting": 3, "kv_usage": 0.42})
            read.assert_called_once_with("http://127.0.0.1:8100/metrics", timeout=1)
            response.__enter__.return_value.read.return_value = b"vllm:gpu_cache_usage_perc 0.5\n"
            self.assertEqual(model_load("http://127.0.0.1:8100"), {"kv_usage": 0.5})
            read.side_effect = TimeoutError
            self.assertEqual(model_load("http://127.0.0.1:8100"), {})

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
