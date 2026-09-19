from __future__ import annotations

import json
import threading
import time
from http.client import HTTPConnection

from mnemonics import server as srv


def _post(port: int, path: str, body: dict) -> tuple[int, str]:
    conn = HTTPConnection("127.0.0.1", port, timeout=2)
    try:
        payload = json.dumps(body)
        conn.request(
            "POST",
            path,
            body=payload,
            headers={"content-type": "application/json"},
        )
        response = conn.getresponse()
        return response.status, response.read().decode()
    finally:
        conn.close()


def _get(port: int, path: str) -> tuple[int, str]:
    conn = HTTPConnection("127.0.0.1", port, timeout=2)
    try:
        conn.request("GET", path)
        response = conn.getresponse()
        return response.status, response.read().decode()
    finally:
        conn.close()


def test_warmup_does_not_block_health_requests(monkeypatch):
    warm_started = threading.Event()
    release_warm = threading.Event()

    def slow_warm(ns: str):
        warm_started.set()
        assert release_warm.wait(timeout=2)
        return {"status": "ready", "ns": ns}

    monkeypatch.setattr(srv, "_warm_store", slow_warm)
    server = srv.ThreadingHTTPServer(("127.0.0.1", 0), srv._Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = int(server.server_address[1])

    warm_result: list[tuple[int, str]] = []
    warm_thread = threading.Thread(
        target=lambda: warm_result.append(_post(port, "/warmup", {"ns": "sessions"})),
        daemon=True,
    )
    warm_thread.start()
    assert warm_started.wait(timeout=1)

    started = time.perf_counter()
    status, body = _get(port, "/health")
    elapsed = time.perf_counter() - started

    release_warm.set()
    warm_thread.join(timeout=2)
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)

    assert status == 200
    assert json.loads(body)["status"] == "ok"
    assert elapsed < 0.2
    assert warm_result and warm_result[0][0] == 200
