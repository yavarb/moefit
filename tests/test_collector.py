"""Tests: collect_silicon_run.one_run against a mock SSE server.

Regression for the T8 counterexamples (astra_local_exp 175d734):
  1. A slow usage TRAILER after the last text token must not deflate
     decode_tps (was: wall-EOF closed the decode interval, 5s trailer
     cut a 12.50 tok/s stream to 10.05).
  2. Control chunks (empty-delta role/finish chunks) must not enter
     the per-token gap samples.
  3. First control chunk is not TTFT.
"""
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.collect_silicon_run import one_run  # noqa: E402


def make_server(handler_cls):
    srv = HTTPServer(("127.0.0.1", 0), handler_cls)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/v1/chat/completions"


def sse(j):
    return b"data: " + json.dumps(j).encode() + b"\n\n"


class Handler(BaseHTTPRequestHandler):
    behavior = "trailer"

    def do_POST(self):
        n = 40
        gen_ms = 5.0
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        t0 = time.monotonic()
        # leading control chunk (empty delta): must not set TTFT
        self.wfile.write(sse(dict(choices=[dict(delta={}, finish_reason=None)])))
        for i in range(n):
            time.sleep(max(0.0, t0 + (i + 1) * gen_ms / 1000 - time.monotonic()))
            self.wfile.write(sse(dict(choices=[dict(
                delta=dict(content=f"t{i}"), finish_reason=None)])))
            # interleaved control chunk every 4th token
            if i % 4 == 3:
                self.wfile.write(sse(dict(choices=[dict(
                    delta={}, finish_reason=None)])))
        self.wfile.write(sse(dict(choices=[dict(
            delta={}, finish_reason="stop")])))
        if self.behavior == "trailer":
            time.sleep(1.0)   # the T8 counterexample: slow usage trailer
        self.wfile.write(sse(dict(usage=dict(
            completion_tokens=n, prompt_tokens=5))))
        self.wfile.write(b"data: \n\n")

    def log_message(self, format, *a):  # noqa: A002
        pass


def test_all():
    srv, url = make_server(Handler)
    try:
        # counterexample 1: 1s trailer must NOT deflate decode_tps
        r = one_run(url, "m", "hi", 64, timeout=30)
        expect_tps = 1000.0 / 5.0
        assert r["tokens"] == 40, r
        assert abs(r["decode_tps"] - expect_tps) < 12, r
        assert r["decode_s"] < 0.6, r          # closed at last token event
        assert r["wall_s"] >= 1.0              # trailer is in wall only
        # counterexample 2+3: control chunks out of gap samples, not TTFT
        assert len(r["per_token_ms"]) == 39, r  # n_text-1 only
        assert 0.0 <= r["ttft_s"] < 0.05, r
        assert r["n_control_chunks"] >= 1 + 10 + 1, r
        assert r["timing_provenance"].startswith("client SSE chunk")
        # per-token gaps: coalescing detector must NOT fire (uniform ~5ms)
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from moefit.metrics import detect_coalescing
        assert not detect_coalescing(r["per_token_ms"])["coalesced"]
    finally:
        srv.shutdown()


if __name__ == "__main__":
    test_all()
    print("ok test_all")
