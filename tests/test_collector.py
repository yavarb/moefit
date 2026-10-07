"""Tests: collect_silicon_run.one_run against a mock SSE server.

Regression for the T8 counterexamples:
  astra_local_exp 175d734 (timing):
  1. A slow usage TRAILER after the last text token must not deflate
     decode_tps (was: wall-EOF closed the decode interval, 5s trailer
     cut a 12.50 tok/s stream to 10.05).
  2. Control chunks (empty-delta role/finish chunks) must not enter
     the per-token gap samples.
  3. First control chunk is not TTFT.
  astra_local_exp 8f494f3 (completion integrity — production form):
  4. Truncated EOF (no finish/usage/DONE) -> stream_integrity False;
     damaged streams report problems and are marked ineligible.
  5. Usage-without-finish -> ineligible.
  6. Malformed data lines are COUNTED, not silently skipped.
  7. Text after [DONE] must not count.
  8. A combined usage+final-text event must not lose the text or the
     finish (T8: 16 tokens -> 15 deltas, finish lost).
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
        body = []
        for i in range(n):
            body.append(((i + 1) * gen_ms / 1000,
                         sse(dict(choices=[dict(
                             delta=dict(content=f"t{i}"),
                             finish_reason=None)]))))
        if self.behavior == "malformed":
            body.insert(10, (10.5 * gen_ms / 1000, b"data: {broken\n\n"))
        b = self.behavior
        if b == "truncated":            # EOF mid-stream, nothing valid after
            for t, w in body[:30]:
                time.sleep(max(0.0, t0 + t - time.monotonic()))
                self.wfile.write(w)
            return                       # abrupt EOF: no finish/usage/DONE
        if b == "usage_no_finish":
            for t, w in body:
                time.sleep(max(0.0, t0 + t - time.monotonic()))
                self.wfile.write(w)
            self.wfile.write(sse(dict(usage=dict(
                completion_tokens=n, prompt_tokens=5))))
            self.wfile.write(b"data: \n\n")   # no finish_reason anywhere
            return
        if b == "post_done_text":
            for t, w in body:
                time.sleep(max(0.0, t0 + t - time.monotonic()))
                self.wfile.write(w)
            self.wfile.write(sse(dict(choices=[dict(
                delta={}, finish_reason="stop")])))
            self.wfile.write(b"data: [DONE]\n\n")
            # damage: tokens AFTER the terminal marker
            self.wfile.write(sse(dict(choices=[dict(
                delta=dict(content="GHOST"), finish_reason=None)])))
            self.wfile.write(sse(dict(usage=dict(
                completion_tokens=n, prompt_tokens=5))))
            return
        if b == "combined_final":
            # final text + finish + usage in ONE event, then [DONE]:
            # old parser hit `continue` and dropped text and finish
            for t, w in body[:-1]:
                time.sleep(max(0.0, t0 + t - time.monotonic()))
                self.wfile.write(w)
            time.sleep(max(0.0, t0 + body[-1][0] - time.monotonic()))
            self.wfile.write(sse(dict(
                choices=[dict(delta=dict(content="last"),
                              finish_reason="stop")],
                usage=dict(completion_tokens=n, prompt_tokens=5))))
            self.wfile.write(b"data: [DONE]\n\n")
            return
        # default "trailer": complete stream + interleaved controls
        for t, w in body:
            time.sleep(max(0.0, t0 + t - time.monotonic()))
            self.wfile.write(w)
            i = round(t * 1000 / gen_ms) - 1
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
    # timing counterexamples (complete stream + 1s trailer)
    srv, url = make_server(Handler)
    try:
        r = one_run(url, "m", "hi", 64, timeout=30)
        expect_tps = 1000.0 / 5.0
        assert r["tokens"] == 40, r
        assert abs(r["decode_tps"] - expect_tps) < 12, r
        assert r["decode_s"] < 0.6, r          # closed at last token event
        assert r["wall_s"] >= 1.0              # trailer is in wall only
        assert len(r["per_token_ms"]) == 39, r  # n_text-1 only
        assert 0.0 <= r["ttft_s"] < 0.05, r
        assert r["n_control_chunks"] >= 1 + 10 + 1, r
        assert r["timing_provenance"].startswith("client SSE chunk")
        si = r["stream_integrity"]
        assert si["eligible"] and si["finish_reason"] == "stop", si
        from moefit.metrics import detect_coalescing
        assert not detect_coalescing(r["per_token_ms"])["coalesced"]
    finally:
        srv.shutdown()

    # completion-integrity counterexamples (T8 8f494f3, production form)
    for behavior, checks in [
        ("truncated", lambda r: (
            not r["stream_integrity"]["eligible"],
            "missing_or_unsupported_finish" in r["stream_integrity"]["problems"],
            "no_authoritative_count" in r["stream_integrity"]["problems"],
            # tokens fall back to streamed deltas and stay descriptive;
            # stream_interrupted is best-effort (depends on whether the
            # transport yields RST vs clean EOF), not part of the gate
            r["tokens"] == 30)),
        ("usage_no_finish", lambda r: (
            r["tokens"] == 40,                      # usage still counted
            not r["stream_integrity"]["eligible"],  # but ineligible
            "missing_or_unsupported_finish" in r["stream_integrity"]["problems"])),
        ("post_done_text", lambda r: (
            r["tokens"] == 40,
            r["streamed_deltas"] == 40,             # GHOST token NOT counted
            r["stream_integrity"]["done"],
            # ghost content is protocol damage: throughput is correct
            # but the run is ineligible for scoring (fail closed)
            not r["stream_integrity"]["eligible"],
            "content_after_done" in r["stream_integrity"]["problems"])),
        ("combined_final", lambda r: (
            r["tokens"] == 40,
            r["streamed_deltas"] == 40,             # last text not dropped
            r["stream_integrity"]["finish_reason"] == "stop",
            r["stream_integrity"]["eligible"])),
        ("malformed", lambda r: (
            r["tokens"] == 40,
            r["stream_integrity"]["parse_errors"] == 1,
            not r["stream_integrity"]["eligible"],
            any("malformed" in p
                for p in r["stream_integrity"]["problems"]))),
    ]:
        Handler.behavior = behavior
        srv, url = make_server(Handler)
        try:
            r = one_run(url, "m", "hi", 64, timeout=30)
            ok = checks(r)
            assert all(ok), (behavior, r["stream_integrity"], r["tokens"],
                            r["streamed_deltas"])
        finally:
            srv.shutdown()


if __name__ == "__main__":
    test_all()
    print("ok test_all")
