#!/usr/bin/env python3
"""Serves chat.html and runs the Jira agent locally.

Runs the LangGraph agent in-process so Jira API calls go through the local
machine (which can reach atlassian.net), bypassing cloud network restrictions.
"""

import json
import os
import sys
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Lock

# ── Point to jira-agent and its venv ─────────────────────────────────────────
_BASE   = os.path.dirname(os.path.abspath(__file__))
_AGENT  = os.path.join(_BASE, "jira-agent")
_VENV   = os.path.join(_AGENT, "venv", "lib", "python3.11", "site-packages")

sys.path.insert(0, _VENV)
sys.path.insert(0, _AGENT)

# Load .env before importing main so env vars are available at module load time
from dotenv import load_dotenv
load_dotenv(os.path.join(_AGENT, ".env"))

# Import the compiled LangGraph from jira-agent/main.py
from main import graph  # noqa: E402  (module-level side effects are intentional)

PORT      = int(os.environ.get("PORT", 8080))
CHAT_HTML = os.path.join(_BASE, "chat.html")
_lock     = Lock()   # graph.invoke is not thread-safe


# ── HTTP handler ─────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):  # quieter logs
        print(" ", self.address_string(), fmt % args)

    # ── CORS pre-flight ───────────────────────────────────────────────────────
    def do_OPTIONS(self):
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    # ── Serve chat.html ───────────────────────────────────────────────────────
    def do_GET(self):
        if self.path in ("/", "/chat.html"):
            try:
                with open(CHAT_HTML, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self._cors_headers()
                self.end_headers()
                self.wfile.write(content)
            except FileNotFoundError:
                self._json(404, {"error": "chat.html not found"})
        else:
            self._json(404, {"error": "not found"})

    # ── Agent call ────────────────────────────────────────────────────────────
    def do_POST(self):
        if self.path != "/api/chat":
            self._json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", 0))
        body   = self.rfile.read(length)

        try:
            payload = json.loads(body)
        except Exception:
            self._json(400, {"status": "error", "response": "Invalid JSON"})
            return

        message = payload.get("message", "").strip()
        if not message:
            self._json(400, {"status": "error", "response": "Missing 'message' in payload"})
            return

        try:
            with _lock:
                result = graph.invoke({"messages": [("user", message)]})
            msgs  = result.get("messages", [])
            reply = msgs[-1].content if msgs else "Agent returned no response."
            self._json(200, {
                "status":    "success",
                "response":  reply,
                "timestamp": datetime.now().isoformat(),
            })
        except Exception as exc:
            self._json(500, {"status": "error", "response": f"Agent error: {exc}"})

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _json(self, status, body):
        data = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(data)

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Jira Agent (local) → http://localhost:{PORT}/chat.html")
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
