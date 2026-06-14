#!/usr/bin/env python3
"""
Proxy server: serves chat.html and forwards /api/chat requests to the agent endpoint.
Run: python proxy_server.py
Open: http://localhost:8080
"""

import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

AGENT_ENDPOINT = os.environ.get(
    "AGENT_ENDPOINT",
    "https://endpoint-c9b79e7b-2b3a-4bb5-be5c-b6406cfa5d72.agentbase-runtime.aiplatform.vngcloud.vn/invocations",
)
PORT = int(os.environ.get("PORT", 8080))
CHAT_HTML = os.path.join(os.path.dirname(__file__), "chat.html")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"  {self.address_string()} - {fmt % args}")

    def send_json(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path in ("/", "/chat.html"):
            try:
                with open(CHAT_HTML, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            except FileNotFoundError:
                self.send_json(404, {"error": "chat.html not found"})
        else:
            self.send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/api/chat":
            self.send_json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        try:
            req = Request(
                AGENT_ENDPOINT,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=120) as resp:
                response_body = resp.read()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(response_body)

        except HTTPError as e:
            err_body = e.read()
            try:
                detail = json.loads(err_body)
            except Exception:
                detail = err_body.decode(errors="replace")
            self.send_json(e.code, {"error": f"Agent returned {e.code}", "detail": detail})

        except URLError as e:
            self.send_json(502, {"error": f"Could not reach agent: {e.reason}"})

        except Exception as e:
            self.send_json(500, {"error": str(e)})


if __name__ == "__main__":
    print(f"Proxy server running at http://localhost:{PORT}")
    print(f"Forwarding /api/chat -> {AGENT_ENDPOINT}")
    print("Press Ctrl+C to stop.\n")
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
