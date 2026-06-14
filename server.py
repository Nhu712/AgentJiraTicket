#!/usr/bin/env python3
"""Serves chat.html and proxies /api/chat → AgentBase endpoint (bypasses CORS)."""

from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.request
import urllib.error
import json
import os

ENDPOINT = (
    'https://endpoint-c9b79e7b-2b3a-4bb5-be5c-b6406cfa5d72'
    '.agentbase-runtime.aiplatform.vngcloud.vn/invocations'
)
PORT = 8080


class Handler(SimpleHTTPRequestHandler):

    def do_OPTIONS(self):
        self._cors(200)

    def do_POST(self):
        if self.path != '/api/chat':
            self.send_error(404)
            return

        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length)

        try:
            req = urllib.request.Request(
                ENDPOINT,
                data=body,
                headers={'Content-Type': 'application/json'},
                method='POST',
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            self._cors(200, 'application/json')
            self.wfile.write(data)

        except urllib.error.HTTPError as e:
            msg = json.dumps({'status': 'error', 'response': f'Upstream {e.code}: {e.reason}'})
            self._cors(502, 'application/json')
            self.wfile.write(msg.encode())

        except Exception as e:
            msg = json.dumps({'status': 'error', 'response': str(e)})
            self._cors(500, 'application/json')
            self.wfile.write(msg.encode())

    def _cors(self, code, content_type=None):
        self.send_response(code)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        if content_type:
            self.send_header('Content-Type', content_type)
        self.end_headers()

    def log_message(self, fmt, *args):
        print(' ', args[0], args[1])


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    server = HTTPServer(('localhost', PORT), Handler)
    print(f'Running at http://localhost:{PORT}')
    print(f'Open  → http://localhost:{PORT}/chat.html')
    server.serve_forever()
