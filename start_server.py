#!/usr/bin/env python3
"""
Simple HTTP server for local development of Tessellate
Serves files without CORS issues
"""

import http.server
import socketserver
import os

PORT = 8000

class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add CORS headers to allow local file access
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()

os.chdir(os.path.dirname(os.path.abspath(__file__)))

with socketserver.TCPServer(("", PORT), CORSHTTPRequestHandler) as httpd:
    print(f"Tessellate server running at http://localhost:{PORT}/")
    print(f"Open http://localhost:{PORT}/ to play the game")
    print(f"Open http://localhost:{PORT}/browser.html to browse games")
    print("Press Ctrl+C to stop the server")
    httpd.serve_forever()