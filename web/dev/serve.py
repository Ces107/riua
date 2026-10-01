"""Local server with caching disabled: py -3.11 web/dev/serve.py  ->  http://localhost:8765/"""
import functools, http.server, pathlib


class NoCache(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


http.server.ThreadingHTTPServer(("127.0.0.1", 8765), functools.partial(NoCache, directory=str(pathlib.Path(__file__).resolve().parents[1]))).serve_forever()
