"""Minimal HTTP-range backed file object so zipfile can list / extract single
members of a large remote zip without downloading the whole archive.

Used for the HydroSHEDS v1 3 arc-second continental archives (0.8-2.4 GB each),
of which we only need a handful of 5x5 degree tiles.
"""
import io
import requests

UA = "Mozilla/5.0 (riua-research; h1-catchments)"


class HttpRangeFile(io.RawIOBase):
    def __init__(self, url, block=4 * 1024 * 1024):
        self.url = url
        self.s = requests.Session()
        self.s.headers["User-Agent"] = UA
        r = self.s.head(url, allow_redirects=True, timeout=60)
        r.raise_for_status()
        self.size = int(r.headers["Content-Length"])
        self.pos = 0
        self.block = block
        self._cache = {}
        self.fetched = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        if whence == 0:
            self.pos = off
        elif whence == 1:
            self.pos += off
        else:
            self.pos = self.size + off
        return self.pos

    def _get_block(self, bi):
        if bi not in self._cache:
            a = bi * self.block
            b = min(self.size, a + self.block) - 1
            for attempt in range(5):
                try:
                    r = self.s.get(self.url, headers={"Range": f"bytes={a}-{b}"}, timeout=180)
                    r.raise_for_status()
                    data = r.content
                    if len(data) != b - a + 1:
                        raise IOError("short read")
                    break
                except Exception as e:  # noqa
                    if attempt == 4:
                        raise
            self.fetched += len(data)
            if len(self._cache) > 16:
                self._cache.pop(next(iter(self._cache)))
            self._cache[bi] = data
        return self._cache[bi]

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        out = bytearray()
        while n > 0:
            bi = self.pos // self.block
            data = self._get_block(bi)
            o = self.pos - bi * self.block
            chunk = data[o:o + n]
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)
