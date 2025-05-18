# stream_bz2.py  ── tiny helper that turns a requests stream into a bz2‑decompressed file‑like object
import io, bz2, requests
from typing import Iterator

class StreamDecompressor(io.RawIOBase):
    def __init__(self, iterator: Iterator[bytes]):
        self._it = iterator
        self._dcmp = bz2.BZ2Decompressor()
        self._buf = b""

    def readable(self):                # ElementTree needs this
        return True

    def readinto(self, b):             # core of Python's file API
        while not self._buf:
            try:
                self._buf += self._dcmp.decompress(next(self._it))
            except StopIteration:
                return 0               # EOF
        n = min(len(b), len(self._buf))
        b[:n] = self._buf[:n]
        self._buf = self._buf[n:]
        return n

def open_url_bz2(url: str, chunk=64_000) -> io.BufferedReader:
    """Return a *buffered* file‑like object that streams + decompresses the URL."""
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    raw = StreamDecompressor(resp.iter_content(chunk_size=chunk))
    return io.BufferedReader(raw)