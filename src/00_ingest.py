#!/usr/bin/env python
"""Stage 0 – download the compressed XML dump to data/00_raw/"""

import requests, pathlib

URL = "https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2"
OUT = pathlib.Path("data/00_raw/simplewiki-latest-pages-articles.xml.bz2")
OUT.parent.mkdir(parents=True, exist_ok=True)

if OUT.exists():
    print("✓ already downloaded:", OUT)
else:
    print("⇣ downloading", URL)
    with requests.get(URL, stream=True, timeout=60) as r, OUT.open("wb") as f:
        r.raise_for_status()
        for chunk in r.iter_content(64_000):
            f.write(chunk)
    print("✓ saved to", OUT)
