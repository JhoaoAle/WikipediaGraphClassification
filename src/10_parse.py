#!/usr/bin/env python
"""Stage 1 – stream‑parse raw XML dump ⇒ Parquet with title + body"""

import pathlib, xml.etree.ElementTree as ET, pandas as pd
from tqdm import tqdm
from utils.stream_bz2 import open_url_bz2          # pull straight from web
# If you want to parse the local .bz2 instead, swap to `bz2.open(...)`

URL = "https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2"
ARTICLES = pathlib.Path("data/10_parsed/articles.parquet")
ARTICLES.parent.mkdir(parents=True, exist_ok=True)

titles, bodies = [], []
with open_url_bz2(URL) as stream:
    for _, elem in tqdm(ET.iterparse(stream, events=("end",)), desc="XML"):
        if elem.tag.endswith("page") and elem.findtext("./{*}ns") == "0":
            text = elem.findtext("./{*}revision/{*}text") or ""
            if text:
                titles.append(elem.findtext("./{*}title"))
                bodies.append(text)
            elem.clear()

df = pd.DataFrame({"title": titles, "body": bodies})
df.to_parquet(ARTICLES, index=False)
print("✓ wrote", ARTICLES, len(df), "rows")
