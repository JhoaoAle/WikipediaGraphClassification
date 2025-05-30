#!/usr/bin/env python
"""Stage 1 – stream‑parse raw XML dump ⇒ Parquet with title + body"""

import pathlib
import xml.etree.ElementTree as ET
import pandas as pd
from tqdm import tqdm
import bz2

from utils.stream_bz2 import open_url_bz2

URL = "https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2"
LOCAL = pathlib.Path("data/00_raw/simplewiki-latest-pages-articles.xml.bz2")
ARTICLES = pathlib.Path("data/10_parsed/articles.parquet")
ARTICLES.parent.mkdir(parents=True, exist_ok=True)

def main():

    if ARTICLES.exists():
        df = pd.read_parquet(ARTICLES)
        print("✓ loaded", ARTICLES, len(df), "rows")
    else:
        # Choose source: local file if exists, otherwise URL
        if LOCAL.exists():
            print("Using local file:", LOCAL)
            stream = bz2.open(LOCAL, "rb")
        else:
            print("Downloading from URL:", URL)
            stream = open_url_bz2(URL)

        titles, bodies = [], []
        with stream:
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

if __name__ == "__main__":
    main()