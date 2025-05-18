#!/usr/bin/env python
"""Stage 2 – read Wikitext rows ⇒ add dest_articles + clean_body"""

import pathlib, pandas as pd
from utils.wikiclean import extract_links, clean

IN_PARQUET  = pathlib.Path("data/10_parsed/articles.parquet")
OUT_PARQUET = pathlib.Path("data/20_clean/articles.parquet")
OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(IN_PARQUET)
df["dest_articles"] = df["body"].apply(extract_links)
df["clean_body"]    = df["body"].apply(clean)
df.to_parquet(OUT_PARQUET, index=False)
print("✓ wrote", OUT_PARQUET, len(df), "rows")
