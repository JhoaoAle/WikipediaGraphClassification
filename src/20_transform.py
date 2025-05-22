#!/usr/bin/env python
"""Stage 2 – read Wikitext rows ⇒ add destination_articles + clean_body"""

import pathlib
import pandas as pd
from utils.wikiclean import extract_links
import wikitextparser as wtp

IN_PARQUET = pathlib.Path("data/10_parsed/articles.parquet")
OUT_PARQUET = pathlib.Path("data/20_clean/articles.parquet")
OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)


def clean_wiki_text(text):
    if not isinstance(text, str):
        return ""
    
    parsed = wtp.parse(text)

    # Get plain text without templates, links, and formatting
    clean = parsed.plain_text()

    # Optionally remove excessive whitespace
    clean = ' '.join(clean.split())
    
    return clean


df = pd.read_parquet(IN_PARQUET)


df["linked_article_titles"] = df["body"].apply(extract_links)
df["cleaned_article_body"] = df["body"].apply(clean_wiki_text)
df.to_parquet(OUT_PARQUET, index=False)

print("✓ wrote", OUT_PARQUET, len(df), "rows")
