# wikiclean.py  ── regex helpers to strip/parse Wikitext
import re
from typing import List
import wikitextparser as wtp
import ast
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

WIKI_LINK = re.compile(r"\[\[([^\]\|#]+)")

JUNK_PATTERNS = re.compile(
    r"""
    \{\{[^}]+\}\}          |   # templates
    \'\'+                  |   # '' or '''
    \[\[File:[^\]]+\]\]    |   # files
    \[\[Category:[^\]]+\]\]|   # categories
    \[[^\[]+?\.jpg[^\]]*\]     # bare jpg links
    """,
    re.VERBOSE | re.IGNORECASE,
)

def extract_links(text: str) -> List[str]:
    seen, out = set(), []
    for m in WIKI_LINK.finditer(text):
        tgt = m.group(1).strip()
        if tgt and tgt not in seen:
            seen.add(tgt); out.append(tgt)
    return out

def clean_wiki_text(text):
    if not isinstance(text, str):
        return ""
    parsed = wtp.parse(text)
    clean = parsed.plain_text()
    clean = ' '.join(clean.split())
    return clean

def parallel_clean_wiki_text(series, chunk_size=15000, workers=10):
    results = []
    # Iterate over chunks with a tqdm progress bar
    for i in tqdm(range(0, len(series), chunk_size), desc="Chunks"):
        chunk = series.iloc[i:i+chunk_size]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            # Map the single-item clean function across the chunk in parallel
            cleaned_chunk = list(executor.map(clean_wiki_text, chunk))
        results.extend(cleaned_chunk)
    return results

# Extract all category values
def extract_categories(text):
    return re.findall(r'\[\[Category:([^\|\]]+)', text)

def clean_linked_articles(cell):
    try:
        articles = ast.literal_eval(cell) if isinstance(cell, str) else cell
        return [a.lower() for a in articles]
    except:
        return []