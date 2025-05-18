# wikiclean.py  ── regex helpers to strip/parse Wikitext
import re
from typing import List

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

def clean(text: str) -> str:
    text = JUNK_PATTERNS.sub(" ", text)
    text = re.sub(r"\[\[([^|\]]+\|)?([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"\]\]|\[\[", " ", text)
    text = re.sub(r"[ .]*\.[ .]*", ". ", text)
    return re.sub(r"\s+", " ", text).strip()
