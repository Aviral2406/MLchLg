"""
Multi-representation normalization for business_name / business_address / country.
See docs/architecture.md §3. Never collapse a field to a single normalized string -
different similarity features need different levels of aggression.

Suffix/abbreviation maps are loaded from configs/normalization/*.json, which MUST be
mined from the provided train/test corpus (see docs/data_dictionary_template.md),
never hand-authored from general knowledge or external sources (CLAUDE.md §2).
"""
from __future__ import annotations
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs" / "normalization"

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[.,;:()\[\]\"']")
_DIGIT_RE = re.compile(r"\d+")


def _load_json_map(filename: str) -> dict:
    path = CONFIG_DIR / filename
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


SUFFIX_MAP = _load_json_map("suffixes.json")             # e.g. {"pvt ltd": "private limited"}
ABBREVIATION_MAP = _load_json_map("abbreviations.json")  # e.g. {"rd": "road", "st": "street"}


@dataclass
class NormalizedField:
    raw: str
    normalized: str
    tokenized: list = field(default_factory=list)
    char_ngrams: list = field(default_factory=list)
    digits_only: list = field(default_factory=list)
    sorted_tokens: str = ""


def _base_clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.lower().strip()
    text = text.replace("&", " and ")
    text = _PUNCT_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def _apply_map(tokens: list, mapping: dict) -> list:
    # Longest-key-first so multi-word keys (e.g. "pvt ltd") match before single tokens.
    if not mapping:
        return tokens
    text = " ".join(tokens)
    for key in sorted(mapping, key=len, reverse=True):
        text = re.sub(r"\b" + re.escape(key) + r"\b", mapping[key], text)
    return text.split()


def char_ngrams(text: str, n: int = 4) -> list:
    compact = text.replace(" ", "")
    if len(compact) < n:
        return [compact] if compact else []
    return [compact[i:i + n] for i in range(len(compact) - n + 1)]


def normalize_field(raw_text: str, mapping: dict = None, ngram_n: int = 4) -> NormalizedField:
    raw_text = raw_text or ""
    cleaned = _base_clean(raw_text)
    tokens = cleaned.split()
    if mapping:
        tokens = _apply_map(tokens, mapping)
    normalized = " ".join(tokens)
    digits = _DIGIT_RE.findall(raw_text)
    return NormalizedField(
        raw=raw_text,
        normalized=normalized,
        tokenized=tokens,
        char_ngrams=char_ngrams(normalized, n=ngram_n),
        digits_only=digits,
        sorted_tokens=" ".join(sorted(tokens)),
    )


@dataclass
class NormalizedRecord:
    entity_id: str
    name: NormalizedField
    address: NormalizedField
    country_normalized: str


def normalize(record: dict) -> NormalizedRecord:
    """record: dict with keys entity_id, business_name, business_address, country."""
    return NormalizedRecord(
        entity_id=record["entity_id"],
        name=normalize_field(record.get("business_name", ""), mapping=SUFFIX_MAP),
        address=normalize_field(record.get("business_address", ""), mapping=ABBREVIATION_MAP),
        country_normalized=_base_clean(record.get("country", "")),
    )
