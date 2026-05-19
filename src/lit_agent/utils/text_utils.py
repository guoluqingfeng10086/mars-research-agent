# src/lit_agent/utils/text_utils.py

import re
from typing import List


def clean_text(value) -> str:
    if value is None:
        return ""

    text = str(value)
    text = text.replace("\r", " ").replace("\n", " ")
    text = " ".join(text.split())
    return text.strip()


def normalize_text(value) -> str:
    return clean_text(value).lower()

def simple_word_tokenize(text: str) -> List[str]:
    text = normalize_text(text)

    tokens = re.findall(
        r"[a-zA-Z0-9]+(?:[-'][a-zA-Z0-9]+)?|[\u4e00-\u9fff]",
        text
    )

    return tokens

def normalize_phrase(phrase: str) -> str:
    tokens = simple_word_tokenize(phrase)
    return " ".join(tokens)

def phrase_ngram_tokenize(text: str, max_n: int = 6) -> List[str]:
    """
    Generate phrase tokens for BM25.

    Example:
        "Gale Crater methane"
    tokens include:
        "gale"
        "crater"
        "methane"
        "gale crater"
        "crater methane"
        "gale crater methane"

    Query side will only use complete phrases extracted by LLM.
    """

    words = simple_word_tokenize(text)

    if not words:
        return []

    tokens = []

    max_n = min(max_n, len(words))
    for n in range(1, max_n + 1):
        for i in range(len(words) - n + 1):
            tokens.append(" ".join(words[i:i + n]))
    return tokens