"""Solutions for easy_03_tokenisation_intro.py"""
import re
from collections import Counter

def whitespace_tokenise(text: str) -> list[str]:
    return text.split()

def simple_word_tokenise(text: str) -> list[str]:
    return re.findall(r"\w+", text)

def sentence_split(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]

def count_tokens(text: str) -> dict[str, int]:
    tokens = re.findall(r"\w+", text.lower())
    return dict(Counter(tokens))
