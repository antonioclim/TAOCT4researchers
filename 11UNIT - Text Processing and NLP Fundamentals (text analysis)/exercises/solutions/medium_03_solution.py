"""Solutions for medium_03_frequency_analysis.py"""
import re
from collections import Counter

def word_frequencies(text: str) -> Counter[str]:
    tokens = re.findall(r"\w+", text.lower())
    return Counter(tokens)

def top_n_words(text: str, n: int = 10) -> list[tuple[str, int]]:
    return word_frequencies(text).most_common(n)

def hapax_legomena(text: str) -> list[str]:
    freq = word_frequencies(text)
    return [word for word, count in freq.items() if count == 1]

def type_token_ratio(text: str) -> float:
    tokens = re.findall(r"\w+", text.lower())
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)

def vocabulary_richness(text: str) -> dict[str, float]:
    tokens = re.findall(r"\w+", text.lower())
    if not tokens:
        return {"ttr": 0, "hapax_ratio": 0, "avg_word_length": 0}
    freq = Counter(tokens)
    hapax = sum(1 for c in freq.values() if c == 1)
    return {
        "ttr": len(set(tokens)) / len(tokens),
        "hapax_ratio": hapax / len(freq),
        "avg_word_length": sum(len(t) for t in tokens) / len(tokens)
    }
