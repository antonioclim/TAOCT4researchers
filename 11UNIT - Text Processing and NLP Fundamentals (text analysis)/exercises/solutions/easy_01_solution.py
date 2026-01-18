"""Solutions for easy_01_string_operations.py"""

def normalise_whitespace(text: str) -> str:
    return " ".join(text.split())

def count_words(text: str) -> int:
    words = text.split()
    return len(words)

def reverse_words(text: str) -> str:
    return " ".join(text.split()[::-1])

def capitalise_words(text: str) -> str:
    return text.title()

def find_longest_word(text: str) -> str:
    words = text.split()
    if not words:
        return ""
    return max(words, key=len)

def replace_multiple(text: str, replacements: dict[str, str]) -> str:
    result = text
    for old, new in replacements.items():
        result = result.replace(old, new)
    return result

def extract_initials(text: str) -> str:
    return "".join(word[0].upper() for word in text.split() if word)
