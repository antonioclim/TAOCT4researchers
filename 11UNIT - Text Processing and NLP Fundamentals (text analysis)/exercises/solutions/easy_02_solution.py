"""Solutions for easy_02_regex_basics.py"""
import re

def find_all_digits(text: str) -> list[str]:
    return re.findall(r"\d+", text)

def find_all_words(text: str) -> list[str]:
    return re.findall(r"\w+", text)

def has_email(text: str) -> bool:
    return bool(re.search(r"\b[\w.+-]+@[\w.-]+\.\w+\b", text))

def replace_digits(text: str, replacement: str = "#") -> str:
    return re.sub(r"\d", replacement, text)

def split_on_punctuation(text: str) -> list[str]:
    return re.split(r"[.!?]", text)
