"""Solutions for medium_01_regex_extraction.py"""
import re
from dataclasses import dataclass

@dataclass
class EmailInfo:
    username: str
    domain: str
    tld: str

def extract_emails_detailed(text: str) -> list[EmailInfo]:
    pattern = r"([\w.+-]+)@([\w.-]+)\.(\w+)"
    matches = re.findall(pattern, text)
    return [EmailInfo(m[0], m[1], m[2]) for m in matches]

def extract_dates(text: str) -> list[tuple[str, str, str]]:
    pattern = r"(\d{4})-(\d{2})-(\d{2})"
    return re.findall(pattern, text)

def extract_urls_with_parts(text: str) -> list[dict[str, str]]:
    pattern = r"(?P<protocol>https?)://(?P<domain>[\w.-]+)(?P<path>/[\w./?&=-]*)?"
    results = []
    for m in re.finditer(pattern, text):
        results.append(m.groupdict())
    return results

def extract_quoted_strings(text: str) -> list[str]:
    return re.findall(r'"([^"]*)"', text)

def extract_hashtags(text: str) -> list[str]:
    return re.findall(r"#(\w+)", text)
