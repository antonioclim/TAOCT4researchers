"""
Exercise: Regex Extraction (Medium)

Practice data extraction using regular expressions with groups.

Duration: 20-25 minutes
Difficulty: ★★★☆☆

Author: Antonio Clim
Version: 1.0.0
"""

from __future__ import annotations
import re
from dataclasses import dataclass


@dataclass
class EmailInfo:
    username: str
    domain: str
    tld: str


def extract_emails_detailed(text: str) -> list[EmailInfo]:
    """
    Extract email addresses with component parts.
    
    Use capturing groups to separate username, domain, and TLD.
    
    Example:
        >>> emails = extract_emails_detailed("Contact: user@example.com")
        >>> emails[0].username
        'user'
    """
    # TODO: Implement with capturing groups
    pass


def extract_dates(text: str) -> list[tuple[str, str, str]]:
    """
    Extract dates in format YYYY-MM-DD as (year, month, day) tuples.
    
    Example:
        >>> extract_dates("Date: 2025-01-17")
        [('2025', '01', '17')]
    """
    # TODO: Implement
    pass


def extract_urls_with_parts(text: str) -> list[dict[str, str]]:
    """
    Extract URLs and parse into components.
    
    Returns list of dicts with keys: protocol, domain, path
    """
    # TODO: Implement using named groups
    pass


def extract_quoted_strings(text: str) -> list[str]:
    """
    Extract all strings enclosed in double quotes.
    
    Example:
        >>> extract_quoted_strings('He said "Hello" and "Goodbye"')
        ['Hello', 'Goodbye']
    """
    # TODO: Implement
    pass


def extract_hashtags(text: str) -> list[str]:
    """
    Extract hashtags from social media text.
    
    Example:
        >>> extract_hashtags("Learning #Python and #NLP today!")
        ['Python', 'NLP']
    """
    # TODO: Implement
    pass


def run_tests() -> None:
    emails = extract_emails_detailed("Contact: user@example.com")
    if emails:
        assert emails[0].username == "user"
    
    dates = extract_dates("Date: 2025-01-17")
    if dates:
        assert dates[0] == ("2025", "01", "17")
    
    quoted = extract_quoted_strings('He said "Hello"')
    if quoted:
        assert "Hello" in quoted
    
    print("All tests passed!")


if __name__ == "__main__":
    run_tests()
