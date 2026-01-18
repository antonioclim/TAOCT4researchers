"""
Lab 11_01 Solutions: Regular Expressions and String Operations

This module provides worked solutions and additional examples
for the regex and string operations laboratory.

Author: Antonio Clim
Version: 1.0.0
"""

from __future__ import annotations

import re
import unicodedata
from lab.lab_11_01_regex_string_ops import (
    StringOperations,
    RegexMatcher,
    PatternLibrary,
    AdvancedPatternMatcher,
    PatternValidator,
    UnicodeHandler,
    TextCleaner,
    CleaningConfig,
)


# =============================================================================
# SOLUTION EXAMPLES
# =============================================================================

def solution_string_methods() -> None:
    """Demonstrate string method solutions."""
    # Task: Process a messy string
    messy = "  Hello,   World!  This   is   a   TEST.  "
    
    ops = StringOperations(messy)
    
    # Clean and normalise
    cleaned = ops.normalise_whitespace()
    assert cleaned == "Hello, World! This is a TEST."
    
    # Case transformations
    assert ops.to_lowercase() == "  hello,   world!  this   is   a   test.  "
    
    # Splitting and counting
    words = cleaned.split()
    assert len(words) == 6
    assert ops.count_occurrences("a") == 2


def solution_regex_extraction() -> None:
    """Demonstrate regex extraction solutions."""
    text = """
    Contact Information:
    Email: john.doe@example.com, jane_smith@company.co.uk
    Phone: +44 123 456 7890, (555) 123-4567
    Website: https://example.com/page?id=123
    Date: 2025-01-17, 17/01/2025
    """
    
    lib = PatternLibrary()
    
    # Extract all emails
    emails = lib.extract_emails(text)
    assert "john.doe@example.com" in emails
    assert "jane_smith@company.co.uk" in emails
    
    # Extract all URLs
    urls = lib.extract_urls(text)
    assert any("example.com" in url for url in urls)
    
    # Extract ISO dates
    iso_dates = lib.extract_dates_iso(text)
    assert "2025-01-17" in iso_dates
    
    # Extract UK dates
    uk_dates = lib.extract_dates_uk(text)
    assert "17/01/2025" in uk_dates


def solution_advanced_patterns() -> None:
    """Demonstrate advanced pattern solutions."""
    matcher = AdvancedPatternMatcher()
    
    # Parse log entries
    log = "2025-01-17 14:30:25 ERROR Database connection failed"
    result = matcher.parse_log_entry(log)
    
    assert result is not None
    assert result.groups["date"] == "2025-01-17"
    assert result.groups["level"] == "ERROR"
    assert result.groups["message"] == "Database connection failed"
    
    # Extract currency amounts
    prices = "Items cost $100, £50.99, and €200"
    amounts = matcher.extract_currency_amounts(prices)
    assert "100" in amounts
    assert "50.99" in amounts
    
    # Custom pattern with lookahead
    # Find words followed by a comma
    pattern = re.compile(r"\b\w+(?=,)")
    text = "apple, banana, cherry"
    matches = pattern.findall(text)
    assert matches == ["apple", "banana"]


def solution_validation() -> None:
    """Demonstrate validation pattern solutions."""
    validator = PatternValidator()
    
    # Email validation
    assert validator.is_valid_email("user@example.com") is True
    assert validator.is_valid_email("invalid-email") is False
    assert validator.is_valid_email("user@.com") is False
    
    # URL validation
    assert validator.is_valid_url("https://example.com") is True
    assert validator.is_valid_url("ftp://example.com") is False
    
    # UK postcode validation
    assert validator.is_valid_uk_postcode("SW1A 1AA") is True
    assert validator.is_valid_uk_postcode("12345") is False


def solution_unicode_handling() -> None:
    """Demonstrate Unicode handling solutions."""
    handler = UnicodeHandler()
    
    # Normalisation comparison
    composed = "café"  # é as single character
    decomposed = "cafe\u0301"  # e + combining accent
    
    # Without normalisation, these are different
    assert composed != decomposed
    
    # With NFC normalisation, they become equal
    norm_composed = handler.normalise_nfc(composed)
    norm_decomposed = handler.normalise_nfc(decomposed)
    assert norm_composed == norm_decomposed
    
    # Accent removal
    accented = "Crème brûlée"
    no_accents = handler.remove_accents(accented)
    assert no_accents == "Creme brulee"
    
    # Character information
    info = handler.get_char_info("é")
    assert info["hex"] == "U+00E9"


def solution_text_cleaning() -> None:
    """Demonstrate text cleaning solutions."""
    # HTML and URL removal
    html_text = """
    <html>
    <body>
    <p>Welcome to our website!</p>
    <p>Visit https://example.com for more info.</p>
    <p>Contact: info@example.com</p>
    </body>
    </html>
    """
    
    config = CleaningConfig(
        remove_html=True,
        remove_urls=True,
        remove_emails=True,
        normalise_whitespace=True
    )
    
    cleaner = TextCleaner(config)
    result = cleaner.clean(html_text)
    
    assert "<" not in result.cleaned
    assert "https://" not in result.cleaned
    assert "@" not in result.cleaned
    assert "Welcome" in result.cleaned


def solution_custom_pattern() -> None:
    """Demonstrate creating custom extraction patterns."""
    # Extract citations in format: (Author, Year)
    citation_pattern = re.compile(r"\(([A-Z][a-z]+),\s*(\d{4})\)")
    
    text = "As noted by (Smith, 2020) and (Jones, 2019), the findings..."
    
    citations = citation_pattern.findall(text)
    assert ("Smith", "2020") in citations
    assert ("Jones", "2019") in citations
    
    # Extract DOIs
    doi_pattern = re.compile(r"10\.\d{4,}/[^\s]+")
    
    text_with_doi = "See DOI: 10.1234/example.2025.001 for details."
    dois = doi_pattern.findall(text_with_doi)
    assert "10.1234/example.2025.001" in dois


# =============================================================================
# EXERCISE SOLUTIONS
# =============================================================================

def exercise_1_solution() -> dict[str, list[str]]:
    """
    Exercise 1: Extract all structured data from a document.
    
    Returns:
        Dictionary with extracted emails, phones, dates and URLs.
    """
    document = """
    Meeting Notes - 2025-01-17
    
    Attendees: john@company.com, mary@company.com
    Conference call: +1 (555) 123-4567
    
    Action items due: 2025-02-01
    Resources: https://wiki.company.com/meeting-notes
    
    Next meeting: 24/01/2025
    Contact support: support@company.com, 1-800-555-0123
    """
    
    lib = PatternLibrary()
    
    return {
        "emails": lib.extract_emails(document),
        "phones": lib.extract_phone_numbers(document),
        "dates_iso": lib.extract_dates_iso(document),
        "dates_uk": lib.extract_dates_uk(document),
        "urls": lib.extract_urls(document),
    }


def exercise_2_solution() -> str:
    """
    Exercise 2: Clean and normalise a messy OCR output.
    
    Returns:
        Cleaned text.
    """
    ocr_output = """
    <p>Tlie quick  brown   fox jurnps
    over   tlie lazy  dog.</p>
    
    <br>
    
    Conlact: exarnple@test.corn
    """
    
    # Custom cleaning with OCR corrections
    text = ocr_output
    
    # Remove HTML
    text = re.sub(r"<[^>]+>", "", text)
    
    # OCR corrections
    corrections = {
        "Tlie": "The",
        "tlie": "the",
        "jurnps": "jumps",
        "Conlact": "Contact",
        "exarnple": "example",
        ".corn": ".com",
    }
    
    for error, fix in corrections.items():
        text = text.replace(error, fix)
    
    # Normalise whitespace
    text = " ".join(text.split())
    
    return text


if __name__ == "__main__":
    # Run all solutions
    solution_string_methods()
    solution_regex_extraction()
    solution_advanced_patterns()
    solution_validation()
    solution_unicode_handling()
    solution_text_cleaning()
    solution_custom_pattern()
    
    print("All solutions verified successfully!")
