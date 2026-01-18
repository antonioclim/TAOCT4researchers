"""
Tests for Lab 11_01: Regular Expressions and String Operations

This module provides comprehensive tests for the regex and string
operations laboratory components.

Run with: pytest tests/test_lab_11_01.py -v

Author: Antonio Clim
Version: 1.0.0
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    pass

# Import lab modules (path configured in conftest.py)
try:
    from lab_11_01_regex_string_ops import (
        StringOperations,
        RegexMatcher,
        PatternLibrary,
        AdvancedPatternMatcher,
        PatternValidator,
        UnicodeHandler,
        TextCleaner,
        CleaningConfig,
        CleaningResult,
    )
    LAB_AVAILABLE = True
except ImportError:
    LAB_AVAILABLE = False


# =============================================================================
# STRING OPERATIONS TESTS
# =============================================================================

@pytest.mark.skipif(not LAB_AVAILABLE, reason="Lab module not available")
class TestStringOperations:
    """Tests for the StringOperations class."""
    
    def test_lowercase(self) -> None:
        """Test lowercase conversion."""
        ops = StringOperations("Hello WORLD")
        assert ops.to_lowercase() == "hello world"
    
    def test_uppercase(self) -> None:
        """Test uppercase conversion."""
        ops = StringOperations("Hello World")
        assert ops.to_uppercase() == "HELLO WORLD"
    
    def test_titlecase(self) -> None:
        """Test title case conversion."""
        ops = StringOperations("hello world")
        assert ops.to_titlecase() == "Hello World"
    
    def test_normalise_whitespace(self) -> None:
        """Test whitespace normalisation."""
        ops = StringOperations("  Hello   World  ")
        assert ops.normalise_whitespace() == "Hello World"
    
    def test_normalise_whitespace_empty(self) -> None:
        """Test whitespace normalisation on empty string."""
        ops = StringOperations("   ")
        assert ops.normalise_whitespace() == ""
    
    def test_split_on_delimiter(self) -> None:
        """Test splitting on delimiter."""
        ops = StringOperations("a,b,c,d")
        assert ops.split_on_delimiter(",") == ["a", "b", "c", "d"]
    
    def test_split_with_maxsplit(self) -> None:
        """Test splitting with maximum splits."""
        ops = StringOperations("a-b-c-d")
        assert ops.split_on_delimiter("-", 2) == ["a", "b", "c-d"]
    
    def test_join_with_delimiter(self) -> None:
        """Test joining with delimiter."""
        ops = StringOperations("")
        assert ops.join_with_delimiter(["a", "b", "c"], " | ") == "a | b | c"
    
    def test_count_occurrences(self) -> None:
        """Test substring counting."""
        ops = StringOperations("abracadabra")
        assert ops.count_occurrences("a") == 5
    
    def test_find_position(self) -> None:
        """Test finding substring position."""
        ops = StringOperations("Hello World")
        assert ops.find_position("World") == 6
        assert ops.find_position("xyz") == -1
    
    def test_replace_substring(self) -> None:
        """Test substring replacement."""
        ops = StringOperations("Hello World")
        assert ops.replace_substring("World", "Python") == "Hello Python"
    
    def test_check_prefix(self) -> None:
        """Test prefix checking."""
        ops = StringOperations("Hello World")
        assert ops.check_prefix("Hello") is True
        assert ops.check_prefix("World") is False
    
    def test_check_suffix(self) -> None:
        """Test suffix checking."""
        ops = StringOperations("Hello World")
        assert ops.check_suffix("World") is True
        assert ops.check_suffix("Hello") is False


# =============================================================================
# REGEX MATCHER TESTS
# =============================================================================

@pytest.mark.skipif(not LAB_AVAILABLE, reason="Lab module not available")
class TestRegexMatcher:
    """Tests for the RegexMatcher class."""
    
    def test_search_found(self) -> None:
        """Test search finding a match."""
        matcher = RegexMatcher(r"\d+")
        match = matcher.search("Order 123")
        assert match is not None
        assert match.group() == "123"
    
    def test_search_not_found(self) -> None:
        """Test search with no match."""
        matcher = RegexMatcher(r"\d+")
        match = matcher.search("No numbers here")
        assert match is None
    
    def test_match_at_start(self) -> None:
        """Test match at string start."""
        matcher = RegexMatcher(r"\d+")
        assert matcher.match("123abc") is not None
        assert matcher.match("abc123") is None
    
    def test_fullmatch(self) -> None:
        """Test full string matching."""
        matcher = RegexMatcher(r"\d+")
        assert matcher.fullmatch("123") is not None
        assert matcher.fullmatch("123abc") is None
    
    def test_find_all(self) -> None:
        """Test finding all matches."""
        matcher = RegexMatcher(r"\d+")
        matches = matcher.find_all("Order 123 and 456")
        assert matches == ["123", "456"]
    
    def test_find_iter(self) -> None:
        """Test iterating over matches."""
        matcher = RegexMatcher(r"\d+")
        matches = list(matcher.find_iter("A1 B2 C3"))
        assert len(matches) == 3
        assert matches[0].group() == "1"
    
    def test_substitute(self) -> None:
        """Test substitution."""
        matcher = RegexMatcher(r"\d+")
        result = matcher.substitute("NUM", "Order 123 and 456")
        assert result == "Order NUM and NUM"
    
    def test_split(self) -> None:
        """Test splitting on pattern."""
        matcher = RegexMatcher(r"[,;]\s*")
        result = matcher.split("a, b; c, d")
        assert result == ["a", "b", "c", "d"]
    
    def test_case_insensitive(self) -> None:
        """Test case-insensitive matching."""
        matcher = RegexMatcher(r"hello", re.IGNORECASE)
        assert matcher.search("HELLO World") is not None


# =============================================================================
# PATTERN LIBRARY TESTS
# =============================================================================

@pytest.mark.skipif(not LAB_AVAILABLE, reason="Lab module not available")
class TestPatternLibrary:
    """Tests for the PatternLibrary class."""
    
    @pytest.fixture
    def library(self) -> PatternLibrary:
        """Create a pattern library instance."""
        return PatternLibrary()
    
    def test_extract_emails(self, library: PatternLibrary) -> None:
        """Test email extraction."""
        text = "Contact: john@example.com and jane@test.co.uk"
        emails = library.extract_emails(text)
        assert "john@example.com" in emails
        assert "jane@test.co.uk" in emails
    
    def test_extract_urls(self, library: PatternLibrary) -> None:
        """Test URL extraction."""
        text = "Visit https://example.com or http://test.org"
        urls = library.extract_urls(text)
        assert any("example.com" in url for url in urls)
    
    def test_extract_dates_iso(self, library: PatternLibrary) -> None:
        """Test ISO date extraction."""
        text = "Meeting on 2025-01-17 and 2025-02-20"
        dates = library.extract_dates_iso(text)
        assert "2025-01-17" in dates
        assert "2025-02-20" in dates
    
    def test_extract_integers(self, library: PatternLibrary) -> None:
        """Test integer extraction."""
        text = "Values: 42, -17, and 100"
        integers = library.extract_integers(text)
        assert 42 in integers
        assert -17 in integers
    
    def test_extract_words(self, library: PatternLibrary) -> None:
        """Test word extraction."""
        text = "Hello, World! How are you?"
        words = library.extract_words(text)
        assert "Hello" in words
        assert "World" in words


# =============================================================================
# PATTERN VALIDATOR TESTS
# =============================================================================

@pytest.mark.skipif(not LAB_AVAILABLE, reason="Lab module not available")
class TestPatternValidator:
    """Tests for the PatternValidator class."""
    
    @pytest.fixture
    def validator(self) -> PatternValidator:
        """Create a validator instance."""
        return PatternValidator()
    
    def test_valid_email(self, validator: PatternValidator) -> None:
        """Test valid email validation."""
        assert validator.is_valid_email("user@example.com") is True
        assert validator.is_valid_email("john.doe@company.co.uk") is True
    
    def test_invalid_email(self, validator: PatternValidator) -> None:
        """Test invalid email validation."""
        assert validator.is_valid_email("invalid") is False
        assert validator.is_valid_email("@nodomain.com") is False
    
    def test_valid_url(self, validator: PatternValidator) -> None:
        """Test valid URL validation."""
        assert validator.is_valid_url("https://example.com") is True
        assert validator.is_valid_url("http://test.org/page") is True
    
    def test_invalid_url(self, validator: PatternValidator) -> None:
        """Test invalid URL validation."""
        assert validator.is_valid_url("ftp://invalid.com") is False
        assert validator.is_valid_url("not-a-url") is False
    
    def test_valid_iso_date(self, validator: PatternValidator) -> None:
        """Test valid ISO date validation."""
        assert validator.is_valid_iso_date("2025-01-17") is True
        assert validator.is_valid_iso_date("2025-12-31") is True
    
    def test_invalid_iso_date(self, validator: PatternValidator) -> None:
        """Test invalid ISO date validation."""
        assert validator.is_valid_iso_date("2025/01/17") is False
        assert validator.is_valid_iso_date("17-01-2025") is False


# =============================================================================
# UNICODE HANDLER TESTS
# =============================================================================

@pytest.mark.skipif(not LAB_AVAILABLE, reason="Lab module not available")
class TestUnicodeHandler:
    """Tests for the UnicodeHandler class."""
    
    @pytest.fixture
    def handler(self) -> UnicodeHandler:
        """Create a Unicode handler instance."""
        return UnicodeHandler()
    
    def test_encode_utf8(self, handler: UnicodeHandler) -> None:
        """Test UTF-8 encoding."""
        result = handler.encode("café")
        assert isinstance(result, bytes)
        assert result == b"caf\xc3\xa9"
    
    def test_decode_utf8(self, handler: UnicodeHandler) -> None:
        """Test UTF-8 decoding."""
        result = handler.decode(b"caf\xc3\xa9")
        assert result == "café"
    
    def test_normalise_nfc(self, handler: UnicodeHandler) -> None:
        """Test NFC normalisation."""
        composed = handler.normalise_nfc("café")
        decomposed = handler.normalise_nfc("cafe\u0301")
        assert composed == decomposed
    
    def test_remove_accents(self, handler: UnicodeHandler) -> None:
        """Test accent removal."""
        result = handler.remove_accents("Crème brûlée")
        assert result == "Creme brulee"
    
    def test_get_char_info(self, handler: UnicodeHandler) -> None:
        """Test character information retrieval."""
        info = handler.get_char_info("é")
        assert info["hex"] == "U+00E9"
        assert "LATIN" in info["name"]


# =============================================================================
# TEXT CLEANER TESTS
# =============================================================================

@pytest.mark.skipif(not LAB_AVAILABLE, reason="Lab module not available")
class TestTextCleaner:
    """Tests for the TextCleaner class."""
    
    def test_remove_html(self) -> None:
        """Test HTML tag removal."""
        cleaner = TextCleaner()
        result = cleaner.clean("<p>Hello</p>")
        assert "<" not in result.cleaned
        assert "Hello" in result.cleaned
    
    def test_normalise_whitespace(self) -> None:
        """Test whitespace normalisation."""
        cleaner = TextCleaner()
        result = cleaner.clean("Hello   World")
        assert result.cleaned == "Hello World"
    
    def test_remove_urls(self) -> None:
        """Test URL removal when configured."""
        config = CleaningConfig(remove_urls=True)
        cleaner = TextCleaner(config)
        result = cleaner.clean("Visit https://example.com today")
        assert "https://" not in result.cleaned
    
    def test_cleaning_result_structure(self) -> None:
        """Test CleaningResult contains expected fields."""
        cleaner = TextCleaner()
        result = cleaner.clean("<p>Test</p>")
        assert isinstance(result, CleaningResult)
        assert result.original == "<p>Test</p>"
        assert len(result.changes) > 0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

@pytest.mark.skipif(not LAB_AVAILABLE, reason="Lab module not available")
class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_extract_and_validate_emails(self) -> None:
        """Test extracting emails and validating them."""
        library = PatternLibrary()
        validator = PatternValidator()
        
        text = "Contact: valid@example.com and invalid@"
        emails = library.extract_emails(text)
        
        valid_emails = [e for e in emails if validator.is_valid_email(e)]
        assert len(valid_emails) >= 1
    
    def test_clean_and_extract(self) -> None:
        """Test cleaning text then extracting data."""
        cleaner = TextCleaner()
        library = PatternLibrary()
        
        html = "<p>Order 123 from user@test.com</p>"
        result = cleaner.clean(html)
        
        integers = library.extract_integers(result.cleaned)
        assert 123 in integers
