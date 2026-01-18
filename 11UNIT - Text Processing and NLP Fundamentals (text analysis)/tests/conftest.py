"""
Pytest Configuration for Unit 11 Tests

This module provides shared fixtures and configuration for testing
the text processing and NLP laboratories.

Author: Antonio Clim
Version: 1.0.0
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

# Add lab directory to path for imports
LAB_DIR = Path(__file__).parent.parent / "lab"
sys.path.insert(0, str(LAB_DIR))


# =============================================================================
# SAMPLE TEXT FIXTURES
# =============================================================================

@pytest.fixture
def sample_text() -> str:
    """Provide sample text for testing."""
    return """
    The quick brown fox jumps over the lazy dog.
    This is a test sentence with numbers like 123 and 456.
    Contact us at info@example.com or visit https://example.com
    """


@pytest.fixture
def sample_html() -> str:
    """Provide sample HTML text for cleaning tests."""
    return """
    <html>
    <body>
    <h1>Welcome</h1>
    <p>This is a <strong>test</strong> paragraph.</p>
    <p>Contact: user@test.com</p>
    </body>
    </html>
    """


@pytest.fixture
def sample_log_lines() -> list[str]:
    """Provide sample log lines for parsing tests."""
    return [
        "[2025-01-17 10:30:00] INFO Server: Started successfully",
        "[2025-01-17 10:30:05] DEBUG Database: Connection established",
        "[2025-01-17 10:30:10] ERROR Network: Connection timeout",
        "[2025-01-17 10:30:15] WARN Memory: Usage at 80%",
    ]


@pytest.fixture
def unicode_samples() -> dict[str, str]:
    """Provide Unicode text samples for normalisation tests."""
    return {
        "composed": "café",
        "decomposed": "cafe\u0301",
        "accented": "Crème brûlée",
        "multilingual": "Hello 世界 مرحبا",
    }


# =============================================================================
# CORPUS FIXTURES
# =============================================================================

@pytest.fixture
def sample_corpus() -> list[str]:
    """Provide sample corpus for NLP tests."""
    return [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks for pattern recognition.",
        "Natural language processing enables computers to understand text.",
        "Data science combines statistics and programming.",
        "Python is popular for machine learning applications.",
    ]


@pytest.fixture
def tokenised_corpus() -> list[list[str]]:
    """Provide pre-tokenised corpus for feature extraction tests."""
    return [
        ["machine", "learning", "subset", "artificial", "intelligence"],
        ["deep", "learning", "neural", "networks", "pattern"],
        ["natural", "language", "processing", "computers", "text"],
        ["data", "science", "statistics", "programming"],
        ["python", "popular", "machine", "learning", "applications"],
    ]


# =============================================================================
# EMAIL AND URL FIXTURES
# =============================================================================

@pytest.fixture
def email_samples() -> dict[str, bool]:
    """Provide email samples with expected validity."""
    return {
        "user@example.com": True,
        "john.doe@company.co.uk": True,
        "name+tag@domain.org": True,
        "invalid-email": False,
        "@nodomain.com": False,
        "user@.com": False,
    }


@pytest.fixture
def url_samples() -> dict[str, bool]:
    """Provide URL samples with expected validity."""
    return {
        "https://example.com": True,
        "http://test.org/page": True,
        "https://sub.domain.com/path?q=1": True,
        "ftp://invalid.com": False,
        "not-a-url": False,
    }


# =============================================================================
# STOPWORD FIXTURES
# =============================================================================

@pytest.fixture
def english_stopwords() -> set[str]:
    """Provide basic English stopwords for testing without NLTK."""
    return {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "must",
        "i", "you", "he", "she", "it", "we", "they", "this", "that",
    }


# =============================================================================
# HELPER FIXTURES
# =============================================================================

@pytest.fixture
def temp_text_file(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a temporary text file for I/O tests."""
    file_path = tmp_path / "test_document.txt"
    content = "This is a test document.\nIt has multiple lines.\n"
    file_path.write_text(content, encoding="utf-8")
    yield file_path


@pytest.fixture
def regex_test_cases() -> list[tuple[str, str, list[str]]]:
    """Provide regex pattern test cases: (pattern, text, expected_matches)."""
    return [
        (r"\d+", "Order 123 and 456", ["123", "456"]),
        (r"\b\w+\b", "Hello, World!", ["Hello", "World"]),
        (r"[A-Z][a-z]+", "Hello World Test", ["Hello", "World", "Test"]),
        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
         "Contact: test@example.com", ["test@example.com"]),
    ]


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "nltk: marks tests requiring NLTK (deselect with '-m \"not nltk\"')"
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item]
) -> None:
    """Modify test collection to add markers based on test names."""
    for item in items:
        if "nltk" in item.nodeid.lower():
            item.add_marker(pytest.mark.nltk)
        if "slow" in item.nodeid.lower():
            item.add_marker(pytest.mark.slow)
