"""
Lab 11_01: Regular Expressions and String Operations

This laboratory develops proficiency with Python's re module and advanced
string manipulation techniques for text processing applications.

Sections:
    §1. String Methods (~80 lines)
    §2. Regex Fundamentals (~120 lines)
    §3. Advanced Patterns (~120 lines)
    §4. Unicode Handling (~80 lines)
    §5. Text Cleaning Pipeline (~100 lines)

Learning Objectives:
    LO1: Explain regular expression syntax, metacharacters and matching semantics
    LO2: Implement text extraction and validation using regex patterns

Duration: 50 minutes
Difficulty: ★★★☆☆

Author: Antonio Clim
Version: 1.0.0
Date: January 2025
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Pattern

if TYPE_CHECKING:
    from collections.abc import Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# §1. STRING METHODS
# =============================================================================

class StringOperations:
    """
    Demonstrates essential string methods for text processing.
    
    This class encapsulates common string manipulation operations used
    in text preprocessing pipelines.
    
    Attributes:
        text: The input text to process.
    
    Example:
        >>> ops = StringOperations("  Hello World  ")
        >>> ops.normalise_whitespace()
        'Hello World'
    """
    
    def __init__(self, text: str) -> None:
        """
        Initialise with input text.
        
        Args:
            text: The input text to process.
        """
        self.text = text
        logger.debug("StringOperations initialised with text of length %d", len(text))
    
    def to_lowercase(self) -> str:
        """
        Convert text to lowercase.
        
        Returns:
            Lowercase version of the text.
        """
        return self.text.lower()
    
    def to_uppercase(self) -> str:
        """
        Convert text to uppercase.
        
        Returns:
            Uppercase version of the text.
        """
        return self.text.upper()
    
    def to_titlecase(self) -> str:
        """
        Convert text to title case.
        
        Returns:
            Title case version of the text.
        """
        return self.text.title()
    
    def normalise_whitespace(self) -> str:
        """
        Normalise whitespace by stripping edges and collapsing internal spaces.
        
        Returns:
            Text with normalised whitespace.
        """
        return " ".join(self.text.split())
    
    def split_on_delimiter(self, delimiter: str, maxsplit: int = -1) -> list[str]:
        """
        Split text on a specified delimiter.
        
        Args:
            delimiter: The string to split on.
            maxsplit: Maximum number of splits (-1 for unlimited).
        
        Returns:
            List of substrings.
        """
        return self.text.split(delimiter, maxsplit)
    
    def split_into_lines(self, keepends: bool = False) -> list[str]:
        """
        Split text into lines.
        
        Args:
            keepends: Whether to keep line ending characters.
        
        Returns:
            List of lines.
        """
        return self.text.splitlines(keepends)
    
    def join_with_delimiter(self, parts: list[str], delimiter: str = " ") -> str:
        """
        Join a list of strings with a delimiter.
        
        Args:
            parts: List of strings to join.
            delimiter: String to insert between parts.
        
        Returns:
            Joined string.
        """
        return delimiter.join(parts)
    
    def count_occurrences(self, substring: str) -> int:
        """
        Count occurrences of a substring.
        
        Args:
            substring: The substring to count.
        
        Returns:
            Number of non-overlapping occurrences.
        """
        return self.text.count(substring)
    
    def find_position(self, substring: str, start: int = 0) -> int:
        """
        Find the position of a substring.
        
        Args:
            substring: The substring to find.
            start: Starting index for search.
        
        Returns:
            Index of first occurrence, or -1 if not found.
        """
        return self.text.find(substring, start)
    
    def replace_substring(self, old: str, new: str, count: int = -1) -> str:
        """
        Replace occurrences of a substring.
        
        Args:
            old: Substring to replace.
            new: Replacement string.
            count: Maximum replacements (-1 for all).
        
        Returns:
            Text with replacements made.
        """
        return self.text.replace(old, new, count)
    
    def check_prefix(self, prefix: str) -> bool:
        """
        Check if text starts with a prefix.
        
        Args:
            prefix: The prefix to check.
        
        Returns:
            True if text starts with prefix.
        """
        return self.text.startswith(prefix)
    
    def check_suffix(self, suffix: str) -> bool:
        """
        Check if text ends with a suffix.
        
        Args:
            suffix: The suffix to check.
        
        Returns:
            True if text ends with suffix.
        """
        return self.text.endswith(suffix)


# =============================================================================
# §2. REGEX FUNDAMENTALS
# =============================================================================

class RegexMatcher:
    """
    Provides fundamental regex matching operations.
    
    This class wraps Python's re module to provide a clean interface
    for common pattern matching tasks.
    
    Attributes:
        pattern: The compiled regex pattern.
        flags: Regex flags applied to the pattern.
    
    Example:
        >>> matcher = RegexMatcher(r'\\d+')
        >>> matcher.find_all("Price: 100 euros, 200 dollars")
        ['100', '200']
    """
    
    def __init__(self, pattern: str, flags: int = 0) -> None:
        """
        Initialise with a regex pattern.
        
        Args:
            pattern: The regex pattern string.
            flags: Regex flags (e.g., re.IGNORECASE).
        
        Raises:
            re.error: If the pattern is invalid.
        """
        self.pattern_string = pattern
        self.flags = flags
        self.pattern: Pattern[str] = re.compile(pattern, flags)
        logger.debug("Compiled pattern: %s with flags: %d", pattern, flags)
    
    def search(self, text: str) -> re.Match[str] | None:
        """
        Search for the first match in text.
        
        Args:
            text: The text to search.
        
        Returns:
            Match object if found, None otherwise.
        """
        return self.pattern.search(text)
    
    def match(self, text: str) -> re.Match[str] | None:
        """
        Match pattern at the beginning of text.
        
        Args:
            text: The text to match against.
        
        Returns:
            Match object if pattern matches at start, None otherwise.
        """
        return self.pattern.match(text)
    
    def fullmatch(self, text: str) -> re.Match[str] | None:
        """
        Match pattern against the entire text.
        
        Args:
            text: The text to match against.
        
        Returns:
            Match object if entire text matches, None otherwise.
        """
        return self.pattern.fullmatch(text)
    
    def find_all(self, text: str) -> list[str]:
        """
        Find all non-overlapping matches.
        
        Args:
            text: The text to search.
        
        Returns:
            List of matched strings.
        """
        return self.pattern.findall(text)
    
    def find_iter(self, text: str) -> Iterator[re.Match[str]]:
        """
        Iterate over all matches with position information.
        
        Args:
            text: The text to search.
        
        Yields:
            Match objects for each match.
        """
        return self.pattern.finditer(text)
    
    def substitute(self, replacement: str, text: str, count: int = 0) -> str:
        """
        Replace matches with a replacement string.
        
        Args:
            replacement: The replacement string (may include backreferences).
            text: The text to modify.
            count: Maximum replacements (0 for all).
        
        Returns:
            Modified text.
        """
        return self.pattern.sub(replacement, text, count)
    
    def split(self, text: str, maxsplit: int = 0) -> list[str]:
        """
        Split text on pattern matches.
        
        Args:
            text: The text to split.
            maxsplit: Maximum number of splits (0 for unlimited).
        
        Returns:
            List of substrings.
        """
        return self.pattern.split(text, maxsplit)


class PatternLibrary:
    """
    Collection of commonly used regex patterns.
    
    Provides pre-compiled patterns for common extraction tasks
    such as emails, URLs, dates and phone numbers.
    
    Example:
        >>> lib = PatternLibrary()
        >>> lib.extract_emails("Contact: user@example.com")
        ['user@example.com']
    """
    
    # Pattern definitions
    EMAIL_PATTERN = r"\b[\w.+-]+@[\w.-]+\.\w{2,}\b"
    URL_PATTERN = r"https?://[\w.-]+(?:/[\w./?&=-]*)?"
    DATE_ISO_PATTERN = r"\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])"
    DATE_UK_PATTERN = r"(?:0[1-9]|[12]\d|3[01])/(?:0[1-9]|1[0-2])/\d{4}"
    PHONE_PATTERN = r"(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
    INTEGER_PATTERN = r"-?\d+"
    DECIMAL_PATTERN = r"-?\d+\.?\d*"
    WORD_PATTERN = r"\b\w+\b"
    SENTENCE_PATTERN = r"[^.!?]*[.!?]"
    
    def __init__(self) -> None:
        """Initialise pattern library with compiled patterns."""
        self._email = re.compile(self.EMAIL_PATTERN)
        self._url = re.compile(self.URL_PATTERN)
        self._date_iso = re.compile(self.DATE_ISO_PATTERN)
        self._date_uk = re.compile(self.DATE_UK_PATTERN)
        self._phone = re.compile(self.PHONE_PATTERN)
        self._integer = re.compile(self.INTEGER_PATTERN)
        self._decimal = re.compile(self.DECIMAL_PATTERN)
        self._word = re.compile(self.WORD_PATTERN)
        logger.debug("PatternLibrary initialised with %d patterns", 8)
    
    def extract_emails(self, text: str) -> list[str]:
        """Extract email addresses from text."""
        return self._email.findall(text)
    
    def extract_urls(self, text: str) -> list[str]:
        """Extract URLs from text."""
        return self._url.findall(text)
    
    def extract_dates_iso(self, text: str) -> list[str]:
        """Extract ISO format dates (YYYY-MM-DD) from text."""
        return self._date_iso.findall(text)
    
    def extract_dates_uk(self, text: str) -> list[str]:
        """Extract UK format dates (DD/MM/YYYY) from text."""
        return self._date_uk.findall(text)
    
    def extract_phone_numbers(self, text: str) -> list[str]:
        """Extract phone numbers from text."""
        return self._phone.findall(text)
    
    def extract_integers(self, text: str) -> list[int]:
        """Extract integers from text."""
        return [int(x) for x in self._integer.findall(text)]
    
    def extract_decimals(self, text: str) -> list[float]:
        """Extract decimal numbers from text."""
        return [float(x) for x in self._decimal.findall(text)]
    
    def extract_words(self, text: str) -> list[str]:
        """Extract words from text."""
        return self._word.findall(text)


# =============================================================================
# §3. ADVANCED PATTERNS
# =============================================================================

@dataclass
class NamedMatch:
    """
    Represents a match with named groups.
    
    Attributes:
        full_match: The complete matched string.
        groups: Dictionary mapping group names to matched values.
        start: Start position in source text.
        end: End position in source text.
    """
    
    full_match: str
    groups: dict[str, str]
    start: int
    end: int


class AdvancedPatternMatcher:
    """
    Demonstrates advanced regex features including lookahead,
    lookbehind, named groups and verbose patterns.
    
    Example:
        >>> matcher = AdvancedPatternMatcher()
        >>> matches = matcher.parse_log_entry(
        ...     "2025-01-17 14:30:25 ERROR Connection failed"
        ... )
        >>> matches.groups['level']
        'ERROR'
    """
    
    # Verbose pattern for log entries
    LOG_PATTERN = re.compile(r"""
        (?P<date>\d{4}-\d{2}-\d{2})\s+
        (?P<time>\d{2}:\d{2}:\d{2})\s+
        (?P<level>\w+)\s+
        (?P<message>.+)
    """, re.VERBOSE)
    
    # Pattern with lookahead: numbers followed by specific units
    CURRENCY_PATTERN = re.compile(r"(?<=[\$£€])\d+(?:\.\d{2})?")
    
    # Pattern with negative lookahead: words NOT followed by 'ing'
    NON_ING_WORD_PATTERN = re.compile(r"\b\w+\b(?!ing)")
    
    # Pattern with lookbehind: numbers preceded by currency symbols
    AMOUNT_PATTERN = re.compile(r"(?<=\$)\d+(?:\.\d+)?")
    
    def __init__(self) -> None:
        """Initialise the advanced pattern matcher."""
        logger.debug("AdvancedPatternMatcher initialised")
    
    def parse_log_entry(self, line: str) -> NamedMatch | None:
        """
        Parse a log entry using named groups.
        
        Args:
            line: A log line in format "YYYY-MM-DD HH:MM:SS LEVEL Message".
        
        Returns:
            NamedMatch object if pattern matches, None otherwise.
        """
        match = self.LOG_PATTERN.match(line)
        if match:
            return NamedMatch(
                full_match=match.group(0),
                groups=match.groupdict(),
                start=match.start(),
                end=match.end()
            )
        return None
    
    def extract_currency_amounts(self, text: str) -> list[str]:
        """
        Extract numeric amounts following currency symbols.
        
        Uses positive lookbehind to match numbers after $, £, or €
        without including the symbol in the match.
        
        Args:
            text: Text containing currency amounts.
        
        Returns:
            List of numeric amounts as strings.
        """
        return self.CURRENCY_PATTERN.findall(text)
    
    def extract_dollar_amounts(self, text: str) -> list[float]:
        """
        Extract dollar amounts from text.
        
        Args:
            text: Text containing dollar amounts.
        
        Returns:
            List of amounts as floats.
        """
        matches = self.AMOUNT_PATTERN.findall(text)
        return [float(m) for m in matches]
    
    def find_words_not_ending_ing(self, text: str) -> list[str]:
        """
        Find words that do not end with 'ing'.
        
        Uses negative lookahead to exclude -ing words.
        
        Args:
            text: Text to search.
        
        Returns:
            List of words not ending in 'ing'.
        """
        # This pattern is simplified; full implementation would be more complex
        words = re.findall(r"\b\w+\b", text)
        return [w for w in words if not w.endswith("ing")]


class PatternValidator:
    """
    Validates text against common format patterns.
    
    Provides validation methods that return boolean results
    for common data formats.
    
    Example:
        >>> validator = PatternValidator()
        >>> validator.is_valid_email("user@example.com")
        True
        >>> validator.is_valid_email("invalid-email")
        False
    """
    
    EMAIL_VALIDATOR = re.compile(
        r"^[\w.+-]+@[\w.-]+\.\w{2,}$"
    )
    
    URL_VALIDATOR = re.compile(
        r"^https?://[\w.-]+(?:/[\w./?&=-]*)?$"
    )
    
    DATE_ISO_VALIDATOR = re.compile(
        r"^\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])$"
    )
    
    POSTCODE_UK_VALIDATOR = re.compile(
        r"^[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}$",
        re.IGNORECASE
    )
    
    def is_valid_email(self, text: str) -> bool:
        """Check if text is a valid email address."""
        return bool(self.EMAIL_VALIDATOR.match(text))
    
    def is_valid_url(self, text: str) -> bool:
        """Check if text is a valid URL."""
        return bool(self.URL_VALIDATOR.match(text))
    
    def is_valid_iso_date(self, text: str) -> bool:
        """Check if text is a valid ISO date (YYYY-MM-DD)."""
        return bool(self.DATE_ISO_VALIDATOR.match(text))
    
    def is_valid_uk_postcode(self, text: str) -> bool:
        """Check if text is a valid UK postcode."""
        return bool(self.POSTCODE_UK_VALIDATOR.match(text))


# =============================================================================
# §4. UNICODE HANDLING
# =============================================================================

class UnicodeHandler:
    """
    Handles Unicode text processing including encoding,
    decoding and normalisation.
    
    Attributes:
        default_encoding: Default encoding for byte operations.
    
    Example:
        >>> handler = UnicodeHandler()
        >>> handler.normalise_nfc("cafe\\u0301")  # e + combining accent
        'café'  # precomposed é
    """
    
    NORMALISATION_FORMS = ("NFC", "NFD", "NFKC", "NFKD")
    
    def __init__(self, default_encoding: str = "utf-8") -> None:
        """
        Initialise Unicode handler.
        
        Args:
            default_encoding: Default encoding for byte operations.
        """
        self.default_encoding = default_encoding
        logger.debug("UnicodeHandler initialised with encoding: %s", default_encoding)
    
    def encode(self, text: str, encoding: str | None = None) -> bytes:
        """
        Encode string to bytes.
        
        Args:
            text: The string to encode.
            encoding: Encoding to use (defaults to instance default).
        
        Returns:
            Encoded bytes.
        """
        enc = encoding or self.default_encoding
        return text.encode(enc)
    
    def decode(self, data: bytes, encoding: str | None = None) -> str:
        """
        Decode bytes to string.
        
        Args:
            data: The bytes to decode.
            encoding: Encoding to use (defaults to instance default).
        
        Returns:
            Decoded string.
        """
        enc = encoding or self.default_encoding
        return data.decode(enc)
    
    def normalise(self, text: str, form: str = "NFC") -> str:
        """
        Apply Unicode normalisation.
        
        Args:
            text: The text to normalise.
            form: Normalisation form (NFC, NFD, NFKC, NFKD).
        
        Returns:
            Normalised text.
        
        Raises:
            ValueError: If form is not a valid normalisation form.
        """
        if form not in self.NORMALISATION_FORMS:
            raise ValueError(f"Invalid normalisation form: {form}")
        return unicodedata.normalize(form, text)
    
    def normalise_nfc(self, text: str) -> str:
        """Apply NFC normalisation (canonical composition)."""
        return unicodedata.normalize("NFC", text)
    
    def normalise_nfd(self, text: str) -> str:
        """Apply NFD normalisation (canonical decomposition)."""
        return unicodedata.normalize("NFD", text)
    
    def normalise_nfkc(self, text: str) -> str:
        """Apply NFKC normalisation (compatibility composition)."""
        return unicodedata.normalize("NFKC", text)
    
    def normalise_nfkd(self, text: str) -> str:
        """Apply NFKD normalisation (compatibility decomposition)."""
        return unicodedata.normalize("NFKD", text)
    
    def get_char_info(self, char: str) -> dict[str, str | int]:
        """
        Get Unicode information about a character.
        
        Args:
            char: A single character.
        
        Returns:
            Dictionary with character information.
        """
        return {
            "character": char,
            "code_point": ord(char),
            "hex": f"U+{ord(char):04X}",
            "name": unicodedata.name(char, "UNKNOWN"),
            "category": unicodedata.category(char),
        }
    
    def remove_accents(self, text: str) -> str:
        """
        Remove diacritical marks from text.
        
        Decomposes characters and removes combining marks.
        
        Args:
            text: The text to process.
        
        Returns:
            Text with accents removed.
        """
        nfkd = unicodedata.normalize("NFKD", text)
        return "".join(c for c in nfkd if not unicodedata.combining(c))
    
    def is_printable(self, text: str) -> bool:
        """Check if all characters in text are printable."""
        return all(c.isprintable() for c in text)
    
    def remove_non_printable(self, text: str) -> str:
        """Remove non-printable characters from text."""
        return "".join(c for c in text if c.isprintable())


# =============================================================================
# §5. TEXT CLEANING PIPELINE
# =============================================================================

@dataclass
class CleaningConfig:
    """
    Configuration for text cleaning pipeline.
    
    Attributes:
        normalise_unicode: Apply NFC normalisation.
        remove_html: Strip HTML tags.
        normalise_whitespace: Collapse whitespace.
        remove_non_printable: Remove non-printable characters.
        lowercase: Convert to lowercase.
        remove_urls: Remove URLs.
        remove_emails: Remove email addresses.
        fix_common_ocr_errors: Correct common OCR mistakes.
    """
    
    normalise_unicode: bool = True
    remove_html: bool = True
    normalise_whitespace: bool = True
    remove_non_printable: bool = True
    lowercase: bool = False
    remove_urls: bool = False
    remove_emails: bool = False
    fix_common_ocr_errors: bool = False


@dataclass
class CleaningResult:
    """
    Result of text cleaning operation.
    
    Attributes:
        original: Original input text.
        cleaned: Cleaned output text.
        changes: List of transformations applied.
    """
    
    original: str
    cleaned: str
    changes: list[str] = field(default_factory=list)


class TextCleaner:
    """
    Comprehensive text cleaning pipeline.
    
    Applies configurable cleaning transformations to text,
    suitable for preprocessing before NLP analysis.
    
    Attributes:
        config: Cleaning configuration.
    
    Example:
        >>> cleaner = TextCleaner()
        >>> result = cleaner.clean("<p>Hello  World</p>")
        >>> result.cleaned
        'Hello World'
    """
    
    # HTML tag pattern
    HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
    
    # URL pattern
    URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
    
    # Email pattern
    EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w.-]+\.\w+\b")
    
    # Multiple whitespace pattern
    WHITESPACE_PATTERN = re.compile(r"\s+")
    
    # Common OCR error mappings
    OCR_CORRECTIONS: dict[str, str] = {
        "rn": "m",   # Common OCR confusion
        "l1": "ll",  # 1 mistaken for l
        "0": "o",    # Context-dependent
        "vv": "w",   # v-v to w
    }
    
    def __init__(self, config: CleaningConfig | None = None) -> None:
        """
        Initialise text cleaner.
        
        Args:
            config: Cleaning configuration (uses defaults if None).
        """
        self.config = config or CleaningConfig()
        logger.debug("TextCleaner initialised with config: %s", self.config)
    
    def clean(self, text: str) -> CleaningResult:
        """
        Apply full cleaning pipeline to text.
        
        Args:
            text: The text to clean.
        
        Returns:
            CleaningResult with original, cleaned text and changes list.
        """
        result = CleaningResult(original=text, cleaned=text)
        
        if self.config.normalise_unicode:
            result.cleaned = self._normalise_unicode(result.cleaned)
            result.changes.append("unicode_normalised")
        
        if self.config.remove_html:
            result.cleaned = self._remove_html(result.cleaned)
            result.changes.append("html_removed")
        
        if self.config.remove_urls:
            result.cleaned = self._remove_urls(result.cleaned)
            result.changes.append("urls_removed")
        
        if self.config.remove_emails:
            result.cleaned = self._remove_emails(result.cleaned)
            result.changes.append("emails_removed")
        
        if self.config.remove_non_printable:
            result.cleaned = self._remove_non_printable(result.cleaned)
            result.changes.append("non_printable_removed")
        
        if self.config.normalise_whitespace:
            result.cleaned = self._normalise_whitespace(result.cleaned)
            result.changes.append("whitespace_normalised")
        
        if self.config.lowercase:
            result.cleaned = result.cleaned.lower()
            result.changes.append("lowercased")
        
        if self.config.fix_common_ocr_errors:
            result.cleaned = self._fix_ocr_errors(result.cleaned)
            result.changes.append("ocr_errors_fixed")
        
        logger.info("Cleaned text: %d → %d chars", len(text), len(result.cleaned))
        return result
    
    def _normalise_unicode(self, text: str) -> str:
        """Apply NFC Unicode normalisation."""
        return unicodedata.normalize("NFC", text)
    
    def _remove_html(self, text: str) -> str:
        """Remove HTML tags from text."""
        return self.HTML_TAG_PATTERN.sub("", text)
    
    def _remove_urls(self, text: str) -> str:
        """Remove URLs from text."""
        return self.URL_PATTERN.sub("", text)
    
    def _remove_emails(self, text: str) -> str:
        """Remove email addresses from text."""
        return self.EMAIL_PATTERN.sub("", text)
    
    def _remove_non_printable(self, text: str) -> str:
        """Remove non-printable characters."""
        return "".join(c for c in text if c.isprintable() or c in "\n\r\t")
    
    def _normalise_whitespace(self, text: str) -> str:
        """Normalise whitespace to single spaces."""
        return self.WHITESPACE_PATTERN.sub(" ", text).strip()
    
    def _fix_ocr_errors(self, text: str) -> str:
        """Apply common OCR error corrections."""
        result = text
        for error, correction in self.OCR_CORRECTIONS.items():
            result = result.replace(error, correction)
        return result


class TextCleaningPipeline:
    """
    Composable text cleaning pipeline with custom stages.
    
    Allows building custom pipelines by composing transformation
    functions in a specified order.
    
    Example:
        >>> pipeline = TextCleaningPipeline()
        >>> pipeline.add_stage("lowercase", str.lower)
        >>> pipeline.add_stage("strip", str.strip)
        >>> pipeline.process("  HELLO  ")
        'hello'
    """
    
    def __init__(self) -> None:
        """Initialise empty pipeline."""
        self._stages: list[tuple[str, Callable[[str], str]]] = []
        logger.debug("TextCleaningPipeline initialised")
    
    def add_stage(self, name: str, func: Callable[[str], str]) -> None:
        """
        Add a processing stage to the pipeline.
        
        Args:
            name: Name identifier for the stage.
            func: Transformation function (str -> str).
        """
        self._stages.append((name, func))
        logger.debug("Added stage: %s", name)
    
    def remove_stage(self, name: str) -> bool:
        """
        Remove a stage by name.
        
        Args:
            name: Name of the stage to remove.
        
        Returns:
            True if stage was removed, False if not found.
        """
        for i, (stage_name, _) in enumerate(self._stages):
            if stage_name == name:
                del self._stages[i]
                logger.debug("Removed stage: %s", name)
                return True
        return False
    
    def process(self, text: str) -> str:
        """
        Process text through all pipeline stages.
        
        Args:
            text: Input text.
        
        Returns:
            Processed text after all stages.
        """
        result = text
        for name, func in self._stages:
            result = func(result)
            logger.debug("Applied stage '%s': %d chars", name, len(result))
        return result
    
    def list_stages(self) -> list[str]:
        """Return list of stage names in order."""
        return [name for name, _ in self._stages]


# =============================================================================
# DEMONSTRATION FUNCTIONS
# =============================================================================

def demonstrate_string_operations() -> None:
    """Demonstrate string operation methods."""
    ops = StringOperations("  Hello, World! This is a TEST.  ")
    
    logger.info("Original: '%s'", ops.text)
    logger.info("Lowercase: '%s'", ops.to_lowercase())
    logger.info("Uppercase: '%s'", ops.to_uppercase())
    logger.info("Titlecase: '%s'", ops.to_titlecase())
    logger.info("Normalised: '%s'", ops.normalise_whitespace())


def demonstrate_regex_matching() -> None:
    """Demonstrate regex matching capabilities."""
    text = "Contact us at info@example.com or visit https://example.com"
    
    lib = PatternLibrary()
    logger.info("Emails found: %s", lib.extract_emails(text))
    logger.info("URLs found: %s", lib.extract_urls(text))


def demonstrate_text_cleaning() -> None:
    """Demonstrate text cleaning pipeline."""
    html_text = "<p>Hello   World!</p><br>Visit https://example.com"
    
    cleaner = TextCleaner(CleaningConfig(remove_urls=True))
    result = cleaner.clean(html_text)
    
    logger.info("Original: '%s'", result.original)
    logger.info("Cleaned: '%s'", result.cleaned)
    logger.info("Changes applied: %s", result.changes)


if __name__ == "__main__":
    # Run demonstrations when executed directly
    logger.info("=== String Operations ===")
    demonstrate_string_operations()
    
    logger.info("\n=== Regex Matching ===")
    demonstrate_regex_matching()
    
    logger.info("\n=== Text Cleaning ===")
    demonstrate_text_cleaning()
