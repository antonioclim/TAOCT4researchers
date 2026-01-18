"""
Exercise: Document Parser (Hard)

Parse structured documents using regex patterns.

Duration: 30-40 minutes
Difficulty: ★★★★☆

Author: Antonio Clim
Version: 1.0.0
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field


@dataclass
class DocumentSection:
    title: str
    level: int
    content: str
    subsections: list["DocumentSection"] = field(default_factory=list)


@dataclass
class ParsedDocument:
    title: str
    metadata: dict[str, str]
    sections: list[DocumentSection]


class MarkdownParser:
    """
    Parse Markdown-like documents into structured format.
    
    Handles:
    - Headers (# ## ###)
    - Metadata blocks (key: value)
    - Code blocks
    - Lists
    """
    
    HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    METADATA_PATTERN = re.compile(r"^(\w+):\s*(.+)$", re.MULTILINE)
    
    def parse(self, text: str) -> ParsedDocument:
        """Parse document into structured format."""
        # TODO: Implement
        pass
    
    def extract_headers(self, text: str) -> list[tuple[int, str]]:
        """Extract all headers with their levels."""
        # TODO: Implement
        pass
    
    def extract_code_blocks(self, text: str) -> list[str]:
        """Extract fenced code blocks."""
        # TODO: Implement
        pass


class LogParser:
    """
    Parse structured log files.
    
    Log format: [TIMESTAMP] LEVEL COMPONENT: Message
    """
    
    def parse_line(self, line: str) -> dict[str, str] | None:
        """Parse a single log line."""
        # TODO: Implement
        pass
    
    def filter_by_level(self, lines: list[str], level: str) -> list[dict[str, str]]:
        """Filter log lines by level (ERROR, WARN, INFO, DEBUG)."""
        # TODO: Implement
        pass


def run_tests() -> None:
    parser = MarkdownParser()
    headers = parser.extract_headers("# Title\n## Section\n### Subsection")
    if headers:
        assert headers[0] == (1, "Title")
    print("All tests passed!")


if __name__ == "__main__":
    run_tests()
