"""Solutions for hard_01_document_parser.py"""
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
    HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    METADATA_PATTERN = re.compile(r"^(\w+):\s*(.+)$", re.MULTILINE)
    CODE_BLOCK_PATTERN = re.compile(r"```[\w]*\n(.*?)```", re.DOTALL)
    
    def extract_headers(self, text: str) -> list[tuple[int, str]]:
        matches = self.HEADER_PATTERN.findall(text)
        return [(len(hashes), title) for hashes, title in matches]
    
    def extract_code_blocks(self, text: str) -> list[str]:
        return self.CODE_BLOCK_PATTERN.findall(text)
    
    def parse(self, text: str) -> ParsedDocument:
        headers = self.extract_headers(text)
        title = headers[0][1] if headers else "Untitled"
        metadata = dict(self.METADATA_PATTERN.findall(text))
        sections = [DocumentSection(h[1], h[0], "") for h in headers]
        return ParsedDocument(title=title, metadata=metadata, sections=sections)

class LogParser:
    PATTERN = re.compile(r"\[([^\]]+)\]\s+(\w+)\s+(\w+):\s*(.+)")
    
    def parse_line(self, line: str) -> dict[str, str] | None:
        match = self.PATTERN.match(line)
        if match:
            return {
                "timestamp": match.group(1),
                "level": match.group(2),
                "component": match.group(3),
                "message": match.group(4)
            }
        return None
    
    def filter_by_level(self, lines: list[str], level: str) -> list[dict[str, str]]:
        results = []
        for line in lines:
            parsed = self.parse_line(line)
            if parsed and parsed["level"] == level:
                results.append(parsed)
        return results
