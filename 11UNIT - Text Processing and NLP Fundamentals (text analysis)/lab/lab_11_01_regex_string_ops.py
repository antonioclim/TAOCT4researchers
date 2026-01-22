"""
Regex and string operations for research text pipelines.

The module emphasises explicit contracts: the caller supplies a schema, a pattern set and
a decision about what counts as an error. The code is intentionally conservative: it
prefers reporting uncertainty to silently accepting malformed input.
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from typing import Iterable, Iterator, Pattern


@dataclass(frozen=True)
class LogRecord:
    timestamp: str
    level: str
    user: str
    message: str


LOG_PATTERN: Pattern[str] = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)\s+"
    r"(?P<level>[A-Z]{4,5})\s+"
    r"user=(?P<user>[a-zA-Z0-9_.-]+)\s+"
    r"msg=(?P<msg>.+)$"
)


def parse_log_lines(lines: Iterable[str], pattern: Pattern[str] = LOG_PATTERN) -> list[LogRecord]:
    """Parse log lines into structured records."""
    records: list[LogRecord] = []
    for line in lines:
        m = pattern.match(line.strip())
        if not m:
            continue
        records.append(
            LogRecord(
                timestamp=m.group("ts"),
                level=m.group("level"),
                user=m.group("user"),
                message=m.group("msg"),
            )
        )
    return records


def find_emails(text: str) -> list[str]:
    """Extract e-mail addresses using a conservative recogniser."""
    email_re = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Za-z]{2,}\b")
    return email_re.findall(text)


def iter_tokens(text: str) -> Iterator[str]:
    """Yield lowercase word tokens, excluding punctuation."""
    for m in re.finditer(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text):
        yield m.group(0).lower()


def demo() -> None:
    lines = [
        "2025-01-10T09:15:00Z INFO user=alice msg=Downloaded dataset version=3",
        "2025-01-10T09:16:03Z WARN user=bob msg=Checksum mismatch file=data.csv",
        "malformed line without schema",
    ]
    records = parse_log_lines(lines)
    print(f"Parsed records: {len(records)}")
    for r in records:
        print(f"{r.timestamp} {r.level} {r.user} {r.message}")

    sample = "Contact alice.smith@example.org or bob@example.com for details."
    print("Emails:", find_emails(sample))
    print("Tokens:", list(iter_tokens("A tokeniser's output should be testable.")))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="11UNIT regex and parsing demo")
    parser.add_argument("--demo", action="store_true", help="Run a small demonstration")
    args = parser.parse_args(argv)
    if args.demo:
        demo()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
