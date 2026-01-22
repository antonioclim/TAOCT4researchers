"""API consumption patterns for research data acquisition.

The module focuses on pagination, authentication headers, rate limiting and
reproducible request tracing. The demonstrations are self-contained and the
test suite uses response mocking, so no external network access is required.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

import requests


@dataclass(frozen=True)
class Page:
    items: List[Dict[str, Any]]
    next_url: Optional[str]


class RateLimiter:
    """A minimal request spacer.

    It enforces a lower bound on the time between consecutive requests.
    """

    def __init__(self, min_interval_s: float) -> None:
        if min_interval_s < 0:
            raise ValueError("min_interval_s must be non-negative")
        self._min_interval_s = min_interval_s
        self._last_t: Optional[float] = None

    def wait(self) -> None:
        now = time.monotonic()
        if self._last_t is None:
            self._last_t = now
            return
        elapsed = now - self._last_t
        remaining = self._min_interval_s - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_t = time.monotonic()


class ApiClient:
    """A small HTTP client with explicit pagination."""

    def __init__(
        self,
        session: Optional[requests.Session] = None,
        *,
        auth_header_factory: Optional[Callable[[], Mapping[str, str]]] = None,
        rate_limiter: Optional[RateLimiter] = None,
        timeout_s: float = 10.0,
    ) -> None:
        self._session = session or requests.Session()
        self._auth_header_factory = auth_header_factory
        self._rate_limiter = rate_limiter
        self._timeout_s = timeout_s

    def _headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {"Accept": "application/json"}
        if self._auth_header_factory is not None:
            headers.update(dict(self._auth_header_factory()))
        return headers

    def fetch_page(self, url: str) -> Page:
        if self._rate_limiter is not None:
            self._rate_limiter.wait()
        resp = self._session.get(url, headers=self._headers(), timeout=self._timeout_s)
        resp.raise_for_status()
        payload = resp.json()

        items = list(payload.get("items", []))
        next_url = payload.get("next")
        if next_url is not None and not isinstance(next_url, str):
            raise TypeError("'next' must be a string URL or null")
        return Page(items=items, next_url=next_url)

    def iter_items(self, first_url: str, *, max_pages: int = 50) -> Iterable[Dict[str, Any]]:
        url: Optional[str] = first_url
        pages = 0
        while url is not None:
            pages += 1
            if pages > max_pages:
                raise RuntimeError("Pagination exceeded max_pages; possible loop")
            page = self.fetch_page(url)
            for item in page.items:
                yield item
            url = page.next_url


def _demo() -> None:
    print("Demo uses placeholder URLs; run tests for functional verification.")
    example = {
        "pagination": "https://example.invalid/api/items?page=1",
        "auth": "https://example.invalid/api/protected",
    }
    print(json.dumps(example, indent=2))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="lab_12_01_api_consumption")
    parser.add_argument("--demo", nargs="?", const="all", help="Run small demo (no network calls)")
    args = parser.parse_args(argv)

    if args.demo is not None:
        _demo()
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
