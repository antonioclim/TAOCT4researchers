"""Web scraping and a small Flask wrapper API."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Optional

import requests
from bs4 import BeautifulSoup
from flask import Flask, jsonify, request


@dataclass(frozen=True)
class ScrapedItem:
    title: str
    href: str


def parse_links(html: str, *, selector: str = "a") -> List[ScrapedItem]:
    soup = BeautifulSoup(html, "html.parser")
    out: List[ScrapedItem] = []
    for a in soup.select(selector):
        href = a.get("href") or ""
        title = (a.get_text() or "").strip()
        if not title or not href:
            continue
        out.append(ScrapedItem(title=title, href=href))
    return out


def scrape(url: str, *, timeout_s: float = 10.0) -> List[ScrapedItem]:
    resp = requests.get(url, timeout=timeout_s, headers={"User-Agent": "12UNIT-demo/1.0"})
    resp.raise_for_status()
    return parse_links(resp.text)


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.get("/links")
    def links():
        html = request.args.get("html")
        if html is None:
            return jsonify({"error": "missing 'html' query parameter"}), 400
        selector = request.args.get("selector", "a")
        items = parse_links(html, selector=selector)
        return jsonify([{"title": i.title, "href": i.href} for i in items])

    return app


def _demo() -> None:
    html = "<html><body><a href='https://example.org'>Example</a></body></html>"
    items = parse_links(html)
    print("Parsed:", items)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="lab_12_02_web_scraping_flask")
    parser.add_argument("--demo", nargs="?", const="all", help="Run local demo")
    parser.add_argument("--serve", action="store_true", help="Run Flask server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args(argv)

    if args.demo is not None:
        _demo()
        return 0
    if args.serve:
        app = create_app()
        app.run(host=args.host, port=args.port, debug=False)
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
