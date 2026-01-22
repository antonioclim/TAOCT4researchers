from __future__ import annotations

from lab.lab_12_02_web_scraping_flask import create_app, parse_links


def test_parse_links_extracts_title_and_href():
    html = "<html><body><a href='x'>X</a><a href='y'>Y</a></body></html>"
    items = parse_links(html)
    assert [(i.title, i.href) for i in items] == [("X", "x"), ("Y", "y")]


def test_flask_links_endpoint_returns_json():
    app = create_app()
    client = app.test_client()
    html = "<a href='https://a'>A</a>"
    r = client.get("/links", query_string={"html": html})
    assert r.status_code == 200
    assert r.get_json() == [{"title": "A", "href": "https://a"}]
