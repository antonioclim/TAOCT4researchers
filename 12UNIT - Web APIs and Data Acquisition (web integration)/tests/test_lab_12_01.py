from __future__ import annotations

import json

import responses

from lab.lab_12_01_api_consumption import ApiClient, RateLimiter


@responses.activate
def test_pagination_iter_items_collects_all_pages():
    url1 = "https://api.test/items?page=1"
    url2 = "https://api.test/items?page=2"
    responses.add(responses.GET, url1, json={"items": [{"id": 1}, {"id": 2}], "next": url2}, status=200)
    responses.add(responses.GET, url2, json={"items": [{"id": 3}], "next": None}, status=200)

    client = ApiClient(rate_limiter=RateLimiter(0.0))
    items = list(client.iter_items(url1))
    assert [i["id"] for i in items] == [1, 2, 3]


@responses.activate
def test_auth_header_is_sent():
    url = "https://api.test/protected"

    def request_callback(request):
        assert request.headers.get("Authorization") == "Bearer TOKEN"
        body = json.dumps({"items": [], "next": None})
        return (200, {"Content-Type": "application/json"}, body)

    responses.add_callback(responses.GET, url, callback=request_callback, content_type="application/json")

    client = ApiClient(auth_header_factory=lambda: {"Authorization": "Bearer TOKEN"})
    page = client.fetch_page(url)
    assert page.items == []
