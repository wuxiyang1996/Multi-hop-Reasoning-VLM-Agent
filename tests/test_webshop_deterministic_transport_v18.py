from __future__ import annotations

from types import SimpleNamespace

from motif_transfer.webshop_deterministic_transport_v18 import (
    install_native_query_cache,
)


def test_native_query_cache_preserves_first_order_and_replays_it() -> None:
    calls = 0
    products = {key: {"asin": key} for key in ("a", "b", "c")}

    def unstable(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        order = ("b", "a") if calls == 1 else ("a", "c")
        return [products[key] for key in order]

    app = SimpleNamespace(get_top_n_product_from_keywords=unstable)
    install_native_query_cache(app)
    first = app.get_top_n_product_from_keywords(
        ["same", "query"], None, list(products.values()), products, None,
    )
    second = app.get_top_n_product_from_keywords(
        ["same", "query"], None, list(products.values()), products, None,
    )
    assert [row["asin"] for row in first] == ["b", "a"]
    assert [row["asin"] for row in second] == ["b", "a"]
    assert calls == 1


def test_native_query_cache_is_query_specific_and_idempotent() -> None:
    calls = 0
    products = {key: {"asin": key} for key in ("a", "b")}

    def native(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return list(products.values())

    app = SimpleNamespace(get_top_n_product_from_keywords=native)
    install_native_query_cache(app)
    install_native_query_cache(app)
    for query in (["one"], ["two"], ["one"]):
        app.get_top_n_product_from_keywords(
            query, None, list(products.values()), products, None,
        )
    assert calls == 2


def test_native_query_cache_replays_immutable_product_snapshot() -> None:
    products = [
        {"asin": "a", "name": "first"},
        {"asin": "b", "name": "second"},
    ]
    calls = 0

    def native(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return products

    app = SimpleNamespace(get_top_n_product_from_keywords=native)
    install_native_query_cache(app)
    product_dict = {row["asin"]: row for row in products}
    first = app.get_top_n_product_from_keywords(
        ["query"], None, products, product_dict, None,
    )
    first[0]["name"] = "mutated by prior arm"
    products[1]["name"] = "mutated shared native object"
    replay = app.get_top_n_product_from_keywords(
        ["query"], None, products, product_dict, None,
    )
    assert replay == [
        {"asin": "a", "name": "first"},
        {"asin": "b", "name": "second"},
    ]
    assert calls == 1
