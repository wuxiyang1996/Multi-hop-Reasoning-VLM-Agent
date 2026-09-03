"""Deterministic matched-arm transport for native WebShop search results."""

from __future__ import annotations

from copy import deepcopy
from threading import Lock
from typing import Any


def install_native_query_cache(app_module: Any) -> None:
    """Replay the first native result ordering for every exact query.

    Some WebShop search backends return a different tie ordering on the first
    and subsequent identical queries.  Because experimental arms execute
    sequentially, that silently assigns the cold ordering to the first arm.
    This adapter leaves the first native search untouched, stores only its
    product snapshot, and replays an independent deep copy for every repeated
    query.  Caching ASINs alone is insufficient because upstream templates and
    page handlers share mutable product dictionaries across sequential arms.
    It never reads the goal, condition, reward, or source artifact.
    """

    if getattr(app_module, "_V18_NATIVE_QUERY_CACHE_INSTALLED", False):
        return
    original = app_module.get_top_n_product_from_keywords
    cache: dict[tuple[str, ...], tuple[dict[str, Any], ...]] = {}
    lock = Lock()

    def deterministic_search(
        keywords: Any, search_engine: Any, all_products: Any,
        product_item_dict: Any, attribute_to_asins: Any = None,
    ) -> list[Any]:
        key = tuple(map(str, keywords))
        with lock:
            cached = cache.get(key)
        if cached is None:
            products = original(
                keywords, search_engine, all_products, product_item_dict,
                attribute_to_asins,
            )
            snapshot = tuple(deepcopy(dict(row)) for row in products)
            with lock:
                cached = cache.setdefault(key, snapshot)
        return [deepcopy(row) for row in cached]

    app_module.get_top_n_product_from_keywords = deterministic_search
    app_module._V18_NATIVE_QUERY_CACHE = cache
    app_module._V18_NATIVE_QUERY_CACHE_INSTALLED = True


__all__ = ["install_native_query_cache"]
