"""Runtime adapter that enables WebShop's native synthetic goal generator."""

from __future__ import annotations

import random
from typing import Any


def install_synthetic_goal_mode(app_module: Any) -> None:
    """Patch an imported WebShop app module without modifying vendor files.

    The upstream app imports ``load_products`` and ``get_goals`` into module
    globals and calls both without exposing ``human_goals``.  Wrapping those
    two references activates the already-shipped synthetic goal path while
    leaving search, reward, and page execution unchanged.
    """

    if getattr(app_module, "_V14_SYNTHETIC_GOAL_MODE", False):
        return
    original_load_products = app_module.load_products
    original_get_goals = app_module.get_goals

    def load_products(*args: Any, **kwargs: Any) -> Any:
        # Upstream seeds only after load_products(), although product-price
        # generation inside that call consumes randomness.  Seed first so
        # price thresholds and instruction text are reproducible too.
        random.seed(int(app_module.GOAL_SEED))
        kwargs["human_goals"] = False
        return original_load_products(*args, **kwargs)

    def get_goals(*args: Any, **kwargs: Any) -> Any:
        kwargs["human_goals"] = False
        return original_get_goals(*args, **kwargs)

    app_module.load_products = load_products
    app_module.get_goals = get_goals
    # Fail closed if this is installed after a human-goal request initialized
    # the global server state.
    app_module.all_products = None
    app_module.product_item_dict = None
    app_module.product_prices = None
    app_module.attribute_to_asins = None
    app_module.search_engine = None
    app_module.goals = None
    app_module.weights = None
    app_module.user_sessions = {}
    app_module._V14_SYNTHETIC_GOAL_MODE = True


__all__ = ["install_synthetic_goal_mode"]
