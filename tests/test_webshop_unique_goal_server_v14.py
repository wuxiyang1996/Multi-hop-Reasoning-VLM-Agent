from __future__ import annotations

from types import SimpleNamespace
import random

from motif_transfer.webshop_unique_goal_server_v14 import install_synthetic_goal_mode


def test_server_adapter_forces_native_synthetic_goal_path_and_resets_state() -> None:
    calls = []

    def load_products(*args, **kwargs):
        calls.append(("load", args, kwargs))
        return "products"

    def get_goals(*args, **kwargs):
        calls.append(("goals", args, kwargs))
        return "goals"

    module = SimpleNamespace(
        load_products=load_products,
        get_goals=get_goals,
        all_products=["old"],
        product_item_dict={"old": 1},
        product_prices={"old": 1},
        attribute_to_asins={"old": 1},
        search_engine="old",
        goals=["old"],
        weights=[1],
        user_sessions={"old": {}},
        GOAL_SEED=233,
    )
    install_synthetic_goal_mode(module)
    assert module.load_products("x", human_goals=True) == "products"
    assert module.get_goals("p", "prices", human_goals=True) == "goals"
    assert calls[0][2]["human_goals"] is False
    assert calls[1][2]["human_goals"] is False
    assert module.search_engine is None
    assert module.goals is None
    assert module.user_sessions == {}


def test_server_adapter_seeds_before_product_price_generation() -> None:
    observed = []

    def load_products(*args, **kwargs):
        observed.append(random.random())

    module = SimpleNamespace(
        load_products=load_products,
        get_goals=lambda *args, **kwargs: None,
        GOAL_SEED=233,
        all_products=None,
        product_item_dict=None,
        product_prices=None,
        attribute_to_asins=None,
        search_engine=None,
        goals=None,
        weights=None,
        user_sessions={},
    )
    install_synthetic_goal_mode(module)
    module.load_products()
    first = observed[-1]
    random.seed(999)
    module.load_products()
    assert observed[-1] == first
