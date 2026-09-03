from motif_transfer.clevrer_descriptive_compiler import compile_descriptive_question


def test_count_entry_and_temporal_collision() -> None:
    assert compile_descriptive_question("How many objects enter the scene?", "count") == [
        "events", "objects", "filter_in", "count",
    ]
    assert compile_descriptive_question(
        "How many collisions happen after the blue cube enters the scene?", "count",
    ) == [
        "events", "events", "objects", "blue", "filter_color", "cube", "filter_shape",
        "unique", "filter_in", "unique", "filter_after", "objects", "filter_collision", "count",
    ]


def test_dynamic_and_collision_partner_queries() -> None:
    assert compile_descriptive_question(
        "Are there any stationary purple objects when the video ends?", "exist",
    ) == [
        "objects", "purple", "filter_color", "events", "filter_end", "query_frame",
        "filter_stationary", "exist",
    ]
    assert compile_descriptive_question(
        "What material is the first object to collide with the cyan sphere?", "query_material",
    ) == [
        "events", "objects", "cyan", "filter_color", "sphere", "filter_shape", "unique",
        "filter_collision", "first", "filter_order", "objects", "cyan", "filter_color",
        "sphere", "filter_shape", "unique", "query_collision_partner", "query_material",
    ]


def test_ordered_entry_attribute_query() -> None:
    assert compile_descriptive_question(
        "What is the shape of the last object that enters the scene?", "query_shape",
    ) == [
        "events", "objects", "filter_in", "last", "filter_order", "query_object", "query_shape",
    ]
