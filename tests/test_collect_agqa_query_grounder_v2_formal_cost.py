from scripts.collect_agqa_query_grounder_v2_formal_cost import parse_sacct


def test_cost_parser_counts_array_gpu_seconds_without_steps():
    text = "\n".join((
        "10|download|COMPLETED|12|billing=4,cpu=4,mem=16G",
        "11|parser|COMPLETED|20|billing=1,cpu=6,gres/gpu=1,gres/gpu:l40s=1",
        "12_0|slowfast|COMPLETED|3|cpu=6,gres/gpu=1,gres/gpu:l40s=1",
        "12_0.batch|batch|COMPLETED|3|cpu=6,gres/gpu=1,gres/gpu:l40s=1",
        "12_1|slowfast|COMPLETED|4|cpu=6,gres/gpu=1,gres/gpu:l40s=1",
    ))
    rows, totals = parse_sacct(text, {"10": "download", "11": "parser", "12": "slowfast"})
    assert len(rows) == 4
    assert totals["total_gpu_seconds"] == 27
    assert totals["gpu_seconds_by_type"] == {"l40s": 27}
