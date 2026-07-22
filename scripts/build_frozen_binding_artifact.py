#!/usr/bin/env python3
"""Run one-shot adaptation once and freeze its mechanically checked bindings."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import runpy
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.artifact_io import (  # noqa: E402
    adaptation_example_view,
    load_first_source_motif,
    write_frozen_binding_artifact,
)
from motif_transfer.contracts import Lifecycle  # noqa: E402
from motif_transfer.frozen_motif_agent import (  # noqa: E402
    FrozenJSONMotifAgent,
    OpenAICompatibleBackend,
)


def _load_key(path: Path, name: str) -> None:
    value = runpy.run_path(str(path)).get(name)
    if value and not os.environ.get(name):
        os.environ[name] = str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--adaptation-example", type=Path, required=True)
    parser.add_argument("--source-motif", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--max-candidates", type=int, default=4)
    parser.add_argument("--skip-alpha-control", action="store_true")
    args = parser.parse_args()

    _load_key(args.keys, "OPENAI_API_KEY")
    example = adaptation_example_view(args.adaptation_example)
    motif = load_first_source_motif(args.source_motif, status=Lifecycle.GENERIC_ONLY)
    backend = OpenAICompatibleBackend(
        "https://us.api.openai.com/v1",
        {"binding": args.model},
        api_key_env="OPENAI_API_KEY",
        json_mode=True,
        temperature=None,
    )
    agent = FrozenJSONMotifAgent(
        backend, allowed_verifier_ids=("official_transition_and_outcome",),
    )
    artifact = agent.build_binding_artifact(
        motif,
        example,
        max_candidates=args.max_candidates,
        run_alpha_control=not args.skip_alpha_control,
        induction_repetitions=args.repetitions,
    )
    write_frozen_binding_artifact(args.output, artifact)
    print(f"{artifact.status.value} {len(artifact.bindings)} {artifact.artifact_hash}")


if __name__ == "__main__":
    main()
