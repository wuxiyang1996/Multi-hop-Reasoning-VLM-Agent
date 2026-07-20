#!/usr/bin/env python3
"""Fail fast when Slurm exposes GPUs that CUDA cannot initialize."""

from __future__ import annotations

import argparse
import json
import os

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("expected", type=int)
    args = parser.parse_args()

    count = torch.cuda.device_count()
    if count != args.expected:
        raise SystemExit(
            f"expected {args.expected} CUDA devices, torch reported {count}; "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}"
        )

    # device_count() can agree with the mask even when a node has one or more
    # unusable GPUs.  Querying every property forces CUDA to initialize each
    # device and catches that failure before model staging or evaluation.
    devices = []
    for index in range(count):
        props = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": props.name,
                "total_memory": props.total_memory,
                "capability": list(torch.cuda.get_device_capability(index)),
            }
        )
    print(json.dumps({"status": "ok", "devices": devices}, sort_keys=True))


if __name__ == "__main__":
    main()
