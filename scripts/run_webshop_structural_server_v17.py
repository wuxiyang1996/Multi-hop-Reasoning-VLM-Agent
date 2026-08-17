#!/usr/bin/env python3
"""Launch WebShop synthetic goals for a frozen V17 seed."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.webshop_unique_goal_server_v14 import (  # noqa: E402
    install_synthetic_goal_mode,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vendor-root", type=Path,
        default=Path(
            "/fs/gamma-projects/vlm-robot/emnlp2026_download/workspace/vendor/WebShop"
        ),
    )
    parser.add_argument("--goal-seed", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3000)
    args = parser.parse_args()
    if not (args.vendor_root / "web_agent_site/app.py").exists():
        raise SystemExit(f"invalid WebShop vendor root: {args.vendor_root}")
    sys.path.insert(0, str(args.vendor_root))
    import web_agent_site.app as app_module

    # Must be assigned before the adapter initializes any server globals.
    app_module.GOAL_SEED = int(args.goal_seed)
    install_synthetic_goal_mode(app_module)
    app_module.app.run(
        host=args.host, port=args.port, debug=False,
        use_reloader=False, threaded=True,
    )


if __name__ == "__main__":
    main()
