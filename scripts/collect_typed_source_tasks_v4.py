#!/usr/bin/env python3
from pathlib import Path

from collect_typed_source_tasks_v3 import main


if __name__ == "__main__":
    main(Path("configs/typed_multisource_v4.json"))
