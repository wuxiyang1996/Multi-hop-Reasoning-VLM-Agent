#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
REPO_ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO_ROOT/"src"),str(REPO_ROOT)]
from scripts.collect_agqa2_robust_temporal_v36_development import collect_development
from scripts.evaluate_agqa2_directional_support_v52 import evaluate_calibrated
def main():
    p=argparse.ArgumentParser();p.add_argument("--config",type=Path,default=REPO_ROOT/"configs/agqa2_directional_support_v52_qualification.json");p.add_argument("--keys",type=Path,default=Path("/fs/gamma-projects/vlm-robot/keys.py"));p.add_argument("--base-report",type=Path,default=REPO_ROOT/"runs/agqa2_directional_support_v52_qualification/base_report.json");p.add_argument("--output",type=Path,default=REPO_ROOT/"runs/agqa2_directional_support_v52_qualification/report.json");p.add_argument("--workers",type=int,default=6);a=p.parse_args();collect_development(config_path=a.config.resolve(),keys_path=a.keys.resolve(),output_path=a.base_report.resolve(),workers=a.workers,limit=None);r=evaluate_calibrated(config_path=a.config.resolve(),base_report_path=a.base_report.resolve(),output_path=a.output.resolve(),formal=False);print(json.dumps({k:r[k] for k in ("status","rows","source_executor_authorizations","source_vs_target_native","qualification_gates","provider_calls","reported_provider_cost_usd","report_sha256")},indent=2,sort_keys=True))
if __name__=="__main__":main()
