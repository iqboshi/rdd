#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gate whether a candidate review pack is strong enough for human blind scoring."
    )
    parser.add_argument("--baseline_summary", type=str, required=True, help="Path to baseline review_summary.csv")
    parser.add_argument("--candidate_summary", type=str, required=True, help="Path to candidate review_summary.csv")
    parser.add_argument(
        "--min_rel_improve",
        type=float,
        default=0.08,
        help="Minimum relative MAE improvement required to pass, e.g. 0.08 = 8%%.",
    )
    parser.add_argument(
        "--min_dense_abs_improve",
        type=float,
        default=2.0,
        help="Minimum absolute MAE improvement required on dense bucket.",
    )
    parser.add_argument(
        "--max_regress_bucket",
        type=float,
        default=1.5,
        help="Maximum allowed regression MAE on any bucket.",
    )
    parser.add_argument("--save_json", type=str, default="", help="Optional output path for gate report JSON.")
    return parser.parse_args()


def read_rows(path: Path) -> List[Dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if len(rows) == 0:
        raise ValueError(f"Empty csv: {path}")
    return rows


def to_float(row: Dict, key: str) -> float:
    v = row.get(key, "")
    if v is None or str(v).strip() == "":
        raise ValueError(f"Missing value `{key}` in row: {row}")
    return float(v)


def mae(rows: List[Dict]) -> float:
    errs = [abs(to_float(r, "pred_inst_count") - to_float(r, "gt_inst_count")) for r in rows]
    return sum(errs) / len(errs)


def bucket_mae(rows: List[Dict]) -> Dict[str, float]:
    bucket_errors = defaultdict(list)
    for r in rows:
        b = str(r.get("bucket", "unknown"))
        bucket_errors[b].append(abs(to_float(r, "pred_inst_count") - to_float(r, "gt_inst_count")))
    return {k: (sum(v) / len(v)) for k, v in bucket_errors.items()}


def main():
    args = parse_args()
    baseline_path = Path(args.baseline_summary)
    candidate_path = Path(args.candidate_summary)

    baseline_rows = read_rows(baseline_path)
    candidate_rows = read_rows(candidate_path)

    if len(baseline_rows) != len(candidate_rows):
        raise ValueError(
            f"Row count mismatch: baseline={len(baseline_rows)} vs candidate={len(candidate_rows)}"
        )

    base_mae = mae(baseline_rows)
    cand_mae = mae(candidate_rows)
    abs_improve = base_mae - cand_mae
    rel_improve = abs_improve / max(base_mae, 1e-6)

    base_bucket = bucket_mae(baseline_rows)
    cand_bucket = bucket_mae(candidate_rows)

    bucket_delta = {}
    for b in sorted(set(base_bucket.keys()) | set(cand_bucket.keys())):
        bucket_delta[b] = base_bucket.get(b, 0.0) - cand_bucket.get(b, 0.0)

    dense_improve = bucket_delta.get("dense", 0.0)
    worst_bucket_regress = 0.0
    for b, v in bucket_delta.items():
        if v < 0:
            worst_bucket_regress = min(worst_bucket_regress, v)

    pass_overall = rel_improve >= float(args.min_rel_improve)
    pass_dense = dense_improve >= float(args.min_dense_abs_improve)
    pass_regress = abs(worst_bucket_regress) <= float(args.max_regress_bucket)
    passed = bool(pass_overall and pass_dense and pass_regress)

    report = {
        "baseline_summary": str(baseline_path),
        "candidate_summary": str(candidate_path),
        "baseline_mae": base_mae,
        "candidate_mae": cand_mae,
        "abs_improve": abs_improve,
        "rel_improve": rel_improve,
        "dense_abs_improve": dense_improve,
        "worst_bucket_regress": worst_bucket_regress,
        "bucket_abs_improve": bucket_delta,
        "thresholds": {
            "min_rel_improve": float(args.min_rel_improve),
            "min_dense_abs_improve": float(args.min_dense_abs_improve),
            "max_regress_bucket": float(args.max_regress_bucket),
        },
        "checks": {
            "pass_overall": pass_overall,
            "pass_dense": pass_dense,
            "pass_regress": pass_regress,
        },
        "passed": passed,
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.save_json:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[INFO] Saved gate report: {out}")

    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
