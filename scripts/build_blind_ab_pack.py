#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Build blind A/B visual review pack from two review packs.")
    parser.add_argument("--pack_a", type=str, required=True, help="Baseline-like pack directory.")
    parser.add_argument("--pack_b", type=str, required=True, help="Candidate/innovation pack directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output blind review directory.")
    parser.add_argument("--seed", type=int, default=20260316, help="Random seed for A/B assignment.")
    parser.add_argument(
        "--name_a",
        type=str,
        default="baseline",
        help="Internal source label for pack_a (used in mapping file only).",
    )
    parser.add_argument(
        "--name_b",
        type=str,
        default="candidate",
        help="Internal source label for pack_b (used in mapping file only).",
    )
    return parser.parse_args()


def load_manifest(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda x: int(x["index"]))
    return rows


def find_panel_file(panel_dir: Path, idx: int) -> Path:
    candidates = sorted(panel_dir.glob(f"{idx:02d}_*.png"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly 1 panel for idx={idx} in {panel_dir}, got {len(candidates)}")
    return candidates[0]


def main():
    args = parse_args()
    pack_a = Path(args.pack_a)
    pack_b = Path(args.pack_b)
    out_dir = Path(args.output_dir)
    out_pairs = out_dir / "pairs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pairs.mkdir(parents=True, exist_ok=True)

    manifest_a = load_manifest(pack_a / "review_manifest.csv")
    manifest_b = load_manifest(pack_b / "review_manifest.csv")
    if len(manifest_a) != len(manifest_b):
        raise RuntimeError(f"Manifest length mismatch: {len(manifest_a)} vs {len(manifest_b)}")

    for ra, rb in zip(manifest_a, manifest_b):
        if int(ra["index"]) != int(rb["index"]):
            raise RuntimeError(f"Index mismatch: {ra['index']} vs {rb['index']}")
        if ra["image_name"] != rb["image_name"]:
            raise RuntimeError(f"Image mismatch on index {ra['index']}: {ra['image_name']} vs {rb['image_name']}")

    panel_a_dir = pack_a / "panels"
    panel_b_dir = pack_b / "panels"
    if not panel_a_dir.is_dir() or not panel_b_dir.is_dir():
        raise RuntimeError("Both input packs must contain panels/ directory.")

    rng = np.random.default_rng(int(args.seed))
    mapping_rows = []
    scoring_rows = []

    for row in manifest_a:
        idx = int(row["index"])
        image_name = row["image_name"]
        bucket = row["bucket"]
        pa = find_panel_file(panel_a_dir, idx)
        pb = find_panel_file(panel_b_dir, idx)

        if bool(rng.integers(0, 2) == 0):
            a_src_name, a_src_file = args.name_a, pa
            b_src_name, b_src_file = args.name_b, pb
        else:
            a_src_name, a_src_file = args.name_b, pb
            b_src_name, b_src_file = args.name_a, pa

        out_a = out_pairs / f"{idx:02d}_A.png"
        out_b = out_pairs / f"{idx:02d}_B.png"
        shutil.copyfile(str(a_src_file), str(out_a))
        shutil.copyfile(str(b_src_file), str(out_b))

        mapping_rows.append(
            {
                "index": idx,
                "image_name": image_name,
                "bucket": bucket,
                "A_source": a_src_name,
                "B_source": b_src_name,
                "A_file": out_a.name,
                "B_file": out_b.name,
            }
        )
        scoring_rows.append(
            {
                "index": idx,
                "image_name": image_name,
                "bucket": bucket,
                "winner_A_or_B": "",
                "score_A_1to5": "",
                "score_B_1to5": "",
                "notes": "",
            }
        )

    with (out_dir / "blind_mapping_private.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["index", "image_name", "bucket", "A_source", "B_source", "A_file", "B_file"],
        )
        writer.writeheader()
        writer.writerows(mapping_rows)

    with (out_dir / "blind_scoring_template.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["index", "image_name", "bucket", "winner_A_or_B", "score_A_1to5", "score_B_1to5", "notes"],
        )
        writer.writeheader()
        writer.writerows(scoring_rows)

    lines = [
        "# Blind A/B Review Index",
        "",
        "请逐条看图并填写 `blind_scoring_template.csv`。",
        "注意：不要参考映射文件，避免评审偏差。",
        "",
    ]
    for row in mapping_rows:
        idx = int(row["index"])
        lines.append(f"## {idx:02d} | {row['bucket']} | {row['image_name']}")
        lines.append(f"![](pairs/{idx:02d}_A.png)")
        lines.append(f"![](pairs/{idx:02d}_B.png)")
        lines.append("")

    (out_dir / "BLIND_REVIEW_INDEX.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] Blind A/B pack saved to: {out_dir}")
    print(f"[INFO] Review index: {out_dir / 'BLIND_REVIEW_INDEX.md'}")
    print(f"[INFO] Scoring template: {out_dir / 'blind_scoring_template.csv'}")
    print(f"[INFO] Private mapping: {out_dir / 'blind_mapping_private.csv'}")


if __name__ == "__main__":
    main()
