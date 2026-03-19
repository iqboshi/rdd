#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build blind A/B/C visual review pack from three review packs."
    )
    parser.add_argument("--pack_a", type=str, required=True, help="First input review pack directory.")
    parser.add_argument("--pack_b", type=str, required=True, help="Second input review pack directory.")
    parser.add_argument("--pack_c", type=str, required=True, help="Third input review pack directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output blind review directory.")
    parser.add_argument("--seed", type=int, default=20260319, help="Random seed for A/B/C assignment.")
    parser.add_argument("--name_a", type=str, default="baseline", help="Internal source label for pack_a.")
    parser.add_argument("--name_b", type=str, default="candidate_b", help="Internal source label for pack_b.")
    parser.add_argument("--name_c", type=str, default="candidate_c", help="Internal source label for pack_c.")
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


def validate_manifests(man_a, man_b, man_c):
    if not (len(man_a) == len(man_b) == len(man_c)):
        raise RuntimeError(
            f"Manifest length mismatch: {len(man_a)} vs {len(man_b)} vs {len(man_c)}"
        )
    for ra, rb, rc in zip(man_a, man_b, man_c):
        ia, ib, ic = int(ra["index"]), int(rb["index"]), int(rc["index"])
        if not (ia == ib == ic):
            raise RuntimeError(f"Index mismatch: {ia} vs {ib} vs {ic}")
        name_a, name_b, name_c = ra["image_name"], rb["image_name"], rc["image_name"]
        if not (name_a == name_b == name_c):
            raise RuntimeError(f"Image mismatch on index {ia}: {name_a} vs {name_b} vs {name_c}")
        bucket_a, bucket_b, bucket_c = ra["bucket"], rb["bucket"], rc["bucket"]
        if not (bucket_a == bucket_b == bucket_c):
            raise RuntimeError(
                f"Bucket mismatch on index {ia}: {bucket_a} vs {bucket_b} vs {bucket_c}"
            )


def main():
    args = parse_args()
    pack_a = Path(args.pack_a)
    pack_b = Path(args.pack_b)
    pack_c = Path(args.pack_c)

    out_dir = Path(args.output_dir)
    out_triplets = out_dir / "triplets"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_triplets.mkdir(parents=True, exist_ok=True)

    manifest_a = load_manifest(pack_a / "review_manifest.csv")
    manifest_b = load_manifest(pack_b / "review_manifest.csv")
    manifest_c = load_manifest(pack_c / "review_manifest.csv")
    validate_manifests(manifest_a, manifest_b, manifest_c)

    panel_a_dir = pack_a / "panels"
    panel_b_dir = pack_b / "panels"
    panel_c_dir = pack_c / "panels"
    if not panel_a_dir.is_dir() or not panel_b_dir.is_dir() or not panel_c_dir.is_dir():
        raise RuntimeError("All input packs must contain panels/ directory.")

    rng = np.random.default_rng(int(args.seed))
    mapping_rows = []
    scoring_rows = []

    source_items = [
        (args.name_a, panel_a_dir),
        (args.name_b, panel_b_dir),
        (args.name_c, panel_c_dir),
    ]

    for row in manifest_a:
        idx = int(row["index"])
        image_name = row["image_name"]
        bucket = row["bucket"]

        src_triplet = []
        for src_name, panel_dir in source_items:
            panel_file = find_panel_file(panel_dir, idx)
            src_triplet.append((src_name, panel_file))

        perm = rng.permutation(3)
        assigned = [src_triplet[int(k)] for k in perm]

        out_a = out_triplets / f"{idx:02d}_A.png"
        out_b = out_triplets / f"{idx:02d}_B.png"
        out_c = out_triplets / f"{idx:02d}_C.png"
        shutil.copyfile(str(assigned[0][1]), str(out_a))
        shutil.copyfile(str(assigned[1][1]), str(out_b))
        shutil.copyfile(str(assigned[2][1]), str(out_c))

        mapping_rows.append(
            {
                "index": idx,
                "image_name": image_name,
                "bucket": bucket,
                "A_source": assigned[0][0],
                "B_source": assigned[1][0],
                "C_source": assigned[2][0],
                "A_file": out_a.name,
                "B_file": out_b.name,
                "C_file": out_c.name,
            }
        )
        scoring_rows.append(
            {
                "index": idx,
                "image_name": image_name,
                "bucket": bucket,
                "winner_A_B_or_C": "",
                "score_A_1to5": "",
                "score_B_1to5": "",
                "score_C_1to5": "",
                "notes": "",
            }
        )

    with (out_dir / "blind_mapping_private.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "image_name",
                "bucket",
                "A_source",
                "B_source",
                "C_source",
                "A_file",
                "B_file",
                "C_file",
            ],
        )
        writer.writeheader()
        writer.writerows(mapping_rows)

    with (out_dir / "blind_scoring_template.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "image_name",
                "bucket",
                "winner_A_B_or_C",
                "score_A_1to5",
                "score_B_1to5",
                "score_C_1to5",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(scoring_rows)

    lines = [
        "# Blind A/B/C Review Index",
        "",
        "Please review each triplet and fill `blind_scoring_template.csv`.",
        "Do not open `blind_mapping_private.csv` until scoring is complete.",
        "",
    ]
    for row in mapping_rows:
        idx = int(row["index"])
        lines.append(f"## {idx:02d} | {row['bucket']} | {row['image_name']}")
        lines.append(f"![](triplets/{idx:02d}_A.png)")
        lines.append(f"![](triplets/{idx:02d}_B.png)")
        lines.append(f"![](triplets/{idx:02d}_C.png)")
        lines.append("")

    (out_dir / "BLIND_REVIEW_INDEX.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"[INFO] Blind A/B/C pack saved to: {out_dir}")
    print(f"[INFO] Review index: {out_dir / 'BLIND_REVIEW_INDEX.md'}")
    print(f"[INFO] Scoring template: {out_dir / 'blind_scoring_template.csv'}")
    print(f"[INFO] Private mapping: {out_dir / 'blind_mapping_private.csv'}")


if __name__ == "__main__":
    main()
