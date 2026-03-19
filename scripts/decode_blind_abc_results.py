#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Decode blind A/B/C scoring results.")
    parser.add_argument("--blind_dir", type=str, required=True, help="Directory containing blind CSV files.")
    parser.add_argument(
        "--out_detail",
        type=str,
        default="blind_decode_detailed.csv",
        help="Detailed decode CSV filename (relative to blind_dir if not absolute).",
    )
    parser.add_argument(
        "--out_summary",
        type=str,
        default="blind_decode_summary.json",
        help="Summary JSON filename (relative to blind_dir if not absolute).",
    )
    return parser.parse_args()


def maybe_join(base: Path, target: str) -> Path:
    p = Path(target)
    return p if p.is_absolute() else base / p


def to_float_or_none(val: str):
    if val is None:
        return None
    v = str(val).strip()
    if v == "":
        return None
    try:
        return float(v)
    except Exception:
        return None


def normalize_winner(raw: str):
    if raw is None:
        return None
    v = str(raw).strip().upper()
    if v in ("A", "B", "C"):
        return v
    return None


def main():
    args = parse_args()
    blind_dir = Path(args.blind_dir)
    mapping_path = blind_dir / "blind_mapping_private.csv"
    scoring_path = blind_dir / "blind_scoring_template.csv"

    if not mapping_path.is_file():
        raise FileNotFoundError(f"Missing mapping file: {mapping_path}")
    if not scoring_path.is_file():
        raise FileNotFoundError(f"Missing scoring file: {scoring_path}")

    with mapping_path.open("r", encoding="utf-8", newline="") as f:
        map_rows = list(csv.DictReader(f))
    with scoring_path.open("r", encoding="utf-8", newline="") as f:
        score_rows = list(csv.DictReader(f))

    map_by_idx = {int(r["index"]): r for r in map_rows}
    score_by_idx = {int(r["index"]): r for r in score_rows}
    idx_all = sorted(set(map_by_idx.keys()) & set(score_by_idx.keys()))

    all_sources = set()
    for r in map_rows:
        all_sources.add(r["A_source"])
        all_sources.add(r["B_source"])
        all_sources.add(r["C_source"])

    slot_win_counter = Counter()
    source_win_counter = Counter()
    slot_score_sums = Counter()
    slot_score_cnts = Counter()
    source_score_sums = Counter()
    source_score_cnts = Counter()
    bucket_source_wins = defaultdict(Counter)

    detail_rows = []
    for idx in idx_all:
        m = map_by_idx[idx]
        s = score_by_idx[idx]
        bucket = m["bucket"]

        winner_slot = normalize_winner(s.get("winner_A_B_or_C", ""))
        winner_source = m.get(f"{winner_slot}_source") if winner_slot is not None else ""

        if winner_slot is not None:
            slot_win_counter[winner_slot] += 1
        if winner_source:
            source_win_counter[winner_source] += 1
            bucket_source_wins[bucket][winner_source] += 1

        slot_scores = {
            "A": to_float_or_none(s.get("score_A_1to5", "")),
            "B": to_float_or_none(s.get("score_B_1to5", "")),
            "C": to_float_or_none(s.get("score_C_1to5", "")),
        }
        for slot, score_val in slot_scores.items():
            if score_val is None:
                continue
            slot_score_sums[slot] += score_val
            slot_score_cnts[slot] += 1

            src = m.get(f"{slot}_source", "")
            if src:
                source_score_sums[src] += score_val
                source_score_cnts[src] += 1

        detail_rows.append(
            {
                "index": idx,
                "image_name": m["image_name"],
                "bucket": bucket,
                "A_source": m["A_source"],
                "B_source": m["B_source"],
                "C_source": m["C_source"],
                "winner_slot": winner_slot or "",
                "winner_source": winner_source or "",
                "score_A_1to5": s.get("score_A_1to5", ""),
                "score_B_1to5": s.get("score_B_1to5", ""),
                "score_C_1to5": s.get("score_C_1to5", ""),
                "notes": s.get("notes", ""),
            }
        )

    total_samples = len(idx_all)
    slot_win_rate = {
        k: (slot_win_counter[k] / total_samples if total_samples else 0.0) for k in ("A", "B", "C")
    }
    source_win_rate = {
        src: (source_win_counter[src] / total_samples if total_samples else 0.0)
        for src in sorted(all_sources)
    }

    slot_mean_score = {
        k: (slot_score_sums[k] / slot_score_cnts[k] if slot_score_cnts[k] else None) for k in ("A", "B", "C")
    }
    source_mean_score = {
        src: (source_score_sums[src] / source_score_cnts[src] if source_score_cnts[src] else None)
        for src in sorted(all_sources)
    }

    source_rank_by_wins = sorted(
        sorted(all_sources),
        key=lambda x: (
            -source_win_counter[x],
            -(source_mean_score[x] if source_mean_score[x] is not None else -1e9),
            x,
        ),
    )
    source_rank_by_score = sorted(
        sorted(all_sources),
        key=lambda x: (
            -(source_mean_score[x] if source_mean_score[x] is not None else -1e9),
            -source_win_counter[x],
            x,
        ),
    )

    out_detail = maybe_join(blind_dir, args.out_detail)
    out_summary = maybe_join(blind_dir, args.out_summary)

    with out_detail.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "image_name",
                "bucket",
                "A_source",
                "B_source",
                "C_source",
                "winner_slot",
                "winner_source",
                "score_A_1to5",
                "score_B_1to5",
                "score_C_1to5",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(detail_rows)

    summary = {
        "blind_dir": str(blind_dir),
        "total_samples": total_samples,
        "slot_win_count": dict(slot_win_counter),
        "slot_win_rate": slot_win_rate,
        "slot_mean_score": slot_mean_score,
        "source_win_count": {src: int(source_win_counter[src]) for src in sorted(all_sources)},
        "source_win_rate": source_win_rate,
        "source_mean_score": source_mean_score,
        "rank_by_wins": source_rank_by_wins,
        "rank_by_mean_score": source_rank_by_score,
        "bucket_source_wins": {k: dict(v) for k, v in bucket_source_wins.items()},
    }

    with out_summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Detailed decode saved: {out_detail}")
    print(f"[INFO] Summary decode saved: {out_summary}")
    if source_rank_by_wins:
        print(f"[INFO] Top by wins: {source_rank_by_wins[0]}")
    if source_rank_by_score:
        print(f"[INFO] Top by mean score: {source_rank_by_score[0]}")


if __name__ == "__main__":
    main()
