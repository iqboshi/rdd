#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run train-v5 in chunks with automatic resume to reduce long-run interruption risk."
    )
    parser.add_argument("--python_exe", type=str, default=r"D:\anaconda\envs\dl40\python.exe")
    parser.add_argument("--train_script", type=str, default="train-v5.py")
    parser.add_argument("--save_dir", type=str, required=True, help="Same --save_dir passed to train-v5.py")
    parser.add_argument("--total_epochs", type=int, default=50, help="Final target epoch.")
    parser.add_argument("--chunk_size", type=int, default=10, help="Epochs per chunk.")
    parser.add_argument("--force_device", type=str, default="cuda", help="Default device when not set in extra args.")
    parser.add_argument("--no_kmp_env", action="store_true", help="Do not set KMP_DUPLICATE_LIB_OK=TRUE.")
    parser.add_argument("--dry_run", action="store_true")
    args, extra_args = parser.parse_known_args()
    return args, extra_args


def sanitize_extra_args(extra_args: List[str]) -> List[str]:
    skip_next = False
    clean = []
    for i, tok in enumerate(extra_args):
        if skip_next:
            skip_next = False
            continue
        if tok in {"--resume", "--epochs", "--save_dir"}:
            if i + 1 < len(extra_args) and not str(extra_args[i + 1]).startswith("--"):
                skip_next = True
            continue
        if tok.startswith("--resume=") or tok.startswith("--epochs=") or tok.startswith("--save_dir="):
            continue
        clean.append(tok)
    return clean


def has_flag(extra_args: List[str], key: str) -> bool:
    if key in extra_args:
        return True
    prefix = f"{key}="
    return any(str(x).startswith(prefix) for x in extra_args)


def read_ckpt_meta(path: Path) -> Tuple[int, Optional[float]]:
    ckpt = torch.load(str(path), map_location="cpu")
    epoch = int(ckpt.get("epoch", 0)) if isinstance(ckpt, dict) else 0
    best_val = None
    if isinstance(ckpt, dict) and "best_val" in ckpt:
        try:
            best_val = float(ckpt["best_val"])
        except Exception:
            best_val = None
    return epoch, best_val


def find_latest_checkpoint(save_dir: Path) -> Optional[Path]:
    if (save_dir / "checkpoints" / "latest.pth").is_file():
        return save_dir / "checkpoints" / "latest.pth"

    candidates = sorted(
        save_dir.glob("exp_*/checkpoints/latest.pth"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if len(candidates) == 0:
        return None
    return candidates[0]


def main():
    args, extra_args_raw = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    state_path = save_dir / "resumable_state.json"

    extra_args = sanitize_extra_args(extra_args_raw)
    if not has_flag(extra_args, "--device"):
        extra_args.extend(["--device", args.force_device])

    chunk_size = max(1, int(args.chunk_size))
    total_epochs = max(1, int(args.total_epochs))

    while True:
        latest_ckpt = find_latest_checkpoint(save_dir)
        completed_epoch = 0
        best_val = None
        resume_args: List[str] = []

        if latest_ckpt is not None and latest_ckpt.is_file():
            completed_epoch, best_val = read_ckpt_meta(latest_ckpt)
            resume_args = ["--resume", str(latest_ckpt)]

        if completed_epoch >= total_epochs:
            print(f"[DONE] completed_epoch={completed_epoch} >= total_epochs={total_epochs}")
            break

        target_epoch = min(total_epochs, completed_epoch + chunk_size)
        cmd = [
            str(args.python_exe),
            "-u",
            str(args.train_script),
            "--save_dir",
            str(save_dir),
            "--epochs",
            str(target_epoch),
        ] + extra_args + resume_args

        run_meta: Dict = {
            "save_dir": str(save_dir),
            "python_exe": str(args.python_exe),
            "train_script": str(args.train_script),
            "completed_epoch_before": int(completed_epoch),
            "target_epoch": int(target_epoch),
            "resume_checkpoint": str(latest_ckpt) if latest_ckpt else "",
            "best_val_before": best_val,
            "cmd": cmd,
        }
        state_path.write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")

        print(
            f"[RUN] epoch {completed_epoch + 1} -> {target_epoch} | "
            f"resume={str(latest_ckpt) if latest_ckpt else 'none'}"
        )
        print(f"[CMD] {' '.join(cmd)}")

        if args.dry_run:
            print("[DRY-RUN] stop before execution.")
            break

        env = os.environ.copy()
        if not bool(args.no_kmp_env):
            env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        proc = subprocess.Popen(cmd, env=env)
        ret = proc.wait()
        if ret != 0:
            print(f"[ERROR] train process exited with code={ret}")
            return int(ret)

        new_latest = find_latest_checkpoint(save_dir)
        if new_latest is None or not new_latest.is_file():
            print("[ERROR] training finished but latest checkpoint not found.")
            return 2

        new_epoch, new_best = read_ckpt_meta(new_latest)
        print(
            f"[CHECK] latest={new_latest} | epoch={new_epoch} | "
            f"best_val={new_best if new_best is not None else 'NA'}"
        )
        if new_epoch < target_epoch:
            print(
                f"[ERROR] expected epoch >= {target_epoch}, got {new_epoch}. "
                "Please inspect logs/checkpoints."
            )
            return 3

    return 0


if __name__ == "__main__":
    sys.exit(main())
