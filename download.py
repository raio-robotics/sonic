#!/usr/bin/env python3
"""Fetch checkpoints, robot model, and reference motions for sonic.

Two upstream sources are involved:

  1. Hugging Face — `nvidia/GEAR-SONIC` hosts the ONNX checkpoints and the
     observation config:
         https://huggingface.co/nvidia/GEAR-SONIC

  2. GitHub — `NVlabs/GR00T-WholeBodyControl` ships the G1 URDF + meshes
     and the reference motion CSV library (these are not on Hugging Face,
     they live inside the deploy package alongside the C++ code):
         https://github.com/NVlabs/GR00T-WholeBodyControl

Files land at the locations sonic expects relative to the repo root:

    checkpoints/
        model_encoder.onnx
        model_decoder.onnx
        planner_sonic.onnx
    assets/
        observation_config.yaml
        reference/example/<motion>/...   # from gear_sonic_deploy/reference/example/
        robot/                            # from gear_sonic/data/robot_model/model_data/g1/

Usage:
    python download.py                 # download everything that's missing
    python download.py --force         # re-download even if files exist
    python download.py --no-planner    # skip the planner ONNX
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HF_REPO_ID = "nvidia/GEAR-SONIC"
GROOT_REPO_URL = "https://github.com/NVlabs/GR00T-WholeBodyControl.git"

REPO_ROOT = Path(__file__).resolve().parent

# (hf_filename, local_destination_relative_to_repo_root)
HF_FILES = [
    ("model_encoder.onnx",      "checkpoints/model_encoder.onnx"),
    ("model_decoder.onnx",      "checkpoints/model_decoder.onnx"),
    ("observation_config.yaml", "assets/observation_config.yaml"),
]
HF_PLANNER = ("planner_sonic.onnx", "checkpoints/planner_sonic.onnx")

# Paths to pull from the GR00T-WholeBodyControl git repo (src -> dst).
GIT_PATHS = [
    ("gear_sonic_deploy/reference/example",       "assets/reference/example"),
    ("gear_sonic/data/robot_model/model_data/g1", "assets/robot"),
]


def _hf_download():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("huggingface_hub is required. Install with: uv pip install huggingface_hub")
    return hf_hub_download


def fetch_from_hf(force: bool, want_planner: bool) -> None:
    hf_hub_download = _hf_download()
    files = list(HF_FILES) + ([HF_PLANNER] if want_planner else [])

    print(f"\n[Hugging Face] {HF_REPO_ID}")
    for hf_name, local_rel in files:
        dst = REPO_ROOT / local_rel
        if dst.exists() and not force:
            print(f"  (skip)  {local_rel}  already exists")
            continue
        print(f"  fetch   {hf_name}")
        cached = hf_hub_download(repo_id=HF_REPO_ID, filename=hf_name)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached, dst)
        print(f"     ->   {local_rel}")


def fetch_from_git(force: bool) -> None:
    needed = [(src, dst) for src, dst in GIT_PATHS
              if force or not (REPO_ROOT / dst).exists()]
    print(f"\n[GitHub] {GROOT_REPO_URL}")
    if not needed:
        for _, dst in GIT_PATHS:
            print(f"  (skip)  {dst}  already exists")
        return

    with tempfile.TemporaryDirectory(prefix="groot-sparse-") as tmp:
        tmp_path = Path(tmp) / "GR00T-WholeBodyControl"
        print(f"  shallow sparse-clone -> {tmp_path}")
        subprocess.run(
            ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
             GROOT_REPO_URL, str(tmp_path)],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(tmp_path), "sparse-checkout", "set",
             *[src for src, _ in needed]],
            check=True,
        )
        for src, dst in needed:
            src_path = tmp_path / src
            dst_path = REPO_ROOT / dst
            if not src_path.exists():
                sys.exit(f"  ERROR: expected {src} in upstream repo, not found")
            if dst_path.exists():
                shutil.rmtree(dst_path)
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src_path, dst_path)
            print(f"  copied  {src}  ->  {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch sonic checkpoints + assets from Hugging Face and GitHub.",
    )
    parser.add_argument("--force", action="store_true",
                        help="re-download even if destination files already exist")
    parser.add_argument("--no-planner", action="store_true",
                        help="skip the kinematic planner ONNX")
    parser.add_argument("--hf-only", action="store_true",
                        help="only fetch from Hugging Face (skip the git clone)")
    parser.add_argument("--git-only", action="store_true",
                        help="only fetch from GitHub (skip Hugging Face)")
    args = parser.parse_args()

    print("=" * 60)
    print("  sonic asset / checkpoint downloader")
    print(f"  repo root : {REPO_ROOT}")
    print("=" * 60)

    if not args.git_only:
        fetch_from_hf(force=args.force, want_planner=not args.no_planner)
    if not args.hf_only:
        fetch_from_git(force=args.force)

    print("\nDone.")


if __name__ == "__main__":
    main()
