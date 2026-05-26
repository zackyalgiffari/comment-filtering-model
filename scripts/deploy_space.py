#!/usr/bin/env python3
"""Deploy the Gradio Space to HuggingFace Hub.

Usage:
    export HF_TOKEN=hf_xxxxxxxxxxxx   # write-scoped token
    python3 scripts/deploy_space.py [--repo USER/SPACE] [--folder spaces/]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        default="zackyalgiffari/comment-filtering-demo",
        help="HF Space repo ID (USER/SPACE_NAME)",
    )
    parser.add_argument(
        "--folder",
        type=Path,
        default=Path("spaces"),
        help="Local folder containing the Space files",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the Space as private",
    )
    parser.add_argument(
        "--commit-message",
        default="Deploy Gradio Space",
        help="Commit message for the upload",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print(
            "ERROR: HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) is not set.\n"
            "  Create a write-scoped token at https://huggingface.co/settings/tokens\n"
            "  Then: export HF_TOKEN=hf_xxxxxxxxxxxx",
            file=sys.stderr,
        )
        return 1

    if not args.folder.exists() or not args.folder.is_dir():
        print(f"ERROR: folder not found: {args.folder}", file=sys.stderr)
        return 1

    required = ["app.py", "requirements.txt", "README.md"]
    missing = [name for name in required if not (args.folder / name).exists()]
    if missing:
        print(f"ERROR: {args.folder} is missing required files: {missing}", file=sys.stderr)
        return 1

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print(
            "ERROR: huggingface_hub is not installed.\n"
            "  Install with: pip install --user huggingface_hub",
            file=sys.stderr,
        )
        return 1

    print(f"Creating Space repo (if needed): {args.repo}")
    create_repo(
        repo_id=args.repo,
        repo_type="space",
        space_sdk="gradio",
        token=token,
        private=args.private,
        exist_ok=True,
    )

    api = HfApi(token=token)
    print(f"Uploading {args.folder}/ to {args.repo} ...")
    api.upload_folder(
        folder_path=str(args.folder),
        repo_id=args.repo,
        repo_type="space",
        commit_message=args.commit_message,
    )

    url = f"https://huggingface.co/spaces/{args.repo}"
    print(f"\nDeployed. Build status: {url}")
    print(f"Settings (to enable ZeroGPU): {url}/settings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
