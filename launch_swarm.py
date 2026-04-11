#!/usr/bin/env python3
"""Launch a Synapse Brain swarm on HuggingFace Private Spaces.

Creates N private Spaces, each running a spore that:
  - Reasons via free-tier LLM providers (Groq, Google, Cerebras, Mistral, etc.)
  - Routes complex tasks to GLM-5.1 via Z.ai API when available
  - Falls back to free providers if GLM-5.1 is unavailable
  - Gossips with peers via HTTP mesh
  - Reports status via Gradio UI on port 7860

Usage:
    python launch_swarm.py --count 5 --hf-token hf_xxx
    python launch_swarm.py --count 10 --hf-token hf_xxx --zai-key sk-xxx
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    print("Installing huggingface_hub...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
    from huggingface_hub import HfApi, create_repo


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Launch Synapse Brain swarm on HF Spaces")
    p.add_argument("--count", type=int, default=6, help="Number of spores to deploy")
    p.add_argument("--hf-token", required=True, help="HuggingFace token (Pro account)")
    p.add_argument("--hf-owner", default=None, help="HF username or org (auto-detected if omitted)")
    p.add_argument("--prefix", default="synapse-spore", help="Space name prefix")
    p.add_argument("--private", action="store_true", default=True, help="Create as private (default)")
    p.add_argument("--public", action="store_true", help="Create as public")
    p.add_argument("--zai-key", default="", help="Z.ai API key for GLM-5.1 brain tier")
    p.add_argument("--groq-key", default="", help="Groq API key")
    p.add_argument("--google-key", default="", help="Google AI Studio key")
    p.add_argument("--cerebras-key", default="", help="Cerebras API key")
    p.add_argument("--mistral-key", default="", help="Mistral API key")
    p.add_argument("--dry-run", action="store_true", help="Generate files but don't push")
    return p.parse_args()


def get_hf_api(token: str) -> HfApi:
    """Create an HfApi instance."""
    return HfApi(token=token)


def get_hf_username(token: str) -> str:
    """Get the HF username from the token."""
    api = get_hf_api(token)
    return api.whoami()["name"]


def create_space_repo(owner: str, name: str, token: str, private: bool = True) -> str:
    """Create a new HF Space repo. Returns the repo URL."""
    api = get_hf_api(token)
    repo_id = f"{owner}/{name}"
    try:
        url = api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="gradio",
            private=private,
            exist_ok=True,
        )
        print(f"  Created Space: {url}")
        return str(url)
    except Exception as e:
        print(f"  Warning creating {repo_id}: {e}")
        return f"https://huggingface.co/spaces/{repo_id}"


def set_space_secrets(owner: str, name: str, token: str, secrets: dict[str, str]):
    """Set environment secrets on a Space."""
    api = get_hf_api(token)
    repo_id = f"{owner}/{name}"
    for key, value in secrets.items():
        if not value:
            continue
        try:
            api.add_space_secret(repo_id=repo_id, key=key, value=value)
            print(f"  Set secret {key}")
        except Exception as e:
            print(f"  Warning: failed to set secret {key}: {e}")


# Model diversity: each spore gets a different primary model from a different family.
# This is the core innovation: genuinely different reasoning patterns, not clones.
MODEL_ASSIGNMENTS = [
    "Qwen/Qwen3-235B-A22B",                    # 0: Explorer   -- massive 235B MoE thinking model
    "meta-llama/Llama-3.3-70B-Instruct",        # 1: Synthesizer -- Meta's strongest instruct model
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", # 2: Adversarial -- chain-of-thought reasoning
    "google/gemma-3-27b-it",                     # 3: Validator   -- Google's training + Z.ai brain tier
    "meta-llama/Llama-4-Scout-17B-16E-Instruct", # 4: Generalist  -- newest MoE architecture
    "glm-4.7-flash",                            # 5: Brain       -- Z.ai GLM-4.7-Flash (free tier)
]

# Spores that get the Z.ai API key for brain-tier access
ZAI_SPORE_INDICES = {3, 5}  # Validator (brain-tier escalation) + Brain (primary)


def generate_spore_app(
    spore_id: str,
    spore_index: int,
    total_spores: int,
    owner: str,
    prefix: str,
    provider_keys: dict[str, str],
) -> dict[str, str]:
    """Generate all files for one spore Space.

    Reads the v3 template and substitutes per-spore values.
    Each spore gets a different primary LLM for model diversity.
    """
    # Build peer URLs -- each spore knows about all others
    peers = []
    for i in range(total_spores):
        if i != spore_index:
            peers.append(f"https://{owner}-{prefix}-{i:03d}.hf.space")

    peers_json = json.dumps(peers)

    # Assign primary model -- wraps around if more spores than models
    primary_model = MODEL_ASSIGNMENTS[spore_index % len(MODEL_ASSIGNMENTS)]

    # Read the v5 template and substitute per-spore values
    template_path = Path(__file__).parent / "spore.py"
    if template_path.exists():
        app_py = template_path.read_text()
    else:
        # Fallback: try alternate names
        for alt in ["spore_v5.py", "spore_v3.py"]:
            alt_path = Path(__file__).parent / alt
            if alt_path.exists():
                app_py = alt_path.read_text()
                break
        else:
            raise FileNotFoundError("No spore template found")

    app_py = (
        app_py
        .replace("__SPORE_ID__", spore_id)
        .replace("__PEERS_JSON__", peers_json.replace("'", "\\'"))
        .replace("__SPORE_INDEX__", str(spore_index))
        .replace("__PRIMARY_MODEL__", primary_model)
    )

    requirements_txt = """crdt-merge>=0.9.5
httpx>=0.27
numpy>=1.24
sentence-transformers>=3.0
fastapi>=0.115
uvicorn>=0.30
"""

    readme_md = f"""---
title: Synapse Brain Spore
emoji: "\U0001F9E0"
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: "5.25.2"
app_file: app.py
pinned: false
---

Synapse Brain distributed reasoning node.
Spore ID: `{spore_id}`
"""

    return {
        "app.py": app_py,
        "requirements.txt": requirements_txt,
        "README.md": readme_md,
    }


def push_space_files(owner: str, name: str, token: str, files: dict[str, str]):
    """Push files to an HF Space using the huggingface_hub API."""
    from huggingface_hub import CommitOperationAdd

    api = get_hf_api(token)
    repo_id = f"{owner}/{name}"

    operations = []
    for fname, content in files.items():
        operations.append(
            CommitOperationAdd(
                path_in_repo=fname,
                path_or_fileobj=content.encode("utf-8"),
            )
        )

    try:
        api.create_commit(
            repo_id=repo_id,
            repo_type="space",
            operations=operations,
            commit_message="deploy spore",
        )
        print(f"  Pushed {len(files)} files to {repo_id}")
    except Exception as e:
        print(f"  Warning pushing to {repo_id}: {e}")


def main():
    args = parse_args()

    private = not args.public
    token = args.hf_token

    print("Synapse Brain Swarm Launcher")
    print("=" * 40)

    # Get username
    if args.hf_owner:
        owner = args.hf_owner
    else:
        owner = get_hf_username(token)
    print(f"HF account: {owner}")
    print(f"Deploying {args.count} spores ({'private' if private else 'public'})")
    print(f"Brain tier (GLM-5.1): {'YES' if args.zai_key else 'NO -- using free providers only'}")
    print()

    # Collect provider keys for Space secrets
    secrets = {"HF_TOKEN": token}
    if args.zai_key:
        secrets["ZAI_API_KEY"] = args.zai_key
    if args.groq_key:
        secrets["GROQ_API_KEY"] = args.groq_key
    if args.google_key:
        secrets["GOOGLE_AI_KEY"] = args.google_key
    if args.cerebras_key:
        secrets["CEREBRAS_API_KEY"] = args.cerebras_key
    if args.mistral_key:
        secrets["MISTRAL_API_KEY"] = args.mistral_key

    # Also pull from environment if not passed as args
    for env_key in ["ZAI_API_KEY", "GROQ_API_KEY", "GOOGLE_AI_KEY", "CEREBRAS_API_KEY", "MISTRAL_API_KEY"]:
        if env_key not in secrets and os.environ.get(env_key):
            secrets[env_key] = os.environ[env_key]

    # Deploy each spore
    space_urls = []
    for i in range(args.count):
        name = f"{args.prefix}-{i:03d}"
        spore_id = f"{name}-{hashlib.sha256(f'{owner}{name}'.encode()).hexdigest()[:6]}"
        print(f"[{i+1}/{args.count}] Deploying {owner}/{name} (spore: {spore_id})")

        # Generate files
        files = generate_spore_app(
            spore_id=spore_id,
            spore_index=i,
            total_spores=args.count,
            owner=owner,
            prefix=args.prefix,
            provider_keys=secrets,
        )

        if args.dry_run:
            outdir = Path(f"/tmp/swarm-staging/{name}")
            outdir.mkdir(parents=True, exist_ok=True)
            for fname, content in files.items():
                (outdir / fname).write_text(content)
            print(f"  Written to {outdir}")
            space_urls.append(f"https://{owner}-{name}.hf.space")
            continue

        # Create Space
        create_space_repo(owner, name, token, private=private)
        time.sleep(1)

        # Set secrets -- Z.ai key only goes to designated spores
        spore_secrets = dict(secrets)
        if i not in ZAI_SPORE_INDICES and "ZAI_API_KEY" in spore_secrets:
            del spore_secrets["ZAI_API_KEY"]
        set_space_secrets(owner, name, token, spore_secrets)
        time.sleep(0.5)

        # Push code
        push_space_files(owner, name, token, files)
        url = f"https://huggingface.co/spaces/{owner}/{name}"
        space_urls.append(url)
        print(f"  Live at {url}")
        print()

        # Small delay between deployments to avoid rate limits
        if i < args.count - 1:
            time.sleep(2)

    print()
    print("=" * 40)
    print(f"Swarm deployed: {len(space_urls)} spores")
    print()
    for url in space_urls:
        print(f"  {url}")
    print()
    print("Spores will begin reasoning cycles within 60 seconds of startup.")
    if not args.zai_key:
        print("Note: No Z.ai key provided. All reasoning uses free-tier providers.")
        print("Add --zai-key to enable GLM-5.1 as the brain tier.")


if __name__ == "__main__":
    main()
