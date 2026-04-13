#!/usr/bin/env python3
"""Launch a Synapse Brain swarm on HuggingFace Spaces.

Creates N Spaces, each running a spore with a different LLM family.
Spores gossip via HTTP mesh, accumulate CRDT memory, and converge
on synthesized answers through structured debate.

Usage:
    # From config file (recommended):
    python launch_swarm.py --config config.yaml

    # From CLI args:
    python launch_swarm.py --hf-token hf_xxx --count 3

    # Mix both (CLI overrides config):
    python launch_swarm.py --config config.yaml --count 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    from huggingface_hub import HfApi
except ImportError:
    print("Installing huggingface_hub...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
    from huggingface_hub import HfApi

try:
    import yaml
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pyyaml"])
    import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Launch Synapse Brain swarm")
    p.add_argument("--config", type=str, help="Path to config.yaml")
    p.add_argument("--count", type=int, help="Number of spores (default: 3)")
    p.add_argument("--hf-token", type=str, help="HuggingFace token")
    p.add_argument("--hf-owner", type=str, help="HF username or org")
    p.add_argument("--prefix", type=str, help="Space name prefix")
    p.add_argument("--private", action="store_true", default=None)
    p.add_argument("--public", action="store_true")
    p.add_argument("--zai-key", type=str, help="Z.ai API key")
    p.add_argument("--groq-key", type=str, help="Groq API key")
    p.add_argument("--google-key", type=str, help="Google AI Studio key")
    p.add_argument("--cerebras-key", type=str, help="Cerebras API key")
    p.add_argument("--mistral-key", type=str, help="Mistral API key")
    p.add_argument("--peers", type=str, nargs="*", help="Peer URLs to join")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def load_config(path: str) -> dict:
    """Load YAML config file."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def merge_config(args: argparse.Namespace) -> dict:
    """Merge config file + CLI args + env vars. CLI wins over config."""
    cfg = {}

    # Load config file first
    if args.config:
        cfg = load_config(args.config)

    # CLI overrides
    if args.hf_token:
        cfg["hf_token"] = args.hf_token
    if args.count is not None:
        cfg["count"] = args.count
    if args.hf_owner:
        cfg["hf_owner"] = args.hf_owner
    if args.prefix:
        cfg["prefix"] = args.prefix
    if args.public:
        cfg["private"] = False
    if args.peers:
        cfg["peers"] = args.peers

    # API keys from CLI
    api_keys = cfg.get("api_keys", {})
    if args.zai_key:
        api_keys["zai"] = args.zai_key
    if args.groq_key:
        api_keys["groq"] = args.groq_key
    if args.google_key:
        api_keys["google_ai"] = args.google_key
    if args.cerebras_key:
        api_keys["cerebras"] = args.cerebras_key
    if args.mistral_key:
        api_keys["mistral"] = args.mistral_key
    cfg["api_keys"] = api_keys

    # Environment fallbacks
    if not cfg.get("hf_token"):
        cfg["hf_token"] = os.environ.get("HF_TOKEN", "")
    for env_key, cfg_key in [
        ("ZAI_API_KEY", "zai"), ("OPENROUTER_KEY", "openrouter"),
        ("GOOGLE_AI_KEY", "google_ai"),
    ]:
        if not api_keys.get(cfg_key) and os.environ.get(env_key):
            api_keys[cfg_key] = os.environ[env_key]

    # Defaults
    cfg.setdefault("count", 3)
    cfg.setdefault("prefix", "synapse-spore")
    cfg.setdefault("private", True)
    cfg.setdefault("sentinel", False)
    cfg.setdefault("cortex", False)
    cfg["dry_run"] = args.dry_run

    return cfg


def validate_config(cfg: dict) -> bool:
    """Check required fields and print helpful messages."""
    if not cfg.get("hf_token"):
        print("ERROR: No HuggingFace token provided.")
        print("")
        print("Set it in one of these ways:")
        print("  1. In config.yaml:  hf_token: \"hf_your_token\"")
        print("  2. CLI flag:        --hf-token hf_your_token")
        print("  3. Environment:     export HF_TOKEN=hf_your_token")
        print("")
        print("Get a token at: https://huggingface.co/settings/tokens")
        return False
    return True


def get_hf_api(token: str) -> HfApi:
    return HfApi(token=token)


def get_hf_username(token: str) -> str:
    try:
        api = get_hf_api(token)
        info = api.whoami()
        return info.get("name", "unknown")
    except Exception as e:
        print(f"ERROR: Could not authenticate with HuggingFace.")
        print(f"  Check your token is valid: https://huggingface.co/settings/tokens")
        print(f"  Detail: {e}")
        sys.exit(1)


def create_space_repo(owner: str, name: str, token: str, private: bool = True) -> str:
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
        print(f"  Created: {url}")
        return str(url)
    except Exception as e:
        print(f"  Warning: {e}")
        return f"https://huggingface.co/spaces/{repo_id}"


def set_space_secrets(owner: str, name: str, token: str, secrets: dict[str, str]):
    api = get_hf_api(token)
    repo_id = f"{owner}/{name}"
    for key, value in secrets.items():
        if not value:
            continue
        try:
            api.add_space_secret(repo_id=repo_id, key=key, value=value)
        except Exception as e:
            print(f"  Warning: secret {key}: {e}")


# Each spore gets a different model family for reasoning diversity.
MODEL_ASSIGNMENTS = [
    "Qwen/Qwen3-235B-A22B",                    # 0: Explorer
    "meta-llama/Llama-3.3-70B-Instruct",        # 1: Synthesizer
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", # 2: Adversarial
    "google/gemma-3-27b-it",                     # 3: Validator
    "meta-llama/Llama-4-Scout-17B-16E-Instruct", # 4: Generalist
    "glm-4.7-flash",                            # 5: Brain
    "deepseek-ai/DeepSeek-R1",                   # 6: Sentinel
]

ROLE_NAMES = ["Explorer", "Synthesizer", "Adversarial", "Validator", "Generalist", "Brain", "Sentinel"]


def generate_spore_app(
    spore_id: str,
    spore_index: int,
    total_spores: int,
    owner: str,
    prefix: str,
    cfg: dict,
) -> dict[str, str]:
    """Generate all files for one spore Space."""
    # Build peer URLs -- every spore knows all others + any external peers
    peers = []
    for i in range(total_spores):
        if i != spore_index:
            peers.append(f"https://{owner}-{prefix}-{i:03d}.hf.space")

    # Add external peers from config (for joining existing swarms)
    extra_peers = cfg.get("peers", [])
    for p in extra_peers:
        url = p.strip()
        if url and not url.startswith("<") and url not in peers:
            peers.append(url)

    peers_json = json.dumps(peers)
    primary_model = MODEL_ASSIGNMENTS[spore_index % len(MODEL_ASSIGNMENTS)]

    # Read the template
    template_path = Path(__file__).parent / "spore.py"
    if not template_path.exists():
        raise FileNotFoundError("spore.py template not found")
    app_py = template_path.read_text()

    # Substitute all 4 placeholders
    # Template substitution: single-pass via string.Template
    # Prevents future comments/docstrings containing placeholder strings
    # from silently corrupting deployments.
    from string import Template as _Tmpl
    _tmpl_src = (
        app_py
        .replace("__SPORE_ID__", "${_SPORE_ID}")
        .replace("__PEERS_JSON__", "${_PEERS_JSON}")
        .replace("__SPORE_INDEX__", "${_SPORE_INDEX}")
        .replace("__PRIMARY_MODEL__", "${_PRIMARY_MODEL}")
    )
    app_py = _Tmpl(_tmpl_src).safe_substitute(
        _SPORE_ID=spore_id,
        _PEERS_JSON=peers_json.replace("'", "\\'"),
        _SPORE_INDEX=str(spore_index),
        _PRIMARY_MODEL=primary_model,
    )

    is_sentinel = ROLE_NAMES[spore_index] == "Sentinel"

    # llama-cpp-python installed at runtime by cortex.py (avoids slow build phase)
    requirements_txt = (
        "crdt-merge>=0.9.5\n"
        "httpx>=0.27\n"
        "numpy>=1.24\n"
        "fastapi>=0.115\n"
        "uvicorn>=0.30\n"
    )

    python_line = 'python_version: "3.10"\n' if is_sentinel else ""
    readme_md = (
        "---\n"
        "title: Synapse Brain Spore\n"
        'emoji: "\U0001F9E0"\n'
        "colorFrom: purple\n"
        "colorTo: blue\n"
        "sdk: gradio\n"
        'sdk_version: "5.25.2"\n'
        f"{python_line}"
        "app_file: app.py\n"
        "pinned: false\n"
        "---\n\n"
        f"Synapse Brain distributed reasoning node.\n"
        f"Spore ID: `{spore_id}`\n"
    )

    # Read companion modules
    module_dir = Path(__file__).parent
    module_files = {}
    for mod_name in ["cortex.py", "knowledge_wall.py", "mcp_server.py", "federation.py"]:
        mod_path = module_dir / mod_name
        if mod_path.exists():
            module_files[mod_name] = mod_path.read_text()

    result = {
        "app.py": app_py,
        "requirements.txt": requirements_txt,
        "README.md": readme_md,
    }
    result.update(module_files)
    return result


def push_space_files(owner: str, name: str, token: str, files: dict[str, str]):
    from huggingface_hub import CommitOperationAdd

    api = get_hf_api(token)
    repo_id = f"{owner}/{name}"

    operations = [
        CommitOperationAdd(
            path_in_repo=fname,
            path_or_fileobj=content.encode("utf-8"),
        )
        for fname, content in files.items()
    ]

    try:
        api.create_commit(
            repo_id=repo_id,
            repo_type="space",
            operations=operations,
            commit_message="deploy spore",
        )
        print(f"  Pushed {len(files)} files")
    except Exception as e:
        print(f"  Warning: {e}")


def main():
    args = parse_args()
    cfg = merge_config(args)

    if not validate_config(cfg):
        sys.exit(1)

    token = cfg["hf_token"]
    count = cfg["count"]
    prefix = cfg["prefix"]
    private = cfg.get("private", True)
    api_keys = cfg.get("api_keys", {})

    print("Synapse Brain Swarm Launcher")
    print("=" * 40)

    # Detect owner -- skip API call in dry-run if not needed
    owner = cfg.get("hf_owner") or cfg.get("commander")
    if not owner:
        if cfg.get("dry_run"):
            owner = "dry-run-user"
        else:
            owner = get_hf_username(token)
    print(f"Account:  {owner}")
    print(f"Spores:   {count}")
    print(f"Private:  {private}")

    keys_active = [k for k, v in api_keys.items() if v]
    if keys_active:
        print(f"API keys: {', '.join(keys_active)}")
    else:
        print("API keys: none (using free-tier providers)")
    print()

    # Build secrets -- ALL spores get ALL keys for maximum fallback
    secrets = {"HF_TOKEN": token}
    key_map = {
        "zai": "ZAI_API_KEY",
        "openrouter": "OPENROUTER_KEY",
        "google_ai": "GOOGLE_AI_KEY",
    }
    for cfg_key, env_key in key_map.items():
        if api_keys.get(cfg_key):
            secrets[env_key] = api_keys[cfg_key]

    # Deploy
    space_urls = []
    for i in range(count):
        name = f"{prefix}-{i:03d}"
        spore_id = f"{name}-{hashlib.sha256(f'{owner}{name}'.encode()).hexdigest()[:6]}"
        role = ROLE_NAMES[i % len(ROLE_NAMES)]
        model = MODEL_ASSIGNMENTS[i % len(MODEL_ASSIGNMENTS)].split("/")[-1]
        print(f"[{i+1}/{count}] {owner}/{name}  ({role} / {model})")

        files = generate_spore_app(
            spore_id=spore_id,
            spore_index=i,
            total_spores=count,
            owner=owner,
            prefix=prefix,
            cfg=cfg,
        )

        if cfg.get("dry_run"):
            outdir = Path(f"/tmp/swarm-staging/{name}")
            outdir.mkdir(parents=True, exist_ok=True)
            for fname, content in files.items():
                (outdir / fname).write_text(content)
            print(f"  Staged at {outdir}")
            space_urls.append(f"https://{owner}-{name}.hf.space")
            continue

        create_space_repo(owner, name, token, private=private)
        time.sleep(1)
        set_space_secrets(owner, name, token, secrets)
        time.sleep(0.5)
        push_space_files(owner, name, token, files)

        url = f"https://{owner}-{name}.hf.space"
        space_urls.append(url)
        print(f"  Live: {url}")
        print()

        if i < count - 1:
            time.sleep(2)

    print()
    print("=" * 40)
    print(f"Swarm deployed: {len(space_urls)} spores")
    print()
    for url in space_urls:
        print(f"  {url}")
    print()
    print("Spores begin reasoning within 60 seconds of startup.")
    print("Submit a task:  curl -X POST -d '{\"task\": \"your question\"}' <spore-url>/api/task")
    if not keys_active:
        print()
        print("No API keys set. All reasoning uses free-tier providers.")
        print("Add keys to config.yaml for faster, more reliable responses.")


if __name__ == "__main__":
    main()
