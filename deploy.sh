#!/usr/bin/env bash
# Synapse Brain -- one-command deploy
#
# Usage:
#   ./deploy.sh                     # interactive setup
#   ./deploy.sh --hf-token hf_xxx   # skip prompts
#   HF_TOKEN=hf_xxx ./deploy.sh     # from environment
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="${SCRIPT_DIR}/config.yaml"

echo "Synapse Brain -- Quick Deploy"
echo "============================="
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "ERROR: Python 3 required. Install from https://python.org"
    exit 1
fi

# Install deps
echo "Installing dependencies..."
pip install -q pyyaml huggingface_hub 2>/dev/null || pip3 install -q pyyaml huggingface_hub

# If no config.yaml, create one interactively
if [ ! -f "$CONFIG" ]; then
    echo ""
    echo "No config.yaml found. Let's set one up."
    echo ""

    # Get HF token
    HF_TOKEN="${HF_TOKEN:-}"
    if [ -z "$HF_TOKEN" ]; then
        for arg in "$@"; do
            case "$prev" in
                --hf-token) HF_TOKEN="$arg" ;;
            esac
            prev="$arg"
        done
    fi

    if [ -z "$HF_TOKEN" ]; then
        echo "You need a HuggingFace token (free account works)."
        echo "Get one at: https://huggingface.co/settings/tokens"
        echo ""
        read -rp "HF Token: " HF_TOKEN
    fi

    if [ -z "$HF_TOKEN" ]; then
        echo "ERROR: No token provided."
        exit 1
    fi

    # Get spore count
    echo ""
    echo "How many spores? (3 is good to start, 7 for full diversity)"
    read -rp "Count [3]: " COUNT
    COUNT="${COUNT:-3}"

    # Optional: Z.ai key
    echo ""
    echo "Z.ai API key (optional, free at https://z.ai -- improves reliability):"
    read -rp "Z.ai key [skip]: " ZAI_KEY

    # Write config
    cat > "$CONFIG" << YAML
hf_token: "${HF_TOKEN}"
count: ${COUNT}
api_keys:
  zai: "${ZAI_KEY:-}"
  groq: ""
  google_ai: ""
  cerebras: ""
YAML

    echo ""
    echo "Config saved to config.yaml"
    echo "You can edit it later to add more API keys."
fi

echo ""
echo "Deploying swarm..."
echo ""

# Launch
python3 "${SCRIPT_DIR}/launch_swarm.py" --config "$CONFIG" "$@"
