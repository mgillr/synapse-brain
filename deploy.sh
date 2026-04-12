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

    # LLM provider keys (more = more resilient)
    echo ""
    echo "LLM provider keys make your swarm more resilient."
    echo "Each key is a separate fallback path. All are free tier."
    echo "Press Enter to skip any you don't have."
    echo ""

    read -rp "xAI key (https://console.x.ai): " XAI_KEY
    read -rp "Google AI key (https://aistudio.google.com): " GOOGLE_KEY
    read -rp "Z.ai key (https://open.bigmodel.cn): " ZAI_KEY
    read -rp "OpenRouter key (https://openrouter.ai): " OR_KEY

    # Write config
    cat > "$CONFIG" << YAML
hf_token: "${HF_TOKEN}"
count: ${COUNT}
api_keys:
  xai: "${XAI_KEY:-}"
  google_ai: "${GOOGLE_KEY:-}"
  zai: "${ZAI_KEY:-}"
  openrouter: "${OR_KEY:-}"
  github_models: ""
  llmapi: ""
  groq: ""
  cerebras: ""
YAML

    echo ""
    echo "Config saved to config.yaml"
    echo "Edit it anytime to add more API keys for extra resilience."
fi

echo ""
echo "Deploying swarm..."
echo ""

# Launch
python3 "${SCRIPT_DIR}/launch_swarm.py" --config "$CONFIG" "$@"
