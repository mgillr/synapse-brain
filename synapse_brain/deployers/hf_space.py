"""HuggingFace Spaces deployer.

Generates files for deploying a spore as a free HF Space:
- app.py (Gradio interface + gossip server)
- requirements.txt
- Dockerfile (optional, for Docker Spaces)

The Space runs a FastAPI server with:
- /gossip/* endpoints for mesh communication
- /health endpoint for keep-alive pings
- Gradio UI showing spore status and task progress
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path


def generate_hf_space(
    output_dir: str,
    spore_id: str,
    mesh_seeds: list[str],
    provider_keys: dict[str, str],
) -> dict[str, str]:
    """Generate all files needed for an HF Space deployment.

    Returns dict of filename -> content.
    """
    files = {}

    # app.py
    seeds_str = ",".join(mesh_seeds)
    env_lines = "\n".join(
        f'os.environ["{k}"] = "{v}"' for k, v in provider_keys.items()
    )

    files["app.py"] = textwrap.dedent(f'''\
        """Synapse Brain Spore -- HuggingFace Space deployment."""
        import asyncio
        import os
        import threading

        os.environ["SYNAPSE_SPORE_ID"] = "{spore_id}"
        os.environ["SYNAPSE_MESH_SEEDS"] = "{seeds_str}"
        os.environ["SYNAPSE_PORT"] = "7860"
        {env_lines}

        import gradio as gr
        from synapse_brain.spore.runtime import Spore, SporeConfig

        config = SporeConfig.from_env()
        spore = Spore(config)

        def start_spore():
            asyncio.run(spore.run())

        # Run spore in background thread
        spore_thread = threading.Thread(target=start_spore, daemon=True)
        spore_thread.start()

        def get_status():
            metrics = spore.metrics()
            lines = [
                f"Spore ID: {{metrics['spore_id']}}",
                f"State: {{metrics['state']}}",
                f"Clock: {{metrics['clock']}}",
                f"Total Deltas: {{metrics['total_deltas']}}",
                f"Active Tasks: {{metrics['active_tasks']}}",
                f"Peers: {{metrics['peer_count']}}",
                f"Merkle Root: {{metrics['merkle_root'][:16]}}...",
            ]
            return "\\n".join(lines)

        with gr.Blocks(title="Synapse Brain Spore") as demo:
            gr.Markdown("# Synapse Brain -- Spore Node")
            gr.Markdown(f"Spore ID: `{spore_id}`")
            status = gr.Textbox(label="Status", lines=8)
            refresh = gr.Button("Refresh")
            refresh.click(fn=get_status, outputs=status)
            demo.load(fn=get_status, outputs=status)

        demo.launch(server_name="0.0.0.0", server_port=7860)
    ''')

    # requirements.txt
    files["requirements.txt"] = textwrap.dedent("""\
        crdt-merge>=0.9.5
        httpx>=0.27
        msgpack>=1.0
        cryptography>=42.0
        gradio>=4.0
    """)

    # README.md (required for HF Spaces)
    files["README.md"] = textwrap.dedent(f"""\
        ---
        title: Synapse Brain Spore
        emoji: &#x1F9E0;
        colorFrom: purple
        colorTo: blue
        sdk: gradio
        sdk_version: "4.44.0"
        app_file: app.py
        pinned: false
        ---

        Synapse Brain distributed reasoning node.
        Spore ID: `{spore_id}`
    """)

    # Write files if output_dir specified
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for fname, content in files.items():
            (out / fname).write_text(content)

    return files


def generate_hf_space_batch(
    base_dir: str,
    count: int,
    mesh_seeds: list[str],
    provider_keys: dict[str, str],
    prefix: str = "synapse-spore",
) -> list[str]:
    """Generate multiple HF Space deployments at once.

    Returns list of generated spore IDs.
    """
    import uuid

    spore_ids = []
    for i in range(count):
        spore_id = f"{prefix}-{i:03d}-{uuid.uuid4().hex[:6]}"
        output_dir = os.path.join(base_dir, spore_id)
        generate_hf_space(output_dir, spore_id, mesh_seeds, provider_keys)
        spore_ids.append(spore_id)

    return spore_ids
