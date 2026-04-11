"""Render free-tier deployer.

Generates deployment files for Render:
- render.yaml (Blueprint spec)
- Dockerfile
- Keep-alive script (prevents 15min idle spin-down)

Render free tier: 750 hours/month, spins down after 15min idle.
Keep-alive cron hits /health every 10 minutes from an external source.
"""

from __future__ import annotations

import textwrap
from pathlib import Path


def generate_render_deployment(
    output_dir: str,
    spore_id: str,
    mesh_seeds: list[str],
    provider_keys: dict[str, str],
) -> dict[str, str]:
    """Generate Render deployment files."""
    files = {}

    seeds_str = ",".join(mesh_seeds)

    # render.yaml blueprint
    env_vars = "\n".join(
        f"      - key: {k}\n        value: {v}" for k, v in provider_keys.items()
    )

    files["render.yaml"] = textwrap.dedent(f"""\
        services:
          - type: web
            name: synapse-spore-{spore_id[:8]}
            runtime: python
            plan: free
            buildCommand: pip install synapse-brain
            startCommand: python -m synapse_brain.spore.runtime --port $PORT --id {spore_id}
            envVars:
              - key: SYNAPSE_SPORE_ID
                value: {spore_id}
              - key: SYNAPSE_MESH_SEEDS
                value: {seeds_str}
        {env_vars}
    """)

    # Dockerfile
    files["Dockerfile"] = textwrap.dedent("""\
        FROM python:3.12-slim
        WORKDIR /app
        RUN pip install --no-cache-dir synapse-brain
        COPY . .
        CMD ["python", "-m", "synapse_brain.spore.runtime"]
    """)

    # Keep-alive script (run externally, e.g., from Cloudflare Worker)
    files["keepalive.py"] = textwrap.dedent(f"""\
        \"\"\"Keep-alive pinger for Render free tier.
        Run this from any always-on service (Oracle VM, Cloudflare Worker).
        Pings the Render spore every 10 minutes to prevent spin-down.\"\"\"

        import time
        import httpx

        RENDER_URL = "https://synapse-spore-{spore_id[:8]}.onrender.com"
        INTERVAL = 600  # 10 minutes

        def main():
            client = httpx.Client(timeout=30)
            while True:
                try:
                    resp = client.get(f"{{RENDER_URL}}/health")
                    print(f"Ping: {{resp.status_code}}")
                except Exception as e:
                    print(f"Ping failed: {{e}}")
                time.sleep(INTERVAL)

        if __name__ == "__main__":
            main()
    """)

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for fname, content in files.items():
            (out / fname).write_text(content)

    return files
