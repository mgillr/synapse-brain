"""Synapse Brain CLI -- launch dashboard, deploy spores, run tasks."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys


def cmd_dashboard(args):
    """Launch the localhost command center."""
    from synapse_brain.dashboard.server import DashboardServer

    server = DashboardServer(
        spore_count=args.spores,
        port=args.port,
        provider_keys=_collect_provider_keys(),
    )
    server.run()


def cmd_run(args):
    """Run a single task through the swarm."""
    from synapse_brain import Orchestrator

    swarm = Orchestrator(
        spore_count=args.spores,
        provider_keys=_collect_provider_keys(),
    )

    # Load documents if provided
    documents = []
    if args.documents:
        for path in args.documents:
            with open(path, "r") as f:
                documents.append({"name": os.path.basename(path), "content": f.read()})

    task = swarm.submit_task(
        description=args.task,
        documents=documents,
    )

    async def _run():
        result = await swarm.run_task(task, max_cycles=args.cycles)
        return result

    result = asyncio.run(_run())

    print(f"\nTask: {result.description}")
    print(f"Status: {result.status}")
    if result.convergence:
        print(f"Agreement: {result.convergence.agreement_ratio:.0%}")
        print(f"Cycles: {result.convergence.cycle_count}")
    print(f"\nAnswer:\n{result.final_answer}")


def cmd_deploy(args):
    """Deploy spores to free hosting platforms."""
    target = args.target
    count = args.count

    if target == "hf-spaces":
        from synapse_brain.deployers.hf_spaces import deploy_spores
        deploy_spores(count)
    elif target == "oracle":
        from synapse_brain.deployers.oracle_cloud import deploy_spores
        deploy_spores(count)
    elif target == "render":
        from synapse_brain.deployers.render import deploy_spores
        deploy_spores(count)
    elif target == "cloudflare-relay":
        from synapse_brain.deployers.cloudflare_relay import deploy_relay
        deploy_relay()
    else:
        print(f"Unknown target: {target}")
        print("Available: hf-spaces, oracle, render, cloudflare-relay")
        sys.exit(1)


def _collect_provider_keys() -> dict[str, str]:
    """Collect API keys from environment."""
    keys = {}
    for env_var in [
        "GOOGLE_AI_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY",
        "CEREBRAS_API_KEY", "MISTRAL_API_KEY", "HF_TOKEN",
        "CLOUDFLARE_API_TOKEN", "GITHUB_TOKEN", "TOGETHER_API_KEY",
        "NVIDIA_API_KEY", "COHERE_API_KEY", "GLM_API_KEY",
    ]:
        val = os.getenv(env_var)
        if val:
            keys[env_var] = val
    return keys


def main():
    parser = argparse.ArgumentParser(
        description="Synapse Brain -- distributed reasoning swarm",
    )
    subparsers = parser.add_subparsers(dest="command")

    # Dashboard
    dash = subparsers.add_parser("dashboard", help="Launch command center")
    dash.add_argument("--spores", type=int, default=20, help="Local spore count")
    dash.add_argument("--port", type=int, default=7770, help="Dashboard port")

    # Run
    run = subparsers.add_parser("run", help="Run a task through the swarm")
    run.add_argument("task", help="Task description")
    run.add_argument("--spores", type=int, default=10, help="Spore count")
    run.add_argument("--cycles", type=int, default=8, help="Max reasoning cycles")
    run.add_argument("--documents", nargs="*", help="Document paths to attach")

    # Deploy
    deploy = subparsers.add_parser("deploy", help="Deploy spores to free hosts")
    deploy.add_argument("--target", required=True, help="Target platform")
    deploy.add_argument("--count", type=int, default=5, help="Spore count")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
    )

    if args.command == "dashboard":
        cmd_dashboard(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "deploy":
        cmd_deploy(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
