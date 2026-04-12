"""
Command Center -- localhost dashboard for the Synapse Brain swarm.

Runs on :7700 by default. Provides:
  - Real-time swarm topology map
  - Per-spore health, latency, throughput
  - Task submission with document attachment
  - Provider configuration (Groq, OpenRouter, GLM, Mistral, local)
  - Live WebSocket feed of reasoning deltas
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aiohttp import web, WSMsgType
import aiohttp

log = logging.getLogger("synapse.dashboard")

STATIC_DIR = Path(__file__).parent / "static"
UPLOAD_DIR = Path(__file__).parent.parent.parent / "uploads"

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class SwarmState:
    """In-memory representation of the swarm visible to the dashboard."""

    spores: dict[str, dict[str, Any]] = field(default_factory=dict)
    tasks: dict[str, dict[str, Any]] = field(default_factory=dict)
    providers: dict[str, dict[str, Any]] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    ws_clients: list[web.WebSocketResponse] = field(default_factory=list)
    _start_time: float = field(default_factory=time.time)

    def uptime(self) -> float:
        return time.time() - self._start_time

    def summary(self) -> dict[str, Any]:
        active = sum(1 for s in self.spores.values() if s.get("status") == "active")
        total_deltas = sum(s.get("delta_count", 0) for s in self.spores.values())
        avg_latency = 0.0
        latencies = [s.get("latency_ms", 0) for s in self.spores.values() if s.get("latency_ms")]
        if latencies:
            avg_latency = sum(latencies) / len(latencies)

        return {
            "total_spores": len(self.spores),
            "active_spores": active,
            "total_tasks": len(self.tasks),
            "active_tasks": sum(1 for t in self.tasks.values() if t.get("status") == "running"),
            "completed_tasks": sum(1 for t in self.tasks.values() if t.get("status") == "converged"),
            "total_deltas": total_deltas,
            "avg_latency_ms": round(avg_latency, 2),
            "uptime_s": round(self.uptime(), 1),
            "provider_count": len(self.providers),
        }

    async def broadcast(self, event: dict[str, Any]) -> None:
        self.events.append(event)
        if len(self.events) > 5000:
            self.events = self.events[-2500:]
        dead = []
        for ws in self.ws_clients:
            try:
                await ws.send_json(event)
            except (ConnectionError, RuntimeError):
                dead.append(ws)
        for ws in dead:
            self.ws_clients.remove(ws)


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------

async def index(request: web.Request) -> web.Response:
    return web.FileResponse(STATIC_DIR / "index.html")


async def api_summary(request: web.Request) -> web.Response:
    state: SwarmState = request.app["state"]
    return web.json_response(state.summary())


async def api_spores(request: web.Request) -> web.Response:
    state: SwarmState = request.app["state"]
    return web.json_response(list(state.spores.values()))


async def api_spore_detail(request: web.Request) -> web.Response:
    state: SwarmState = request.app["state"]
    spore_id = request.match_info["spore_id"]
    spore = state.spores.get(spore_id)
    if not spore:
        return web.json_response({"error": "unknown spore"}, status=404)
    return web.json_response(spore)


async def api_tasks(request: web.Request) -> web.Response:
    state: SwarmState = request.app["state"]
    return web.json_response(list(state.tasks.values()))


async def api_submit_task(request: web.Request) -> web.Response:
    """Submit a new task to the swarm, optionally with attached documents."""
    state: SwarmState = request.app["state"]

    content_type = request.content_type
    task_id = f"task-{uuid.uuid4().hex[:12]}"
    attachments = []

    if "multipart" in content_type:
        reader = await request.multipart()
        task_text = ""
        task_config = {}
        async for part in reader:
            if part.name == "task":
                task_text = await part.text()
            elif part.name == "config":
                raw = await part.text()
                task_config = json.loads(raw) if raw else {}
            elif part.name == "documents":
                UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
                filename = part.filename or f"doc-{uuid.uuid4().hex[:8]}"
                dest = UPLOAD_DIR / task_id / filename
                dest.parent.mkdir(parents=True, exist_ok=True)
                with open(dest, "wb") as f:
                    while True:
                        chunk = await part.read_chunk(8192)
                        if not chunk:
                            break
                        f.write(chunk)
                attachments.append({"filename": filename, "path": str(dest)})
    else:
        body = await request.json()
        task_text = body.get("task", "")
        task_config = body.get("config", {})

    task_record = {
        "task_id": task_id,
        "text": task_text,
        "config": task_config,
        "attachments": attachments,
        "status": "pending",
        "submitted_at": time.time(),
        "deltas": [],
        "result": None,
    }
    state.tasks[task_id] = task_record

    await state.broadcast({
        "type": "task_submitted",
        "task_id": task_id,
        "text": task_text[:200],
        "attachment_count": len(attachments),
        "timestamp": time.time(),
    })

    # Dispatch to swarm if spores are connected
    asyncio.create_task(_dispatch_task(state, task_record))

    return web.json_response({"task_id": task_id, "status": "pending"})


async def _dispatch_task(state: SwarmState, task: dict[str, Any]) -> None:
    """Forward a task to all active spores via their mesh endpoints."""
    task["status"] = "running"
    await state.broadcast({
        "type": "task_dispatched",
        "task_id": task["task_id"],
        "spore_count": len([s for s in state.spores.values() if s.get("status") == "active"]),
        "timestamp": time.time(),
    })

    for spore_id, spore in state.spores.items():
        endpoint = spore.get("endpoint")
        if not endpoint or spore.get("status") != "active":
            continue
        try:
            async with aiohttp.ClientSession() as session:
                await session.post(
                    f"{endpoint}/task",
                    json={"task_id": task["task_id"], "text": task["text"], "config": task.get("config", {})},
                    timeout=aiohttp.ClientTimeout(total=5),
                )
        except Exception as e:
            log.warning("failed to dispatch to %s: %s", spore_id, e)


async def api_providers(request: web.Request) -> web.Response:
    state: SwarmState = request.app["state"]
    return web.json_response(list(state.providers.values()))


async def api_set_provider(request: web.Request) -> web.Response:
    """Configure an LLM provider."""
    state: SwarmState = request.app["state"]
    body = await request.json()
    provider_id = body.get("provider_id", f"provider-{uuid.uuid4().hex[:8]}")
    record = {
        "provider_id": provider_id,
        "name": body.get("name", "unknown"),
        "api_key": body.get("api_key", ""),
        "base_url": body.get("base_url", ""),
        "model": body.get("model", ""),
        "max_tokens": body.get("max_tokens", 4096),
        "enabled": body.get("enabled", True),
    }
    state.providers[provider_id] = record
    await state.broadcast({"type": "provider_configured", "provider_id": provider_id, "name": record["name"], "timestamp": time.time()})
    return web.json_response({"provider_id": provider_id, "status": "configured"})


async def api_events(request: web.Request) -> web.Response:
    state: SwarmState = request.app["state"]
    limit = int(request.query.get("limit", "100"))
    return web.json_response(state.events[-limit:])


# -- Spore registration (spores call this to announce themselves) --

async def api_register_spore(request: web.Request) -> web.Response:
    state: SwarmState = request.app["state"]
    body = await request.json()
    spore_id = body.get("spore_id", f"spore-{uuid.uuid4().hex[:8]}")
    record = {
        "spore_id": spore_id,
        "endpoint": body.get("endpoint", ""),
        "status": "active",
        "registered_at": time.time(),
        "last_heartbeat": time.time(),
        "delta_count": body.get("delta_count", 0),
        "latency_ms": body.get("latency_ms", 0),
        "trust_score": body.get("trust_score", 1.0),
        "provider": body.get("provider", "unknown"),
        "host": body.get("host", "unknown"),
        "region": body.get("region", "unknown"),
        "merkle_root": body.get("merkle_root", ""),
    }
    state.spores[spore_id] = record
    await state.broadcast({"type": "spore_registered", "spore_id": spore_id, "host": record["host"], "timestamp": time.time()})
    return web.json_response({"status": "registered", "spore_id": spore_id})


async def api_heartbeat(request: web.Request) -> web.Response:
    state: SwarmState = request.app["state"]
    body = await request.json()
    spore_id = body.get("spore_id")
    if spore_id and spore_id in state.spores:
        state.spores[spore_id].update({
            "last_heartbeat": time.time(),
            "delta_count": body.get("delta_count", state.spores[spore_id].get("delta_count", 0)),
            "latency_ms": body.get("latency_ms", state.spores[spore_id].get("latency_ms", 0)),
            "trust_score": body.get("trust_score", state.spores[spore_id].get("trust_score", 1.0)),
            "merkle_root": body.get("merkle_root", state.spores[spore_id].get("merkle_root", "")),
            "status": "active",
        })
    return web.json_response({"status": "ok"})


async def api_delta_feed(request: web.Request) -> web.Response:
    """Receive a reasoning delta from a spore and broadcast it."""
    state: SwarmState = request.app["state"]
    body = await request.json()
    task_id = body.get("task_id")
    if task_id and task_id in state.tasks:
        state.tasks[task_id].setdefault("deltas", []).append(body)

    await state.broadcast({
        "type": "delta_received",
        "spore_id": body.get("spore_id", "unknown"),
        "task_id": task_id,
        "confidence": body.get("confidence", 0),
        "content_preview": str(body.get("content", ""))[:120],
        "timestamp": time.time(),
    })
    return web.json_response({"status": "accepted"})


async def api_task_result(request: web.Request) -> web.Response:
    """Mark a task as converged with a final result."""
    state: SwarmState = request.app["state"]
    body = await request.json()
    task_id = body.get("task_id")
    if task_id and task_id in state.tasks:
        state.tasks[task_id]["status"] = "converged"
        state.tasks[task_id]["result"] = body.get("result")
        state.tasks[task_id]["converged_at"] = time.time()
        await state.broadcast({
            "type": "task_converged",
            "task_id": task_id,
            "timestamp": time.time(),
        })
    return web.json_response({"status": "ok"})


# -- WebSocket for live feed --

async def ws_feed(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    state: SwarmState = request.app["state"]
    state.ws_clients.append(ws)

    # Send current state snapshot
    await ws.send_json({"type": "snapshot", "data": state.summary(), "spores": list(state.spores.values()), "tasks": list(state.tasks.values())})

    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                data = json.loads(msg.data)
                if data.get("type") == "ping":
                    await ws.send_json({"type": "pong", "timestamp": time.time()})
            elif msg.type == WSMsgType.ERROR:
                break
    finally:
        if ws in state.ws_clients:
            state.ws_clients.remove(ws)

    return ws


# -- Health check for spore liveness --

async def _health_checker(app: web.Application) -> None:
    """Background task: mark spores stale if no heartbeat in 90s."""
    state: SwarmState = app["state"]
    while True:
        await asyncio.sleep(30)
        now = time.time()
        for spore_id, spore in state.spores.items():
            if now - spore.get("last_heartbeat", 0) > 90:
                if spore.get("status") != "stale":
                    spore["status"] = "stale"
                    await state.broadcast({
                        "type": "spore_stale",
                        "spore_id": spore_id,
                        "timestamp": now,
                    })


async def start_background(app: web.Application) -> None:
    app["health_checker"] = asyncio.create_task(_health_checker(app))


async def stop_background(app: web.Application) -> None:
    app["health_checker"].cancel()


# ---------------------------------------------------------------------------
# Live HF Spaces proxy -- polls deployed spores directly
# ---------------------------------------------------------------------------

HF_SPORE_URLS = [
    url.strip() for url in os.environ.get("SYNAPSE_SPORE_URLS", "").split(",") if url.strip()
] or [
    f"https://{os.environ.get('HF_SPACE_PREFIX', 'synapse-spore')}-{i:03d}.hf.space" for i in range(5)
]

async def _fetch_spore(session: aiohttp.ClientSession, url: str, hf_token: str) -> dict[str, Any]:
    """Fetch health + task data from one live HF spore."""
    headers = {"Authorization": f"Bearer {hf_token}"}
    sid = url.split("-")[-1].replace(".hf.space", "")
    result: dict[str, Any] = {"spore": f"spore-{sid}", "url": url, "status": "offline", "health": {}, "tasks": {}, "trust": []}
    try:
        async with session.get(f"{url}/api/health", headers=headers, timeout=aiohttp.ClientTimeout(total=12)) as r:
            if r.status == 200:
                result["health"] = await r.json()
                result["status"] = "online"
            else:
                result["status"] = "building"
                return result
    except Exception as e:
        result["error"] = str(e)[:100]
        return result

    # Trust
    try:
        async with session.get(f"{url}/api/trust", headers=headers, timeout=aiohttp.ClientTimeout(total=8)) as r:
            if r.status == 200:
                result["trust"] = (await r.json()).get("trust_scores", [])
    except Exception:
        pass

    # Tasks
    try:
        async with session.get(f"{url}/api/tasks", headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as r:
            if r.status == 200:
                task_ids = (await r.json()).get("tasks", [])
                for tid in task_ids:
                    try:
                        async with session.get(f"{url}/api/task/{tid}", headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as tr:
                            if tr.status == 200:
                                td = await tr.json()
                                td["task_id"] = tid
                                result["tasks"][tid] = td
                    except Exception:
                        pass
    except Exception:
        pass

    return result


async def api_live_snapshot(request: web.Request) -> web.Response:
    """Poll all HF spores and return a unified snapshot with full conversation data."""
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        token_path = Path.home() / ".hf_token"
        if not token_path.exists():
            token_path = Path.home() / ".cache" / "huggingface" / "token"
        if token_path.exists():
            hf_token = token_path.read_text().strip()

    if not hf_token:
        return web.json_response({"error": "No HF_TOKEN configured"}, status=500)

    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(*[_fetch_spore(session, url, hf_token) for url in HF_SPORE_URLS])

    # Merge task data across spores
    all_tasks: dict[str, dict[str, Any]] = {}
    conversation: list[dict[str, Any]] = []

    for spore_data in results:
        for tid, task in spore_data.get("tasks", {}).items():
            if tid not in all_tasks:
                all_tasks[tid] = task
            # Extract conversation entries from contributor_detail
            detail = task.get("contributor_detail", {})
            for contrib_id, cd in detail.items():
                conversation.append({
                    "spore": contrib_id,
                    "role": cd.get("role", "?"),
                    "model": cd.get("model", "?"),
                    "cycles": cd.get("cycles", 0),
                    "hypothesis": cd.get("hypothesis", ""),
                    "claims": cd.get("claims", []),
                    "confidence": cd.get("confidence", 0),
                    "phase": cd.get("phase", "?"),
                    "task_id": tid,
                })

    # Sort conversation by spore name for consistent display
    conversation.sort(key=lambda c: (c["task_id"], c["spore"]))

    snapshot = {
        "timestamp": time.time(),
        "spores": [
            {
                "spore": r["spore"],
                "url": r["url"],
                "status": r["status"],
                "role": r.get("health", {}).get("role", "?"),
                "primary_model": r.get("health", {}).get("primary_model", "?"),
                "clock": r.get("health", {}).get("clock", 0),
                "cycles": r.get("health", {}).get("cycles", 0),
                "peers": r.get("health", {}).get("peers", 0),
                "active_tasks": r.get("health", {}).get("active_tasks", 0),
                "trust": r.get("trust", []),
                "errors": r.get("health", {}).get("errors", []),
            }
            for r in results
        ],
        "tasks": list(all_tasks.values()),
        "conversation": conversation,
    }
    return web.json_response(snapshot)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(state: SwarmState | None = None) -> web.Application:
    app = web.Application(client_max_size=100 * 1024 * 1024)  # 100MB upload limit
    app["state"] = state or SwarmState()

    # Static assets
    app.router.add_get("/", index)
    app.router.add_static("/static", STATIC_DIR, name="static")

    # Dashboard API
    app.router.add_get("/api/summary", api_summary)
    app.router.add_get("/api/spores", api_spores)
    app.router.add_get("/api/spores/{spore_id}", api_spore_detail)
    app.router.add_get("/api/tasks", api_tasks)
    app.router.add_post("/api/tasks", api_submit_task)
    app.router.add_get("/api/providers", api_providers)
    app.router.add_post("/api/providers", api_set_provider)
    app.router.add_get("/api/events", api_events)

    # Spore registration
    app.router.add_post("/api/spores/register", api_register_spore)
    app.router.add_post("/api/spores/heartbeat", api_heartbeat)

    # Delta and task result feed
    app.router.add_post("/api/deltas", api_delta_feed)
    app.router.add_post("/api/tasks/result", api_task_result)

    # Live HF spore proxy
    app.router.add_get("/api/live/snapshot", api_live_snapshot)

    # WebSocket
    app.router.add_get("/ws", ws_feed)

    # Background health checker
    app.on_startup.append(start_background)
    app.on_cleanup.append(stop_background)

    return app


def launch(host: str = "0.0.0.0", port: int = 7700) -> None:
    """Launch the Command Center."""
    print(f"\n  Synapse Brain -- Command Center")
    print(f"  http://localhost:{port}\n")
    app = create_app()
    web.run_app(app, host=host, port=port, print=None)


if __name__ == "__main__":
    launch()
