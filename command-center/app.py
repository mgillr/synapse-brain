"""
Synapse Brain Command Center -- HuggingFace Space deployment.

Monitors all private spores in the Optitransfer swarm. Provides:
  - Real-time spore health, model identity, peer mesh status
  - Full conversation library: every reasoning delta from every spore
  - Task submission to the swarm
  - Convergence tracking with agreement history
  - Trust score matrix across the mesh
"""

import asyncio
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import gradio as gr
import httpx

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HF_TOKEN = os.environ.get("HF_TOKEN", "")
NUM_SPORES = 5
SPORE_URLS = [f"https://optitransfer-synapse-spore-{i:03d}.hf.space" for i in range(NUM_SPORES)]
POLL_TIMEOUT = 15.0

# Thread pool for parallel HTTP
executor = ThreadPoolExecutor(max_workers=NUM_SPORES + 2)


# ---------------------------------------------------------------------------
# Spore polling
# ---------------------------------------------------------------------------

def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}


def fetch_spore(url: str) -> dict[str, Any]:
    """Fetch full state from a single spore."""
    sid = url.split("-")[-1].replace(".hf.space", "")
    name = f"spore-{sid}"
    result: dict[str, Any] = {
        "name": name, "url": url, "status": "offline",
        "health": {}, "tasks": {}, "trust": [], "error": None,
    }
    try:
        with httpx.Client(timeout=POLL_TIMEOUT) as client:
            # Health
            r = client.get(f"{url}/api/health", headers=_headers())
            if r.status_code == 200:
                result["health"] = r.json()
                result["status"] = "online"
            elif r.status_code == 503:
                result["status"] = "building"
                return result
            else:
                result["status"] = f"error-{r.status_code}"
                return result

            # Trust
            try:
                r = client.get(f"{url}/api/trust", headers=_headers())
                if r.status_code == 200:
                    result["trust"] = r.json().get("trust_scores", [])
            except Exception:
                pass

            # Tasks
            try:
                r = client.get(f"{url}/api/tasks", headers=_headers())
                if r.status_code == 200:
                    task_ids = r.json().get("tasks", [])
                    for tid in task_ids:
                        try:
                            r2 = client.get(f"{url}/api/task/{tid}", headers=_headers())
                            if r2.status_code == 200:
                                td = r2.json()
                                td["_task_id"] = tid
                                result["tasks"][tid] = td
                        except Exception:
                            pass
            except Exception:
                pass

    except httpx.ConnectError:
        result["status"] = "unreachable"
        result["error"] = "connection refused"
    except httpx.TimeoutException:
        result["status"] = "timeout"
        result["error"] = "request timed out"
    except Exception as e:
        result["error"] = str(e)[:120]

    return result


def poll_all_spores() -> list[dict[str, Any]]:
    """Poll all spores in parallel."""
    futures = [executor.submit(fetch_spore, url) for url in SPORE_URLS]
    return [f.result() for f in futures]


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def build_snapshot() -> dict[str, Any]:
    """Build a complete snapshot of the swarm state."""
    spores = poll_all_spores()

    # Merge tasks across all spores (same task seen by multiple spores)
    all_tasks: dict[str, dict[str, Any]] = {}
    for s in spores:
        for tid, td in s.get("tasks", {}).items():
            if tid not in all_tasks:
                all_tasks[tid] = td
            else:
                # Merge contributor details
                existing = all_tasks[tid].get("contributor_detail", {})
                incoming = td.get("contributor_detail", {})
                existing.update(incoming)
                all_tasks[tid]["contributor_detail"] = existing
                # Take higher delta count
                if td.get("delta_count", 0) > all_tasks[tid].get("delta_count", 0):
                    all_tasks[tid]["delta_count"] = td["delta_count"]

    # Build conversation from contributor details
    conversation = []
    for tid, td in all_tasks.items():
        task_desc = td.get("task", td.get("description", "Unknown task"))
        detail = td.get("contributor_detail", {})
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
                "task_desc": task_desc[:120],
            })

        # Check for synthesis
        synthesis = td.get("synthesis")
        if synthesis:
            conversation.append({
                "spore": "SYNTHESIS",
                "role": "collective",
                "model": "swarm-consensus",
                "cycles": 0,
                "hypothesis": synthesis if isinstance(synthesis, str) else json.dumps(synthesis),
                "claims": [],
                "confidence": td.get("agreement", 0),
                "phase": "converged",
                "task_id": tid,
                "task_desc": task_desc[:120],
            })

    conversation.sort(key=lambda c: (c["task_id"], c.get("cycles", 0), c["spore"]))

    return {
        "timestamp": time.time(),
        "spores": spores,
        "tasks": list(all_tasks.values()),
        "conversation": conversation,
    }


# ---------------------------------------------------------------------------
# Display formatters
# ---------------------------------------------------------------------------

ROLE_COLORS = {
    "explorer": "#3b82f6",
    "synthesizer": "#8b5cf6",
    "adversarial": "#ef4444",
    "validator": "#22c55e",
    "generalist": "#f59e0b",
    "collective": "#06b6d4",
}


def format_spore_grid(snapshot: dict) -> str:
    """Format spore status as HTML grid."""
    html = '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px;margin-bottom:16px;">'

    for s in snapshot.get("spores", []):
        health = s.get("health", {})
        status = s.get("status", "offline")
        role = health.get("role", "?")
        model = health.get("primary_model", "?")
        clock = health.get("clock", 0)
        cycles = health.get("cycles", 0)
        peers = health.get("peers", 0)
        deltas_out = health.get("deltas_produced", 0)
        deltas_in = health.get("deltas_received", 0)
        active_tasks = health.get("active_tasks", 0)
        color = ROLE_COLORS.get(role, "#666")

        if status == "online":
            dot = '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#22c55e;margin-right:6px;"></span>'
        elif status == "building":
            dot = '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#f59e0b;margin-right:6px;animation:pulse 1s infinite;"></span>'
        else:
            dot = '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#ef4444;margin-right:6px;"></span>'

        # Trim model name for display
        model_short = model.split("/")[-1] if "/" in model else model
        if len(model_short) > 35:
            model_short = model_short[:32] + "..."

        html += f'''
        <div style="background:#1a1a2e;border:1px solid {color}40;border-radius:8px;padding:12px;position:relative;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
            <div style="display:flex;align-items:center;">
              {dot}
              <span style="font-weight:600;color:#e0e0e0;font-size:13px;">{s["name"]}</span>
            </div>
            <span style="background:{color}30;color:{color};padding:2px 8px;border-radius:4px;font-size:11px;font-weight:500;text-transform:uppercase;">{role}</span>
          </div>
          <div style="font-size:11px;color:#888;margin-bottom:6px;font-family:monospace;">{model_short}</div>
          <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px;font-size:11px;">
            <div style="color:#888;">Clock: <span style="color:#e0e0e0;">{clock}</span></div>
            <div style="color:#888;">Cycles: <span style="color:#e0e0e0;">{cycles}</span></div>
            <div style="color:#888;">Peers: <span style="color:#e0e0e0;">{peers}</span></div>
            <div style="color:#888;">Deltas out: <span style="color:#e0e0e0;">{deltas_out}</span></div>
            <div style="color:#888;">Deltas in: <span style="color:#e0e0e0;">{deltas_in}</span></div>
            <div style="color:#888;">Tasks: <span style="color:#e0e0e0;">{active_tasks}</span></div>
          </div>
        </div>'''

    html += '</div>'
    return html


def format_conversation(snapshot: dict, filter_spore: str = "all", filter_task: str = "all") -> str:
    """Format the full conversation as an HTML timeline."""
    convos = snapshot.get("conversation", [])

    if filter_spore != "all":
        convos = [c for c in convos if c["spore"] == filter_spore or c["spore"] == "SYNTHESIS"]
    if filter_task != "all":
        convos = [c for c in convos if c["task_id"] == filter_task]

    if not convos:
        return '<div style="text-align:center;padding:40px;color:#666;">No conversation data yet. Submit a task and wait for reasoning cycles.</div>'

    # Group by task
    tasks: dict[str, list] = {}
    for c in convos:
        tid = c["task_id"]
        if tid not in tasks:
            tasks[tid] = []
        tasks[tid].append(c)

    html = ''
    for tid, entries in tasks.items():
        task_desc = entries[0].get("task_desc", "Unknown")
        html += f'''
        <div style="margin-bottom:24px;">
          <div style="background:#0d0d1a;border:1px solid #333;border-radius:8px;padding:12px 16px;margin-bottom:12px;">
            <div style="font-size:11px;color:#666;margin-bottom:4px;">TASK {tid[:12]}</div>
            <div style="color:#e0e0e0;font-size:13px;">{task_desc}</div>
          </div>
          <div style="position:relative;padding-left:24px;border-left:2px solid #333;">'''

        for entry in entries:
            role = entry.get("role", "?")
            color = ROLE_COLORS.get(role, "#666")
            spore = entry.get("spore", "?")
            model = entry.get("model", "?")
            model_short = model.split("/")[-1] if "/" in model else model
            phase = entry.get("phase", "?")
            confidence = entry.get("confidence", 0)
            hypothesis = entry.get("hypothesis", "")
            claims = entry.get("claims", [])
            cycles = entry.get("cycles", 0)

            is_synthesis = spore == "SYNTHESIS"

            if is_synthesis:
                html += f'''
            <div style="position:relative;margin-bottom:16px;margin-left:8px;">
              <div style="position:absolute;left:-33px;top:8px;width:12px;height:12px;border-radius:50%;background:#06b6d4;border:2px solid #0d0d1a;"></div>
              <div style="background:linear-gradient(135deg,#06b6d420,#8b5cf620);border:1px solid #06b6d450;border-radius:8px;padding:14px;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                  <span style="font-weight:700;color:#06b6d4;font-size:14px;">CONVERGED SYNTHESIS</span>
                  <span style="color:#06b6d4;font-size:12px;">Agreement: {confidence:.0%}</span>
                </div>
                <div style="color:#e0e0e0;font-size:13px;line-height:1.6;white-space:pre-wrap;">{hypothesis}</div>
              </div>
            </div>'''
            else:
                conf_pct = f"{confidence:.0%}" if isinstance(confidence, float) else f"{confidence}%"
                claims_html = ""
                if claims:
                    claims_html = '<div style="margin-top:6px;">'
                    for cl in claims[:5]:
                        claims_html += f'<div style="font-size:11px;color:#aaa;padding:2px 0;border-left:2px solid {color}40;padding-left:8px;margin-top:2px;">{cl}</div>'
                    claims_html += '</div>'

                html += f'''
            <div style="position:relative;margin-bottom:12px;margin-left:8px;">
              <div style="position:absolute;left:-33px;top:8px;width:10px;height:10px;border-radius:50%;background:{color};border:2px solid #0d0d1a;"></div>
              <div style="background:#1a1a2e;border:1px solid {color}30;border-radius:8px;padding:12px;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                  <div style="display:flex;align-items:center;gap:8px;">
                    <span style="font-weight:600;color:{color};font-size:12px;">{spore}</span>
                    <span style="background:{color}20;color:{color};padding:1px 6px;border-radius:3px;font-size:10px;text-transform:uppercase;">{role}</span>
                    <span style="color:#555;font-size:10px;font-family:monospace;">{model_short}</span>
                  </div>
                  <div style="display:flex;align-items:center;gap:8px;">
                    <span style="color:#555;font-size:10px;">Cycle {cycles}</span>
                    <span style="color:#888;font-size:10px;">Phase: {phase}</span>
                    <span style="color:{color};font-size:11px;font-weight:500;">{conf_pct}</span>
                  </div>
                </div>
                <div style="color:#ccc;font-size:12px;line-height:1.5;white-space:pre-wrap;">{hypothesis}</div>
                {claims_html}
              </div>
            </div>'''

        html += '</div></div>'

    return html


def format_trust_matrix(snapshot: dict) -> str:
    """Format trust scores as a matrix."""
    spores = snapshot.get("spores", [])
    all_trust = {}
    for s in spores:
        name = s["name"]
        for t in s.get("trust", []):
            peer = t.get("peer", "?")
            score = t.get("trust", t.get("score", 0))
            all_trust.setdefault(name, {})[peer] = score

    if not all_trust:
        return '<div style="text-align:center;padding:20px;color:#666;">No trust data available yet.</div>'

    names = sorted(set(list(all_trust.keys()) + [p for peers in all_trust.values() for p in peers]))

    html = '<table style="width:100%;border-collapse:collapse;font-size:11px;font-family:monospace;">'
    html += '<tr><th style="padding:6px;color:#666;text-align:left;border-bottom:1px solid #333;"></th>'
    for n in names:
        short = n.replace("spore-", "S")
        html += f'<th style="padding:6px;color:#888;text-align:center;border-bottom:1px solid #333;">{short}</th>'
    html += '</tr>'

    for row_name in names:
        short_row = row_name.replace("spore-", "S")
        html += f'<tr><td style="padding:6px;color:#888;border-bottom:1px solid #222;">{short_row}</td>'
        for col_name in names:
            if row_name == col_name:
                html += '<td style="padding:6px;text-align:center;color:#333;border-bottom:1px solid #222;">--</td>'
            else:
                score = all_trust.get(row_name, {}).get(col_name, 0)
                # Color based on score
                if score >= 0.3:
                    c = "#22c55e"
                elif score >= 0.15:
                    c = "#f59e0b"
                else:
                    c = "#ef4444"
                html += f'<td style="padding:6px;text-align:center;color:{c};border-bottom:1px solid #222;">{score:.2f}</td>'
        html += '</tr>'

    html += '</table>'
    return html


def format_summary_stats(snapshot: dict) -> str:
    """Top-level stats bar."""
    spores = snapshot.get("spores", [])
    online = sum(1 for s in spores if s["status"] == "online")
    total_cycles = sum(s.get("health", {}).get("cycles", 0) for s in spores)
    total_deltas = sum(s.get("health", {}).get("deltas_produced", 0) for s in spores)
    total_received = sum(s.get("health", {}).get("deltas_received", 0) for s in spores)
    n_tasks = len(snapshot.get("tasks", []))
    n_convos = len(snapshot.get("conversation", []))

    return f'''
    <div style="display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin-bottom:16px;">
      <div style="background:#1a1a2e;border-radius:8px;padding:12px;text-align:center;">
        <div style="font-size:24px;font-weight:700;color:{"#22c55e" if online == NUM_SPORES else "#f59e0b"}">{online}/{NUM_SPORES}</div>
        <div style="font-size:10px;color:#666;text-transform:uppercase;">Spores Online</div>
      </div>
      <div style="background:#1a1a2e;border-radius:8px;padding:12px;text-align:center;">
        <div style="font-size:24px;font-weight:700;color:#3b82f6;">{total_cycles}</div>
        <div style="font-size:10px;color:#666;text-transform:uppercase;">Total Cycles</div>
      </div>
      <div style="background:#1a1a2e;border-radius:8px;padding:12px;text-align:center;">
        <div style="font-size:24px;font-weight:700;color:#8b5cf6;">{total_deltas}</div>
        <div style="font-size:10px;color:#666;text-transform:uppercase;">Deltas Produced</div>
      </div>
      <div style="background:#1a1a2e;border-radius:8px;padding:12px;text-align:center;">
        <div style="font-size:24px;font-weight:700;color:#06b6d4;">{total_received}</div>
        <div style="font-size:10px;color:#666;text-transform:uppercase;">Deltas Received</div>
      </div>
      <div style="background:#1a1a2e;border-radius:8px;padding:12px;text-align:center;">
        <div style="font-size:24px;font-weight:700;color:#f59e0b;">{n_tasks}</div>
        <div style="font-size:10px;color:#666;text-transform:uppercase;">Active Tasks</div>
      </div>
      <div style="background:#1a1a2e;border-radius:8px;padding:12px;text-align:center;">
        <div style="font-size:24px;font-weight:700;color:#22c55e;">{n_convos}</div>
        <div style="font-size:10px;color:#666;text-transform:uppercase;">Contributions</div>
      </div>
    </div>'''


# ---------------------------------------------------------------------------
# Gradio callbacks
# ---------------------------------------------------------------------------

def refresh_dashboard():
    """Main refresh callback -- polls all spores and returns formatted HTML."""
    snapshot = build_snapshot()
    stats = format_summary_stats(snapshot)
    grid = format_spore_grid(snapshot)
    convo = format_conversation(snapshot)
    trust = format_trust_matrix(snapshot)
    return stats, grid, convo, trust


def submit_task(task_text: str) -> str:
    """Submit a task to the first online spore."""
    if not task_text.strip():
        return "Enter a task description."

    for url in SPORE_URLS:
        try:
            with httpx.Client(timeout=10) as client:
                r = client.post(
                    f"{url}/api/task",
                    json={"task": task_text.strip()},
                    headers=_headers(),
                )
                if r.status_code == 200:
                    data = r.json()
                    tid = data.get("task_id", "?")
                    return f"Task {tid[:12]} submitted via {url.split('/')[-1]}. Gossip will propagate to all spores."
        except Exception:
            continue

    return "Failed to submit -- no spores reachable."


def filter_conversation(spore_filter: str, task_filter: str) -> str:
    """Refresh conversation with filters applied."""
    snapshot = build_snapshot()
    return format_conversation(snapshot, filter_spore=spore_filter, filter_task=task_filter)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

CSS = """
.gradio-container { background: #0d0d1a !important; max-width: 100% !important; }
.tab-nav button { background: #1a1a2e !important; color: #888 !important; border: 1px solid #333 !important; }
.tab-nav button.selected { background: #2a2a4e !important; color: #e0e0e0 !important; border-color: #3b82f6 !important; }
footer { display: none !important; }
"""

with gr.Blocks(css=CSS, theme=gr.themes.Base(), title="Synapse Brain Command Center") as app:
    gr.Markdown(
        "# Synapse Brain -- Command Center\n"
        "Real-time monitoring of the cognitive swarm. "
        "Each spore runs a different LLM, reasonoing from a unique cognitive role."
    )

    with gr.Tab("Dashboard"):
        stats_html = gr.HTML(label="Stats")
        grid_html = gr.HTML(label="Spore Grid")
        refresh_btn = gr.Button("Refresh Dashboard", variant="primary", size="sm")

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Submit Task to Swarm")
                task_input = gr.Textbox(
                    placeholder="Describe a reasoning task for the swarm...",
                    lines=3, label="Task",
                )
                submit_btn = gr.Button("Submit", variant="secondary", size="sm")
                submit_result = gr.Textbox(label="Result", interactive=False)
            with gr.Column(scale=1):
                gr.Markdown("### Trust Matrix")
                trust_html = gr.HTML()

    with gr.Tab("Library"):
        gr.Markdown(
            "### Conversation Library\n"
            "Full reasoning history from every spore. "
            "Each contribution shows the spore's role, model, phase, confidence, and claims."
        )
        with gr.Row():
            spore_filter = gr.Dropdown(
                choices=["all"] + [f"spore-{i:03d}" for i in range(NUM_SPORES)],
                value="all", label="Filter by Spore", scale=1,
            )
            task_filter = gr.Dropdown(
                choices=["all"], value="all", label="Filter by Task", scale=1,
            )
            lib_refresh_btn = gr.Button("Refresh", variant="primary", size="sm", scale=0)

        convo_html = gr.HTML()

    # Wire up Dashboard tab
    refresh_btn.click(
        fn=refresh_dashboard,
        outputs=[stats_html, grid_html, convo_html, trust_html],
    )

    submit_btn.click(
        fn=submit_task,
        inputs=[task_input],
        outputs=[submit_result],
    )

    # Wire up Library tab
    lib_refresh_btn.click(
        fn=filter_conversation,
        inputs=[spore_filter, task_filter],
        outputs=[convo_html],
    )

    # Auto-refresh on load
    app.load(
        fn=refresh_dashboard,
        outputs=[stats_html, grid_html, convo_html, trust_html],
    )


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
