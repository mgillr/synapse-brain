"""Synapse Command Center -- Dashboard, Sentinel changelog, Library archive."""
import os, json, time, httpx, gradio as gr, threading, html as html_mod
from datetime import datetime

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
SPORES = [
    {"id": "000", "url": "https://optitransfer-synapse-spore-000.hf.space", "role": "Explorer", "model": "Qwen3 235B MoE", "color": "#3b82f6"},
    {"id": "001", "url": "https://optitransfer-synapse-spore-001.hf.space", "role": "Synthesizer", "model": "Llama 3.3 70B", "color": "#8b5cf6"},
    {"id": "002", "url": "https://optitransfer-synapse-spore-002.hf.space", "role": "Adversarial", "model": "DeepSeek R1 32B", "color": "#ef4444"},
    {"id": "003", "url": "https://optitransfer-synapse-spore-003.hf.space", "role": "Validator", "model": "Gemma 3 27B", "color": "#f59e0b"},
    {"id": "004", "url": "https://optitransfer-synapse-spore-004.hf.space", "role": "Generalist", "model": "Llama 4 Scout 17B", "color": "#10b981"},
    {"id": "005", "url": "https://optitransfer-synapse-spore-005.hf.space", "role": "Brain", "model": "GLM-4.7-Flash", "color": "#ec4899"},
    {"id": "006", "url": "https://optitransfer-synapse-spore-006.hf.space", "role": "Sentinel", "model": "DeepSeek R1 671B", "color": "#f97316"},
]
N_SPORES = len(SPORES)

SENTINEL_URL = SPORES[-1]["url"]

# Lookup maps -- handle "spore-001" and "synapse-spore-005-80d595" formats
SPORE_BY_KEY = {}
for _s in SPORES:
    SPORE_BY_KEY[f"spore-{_s['id']}"] = _s
    SPORE_BY_KEY[_s["id"]] = _s

def resolve_spore(author_str):
    """Resolve any author string to its spore info dict."""
    if author_str in SPORE_BY_KEY:
        return SPORE_BY_KEY[author_str]
    for s in SPORES:
        if f"spore-{s['id']}" in str(author_str):
            return s
    return {"id": "?", "role": "Unknown", "model": str(author_str), "color": "#6b7280"}

PHASE_COLORS = {"DIVERGE": "#3b82f6", "DEEPEN": "#f59e0b", "CONVERGE": "#10b981", "SYNTHESIZE": "#8b5cf6"}

_state = {"current_task": None, "lock": threading.Lock()}


def safe_get(url, timeout=8):
    try:
        r = httpx.get(url, headers=HEADERS, timeout=timeout, follow_redirects=True)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def safe_post(url, data, timeout=15):
    try:
        r = httpx.post(url, json=data, headers=HEADERS, timeout=timeout, follow_redirects=True)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def esc(text):
    return html_mod.escape(str(text)) if text else ""


# -----------------------------------------------------------------------
#  STATS BAR + SPORE CARDS
# -----------------------------------------------------------------------
def build_stats_html():
    total_mem = 0; total_deltas = 0; total_cycles = 0; online = 0
    cards = []
    for s in SPORES:
        h = safe_get(f"{s['url']}/api/health")
        if h:
            online += 1
            mem = h.get("memories", h.get("memory_size", 0))
            cyc = h.get("cycles", h.get("reasoning_cycles", 0))
            dp = h.get("deltas_produced", 0)
            dr = h.get("deltas_received", 0)
            d_total = dp + dr
            peers_raw = h.get("peers", h.get("connected_peers", []))
            n_peers = len(peers_raw) if isinstance(peers_raw, list) else int(peers_raw)
            conf = h.get("confidence", 0)
            tier = h.get("last_tier", h.get("tier", "worker"))
            total_mem += mem; total_deltas += d_total; total_cycles += cyc
            conf_pct = int(conf * 100) if conf <= 1 else int(conf)
            tier_bg = "#ec4899" if "brain" in str(tier).lower() else "#6b7280"
            if s["role"] == "Sentinel":
                tier_bg = "#f97316"
            cards.append(f'''
            <div style="background:#1e293b;border:1px solid #334155;border-radius:8px;padding:12px;min-width:160px">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
                <span style="background:{s['color']};color:#fff;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600">{s['role']}</span>
                <span style="color:#10b981">&#9679;</span>
                <span style="background:{tier_bg};color:#fff;padding:1px 6px;border-radius:8px;font-size:10px">{esc(tier).upper()}</span>
              </div>
              <div style="font-size:12px;color:#94a3b8;margin-bottom:4px">Spore {s['id']} -- {s['model']}</div>
              <div style="display:flex;gap:10px;font-size:11px;color:#64748b;flex-wrap:wrap">
                <span>Mem {mem:,}</span><span>Cyc {cyc}</span><span>Del {d_total}</span><span>Peers {n_peers}</span>
              </div>
              <div style="background:#374151;border-radius:4px;height:5px;width:100%;margin-top:6px">
                <div style="background:{s['color']};height:5px;border-radius:4px;width:{conf_pct}%"></div>
              </div>
              <div style="font-size:10px;color:#475569;margin-top:2px">Confidence {conf_pct}%</div>
            </div>''')
        else:
            cards.append(f'''
            <div style="background:#1e293b;border:1px solid #334155;border-radius:8px;padding:12px;min-width:160px;opacity:0.45">
              <span style="background:{s['color']};color:#fff;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600">{s['role']}</span>
              <span style="color:#ef4444;margin-left:6px">&#9679; OFFLINE</span>
              <div style="font-size:12px;color:#64748b;margin-top:4px">Spore {s['id']} -- {s['model']}</div>
            </div>''')

    summary = f'''
    <div style="display:flex;gap:16px;padding:8px 0;margin-bottom:10px;flex-wrap:wrap">
      <div style="background:#1e293b;border:1px solid #334155;border-radius:8px;padding:6px 14px;text-align:center">
        <div style="font-size:20px;font-weight:700;color:#10b981">{online}/{N_SPORES}</div><div style="font-size:10px;color:#64748b">Online</div>
      </div>
      <div style="background:#1e293b;border:1px solid #334155;border-radius:8px;padding:6px 14px;text-align:center">
        <div style="font-size:20px;font-weight:700;color:#3b82f6">{total_mem:,}</div><div style="font-size:10px;color:#64748b">Memories</div>
      </div>
      <div style="background:#1e293b;border:1px solid #334155;border-radius:8px;padding:6px 14px;text-align:center">
        <div style="font-size:20px;font-weight:700;color:#8b5cf6">{total_deltas:,}</div><div style="font-size:10px;color:#64748b">Deltas</div>
      </div>
      <div style="background:#1e293b;border:1px solid #334155;border-radius:8px;padding:6px 14px;text-align:center">
        <div style="font-size:20px;font-weight:700;color:#f59e0b">{total_cycles:,}</div><div style="font-size:10px;color:#64748b">Cycles</div>
      </div>
    </div>'''
    # 4 columns for 7 spores (4+3)
    grid = f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px">{chr(10).join(cards)}</div>'
    return summary + grid


# -----------------------------------------------------------------------
#  TRUST MATRIX
# -----------------------------------------------------------------------
def _resolve_trust_val(t, spore_id):
    candidates = [f"spore-{spore_id}", spore_id]
    for k in t:
        if f"spore-{spore_id}" in k or f"-{spore_id}-" in k or k.endswith(f"-{spore_id}"):
            candidates.append(k)
    for key in candidates:
        if key in t:
            val = t[key]
            if isinstance(val, dict):
                return float(val.get("overall", val.get("trust", 0)))
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return None


def build_trust_html():
    rows = []
    for s in SPORES:
        t = safe_get(f"{s['url']}/api/trust")
        if t:
            cells = []
            for s2 in SPORES:
                v = _resolve_trust_val(t, s2["id"])
                if v is not None:
                    bg = f"rgba(16,185,129,{v})" if v > 0.5 else f"rgba(239,68,68,{1-v})"
                    cells.append(f'<td style="background:{bg};text-align:center;padding:3px;font-size:10px;color:#fff">{v:.2f}</td>')
                else:
                    cells.append('<td style="text-align:center;padding:3px;font-size:10px;color:#475569">--</td>')
            rows.append(f'<tr><td style="padding:3px;font-size:10px;color:{s["color"]};font-weight:600">{s["id"]}</td>{"".join(cells)}</tr>')
    if not rows:
        return ""
    hdrs = "".join(f'<th style="padding:3px;font-size:9px;color:#94a3b8">{s["id"]}</th>' for s in SPORES)
    return f'''<div style="margin-top:10px"><div style="font-size:12px;font-weight:600;color:#e2e8f0;margin-bottom:4px">Trust Matrix</div>
    <table style="border-collapse:collapse;width:100%;background:#0f172a;border-radius:6px"><tr><th></th>{hdrs}</tr>{"".join(rows)}</table></div>'''


# -----------------------------------------------------------------------
#  TASK FETCHING
# -----------------------------------------------------------------------
def fetch_all_tasks():
    merged = {}
    for s in SPORES:
        raw = safe_get(f"{s['url']}/api/tasks")
        if not raw:
            continue
        if isinstance(raw, dict):
            for tid, tdata in raw.items():
                if tid not in merged or (tdata.get("delta_count", 0) > merged[tid].get("delta_count", 0)):
                    tdata["task_id"] = tid
                    merged[tid] = tdata
        elif isinstance(raw, list):
            for t in raw:
                tid = t.get("task_id", t.get("id", ""))
                if tid and (tid not in merged or t.get("delta_count", 0) > merged[tid].get("delta_count", 0)):
                    merged[tid] = t
    return merged


def fetch_task_detail(task_id):
    best = None
    longest_answer = ""
    for s in SPORES:
        detail = safe_get(f"{s['url']}/api/task/{task_id}")
        if detail:
            deltas = detail.get("deltas", [])
            if best is None or len(deltas) > len(best.get("deltas", [])):
                best = detail
            fa = detail.get("final_answer", "") or ""
            if len(fa) > len(longest_answer):
                longest_answer = fa
    if best and longest_answer:
        best["final_answer"] = longest_answer
    return best


# -----------------------------------------------------------------------
#  CONVERSATION RENDERER -- latest per spore, full converged answer
# -----------------------------------------------------------------------
def render_spore_card(d, is_latest=True):
    author = d.get("author", d.get("spore", "???"))
    info = resolve_spore(author)
    role = d.get("role", info.get("role", "?")).capitalize()
    model_short = d.get("model", info.get("model", "?"))
    if "/" in model_short:
        model_short = model_short.split("/")[-1]
    cycle = d.get("cycle", "?")
    conf = d.get("confidence", 0)
    conf_pct = int(conf * 100) if conf <= 1 else int(conf)
    ts = d.get("timestamp", 0)
    try:
        time_str = datetime.fromtimestamp(float(ts)).strftime("%H:%M:%S") if float(ts) > 1000000 else ""
    except Exception:
        time_str = ""
    text = d.get("content", "") or d.get("hypothesis", "") or d.get("reasoning", "")
    text = esc(text)
    claims = d.get("claims", [])
    claims_html = ""
    if claims:
        tags = "".join(f'<span style="background:#1e3a5f;color:#93c5fd;padding:1px 5px;border-radius:6px;font-size:9px;margin-right:3px">{esc(c)}</span>' for c in claims[:8])
        claims_html = f'<div style="margin-top:4px;display:flex;flex-wrap:wrap;gap:2px">{tags}</div>'
    color = info.get("color", "#6b7280")
    opacity = "1" if is_latest else "0.5"
    return f'''<div style="background:#1e293b;border-left:3px solid {color};border-radius:4px;padding:8px 12px;margin:4px 0;opacity:{opacity}">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;flex-wrap:wrap">
        <div>
          <span style="background:{color};color:#fff;padding:1px 6px;border-radius:8px;font-size:10px;font-weight:600">{esc(role)}</span>
          <span style="color:#64748b;font-size:10px;margin-left:4px">{esc(model_short)}</span>
        </div>
        <div style="font-size:10px;color:#475569">Cycle {cycle} | {conf_pct}% | {time_str}</div>
      </div>
      <div style="font-size:12px;color:#cbd5e1;line-height:1.5;white-space:pre-wrap">{text}</div>
      {claims_html}
    </div>'''


def build_conversation_stream(task_id):
    detail = fetch_task_detail(task_id)
    if not detail:
        return f'<div style="color:#64748b;padding:12px;text-align:center">Task {task_id[:16]}... -- waiting for data.</div>'
    deltas = detail.get("deltas", [])
    converged = detail.get("converged", False)
    final_answer = detail.get("final_answer", "")
    contribs = detail.get("contributors", [])
    desc = detail.get("description", "")[:120]
    max_cycle = max((d.get("cycle", 0) for d in deltas), default=0) if deltas else 0

    status_color = "#10b981" if converged else "#f59e0b"
    status_text = "CONVERGED" if converged else "REASONING"
    parts = [
        f'<div style="padding:6px 0;margin-bottom:6px;border-bottom:1px solid #334155">',
        f'  <div style="font-size:13px;color:#e2e8f0;font-weight:600">{esc(desc)}</div>',
        f'  <div style="font-size:11px;color:#64748b;margin-top:2px">',
        f'    <span style="color:{status_color};font-weight:600">{status_text}</span>',
        f'    | {len(deltas)} deltas across {max_cycle} cycles | {len(set(contribs))} contributors',
        f'  </div>',
        f'</div>',
    ]

    # Converged answer -- full text, no truncation, prominent
    if converged and final_answer:
        parts.append(f'''<div style="background:#064e3b;border:2px solid #10b981;border-radius:8px;padding:14px;margin:8px 0">
          <div style="font-size:14px;font-weight:700;color:#10b981;margin-bottom:8px">CONVERGED ANSWER</div>
          <div style="font-size:13px;color:#e2e8f0;line-height:1.7;white-space:pre-wrap">{esc(final_answer)}</div>
        </div>''')
    elif converged:
        synth_deltas = [d for d in deltas if d.get("role", "").lower() in ("synthesizer", "brain")]
        if synth_deltas:
            last = synth_deltas[-1]
            text = last.get("content", "") or last.get("hypothesis", "") or last.get("reasoning", "")
            if text:
                parts.append(f'''<div style="background:#064e3b;border:2px solid #10b981;border-radius:8px;padding:14px;margin:8px 0">
                  <div style="font-size:14px;font-weight:700;color:#10b981;margin-bottom:8px">CONVERGED ANSWER</div>
                  <div style="font-size:13px;color:#e2e8f0;line-height:1.7;white-space:pre-wrap">{esc(text)}</div>
                </div>''')

    # Latest contribution from each spore only
    latest_per_spore = {}
    for d in deltas:
        author = d.get("author", d.get("spore", ""))
        info = resolve_spore(author)
        sid = info.get("id", author)
        latest_per_spore[sid] = d

    if latest_per_spore:
        parts.append(f'<div style="font-size:11px;font-weight:600;color:#94a3b8;margin:10px 0 4px 0">Latest from each spore (cycle {max_cycle})</div>')
        for sid in sorted(latest_per_spore.keys()):
            d = latest_per_spore[sid]
            parts.append(render_spore_card(d, is_latest=True))

    return "\n".join(parts)


# -----------------------------------------------------------------------
#  SENTINEL TAB -- Proposals, consensus, changelog
# -----------------------------------------------------------------------
def fetch_sentinel_status():
    """Get sentinel state from spore-006."""
    return safe_get(f"{SENTINEL_URL}/api/sentinel/status") or {}


def build_sentinel_html():
    """Build the full Sentinel tab content."""
    status = fetch_sentinel_status()

    if not status.get("sentinel"):
        # Spore-006 not online or not sentinel role
        return f'''<div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0f172a;color:#e2e8f0;padding:16px;border-radius:10px">
          <div style="font-size:16px;font-weight:700;color:#f97316;margin-bottom:8px">Sentinel (Spore 006 -- DeepSeek R1 671B)</div>
          <div style="color:#f59e0b;padding:20px;text-align:center;background:#1e293b;border-radius:8px;border:1px solid #334155">
            Sentinel offline or starting up. Waiting for first analysis cycle...
          </div>
        </div>'''

    proposals = status.get("proposals", {})
    active = status.get("active_proposal")
    total_deploy = status.get("total_deployments", 0)
    snapshots = status.get("telemetry_snapshots", 0)
    last_analysis = status.get("last_analysis", 0)
    cooldown = status.get("deploy_cooldown_remaining", 0)

    last_str = ""
    if last_analysis > 0:
        try:
            last_str = datetime.fromtimestamp(last_analysis).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            last_str = str(last_analysis)

    # Status header
    parts = [f'''<div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0f172a;color:#e2e8f0;padding:16px;border-radius:10px">
      <div style="font-size:16px;font-weight:700;color:#f97316;margin-bottom:12px">Sentinel (Spore 006 -- DeepSeek R1 671B)</div>
      <div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:12px">
        <div style="background:#1e293b;border:1px solid #334155;border-radius:8px;padding:6px 14px;text-align:center">
          <div style="font-size:18px;font-weight:700;color:#f97316">{len(proposals)}</div>
          <div style="font-size:10px;color:#64748b">Proposals</div>
        </div>
        <div style="background:#1e293b;border:1px solid #334155;border-radius:8px;padding:6px 14px;text-align:center">
          <div style="font-size:18px;font-weight:700;color:#10b981">{total_deploy}</div>
          <div style="font-size:10px;color:#64748b">Deployments</div>
        </div>
        <div style="background:#1e293b;border:1px solid #334155;border-radius:8px;padding:6px 14px;text-align:center">
          <div style="font-size:18px;font-weight:700;color:#3b82f6">{snapshots}</div>
          <div style="font-size:10px;color:#64748b">Telemetry Snaps</div>
        </div>
        <div style="background:#1e293b;border:1px solid #334155;border-radius:8px;padding:6px 14px;text-align:center">
          <div style="font-size:14px;font-weight:700;color:#94a3b8">{last_str or "N/A"}</div>
          <div style="font-size:10px;color:#64748b">Last Analysis</div>
        </div>
      </div>''']

    # Active proposal
    if active and active in proposals:
        p = proposals[active]
        parts.append(f'''<div style="background:#1e3a5f;border:2px solid #3b82f6;border-radius:8px;padding:12px;margin-bottom:12px">
          <div style="font-size:13px;font-weight:700;color:#3b82f6;margin-bottom:6px">ACTIVE PROPOSAL -- AWAITING CONSENSUS</div>
          <div style="font-size:12px;color:#e2e8f0;line-height:1.5;white-space:pre-wrap">{esc(p.get("description", ""))}</div>
          <div style="font-size:10px;color:#64748b;margin-top:4px">Type: {esc(p.get("change_type", "?"))} | Status: {esc(p.get("status", "?"))}</div>
        </div>''')
    elif not proposals:
        parts.append(f'''<div style="background:#1e293b;border:1px solid #334155;border-radius:8px;padding:16px;margin-bottom:12px;text-align:center">
          <div style="font-size:12px;color:#64748b">No proposals yet. Sentinel is collecting telemetry and analyzing swarm performance...</div>
          <div style="font-size:10px;color:#475569;margin-top:4px">First proposal after 3 telemetry snapshots (approx 10 min)</div>
        </div>''')

    # Cooldown indicator
    if cooldown > 0:
        parts.append(f'''<div style="background:#1e293b;border-radius:6px;padding:6px 12px;margin-bottom:12px">
          <div style="font-size:11px;color:#f59e0b">Deploy cooldown: {int(cooldown)}s remaining (safety interval between deployments)</div>
        </div>''')

    # Proposal changelog
    if proposals:
        parts.append('<div style="font-size:13px;font-weight:600;color:#e2e8f0;margin:8px 0 6px 0">Proposal History</div>')
        STATUS_COLORS = {
            "proposed": "#3b82f6",
            "awaiting_consensus": "#f59e0b",
            "consensus_reached": "#10b981",
            "testing": "#8b5cf6",
            "deployed": "#10b981",
            "rejected": "#ef4444",
            "failed": "#ef4444",
            "consensus_timeout": "#6b7280",
        }
        # Reverse order -- newest first
        for pid in reversed(list(proposals.keys())):
            p = proposals[pid]
            st = p.get("status", "unknown")
            st_color = STATUS_COLORS.get(st, "#6b7280")
            ts_val = p.get("submitted_at", 0)
            ts_str = ""
            if ts_val > 0:
                try:
                    ts_str = datetime.fromtimestamp(ts_val).strftime("%H:%M:%S")
                except Exception:
                    ts_str = ""
            desc = p.get("description", "")
            change_type = p.get("change_type", "")
            parts.append(f'''<div style="background:#1e293b;border-left:3px solid {st_color};border-radius:4px;padding:8px 12px;margin:4px 0">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
                <span style="background:{st_color};color:#fff;padding:1px 6px;border-radius:8px;font-size:10px;font-weight:600">{esc(st).upper()}</span>
                <span style="color:#475569;font-size:10px">{esc(change_type)} | {ts_str}</span>
              </div>
              <div style="font-size:12px;color:#cbd5e1;line-height:1.5;white-space:pre-wrap">{esc(desc)}</div>
            </div>''')

    parts.append('</div>')
    return "\n".join(parts)


# -----------------------------------------------------------------------
#  DASHBOARD TAB
# -----------------------------------------------------------------------
def refresh_dashboard():
    stats = build_stats_html()
    trust = build_trust_html()
    dash = f'''<div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0f172a;color:#e2e8f0;padding:12px;border-radius:10px">
      <div style="font-size:16px;font-weight:700;margin-bottom:8px">Synapse Swarm</div>
      {stats}
      {trust}
    </div>'''
    with _state["lock"]:
        tid = _state["current_task"]
    if not tid:
        all_tasks = fetch_all_tasks()
        if all_tasks:
            best_tid = max(all_tasks.keys(), key=lambda k: all_tasks[k].get("delta_count", 0))
            tid = best_tid
    if tid:
        conv = build_conversation_stream(tid)
    else:
        conv = '<div style="color:#64748b;padding:20px;text-align:center">No conversations yet. Submit a prompt above.</div>'
    return dash, conv


def refresh_conversation_only():
    with _state["lock"]:
        tid = _state["current_task"]
    if not tid:
        all_tasks = fetch_all_tasks()
        if all_tasks:
            tid = max(all_tasks.keys(), key=lambda k: all_tasks[k].get("delta_count", 0))
    if tid:
        return build_conversation_stream(tid)
    return '<div style="color:#64748b;padding:20px;text-align:center">No active conversation.</div>'


def submit_prompt(prompt):
    if not prompt or not prompt.strip():
        return "Enter a prompt.", ""
    task_data = {"task": prompt.strip()}
    submitted = 0; task_id = None
    for s in SPORES:
        r = safe_post(f"{s['url']}/api/task", task_data, timeout=15)
        if r:
            submitted += 1
            if not task_id:
                task_id = r.get("task_id", r.get("id", ""))
    if submitted == 0:
        return '<div style="color:#ef4444;font-size:12px">Failed to submit to any spore.</div>', ""
    with _state["lock"]:
        _state["current_task"] = task_id
    msg = f'<div style="color:#10b981;font-size:12px">Submitted to {submitted}/{N_SPORES} spores. Task: {task_id[:16] if task_id else "?"}... Auto-refreshing below.</div>'
    conv = build_conversation_stream(task_id) if task_id else ""
    return msg, conv


def new_conversation():
    with _state["lock"]:
        _state["current_task"] = None
    return (
        '<div style="color:#94a3b8;font-size:12px">Previous conversation archived. Enter a new prompt.</div>',
        '<div style="color:#64748b;padding:20px;text-align:center">Submit a prompt above to start a new conversation.</div>',
        ""
    )


# -----------------------------------------------------------------------
#  LIBRARY TAB
# -----------------------------------------------------------------------
def refresh_library():
    all_tasks = fetch_all_tasks()
    if not all_tasks:
        return '<div style="color:#64748b;padding:20px">No conversations yet.</div>', gr.update(choices=[], value=None)
    sorted_tasks = sorted(all_tasks.values(), key=lambda t: t.get("delta_count", 0), reverse=True)
    choices = []
    for t in sorted_tasks:
        tid = t.get("task_id", "")
        desc = t.get("description", "?")[:60]
        dc = t.get("delta_count", 0)
        conv = "CONVERGED" if t.get("converged") else "ACTIVE"
        choices.append((f"[{conv}] {desc} ({dc} deltas)", tid))
    summary = f'<div style="color:#94a3b8;font-size:12px">{len(choices)} conversations ({sum(1 for t in sorted_tasks if t.get("converged"))} converged)</div>'
    return summary, gr.update(choices=choices, value=None)


def view_library_task(task_id):
    if not task_id:
        return '<div style="color:#64748b;padding:20px">Select a conversation from the dropdown.</div>'
    return build_conversation_stream(task_id)


# -----------------------------------------------------------------------
#  SENTINEL TAB REFRESH
# -----------------------------------------------------------------------
def refresh_sentinel():
    return build_sentinel_html()


# -----------------------------------------------------------------------
#  GRADIO APP -- 3 tabs: Dashboard, Sentinel, Library
# -----------------------------------------------------------------------
css = """
.gradio-container { background: #0f172a !important; }
footer { display: none !important; }
.dark { background: #0f172a !important; }
"""

with gr.Blocks(css=css, theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate"), title="Synapse Command Center") as demo:

    # -- DASHBOARD TAB --
    with gr.Tab("Dashboard"):
        dashboard_html = gr.HTML()
        refresh_dash_btn = gr.Button("Refresh Dashboard", size="sm")

        gr.HTML('<div style="border-top:1px solid #334155;margin:8px 0"></div>')

        with gr.Row():
            prompt_box = gr.Textbox(placeholder="Enter a prompt for the swarm...", label="Submit to Swarm", scale=4, lines=2)
            with gr.Column(scale=1, min_width=120):
                submit_btn = gr.Button("Send", variant="primary")
                new_conv_btn = gr.Button("New Topic", variant="secondary", size="sm")
        status_msg = gr.HTML()

        gr.HTML('<div style="border-top:1px solid #334155;margin:8px 0"></div>')
        gr.HTML('<div style="font-size:13px;font-weight:600;color:#e2e8f0;padding:2px 0">Live Conversation (auto-refreshes every 15s)</div>')
        conversation_html = gr.HTML()

        conv_timer = gr.Timer(value=15)
        conv_timer.tick(fn=refresh_conversation_only, outputs=[conversation_html])

        refresh_dash_btn.click(fn=refresh_dashboard, outputs=[dashboard_html, conversation_html])
        submit_btn.click(fn=submit_prompt, inputs=[prompt_box], outputs=[status_msg, conversation_html])
        prompt_box.submit(fn=submit_prompt, inputs=[prompt_box], outputs=[status_msg, conversation_html])
        new_conv_btn.click(fn=new_conversation, outputs=[status_msg, conversation_html, prompt_box])
        demo.load(fn=refresh_dashboard, outputs=[dashboard_html, conversation_html])

    # -- SENTINEL TAB --
    with gr.Tab("Sentinel"):
        sentinel_html = gr.HTML()
        refresh_sentinel_btn = gr.Button("Refresh Sentinel", size="sm")
        sentinel_timer = gr.Timer(value=30)
        sentinel_timer.tick(fn=refresh_sentinel, outputs=[sentinel_html])
        refresh_sentinel_btn.click(fn=refresh_sentinel, outputs=[sentinel_html])
        demo.load(fn=refresh_sentinel, outputs=[sentinel_html])

    # -- LIBRARY TAB --
    with gr.Tab("Library"):
        with gr.Row():
            lib_status = gr.HTML()
            refresh_lib_btn = gr.Button("Refresh", size="sm", scale=0)
        task_dropdown = gr.Dropdown(label="Select Conversation", interactive=True)
        archived_html = gr.HTML()

        refresh_lib_btn.click(fn=refresh_library, outputs=[lib_status, task_dropdown])
        task_dropdown.change(fn=view_library_task, inputs=[task_dropdown], outputs=[archived_html])
        demo.load(fn=refresh_library, outputs=[lib_status, task_dropdown])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)
