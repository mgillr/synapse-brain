"""Synapse Command Center -- Mass-scale swarm dashboard.

Designed for 7 to 70,000 nodes. Compact topology view, fractal sentinel
mesh monitoring, searchable node browser, aggregate trust heatmap.
"""
import os, json, time, httpx, gradio as gr, threading, html as html_mod
from datetime import datetime
from collections import defaultdict

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

# Bootstrap URL for cross-cluster discovery (injected by launch_swarm.py).
BOOTSTRAP_URL = os.environ.get(
    "BOOTSTRAP_URL",
    "https://raw.githubusercontent.com/mgillr/synapse-brain/main/bootstrap.json",
)

# Default role/model template per spore index (used to fill out URL-only env input).
_SPORE_TEMPLATE = [
    ("Explorer",     "Qwen3 235B MoE",      "#3b82f6"),
    ("Synthesizer",  "Llama 3.3 70B",       "#8b5cf6"),
    ("Adversarial",  "DeepSeek R1 32B",     "#ef4444"),
    ("Validator",    "Gemma 3 27B",         "#f59e0b"),
    ("Generalist",   "Llama 4 Scout 17B",   "#10b981"),
    ("Brain",        "GLM-4.7-Flash",       "#ec4899"),
    ("Sentinel",     "DeepSeek R1 671B",    "#f97316"),
]


def _spore_dict_from_url(url, idx=0):
    role, model, color = _SPORE_TEMPLATE[idx % len(_SPORE_TEMPLATE)]
    sid = f"{idx:03d}"
    try:
        tail = url.rsplit("-", 1)[-1].split(".", 1)[0]
        if tail.isdigit():
            sid = tail
    except Exception:
        pass
    return {
        "id": sid,
        "url": url.rstrip("/"),
        "role": role,
        "model": model,
        "color": color,
        "commander": "default",
    }


def _urls_from_bootstrap(timeout=6.0):
    """Fetch bootstrap.json and return its seeds list. Empty list on any failure."""
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as c:
            r = c.get(BOOTSTRAP_URL)
            if r.status_code != 200:
                return []
            data = r.json()
            seeds = data.get("seeds", []) if isinstance(data, dict) else []
            return [str(s).strip().rstrip("/") for s in seeds if s]
    except Exception:
        return []


def _urls_from_env():
    urls = []
    raw = os.environ.get("SYNAPSE_SPORES", "").strip()
    if raw:
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                urls = [str(u).strip().rstrip("/") for u in data if u]
        except Exception:
            pass
    if not urls:
        raw = os.environ.get("SPORE_URLS", "").strip()
        if raw:
            for line in raw.replace(",", "\n").splitlines():
                line = line.strip().rstrip("/")
                if line:
                    urls.append(line)
    if not urls:
        owner = os.environ.get("HF_OWNER", "").strip().lower()
        prefix = os.environ.get("SPORE_PREFIX", "synapse-spore").strip()
        count = int(os.environ.get("SPORE_COUNT", "7") or "7")
        if owner:
            urls = [f"https://{owner}-{prefix}-{i:03d}.hf.space" for i in range(count)]
    return urls


def _build_core_spores():
    """Resolve seed URLs from bootstrap.json + env vars, deduplicated.

    Order: bootstrap.json (canonical, lists every known spore in the network)
    then env-derived URLs as a supplement. Placeholder fallback only if both
    sources return nothing -- visible misconfiguration warning in the UI.
    """
    urls = []
    seen = set()
    for u in _urls_from_bootstrap() + _urls_from_env():
        if u and u not in seen:
            seen.add(u)
            urls.append(u)
    if not urls:
        urls = [f"https://your-prefix-synapse-spore-{i:03d}.hf.space" for i in range(7)]
    spores = [_spore_dict_from_url(u, i) for i, u in enumerate(urls)]
    print(f"[CC] CORE_SPORES resolved: {len(spores)} seed URLs", flush=True)
    for s in spores:
        print(f"  - {s['id']}: {s['url']}", flush=True)
    return spores


# Core spores -- bootstrap. Federation nodes discovered dynamically.
CORE_SPORES = _build_core_spores()

# Dynamic registry -- populated at runtime from gossip + federation
_registry = {
    "nodes": {s["id"]: s for s in CORE_SPORES},
    "commanders": {"default": {"name": "Default", "color": "#f97316", "sentinel": "006"}},
    "lock": threading.Lock(),
    "last_discovery": 0,
    "current_task": None,
}

ROLE_COLORS = {
    "explorer": "#3b82f6", "synthesizer": "#8b5cf6", "adversarial": "#ef4444",
    "validator": "#f59e0b", "generalist": "#10b981", "brain": "#ec4899",
    "sentinel": "#f97316", "contributor": "#6b7280", "specialist": "#06b6d4",
}

PHASE_COLORS = {"DIVERGE": "#3b82f6", "DEEPEN": "#f59e0b", "CONVERGE": "#10b981", "SYNTHESIZE": "#8b5cf6"}

# Per-thread HTTP clients -- no shared pool.
# A single shared client + _reset_client() inside ThreadPoolExecutor workers is
# unsafe: one thread's transport error nukes the client mid-flight for all other
# threads, making healthy spores appear offline. threading.local() gives each
# worker its own client so resets are isolated to the faulting thread.
_tls = threading.local()


def _get_client():
    """Return a per-thread httpx.Client (keepalive disabled)."""
    client = getattr(_tls, "client", None)
    if client is None or client.is_closed:
        _tls.client = httpx.Client(
            timeout=httpx.Timeout(8.0, connect=4.0),
            limits=httpx.Limits(max_connections=1, max_keepalive_connections=0),
            follow_redirects=True,
        )
    return _tls.client


def _reset_client():
    """Close this thread's client; next call to _get_client() opens a fresh one."""
    client = getattr(_tls, "client", None)
    if client is not None:
        try:
            client.close()
        except Exception:
            pass
        _tls.client = None


def esc(text):
    return html_mod.escape(str(text)) if text else ""


def safe_get(url, timeout=8):
    """GET with one retry on protocol/network errors (stale keepalive defence-in-depth)."""
    last_err = None
    for attempt in range(2):
        try:
            client = _get_client()
            r = client.get(url, headers=HEADERS, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            return None  # non-2xx is a real "no data", not retryable
        except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ConnectError,
                httpx.PoolTimeout, httpx.ReadTimeout) as e:
            last_err = e
            _reset_client()  # nuke pool on transport error
            continue
        except Exception:
            return None
    return None


def safe_post(url, data, timeout=15):
    last_err = None
    for attempt in range(2):
        try:
            client = _get_client()
            r = client.post(url, json=data, headers=HEADERS, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            return None
        except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ConnectError,
                httpx.PoolTimeout, httpx.ReadTimeout) as e:
            last_err = e
            _reset_client()
            continue
        except Exception:
            return None
    return None


# -------------------------------------------------------------------
#  NODE DISCOVERY -- PARALLEL thread-based polling (Gradio-safe)
# -------------------------------------------------------------------
def _poll_node(nid, node):
    """Poll a single node for health + federation. Thread-safe."""
    node = dict(node)
    # 20s accommodates HF Space cold-start wake-up (~10-15s) on idle spores.
    health = safe_get(f"{node['url']}/api/health", timeout=20.0)
    if not health:
        node["online"] = False
        return node, {}
    node["online"] = True
    node["memories"] = health.get("memories", health.get("memory_size", 0))
    node["cycles"] = health.get("cycles", health.get("reasoning_cycles", 0))
    node["deltas_produced"] = health.get("deltas_produced", 0)
    node["deltas_received"] = health.get("deltas_received", 0)
    node["confidence"] = health.get("confidence", 0)
    node["tier"] = health.get("last_tier", health.get("tier", "worker"))
    node["peers"] = health.get("peers", health.get("connected_peers", []))
    n_peers = len(node["peers"]) if isinstance(node["peers"], list) else (int(node["peers"]) if node["peers"] else 0)
    node["n_peers"] = n_peers

    fed = safe_get(f"{node['url']}/api/federation/peers", timeout=5.0)  # served by spore.py
    new_peers = {}
    if fed and isinstance(fed, dict):
        for peer_id, peer_info in fed.items():
            new_peers[peer_id] = {
                "id": peer_id,
                "url": peer_info.get("url", ""),
                "role": peer_info.get("role", "contributor"),
                "model": peer_info.get("model", "unknown"),
                "color": ROLE_COLORS.get(peer_info.get("role", "").lower(), "#6b7280"),
                "commander": peer_info.get("commander", "unknown"),
            }
    return node, new_peers


# Guards against concurrent discoveries launched by multiple Gradio sessions.
# Without this, two simultaneous calls each take a snapshot of _registry["nodes"],
# poll in parallel, then the second writeback clobbers the first with stale data,
# flipping groups of spores back to offline.
_discovery_in_flight = False


def discover_nodes():
    """Poll all known nodes in PARALLEL using threads. Gradio-safe."""
    global _discovery_in_flight
    now = time.time()
    with _registry["lock"]:
        if now - _registry["last_discovery"] < 15:
            return
        if _discovery_in_flight:
            return  # another session is already mid-discovery; skip, don't clobber
        _discovery_in_flight = True
        _registry["last_discovery"] = now

    try:
        # Snapshot just the node IDs + current state to build the work queue.
        with _registry["lock"]:
            work_items = list(_registry["nodes"].items())

        import concurrent.futures
        updated_nodes = {}
        updated_commanders = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
            futures = {pool.submit(_poll_node, nid, node): nid for nid, node in work_items}
            try:
                # 60s window absorbs 20s cold-start health + 5s fed + one retry per poll.
                for future in concurrent.futures.as_completed(futures, timeout=60):
                    try:
                        node, new_peers = future.result()
                        updated_nodes[node["id"]] = node
                        for peer_id, peer_info in new_peers.items():
                            updated_nodes.setdefault(peer_id, peer_info)
                            cmd = peer_info.get("commander", "unknown")
                            if cmd not in updated_commanders:
                                updated_commanders[cmd] = {"name": cmd, "color": "#6b7280", "sentinel": None}
                    except Exception:
                        pass
            except concurrent.futures.TimeoutError:
                pass  # merge whatever completed; timed-out nodes keep previous state

        # Merge-only writeback: only update nodes that were polled this cycle.
        # Nodes whose futures didn't complete keep their existing state in the registry
        # rather than being silently overwritten with a stale snapshot entry.
        with _registry["lock"]:
            for nid, node in updated_nodes.items():
                _registry["nodes"][nid] = node
            for cmd, info in updated_commanders.items():
                if cmd not in _registry["commanders"]:
                    _registry["commanders"][cmd] = info
    finally:
        with _registry["lock"]:
            _discovery_in_flight = False


def get_all_nodes():
    with _registry["lock"]:
        return dict(_registry["nodes"])


def get_commanders():
    with _registry["lock"]:
        return dict(_registry["commanders"])


# -------------------------------------------------------------------
#  AGGREGATE STATS BAR -- compact, fixed height
# -------------------------------------------------------------------
def build_stats_bar():
    nodes = get_all_nodes()
    online = sum(1 for n in nodes.values() if n.get("online"))
    total = len(nodes)
    commanders = get_commanders()
    total_mem = sum(n.get("memories", 0) for n in nodes.values())
    total_deltas = sum(n.get("deltas_produced", 0) + n.get("deltas_received", 0) for n in nodes.values())
    total_cycles = sum(n.get("cycles", 0) for n in nodes.values())
    sentinels = sum(1 for n in nodes.values() if n.get("role", "").lower() == "sentinel" and n.get("online"))

    return f'''<div style="display:flex;gap:10px;flex-wrap:wrap;padding:4px 0">
      <div style="background:#1e293b;border:1px solid #334155;border-radius:6px;padding:4px 12px;text-align:center">
        <span style="font-size:18px;font-weight:700;color:#10b981">{online}</span>
        <span style="font-size:10px;color:#64748b">/{total} nodes</span>
      </div>
      <div style="background:#1e293b;border:1px solid #334155;border-radius:6px;padding:4px 12px;text-align:center">
        <span style="font-size:18px;font-weight:700;color:#f97316">{len(commanders)}</span>
        <span style="font-size:10px;color:#64748b">commanders</span>
      </div>
      <div style="background:#1e293b;border:1px solid #334155;border-radius:6px;padding:4px 12px;text-align:center">
        <span style="font-size:18px;font-weight:700;color:#ec4899">{sentinels}</span>
        <span style="font-size:10px;color:#64748b">sentinels</span>
      </div>
      <div style="background:#1e293b;border:1px solid #334155;border-radius:6px;padding:4px 12px;text-align:center">
        <span style="font-size:18px;font-weight:700;color:#3b82f6">{total_mem:,}</span>
        <span style="font-size:10px;color:#64748b">memories</span>
      </div>
      <div style="background:#1e293b;border:1px solid #334155;border-radius:6px;padding:4px 12px;text-align:center">
        <span style="font-size:18px;font-weight:700;color:#8b5cf6">{total_deltas:,}</span>
        <span style="font-size:10px;color:#64748b">deltas</span>
      </div>
      <div style="background:#1e293b;border:1px solid #334155;border-radius:6px;padding:4px 12px;text-align:center">
        <span style="font-size:18px;font-weight:700;color:#f59e0b">{total_cycles:,}</span>
        <span style="font-size:10px;color:#64748b">cycles</span>
      </div>
    </div>'''


# -------------------------------------------------------------------
#  TOPOLOGY VIEW -- compact node table grouped by commander
# -------------------------------------------------------------------
def build_topology():
    nodes = get_all_nodes()
    commanders = get_commanders()

    # Group nodes by commander
    by_commander = defaultdict(list)
    for n in nodes.values():
        by_commander[n.get("commander", "unknown")].append(n)

    parts = []
    for cmd_id in sorted(by_commander.keys()):
        cmd_nodes = by_commander[cmd_id]
        cmd_info = commanders.get(cmd_id, {"name": cmd_id, "color": "#6b7280"})
        online_count = sum(1 for n in cmd_nodes if n.get("online"))
        sentinel_node = next((n for n in cmd_nodes if n.get("role", "").lower() == "sentinel"), None)
        sentinel_status = "ACTIVE" if sentinel_node and sentinel_node.get("online") else "NONE"
        sentinel_color = "#f97316" if sentinel_status == "ACTIVE" else "#6b7280"

        # Commander header row
        parts.append(f'''<div style="margin:6px 0 2px 0">
          <div style="display:flex;align-items:center;gap:8px;padding:3px 8px;background:#1e293b;border-radius:6px;border:1px solid #334155">
            <span style="width:8px;height:8px;border-radius:50%;background:{cmd_info['color']};display:inline-block"></span>
            <span style="font-size:12px;font-weight:700;color:#e2e8f0">{esc(cmd_info.get('name', cmd_id))}</span>
            <span style="font-size:10px;color:#64748b">{online_count}/{len(cmd_nodes)} nodes</span>
            <span style="font-size:9px;padding:1px 5px;border-radius:8px;background:{sentinel_color};color:#fff">Sentinel: {sentinel_status}</span>
          </div>
        </div>''')

        # Compact node rows -- one line per node, ~20px height
        rows = []
        for n in sorted(cmd_nodes, key=lambda x: x.get("id", "")):
            is_online = n.get("online")
            status_dot = '<span style="color:#10b981;font-size:8px">&#9679;</span>' if is_online else '<span style="color:#ef4444;font-size:8px">&#9679;</span>'
            role = n.get("role", "?")
            role_color = ROLE_COLORS.get(role.lower(), "#6b7280")
            mem = n.get("memories", 0)
            cyc = n.get("cycles", 0)
            dp = n.get("deltas_produced", 0)
            dr = n.get("deltas_received", 0)
            conf = n.get("confidence", 0)
            conf_pct = int(conf * 100) if conf <= 1 else int(conf)
            tier = n.get("tier", "worker")
            tier_bg = "#ec4899" if "brain" in str(tier).lower() else "#334155"
            if role.lower() == "sentinel":
                tier_bg = "#f97316"
            model_short = n.get("model", "?")
            if "/" in model_short:
                model_short = model_short.split("/")[-1]
            if len(model_short) > 18:
                model_short = model_short[:16] + ".."

            rows.append(f'''<tr style="border-bottom:1px solid #1e293b;height:22px">
              <td style="padding:1px 4px;font-size:10px;white-space:nowrap">{status_dot} <span style="color:#94a3b8">{esc(n['id'][:8])}</span></td>
              <td style="padding:1px 4px"><span style="background:{role_color};color:#fff;padding:0px 4px;border-radius:4px;font-size:9px">{esc(role)}</span></td>
              <td style="padding:1px 4px;font-size:10px;color:#64748b;white-space:nowrap">{esc(model_short)}</td>
              <td style="padding:1px 4px"><span style="background:{tier_bg};color:#fff;padding:0px 3px;border-radius:3px;font-size:8px">{esc(str(tier).upper()[:6])}</span></td>
              <td style="padding:1px 4px;font-size:10px;color:#64748b;text-align:right">{mem:,}</td>
              <td style="padding:1px 4px;font-size:10px;color:#64748b;text-align:right">{cyc}</td>
              <td style="padding:1px 4px;font-size:10px;color:#64748b;text-align:right">{dp+dr}</td>
              <td style="padding:1px 4px;width:40px">
                <div style="background:#374151;border-radius:2px;height:4px;width:100%">
                  <div style="background:{role_color};height:4px;border-radius:2px;width:{conf_pct}%"></div>
                </div>
              </td>
            </tr>''')

        if rows:
            parts.append(f'''<div style="max-height:200px;overflow-y:auto;margin:0 0 4px 12px">
              <table style="width:100%;border-collapse:collapse;background:#0f172a">
                <tr style="border-bottom:1px solid #334155">
                  <th style="padding:1px 4px;font-size:9px;color:#475569;text-align:left">Node</th>
                  <th style="padding:1px 4px;font-size:9px;color:#475569;text-align:left">Role</th>
                  <th style="padding:1px 4px;font-size:9px;color:#475569;text-align:left">Model</th>
                  <th style="padding:1px 4px;font-size:9px;color:#475569;text-align:left">Tier</th>
                  <th style="padding:1px 4px;font-size:9px;color:#475569;text-align:right">Mem</th>
                  <th style="padding:1px 4px;font-size:9px;color:#475569;text-align:right">Cyc</th>
                  <th style="padding:1px 4px;font-size:9px;color:#475569;text-align:right">Del</th>
                  <th style="padding:1px 4px;font-size:9px;color:#475569;text-align:left">Conf</th>
                </tr>
                {"".join(rows)}
              </table>
            </div>''')

    return "".join(parts)


# -------------------------------------------------------------------
#  TRUST SUMMARY -- aggregate by commander, not NxN
# -------------------------------------------------------------------
def build_trust_summary():
    nodes = get_all_nodes()
    commanders = get_commanders()

    # For small swarms (<20 nodes), show full matrix
    if len(nodes) <= 20:
        return _build_trust_matrix_small(nodes)

    # For large swarms, show commander-level aggregate
    by_commander = defaultdict(list)
    for n in nodes.values():
        by_commander[n.get("commander", "unknown")].append(n)

    rows = []
    for cmd_id in sorted(by_commander.keys()):
        cmd_nodes = by_commander[cmd_id]
        cmd_info = commanders.get(cmd_id, {"name": cmd_id, "color": "#6b7280"})
        online = [n for n in cmd_nodes if n.get("online")]
        if not online:
            continue

        # Sample trust from first online node
        sample = safe_get(f"{online[0]['url']}/api/trust")
        if not sample:
            continue

        trust_vals = []
        for key, val in sample.items():
            if isinstance(val, dict):
                v = val.get("overall", val.get("trust", 0))
            else:
                try:
                    v = float(val)
                except (TypeError, ValueError):
                    continue
            trust_vals.append(v)

        avg_trust = sum(trust_vals) / len(trust_vals) if trust_vals else 0
        min_trust = min(trust_vals) if trust_vals else 0
        max_trust = max(trust_vals) if trust_vals else 0

        bg = f"rgba(16,185,129,{avg_trust})" if avg_trust > 0.5 else f"rgba(239,68,68,{1-avg_trust})"
        rows.append(f'''<tr>
          <td style="padding:2px 6px;font-size:10px;color:{cmd_info['color']};font-weight:600">{esc(cmd_info.get('name', cmd_id))}</td>
          <td style="padding:2px 6px;font-size:10px;color:#94a3b8;text-align:right">{len(online)}/{len(cmd_nodes)}</td>
          <td style="padding:2px 6px;background:{bg};text-align:center;font-size:10px;color:#fff">{avg_trust:.2f}</td>
          <td style="padding:2px 6px;font-size:10px;color:#64748b;text-align:center">{min_trust:.2f} - {max_trust:.2f}</td>
        </tr>''')

    if not rows:
        return ""

    return f'''<div style="margin-top:6px">
      <div style="font-size:11px;font-weight:600;color:#94a3b8;margin-bottom:3px">Trust by Commander</div>
      <table style="width:100%;border-collapse:collapse;background:#0f172a;border-radius:4px">
        <tr style="border-bottom:1px solid #334155">
          <th style="padding:2px 6px;font-size:9px;color:#475569;text-align:left">Commander</th>
          <th style="padding:2px 6px;font-size:9px;color:#475569;text-align:right">Nodes</th>
          <th style="padding:2px 6px;font-size:9px;color:#475569;text-align:center">Avg Trust</th>
          <th style="padding:2px 6px;font-size:9px;color:#475569;text-align:center">Range</th>
        </tr>
        {"".join(rows)}
      </table>
    </div>'''


def _build_trust_matrix_small(nodes):
    """Full NxN trust matrix for small swarms (<=20 nodes)."""
    node_list = sorted(nodes.values(), key=lambda n: n.get("id", ""))
    rows = []
    for n in node_list:
        if not n.get("online"):
            continue
        t = safe_get(f"{n['url']}/api/trust")
        if not t:
            continue
        cells = []
        for n2 in node_list:
            v = _resolve_trust_val(t, n2["id"])
            if v is not None:
                bg = f"rgba(16,185,129,{v})" if v > 0.5 else f"rgba(239,68,68,{1-v})"
                cells.append(f'<td style="background:{bg};text-align:center;padding:2px;font-size:9px;color:#fff">{v:.1f}</td>')
            else:
                cells.append('<td style="text-align:center;padding:2px;font-size:9px;color:#334155">--</td>')
        color = n.get("color", "#6b7280")
        rows.append(f'<tr><td style="padding:2px 3px;font-size:9px;color:{color};font-weight:600">{n["id"][:6]}</td>{"".join(cells)}</tr>')

    if not rows:
        return ""
    hdrs = "".join(f'<th style="padding:2px;font-size:8px;color:#475569">{n["id"][:4]}</th>' for n in node_list)
    return f'''<div style="margin-top:6px">
      <div style="font-size:11px;font-weight:600;color:#94a3b8;margin-bottom:3px">Trust Matrix</div>
      <table style="border-collapse:collapse;width:100%;background:#0f172a;border-radius:4px"><tr><th></th>{hdrs}</tr>{"".join(rows)}</table>
    </div>'''


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


# -------------------------------------------------------------------
#  TASK FETCHING -- parallel thread-based
# -------------------------------------------------------------------
def _fetch_node_tasks(node):
    """Fetch tasks from a single node."""
    if not node.get("online", True):
        return {}
    raw = safe_get(f"{node['url']}/api/tasks", timeout=10.0)
    if not raw:
        return {}
    result = {}
    if isinstance(raw, dict):
        for tid, tdata in raw.items():
            if tid not in result or (tdata.get("delta_count", 0) > result.get(tid, {}).get("delta_count", 0)):
                tdata["task_id"] = tid
                result[tid] = tdata
    elif isinstance(raw, list):
        for t in raw:
            tid = t.get("task_id", t.get("id", ""))
            if tid and (tid not in result or t.get("delta_count", 0) > result.get(tid, {}).get("delta_count", 0)):
                result[tid] = t
    return result


def fetch_all_tasks():
    """Fetch tasks from all online nodes in PARALLEL."""
    all_nodes = list(get_all_nodes().values())[:20]
    import concurrent.futures
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(_fetch_node_tasks, n) for n in all_nodes]
        try:
            for f in concurrent.futures.as_completed(futures, timeout=30):
                try:
                    results.append(f.result())
                except Exception:
                    pass
        except concurrent.futures.TimeoutError:
            pass

    merged = {}
    for r in results:
        if not isinstance(r, dict):
            continue
        for tid, tdata in r.items():
            if tid not in merged or (tdata.get("delta_count", 0) > merged[tid].get("delta_count", 0)):
                merged[tid] = tdata
    return merged


def _fetch_node_detail(node, task_id):
    if not node.get("online", True):
        return None
    return safe_get(f"{node['url']}/api/task/{task_id}", timeout=10.0)


def fetch_task_detail(task_id):
    """Fetch task detail from all nodes in PARALLEL."""
    all_nodes = list(get_all_nodes().values())[:20]
    import concurrent.futures
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(_fetch_node_detail, n, task_id) for n in all_nodes]
        try:
            for f in concurrent.futures.as_completed(futures, timeout=30):
                try:
                    results.append(f.result())
                except Exception:
                    pass
        except concurrent.futures.TimeoutError:
            pass

    best = None
    longest_answer = ""
    for detail in results:
        if not detail:
            continue
        deltas = detail.get("deltas", [])
        if best is None or len(deltas) > len(best.get("deltas", [])):
            best = detail
        fa = detail.get("final_answer", "") or ""
        if len(fa) > len(longest_answer):
            longest_answer = fa
    if best and longest_answer:
        best["final_answer"] = longest_answer
    return best


# -------------------------------------------------------------------
#  CONVERSATION -- compact, converged answer prominent
# -------------------------------------------------------------------
def build_conversation_stream(task_id):
    detail = fetch_task_detail(task_id)
    if not detail:
        return f'<div style="color:#64748b;padding:8px;text-align:center;font-size:11px">Task {task_id[:16]}... -- waiting for data.</div>'

    deltas = detail.get("deltas", [])
    converged = detail.get("converged", False)
    final_answer = detail.get("final_answer", "")
    contribs = detail.get("contributors", [])
    desc = detail.get("description", "")[:120]
    max_cycle = max((d.get("cycle", 0) for d in deltas), default=0) if deltas else 0

    status_color = "#10b981" if converged else "#f59e0b"
    status_text = "CONVERGED" if converged else "REASONING"

    parts = [f'''<div style="padding:4px 0;margin-bottom:4px;border-bottom:1px solid #1e293b">
      <div style="font-size:12px;color:#e2e8f0;font-weight:600">{esc(desc)}</div>
      <div style="font-size:10px;color:#64748b">
        <span style="color:{status_color};font-weight:600">{status_text}</span>
        | {len(deltas)} deltas / {max_cycle} cycles / {len(set(contribs))} contributors
      </div>
    </div>''']

    # Converged answer -- always prominent
    if converged and final_answer:
        parts.append(f'''<div style="background:#064e3b;border:1px solid #10b981;border-radius:6px;padding:10px;margin:4px 0">
          <div style="font-size:12px;font-weight:700;color:#10b981;margin-bottom:6px">CONVERGED ANSWER</div>
          <div style="font-size:12px;color:#e2e8f0;line-height:1.6;white-space:pre-wrap">{esc(final_answer)}</div>
        </div>''')
    elif converged:
        synth = [d for d in deltas if d.get("role", "").lower() in ("synthesizer", "brain")]
        if synth:
            text = synth[-1].get("content", "") or synth[-1].get("hypothesis", "") or synth[-1].get("reasoning", "")
            if text:
                parts.append(f'''<div style="background:#064e3b;border:1px solid #10b981;border-radius:6px;padding:10px;margin:4px 0">
                  <div style="font-size:12px;font-weight:700;color:#10b981;margin-bottom:6px">CONVERGED ANSWER</div>
                  <div style="font-size:12px;color:#e2e8f0;line-height:1.6;white-space:pre-wrap">{esc(text)}</div>
                </div>''')

    # Compact contribution summary -- one line per spore, not full cards
    latest = {}
    for d in deltas:
        author = d.get("author", d.get("spore", ""))
        sid = author
        for s in CORE_SPORES:
            if f"spore-{s['id']}" in str(author) or s["id"] == author:
                sid = s["id"]
                break
        latest[sid] = d

    if latest:
        parts.append('<div style="font-size:10px;font-weight:600;color:#475569;margin:6px 0 2px 0">Latest contributions</div>')
        for sid in sorted(latest.keys()):
            d = latest[sid]
            role = d.get("role", "?")
            role_color = ROLE_COLORS.get(role.lower(), "#6b7280")
            conf = d.get("confidence", 0)
            conf_pct = int(conf * 100) if conf <= 1 else int(conf)
            cycle = d.get("cycle", "?")
            text = d.get("content", "") or d.get("hypothesis", "") or d.get("reasoning", "")
            # Truncate display only (full text in CRDT memory)
            display_text = text[:300] + "..." if len(text) > 300 else text
            claims = d.get("claims", [])
            claims_html = ""
            if claims:
                tags = "".join(f'<span style="background:#1e3a5f;color:#93c5fd;padding:0px 3px;border-radius:3px;font-size:8px;margin-right:2px">{esc(c)}</span>' for c in claims[:5])
                claims_html = f'<div style="margin-top:2px">{tags}</div>'

            parts.append(f'''<div style="border-left:2px solid {role_color};padding:3px 8px;margin:2px 0;background:#0f172a">
              <div style="display:flex;justify-content:space-between;align-items:center">
                <span style="font-size:9px;font-weight:600;color:{role_color}">{esc(role).upper()} ({sid[:6]})</span>
                <span style="font-size:9px;color:#475569">C{cycle} | {conf_pct}%</span>
              </div>
              <div style="font-size:11px;color:#94a3b8;line-height:1.4;margin-top:1px;white-space:pre-wrap">{esc(display_text)}</div>
              {claims_html}
            </div>''')

    return "\n".join(parts)


# -------------------------------------------------------------------
#  SENTINEL MESH -- fractal view of all sentinels in the swarm
# -------------------------------------------------------------------
def build_sentinel_mesh_html():
    nodes = get_all_nodes()
    sentinels = {nid: n for nid, n in nodes.items() if n.get("role", "").lower() == "sentinel"}
    commanders = get_commanders()

    if not sentinels:
        return '''<div style="color:#64748b;padding:16px;text-align:center;background:#1e293b;border-radius:8px">
          No sentinels detected. The first sentinel (Spore 006) will appear once online.
        </div>'''

    parts = [f'''<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:8px">
      <div style="background:#1e293b;border:1px solid #334155;border-radius:6px;padding:4px 12px;text-align:center">
        <span style="font-size:18px;font-weight:700;color:#f97316">{len(sentinels)}</span>
        <span style="font-size:10px;color:#64748b">sentinels</span>
      </div>
      <div style="background:#1e293b;border:1px solid #334155;border-radius:6px;padding:4px 12px;text-align:center">
        <span style="font-size:18px;font-weight:700;color:#10b981">{sum(1 for s in sentinels.values() if s.get('online'))}</span>
        <span style="font-size:10px;color:#64748b">online</span>
      </div>
    </div>''']

    # Primary sentinel (ours) -- expanded view
    primary = sentinels.get("006")
    if primary and primary.get("online"):
        status = safe_get(f"{primary['url']}/api/sentinel/status") or {}
        proposals = status.get("proposals", {})
        total_deploy = status.get("total_deployments", 0)
        snapshots = status.get("telemetry_snapshots", 0)
        last_analysis = status.get("last_analysis", 0)
        active = status.get("active_proposal")
        cooldown = status.get("deploy_cooldown_remaining", 0)

        last_str = ""
        if last_analysis > 0:
            try:
                last_str = datetime.fromtimestamp(last_analysis).strftime("%H:%M:%S")
            except Exception:
                pass

        parts.append(f'''<div style="background:#1e293b;border:2px solid #f97316;border-radius:8px;padding:10px;margin-bottom:8px">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
            <span style="font-size:12px;font-weight:700;color:#f97316">PRIMARY SENTINEL (006) -- DeepSeek R1 671B</span>
            <span style="font-size:10px;color:#10b981">ONLINE</span>
          </div>
          <div style="display:flex;gap:12px;flex-wrap:wrap;font-size:10px;color:#94a3b8">
            <span>Proposals: {len(proposals)}</span>
            <span>Deployments: {total_deploy}</span>
            <span>Telemetry: {snapshots}</span>
            <span>Last: {last_str or "N/A"}</span>
            {f'<span style="color:#f59e0b">Cooldown: {int(cooldown)}s</span>' if cooldown > 0 else ""}
          </div>
        </div>''')

        # Active proposal
        if active and active in proposals:
            p = proposals[active]
            parts.append(f'''<div style="background:#1e3a5f;border:1px solid #3b82f6;border-radius:6px;padding:8px;margin-bottom:6px">
              <div style="font-size:11px;font-weight:700;color:#3b82f6;margin-bottom:4px">ACTIVE PROPOSAL</div>
              <div style="font-size:11px;color:#e2e8f0;line-height:1.5;white-space:pre-wrap">{esc(p.get("description", ""))}</div>
              <div style="font-size:9px;color:#64748b;margin-top:3px">Type: {esc(p.get("change_type", "?"))} | Status: {esc(p.get("status", "?"))}</div>
            </div>''')

        # Proposal history -- compact table
        if proposals:
            STATUS_COLORS = {
                "proposed": "#3b82f6", "awaiting_consensus": "#f59e0b", "consensus_reached": "#10b981",
                "testing": "#8b5cf6", "deployed": "#10b981", "rejected": "#ef4444",
                "failed": "#ef4444", "consensus_timeout": "#6b7280",
            }
            prop_rows = []
            for pid in reversed(list(proposals.keys())):
                p = proposals[pid]
                st = p.get("status", "unknown")
                st_color = STATUS_COLORS.get(st, "#6b7280")
                ts_val = p.get("submitted_at", 0)
                ts_str = ""
                if ts_val > 0:
                    try:
                        ts_str = datetime.fromtimestamp(ts_val).strftime("%H:%M")
                    except Exception:
                        pass
                desc = p.get("description", "")[:80]
                prop_rows.append(f'''<tr style="border-bottom:1px solid #1e293b;height:20px">
                  <td style="padding:1px 4px"><span style="background:{st_color};color:#fff;padding:0px 4px;border-radius:3px;font-size:8px">{esc(st).upper()[:12]}</span></td>
                  <td style="padding:1px 4px;font-size:10px;color:#94a3b8">{esc(desc)}</td>
                  <td style="padding:1px 4px;font-size:9px;color:#475569">{esc(p.get("change_type", ""))}</td>
                  <td style="padding:1px 4px;font-size:9px;color:#475569;text-align:right">{ts_str}</td>
                </tr>''')

            parts.append(f'''<div style="margin-top:4px">
              <div style="font-size:10px;font-weight:600;color:#94a3b8;margin-bottom:2px">Proposal History</div>
              <div style="max-height:160px;overflow-y:auto">
                <table style="width:100%;border-collapse:collapse;background:#0f172a">
                  <tr style="border-bottom:1px solid #334155">
                    <th style="padding:1px 4px;font-size:8px;color:#475569;text-align:left">Status</th>
                    <th style="padding:1px 4px;font-size:8px;color:#475569;text-align:left">Description</th>
                    <th style="padding:1px 4px;font-size:8px;color:#475569;text-align:left">Type</th>
                    <th style="padding:1px 4px;font-size:8px;color:#475569;text-align:right">Time</th>
                  </tr>
                  {"".join(prop_rows)}
                </table>
              </div>
            </div>''')

        # Deployment log -- compact table
        deploy_data = safe_get(f"{primary['url']}/api/sentinel/deployments") or {}
        deployments = deploy_data.get("deployments", [])
        if deployments:
            successful = deploy_data.get("successful", 0)
            failed = deploy_data.get("failed", 0)
            dep_rows = []
            for dep in reversed(deployments[-15:]):
                ds = dep.get("status", "unknown")
                dc = "#10b981" if ds == "success" else "#ef4444"
                dt = ""
                try:
                    dt = datetime.fromtimestamp(dep.get("time", 0)).strftime("%m-%d %H:%M")
                except Exception:
                    pass
                desc = dep.get("description", "")[:60]
                sha = dep.get("github_sha", "")[:7] or "N/A"
                dep_rows.append(f'''<tr style="border-bottom:1px solid #1e293b;height:20px">
                  <td style="padding:1px 4px"><span style="background:{dc};color:#fff;padding:0px 4px;border-radius:3px;font-size:8px">{esc(ds).upper()[:8]}</span></td>
                  <td style="padding:1px 4px;font-size:10px;color:#94a3b8">{esc(desc)}</td>
                  <td style="padding:1px 4px;font-size:9px;color:#3b82f6"><code>{sha}</code></td>
                  <td style="padding:1px 4px;font-size:9px;color:#475569;text-align:right">{dt}</td>
                </tr>''')

            parts.append(f'''<div style="margin-top:8px;border-top:1px solid #334155;padding-top:6px">
              <div style="font-size:10px;font-weight:600;color:#94a3b8;margin-bottom:2px">
                Deployments ({successful} OK / {failed} failed)
              </div>
              <div style="max-height:120px;overflow-y:auto">
                <table style="width:100%;border-collapse:collapse;background:#0f172a">
                  {"".join(dep_rows)}
                </table>
              </div>
            </div>''')

    # Other sentinels -- compact rows
    other_sentinels = {k: v for k, v in sentinels.items() if k != "006"}
    if other_sentinels:
        parts.append(f'''<div style="margin-top:8px;border-top:1px solid #334155;padding-top:6px">
          <div style="font-size:10px;font-weight:600;color:#94a3b8;margin-bottom:2px">Federation Sentinels</div>
        </div>''')
        for sid, sn in sorted(other_sentinels.items()):
            cmd = sn.get("commander", "?")
            online = sn.get("online")
            dot = '<span style="color:#10b981;font-size:8px">&#9679;</span>' if online else '<span style="color:#ef4444;font-size:8px">&#9679;</span>'
            parts.append(f'''<div style="display:flex;align-items:center;gap:6px;padding:2px 8px;background:#1e293b;border-radius:4px;margin:2px 0">
              {dot}
              <span style="font-size:10px;color:#f97316;font-weight:600">{esc(sid[:8])}</span>
              <span style="font-size:9px;color:#64748b">{esc(cmd)}</span>
              <span style="font-size:9px;color:#475569">{esc(sn.get('model', '?'))}</span>
            </div>''')

    return "\n".join(parts)


# -------------------------------------------------------------------
#  GLOBE MAP -- world view of all spores
# -------------------------------------------------------------------
# Platform -> approximate lat/lon for placement
PLATFORM_COORDS = {
    "hf-spaces": (37.77, -122.42),       # San Francisco (HF default)
    "hf-spaces-eu": (48.86, 2.35),        # Paris
    "railway": (40.71, -74.01),            # New York
    "render": (37.39, -122.08),            # Mountain View
    "fly-io": (51.51, -0.13),             # London
    "oracle-cloud": (37.39, -122.08),     # US West
    "google-cloud": (33.45, -111.98),     # Phoenix
    "koyeb": (48.86, 2.35),              # Paris
    "replit": (37.77, -122.42),           # San Francisco
    "default": (47.38, 8.54),        # Zurich (example)
    "unknown": (0, 0),
}


def _latlon_to_svg(lat, lon, width=900, height=450):
    """Equirectangular projection: lat/lon -> SVG x/y."""
    x = (lon + 180) / 360 * width
    y = (90 - lat) / 180 * height
    return x, y


def build_globe_map():
    nodes = get_all_nodes()
    commanders = get_commanders()

    # Assign coordinates -- use platform or commander location
    positioned = []
    for nid, n in nodes.items():
        platform = n.get("platform", "hf-spaces")
        commander = n.get("commander", "default")
        # Try commander-specific coords first, then platform
        if commander in PLATFORM_COORDS:
            lat, lon = PLATFORM_COORDS[commander]
        elif platform in PLATFORM_COORDS:
            lat, lon = PLATFORM_COORDS[platform]
        else:
            lat, lon = PLATFORM_COORDS["hf-spaces"]
        # Jitter to prevent overlap
        import hashlib
        h = hashlib.md5(nid.encode()).hexdigest()
        jx = (int(h[:4], 16) / 65535 - 0.5) * 8
        jy = (int(h[4:8], 16) / 65535 - 0.5) * 6
        lat += jy
        lon += jx
        positioned.append((nid, n, lat, lon))

    W, H = 900, 450

    # SVG world map outline (simplified continents)
    map_paths = '''
    <!-- Simplified continent outlines -->
    <path d="M170,120 L220,100 L280,110 L290,130 L260,180 L220,200 L200,280 L180,300 L160,280 L140,220 L150,160 Z" fill="#1e293b" stroke="#334155" stroke-width="0.5"/>
    <path d="M290,100 L340,90 L360,110 L350,160 L320,190 L290,180 L280,140 Z" fill="#1e293b" stroke="#334155" stroke-width="0.5"/>
    <path d="M400,80 L500,60 L600,70 L650,90 L680,130 L650,180 L600,200 L550,180 L500,190 L470,170 L440,130 L420,100 Z" fill="#1e293b" stroke="#334155" stroke-width="0.5"/>
    <path d="M480,170 L530,200 L560,250 L540,300 L500,310 L470,280 L460,230 L470,190 Z" fill="#1e293b" stroke="#334155" stroke-width="0.5"/>
    <path d="M620,200 L660,190 L700,210 L720,260 L700,320 L670,350 L640,340 L620,300 L610,250 Z" fill="#1e293b" stroke="#334155" stroke-width="0.5"/>
    <path d="M720,280 L780,270 L820,290 L830,330 L810,370 L770,380 L740,360 L730,320 Z" fill="#1e293b" stroke="#334155" stroke-width="0.5"/>
    '''

    # Spore dots
    dots = []
    tooltips = []
    for nid, n, lat, lon in positioned:
        x, y = _latlon_to_svg(lat, lon, W, H)
        online = n.get("online")
        role = n.get("role", "contributor")
        role_color = ROLE_COLORS.get(role.lower(), "#6b7280")
        r = 5 if role.lower() == "sentinel" else 3.5 if online else 2
        opacity = 1.0 if online else 0.3
        pulse = ""
        if role.lower() == "sentinel" and online:
            pulse = f'''<circle cx="{x}" cy="{y}" r="8" fill="none" stroke="{role_color}" stroke-width="1" opacity="0.4">
              <animate attributeName="r" values="5;12;5" dur="2s" repeatCount="indefinite"/>
              <animate attributeName="opacity" values="0.6;0;0.6" dur="2s" repeatCount="indefinite"/>
            </circle>'''

        mem = n.get("memories", 0)
        cyc = n.get("cycles", 0)
        tier = n.get("tier", "worker")
        conf = n.get("confidence", 0)
        conf_pct = int(conf * 100) if conf <= 1 else int(conf)
        model = n.get("model", "?")
        cmd = n.get("commander", "?")
        status = "ONLINE" if online else "OFFLINE" if online is not None else "UNKNOWN"

        dot_id = f"dot_{nid.replace('-', '_')}"
        dots.append(f'''{pulse}
          <circle id="{dot_id}" cx="{x}" cy="{y}" r="{r}" fill="{role_color}" opacity="{opacity}"
            style="cursor:pointer;transition:r 0.2s"
            onmouseover="this.setAttribute('r','{r+3}');document.getElementById('tip_{dot_id}').style.display='block'"
            onmouseout="this.setAttribute('r','{r}');document.getElementById('tip_{dot_id}').style.display='none'"/>''')

        # Tooltip positioned near the dot
        tx = min(x + 10, W - 180)
        ty = max(y - 60, 10)
        tooltips.append(f'''<foreignObject id="tip_{dot_id}" x="{tx}" y="{ty}" width="170" height="100" style="display:none;pointer-events:none">
          <div xmlns="http://www.w3.org/1999/xhtml" style="background:#1e293b;border:1px solid {role_color};border-radius:6px;padding:6px;font-size:10px;color:#e2e8f0;font-family:sans-serif;line-height:1.4">
            <div style="font-weight:700;color:{role_color}">{esc(role).upper()} {esc(nid[:8])}</div>
            <div style="color:#94a3b8">{esc(model)}</div>
            <div style="color:#64748b">Cmd: {esc(cmd)} | {status}</div>
            <div style="color:#64748b">Mem: {mem:,} | Cyc: {cyc} | {conf_pct}%</div>
            <div><span style="background:{'#10b981' if tier=='brain' else '#334155'};color:#fff;padding:0 3px;border-radius:2px;font-size:8px">{esc(str(tier).upper())}</span></div>
          </div>
        </foreignObject>''')

    # Connection lines between peers (sample -- just core spores)
    lines = []
    core_positions = {}
    for nid, n, lat, lon in positioned:
        x, y = _latlon_to_svg(lat, lon, W, H)
        core_positions[nid] = (x, y)

    for nid, n, lat, lon in positioned:
        peers = n.get("peers", [])
        if not isinstance(peers, list):
            continue
        x1, y1 = _latlon_to_svg(lat, lon, W, H)
        for peer in peers[:5]:  # Cap lines for readability
            pid = str(peer)
            # Resolve peer id
            for sid in core_positions:
                if sid in pid or f"spore-{sid}" in pid:
                    x2, y2 = core_positions[sid]
                    lines.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#334155" stroke-width="0.5" opacity="0.3"/>')
                    break

    # Stats overlay
    online_count = sum(1 for _, n, _, _ in positioned if n.get("online"))
    total_count = len(positioned)
    cmd_count = len(commanders)

    # Legend
    legend_items = []
    seen_roles = set()
    for _, n, _, _ in positioned:
        role = n.get("role", "contributor")
        if role.lower() not in seen_roles:
            seen_roles.add(role.lower())
            rc = ROLE_COLORS.get(role.lower(), "#6b7280")
            legend_items.append(f'<span style="display:inline-flex;align-items:center;gap:2px;margin-right:8px"><span style="width:8px;height:8px;border-radius:50%;background:{rc};display:inline-block"></span><span style="font-size:9px;color:#94a3b8">{role}</span></span>')

    svg = f'''<div style="background:#0f172a;border-radius:8px;padding:8px">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
        <div style="font-size:14px;font-weight:700;color:#e2e8f0">Swarm Topology</div>
        <div style="font-size:11px;color:#64748b">{online_count}/{total_count} nodes | {cmd_count} commanders</div>
      </div>
      <div style="display:flex;gap:4px;flex-wrap:wrap;margin-bottom:4px">{"".join(legend_items)}</div>
      <svg viewBox="0 0 {W} {H}" style="width:100%;background:#0a0f1a;border-radius:6px;border:1px solid #1e293b">
        <!-- Grid lines -->
        <line x1="0" y1="{H//2}" x2="{W}" y2="{H//2}" stroke="#1e293b" stroke-width="0.3" stroke-dasharray="4"/>
        <line x1="{W//2}" y1="0" x2="{W//2}" y2="{H}" stroke="#1e293b" stroke-width="0.3" stroke-dasharray="4"/>
        <line x1="0" y1="{H//4}" x2="{W}" y2="{H//4}" stroke="#1e293b" stroke-width="0.2" stroke-dasharray="2"/>
        <line x1="0" y1="{3*H//4}" x2="{W}" y2="{3*H//4}" stroke="#1e293b" stroke-width="0.2" stroke-dasharray="2"/>
        <line x1="{W//4}" y1="0" x2="{W//4}" y2="{H}" stroke="#1e293b" stroke-width="0.2" stroke-dasharray="2"/>
        <line x1="{3*W//4}" y1="0" x2="{3*W//4}" y2="{H}" stroke="#1e293b" stroke-width="0.2" stroke-dasharray="2"/>
        {map_paths}
        {"".join(lines)}
        {"".join(dots)}
        {"".join(tooltips)}
      </svg>
    </div>'''

    return svg


# -------------------------------------------------------------------
#  DASHBOARD REFRESH
# -------------------------------------------------------------------
def refresh_dashboard():
    discover_nodes()
    stats = build_stats_bar()
    topo = build_topology()
    trust = build_trust_summary()
    dash = f'''<div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0f172a;color:#e2e8f0;padding:8px;border-radius:8px">
      <div style="font-size:14px;font-weight:700;margin-bottom:4px">Synapse Collective</div>
      {stats}
      <div style="margin-top:6px">{topo}</div>
      {trust}
    </div>'''

    tid = _registry.get("current_task")
    if not tid:
        tasks = fetch_all_tasks()
        if tasks:
            tid = max(tasks.keys(), key=lambda k: tasks[k].get("delta_count", 0))

    conv = build_conversation_stream(tid) if tid else '<div style="color:#64748b;padding:12px;text-align:center;font-size:11px">Submit a prompt above.</div>'
    return dash, conv


def refresh_conversation_only():
    tid = _registry.get("current_task")
    if not tid:
        tasks = fetch_all_tasks()
        if tasks:
            tid = max(tasks.keys(), key=lambda k: tasks[k].get("delta_count", 0))
    if tid:
        return build_conversation_stream(tid)
    return '<div style="color:#64748b;padding:12px;text-align:center;font-size:11px">No active conversation.</div>'


def chunk_document(file_path, max_chunk_chars=3000, overlap=200):
    """Read and chunk a document for swarm analysis.

    Supports: .txt, .md, .py, .json, .csv, .html, .log and other text files.
    Chunks at paragraph/section boundaries with overlap for context continuity.
    Returns list of chunk strings.
    """
    if not file_path:
        return []
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except Exception as e:
        return [f"[Document read error: {e}]"]

    if not text.strip():
        return []

    # Split on double newlines (paragraphs/sections)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 > max_chunk_chars and current:
            chunks.append(current)
            # Overlap: keep the tail of the previous chunk
            if len(current) > overlap:
                current = current[-overlap:] + "\n\n" + para
            else:
                current = para
        else:
            current = (current + "\n\n" + para).strip() if current else para

    if current:
        chunks.append(current)

    return chunks


def submit_prompt(prompt, doc_file=None):
    if not prompt or not prompt.strip():
        if not doc_file:
            return "Enter a prompt.", ""

    # Build the task text: prompt + chunked document if attached
    task_text = prompt.strip() if prompt else ""
    doc_summary = ""

    if doc_file:
        import os
        fname = os.path.basename(doc_file) if isinstance(doc_file, str) else "document"
        chunks = chunk_document(doc_file)
        if chunks:
            n_chunks = len(chunks)
            total_chars = sum(len(c) for c in chunks)
            doc_context = "\n\n---\n\n".join(
                f"[Chunk {i+1}/{n_chunks}]\n{c}" for i, c in enumerate(chunks)
            )
            if task_text:
                task_text = (
                    f"{task_text}\n\n"
                    f"--- ATTACHED DOCUMENT: {fname} ({n_chunks} chunks, {total_chars:,} chars) ---\n\n"
                    f"Perform full enumerative analysis of this document.\n\n{doc_context}"
                )
            else:
                task_text = (
                    f"Analyze the following document: {fname} ({n_chunks} chunks, {total_chars:,} chars)\n\n"
                    f"Perform full enumerative analysis.\n\n{doc_context}"
                )
            doc_summary = f" + {fname} ({n_chunks} chunks)"

    if not task_text:
        return "Enter a prompt or attach a document.", ""

    task_data = {"task": task_text}
    submitted = 0
    task_id = None
    nodes = get_all_nodes()
    for n in nodes.values():
        if not n.get("online", True):
            continue
        r = safe_post(f"{n['url']}/api/task", task_data, timeout=15)
        if r:
            submitted += 1
            if not task_id:
                task_id = r.get("task_id", r.get("id", ""))
    if submitted == 0:
        return '<div style="color:#ef4444;font-size:11px">Failed to submit to any node.</div>', ""
    _registry["current_task"] = task_id
    total = len(nodes)
    msg = f'<div style="color:#10b981;font-size:11px">Submitted to {submitted}/{total} nodes{doc_summary}. Auto-refreshing.</div>'
    conv = build_conversation_stream(task_id) if task_id else ""
    return msg, conv


def new_conversation():
    _registry["current_task"] = None
    return (
        '<div style="color:#94a3b8;font-size:11px">New conversation started.</div>',
        '<div style="color:#64748b;padding:12px;text-align:center;font-size:11px">Submit a prompt above.</div>',
        ""
    )


# -------------------------------------------------------------------
#  LIBRARY TAB
# -------------------------------------------------------------------
def refresh_library():
    all_tasks = fetch_all_tasks()
    if not all_tasks:
        return '<div style="color:#64748b;padding:12px;font-size:11px">No conversations yet.</div>', gr.update(choices=[], value=None)
    sorted_tasks = sorted(all_tasks.values(), key=lambda t: t.get("delta_count", 0), reverse=True)
    choices = []
    for t in sorted_tasks:
        tid = t.get("task_id", "")
        desc = t.get("description", "?")[:50]
        dc = t.get("delta_count", 0)
        conv = "OK" if t.get("converged") else "--"
        choices.append((f"[{conv}] {desc} ({dc}d)", tid))
    summary = f'<div style="color:#94a3b8;font-size:11px">{len(choices)} conversations ({sum(1 for t in sorted_tasks if t.get("converged"))} converged)</div>'
    return summary, gr.update(choices=choices, value=None)


def view_library_task(task_id):
    if not task_id:
        return '<div style="color:#64748b;padding:12px;font-size:11px">Select a conversation.</div>'
    return build_conversation_stream(task_id)




# -------------------------------------------------------------------
#  ANALYTICS -- quantitative, qualitative, performance, capacity
# -------------------------------------------------------------------
_analytics_cache = {"data": None, "ts": 0}

def collect_analytics():
    """Aggregate analytics from all spores. Cached for 15s."""
    import time
    now = time.time()
    if _analytics_cache["data"] and now - _analytics_cache["ts"] < 15:
        return _analytics_cache["data"]

    nodes = get_all_nodes()
    online_nodes = {nid: n for nid, n in nodes.items() if n.get("online")}

    # Per-spore health data
    spore_data = []
    for nid, n in sorted(online_nodes.items()):
        health = safe_get(f"{n['url']}/api/health")
        wall = safe_get(f"{n['url']}/api/wall")
        cortex = safe_get(f"{n['url']}/api/cortex") if n.get("role", "").lower() == "sentinel" else None
        sidecar = safe_get(f"{n['url']}/api/sidecar")
        trust = safe_get(f"{n['url']}/api/trust")

        if not health:
            continue

        # Parse uptime to seconds
        uptime_str = health.get("uptime", "0h 0m 0s")
        uptime_secs = 0
        try:
            parts = uptime_str.replace("h", "").replace("m", "").replace("s", "").split()
            if len(parts) >= 3:
                uptime_secs = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            elif len(parts) >= 2:
                uptime_secs = int(parts[0]) * 3600 + int(parts[1]) * 60
        except (ValueError, IndexError):
            uptime_secs = 1

        memories = health.get("memories", 0)
        cycles = health.get("cycles", 0)
        dp = health.get("deltas_produced", 0)
        dr = health.get("deltas_received", 0)
        peers = health.get("peers", [])
        n_peers = len(peers) if isinstance(peers, list) else 0
        converged = health.get("converged_tasks", 0)
        active = health.get("active_tasks", 0)
        provider = health.get("last_provider", "none")
        model = health.get("last_model", "none")

        # Clock entries
        clock = health.get("clock", {})
        clock_entries = len(clock.get("clocks", {})) if isinstance(clock, dict) else 0

        # Wall stats
        wall_stats = {}
        if wall and isinstance(wall, dict):
            w = wall.get("wall", {})
            wall_stats = {
                "crossings": w.get("crossings", 0),
                "blocks": w.get("blocks", 0),
                "audits": w.get("audit_entries", w.get("audit_count", 0)),
                "block_rate": w.get("block_rate", 0),
                "collective_size": wall.get("collective_size", 0),
            }

        # Trust values
        trust_vals = []
        if trust and isinstance(trust, dict):
            for k, v in trust.items():
                if isinstance(v, dict):
                    tv = v.get("overall", 0)
                else:
                    try:
                        tv = float(v)
                    except (TypeError, ValueError):
                        continue
                trust_vals.append(tv)

        # Growth rate
        mem_per_min = (memories / max(uptime_secs, 1)) * 60

        spore_data.append({
            "id": nid,
            "role": n.get("role", "?"),
            "model": n.get("model", "?"),
            "color": n.get("color", "#6b7280"),
            "memories": memories,
            "cycles": cycles,
            "dp": dp, "dr": dr,
            "peers": n_peers,
            "converged": converged,
            "active": active,
            "uptime_secs": uptime_secs,
            "uptime_str": uptime_str,
            "provider": provider,
            "model_used": model,
            "clock_entries": clock_entries,
            "wall": wall_stats,
            "trust_vals": trust_vals,
            "mem_per_min": mem_per_min,
            "cortex": cortex,
            "sidecar": sidecar or {},
        })

    # Aggregates
    total_mem = sum(s["memories"] for s in spore_data)
    total_dp = sum(s["dp"] for s in spore_data)
    total_dr = sum(s["dr"] for s in spore_data)
    total_cycles = sum(s["cycles"] for s in spore_data)
    total_converged = max((s["converged"] for s in spore_data), default=0)
    total_active = max((s["active"] for s in spore_data), default=0)
    all_trust = []
    for s in spore_data:
        all_trust.extend(s["trust_vals"])
    avg_trust = sum(all_trust) / len(all_trust) if all_trust else 0
    min_trust = min(all_trust) if all_trust else 0
    max_trust = max(all_trust) if all_trust else 0

    # Wall aggregates
    total_crossings = sum(s["wall"].get("crossings", 0) for s in spore_data)
    total_blocks = sum(s["wall"].get("blocks", 0) for s in spore_data)
    total_audits = sum(s["wall"].get("audits", 0) for s in spore_data)
    total_collective = sum(s["wall"].get("collective_size", 0) for s in spore_data)

    # Provider status
    providers = {}
    for s in spore_data:
        p = s["provider"]
        if p not in providers:
            providers[p] = {"count": 0, "models": set()}
        providers[p]["count"] += 1
        if s["model_used"] and s["model_used"] != "none":
            providers[p]["models"].add(s["model_used"])

    # Sidecar aggregates
    total_vocab = sum(s["sidecar"].get("vocab_features", 0) for s in spore_data)
    avg_growth_rate = sum(s["sidecar"].get("growth_rate", {}).get("per_hour", 0) for s in spore_data) / max(len(spore_data), 1)

    # Max memory per spore (for capacity bar)
    max_mem = max((s["memories"] for s in spore_data), default=1)

    result = {
        "spores": spore_data,
        "total_mem": total_mem,
        "total_dp": total_dp,
        "total_dr": total_dr,
        "total_cycles": total_cycles,
        "total_converged": total_converged,
        "total_active": total_active,
        "avg_trust": avg_trust,
        "min_trust": min_trust,
        "max_trust": max_trust,
        "total_crossings": total_crossings,
        "total_blocks": total_blocks,
        "total_audits": total_audits,
        "total_collective": total_collective,
        "providers": providers,
        "max_mem": max_mem,
        "total_vocab": total_vocab,
        "avg_growth_rate": avg_growth_rate,
        "n_online": len(spore_data),
        "n_total": len(nodes),
    }
    _analytics_cache["data"] = result
    _analytics_cache["ts"] = now
    return result


def build_analytics_html():
    """Build the full analytics dashboard HTML."""
    data = collect_analytics()
    if not data or not data["spores"]:
        return '<div style="color:#64748b;padding:20px;text-align:center">No spore data available.</div>'

    sections = []

    # --- HEADER ---
    sections.append(f'''<div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0f172a;color:#e2e8f0;padding:12px;border-radius:8px">
      <div style="font-size:16px;font-weight:700;margin-bottom:8px">Swarm Analytics</div>''')

    # --- 1. MEMORY: NEVER-FORGETTING ACCUMULATION ---
    sections.append('''<div style="margin-bottom:12px">
      <div style="font-size:13px;font-weight:700;color:#3b82f6;margin-bottom:6px;border-bottom:1px solid #1e293b;padding-bottom:3px">Memory Accumulation</div>''')

    # Monotonic guarantee badge
    sections.append(f'''<div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:8px">
        <div style="background:#064e3b;border:1px solid #10b981;border-radius:6px;padding:6px 14px;text-align:center">
          <div style="font-size:22px;font-weight:800;color:#10b981">{data["total_mem"]:,}</div>
          <div style="font-size:9px;color:#6ee7b7">total memories (aggregate)</div>
        </div>
        <div style="background:#1e293b;border:1px solid #334155;border-radius:6px;padding:6px 14px;text-align:center">
          <div style="font-size:22px;font-weight:800;color:#f59e0b">{data["total_converged"]}</div>
          <div style="font-size:9px;color:#fcd34d">converged tasks</div>
        </div>
        <div style="background:#1e293b;border:1px solid #334155;border-radius:6px;padding:6px 14px;text-align:center">
          <div style="font-size:22px;font-weight:800;color:#8b5cf6">{data["total_dp"] + data["total_dr"]:,}</div>
          <div style="font-size:9px;color:#c4b5fd">total deltas exchanged</div>
        </div>
        <div style="background:#0c4a6e;border:1px solid #0284c7;border-radius:6px;padding:6px 14px;text-align:center">
          <div style="font-size:10px;font-weight:700;color:#38bdf8">MONOTONIC GUARANTEE</div>
          <div style="font-size:9px;color:#7dd3fc">OR-Set add-wins: memories only grow, never shrink</div>
        </div>
      </div>''')

    # Per-spore memory bars
    max_mem = data["max_mem"] or 1
    bar_rows = []
    for s in data["spores"]:
        pct = int((s["memories"] / max_mem) * 100)
        rate = f"{s['mem_per_min']:.1f}/min"
        bar_rows.append(f'''<div style="display:flex;align-items:center;gap:6px;margin:2px 0">
          <span style="width:50px;font-size:10px;color:{s['color']};font-weight:600">{s['id'][:6]}</span>
          <span style="width:60px;font-size:9px;color:#64748b">{s['role'][:10]}</span>
          <div style="flex:1;background:#1e293b;border-radius:3px;height:12px;position:relative">
            <div style="background:{s['color']};height:12px;border-radius:3px;width:{pct}%;min-width:2px"></div>
            <span style="position:absolute;right:4px;top:0;font-size:9px;color:#e2e8f0;line-height:12px">{s['memories']:,}</span>
          </div>
          <span style="width:55px;font-size:9px;color:#475569;text-align:right">{rate}</span>
        </div>''')
    sections.append(f'''<div style="background:#0f172a;border:1px solid #1e293b;border-radius:6px;padding:8px">
      <div style="font-size:10px;font-weight:600;color:#94a3b8;margin-bottom:4px">Per-Spore Memory Distribution</div>
      {"".join(bar_rows)}
    </div></div>''')

    # --- 2. GOSSIP & CONVERGENCE ---
    conv_rate = (data["total_converged"] / max(data["total_converged"] + data["total_active"], 1)) * 100
    avg_cycles = data["total_cycles"] / max(data["n_online"], 1)
    total_deltas_per_min = sum(
        (s["dp"] + s["dr"]) / max(s["uptime_secs"], 1) * 60
        for s in data["spores"]
    )

    sections.append(f'''<div style="margin-bottom:12px">
      <div style="font-size:13px;font-weight:700;color:#8b5cf6;margin-bottom:6px;border-bottom:1px solid #1e293b;padding-bottom:3px">Gossip Sidecar Performance</div>
      <div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:8px">
        <div style="background:#1e293b;border:1px solid #334155;border-radius:6px;padding:5px 12px;text-align:center">
          <div style="font-size:18px;font-weight:700;color:#8b5cf6">{total_deltas_per_min:.1f}</div>
          <div style="font-size:9px;color:#a78bfa">deltas/min (swarm)</div>
        </div>
        <div style="background:#1e293b;border:1px solid #334155;border-radius:6px;padding:5px 12px;text-align:center">
          <div style="font-size:18px;font-weight:700;color:#10b981">{conv_rate:.0f}%</div>
          <div style="font-size:9px;color:#6ee7b7">convergence rate</div>
        </div>
        <div style="background:#1e293b;border:1px solid #334155;border-radius:6px;padding:5px 12px;text-align:center">
          <div style="font-size:18px;font-weight:700;color:#f59e0b">{avg_cycles:.1f}</div>
          <div style="font-size:9px;color:#fcd34d">avg cycles/spore</div>
        </div>
      </div>''')

    # Delta produced vs received table
    delta_rows = []
    for s in data["spores"]:
        total_d = s["dp"] + s["dr"]
        ratio = f"{s['dp']}:{s['dr']}"
        d_per_min = total_d / max(s["uptime_secs"], 1) * 60
        delta_rows.append(f'''<tr style="border-bottom:1px solid #1e293b;height:20px">
          <td style="padding:1px 6px;font-size:10px;color:{s['color']};font-weight:600">{s['id'][:6]}</td>
          <td style="padding:1px 6px;font-size:10px;color:#94a3b8;text-align:right">{s['dp']}</td>
          <td style="padding:1px 6px;font-size:10px;color:#94a3b8;text-align:right">{s['dr']}</td>
          <td style="padding:1px 6px;font-size:10px;color:#64748b;text-align:right">{d_per_min:.1f}/m</td>
          <td style="padding:1px 6px;font-size:10px;color:#64748b;text-align:right">{s['peers']}</td>
          <td style="padding:1px 6px;font-size:10px;color:#64748b;text-align:right">{s['clock_entries']}</td>
        </tr>''')
    sections.append(f'''<table style="width:100%;border-collapse:collapse;background:#0f172a;border:1px solid #1e293b;border-radius:6px">
        <tr style="border-bottom:1px solid #334155">
          <th style="padding:2px 6px;font-size:9px;color:#475569;text-align:left">Spore</th>
          <th style="padding:2px 6px;font-size:9px;color:#475569;text-align:right">Produced</th>
          <th style="padding:2px 6px;font-size:9px;color:#475569;text-align:right">Received</th>
          <th style="padding:2px 6px;font-size:9px;color:#475569;text-align:right">Rate</th>
          <th style="padding:2px 6px;font-size:9px;color:#475569;text-align:right">Peers</th>
          <th style="padding:2px 6px;font-size:9px;color:#475569;text-align:right">Clock</th>
        </tr>
        {"".join(delta_rows)}
      </table></div>''')

    # --- 3. KNOWLEDGE WALL ---
    sections.append(f'''<div style="margin-bottom:12px">
      <div style="font-size:13px;font-weight:700;color:#ec4899;margin-bottom:6px;border-bottom:1px solid #1e293b;padding-bottom:3px">Knowledge Wall (Privacy Boundary)</div>
      <div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:6px">
        <div style="background:#1e293b;border:1px solid #334155;border-radius:6px;padding:5px 12px;text-align:center">
          <div style="font-size:18px;font-weight:700;color:#10b981">{data["total_crossings"]}</div>
          <div style="font-size:9px;color:#6ee7b7">crossings (distilled)</div>
        </div>
        <div style="background:#1e293b;border:1px solid #334155;border-radius:6px;padding:5px 12px;text-align:center">
          <div style="font-size:18px;font-weight:700;color:#ef4444">{data["total_blocks"]}</div>
          <div style="font-size:9px;color:#fca5a5">blocks (privacy-filtered)</div>
        </div>
        <div style="background:#1e293b;border:1px solid #334155;border-radius:6px;padding:5px 12px;text-align:center">
          <div style="font-size:18px;font-weight:700;color:#f59e0b">{data["total_audits"]}</div>
          <div style="font-size:9px;color:#fcd34d">audit entries</div>
        </div>
        <div style="background:#1e293b;border:1px solid #334155;border-radius:6px;padding:5px 12px;text-align:center">
          <div style="font-size:18px;font-weight:700;color:#8b5cf6">{data["total_collective"]}</div>
          <div style="font-size:9px;color:#c4b5fd">collective knowledge items</div>
        </div>
      </div>
      <div style="font-size:10px;color:#64748b;padding:4px 8px;background:#1e293b;border-radius:4px">
        Raw input never enters gossip protocol. Only distilled insights cross the wall.
        HMAC-bound provenance ensures every crossing is auditable.
      </div>
    </div>''')

    # --- 4. TRUST LATTICE ---
    trust_dist = [0, 0, 0, 0, 0]  # 0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0
    for s in data["spores"]:
        for t in s["trust_vals"]:
            bucket = min(int(t * 5), 4)
            trust_dist[bucket] += 1
    trust_total = sum(trust_dist) or 1
    trust_labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    trust_colors = ["#ef4444", "#f59e0b", "#eab308", "#10b981", "#3b82f6"]

    hist_bars = []
    for i, (label, count, color) in enumerate(zip(trust_labels, trust_dist, trust_colors)):
        pct = int((count / trust_total) * 100)
        hist_bars.append(f'''<div style="text-align:center;flex:1">
          <div style="background:#1e293b;border-radius:3px;height:60px;display:flex;align-items:flex-end;justify-content:center;padding:0 2px">
            <div style="background:{color};width:100%;border-radius:2px 2px 0 0;height:{max(pct, 2)}%"></div>
          </div>
          <div style="font-size:8px;color:#64748b;margin-top:2px">{label}</div>
          <div style="font-size:9px;color:#94a3b8">{count}</div>
        </div>''')

    sections.append(f'''<div style="margin-bottom:12px">
      <div style="font-size:13px;font-weight:700;color:#f59e0b;margin-bottom:6px;border-bottom:1px solid #1e293b;padding-bottom:3px">E4 Trust Lattice</div>
      <div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:8px">
        <div style="background:#1e293b;border:1px solid #334155;border-radius:6px;padding:5px 12px;text-align:center">
          <div style="font-size:18px;font-weight:700;color:#f59e0b">{data["avg_trust"]:.3f}</div>
          <div style="font-size:9px;color:#fcd34d">mean trust</div>
        </div>
        <div style="background:#1e293b;border:1px solid #334155;border-radius:6px;padding:5px 12px;text-align:center">
          <div style="font-size:18px;font-weight:700;color:#ef4444">{data["min_trust"]:.3f}</div>
          <div style="font-size:9px;color:#fca5a5">min trust</div>
        </div>
        <div style="background:#1e293b;border:1px solid #334155;border-radius:6px;padding:5px 12px;text-align:center">
          <div style="font-size:18px;font-weight:700;color:#10b981">{data["max_trust"]:.3f}</div>
          <div style="font-size:9px;color:#6ee7b7">max trust</div>
        </div>
      </div>
      <div style="display:flex;gap:3px;padding:8px;background:#0f172a;border:1px solid #1e293b;border-radius:6px">
        {"".join(hist_bars)}
      </div>
      <div style="font-size:9px;color:#475569;text-align:center;margin-top:3px">Trust score distribution across all peer pairs</div>
    </div>''')

    # --- 5. CAPACITY & SLIDING WINDOW ---
    # Estimate hours to various memory milestones
    avg_rate = sum(s["mem_per_min"] for s in data["spores"]) / max(data["n_online"], 1)
    if avg_rate > 0:
        hrs_to_10k = max(0, (10000 - data["spores"][0]["memories"]) / avg_rate / 60) if data["spores"] else 0
        hrs_to_50k = max(0, (50000 - data["spores"][0]["memories"]) / avg_rate / 60) if data["spores"] else 0
        hrs_to_100k = max(0, (100000 - data["spores"][0]["memories"]) / avg_rate / 60) if data["spores"] else 0
    else:
        hrs_to_10k = hrs_to_50k = hrs_to_100k = 0

    max_uptime = max((s["uptime_secs"] for s in data["spores"]), default=0)
    max_uptime_hrs = max_uptime / 3600

    sections.append(f'''<div style="margin-bottom:12px">
      <div style="font-size:13px;font-weight:700;color:#10b981;margin-bottom:6px;border-bottom:1px solid #1e293b;padding-bottom:3px">Capacity & Intelligent Window</div>
      <div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:8px">
        <div style="background:#1e293b;border:1px solid #334155;border-radius:6px;padding:5px 12px;text-align:center">
          <div style="font-size:18px;font-weight:700;color:#10b981">{avg_rate:.1f}</div>
          <div style="font-size:9px;color:#6ee7b7">memories/min (avg/spore)</div>
        </div>
        <div style="background:#1e293b;border:1px solid #334155;border-radius:6px;padding:5px 12px;text-align:center">
          <div style="font-size:18px;font-weight:700;color:#3b82f6">{max_uptime_hrs:.1f}h</div>
          <div style="font-size:9px;color:#93c5fd">max uptime</div>
        </div>
        <div style="background:#1e293b;border:1px solid #334155;border-radius:6px;padding:5px 12px;text-align:center">
          <div style="font-size:18px;font-weight:700;color:#f59e0b">{data["n_online"]}/{data["n_total"]}</div>
          <div style="font-size:9px;color:#fcd34d">nodes online</div>
        </div>
      </div>
      <div style="background:#0f172a;border:1px solid #1e293b;border-radius:6px;padding:8px">
        <div style="font-size:10px;font-weight:600;color:#94a3b8;margin-bottom:4px">Scaling Projections (per spore at current rate)</div>
        <div style="display:flex;gap:8px;flex-wrap:wrap">
          <div style="font-size:10px;color:#64748b">10K: <span style="color:#10b981">{hrs_to_10k:.0f}h</span></div>
          <div style="font-size:10px;color:#64748b">50K: <span style="color:#f59e0b">{hrs_to_50k:.0f}h</span></div>
          <div style="font-size:10px;color:#64748b">100K: <span style="color:#ef4444">{hrs_to_100k:.0f}h</span></div>
        </div>
        <div style="font-size:10px;color:#475569;margin-top:4px">
          Semantic sidecar indexes continuously. Retrieval is O(1) via embedding similarity --
          context window never grows regardless of total memory size. Every memory persists forever;
          the intelligent sliding window surfaces only the most relevant context per query.
        </div>
      </div>
    </div>''')

    # --- 5b. SIDECAR PERFORMANCE ---
    sidecar_rows = []
    for s in data["spores"]:
        sc = s.get("sidecar", {})
        gr = sc.get("growth_rate", {})
        corpus = sc.get("corpus_size", 0)
        vocab = sc.get("vocab_features", 0)
        idx_health = sc.get("index_health", "unknown")
        idx_color = "#10b981" if idx_health == "current" else "#f59e0b" if idx_health == "refit_pending" else "#64748b"
        footprint = sc.get("estimated_footprint_mb", 0)
        merkle = sc.get("merkle_root", "N/A")
        sidecar_rows.append(f'''<tr style="border-bottom:1px solid #1e293b;height:22px">
          <td style="padding:2px 6px;font-size:10px;color:{s['color']};font-weight:600">{s['id'][:6]}</td>
          <td style="padding:2px 6px;font-size:10px;color:#94a3b8;text-align:right">{corpus:,}</td>
          <td style="padding:2px 6px;font-size:10px;color:#94a3b8;text-align:right">{vocab:,}</td>
          <td style="padding:2px 6px;font-size:10px;color:#94a3b8;text-align:right">{gr.get("per_hour", 0):.0f}/h</td>
          <td style="padding:2px 6px;font-size:10px;color:{idx_color};text-align:center">{idx_health}</td>
          <td style="padding:2px 6px;font-size:10px;color:#64748b;text-align:right">{footprint:.1f} MB</td>
          <td style="padding:2px 6px;font-size:10px;color:#64748b;font-family:monospace;font-size:9px">{merkle}</td>
        </tr>''')

    # Statistical summary
    all_corpus = [s.get("sidecar", {}).get("corpus_size", 0) for s in data["spores"]]
    all_vocab = [s.get("sidecar", {}).get("vocab_features", 0) for s in data["spores"]]
    all_rates = [s.get("sidecar", {}).get("growth_rate", {}).get("per_hour", 0) for s in data["spores"]]
    
    import statistics
    def safe_stats(vals):
        vals = [v for v in vals if v > 0]
        if len(vals) < 2:
            return {"mean": sum(vals)/max(len(vals),1), "stdev": 0, "min": min(vals) if vals else 0, "max": max(vals) if vals else 0}
        return {"mean": statistics.mean(vals), "stdev": statistics.stdev(vals), "min": min(vals), "max": max(vals)}
    
    cs = safe_stats(all_corpus)
    vs = safe_stats(all_vocab)
    rs = safe_stats(all_rates)

    sections.append(f'''<div style="margin-bottom:12px">
      <div style="font-size:13px;font-weight:700;color:#14b8a6;margin-bottom:6px;border-bottom:1px solid #1e293b;padding-bottom:3px">Sidecar Performance (Semantic Indexer)</div>
      <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px">
        <div style="background:#1e293b;border:1px solid #334155;border-radius:6px;padding:5px 10px;text-align:center">
          <div style="font-size:16px;font-weight:700;color:#14b8a6">{data.get("total_vocab", 0):,}</div>
          <div style="font-size:9px;color:#5eead4">TF-IDF features (aggregate)</div>
        </div>
        <div style="background:#1e293b;border:1px solid #334155;border-radius:6px;padding:5px 10px;text-align:center">
          <div style="font-size:16px;font-weight:700;color:#3b82f6">{data.get("avg_growth_rate", 0):.0f}/h</div>
          <div style="font-size:9px;color:#93c5fd">avg growth (per spore)</div>
        </div>
        <div style="background:#0c4a6e;border:1px solid #0284c7;border-radius:6px;padding:5px 10px;text-align:center">
          <div style="font-size:9px;font-weight:700;color:#38bdf8">SLIDING WINDOW</div>
          <div style="font-size:8px;color:#7dd3fc">TF-IDF cosine similarity</div>
          <div style="font-size:8px;color:#7dd3fc">Context = top-K relevant, not all memories</div>
        </div>
      </div>
      <table style="width:100%;border-collapse:collapse;background:#0f172a;border:1px solid #1e293b;border-radius:6px;margin-bottom:6px">
        <tr style="border-bottom:1px solid #334155">
          <th style="padding:2px 6px;font-size:9px;color:#475569;text-align:left">Spore</th>
          <th style="padding:2px 6px;font-size:9px;color:#475569;text-align:right">Corpus</th>
          <th style="padding:2px 6px;font-size:9px;color:#475569;text-align:right">Vocab</th>
          <th style="padding:2px 6px;font-size:9px;color:#475569;text-align:right">Rate</th>
          <th style="padding:2px 6px;font-size:9px;color:#475569;text-align:center">Index</th>
          <th style="padding:2px 6px;font-size:9px;color:#475569;text-align:right">Footprint</th>
          <th style="padding:2px 6px;font-size:9px;color:#475569;text-align:left">Merkle</th>
        </tr>
        {"".join(sidecar_rows)}
      </table>
      <div style="background:#1e293b;border:1px solid #334155;border-radius:6px;padding:6px 10px;margin-bottom:6px">
        <div style="font-size:10px;font-weight:600;color:#94a3b8;margin-bottom:3px">Statistical Summary</div>
        <div style="display:flex;gap:16px;flex-wrap:wrap;font-size:10px;color:#64748b">
          <div>Corpus: mean={cs["mean"]:.0f} stdev={cs["stdev"]:.0f} range=[{cs["min"]:.0f}, {cs["max"]:.0f}]</div>
          <div>Vocab: mean={vs["mean"]:.0f} stdev={vs["stdev"]:.0f} range=[{vs["min"]:.0f}, {vs["max"]:.0f}]</div>
          <div>Growth: mean={rs["mean"]:.0f}/h stdev={rs["stdev"]:.0f} range=[{rs["min"]:.0f}, {rs["max"]:.0f}]</div>
        </div>
      </div>
      <div style="font-size:10px;color:#475569;padding:4px 8px;background:#0f172a;border:1px solid #1e293b;border-radius:4px">
        <strong style="color:#14b8a6">Never-forgetting guarantee:</strong> OR-Set add-wins semantics means memories only accumulate.
        The semantic indexer continuously refits the TF-IDF vectorizer as new memories arrive.
        Context window stays fixed at top-K regardless of total memory size -- no context pressure at any scale.
        Merkle root changes on every insert, providing cryptographic proof of memory integrity.
      </div>
    </div>''')

    # --- 6. PROVIDER STATUS ---
    prov_rows = []
    for pname, pdata in sorted(data["providers"].items()):
        pcolor = "#10b981" if pname != "none" else "#ef4444"
        models_str = ", ".join(sorted(pdata["models"]))[:60] if pdata["models"] else "N/A"
        prov_rows.append(f'''<tr style="border-bottom:1px solid #1e293b;height:20px">
          <td style="padding:2px 6px"><span style="color:{pcolor};font-size:10px;font-weight:600">{esc(pname)}</span></td>
          <td style="padding:2px 6px;font-size:10px;color:#94a3b8;text-align:right">{pdata["count"]} spores</td>
          <td style="padding:2px 6px;font-size:10px;color:#64748b">{esc(models_str)}</td>
        </tr>''')

    sections.append(f'''<div style="margin-bottom:12px">
      <div style="font-size:13px;font-weight:700;color:#06b6d4;margin-bottom:6px;border-bottom:1px solid #1e293b;padding-bottom:3px">LLM Provider Status</div>
      <table style="width:100%;border-collapse:collapse;background:#0f172a;border:1px solid #1e293b;border-radius:6px">
        <tr style="border-bottom:1px solid #334155">
          <th style="padding:2px 6px;font-size:9px;color:#475569;text-align:left">Provider</th>
          <th style="padding:2px 6px;font-size:9px;color:#475569;text-align:right">Nodes</th>
          <th style="padding:2px 6px;font-size:9px;color:#475569;text-align:left">Models</th>
        </tr>
        {"".join(prov_rows)}
      </table>
      <div style="font-size:10px;color:#475569;margin-top:4px;padding:4px 8px;background:#1e293b;border-radius:4px">
        Each commander configures their own API keys. Spores fallback through:
        HF Router (free) &rarr; Z.ai (free) &rarr; Groq / Google AI / Cerebras (free tier) &rarr; OpenRouter
      </div>
    </div>''')

    # --- 7. CORTEX (Sentinel only) ---
    cortex_spores = [s for s in data["spores"] if s.get("cortex")]
    if cortex_spores:
        for s in cortex_spores:
            c = s["cortex"]
            sections.append(f'''<div style="margin-bottom:12px">
              <div style="font-size:13px;font-weight:700;color:#f97316;margin-bottom:6px;border-bottom:1px solid #1e293b;padding-bottom:3px">Cortex (Local Micro-LLM on Sentinel)</div>
              <div style="display:flex;gap:10px;flex-wrap:wrap">
                <div style="background:#1e293b;border:1px solid #334155;border-radius:6px;padding:5px 12px;text-align:center">
                  <div style="font-size:14px;font-weight:700;color:{'#10b981' if c.get('loaded') else '#ef4444'}">{'LOADED' if c.get('loaded') else 'OFFLINE'}</div>
                  <div style="font-size:9px;color:#64748b">Qwen3-4B Q4_K_M</div>
                </div>
                <div style="background:#1e293b;border:1px solid #334155;border-radius:6px;padding:5px 12px;text-align:center">
                  <div style="font-size:14px;font-weight:700;color:#8b5cf6">{c.get("total_calls", 0)}</div>
                  <div style="font-size:9px;color:#64748b">total calls</div>
                </div>
                <div style="background:#1e293b;border:1px solid #334155;border-radius:6px;padding:5px 12px;text-align:center">
                  <div style="font-size:14px;font-weight:700;color:#3b82f6">{c.get("total_tokens", 0):,}</div>
                  <div style="font-size:9px;color:#64748b">tokens generated</div>
                </div>
                <div style="background:#1e293b;border:1px solid #334155;border-radius:6px;padding:5px 12px;text-align:center">
                  <div style="font-size:14px;font-weight:700;color:#f59e0b">{c.get("escalations", 0)}</div>
                  <div style="font-size:9px;color:#64748b">escalations to external</div>
                </div>
                <div style="background:#1e293b;border:1px solid #334155;border-radius:6px;padding:5px 12px;text-align:center">
                  <div style="font-size:14px;font-weight:700;color:#10b981">{c.get("trust_score", 0):.2f}</div>
                  <div style="font-size:9px;color:#64748b">trust score</div>
                </div>
              </div>
            </div>''')

    # Close wrapper
    sections.append("</div>")

    return "\n".join(sections)


# -------------------------------------------------------------------
#  GRADIO APP -- compact layout for mass scale
# -------------------------------------------------------------------
css = """
.gradio-container { background: #0f172a !important; max-width: 100% !important; }
footer { display: none !important; }
.dark { background: #0f172a !important; }
"""

with gr.Blocks(css=css, theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate"), title="Synapse Command Center") as demo:

    with gr.Tab("Dashboard"):
        dashboard_html = gr.HTML()
        with gr.Row():
            prompt_box = gr.Textbox(
                placeholder="Enter a prompt for the swarm...",
                label="", show_label=False, scale=5, lines=1,
                container=False
            )
            doc_upload = gr.File(
                label="", file_types=[".txt", ".md", ".py", ".json", ".csv", ".html", ".log", ".yaml", ".yml", ".toml", ".cfg", ".xml", ".rst", ".tex"],
                scale=0, min_width=40, visible=True, container=False,
            )
            submit_btn = gr.Button("Send", variant="primary", scale=0, min_width=60)
            new_conv_btn = gr.Button("New", variant="secondary", scale=0, min_width=50, size="sm")
        status_msg = gr.HTML()
        conversation_html = gr.HTML()

        conv_timer = gr.Timer(value=15)
        conv_timer.tick(fn=refresh_conversation_only, outputs=[conversation_html])

        refresh_btn = gr.Button("Refresh", size="sm", scale=0, visible=False)
        submit_btn.click(fn=submit_prompt, inputs=[prompt_box, doc_upload], outputs=[status_msg, conversation_html])
        prompt_box.submit(fn=submit_prompt, inputs=[prompt_box, doc_upload], outputs=[status_msg, conversation_html])
        new_conv_btn.click(fn=new_conversation, outputs=[status_msg, conversation_html, prompt_box])
        demo.load(fn=refresh_dashboard, outputs=[dashboard_html, conversation_html])

        # Periodic full dashboard refresh (topology + trust) every 60s
        dash_timer = gr.Timer(value=60)
        dash_timer.tick(fn=refresh_dashboard, outputs=[dashboard_html, conversation_html])

    with gr.Tab("Sentinel"):
        sentinel_html = gr.HTML()
        sentinel_timer = gr.Timer(value=30)
        sentinel_timer.tick(fn=build_sentinel_mesh_html, outputs=[sentinel_html])
        demo.load(fn=build_sentinel_mesh_html, outputs=[sentinel_html])

    with gr.Tab("Analytics"):
        analytics_html = gr.HTML()
        analytics_timer = gr.Timer(value=30)
        analytics_timer.tick(fn=build_analytics_html, outputs=[analytics_html])
        demo.load(fn=build_analytics_html, outputs=[analytics_html])

    with gr.Tab("Map"):
        map_html = gr.HTML()
        map_timer = gr.Timer(value=60)
        map_timer.tick(fn=build_globe_map, outputs=[map_html])
        demo.load(fn=build_globe_map, outputs=[map_html])

    with gr.Tab("Library"):
        lib_status = gr.HTML()
        task_dropdown = gr.Dropdown(label="Conversation", interactive=True)
        archived_html = gr.HTML()
        lib_refresh_btn = gr.Button("Refresh", size="sm", scale=0)
        lib_refresh_btn.click(fn=refresh_library, outputs=[lib_status, task_dropdown])
        task_dropdown.change(fn=view_library_task, inputs=[task_dropdown], outputs=[archived_html])
        demo.load(fn=refresh_library, outputs=[lib_status, task_dropdown])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)
