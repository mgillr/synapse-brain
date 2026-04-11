"""Cloudflare Worker deployer for gossip relay.

Cloudflare Workers are not suitable for running full spores (10ms CPU limit).
Instead, they serve as:
- Gossip relay: forward messages between spores
- Peer registry: maintain list of active spores in KV
- Keep-alive: ping Render/HF spores to prevent sleep
- Task router: assign incoming tasks to available spores

Free tier: 100K requests/day, KV storage 1GB.
"""

from __future__ import annotations

import textwrap
from pathlib import Path


def generate_cloudflare_relay(
    output_dir: str,
    kv_namespace: str = "SYNAPSE_PEERS",
) -> dict[str, str]:
    """Generate Cloudflare Worker relay files."""
    files = {}

    files["wrangler.toml"] = textwrap.dedent(f"""\
        name = "synapse-relay"
        main = "src/worker.js"
        compatibility_date = "2024-01-01"

        [[kv_namespaces]]
        binding = "{kv_namespace}"
        id = "REPLACE_WITH_KV_NAMESPACE_ID"
    """)

    files["src/worker.js"] = textwrap.dedent("""\
        /**
         * Synapse Brain Gossip Relay -- Cloudflare Worker
         *
         * Routes gossip messages between spores, maintains peer registry,
         * and pings sleeping services to keep them alive.
         *
         * Endpoints:
         *   POST /register   -- register a spore in the peer registry
         *   GET  /peers      -- list all known peers
         *   POST /relay      -- forward a gossip message to target peers
         *   POST /task       -- submit a task for the swarm
         *   GET  /health     -- liveness check
         */

        export default {
          async fetch(request, env) {
            const url = new URL(request.url);
            const path = url.pathname;

            try {
              if (path === '/health') {
                return new Response(JSON.stringify({ status: 'alive', role: 'relay' }), {
                  headers: { 'Content-Type': 'application/json' },
                });
              }

              if (path === '/register' && request.method === 'POST') {
                const body = await request.json();
                const { spore_id, url: sporeUrl } = body;

                if (!spore_id || !sporeUrl) {
                  return new Response(JSON.stringify({ error: 'missing spore_id or url' }), { status: 400 });
                }

                // Store peer in KV with 10 minute TTL
                await env.SYNAPSE_PEERS.put(spore_id, JSON.stringify({
                  spore_id,
                  url: sporeUrl,
                  registered: Date.now(),
                }), { expirationTtl: 600 });

                return new Response(JSON.stringify({ registered: true }));
              }

              if (path === '/peers') {
                // List all known peers
                const list = await env.SYNAPSE_PEERS.list();
                const peers = [];
                for (const key of list.keys) {
                  const val = await env.SYNAPSE_PEERS.get(key.name);
                  if (val) peers.push(JSON.parse(val));
                }
                return new Response(JSON.stringify({ peers, count: peers.length }), {
                  headers: { 'Content-Type': 'application/json' },
                });
              }

              if (path === '/relay' && request.method === 'POST') {
                const body = await request.json();
                const { targets, message } = body;

                if (!targets || !message) {
                  return new Response(JSON.stringify({ error: 'missing targets or message' }), { status: 400 });
                }

                // Forward message to each target
                const results = await Promise.allSettled(
                  targets.map(async (targetId) => {
                    const peerData = await env.SYNAPSE_PEERS.get(targetId);
                    if (!peerData) return { target: targetId, status: 'unknown' };

                    const peer = JSON.parse(peerData);
                    const resp = await fetch(`${peer.url}/gossip/digest`, {
                      method: 'POST',
                      headers: { 'Content-Type': 'application/json' },
                      body: JSON.stringify(message),
                    });

                    return { target: targetId, status: resp.status };
                  })
                );

                return new Response(JSON.stringify({
                  relayed: results.map(r => r.status === 'fulfilled' ? r.value : { error: r.reason }),
                }));
              }

              if (path === '/task' && request.method === 'POST') {
                const body = await request.json();
                // Find a random healthy peer and forward the task
                const list = await env.SYNAPSE_PEERS.list();
                if (list.keys.length === 0) {
                  return new Response(JSON.stringify({ error: 'no peers available' }), { status: 503 });
                }

                const randomKey = list.keys[Math.floor(Math.random() * list.keys.length)];
                const peerData = await env.SYNAPSE_PEERS.get(randomKey.name);
                const peer = JSON.parse(peerData);

                const resp = await fetch(`${peer.url}/task`, {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify(body),
                });

                const result = await resp.json();
                return new Response(JSON.stringify({
                  assigned_to: peer.spore_id,
                  result,
                }));
              }

              return new Response('Not Found', { status: 404 });

            } catch (err) {
              return new Response(JSON.stringify({ error: err.message }), { status: 500 });
            }
          },

          // Scheduled handler: ping sleeping services every 10 minutes
          async scheduled(event, env) {
            const list = await env.SYNAPSE_PEERS.list();
            for (const key of list.keys) {
              const val = await env.SYNAPSE_PEERS.get(key.name);
              if (!val) continue;
              const peer = JSON.parse(val);
              try {
                await fetch(`${peer.url}/health`, { method: 'GET' });
              } catch (e) {
                // Peer unreachable, will expire from KV naturally
              }
            }
          },
        };
    """)

    files["package.json"] = textwrap.dedent("""\
        {
          "name": "synapse-relay",
          "version": "0.1.0",
          "private": true,
          "scripts": {
            "dev": "wrangler dev",
            "deploy": "wrangler deploy"
          },
          "devDependencies": {
            "wrangler": "^3.0.0"
          }
        }
    """)

    if output_dir:
        out = Path(output_dir)
        (out / "src").mkdir(parents=True, exist_ok=True)
        for fname, content in files.items():
            fpath = out / fname
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_text(content)

    return files
