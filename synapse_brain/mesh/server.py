"""Lightweight HTTP server for gossip endpoints.

Provides two endpoints:
- POST /gossip/digest -- receive a state digest, respond with missing deltas
- POST /gossip/pull   -- respond to a full pull request
- GET  /health        -- liveness probe (keeps free-tier services awake)
- GET  /metrics       -- operational metrics
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from synapse_brain.mesh.gossip import GossipProtocol

logger = logging.getLogger(__name__)


class GossipServer:
    """Minimal async HTTP server for gossip protocol.

    Uses raw asyncio to avoid heavy web framework dependencies.
    Keeps the spore footprint minimal for free-tier deployment.
    """

    def __init__(self, gossip: GossipProtocol, get_snapshot=None):
        self.gossip = gossip
        self._get_snapshot = get_snapshot
        self._server = None

    async def handle_request(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle a single HTTP request."""
        try:
            request_line = await asyncio.wait_for(reader.readline(), timeout=10)
            if not request_line:
                writer.close()
                return

            method, path, _ = request_line.decode().strip().split(" ", 2)

            # Read headers
            headers = {}
            while True:
                line = await reader.readline()
                if line == b"\r\n" or not line:
                    break
                key, val = line.decode().strip().split(": ", 1)
                headers[key.lower()] = val

            # Read body
            body = b""
            content_length = int(headers.get("content-length", 0))
            if content_length > 0:
                body = await reader.readexactly(content_length)

            # Route
            if path == "/health" and method == "GET":
                response = {"status": "alive", "spore_id": self.gossip.spore_id}
                await self._send_json(writer, 200, response)

            elif path == "/metrics" and method == "GET":
                response = self.gossip.status()
                await self._send_json(writer, 200, response)

            elif path == "/gossip/digest" and method == "POST":
                digest = json.loads(body)
                snapshot = self._get_snapshot() if self._get_snapshot else {}
                result = await self.gossip.handle_digest(digest, snapshot)
                await self._send_json(writer, 200, result)

            elif path == "/gossip/pull" and method == "POST":
                request = json.loads(body)
                snapshot = self._get_snapshot() if self._get_snapshot else {}
                known = set(request.get("known_fingerprints", []))
                deltas = []
                for fp, delta in snapshot.get("delta_store", {}).items():
                    if fp not in known:
                        deltas.append(delta)
                await self._send_json(writer, 200, {"deltas": deltas})

            else:
                await self._send_json(writer, 404, {"error": "not found"})

        except Exception as e:
            logger.debug("Request handling error: %s", e)
            try:
                await self._send_json(writer, 500, {"error": str(e)})
            except Exception:
                pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _send_json(self, writer: asyncio.StreamWriter, status: int, data: Any):
        body = json.dumps(data).encode()
        status_text = {200: "OK", 404: "Not Found", 500: "Internal Server Error"}.get(status, "OK")
        response = (
            f"HTTP/1.1 {status} {status_text}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"Connection: close\r\n"
            f"\r\n"
        ).encode() + body
        writer.write(response)
        await writer.drain()

    async def serve(self, host: str = "0.0.0.0", port: int = 8470):
        """Start serving gossip endpoints."""
        self._server = await asyncio.start_server(
            self.handle_request, host, port,
        )
        logger.info("Gossip server listening on %s:%d", host, port)
        async with self._server:
            await self._server.serve_forever()


def create_gossip_server(gossip: GossipProtocol, get_snapshot=None) -> GossipServer:
    """Factory function for creating a gossip server."""
    return GossipServer(gossip, get_snapshot)
