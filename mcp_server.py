"""Synapse MCP Server -- Model Context Protocol endpoint for every spore.

Exposes spore capabilities as MCP tools that any MCP-compatible client
(Claude, Cursor, Windsurf, custom agents) can discover and invoke.

Transport: Streamable HTTP on /mcp (POST for requests, GET+SSE for streaming)
Protocol: MCP 2025-03-26 (latest stable)

Tools exposed:
  - submit_task: Submit a reasoning task to the swarm
  - query_memory: Semantic search over CRDT memory
  - get_trust: Query trust lattice for any peer
  - swarm_health: Full swarm health snapshot
  - get_task: Get detailed task state including deltas and convergence
  - list_tasks: List all tasks with status
  - collective_knowledge: Query the collective intelligence layer

Non-breaking: MCP server is an additional FastAPI route. If MCP library
is unavailable, the route simply returns 501.
"""
import json
import logging
import time
import hashlib
from typing import Any

log = logging.getLogger("mcp_server")

# MCP protocol constants
MCP_VERSION = "2025-03-26"
JSONRPC_VERSION = "2.0"


class MCPToolDefinition:
    """MCP tool definition with JSON Schema parameters."""

    def __init__(self, name: str, description: str,
                 input_schema: dict, handler):
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.handler = handler

    def to_mcp(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


class SynapseMCPServer:
    """MCP server implementation for Synapse spores.

    Handles the MCP JSON-RPC protocol over HTTP:
    - initialize: capability negotiation
    - tools/list: enumerate available tools
    - tools/call: execute a tool
    - ping: health check

    Mounted as a FastAPI route at /mcp on every spore.
    """

    def __init__(self, spore_id: str, role: str, model: str):
        self.spore_id = spore_id
        self.role = role
        self.model = model
        self.tools: dict[str, MCPToolDefinition] = {}
        self.sessions: dict[str, dict] = {}
        self._initialized = False

        # Bind points set by spore.py after construction
        self.memory = None
        self.dual_memory = None
        self.trust_store = None
        self.spore_state = None
        self.cortex = None

    def register_tool(self, name: str, description: str,
                      input_schema: dict, handler):
        self.tools[name] = MCPToolDefinition(
            name, description, input_schema, handler
        )

    def bind(self, memory=None, dual_memory=None, trust_store=None,
             spore_state=None, cortex=None):
        """Bind spore components after initialization."""
        if memory:
            self.memory = memory
        if dual_memory:
            self.dual_memory = dual_memory
        if trust_store:
            self.trust_store = trust_store
        if spore_state:
            self.spore_state = spore_state
        if cortex:
            self.cortex = cortex
        self._register_default_tools()

    def _register_default_tools(self):
        """Register all default spore tools."""

        # --- submit_task ---
        def handle_submit_task(description: str = "",
                               priority: str = "normal") -> dict:
            if not self.spore_state or not description:
                return {"error": "No task description provided"}
            tid = hashlib.sha256(
                f"{description}:{time.time()}".encode()
            ).hexdigest()[:16]
            task = self.spore_state.get_or_create_task(tid, description)
            return {
                "task_id": tid,
                "status": "submitted",
                "description": description,
                "spore": self.spore_id,
            }

        self.register_tool(
            "submit_task",
            "Submit a reasoning task to the Synapse swarm. The task will be "
            "debated by all spores using structured cognitive protocols until "
            "convergence is reached.",
            {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "The task or question to reason about",
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["low", "normal", "high"],
                        "description": "Task priority level",
                    },
                },
                "required": ["description"],
            },
            handle_submit_task,
        )

        # --- query_memory ---
        def handle_query_memory(query: str = "",
                                top_k: int = 5,
                                layer: str = "private") -> dict:
            if not query:
                return {"error": "No query provided"}
            if layer == "collective" and self.dual_memory:
                records = list(
                    self.dual_memory._collective_records.values()
                )[-top_k:]
                return {
                    "layer": "collective",
                    "results": records,
                    "total_collective": (
                        self.dual_memory.collective_size
                    ),
                }
            if self.memory:
                results = self.memory.recall(query, top_k)
                return {
                    "layer": "private",
                    "results": results,
                    "total_memories": self.memory.size,
                }
            return {"error": "Memory not available"}

        self.register_tool(
            "query_memory",
            "Semantic search over the spore's CRDT-backed memory. Query the "
            "private layer (local knowledge) or collective layer (swarm-wide "
            "distilled intelligence).",
            {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for semantic matching",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 5,
                    },
                    "layer": {
                        "type": "string",
                        "enum": ["private", "collective"],
                        "description": "Memory layer to search",
                        "default": "private",
                    },
                },
                "required": ["query"],
            },
            handle_query_memory,
        )

        # --- get_trust ---
        def handle_get_trust(peer_id: str = "") -> dict:
            if not self.trust_store:
                return {"error": "Trust store not available"}
            if peer_id:
                return {
                    "peer_id": peer_id,
                    "trust": self.trust_store.get(peer_id),
                }
            return {"all_trust": self.trust_store.get_all()}

        self.register_tool(
            "get_trust",
            "Query the E4 trust lattice. Get trust score for a specific peer "
            "or the full trust state across the swarm.",
            {
                "type": "object",
                "properties": {
                    "peer_id": {
                        "type": "string",
                        "description": "Peer ID to query (empty for all)",
                        "default": "",
                    },
                },
            },
            handle_get_trust,
        )

        # --- swarm_health ---
        def handle_swarm_health() -> dict:
            if not self.spore_state:
                return {"error": "Spore state not available"}
            return {
                "spore": self.spore_id,
                "role": self.role,
                "model": self.model,
                "cycles": self.spore_state.reasoning_cycles,
                "deltas_produced": self.spore_state.deltas_produced,
                "deltas_received": self.spore_state.deltas_received,
                "memories": self.memory.size if self.memory else 0,
                "collective_memories": (
                    self.dual_memory.collective_size
                    if self.dual_memory else 0
                ),
                "peers_seen": list(self.spore_state.peers_seen),
                "cortex": self.cortex.stats() if self.cortex else None,
            }

        self.register_tool(
            "swarm_health",
            "Get comprehensive health status of this spore including "
            "reasoning cycles, delta throughput, memory size, peer "
            "connectivity, and Cortex status.",
            {"type": "object", "properties": {}},
            handle_swarm_health,
        )

        # --- get_task ---
        def handle_get_task(task_id: str = "") -> dict:
            if not self.spore_state or not task_id:
                return {"error": "No task_id provided"}
            task = self.spore_state.tasks.get(task_id)
            if not task:
                return {"error": f"Task {task_id} not found"}
            return {
                "task_id": task_id,
                "description": task.description,
                "converged": task.converged,
                "final_answer": task.final_answer,
                "delta_count": len(task.deltas),
                "contributors": task.contributors(),
                "deltas": task.deltas[-10:],
            }

        self.register_tool(
            "get_task",
            "Get detailed state of a specific reasoning task including "
            "convergence status, final answer, and recent reasoning deltas.",
            {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The task ID to query",
                    },
                },
                "required": ["task_id"],
            },
            handle_get_task,
        )

        # --- list_tasks ---
        def handle_list_tasks(limit: int = 20) -> dict:
            if not self.spore_state:
                return {"error": "Spore state not available"}
            tasks = []
            for tid, task in list(self.spore_state.tasks.items())[-limit:]:
                tasks.append({
                    "task_id": tid,
                    "description": task.description[:200],
                    "converged": task.converged,
                    "has_answer": bool(task.final_answer),
                    "delta_count": len(task.deltas),
                })
            return {"tasks": tasks, "total": len(self.spore_state.tasks)}

        self.register_tool(
            "list_tasks",
            "List all reasoning tasks with their convergence status.",
            {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Max tasks to return",
                        "default": 20,
                    },
                },
            },
            handle_list_tasks,
        )

        # --- collective_knowledge ---
        def handle_collective(query: str = "", limit: int = 10) -> dict:
            if not self.dual_memory:
                return {"error": "Dual memory not initialized"}
            records = list(
                self.dual_memory._collective_records.values()
            )
            if query:
                # Simple keyword match on collective
                lower_q = query.lower()
                records = [
                    r for r in records
                    if lower_q in r.get("content", "").lower()
                ]
            return {
                "results": records[-limit:],
                "total_collective": self.dual_memory.collective_size,
                "wall_stats": self.dual_memory.wall.stats(),
            }

        self.register_tool(
            "collective_knowledge",
            "Query the collective intelligence layer -- distilled insights "
            "from across the entire swarm. This is privacy-safe knowledge "
            "that has passed through the Knowledge Wall.",
            {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Optional keyword filter",
                        "default": "",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results",
                        "default": 10,
                    },
                },
            },
            handle_collective,
        )

        log.info("MCP server: %d tools registered", len(self.tools))

    async def handle_request(self, body: dict) -> dict:
        """Process a single MCP JSON-RPC request."""
        method = body.get("method", "")
        req_id = body.get("id")
        params = body.get("params", {})

        if method == "initialize":
            return self._handle_initialize(req_id, params)
        elif method == "ping":
            return self._success(req_id, {})
        elif method == "tools/list":
            return self._handle_tools_list(req_id)
        elif method == "tools/call":
            return await self._handle_tools_call(req_id, params)
        elif method == "notifications/initialized":
            return None  # Client notification, no response
        else:
            return self._error(req_id, -32601, f"Method not found: {method}")

    def _handle_initialize(self, req_id, params: dict) -> dict:
        session_id = hashlib.sha256(
            f"{time.time()}:{self.spore_id}".encode()
        ).hexdigest()[:16]
        self.sessions[session_id] = {
            "created": time.time(),
            "client_info": params.get("clientInfo", {}),
        }
        return self._success(req_id, {
            "protocolVersion": MCP_VERSION,
            "capabilities": {
                "tools": {"listChanged": False},
            },
            "serverInfo": {
                "name": f"synapse-{self.spore_id}",
                "version": "5.1.0",
            },
            "instructions": (
                f"Synapse Brain spore {self.spore_id} ({self.role}). "
                f"Model: {self.model}. CRDT-backed persistent memory. "
                f"Submit tasks for multi-agent reasoning, query collective "
                f"intelligence, inspect trust lattice."
            ),
        })

    def _handle_tools_list(self, req_id) -> dict:
        tools = [t.to_mcp() for t in self.tools.values()]
        return self._success(req_id, {"tools": tools})

    async def _handle_tools_call(self, req_id, params: dict) -> dict:
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        tool = self.tools.get(tool_name)
        if not tool:
            return self._error(
                req_id, -32602,
                f"Unknown tool: {tool_name}. "
                f"Available: {list(self.tools.keys())}"
            )

        try:
            result = tool.handler(**arguments)
            text = json.dumps(result, default=str, indent=2)
            return self._success(req_id, {
                "content": [{"type": "text", "text": text}],
            })
        except Exception as e:
            return self._success(req_id, {
                "content": [{"type": "text", "text": f"Error: {e}"}],
                "isError": True,
            })

    def _success(self, req_id, result: dict) -> dict:
        return {
            "jsonrpc": JSONRPC_VERSION,
            "id": req_id,
            "result": result,
        }

    def _error(self, req_id, code: int, message: str) -> dict:
        return {
            "jsonrpc": JSONRPC_VERSION,
            "id": req_id,
            "error": {"code": code, "message": message},
        }


def mount_mcp_routes(fastapi_app, mcp_server: SynapseMCPServer):
    """Mount MCP endpoint onto existing FastAPI app.

    Route: POST /mcp -- handles all MCP JSON-RPC requests
    Route: GET /mcp -- SSE transport (stubbed for future streaming)

    Non-breaking: adds routes without affecting existing endpoints.
    """
    from starlette.requests import Request
    from starlette.responses import JSONResponse

    @fastapi_app.post("/mcp")
    async def mcp_endpoint(request: Request):
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                {"jsonrpc": "2.0", "id": None,
                 "error": {"code": -32700, "message": "Parse error"}},
                status_code=400,
            )

        # Handle batch requests
        if isinstance(body, list):
            responses = []
            for item in body:
                resp = await mcp_server.handle_request(item)
                if resp is not None:
                    responses.append(resp)
            return JSONResponse(responses if responses else {"ok": True})

        response = await mcp_server.handle_request(body)
        if response is None:
            return JSONResponse({"ok": True}, status_code=202)
        return JSONResponse(response)

    @fastapi_app.get("/mcp")
    async def mcp_sse():
        """SSE endpoint placeholder for future streaming support."""
        return JSONResponse(
            {"error": "SSE transport not yet implemented. Use POST /mcp."},
            status_code=501,
        )

    @fastapi_app.get("/mcp/info")
    async def mcp_info():
        """Public info about this MCP server."""
        return JSONResponse({
            "spore": mcp_server.spore_id,
            "role": mcp_server.role,
            "model": mcp_server.model,
            "protocol_version": MCP_VERSION,
            "tools": [t.name for t in mcp_server.tools.values()],
            "tool_count": len(mcp_server.tools),
        })

    log.info("MCP routes mounted at /mcp (%d tools)", len(mcp_server.tools))
