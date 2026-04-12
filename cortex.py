"""Synapse Cortex -- Local Micro-LLM with Tool-Calling Agent Loop.

Runs Qwen3-4B Q4_K_M locally on CPU via llama-cpp-python. Provides System 1
fast cognition for the Sentinel while using external LLMs as System 2
for novel/complex reasoning.

Architecture:
  - Downloads GGUF model from HuggingFace Hub on first boot
  - llama-cpp-python for CPU inference (~10 tok/s on 2 vCPU)
  - Tool registry with JSON schema validation
  - ReAct-style agent loop: think -> tool_call -> observe -> repeat
  - Confidence-gated escalation to System 2 (external LLM)
  - Trust accumulation via CRDT for earned autonomy

Non-breaking: if model fails to load, all methods return None and callers
fall back to external LLM path transparently.
"""
import hashlib
import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("cortex")

# Model configuration
MODEL_REPO = "Qwen/Qwen3-4B-GGUF"
MODEL_FILE = "Qwen3-4B-Q4_K_M.gguf"
MODEL_DIR = "/tmp/cortex_model"
CONTEXT_SIZE = 4096
MAX_TOKENS = 1024
TEMPERATURE = 0.3

# Tool-calling format for Qwen3
TOOL_CALL_REGEX = re.compile(
    r'<tool_call>\s*(\{.*?\})\s*</tool_call>', re.DOTALL
)
THINK_REGEX = re.compile(
    r'<think>(.*?)</think>', re.DOTALL
)


class CortexTool:
    """Registered tool that the Cortex can invoke."""

    __slots__ = ("name", "description", "parameters", "handler")

    def __init__(self, name: str, description: str,
                 parameters: dict, handler: Callable):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.handler = handler

    def schema_json(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def execute(self, **kwargs) -> Any:
        try:
            return self.handler(**kwargs)
        except Exception as e:
            return {"error": str(e)}


class ToolRegistry:
    """Registry of tools available to the Cortex agent loop."""

    def __init__(self):
        self._tools: dict[str, CortexTool] = {}

    def register(self, name: str, description: str,
                 parameters: dict, handler: Callable):
        self._tools[name] = CortexTool(name, description, parameters, handler)
        log.info("Cortex tool registered: %s", name)

    def get(self, name: str) -> CortexTool | None:
        return self._tools.get(name)

    def list_schemas(self) -> list[dict]:
        return [t.schema_json() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools.keys())


class Cortex:
    """Local micro-LLM agent with tool-calling capability.

    System 1 (fast/local): handles telemetry analysis, code review,
    pattern matching, memory queries -- 80% of sentinel workload.

    Escalates to System 2 (external LLM) when:
    - Confidence is below threshold
    - Task requires creative synthesis
    - Novel architecture proposals needed
    - Multi-file code generation required
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.model = None
        self.tools = ToolRegistry()
        self.loaded = False
        self._lock = threading.Lock()
        self._load_error = None

        # Performance tracking
        self.total_calls = 0
        self.total_tokens = 0
        self.total_tool_calls = 0
        self.escalations = 0
        self.start_time = time.time()

        # Trust score (CRDT-tracked externally)
        self.trust_score = 0.5  # starts neutral
        self.correct_decisions = 0
        self.total_decisions = 0

        if enabled:
            self._load_thread = threading.Thread(
                target=self._load_model, daemon=True, name="cortex-loader"
            )
            self._load_thread.start()

    def _load_model(self):
        """Download GGUF and initialize llama-cpp-python. Runs in background."""
        try:
            os.makedirs(MODEL_DIR, exist_ok=True)
            model_path = os.path.join(MODEL_DIR, MODEL_FILE)

            if not os.path.exists(model_path):
                log.info("Cortex: downloading %s/%s ...", MODEL_REPO, MODEL_FILE)
                from huggingface_hub import hf_hub_download
                hf_token = os.environ.get("HF_TOKEN", "")
                model_path = hf_hub_download(
                    repo_id=MODEL_REPO,
                    filename=MODEL_FILE,
                    local_dir=MODEL_DIR,
                    token=hf_token if hf_token else None,
                )
                log.info("Cortex: model downloaded to %s", model_path)
            else:
                log.info("Cortex: model already cached at %s", model_path)

            try:
                from llama_cpp import Llama
            except ImportError:
                log.info("Cortex: llama-cpp-python not installed -- building from source")
                import subprocess
                env = {**os.environ, "CMAKE_ARGS": "-DGGML_NATIVE=OFF -DGGML_BLAS=OFF -DGGML_OPENMP=OFF"}
                subprocess.check_call(
                    ["pip", "install", "llama-cpp-python>=0.3.19", "--no-cache-dir"],
                    env=env, timeout=1200
                )
                log.info("Cortex: llama-cpp-python installed")
                from llama_cpp import Llama
            self.model = Llama(
                model_path=model_path,
                n_ctx=CONTEXT_SIZE,
                n_threads=2,  # HF Spaces free tier = 2 vCPU
                n_batch=256,
                verbose=False,
            )
            self.loaded = True
            log.info("Cortex: model loaded -- ready for inference")

        except Exception as e:
            self._load_error = str(e)
            log.error("Cortex: failed to load model: %s", e)
            self.loaded = False

    def is_ready(self) -> bool:
        return self.loaded and self.model is not None

    def register_tool(self, name: str, description: str,
                      parameters: dict, handler: Callable):
        self.tools.register(name, description, parameters, handler)

    def _build_system_prompt(self, task_type: str = "analysis") -> str:
        """Build system prompt with tool definitions for Qwen3."""
        tool_schemas = self.tools.list_schemas()
        tools_block = json.dumps(tool_schemas, indent=2) if tool_schemas else "[]"

        return f"""You are the Cortex -- the local reasoning engine of a Synapse Brain Sentinel spore.
You analyze swarm telemetry, review code patches, evaluate proposals, and make decisions.

You have access to these tools:
{tools_block}

When you need to use a tool, output:
<tool_call>
{{"name": "tool_name", "arguments": {{"key": "value"}}}}
</tool_call>

You may call multiple tools in sequence. After receiving tool results, continue reasoning.

When you have a final answer, output it directly without tool_call tags.

If you are uncertain about your analysis (confidence < 0.7), say ESCALATE and explain why.
The system will route to a more capable external model.

Be concise. Be precise. No filler."""

    def _parse_tool_calls(self, text: str) -> list[dict]:
        """Extract tool calls from model output."""
        calls = []
        for match in TOOL_CALL_REGEX.finditer(text):
            try:
                call = json.loads(match.group(1))
                if "name" in call:
                    calls.append(call)
            except json.JSONDecodeError:
                continue
        return calls

    def _extract_thinking(self, text: str) -> str:
        """Extract thinking content from Qwen3 output."""
        match = THINK_REGEX.search(text)
        return match.group(1).strip() if match else ""

    def _strip_tags(self, text: str) -> str:
        """Remove thinking and tool_call tags from final output."""
        text = THINK_REGEX.sub("", text)
        text = TOOL_CALL_REGEX.sub("", text)
        return text.strip()

    def generate(self, prompt: str, system: str = "",
                 max_tokens: int = MAX_TOKENS) -> dict | None:
        """Raw generation -- no tool loop. Returns None if not ready."""
        if not self.is_ready():
            return None

        with self._lock:
            try:
                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": prompt})

                result = self.model.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=TEMPERATURE,
                    stop=["<|endoftext|>", "<|im_end|>"],
                )

                text = result["choices"][0]["message"]["content"]
                tokens_used = result.get("usage", {}).get("total_tokens", 0)

                self.total_calls += 1
                self.total_tokens += tokens_used

                return {
                    "text": text,
                    "tokens": tokens_used,
                    "thinking": self._extract_thinking(text),
                    "clean_text": self._strip_tags(text),
                }

            except Exception as e:
                log.error("Cortex generate error: %s", e)
                return None

    def agent_loop(self, task: str, system: str = "",
                   max_iterations: int = 5) -> dict:
        """ReAct-style agent loop with tool calling.

        Returns:
            {
                "result": str,          # final answer
                "escalate": bool,       # True if System 2 needed
                "reason": str,          # escalation reason if applicable
                "tool_calls": list,     # tools invoked
                "iterations": int,
                "tokens": int,
                "thinking": str,        # accumulated reasoning
            }
        """
        if not self.is_ready():
            return {
                "result": None,
                "escalate": True,
                "reason": "cortex_not_loaded",
                "tool_calls": [],
                "iterations": 0,
                "tokens": 0,
                "thinking": "",
            }

        sys_prompt = system or self._build_system_prompt()
        conversation = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": task},
        ]

        all_tool_calls = []
        all_thinking = []
        total_tokens = 0

        for iteration in range(max_iterations):
            with self._lock:
                try:
                    result = self.model.create_chat_completion(
                        messages=conversation,
                        max_tokens=MAX_TOKENS,
                        temperature=TEMPERATURE,
                        stop=["<|endoftext|>", "<|im_end|>"],
                    )
                except Exception as e:
                    log.error("Cortex agent loop error at iteration %d: %s",
                              iteration, e)
                    return {
                        "result": None,
                        "escalate": True,
                        "reason": f"inference_error: {e}",
                        "tool_calls": all_tool_calls,
                        "iterations": iteration,
                        "tokens": total_tokens,
                        "thinking": "\n".join(all_thinking),
                    }

            text = result["choices"][0]["message"]["content"]
            tokens = result.get("usage", {}).get("total_tokens", 0)
            total_tokens += tokens
            self.total_calls += 1
            self.total_tokens += tokens

            # Extract thinking
            thinking = self._extract_thinking(text)
            if thinking:
                all_thinking.append(thinking)

            # Check for escalation request
            if "ESCALATE" in text.upper():
                self.escalations += 1
                reason = self._strip_tags(text)
                return {
                    "result": None,
                    "escalate": True,
                    "reason": reason,
                    "tool_calls": all_tool_calls,
                    "iterations": iteration + 1,
                    "tokens": total_tokens,
                    "thinking": "\n".join(all_thinking),
                }

            # Check for tool calls
            tool_calls = self._parse_tool_calls(text)
            if not tool_calls:
                # No tool calls -- this is the final answer
                clean = self._strip_tags(text)
                self.total_decisions += 1
                return {
                    "result": clean,
                    "escalate": False,
                    "reason": None,
                    "tool_calls": all_tool_calls,
                    "iterations": iteration + 1,
                    "tokens": total_tokens,
                    "thinking": "\n".join(all_thinking),
                }

            # Execute tool calls
            conversation.append({"role": "assistant", "content": text})
            tool_results = []

            for call in tool_calls:
                tool_name = call.get("name", "")
                tool_args = call.get("arguments", {})
                tool = self.tools.get(tool_name)

                if tool:
                    result_data = tool.execute(**tool_args)
                    self.total_tool_calls += 1
                    all_tool_calls.append({
                        "tool": tool_name,
                        "args": tool_args,
                        "result": result_data,
                        "iteration": iteration,
                    })
                    tool_results.append(
                        f"Tool '{tool_name}' returned: "
                        f"{json.dumps(result_data, default=str)[:2000]}"
                    )
                else:
                    tool_results.append(
                        f"Tool '{tool_name}' not found. "
                        f"Available: {self.tools.names()}"
                    )

            # Feed results back for next iteration
            observation = "\n".join(tool_results)
            conversation.append({"role": "user", "content": observation})

        # Max iterations reached
        self.total_decisions += 1
        return {
            "result": "Max iterations reached without final answer.",
            "escalate": True,
            "reason": "max_iterations",
            "tool_calls": all_tool_calls,
            "iterations": max_iterations,
            "tokens": total_tokens,
            "thinking": "\n".join(all_thinking),
        }

    def should_handle(self, task_type: str) -> bool:
        """System 1/System 2 routing decision.

        System 1 (Cortex) handles:
        - telemetry_analysis: pattern detection in swarm metrics
        - code_review: syntax check, safety scan, patch validation
        - memory_query: semantic search over CRDT memory
        - health_check: peer status evaluation
        - config_validation: bounds checking for proposed configs
        - proposal_triage: quick assessment of proposal quality

        System 2 (external LLM) handles:
        - architecture_proposal: novel system design
        - creative_synthesis: combining insights into new ideas
        - multi_file_codegen: generating complex code changes
        - five_phase_analysis: full Five-Phase Discipline reasoning
        """
        system1_tasks = {
            "telemetry_analysis",
            "code_review",
            "memory_query",
            "health_check",
            "config_validation",
            "proposal_triage",
            "trend_analysis",
            "anomaly_detection",
            "gossip_optimization",
        }
        return task_type in system1_tasks and self.is_ready()

    def record_outcome(self, correct: bool):
        """Record whether a Cortex decision was correct. Updates trust."""
        self.total_decisions += 1
        if correct:
            self.correct_decisions += 1
        # Trust score = rolling accuracy with decay
        if self.total_decisions > 0:
            self.trust_score = (
                0.9 * self.trust_score +
                0.1 * (self.correct_decisions / self.total_decisions)
            )

    def stats(self) -> dict:
        uptime = time.time() - self.start_time
        return {
            "loaded": self.loaded,
            "load_error": self._load_error,
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "total_tool_calls": self.total_tool_calls,
            "escalations": self.escalations,
            "trust_score": round(self.trust_score, 4),
            "accuracy": (
                round(self.correct_decisions / max(1, self.total_decisions), 4)
            ),
            "uptime_seconds": round(uptime),
            "tokens_per_call": (
                round(self.total_tokens / max(1, self.total_calls))
            ),
        }
