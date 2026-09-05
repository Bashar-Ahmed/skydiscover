"""Claude Code CLI LLM backend.

Drives the locally installed ``claude`` binary in non-interactive print mode
(``claude -p``) instead of calling an HTTP API. This lets SkyDiscover run
against a Claude subscription (Pro/Max) rather than a metered API key: the CLI
reuses whatever credentials ``claude auth`` already established.

The CLI is invoked as a plain text generator, not as a coding agent:
``--tools ""`` disables every built-in tool, ``--max-turns 1`` forbids
multi-turn tool loops, and settings/MCP/slash-command discovery is switched off
so a run is not perturbed by whatever is configured on the machine. Nothing is
containerized — the binary runs directly on this host.

Model selection uses the ``claude_cli/`` provider prefix::

    llm:
      models:
        - name: "claude_cli/sonnet"     # or opus, haiku, or a full model id
          weight: 1.0

Usage limits are handled by :mod:`skydiscover.llm.rate_limit`: a rejected call
parks every caller until the quota resets instead of failing the iteration.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
import threading
from typing import Any, Dict, List, Optional, Tuple

from skydiscover.config import LLMModelConfig
from skydiscover.llm.base import LLMInterface, LLMResponse
from skydiscover.llm.call_log import GLOBAL_CALL_LOG
from skydiscover.llm.rate_limit import (
    UsageLimitError,
    get_usage_limit_gate,
    looks_like_usage_limit,
    parse_usage_limit,
)

logger = logging.getLogger("skydiscover.llm")

DEFAULT_CLI_BINARY = "claude"

# Effort levels the CLI accepts, weakest first.
CLI_EFFORT_LEVELS: Tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")

# OpenAI-style reasoning_effort values that have no direct CLI equivalent.
_EFFORT_ALIASES = {
    "minimal": "low",
    "none": "low",
    "default": "medium",
    "standard": "medium",
    "xhigh": "xhigh",
    "very_high": "xhigh",
    "maximum": "max",
}

# Refuse to loop forever if a quota never clears.
DEFAULT_MAX_USAGE_LIMIT_WAITS = 24


def normalize_effort(effort: Optional[str]) -> Optional[str]:
    """Coerce an effort string to one the CLI accepts, or None."""
    if not effort:
        return None
    value = str(effort).strip().lower()
    value = _EFFORT_ALIASES.get(value, value)
    if value in CLI_EFFORT_LEVELS:
        return value
    logger.warning(
        "Unknown reasoning effort %r for the Claude CLI; expected one of %s. Ignoring it.",
        effort,
        ", ".join(CLI_EFFORT_LEVELS),
    )
    return None


def _downgrade_effort(effort: Optional[str]) -> Optional[str]:
    """One rung down the effort ladder, or None if already lowest/unknown."""
    if effort not in CLI_EFFORT_LEVELS:
        return None
    idx = CLI_EFFORT_LEVELS.index(effort)
    if idx == 0:
        return None
    return CLI_EFFORT_LEVELS[idx - 1]


class CostTracker:
    """Process-wide tally of what the CLI reports spending.

    The rest of the framework does no cost accounting at all; the CLI hands us
    ``total_cost_usd`` per call for free, so it is worth keeping. On a
    subscription plan this is an equivalent-value figure, not a real charge.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.total_usd = 0.0
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self._next_log_at = 1.0

    def record(self, payload: Dict[str, Any]) -> None:
        usage = payload.get("usage") or {}
        with self._lock:
            self.calls += 1
            self.total_usd += float(payload.get("total_cost_usd") or 0.0)
            self.input_tokens += int(usage.get("input_tokens") or 0)
            self.output_tokens += int(usage.get("output_tokens") or 0)
            should_log = self.total_usd >= self._next_log_at
            if should_log:
                self._next_log_at = self.total_usd + 1.0
            snapshot = (self.calls, self.total_usd, self.input_tokens, self.output_tokens)

        if should_log:
            calls, total, tin, tout = snapshot
            logger.info(
                "Claude CLI usage so far: %d calls, ~$%.2f equivalent, %d in / %d out tokens.",
                calls,
                total,
                tin,
                tout,
            )

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "calls": self.calls,
                "total_cost_usd": round(self.total_usd, 6),
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
            }


GLOBAL_COST_TRACKER = CostTracker()


class ClaudeCLILLM(LLMInterface):
    """LLM backend that shells out to the local ``claude`` CLI."""

    # The CLI is itself an agent, so agentic mode is served by enabling its own
    # read-only tools rather than by the framework's function-calling loop
    # (which this backend cannot participate in -- ``claude -p`` returns final
    # text, not tool_calls).
    supports_native_agentic = True

    #: Read-only tools exposed in agentic mode. Deliberately excludes Edit,
    #: Write, and Bash: the agent's job is to explore the codebase before
    #: proposing a solution, not to modify the machine it runs on.
    #: WebSearch/WebFetch are read-only research tools: the agent may
    #: consult literature independently before proposing a solution.
    AGENTIC_TOOLS = ("Read", "Grep", "Glob", "WebSearch", "WebFetch")

    def __init__(self, model_cfg: Optional[LLMModelConfig] = None):
        model_cfg = model_cfg or LLMModelConfig()

        self.model = _strip_provider_prefix(model_cfg.name)
        self.max_tokens = model_cfg.max_tokens
        self.timeout = model_cfg.timeout
        self.retries = model_cfg.retries
        self.retry_delay = model_cfg.retry_delay
        self.reasoning_effort = normalize_effort(getattr(model_cfg, "reasoning_effort", None))

        # Extras that only apply to this backend; read defensively because they
        # arrive via DatabaseConfig-style untyped passthrough.
        self.binary = getattr(model_cfg, "cli_binary", None) or os.environ.get(
            "SKYDISCOVER_CLAUDE_BINARY", DEFAULT_CLI_BINARY
        )
        self.max_budget_usd = getattr(model_cfg, "max_budget_usd", None)
        self.extra_args: List[str] = list(getattr(model_cfg, "cli_extra_args", None) or [])
        self.max_usage_limit_waits = int(
            getattr(model_cfg, "max_usage_limit_waits", None) or DEFAULT_MAX_USAGE_LIMIT_WAITS
        )
        self.fallback_model = getattr(model_cfg, "fallback_model", None)

        resolved = shutil.which(self.binary)
        if not resolved:
            raise RuntimeError(
                f"Claude CLI binary {self.binary!r} not found on PATH. Install Claude Code "
                f"(https://claude.com/claude-code) and run `claude auth` once, or set "
                f"SKYDISCOVER_CLAUDE_BINARY to its absolute path."
            )
        self.binary = resolved

        # temperature is meaningless here: the CLI exposes no sampling knob.
        # skydiscover.llm.temperature emulates it via model/effort instead.
        self.temperature = None
        self.top_p = None

        if not hasattr(logger, "_initialized_models"):
            logger._initialized_models = set()
        key = f"claude_cli:{self.model or 'default'}"
        if key not in logger._initialized_models:
            logger.info(
                "Claude CLI LLM: %s (effort=%s, binary=%s)",
                self.model or "<cli default>",
                self.reasoning_effort or "<cli default>",
                self.binary,
            )
            logger._initialized_models.add(key)

    # ------------------------------------------------------------------
    # LLMInterface
    # ------------------------------------------------------------------

    async def generate(
        self, system_message: str, messages: List[Dict[str, Any]], **kwargs
    ) -> LLMResponse:
        if kwargs.get("image_output"):
            raise NotImplementedError(
                "The Claude CLI backend cannot generate images. Use an OpenAI-compatible "
                "model for image mode (language: image)."
            )

        prompt = _flatten_messages(messages)
        if not prompt.strip():
            raise ValueError("Claude CLI backend received an empty prompt")

        text = await self._run_with_limits(system_message or "", prompt, **kwargs)
        return LLMResponse(text=text)

    # ------------------------------------------------------------------
    # Invocation
    # ------------------------------------------------------------------

    async def _run_with_limits(self, system_message: str, prompt: str, **kwargs) -> str:
        gate = get_usage_limit_gate()
        retries = kwargs.get("retries", self.retries)
        retries = 0 if retries is None else int(retries)
        retry_delay = kwargs.get("retry_delay", self.retry_delay)
        retry_delay = 5 if retry_delay is None else retry_delay

        attempt = 0
        limit_waits = 0

        while True:
            # Respect a block another call is already serving.
            await gate.wait_until_clear()

            try:
                payload = await self._invoke_cli(system_message, prompt, **kwargs)
                return self._text_from_payload(payload)

            except UsageLimitError as exc:
                # A quota pause is scheduled downtime, not a failed attempt: it
                # must not consume the retry budget, or the run would give up
                # on exactly the condition we are trying to survive.
                limit_waits += 1
                if limit_waits > self.max_usage_limit_waits:
                    raise RuntimeError(
                        f"Claude CLI still usage-limited after {self.max_usage_limit_waits} "
                        f"waits; giving up. Last message: {exc}"
                    ) from exc
                GLOBAL_CALL_LOG.record(
                    "usage_limit_pause",
                    source="claude_cli",
                    model=self.model,
                    wait_number=limit_waits,
                    reset_at=exc.reset_at,
                    message=str(exc)[:200],
                )
                await gate.pause_for(exc.reset_at, str(exc)[:200])
                continue

            except asyncio.TimeoutError:
                if attempt < retries:
                    attempt += 1
                    # A timeout at high effort usually means the model thought
                    # past the deadline; retrying the identical call tends to
                    # time out again. Step the effort down one rung per timeout
                    # so the retry budget degrades toward a call that can finish.
                    effort = normalize_effort(kwargs.get("reasoning_effort", self.reasoning_effort))
                    downgraded = _downgrade_effort(effort)
                    if downgraded is not None:
                        kwargs["reasoning_effort"] = downgraded
                    GLOBAL_CALL_LOG.record(
                        "retry",
                        source="claude_cli",
                        model=self.model,
                        attempt=attempt,
                        max_attempts=retries + 1,
                        reason="timeout",
                        effort_downgraded_to=downgraded,
                    )
                    logger.warning(
                        "Claude CLI timed out (attempt %s/%s), retrying%s...",
                        attempt,
                        retries + 1,
                        f" at effort={downgraded}" if downgraded is not None else "",
                    )
                    await asyncio.sleep(retry_delay)
                    continue
                raise

            except Exception as exc:
                if attempt < retries:
                    attempt += 1
                    GLOBAL_CALL_LOG.record(
                        "retry",
                        source="claude_cli",
                        model=self.model,
                        attempt=attempt,
                        max_attempts=retries + 1,
                        reason=type(exc).__name__,
                        error=str(exc)[:200],
                    )
                    logger.warning(
                        "Claude CLI error (attempt %s/%s): %s, retrying...",
                        attempt,
                        retries + 1,
                        exc,
                    )
                    await asyncio.sleep(retry_delay)
                    continue
                raise

    def _build_command(self, system_prompt_path: Optional[str], **kwargs) -> List[str]:
        agentic = bool(kwargs.get("agentic"))

        cmd = [
            self.binary,
            "--print",
            "--output-format",
            "json",
        ]

        if agentic:
            # Let the CLI explore the codebase with its own read-only tools.
            tools = ",".join(self.AGENTIC_TOOLS)
            max_steps = int(kwargs.get("max_steps") or 5)
            # Every tool use costs a turn, so a budget equal to max_steps is
            # spent purely on exploration and the run ends in error_max_turns
            # with no answer. Leave room for the tool calls *and* the final
            # response.
            max_turns = max(4, 2 * max_steps + 2)
            cmd += [
                "--tools",
                tools,
                # Pre-approve them so a non-interactive run never blocks on a
                # permission prompt.
                "--allowed-tools",
                tools,
                "--max-turns",
                str(max_turns),
            ]
            codebase_root = kwargs.get("codebase_root")
            if codebase_root and os.path.isdir(codebase_root):
                cmd += ["--add-dir", str(codebase_root)]
        else:
            # Pure text generation: no tools, no agent loop.
            cmd += ["--tools", "", "--max-turns", "1"]

        # Keep the run hermetic and side-effect free.
        cmd += [
            "--no-session-persistence",
            "--strict-mcp-config",
            "--disable-slash-commands",
            "--setting-sources",
            "",
        ]

        model = kwargs.get("model", self.model)
        if model:
            cmd += ["--model", str(model)]

        effort = normalize_effort(kwargs.get("reasoning_effort", self.reasoning_effort))
        if effort:
            cmd += ["--effort", effort]

        if system_prompt_path:
            cmd += ["--system-prompt-file", system_prompt_path]

        fallback = kwargs.get("fallback_model", self.fallback_model)
        if fallback:
            cmd += ["--fallback-model", str(fallback)]

        budget = kwargs.get("max_budget_usd", self.max_budget_usd)
        if budget:
            cmd += ["--max-budget-usd", str(budget)]

        schema = _json_schema_from_response_format(kwargs.get("response_format"))
        if schema is not None:
            cmd += ["--json-schema", json.dumps(schema)]

        cmd += self.extra_args
        return cmd

    @staticmethod
    def _resolve_cwd(workdir: str, **kwargs) -> str:
        """Working directory for the CLI process.

        Agentic mode runs from the codebase root so the agent's own file tools
        have something to explore; everything else runs from an isolated temp
        directory.
        """
        if kwargs.get("agentic"):
            root = kwargs.get("codebase_root")
            if root and os.path.isdir(root):
                return str(root)
        return workdir

    async def _invoke_cli(self, system_message: str, prompt: str, **kwargs) -> Dict[str, Any]:
        timeout = kwargs.get("timeout", self.timeout) or 600

        # The system prompt goes through a file rather than argv: templates plus
        # injected evaluator source routinely exceed a comfortable argument size.
        workdir = tempfile.mkdtemp(prefix="skydiscover-claude-cli-")
        system_prompt_path = None
        try:
            if system_message.strip():
                system_prompt_path = os.path.join(workdir, "system_prompt.txt")
                with open(system_prompt_path, "w", encoding="utf-8") as fh:
                    fh.write(system_message)

            cmd = self._build_command(system_prompt_path, **kwargs)

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                # Non-agentic calls run from an empty temp dir so ambient
                # CLAUDE.md files cannot leak into the prompt. Agentic calls
                # must start *in* the codebase: --add-dir only widens the
                # permission boundary, it does not point the agent anywhere, so
                # from a temp cwd its Glob/Grep would search an empty directory.
                cwd=self._resolve_cwd(workdir, **kwargs),
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(prompt.encode("utf-8")), timeout=timeout
                )
            except asyncio.TimeoutError:
                _terminate(proc)
                # Reap the process so it cannot linger as a zombie.
                try:
                    await asyncio.wait_for(proc.wait(), timeout=10)
                except asyncio.TimeoutError:
                    logger.warning("Claude CLI did not exit after termination.")
                raise

            return self._parse_payload(stdout, stderr, proc.returncode)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def _parse_payload(
        self, stdout: bytes, stderr: bytes, returncode: Optional[int]
    ) -> Dict[str, Any]:
        out = (stdout or b"").decode("utf-8", errors="replace").strip()
        err = (stderr or b"").decode("utf-8", errors="replace").strip()

        if not out:
            combined = err or f"claude exited with code {returncode} and no output"
            reset_at = parse_usage_limit(combined, None, None)
            if reset_at is not None:
                raise UsageLimitError(combined, reset_at=reset_at, source="claude_cli")
            raise RuntimeError(f"Claude CLI produced no output. stderr: {combined[:500]}")

        try:
            payload = json.loads(out)
        except json.JSONDecodeError:
            # Fall back to the last JSON object on stdout; some CLI versions
            # prepend diagnostics before the result object.
            payload = _last_json_object(out)
            if payload is None:
                reset_at = parse_usage_limit(out, None, None)
                if reset_at is not None:
                    raise UsageLimitError(out, reset_at=reset_at, source="claude_cli")
                raise RuntimeError(f"Claude CLI returned non-JSON output: {out[:500]}")

        if not isinstance(payload, dict):
            raise RuntimeError(f"Claude CLI returned unexpected JSON: {out[:300]}")

        GLOBAL_COST_TRACKER.record(payload)
        return payload

    def _text_from_payload(self, payload: Dict[str, Any]) -> str:
        result = payload.get("result")
        status = payload.get("api_error_status")
        is_error = bool(payload.get("is_error"))
        subtype = payload.get("subtype")

        message = result if isinstance(result, str) else json.dumps(result)

        # The CLI exits 0 and reports subtype "success" even for API errors, so
        # is_error / api_error_status are the only reliable signals.
        if is_error or (isinstance(status, int) and status >= 400):
            reset_at = parse_usage_limit(message, status if isinstance(status, int) else None)
            if reset_at is not None:
                raise UsageLimitError(message, reset_at=reset_at, source="claude_cli")
            raise RuntimeError(
                f"Claude CLI call failed (status={status}, subtype={subtype}): {message[:500]}"
            )

        if not isinstance(result, str):
            if subtype == "error_max_turns":
                raise RuntimeError(
                    "Claude CLI exhausted its turn budget before answering. In agentic "
                    "mode, raise agentic.max_steps; otherwise the prompt likely asked "
                    "for tool use that this backend does not allow."
                )
            raise RuntimeError(f"Claude CLI returned no text result (subtype={subtype})")

        # A limit can also surface as a normal-looking result body.
        if looks_like_usage_limit(result, None) and len(result) < 400:
            reset_at = parse_usage_limit(result, None)
            if reset_at is not None:
                raise UsageLimitError(result, reset_at=reset_at, source="claude_cli")

        return result


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────


def _strip_provider_prefix(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    for prefix in ("claude_cli/", "claude-cli/", "claude_code/"):
        if name.startswith(prefix):
            return name[len(prefix) :] or None
    return name


def _flatten_messages(messages: List[Dict[str, Any]]) -> str:
    """Render a Chat-Completions message list as one prompt string.

    ``claude -p`` takes a single prompt on stdin. SkyDiscover's non-agentic
    path sends exactly one user message, which passes through untouched; a
    longer history is rendered as a labelled transcript.
    """
    if not messages:
        return ""

    texts = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            parts = [p.get("text", "") for p in content if p.get("type") == "text"]
            content = "\n".join(p for p in parts if p)
        texts.append((msg.get("role", "user"), content or ""))

    if len(texts) == 1 and texts[0][0] == "user":
        return texts[0][1]

    return "\n\n".join(f"[{role}]\n{content}" for role, content in texts if content)


def _json_schema_from_response_format(response_format: Any) -> Optional[Dict[str, Any]]:
    """Extract a bare JSON Schema from an OpenAI-style response_format."""
    if not isinstance(response_format, dict):
        return None
    if response_format.get("type") != "json_schema":
        return None
    block = response_format.get("json_schema")
    if isinstance(block, dict):
        schema = block.get("schema")
        if isinstance(schema, dict):
            return schema
        return block
    return None


def _last_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Return the last top-level JSON object in *text*, if any."""
    end = text.rfind("}")
    while end != -1:
        depth = 0
        for start in range(end, -1, -1):
            char = text[start]
            if char == "}":
                depth += 1
            elif char == "{":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start : end + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(parsed, dict):
                        return parsed
                    break
        end = text.rfind("}", 0, end)
    return None


def _terminate(proc) -> None:
    try:
        proc.kill()
    except ProcessLookupError:
        pass
    except Exception:
        logger.debug("Failed to terminate Claude CLI process", exc_info=True)
