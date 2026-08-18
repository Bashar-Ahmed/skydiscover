"""Codex CLI LLM backend.

Drives the locally installed ``codex`` binary in non-interactive mode
(``codex exec``) instead of calling an HTTP API. Like the Claude Code CLI
backend this lets SkyDiscover run against a ChatGPT subscription rather than a
metered API key: the CLI reuses whatever credentials ``codex login`` already
established. Nothing is containerized — the binary runs directly on this host.

Model selection uses the ``codex_cli/`` provider prefix::

    llm:
      models:
        - name: "codex_cli/gpt-5.6-terra"
          weight: 1.0

Usage limits are handled by :mod:`skydiscover.llm.rate_limit`: a rejected call
parks every caller until the quota resets instead of failing the iteration.

Differences from the Claude CLI backend, all forced by the tool
=============================================================

* **No system-prompt channel.** ``codex exec`` accepts one prompt and nothing
  else, so the system message is prepended to it under a header. The Claude
  backend can pass ``--system-prompt-file``; this one cannot.
* **No way to disable tools.** Codex is a coding agent and always has its shell
  available — there is no ``--tools ""`` equivalent. Non-agentic generation is
  therefore constrained rather than disarmed: it runs under
  ``--sandbox read-only`` from an empty temp directory, so the model can look
  around but cannot write anything or reach the network. Treat "non-agentic"
  here as "no useful context to explore", not as "no tools".
* **No cost figure.** ``turn.completed`` reports token counts but no dollar
  amount, so :data:`GLOBAL_USAGE_TRACKER` counts tokens only.
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
from skydiscover.llm.rate_limit import (
    UsageLimitError,
    get_usage_limit_gate,
    looks_like_usage_limit,
    parse_usage_limit,
)

logger = logging.getLogger("skydiscover.llm")

DEFAULT_CLI_BINARY = "codex"

#: Values ``model_reasoning_effort`` accepts, weakest first.
#:
#: Taken from ``~/.codex/models_cache.json`` (``supported_reasoning_levels``),
#: not from the published docs, which still list a ``minimal`` rung that the API
#: now rejects outright with ``unsupported_value``. The top of the ladder is
#: model-dependent — ``ultra`` exists only on the largest models, and ``max``
#: only on some — but Codex coerces a too-high level down to the nearest
#: supported one, so sending one is safe while sending ``minimal`` is not.
CLI_EFFORT_LEVELS: Tuple[str, ...] = ("low", "medium", "high", "xhigh", "max", "ultra")

#: Effort spellings that have no direct Codex equivalent. ``minimal`` matters
#: most: OpenAI's own reasoning_effort vocabulary includes it and Codex does
#: not, so it is mapped down rather than dropped — dropping it would silently
#: promote the call to the model's default effort.
_EFFORT_ALIASES = {
    "minimal": "low",
    "none": "low",
    "default": "medium",
    "standard": "medium",
    "very_high": "xhigh",
    "extra_high": "xhigh",
    "extra-high": "xhigh",
    "maximum": "max",
}

# Refuse to loop forever if a quota never clears.
DEFAULT_MAX_USAGE_LIMIT_WAITS = 24

#: Sandbox policy. Read-only in both modes: in agentic mode the agent's job is
#: to understand the codebase before proposing a solution, not to edit the
#: machine it runs on, and the framework applies the model's answer itself.
SANDBOX_MODE = "read-only"


def normalize_effort(effort: Optional[str]) -> Optional[str]:
    """Coerce an effort string to one Codex accepts, or None."""
    if not effort:
        return None
    value = str(effort).strip().lower()
    value = _EFFORT_ALIASES.get(value, value)
    if value in CLI_EFFORT_LEVELS:
        return value
    logger.warning(
        "Unknown reasoning effort %r for the Codex CLI; expected one of %s. Ignoring it.",
        effort,
        ", ".join(CLI_EFFORT_LEVELS),
    )
    return None


#: Where the CLI caches the model list it fetched for this account.
MODELS_CACHE_FILENAME = "models_cache.json"


def available_models(codex_home: Optional[str] = None) -> List[str]:
    """Model slugs this account can actually select, best-first.

    Read from the CLI's own ``models_cache.json`` rather than hardcoded: the
    list is per-account and moves quickly (three slugs appeared and one model
    family was superseded within a single week), so a baked-in list would be
    wrong almost immediately. Entries marked ``visibility: hide`` are internal —
    ``codex-auto-review`` is the approval-review model, not something to evolve
    with — and are left out.

    Returns an empty list if the cache is missing or unreadable, which callers
    must treat as "unknown", never as "no models".
    """
    home = (
        codex_home
        or os.environ.get("CODEX_HOME")
        or os.path.join(os.path.expanduser("~"), ".codex")
    )
    try:
        with open(os.path.join(home, MODELS_CACHE_FILENAME), encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return []

    entries = payload.get("models")
    if not isinstance(entries, list):
        return []

    listed = [
        m
        for m in entries
        if isinstance(m, dict) and m.get("slug") and m.get("visibility") != "hide"
    ]
    # `priority` orders the CLI's own picker; lower is better.
    listed.sort(key=lambda m: m.get("priority", 1_000_000))
    return [m["slug"] for m in listed]


class UsageTracker:
    """Process-wide tally of the tokens Codex reports using.

    Codex reports no cost, so unlike the Claude CLI tracker there is no dollar
    figure to accumulate. Cached input is counted separately because it is the
    number that explains a run suddenly getting slower or hitting its window
    earlier: a prompt-builder change that breaks the cache prefix shows up here
    first.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls = 0
        self.input_tokens = 0
        self.cached_input_tokens = 0
        self.output_tokens = 0
        self.reasoning_tokens = 0
        self._next_log_at = 1_000_000

    def record(self, usage: Dict[str, Any]) -> None:
        with self._lock:
            self.calls += 1
            self.input_tokens += int(usage.get("input_tokens") or 0)
            self.cached_input_tokens += int(usage.get("cached_input_tokens") or 0)
            self.output_tokens += int(usage.get("output_tokens") or 0)
            self.reasoning_tokens += int(usage.get("reasoning_output_tokens") or 0)
            total = self.input_tokens + self.output_tokens
            should_log = total >= self._next_log_at
            if should_log:
                self._next_log_at = total + 1_000_000
            snapshot = (self.calls, self.input_tokens, self.cached_input_tokens, self.output_tokens)

        if should_log:
            calls, tin, tcached, tout = snapshot
            logger.info(
                "Codex CLI usage so far: %d calls, %d in (%d cached) / %d out tokens.",
                calls,
                tin,
                tcached,
                tout,
            )

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "calls": self.calls,
                "input_tokens": self.input_tokens,
                "cached_input_tokens": self.cached_input_tokens,
                "output_tokens": self.output_tokens,
                "reasoning_output_tokens": self.reasoning_tokens,
            }


GLOBAL_USAGE_TRACKER = UsageTracker()


class CodexCLILLM(LLMInterface):
    """LLM backend that shells out to the local ``codex`` CLI."""

    # Codex is itself an agent, so agentic mode is served by pointing it at the
    # codebase and letting it use its own tools, rather than by the framework's
    # function-calling loop (which this backend cannot join -- `codex exec`
    # returns final text, not tool_calls).
    supports_native_agentic = True

    def __init__(self, model_cfg: Optional[LLMModelConfig] = None):
        model_cfg = model_cfg or LLMModelConfig()

        self.model = _strip_provider_prefix(model_cfg.name)
        self.max_tokens = model_cfg.max_tokens
        self.timeout = model_cfg.timeout
        self.retries = model_cfg.retries
        self.retry_delay = model_cfg.retry_delay
        self.reasoning_effort = normalize_effort(getattr(model_cfg, "reasoning_effort", None))

        # Extras that only apply to CLI backends; read defensively because they
        # arrive via DatabaseConfig-style untyped passthrough.
        self.binary = getattr(model_cfg, "cli_binary", None) or os.environ.get(
            "SKYDISCOVER_CODEX_BINARY", DEFAULT_CLI_BINARY
        )
        self.extra_args: List[str] = list(getattr(model_cfg, "cli_extra_args", None) or [])
        self.max_usage_limit_waits = int(
            getattr(model_cfg, "max_usage_limit_waits", None) or DEFAULT_MAX_USAGE_LIMIT_WAITS
        )

        resolved = shutil.which(self.binary)
        if not resolved:
            raise RuntimeError(
                f"Codex CLI binary {self.binary!r} not found on PATH. Install Codex "
                f"(https://developers.openai.com/codex) and run `codex login` once, or set "
                f"SKYDISCOVER_CODEX_BINARY to its absolute path."
            )
        self.binary = resolved
        self._warn_if_model_unknown()

        # temperature is meaningless here: the CLI exposes no sampling knob.
        # skydiscover.llm.temperature emulates it via model/effort instead.
        self.temperature = None
        self.top_p = None

        if not hasattr(logger, "_initialized_models"):
            logger._initialized_models = set()
        key = f"codex_cli:{self.model or 'default'}"
        if key not in logger._initialized_models:
            logger.info(
                "Codex CLI LLM: %s (effort=%s, binary=%s)",
                self.model or "<cli default>",
                self.reasoning_effort or "<cli default>",
                self.binary,
            )
            logger._initialized_models.add(key)

    def _warn_if_model_unknown(self) -> None:
        """Flag a model slug this account does not offer, at construction time.

        A wrong slug is not rejected until the first generation, and then only
        as an opaque 400 ("not supported when using Codex with a ChatGPT
        account") several retries deep. Since the valid names are unguessable —
        the family name `gpt-5.6` is invalid while `gpt-5.6-sol` is not — a run
        can burn its first iterations on a typo.

        This warns rather than raises: the cache is written by the CLI, not by
        us, so an empty or stale one must not be able to block a model that
        actually works.
        """
        if not self.model:
            return
        known = available_models()
        if not known or self.model in known:
            return
        logger.warning(
            "Codex model %r is not in this account's model list (%s). "
            "The CLI will likely reject it; run `codex exec` once to refresh "
            "%s, or pick one of the listed slugs.",
            self.model,
            ", ".join(known),
            MODELS_CACHE_FILENAME,
        )

    # ------------------------------------------------------------------
    # LLMInterface
    # ------------------------------------------------------------------

    async def generate(
        self, system_message: str, messages: List[Dict[str, Any]], **kwargs
    ) -> LLMResponse:
        if kwargs.get("image_output"):
            raise NotImplementedError(
                "The Codex CLI backend cannot generate images. Use an OpenAI-compatible "
                "model for image mode (language: image)."
            )

        prompt = _flatten_messages(messages)
        if not prompt.strip():
            raise ValueError("Codex CLI backend received an empty prompt")

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
                return await self._invoke_cli(system_message, prompt, **kwargs)

            except UsageLimitError as exc:
                # A quota pause is scheduled downtime, not a failed attempt: it
                # must not consume the retry budget, or the run would give up
                # on exactly the condition we are trying to survive.
                limit_waits += 1
                if limit_waits > self.max_usage_limit_waits:
                    raise RuntimeError(
                        f"Codex CLI still usage-limited after {self.max_usage_limit_waits} "
                        f"waits; giving up. Last message: {exc}"
                    ) from exc
                await gate.pause_for(exc.reset_at, str(exc)[:200])
                continue

            except asyncio.TimeoutError:
                if attempt < retries:
                    attempt += 1
                    logger.warning(
                        "Codex CLI timed out (attempt %s/%s), retrying...", attempt, retries + 1
                    )
                    await asyncio.sleep(retry_delay)
                    continue
                raise

            except Exception as exc:
                if attempt < retries:
                    attempt += 1
                    logger.warning(
                        "Codex CLI error (attempt %s/%s): %s, retrying...",
                        attempt,
                        retries + 1,
                        exc,
                    )
                    await asyncio.sleep(retry_delay)
                    continue
                raise

    def _build_command(self, schema_path: Optional[str], **kwargs) -> List[str]:
        cmd = [
            self.binary,
            "exec",
            # JSONL on stdout; progress goes to stderr either way.
            "--json",
            # A run must not depend on the cwd happening to be a git repo.
            "--skip-git-repo-check",
            # Leave no session files behind.
            "--ephemeral",
            # An evolutionary run makes thousands of calls; whatever is in the
            # user's config.toml must not silently redefine what they mean.
            "--ignore-user-config",
            "--sandbox",
            SANDBOX_MODE,
            # Never block for a human: this is a batch process. `codex exec` has
            # no --ask-for-approval flag (that is an interactive-mode option);
            # the policy is only reachable as a config key.
            "--config",
            "approval_policy=never",
        ]

        model = kwargs.get("model", self.model)
        if model:
            cmd += ["--model", str(model)]

        effort = normalize_effort(kwargs.get("reasoning_effort", self.reasoning_effort))
        if effort:
            cmd += ["--config", f"model_reasoning_effort={effort}"]

        if schema_path:
            cmd += ["--output-schema", schema_path]

        cmd += self.extra_args

        # The prompt arrives on stdin. Prompts here routinely carry a whole
        # program plus context and would be awkward as an argv entry.
        cmd.append("-")
        return cmd

    @staticmethod
    def _resolve_cwd(workdir: str, **kwargs) -> str:
        """Working directory for the CLI process.

        Agentic mode runs from the codebase root so the agent's own file tools
        have something to explore; everything else runs from an isolated temp
        directory, which is also what keeps a non-agentic call from finding any
        project files to read.
        """
        if kwargs.get("agentic"):
            root = kwargs.get("codebase_root")
            if root and os.path.isdir(root):
                return str(root)
        return workdir

    async def _invoke_cli(self, system_message: str, prompt: str, **kwargs) -> str:
        timeout = kwargs.get("timeout", self.timeout) or 600

        workdir = tempfile.mkdtemp(prefix="skydiscover-codex-cli-")
        try:
            schema_path = None
            schema = _json_schema_from_response_format(kwargs.get("response_format"))
            if schema is not None:
                schema_path = os.path.join(workdir, "output_schema.json")
                with open(schema_path, "w", encoding="utf-8") as fh:
                    json.dump(schema, fh)

            cmd = self._build_command(schema_path, **kwargs)
            stdin_text = _merge_system_prompt(system_message, prompt)

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._resolve_cwd(workdir, **kwargs),
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(stdin_text.encode("utf-8")), timeout=timeout
                )
            except asyncio.TimeoutError:
                _terminate(proc)
                # Reap the process so it cannot linger as a zombie.
                try:
                    await asyncio.wait_for(proc.wait(), timeout=10)
                except asyncio.TimeoutError:
                    logger.warning("Codex CLI did not exit after termination.")
                raise

            return self._text_from_stream(stdout, stderr, proc.returncode)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def _text_from_stream(self, stdout: bytes, stderr: bytes, returncode: Optional[int]) -> str:
        out = (stdout or b"").decode("utf-8", errors="replace").strip()
        err = (stderr or b"").decode("utf-8", errors="replace").strip()

        events = _parse_jsonl(out)

        if not events:
            combined = err or out or f"codex exited with code {returncode} and no output"
            _raise_usage_limit_if_any(combined)
            raise RuntimeError(f"Codex CLI produced no parseable events. stderr: {combined[:500]}")

        message = None
        failure = None
        for event in events:
            kind = event.get("type")
            if kind == "item.completed":
                item = event.get("item") or {}
                if item.get("type") == "agent_message":
                    text = item.get("text")
                    if isinstance(text, str) and text.strip():
                        # Keep the last one: earlier agent messages are progress
                        # narration, the final one is the answer.
                        message = text
            elif kind == "turn.completed":
                usage = event.get("usage")
                if isinstance(usage, dict):
                    GLOBAL_USAGE_TRACKER.record(usage)
            elif kind in ("turn.failed", "error"):
                failure = _event_error_text(event) or failure

        if failure:
            _raise_usage_limit_if_any(failure)
            raise RuntimeError(f"Codex CLI call failed: {failure[:500]}")

        if message is None:
            combined = err or out
            _raise_usage_limit_if_any(combined)
            raise RuntimeError(
                "Codex CLI returned no agent message "
                f"(exit={returncode}). stderr: {combined[:500]}"
            )

        # A limit can also surface as an ordinary-looking answer.
        if looks_like_usage_limit(message, None) and len(message) < 400:
            _raise_usage_limit_if_any(message)

        return message


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────


def _raise_usage_limit_if_any(text: str) -> None:
    """Raise UsageLimitError if *text* describes a quota block."""
    reset_at = parse_usage_limit(text, None, None)
    if reset_at is not None:
        raise UsageLimitError(text, reset_at=reset_at, source="codex_cli")


def _parse_jsonl(text: str) -> List[Dict[str, Any]]:
    """Parse the JSONL event stream, skipping any non-JSON noise."""
    events = []
    for line in text.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            events.append(parsed)
    return events


def _event_error_text(event: Dict[str, Any]) -> Optional[str]:
    """Best-effort human-readable message from a turn.failed / error event."""
    for key in ("error", "message"):
        value = event.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, dict):
            nested = value.get("message")
            if isinstance(nested, str) and nested.strip():
                return nested
            return json.dumps(value)
    return None


def _strip_provider_prefix(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    for prefix in ("codex_cli/", "codex-cli/", "codex/"):
        if name.startswith(prefix):
            return name[len(prefix) :] or None
    return name


def _merge_system_prompt(system_message: str, prompt: str) -> str:
    """Fold the system message into the prompt.

    ``codex exec`` takes one prompt and offers no system-prompt channel, so the
    instructions have to travel with it. The header keeps the boundary visible
    to the model instead of running the two together.
    """
    system_message = (system_message or "").strip()
    if not system_message:
        return prompt
    return f"# Instructions\n\n{system_message}\n\n# Task\n\n{prompt}"


def _flatten_messages(messages: List[Dict[str, Any]]) -> str:
    """Render a Chat-Completions message list as one prompt string.

    ``codex exec`` takes a single prompt on stdin. SkyDiscover's non-agentic
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


def _terminate(proc) -> None:
    try:
        proc.kill()
    except ProcessLookupError:
        pass
    except Exception:
        logger.debug("Failed to terminate Codex CLI process", exc_info=True)
