# LLM backends

## Dispatch

`llm/llm_pool.py::create_llm_backend(model_cfg)`:

1. `model_cfg.init_client` if set (Python-API only, not expressible in YAML).
2. `is_local_provider(provider)` → a CLI backend, chosen by provider:
   `codex_cli` → `CodexCLILLM`, otherwise `ClaudeCLILLM`. Both are imported
   lazily, so runs that never use a CLI don't probe for its binary.
3. otherwise `OpenAILLM`.

`LLMModelConfig.provider` is filled from the model name's `provider/` prefix by
`LLMConfig.__post_init__`.

## `OpenAILLM` — every HTTP provider

`llm/openai.py`. One class reaches OpenAI, Azure, Anthropic, Gemini, DeepSeek,
Mistral, Ollama, and vLLM through OpenAI-compatible base URLs. There is **no
litellm in the native path** (`grep 'import litellm' skydiscover/` → zero hits),
despite the README claiming "any LiteLLM-compatible model works"; litellm
appears only in `extras/external/gepa_backend.py`.

- Reasoning models (`is_openai_reasoning_model`) get `max_completion_tokens` +
  `reasoning_effort` instead of `max_tokens` + `temperature`.
- `response_format` downgrade ladder on error: `json_schema` → `json_object` → dropped.
- Azure deployments that only expose the Responses API fall back transparently
  via `_call_api_via_responses`.
- Uses the **sync** `openai` SDK on `run_in_executor(None, …)` — the shared
  default thread pool.

## `ClaudeCLILLM` — Claude subscription, no API key

`llm/claude_cli.py`. Drives the locally installed `claude` binary in print mode.
Nothing is containerized; the binary runs on the same host, once per LLM call.

```yaml
llm:
  models:
    - name: "claude_cli/sonnet"     # or opus / haiku / fable, or a full model id
      weight: 1.0
```

Setup: install Claude Code, run `claude auth` once. No `ANTHROPIC_API_KEY`.

**Invoked as a pure generator by default:**
`--print --output-format json --tools "" --max-turns 1 --no-session-persistence
--strict-mcp-config --disable-slash-commands --setting-sources ""`, run from an
isolated temp cwd so ambient `CLAUDE.md` files can't leak into the prompt. The
system prompt goes through `--system-prompt-file` (templates plus injected
evaluator source routinely exceed a comfortable argv size); the user prompt goes
on stdin.

**Critical contract detail:** the CLI exits **0** and reports
`subtype: "success"` even for API errors. `is_error` and `api_error_status` are
the only reliable signals.

**Agentic mode** — the CLI is already an agent, so `--agentic` enables its own
read-only tools (`Read,Grep,Glob`; never `Edit`/`Write`/`Bash`) scoped by
`--add-dir`, and the process runs **from `codebase_root`** (`--add-dir` only
widens the permission boundary; it does not point the agent anywhere). Turn
budget is `max(4, 2*max_steps+2)` because every tool use costs a turn.

Per-model options (all optional): `cli_binary` (or `$SKYDISCOVER_CLAUDE_BINARY`),
`cli_extra_args`, `max_budget_usd`, `fallback_model`, `max_usage_limit_waits`.

Cost/token accounting is recorded in `GLOBAL_COST_TRACKER` — the only cost
tracking among the LLM backends. (The separate `--search claude_code` baseline
tracks its own `cost_usd` scraped from the CLI's JSON stream.)

Not supported: image generation (`language: image` needs an OpenAI-compatible
model).

## `CodexCLILLM` — ChatGPT subscription, no API key

`llm/codex_cli.py`. Drives the locally installed `codex` binary via
`codex exec`. Same shape as the Claude CLI backend — local binary, subscription
auth, no HTTP, nothing containerized. Template: `configs/codex_cli.yaml`.

```yaml
llm:
  models:
    - name: "codex_cli/gpt-5.6-sol"
      weight: 1.0
```

Setup: install Codex, run `codex login` once. No `OPENAI_API_KEY`.

Every call is:
`codex exec --json --skip-git-repo-check --ephemeral --ignore-user-config
--sandbox read-only --config approval_policy=never [--model M]
[--config model_reasoning_effort=E] [--output-schema F] -`
with the prompt on **stdin** (the trailing `-`).

⚠️ **`codex exec` has no `--ask-for-approval` flag** — that is interactive-mode
only, and passing it makes clap reject the whole invocation. The policy is
reachable only as a config key. It is load-bearing: `codex doctor` reports the
default as `OnRequest`, which would block a batch run.

⚠️ **The model must be a slug the account actually offers.** A bare family name
like `gpt-5.6` fails with *"not supported when using Codex with a ChatGPT
account"* — the names are unguessable, and the list is per-account and moves
fast (three slugs appeared in one week). It is therefore **never hardcoded**:
`codex_cli.py::available_models()` reads `$CODEX_HOME/models_cache.json`
(default `~/.codex`), drops `visibility: hide` entries (`codex-auto-review` is
the approval-review model), and sorts by the CLI's own `priority`.

`CodexCLILLM.__init__` **warns** — never raises — when the configured slug is
not in that list, naming the valid ones. Warning rather than raising is
deliberate: the cache belongs to the CLI, so a stale or missing one must not be
able to block a model that works. An empty list means "unknown", not "none".

As of writing, this account offers:

| slug | notes | top effort |
|---|---|---|
| `gpt-5.6-sol` | latest frontier agentic model | `ultra` |
| `gpt-5.6-terra` | balanced, everyday work | `ultra` |
| `gpt-5.6-luna` | fast and affordable | `max` |
| `gpt-5.5` | frontier: complex coding + research | `xhigh` |
| `gpt-5.4` | strong everyday coding | `xhigh` |
| `gpt-5.4-mini` | small, fast, cost-efficient | `xhigh` |
| `gpt-5.3-codex-spark` | ultra-fast, **128k** context (others are 272k) | `xhigh` |

Note `vary_model` in `temperature_emulation` reshapes the *pool* weights, so it
is inert with a single model configured — list several to make that axis live.

**Three differences from the Claude CLI backend, all forced by the tool:**

| | Claude CLI | Codex CLI |
|---|---|---|
| System prompt | `--system-prompt-file` | none — folded into the prompt under an `# Instructions` header |
| Tools off | `--tools "" --max-turns 1` | **impossible**; constrained by `--sandbox read-only` + an empty temp cwd |
| Cost | `total_cost_usd` per call | tokens only (`GLOBAL_USAGE_TRACKER`) |

So "non-agentic" here means *no useful context to explore*, not *no tools*:
Codex is a coding agent and always has its shell. It can never write files or
reach the network in either mode.

**Output is JSONL, not a single object.** The answer is the **last**
`item.completed` whose `item.type == "agent_message"` (earlier ones are progress
narration; `reasoning` and `command_execution` items are skipped).
`turn.completed.usage` feeds the tracker; `turn.failed` / `error` raise.

**Effort** is `low|medium|high|xhigh|max` (+`ultra` on the largest models),
passed as `--config model_reasoning_effort=…`. That matches
`DEFAULT_EFFORT_LADDER`, so one `temperature_emulation` block works across both
CLI backends unchanged.

⚠️ The published docs still list a **`minimal`** rung. It is gone — the API
rejects it with `unsupported_value`. `normalize_effort` maps it down to `low`.
The authoritative per-model list is `supported_reasoning_levels` in
`~/.codex/models_cache.json`; a too-high level is coerced down by Codex, so only
the bottom of the ladder is dangerous.

Per-model options: `cli_binary` (or `$SKYDISCOVER_CODEX_BINARY`),
`cli_extra_args`, `max_usage_limit_waits`. **`max_budget_usd` and
`fallback_model` are Claude-CLI-only** and are ignored here.

Not supported: image generation.

## Per-call log — what actually ran

`llm/call_log.py`. With temperature emulation the model *and* the effort are
sampled per call, so neither the config nor the ordinary log says what a given
iteration ran on. `Runner._setup_logging` points `GLOBAL_CALL_LOG` at
`<output_dir>/logs/llm_calls.jsonl`, one JSON object per line:

```json
{"ts":…, "event":"generate", "iteration":2, "phase":"iteration",
 "backend":"CodexCLILLM", "model":"gpt-5.6-luna", "reasoning_effort":"medium",
 "emulated":true, "temperature":0.7, "duration_s":7.16, "response_chars":594}
```

Events: `generate`, `generate_failed`, `generate_all`, `retry` (with `attempt`,
`reason`, and the Claude backend's `effort_downgraded_to`), and
`usage_limit_pause` (with `wait_number` and `reset_at`).

`iteration` / `phase` ride a **ContextVar** rather than call signatures — the
sampling happens deep inside `LLMPool.generate`, and threading an iteration
number down to it would touch every controller and backend. Set it with
`set_call_context(...)`; it is set in both `_run_iteration` bodies and in the
paradigm generator. A new controller that wants iteration numbers in the log
must call it too, otherwise its rows simply carry no `iteration` key.

The log is disabled until `configure()` is called, and every write failure
disables it with a warning rather than propagating — observability must not
fail a run.

## Usage limits — wait, don't skip

`llm/rate_limit.py`. A quota rejection is treated as **scheduled downtime**: the
reset instant is parsed, recorded in a process-wide gate, and every caller sleeps
until it passes. **A pause never consumes the retry budget.**

Two tiers, because the correct response differs:

| Tier | Examples | Behaviour |
|---|---|---|
| **Hard quota** — allowance spent | `usage limit reached`, weekly/5-hour limit, `quota exceeded`, `insufficient_quota` | park; fall back to 15 min if no reset time is given |
| **Soft rate limit** — transient cap | bare `429`, `rate_limit`, `too many requests` | park **only** if the provider gave a reset time; otherwise fall through to ordinary backoff |

`overloaded_error` (HTTP 529) is deliberately in neither list — it is server
capacity, handled by ordinary retry.

Reset times come from **response headers first** (`retry-after`,
`anthropic-ratelimit-*-reset`, `x-ratelimit-reset*`; earliest usable value wins).
Only if headers yield nothing is the error text parsed, in order:
`usage limit reached|<epoch>` (the CLI's machine-readable form), relative
durations ("in 42 minutes", "in 120ms"), compound Go durations ("6m0s"), bare
epochs, then wall-clock ("resets at 3pm"). The wall-clock branch requires real time evidence (am/pm or
`:MM`) and rejects digits followed by a duration unit — otherwise `"try again in
120ms"` would read as 12:00 and park the process for most of a day.

**The gate holds no loop-bound state** (`threading.Lock` + timestamp polling, not
`asyncio.Event`) because EvoX drives coroutines from worker threads with their
own event loops.

⚠️ When adding a new call path, **the gate wait must sit outside any enclosing
`asyncio.wait_for`** — otherwise a multi-hour pause gets cancelled by a
60-second timeout. See `agentic_generator.py::_call_llm_with_limits` for the
pattern.

## Temperature emulation

`llm/temperature.py`. Sampling parameters were removed starting with **Claude
Opus 4.7** — rejected with a 400 on Opus 4.7/4.8, Opus 5, Sonnet 5, Fable 5,
Mythos 5 — while **Opus 4.6, Sonnet 4.6, and the entire 4.5 family still accept
them**. Neither the Claude Code CLI nor the Codex CLI exposes a sampling knob at
all, so `model_supports_temperature` returns `False` for both providers and
`enabled: auto` activates emulation for them.

That matters here because diversity-per-iteration is a first-class search
parameter: AdaEvolve and EvoX both steer it. So `llm.temperature` is
reconstructed from the two knobs that *are* available:

| Lever | Mechanism | `T=0` | `T=1` | `T=2` |
|---|---|---|---|---|
| Model choice | `p_i ∝ w_i^(1/T)` over pool weights | argmax (deterministic) | configured weights exactly | flattened toward uniform |
| Reasoning effort | discrete Gaussian over `[low … max]` centred on `base_effort`, `σ = T × effort_spread` | pinned to `base_effort` | ±1 rung | spread across the ladder |

```yaml
llm:
  temperature: 0.7
  temperature_emulation:
    enabled: auto     # auto | true | false — "auto" activates only when a
                      # pooled model cannot accept a real temperature
    vary_model: true
    vary_effort: true
    effort_ladder: ["low", "medium", "high", "xhigh", "max"]
    base_effort: "medium"
    effort_spread: 1.0
    max_temperature: 2.0
```

Under the default `auto` it is inert for OpenAI/Gemini, so switching providers
needs no config change.

⚠️ Once emulation is **active**, the pool injects `reasoning_effort` into every
call. `vary_effort: false` stops the *jitter*, not the injection — calls still
get `base_effort` (`"medium"` by default). To leave requests completely
untouched set `temperature_emulation.enabled: false`.

**Per-model overrides:** a model entry in `llm.models` may set its own
`effort_ladder`, `base_effort`, and/or `effort_spread`; while pool-level
emulation is active, that model draws effort from its own emulator (unset
fields inherit the pool's). Use it when models saturate at different
rungs — one shared centre mis-places them. Applies in `LLMPool.generate`
and on the agentic delegation path; `generate_all` stays pool-level.

`OpenAILLM.__init__` separately drops `temperature` for models that reject it,
so a stale config value can't 400 a run.

## Agentic generation

`llm/agentic_generator.py::AgenticGenerator` — a bounded ReAct loop over two
sandboxed tools (`read_file`, `search`, schemas in `llm/tool_schemas/`).

- If the sampled backend advertises `supports_native_agentic` (both CLI
  backends), the whole loop is delegated to it instead.
- Otherwise it calls the raw OpenAI client directly — **it does not go through
  `OpenAILLM._generate_text`**, so anything added there (retries, downgrades)
  must be added here too.
- Quota pauses sit outside the per-step `wait_for` and are subtracted from
  `overall_timeout`.
- Returns `None` on failure; the caller falls back to direct generation.

## Adding an LLM backend

```python
from skydiscover.llm.base import LLMInterface, LLMResponse

class MyLLM(LLMInterface):
    supports_native_agentic = False        # True → AgenticGenerator delegates

    def __init__(self, model_cfg): ...
    async def generate(self, system_message, messages, **kwargs) -> LLMResponse: ...
```

`kwargs` the framework may pass: `temperature`, `top_p`, `max_tokens`,
`reasoning_effort`, `timeout`, `retries`, `retry_delay`, `response_format`,
`image_output`, `output_dir`, `program_id`.

Wire it up by adding the provider to `_PROVIDERS` in `config.py` (plus
`_LOCAL_PROVIDERS` if it needs no endpoint/key) and branching in
`llm_pool.py::create_llm_backend`.

Honour the usage-limit gate: `await get_usage_limit_gate().wait_until_clear()`
before each attempt, and `await gate.pause_for(reset_at, reason)` on a quota
rejection — without consuming a retry.
