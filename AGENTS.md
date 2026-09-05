# AGENTS.md

SkyDiscover is an LLM-driven algorithmic-discovery framework: you supply a
scoring function and (optionally) a seed program, an LLM rewrites the program,
an evaluator scores it, and a pluggable **search strategy** decides what to
mutate next.

## Read this before exploring

`.claude/skills/skydiscover/` is a checked-in, exhaustive description of this
repo — architecture, search strategies, benchmarks, evaluators, configuration,
LLM backends, and a long list of sharp edges. **Read it instead of re-deriving
the codebase.** Start at `SKILL.md`, then the file in `references/` that matches
your task:

| Task | Read |
|---|---|
| Understand the system | `references/architecture.md` |
| Add or pick a search strategy | `references/search-strategies.md` |
| Add or run a benchmark | `references/benchmarks.md` |
| Write an evaluator | `references/evaluation.md` |
| Any config question | `references/configuration.md` |
| LLM backends, rate limits, temperature | `references/llm-backends.md` |
| Something behaves oddly | `references/gotchas.md` |

If you change behaviour these files describe, update them in the same commit —
they are the reason the next agent does not have to re-explore.

## Commands

```bash
uv sync --extra dev                                  # pytest lives in `dev`
uv run python -m pytest tests/ -q -m "not integration"
uv run black skydiscover/ && uv run isort skydiscover/
uv run skydiscover-run <initial_program> <evaluator> -c config.yaml -i 100
```

- **Integration tests are not deselected for you.** `addopts` is only
  `--strict-markers` and there is no `conftest.py`, so pass `-m "not
  integration"` yourself. CI does not — it runs `pytest tests/ -v`.
- **Lint is scoped to `skydiscover/` only.** `tests/`, `benchmarks/`,
  `examples/`, and `scripts/` are unlinted — do not reformat them in passing, it
  buries your real diff.
- `mypy` is configured strictly in `pyproject.toml` and never run.

## Invariants worth knowing before you edit

These fail *silently* rather than loudly:

- **`ProgramDatabase` has exactly two abstract methods**, `add` and `sample`.
  That is the whole search extension surface.
- **Controllers have no `save`/`load`.** State kept on a controller is lost on
  resume; resumable state belongs on the database.
- **Config keys are validated inconsistently.** A misspelled *top-level* key is
  dropped by a `hasattr` guard; a misspelled key under
  `llm`/`prompt`/`evaluator`/`agentic`/`monitor`/`search` raises `TypeError`,
  because those are splatted with `**dict`. Add new top-level keys accordingly.
- **A diff that applies zero blocks must be rejected.** `apply_diff` skips
  non-matching blocks, so a child can come back byte-identical to its parent and
  be stored as a genuine candidate. Use `apply_diff_detailed`, which returns
  `(result, applied, total)`.
- **The eval-failure screen is duplicated** in `default_discovery_controller.py`
  and `adaevolve/controller.py`. Change both.
- **The evaluator is not sandboxed.** LLM-generated code runs with full default
  container privileges and unrestricted network. Do not assume otherwise.

## Working conventions

- **The default LLM backend is a local CLI, not an API key.**
  `claude_cli/claude-opus-5` by default; `codex_cli/<slug>` also works. Both
  drive a binary on this host and authenticate from `claude auth` / `codex
  login`. Runs cost subscription quota — keep verification runs to a couple of
  iterations.
- **`uv.lock` is rewritten by `uv run`.** Check `git status` before committing
  and revert it (`git checkout -- uv.lock`) unless you deliberately changed a
  dependency. Never let it ride along in an unrelated commit.
- Verify against the real thing where you can. External CLI contracts in
  particular have drifted from their published docs more than once; prefer
  `--help`, a live call, or the tool's own cache over documentation.
- Say what you actually ran. If a check was skipped or a test fails, state it.
