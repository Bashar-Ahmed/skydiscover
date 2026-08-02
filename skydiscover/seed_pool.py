"""Loading a pool of seed programs to initialise the search population.

By default a run starts from exactly one program, which means every island in
an island-model database starts in the same basin: ``adaevolve`` clones the
single seed into each empty island, so the first several migrations exchange
near-duplicates and the UCB arms are scored against variations of one idea.

A seed directory lets a benchmark author hand the search several genuinely
different starting points — insertion vs. merge vs. radix sort, FIFO vs. SJF vs.
work-stealing schedulers, naive vs. tiled vs. vectorised kernels. Diverse
initialisation is one of the cheapest wins available to an evolutionary search,
and unlike almost everything else here it costs no LLM calls at all: seeds are
scored by the evaluator and nothing else.

The two functions below are deliberately pure — no Runner or database imports —
so they can be tested without constructing either.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# One (path, source) pair per seed.
Seed = Tuple[str, str]


def load_seed_pool(
    seed_dir: Optional[str],
    initial_program_path: Optional[str],
    file_extension: str,
    max_seeds: Optional[int] = None,
) -> List[Seed]:
    """Read additional seed programs from *seed_dir*.

    Args:
        seed_dir: Directory of seed programs. A relative path is resolved
            against the primary seed program's directory, not the process cwd,
            so a task directory stays relocatable.
        initial_program_path: The primary seed. If it also lives in *seed_dir*
            it is skipped, so it is never added twice.
        file_extension: Only files with this extension are read (``".py"``,
            ``".cu"``, …), matching the language of the primary seed.
        max_seeds: Hard cap on how many are returned. Every seed costs one
            evaluator run before iteration 1, so an unbounded directory can
            stall startup for a long time with no visible progress.

    Returns:
        ``(path, source)`` pairs sorted by filename. Empty on any problem —
        a missing or unreadable seed directory downgrades the run to
        single-seed behaviour rather than aborting it.
    """
    if not seed_dir:
        return []

    base = Path(seed_dir)
    if not base.is_absolute() and initial_program_path:
        base = Path(initial_program_path).resolve().parent / base

    if not base.is_dir():
        logger.warning(f"seed_programs_dir does not exist, ignoring: {base}")
        return []

    primary = Path(initial_program_path).resolve() if initial_program_path else None

    seeds: List[Seed] = []
    for path in sorted(base.iterdir()):
        if not path.is_file() or path.suffix != file_extension:
            continue
        if primary is not None and path.resolve() == primary:
            continue
        try:
            seeds.append((str(path), path.read_text(encoding="utf-8")))
        except (OSError, UnicodeDecodeError) as e:
            logger.warning(f"Skipping unreadable seed {path}: {e}")

    if max_seeds is not None and len(seeds) > max_seeds:
        logger.warning(
            f"Found {len(seeds)} seed programs in {base}, using the first "
            f"{max_seeds} (raise max_seed_programs to use more)"
        )
        seeds = seeds[:max_seeds]

    return seeds


def _normalized_digest(source: str) -> str:
    """Whitespace-insensitive content hash.

    Normalisation is whitespace-only — strip each line, drop blank lines — and
    therefore language-agnostic. It deliberately does *not* strip comments or
    docstrings: doing that needs a per-language parser, and the obvious
    line-based approximation collapses genuinely different programs whenever
    they share a header comment.
    """
    normalized = "\n".join(line.strip() for line in source.splitlines() if line.strip())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def dedupe_seeds(seeds: Sequence[Seed], existing: Optional[str] = None) -> List[Seed]:
    """Drop seeds whose source differs only in whitespace.

    Duplicates are pure waste: an identical program costs a full evaluator run
    and then occupies a slot in a capped archive that a different idea could
    have held.

    Args:
        seeds: Candidate seeds, in load order.
        existing: Source of a program already being added — the primary seed.
            Matching against content as well as path matters because a copy of
            the primary sitting inside the seed directory has a different path
            but is the same program.
    """
    seen = {_normalized_digest(existing)} if existing else set()
    unique: List[Seed] = []
    for path, source in seeds:
        digest = _normalized_digest(source)
        if digest in seen:
            logger.info(f"Skipping duplicate seed program: {os.path.basename(path)}")
            continue
        seen.add(digest)
        unique.append((path, source))
    return unique
