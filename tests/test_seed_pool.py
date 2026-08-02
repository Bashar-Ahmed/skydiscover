"""Loading and placing a pool of seed programs."""

import os

from skydiscover.config import AdaEvolveDatabaseConfig
from skydiscover.runner import Runner
from skydiscover.search.adaevolve.database import AdaEvolveDatabase
from skydiscover.search.base_database import Program
from skydiscover.seed_pool import dedupe_seeds, load_seed_pool


def build_task(tmp_path):
    primary = tmp_path / "initial_program.py"
    primary.write_text("def f():\n    return 1\n")
    seeds = tmp_path / "seed_programs"
    seeds.mkdir()
    (seeds / "a.py").write_text("def f():\n    return 2\n")
    (seeds / "b.py").write_text("def f():\n    return 3\n")
    (seeds / "notes.txt").write_text("ignored\n")
    return str(primary), seeds


def names(seeds):
    return [os.path.basename(path) for path, _ in seeds]


def test_loader_filters_by_extension_and_sorts(tmp_path):
    primary, seeds = build_task(tmp_path)
    (seeds / "kernel.cu").write_text("__global__ void k(){}\n")

    assert names(load_seed_pool(str(seeds), primary, ".py")) == ["a.py", "b.py"]
    assert names(load_seed_pool(str(seeds), primary, ".cu")) == ["kernel.cu"]


def test_primary_inside_the_seed_dir_is_skipped_by_path(tmp_path):
    primary, seeds = build_task(tmp_path)
    moved = seeds / "initial_program.py"
    moved.write_text("def f():\n    return 1\n")

    loaded = load_seed_pool(str(seeds), str(moved), ".py")
    assert names(loaded) == ["a.py", "b.py"]


def test_relative_dir_resolves_against_the_primary_not_the_cwd(tmp_path, monkeypatch):
    primary, _ = build_task(tmp_path)
    monkeypatch.chdir(tmp_path.parent)

    assert names(load_seed_pool("seed_programs", primary, ".py")) == ["a.py", "b.py"]


def test_missing_or_unset_dir_degrades_to_single_seed(tmp_path):
    primary, _ = build_task(tmp_path)

    assert load_seed_pool(None, primary, ".py") == []
    assert load_seed_pool(str(tmp_path / "nope"), primary, ".py") == []


def test_max_seeds_caps_the_pool(tmp_path):
    primary, seeds = build_task(tmp_path)

    assert names(load_seed_pool(str(seeds), primary, ".py", max_seeds=1)) == ["a.py"]


def test_dedupe_ignores_whitespace_differences(tmp_path):
    primary, seeds = build_task(tmp_path)
    (seeds / "a_reformatted.py").write_text("\n\ndef f():\n      return 2\n\n")

    loaded = load_seed_pool(str(seeds), primary, ".py")
    assert names(dedupe_seeds(loaded)) == ["a.py", "b.py"]


def test_dedupe_drops_a_copy_of_the_primary(tmp_path):
    """A copy of the primary has a different path but is the same program."""
    primary, seeds = build_task(tmp_path)
    (seeds / "same_as_primary.py").write_text("def f():\n    return 1\n")

    loaded = load_seed_pool(str(seeds), primary, ".py")
    deduped = dedupe_seeds(loaded, existing=open(primary).read())
    assert names(deduped) == ["a.py", "b.py"]


def test_dedupe_keeps_programs_that_share_a_header_comment(tmp_path):
    """Comment stripping would collapse these; whitespace normalisation must not."""
    seeds = [
        ("x.py", "# Sorting benchmark seed\ndef s(a):\n    return sorted(a)\n"),
        ("y.py", "# Sorting benchmark seed\ndef s(a):\n    return list(reversed(a))\n"),
    ]
    assert len(dedupe_seeds(seeds)) == 2


# ----------------------------------------------------------------------
# Island placement
# ----------------------------------------------------------------------


def fake_runner(database):
    runner = object.__new__(Runner)
    runner.database = database
    return runner


def test_island_placement_leaves_island_zero_to_the_primary():
    db = AdaEvolveDatabase("t", AdaEvolveDatabaseConfig(num_islands=4))
    runner = fake_runner(db)

    assert [runner._seed_target_island(i) for i in range(5)] == [1, 2, 3, 0, 1]


def test_island_placement_is_none_without_islands():
    class Plain:
        pass

    runner = fake_runner(Plain())
    assert runner._seed_target_island(0) is None
    assert runner._seed_target_island(3) is None


def test_seeds_populate_every_island_and_count_as_evaluations():
    """Regression: seeds are placed on a named island but are not migrations.

    Without `is_seed`, islands 1..N-1 route through
    `receive_external_improvement`, which leaves their UCB arms with zero
    evaluations — so the bandit ignores the seed fitnesses just paid for.
    """
    db = AdaEvolveDatabase("t", AdaEvolveDatabaseConfig(num_islands=4, population_size=20))
    runner = fake_runner(db)

    db.add(Program(id="primary", solution="a", metrics={"combined_score": 0.5}))
    for index, (pid, score) in enumerate([("s1", 0.6), ("s2", 0.7), ("s3", 0.8)]):
        db.add(
            Program(id=pid, solution=pid, metrics={"combined_score": score}),
            iteration=0,
            target_island=runner._seed_target_island(index),
            is_seed=True,
        )

    assert [db.get_island_size(i) for i in range(4)] == [1, 1, 1, 1]
    assert all(db.adapter.states[i].total_evaluations > 0 for i in range(4))


def test_migrations_are_still_treated_as_migrations():
    """The is_seed flag must not weaken the real migration path."""
    db = AdaEvolveDatabase("t", AdaEvolveDatabaseConfig(num_islands=4, population_size=20))

    db.add(Program(id="p0", solution="a", metrics={"combined_score": 0.5}))
    db.add(
        Program(id="p1", solution="b", metrics={"combined_score": 0.9}),
        iteration=1,
        target_island=2,
    )

    assert db.adapter.states[2].total_evaluations == 0


def test_seeds_do_not_fill_the_paradigm_stagnation_window():
    """Regression: seeds must not count as evolutionary (non-)improvements.

    paradigm_window_size defaults to 10 and max_seed_programs to 16, so a
    seed pool that fed the tracker would report stagnation before iteration 1
    and spend a paradigm breakthrough — several guide-LLM calls — before the
    search had made a single edit.
    """
    db = AdaEvolveDatabase("t", AdaEvolveDatabaseConfig())
    assert db.use_paradigm_breakthrough  # the default this test depends on

    db.add(Program(id="primary", solution="p", metrics={"combined_score": 0.9}))
    for i in range(16):
        db.add(
            Program(id=f"s{i}", solution=f"s{i}", metrics={"combined_score": 0.1}),
            iteration=0,
            target_island=(i + 1) % db.num_islands,
            is_seed=True,
        )

    assert not db.is_paradigm_stagnating()


def test_migrations_still_do_not_pollute_the_paradigm_window():
    """The seed exclusion must not disturb the pre-existing migration rule."""
    db = AdaEvolveDatabase("t", AdaEvolveDatabaseConfig())
    db.add(Program(id="p", solution="p", metrics={"combined_score": 0.9}))
    before = len(db.paradigm_tracker.improvement_history)

    db.add(
        Program(id="m", solution="m", metrics={"combined_score": 0.1}),
        iteration=1,
        target_island=(db.current_island + 1) % db.num_islands,
    )

    assert len(db.paradigm_tracker.improvement_history) == before


def test_ordinary_children_still_reach_the_paradigm_tracker():
    """Guard against over-suppressing: normal adds must still be recorded."""
    db = AdaEvolveDatabase("t", AdaEvolveDatabaseConfig())
    db.add(Program(id="p", solution="p", metrics={"combined_score": 0.9}))
    before = len(db.paradigm_tracker.improvement_history)

    db.add(Program(id="c", solution="c", metrics={"combined_score": 0.95}), iteration=1)

    assert len(db.paradigm_tracker.improvement_history) == before + 1


# ----------------------------------------------------------------------
# Baseline reporting
# ----------------------------------------------------------------------


def test_initial_score_prefers_the_score_recorded_at_insertion():
    """The baseline must survive the seed being evicted from a capped population.

    Without this, `initial_score` re-resolved the program by scanning the
    database; once a pool exists, several programs share `iteration_found == 0`,
    so the fallback could report an unrelated program's score as the baseline.
    """
    runner = object.__new__(Runner)
    runner.initial_program_solution = "seed source"

    class Db:
        initial_program_id = "gone"
        initial_program_score = 0.10
        programs = {
            "other": Program(
                id="other", solution="a different program", metrics={"combined_score": 0.88}
            )
        }

    runner.database = Db()
    assert runner.initial_score == 0.10


def test_initial_score_falls_back_without_reporting_another_program():
    """With no recorded score and the seed gone, report nothing — not a stranger."""
    runner = object.__new__(Runner)
    runner.initial_program_solution = "seed source"

    other = Program(id="other", solution="a different program", metrics={"combined_score": 0.88})
    other.iteration_found = 0

    class Db:
        initial_program_id = None
        initial_program_score = None
        programs = {"other": other}

    runner.database = Db()
    assert runner.initial_score is None
