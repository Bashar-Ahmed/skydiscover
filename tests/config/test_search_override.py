"""Overriding search.type must not silently discard the user's database config."""

from skydiscover.config import (
    AdaEvolveDatabaseConfig,
    Config,
    GEPANativeDatabaseConfig,
    apply_overrides,
)


def test_base_fields_and_extras_survive_a_search_override():
    """db_path and untyped extras carry over to the new database class.

    Before this was fixed, `-s adaevolve` on a config written without an
    explicit search.type replaced the whole database config with defaults,
    nulling db_path — so the run silently lost persistence.
    """
    config = Config.from_dict(
        {
            "search": {
                "database": {
                    "population_size": 64,
                    "num_islands": 7,
                    "pareto_objectives": ["a", "b"],
                    "db_path": "/tmp/some_db",
                }
            }
        }
    )
    apply_overrides(config, search="adaevolve")

    db = config.search.database
    assert isinstance(db, AdaEvolveDatabaseConfig)
    assert db.db_path == "/tmp/some_db"
    assert db.population_size == 64
    assert db.num_islands == 7
    assert db.pareto_objectives == ["a", "b"]


def test_fields_owned_by_the_previous_algorithm_do_not_carry_over():
    """A value tuned for one strategy must not leak into another.

    Several strategies declare `population_size` with different meanings and
    scales, so carrying it across would swap a visible reset for a silent
    misconfiguration. Only shared base fields and unclaimed extras travel.
    """
    config = Config.from_dict(
        {
            "search": {
                "type": "adaevolve",
                "database": {"population_size": 64, "db_path": "/tmp/some_db"},
            }
        }
    )
    apply_overrides(config, search="gepa_native")

    db = config.search.database
    assert isinstance(db, GEPANativeDatabaseConfig)
    assert db.db_path == "/tmp/some_db"
    assert db.population_size == GEPANativeDatabaseConfig().population_size
    assert db.population_size != 64


def test_same_type_override_leaves_the_object_untouched():
    config = Config.from_dict(
        {"search": {"type": "adaevolve", "database": {"population_size": 64}}}
    )
    before = config.search.database
    apply_overrides(config, search="adaevolve")

    assert config.search.database is before
    assert config.search.database.population_size == 64


def test_override_still_sets_the_search_type():
    config = Config.from_dict({"search": {"type": "topk"}})
    apply_overrides(config, search="beam_search")
    assert config.search.type == "beam_search"
