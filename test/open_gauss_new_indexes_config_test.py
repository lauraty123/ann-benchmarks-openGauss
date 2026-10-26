"""The stock AnnBenchmark runner must discover the added openGauss methods."""

from pathlib import Path

import pytest

from ann_benchmarks.definitions import get_definitions, instantiate_algorithm

ALGORITHMS_DIR = str(Path(__file__).resolve().parents[1] / "ann_benchmarks" / "algorithms")


@pytest.mark.parametrize("metric", ["euclidean", "angular"])
def test_native_algorithm_registration(metric):
    expected = {
        "openGauss-diskann",
        "openGauss-hnsw-rabitq",
        "openGauss-ivfflat-rabitq",
    }
    definitions = [
        definition for definition in get_definitions(784, "float", metric, 10, base_dir=ALGORITHMS_DIR)
        if definition.algorithm in expected
    ]
    assert {definition.algorithm for definition in definitions} == expected
    assert len(definitions) == len(expected)
    for definition in definitions:
        algorithm = instantiate_algorithm(definition)
        try:
            assert type(algorithm).__module__ == "ann_benchmarks.algorithms.openGauss.module"
            assert len(definition.query_argument_groups) == 2
            if definition.algorithm == "openGauss-diskann":
                assert algorithm.params["statement_timeout_ms"] == 3600000
            for group in definition.query_argument_groups:
                assert len(group) == 1 and isinstance(group[0], dict)
                algorithm.normalize_query_arguments(group[0])
        finally:
            algorithm.done()
