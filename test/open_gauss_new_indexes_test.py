import json
import warnings
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.pool import ThreadPool
from threading import Barrier, Event, Lock
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from ann_benchmarks.algorithms.openGauss import module as m
from ann_benchmarks.algorithms.base import module as base
from ann_benchmarks import runner
from ann_benchmarks.plotting.metrics import queries_per_second


CLASSES = [m.OpenGaussHNSWRaBitQ, m.OpenGaussIVFFlatRaBitQ, m.OpenGaussDiskANN]


@pytest.fixture
def framework_workers(monkeypatch):
    """Exercise the real BaseANN implementation with a bounded test worker count."""
    pools = []

    def thread_pool():
        pool = ThreadPool(4)
        pools.append(pool)
        return pool

    monkeypatch.setattr(base, "ThreadPool", thread_pool)
    yield
    for pool in pools:
        pool.close()
        pool.join()


@pytest.mark.parametrize("cls", CLASSES)
def test_batch_hooks_are_inherited_from_framework(cls):
    assert cls.batch_query is base.BaseANN.batch_query
    assert cls.get_batch_results is base.BaseANN.get_batch_results
    for hook in ("prepare_batch_query", "run_batch_query", "get_batch_latencies"):
        assert not hasattr(cls, hook)


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("metric,operator,opclass", [
    ("euclidean", "<->", "vector_l2_ops"), ("angular", "<=>", "vector_cosine_ops")])
def test_distance_mapping(cls, metric, operator, opclass):
    algo = cls(metric, {})
    assert algo.operator == operator
    assert algo.opclass == opclass
    assert operator in algo.query_sql.as_string()
    assert "%s::vector" in algo.query_sql.as_string()


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("params", [{"unknown": 1}, {"concurrents": 0}, {"concurrents": True},
    {"parallel_workers": 0}, {"parallel_workers": 129}, {"enable_opfusion": "off"}, {"enable_seqscan": 0},
    {"statement_timeout_ms": 0}, {"lock_timeout_ms": 1.5}, {"schema": ""},
    {"maintenance_work_mem_mb": 0}, {"maintenance_work_mem_mb": True}, {"maintenance_work_mem_mb": "256"}])
def test_invalid_common_parameters(cls, params):
    with pytest.raises(ValueError):
        cls("euclidean", params)


@pytest.mark.parametrize("metric", ["dot", "l2", None, "hamming"])
def test_unsupported_distance(metric):
    with pytest.raises(ValueError):
        m.OpenGaussHNSWRaBitQ(metric, {})


@pytest.mark.parametrize("params", [{"m": 1}, {"m": 101}, {"m": 16.0}, {"m": "16"},
    {"ef_construction": 1001}, {"m": 32, "ef_construction": 63}, {"enable_rabitq": False},
    {"rabitq_refine_type": "PQ"}, {"rabitq_fht": "on"}, {"enable_pq": True}])
def test_invalid_hnsw_parameters(params):
    with pytest.raises(ValueError):
        m.OpenGaussHNSWRaBitQ("euclidean", params)


@pytest.mark.parametrize("value", [0, 32769, True, "20"])
def test_invalid_lists(value):
    with pytest.raises(ValueError):
        m.OpenGaussIVFFlatRaBitQ("euclidean", {"lists": value})


@pytest.mark.parametrize("params", [{"index_size": 15}, {"index_size": 1001}, {"pq_m": 0},
    {"pq_m": 193}, {"enable_pq": "false"}, {"enable_rabitq": True}])
def test_invalid_diskann_parameters(params):
    with pytest.raises(ValueError):
        m.OpenGaussDiskANN("euclidean", params)


@pytest.mark.parametrize("refine", ["fp32", "sq8", "none"])
def test_refine_types(refine):
    algo = m.OpenGaussHNSWRaBitQ("euclidean", {"rabitq_refine_type": refine})
    assert algo._index_options()["rabitq_refine_type"] == refine.upper()


@pytest.mark.parametrize("cls", CLASSES)
def test_not_fitted_and_unique_objects(cls):
    a, b = cls("euclidean", {}), cls("euclidean", {})
    assert a.table_name != b.table_name
    assert len(a.index_name.encode()) < 63
    assert a.get_memory_usage() == 0
    with pytest.raises(RuntimeError):
        a.query([1, 2], 1)
    a.done()
    a.done()


@pytest.mark.parametrize("data", [[], [1, 2], [[float("nan"), 1]], [[1, float("inf")]],
    [[1e100, 2]], [["1", "2"]], np.empty((0, 2)), np.empty((1, 2001)), [[1j, 2j]]])
def test_invalid_data_before_connect(data, monkeypatch):
    algo = m.OpenGaussHNSWRaBitQ("euclidean", {})
    connect = MagicMock()
    monkeypatch.setattr(algo, "_connect", connect)
    with pytest.raises(ValueError):
        algo.fit(data)
    connect.assert_not_called()


def test_dimension_and_pq_divisibility():
    with pytest.raises(ValueError):
        m.OpenGaussDiskANN("euclidean", {})._validate_matrix(np.ones((100, 1537)), fitting=True)
    with pytest.raises(ValueError):
        m.OpenGaussDiskANN("euclidean", {"enable_pq": True, "pq_m": 8})._validate_matrix(np.ones((100, 15)), fitting=True)
    with pytest.raises(ValueError):
        m.OpenGaussIVFFlatRaBitQ("euclidean", {"lists": 101})._validate_matrix(np.ones((100, 64)), fitting=True)
    with pytest.raises(ValueError):
        m.OpenGaussHNSWRaBitQ("angular", {})._validate_matrix(np.zeros((100, 64)), fitting=True)
    with pytest.raises(ValueError):
        m.OpenGaussHNSWRaBitQ("angular", {})._validate_matrix(np.ones((100, 1)), fitting=True)


@pytest.mark.parametrize("key,value", [("hnsw_ef_search", 0), ("hnsw_ef_search", 1000001),
    ("rbq_query_bits", 0), ("rbq_query_bits", 9), ("rbq_query_bits", 1.5),
    ("rbq_refinek", -1), ("rbq_refinek", 1001), ("rbq_refinek", float("nan")),
    ("rbq_refinek", True)])
def test_query_bounds(key, value):
    algo = m.OpenGaussHNSWRaBitQ("euclidean", {})
    params = {**algo.query_defaults, key: value}
    with pytest.raises(ValueError):
        algo._validate_query(params)


def test_fractional_refinek_and_source_bounds():
    algo = m.OpenGaussHNSWRaBitQ("euclidean", {"m": 100, "ef_construction": 1000})
    params = {"hnsw_ef_search": 1000000, "rbq_query_bits": 1, "rbq_refinek": 2.5}
    algo._validate_query(params)
    assert params["rbq_refinek"] == 2.5
    params["rbq_refinek"] = 0
    algo._validate_query(params)
    with pytest.raises(ValueError):
        m.OpenGaussIVFFlatRaBitQ("euclidean", {"lists": 3})._validate_query(
            {"ivfflat_probes": 4, "rbq_query_bits": 8, "rbq_refinek": 5})
    assert m.OpenGaussIVFFlatRaBitQ("euclidean", {"lists": 1}).query_params["ivfflat_probes"] == 1


def fitted_mock(cls=m.OpenGaussHNSWRaBitQ, params=None, metric="euclidean"):
    algo = cls(metric, params or {})
    algo.connection = MagicMock(closed=False, broken=False)
    algo._connect = MagicMock(side_effect=lambda: MagicMock(closed=False, broken=False))
    algo._created = True
    algo._dimension = 2
    algo._count = 100
    algo._probe = np.ones(2, dtype=np.float32)
    return algo


def test_parameter_group_reset_and_validation(monkeypatch):
    algo = fitted_mock()
    monkeypatch.setattr(algo, "_apply_settings", MagicMock())
    monkeypatch.setattr(algo, "_record_settings", MagicMock())
    monkeypatch.setattr(algo, "explain", MagicMock())
    old_pool = algo.search_pool
    old_pool.close = MagicMock()
    algo.set_query_arguments({"hnsw_ef_search": 50, "rbq_query_bits": 2, "rbq_refinek": 2.5})
    old_pool.close.assert_called_once()
    connections = list(algo.search_pool.connections)
    assert len(connections) == algo.params["concurrents"]
    algo.set_query_arguments({"hnsw_ef_search": 80})
    for connection in connections:
        connection.close.assert_called_once()
        assert connection not in algo.search_pool.connections
    assert algo.query_params == {"hnsw_ef_search": 80, "rbq_query_bits": 8, "rbq_refinek": 5.0}
    before = dict(algo.query_params)
    with pytest.raises(ValueError):
        algo.set_query_arguments({"unknown": 1})
    assert algo.query_params == before
    algo.done()


@pytest.mark.parametrize("value,k", [([1], 1), ([1, 2], 0), ([1, 2], 101),
    ([1, 2], True), ([float("nan"), 2], 1), ([[1, 2]], 1)])
def test_invalid_query_shape_and_k(value, k):
    with pytest.raises(ValueError):
        fitted_mock()._query_vector(value, k)


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("batch", [False, True])
def test_sql_parameters_and_bad_result_detection(framework_workers, cls, batch):
    algo = fitted_mock(cls)
    algo.search_pool.prepare()
    cur = algo.search_pool.connections[0].cursor.return_value.__enter__.return_value

    def query():
        if batch:
            algo.batch_query(np.array([[1, 2]], dtype=np.float32), 2)
            return algo.get_batch_results()[0]
        return algo.query([1, 2], 2)

    try:
        cur.fetchall.return_value = [(3,), (1,)]
        assert query() == [3, 1]
        call = cur.execute.call_args
        vector, count = call.args[1]
        np.testing.assert_array_equal(vector, np.array([1, 2], dtype=np.float32))
        assert vector.dtype == np.float32 and count == 2
        assert call.kwargs == {"binary": True, "prepare": True}
        assert "%s::vector" in call.args[0].as_string()
        # Unlike the runner's single-only warning, guards also cover batches.
        for rows in [[(-1,), (1,)], [(100,), (1,)], [(1,), (2,), (3,)]]:
            cur.fetchall.return_value = rows
            with pytest.raises(RuntimeError, match="too many or out-of-range IDs"):
                query()
        for rows in [[], [(1,)], [(3,), (3,)]]:
            cur.fetchall.return_value = rows
            assert query() == [r[0] for r in rows]
    finally:
        algo.done()


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("batch", [False, True])
def test_runner_converts_driver_ids_outside_timing(monkeypatch, framework_workers, cls, batch):
    clock = MagicMock(side_effect=[10.0, 11.0])
    conversion_clock_counts = []

    class DriverID(int):
        def __int__(self):
            conversion_clock_counts.append(clock.call_count)
            return super().__int__()

    algo = fitted_mock(cls)
    algo.search_pool.prepare()
    cur = algo.search_pool.connections[0].cursor.return_value.__enter__.return_value
    candidate = DriverID(1)
    cur.fetchall.return_value = [(candidate,)]
    monkeypatch.setattr(algo, "get_memory_usage", lambda: 0)
    monkeypatch.setattr(runner, "time", SimpleNamespace(time=clock))
    data = np.ones((2, 2), dtype=np.float32)
    try:
        # The adapter only extracts the driver-decoded integer, without coercion.
        assert algo.query(data[0], 1)[0] is candidate
        assert conversion_clock_counts == []
        _, results = runner.run_individual_query(algo, data, data[:1], "euclidean", 1, 1, batch)
        assert conversion_clock_counts == [2]  # after the runner stops its timer
        result_id = results[0][1][0][0]
        assert type(result_id) is int and result_id == 1
    finally:
        algo.done()


def test_normalize_query_uses_effective_ivf_default():
    algo = m.OpenGaussIVFFlatRaBitQ("euclidean", {"lists": 1})
    assert algo.normalize_query_arguments({"rbq_query_bits": 4}) == {
        "ivfflat_probes": 1, "rbq_query_bits": 4, "rbq_refinek": 5.0}
    with pytest.raises(ValueError, match="Unknown"):
        algo.normalize_query_arguments({"typo": 1})


@pytest.mark.parametrize("batch", [False, True])
def test_runner_rejects_duplicate_candidates_outside_timing(monkeypatch, framework_workers, batch):
    algo = fitted_mock()
    algo.search_pool.prepare()
    cur = algo.search_pool.connections[0].cursor.return_value.__enter__.return_value
    cur.fetchall.return_value = [(0,), (0,)]
    # The adapter passes duplicates through; the unchanged runner rejects them.
    assert algo.query([1, 2], 2) == [0, 0]
    clock = MagicMock(side_effect=[10.0, 11.0])
    monkeypatch.setattr(runner, "time", SimpleNamespace(time=clock))
    data = np.ones((2, 2), dtype=np.float32)
    try:
        with pytest.raises(AssertionError, match="Implementation returned duplicated candidates"):
            runner.run_individual_query(algo, data, data[:1], "euclidean", 2, 1, batch)
        assert clock.call_count == 2
    finally:
        algo.done()


@pytest.mark.parametrize("cls", CLASSES)
def test_fit_uses_driver_copy_rows(monkeypatch, cls):
    algo = cls("euclidean", {"lists": 1} if cls is m.OpenGaussIVFFlatRaBitQ else {})
    connection = MagicMock(closed=False, broken=False)
    cur = connection.cursor.return_value.__enter__.return_value
    cur.fetchone.side_effect = [("openGauss test build",), ("ann", "ann")]
    monkeypatch.setattr(algo, "_connect", MagicMock(return_value=connection))
    monkeypatch.setattr(algo, "_record_settings", MagicMock())
    monkeypatch.setattr(algo, "explain", MagicMock())
    data = np.array([[1.23456789, -0.000000123456789], [1e30, -1e-30]], dtype=np.float64)
    try:
        algo.fit(data)
        copy = cur.copy.return_value.__enter__.return_value
        assert "FROM STDIN" in cur.copy.call_args.args[0].as_string()
        assert "BINARY" not in cur.copy.call_args.args[0].as_string()
        copy.write.assert_not_called()
        assert copy.write_row.call_count == len(data)
        for i, call in enumerate(copy.write_row.call_args_list):
            row_id, vector = call.args[0]
            assert row_id == i and vector.dtype == np.float32
            np.testing.assert_array_equal(vector, data[i].astype(np.float32))
    finally:
        algo.done()


@pytest.mark.parametrize("empty_value", [None, "", "  "])
def test_connection_defaults_match_existing_adapters(monkeypatch, empty_value):
    for key in ("USER", "PASSWORD", "DBNAME", "HOST", "PORT"):
        if empty_value is None:
            monkeypatch.delenv("ANN_BENCHMARKS_OG_" + key, raising=False)
        else:
            monkeypatch.setenv("ANN_BENCHMARKS_OG_" + key, empty_value)
    connect = MagicMock()
    monkeypatch.setattr(m.psycopg, "connect", connect)
    register = MagicMock()
    monkeypatch.setattr(m.pgvector.psycopg, "register_vector", register)
    algo = m.OpenGaussDiskANN("euclidean", {})
    assert algo._connect() is connect.return_value
    assert "host" not in connect.call_args.kwargs
    assert "port" not in connect.call_args.kwargs
    for key in ("user", "password", "dbname"):
        assert connect.call_args.kwargs[key] == "ann"
    register.assert_called_once_with(connect.return_value)
    connect.return_value.close.assert_not_called()


def test_connection_environment_overrides_defaults(monkeypatch):
    values = {"user": "test-user", "password": "test-placeholder", "dbname": "test-db",
              "host": "/tmp/test-socket", "port": "55433"}
    for key, value in values.items():
        monkeypatch.setenv(m.get_pg_param_env_var_name(key), value)
    connect = MagicMock()
    monkeypatch.setattr(m.psycopg, "connect", connect)
    register = MagicMock()
    monkeypatch.setattr(m.pgvector.psycopg, "register_vector", register)
    algo = m.OpenGaussDiskANN("euclidean", {})
    algo._connect()
    for key, value in values.items():
        assert connect.call_args.kwargs[key] == (int(value) if key == "port" else value)
    register.assert_called_once_with(connect.return_value)


@pytest.mark.parametrize("error", [RuntimeError("registration failed"), KeyboardInterrupt()])
def test_registration_failure_closes_new_connection(monkeypatch, error):
    connect = MagicMock()
    monkeypatch.setattr(m.psycopg, "connect", connect)
    monkeypatch.setattr(m.pgvector.psycopg, "register_vector", MagicMock(side_effect=error))
    with pytest.raises(type(error)):
        m.OpenGaussDiskANN("euclidean", {})._connect()
    connect.return_value.close.assert_called_once()


def test_done_drops_only_owned_table_and_is_idempotent():
    algo = fitted_mock()
    connection = algo.connection
    cur = connection.cursor.return_value.__enter__.return_value
    algo.done()
    statement = cur.execute.call_args.args[0].as_string()
    assert algo.table_name in statement
    assert "CASCADE" not in statement and "items" not in statement
    assert not algo._created and algo.connection is None
    connection.close.assert_called_once()
    algo.done()


def test_cleanup_reconnects_if_query_connection_broken(monkeypatch):
    algo = fitted_mock()
    algo.connection.broken = True
    fresh = MagicMock(closed=False, broken=False)
    monkeypatch.setattr(algo, "_connect", MagicMock(return_value=fresh))
    algo.done()
    assert not algo._created
    fresh.close.assert_called_once()


def test_fit_error_closes_connection_and_cleans_created_object(monkeypatch):
    algo = m.OpenGaussHNSWRaBitQ("euclidean", {})
    connection = MagicMock(closed=False, broken=False)
    cur = connection.cursor.return_value.__enter__.return_value
    def execute(statement, *args, **kwargs):
        if "CREATE INDEX" in statement.as_string():
            raise RuntimeError("injected index build failure")
    cur.execute.side_effect = execute
    monkeypatch.setattr(algo, "_connect", MagicMock(return_value=connection))
    with pytest.raises(RuntimeError, match="injected index"):
        algo.fit(np.ones((100, 2), dtype=np.float32))
    statements = [c.args[0].as_string() for c in cur.execute.call_args_list]
    assert any(s.startswith("DROP TABLE") for s in statements)
    assert not algo._created
    connection.close.assert_called_once()


def test_pool_failed_prepare_closes_all_partial_connections(monkeypatch):
    algo = m.OpenGaussDiskANN("euclidean", {"concurrents": 2})
    connection = MagicMock()
    monkeypatch.setattr(algo, "_connect", MagicMock(side_effect=[connection, RuntimeError("connect failure")]))
    monkeypatch.setattr(algo, "_apply_settings", MagicMock())
    with pytest.raises(RuntimeError, match="connect failure"):
        algo.search_pool.prepare()
    connection.close.assert_called_once()
    assert not algo.search_pool.connections
    algo.search_pool.close()


def test_failed_session_initialization_can_be_retried(monkeypatch):
    algo = fitted_mock(params={"concurrents": 2})
    monkeypatch.setattr(algo, "_apply_settings", MagicMock(side_effect=RuntimeError("SET failed")))
    with pytest.raises(RuntimeError, match="SET failed"):
        algo.search_pool.prepare()
    assert not algo.search_pool.connections and not algo.search_pool.available
    monkeypatch.setattr(algo, "_apply_settings", MagicMock())
    algo.search_pool.prepare()
    assert len(algo.search_pool.connections) == 2
    assert len(algo.search_pool.available) == 2
    algo.done()


@pytest.mark.parametrize("cls", CLASSES)
def test_fit_after_done_creates_fresh_query_sessions(monkeypatch, cls):
    algo = cls("euclidean", {})

    def connect():
        connection = MagicMock(closed=False, broken=False)
        connection.cursor.return_value.__enter__.return_value.fetchone.side_effect = [
            ("openGauss test build",), ("ann", "ann")]
        return connection

    monkeypatch.setattr(algo, "_connect", MagicMock(side_effect=connect))
    monkeypatch.setattr(algo, "_record_settings", MagicMock())
    monkeypatch.setattr(algo, "explain", MagicMock())
    monkeypatch.setattr(algo, "_execute_query", lambda connection, vector, n: [0])
    for _ in range(2):
        algo.fit(np.ones((100, 2), dtype=np.float32))
        assert algo.query([1, 1], 1) == [0]
        algo.done()
        assert not algo.search_pool.connections


def test_effective_settings_applied_to_all_workers(monkeypatch):
    algo = m.OpenGaussHNSWRaBitQ("euclidean", {"concurrents": 2, "enable_opfusion": False,
                                             "maintenance_work_mem_mb": 128})
    connections = [MagicMock(), MagicMock()]
    monkeypatch.setattr(algo, "_connect", MagicMock(side_effect=connections))
    algo.search_pool.prepare()
    for connection in connections:
        calls = connection.cursor.return_value.__enter__.return_value.execute.call_args_list
        statements = [c.args[0].as_string() for c in calls]
        assert any('"hnsw_ef_search" = 100' in s for s in statements)
        assert any('"enable_opfusion" = false' in s for s in statements)
        assert any('"rbq_refinek" = 5.0' in s for s in statements)
        assert any('"maintenance_work_mem" = \'128MB\'' in s for s in statements)
    algo.search_pool.close()
    for connection in connections:
        connection.close.assert_called_once()


def test_explain_rejects_fallback():
    algo = fitted_mock()
    cur = algo.connection.cursor.return_value.__enter__.return_value
    cur.fetchall.return_value = [("Seq Scan on another_table",)]
    with pytest.raises(RuntimeError, match="planner did not"):
        algo.explain([1, 2], 10)


def test_settings_record_includes_units():
    algo = fitted_mock()
    cur = algo.connection.cursor.return_value.__enter__.return_value
    cur.fetchall.return_value = [("maintenance_work_mem", "262144", "kB"),
                                 ("shared_buffers", "16384", "8kB"), ("enable_seqscan", "off", None)]
    algo._record_settings()
    assert json.loads(algo._metadata["effective_settings_json"])["shared_buffers"] == "16384"
    assert json.loads(algo._metadata["effective_setting_units_json"])["shared_buffers"] == "8kB"


def test_batch_empty_and_order(monkeypatch, framework_workers):
    algo = fitted_mock()
    monkeypatch.setattr(algo, "_connect", MagicMock(side_effect=lambda: MagicMock()))
    monkeypatch.setattr(algo, "_apply_settings", MagicMock())
    monkeypatch.setattr(algo, "_execute_query", lambda conn, v, n: [int(v[0])])
    algo.batch_query(np.empty((0, 2)), 1)
    assert algo.get_batch_results() == []
    algo.batch_query(np.array([[3, 1], [2, 1], [1, 1]], dtype=np.float32), 1)
    assert algo.get_batch_results() == [[3], [2], [1]]
    assert not hasattr(algo, "get_batch_latencies")
    algo.search_pool.close()


def test_runner_cleans_connections_after_batch_failure(monkeypatch, framework_workers):
    algo = fitted_mock()
    monkeypatch.setattr(algo, "_apply_settings", MagicMock())
    monkeypatch.setattr(algo, "_execute_query", MagicMock(side_effect=RuntimeError("injected query failure")))
    monkeypatch.setattr(algo, "fit", MagicMock())
    monkeypatch.setattr(algo, "get_memory_usage", lambda: 0)
    monkeypatch.setattr(runner, "instantiate_algorithm", lambda definition: algo)
    data = np.ones((5, 2), dtype=np.float32)
    monkeypatch.setattr(runner, "load_and_transform_dataset", lambda name: (data, data, "euclidean"))
    algo.search_pool.prepare()
    connections = list(algo.search_pool.connections)
    build_connection = algo.connection
    definition = SimpleNamespace(query_argument_groups=[], algorithm="openGauss-hnsw-rabitq")
    with pytest.raises(RuntimeError, match="injected query"):
        runner.run(definition, "test-data", 1, 1, True)
    assert not algo.search_pool.connections
    assert not algo.search_pool.available
    for connection in connections + [build_connection]:
        connection.close.assert_called_once()
    assert not algo._created


@pytest.mark.parametrize("concurrents", [1, 2, 4, 8])
def test_framework_batch_exclusively_leases_bounded_connections(monkeypatch, framework_workers, concurrents):
    algo = fitted_mock(params={"concurrents": concurrents})
    monkeypatch.setattr(algo, "_apply_settings", MagicMock())
    barrier = Barrier(min(concurrents, 4))
    lock = Lock()
    active = set()
    high_water_mark = 0

    def execute(connection, vector, n):
        nonlocal high_water_mark
        with lock:
            assert id(connection) not in active
            active.add(id(connection))
            high_water_mark = max(high_water_mark, len(active))
        try:
            barrier.wait(timeout=10)
            return [int(vector[0])]
        finally:
            with lock:
                active.remove(id(connection))

    monkeypatch.setattr(algo, "_execute_query", execute)
    data = np.array([[i, 1] for i in range(8)], dtype=np.float32)
    try:
        algo.batch_query(data, 1)
        assert algo.get_batch_results() == [[i] for i in range(8)]
        assert high_water_mark == min(concurrents, 4)
        assert not active
        assert algo._connect.call_count == concurrents
        assert len(algo.search_pool.available) == concurrents
    finally:
        algo.done()


def test_query_exception_returns_its_connection(monkeypatch):
    algo = fitted_mock()
    monkeypatch.setattr(algo, "_apply_settings", MagicMock())
    execute = MagicMock(side_effect=[RuntimeError("query failed"), [0]])
    monkeypatch.setattr(algo, "_execute_query", execute)
    with pytest.raises(RuntimeError, match="query failed"):
        algo.query([1, 2], 1)
    assert len(algo.search_pool.available) == 1
    assert algo.query([1, 2], 1) == [0]
    assert execute.call_args_list[0].args[0] is execute.call_args_list[1].args[0]
    algo.done()


def test_connection_close_waits_for_active_query_and_rejects_new_leases(monkeypatch):
    algo = fitted_mock()
    monkeypatch.setattr(algo, "_apply_settings", MagicMock())
    borrowed, release = Event(), Event()
    sessions = algo.search_pool
    sessions.prepare()
    connection = sessions.connections[0]

    def hold_connection():
        with sessions.connection() as leased:
            assert leased is connection
            borrowed.set()
            assert release.wait(timeout=10)

    with ThreadPoolExecutor(max_workers=2) as executor:
        query = executor.submit(hold_connection)
        try:
            assert borrowed.wait(timeout=10)
            closing = executor.submit(sessions.close)
            with sessions._condition:
                assert sessions._condition.wait_for(lambda: sessions._closed, timeout=10)
            assert not closing.done()
            connection.close.assert_not_called()
            with pytest.raises(RuntimeError, match="connections are closed"):
                with sessions.connection():
                    pytest.fail("a closed session manager must not admit new queries")
        finally:
            release.set()
        query.result(timeout=10)
        closing.result(timeout=10)
    connection.close.assert_called_once()
    sessions.close()
    connection.close.assert_called_once()
    algo.done()


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("metric", ["euclidean", "angular"])
@pytest.mark.parametrize("batch", [False, True])
def test_standard_runner_owns_query_timing(monkeypatch, framework_workers, cls, metric, batch):
    algo = fitted_mock(cls, {"concurrents": 2}, metric)
    monkeypatch.setattr(algo, "_apply_settings", MagicMock())
    monkeypatch.setattr(algo, "_record_settings", MagicMock())
    monkeypatch.setattr(algo, "explain", MagicMock())
    monkeypatch.setattr(algo, "get_memory_usage", lambda: 0)
    monkeypatch.setattr(algo, "_execute_query", lambda connection, vector, n: [int(vector[0])])
    algo.set_query_arguments({})
    connections = list(algo.search_pool.connections)
    data = np.array([[i, 1] for i in range(4)], dtype=np.float32)
    if batch:
        # The unmodified runner measures two whole batches: 2 s, then 1 s.
        timestamps = [10.0, 12.0, 20.0, 21.0]
    else:
        # In single mode it measures each query: 0.5 s, then 0.25 s per run.
        timestamps = [t for start, duration in ((10.0, 0.5), (20.0, 0.25))
                      for i in range(len(data)) for t in (start + i, start + i + duration)]
    clock = MagicMock(side_effect=timestamps)
    monkeypatch.setattr(runner, "time", SimpleNamespace(time=clock))
    try:
        attrs, results = runner.run_individual_query(algo, data, data, metric, 1, 2, batch)
        assert attrs["best_search_time"] == pytest.approx(1.0 / len(data))
        assert queries_per_second([], attrs) == pytest.approx(len(data) / 1.0)
        assert [latency for latency, _ in results] == [0.25] * len(data)
        assert [candidates[0][0] for _, candidates in results] == list(range(4))
        assert "batch_qps" not in attrs and "batch_wall_time_s" not in attrs
        assert attrs["latency_definition"].startswith("runner wall time:")
        assert clock.call_count == len(timestamps)
        assert algo._connect.call_count == 2  # no new sessions inside the timed queries
        for connection in connections:
            connection.close.assert_called_once()
        # The next parameter group recreates sessions after the runner's close.
        algo.set_query_arguments({})
        assert algo._connect.call_count == 4
        assert all(connection not in connections for connection in algo.search_pool.connections)
    finally:
        algo.done()


@pytest.mark.parametrize("build", ["2397a86f", "e024ee66"])
@pytest.mark.parametrize("cls,metric,refine,expected", [
    (m.OpenGaussIVFFlatRaBitQ, "euclidean", "FP32", "ivf-rabitq-refinement"),
    (m.OpenGaussIVFFlatRaBitQ, "angular", "SQ8", "ivf-rabitq-refinement"),
    (m.OpenGaussHNSWRaBitQ, "angular", "FP32", "hnsw-rabitq-cosine-refinement"),
    (m.OpenGaussHNSWRaBitQ, "angular", "SQ8", "hnsw-rabitq-cosine-refinement"),
    (m.OpenGaussHNSWRaBitQ, "euclidean", "FP32", None),
    (m.OpenGaussHNSWRaBitQ, "angular", "NONE", None),
    (m.OpenGaussIVFFlatRaBitQ, "euclidean", "NONE", None),
])
def test_known_server_risk_scope_and_no_parameter_substitution(build, cls, metric, refine, expected):
    algo = cls(metric, {"rabitq_refine_type": refine})
    algo._metadata["database_version"] = f"(openGauss 7.0.0 build {build})"
    before = (dict(algo.params), dict(algo.query_params))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        algo._record_known_server_risks()
        algo._record_known_server_risks()
    assert json.loads(algo._metadata["known_server_risks_json"]) == ([expected] if expected else [])
    assert len(caught) == (1 if expected else 0)
    if expected:
        assert "requested parameters and returned order are unchanged" in str(caught[0].message)
    assert (algo.params, algo.query_params) == before


@pytest.mark.parametrize("version", ["", "openGauss 8.0.0 build abc12345", "build 2397a86f0"])
def test_unknown_server_build_is_not_declared_affected(version):
    algo = m.OpenGaussIVFFlatRaBitQ("euclidean", {})
    algo._metadata["database_version"] = version
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        algo._record_known_server_risks()
    assert not caught
    assert json.loads(algo._metadata["known_server_risks_json"]) == []


def test_risk_warning_when_query_group_enables_refinement(monkeypatch):
    algo = fitted_mock(m.OpenGaussIVFFlatRaBitQ)
    algo._metadata["database_version"] = "(openGauss 7.0.0 build 2397a86f)"
    monkeypatch.setattr(algo, "_apply_settings", MagicMock())
    monkeypatch.setattr(algo, "_record_settings", MagicMock())
    monkeypatch.setattr(algo, "explain", MagicMock())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        algo.set_query_arguments({"rbq_refinek": 0})
    assert not caught
    assert json.loads(algo._metadata["known_server_risks_json"]) == []
    with pytest.warns(RuntimeWarning, match="independent gsql"):
        algo.set_query_arguments({"rbq_refinek": 5})
    assert algo.query_params["rbq_refinek"] == 5
    assert json.loads(algo._metadata["known_server_risks_json"]) == ["ivf-rabitq-refinement"]
    algo.set_query_arguments({"rbq_refinek": 0})
    assert json.loads(algo._metadata["known_server_risks_json"]) == []
