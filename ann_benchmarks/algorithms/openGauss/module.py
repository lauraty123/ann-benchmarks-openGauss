"""
This module supports connecting to a openGauss instance and performing vector
indexing and search using the pgvector extension. The default behavior uses
the "ann" value of openGauss user name, password, and database name, as well
as the default host and port values of the psycopg driver.

If openGaussis managed externally, e.g. in a cloud DBaaS environment, the
environment variable overrides listed below are available for setting openGauss
connection parameters:

ANN_BENCHMARKS_OG_USER
ANN_BENCHMARKS_OG_PASSWORD
ANN_BENCHMARKS_OG_DBNAME
ANN_BENCHMARKS_OG_HOST
ANN_BENCHMARKS_OG_PORT

This module starts the openGauss service automatically using the "service"
command. The environment variable ANN_BENCHMARKS_OG_START_SERVICE could be set
to "false" (or e.g. "0" or "no") in order to disable this behavior.

This module will also attempt to create the pgvector extension inside the
target database, if it has not been already created.

The DiskANN and RaBitQ adapters below use an externally managed openGauss
instance. They do not start a service or create an extension.
"""
import json
import math
import re
import subprocess
import sys
import os
import uuid
import warnings
from contextlib import contextmanager
from numbers import Integral, Real
from threading import Condition

import pgvector.psycopg
import psycopg
from psycopg import sql

from typing import Dict, Any, Optional

from ..base.module import BaseANN
from ...util import get_bool_env_var

import multiprocessing
from multiprocessing import Pool
import numpy
import numpy as np
import time

def get_pg_param_env_var_name(pg_param_name: str) -> str:
    return f'ANN_BENCHMARKS_OG_{pg_param_name.upper()}'


def get_pg_conn_param(
        pg_param_name: str,
        default_value: Optional[str] = None) -> Optional[str]:
    env_var_name = get_pg_param_env_var_name(pg_param_name)
    env_var_value = os.getenv(env_var_name, default_value)
    if env_var_value is None or len(env_var_value.strip()) == 0:
        return default_value
    return env_var_value


class openGaussHNSW(BaseANN):
    def __init__(self, metric, method_param):
        self._metric = metric
        self._m = method_param['M']
        self._ef_construction = method_param['efConstruction']
        self._cur = None
        self.curs = []
        self.connections = []
        self.init_conn_flags = False
        self.search_pool = None
        self.con = method_param['concurrents']

        # default value
        self.user = 'ann'
        self.port = 5432
        self.dbname = 'ann'
        self.host = 'localhost'
        self.password = 'ann'

        if metric == "angular":
            self._query = "SELECT id FROM items ORDER BY embedding <=> %s LIMIT %s"
        elif metric == "euclidean":
            self._query = "SELECT id FROM items ORDER BY embedding <-> %s LIMIT %s"
        else:
            raise RuntimeError(f"unknown metric {metric}")

    def fit(self, X):
        psycopg_connect_kwargs: Dict[str, Any] = dict(
            autocommit=True,
        )
        for arg_name in ['user', 'password', 'dbname']:
            # The default value is "ann" for all of these parameters.
            psycopg_connect_kwargs[arg_name] = get_pg_conn_param(
                arg_name, 'ann')
            if arg_name == 'user':
                self.user = psycopg_connect_kwargs['user']
            if arg_name == 'password':
                self.password = psycopg_connect_kwargs['password']
            if arg_name == 'dbname':
                self.dbname = psycopg_connect_kwargs['dbname']

        # If host/port are not specified, leave the default choice to the
        # psycopg driver.
        og_host: Optional[str] = get_pg_conn_param('host')
        if og_host is not None:
            psycopg_connect_kwargs['host'] = og_host
            self.host = og_host

        og_port_str: Optional[str] = get_pg_conn_param('port')
        if og_port_str is not None:
            psycopg_connect_kwargs['port'] = int(og_port_str)
            self.port = int(og_port_str)

        conn = psycopg.connect(**psycopg_connect_kwargs)

        pgvector.psycopg.register_vector(conn)
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS items")
        cur.execute("CREATE TABLE items (id int, embedding vector(%d))" % X.shape[1])
        cur.execute("ALTER TABLE items ALTER COLUMN embedding SET STORAGE PLAIN")
        cur.execute("ALTER TABLE items SET (parallel_workers = 32);")
        print("copying data...")
        with cur.copy("COPY items (id, embedding) FROM STDIN WITH (FORMAT BINARY)") as copy:
            copy.set_types(["int4", "vector"])
            for i, embedding in enumerate(X):
                copy.write_row((i, embedding))
        print("creating index...")

        if self._metric == "angular":
            cur.execute(
                "CREATE INDEX ON items USING hnsw (embedding vector_cosine_ops) WITH (m = %d, ef_construction = %d)" % (self._m, self._ef_construction)
            )
        elif self._metric == "euclidean":
            cur.execute("CREATE INDEX ON items USING hnsw (embedding vector_l2_ops) WITH (m = %d, ef_construction = %d)" % (self._m, self._ef_construction))
        else:
            raise RuntimeError(f"unknown metric {self._metric}")
        print("done!")
        self._cur = cur


    def set_query_arguments(self, ef_search):
        self._ef_search = ef_search
        self._cur.execute("SET hnsw_ef_search = %d" % ef_search)
        self._cur.execute("set enable_seqscan = off")
        conc = self.con
        self.search_pool = Pool(conc, initializer=self.init_connection)

    def query(self, v, n):
        self._cur.execute(self._query, (v, n), binary=True, prepare=True)
        return [id for id, in self._cur.fetchall()]

    def get_memory_usage(self):
        if self._cur is None:
            return 0
        self._cur.execute("SELECT pg_relation_size('items_embedding_idx')")
        return self._cur.fetchone()[0] / 1024
    
    # batch search
    def init_connection(self):
        conn = psycopg.connect(user=self.user, password=self.password, dbname=self.dbname, host=self.host, autocommit=True, port=self.port)
        pgvector.psycopg.register_vector(conn)
        global cur
        cur = conn.cursor()
        cur.execute("SET hnsw_ef_search = %d" % self._ef_search)
        cur.execute("set enable_seqscan = off")
        global base_query
        base_query = self._query
        self.connections.append(conn)
        self.curs.append(cur)

    def close_connections(self):
        for cur in self.curs:
            if cur is not None:
                cur.close()
                cur = None
        for conn in self.connections:
            if conn is not None:
                conn.close()
                conn = None
        self.curs = []
        self.connections = []
        self.init_conn_flags = False
    
    @staticmethod
    def sub_query(chunk, n):
        ids = []
        for item in chunk:
            cur.execute(base_query, (item, n), binary=True, prepare=True)
            res = cur.fetchall()
            ids.append([i[0] for i in res])
        return ids
    
    def batch_query(self, X: numpy.array, n: int):
        conc = self.con
        chunk_size = len(X) // conc + 1
        chunks = [X[i:i + chunk_size] for i in range(0, len(X), chunk_size)]
        res = self.search_pool.starmap(self.sub_query, [(chunk, n) for chunk in chunks])
        self.res = []
        for item in res:
            for ids in item:
                self.res.append(ids)

    def __str__(self):
        return f"openGaussHNSW(m={self._m}, ef_construction={self._ef_construction}, ef_search={self._ef_search})"

    def __del__(self):
        self.close_connections()

# Database-side vector indexes through the existing BaseANN runner interface.
# Only tables created by these adapters are dropped during cleanup.

def integer(name, value, lower, upper):
    if isinstance(value, bool) or not isinstance(value, Integral) or not lower <= value <= upper:
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
    return int(value)


def real(name, value, lower, upper):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number in [{lower}, {upper}]")
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}]")
    return float(value)


def boolean(name, value):
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a YAML/JSON boolean, not a string")
    return value


class _QueryConnections:
    """Database sessions for BaseANN's existing batch-query workers.

    BaseANN owns task scheduling, result ordering and batching; the runner owns
    timing. Its default ThreadPool has no connection initializer or configurable
    worker count, so this session manager only limits active database queries to
    ``concurrents`` and gives each query exclusive use of an initialized session.
    Actual parallelism is also bounded by the framework's worker count.
    The adapter exposes this as ``search_pool`` for the existing runner's
    close() hook. Connections are closed after the framework's workers finish.
    """

    def __init__(self, adapter):
        self.adapter = adapter
        self.connections = []
        self.available = []
        self._condition = Condition()
        self._closed = False

    def prepare(self):
        with self._condition:
            if self._closed:
                raise RuntimeError("query connections are closed; call set_query_arguments() before querying again")
            if self.connections:
                return
            try:
                for _ in range(self.adapter.params["concurrents"]):
                    connection = self.adapter._connect()
                    self.connections.append(connection)
                    self.adapter._apply_settings(connection, self.adapter.query_params)
            except BaseException:
                for connection in self.connections:
                    connection.close()
                self.connections = []
                raise
            self.available = list(self.connections)

    @contextmanager
    def connection(self):
        self.prepare()
        with self._condition:
            while not self.available and not self._closed:
                self._condition.wait()
            if self._closed:
                raise RuntimeError("query connections are closed")
            connection = self.available.pop()
        try:
            yield connection
        finally:
            with self._condition:
                self.available.append(connection)
                self._condition.notify_all()

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()
            # Also handle runner cleanup after an interrupted batch: reject new
            # leases and let in-flight queries return before closing sessions.
            while len(self.available) < len(self.connections):
                self._condition.wait()
            for connection in self.connections:
                connection.close()
            self.connections = []
            self.available = []


class OpenGaussANN(BaseANN):
    method = None
    defaults = {}
    query_defaults = {}
    max_dimension = 2000

    def __init__(self, metric, method_param):
        if metric not in ("euclidean", "angular"):
            raise ValueError("Only euclidean (L2) and angular (cosine) float vectors are supported")
        if not isinstance(method_param, dict):
            raise ValueError("build parameters must be a mapping")
        self.metric = metric
        common = {
            "concurrents": 1, "parallel_workers": 1, "schema": "public",
            "enable_opfusion": None, "enable_seqscan": False,
            "statement_timeout_ms": 300000, "lock_timeout_ms": 5000,
            "maintenance_work_mem_mb": None,
        }
        unknown = set(method_param) - set(common) - set(self.defaults)
        if unknown:
            raise ValueError(f"Unknown build parameter(s): {sorted(unknown)}")
        self.params = {**common, **self.defaults, **method_param}
        for name, lo, hi in (("concurrents", 1, 256), ("parallel_workers", 1, 128),
                             ("statement_timeout_ms", 1, 2147483647), ("lock_timeout_ms", 1, 2147483647)):
            self.params[name] = integer(name, self.params[name], lo, hi)
        if self.params["maintenance_work_mem_mb"] is not None:
            self.params["maintenance_work_mem_mb"] = integer(
                "maintenance_work_mem_mb", self.params["maintenance_work_mem_mb"], 1, 2097151)
        schema = self.params["schema"]
        if not isinstance(schema, str) or not schema or "\x00" in schema or len(schema.encode()) > 63:
            raise ValueError("schema must be a nonempty SQL identifier of at most 63 bytes")
        boolean("enable_seqscan", self.params["enable_seqscan"])
        if self.params["enable_opfusion"] is not None:
            boolean("enable_opfusion", self.params["enable_opfusion"])
        self._validate_build()
        self.query_params = self.normalize_query_arguments({})
        self.connection = None
        self._created = False
        self.table_name = "annb_" + uuid.uuid4().hex
        self.index_name = self.table_name + "_idx"
        self.table = sql.Identifier(schema, self.table_name)
        self.index = sql.Identifier(schema, self.index_name)
        self.operator, self.opclass = ("<->", "vector_l2_ops") if metric == "euclidean" else ("<=>", "vector_cosine_ops")
        self.query_sql = sql.SQL("SELECT id FROM {} ORDER BY embedding {} %s::vector LIMIT %s").format(
            self.table, sql.SQL(self.operator))
        self.search_pool = _QueryConnections(self)
        self.res = []
        self._metadata = {}
        self._reported_server_risks = set()
        self._dimension = None
        self._count = None
        self._probe = None

    def _validate_build(self):
        raise NotImplementedError

    def _index_options(self):
        raise NotImplementedError

    def _validate_query(self, params):
        raise NotImplementedError

    def _connect(self):
        kwargs = {"autocommit": True, "connect_timeout": 10, "application_name": "annbenchmark-opengauss"}
        for name in ("user", "password", "dbname", "host", "port"):
            # Match the existing openGauss adapters' connection defaults.
            value = get_pg_conn_param(name, "ann" if name in ("user", "password", "dbname") else None)
            if value is not None:
                kwargs[name] = int(value) if name == "port" else value
        connection = psycopg.connect(**kwargs)
        try:
            pgvector.psycopg.register_vector(connection)
        except BaseException:
            connection.close()
            raise
        return connection

    def _apply_settings(self, connection, query_params):
        settings = {
            "statement_timeout": self.params["statement_timeout_ms"],
            "lockwait_timeout": self.params["lock_timeout_ms"],
            "enable_seqscan": self.params["enable_seqscan"],
            **query_params,
        }
        if self.params["enable_opfusion"] is not None:
            settings["enable_opfusion"] = self.params["enable_opfusion"]
        if self.params["maintenance_work_mem_mb"] is not None:
            settings["maintenance_work_mem"] = f"{self.params['maintenance_work_mem_mb']}MB"
        with connection.cursor() as cur:
            for name, value in settings.items():
                cur.execute(sql.SQL("SET {} = {}").format(sql.Identifier(name), sql.Literal(value)))

    def _validate_matrix(self, data, fitting=False):
        raw = np.asarray(data)
        if raw.dtype.kind not in "fiu" or raw.ndim != 2:
            raise ValueError("vectors must be a two-dimensional numeric array")
        with np.errstate(over="ignore", invalid="ignore"):
            matrix = np.asarray(raw, dtype=np.float32)
        if not np.isfinite(matrix).all():
            raise ValueError("vectors must be finite and representable as float32")
        if fitting:
            if not 1 <= len(matrix) <= 2147483647 or not 1 <= matrix.shape[1] <= self.max_dimension:
                raise ValueError(f"nonempty vector data with dimensions 1..{self.max_dimension} required")
            if self.metric == "angular" and matrix.shape[1] < 2:
                raise ValueError("cosine indexes require at least two dimensions")
            if self.method == "ivfflat" and self.params["lists"] > len(matrix):
                raise ValueError("lists must not exceed the number of input vectors")
            if self.method == "diskann" and self.params["enable_pq"] and matrix.shape[1] % self.params["pq_m"]:
                raise ValueError("vector dimension must be divisible by pq_m")
        elif matrix.shape[1] != self._dimension:
            raise ValueError(f"query dimension must be {self._dimension}")
        if self.metric == "angular" and np.any(~np.any(matrix != 0, axis=1)):
            raise ValueError("cosine vectors must be nonzero (zero vectors are not indexed)")
        return matrix

    def _require_fitted(self):
        if self.connection is None or self.connection.closed or not self._created:
            raise RuntimeError("fit() must complete before querying")

    def fit(self, X):
        if self.connection is not None or self._created:
            raise RuntimeError("call done() before fitting this adapter again")
        matrix = self._validate_matrix(X, fitting=True)
        self._metadata = {}
        self._reported_server_risks.clear()
        self.search_pool = _QueryConnections(self)
        self.res = []
        self._dimension, self._count = matrix.shape[1], len(matrix)
        self._probe = matrix[0].copy()
        try:
            self.connection = self._connect()
            self._apply_settings(self.connection, self.query_params)
            start = time.perf_counter()
            with self.connection.cursor() as cur:
                cur.execute(sql.SQL("CREATE TABLE {} (id integer NOT NULL, embedding vector({}))").format(
                    self.table, sql.Literal(self._dimension)))
                self._created = True
                cur.execute(sql.SQL("ALTER TABLE {} ALTER COLUMN embedding SET STORAGE PLAIN").format(self.table))
                cur.execute(sql.SQL("ALTER TABLE {} SET (parallel_workers = {})").format(
                    self.table, sql.Literal(self.params["parallel_workers"])))
                # Ordinary, committed rows are visible to DiskANN's build workers.
                with cur.copy(sql.SQL("COPY {} (id, embedding) FROM STDIN").format(self.table)) as copy:
                    for i, vector in enumerate(matrix):
                        copy.write_row((i, vector))
                self._metadata["data_load_time_s"] = time.perf_counter() - start
                options = sql.SQL(", ").join(sql.SQL("{} = {}").format(sql.Identifier(k), sql.Literal(v))
                                           for k, v in self._index_options().items())
                create = sql.SQL("CREATE INDEX {} ON {} USING {} (embedding {}) WITH ({})").format(
                    sql.Identifier(self.index_name), self.table, sql.Identifier(self.method),
                    sql.Identifier(self.opclass), options)
                start = time.perf_counter()
                cur.execute(create)
                self._metadata["index_build_time_s"] = time.perf_counter() - start
                cur.execute(sql.SQL("ANALYZE {}").format(self.table))
                cur.execute("SELECT version()")
                self._metadata["database_version"] = cur.fetchone()[0]
                cur.execute("SELECT current_database(), current_user")
                database, user = cur.fetchone()
                self._metadata.update(database_name=database, database_user=user,
                                      vector_dimension=self._dimension, vector_count=self._count,
                                      vector_type="vector", adapter_version="0.2.2")
            self._record_settings()
            self._record_known_server_risks()
            self.explain(self._probe, min(10, self._count))
        except BaseException:
            try:
                self.done()
            except Exception as cleanup_error:
                warnings.warn(f"Cleanup after fit failure also failed: {type(cleanup_error).__name__}", RuntimeWarning)
            raise

    def _record_settings(self):
        names = list(self.query_params) + ["enable_seqscan", "enable_opfusion", "statement_timeout", "lockwait_timeout",
                                          "maintenance_work_mem", "work_mem", "shared_buffers", "max_process_memory"]
        with self.connection.cursor() as cur:
            cur.execute("SELECT name, setting, unit FROM pg_settings WHERE name = ANY(%s) ORDER BY name", (names,))
            rows = cur.fetchall()
            self._metadata["effective_settings_json"] = json.dumps({r[0]: r[1] for r in rows}, sort_keys=True)
            self._metadata["effective_setting_units_json"] = json.dumps({r[0]: r[2] for r in rows}, sort_keys=True)

    def _record_known_server_risks(self):
        """Disclose known build/path risks without changing requested semantics.

        Build labels cannot establish whether a downstream kernel has a fix.
        An empty list is not a certificate that other builds/paths are correct.
        """
        version = self._metadata.get("database_version", "")
        risks = {}
        if re.search(r"\bbuild (2397a86f|e024ee66)\b", version):
            refine = self.params.get("rabitq_refine_type")
            if refine in ("FP32", "SQ8") and self.query_params.get("rbq_refinek", 0) > 0:
                if self.method == "ivfflat":
                    risks["ivf-rabitq-refinement"] = (
                        "Official openGauss 7.0.0 IVF RaBitQ FP32 refinement failed with "
                        "'The tuple is not found' in independent gsql reproduction. "
                        "SQ8 also uses the rerank path; its successful samples do not prove stability.")
                elif self.method == "hnsw" and self.metric == "angular":
                    risks["hnsw-rabitq-cosine-refinement"] = (
                        "Official openGauss 7.0.0 HNSW RaBitQ FP32 cosine refinement returned "
                        "incorrect candidate order in independent gsql reproduction. "
                        "SQ8 shares the source scoring branch; that is a source-level risk, "
                        "not a claim of an independently reproduced SQ8 failure.")
        self._metadata["known_server_risks_json"] = json.dumps(sorted(risks))
        for key, message in risks.items():
            if key not in self._reported_server_risks:
                warnings.warn(message + " Check whether your server includes a fix; requested "
                              "parameters and returned order are unchanged.",
                              RuntimeWarning, stacklevel=2)
                self._reported_server_risks.add(key)

    def set_query_arguments(self, *args):
        """Accept a mapping, or ordered positional values matching query_defaults."""
        self._require_fitted()
        if len(args) == 1 and isinstance(args[0], dict):
            supplied = args[0]
        else:
            if len(args) != len(self.query_defaults):
                raise ValueError(f"query parameters must be {list(self.query_defaults)} or one mapping")
            supplied = dict(zip(self.query_defaults, args))
        params = self.normalize_query_arguments(supplied)
        self.search_pool.close()
        self._apply_settings(self.connection, params)
        self.query_params = params
        self.search_pool = _QueryConnections(self)
        self.res = []
        self._record_settings()
        self._record_known_server_risks()
        self.explain(self._probe, min(10, self._count))
        # Like the existing openGauss adapters, initialize query sessions when
        # switching parameter groups, before the runner starts timing queries.
        self.search_pool.prepare()

    def normalize_query_arguments(self, supplied):
        """Pure validation shared by CLI preflight and live parameter switching."""
        if not isinstance(supplied, dict):
            raise ValueError("query parameters must be a mapping")
        unknown = set(supplied) - set(self.query_defaults)
        if unknown:
            raise ValueError(f"Unknown query parameter(s): {sorted(unknown)}")
        defaults = dict(self.query_defaults)
        if self.method == "ivfflat":
            defaults["ivfflat_probes"] = min(defaults["ivfflat_probes"], self.params["lists"])
        params = {**defaults, **supplied}  # reset omitted values, never inherit the prior group
        self._validate_query(params)
        return params

    def _query_vector(self, v, n):
        self._require_fitted()
        n = integer("k", n, 1, self._count)
        raw = np.asarray(v)
        if raw.ndim != 1:
            raise ValueError("query must be a one-dimensional vector")
        matrix = self._validate_matrix(raw.reshape(1, -1))
        return matrix[0], n

    def _execute_query(self, connection, vector, n):
        with connection.cursor() as cur:
            cur.execute(self.query_sql, (vector, n), binary=True, prepare=True)
            ids = [row[0] for row in cur.fetchall()]
        # ANN searches can return fewer than k candidates at small search budgets.
        # The reference runner pads these with -1/+inf and reduces recall.
        # The driver decodes the integer column; the runner handles result ID
        # conversion and uniqueness outside query timing. Keep these guards:
        # its single path only warns on excess results, and its batch path has
        # no equivalent count check; negative IDs would index from the end.
        if len(ids) > n or any(i < 0 or i >= self._count for i in ids):
            raise RuntimeError("index query returned too many or out-of-range IDs")
        return ids

    def query(self, v, n):
        vector, n = self._query_vector(v, n)
        with self.search_pool.connection() as connection:
            return self._execute_query(connection, vector, n)

    def explain(self, v, n):
        vector, n = self._query_vector(v, n)
        with self.connection.cursor() as cur:
            cur.execute(sql.SQL("EXPLAIN ") + self.query_sql, (vector, n), prepare=False)
            plan = "\n".join(str(row[0]) for row in cur.fetchall())
        if self.index_name not in plan or "Index Scan" not in plan:
            raise RuntimeError(f"planner did not select the benchmark index: {plan}")
        self._metadata["query_plan"] = plan
        return plan

    def get_memory_usage(self):
        """Reference hook is in KiB; this is index storage, NOT process memory."""
        if not self._created or self.connection is None or self.connection.closed:
            return 0.0
        with self.connection.cursor() as cur:
            cur.execute("SELECT pg_relation_size(%s::regclass)", (self.index.as_string(self.connection),))
            return int(cur.fetchone()[0]) / 1024.0

    def get_additional(self):
        return {**self._metadata,
                "build_params_json": json.dumps(self.params, sort_keys=True),
                "query_params_json": json.dumps(self.query_params, sort_keys=True),
                "metric": self.metric, "index_method": self.method,
                "index_size_bytes": int(self.get_memory_usage() * 1024),
                "latency_definition": "runner wall time: per query in single mode; per batch / query count in batch mode",
                "index_size_definition": "pg_relation_size of benchmark index, not peak memory"}

    def done(self):
        self.search_pool.close()
        cleanup_error = None
        try:
            if self._created:
                if self.connection is None or self.connection.closed or self.connection.broken:
                    if self.connection is not None:
                        self.connection.close()
                    self.connection = self._connect()
                    self._apply_settings(self.connection, self.query_params)
                with self.connection.cursor() as cur:
                    cur.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(self.table))
                self._created = False
        except Exception as exc:
            cleanup_error = exc
        finally:
            if self.connection is not None:
                self.connection.close()
                self.connection = None
        if cleanup_error is not None:
            raise RuntimeError(f"Could not clean owned table {self.params['schema']}.{self.table_name}") from cleanup_error

    def __str__(self):
        return f"{type(self).__name__}(build={json.dumps(self.params, sort_keys=True)},query={json.dumps(self.query_params, sort_keys=True)})"


class _RaBitQ(OpenGaussANN):
    def _validate_rabitq(self):
        if not boolean("enable_rabitq", self.params["enable_rabitq"]):
            raise ValueError("RaBitQ adapters require enable_rabitq=true")
        boolean("rabitq_fht", self.params["rabitq_fht"])
        value = self.params["rabitq_refine_type"]
        if not isinstance(value, str) or value.upper() not in ("FP32", "SQ8", "NONE"):
            raise ValueError("rabitq_refine_type must be FP32, SQ8, or NONE")
        self.params["rabitq_refine_type"] = value.upper()

    def _rabitq_options(self):
        return {key: self.params[key] for key in ("enable_rabitq", "rabitq_refine_type", "rabitq_fht")}

    def _validate_rabitq_query(self, params):
        params["rbq_query_bits"] = integer("rbq_query_bits", params["rbq_query_bits"], 1, 8)
        # Source/pg_settings in 7.0.0 accepts real 0..1000 (some docs state 1..1000).
        params["rbq_refinek"] = real("rbq_refinek", params["rbq_refinek"], 0, 1000)


RABITQ_DEFAULTS = {"enable_rabitq": True, "rabitq_refine_type": "FP32", "rabitq_fht": False}


class OpenGaussHNSWRaBitQ(_RaBitQ):
    method = "hnsw"
    defaults = {"m": 16, "ef_construction": 64, **RABITQ_DEFAULTS}
    query_defaults = {"hnsw_ef_search": 100, "rbq_query_bits": 8, "rbq_refinek": 5.0}

    def _validate_build(self):
        self.params["m"] = integer("m", self.params["m"], 2, 100)
        self.params["ef_construction"] = integer("ef_construction", self.params["ef_construction"], 4, 1000)
        if self.params["ef_construction"] < 2 * self.params["m"]:
            raise ValueError("ef_construction must be at least 2*m")
        self._validate_rabitq()

    def _index_options(self):
        return {"m": self.params["m"], "ef_construction": self.params["ef_construction"], **self._rabitq_options()}

    def _validate_query(self, params):
        params["hnsw_ef_search"] = integer("hnsw_ef_search", params["hnsw_ef_search"], 1, 1000000)
        self._validate_rabitq_query(params)


class OpenGaussIVFFlatRaBitQ(_RaBitQ):
    method = "ivfflat"
    defaults = {"lists": 100, **RABITQ_DEFAULTS}
    query_defaults = {"ivfflat_probes": 10, "rbq_query_bits": 8, "rbq_refinek": 5.0}

    def _validate_build(self):
        self.params["lists"] = integer("lists", self.params["lists"], 1, 32768)
        self._validate_rabitq()

    def _index_options(self):
        return {"lists": self.params["lists"], **self._rabitq_options()}

    def _validate_query(self, params):
        params["ivfflat_probes"] = integer("ivfflat_probes", params["ivfflat_probes"], 1, 32768)
        if params["ivfflat_probes"] > self.params["lists"]:
            raise ValueError("ivfflat_probes must not exceed lists")
        self._validate_rabitq_query(params)


class OpenGaussDiskANN(OpenGaussANN):
    method = "diskann"
    max_dimension = 1536
    defaults = {"index_size": 100, "enable_pq": False, "pq_m": 8}
    query_defaults = {"diskann_probes": 128}

    def _validate_build(self):
        self.params["index_size"] = integer("index_size", self.params["index_size"], 16, 1000)
        self.params["pq_m"] = integer("pq_m", self.params["pq_m"], 1, 192)
        boolean("enable_pq", self.params["enable_pq"])

    def _index_options(self):
        return {key: self.params[key] for key in ("index_size", "enable_pq", "pq_m")}

    def _validate_query(self, params):
        params["diskann_probes"] = integer("diskann_probes", params["diskann_probes"], 1, 32768)


class openGaussHNSWPQ(BaseANN):
    def __init__(self, metric, method_param):
        self._metric = metric
        self._m = method_param['M']
        self._ef_construction = method_param['efConstruction']
        self._cur = None
        self.curs = []
        self.connections = []
        self.init_conn_flags = False
        self.search_pool = None
        self.con = method_param['concurrents']
        self.hnsw_earlystop_threshold = method_param['hnswEarlystopThreshold']
        self.pq_m = method_param['pqM']

        # default value
        self.user = 'ann'
        self.port = 5432
        self.dbname = 'ann'
        self.host = 'localhost'
        self.password = 'ann'

        if metric == "angular":
            self._query = "SELECT id FROM items ORDER BY embedding <=> %s LIMIT %s"
        elif metric == "euclidean":
            self._query = "SELECT id FROM items ORDER BY embedding <-> %s LIMIT %s"
        else:
            raise RuntimeError(f"unknown metric {metric}")

    def fit(self, X):
        psycopg_connect_kwargs: Dict[str, Any] = dict(
            autocommit=True,
        )
        for arg_name in ['user', 'password', 'dbname']:
            # The default value is "ann" for all of these parameters.
            psycopg_connect_kwargs[arg_name] = get_pg_conn_param(
                arg_name, 'ann')
            if arg_name == 'user':
                self.user = psycopg_connect_kwargs['user']
            if arg_name == 'password':
                self.password = psycopg_connect_kwargs['password']
            if arg_name == 'dbname':
                self.dbname = psycopg_connect_kwargs['dbname']

        # If host/port are not specified, leave the default choice to the
        # psycopg driver.
        og_host: Optional[str] = get_pg_conn_param('host')
        if og_host is not None:
            psycopg_connect_kwargs['host'] = og_host
            self.host = og_host

        og_port_str: Optional[str] = get_pg_conn_param('port')
        if og_port_str is not None:
            psycopg_connect_kwargs['port'] = int(og_port_str)
            self.port = int(og_port_str)

        conn = psycopg.connect(**psycopg_connect_kwargs)

        pgvector.psycopg.register_vector(conn)
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS items")
        cur.execute("CREATE TABLE items (id int, embedding vector(%d))" % X.shape[1])
        cur.execute("ALTER TABLE items ALTER COLUMN embedding SET STORAGE PLAIN")
        cur.execute("ALTER TABLE items SET (parallel_workers = 32);")
        cur.execute("set hnsw_earlystop_threshold = %d" % self.hnsw_earlystop_threshold)
        print("copying data...")
        with cur.copy("COPY items (id, embedding) FROM STDIN WITH (FORMAT BINARY)") as copy:
            copy.set_types(["int4", "vector"])
            for i, embedding in enumerate(X):
                copy.write_row((i, embedding))
        print(f"creating index [hnsw_earlystop_threshold={self.hnsw_earlystop_threshold}]...")
        if self._metric == "angular":
            cur.execute(
                "CREATE INDEX ON items USING hnsw (embedding vector_cosine_ops) WITH (m = %d, ef_construction = %d, enable_pq = on, pq_m = %d)" % (self._m, self._ef_construction, self.pq_m))
        elif self._metric == "euclidean":
            cur.execute("CREATE INDEX ON items USING hnsw (embedding vector_l2_ops) WITH (m = %d, ef_construction = %d, enable_pq = on, pq_m = %d)" % (self._m, self._ef_construction, self.pq_m))
        else:
            raise RuntimeError(f"unknown metric {self._metric}")
        print("done!")
        self._cur = cur

    def set_query_arguments(self, ef_search):
        self._ef_search = ef_search
        self._cur.execute("SET hnsw_ef_search = %d" % ef_search)
        self._cur.execute("set enable_seqscan = off")
        self._cur.execute("set hnsw_earlystop_threshold = %d" % self.hnsw_earlystop_threshold)
        conc = self.con
        self.search_pool = Pool(conc, initializer=self.init_connection)

    def query(self, v, n):
        self._cur.execute(self._query, (v, n), binary=True, prepare=True)
        return [id for id, in self._cur.fetchall()]

    def get_memory_usage(self):
        if self._cur is None:
            return 0
        self._cur.execute("SELECT pg_relation_size('items_embedding_idx')")
        return self._cur.fetchone()[0] / 1024
    
    def init_connection(self):
        conn = psycopg.connect(user=self.user, password=self.password, dbname=self.dbname, host=self.host, autocommit=True, port=self.port)
        pgvector.psycopg.register_vector(conn)
        global cur
        cur = conn.cursor()
        cur.execute("SET hnsw_ef_search = %d" % self._ef_search)
        cur.execute("set enable_seqscan = off")
        cur.execute("set hnsw_earlystop_threshold = %d" % self.hnsw_earlystop_threshold)
        global base_query
        base_query = self._query
        self.connections.append(conn)
        self.curs.append(cur)

    def close_connections(self):
        for cur in self.curs:
            if cur is not None:
                cur.close()
                cur = None
        for conn in self.connections:
            if conn is not None:
                conn.close()
                conn = None
        self.curs = []
        self.connections = []
        self.init_conn_flags = False
    
    @staticmethod
    def sub_query(chunk, n):
        ids = []
        for item in chunk:
            cur.execute(base_query, (item, n), binary=True, prepare=True)
            res = cur.fetchall()
            ids.append([i[0] for i in res])
        return ids
    
    def batch_query(self, X: numpy.array, n: int):
        conc = self.con
        chunk_size = len(X) // conc + 1
        chunks = [X[i:i + chunk_size] for i in range(0, len(X), chunk_size)]
        res = self.search_pool.starmap(self.sub_query, [(chunk, n) for chunk in chunks])
        print(f"openGauss concuiiency_query process num: {conc}")
        self.res = []
        for item in res:
            for ids in item:
                self.res.append(ids)

    def __str__(self):
        return f"openGaussHNSWPQ(m={self._m}, ef_construction={self._ef_construction}, ef_search={self._ef_search}, pq_m={self.pq_m}, hnsw_earlystop_threshold={self.hnsw_earlystop_threshold})"
    
    def __del__(self):
        self.close_connections()


class openGaussIVF(BaseANN):
    def __init__(self, metric, params, lists):
        self._metric = metric
        self._lists = lists
        self._cur = None
        self.curs = []
        self.connections = []
        self.init_conn_flags = False
        self.search_pool = None

        self.enable_npu = params[0]["enable_npu"]
        self.conc = params[1]["concurrents"]

        # default value
        self.user = 'ann'
        self.port = 5432
        self.dbname = 'ann'
        self.host = 'localhost'
        self.password = 'ann'

        if metric == "angular":
            self._query = "SELECT id FROM items ORDER BY embedding <=> %s LIMIT %s"
        elif metric == "euclidean":
            self._query = "SELECT id FROM items ORDER BY embedding <-> %s LIMIT %s"
        else:
            raise RuntimeError(f"unknown metric {metric}")

    def fit(self, X):
        psycopg_connect_kwargs: Dict[str, Any] = dict(
            autocommit=True,
        )
        for arg_name in ['user', 'password', 'dbname']:
            # The default value is "ann" for all of these parameters.
            psycopg_connect_kwargs[arg_name] = get_pg_conn_param(
                arg_name, 'ann')
            if arg_name == 'user':
                self.user = psycopg_connect_kwargs['user']
            if arg_name == 'password':
                self.password = psycopg_connect_kwargs['password']
            if arg_name == 'dbname':
                self.dbname = psycopg_connect_kwargs['dbname']

        # If host/port are not specified, leave the default choice to the
        # psycopg driver.
        og_host: Optional[str] = get_pg_conn_param('host')
        if og_host is not None:
            psycopg_connect_kwargs['host'] = og_host
            self.host = og_host

        og_port_str: Optional[str] = get_pg_conn_param('port')
        if og_port_str is not None:
            psycopg_connect_kwargs['port'] = int(og_port_str)
            self.port = int(og_port_str)

        conn = psycopg.connect(**psycopg_connect_kwargs)

        pgvector.psycopg.register_vector(conn)
        
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS items")
        cur.execute("CREATE TABLE items (id int, embedding vector(%d))" % X.shape[1])
        cur.execute("ALTER TABLE items ALTER COLUMN embedding SET STORAGE PLAIN")
        cur.execute("ALTER TABLE items SET (parallel_workers = 32);")
        print("copying data...")
        with cur.copy("COPY items (id, embedding) FROM STDIN") as copy:
            for i, embedding in enumerate(X):
                copy.write_row((i, embedding))
        print(f"creating index ...")
        start_time = time.perf_counter()
        if self._metric == "angular":
            cur.execute(
                "CREATE INDEX ON items USING ivfflat (embedding vector_cosine_ops) WITH (lists = %d)" % (self._lists))
        elif self._metric == "euclidean":
            cur.execute("CREATE INDEX ON items USING ivfflat (embedding vector_l2_ops) WITH (lists= %d)" % (self._lists))
        else:
            raise RuntimeError(f"unknown metric {self._metric}")
        end_time = time.perf_counter() - start_time
        print(f"done! time = {end_time}.")
        self._cur = cur

    def set_query_arguments(self, probes):
        self._probes = probes
        self._cur.execute("SET ivfflat_probes = %d" % probes)
        self._cur.execute("set enable_seqscan = off")
        self._cur.execute("set enable_npu = %s" % self.enable_npu)
        conc = self.conc
        self.search_pool = Pool(conc, initializer=self.init_connection)

    def query(self, v, n):
        self._cur.execute(self._query, (v, n), binary=True, prepare=True)
        return [id for id, in self._cur.fetchall()]

    def get_memory_usage(self):
        if self._cur is None:
            return 0
        self._cur.execute("SELECT pg_relation_size('items_embedding_idx')")
        return self._cur.fetchone()[0] / 1024
    
    def init_connection(self):
        conn = psycopg.connect(user=self.user, password=self.password, dbname=self.dbname, host=self.host, autocommit=True, port=self.port)
        pgvector.psycopg.register_vector(conn)
        global cur
        cur = conn.cursor()
        cur.execute("SET ivfflat_probes = %d" % self._probes)
        cur.execute("set enable_seqscan = off")
        cur.execute("set enable_npu = %s" % self.enable_npu)
        global base_query
        base_query = self._query
        self.connections.append(conn)
        self.curs.append(cur)

    def close_connections(self):
        for cur in self.curs:
            if cur is not None:
                cur.close()
                cur = None
        for conn in self.connections:
            if conn is not None:
                conn.close()
                conn = None
        self.curs = []
        self.connections = []
        self.init_conn_flags = False
    
    @staticmethod
    def sub_query(chunk, n):
        ids = []
        for item in chunk:
            cur.execute(base_query, (item, n), binary=True, prepare=True)
            res = cur.fetchall()
            ids.append([i[0] for i in res])
        return ids
    
    def batch_query(self, X: numpy.array, n: int):
        conc = self.conc
        chunk_size = len(X) // conc + 1
        chunks = [X[i:i + chunk_size] for i in range(0, len(X), chunk_size)]
        res = self.search_pool.starmap(self.sub_query, [(chunk, n) for chunk in chunks])
        print(f"openGauss concuiiency_query process num: {conc}")
        self.res = []
        for item in res:
            for ids in item:
                self.res.append(ids)


    def __str__(self):
        return f"openGaussIVF(lists={self._lists}, probes={self._probes})"

    def __del__(self):
        self.close_connections()
