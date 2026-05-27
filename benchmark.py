from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import shutil
import socket
import statistics
import subprocess
import sys
import time
import urllib.request
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

try:
    from pymongo import MongoClient, ASCENDING, DESCENDING, ReadPreference
    from pymongo.write_concern import WriteConcern
    from pymongo.errors import (
        AutoReconnect,
        ConnectionFailure,
        OperationFailure,
        ServerSelectionTimeoutError,
    )
except ImportError:
    print("ERROR: pymongo is not installed. Run: pip install -r requirements.txt",
          file=sys.stderr)
    raise

#Config

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
RESULTS_DIR = HERE / "results"
CHARTS_DIR = RESULTS_DIR / "charts"

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DB_NAME = "dse_benchmark"

NODES = [
    {"container": "mongo1", "host": "localhost", "port": 27017},
    {"container": "mongo2", "host": "localhost", "port": 27018},
    {"container": "mongo3", "host": "localhost", "port": 27019},
]

RS_URI_HOSTNAMES = "mongodb://mongo1:27017,mongo2:27017,mongo3:27017/?replicaSet=rs0"
RS_URI_LOCALPORTS = (
    "mongodb://localhost:27017,localhost:27018,localhost:27019/?replicaSet=rs0"
)


# Utilities

def now_ms() -> float:
    return time.perf_counter() * 1000.0


def pct(samples: list[float], p: float) -> float:
    if not samples:
        return float("nan")
    if len(samples) == 1:
        return samples[0]
    s = sorted(samples)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def fmt_ms(v: float) -> str:
    if v != v:  # NaN
        return "—"
    if v < 1:
        return f"{v*1000:.0f} µs"
    if v < 1000:
        return f"{v:.2f} ms"
    return f"{v/1000:.2f} s"


def banner(msg: str) -> None:
    line = "─" * max(60, len(msg) + 4)
    print(f" {msg}")


# Result Managing

@dataclass
class OpResult:
    test: str           
    op_index: int        
    latency_ms: float    
    ok: bool = True
    error: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TestSummary:
    test: str
    n: int
    total_seconds: float
    ops_per_sec: float
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    errors: int


class Recorder:

    def __init__(self) -> None:
        self.ops: list[OpResult] = []
        self.test_started: dict[str, float] = {}
        self.test_ended: dict[str, float] = {}

    def start(self, test: str) -> None:
        self.test_started[test] = time.perf_counter()

    def end(self, test: str) -> None:
        self.test_ended[test] = time.perf_counter()

    def record(self, op: OpResult) -> None:
        self.ops.append(op)

    def summarise(self) -> list[TestSummary]:
        by_test: dict[str, list[OpResult]] = defaultdict(list)
        for op in self.ops:
            by_test[op.test].append(op)

        out: list[TestSummary] = []
        for test, ops in by_test.items():
            latencies = [o.latency_ms for o in ops if o.ok]
            errors = sum(1 for o in ops if not o.ok)
            total = self.test_ended.get(test, 0) - self.test_started.get(test, 0)
            ops_sec = len(ops) / total if total > 0 else 0.0
            out.append(TestSummary(
                test=test,
                n=len(ops),
                total_seconds=total,
                ops_per_sec=ops_sec,
                mean_ms=statistics.fmean(latencies) if latencies else float("nan"),
                p50_ms=pct(latencies, 50),
                p95_ms=pct(latencies, 95),
                p99_ms=pct(latencies, 99),
                max_ms=max(latencies) if latencies else float("nan"),
                errors=errors,
            ))
        out.sort(key=lambda s: s.test)
        return out

    def write_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["test", "op_index", "latency_ms", "ok", "error", "extra_json"])
            for op in self.ops:
                w.writerow([
                    op.test, op.op_index, f"{op.latency_ms:.4f}",
                    int(op.ok), op.error, json.dumps(op.extra) if op.extra else "",
                ])


# Dataset

def download_dataset() -> dict[str, Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    extracted = DATA_DIR / "ml-latest-small"
    if extracted.exists() and (extracted / "ratings.csv").exists():
        print(f"[data] using cached dataset at {extracted}")
    else:
        zip_path = DATA_DIR / "ml-latest-small.zip"
        if not zip_path.exists():
            print(f"[data] downloading {MOVIELENS_URL} ...")
            urllib.request.urlretrieve(MOVIELENS_URL, zip_path)
        print(f"[data] extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(DATA_DIR)

    return {
        "movies": extracted / "movies.csv",
        "ratings": extracted / "ratings.csv",
        "tags": extracted / "tags.csv",
        "links": extracted / "links.csv",
    }


def load_movies(path: Path) -> list[dict]:
    out = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append({
                "_id": int(row["movieId"]),
                "title": row["title"],
                "genres": [g for g in row["genres"].split("|") if g and g != "(no genres listed)"],
            })
    return out


def load_ratings(path: Path) -> list[dict]:
    out = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append({
                "userId": int(row["userId"]),
                "movieId": int(row["movieId"]),
                "rating": float(row["rating"]),
                "timestamp": int(row["timestamp"]),
            })
    return out


def load_tags(path: Path) -> list[dict]:
    out = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append({
                "userId": int(row["userId"]),
                "movieId": int(row["movieId"]),
                "tag": row["tag"],
                "timestamp": int(row["timestamp"]),
            })
    return out


# Connection

def _hostnames_resolve() -> bool:
    try:
        socket.gethostbyname("mongo1")
        socket.gethostbyname("mongo2")
        socket.gethostbyname("mongo3")
        return True
    except OSError:
        return False


def connect_replica_set(timeout_ms: int = 4000) -> MongoClient | None:
    candidates = []
    if _hostnames_resolve():
        candidates.append(("rs-via-hostnames", RS_URI_HOSTNAMES))
    candidates.append(("rs-via-localhost-ports", RS_URI_LOCALPORTS))

    for label, uri in candidates:
        try:
            client = MongoClient(uri, serverSelectionTimeoutMS=timeout_ms)
            # Force discovery
            client.admin.command("ping")
            primary = client.primary
            if primary is None:
                client.close()
                continue
            print(f"[conn] replica-set client ok via {label}; primary={primary}")
            return client
        except (ServerSelectionTimeoutError, ConnectionFailure, OperationFailure) as e:
            print(f"[conn] {label} failed: {e.__class__.__name__}: {e}")
    return None


def connect_directs() -> list[MongoClient]:
    clients = []
    for node in NODES:
        uri = f"mongodb://{node['host']}:{node['port']}/?directConnection=true"
        c = MongoClient(uri, serverSelectionTimeoutMS=2000)
        clients.append(c)
    return clients


def find_primary_direct(directs: list[MongoClient]) -> tuple[MongoClient, dict] | None:
    for c, node in zip(directs, NODES):
        try:
            info = c.admin.command("hello")
            if info.get("isWritablePrimary") or info.get("ismaster"):
                return c, node
        except (ServerSelectionTimeoutError, ConnectionFailure, OperationFailure):
            continue
    return None


# Workload preparation

def reset_db(client: MongoClient) -> None:
    print(f"[setup] dropping database {DB_NAME!r} to start clean")
    client.drop_database(DB_NAME)


def load_seed_data(client: MongoClient, csvs: dict[str, Path]) -> dict[str, int]:
    db = client[DB_NAME]
    movies = load_movies(csvs["movies"])
    ratings = load_ratings(csvs["ratings"])
    tags = load_tags(csvs["tags"])

    # Holding 5000 ratings for the write benchmark
    random.Random(1234).shuffle(ratings)
    holdout = ratings[:5000]
    seed_ratings = ratings[5000:]

    print(f"[setup] seeding {len(movies)} movies, {len(seed_ratings)} ratings, "
          f"{len(tags)} tags (holding back {len(holdout)} ratings for writes)")

    db.movies.insert_many(movies, ordered=False)
    chunk = 5000
    for i in range(0, len(seed_ratings), chunk):
        db.ratings.insert_many(seed_ratings[i:i + chunk], ordered=False)
    if tags:
        db.tags.insert_many(tags, ordered=False)

    db.ratings.create_index([("movieId", ASCENDING)])
    db.ratings.create_index([("userId", ASCENDING)])
    db.movies.create_index([("genres", ASCENDING)])

    (DATA_DIR / "holdout.json").write_text(json.dumps(holdout))

    return {
        "movies": len(movies),
        "ratings_seeded": len(seed_ratings),
        "ratings_holdout": len(holdout),
        "tags": len(tags),
    }


def load_holdout() -> list[dict]:
    p = DATA_DIR / "holdout.json"
    if not p.exists():
        return []
    return json.loads(p.read_text())

# Write benchmarks

def bench_write_single(client: MongoClient, holdout: list[dict],
                       recorder: Recorder, write_concern: str,
                       n: int) -> None:
    test = f"write.single.{write_concern}"
    wc = WriteConcern(w=1 if write_concern == "w1" else "majority")
    coll = client[DB_NAME].get_collection("ratings", write_concern=wc)

    print(f"[write] {test}: inserting {n} docs one-by-one ...")
    recorder.start(test)
    for i in range(n):
        doc = dict(holdout[i % len(holdout)])
        doc["_bench_op"] = i
        doc["_bench_test"] = test
        t0 = now_ms()
        try:
            coll.insert_one(doc)
            recorder.record(OpResult(test, i, now_ms() - t0))
        except Exception as e:
            recorder.record(OpResult(test, i, now_ms() - t0, ok=False, error=str(e)))
    recorder.end(test)


def bench_write_bulk(client: MongoClient, holdout: list[dict],
                     recorder: Recorder, write_concern: str,
                     batches: int, batch_size: int) -> None:
    test = f"write.bulk{batch_size}.{write_concern}"
    wc = WriteConcern(w=1 if write_concern == "w1" else "majority")
    coll = client[DB_NAME].get_collection("ratings", write_concern=wc)

    print(f"[write] {test}: {batches} batches of {batch_size} docs ...")
    recorder.start(test)
    cursor = 0
    for b in range(batches):
        docs = []
        for _ in range(batch_size):
            d = dict(holdout[cursor % len(holdout)])
            d["_bench_op"] = cursor
            d["_bench_test"] = test
            docs.append(d)
            cursor += 1
        t0 = now_ms()
        try:
            coll.insert_many(docs, ordered=False)
            recorder.record(OpResult(test, b, now_ms() - t0,
                                     extra={"batch_size": batch_size}))
        except Exception as e:
            recorder.record(OpResult(test, b, now_ms() - t0, ok=False, error=str(e)))
    recorder.end(test)


def bench_update(client: MongoClient, recorder: Recorder, n: int) -> None:
    test = "write.update.w1"
    coll = client[DB_NAME].ratings
    # pick n random ratings to bump
    sample = list(coll.aggregate([{"$sample": {"size": n}},
                                  {"$project": {"_id": 1}}]))
    print(f"[write] {test}: updating {len(sample)} random ratings ...")
    recorder.start(test)
    for i, doc in enumerate(sample):
        t0 = now_ms()
        try:
            coll.update_one({"_id": doc["_id"]},
                            {"$inc": {"_bench_updates": 1}})
            recorder.record(OpResult(test, i, now_ms() - t0))
        except Exception as e:
            recorder.record(OpResult(test, i, now_ms() - t0, ok=False, error=str(e)))
    recorder.end(test)

# Read benchmarks

def bench_read_by_id(client: MongoClient, recorder: Recorder,
                     read_pref: str, n: int) -> None:
    test = f"read.by_id.{read_pref}"
    coll = _coll_with_read_pref(client[DB_NAME].movies, read_pref)

    # collect a pool of valid _ids first
    pool = [d["_id"] for d in coll.find({}, {"_id": 1}).limit(2000)]
    if not pool:
        print(f"[read] {test}: skipped (no movies in pool)")
        return
    print(f"[read] {test}: {n} point lookups by _id ...")
    recorder.start(test)
    rng = random.Random(7)
    for i in range(n):
        _id = rng.choice(pool)
        t0 = now_ms()
        try:
            coll.find_one({"_id": _id})
            recorder.record(OpResult(test, i, now_ms() - t0))
        except Exception as e:
            recorder.record(OpResult(test, i, now_ms() - t0, ok=False, error=str(e)))
    recorder.end(test)


def bench_read_filter(client: MongoClient, recorder: Recorder,
                      read_pref: str, n: int) -> None:
    test = f"read.filter_genre.{read_pref}"
    coll = _coll_with_read_pref(client[DB_NAME].movies, read_pref)
    genres = ["Action", "Comedy", "Drama", "Horror", "Romance",
              "Sci-Fi", "Thriller", "Animation", "Documentary"]
    print(f"[read] {test}: {n} indexed equality queries on genres ...")
    recorder.start(test)
    rng = random.Random(11)
    for i in range(n):
        g = rng.choice(genres)
        t0 = now_ms()
        try:
            _ = list(coll.find({"genres": g}).limit(50))
            recorder.record(OpResult(test, i, now_ms() - t0, extra={"genre": g}))
        except Exception as e:
            recorder.record(OpResult(test, i, now_ms() - t0, ok=False, error=str(e)))
    recorder.end(test)


def bench_aggregation(client: MongoClient, recorder: Recorder,
                      read_pref: str, n: int) -> None:
    test = f"read.agg_top_rated.{read_pref}"
    coll = _coll_with_read_pref(client[DB_NAME].ratings, read_pref)
    print(f"[read] {test}: {n} top-rated aggregations ...")
    recorder.start(test)
    for i in range(n):
        t0 = now_ms()
        try:
            _ = list(coll.aggregate([
                {"$group": {"_id": "$movieId",
                            "avg_rating": {"$avg": "$rating"},
                            "n_ratings": {"$sum": 1}}},
                {"$match": {"n_ratings": {"$gte": 20}}},
                {"$sort": {"avg_rating": -1}},
                {"$limit": 10},
            ]))
            recorder.record(OpResult(test, i, now_ms() - t0))
        except Exception as e:
            recorder.record(OpResult(test, i, now_ms() - t0, ok=False, error=str(e)))
    recorder.end(test)


def _coll_with_read_pref(coll, read_pref: str):
    if read_pref == "primary":
        return coll.with_options(read_preference=ReadPreference.PRIMARY)
    if read_pref == "secondary_preferred":
        return coll.with_options(read_preference=ReadPreference.SECONDARY_PREFERRED)
    return coll

# Failover test

def docker_available() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        r = subprocess.run(["docker", "ps", "--format", "{{.Names}}"],
                           capture_output=True, text=True, timeout=5)
        if r.returncode != 0:
            return False
        names = set(r.stdout.split())
        return all(node["container"] in names for node in NODES)
    except (subprocess.SubprocessError, OSError):
        return False


def docker_stop(container: str) -> None:
    subprocess.run(["docker", "stop", container], check=True, capture_output=True)


def docker_start(container: str) -> None:
    subprocess.run(["docker", "start", container], check=True, capture_output=True)


def identify_primary(rs_client: MongoClient | None,
                     directs: list[MongoClient]) -> dict | None:
    """Return the NODES entry that currently hosts the primary."""
    if rs_client is not None:
        try:
            addr = rs_client.primary  # ('host', port)
            if addr is not None:
                host, port = addr
                # map to container by matching hostname
                for node in NODES:
                    if node["container"] == host or host.startswith(node["container"]):
                        return node
                # otherwise match by published port
                for node in NODES:
                    if node["port"] == port:
                        return node
        except Exception:
            pass
    # fall back: ask each node directly
    found = find_primary_direct(directs)
    return found[1] if found else None


def bench_failover(rs_client: MongoClient | None,
                   directs: list[MongoClient],
                   recorder: Recorder,
                   probe_count: int = 200,
                   probe_interval_s: float = 0.1) -> dict[str, Any]:
    test = "failover.write_probe"
    primary_node = identify_primary(rs_client, directs)
    if primary_node is None:
        print("[failover] could not identify primary — skipping failover test")
        return {"skipped": True, "reason": "no primary"}

    print(f"[failover] current primary is {primary_node['container']} "
          f"(host port {primary_node['port']})")

    auto = docker_available()
    if not auto:
        print("[failover] `docker` not callable or containers not found.")
        print(f"[failover] In ANOTHER terminal, run:")
        print(f"           docker stop {primary_node['container']}")
        input("[failover] press <Enter> once you have stopped it ...")
    else:
        print(f"[failover] stopping container {primary_node['container']} via docker ...")

    def do_write() -> tuple[bool, float, str]:
        doc = {"_failover_probe": True, "ts": time.time(),
               "userId": random.randint(1, 1000),
               "movieId": random.randint(1, 200000),
               "rating": 3.0, "timestamp": int(time.time())}
        t0 = now_ms()
        try:
            if rs_client is not None:
                rs_client[DB_NAME].ratings.with_options(
                    write_concern=WriteConcern(w=1)
                ).insert_one(doc)
            else:
                # find a writable node among the directs
                fp = find_primary_direct(directs)
                if fp is None:
                    raise ConnectionFailure("no writable node found")
                fp[0][DB_NAME].ratings.insert_one(doc)
            return True, now_ms() - t0, ""
        except (AutoReconnect, ServerSelectionTimeoutError,
                ConnectionFailure, OperationFailure) as e:
            return False, now_ms() - t0, e.__class__.__name__

    kill_time = None
    recovery_time = None
    last_success_before_kill = None
    first_success_after_kill = None

    recorder.start(test)
    warmup = 20
    print(f"[failover] {warmup} warm-up probes before kill ...")
    for i in range(warmup):
        ok, lat, err = do_write()
        recorder.record(OpResult(test, i, lat, ok=ok, error=err,
                                 extra={"phase": "pre_kill"}))
        if ok:
            last_success_before_kill = time.time()
        time.sleep(probe_interval_s)

    if auto:
        kill_t0 = time.time()
        try:
            docker_stop(primary_node["container"])
        except subprocess.CalledProcessError as e:
            print(f"[failover] docker stop failed: {e}; falling back to manual")
            input(f"[failover] please run: docker stop {primary_node['container']} "
                  "and press <Enter> ...")
        kill_time = time.time()
        print(f"[failover] primary stopped at t={kill_time - kill_t0:.3f}s after issue")
    else:
        kill_time = time.time()

    print(f"[failover] probing for recovery (up to {probe_count} probes "
          f"@ {probe_interval_s}s) ...")
    for i in range(warmup, warmup + probe_count):
        ok, lat, err = do_write()
        recorder.record(OpResult(test, i, lat, ok=ok, error=err,
                                 extra={"phase": "post_kill"}))
        if ok and first_success_after_kill is None:
            first_success_after_kill = time.time()
            print(f"[failover] writes resumed after "
                  f"{first_success_after_kill - kill_time:.2f}s")
        time.sleep(probe_interval_s)
        if first_success_after_kill is not None and \
                i > warmup + 30 and (time.time() - first_success_after_kill > 2):
            break
    recorder.end(test)

    if auto:
        try:
            print(f"[failover] restarting {primary_node['container']} ...")
            docker_start(primary_node["container"])
            print(f"[failover] {primary_node['container']} restarted; "
                  "it should rejoin as secondary")
        except subprocess.CalledProcessError as e:
            print(f"[failover] WARNING: docker start failed: {e}; "
                  "please restart manually")
    else:
        print(f"[failover] please restart manually: "
              f"docker start {primary_node['container']}")

    downtime = (first_success_after_kill - kill_time
                if first_success_after_kill else None)
    return {
        "primary_killed": primary_node["container"],
        "kill_time": kill_time,
        "recovery_time": first_success_after_kill,
        "downtime_seconds": downtime,
        "automated": auto,
    }

# Reporting

def write_charts(recorder: Recorder, charts_dir: Path) -> list[Path]:
    if plt is None:
        print("[report] matplotlib not installed — skipping charts")
        return []
    charts_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    summaries = recorder.summarise()
    by_test = {s.test: s for s in summaries}

    # 1. Throughput bar chart (writes vs reads)
    write_tests = [s for s in summaries if s.test.startswith("write.")]
    read_tests = [s for s in summaries if s.test.startswith("read.")]
    if write_tests:
        fig, ax = plt.subplots(figsize=(10, 5))
        names = [s.test for s in write_tests]
        ax.bar(names, [s.ops_per_sec for s in write_tests], color="#c44")
        ax.set_ylabel("ops / sec")
        ax.set_title("Write throughput by test")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        p = charts_dir / "write_throughput.png"
        plt.savefig(p, dpi=120); plt.close(fig); created.append(p)
    if read_tests:
        fig, ax = plt.subplots(figsize=(10, 5))
        names = [s.test for s in read_tests]
        ax.bar(names, [s.ops_per_sec for s in read_tests], color="#46a")
        ax.set_ylabel("ops / sec")
        ax.set_title("Read throughput by test")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        p = charts_dir / "read_throughput.png"
        plt.savefig(p, dpi=120); plt.close(fig); created.append(p)

    # 2. Latency percentile chart
    if write_tests + read_tests:
        fig, ax = plt.subplots(figsize=(11, 6))
        tests = [s for s in summaries if not s.test.startswith("failover.")]
        x = list(range(len(tests)))
        ax.bar([i - 0.3 for i in x], [t.p50_ms for t in tests],
               width=0.3, label="p50", color="#5a5")
        ax.bar(x, [t.p95_ms for t in tests],
               width=0.3, label="p95", color="#dd5")
        ax.bar([i + 0.3 for i in x], [t.p99_ms for t in tests],
               width=0.3, label="p99", color="#d55")
        ax.set_xticks(x)
        ax.set_xticklabels([t.test for t in tests], rotation=30, ha="right")
        ax.set_ylabel("latency (ms)")
        ax.set_title("Operation latency — p50 / p95 / p99")
        ax.legend()
        plt.tight_layout()
        p = charts_dir / "latency_percentiles.png"
        plt.savefig(p, dpi=120); plt.close(fig); created.append(p)

    # 3. Failover timeline
    failover_ops = [o for o in recorder.ops if o.test == "failover.write_probe"]
    if failover_ops:
        fig, ax = plt.subplots(figsize=(11, 4))
        xs = list(range(len(failover_ops)))
        first_post = next((i for i, o in enumerate(failover_ops)
                           if o.extra.get("phase") == "post_kill"), None)
        colors = ["#5a5" if o.ok else "#c44" for o in failover_ops]
        ax.bar(xs, [o.latency_ms for o in failover_ops], color=colors, width=1.0)
        if first_post is not None:
            ax.axvline(first_post - 0.5, color="black", linestyle="--",
                       label="primary killed")
            ax.legend(loc="upper right")
        ax.set_xlabel(f"probe # (one every ~{int(0.1*1000)}ms)")
        ax.set_ylabel("write latency (ms)")
        ax.set_title("Failover probe — green=success, red=failure")
        plt.tight_layout()
        p = charts_dir / "failover_timeline.png"
        plt.savefig(p, dpi=120); plt.close(fig); created.append(p)

    return created


def write_report(recorder: Recorder, info: dict[str, Any],
                 failover_result: dict[str, Any],
                 charts: list[Path], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summaries = recorder.summarise()
    write_rows = [s for s in summaries if s.test.startswith("write.")]
    read_rows = [s for s in summaries if s.test.startswith("read.")]

    def row(s: TestSummary) -> str:
        return (f"| `{s.test}` | {s.n} | {s.total_seconds:.2f} | "
                f"{s.ops_per_sec:.1f} | {fmt_ms(s.mean_ms)} | "
                f"{fmt_ms(s.p50_ms)} | {fmt_ms(s.p95_ms)} | "
                f"{fmt_ms(s.p99_ms)} | {fmt_ms(s.max_ms)} | {s.errors} |")

    header = ("| test | n | total (s) | ops/s | mean | p50 | p95 | p99 | "
              "max | errors |\n"
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
              "---: | ---: |")

    parts: list[str] = []
    parts.append(f"# MongoDB Replica-Set Benchmark — Phase 2\n")
    parts.append(f"_Generated: {datetime.now().isoformat(timespec='seconds')}_\n")
    parts.append("## 1. Setup\n")
    parts.append(f"- Replica set: `rs0` (mongo1 / mongo2 / mongo3)")
    parts.append(f"- Connection used: `{info.get('connection_label')}`")
    parts.append(f"- Dataset: MovieLens latest-small "
                 f"(movies={info.get('movies')}, "
                 f"ratings_seeded={info.get('ratings_seeded')}, "
                 f"ratings_holdout={info.get('ratings_holdout')}, "
                 f"tags={info.get('tags')})")
    parts.append("")

    parts.append("## 2. Write performance\n")
    parts.append(header)
    parts.extend(row(s) for s in write_rows)

    parts.append("\n## 3. Read performance\n")
    parts.append(header)
    parts.extend(row(s) for s in read_rows)

    parts.append("\n## 4. Failover\n")
    if failover_result.get("skipped"):
        parts.append(f"_Skipped: {failover_result.get('reason')}_")
    else:
        downtime = failover_result.get("downtime_seconds")
        parts.append(f"- Primary killed: **{failover_result.get('primary_killed')}**")
        parts.append(f"- Mode: {'automated (docker stop)' if failover_result.get('automated') else 'manual'}")
        if downtime is None:
            parts.append("- Recovery: writes **did not recover** within probe window")
        else:
            parts.append(f"- Write downtime: **{downtime:.2f} s** "
                         "(time from kill to first successful write)")
        fp = [o for o in recorder.ops if o.test == "failover.write_probe"]
        ok = sum(1 for o in fp if o.ok)
        parts.append(f"- Probes: {len(fp)} total, {ok} succeeded, {len(fp) - ok} failed")

    if charts:
        parts.append("\n## 5. Charts\n")
        for c in charts:
            rel = os.path.relpath(c, path.parent)
            parts.append(f"![{c.stem}]({rel})")

    parts.append("\n## 6. Notes\n")
    parts.append("- Latency reported is wall-clock time around each driver call.")
    parts.append("- `w1` = WriteConcern(w=1), `majority` = WriteConcern(w='majority').")
    parts.append("- `primary` reads go to the primary; `secondary_preferred` reads "
                 "are served from a secondary when one is available.")
    parts.append("- Raw per-op timings are in `results.csv` if you want to "
                 "compute different statistics.")
    parts.append("")

    path.write_text("\n".join(parts))
    print(f"[report] wrote {path}")

# Main

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-seed", action="store_true",
                    help="reuse the DB from a previous run (faster iteration)")
    ap.add_argument("--skip-failover", action="store_true",
                    help="don't run the failover test")
    ap.add_argument("--write-single", type=int, default=500,
                    help="number of single-doc inserts per write-concern")
    ap.add_argument("--write-bulk-batches", type=int, default=20,
                    help="number of bulk-insert batches per write-concern")
    ap.add_argument("--write-bulk-size", type=int, default=500,
                    help="docs per bulk batch")
    ap.add_argument("--write-updates", type=int, default=500,
                    help="number of single-doc updates")
    ap.add_argument("--reads", type=int, default=500,
                    help="number of read ops per read test")
    ap.add_argument("--aggregations", type=int, default=20,
                    help="number of aggregation runs per read pref")
    args = ap.parse_args(argv)

    banner("MongoDB replica-set benchmark")
    print(f"output dir: {RESULTS_DIR}")

    # 1. dataset
    banner("Step 1 / 4 — Dataset")
    csvs = download_dataset()

    # 2. connect
    banner("Step 2 / 4 — Connect")
    rs_client = connect_replica_set()
    directs = connect_directs()
    info: dict[str, Any] = {}

    if rs_client is not None:
        client = rs_client
        info["connection_label"] = "replica-set client"
    else:
        # use a direct connection to whichever node is primary
        found = find_primary_direct(directs)
        if found is None:
            print("FATAL: no node responded as primary. Is the stack up?")
            return 2
        client, primary_node = found
        info["connection_label"] = f"direct connection to {primary_node['container']}"
        print(f"[conn] no replica-set client; using direct connection to "
              f"{primary_node['container']} for the benchmark")
        print("[conn] tip: add `127.0.0.1 mongo1 mongo2 mongo3` to /etc/hosts "
              "to get full replica-set semantics from the host")

    # 3. seed
    banner("Step 3 / 4 — Seed dataset")
    if args.skip_seed:
        print("[setup] --skip-seed set; using existing DB contents")
        db = client[DB_NAME]
        info["movies"] = db.movies.estimated_document_count()
        info["ratings_seeded"] = db.ratings.estimated_document_count()
        info["tags"] = db.tags.estimated_document_count()
        info["ratings_holdout"] = len(load_holdout())
        if info["ratings_holdout"] == 0:
            print("WARN: no holdout file found; write tests will reuse old _ids")
    else:
        reset_db(client)
        info.update(load_seed_data(client, csvs))

    holdout = load_holdout()
    if not holdout:
        holdout = [{"userId": i, "movieId": 1, "rating": 5.0,
                    "timestamp": int(time.time())} for i in range(500)]

    # 4. benchmarks
    banner("Step 4 / 4 — Benchmarks")
    rec = Recorder()

    # writes
    bench_write_single(client, holdout, rec, "w1", args.write_single)
    bench_write_single(client, holdout, rec, "majority", max(50, args.write_single // 4))
    bench_write_bulk(client, holdout, rec, "w1",
                     args.write_bulk_batches, args.write_bulk_size)
    bench_write_bulk(client, holdout, rec, "majority",
                     max(5, args.write_bulk_batches // 4), args.write_bulk_size)
    bench_update(client, rec, args.write_updates)

    # reads
    bench_read_by_id(client, rec, "primary", args.reads)
    bench_read_by_id(client, rec, "secondary_preferred", args.reads)
    bench_read_filter(client, rec, "primary", args.reads)
    bench_read_filter(client, rec, "secondary_preferred", args.reads)
    bench_aggregation(client, rec, "primary", args.aggregations)
    bench_aggregation(client, rec, "secondary_preferred", args.aggregations)

    # failover
    failover_result: dict[str, Any] = {"skipped": True, "reason": "--skip-failover"}
    if not args.skip_failover:
        banner("Failover test")
        failover_result = bench_failover(rs_client, directs, rec)

    # output
    banner("Writing results")
    rec.write_csv(RESULTS_DIR / "results.csv")
    charts = write_charts(rec, CHARTS_DIR)
    write_report(rec, info, failover_result, charts, RESULTS_DIR / "report.md")

    # quick console summary
    print("\nSummary:")
    for s in rec.summarise():
        print(f"  {s.test:38s}  n={s.n:5d}  ops/s={s.ops_per_sec:8.1f}  "
              f"p50={fmt_ms(s.p50_ms):>9s}  p95={fmt_ms(s.p95_ms):>9s}  "
              f"errs={s.errors}")
    if not failover_result.get("skipped"):
        d = failover_result.get("downtime_seconds")
        print(f"  failover downtime: "
              f"{d:.2f}s" if d is not None else "  failover: did not recover")

    print(f"\nDone. See {RESULTS_DIR / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
