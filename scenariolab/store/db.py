"""M5 ResultStore: the single write path into lab.sqlite (DESIGN §8, FR-D1..D3).

WAL mode + a generous busy timeout make concurrent worker writes safe; every
write goes through this class so raw SQL never spreads through the codebase.
The DB stores summaries and paths; full PlannerOutput JSON stays in files.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scenariolab.generator.cluster_gen import ClusterSummary
from scenariolab.generator.slo_gen import ServiceSummary

SCHEMA_VERSION = 1
SCHEMA_PATH = Path(__file__).parent / "schema.sql"

#: Scenario state machine (DESIGN §2.3).
PENDING = "PENDING"
RUNNING = "RUNNING"
DONE_FEASIBLE = "DONE_FEASIBLE"
DONE_INFEASIBLE = "DONE_INFEASIBLE"
ERROR = "ERROR"
VERIFIED = "VERIFIED"

DONE_STATES = (DONE_FEASIBLE, DONE_INFEASIBLE, VERIFIED)


class StoreError(RuntimeError):
    """Raised on schema-version mismatch or malformed store usage."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ResultStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._init_schema()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> ResultStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- schema ------------------------------------------------------------ #

    def _init_schema(self) -> None:
        version = self._conn.execute("PRAGMA user_version").fetchone()[0]
        if version == 0:
            self._conn.executescript(SCHEMA_PATH.read_text())
            self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            self._conn.commit()
        elif version != SCHEMA_VERSION:
            raise StoreError(
                f"{self.db_path}: schema version {version} != expected "
                f"{SCHEMA_VERSION}. No automatic migration exists; move the old "
                "DB aside or export it first."
            )

    # -- batch / matrix registration ---------------------------------------- #

    def register_batch(
        self, batch_id: str, config_yaml: str, config_hash: str, master_seed: int
    ) -> None:
        self._conn.execute(
            "INSERT INTO batches (batch_id, config_yaml, config_hash, master_seed, "
            "status, started_at) VALUES (?, ?, ?, ?, 'RUNNING', ?) "
            "ON CONFLICT(batch_id) DO UPDATE SET status='RUNNING'",
            (batch_id, config_yaml, config_hash, master_seed, _now()),
        )
        self._conn.commit()

    def finish_batch(self, batch_id: str, status: str = "DONE") -> None:
        self._conn.execute(
            "UPDATE batches SET status=?, finished_at=? WHERE batch_id=?",
            (status, _now(), batch_id),
        )
        self._conn.commit()

    def upsert_cluster(self, summary: ClusterSummary) -> None:
        row = asdict(summary)
        self._conn.execute(
            "INSERT OR REPLACE INTO clusters (cluster_id, seed, yaml_path, num_nodes, "
            "num_accels, num_free_accels, classes_json, num_islands, has_npu) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                row["cluster_id"], row["seed"], str(row["yaml_path"]),
                row["num_nodes"], row["num_accels"], row["num_free_accels"],
                json.dumps(row["classes"]), row["num_islands"], int(row["has_npu"]),
            ),
        )
        self._conn.commit()

    def upsert_service(self, summary: ServiceSummary) -> None:
        row = asdict(summary)
        self._conn.execute(
            "INSERT OR REPLACE INTO services (service_id, seed, yaml_path, model, "
            "rps, ttft_p99_ms, tpot_p99_ms, power_cap_w) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                row["service_id"], row["seed"], str(row["yaml_path"]), row["model"],
                row["rps"], row["ttft_p99_ms"], row["tpot_p99_ms"], row["power_cap_w"],
            ),
        )
        self._conn.commit()

    def register_scenario(
        self, scenario_id: str, batch_id: str, cluster_id: str, service_id: str, seed: int
    ) -> None:
        """Idempotent: an existing scenario keeps its status (resume, FR-B4)."""
        self._conn.execute(
            "INSERT OR IGNORE INTO scenarios (scenario_id, batch_id, cluster_id, "
            "service_id, seed) VALUES (?, ?, ?, ?, ?)",
            (scenario_id, batch_id, cluster_id, service_id, seed),
        )
        self._conn.commit()

    # -- execution ----------------------------------------------------------- #

    def pending_scenarios(self, batch_id: str) -> list[sqlite3.Row]:
        """Rows a (re)run must execute: PENDING, ERROR (resume retries them,
        DESIGN §2.3) and RUNNING (stale claims from an interrupted run - the
        runner is single-invocation per batch, so nothing else owns them)."""
        return list(
            self._conn.execute(
                "SELECT * FROM scenarios WHERE batch_id=? AND status IN (?, ?, ?) "
                "ORDER BY scenario_id",
                (batch_id, PENDING, ERROR, RUNNING),
            )
        )

    def mark_running(self, scenario_id: str) -> None:
        self._conn.execute(
            "UPDATE scenarios SET status=?, attempts=attempts+1 WHERE scenario_id=?",
            (RUNNING, scenario_id),
        )
        self._conn.commit()

    def mark_error(self, scenario_id: str, error_text: str, elapsed_s: float) -> None:
        self._conn.execute(
            "UPDATE scenarios SET status=?, error_text=?, elapsed_s=? WHERE scenario_id=?",
            (ERROR, error_text, elapsed_s, scenario_id),
        )
        self._conn.commit()

    def insert_result(
        self, scenario_id: str, elapsed_s: float, result: dict[str, Any]
    ) -> None:
        """Store one ScenarioResult summary and flip the scenario to DONE_*."""
        status = DONE_FEASIBLE if result["feasible"] else DONE_INFEASIBLE
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO results (scenario_id, feasible, fidelity, "
                "calibrated, npu_extrapolated, plan_json_path, p99_ttft_ms, "
                "p99_tpot_ms, avg_power_w, peak_power_w, tokens_per_joule, "
                "slo_goodput, active_devices, baseline_power_w, power_saving_pct, "
                "baseline_note, violated_json, provenance_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    scenario_id,
                    int(result["feasible"]),
                    result["fidelity"],
                    int(result.get("calibrated", False)),
                    int(result.get("npu_extrapolated", False)),
                    result["plan_json_path"],
                    result.get("p99_ttft_ms"),
                    result.get("p99_tpot_ms"),
                    result.get("avg_power_w"),
                    result.get("peak_power_w"),
                    result.get("tokens_per_joule"),
                    result.get("slo_goodput"),
                    result.get("active_devices"),
                    result.get("baseline_power_w"),
                    result.get("power_saving_pct"),
                    result.get("baseline_note"),
                    json.dumps(result.get("violated", []), sort_keys=True),
                    json.dumps(result.get("provenance", {}), sort_keys=True),
                ),
            )
            self._conn.execute(
                "UPDATE scenarios SET status=?, error_text=NULL, elapsed_s=? "
                "WHERE scenario_id=?",
                (status, elapsed_s, scenario_id),
            )

    def insert_verification(self, scenario_id: str, record: dict[str, Any]) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO verifications (scenario_id, sim_p99_ttft_ms, "
                "sim_p99_tpot_ms, sim_avg_power_w, err_ttft_pct, err_tpot_pct, "
                "err_power_pct, selection_flipped, feasibility_flipped, "
                "regret_energy_pct) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    scenario_id,
                    record.get("sim_p99_ttft_ms"),
                    record.get("sim_p99_tpot_ms"),
                    record.get("sim_avg_power_w"),
                    record.get("err_ttft_pct"),
                    record.get("err_tpot_pct"),
                    record.get("err_power_pct"),
                    _opt_int(record.get("selection_flipped")),
                    _opt_int(record.get("feasibility_flipped")),
                    record.get("regret_energy_pct"),
                ),
            )
            self._conn.execute(
                "UPDATE scenarios SET status=? WHERE scenario_id=? AND status IN (?, ?)",
                (VERIFIED, scenario_id, DONE_FEASIBLE, DONE_INFEASIBLE),
            )

    def verification_pool(self, batch_id: str) -> list[sqlite3.Row]:
        """DONE scenarios joined with what stratified sampling and error
        computation need (§7.4). VERIFIED rows are excluded: re-verifying
        would overwrite a record without adding information."""
        return list(
            self._conn.execute(
                "SELECT s.scenario_id, s.cluster_id, s.service_id, s.seed, "
                "r.feasible, r.plan_json_path, r.p99_ttft_ms, r.p99_tpot_ms, "
                "r.avg_power_w, c.num_accels, c.has_npu, "
                "cl.yaml_path AS cluster_yaml, sv.yaml_path AS service_yaml "
                "FROM scenarios s "
                "JOIN results r ON r.scenario_id = s.scenario_id "
                "JOIN clusters c ON c.cluster_id = s.cluster_id "
                "JOIN clusters cl ON cl.cluster_id = s.cluster_id "
                "JOIN services sv ON sv.service_id = s.service_id "
                "WHERE s.batch_id=? AND s.status IN (?, ?) "
                "ORDER BY s.scenario_id",
                (batch_id, DONE_FEASIBLE, DONE_INFEASIBLE),
            )
        )

    # -- queries -------------------------------------------------------------- #

    def scenario_counts(self, batch_id: str) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT status, COUNT(*) AS n FROM scenarios WHERE batch_id=? GROUP BY status",
            (batch_id,),
        )
        return {row["status"]: row["n"] for row in rows}

    def batch_summary(self, batch_id: str) -> dict[str, Any]:
        counts = self.scenario_counts(batch_id)
        agg = self._conn.execute(
            "SELECT COUNT(*) AS done, SUM(r.feasible) AS feasible, "
            "AVG(r.power_saving_pct) AS mean_saving "
            "FROM results r JOIN scenarios s ON s.scenario_id = r.scenario_id "
            "WHERE s.batch_id=?",
            (batch_id,),
        ).fetchone()
        savings = [
            row[0] for row in self._conn.execute(
                "SELECT r.power_saving_pct FROM results r "
                "JOIN scenarios s ON s.scenario_id = r.scenario_id "
                "WHERE s.batch_id=? AND r.power_saving_pct IS NOT NULL "
                "ORDER BY r.power_saving_pct",
                (batch_id,),
            )
        ]
        median = savings[len(savings) // 2] if savings else None
        done = agg["done"] or 0
        return {
            "counts": counts,
            "done": done,
            "feasible": agg["feasible"] or 0,
            "feasible_rate": (agg["feasible"] or 0) / done if done else None,
            "median_power_saving_pct": median,
        }

    def query_results(
        self,
        batch_id: str | None = None,
        *,
        feasible: bool | None = None,
        fidelity: str | None = None,
        has_npu: bool | None = None,
        min_saving_pct: float | None = None,
        cluster_id: str | None = None,
        service_id: str | None = None,
        order_by: str = "scenario_id",
        descending: bool = False,
        page: int = 1,
        page_size: int = 50,
    ) -> list[sqlite3.Row]:
        """Filtered, ordered, paginated result rows (FR-D3)."""
        allowed_order = {
            "scenario_id", "avg_power_w", "power_saving_pct", "p99_ttft_ms",
            "p99_tpot_ms", "tokens_per_joule",
        }
        if order_by not in allowed_order:
            raise StoreError(f"order_by must be one of {sorted(allowed_order)}")
        clauses = []
        params: list[Any] = []
        if batch_id is not None:
            clauses.append("s.batch_id = ?")
            params.append(batch_id)
        if feasible is not None:
            clauses.append("r.feasible = ?")
            params.append(int(feasible))
        if fidelity is not None:
            clauses.append("r.fidelity = ?")
            params.append(fidelity)
        if has_npu is not None:
            clauses.append("c.has_npu = ?")
            params.append(int(has_npu))
        if min_saving_pct is not None:
            clauses.append("r.power_saving_pct >= ?")
            params.append(min_saving_pct)
        if cluster_id is not None:
            clauses.append("s.cluster_id = ?")
            params.append(cluster_id)
        if service_id is not None:
            clauses.append("s.service_id = ?")
            params.append(service_id)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        direction = "DESC" if descending else "ASC"
        params.extend([page_size, (page - 1) * page_size])
        return list(
            self._conn.execute(
                "SELECT s.scenario_id, s.batch_id, s.cluster_id, s.service_id, "
                "s.status, s.elapsed_s, r.*, c.has_npu "
                "FROM results r "
                "JOIN scenarios s ON s.scenario_id = r.scenario_id "
                "JOIN clusters c ON c.cluster_id = s.cluster_id "
                f"{where} ORDER BY r.{order_by} {direction}, s.scenario_id "
                "LIMIT ? OFFSET ?",
                params,
            )
        )

    def dump_for_golden(self, batch_id: str) -> dict[str, Any]:
        """Deterministic DB image for golden tests: no timestamps, no elapsed."""
        scenarios = [
            dict(row) for row in self._conn.execute(
                "SELECT scenario_id, cluster_id, service_id, seed, status "
                "FROM scenarios WHERE batch_id=? ORDER BY scenario_id",
                (batch_id,),
            )
        ]
        results = []
        for row in self._conn.execute(
            "SELECT r.* FROM results r JOIN scenarios s ON s.scenario_id=r.scenario_id "
            "WHERE s.batch_id=? ORDER BY r.scenario_id",
            (batch_id,),
        ):
            d = dict(row)
            d.pop("provenance_json", None)  # carries timestamps/hostnames
            results.append(d)
        return {"scenarios": scenarios, "results": results}


def _opt_int(value: Any) -> int | None:
    return None if value is None else int(value)
