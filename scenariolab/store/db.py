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

SCHEMA_VERSION = 3
SCHEMA_PATH = Path(__file__).parent / "schema.sql"

#: Placement state machine (workspace work order §1.3).
PLANNING = "PLANNING"
PLACED = "PLACED"
REJECTED = "REJECTED"
FAILED = "FAILED"
REMOVED = "REMOVED"

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
    def __init__(self, db_path: str | Path, *, readonly: bool = False) -> None:
        """`readonly=True` opens the file with SQLite's mode=ro so a reader
        (the web API, FR-A6) can never mutate results, and refuses to create
        a missing DB rather than serving an empty one."""
        self.db_path = Path(db_path)
        self.readonly = readonly
        if readonly:
            if not self.db_path.is_file():
                raise StoreError(f"{self.db_path}: no result store at this path")
            self._conn = sqlite3.connect(
                f"file:{self.db_path}?mode=ro", uri=True, timeout=30.0,
                check_same_thread=False,
            )
        else:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.db_path, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        if not readonly:
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
            if self.readonly:
                raise StoreError(
                    f"{self.db_path}: not a ScenarioLab result store (no schema)"
                )
            self._conn.executescript(SCHEMA_PATH.read_text())
            self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            self._conn.commit()
        elif version < SCHEMA_VERSION and not self.readonly:
            # Every migration so far is purely additive (v2: plan_queries;
            # v3: workspace/placement tables + origin/link columns), so the
            # chain runs in place. A non-additive change must break this
            # pattern loudly instead of extending it (FR-D2).
            self._conn.executescript(SCHEMA_PATH.read_text())
            if version < 3:
                self._add_column_if_missing(
                    "clusters", "origin", "TEXT NOT NULL DEFAULT 'random'"
                )
                self._add_column_if_missing("clusters", "build_request_json", "TEXT")
                self._add_column_if_missing("clusters", "link_summary", "TEXT")
                self._add_column_if_missing(
                    "services", "origin", "TEXT NOT NULL DEFAULT 'random'"
                )
            self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            self._conn.commit()
        elif version != SCHEMA_VERSION:
            raise StoreError(
                f"{self.db_path}: schema version {version} != expected "
                f"{SCHEMA_VERSION}."
                + (
                    " Open it once read-write to migrate (additive migrations "
                    "are automatic)."
                    if version < SCHEMA_VERSION else
                    " No automatic migration exists; move the old DB aside or "
                    "export it first."
                )
            )

    def _add_column_if_missing(self, table: str, column: str, decl: str) -> None:
        existing = {
            row["name"]
            for row in self._conn.execute(f"PRAGMA table_info({table})")
        }
        if column not in existing:
            self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

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
            "num_accels, num_free_accels, classes_json, num_islands, has_npu, "
            "origin, build_request_json, link_summary) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                row["cluster_id"], row["seed"], str(row["yaml_path"]),
                row["num_nodes"], row["num_accels"], row["num_free_accels"],
                json.dumps(row["classes"]), row["num_islands"], int(row["has_npu"]),
                row["origin"], row["build_request_json"], row["link_summary"],
            ),
        )
        self._conn.commit()

    def upsert_service(self, summary: ServiceSummary, *, origin: str = "random") -> None:
        row = asdict(summary)
        self._conn.execute(
            "INSERT OR REPLACE INTO services (service_id, seed, yaml_path, model, "
            "rps, ttft_p99_ms, tpot_p99_ms, power_cap_w, origin) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                row["service_id"], row["seed"], str(row["yaml_path"]), row["model"],
                row["rps"], row["ttft_p99_ms"], row["tpot_p99_ms"], row["power_cap_w"],
                origin,
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

    # -- workspaces / placements (workspace work order §2.2) ------------------- #

    def create_workspace(
        self,
        cluster_id: str,
        name: str,
        *,
        cluster_yaml_hash: str | None = None,
        total_power_cap_w: float | None = None,
    ) -> str:
        """Create a workspace over a cluster; all devices start FREE (the
        occupancy overlay is derived from PLACED placements, never stored)."""
        with self._conn:
            count = self._conn.execute(
                "SELECT COUNT(*) FROM workspaces"
            ).fetchone()[0]
            workspace_id = f"ws{count + 1:04d}"
            self._conn.execute(
                "INSERT INTO workspaces (workspace_id, cluster_id, name, created_at, "
                "status, cluster_yaml_hash, total_power_cap_w) "
                "VALUES (?, ?, ?, ?, 'ACTIVE', ?, ?)",
                (workspace_id, cluster_id, name, _now(), cluster_yaml_hash,
                 total_power_cap_w),
            )
        return workspace_id

    def get_workspace(self, workspace_id: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM workspaces WHERE workspace_id = ?", (workspace_id,)
        ).fetchone()

    def list_workspaces(self) -> list[sqlite3.Row]:
        return list(
            self._conn.execute(
                "SELECT w.*, "
                "  (SELECT COUNT(*) FROM placements p "
                "   WHERE p.workspace_id = w.workspace_id AND p.status = 'PLACED') "
                "  AS placed_count "
                "FROM workspaces w ORDER BY w.workspace_id"
            )
        )

    def workspace_count_for_cluster(self, cluster_id: str) -> int:
        return int(self._conn.execute(
            "SELECT COUNT(*) FROM workspaces WHERE cluster_id = ?", (cluster_id,)
        ).fetchone()[0])

    def workspace_placements(
        self, workspace_id: str, *, include_removed: bool = True
    ) -> list[sqlite3.Row]:
        query = (
            "SELECT p.*, s.model, s.rps, s.ttft_p99_ms AS slo_ttft_ms, "
            "s.tpot_p99_ms AS slo_tpot_ms, s.power_cap_w, s.origin AS service_origin "
            "FROM placements p JOIN services s ON s.service_id = p.service_id "
            "WHERE p.workspace_id = ?"
        )
        if not include_removed:
            query += " AND p.status != 'REMOVED'"
        return list(self._conn.execute(query + " ORDER BY p.seq", (workspace_id,)))

    def placed_devices(self, workspace_id: str) -> set[str]:
        """The occupancy overlay: union of devices of PLACED placements."""
        devices: set[str] = set()
        for row in self._conn.execute(
            "SELECT devices_json FROM placements WHERE workspace_id = ? "
            "AND status = 'PLACED'",
            (workspace_id,),
        ):
            devices.update(devices_from_json(row["devices_json"]))
        return devices

    def insert_placement(
        self,
        workspace_id: str,
        service_id: str,
        *,
        status: str,
        devices: dict[str, str] | list[str],
        record: dict[str, Any],
    ) -> str:
        """One placement row in its initial state (PLANNING preview, or a
        terminal REJECTED/FAILED that never occupies devices)."""
        with self._conn:
            placement_id = self._insert_placement_row(
                workspace_id, service_id, status, devices, record
            )
        return placement_id

    def _insert_placement_row(
        self,
        workspace_id: str,
        service_id: str,
        status: str,
        devices: dict[str, str] | list[str],
        record: dict[str, Any],
    ) -> str:
        """The bare INSERT; the caller owns the transaction."""
        seq = self._conn.execute(
            "SELECT COALESCE(MAX(seq), 0) + 1 FROM placements "
            "WHERE workspace_id = ?",
            (workspace_id,),
        ).fetchone()[0]
        placement_id = f"{workspace_id}-p{seq:04d}"
        self._conn.execute(
            "INSERT INTO placements (placement_id, workspace_id, service_id, "
            "seq, status, devices_json, plan_json_path, fidelity, calibrated, "
            "slo_ttft_ok, slo_tpot_ok, p99_ttft_ms, p99_tpot_ms, avg_power_w, "
            "peak_power_w, tokens_per_joule, shared_fabric_warning, "
            "npu_extrapolated, rejected_reason_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                placement_id, workspace_id, service_id, seq, status,
                json.dumps(devices, sort_keys=True)
                if isinstance(devices, dict) else json.dumps(sorted(devices)),
                record.get("plan_json_path"),
                record.get("fidelity"),
                _opt_int(record.get("calibrated")),
                _opt_int(record.get("slo_ttft_ok")),
                _opt_int(record.get("slo_tpot_ok")),
                record.get("p99_ttft_ms"),
                record.get("p99_tpot_ms"),
                record.get("avg_power_w"),
                record.get("peak_power_w"),
                record.get("tokens_per_joule"),
                _opt_int(record.get("shared_fabric_warning")),
                _opt_int(record.get("npu_extrapolated")),
                json.dumps(record.get("rejected_reason"))
                if record.get("rejected_reason") is not None else None,
                _now(),
            ),
        )
        return placement_id

    def replace_all_placements(
        self,
        workspace_id: str,
        entries: list[tuple[str, str, dict[str, str] | list[str], dict[str, Any]]],
    ) -> list[str]:
        """Atomic replan swap (§5.5): every current PLACED placement becomes
        REMOVED and the new set is inserted, in ONE transaction - a reader
        never observes a half-replanned workspace."""
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            self._conn.execute(
                "UPDATE placements SET status = 'REMOVED', removed_at = ? "
                "WHERE workspace_id = ? AND status = 'PLACED'",
                (_now(), workspace_id),
            )
            ids = [
                self._insert_placement_row(
                    workspace_id, service_id, status, devices, record
                )
                for service_id, status, devices, record in entries
            ]
            self._conn.execute("COMMIT")
            return ids
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise

    def archive_workspace(self, workspace_id: str) -> None:
        """ACTIVE -> ARCHIVED: the workspace becomes read-only history."""
        with self._conn:
            row = self._conn.execute(
                "SELECT status FROM workspaces WHERE workspace_id = ?",
                (workspace_id,),
            ).fetchone()
            if row is None:
                raise StoreError(f"no workspace '{workspace_id}'")
            self._conn.execute(
                "UPDATE workspaces SET status = 'ARCHIVED' WHERE workspace_id = ?",
                (workspace_id,),
            )

    def set_placement_plan_path(self, placement_id: str, path: str) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE placements SET plan_json_path = ? WHERE placement_id = ?",
                (path, placement_id),
            )

    def get_placement(self, placement_id: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM placements WHERE placement_id = ?", (placement_id,)
        ).fetchone()

    def confirm_placement(self, placement_id: str) -> None:
        """PLANNING -> PLACED, atomically (FR-P2).

        BEGIN IMMEDIATE serializes confirmations; inside the transaction the
        devices are re-checked against the CURRENT overlay, so two previews
        computed from the same FREE state can never both occupy a device.
        """
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            row = self._conn.execute(
                "SELECT workspace_id, status, devices_json FROM placements "
                "WHERE placement_id = ?",
                (placement_id,),
            ).fetchone()
            if row is None:
                raise StoreError(f"no placement '{placement_id}'")
            if row["status"] != PLANNING:
                raise StoreError(
                    f"placement '{placement_id}' is {row['status']}, not PLANNING"
                )
            devices = devices_from_json(row["devices_json"])
            taken = devices & self.placed_devices(row["workspace_id"])
            if taken:
                raise StoreError(
                    f"devices already occupied by a concurrent placement: "
                    f"{sorted(taken)}; re-plan against the current workspace state"
                )
            self._conn.execute(
                "UPDATE placements SET status = 'PLACED' WHERE placement_id = ?",
                (placement_id,),
            )
            self._conn.execute("COMMIT")
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise

    def remove_placement(self, placement_id: str) -> None:
        """PLACED -> REMOVED; the devices return to FREE (overlay is derived)."""
        with self._conn:
            row = self._conn.execute(
                "SELECT status FROM placements WHERE placement_id = ?",
                (placement_id,),
            ).fetchone()
            if row is None:
                raise StoreError(f"no placement '{placement_id}'")
            if row["status"] != PLACED:
                raise StoreError(
                    f"placement '{placement_id}' is {row['status']}, not PLACED"
                )
            self._conn.execute(
                "UPDATE placements SET status = 'REMOVED', removed_at = ? "
                "WHERE placement_id = ?",
                (_now(), placement_id),
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

    def record_plan_query(
        self,
        *,
        cluster_id: str,
        slo: dict[str, Any],
        seed: int,
        num_requests: int,
        feasible: bool | None,
        fidelity: str | None,
        truncated: bool | None,
        elapsed_s: float | None,
    ) -> None:
        """Interactive query history (§2.2). The web layer calls this through a
        dedicated read-write ResultStore, never through its read-only one."""
        if self.readonly:
            raise StoreError("plan-query history requires a read-write store")
        with self._conn:
            self._conn.execute(
                "INSERT INTO plan_queries (created_at, cluster_id, slo_json, seed, "
                "num_requests, feasible, fidelity, truncated, elapsed_s) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    _now(), cluster_id, json.dumps(slo, sort_keys=True), seed,
                    num_requests,
                    None if feasible is None else int(feasible),
                    fidelity,
                    None if truncated is None else int(truncated),
                    elapsed_s,
                ),
            )

    # -- web API queries (all read-only, FR-D3/FR-A6) -------------------------- #

    def list_batches(self) -> list[sqlite3.Row]:
        return list(
            self._conn.execute(
                "SELECT batch_id, config_hash, master_seed, status, started_at, "
                "finished_at FROM batches ORDER BY batch_id"
            )
        )

    def count_results(
        self,
        batch_id: str | None = None,
        *,
        feasible: bool | None = None,
        fidelity: str | None = None,
        has_npu: bool | None = None,
        min_saving_pct: float | None = None,
        cluster_id: str | None = None,
        service_id: str | None = None,
    ) -> int:
        clauses = []
        params: list[Any] = []
        for clause, value in (
            ("s.batch_id = ?", batch_id),
            ("r.feasible = ?", None if feasible is None else int(feasible)),
            ("r.fidelity = ?", fidelity),
            ("c.has_npu = ?", None if has_npu is None else int(has_npu)),
            ("r.power_saving_pct >= ?", min_saving_pct),
            ("s.cluster_id = ?", cluster_id),
            ("s.service_id = ?", service_id),
        ):
            if value is not None:
                clauses.append(clause)
                params.append(value)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM results r "
            "JOIN scenarios s ON s.scenario_id = r.scenario_id "
            "JOIN clusters c ON c.cluster_id = s.cluster_id "
            f"{where}",
            params,
        ).fetchone()
        return int(row[0])

    def get_scenario(self, scenario_id: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT s.*, r.feasible, r.fidelity, r.calibrated, r.npu_extrapolated, "
            "r.plan_json_path, r.p99_ttft_ms, r.p99_tpot_ms, r.avg_power_w, "
            "r.peak_power_w, r.tokens_per_joule, r.slo_goodput, r.active_devices, "
            "r.baseline_power_w, r.power_saving_pct, r.baseline_note, "
            "r.violated_json "
            "FROM scenarios s LEFT JOIN results r ON r.scenario_id = s.scenario_id "
            "WHERE s.scenario_id = ?",
            (scenario_id,),
        ).fetchone()

    def get_verification(self, scenario_id: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM verifications WHERE scenario_id = ?", (scenario_id,)
        ).fetchone()

    def list_clusters(self) -> list[sqlite3.Row]:
        return list(self._conn.execute("SELECT * FROM clusters ORDER BY cluster_id"))

    def get_cluster(self, cluster_id: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM clusters WHERE cluster_id = ?", (cluster_id,)
        ).fetchone()

    def list_services(self) -> list[sqlite3.Row]:
        return list(self._conn.execute("SELECT * FROM services ORDER BY service_id"))

    def get_service(self, service_id: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM services WHERE service_id = ?", (service_id,)
        ).fetchone()

    def verification_rows(self, batch_id: str) -> list[sqlite3.Row]:
        return list(
            self._conn.execute(
                "SELECT v.*, r.p99_ttft_ms AS fast_p99_ttft_ms, "
                "r.p99_tpot_ms AS fast_p99_tpot_ms, r.avg_power_w AS fast_avg_power_w, "
                "r.feasible, r.fidelity "
                "FROM verifications v "
                "JOIN scenarios s ON s.scenario_id = v.scenario_id "
                "JOIN results r ON r.scenario_id = v.scenario_id "
                "WHERE s.batch_id = ? ORDER BY v.scenario_id",
                (batch_id,),
            )
        )

    def dashboard_rows(self, batch_id: str) -> list[sqlite3.Row]:
        """Everything the dashboard bins server-side (FR-U6: the browser never
        pulls the full scenario table)."""
        return list(
            self._conn.execute(
                "SELECT r.feasible, r.fidelity, r.power_saving_pct, "
                "r.npu_extrapolated, r.calibrated, sv.power_cap_w, "
                "sv.tpot_p99_ms, sv.ttft_p99_ms, c.num_accels "
                "FROM results r "
                "JOIN scenarios s ON s.scenario_id = r.scenario_id "
                "JOIN services sv ON sv.service_id = s.service_id "
                "JOIN clusters c ON c.cluster_id = s.cluster_id "
                "WHERE s.batch_id = ?",
                (batch_id,),
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


def devices_from_json(text: str) -> set[str]:
    """devices_json holds either a {device: role} map (placements made since
    roles were recorded) or a plain list; both mean 'these devices'."""
    obj = json.loads(text)
    return set(obj.keys()) if isinstance(obj, dict) else set(obj)
