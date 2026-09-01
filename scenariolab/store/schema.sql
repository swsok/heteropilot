-- ScenarioLab ResultStore schema (DESIGN §8.2).
-- Version is tracked via PRAGMA user_version, set by db.py to SCHEMA_VERSION.
-- Full PlannerOutput JSON lives in files; the DB holds paths + summary columns.

CREATE TABLE IF NOT EXISTS batches (
    batch_id     TEXT PRIMARY KEY,
    config_yaml  TEXT NOT NULL,
    config_hash  TEXT NOT NULL,
    master_seed  INTEGER NOT NULL,
    status       TEXT NOT NULL DEFAULT 'PENDING',
    started_at   TEXT,
    finished_at  TEXT
);

CREATE TABLE IF NOT EXISTS clusters (
    cluster_id      TEXT PRIMARY KEY,
    seed            INTEGER NOT NULL,
    yaml_path       TEXT NOT NULL,
    num_nodes       INTEGER NOT NULL,
    num_accels      INTEGER NOT NULL,
    num_free_accels INTEGER NOT NULL,
    classes_json    TEXT NOT NULL,
    num_islands     INTEGER NOT NULL,
    has_npu         INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS services (
    service_id  TEXT PRIMARY KEY,
    seed        INTEGER NOT NULL,
    yaml_path   TEXT NOT NULL,
    model       TEXT NOT NULL,
    rps         REAL NOT NULL,
    ttft_p99_ms REAL NOT NULL,
    tpot_p99_ms REAL NOT NULL,
    power_cap_w REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS scenarios (
    scenario_id TEXT PRIMARY KEY,
    batch_id    TEXT NOT NULL REFERENCES batches(batch_id),
    cluster_id  TEXT NOT NULL REFERENCES clusters(cluster_id),
    service_id  TEXT NOT NULL REFERENCES services(service_id),
    seed        INTEGER NOT NULL,
    status      TEXT NOT NULL DEFAULT 'PENDING',
    attempts    INTEGER NOT NULL DEFAULT 0,
    error_text  TEXT,
    elapsed_s   REAL
);

CREATE INDEX IF NOT EXISTS idx_scenarios_batch ON scenarios(batch_id, status);

CREATE TABLE IF NOT EXISTS results (
    scenario_id       TEXT PRIMARY KEY REFERENCES scenarios(scenario_id),
    feasible          INTEGER NOT NULL,
    fidelity          TEXT NOT NULL,
    calibrated        INTEGER NOT NULL DEFAULT 0,
    npu_extrapolated  INTEGER NOT NULL DEFAULT 0,
    plan_json_path    TEXT NOT NULL,
    p99_ttft_ms       REAL,
    p99_tpot_ms       REAL,
    avg_power_w       REAL,
    peak_power_w      REAL,
    tokens_per_joule  REAL,
    slo_goodput       REAL,
    active_devices    INTEGER,
    baseline_power_w  REAL,
    power_saving_pct  REAL,
    baseline_note     TEXT,
    violated_json     TEXT,
    provenance_json   TEXT
);

CREATE TABLE IF NOT EXISTS verifications (
    scenario_id        TEXT PRIMARY KEY REFERENCES scenarios(scenario_id),
    sim_p99_ttft_ms    REAL,
    sim_p99_tpot_ms    REAL,
    sim_avg_power_w    REAL,
    err_ttft_pct       REAL,
    err_tpot_pct       REAL,
    err_power_pct      REAL,
    selection_flipped  INTEGER,
    feasibility_flipped INTEGER,
    regret_energy_pct  REAL
);
