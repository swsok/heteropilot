"""`plan --accuracy-domain` end to end (uncertainty work order STEP A4).

One flag serves both entry points (D33): bare, it reads every domain under
profiles/calibration/ (rps STEP 4); with files, exactly those. Under it a
candidate whose operating point a `refuse` domain does not cover is rejected as
`outside_calibration_domain`, the uncertain-input registry is emitted, and a
manual `--tpot-margin-percent` is a FLOOR, never an error.

These are smoke tests: they check the wiring, the exclusions and the
provenance, not the margin arithmetic (that is tests/test_margin_policy.py).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from planner.__main__ import build_parser
from planner.envelope import workload_bucket
from planner.plan import RejectionStage
from planner.predictor.calibration import (
    AccuracyDomain,
    CalibrationModel,
    HardwareCalibration,
    save_calibration,
)
from planner.spec import load_service_spec

ROOT = Path(__file__).resolve().parents[1]
SERVICE = ROOT / "examples/service_specs/llama31-8b.yaml"
CLUSTER = ROOT / "experiments/configs/clusters/a40x8.yaml"
OUTSIDE = RejectionStage.OUTSIDE_CALIBRATION_DOMAIN.value
MISMATCH = RejectionStage.CALIBRATION_CONDITION_MISMATCH.value


@pytest.fixture
def canonical() -> str:
    return workload_bucket(load_service_spec(SERVICE))


@pytest.fixture
def domain_file(tmp_path: Path) -> Path:
    """An A40 domain covering [20, 120], wide enough to admit a mock candidate."""
    model = CalibrationModel(hardware={"A40": HardwareCalibration(
        hardware="A40",
        accuracy_domain=AccuracyDomain(
            fitted_at_concurrency=20.0,
            points=[{"conc": 20.0, "tpot_err_pct": -2.0, "ttft_err_pct": -2.0},
                    {"conc": 120.0, "tpot_err_pct": -20.0, "ttft_err_pct": -20.0}],
        ),
    )})
    path = tmp_path / "a40.domain.yaml"
    save_calibration(model, path)
    return path


def _run(argv: list[str], monkeypatch, served: float | None) -> tuple[int, object]:
    """Run `plan` with the mock predictor standing in for the simulator."""
    from planner.optimizer import exhaustive

    from .conftest import MockPredictor

    captured: dict[str, object] = {}
    real_search = exhaustive.search

    def fake_search(spec, cluster, islands, profiles, predictor, **kwargs):
        out = real_search(
            spec, cluster, islands, profiles,
            MockPredictor(served_concurrency=served), **kwargs
        )
        captured["output"] = out
        return out

    monkeypatch.setattr(exhaustive, "search", fake_search)
    monkeypatch.setattr("planner.__main__.exhaustive.search", fake_search)

    parser = build_parser()
    args = parser.parse_args(argv)
    code = args.func(args)
    return code, captured.get("output")


BASE = [
    "plan", "--service", str(SERVICE), "--cluster", str(CLUSTER),
    "--root", str(ROOT), "--num-requests", "8", "--quiet",
]


# --- exclusions -----------------------------------------------------------

def test_calibration_bucket_requires_accuracy_domain(capsys) -> None:
    parser = build_parser()
    args = parser.parse_args([*BASE, "--calibration-bucket", "in_lt1024-out_ge512-rps_lt20"])
    assert args.func(args) == 1
    assert "only means something with --accuracy-domain" in capsys.readouterr().err


def test_a_non_canonical_override_is_refused(capsys, domain_file) -> None:
    """§2.4.1: a human label is never accepted as a lookup key."""
    parser = build_parser()
    args = parser.parse_args(
        [*BASE, "--accuracy-domain", str(domain_file),
         "--calibration-bucket", "sharegpt-llama31-8b-20"]
    )
    assert args.func(args) == 1
    assert "not a canonical" in capsys.readouterr().err


def test_two_files_carrying_one_hardware_are_refused(capsys, domain_file, tmp_path) -> None:
    twin = tmp_path / "twin.yaml"
    twin.write_text(domain_file.read_text())
    parser = build_parser()
    args = parser.parse_args([*BASE, "--accuracy-domain", str(domain_file), str(twin)])
    assert args.func(args) == 1
    assert "two accuracy domains for A40" in capsys.readouterr().err


# --- the flag does its job ------------------------------------------------

def test_in_domain_candidates_are_margined_and_the_registry_is_emitted(
    monkeypatch, tmp_path, domain_file, canonical
) -> None:
    out_path = tmp_path / "plan.yaml"
    code, output = _run(
        [*BASE, "--accuracy-domain", str(domain_file), "--output", str(out_path)],
        monkeypatch, served=60.0,
    )
    assert output is not None
    assert output.uncertain_inputs is not None
    assert output.rejected_summary.get(OUTSIDE, 0) == 0

    uncertainty = output.provenance["uncertainty"]
    assert uncertainty["policy"] == "accuracy_domain"
    assert uncertainty["bucket"] == canonical
    assert uncertainty["domains"]["A40"]["conc_min"] == 20.0
    assert uncertainty["domains"]["A40"]["conc_max"] == 120.0
    assert uncertainty["domains"]["A40"]["outside_domain"] == "refuse"
    assert uncertainty["grades_digest"] and uncertainty["costs_digest"]
    assert "bucket_override" not in uncertainty
    assert output.provenance["accuracy_domain"] == ["A40"]

    plan = output.recommended.plan if output.recommended else None
    if plan is not None:
        # Linear between (20, -2) and (120, -20): weight (60-20)/100 = 0.4,
        # so the ERROR is -2 - 0.4 * 18 = -9.2 %. The margin is -e/(1+e) of that,
        # 9.2/90.8 = 10.13 % -- not 9.2 %, because the error's denominator is the
        # measurement and the margin multiplies a prediction (D70).
        assert plan.robust_margin_tpot_percent == pytest.approx(10.1322, abs=1e-4)
        assert plan.margin_source == "accuracy_domain"
        assert plan.margin_basis and "in domain" in plan.margin_basis
        assert plan.operating_point[0].hardware == "A40"

    # Opting in keeps the fields in the YAML.
    dumped = yaml.safe_load(out_path.read_text())
    assert "uncertain_inputs" in dumped
    if plan is not None:
        assert dumped["recommended"]["plan"]["predicted"]["served_concurrency"] == 60.0
        assert dumped["recommended"]["plan"]["margin_basis"]
    assert code in (0, 3)


def test_a_manual_margin_is_a_floor_not_an_error(monkeypatch, domain_file) -> None:
    """rps STEP 4.3: the LARGER of the two wins, and provenance records both."""
    code, output = _run(
        [*BASE, "--accuracy-domain", str(domain_file), "--tpot-margin-percent", "18"],
        monkeypatch, served=60.0,
    )
    assert output is not None
    assert output.provenance["manual_margin_percent"] == {"ttft": 0.0, "tpot": 18.0}
    for scored in [output.recommended, *output.alternatives]:
        if scored is not None:
            assert scored.plan.robust_margin_tpot_percent == 18.0
            assert scored.plan.margin_source == "manual"
    assert code in (0, 3)


def test_an_operating_point_outside_the_domain_is_rejected_as_unmeasured(
    monkeypatch, domain_file
) -> None:
    code, output = _run(
        [*BASE, "--accuracy-domain", str(domain_file)], monkeypatch, served=500.0
    )
    assert output is not None
    assert output.rejected_summary.get(OUTSIDE, 0) > 0
    assert output.rejected_summary.get(RejectionStage.SLO_VIOLATED.value, 0) == 0
    assert code == 3


def test_a_predictor_with_no_operating_point_is_unmeasured(monkeypatch, domain_file) -> None:
    """The honest default: no per-request records, so no operating point."""
    code, output = _run([*BASE, "--accuracy-domain", str(domain_file)], monkeypatch, served=None)
    assert output is not None
    assert output.rejected_summary.get(OUTSIDE, 0) > 0
    assert code == 3


def test_an_override_is_recorded_and_warned_about(monkeypatch, tmp_path, canonical) -> None:
    """A scoped domain plus a bucket override for a different token mix: the
    override is recorded, rendered, and genuinely changes the lookup."""
    from planner.render import render

    model = CalibrationModel(hardware={"A40": HardwareCalibration(
        hardware="A40",
        accuracy_domain=AccuracyDomain(
            fitted_at_concurrency=20.0, workload_shape="in_lt1024-out_ge512",
            points=[{"conc": 20.0, "tpot_err_pct": -2.0}, {"conc": 120.0, "tpot_err_pct": -20.0}],
        ),
    )})
    scoped = tmp_path / "scoped.yaml"
    save_calibration(model, scoped)

    other = "in_ge4096-out_lt128-rps_ge20"
    assert other != canonical
    code, output = _run(
        [*BASE, "--accuracy-domain", str(scoped), "--calibration-bucket", other],
        monkeypatch, served=60.0,
    )
    assert output is not None
    override = output.provenance["uncertainty"]["bucket_override"]
    assert override == {"requested": canonical, "used": other}
    assert output.provenance["uncertainty"]["shape"] == "in_ge4096-out_lt128"
    text = render(output)
    assert "BUCKET OVERRIDE" in text
    assert "DIFFERENT workload" in text
    assert output.rejected_summary.get(MISMATCH, 0) > 0
    assert code == 3


def test_the_unmeasured_rows_are_annotated_in_the_render(monkeypatch, domain_file) -> None:
    from planner.render import render

    _, output = _run([*BASE, "--accuracy-domain", str(domain_file)], monkeypatch, served=500.0)
    text = render(output)
    assert "outside_calibration_domain" in text
    assert "were not judged at all" in text
    assert "Uncertain inputs" in text


# --- rule A4 --------------------------------------------------------------

def test_without_the_flag_nothing_changes(monkeypatch, tmp_path) -> None:
    out_path = tmp_path / "plan.yaml"
    code, output = _run([*BASE, "--output", str(out_path)], monkeypatch, served=60.0)
    assert output is not None
    assert output.uncertain_inputs is None
    assert "uncertainty" not in output.provenance
    assert output.provenance["accuracy_domain"] is None
    assert OUTSIDE not in output.rejected_summary

    text = out_path.read_text()
    assert "uncertain_inputs" not in text
    assert "served_concurrency" not in text
    assert "margin_basis" not in text
    assert code in (0, 3)


def test_the_bare_flag_reads_every_committed_domain(monkeypatch) -> None:
    """The rps STEP 4 entry point: no files means profiles/calibration/*.yaml.

    The committed domains opt into `widen_error_bars` explicitly, so a mock A40
    candidate at concurrency 60 sits inside the A40 domain and is margined.
    """
    _, output = _run([*BASE, "--accuracy-domain"], monkeypatch, served=60.0)
    assert output is not None
    domains = output.provenance["uncertainty"]["domains"]
    assert set(domains) >= {"A40", "RNGD-CARD", "RNGD"}
    assert domains["RNGD-CARD"]["outside_domain"] == "widen_error_bars"
    assert output.provenance["accuracy_domain"] == sorted(domains)
    assert output.recommended is not None
    assert output.recommended.plan.margin_source == "accuracy_domain"


def test_the_committed_scalar_calibrations_are_placeable_or_labelled(monkeypatch) -> None:
    """The migrated a40.yaml carries a canonical bucket; the two RNGD fits do not.

    Both facts matter: A40 resolves to a canonical bucket and would be applied
    as a scalar fallback, while the RNGD fits - burst runs against a
    closed-loop client, so with no arrival rate to bucket by - stay invisible
    instead of being applied to a workload they were not measured on (§2.4.1
    migration). Their DOMAINS, measured separately, are what the flag reads.
    """
    from planner.predictor.calibration import load_calibration

    a40 = load_calibration(ROOT / "profiles/calibration/a40.yaml")
    assert a40.hardware["A40"].bucket_for(
        workload_bucket(load_service_spec(SERVICE))
    ) is not None
    for name, hardware in (("rngd.yaml", "RNGD"), ("rngd_card_edf.yaml", "RNGD-CARD")):
        model = load_calibration(ROOT / "profiles/calibration" / name)
        cal = model.hardware[hardware]
        assert cal.errors, "the fit is still there"
        assert all(not e.workload_bucket for e in cal.errors.values())
        assert all(e.label for e in cal.errors.values())
        assert "no canonical bucket" in " ".join(cal.available_buckets())


# --- --measurement-plan (STEP B3) -----------------------------------------

def test_measurement_plan_requires_the_accuracy_domain(capsys) -> None:
    parser = build_parser()
    args = parser.parse_args([*BASE, "--measurement-plan"])
    assert args.func(args) == 1
    assert "needs --accuracy-domain" in capsys.readouterr().err


def test_resimulate_top_needs_a_ranking_to_refine(capsys) -> None:
    """It refines --measurement-plan's order; alone it has nothing to work on."""
    parser = build_parser()
    args = parser.parse_args([*BASE, "--accuracy-domain", "x", "--resimulate-top", "3"])
    assert args.func(args) == 1
    assert "no ranking to refine" in capsys.readouterr().err


def test_resimulate_top_is_off_by_default(monkeypatch, tmp_path, domain_file) -> None:
    """Absolute rule A4: an opt-in feature must not move the default path."""
    parser = build_parser()
    args = parser.parse_args([*BASE, "--accuracy-domain", str(domain_file),
                              "--measurement-plan"])
    assert args.resimulate_top == 0


def test_a_measurement_plan_is_emitted_and_rendered(monkeypatch, tmp_path,
                                                    domain_file) -> None:
    from planner.render import render

    out_path = tmp_path / "plan.yaml"
    _code, output = _run(
        [*BASE, "--accuracy-domain", str(domain_file), "--measurement-plan",
         "--output", str(out_path)],
        monkeypatch, served=60.0,
    )
    assert output is not None
    plan = output.measurement_plan
    assert plan is not None

    # Every registry entry is accounted for: ranked, deferred, or undecidable.
    seen = (
        {i.input_id for i in plan.items}
        | {i.input_id for i in plan.uncovered}
        | set(plan.undecidable)
    )
    assert seen == {i.id for i in output.uncertain_inputs.items}

    # The knobs that decide the ranking travel with the result (§2.5).
    assert output.provenance["uncertainty"]["slo_penalty"] > 0
    assert output.provenance["uncertainty"]["grid_weighting"] == "uniform"

    text = render(output)
    assert "Measurement plan" in text
    # a40x8 declares measured links and the fixture calibration carries no
    # scalar bucket, so the registry can legitimately be empty here; the section
    # then says so instead of printing an empty table.
    assert "decision regret" in text or "nothing to measure" in text
    assert "measurement_plan" in yaml.safe_load(out_path.read_text())


def test_a_measurement_plan_ranks_the_placeholder_links_of_a_mixed_cluster(
    monkeypatch, domain_file
) -> None:
    """On `pd-rngd-gpu.yaml` (14 placeholder links, D33 registry) every uncertain
    input is either ranked, deferred or listed as undecidable, and the render
    states the regret the budget removes."""
    from planner.render import render

    argv = [
        "plan", "--service", str(SERVICE),
        "--cluster", str(ROOT / "experiments/configs/clusters/pd-rngd-gpu.yaml"),
        "--root", str(ROOT), "--num-requests", "8", "--quiet",
        "--accuracy-domain", str(domain_file), "--measurement-plan",
    ]
    _code, output = _run(argv, monkeypatch, served=60.0)
    assert output is not None and output.uncertain_inputs is not None
    assert output.uncertain_inputs.items, "placeholder links make this registry non-empty"
    plan = output.measurement_plan
    assert plan is not None
    seen = (
        {i.input_id for i in plan.items}
        | {i.input_id for i in plan.uncovered}
        | set(plan.undecidable)
        | set(plan.inert)
    )
    assert seen == {i.id for i in output.uncertain_inputs.items}
    text = render(output)
    assert "decision regret" in text or "moves the recommendation" in text


def test_a_budget_defers_what_does_not_fit(monkeypatch, domain_file) -> None:
    _code, output = _run(
        [*BASE, "--accuracy-domain", str(domain_file), "--measurement-plan",
         "--budget-hours", "0"],
        monkeypatch, served=60.0,
    )
    plan = output.measurement_plan
    assert plan.items == []
    assert plan.covered_regret == 0.0


def test_without_the_flag_there_is_no_plan_in_the_yaml(monkeypatch, tmp_path,
                                                      domain_file) -> None:
    out_path = tmp_path / "plan.yaml"
    _code, output = _run(
        [*BASE, "--accuracy-domain", str(domain_file), "--output", str(out_path)],
        monkeypatch, served=60.0,
    )
    assert output.measurement_plan is None
    # Against the PARSED document, not the raw text: provenance records the
    # command line, so `pytest tests/test_measurement_plan.py` put the string
    # "measurement_plan" in the file and failed this on a substring that had
    # nothing to do with the plan.
    import yaml

    assert "measurement_plan" not in yaml.safe_load(out_path.read_text())


def test_a_domain_scoped_to_another_model_is_refused(monkeypatch, tmp_path) -> None:
    """§2.4.1 rev 2 as D33 carries it: the verification conditions are model,
    precision and token mix, held as scope fields on the domain."""
    model = CalibrationModel(hardware={"A40": HardwareCalibration(
        hardware="A40",
        accuracy_domain=AccuracyDomain(
            fitted_at_concurrency=20.0, model="Qwen/Qwen3-32B",
            points=[{"conc": 20.0, "tpot_err_pct": -2.0}, {"conc": 120.0, "tpot_err_pct": -20.0}],
        ),
    )})
    scoped = tmp_path / "scoped.yaml"
    save_calibration(model, scoped)
    code, output = _run([*BASE, "--accuracy-domain", str(scoped)], monkeypatch, served=60.0)
    assert output is not None
    # `calibration_condition_mismatch`, not `outside_calibration_domain`: the
    # model is an APPLICATION CONDITION, so the domain may not be consulted at
    # all, which is a different gap from an operating point past its load axis
    # (domain-scoping S1, D110).
    assert output.rejected_summary.get(MISMATCH, 0) > 0
    assert output.rejected_summary.get(OUTSIDE, 0) == 0
    assert code == 3
