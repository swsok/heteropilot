"""What each committed accuracy domain states, and what it deliberately does not.

Domain-scoping S4; deviations D113. S1 (D110) added the application-condition
fields and left `arrival_process` and `model`/`variant` unstated; S4 owns them.
This pins the decision that was made and, for each field, the reason - because
"not stated" and "nobody got round to it" look identical in a YAML file and
only one of them is a finding.

The arrival process is the field that changes verdicts: a domain fitted against
a closed-loop burst does not answer for the arrival-trace replay `plan`
performs (D19), and stating it moves E-A1's 36 RNGD-touching candidates from
`outside_calibration_domain` to `calibration_condition_mismatch`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from planner.predictor.calibration import load_accuracy_domains

ROOT = Path(__file__).resolve().parents[1]

#: hardware -> the arrival process its REAL side was measured under, and where
#: that is written down independently of the field this test checks.
EXPECTED_ARRIVAL = {
    "A40": ("open_loop", "a40.accuracy.yaml's header: 'OPEN-LOOP, deliberately'"),
    "RNGD-CARD": ("closed_loop",
                  "rngd_card_edf.yaml's why_unresolved: a burst against a "
                  "closed-loop client; bench_furiosa_endpoint.py ignores "
                  "arrival_time_ns (D19)"),
    "RNGD": ("closed_loop",
             "rngd_perpe.yaml's measured half is the RNGD-CARD envelope, whose "
             "file states closed_loop: true"),
}

#: hardware -> the dtype variant, where a source states it. RNGD-CARD is absent
#: on purpose: its one `fitted_from` artifact records no dtype at all.
EXPECTED_VARIANT = {"A40": "bf16", "RNGD": "bf16"}


@pytest.fixture(scope="module")
def domains():
    return load_accuracy_domains(ROOT)


def test_every_domain_states_its_arrival_process(domains):
    """`unknown` here is not a safe default - it is a condition that will not be
    compared, on the axis D19 says is the one that differs."""
    for hardware, (expected, source) in EXPECTED_ARRIVAL.items():
        assert hardware in domains, f"no domain for {hardware}"
        got = domains[hardware].arrival_process
        assert got == expected, (
            f"{hardware}: arrival_process is {got!r}, expected {expected!r} "
            f"from {source}"
        )


def test_variant_is_stated_only_where_a_source_states_it(domains):
    for hardware, expected in EXPECTED_VARIANT.items():
        assert domains[hardware].variant == expected

    # RNGD-CARD's `fitted_from` is one summary table that records no model and
    # no dtype, so there is nothing to state and a guess would be the thing
    # absolute rule 3 forbids.
    assert domains["RNGD-CARD"].variant == "", (
        "RNGD-CARD's dtype is not recorded anywhere in its provenance; stating "
        "one would be a guess, and a guessed condition silently accepts or "
        "refuses candidates"
    )


def test_no_domain_states_a_model(domains):
    """The one field S4 decided NOT to fill, and why.

    This repository holds two strings for one set of weights -
    `meta-llama/Llama-3.1-8B` in the envelope paths and
    `NousResearch/Meta-Llama-3.1-8B` in the open-loop A40 domain's own
    `provenance.deployment`. `check_conditions` compares strings, so stating
    either would refuse a run on the MIRROR NAME rather than on a model
    difference. Remove this test only together with an alias policy.
    """
    for hardware, domain in domains.items():
        assert domain.model == "", (
            f"{hardware} states model={domain.model!r}. Two strings for one set "
            f"of weights exist here, so a raw string comparison has a "
            f"false-refusal mode; see D113"
        )


def test_the_open_loop_a40_domain_is_still_the_open_loop_one():
    """D102 keeps a second A40 domain in a subdirectory. It has always stated
    `open_loop`, and it is the precedent S4 followed - so if it ever stops
    saying so, the default-path domain's `open_loop` loses its counterpart."""
    openloop = load_accuracy_domains(
        ROOT, [ROOT / "profiles/calibration/openloop/a40.accuracy.openloop.yaml"]
    )
    assert openloop["A40"].arrival_process == "open_loop"


def test_a_closed_loop_domain_refuses_an_open_loop_candidate(domains):
    """The behaviour the field exists for, on the two domains that carry it."""
    from planner.predictor.calibration import CandidateConditions

    for hardware in ("RNGD-CARD", "RNGD"):
        domain = domains[hardware]
        check = domain.check_conditions(CandidateConditions(
            hardware=hardware, arrival_process="open_loop",
            tp=domain.parallelism.tp if domain.parallelism else 1,
            pp=1, dp=1, islands=1,
        ))
        assert "arrival_process" in check.mismatch, (
            f"{hardware} is fitted closed-loop and must refuse an open-loop "
            f"candidate; mismatch={check.mismatch} skipped={check.skipped}"
        )

    # And the A40 domain, fitted open-loop, must NOT refuse one. A field that
    # refused everything would prove nothing.
    a40 = domains["A40"]
    check = a40.check_conditions(CandidateConditions(
        hardware="A40", arrival_process="open_loop", tp=1, pp=1, dp=1, islands=1,
    ))
    assert "arrival_process" not in check.mismatch
