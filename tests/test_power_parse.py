"""Golden-fixture tests for the stdout power parser (deviations D2).

The fixture is verbatim from a real run at the pinned commit. If upstream
changes the format these fail loudly here, which is the entire point of keeping
every regex in one module rather than scattering them through the predictor.
"""

from __future__ import annotations

import pytest

from planner.util.power_parse import PowerParseError, parse_power

# Captured from `python -m serving --cluster-config single_node_power_instance.json`
GOLDEN = """\
Total clocks (ns):                                                  1665077255
Total latency (s):                                                  1.665
Request throughput (req/s):                                         6.01
────────────────────────────── Power Modeling Results ──────────────────────────
Total energy consumption (kJ):                                      1.42
────────────────────────────────────────────────────────────────────────────────
Node 0 total energy consumption (kJ):                               1.42
├─ Base Node energy consumption (J):                                99.90
├─ NPU energy consumption (J):                                      972.14
├─ CPU energy consumption (J):                                      233.13
├─ Memory energy consumption (J):                                   53.28
├─ Link energy consumption (J):                                     8.33
├─ NIC energy consumption (J):                                      33.30
└─ Storage energy consumption (J):                                  16.65
────────────────────────────────────────────────────────────────────────────────
Power per 1.0 sec (W): [845.91]
"""

MULTI_SAMPLE = GOLDEN.replace(
    "Power per 1.0 sec (W): [845.91]", "Power per 1.0 sec (W): [800.0, 900.0, 850.0]"
)

NO_POWER = "Total clocks (ns): 123\nRequest throughput (req/s): 6.01\n"


def test_golden_total_energy() -> None:
    result = parse_power(GOLDEN)
    assert result.present
    # 1.42 kJ -> J
    assert result.summary.total_energy_j == pytest.approx(1420.0)


def test_golden_per_node_and_components() -> None:
    s = parse_power(GOLDEN).summary
    assert s.per_node_energy_j == {0: pytest.approx(1420.0)}
    assert s.per_component_j["NPU"] == pytest.approx(972.14)
    assert s.per_component_j["Storage"] == pytest.approx(16.65)
    # "Base Node" keeps its space rather than being split into two components.
    assert "Base Node" in s.per_component_j


def test_golden_power_series() -> None:
    s = parse_power(GOLDEN).summary
    assert s.power_series_w == [pytest.approx(845.91)]
    assert s.interval_s == pytest.approx(1.0)
    assert s.average_power_w == pytest.approx(845.91)
    assert s.peak_power_w == pytest.approx(845.91)


def test_peak_and_average_differ_on_a_real_series() -> None:
    s = parse_power(MULTI_SAMPLE).summary
    assert s.average_power_w == pytest.approx(850.0)
    assert s.peak_power_w == pytest.approx(900.0)


def test_tokens_per_joule() -> None:
    s = parse_power(GOLDEN).summary
    assert s.tokens_per_joule(1420) == pytest.approx(1.0)
    assert s.tokens_per_joule(0) == pytest.approx(0.0)


def test_absent_power_block_is_not_an_error() -> None:
    """The simulator emits no power unless the config has a `power:` block."""
    result = parse_power(NO_POWER)
    assert not result.present
    assert any("no power block" in w for w in result.warnings)


def test_series_without_total_is_a_hard_error() -> None:
    """Half a power block means the format moved; better to fail than guess."""
    broken = "Power per 1.0 sec (W): [800.0]\n"
    with pytest.raises(PowerParseError, match="format has changed"):
        parse_power(broken)


def test_provenance_marks_peak_as_an_interval_average() -> None:
    """Resolution equals --log-interval, so the peak is not instantaneous."""
    prov = parse_power(GOLDEN).summary.as_provenance()
    assert prov["peak_is_interval_average"] is True
    assert prov["power_sample_interval_s"] == pytest.approx(1.0)


def test_box_drawing_characters_are_not_load_bearing() -> None:
    """Anchor on labels, not on the tree glyphs, which upstream may restyle."""
    plain = GOLDEN.replace("├─", "  ").replace("└─", "  ")
    assert parse_power(plain).summary.per_component_j["NPU"] == pytest.approx(972.14)
