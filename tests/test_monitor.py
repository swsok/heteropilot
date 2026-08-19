"""Monitor tests: Prometheus parsing and energy integration.

Pure functions only - no socket, no nvidia-smi, no GPU (work order §9).
"""

from __future__ import annotations

import math

from planner.monitor import (
    PowerSeries,
    histogram_quantile,
    integrate_energy,
    parse_prometheus,
    parse_vllm_metrics,
)

_METRICS_TEXT = """
# HELP vllm:time_to_first_token_seconds TTFT histogram
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{le="0.1"} 0
vllm:time_to_first_token_seconds_bucket{le="0.5"} 2
vllm:time_to_first_token_seconds_bucket{le="1.0"} 8
vllm:time_to_first_token_seconds_bucket{le="+Inf"} 10
vllm:time_to_first_token_seconds_sum 5.0
vllm:time_to_first_token_seconds_count 10
# TYPE vllm:time_per_output_token_seconds histogram
vllm:time_per_output_token_seconds_bucket{le="0.01"} 4
vllm:time_per_output_token_seconds_bucket{le="0.05"} 10
vllm:time_per_output_token_seconds_bucket{le="+Inf"} 10
vllm:time_per_output_token_seconds_sum 0.3
vllm:time_per_output_token_seconds_count 10
vllm:generation_tokens_total 1234
vllm:prompt_tokens_total 5678
vllm:request_success_total{finished_reason="stop"} 8
vllm:request_success_total{finished_reason="length"} 2
"""


def test_parse_prometheus_skips_comments_and_reads_labels() -> None:
    samples = parse_prometheus(_METRICS_TEXT)
    named = [s for s in samples if s.name == "vllm:request_success_total"]
    assert len(named) == 2
    assert named[0].labels["finished_reason"] == "stop"
    assert named[0].value == 8.0


def test_histogram_quantile_linear_interpolation() -> None:
    buckets = [(0.1, 0.0), (0.5, 2.0), (1.0, 8.0), (float("inf"), 10.0)]
    # rank = 5 of 10 lands in (0.5, 1.0]: 0.5 + 0.5 * (5-2)/6 = 0.75
    assert histogram_quantile(buckets, 0.50) == 0.75


def test_histogram_quantile_empty_is_nan() -> None:
    assert math.isnan(histogram_quantile([], 0.5))
    assert math.isnan(histogram_quantile([(1.0, 0.0)], 0.5))


def test_parse_vllm_metrics_percentiles_and_counters() -> None:
    scrape = parse_vllm_metrics(_METRICS_TEXT)
    assert scrape.p50_ttft_ms == 750.0  # 0.75 s -> ms
    assert scrape.completed_tokens == 1234
    assert scrape.prompt_tokens == 5678
    assert scrape.completed_requests == 10  # both success series summed
    assert scrape.p50_tpot_ms > 0


def test_integrate_energy_trapezoid_constant() -> None:
    # 100 W held for 2 s -> 200 J.
    assert integrate_energy([0.0, 1.0, 2.0], [100.0, 100.0, 100.0]) == 200.0


def test_integrate_energy_trapezoid_ramp() -> None:
    # linear ramp 0 -> 100 W over 2 s -> triangle area 100 J.
    assert integrate_energy([0.0, 2.0], [0.0, 100.0]) == 100.0


def test_integrate_energy_needs_two_samples() -> None:
    assert integrate_energy([], []) == 0.0
    assert integrate_energy([1.0], [50.0]) == 0.0


def test_power_series_aggregates() -> None:
    series = PowerSeries()
    series.add(100.0, at=0.0)
    series.add(100.0, at=1.0)
    series.add(200.0, at=2.0)
    assert series.average_power_w is not None and abs(series.average_power_w - 400.0 / 3) < 1e-9
    assert series.peak_power_w == 200.0
    assert series.window_seconds == 2.0
    # 0->1: (100+100)/2, 1->2: (100+200)/2 = 100 + 150 = 250 J
    assert series.total_energy_j == 250.0


def test_power_series_empty() -> None:
    series = PowerSeries()
    assert series.average_power_w is None
    assert series.peak_power_w is None
    assert series.total_energy_j == 0.0


def test_parse_prometheus_accepts_negative_exponent() -> None:
    samples = parse_prometheus("foo_metric 1.5e-08\nbar_metric{k=\"v\"} -2.0e+03\n")
    by_name = {s.name: s.value for s in samples}
    assert math.isclose(by_name["foo_metric"], 1.5e-08)
    assert math.isclose(by_name["bar_metric"], -2000.0)


def test_histogram_buckets_summed_across_series() -> None:
    # Same le split across two label series must collapse to one cumulative
    # curve: a single 0/2/8/10 series and two 0/1/4/5 series give equal p95.
    single = """
vllm:time_to_first_token_seconds_bucket{le="0.1"} 0
vllm:time_to_first_token_seconds_bucket{le="0.5"} 2
vllm:time_to_first_token_seconds_bucket{le="1.0"} 8
vllm:time_to_first_token_seconds_bucket{le="+Inf"} 10
"""
    split = """
vllm:time_to_first_token_seconds_bucket{le="0.1",m="a"} 0
vllm:time_to_first_token_seconds_bucket{le="0.5",m="a"} 1
vllm:time_to_first_token_seconds_bucket{le="1.0",m="a"} 4
vllm:time_to_first_token_seconds_bucket{le="+Inf",m="a"} 5
vllm:time_to_first_token_seconds_bucket{le="0.1",m="b"} 0
vllm:time_to_first_token_seconds_bucket{le="0.5",m="b"} 1
vllm:time_to_first_token_seconds_bucket{le="1.0",m="b"} 4
vllm:time_to_first_token_seconds_bucket{le="+Inf",m="b"} 5
"""
    assert parse_vllm_metrics(single).p95_ttft_ms == parse_vllm_metrics(split).p95_ttft_ms


def test_tpot_reads_inter_token_latency_metric() -> None:
    # vLLM >= 0.19 renamed per-output-token latency to inter_token_latency_seconds.
    # A live A40 run surfaced this as all-nan TPOT under the old name; the parser
    # must fall back to the new name. (Regression guard.)
    text = """
vllm:inter_token_latency_seconds_bucket{le="0.05"} 0
vllm:inter_token_latency_seconds_bucket{le="0.1"} 8
vllm:inter_token_latency_seconds_bucket{le="+Inf"} 10
vllm:inter_token_latency_seconds_count 10
"""
    scrape = parse_vllm_metrics(text)
    assert not math.isnan(scrape.p50_tpot_ms)
    assert scrape.p50_tpot_ms > 0.0
