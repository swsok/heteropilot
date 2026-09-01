"""ScenarioLab CLI (DESIGN §6.4).

P1 ships `generate` and `run`. `verify` arrives with the P2 tier work,
`serve` with P3 and `export` with P5; they are declared so --help shows the
intended shape, and they exit loudly instead of pretending.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scenariolab.config import LabConfigError, load_lab_config
from scenariolab.generator.cluster_gen import ClusterGenError
from scenariolab.generator.slo_gen import SloGenError
from scenariolab.runner.batch import BatchRunner
from scenariolab.store.db import ResultStore


def cmd_generate(args: argparse.Namespace) -> int:
    config, digest = load_lab_config(args.config, args.root)
    runner = BatchRunner(config, digest, Path(args.config).read_text(), args.root)
    clusters, services = runner.generate()
    print(f"generated {len(clusters)} clusters -> {config.store.clusters_dir}")
    for c in clusters:
        print(
            f"  {c.cluster_id}: nodes={c.num_nodes} accels={c.num_accels} "
            f"free={c.num_free_accels} islands={c.num_islands} "
            f"classes={','.join(c.classes)}{' [NPU]' if c.has_npu else ''}"
        )
    print(f"generated {len(services)} service specs -> {config.store.services_dir}")
    for s in services:
        print(
            f"  {s.service_id}: {s.model} rps={s.rps:.2f} "
            f"ttft_p99={s.ttft_p99_ms:.0f}ms tpot_p99={s.tpot_p99_ms:.0f}ms "
            f"cap={s.power_cap_w:.0f}W"
        )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    config, digest = load_lab_config(args.config, args.root)
    if args.workers is not None:
        config.runner.workers = args.workers
    runner = BatchRunner(config, digest, Path(args.config).read_text(), args.root)
    with ResultStore(Path(args.root) / config.store.db_path) as store:
        runner.run(store, quiet=args.quiet)
    # Scenario errors are isolated, recorded in the DB and printed in the
    # summary; the batch itself completed, so the exit code stays 0 (FR-B5).
    return 0


def _not_yet(phase: str):
    def handler(args: argparse.Namespace) -> int:
        print(f"'{args.command}' is scheduled for {phase} and is not implemented yet.",
              file=sys.stderr)
        return 2
    return handler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scenariolab",
        description="ScenarioLab: random-scenario power-optimal placement experiments.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Generate random clusters and SLOs (M1+M2).")
    gen.add_argument("--config", required=True, type=Path)
    gen.add_argument("--root", type=Path, default=Path("."))
    gen.set_defaults(func=cmd_generate)

    run = sub.add_parser("run", help="Run (or resume) the full scenario batch.")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--root", type=Path, default=Path("."))
    run.add_argument("--workers", type=int, default=None,
                     help="Override runner.workers from the config.")
    run.add_argument("--quiet", action="store_true")
    run.set_defaults(func=cmd_run)

    verify = sub.add_parser("verify", help="(P2) Cross-check sampled scenarios with full sim.")
    verify.add_argument("--batch", required=False)
    verify.add_argument("--fraction", type=float, default=None)
    verify.set_defaults(func=_not_yet("P2 (feat/scenariolab-tiers)"))

    serve = sub.add_parser("serve", help="(P3) Serve the web UI over the result store.")
    serve.add_argument("--db", type=Path, default=None)
    serve.add_argument("--port", type=int, default=8080)
    serve.set_defaults(func=_not_yet("P3 (feat/scenariolab-web)"))

    export = sub.add_parser("export", help="(P5) Static report export.")
    export.add_argument("--batch", required=False)
    export.add_argument("--out", type=Path, default=None)
    export.set_defaults(func=_not_yet("P5"))

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (LabConfigError, ClusterGenError, SloGenError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
