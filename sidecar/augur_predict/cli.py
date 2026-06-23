"""Command-line interface for the sidecar.

Four subcommands, each a thin shell over the library so the logic stays testable
without argv:

    fit          trace.jsonl            -> model.json
    report       model.json            -> human-readable fit quality
    predict      model.json + inputs   -> predicted output length (+ optional cost)
    emit-trace   model.json            -> predicted trace.jsonl (for the Go gate)

The contract with Augur's Go core is files only: this never calls the Go binary
and the Go binary never calls this. ``emit-trace`` writes a trace; everything
else reads one.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from . import __version__
from .emit import emit
from .model import Model, fit
from .trace import load_trace, write_trace


def _cmd_fit(args: argparse.Namespace) -> int:
    records = load_trace(args.trace)
    if not records:
        print(f"fit: trace {args.trace!r} is empty — nothing to learn", file=sys.stderr)
        return 1
    model = fit(records)
    model.source = args.trace
    model.save(args.out)
    print(f"fit: learned {len(model.fits)} model(s) from {model.n_records} "
          f"call(s) across {len(model.templates)} run(s) -> {args.out}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    model = Model.load(args.model)
    print(_render_report(model))
    return 0


def _render_report(model: Model) -> str:
    lines: List[str] = []
    lines.append(f"Output-length model  (source: {model.source or 'unknown'})")
    lines.append(f"  {model.n_records} calls, {len(model.templates)} run templates, "
                 f"{len(model.fits)} model(s)")
    lines.append("")
    header = f"  {'model':<28} {'n':>5} {'method':>7} {'out~in':>16} {'R2':>6} {'±resid':>8}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for name in sorted(model.fits):
        f = model.fits[name]
        if f.method == "ols":
            rel = f"{f.intercept:.0f}+{f.slope:.3f}·in"
            r2 = f"{f.r2:.2f}"
        else:
            rel = f"~{f.output_mean:.0f}"
            r2 = "  — "
        lines.append(f"  {name:<28} {f.n:>5} {f.method:>7} {rel:>16} {r2:>6} {f.resid_std:>8.0f}")
    lines.append("")
    # An honest read on whether the fit is worth trusting — the same spirit as
    # the Go side surfacing CIs instead of bare point estimates.
    weak = [name for name, f in model.fits.items()
            if f.method == "mean" or f.r2 < 0.3]
    if weak:
        lines.append("  note: weak/absent input→output signal for: "
                     + ", ".join(sorted(weak)))
        lines.append("        predictions fall back to the observed mean output; "
                     "treat them as rough.")
    return "\n".join(lines)


def _cmd_predict(args: argparse.Namespace) -> int:
    model = Model.load(args.model)
    f = model.fit_for(args.model_name)
    if f is None:
        avail = ", ".join(sorted(model.fits)) or "(none)"
        print(f"predict: no fit for model {args.model_name!r}; trace covered: {avail}",
              file=sys.stderr)
        return 1

    mid = f.predict(args.input_tokens)
    lo, hi = f.band(args.input_tokens)
    print(f"model={args.model_name} input_tokens={args.input_tokens}")
    print(f"  predicted output tokens: {mid:.0f}  (~95% band {lo:.0f}–{hi:.0f}, method={f.method})")

    if args.price_out is not None:
        # Optional dollar estimate. We price only the completion side here: the
        # sidecar's job is the unknown (output length); prompt cost is already
        # known exactly from the input you supplied. Full per-call pricing lives
        # in the Go `aggregate` stage, which emit-trace feeds.
        cost_mid = mid / 1_000_000 * args.price_out
        cost_hi = hi / 1_000_000 * args.price_out
        print(f"  est. output cost @ ${args.price_out}/Mtok: "
              f"${cost_mid:.6f}  (p~95 ${cost_hi:.6f})")
    return 0


def _cmd_emit_trace(args: argparse.Namespace) -> int:
    model = Model.load(args.model)
    try:
        records = emit(
            model,
            runs=args.runs,
            input_scale=args.input_scale,
            seed=args.seed,
            scenario_filter=args.scenario,
            run_prefix=args.run_prefix,
        )
    except ValueError as e:
        print(f"emit-trace: {e}", file=sys.stderr)
        return 1

    n = write_trace(args.out, records)
    print(f"emit-trace: wrote {n} predicted call(s) over {args.runs} run(s) "
          f"(input_scale={args.input_scale}, seed={args.seed}) -> {args.out}")
    print(f"  feed it to the Go gate, e.g.:  augur aggregate --trace {args.out} | ...")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="augur-predict",
        description="Augur's predictive output-length sidecar: learn completion "
                    "length from a recorded trace, then estimate cost for inputs "
                    "you have not run.",
    )
    p.add_argument("--version", action="version", version=f"augur-predict {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    pf = sub.add_parser("fit", help="learn an output-length model from a trace")
    pf.add_argument("--trace", required=True, help="recorded JSONL trace to learn from")
    pf.add_argument("--out", default="model.json", help="model artifact to write")
    pf.set_defaults(func=_cmd_fit)

    pr = sub.add_parser("report", help="print fit quality for a model artifact")
    pr.add_argument("--model", default="model.json", help="model artifact to read")
    pr.set_defaults(func=_cmd_report)

    pp = sub.add_parser("predict", help="predict output length for one input size")
    pp.add_argument("--model", default="model.json", help="model artifact to read")
    pp.add_argument("--model-name", required=True, dest="model_name",
                    help="billed model to predict for (must appear in the trace)")
    pp.add_argument("--input-tokens", required=True, type=float, dest="input_tokens",
                    help="prompt size to predict the completion length for")
    pp.add_argument("--price-out", type=float, default=None, dest="price_out",
                    help="optional $/Mtok for output, to print an output-cost estimate")
    pp.set_defaults(func=_cmd_predict)

    pe = sub.add_parser("emit-trace",
                        help="synthesise a predicted trace for the Go gate")
    pe.add_argument("--model", default="model.json", help="model artifact to read")
    pe.add_argument("--out", default="predicted-trace.jsonl", help="trace file to write")
    pe.add_argument("--runs", type=int, default=20,
                    help="number of synthetic runs to generate")
    pe.add_argument("--input-scale", type=float, default=1.0, dest="input_scale",
                    help="multiply every prompt size (and re-predict output); "
                         "the predictive analogue of --context-growth")
    pe.add_argument("--scenario", default=None,
                    help="restrict to one scenario id from the templates")
    pe.add_argument("--seed", type=int, default=0,
                    help="RNG seed; same seed reproduces the same trace")
    pe.add_argument("--run-prefix", default="pred", dest="run_prefix",
                    help="prefix for synthetic run ids")
    pe.set_defaults(func=_cmd_emit_trace)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
