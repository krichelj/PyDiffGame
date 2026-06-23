"""Orchestrate a head-to-head comparison: metrics table, robustness, GIF, JSON."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.harness import (
    Controller,
    ControlProblem,
    comparison_panels_gif,
    metrics,
    monte_carlo_disturbance_cost,
    simulate,
    worst_case_impulse_cost,
)

RESULTS = Path(__file__).resolve().parent / "results"


def _fmt(v: float) -> str:
    if v == float("inf"):
        return "  inf"
    if abs(v) >= 1e4 or (v != 0 and abs(v) < 1e-3):
        return f"{v:.3e}"
    return f"{v:.3f}"


def compare(
    problem: ControlProblem,
    controllers: list[Controller],
    *,
    out_prefix: str,
    render_gif: bool = True,
    robustness: bool = True,
) -> dict:
    """Run all controllers on ``problem``, emit a GIF + metrics JSON, return the report."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    rollouts = [simulate(problem, c) for c in controllers]
    report: dict = {"problem": problem.name, "controllers": {}}

    nominal = {c.name: metrics(problem, r) for c, r in zip(controllers, rollouts)}
    for c in controllers:
        report["controllers"][c.name] = {"meta": c.meta, "nominal": nominal[c.name]}

    if robustness:
        for c in controllers:
            wc = worst_case_impulse_cost(problem, c, magnitude=1.0)
            mc = monte_carlo_disturbance_cost(problem, c, trials=48, sigma=0.5)
            report["controllers"][c.name]["robustness"] = {"worst_case_impulse_cost": wc, "monte_carlo": mc}

    # ---- console report ----
    metric_keys = list(next(iter(nominal.values())).keys())
    width = max(len(c.name) for c in controllers)
    print(f"\n=== {problem.name} ===")
    header = "metric".ljust(20) + "".join(c.name.rjust(width + 3) for c in controllers)
    print(header)
    print("-" * len(header))
    for key in metric_keys:
        row = key.ljust(20) + "".join(_fmt(nominal[c.name][key]).rjust(width + 3) for c in controllers)
        print(row)
    if robustness:
        print("-" * len(header))
        for label, path in [
            ("worst-case dist cost", ("robustness", "worst_case_impulse_cost")),
            ("MC mean cost", ("robustness", "monte_carlo", "mean")),
            ("MC p95 cost", ("robustness", "monte_carlo", "p95")),
        ]:
            vals = []
            for c in controllers:
                node = report["controllers"][c.name]
                for p in path:
                    node = node[p]
                vals.append(node)
            print(label.ljust(20) + "".join(_fmt(v).rjust(width + 3) for v in vals))

    # ---- honest verdict on the shared cost ----
    costs = {c.name: nominal[c.name]["cost"] for c in controllers}
    best = min(costs, key=costs.get)
    spread = (max(costs.values()) - min(costs.values())) / (abs(min(costs.values())) + 1e-12)
    report["verdict"] = {"best_nominal_cost": best, "relative_spread": spread}
    if spread < 1e-4:
        print(f"\nVERDICT: nominal costs match to {spread:.1e} (decomposition is lossless here).")
    else:
        print(f"\nVERDICT: lowest nominal cost = {best} (relative spread {spread:.3%}).")

    if render_gif:
        gif = comparison_panels_gif(problem, rollouts, str(RESULTS / f"{out_prefix}.gif"))
        report["gif"] = gif
        print(f"GIF: {gif}")

    (RESULTS / f"{out_prefix}.json").write_text(json.dumps(report, indent=2, default=float))
    return report
