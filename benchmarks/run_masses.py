"""Benchmark: masses-on-springs — PyDiffGame modal Nash game vs centralized LQR."""

from __future__ import annotations

from benchmarks.pdg_design import game_controller, lqr_controller
from benchmarks.runner import compare
from benchmarks.systems.masses import build


def main() -> None:
    problem, game_objectives = build(N=2)
    controllers = [
        lqr_controller(problem),
        game_controller(problem, game_objectives),
    ]
    compare(problem, controllers, out_prefix="masses_pdg_vs_lqr")


if __name__ == "__main__":
    main()
