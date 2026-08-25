#!/usr/bin/env python
"""
scripts/plot_injections.py — efficiency bar charts for injection campaigns,
sorted most -> least efficient. Handles a single run dir, a batch parent
(combines all run_*/summary.pkl found recursively), or an explicit
summary.pkl / summary.txt. All plotting logic lives in plotting.injections.

Usage:
    python -m bilbyflow.scripts.plot_injections /path/to/run_..._seed1_abc
    python -m bilbyflow.scripts.plot_injections /path/to/reweighting_injections_...
    python -m bilbyflow.scripts.plot_injections /path/to/summary.txt --from-txt
    python -m bilbyflow.scripts.plot_injections <batch> --dl-max 2000
"""

import argparse
import os

from bilbyflow.plotting.injections import (resolve_and_load, plot_sorted_injections,
                                   plot_snr_panel)


def main():
    parser = argparse.ArgumentParser(
        description="Efficiency bar charts for injection reweighting "
                    "(single run or combined batch), sorted by efficiency")
    parser.add_argument("path",
                        help="a run dir, a batch parent dir (combines run_*/), "
                             "or a summary.pkl / summary.txt")
    parser.add_argument("--from-txt", action="store_true",
                        help="Force parsing the path as a summary.txt")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Where to write the PNGs (default: alongside the input)")
    parser.add_argument("--dl-max", type=float, default=None,
                        help="Only plot injections with dL <= this (Mpc). "
                             "Forces the event-pkl path (summary.pkl lacks dL).")
    parser.add_argument("--dl-min", type=float, default=None)
    args = parser.parse_args()

    cut_active = (args.dl_max is not None) or (args.dl_min is not None)
    rows, default_dir = resolve_and_load(args.path, from_txt=args.from_txt,
                                         prefer_events=cut_active)
    print(f"Loaded {len(rows)} injections total")
    if not rows:
        raise SystemExit("No injections found.")

    out_dir = args.outdir or default_dir
    os.makedirs(out_dir, exist_ok=True)
    tag = "" if args.dl_max is None else f"_dlmax{args.dl_max:.0f}"
    plot_sorted_injections(rows, os.path.join(out_dir, f"efficiency_sorted_linear{tag}.png"),
                           log_y=False, dl_max=args.dl_max, dl_min=args.dl_min)
    plot_sorted_injections(rows, os.path.join(out_dir, f"efficiency_sorted_log{tag}.png"),
                           log_y=True, dl_max=args.dl_max, dl_min=args.dl_min)
    plot_snr_panel(rows, out_dir)


if __name__ == "__main__":
    main()