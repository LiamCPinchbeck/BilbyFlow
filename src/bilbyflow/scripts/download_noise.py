#!/usr/bin/env python
"""
download_noise.py -- grab off-source noise from GWOSC.

Needs internet + gwpy + gwosc, so probably login node only if you're on a cluster.
For large runs, e.g. you want the maximum number of noise segments, then this can
take upwards of 120+ minutes depending on configuration/internet speed. Best to 
run a small test case first if you want to eventually get as much as possible (which
is generally recommended to mitigate over-training issues).

For default settings and 5 segments per era, it took ~20 minutes on my computer.
(~5-6 mins per era)


Output layout is what data/psd.py and io/strain.py already glob for:

    {noise-outdir}/{era}/{name}_H1_noise.npy
    {noise-outdir}/{era}/{name}_L1_noise.npy

Each file is np.array([t0, dt, strain], dtype=object). Strain is raw; the
15 Hz highpass happens on load.

Two things this is used for:

  --events GW150914 ...
      Noise sitting just before a specific event, so that event can get a
      PSD. Falls back to a post-event window if the pre-event data has a
      gap. A new event is useless without this.

  --eras O1 O2 O3a O3b
      Bulk noise for the training PSD bank. Windows are taken from H1_DATA
      & L1_DATA (both detectors up), and anything near a catalogued event is
      skipped so we don't train on a real signal by accident.

      With --n-per-era N: N windows drawn at random.
      Without it:        every non-overlapping window that fits. This is a
                         LOT of data -- run --dry-run first, it prints the
                         count and doesn't touch the network.

--config pulls sampling_frequency / duration / noise_data_dir straight out
of the training config. A file at the wrong sample rate gets dropped
by _segment_psds without a word and you find out ~an hour later when the
bank comes back with 0 PSDs (latest time this happened, 22/08/2026).

    python download_noise.py --config cfg.yaml --eras O3a --dry-run
    python download_noise.py --config cfg.yaml --eras O3a
    python download_noise.py --config cfg.yaml --eras O1 O2 O3a O3b --n-per-era 25
    python download_noise.py --config cfg.yaml --events GW200129_065458
    python download_noise.py --eras O3a --n-per-era 5 --sr 2048 --noise-outdir /tmp/n 
                               (typically using absolute path for this ^ leads to 
                                                                     less issues)
"""

import argparse
import os
import time
from tqdm import tqdm

import numpy as np

# run boundaries, for the era/ subdir name and for where to look for segments
ERA_GPS = {
    "O1":  (1126051217, 1137254417),
    "O2":  (1164556817, 1187733618),
    "O3a": (1238166018, 1253977218),
    "O3b": (1256655618, 1269363618),
}

MIN_WELCH_WINDOW = 32.0    # data/psd.py _WINDOW_LENGTHS starts at 32 s


def era_of_gps(gps):
    try:
        from gwosc.datasets import run_at_gps
        run = str(run_at_gps(gps))
        for era in ("O3a", "O3b", "O2", "O1"):   # O3a/b first, they'd match O3
            if run.startswith(era):
                return era
    except Exception:
        pass
    for era, (s, e) in ERA_GPS.items():
        if s <= gps < e:
            return era
    return "unknown_era"


def _unwrap(exc):
    # gwpy raises ExceptionGroup and the useful bit is inside. ValueError ==
    # data gap (don't retry), URL/HTTP == server (do retry).
    subs = getattr(exc, "exceptions", None)
    if subs:
        return f"{type(exc).__name__}: " + " | ".join(_unwrap(e) for e in subs)
    return f"{type(exc).__name__}: {exc}"


def _save(ts, outfile):
    np.save(outfile, np.array(
        [float(ts.t0.value), float(ts.dt.value), ts.value.astype(np.float64)],
        dtype=object))


def fetch(det, start, end, sr, retries=3, backoff=5.0):
    from gwpy.timeseries import TimeSeries
    last = None
    for attempt in range(1, retries + 1):
        try:
            raw = TimeSeries.fetch_open_data(det, start, end,
                                             sample_rate=4096, cache=False)
            return raw.resample(sr) if sr != 4096 else raw
        except Exception as e:
            last = e
            msg = _unwrap(e)
            print(f"      attempt {attempt}/{retries}: {msg}")
            transient = any(k in msg.lower() for k in
                            ("http", "url", "timeout", "connection", "503",
                             "500", "429", "temporarily", "reset"))
            if attempt < retries and transient:
                time.sleep(backoff * attempt)
                continue
            break
    raise RuntimeError(_unwrap(last))


def check(ts, sr, min_seconds):
    """Returns a reason string if this segment would be junk, else None.
    Cheaper to catch it here than to have the bank builder drop it."""
    got_sr = int(round(1.0 / float(ts.dt.value)))
    if got_sr != sr:
        return f"sr {got_sr} != {sr}"
    if len(ts.value) / sr < min_seconds:
        return f"{len(ts.value) / sr:.0f}s < {min_seconds:.0f}s"
    finite = np.isfinite(ts.value)
    if not finite.all():
        return f"{int((~finite).sum())} non-finite samples"
    if float(np.std(ts.value[finite])) == 0.0:
        return "flat"
    return None


# ---- segment finding (era mode) ----

def science_segments(era, min_length):
    """Stretches where both detectors were up, longer than min_length."""
    from gwpy.segments import DataQualityFlag
    start, end = ERA_GPS[era]
    print(f"                {era}: fetching segment lists ({(end - start) / 86400:.0f} d)")
    h1 = DataQualityFlag.fetch_open_data("H1_DATA", start, end)
    l1 = DataQualityFlag.fetch_open_data("L1_DATA", start, end)
    segs = [(float(s.start), float(s.end)) for s in (h1 & l1).active
            if float(s.end - s.start) >= min_length]
    print(f"                {era}: {len(segs)} coincident stretches >= {min_length:.0f}s "
                            f"({sum(e - s for s, e in segs) / 3600:.0f} h)")
    return segs


def event_times(max_query=None):
    """GPS times of catalogued events, for vetoing noise windows.

    max_query caps how many events are actually queried (each ~0.5 s).
    For small n_per_era the veto doesn't need to be exhaustive — the
    chance of a random window landing on an un-vetoed event is negligible.
    Subsampled deterministically (seed 0) so reruns are consistent."""
    from gwosc.datasets import find_datasets, event_gps as _gps

    all_names = find_datasets(type="events")
    # dedup versions: GW150914-v1, -v2, -v3 -> one query
    seen = {}
    for v in all_names:
        base = v.partition("-v")[0]
        if base not in seen:
            seen[base] = v
    candidates = list(seen.values())

    if max_query is not None and len(candidates) > max_query:
        rng = np.random.default_rng(0)
        candidates = list(rng.choice(candidates, size=max_query, replace=False))
        print(f"  veto: querying {max_query} of {len(seen)} events "
              f"(capped at 2 * n_per_era * n_eras)")

    out = set()
    for vname in candidates:
        try:
            out.add(float(_gps(vname)))
        except Exception:
            continue
    print(f"  {len(out)} event times to veto")
    return np.array(sorted(out))



def _tile(segs, length, vetoes, veto_pad):
    """All non-overlapping windows, laid end to end from each stretch start."""
    out = []
    for starting, ending in segs:
        start = starting
        while start + length <= ending:
            stop = start + length
            near_event = vetoes.size and np.any(
                (vetoes > start - veto_pad) & (vetoes < stop + veto_pad))
            if not near_event:
                out.append((start, stop))
            start = stop
    return out


def pick_windows(segs, n, length, vetoes, veto_pad, rng):
    avail = _tile(segs, length, vetoes, veto_pad)

    if n is None:
        print(f"    {len(avail)} windows of {length:.0f}s available")
        return avail
    if n >= len(avail):
        print(f"    asked for {n}, only {len(avail)} exist -> taking all")
        return avail
    if n > 0.5 * len(avail):
        # rejection sampling can't place the last few when the space is
        # nearly full, so thin the tiling instead
        print(f"    taking {n} of {len(avail)}")
        return [avail[i] for i in sorted(rng.choice(len(avail), n,
                                                    replace=False))]

    # sparse case: sample directly, weighted by stretch length
    w = np.array([e - s for s, e in segs], float)
    w /= w.sum()
    picked, tries = [], 0
    while len(picked) < n and tries < 200 * n:
        tries += 1
        s, e = segs[rng.choice(len(segs), p=w)]
        start = rng.uniform(s, e - length)
        stop = start + length
        if vetoes.size and np.any((vetoes > start - veto_pad)
                                  & (vetoes < stop + veto_pad)):
            continue
        if any(not (stop <= a or start >= b) for a, b in picked):
            continue
        picked.append((start, stop))
    if len(picked) < n:
        print(f"    only got {len(picked)}/{n} after {tries} tries")
    return sorted(picked)


# ---- two fetch modes, can either get noise data based on events or eras ----

def noise_for_events(events, args, sr, min_seconds):
    from gwosc.datasets import event_gps

    for name in events:
        try:
            gps = float(event_gps(name))
        except Exception as e:
            print(f"{name}: no GPS ({e})")
            continue

        era = era_of_gps(gps)
        if era == "unknown_era":
            print(f"{name}: GPS {gps:.0f} is outside O1-O3b, so its PSD is "
                  f"outside what the flow was conditioned on. Continuing.")
        era_dir = os.path.join(args.noise_outdir, era)
        os.makedirs(era_dir, exist_ok=True)
        print(f"{name}: {gps:.1f} ({era})")

        # before the event by preference; after it if there's a gap
        windows = [(gps - args.gap - args.length, gps - args.gap, "pre"),
                   (gps + args.gap, gps + args.gap + args.length, "post")]

        for det in ("H1", "L1"):
            out = os.path.join(era_dir, f"{name}_{det}_noise.npy")
            if os.path.exists(out) and not args.overwrite:
                print(f"  {det}: have it")
                continue
            for start, stop, label in windows:
                if args.dry_run:
                    print(f"  {det}: would fetch {label} "
                          f"[{start:.0f}, {stop:.0f}]")
                    break
                try:
                    print(f"  {det}: {label}, {args.length:.0f}s")
                    ts = fetch(det, start, stop, sr)
                    bad = check(ts, sr, min_seconds)
                    if bad:
                        print(f"  {det}: no good ({bad})")
                        continue
                    _save(ts, out)
                    print(f"  {det}: -> {out}")
                    break
                except Exception as e:
                    print(f"  {det}: {label} window failed ({e})")
            else:
                print(f"  {name} {det}: FAILED, no usable off-source data. "
                      f"This event can't be PSD-estimated.")


def noise_for_eras(args, sr, min_seconds):
    rng = np.random.default_rng(args.seed)
    max_q = (2 * args.n_per_era if args.n_per_era is not None else None)
    vetoes = np.array([]) if args.dry_run else event_times(max_query=max_q)

    for era in tqdm(args.eras, desc="Getting data for eras"):
        print("\n")
        print(f"--- ")

        print(f"  Getting noise for {era}")
        era_dir = os.path.join(args.noise_outdir, era)
        os.makedirs(era_dir, exist_ok=True)

        print(f"                Running  [science_segments]")
        segs = science_segments(era, args.length)
        if not segs:
            print(f"  {era}: nothing long enough")
            continue

        print(f"                Running  [pick_windows]")
        windows = pick_windows(segs, args.n_per_era, args.length,
                               vetoes, args.event_veto, rng)
        
        print(f"                Running  [fetching and cleaning data]")

        kept = 0
        for i, (start, stop) in tqdm(enumerate(windows), total=len(windows), leave=False, desc="Getting noise from the   segment(s)"):
            tag = f"{era}_seg{i:04d}_{int(start)}"
            outs = {d: os.path.join(era_dir, f"{tag}_{d}_noise.npy")
                    for d in ("H1", "L1")}

            if all(map(os.path.exists, outs.values())) and not args.overwrite:
                kept += 1
                continue
            if args.dry_run:
                print(f"  {tag}: [{start:.0f}, {stop:.0f}]")
                kept += 1
                continue

            print(f"  {tag}")
            got = {}
            for det in ("H1", "L1"):
                try:
                    ts = fetch(det, start, stop, sr)
                    bad = check(ts, sr, min_seconds)
                    if bad:
                        print(f"    {det}: {bad}")
                        break
                    got[det] = ts
                except Exception as e:
                    print(f"    {det}: {e}")
                    break

            # both or neither -- the bank pairs them by era and a lone H1
            # file just sits there unused
            if len(got) != 2:
                print(f"    dropped (only got {list(got)})")
                continue
            for det, ts in got.items():
                _save(ts, outs[det])
            kept += 1

        print(f"  {era}: {kept}/{len(windows)} pairs")


def main():
    p = argparse.ArgumentParser(description="Off-source GWOSC noise")
    p.add_argument("--config", default=None,
                   help="training config.yaml: takes sampling_frequency, "
                        "duration and noise_data_dir from it")
    p.add_argument("--noise-outdir", default=None,
                   help="where the era/ dirs go (default: noise_data_dir "
                        "from --config)")

    p.add_argument("--events", nargs="+", default=None)
    p.add_argument("--eras", nargs="+", default=None, choices=sorted(ERA_GPS))
    p.add_argument("--n-per-era", type=int, default=None,
                   help="how many window pairs per era. Leave unset to take "
                        "everything that fits (check --dry-run first)")

    p.add_argument("--length", type=float, default=4096.0,
                   help="seconds per file (default 4096)")
    p.add_argument("--sr", type=int, default=None, help="overridden by --config")
    p.add_argument("--gap", type=float, default=16.0,
                   help="keep this far away from the event itself")
    p.add_argument("--event-veto", type=float, default=64.0,
                   help="era windows this close to any catalogued event are "
                        "skipped")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if not (args.events or args.eras):
        p.error("need --events or --eras")

    sr, duration = args.sr, 4.0
    if args.config:
        import yaml
        cfg = yaml.safe_load(open(args.config))
        sr = int(cfg["sampling_frequency"])
        duration = float(cfg["duration"])
        if args.noise_outdir is None:
            args.noise_outdir = cfg.get("noise_data_dir")
        print(f"from config: sr={sr}, duration={duration}s, "
              f"outdir={args.noise_outdir}")
    if args.noise_outdir is None:
        p.error("need --noise-outdir, or --config with noise_data_dir in it")
    if sr is None:
        sr = 2048
        print(f"WARNING: no --config or --sr, assuming {sr} Hz. If the "
              f"training config says otherwise these files are wasted.")

    min_seconds = max(MIN_WELCH_WINDOW, duration)

    os.makedirs(args.noise_outdir, exist_ok=True)
    print(f"-> {args.noise_outdir}, {args.length:.0f}s @ {sr} Hz "
          f"(~{args.length * sr * 8 / 1e6:.0f} MB per file)\n")

    if args.events:
        print(f"Getting noise around events: {args.events}")
        noise_for_events(args.events, args, sr, min_seconds)
    if args.eras:
        print(f"Getting noise for eras: {args.eras}")
        noise_for_eras(args, sr, min_seconds)


if __name__ == "__main__":
    main()