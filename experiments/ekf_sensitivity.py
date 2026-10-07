"""Sensitivity of the A-versus-B comparison. Run from the repository root: python experiments/ekf_sensitivity.py [runs]

1. Calibration error of the fixed-gain filter (A) from 0 to 10 percent; B does not depend on it.
2. Marker visibility, changed through the maximum detection range.
"""

import dataclasses
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ekf_synthetic import CHI2_3_95, run_variant  # noqa: E402
from sim.synthetic import SyntheticConfig, simulate  # noqa: E402


def stats(cfg, seeds, name, whole=False, **kw):
    out = []
    for s in seeds:
        errs, nees, _ = run_variant(cfg, simulate(cfg, s), name, s, **kw)
        half = 0 if whole else len(errs) // 2
        out.append([1000 * np.sqrt(np.mean(np.sum(errs[half:, :2] ** 2, axis=1))),
                    np.degrees(np.sqrt(np.mean(errs[half:, 2] ** 2))), float(np.mean(nees)),
                    float(np.mean(nees <= CHI2_3_95))])
    a = np.array(out)
    return a.mean(axis=0), a.std(axis=0)


def fmt(m):
    return f"{m[0]:8.1f} mm | {m[1]:6.2f} deg | ANEES {m[2]:9.2f} | inside {m[3]:.2f}"


def duration_part(seeds):
    print("3. mission length: whole-run RMSE (includes the convergence phase of B), A is 5% miscalibrated")
    for dur in (60.0, 120.0, 300.0, 600.0):
        c = SyntheticConfig(duration=dur)
        ma, _ = stats(c, seeds, "A2", whole=True)
        m1, _ = stats(c, seeds, "A1", whole=True)
        mb, _ = stats(c, seeds, "B", whole=True)
        print(f"   {dur:4.0f} s")
        print(f"      A (5% miscalibrated): {fmt(ma)}")
        print(f"      A (exact)           : {fmt(m1)}")
        print(f"      B                   : {fmt(mb)}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    seeds = range(n)
    cfg = SyntheticConfig()
    if len(sys.argv) > 2 and sys.argv[2] == "duration":
        duration_part(seeds)
        sys.exit(0)
    print(f"{n} runs per row, {cfg.duration:.0f} s each, second-half position/heading RMSE")
    print("1. calibration error of the fixed-gain filter (both gains off by +e and -e)")
    mb, _ = stats(cfg, seeds, "B")
    for e in (0.0, 0.01, 0.02, 0.05, 0.10):
        ma, _ = stats(cfg, seeds, "A2", miscal=(1 + e, 1 - e))
        print(f"   e = {100 * e:4.0f}%  A: {fmt(ma)}")
    print(f"   B (independent of e): {fmt(mb)}")
    print("2. marker visibility (maximum detection range)")
    for rng_max in (1.5, 2.0, 3.0, 4.0):
        c = dataclasses.replace(cfg, max_range=rng_max)
        vis = 100 * np.mean([np.mean(np.all(np.isfinite(simulate(c, s)[2]), axis=1)) for s in seeds])
        ma, _ = stats(c, seeds, "A2")
        mb, _ = stats(c, seeds, "B")
        print(f"   range {rng_max:3.1f} m, marker visible {vis:4.0f}% of samples")
        print(f"      A (5% miscalibrated): {fmt(ma)}")
        print(f"      B                   : {fmt(mb)}")
