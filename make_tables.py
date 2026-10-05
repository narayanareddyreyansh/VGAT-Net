"""Regenerate the result tables and the paired significance tests from per-image metric files.

Each experiment directory holds seed*/per_image_test.csv written by scripts/per_image_metrics.py.

    # mean +/- standard deviation over seeds (Tables 6-8 style)
    python scripts/make_tables.py summary runs/vgatnet_octa500 runs/baseline_unet_octa500

    # paired comparison of two methods on the same test set (Table 10 style):
    # per-image metrics are averaged over seeds, then a two-tailed paired t-test per metric,
    # Holm-Bonferroni correction within the (DSC, IoU, HD95) family, and Cohen's d for paired samples
    python scripts/make_tables.py compare runs/vgatnet_octa500 runs/baseline_ddnet_octa500 --task vessel
"""
import argparse
import csv
import glob
import os

import numpy as np

METRICS = ["DSC", "IoU", "SE", "SP", "HD95"]


def load_experiment(exp_dir):
    """Return {seed: {image_id: {metric_name: value}}}."""
    seeds = {}
    for f in sorted(glob.glob(os.path.join(exp_dir, "seed*", "per_image_test.csv"))):
        seed = os.path.basename(os.path.dirname(f))
        rows = list(csv.DictReader(open(f)))
        seeds[seed] = {r["id"]: {k: float(v) for k, v in r.items() if k != "id"} for r in rows}
    if not seeds:
        raise SystemExit(f"no per_image_test.csv under {exp_dir}")
    return seeds


def summary(exp_dirs):
    for d in exp_dirs:
        seeds = load_experiment(d)
        print(f"\n{d}  ({len(seeds)} seeds, {len(next(iter(seeds.values())))} test images)")
        for task in ("vessel", "faz"):
            line = []
            for m in METRICS:
                per_seed = [np.nanmean([img[f"{task}_{m}"] for img in s.values()]) for s in seeds.values()]
                line.append(f"{m} {np.mean(per_seed):6.2f} +/- {np.std(per_seed):.2f}")
            print(f"  {task:6s} " + "   ".join(line))


def seed_averaged(seeds, task, metric):
    ids = sorted(set.intersection(*[set(s) for s in seeds.values()]))
    return ids, np.array([np.nanmean([seeds[s][i][f"{task}_{metric}"] for s in seeds]) for i in ids])


def holm(pvals):
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, pvals[idx] * (len(pvals) - rank))
        running = max(running, val)
        adj[idx] = running
    return adj


def compare(exp_a, exp_b, task):
    from scipy import stats

    a, b = load_experiment(exp_a), load_experiment(exp_b)
    results = []
    for m in ("DSC", "IoU", "HD95"):
        ids_a, xa = seed_averaged(a, task, m)
        ids_b, xb = seed_averaged(b, task, m)
        assert ids_a == ids_b, "the two experiments must share the same test images"
        diff = xa - xb
        n = len(diff)
        t, p = stats.ttest_rel(xa, xb)
        se = diff.std(ddof=1) / np.sqrt(n)
        ci = stats.t.ppf(0.975, n - 1) * se
        d = diff.mean() / diff.std(ddof=1)
        results.append((m, n, diff.mean(), ci, t, p, d))
    adj = holm([r[5] for r in results])
    print(f"\n{task}: {exp_a}  vs  {exp_b}")
    print("metric   n   mean diff (95% CI)          t (df)        p raw      p Holm     Cohen d")
    for (m, n, md, ci, t, p, d), ph in zip(results, adj):
        print(f"{m:6s} {n:3d}   {md:+6.2f} ({md-ci:+.2f}, {md+ci:+.2f})   {t:6.2f} ({n-1})   {p:.2e}   {ph:.2e}   {d:5.2f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summary")
    s.add_argument("exp_dirs", nargs="+")
    c = sub.add_parser("compare")
    c.add_argument("exp_a")
    c.add_argument("exp_b")
    c.add_argument("--task", choices=["vessel", "faz"], default="vessel")
    a = ap.parse_args()
    if a.cmd == "summary":
        summary(a.exp_dirs)
    else:
        compare(a.exp_a, a.exp_b, a.task)


if __name__ == "__main__":
    main()
