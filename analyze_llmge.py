"""Summarise an LLM-GE run on Titanic from its global_gen_*.pkl files.

Run from the repo root (on a compute node, not the login node):
    uv run python analyze_llmge.py                      # uses titanic_test/global_data
    uv run python analyze_llmge.py --run-dir titanic_test --out-dir analysis_out

Writes into --out-dir:
    llmge_individuals.csv  every individual: id, first generation, FP, FN, FPR, FNR, status, operator
    llmge_per_generation.csv  per-generation counts, errors, front size, new front members, AUC
    llmge_front.csv        the final Pareto front (what goes on the comparison plot)
    llmge_pareto.png       all individuals + final front, coloured by generation

Fitness in the pickles is (FP count, FN count), both minimised. Errored individuals
carry (inf, inf).
"""
import argparse
import glob
import math
import os
import pickle
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_generations(global_dir):
    files = {}
    for path in glob.glob(os.path.join(global_dir, "global_gen_*.pkl")):
        m = re.search(r"global_gen_(\d+)\.pkl$", path)
        if m:
            files[int(m.group(1))] = path
    if not files:
        raise SystemExit(f"No global_gen_*.pkl files found in {global_dir}")
    out = {}
    for gen in sorted(files):
        with open(files[gen], "rb") as fh:
            out[gen] = pickle.load(fh)
    return out


def is_valid(fit):
    return (
        fit is not None
        and hasattr(fit, "__len__")
        and len(fit) >= 2
        and not any(math.isinf(float(v)) or math.isnan(float(v)) for v in fit[:2])
    )


def snapshot(blob):
    """All individuals known at the end of a generation (history first, current overrides)."""
    snap = {}
    snap.update(blob.get("GLOBAL_DATA_HIST") or {})
    snap.update(blob.get("GLOBAL_DATA") or {})
    return snap


def nondominated(points):
    """points: iterable of (x, y) minimised. Returns the sorted set of non-dominated points."""
    pts = sorted(set(points))
    front, best_y = [], math.inf
    for x, y in pts:
        if y < best_y:
            front.append((x, y))
            best_y = y
    return front


def pareto_auc(points):
    """Area under an FPR/FNR staircase (lower is better); same definition as the MOGP notebook."""
    pts = sorted(set(points))
    fpr = [0.0] + [p[0] for p in pts] + [1.0]
    fnr = [1.0] + [p[1] for p in pts] + [0.0]
    area, best = 0.0, 1.0
    for i in range(len(fpr) - 1):
        best = min(best, fnr[i])
        area += (fpr[i + 1] - fpr[i]) * best
    return area


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="titanic_test", help="folder that holds global_data/")
    ap.add_argument("--out-dir", default="analysis_out")
    ap.add_argument("--n-neg", type=int, default=172, help="negatives in the validation split")
    ap.add_argument("--n-pos", type=int, default=123, help="positives in the validation split")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    gens = load_generations(os.path.join(args.run_dir, "global_data"))
    gen_ids = sorted(gens)
    print(f"Found generation files: {gen_ids}")

    # first generation in which each individual appears
    birth, record, ancestry = {}, {}, {}
    for g in gen_ids:
        snap = snapshot(gens[g])
        for gid, rec in snap.items():
            birth.setdefault(gid, g)
            record[gid] = rec
        ancestry.update(gens[g].get("GLOBAL_DATA_ANCESTRY") or {})

    def rates(fit):
        return float(fit[0]) / args.n_neg, float(fit[1]) / args.n_pos

    rows = []
    for gid, rec in record.items():
        fit = rec.get("fitness")
        valid = is_valid(fit)
        op = (ancestry.get(gid) or {}).get("MUTATE_TYPE") or []
        rows.append(
            {
                "gene_id": gid,
                "gen": birth[gid],
                "status": rec.get("status"),
                "valid": valid,
                "fp": float(fit[0]) if valid else "",
                "fn": float(fit[1]) if valid else "",
                "fpr": rates(fit)[0] if valid else "",
                "fnr": rates(fit)[1] if valid else "",
                "operator": op[-1] if op else "",
            }
        )
    rows.sort(key=lambda r: (r["gen"], r["gene_id"]))

    def write_csv(path, header, data):
        with open(path, "w") as fh:
            fh.write(",".join(header) + "\n")
            for r in data:
                fh.write(",".join(str(r[h]) for h in header) + "\n")

    write_csv(
        os.path.join(args.out_dir, "llmge_individuals.csv"),
        ["gene_id", "gen", "status", "valid", "fp", "fn", "fpr", "fnr", "operator"],
        rows,
    )

    # per generation: cumulative front, new members, errors
    per_gen, prev_front_ids = [], set()
    for g in gen_ids:
        upto = [r for r in rows if r["gen"] <= g and r["valid"]]
        front_pts = nondominated((r["fp"], r["fn"]) for r in upto)
        front_set = set(front_pts)
        # individuals sitting on the front: one id per distinct point is enough for "new"
        front_ids = {r["gene_id"] for r in upto if (r["fp"], r["fn"]) in front_set}
        new_ids = front_ids - prev_front_ids
        born_now = [r for r in rows if r["gen"] == g]
        per_gen.append(
            {
                "gen": g,
                "individuals": len(born_now),
                "valid": sum(r["valid"] for r in born_now),
                "errored": sum(not r["valid"] for r in born_now),
                "front_points": len(front_pts),
                "front_individuals": len(front_ids),
                "new_on_front": len([i for i in new_ids if birth[i] == g]),
                "auc_rates": round(pareto_auc([rates(p) for p in front_pts]), 5),
            }
        )
        prev_front_ids = front_ids
    write_csv(
        os.path.join(args.out_dir, "llmge_per_generation.csv"),
        ["gen", "individuals", "valid", "errored", "front_points", "front_individuals", "new_on_front", "auc_rates"],
        per_gen,
    )

    final_valid = [r for r in rows if r["valid"]]
    final_pts = nondominated((r["fp"], r["fn"]) for r in final_valid)
    front_rows = []
    for fp, fn in final_pts:
        owner = next(r for r in final_valid if (r["fp"], r["fn"]) == (fp, fn))
        front_rows.append({"gene_id": owner["gene_id"], "gen": owner["gen"], "fp": fp, "fn": fn,
                           "fpr": fp / args.n_neg, "fnr": fn / args.n_pos})
    write_csv(os.path.join(args.out_dir, "llmge_front.csv"),
              ["gene_id", "gen", "fp", "fn", "fpr", "fnr"], front_rows)

    # ---- printed summary
    print("\nPer generation (cumulative front):")
    print(f"{'gen':>4} {'indiv':>6} {'valid':>6} {'error':>6} {'front':>6} {'new':>4} {'AUC':>8}")
    for p in per_gen:
        print(f"{p['gen']:>4} {p['individuals']:>6} {p['valid']:>6} {p['errored']:>6} "
              f"{p['front_points']:>6} {p['new_on_front']:>4} {p['auc_rates']:>8}")
    status_counts = {}
    for r in rows:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
    print(f"\nTotal individuals: {len(rows)}   valid: {len(final_valid)}   errored: {len(rows) - len(final_valid)}")
    print(f"Status counts: {status_counts}")
    print("\nFinal Pareto front (FP, FN):", final_pts)

    # ---- plot
    fig, ax = plt.subplots(figsize=(8, 5.5))
    cmap = plt.get_cmap("viridis")
    for i, g in enumerate(gen_ids):
        pts = [(r["fpr"], r["fnr"]) for r in final_valid if r["gen"] == g]
        if pts:
            ax.scatter(*zip(*pts), s=28, alpha=0.65, color=cmap(i / max(1, len(gen_ids) - 1)),
                       label=f"gen {g}")
    fp_pts = [(p[0] / args.n_neg, p[1] / args.n_pos) for p in final_pts]
    xs = [fp_pts[0][0]] + [p[0] for p in fp_pts] + [1.0]
    ys = [1.0] + [p[1] for p in fp_pts] + [fp_pts[-1][1]]
    ax.plot(xs, ys, drawstyle="steps-post", color="black", linewidth=1.8, label="final Pareto front")
    # zoom to the data (the staircase runs out to 1.0 and would squash the points)
    all_x = [r["fpr"] for r in final_valid]
    all_y = [r["fnr"] for r in final_valid]
    pad = 0.03
    ax.set_xlim(max(0.0, min(all_x) - pad), min(1.0, max(all_x) + pad))
    ax.set_ylim(max(0.0, min(all_y) - pad), min(1.0, max(all_y) + pad))
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("False Negative Rate (FNR)")
    ax.set_title("LLM-GE individuals (validation split)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "llmge_pareto.png"), dpi=150)
    print(f"\nWrote files to {args.out_dir}/")


if __name__ == "__main__":
    main()
