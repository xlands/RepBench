#!/usr/bin/env python3
"""Contribution 3 (method axis): compare linear probing methods under the
identical leave-one-benchmark-out (LOBO) protocol used by probe_clusters.py.

Methods (all trained WITHOUT the held-out dataset's texts, tested on it vs
held-out negatives, exactly as diff-mean's LOBO):
  diffmean - mean(pos) - mean(neg), standardized dims (the benchmark's main)
  pca      - top-K PCs of the pooled train activations; the component with the
             best TRAIN AUC is selected and sign-oriented (unsupervised-direction
             baseline given its best shot; K reported)
  lr       - L2 logistic regression (C=0.1); capacity reference

For each cluster with >=2 datasets we take the median held-out AUC across
folds, then aggregate across clusters. Runs at each model's own best layer
(read from probe_results_{tag}.jsonl). Reuses hidden/{tag}/ on disk.

Usage: python3 method_compare.py --tag qwen3-8b
Reads  hidden/{tag}/, probe_results_{tag}.jsonl
Writes method_results_{tag}.jsonl  (per cluster: {diffmean, pca, lr})
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from probe_clusters import auc, diff_mean, load, stratified_neg

PCA_K = 10  # candidate components; the best-train-AUC one is picked


def pca_dir(X, pos_tr, neg_tr):
    """Top-K PCA on pooled train activations; return the component (as a unit
    direction) with the highest training AUC, sign-oriented toward positives."""
    tr = np.r_[pos_tr, neg_tr]
    y = np.r_[np.ones(len(pos_tr)), np.zeros(len(neg_tr))]
    Xtr = X[tr]
    k = min(PCA_K, Xtr.shape[0] - 1, Xtr.shape[1])
    if k < 1:
        return None
    comps = PCA(n_components=k, random_state=0).fit(Xtr).components_
    best, best_a = None, -1.0
    for v in comps:
        v = v / (np.linalg.norm(v) + 1e-8)
        s = Xtr @ v
        a = auc(s[y == 1], s[y == 0])
        if a < 0.5:            # allow either orientation
            v, a = -v, 1.0 - a
        if a > best_a:
            best, best_a = v, a
    return best


def lobo_median(X, pos, neg_tr, neg_te, ds, method):
    pos_ds = sorted(set(ds[pos]))
    if len(pos_ds) < 2:
        return None
    aucs = []
    for hold in pos_ds:
        tr = pos[ds[pos] != hold]
        te = pos[ds[pos] == hold]
        if len(tr) < 10 or len(te) < 10:
            continue
        if method == "diffmean":
            v = diff_mean(X, tr, neg_tr)
            a = auc(X[te] @ v, X[neg_te] @ v)
        elif method == "pca":
            v = pca_dir(X, tr, neg_tr)
            if v is None:
                continue
            a = auc(X[te] @ v, X[neg_te] @ v)
        elif method == "lr":
            Xtr = np.r_[X[tr], X[neg_tr]]
            ytr = np.r_[np.ones(len(tr)), np.zeros(len(neg_tr))]
            lr = LogisticRegression(C=0.1, max_iter=300).fit(Xtr, ytr)
            a = auc(lr.decision_function(X[te]), lr.decision_function(X[neg_te]))
        aucs.append(a)
    return float(np.median(aucs)) if aucs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--layer", type=int, default=None,
                    help="run at this layer instead of the best diff-mean "
                         "layer (per-depth ablation; output gets a _L{layer} suffix)")
    args = ap.parse_args()
    tag = args.tag

    rows = [json.loads(l) for l in Path(f"probe_results_{tag}.jsonl").open()]
    best_L = rows[0]["best_layer"] if args.layer is None else args.layer
    layers, H, ds, clusters, lens = load(tag)
    X = H[best_L]
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    all_cids = sorted({c for cl in clusters for c in cl})
    cl_idx = {c: np.array([i for i, cl in enumerate(clusters) if c in cl])
              for c in all_cids}
    print(f"[{tag}] best layer L{best_L}, {len(all_cids)} clusters")

    METHODS = ["diffmean", "pca", "lr"]
    out = []
    for c in all_cids:
        pos = cl_idx[c]
        neg = stratified_neg(c, clusters, ds, 6 * len(pos))
        half = len(neg) // 2
        neg_tr, neg_te = neg[:half], neg[half:]
        rec = {"cluster_id": c, "n_datasets": int(len(set(ds[pos])))}
        for m in METHODS:
            v = lobo_median(X, pos, neg_tr, neg_te, ds, m)
            rec[m] = round(v, 3) if v is not None else None
        out.append(rec)

    suffix = "" if args.layer is None else f"_L{best_L}"
    with Path(f"method_results_{tag}{suffix}.jsonl").open("w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[{tag}] best-layer L{best_L} — mean median-LOBO by method:")
    for m in METHODS:
        vals = [r[m] for r in out if r[m] is not None]
        strong = sum(v >= 0.8 for v in vals)
        print(f"  {m:9s} n={len(vals):3d}  mean {np.mean(vals):.3f}  "
              f"median {np.median(vals):.3f}  strong {strong}")


if __name__ == "__main__":
    main()
