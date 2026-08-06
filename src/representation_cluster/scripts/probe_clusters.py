#!/usr/bin/env python3
"""Probe step 3: diff-mean capability directions + generalization tests.

Per cluster c and layer L:
  v_c = mean(h | c) - mean(h | stratified negative pool)   (standardized dims)

Protocols:
  within  - random half-split of the cluster's texts (upper bound; the
            direction may exploit dataset fingerprints)
  lobo    - leave-one-benchmark-out: direction trained WITHOUT any text
            from the held-out dataset, tested on that dataset vs held-out
            negatives. This is the cross-benchmark generalization claim.
  spec    - 94x94 transfer matrix: projection of cluster j's texts onto
            v_i; diagonal rank measures specificity.
  lenr    - |spearman| of projection vs log length on negatives (confound).

Logistic-regression probe (same LOBO protocol) runs at the best diff-mean
layer as a capacity upper reference.

Usage: python3 probe_clusters.py --tag qwen3-8b
Reads  hidden/{tag}/, writes probe_results_{tag}.jsonl,
probe_report_{tag}.md, directions_{tag}_L{best}.npy, spec_matrix_{tag}.npy
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

NSHARDS = 4
RNG = np.random.default_rng(0)
NEG_PER_DS = 30  # stratification cap when sampling the negative pool


def load(tag):
    d = Path("hidden") / tag
    layers = sorted(int(re.search(r"_L(\d+)\.npy", p.name).group(1))
                    for p in d.glob("h_0_L*.npy"))
    shards = sorted((int(p.stem.split("_")[1]) for p in d.glob("uids_*.json")))
    uids, feats = [], {L: [] for L in layers}
    for k in shards:
        uids += json.loads((d / f"uids_{k}.json").read_text())
        for L in layers:
            feats[L].append(np.load(d / f"h_{k}_L{L}.npy"))
    H = {L: np.concatenate(feats[L]).astype(np.float32) for L in layers}
    meta = {t["uid"]: t for t in
            (json.loads(l) for l in Path("probe_texts.jsonl").open())}
    keep = [i for i, u in enumerate(uids) if u in meta]
    if len(keep) != len(uids):
        print(f"[{tag}] dropping {len(uids) - len(keep)} uids not in probe_texts")
        uids = [uids[i] for i in keep]
        H = {L: H[L][keep] for L in layers}
    ds = np.array([meta[u]["dataset_id"] for u in uids])
    clusters = [set(meta[u]["clusters"]) for u in uids]
    lens = np.log1p([len(meta[u]["text"]) for u in uids]).astype(np.float32)
    return layers, H, ds, clusters, lens


def auc(pos_scores, neg_scores):
    y = np.r_[np.ones(len(pos_scores)), np.zeros(len(neg_scores))]
    return roc_auc_score(y, np.r_[pos_scores, neg_scores])


def diff_mean(X, pos_idx, neg_idx):
    v = X[pos_idx].mean(0) - X[neg_idx].mean(0)
    return v / (np.linalg.norm(v) + 1e-8)


def stratified_neg(cid, all_clusters, ds, n_target):
    """Negatives: texts sharing no cluster with cid, capped per dataset."""
    ok = np.array([cid not in cl for cl in all_clusters])
    by_ds = defaultdict(list)
    for i in np.nonzero(ok)[0]:
        by_ds[ds[i]].append(i)
    picked = []
    for d, idxs in by_ds.items():
        idxs = np.array(idxs)
        if len(idxs) > NEG_PER_DS:
            idxs = RNG.choice(idxs, NEG_PER_DS, replace=False)
        picked.append(idxs)
    neg = np.concatenate(picked)
    RNG.shuffle(neg)
    return neg[:n_target] if len(neg) > n_target else neg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    tag = args.tag

    layers, H, ds, clusters, lens = load(tag)
    n = len(ds)
    all_cids = sorted({c for cl in clusters for c in cl})
    print(f"[{tag}] {n} texts, {len(all_cids)} clusters, layers {layers}")

    Xs = {}
    for L in layers:
        mu, sd = H[L].mean(0), H[L].std(0) + 1e-6
        Xs[L] = (H[L] - mu) / sd

    cl_idx = {c: np.array([i for i, cl in enumerate(clusters) if c in cl])
              for c in all_cids}

    results = {c: {"cluster_id": c, "n_pos": int(len(cl_idx[c])),
                   "n_datasets": int(len(set(ds[cl_idx[c]]))), "layers": {}}
               for c in all_cids}

    for L in layers:
        X = Xs[L]
        for c in all_cids:
            pos = cl_idx[c]
            neg = stratified_neg(c, clusters, ds, 6 * len(pos))
            half = len(neg) // 2
            neg_tr, neg_te = neg[:half], neg[half:]

            p = RNG.permutation(pos)
            v = diff_mean(X, p[:len(p) // 2], neg_tr)
            within = auc(X[p[len(p) // 2:]] @ v, X[neg_te] @ v)

            pos_ds = sorted(set(ds[pos]))
            lobo_aucs, per_ds = [], {}
            if len(pos_ds) >= 2:
                for hold in pos_ds:
                    tr = pos[ds[pos] != hold]
                    te = pos[ds[pos] == hold]
                    if len(tr) < 10 or len(te) < 10:
                        continue
                    v = diff_mean(X, tr, neg_tr)
                    a = auc(X[te] @ v, X[neg_te] @ v)
                    lobo_aucs.append(a)
                    per_ds[hold] = round(float(a), 3)
            lobo = float(np.mean(lobo_aucs)) if lobo_aucs else None
            lobo_med = float(np.median(lobo_aucs)) if lobo_aucs else None

            v_full = diff_mean(X, pos, neg_tr)
            s_neg = X[neg_te] @ v_full
            lenr = abs(spearmanr(s_neg, lens[neg_te]).statistic)

            results[c]["layers"][L] = {
                "within": round(float(within), 3),
                "lobo": round(lobo, 3) if lobo is not None else None,
                "lobo_med": round(lobo_med, 3) if lobo_med is not None else None,
                "lobo_per_ds": per_ds, "len_r": round(float(lenr), 3)}
        vals = [r["layers"][L]["lobo_med"] for r in results.values()
                if r["layers"][L]["lobo_med"] is not None]
        print(f"[{tag}] layer {L}: mean median-LOBO {np.mean(vals):.3f}")

    def score(L):
        vals = [r["layers"][L]["lobo_med"] if r["layers"][L]["lobo_med"]
                is not None else r["layers"][L]["within"]
                for r in results.values()]
        return np.mean(vals)
    best_L = max(layers, key=score)
    print(f"[{tag}] best layer: {best_L}")

    # LR upper reference at best layer, LOBO protocol
    X = Xs[best_L]
    for c in all_cids:
        pos = cl_idx[c]
        neg = stratified_neg(c, clusters, ds, 6 * len(pos))
        half = len(neg) // 2
        neg_tr, neg_te = neg[:half], neg[half:]
        pos_ds = sorted(set(ds[pos]))
        aucs = []
        if len(pos_ds) >= 2:
            for hold in pos_ds:
                tr, te = pos[ds[pos] != hold], pos[ds[pos] == hold]
                if len(tr) < 10 or len(te) < 10:
                    continue
                Xtr = np.r_[X[tr], X[neg_tr]]
                ytr = np.r_[np.ones(len(tr)), np.zeros(len(neg_tr))]
                lr = LogisticRegression(C=0.1, max_iter=300).fit(Xtr, ytr)
                aucs.append(auc(lr.decision_function(X[te]),
                                lr.decision_function(X[neg_te])))
        results[c]["lobo_lr"] = round(float(np.mean(aucs)), 3) if aucs else None

    # specificity matrix at best layer (full directions)
    V = np.stack([diff_mean(X, cl_idx[c],
                            stratified_neg(c, clusters, ds, 6 * len(cl_idx[c])))
                  for c in all_cids])
    np.save(f"directions_{tag}_L{best_L}.npy", V)
    P = X @ V.T
    P = (P - P.mean(0)) / (P.std(0) + 1e-8)
    M = np.stack([P[cl_idx[c]].mean(0) for c in all_cids])
    np.save(f"spec_matrix_{tag}.npy", M)
    for i, c in enumerate(all_cids):
        results[c]["spec_rank"] = int((M[:, i] >= M[i, i]).sum())
        results[c]["best_layer"] = best_L

    with Path(f"probe_results_{tag}.jsonl").open("w") as f:
        for c in all_cids:
            f.write(json.dumps(results[c], ensure_ascii=False) + "\n")

    # report
    tax = {r["cluster_id"]: r for r in
           (json.loads(l) for l in Path("experiment_clusters.jsonl").open())}
    key = lambda r: r["layers"][best_L]["lobo_med"]
    rows = sorted(results.values(), key=lambda r: -(key(r) or 0))
    lobo_all = [key(r) for r in rows if key(r) is not None]
    tier = lambda a: ("strong" if a >= 0.8 else
                      "moderate" if a >= 0.65 else "weak")
    suspects = []
    for r in rows:
        ly = r["layers"][best_L]
        if ly["lobo_med"] is not None and ly["lobo_med"] >= 0.65:
            for d, a in ly["lobo_per_ds"].items():
                if a < 0.5:
                    suspects.append((r["cluster_id"], d, a, ly["lobo_med"]))
    out = [f"# Diff-mean 能力方向 probe 报告 — {tag}\n",
           f"- last-token hidden states，最佳层 **L{best_L}**/{layers}"
           f"（按各层平均中位 LOBO 选出）",
           f"- 有 LOBO（≥2 数据集）的簇 {len(lobo_all)}/{len(rows)}；"
           f"中位 LOBO AUC（簇内取中位）平均 **{np.mean(lobo_all):.3f}**，"
           f"strong(≥0.8) {sum(a >= .8 for a in lobo_all)}、"
           f"moderate {sum(.65 <= a < .8 for a in lobo_all)}、"
           f"weak {sum(a < .65 for a in lobo_all)}",
           f"- 特异性：spec_rank=1（自簇投影最高）的簇 "
           f"{sum(r['spec_rank'] == 1 for r in rows)}/{len(rows)}",
           "- within 接近 1.0 只说明可分数据集指纹，不作依据；"
           "泛化结论看 LOBO（方向未见过被测 benchmark）\n",
           "| cluster | n | ds | within | LOBO均值 | LOBO中位 | LR-LOBO "
           "| spec | len_r | 档 |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        ly = r["layers"][best_L]
        name = tax[r["cluster_id"]]["cluster_name"]
        lb, lm = ly["lobo"], ly["lobo_med"]
        out.append(
            f"| {name} ({r['cluster_id']}) | {r['n_pos']} | {r['n_datasets']} "
            f"| {ly['within']} | {lb if lb is not None else '—'} "
            f"| {lm if lm is not None else '—'} "
            f"| {r['lobo_lr'] if r['lobo_lr'] is not None else '—'} "
            f"| {r['spec_rank']} | {ly['len_r']} "
            f"| {tier(lm) if lm is not None else 'no-lobo'} |")
    if suspects:
        out.append("\n## 可疑 benchmark→簇 映射（簇整体泛化好，单数据集 AUC<0.5）\n")
        for cid, d, a, med in suspects:
            out.append(f"- {cid} ({tax[cid]['cluster_name']}): **{d}** "
                       f"AUC {a}（簇中位 {med}）")
    Path(f"probe_report_{tag}.md").write_text("\n".join(out), encoding="utf-8")
    print(f"[{tag}] mean median-LOBO {np.mean(lobo_all):.3f}; "
          f"{len(suspects)} suspect mappings")


if __name__ == "__main__":
    main()
