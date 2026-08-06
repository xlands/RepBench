#!/usr/bin/env python3
"""Gemma Scope SAE concept detection under the benchmark's LOBO protocol.

This implements the SAE-AUC baseline used by AxBench: on every training fold,
select the *single* SAE latent with the largest training ROC AUC (allowing
either sign), then evaluate that frozen feature and sign on the held-out
benchmark.  Feature selection never sees held-out examples.

The existing Hugging Face extraction stores hidden_states[L], which is the
residual stream after transformer block L-1.  Consequently these pairs align:
  hidden L10 <-> Gemma Scope layer_9
  hidden L21 <-> Gemma Scope layer_20
  hidden L32 <-> Gemma Scope layer_31

The Gemma Scope JumpReLU encoder is:
  pre = x @ W_enc + b_enc
  z   = relu(pre) * (pre > threshold)

Usage:
  python3 sae_probe.py --encode
  python3 sae_probe.py --probe
  python3 sae_probe.py --encode --probe
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

from probe_clusters import auc, load, stratified_neg


TAG = "gemma2-9b"
SCOPE_REPO = "google/gemma-scope-9b-it-res"
SCOPE_REVISION = "e86af97a5b6fbbccca28ab654f2fda1b0768f770"
LAYER_CONFIG = {
    10: ("layer_9", "width_16k", "average_l0_47",
         "layer_9_width_16k_l0_47.npz"),
    21: ("layer_20", "width_16k", "average_l0_47",
         "layer_20_width_16k_l0_47.npz"),
    32: ("layer_31", "width_16k", "average_l0_43",
         "layer_31_width_16k_l0_43.npz"),
}
WEIGHT_DIR = Path("sae_weights/gemma-scope-9b-it-res")
ACT_DIR = Path("sae_acts/gemma2-9b")


def encode_layer(X, weight_path, output_path, batch_size=1024):
    """Encode all last-token residual activations to an on-disk float16 array."""
    params = np.load(weight_path)
    W = torch.from_numpy(params["W_enc"]).to("cuda", dtype=torch.float32)
    b = torch.from_numpy(params["b_enc"]).to("cuda", dtype=torch.float32)
    threshold = torch.from_numpy(params["threshold"]).to(
        "cuda", dtype=torch.float32)
    if X.shape[1] != W.shape[0]:
        raise ValueError(f"activation/SAE mismatch: {X.shape} vs {tuple(W.shape)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Z = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.float16,
        shape=(len(X), W.shape[1]))
    l0_sum = 0
    with torch.inference_mode():
        for start in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[start:start + batch_size]).to(
                "cuda", dtype=torch.float32)
            pre = xb @ W + b
            z = torch.where(pre > threshold, torch.relu(pre),
                            torch.zeros((), device=pre.device))
            l0_sum += int((z > 0).sum().item())
            Z[start:start + len(xb)] = z.to(torch.float16).cpu().numpy()
            if start % (10 * batch_size) == 0:
                print(f"  encoded {start}/{len(X)}", flush=True)
    Z.flush()
    observed_l0 = l0_sum / len(X)
    print(f"  wrote {output_path}; observed mean L0={observed_l0:.1f}")
    return observed_l0


def feature_aucs(X, pos_idx, neg_idx, chunk=512):
    """Exact tie-aware AUC for every feature, in bounded-memory chunks."""
    idx = np.r_[pos_idx, neg_idx]
    y = np.r_[np.ones(len(pos_idx), dtype=np.float64),
              np.zeros(len(neg_idx), dtype=np.float64)]
    n_pos, n_neg = len(pos_idx), len(neg_idx)
    out = np.empty(X.shape[1], dtype=np.float64)
    for j in range(0, X.shape[1], chunk):
        # Average ranks make this exactly equivalent to ROC AUC even with the
        # large zero ties characteristic of SAE activations.
        values = np.asarray(X[idx, j:j + chunk], dtype=np.float32)
        ranks = rankdata(values, method="average", axis=0)
        rank_sum = (ranks * y[:, None]).sum(axis=0)
        out[j:j + chunk] = (
            rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return out


def run_probe(workers=8):
    _, _, ds, clusters, _ = load(TAG)
    all_cids = sorted({c for cl in clusters for c in cl})
    cl_idx = {
        c: np.array([i for i, cl in enumerate(clusters) if c in cl])
        for c in all_cids
    }
    results = {
        c: {"cluster_id": c, "n_pos": int(len(cl_idx[c])),
            "n_datasets": int(len(set(ds[cl_idx[c]]))), "layers": {}}
        for c in all_cids
    }

    for hidden_layer, (scope_layer, width, l0_name, _) in LAYER_CONFIG.items():
        act_path = ACT_DIR / f"L{hidden_layer}.npy"
        if not act_path.exists():
            raise FileNotFoundError(f"missing encoded activations: {act_path}")
        Z = np.load(act_path, mmap_mode="r")
        print(f"[SAE] L{hidden_layer}/{scope_layer}: {Z.shape}", flush=True)
        for ci, c in enumerate(all_cids):
            pos = cl_idx[c]
            neg = stratified_neg(c, clusters, ds, 6 * len(pos))
            half = len(neg) // 2
            neg_tr, neg_te = neg[:half], neg[half:]
            fold_inputs = []
            for hold in sorted(set(ds[pos])):
                tr = pos[ds[pos] != hold]
                te = pos[ds[pos] == hold]
                if len(tr) < 10 or len(te) < 10:
                    continue
                fold_inputs.append((hold, tr, te))

            def evaluate_fold(fold):
                hold, tr, te = fold
                train_auc = feature_aucs(Z, tr, neg_tr)
                oriented = np.maximum(train_auc, 1.0 - train_auc)
                feature = int(np.argmax(oriented))
                sign = 1 if train_auc[feature] >= 0.5 else -1
                heldout_auc = auc(sign * np.asarray(Z[te, feature]),
                                  sign * np.asarray(Z[neg_te, feature]))
                return hold, feature, sign, float(oriented[feature]), float(heldout_auc)

            # Fold calculations are independent and scipy's ranking kernels
            # release the GIL. Threads also share the read-only activation
            # memmap, avoiding multi-GB copies.
            with ThreadPoolExecutor(max_workers=min(workers, len(fold_inputs) or 1)) as ex:
                fold_outputs = list(ex.map(evaluate_fold, fold_inputs))
            folds, per_ds, selected = [], {}, {}
            for hold, feature, sign, train_auc, heldout_auc in fold_outputs:
                folds.append(heldout_auc)
                per_ds[hold] = round(float(heldout_auc), 3)
                selected[hold] = {
                    "feature": feature, "sign": sign,
                    "train_auc": round(train_auc, 3)}
            med = float(np.median(folds)) if folds else None
            mean = float(np.mean(folds)) if folds else None
            results[c]["layers"][str(hidden_layer)] = {
                "scope_layer": int(scope_layer.split("_")[1]),
                "width": width, "average_l0_release": l0_name,
                "lobo": round(mean, 3) if mean is not None else None,
                "lobo_med": round(med, 3) if med is not None else None,
                "lobo_per_ds": per_ds, "selected_features": selected}
            print(f"  {ci + 1:02d}/{len(all_cids)} {c}: "
                  f"{med if med is not None else float('nan'):.3f}", flush=True)

        vals = [r["layers"][str(hidden_layer)]["lobo_med"]
                for r in results.values()
                if r["layers"][str(hidden_layer)]["lobo_med"] is not None]
        print(f"[SAE] L{hidden_layer}: mean median-LOBO={np.mean(vals):.3f}",
              flush=True)
        del Z

    def layer_score(layer):
        vals = [r["layers"][str(layer)]["lobo_med"] for r in results.values()
                if r["layers"][str(layer)]["lobo_med"] is not None]
        return float(np.mean(vals))

    best_layer = max(LAYER_CONFIG, key=layer_score)
    for rec in results.values():
        rec["best_layer"] = best_layer
        rec["sae"] = {
            "repo": SCOPE_REPO,
            "revision": SCOPE_REVISION,
            "selection": "single feature with maximum train-fold ROC AUC",
            "heldout_used_for_selection": False}

    out_path = Path(f"sae_results_{TAG}.jsonl")
    with out_path.open("w") as f:
        for c in all_cids:
            f.write(json.dumps(results[c], ensure_ascii=False) + "\n")

    lines = [
        "# Gemma Scope SAE probe — Gemma-2-9B-IT",
        "",
        "Protocol: for each LOBO fold, select the single 16k SAE feature with "
        "the highest training-fold ROC AUC (either orientation), then evaluate "
        "that frozen feature on the held-out benchmark.",
        "",
        "| hidden / Scope layer | release | mean median-LOBO |",
        "|---|---|---|",
    ]
    for hidden_layer, (scope_layer, width, l0_name, _) in LAYER_CONFIG.items():
        lines.append(
            f"| L{hidden_layer} / {scope_layer} | {width}/{l0_name} "
            f"| {layer_score(hidden_layer):.3f} |")
    best_vals = [r["layers"][str(best_layer)]["lobo_med"]
                 for r in results.values()
                 if r["layers"][str(best_layer)]["lobo_med"] is not None]
    lines += [
        "",
        f"Best SAE layer: **L{best_layer}** (median {np.median(best_vals):.3f}; "
        f"strong >=0.8: {sum(v >= .8 for v in best_vals)}/{len(best_vals)}).",
    ]
    method_path = Path(f"method_results_{TAG}.jsonl")
    if method_path.exists():
        method_rows = [json.loads(line) for line in method_path.open()]
        lines += [
            "",
            "Same-model comparison (all 94 capability clusters):",
            "",
            "| method | mean median-LOBO | median | strong >=0.8 |",
            "|---|---|---|---|",
            f"| SAE-AUC | {np.mean(best_vals):.3f} "
            f"| {np.median(best_vals):.3f} "
            f"| {sum(v >= .8 for v in best_vals)} |",
        ]
        for key, label in (("diffmean", "DiffMean"), ("lr", "LR"),
                           ("pca", "PCA")):
            vals = [r[key] for r in method_rows if r.get(key) is not None]
            lines.append(f"| {label} | {np.mean(vals):.3f} "
                         f"| {np.median(vals):.3f} "
                         f"| {sum(v >= .8 for v in vals)} |")
    lines += ["",
              f"Gemma Scope revision: `{SCOPE_REVISION}`.",
              f"Result file: `{out_path}`."]
    Path(f"sae_report_{TAG}.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[SAE] best layer L{best_layer}; wrote {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encode", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=8,
                    help="parallel LOBO folds during feature selection")
    args = ap.parse_args()
    if not args.encode and not args.probe:
        ap.error("choose --encode and/or --probe")

    if args.encode:
        layers, H, _, _, _ = load(TAG)
        for hidden_layer, (_, _, _, filename) in LAYER_CONFIG.items():
            if hidden_layer not in layers:
                raise ValueError(f"hidden L{hidden_layer} is unavailable: {layers}")
            weight_path = WEIGHT_DIR / filename
            if not weight_path.exists():
                raise FileNotFoundError(weight_path)
            encode_layer(H[hidden_layer], weight_path,
                         ACT_DIR / f"L{hidden_layer}.npy", args.batch_size)
        del H
    if args.probe:
        run_probe(args.workers)


if __name__ == "__main__":
    main()
