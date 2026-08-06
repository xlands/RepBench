#!/usr/bin/env python3
"""Evaluate J-Lens as a label-addressed concept detector under LOBO.

J-Lens is a token-indexed dictionary, not a trained binary classifier. For
each capability, candidate concept tokens are obtained deterministically from
its taxonomy label. Within each LOBO training fold, we choose the candidate
token with the highest training AUC (positive orientation only), freeze it,
and evaluate its J-Lens logit on the held-out benchmark.

The Jacobian lens itself is fitted only on external WikiText-103. It never
sees RepBench evaluation texts while fitting.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer

import jlens
from jlens_models import MODELS, model_spec
from probe_clusters import auc, load, stratified_neg


TAG = None
MODEL = None
LENS_PATH = None
SCORE_DIR = None
LAYER_MAP = None
VERBALIZER_MODE = None
STOPWORDS = {"and", "to", "of", "the", "in", "for", "with", "from", "on"}
WHOLE_PATH = Path("jlens_whole_token_verbalizers.json")


def taxonomy():
    return {
        row["cluster_id"]: row
        for row in (json.loads(line)
                    for line in Path("experiment_clusters.jsonl").open())
    }


def label_token_ids(tokenizer, name):
    words = [w for w in re.split(r"[_\W]+", name.lower())
             if w and w not in STOPWORDS]
    ids = []
    for word in words:
        # Include both standalone and word-in-context SentencePiece forms.
        word_ids = []
        exact_ids = []
        for text in (word, " " + word):
            encoded = tokenizer.encode(text, add_special_tokens=False)
            word_ids.extend(encoded)
            if (len(encoded) == 1
                    and tokenizer.decode(encoded).strip().lower() == word):
                exact_ids.extend(encoded)
        # J-Lens is token-indexed. Prefer an exact whole-word token; only fall
        # back to subword pieces when the vocabulary has no exact token.
        ids.extend(exact_ids if exact_ids else word_ids)
    special = set(tokenizer.all_special_ids)
    return sorted({int(i) for i in ids if int(i) not in special})


def whole_token_ids(tokenizer, name, verbalizers):
    """Return only exact, whitespace-prefixed single-token verbalizers."""
    ids = []
    for word in verbalizers[name]:
        encoded = tokenizer.encode(" " + word, add_special_tokens=False)
        if (len(encoded) == 1
                and tokenizer.decode(encoded).strip().lower() == word.lower()
                and int(encoded[0]) not in tokenizer.all_special_ids):
            ids.append(int(encoded[0]))
    return list(dict.fromkeys(ids))


def load_readout_weights(token_ids):
    index_path = MODEL / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
    else:
        only = MODEL / "model.safetensors"
        with safe_open(only, framework="pt", device="cpu") as f:
            weight_map = {key: only.name for key in f.keys()}
    embed_keys = [k for k in weight_map if k.endswith("embed_tokens.weight")]
    if not embed_keys and "embed.weight" in weight_map:
        embed_keys = ["embed.weight"]
    norm_keys = [k for k in weight_map
                 if k.endswith(".norm.weight") and "layers." not in k
                 and ("language_model.norm.weight" in k
                      or k == "model.norm.weight")]
    head_keys = [k for k in weight_map if k == "lm_head.weight"]
    if not head_keys and "head.weight" in weight_map:
        head_keys = ["head.weight"]
    if not norm_keys and "norm.weight" in weight_map:
        norm_keys = ["norm.weight"]
    if not embed_keys or not norm_keys:
        raise ValueError(f"cannot locate readout weights for {TAG}")
    embed_key, norm_key = embed_keys[0], norm_keys[0]
    unembed_key = head_keys[0] if head_keys else embed_key
    with safe_open(MODEL / weight_map[unembed_key],
                   framework="pt", device="cpu") as f:
        unembed = f.get_tensor(unembed_key)[token_ids].float()
    with safe_open(MODEL / weight_map[norm_key],
                   framework="pt", device="cpu") as f:
        norm_weight = f.get_tensor(norm_key).float()
    cfg = json.loads((MODEL / "config.json").read_text())
    text_cfg = cfg.get("text_config", cfg)
    eps = float(text_cfg.get("rms_norm_eps", 1e-6))
    softcap = text_cfg.get("final_logit_softcapping")
    # Gemma RMSNorm stores a zero-centred scale; other families store the
    # multiplicative scale directly.
    zero_centered = "gemma" in str(text_cfg.get("model_type", cfg.get(
        "model_type", ""))).lower()
    return unembed, norm_weight, eps, softcap, zero_centered


def build_scores(batch_size=2048):
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    lens = jlens.JacobianLens.load(str(LENS_PATH))
    tax = taxonomy()
    verbalizers = None
    if VERBALIZER_MODE == "whole":
        raw = json.loads(WHOLE_PATH.read_text())
        verbalizers = {k: v for k, v in raw.items() if k != "_metadata"}
        candidates = {
            cid: whole_token_ids(
                tokenizer, row["cluster_name"], verbalizers)
            for cid, row in tax.items()
        }
    else:
        candidates = {
            cid: label_token_ids(tokenizer, row["cluster_name"])
            for cid, row in tax.items()
        }
    uncovered = [cid for cid, ids in candidates.items() if not ids]
    if uncovered:
        raise ValueError(
            f"{TAG}: {len(uncovered)} capabilities have no valid "
            f"{VERBALIZER_MODE} candidate: {uncovered}")
    token_ids = sorted({t for ids in candidates.values() for t in ids})
    token_col = {t: i for i, t in enumerate(token_ids)}
    metadata = {
        "lens": str(LENS_PATH),
        "lens_fit_corpus": "Salesforce/wikitext wikitext-103-raw-v1",
        "lens_fit_prompts": lens.n_prompts,
        "candidate_rule": (
            "pre-registered semantically complete whitespace-prefixed exact "
            "single-token verbalizers; no subword fallback"
            if VERBALIZER_MODE == "whole"
            else "taxonomy-label constituent tokens"),
        "verbalizer_mode": VERBALIZER_MODE,
        "token_ids": token_ids,
        "decoded_tokens": {str(t): tokenizer.decode([t]) for t in token_ids},
        "candidates": candidates,
    }
    if verbalizers is not None:
        mapping_bytes = WHOLE_PATH.read_bytes()
        metadata["verbalizer_file"] = str(WHOLE_PATH)
        metadata["verbalizer_sha256"] = hashlib.sha256(
            mapping_bytes).hexdigest()
        metadata["verbalizer_words"] = {
            cid: verbalizers[row["cluster_name"]]
            for cid, row in tax.items()}
        metadata["coverage"] = {
            "capabilities_total": len(candidates),
            "capabilities_covered": sum(bool(x) for x in candidates.values()),
            "valid_candidates_min": min(map(len, candidates.values())),
            "valid_candidates_mean": float(np.mean(
                list(map(len, candidates.values())))),
            "valid_candidates_max": max(map(len, candidates.values())),
        }
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    (SCORE_DIR / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2))

    _, H, _, _, _ = load(TAG)
    unembed, norm_weight, eps, softcap, zero_centered = \
        load_readout_weights(token_ids)
    U = unembed.cuda()
    norm_weight = norm_weight.cuda()

    with torch.inference_mode():
        for hidden_layer, lens_layer in LAYER_MAP.items():
            J = lens.jacobians[lens_layer].cuda()
            out_path = SCORE_DIR / f"L{hidden_layer}.npy"
            S = np.lib.format.open_memmap(
                out_path, mode="w+", dtype=np.float16,
                shape=(len(H[hidden_layer]), len(token_ids)))
            for start in range(0, len(S), batch_size):
                x = torch.from_numpy(H[hidden_layer][start:start + batch_size]).cuda()
                transported = x @ J.T
                transported = transported * torch.rsqrt(
                    transported.float().pow(2).mean(-1, keepdim=True) + eps)
                transported = transported * (
                    1.0 + norm_weight if zero_centered else norm_weight)
                logits = transported @ U.T
                if softcap is not None:
                    logits = softcap * torch.tanh(logits / softcap)
                S[start:start + len(x)] = logits.half().cpu().numpy()
                if start % (10 * batch_size) == 0:
                    print(f"L{hidden_layer}: {start}/{len(S)}", flush=True)
            S.flush()
            print(f"wrote {out_path} {S.shape}", flush=True)
    # Keep this local mapping out of metadata's stringified JSON keys.
    return candidates, token_col


def run_probe():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    meta = json.loads((SCORE_DIR / "metadata.json").read_text())
    fit_prompts = int(meta["lens_fit_prompts"])
    token_ids = [int(t) for t in meta["token_ids"]]
    token_col = {t: i for i, t in enumerate(token_ids)}
    candidates = {cid: [int(t) for t in ids]
                  for cid, ids in meta["candidates"].items()}
    tax = taxonomy()
    _, _, ds, clusters, _ = load(TAG)
    cids = sorted({c for cl in clusters for c in cl})
    cl_idx = {c: np.array([i for i, cl in enumerate(clusters) if c in cl])
              for c in cids}

    # One deterministic negative split per capability, shared by all layers.
    splits = {}
    for c in cids:
        pos = cl_idx[c]
        neg = stratified_neg(c, clusters, ds, 6 * len(pos))
        half = len(neg) // 2
        splits[c] = (neg[:half], neg[half:])

    results = {
        c: {"cluster_id": c, "cluster_name": tax[c]["cluster_name"],
            "n_pos": int(len(cl_idx[c])),
            "n_datasets": int(len(set(ds[cl_idx[c]]))), "layers": {}}
        for c in cids
    }
    for hidden_layer, lens_layer in LAYER_MAP.items():
        S = np.load(SCORE_DIR / f"L{hidden_layer}.npy", mmap_mode="r")
        for c in cids:
            pos = cl_idx[c]
            neg_tr, neg_te = splits[c]
            cols = np.array([token_col[t] for t in candidates[c]])
            folds, per_ds, selected = [], {}, {}
            for hold in sorted(set(ds[pos])):
                tr, te = pos[ds[pos] != hold], pos[ds[pos] == hold]
                if len(tr) < 10 or len(te) < 10:
                    continue
                y = np.r_[np.ones(len(tr)), np.zeros(len(neg_tr))]
                train = np.r_[np.asarray(S[tr][:, cols]),
                              np.asarray(S[neg_tr][:, cols])]
                train_aucs = np.array(
                    [roc_auc_score(y, train[:, j])
                     for j in range(train.shape[1])])
                best = int(np.argmax(train_aucs))
                token_id = candidates[c][best]
                heldout = auc(np.asarray(S[te, cols[best]]),
                              np.asarray(S[neg_te, cols[best]]))
                folds.append(float(heldout))
                per_ds[hold] = round(float(heldout), 3)
                selected[hold] = {
                    "token_id": token_id,
                    "token": tokenizer.decode([token_id]),
                    "train_auc": round(float(train_aucs[best]), 3)}
            results[c]["layers"][str(hidden_layer)] = {
                "lens_layer": lens_layer,
                "lobo": round(float(np.mean(folds)), 3) if folds else None,
                "lobo_med": round(float(np.median(folds)), 3) if folds else None,
                "lobo_per_ds": per_ds,
                "selected_tokens": selected,
                "candidate_tokens": [
                    {"token_id": t, "token": tokenizer.decode([t])}
                    for t in candidates[c]],
            }
        vals = [r["layers"][str(hidden_layer)]["lobo_med"]
                for r in results.values()
                if r["layers"][str(hidden_layer)]["lobo_med"] is not None]
        print(f"L{hidden_layer}: mean median-LOBO {np.mean(vals):.3f}")

    def score(layer):
        vals = [r["layers"][str(layer)]["lobo_med"] for r in results.values()
                if r["layers"][str(layer)]["lobo_med"] is not None]
        return float(np.mean(vals))

    best_layer = max(LAYER_MAP, key=score)
    for row in results.values():
        row["best_layer"] = best_layer
        row["jlens"] = {
            "implementation": "anthropics/jacobian-lens",
            "fit_prompts": fit_prompts,
            "fit_corpus": "Salesforce/wikitext-103-raw-v1",
            "selection": "best positive-orientation taxonomy-label token "
                         "by training-fold AUC",
            "heldout_used_for_selection": False,
            "verbalizer_mode": VERBALIZER_MODE,
        }
    prefix = "jlens_whole_results" if VERBALIZER_MODE == "whole" \
        else "jlens_results"
    out = Path(f"{prefix}_{TAG}.jsonl")
    with out.open("w") as f:
        for c in cids:
            f.write(json.dumps(results[c], ensure_ascii=False) + "\n")

    best_vals = [r["layers"][str(best_layer)]["lobo_med"]
                 for r in results.values()]
    lines = [
        f"# J-Lens {'whole-token verbalizer' if VERBALIZER_MODE == 'whole' else 'label-token'} probe — {TAG}",
        "",
        f"The Jacobian lens was fitted on {fit_prompts:,} external "
        "WikiText-103 sequences "
        "(128 tokens), following the reference implementation. Each LOBO fold "
        "selects the positive-orientation J-Lens token with highest training "
        "AUC from a frozen label-derived candidate set, then evaluates it on the "
        "held-out benchmark.",
        "",
        "| hidden / J-Lens layer | mean median-LOBO |",
        "|---|---|",
    ]
    for hidden_layer, lens_layer in LAYER_MAP.items():
        lines.append(f"| L{hidden_layer} / layer {lens_layer} "
                     f"| {score(hidden_layer):.3f} |")
    lines += [
        "",
        f"Best layer: **L{best_layer}**; mean {np.mean(best_vals):.3f}; "
        f"median {np.median(best_vals):.3f}; "
        f"strong >=0.8: {sum(v >= .8 for v in best_vals)}/{len(best_vals)}.",
        "",
        ("Candidate constraint: only semantically complete, exact single-token "
         "verbalizers are retained; tokenizer fragments and subword fallback "
         "are forbidden."
         if VERBALIZER_MODE == "whole" else
         "Caveat: J-Lens is token-indexed, so multi-token capability names are "
         "represented by their constituent tokenizer pieces; this is a known "
         "limitation of J-Lens rather than the template lens."),
        f"Result file: `{out}`.",
    ]
    report_prefix = "jlens_whole_report" if VERBALIZER_MODE == "whole" \
        else "jlens_report"
    Path(f"{report_prefix}_{TAG}.md").write_text("\n".join(lines))
    print(f"best L{best_layer}; wrote {out}")


def main():
    global TAG, MODEL, LENS_PATH, SCORE_DIR, LAYER_MAP, VERBALIZER_MODE
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="gemma2-9b", choices=MODELS)
    ap.add_argument("--verbalizer", choices=["fragment", "whole"],
                    default="fragment")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--batch-size", type=int, default=2048)
    args = ap.parse_args()
    TAG = args.tag
    VERBALIZER_MODE = args.verbalizer
    model_path, _, LAYER_MAP = model_spec(TAG)
    MODEL = Path(model_path)
    LENS_PATH = Path("jlens_weights") / TAG / "wikitext_n1000.lens.pt"
    # Backward compatibility with the already completed Gemma-2 run.
    if TAG == "gemma2-9b" and not LENS_PATH.exists():
        LENS_PATH = Path(
            "jlens_weights/gemma2-9b-it/wikitext_n1000.lens.pt")
    SCORE_DIR = Path(
        "jlens_scores_whole" if VERBALIZER_MODE == "whole"
        else "jlens_scores") / TAG
    if not args.score and not args.probe:
        ap.error("choose --score and/or --probe")
    if args.score:
        build_scores(args.batch_size)
    if args.probe:
        run_probe()


if __name__ == "__main__":
    main()
