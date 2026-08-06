#!/usr/bin/env python3
"""Build the 12-model best-valid-depth four-method paper table."""

import json
from pathlib import Path

import numpy as np

MODELS = {
    "qwen3-0.6b": [7, 14, 21, 28],
    "qwen3-1.7b": [7, 14, 21, 28],
    "qwen3-4b": [9, 18, 27, 36],
    "qwen3-8b": [9, 18, 27, 36],
    "qwen3-32b": [16, 32, 48, 64],
    "qwen3.5-9b": [8, 16, 24, 32],
    "llama3.1-8b": [8, 16, 24, 32],
    "deepseek-r1-8b": [9, 18, 27, 36],
    "gemma2-9b": [10, 21, 32, 42],
    "gemma4-12b": [12, 24, 36, 48],
    "gemma4-31b": [15, 30, 45, 60],
    "deepseek-v4-flash": [11, 22, 32, 43],
}
DISPLAY = {
    "deepseek-r1-8b": "R1-Distill-Qwen3-8B",
    "gemma2-9b": "Gemma-2-9B-IT",
    "deepseek-v4-flash": "DeepSeek-V4-Flash-Base",
}


def method_score(tag, layer, method):
    rows = [json.loads(x) for x in Path(
        f"method_results_{tag}_L{layer}.jsonl").open()]
    vals = [r[method] for r in rows if r[method] is not None]
    return float(np.mean(vals))


def main():
    records = []
    for tag, layers in MODELS.items():
        rec = {"tag": tag}
        for method in ("diffmean", "pca", "lr"):
            scores = {layer: method_score(tag, layer, method)
                      for layer in layers}
            best = max(layers, key=lambda x: scores[x])
            rec[method] = {"auc": scores[best], "layer": best,
                           "all_layers": scores}
        jlens = [json.loads(x) for x in Path(
            f"jlens_whole_results_{tag}.jsonl").open()]
        jl = int(jlens[0]["best_layer"])
        vals = [r["layers"][str(jl)]["lobo_med"] for r in jlens]
        rec["jlens"] = {"auc": float(np.mean(vals)), "layer": jl}
        records.append(rec)

    Path("four_method_best_depth.json").write_text(
        json.dumps(records, indent=2))
    lines = [
        "# Four-method comparison at each method's best valid depth",
        "",
        "| Model | DiffMean | PCA | LR | J-Lens verbalizer |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in records:
        cell = lambda m: f"{r[m]['auc']:.3f} (L{r[m]['layer']})"
        lines.append(
            f"| {DISPLAY.get(r['tag'], r['tag'])} | {cell('diffmean')} | "
            f"{cell('pca')} | {cell('lr')} | {cell('jlens')} |")
    lines += ["", "Each method independently selects the depth with the "
              "highest mean capability-level median LOBO AUC. J-Lens selects "
              "among the three valid intermediate depths; the other linear "
              "methods select among all four captured depths."]
    Path("four_method_best_depth.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
