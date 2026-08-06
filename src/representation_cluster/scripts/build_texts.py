#!/usr/bin/env python3
"""Probe step 1: assemble per-cluster task texts from crawled rows.

Heuristic field mapping over 374 heterogeneous schemas: prefer known
task-text fields (question/prompt/instruction/...), attach context and
options when present, fall back to the longest string field. Datasets
whose assembled texts stay too short are dropped and logged.

Outputs:
  probe_texts.jsonl    - {uid, dataset_id, clusters, text}
  text_mapping_log.md  - per-dataset chosen fields + kept/dropped
"""

import hashlib
import json
import random
from pathlib import Path

MAX_PER_DS = 150
MAX_CHARS = 4000
MIN_CHARS = 30

# ordered by how likely the field is THE task statement
PRIMARY = ["question", "prompt", "instruction", "problem", "query", "src",
           "statement", "user_message", "input", "sentence", "text", "claim",
           "premise", "anchor_post", "post", "original", "code", "story",
           "dialogue", "conversation", "title", "content"]
CONTEXT = ["context", "paragraph", "passage", "background", "document", "article"]
OPTIONS = ["options", "choices", "answer_candidates", "candidates"]


def as_text(v, depth=0):
    if isinstance(v, str):
        return v
    if isinstance(v, (int, float, bool)) or v is None:
        return ""
    if isinstance(v, list):
        parts = [as_text(x, depth + 1) for x in v[:8]]
        return "\n".join(p for p in parts if p)
    if isinstance(v, dict) and depth < 2:
        if set(v) & {"role", "content"}:  # chat turn
            return f"{v.get('role', '')}: {as_text(v.get('content', ''), depth + 1)}"
        return "\n".join(f"{k}: {as_text(x, depth + 1)}"
                         for k, x in list(v.items())[:8] if as_text(x, depth + 1))
    return ""


def pick_fields(row):
    """Return (primary_key, context_key, options_key) by name heuristic."""
    low = {k.lower(): k for k in row}
    prim = next((low[f] for f in PRIMARY if f in low), None)
    ctx = next((low[f] for f in CONTEXT if f in low and low[f] != prim), None)
    opt = next((low[f] for f in OPTIONS if f in low), None)
    return prim, ctx, opt


def longest_str_field(row):
    best, best_len = None, 0
    for k, v in row.items():
        t = as_text(v)
        if len(t) > best_len:
            best, best_len = k, len(t)
    return best


def assemble(row, prim, ctx, opt):
    parts = []
    if ctx:
        t = as_text(row.get(ctx)).strip()
        if t:
            parts.append(t[:2000])
    if prim:
        t = as_text(row.get(prim)).strip()
        if t:
            parts.append(t)
    if opt:
        t = as_text(row.get(opt)).strip()
        if t:
            parts.append("Options:\n" + t[:600])
    return "\n\n".join(parts)[:MAX_CHARS]


def main():
    random.seed(0)
    clusters = [json.loads(l) for l in Path("experiment_clusters.jsonl").open()]
    # dataset file -> set of cluster ids (a dataset may serve several)
    ds_clusters, ds_file = {}, {}
    for c in clusters:
        for d in c["datasets"]:
            if d["status"] == "ok" and d["file"]:
                ds_clusters.setdefault(d["hf_dataset"], set()).add(c["cluster_id"])
                ds_file[d["hf_dataset"]] = d["file"]

    out, log = [], ["# 文本映射日志\n", "| dataset | 字段 | 保留行 | 中位长 | 状态 |",
                    "|---|---|---|---|---|"]
    n_drop = 0
    for ds_id, fname in sorted(ds_file.items()):
        p = Path("data") / fname
        if not p.exists():
            continue
        rows = [json.loads(l) for l in p.open()]
        if not rows:
            continue
        prim, ctx, opt = pick_fields(rows[0])
        if prim is None:
            prim = longest_str_field(rows[0])

        def collect(sec=None):
            texts, seen = [], set()
            for r in rows:
                t = assemble(r, prim, ctx, opt).strip()
                if sec:
                    t2 = as_text(r.get(sec)).strip()
                    if t2:
                        t = (t + "\n\n" + t2[:1500]).strip()
                if len(t) < MIN_CHARS:
                    continue
                h = hashlib.md5(t.encode()).hexdigest()
                if h not in seen:
                    seen.add(h)
                    texts.append(t)
            return texts

        texts, sec = collect(), None
        if len(texts) < 20:
            # primary field repeats across rows (one premise, many hypotheses):
            # append the string field with the highest row-wise cardinality
            cands = {}
            for k in rows[0]:
                if k in (prim, ctx, opt):
                    continue
                vals = {as_text(r.get(k))[:500] for r in rows}
                if sorted(map(len, vals))[len(vals) // 2] >= 15:
                    cands[k] = len(vals)
            if cands:
                sec = max(cands, key=cands.get)
                texts = collect(sec)
        if len(texts) > MAX_PER_DS:
            texts = random.sample(texts, MAX_PER_DS)
        med = sorted(map(len, texts))[len(texts) // 2] if texts else 0
        status = "ok" if len(texts) >= 20 and med >= 40 else "dropped"
        fields = "+".join(x for x in (ctx, prim, opt, sec) if x)
        log.append(f"| {ds_id} | {fields} | {len(texts)} | {med} | {status} |")
        if status == "dropped":
            n_drop += 1
            continue
        for i, t in enumerate(texts):
            out.append({"uid": f"{ds_id}::{i}", "dataset_id": ds_id,
                        "clusters": sorted(ds_clusters[ds_id]), "text": t})

    with Path("probe_texts.jsonl").open("w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    Path("text_mapping_log.md").write_text("\n".join(log), encoding="utf-8")

    from collections import Counter
    cc = Counter(c for r in out for c in r["clusters"])
    print(f"{len(out)} texts from {len(ds_file) - n_drop}/{len(ds_file)} datasets "
          f"({n_drop} dropped)")
    print(f"clusters covered: {len(cc)}/94; min texts per cluster: "
          f"{min(cc.values()) if cc else 0}")
    weak = [c for c, n in cc.items() if n < 60]
    print(f"clusters with <60 texts: {len(weak)} {sorted(weak)[:10]}")


if __name__ == "__main__":
    main()
