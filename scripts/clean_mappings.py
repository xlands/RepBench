#!/usr/bin/env python3
"""Clean benchmark->cluster mappings by cross-model suspect voting.

A (cluster, dataset) mapping is removed when, across the 10 probed models,
the dataset's held-out LOBO AUC was < 0.5 in >= 60% of eligible votes
(eligible = the cluster's median LOBO was >= 0.65 in that model, i.e. the
cluster itself generalizes and this dataset is the outlier), with >= 3
eligible votes. Manual removals cover known whole-cluster mismatches.

Updates hf_matches.jsonl and patches probe_texts.jsonl IN PLACE (clusters
field only — texts untouched so uid<->hidden alignment survives).
Writes cleaning_log.md.
"""

import json
from collections import defaultdict
from pathlib import Path

TAGS = ["qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b", "qwen3-32b",
        "qwen3.5-9b", "llama3.1-8b", "gemma2-9b", "gemma4-12b", "gemma4-31b"]
MANUAL = {("cap071", "Putnam-AXIOM/putnam-axiom-dataset-ICML-2025-522")}


def main():
    votes = defaultdict(lambda: [0, 0])  # (cid, ds) -> [suspect, eligible]
    for t in TAGS:
        p = Path(f"probe_results_{t}.jsonl")
        if not p.exists():
            continue
        for line in p.open():
            r = json.loads(line)
            ly = r["layers"][str(r["best_layer"])] \
                if str(r["best_layer"]) in r["layers"] \
                else r["layers"][r["best_layer"]]
            if ly.get("lobo_med") is None or ly["lobo_med"] < 0.65:
                continue
            for d, a in ly["lobo_per_ds"].items():
                key = (r["cluster_id"], d)
                votes[key][1] += 1
                if a < 0.5:
                    votes[key][0] += 1

    remove = set(MANUAL)
    log = ["# 映射清洗日志（跨 10 模型投票）\n",
           "| cluster | dataset | 可疑票/有效票 | 处置 |", "|---|---|---|---|"]
    for (cid, d), (s, e) in sorted(votes.items()):
        if e >= 3 and s / e >= 0.6:
            remove.add((cid, d))
            log.append(f"| {cid} | {d} | {s}/{e} | 移除 |")
        elif s > 0:
            log.append(f"| {cid} | {d} | {s}/{e} | 保留 |")
    for cid, d in MANUAL:
        log.append(f"| {cid} | {d} | 人工 | 移除（跨领域错配） |")
    removed_by_ds = defaultdict(set)
    for cid, d in remove:
        removed_by_ds[d].add(cid)

    # 1. hf_matches.jsonl: drop cluster tags from affected records
    recs = [json.loads(l) for l in Path("hf_matches.jsonl").open()]
    n_edit = 0
    for r in recs:
        if not r.get("match") or r["match"] not in removed_by_ds:
            continue
        bad = removed_by_ds[r["match"]]
        kept = [c for c in r["clusters"]
                if not any(f"({cid})" in c for cid in bad)]
        if len(kept) != len(r["clusters"]):
            r["clusters"] = kept
            n_edit += 1
            if not kept:
                r["match"] = None
                r["note"] = "removed_by_probe_cleaning"
    with Path("hf_matches.jsonl").open("w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 2. probe_texts.jsonl: patch clusters per dataset (texts untouched)
    texts = [json.loads(l) for l in Path("probe_texts.jsonl").open()]
    n_patch = 0
    for t in texts:
        bad = removed_by_ds.get(t["dataset_id"], set())
        kept = [c for c in t["clusters"] if c not in bad]
        if len(kept) != len(t["clusters"]):
            t["clusters"] = kept
            n_patch += 1
    with Path("probe_texts.jsonl").open("w") as f:
        for t in texts:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    log.append(f"\n共移除 **{len(remove)}** 条映射；hf_matches 改动 {n_edit} 条记录；"
               f"probe_texts 打补丁 {n_patch} 条文本。")
    Path("cleaning_log.md").write_text("\n".join(log), encoding="utf-8")
    print(f"removed {len(remove)} mappings ({n_edit} match records, "
          f"{n_patch} texts patched)")


if __name__ == "__main__":
    main()
