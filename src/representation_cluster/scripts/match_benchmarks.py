#!/usr/bin/env python3
"""Crawl step 1: collect target benchmarks and match them to HF datasets.

For every kept cluster in ../capability_clustering_v2/selection_matrix.jsonl,
take its top scorer-available benchmarks, dedupe globally, search the
HuggingFace datasets API for candidates, and let the local LLM pick the
correct match (benchmark papers and HF ids rarely share exact strings).

Outputs (in this directory):
  benchmarks_to_crawl.jsonl - benchmark -> clusters, mentions, paper titles
  hf_matches.jsonl          - benchmark -> matched HF dataset id (or none)
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "capability_clustering_v2"))
from llm_refine import LLM  # noqa: E402

V2 = Path(__file__).resolve().parent.parent / "capability_clustering_v2"
TOP_PER_CLUSTER = 6

MATCH_PROMPT = """A benchmark named "{bench}" is cited in LLM-evaluation papers such as:
{titles}
It is used to test the capability cluster(s): {clusters}.

Below are HuggingFace dataset search results (id | downloads | likes | description):
{candidates}

Which dataset id IS this benchmark (the actual eval data, not a derivative like
model outputs, embeddings, train-only rewrites, or a different benchmark with a
similar name)? Prefer the official/canonical upload; popular community mirrors
(e.g. openai/gsm8k, cais/mmlu) count as canonical.
Return ONLY JSON: {{"match": "<dataset_id or none>", "confidence": "high|medium|low"}}"""


def collect():
    rows = [json.loads(l) for l in (V2 / "selection_matrix.jsonl").open()]
    kept = {r["cluster_id"]: r for r in rows if not r["excluded"]}

    # paper titles per benchmark for match context
    titles = {}
    for line in (V2 / "mentions_final.jsonl").open():
        m = json.loads(line)
        if m["final_cluster_id"] in kept:
            t = titles.setdefault(m["benchmark_name"], set())
            if len(t) < 3:
                t.add((m.get("paper_title") or "")[:120])

    agg = {}
    for cid, r in kept.items():
        for b, n in r["top_scorer_benchmarks"][:TOP_PER_CLUSTER]:
            a = agg.setdefault(b, {"benchmark": b, "mentions": 0, "clusters": []})
            a["mentions"] += n
            a["clusters"].append(f"{r['cluster_name']} ({cid})")
    for a in agg.values():
        a["paper_titles"] = sorted(titles.get(a["benchmark"], []))

    out = sorted(agg.values(), key=lambda a: -a["mentions"])
    with Path("benchmarks_to_crawl.jsonl").open("w") as f:
        for a in out:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    print(f"{len(out)} unique benchmarks to match")
    return out


def hf_search(query, limit=6):
    r = requests.get("https://huggingface.co/api/datasets",
                     params={"search": query, "limit": limit, "full": "false"},
                     timeout=20)
    r.raise_for_status()
    out = []
    for d in r.json():
        out.append({"id": d["id"], "downloads": d.get("downloads", 0),
                    "likes": d.get("likes", 0),
                    "desc": (d.get("description") or "")[:150]})
    return out


def main():
    targets = collect()
    llm = LLM("http://localhost:30200")
    done = {}
    out_path = Path("hf_matches.jsonl")
    if out_path.exists():  # resumable
        for line in out_path.open():
            r = json.loads(line)
            done[r["benchmark"]] = r
    fout = out_path.open("a")

    def work(t):
        b = t["benchmark"]
        if b in done:
            return None
        cands, err = [], ""
        for q in dict.fromkeys([b, b.replace(" ", "-"), b.split(" (")[0]]):
            try:
                for c in hf_search(q):
                    if c["id"] not in {x["id"] for x in cands}:
                        cands.append(c)
            except Exception as e:  # noqa: BLE001
                err = str(e)
                time.sleep(1)
        rec = {"benchmark": b, "mentions": t["mentions"], "clusters": t["clusters"],
               "candidates": cands[:10], "match": None, "confidence": None, "error": err}
        if cands:
            try:
                v = llm.chat_json(MATCH_PROMPT.format(
                    bench=b, titles="\n".join(t["paper_titles"]) or "(none)",
                    clusters="; ".join(t["clusters"][:3]),
                    candidates="\n".join(
                        f"{c['id']} | {c['downloads']} | {c['likes']} | {c['desc']}"
                        for c in cands[:10])))
                m = v.get("match")
                if m and m.lower() != "none" and m in {c["id"] for c in cands}:
                    rec["match"] = m
                    rec["confidence"] = v.get("confidence")
            except Exception as e:  # noqa: BLE001
                rec["error"] = str(e)
        return rec

    n_matched = 0
    with ThreadPoolExecutor(8) as ex:
        for rec in ex.map(work, targets):
            if rec is None:
                continue
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            n_matched += bool(rec["match"])
    fout.close()
    total = len(targets) - len(done)
    print(f"matched {n_matched}/{total} new (plus {len(done)} cached)")


if __name__ == "__main__":
    main()
