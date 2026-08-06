#!/usr/bin/env python3
"""Crawl step 2: download rows for every matched HF dataset.

Uses the datasets-server API (no auth, no full downloads):
  /splits?dataset=ID          -> configs & splits
  /rows?dataset&config&split  -> 100-row pages

Per dataset: prefer test > validation > dev > train split; prefer an
"all"/"default"/"main" config, else the largest, else the first. Caps at
--max-rows rows. Falls back to /first-rows for datasets where /rows is
disabled. Writes data/<sanitized_id>.jsonl and a resumable manifest.jsonl.
"""

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

API = "https://datasets-server.huggingface.co"
SPLIT_PREF = ["test", "testing", "validation", "valid", "val", "dev", "eval", "train"]


def get(path, params, retries=3):
    last = None
    for i in range(retries):
        try:
            r = requests.get(API + path, params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}: {r.text[:150]}"
            if r.status_code in (400, 401, 404, 501):
                break  # permanent
        except Exception as e:  # noqa: BLE001
            last = str(e)
        time.sleep(2 * (i + 1))
    raise RuntimeError(last)


def pick_split(splits):
    """splits: list of {config, split}. Return (config, split)."""
    configs = {}
    for s in splits:
        configs.setdefault(s["config"], []).append(s["split"])
    cfg_names = list(configs)
    cfg = next((c for c in cfg_names if c.lower() in ("all", "default", "main")), None)
    if cfg is None:
        cfg = cfg_names[0]
    avail = configs[cfg]
    for pref in SPLIT_PREF:
        hit = next((s for s in avail if s.lower() == pref), None)
        if hit:
            return cfg, hit
    return cfg, avail[0]


def crawl_one(ds_id, out_dir, max_rows):
    info = get("/splits", {"dataset": ds_id})
    splits = info.get("splits", [])
    if not splits:
        raise RuntimeError("no splits")
    config, split = pick_split(splits)
    rows, offset = [], 0
    while len(rows) < max_rows:
        length = min(100, max_rows - len(rows))
        try:
            page = get("/rows", {"dataset": ds_id, "config": config,
                                 "split": split, "offset": offset, "length": length})
        except RuntimeError as e:
            if offset == 0:  # /rows disabled -> first-rows fallback
                page = get("/first-rows", {"dataset": ds_id, "config": config,
                                           "split": split})
                rows = [r["row"] for r in page.get("rows", [])][:max_rows]
                return config, split, rows, "first_rows_only"
            raise e
        batch = [r["row"] for r in page.get("rows", [])]
        rows.extend(batch)
        offset += len(batch)
        if len(batch) < length:
            break
    fname = re.sub(r"[^A-Za-z0-9_.-]", "__", ds_id) + ".jsonl"
    with (out_dir / fname).open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return config, split, rows, fname


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rows", type=int, default=500)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    recs = [json.loads(l) for l in Path("hf_matches.jsonl").open()]
    by_id = {}
    for r in recs:
        if r["match"]:
            by_id.setdefault(r["match"], []).append(r)

    manifest_path = Path("manifest.jsonl")
    done = set()
    if manifest_path.exists():
        for line in manifest_path.open():
            done.add(json.loads(line)["dataset_id"])
    todo = [i for i in by_id if i not in done]
    print(f"{len(by_id)} datasets, {len(done)} done, {len(todo)} to crawl")
    mf = manifest_path.open("a")

    def work(ds_id):
        entry = {
            "dataset_id": ds_id,
            "benchmarks": [r["benchmark"] for r in by_id[ds_id]],
            "clusters": sorted({c for r in by_id[ds_id] for c in r["clusters"]}),
        }
        try:
            config, split, rows, fname = crawl_one(ds_id, out_dir, args.max_rows)
            entry.update(status="ok", config=config, split=split,
                         n_rows=len(rows), file=fname if isinstance(fname, str) else None,
                         columns=sorted(rows[0].keys()) if rows else [])
            if fname == "first_rows_only":
                f2 = re.sub(r"[^A-Za-z0-9_.-]", "__", ds_id) + ".jsonl"
                with (out_dir / f2).open("w") as f:
                    for r in rows:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                entry.update(file=f2, note="first_rows_only")
        except Exception as e:  # noqa: BLE001
            entry.update(status="error", error=str(e)[:300])
        return entry

    ok = err = 0
    with ThreadPoolExecutor(args.workers) as ex:
        for entry in ex.map(work, todo):
            mf.write(json.dumps(entry, ensure_ascii=False) + "\n")
            mf.flush()
            ok += entry["status"] == "ok"
            err += entry["status"] == "error"
            if (ok + err) % 25 == 0:
                print(f"  progress {ok + err}/{len(todo)} (ok {ok}, err {err})")
    mf.close()
    print(f"done: ok {ok}, error {err}")


if __name__ == "__main__":
    main()
