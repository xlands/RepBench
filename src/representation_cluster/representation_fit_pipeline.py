#!/usr/bin/env python3
"""Representation-facing capability refinement pipeline.

Pipeline stage covered by this script:

  clustered task-facing capability mentions
    -> LLM judge representation fit
    -> merge / split / drop
    -> final RepBench latent ability candidates

The script is resumable. It writes all outputs under ./results by default.
"""

import argparse
import concurrent.futures
import json
import os
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLUSTER_DIR = PROJECT_ROOT / "tmp_scripts" / "capability_clustering_baseline"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "results"
COMPASS_BASE_URL = "https://compass.llm.shopee.io/compass-api/v1"
DEFAULT_MODEL = "DeepSeek-V3.2"

_THREAD_LOCAL = threading.local()


SYSTEM_GUIDANCE = """You are designing RepBench, a latent representation benchmark for language models and agents.
Your job is not to preserve benchmark task labels. Your job is to decide which task-facing clusters can be turned into representation-facing latent abilities.

A representation-facing ability must satisfy most of these:
1. It admits a clear positive/negative contrast pair for probing.
2. It is not merely a domain such as medical, legal, finance, science, or coding.
3. It is not an entire task or product workflow; split workflows into latent operations.
4. It is likely to correspond to a readable hidden-state direction/subspace or a steerable control variable.
5. It can transfer across multiple benchmarks or task surfaces.
6. It can be evaluated through an output scorer after steering, or at least has a plausible behavioral proxy.

Use hard judgment:
- DROP pure domains, benchmark-quality/meta-evaluation clusters, and narrow dataset artifacts.
- SPLIT compound clusters into 2-5 representation-facing latent abilities.
- MERGE by suggesting canonical ability names if several clusters clearly describe the same latent operation.
- KEEP only if the cluster already describes a sufficiently atomic latent operation.

Return strict JSON only. Do not use markdown.
"""

SCHEMA = {
    "cluster_id": "family__cNNN",
    "family": "original family",
    "task_facing_cluster_summary": "one sentence summary of what this cluster of benchmarks measures",
    "representation_fit": "keep | split | merge | drop",
    "fit_score": "integer 1-5 where 5 is very suitable for RepBench",
    "fit_rationale": "brief reason",
    "drop_reason": "null unless representation_fit=drop",
    "merge_targets_suggested": ["snake_case names of equivalent abilities from other clusters, if obvious"],
    "latent_abilities": [
        {
            "ability_name": "snake_case canonical representation-facing ability name",
            "display_name": "Human-readable ability name",
            "representation_family": "factuality_grounding | uncertainty_metacognition | reasoning | math_symbolic | planning_tool_use | memory_state_tracking | instruction_policy_following | safety_robustness | social_pragmatic | multimodal_grounding | coding_debugging | other",
            "status": "keep | split_from_cluster | merge_candidate",
            "definition": "one-sentence operational definition of the latent ability",
            "positive_state": "what a positive hidden-state example represents",
            "negative_state": "what a negative hidden-state example represents",
            "probe_hypothesis": "what should be linearly readable, from which kind of examples",
            "steering_hypothesis": "what behavior should change if this direction is steered",
            "contrast_pair_feasibility": "yes | unclear | no",
            "behavioral_scorer_feasibility": "yes | unclear | no",
            "generality": "cross_benchmark | family_specific | narrow",
            "source_benchmark_examples": ["benchmark names from evidence"],
            "source_task_examples": ["task names from evidence"],
            "notes": "important caveats"
        }
    ]
}


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def compact(value, limit=1200):
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def top_list(items, n=10):
    return [{"name": name, "count": count} for name, count in (items or [])[:n]]


def load_cluster_inputs(cluster_dir):
    cluster_dir = Path(cluster_dir)
    clusters = read_jsonl(cluster_dir / "clusters.jsonl")
    mentions = read_jsonl(cluster_dir / "mentions.jsonl")
    mentions_by_cluster = defaultdict(list)
    for mention in mentions:
        mentions_by_cluster[mention["cluster_id"]].append(mention)
    return clusters, mentions_by_cluster


def representative_mentions(mentions, max_mentions):
    out = []
    for m in mentions[:max_mentions]:
        out.append(
            {
                "paper_title": m.get("paper_title"),
                "benchmark_name": m.get("benchmark_name"),
                "task_name": m.get("task_name"),
                "capability_name": m.get("canonical_name"),
                "definition": compact(m.get("definition"), 450),
                "cognitive_operation": compact(m.get("cognitive_operation"), 350),
                "surface_task": compact(m.get("surface_task"), 300),
                "evaluated_behavior": compact(m.get("evaluated_behavior"), 300),
                "agentic_dimensions": m.get("agentic_dimensions") or [],
                "domain_metadata": m.get("domain_metadata") or [],
                "confidence": m.get("confidence"),
            }
        )
    return out


def build_cluster_payload(cluster, mentions, max_mentions):
    return {
        "cluster_id": cluster.get("cluster_id"),
        "family": cluster.get("family"),
        "size": cluster.get("size"),
        "top_capability_names": top_list(cluster.get("top_capability_names"), 12),
        "top_operation_terms": top_list(cluster.get("top_operation_terms"), 16),
        "top_agentic_dimensions": top_list(cluster.get("top_agentic_dimensions"), 10),
        "top_domains": top_list(cluster.get("top_domains"), 10),
        "top_target_model_types": top_list(cluster.get("top_target_model_types"), 10),
        "top_benchmarks": top_list(cluster.get("top_benchmarks"), 12),
        "top_tasks": top_list(cluster.get("top_tasks"), 12),
        "representative_mentions": representative_mentions(mentions, max_mentions),
    }


def build_prompt(cluster_payload):
    return "\n\n".join(
        [
            SYSTEM_GUIDANCE,
            "Output JSON schema:",
            json.dumps(SCHEMA, ensure_ascii=False, indent=2),
            "Now judge this task-facing benchmark cluster for representation-facing RepBench abilities:",
            json.dumps(cluster_payload, ensure_ascii=False, indent=2),
        ]
    )


def cmd_prepare_prompts(args):
    clusters, mentions_by_cluster = load_cluster_inputs(args.cluster_dir)
    rows = []
    for cluster in clusters:
        cluster_id = cluster["cluster_id"]
        payload = build_cluster_payload(cluster, mentions_by_cluster.get(cluster_id, []), args.max_mentions)
        rows.append(
            {
                "cluster_id": cluster_id,
                "family": cluster.get("family"),
                "size": cluster.get("size"),
                "prompt": build_prompt(payload),
                "cluster_payload": payload,
            }
        )
    out = Path(args.out_dir) / "representation_fit_prompts.jsonl"
    write_jsonl(out, rows)
    print(f"prompts={len(rows)} out={out}")


def get_client(api_key, timeout, max_retries):
    client = getattr(_THREAD_LOCAL, "client", None)
    if client is None:
        client = OpenAI(api_key=api_key, base_url=COMPASS_BASE_URL, timeout=timeout, max_retries=max_retries)
        _THREAD_LOCAL.client = client
    return client


def completed_ids(path):
    ids = set()
    if not Path(path).exists():
        return ids
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("cluster_id"):
                ids.add(row["cluster_id"])
    return ids


def run_one(row, args):
    started = time.time()
    try:
        client = get_client(args.api_key, args.timeout, args.max_retries)
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": row["prompt"]}],
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            extra_body={"chat_template_kwargs": {"thinking": args.thinking}},
        )
        message = completion.choices[0].message
        usage = None
        if getattr(completion, "usage", None) is not None:
            try:
                usage = completion.usage.model_dump(mode="json", exclude_none=True)
            except AttributeError:
                usage = str(completion.usage)
        return {
            "cluster_id": row["cluster_id"],
            "family": row.get("family"),
            "size": row.get("size"),
            "status": "ok",
            "elapsed_sec": round(time.time() - started, 3),
            "model": args.model,
            "thinking": args.thinking,
            "response_text": message.content or "",
            "reasoning_content": getattr(message, "reasoning_content", None),
            "usage": usage,
        }
    except Exception as exc:
        return {
            "cluster_id": row["cluster_id"],
            "family": row.get("family"),
            "size": row.get("size"),
            "status": "error",
            "elapsed_sec": round(time.time() - started, 3),
            "error": str(exc),
        }


def cmd_run_judge(args):
    if not args.api_key:
        raise SystemExit(
            "COMPASS_API_KEY is required for judge-backed stages; "
            "set it in the environment or pass --api-key."
        )
    prompt_path = Path(args.out_dir) / "representation_fit_prompts.jsonl"
    if not prompt_path.exists():
        raise SystemExit(f"Missing prompts: {prompt_path}. Run prepare-prompts first.")
    rows = read_jsonl(prompt_path)
    if args.limit:
        rows = rows[: args.limit]
    out = Path(args.out_dir) / "representation_fit_judge_results.jsonl"
    done = completed_ids(out) if args.resume else set()
    pending = [r for r in rows if r["cluster_id"] not in done]
    print(f"total={len(rows)} completed={len(done)} pending={len(pending)} out={out}")
    if args.dry_run:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    count = 0
    with out.open("a", encoding="utf-8") as f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            future_to_row = {executor.submit(run_one, row, args): row for row in pending}
            for future in concurrent.futures.as_completed(future_to_row):
                result = future.result()
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                count += 1
                if count == 1 or count % args.log_every == 0:
                    rate = count / max(time.time() - start, 1e-6)
                    print(f"[{count}/{len(pending)}] {result.get('status')} {result.get('cluster_id')} rate={rate:.2f}/s")


def parse_json_text(text):
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def cmd_parse_judge(args):
    results_path = Path(args.out_dir) / "representation_fit_judge_results.jsonl"
    judgments_path = Path(args.out_dir) / "cluster_representation_judgments.jsonl"
    abilities_path = Path(args.out_dir) / "latent_abilities_raw.jsonl"
    rejects_path = Path(args.out_dir) / "judge_rejects.jsonl"
    counts = Counter()
    abilities = []
    judgments = []
    rejects = []
    with results_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            result = json.loads(line)
            counts["results"] += 1
            if result.get("status") != "ok":
                counts["request_errors"] += 1
                rejects.append(result)
                continue
            try:
                judgment = parse_json_text(result.get("response_text", ""))
            except Exception as exc:
                counts["parse_errors"] += 1
                rejects.append({"line_no": line_no, "cluster_id": result.get("cluster_id"), "error": str(exc), "text": result.get("response_text", "")[:2000]})
                continue
            counts["parsed"] += 1
            judgment.setdefault("cluster_id", result.get("cluster_id"))
            judgment.setdefault("family", result.get("family"))
            judgment["_source_size"] = result.get("size")
            judgments.append(judgment)
            counts[f"fit:{judgment.get('representation_fit', 'missing')}"] += 1
            for idx, ability in enumerate(judgment.get("latent_abilities") or []):
                row = dict(ability)
                row["source_cluster_id"] = judgment.get("cluster_id")
                row["source_family"] = judgment.get("family")
                row["cluster_fit"] = judgment.get("representation_fit")
                row["cluster_fit_score"] = judgment.get("fit_score")
                row["ability_index"] = idx
                abilities.append(row)
    write_jsonl(judgments_path, judgments)
    write_jsonl(abilities_path, abilities)
    write_jsonl(rejects_path, rejects)
    summary = dict(counts)
    summary["raw_abilities"] = len(abilities)
    (Path(args.out_dir) / "representation_fit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"judgments={judgments_path}")
    print(f"abilities={abilities_path}")
    print(f"rejects={rejects_path}")


def norm_name(name):
    name = re.sub(r"[^a-zA-Z0-9]+", "_", str(name or "").lower()).strip("_")
    return re.sub(r"_+", "_", name) or "unnamed_ability"


def cmd_build_final(args):
    raw_path = Path(args.out_dir) / "latent_abilities_raw.jsonl"
    if not raw_path.exists():
        raise SystemExit(f"Missing raw abilities: {raw_path}. Run parse-judge first.")
    rows = read_jsonl(raw_path)
    grouped = defaultdict(list)
    for row in rows:
        name = norm_name(row.get("ability_name"))
        if row.get("contrast_pair_feasibility") == "no" and not args.keep_no_contrast:
            continue
        if row.get("generality") == "narrow" and not args.keep_narrow:
            continue
        grouped[name].append(row)
    final = []
    for name, members in grouped.items():
        families = Counter(m.get("representation_family") or m.get("source_family") for m in members)
        display = Counter(m.get("display_name") or name.replace("_", " ").title() for m in members).most_common(1)[0][0]
        definitions = [m.get("definition") for m in members if m.get("definition")]
        positive = [m.get("positive_state") for m in members if m.get("positive_state")]
        negative = [m.get("negative_state") for m in members if m.get("negative_state")]
        probe = [m.get("probe_hypothesis") for m in members if m.get("probe_hypothesis")]
        steering = [m.get("steering_hypothesis") for m in members if m.get("steering_hypothesis")]
        benchmarks = []
        tasks = []
        source_clusters = []
        for m in members:
            benchmarks.extend(m.get("source_benchmark_examples") or [])
            tasks.extend(m.get("source_task_examples") or [])
            source_clusters.append(m.get("source_cluster_id"))
        final.append(
            {
                "ability_name": name,
                "display_name": display,
                "representation_family": families.most_common(1)[0][0] if families else "missing",
                "source_cluster_count": len(set(source_clusters)),
                "source_clusters": sorted(set(x for x in source_clusters if x)),
                "mention_count": len(members),
                "definition_candidates": Counter(definitions).most_common(3),
                "positive_state_candidates": Counter(positive).most_common(3),
                "negative_state_candidates": Counter(negative).most_common(3),
                "probe_hypothesis_candidates": Counter(probe).most_common(3),
                "steering_hypothesis_candidates": Counter(steering).most_common(3),
                "top_benchmark_examples": Counter(benchmarks).most_common(12),
                "top_task_examples": Counter(tasks).most_common(12),
                "generalities": Counter(m.get("generality") for m in members).most_common(),
                "contrast_pair_feasibility": Counter(m.get("contrast_pair_feasibility") for m in members).most_common(),
                "behavioral_scorer_feasibility": Counter(m.get("behavioral_scorer_feasibility") for m in members).most_common(),
            }
        )
    final.sort(key=lambda x: (x["representation_family"], -x["source_cluster_count"], x["display_name"]))
    out = Path(args.out_dir) / "final_repbench_latent_abilities.jsonl"
    write_jsonl(out, final)
    md = Path(args.out_dir) / "final_repbench_latent_abilities.md"
    with md.open("w", encoding="utf-8") as f:
        f.write("# Final RepBench Latent Ability Candidates\n\n")
        f.write("These are merged candidates from LLM representation-fit judgments. They still need human review.\n\n")
        f.write("| family | ability | source clusters | top benchmarks |\n|---|---|---:|---|\n")
        for a in final:
            b = ", ".join(name for name, _ in a["top_benchmark_examples"][:4])
            f.write(f"| {a['representation_family']} | {a['display_name']} | {a['source_cluster_count']} | {b} |\n")
    print(f"final_abilities={len(final)} out={out}")
    print(f"markdown={md}")


def cmd_run_all(args):
    cmd_prepare_prompts(args)
    cmd_run_judge(args)
    cmd_parse_judge(args)
    cmd_build_final(args)


def add_common(parser):
    parser.add_argument("--cluster-dir", default=str(DEFAULT_CLUSTER_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--max-mentions", type=int, default=10)


def add_api(parser):
    parser.add_argument("--api-key", default=os.environ.get("COMPASS_API_KEY"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-every", type=int, default=20)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare-prompts")
    add_common(p)
    p.set_defaults(func=cmd_prepare_prompts)
    p = sub.add_parser("run-judge")
    add_common(p)
    add_api(p)
    p.set_defaults(func=cmd_run_judge)
    p = sub.add_parser("parse-judge")
    add_common(p)
    p.set_defaults(func=cmd_parse_judge)
    p = sub.add_parser("build-final")
    add_common(p)
    p.add_argument("--keep-no-contrast", action="store_true")
    p.add_argument("--keep-narrow", action="store_true")
    p.set_defaults(func=cmd_build_final)
    p = sub.add_parser("run-all")
    add_common(p)
    add_api(p)
    p.add_argument("--keep-no-contrast", action="store_true")
    p.add_argument("--keep-narrow", action="store_true")
    p.set_defaults(func=cmd_run_all)
    return parser.parse_args()


def main():
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
