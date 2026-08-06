# Representation Cluster Pipeline

This directory refines task-facing benchmark clusters into representation-facing RepBench latent ability candidates.

## Pipeline

```text
papers
  -> LLM extract benchmark units
  -> cluster task-facing capability mentions
  -> LLM judge representation fit
  -> merge / split / drop
  -> final RepBench latent abilities
```

The first two stages were produced under `tmp_scripts/`:

- `first_round_deepseek_v32_no_thinking/benchmark_units.jsonl`
- `capability_clustering_baseline/clusters.jsonl`
- `capability_clustering_baseline/mentions.jsonl`

This directory owns the representation-facing refinement stage.

## Main Script

```bash
python3 representation_fit_pipeline.py prepare-prompts
python3 representation_fit_pipeline.py run-judge
python3 representation_fit_pipeline.py parse-judge
python3 representation_fit_pipeline.py build-final
```

Or run all stages:

```bash
python3 representation_fit_pipeline.py run-all
```

Set the API credential in the environment before running judge-backed stages:

```bash
export COMPASS_API_KEY=your_api_key
```

Defaults:

- model: `DeepSeek-V3.2`
- thinking: off
- concurrency: 20
- output directory: `results/`
- input cluster directory: `../../tmp_scripts/capability_clustering_baseline`

Use `--limit N` with `run-judge` or `run-all` for a pilot.

## Outputs

All outputs are written to `results/`:

- `representation_fit_prompts.jsonl`: prompts for each task-facing cluster
- `representation_fit_judge_results.jsonl`: raw LLM responses
- `cluster_representation_judgments.jsonl`: parsed keep/split/merge/drop judgments
- `latent_abilities_raw.jsonl`: raw representation-facing abilities emitted by the judge
- `final_repbench_latent_abilities.jsonl`: merged candidate latent abilities
- `final_repbench_latent_abilities.md`: human-readable table
- `representation_fit_summary.json`: parse and decision counts
- `judge_rejects.jsonl`: request or parse failures to retry/debug

## Interpretation

The input clusters are not final abilities. They are task-facing capability groups from benchmark literature. The representation-fit judge applies stricter criteria:

- Keep if the cluster is already an atomic latent operation.
- Split if the cluster is a compound task/workflow.
- Merge if it is equivalent to another latent operation.
- Drop if it is a domain, dataset artifact, benchmark meta-evaluation, or too narrow for RepBench.

Final outputs are still candidates and should be human-reviewed before becoming the RepBench taxonomy.
