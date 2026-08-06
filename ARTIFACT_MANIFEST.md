# Artifact manifest

| Claim or table | Supporting files |
|---|---|
| Corpus size and 94-capability coverage | `data/probe_texts.jsonl`, `data/experiment_clusters.jsonl`, `data/manifest.jsonl` |
| Diff-mean/PCA/LR depth sweep | `results/common_grid/method_results_*.jsonl` |
| Four-readout main table | `results/four_method_best_depth.json` |
| Checkpoint-aligned SAE result | `results/sae/sae_results_gemma2-9b.jsonl` |
| Whole-token J-Lens result | `results/jlens/jlens_whole_results_*.jsonl` |
| Frozen semantic verbalizers | `results/jlens/jlens_whole_token_verbalizers.json` |
| Corpus construction | `scripts/match_benchmarks.py`, `scripts/crawl_rows.py`, `scripts/clean_mappings.py`, `scripts/build_texts.py` |
| Hidden-state extraction | `scripts/extract_hidden.py` |
| Diff-mean/PCA/LR evaluation | `scripts/method_compare.py`, `scripts/probe_clusters.py` |
| SAE evaluation | `scripts/sae_probe.py` |
| J-Lens fitting and evaluation | `scripts/fit_jlens.py`, `scripts/jlens_probe.py`, `scripts/jlens_models.py` |
| Table aggregation | `scripts/summarize_four_method_best_depth.py` |

Model weights, fitted Jacobian matrices, and raw hidden-state tensors are
omitted because of the 50 MB review-upload limit. The included scripts and
machine-readable scores expose the evaluation logic and permit independent
verification of all reported table entries.
