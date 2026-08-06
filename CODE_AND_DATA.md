# RepBench code and data release

This release accompanies the AAAI 2027 submission
“RepBench: Compiling Benchmarks into Capability Representations for Large
Language Models.” It is self-contained for inspecting the corpus, evaluation
logic, frozen verbalizers, and reported result files. Model weights and
third-party benchmark repositories are not redistributed.

## Contents

- `data/probe_texts.jsonl`: the 46,149 processed probe records used by the
  representation experiments, including capability and source identifiers.
- `data/experiment_clusters.jsonl`: the 94 evaluated capability definitions.
- `data/manifest.jsonl`: machine-readable corpus/source manifest.
- `scripts/`: corpus preprocessing, hidden-state extraction, cross-benchmark
  readouts, SAE evaluation, J-Lens fitting/evaluation, and result aggregation.
- `results/common_grid/`: depth-level Diff-mean/PCA/LR result files.
- `results/jlens/`: whole-token J-Lens results and the frozen verbalizer map.
- `results/sae/`: checkpoint-aligned Gemma Scope result records.
- `results/four_method_best_depth.json`: values used in the main comparison.

## Data format

`probe_texts.jsonl` contains one JSON object per probe. The central fields are
the capability identifier, capability name, source benchmark identifier, and
normalized text. Upstream examples remain governed by their source licenses;
source identifiers are retained so that reviewers can inspect provenance.

## Reproduction outline

1. Create an environment using `requirements.txt`.
2. Inspect or rebuild normalized probes with `scripts/build_texts.py`.
3. Extract last-token hidden states with `scripts/extract_hidden.py`. Supply
   model paths through command-line arguments; no local model path is required
   by the archive.
4. Run `scripts/method_compare.py` at each captured depth to compute
   Diff-mean, PCA, and LR under leave-one-benchmark-out evaluation.
5. Aggregate method-specific best observed depths with
   `scripts/summarize_four_method_best_depth.py`.
6. For the checkpoint-aligned SAE experiment, use `scripts/sae_probe.py`.
7. For J-Lens, fit model-specific lenses with `scripts/fit_jlens.py`, then run
   `scripts/jlens_probe.py` with
   `results/jlens/jlens_whole_token_verbalizers.json`.

The supplied result files allow the reported tables to be checked without
downloading model weights or rerunning hidden-state extraction.

## Fixed choices

- Negative cap per source dataset: 30.
- Maximum negative-to-positive ratio: approximately 6:1.
- Captured depths: 25%, 50%, 75%, and 100% of model depth.
- PCA components: 10; component and orientation chosen on training benchmarks.
- Logistic regression: L2 penalty, `C=0.1`.
- Corpus sampling seed: 0.
- Representation clustering: fixed random states recorded in the scripts.
- J-Lens fitting corpus: 1,000 external WikiText-103 sequences, 128 tokens.
- J-Lens verbalizers: complete single tokens only; no subword fallback.

## Scope

The release contains no model weights, fitted Jacobian matrices, raw
hidden-state tensors, or local filesystem paths. Upstream examples remain
governed by their source licenses; consult the retained source identifiers
before redistribution or downstream use.
