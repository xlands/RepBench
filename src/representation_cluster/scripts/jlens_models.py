"""Model registry for the RepBench Jacobian-Lens runs."""

import os

BASE = os.environ.get("REPBENCH_MODEL_ROOT", "").rstrip("/")


def checkpoint(repo_id):
    """Use a local model root when supplied, otherwise use the Hub ID."""
    return f"{BASE}/{repo_id}" if BASE else repo_id

MODELS = {
    "qwen3-0.6b": (checkpoint("Qwen/Qwen3-0.6B"), [7, 14, 21, 28]),
    "qwen3-1.7b": (checkpoint("Qwen/Qwen3-1.7B"), [7, 14, 21, 28]),
    "qwen3-4b": (checkpoint("Qwen/Qwen3-4B"), [9, 18, 27, 36]),
    "qwen3-8b": (checkpoint("Qwen/Qwen3-8B"), [9, 18, 27, 36]),
    "qwen3-32b": (checkpoint("Qwen/Qwen3-32B"), [16, 32, 48, 64]),
    "qwen3.5-9b": (checkpoint("Qwen/Qwen3.5-9B"), [8, 16, 24, 32]),
    "llama3.1-8b": (
        checkpoint("meta-llama/Llama-3.1-8B-Instruct"), [8, 16, 24, 32]),
    "deepseek-r1-8b": (
        checkpoint("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"), [9, 18, 27, 36]),
    "gemma2-9b": (checkpoint("google/gemma-2-9b-it"), [10, 21, 32, 42]),
    "gemma4-12b": (checkpoint("google/gemma-4-12B-it"), [12, 24, 36, 48]),
    "gemma4-31b": (checkpoint("google/gemma-4-31B-it"), [15, 30, 45, 60]),
    "deepseek-v4-flash": (
        checkpoint("deepseek-ai/DeepSeek-V4-Flash-Base"), [11, 22, 32, 43]),
}


def model_spec(tag):
    path, hidden_layers = MODELS[tag]
    # HF hidden_states[L] is the output of block L-1. The fourth captured
    # depth is the final block output and cannot be a source below that same
    # target, so J-Lens uses the first three captured depths.
    layer_map = {hidden: hidden - 1 for hidden in hidden_layers[:-1]}
    return path, hidden_layers, layer_map
