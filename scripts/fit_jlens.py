#!/usr/bin/env python3
"""Fit one sharded Jacobian lens for a RepBench model."""

import argparse
import os
from pathlib import Path

import datasets
import torch
import transformers

import jlens
from jlens_models import MODELS, model_spec

CORPUS = Path("jlens_data/wikitext-train-00000.parquet")


def load_prompts(n):
    ds = datasets.Dataset.from_parquet(str(CORPUS))
    prompts = [x for x in ds["text"] if len(x.strip()) >= 600][:n]
    if len(prompts) != n:
        raise ValueError(f"only found {len(prompts)} qualifying prompts")
    return prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=MODELS)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshards", type=int, default=8)
    ap.add_argument("--n-prompts", type=int, default=1000)
    ap.add_argument("--dim-batch", type=int, default=256)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    model_path, _, layer_map = model_spec(args.tag)
    prompts = load_prompts(args.n_prompts)[args.shard::args.nshards]
    if args.smoke:
        prompts = prompts[:1]

    outdir = Path("jlens_weights") / args.tag
    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"n{args.n_prompts}_of_{args.nshards}_shard{args.shard}"
    if args.smoke:
        stem = "smoke"
    checkpoint = outdir / f"{stem}.checkpoint.pt"
    output = outdir / f"{stem}.lens.pt"
    print(f"pid={os.getpid()} tag={args.tag} shard={args.shard}/"
          f"{args.nshards} prompts={len(prompts)}", flush=True)

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    cls = transformers.AutoModelForCausalLM
    try:
        hf_model = cls.from_pretrained(
            model_path, dtype=torch.bfloat16,
            attn_implementation="eager").cuda()
    except ValueError:
        hf_model = transformers.AutoModelForImageTextToText.from_pretrained(
            model_path, dtype=torch.bfloat16,
            attn_implementation="eager").cuda()
    model = jlens.from_hf(hf_model, tokenizer, force_bos=True)
    print(model, model.layout, flush=True)
    lens = jlens.fit(
        model, prompts, source_layers=list(layer_map.values()),
        target_layer=model.n_layers - 1, dim_batch=args.dim_batch,
        max_seq_len=128, skip_first=16,
        checkpoint_path=None if args.smoke else str(checkpoint),
        checkpoint_every=5, resume=not args.smoke)
    lens.save(str(output))
    print(f"wrote {output} ({lens.n_prompts} prompts)", flush=True)


if __name__ == "__main__":
    main()
