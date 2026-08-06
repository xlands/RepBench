#!/usr/bin/env python3
"""Probe step 2: extract last-token hidden states (multi-model).

Each text is wrapped in the model's chat template as a user turn; we take
the hidden state at the final prompt token at 4 fractional depths
(25/50/75/100% of num_hidden_layers). Long inputs keep head+tail halves.

Run one shard per GPU:
  CUDA_VISIBLE_DEVICES=k python3 extract_hidden.py --model <path> --tag qwen3-8b \
      --shard k --nshards 4

Outputs per shard in hidden/{tag}/:
  uids_{k}.json, h_{k}_L{...}.npy  (float16, [n, hidden])
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_TOK = 1024
BATCH = 32
FRACS = [0.25, 0.5, 0.75, 1.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=4)
    ap.add_argument("--missing", action="store_true",
                    help="extract only texts absent from hidden/{tag}, "
                         "saved as one new shard")
    ap.add_argument("--device-map", default="cuda",
                    help='"cuda" (single GPU) or "auto" (shard across visible GPUs)')
    ap.add_argument("--dequant-fp8", action="store_true",
                    help="load an fp8 checkpoint dequantized to bf16 "
                         "(offline; avoids the runtime fp8 kernel dependency)")
    ap.add_argument("--batch", type=int, default=BATCH,
                    help="forward batch size (lower it for models whose eager "
                         "attention materializes B×H×T×T logits, e.g. DSv4 DSA)")
    args = ap.parse_args()

    texts = [json.loads(l) for l in Path("probe_texts.jsonl").open()]
    outdir = Path("hidden") / args.tag
    if args.missing:
        have = set()
        for uf in outdir.glob("uids_*.json"):
            have.update(json.loads(uf.read_text()))
        texts = [t for t in texts if t["uid"] not in have]
        shard_idx = 1 + max(int(p.stem.split("_")[1])
                            for p in outdir.glob("uids_*.json"))
        print(f"[{args.tag}] missing mode: {len(texts)} new texts "
              f"-> shard {shard_idx}")
        if not texts:
            return
    else:
        texts = [t for i, t in enumerate(texts) if i % args.nshards == args.shard]
        shard_idx = args.shard
        print(f"[{args.tag}] shard {args.shard}: {len(texts)} texts")

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    load_kw = dict(torch_dtype=torch.bfloat16, device_map=args.device_map)
    if args.dequant_fp8:
        from transformers import FineGrainedFP8Config
        load_kw["quantization_config"] = FineGrainedFP8Config(
            activation_scheme="dynamic", weight_block_size=(128, 128),
            scale_fmt="ue8m0", dequantize=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kw)
    except ValueError:  # multimodal ForConditionalGeneration (gemma4 etc.)
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(args.model, **load_kw)
    model.eval()
    cfg = getattr(model.config, "text_config", model.config)
    n_layers = cfg.num_hidden_layers
    hidden_size = cfg.hidden_size
    layers = sorted({max(1, round(n_layers * f)) for f in FRACS})
    print(f"[{args.tag}] {n_layers} layers -> capture {layers}")

    def template(text):
        msgs = [{"role": "user", "content": text}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True)
        except ValueError:  # base model without a chat template: BOS + raw text
            return (tok.bos_token or "") + text

    def prep(text):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) > MAX_TOK - 40:  # leave room for template tokens
            half = (MAX_TOK - 40) // 2
            text = tok.decode(ids[:half]) + "\n...\n" + tok.decode(ids[-half:])
        return template(text)

    prompts = [prep(t["text"]) for t in texts]
    order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
    feats = {L: np.zeros((len(prompts), hidden_size), np.float16)
             for L in layers}

    with torch.no_grad():
        bs = args.batch
        for b0 in range(0, len(order), bs):
            idx = order[b0:b0 + bs]
            enc = tok([prompts[i] for i in idx], return_tensors="pt",
                      padding=True, truncation=True, max_length=MAX_TOK + 64,
                      add_special_tokens=False).to("cuda")
            out = model(**enc, output_hidden_states=True)
            for L in layers:
                h = out.hidden_states[L]
                if h.dim() == 4:   # hyper-connection residual streams (B,T,n,H):
                    h = h.mean(2)  # average streams (≡ sum after per-dim standardize)
                h = h[:, -1, :].float().cpu().numpy()
                for j, i in enumerate(idx):
                    feats[L][i] = h[j].astype(np.float16)
            if (b0 // bs) % 40 == 0:
                print(f"  {b0}/{len(order)}", flush=True)

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"uids_{shard_idx}.json").write_text(
        json.dumps([t["uid"] for t in texts]))
    for L in layers:
        np.save(outdir / f"h_{shard_idx}_L{L}.npy", feats[L])
    print(f"[{args.tag}] shard {shard_idx} done")


if __name__ == "__main__":
    main()
