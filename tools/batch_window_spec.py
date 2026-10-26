#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""The spec batch_window_ubench reads: the model's geometry and each layer's tensor types, from the GGUF header.

    tools/batch_window_spec.py SHARD1.gguf [--layers 0-23] > spec.txt

One `dims key=value ...` line, then one `layer L key=type ...` line per layer (ggml type ids; a layer with attn_q
is a QSA layer, the rest GDN)."""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gguf_reader import GGUFFile  # noqa: E402

TENSORS = ("attn_qkv", "attn_gate", "ssm_out", "attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate_exps",
           "ffn_up_exps", "ffn_down_exps", "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--layers", default="", help="first-last (default: every layer)")
    a = ap.parse_args()
    g = GGUFFile(pathlib.Path(a.gguf))
    m = g.metadata
    arch = m["general.architecture"]
    v = lambda key: m[f"{arch}.{key}"]  # noqa: E731
    dims = {
        "n_embd": v("embedding_length"), "n_ff": v("expert_feed_forward_length"), "n_expert": v("expert_count"),
        "k": v("expert_used_count"), "hc": v("hyper_connection.count"), "hc_lr": v("hyper_connection.low_rank"),
        "n_head": v("attention.head_count"), "n_head_kv": v("attention.head_count_kv"),
        "head_dim": v("attention.key_length"), "idx_heads": v("attention.indexer.head_count"),
        "idx_dim": v("attention.indexer.key_length"), "ssm_d_state": v("ssm.state_size"),
        "ssm_groups": v("ssm.group_count"), "ssm_heads": v("ssm.time_step_rank"), "ssm_inner": v("ssm.inner_size"),
        "ssm_d_conv": v("ssm.conv_kernel"), "n_layers": v("block_count"),
    }
    print("dims " + " ".join(f"{k}={int(x)}" for k, x in dims.items()))
    lo, hi = 0, int(dims["n_layers"]) - 1
    if a.layers:
        lo, hi = (int(x) for x in a.layers.split("-"))
    types: dict[int, dict[str, int]] = {}
    for t in g.tensors:
        parts = t.name.split(".")
        if parts[0] != "blk" or not lo <= int(parts[1]) <= hi:
            continue
        rest = ".".join(parts[2:])
        for name in TENSORS:
            if rest == name + ".weight":
                types.setdefault(int(parts[1]), {})[name] = t.type_id
    for layer in sorted(types):
        print(f"layer {layer} " + " ".join(f"{k}={x}" for k, x in types[layer].items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
