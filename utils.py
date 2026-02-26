import torch
import torch.distributed as dist
import numpy as np
import torch.nn.functional as F
import math

def is_main_process():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

def layer_wise_decay(named_parameters, num_layers, base_lr, weight_decay=1e-2, no_weight_decay_list=[], decay_rate=0.75):

    def _is_no_decay(name: str) -> bool:
        n = name.lower()
        return (
            n.endswith(".bias") or
            ".norm" in n or        # timm norm 모듈 공통 처리
            ".bn" in n or
            ".ln" in n or
            "pos_embed" in n or
            "cls_token" in n or
            "dist_token" in n or
            "relative_position_bias_table" in n or
            "mlp_regression" in n
        )

    def _get_layer_id(name: str) -> int:
        n = name
        if n.startswith("backbone."):
            n = n[len("backbone."):]
        # embeddings/pos/cls
        if ("patch_embed" in n) or ("pos_embed" in n) or ("cls_token" in n) or ("dist_token" in n) or ("reg_token" in n):
            return 0
        # transformer blocks
        if ".blocks." in n:
            # e.g. blocks.0.attn.qkv.weight
            try:
                k_str = n.split(".blocks.")[1].split(".")[0]
                k = int(k_str)
            except Exception:
                k = 0
            return k + 1
        return num_layers + 1

    groups = {}
    for name, param in named_parameters:
        if not param.requires_grad:
            continue

        layer_id = _get_layer_id(name)
        use_decay = not _is_no_decay(name)
        lr_scale = decay_rate ** (num_layers + 1 - layer_id)
        if is_main_process():
            print(f"{name} | layer_id: {layer_id}, weight_decay: {use_decay}, lr_scale: {lr_scale:.4f}")

        key = (layer_id, use_decay)
        if key not in groups:
            groups[key] = {
                "params": [],
                "lr": base_lr * lr_scale,
                "weight_decay": (weight_decay if use_decay else 0.0),
            }
        groups[key]["params"].append(param)

    return list(groups.values())