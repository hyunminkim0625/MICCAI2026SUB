import torch
import torch.nn as nn
import timm
from timm.layers.mlp import Mlp
import torch.nn.functional as F
from timm.layers.classifier import NormMlpClassifierHead
from huggingface_hub import hf_hub_download
import math
from functools import partial

import timm.models.vision_transformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from timm.models.layers import trunc_normal_

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1,keepdim=True)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome


def RETFound_mae(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model



def Dinov2(args, **kwargs):
    
    if args.model_arch == 'dinov2_vits14':
        arch = 'vit_small_patch14_dinov2.lvd142m'
    elif args.model_arch == 'dinov2_vitb14':
        arch = 'vit_base_patch14_dinov2.lvd142m'
    elif args.model_arch == 'dinov2_vitl14':
        arch = 'vit_large_patch14_dinov2.lvd142m'
    elif args.model_arch == 'dinov2_vitg14':
        arch = 'vit_giant_patch14_dinov2.lvd142m'
    else:
        raise ValueError(f"Unknown model_arch '{args.model_arch}'. "
                         f"Expected one of: dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14")
        
    model = timm.create_model(
        arch,
        pretrained=True,
        img_size=224,
        **kwargs
    )
    return model



def RETFound_dinov2():
    model = timm.create_model(
        'vit_large_patch14_dinov2.lvd142m',
        pretrained=False,
        img_size=384,
        drop_path_rate=0.2,
    )
    return model


def Dinov3(args, **kwargs):
    # Load ViT-L/16 backbone (hub model has `head = Identity` by default)
    model = torch.hub.load(
        repo_or_dir="facebookresearch/dinov3",
        model=args.model_arch,
        pretrained=False,   # main() will load your checkpoint
        trust_repo=True,
    )

    # Figure out feature dimension for the probe
    feat_dim = getattr(model, "embed_dim", None) or getattr(model, "num_features", None)
    model.head = nn.Linear(feat_dim, args.nb_classes)
    trunc_normal_(model.head.weight, std=2e-5)
    if model.head.bias is not None:
        nn.init.zeros_(model.head.bias)

    return model

class Model(nn.Module):
    def __init__(self, backbone_type, img_size):
        super().__init__()
        if backbone_type == "convnext_small":
            self.backbone = timm.create_model("convnext_small_in22ft1k", pretrained=True, num_classes=1, drop_path_rate=0.2)
            in_features = self.backbone.num_features
            self.backbone.head = torch.nn.Identity()

            self.head = NormMlpClassifierHead(
                in_features=in_features,
                num_classes=1,
                drop_rate=0.2,
            )
        elif backbone_type == "vit_small":
            self.backbone = timm.create_model("vit_small_patch16_384", pretrained=True, num_classes=1, drop_path_rate=0.2)
            in_features = self.backbone.num_features
            self.backbone.head = torch.nn.Identity()

            self.head = nn.Sequential(
                nn.LayerNorm(in_features),
                nn.Dropout(0.2),
                nn.Linear(in_features, 1)
            )
        elif backbone_type == "retfound_dinov2":
            chkpt_dir = 
            checkpoint = torch.load(chkpt_dir, map_location='cpu', weights_only=False)
            
            self.backbone = RETFound_dinov2()

            state_dict = checkpoint['teacher']
            pos_embed_checkpoint = state_dict.get('backbone.pos_embed', None)
            
            if pos_embed_checkpoint is not None:
                
                embedding_size = pos_embed_checkpoint.shape[-1]
                num_extra_tokens = 1
                
                orig_num_patches = pos_embed_checkpoint.shape[1] - num_extra_tokens
                orig_grid_size = int(math.sqrt(orig_num_patches))
                
                patch_size = 14 
                new_grid_size = img_size // patch_size
                
                if orig_grid_size != new_grid_size:
                    print(f"Interpolating position embeddings from {orig_grid_size} to {new_grid_size}")
                    
                    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                    
                    pos_tokens = pos_tokens.reshape(-1, orig_grid_size, orig_grid_size, embedding_size).permute(0, 3, 1, 2)
                    pos_tokens = F.interpolate(
                        pos_tokens, 
                        size=(new_grid_size, new_grid_size), 
                        mode='bicubic', 
                        align_corners=False
                    )
                    
                    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                    
                    state_dict['backbone.pos_embed'] = new_pos_embed

            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('backbone.'):
                    new_key = k.replace('backbone.', '')
                    new_state_dict[new_key] = v
            
            msg = self.backbone.load_state_dict(new_state_dict, strict=False)
            print(msg)
            
            in_features = self.backbone.num_features
            self.backbone.head = torch.nn.Identity()
            
            self.head = nn.Sequential(
                nn.LayerNorm(in_features),
                nn.Dropout(0.2),
                nn.Linear(in_features, 1)
            )

        
    def forward(self, cfp_image):
        # ----------------------------------
        # 1. Backbone Features
        # ----------------------------------
        logits = self.backbone(cfp_image)
        logits = self.head(logits)
        return {
            'logits': logits,
        }