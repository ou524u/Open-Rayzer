import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from .transformer import QK_Norm_TransformerBlock, init_weights, RopeTransformerBlock

def create_tokenizer(
    in_channels: int,
    patch_size: int,
    d_model: int
) -> nn.Sequential:
    """
    Create a patch-based tokenizer that splits input images or pose maps into tokens.

    Args:
        in_channels: Number of input channels (e.g., 3 for RGB, 6 for pose maps).
        patch_size: Spatial size of each patch.
        d_model: Dimension of the output token embeddings.

    Returns:
        An nn.Sequential model that rearranges patches and projects them to d_model.
    """
    tokenizer = nn.Sequential(
        Rearrange(
            "b v c (hh ph) (ww pw) -> (b v) (hh ww) (ph pw c)",
            ph=patch_size,
            pw=patch_size,
        ),
        nn.Linear(
            in_channels * (patch_size ** 2),
            d_model,
            bias=False,
        ),
    )
    tokenizer.apply(init_weights)
    return tokenizer


def create_unpatchifier( # [b, v, n_patches, d] -> [b, v, c, h, w]
    image_size: int,
    patch_size: int,
    d_model: int,
    out_channels: int = 3, # 
) -> nn.Sequential:
    """
    Create an image unpatchifier that reconstructs images from token embeddings.

    Args:
        d_model: Dimension of the input token embeddings.
        patch_size: Spatial size of each patch.
        image_size: Number of patches along each spatial dimension (height and width).
        out_channels: Number of output image channels (e.g., 3 for RGB).

    Returns:
        An nn.Sequential model that normalizes tokens, projects to patch pixels, applies sigmoid,
        and rearranges tokens into full images of shape [b, v, c, H, W].
    """
    unpatchifier = nn.Sequential(
        nn.LayerNorm(d_model, elementwise_affine=False),
        nn.Linear(
            d_model,
            patch_size ** 2 * out_channels,
            bias=False,
        ),
        nn.Sigmoid(),
        Rearrange(
            "b v (h w) (p1 p2 c) -> b v c (h p1) (w p2)",
            h=image_size // patch_size,
            w=image_size // patch_size,
            p1=patch_size,
            p2=patch_size,
            c=out_channels,
        ),
    )
    unpatchifier.apply(init_weights)
    return unpatchifier


def create_transformer_blocks(
    *,
    depth: int,
    d: int,
    d_head: int,
    use_special_init: bool = False,
    use_depth_init: bool = False,
    use_qk_norm: bool = False,
    # new features
    rope: None = None, # rope instance
):
    if rope is None: # original version
        blocks = [
            QK_Norm_TransformerBlock(d, d_head, use_qk_norm=use_qk_norm)
            for _ in range(depth)
        ]
    else: # rope & vggt
        blocks = [
            RopeTransformerBlock(d, d_head, use_qk_norm=use_qk_norm, rope=rope)
            for _ in range(depth)
        ]

# transformer initilization.
    if use_special_init:
        for idx, block in enumerate(blocks):
            if use_depth_init:
                weight_init_std = 0.02 / (2 * (idx + 1)) ** 0.5
            else:
                weight_init_std = 0.02 / (2 * depth) ** 0.5

            block.apply(lambda module: init_weights(module, weight_init_std))
    else:
        for block in blocks:
            block.apply(init_weights)

    return nn.ModuleList(blocks), nn.LayerNorm(d, bias=False)



from src.utils.data_utils import rays_from_c2w
def c2w_to_plucker(
    c2w, # [B, V, 4, 4]
    fxfycxcy: torch.Tensor = None,
    H: int = 256,
    W: int = 256,
):
    ray_o, ray_d = rays_from_c2w(c2w, fxfycxcy, H, W)
    o_cross_d = torch.cross(ray_o, ray_d, dim=2)
    plucker = torch.cat([o_cross_d, ray_d], dim=2)
    return plucker # [B, V, 6, H, W]




class ConcatMLPFuse(nn.Module):
    """
    Fuse two feature maps (feat_a, feat_b) of shape [B, V, N, D].
    Keeps strict one-to-one correspondence across positions (B,V,N).
    Fusion is done by concatenation along the last dimension and passing
    through a small MLP applied per position.

    Args:
        d (int): feature dimension of feat_a and feat_b
        hidden (int): hidden dimension of MLP, defaults to 2*d
        out_dim (int): output feature dimension, defaults to d
        act (nn.Module): activation function, defaults to nn.GELU
        dropout (float): dropout rate
        depth (int): number of MLP layers, defaults to 2
        with_residual (bool): add residual connection if out_dim == d
    """
    def __init__(self, d, hidden=None, out_dim=None,
                 act=nn.GELU, dropout=0.0, depth=2, with_residual=True):
        super().__init__()
        hidden = hidden or 2 * d
        out_dim = out_dim or d

        self.ln_a = nn.LayerNorm(d)
        self.ln_b = nn.LayerNorm(d)

        layers = []
        in_dim = 2 * d
        for i in range(depth):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(act())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, out_dim))
        self.net = nn.Sequential(*layers)

        self.with_residual = with_residual and (out_dim == d)

    def forward(self, feat_a, feat_b):  # feat_a, feat_b: [B, V, N, D]
        a_ = self.ln_a(feat_a)
        b_ = self.ln_b(feat_b)
        z = self.net(torch.cat([a_, b_], dim=-1))  # fuse per-position
        if self.with_residual:
            z = z + 0.5 * (feat_a + feat_b)  # residual connection
        return z  # [B, V, N, out_dim]

from .dinov3.vision_transformer import vit_base
from pathlib import Path

class dinov3_tokenizer(nn.Module):
    def __init__(self, dino_qknorm=False):
        super().__init__()
        # dino_qknorm = self.config.get("unposed", {}).get("dino_qknorm", False)
        # dino_qknorm = True
        print(f"DINO QKNorm status: {dino_qknorm}")
        self.dino_vit = vit_base(
            img_size=256, # dinov3 backbone is trained on 256x256？
            patch_size=16, 
            n_storage_tokens=4,
            layerscale_init=1.0,
            mask_k_bias=True,
            # interpolate_antialias=True,
            # interpolate_offset=0.0,
            # block_chunks=0,
            # init_values=1.0,
            qk_norm=dino_qknorm, # enable qknorm
        )
        dino_file = Path("./datasets/pretrained/dinov3_vitb16_pretrain.pth")
        dino_file.parent.mkdir(exist_ok=True, parents=True)
        if torch.distributed.get_rank() == 0:
            # Download weights if needed
            if not dino_file.exists():
                raise FileNotFoundError(f"DINO weights file not found at {dino_file}")
        torch.distributed.barrier()
        # self.dino_vit.load_state_dict(torch.load(dino_file, map_location='cpu'), strict=False)
        # print(f"DINO weights from {dino_file} loaded!")
        # state = torch.load(dino_file, map_location="cpu")

        missing, unexpected = self.dino_vit.load_state_dict(torch.load(dino_file, map_location='cpu'), strict=False)
        if torch.distributed.get_rank() == 0:
            if missing:
                print("⚠️ Missing keys (in model, not in checkpoint):")
                for k in missing:
                    print("   ", k)
            if unexpected:
                print("⚠️ Unexpected keys (in checkpoint, not in model):")
                for k in unexpected:
                    print("   ", k)
            print(f"DINO weights from {dino_file} loaded with {len(missing)} missing and {len(unexpected)} unexpected keys.")

        # if hasattr(self.dino_vit, "mask_token"):
        #     del self.dino_vit.mask_token

    def forward(self, x, with_cls=False):
        # if x is [b, v, c, h, w], we need to rearrange it to [b*v, c, h, w]
        if x.dim() == 5:
            x = rearrange(x, "b v c h w -> (b v) c h w")

        x = self.dino_vit(x, is_training=True)

        if isinstance(x, dict):
            x_cls = x["x_norm_clstoken"] # [B, D]
            x = x["x_norm_patchtokens"] # [B, N, D]
            if with_cls:
                # make it [B, N+1, D]
                x = torch.cat([x_cls.unsqueeze(1), x], dim=1)
        return x
