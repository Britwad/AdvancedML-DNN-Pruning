# vit.py
# ViT (DeiT-Tiny settings) with maskable Linear layers for pruning.
# - Adds self.register_buffer("weight_mask", torch.ones_like(self.weight)) to all prunable Linear layers
# - Does NOT wrap embeddings/patch-embedding
# - Exposes a VGG(num_classes) wrapper so you can swap it in your existing train.py

from __future__ import annotations
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Maskable Linear
# -----------------------------
class LinearMasked(nn.Linear):
    """nn.Linear with a persistent weight mask for unstructured/structured pruning."""
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        # 1 = keep, 0 = prune
        self.register_buffer("weight_mask", torch.ones_like(self.weight))

    def forward(self, x):
        return F.linear(x, self.weight * self.weight_mask, self.bias)


# -----------------------------
# Patch Embedding  (NOT masked)
# -----------------------------
class PatchEmbed(nn.Module):
    """
    Split image into non-overlapping patches and project to embedding dim.
    Kept as a plain Conv2d so it's NOT pruned (acts like an embedding).
    """
    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_chans: int = 3, embed_dim: int = 192):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        # NOTE: plain Conv2d (no mask) by design
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B, C, H, W -> B, N, D
        x = self.proj(x)                     # [B, D, H/ps, W/ps]
        x = x.flatten(2).transpose(1, 2)     # [B, N, D]
        return x


# -----------------------------
# Attention
# -----------------------------
class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 3, dim_head: int = 64,
                 qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head
        assert inner_dim == dim, \
            f"Expected heads*dim_head == dim; got {heads}*{dim_head}!={dim}."

        self.scale = dim_head ** -0.5
        # Masked qkv & proj (prunable)
        self.to_qkv = LinearMasked(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = LinearMasked(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.to_qkv(x)  # [B, N, 3*C]
        q, k, v = qkv.chunk(3, dim=-1)

        # reshape -> [B, heads, N, dim_head]
        def reshape_heads(t):
            return t.view(B, N, self.heads, self.dim_head).transpose(1, 2)

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

        # attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v                           # [B, heads, N, dim_head]
        out = out.transpose(1, 2).contiguous().view(B, N, C)  # [B, N, C]
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


# -----------------------------
# MLP block
# -----------------------------
class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = LinearMasked(dim, hidden_dim, bias=True)  # prunable
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = LinearMasked(hidden_dim, dim, bias=True)  # prunable
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# -----------------------------
# Encoder block (Pre-LN)
# -----------------------------
class Block(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, drop: float = 0.0, attn_drop: float = 0.0,
                 drop_path: float = 0.0, layer_norm_eps: float = 1e-6):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.attn = Attention(dim, heads=heads, dim_head=dim_head,
                              qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim, eps=layer_norm_eps)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = MLP(dim, mlp_hidden, drop=drop)

        # Optional stochastic depth (DropPath); left as identity by default
        self.drop_path_rate = drop_path
        self._use_drop_path = drop_path > 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_drop_path:
            x = x + F.dropout(self.attn(self.norm1(x)), p=self.drop_path_rate, training=self.training)
            x = x + F.dropout(self.mlp(self.norm2(x)), p=self.drop_path_rate, training=self.training)
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        return x


# -----------------------------
# ViT (DeiT-Tiny defaults)
# -----------------------------
class ViT(nn.Module):
    """
    Vision Transformer with DeiT-Tiny defaults:
        img_size=224, patch_size=16, embed dim=192, depth=12, heads=3, mlp_ratio=4
        qkv_bias=True, all dropout=0 by default
    All prunable Linear layers use LinearMasked (with .weight_mask buffers).
    Embeddings/patch-embedding are NOT wrapped and thus not pruned.
    """
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        num_classes: int = 1000,
        in_chans: int = 3,
        dim: int = 192,
        depth: int = 12,
        heads: int = 3,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        emb_dropout: float = 0.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        layer_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dim = dim

        self.patch_embed = PatchEmbed(image_size, patch_size, in_chans, dim)
        num_patches = self.patch_embed.num_patches

        # cls token & positional embedding (not pruned)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, dim))
        self.pos_drop = nn.Dropout(emb_dropout)

        # transformer encoder
        self.blocks = nn.ModuleList([
            Block(dim, heads, dim_head=(dim // heads), mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
                  drop_path=0.0, layer_norm_eps=layer_norm_eps)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim, eps=layer_norm_eps)

        # classifier head (prunable)
        self.head = LinearMasked(dim, num_classes) if num_classes > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        # Standard ViT/DeiT init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
        self.apply(_init)

        # ensure masks start as "keep all"
        for m in self.modules():
            if isinstance(m, LinearMasked):
                m.weight_mask.data.fill_(1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != (224, 224):
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        x = self.patch_embed(x)                                # [B, N, D]
        B, N, _ = x.shape

        cls = self.cls_token.expand(B, -1, -1)                # [B, 1, D]
        x = torch.cat((cls, x), dim=1)                        # [B, 1+N, D]
        x = x + self.pos_embed[:, : (N + 1), :]
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]                                        # CLS

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.head(x)
        return x


# -----------------------------
# Builder (exact DeiT-Tiny)
# -----------------------------
def vit_tiny_patch16_224(num_classes: int = 1000) -> ViT:
    return ViT(
        image_size=224,
        patch_size=16,
        num_classes=num_classes,
        in_chans=3,
        dim=192,
        depth=12,
        heads=3,
        mlp_ratio=4.0,
        qkv_bias=True,
        emb_dropout=0.0,
        drop=0.0,
        attn_drop=0.0,
        layer_norm_eps=1e-6,   # You can set 1e-12 to mirror HF exactly
    )
