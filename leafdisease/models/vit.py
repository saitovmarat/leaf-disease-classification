import torch
from torch import nn
import math


class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim):
        super().__init__()
        assert image_size % patch_size == 0
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            patch_size,
            patch_size
        )

    def forward(self, images):
        """
        images: (B, 3, H, W)
        emb: (B, N, D)
        """
        emb = self.proj(images).flatten(2).transpose(1, 2)
        return emb
    
    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.attn_dropout = nn.Dropout(dropout)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (B, T, D)
        output, attn_w: (B, T, D), (B, num_heads, T, T)
        """
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, self.num_heads, self.head_dim, 3).transpose(1, 2)
        q, k, v = qkv.unbind(dim=-1)

        attn_w = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.head_dim), dim=-1)
        head_output = self.attn_dropout(attn_w) @ v

        output = head_output.transpose(1, 2).reshape(B, T, D)
        output = self.proj_dropout(self.out_proj(output))
        return (output, attn_w)
    

class FeedForward(nn.Module):
    def __init__(self, embed_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)
    

class ViTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = FeedForward(embed_dim, dropout)

    def forward(self, x):
        """
        x: (B, T, D)
        x, attn_w: (B, T, D), (B, num_heads, T, T)
        """
        attn_output, attn_w = self.attn(self.norm1(x))
        x = x + attn_output
        x = x + self.mlp(self.norm2(x))

        return (x, attn_w)
    
    
class TinyViT(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, 
                 num_classes, embed_dim, num_heads, depth, dropout):
        super().__init__()
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, num_heads, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self.apply(self._init_weights)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, images, **kwargs):
        return self.forward_tokens(self.patch_embed(images), **kwargs)

    def forward_tokens(self, patches, use_positions=True, return_attention=False):
        B, N, D = patches.shape
        assert N == self.num_patches

        x = torch.cat([self.cls_token.expand(B, 1, D), patches], dim=1)
        if use_positions:
            x = x + self.pos_embed

        attentions = []
        for block in self.blocks:
            x, attention = block(x)
            if return_attention:
                attentions.append(attention)

        cls = self.norm(x)[:, 0, :]
        logits = self.head(cls)
        return (logits, attentions) if return_attention else logits