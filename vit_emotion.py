import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """
    224x224 -> 14x14 patches (patch_size=16), each embedded to embed_dim.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        # 用一个 stride = patch_size 的 Conv2d 来做 patch embedding
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)  # [B, embed_dim, H/ps, W/ps]
        x = x.flatten(2)  # [B, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [B, num_patches, embed_dim]
        return x


class TokenSE(nn.Module):
    """
    Token-wise Squeeze-Excitation:
    输入: patch tokens [B, N, D]
    输出: 重加权后的 tokens [B, N, D]
    """
    def __init__(self, num_tokens: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, num_tokens // reduction)
        self.fc = nn.Sequential(
            nn.Linear(num_tokens, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_tokens, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, N, D] (不含 CLS)
        B, N, D = x.shape
        # 在 embedding 维度上做 mean，得到每个 token 的“强度”
        token_stats = x.mean(dim=-1)  # [B, N]
        # 通过 bottleneck FC 得到权重
        w = self.fc(token_stats)      # [B, N]
        w = w.unsqueeze(-1)           # [B, N, 1]
        return x * w                  # [B, N, D]


class EmotionViT(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 5,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ):
        super().__init__()

        # Patch Embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        # [CLS] token + position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=drop_rate,
            batch_first=True,   # 直接 [B, N, D]
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Token-wise SE attention (只对 patch tokens 做)
        self.token_se = TokenSE(num_tokens=num_patches, reduction=4)

        # Classification head: Dual pooling (CLS + mean-patch)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(embed_dim, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        # 简单初始化 Linear/LayerNorm
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_features(self, x):
        """
        返回 penultimate features（用于 t-SNE 等）
        """
        B = x.shape[0]
        x = self.patch_embed(x)  # [B, N, D]
        N = x.shape[1]

        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat((cls_tokens, x), dim=1)          # [B, 1+N, D]

        x = x + self.pos_embed[:, : 1 + N, :]
        x = self.pos_drop(x)

        x = self.encoder(x)  # [B, 1+N, D]

        # 拆开 CLS 和 patch tokens
        cls_token = x[:, 0:1, :]      # [B, 1, D]
        patch_tokens = x[:, 1:, :]    # [B, N, D]

        # 对 patch tokens 做 token-wise SE
        patch_tokens = self.token_se(patch_tokens)     # [B, N, D]

        # 再拼回去（如果后面还要用）
        x = torch.cat([cls_token, patch_tokens], dim=1)

        # LayerNorm
        x = self.norm(x)

        # Dual pooling
        cls_feat = x[:, 0]               # [B, D]
        patch_mean = x[:, 1:].mean(dim=1)  # [B, D]

        feat = torch.cat([cls_feat, patch_mean], dim=-1)  # [B, 2D]
        return feat

    def forward(self, x):
        feat = self.forward_features(x)   # [B, 2D]
        logits = self.head(feat)         # [B, num_classes]
        return logits
