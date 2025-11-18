from typing import Optional, Literal

import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from knn_cuda import KNN
from timm.models.layers import trunc_normal_, DropPath

from utils.checkpoint import (
    get_missing_parameters_message,
    get_unexpected_parameters_message,
)
from utils.logger import *

from .build import MODELS

from utils import misc


class CrossAttention(nn.Module):
    def __init__(
        self,
        d_q: int,
        d_kv: int,
        d_attn: int,
        num_heads: int,
        d_out: int,
        bias: bool = False,
        cross_attn_drop_rate: float = 0.0,
        post_norm: bool = True,
    ):
        super().__init__()
        self.d_q = d_q
        self.d_kv = d_kv
        self.d_attn = d_attn
        self.num_heads = num_heads
        self.d_head = d_attn // num_heads
        self.d_out = d_out
        self.bias = bias

        self.q_norm = nn.LayerNorm(self.d_q)
        self.kv_norm = nn.LayerNorm(self.d_kv)
        self.post_norm_layer = nn.LayerNorm(self.d_q) if post_norm else nn.Identity()

        self.query = nn.Linear(self.d_q, self.d_attn, bias=self.bias)
        self.key = nn.Linear(self.d_kv, self.d_attn, bias=self.bias)
        self.value = nn.Linear(self.d_kv, self.d_attn, bias=self.bias)

        self.attn_drop = nn.Dropout(cross_attn_drop_rate)

        # Mark the weight as unmerged
        self.proj = nn.Sequential(
            nn.Linear(self.d_attn, self.d_q),
            nn.GELU(),
        )
        
    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:

        B, N_q, _ = q.size()
        N_kv = kv.size(1)

        q_norm = self.q_norm(q)
        kv_norm = self.kv_norm(kv)

        # B N D -> B N H d -> B H N d
        queries = (
            self.query(q_norm)
            .reshape(B, N_q, self.num_heads, self.d_head)
            .permute(0, 2, 1, 3)
        )
        keys = (
            self.key(kv_norm)
            .reshape(B, N_kv, self.num_heads, self.d_head)
            .permute(0, 2, 1, 3)
        )
        values = (
            self.value(kv_norm)
            .reshape(B, N_kv, self.num_heads, self.d_head)
            .permute(0, 2, 1, 3)
        )

        attn_scores = queries @ keys.transpose(-2, -1)  # B H N_q N_kv
        attn_weights = torch.softmax(attn_scores / self.d_head**0.5, dim=-1)
        attn_weights = self.attn_drop(attn_weights)
        # B H N_q N_kv @ B H N_kv d -> B H N_q d -> B N_q H d -> B N_q D
        context_vec = (attn_weights @ values).transpose(1, 2).reshape(B, N_q, -1)

        context_vec = q + self.proj(context_vec)
        return context_vec


class EdgeConv(nn.Module):
    def __init__(self, args):
        super(EdgeConv, self).__init__()
        self.args = args
        self.k = args.k
        self.in_dim = args.in_dim
        self.feat_dims = args.feat_dims
        self.emb_dim = args.emb_dim
        self.fuse_dim = sum(self.feat_dims)

        self.conv_blocks = nn.ModuleList()

        # First conv block
        self.conv_blocks.append(
            self.create_conv_block(self.in_dim * 2, self.feat_dims[0])
        )

        # Intermediate conv blocks
        for i, d_in in enumerate(self.feat_dims[:-1]):
            d_out = self.feat_dims[i + 1]
            self.conv_blocks.append(
                self.create_conv_block(in_dim=d_in * 2, out_dim=d_out)
            )

        self.final_conv = nn.Sequential(
            nn.Conv1d(self.fuse_dim, self.emb_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.emb_dim),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.knn = KNN(k=self.k)

    def create_conv_block(
        self, in_dim, out_dim, kernel_size=1, bias=False, negative_slope=0.2
    ):
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(negative_slope=negative_slope),
        )

    def get_graph_feature(self, x: torch.Tensor, idx=None):
        B, C, N = x.shape
        x = x.view(B, -1, N).contiguous()
        if idx is None:
            _, idx = self.knn(x, x)  # (B, k, N)
        idx = idx.transpose(2, 1).contiguous()  # (B, N, k)
        device = x.device

        idx_base = torch.arange(0, B, device=device).view(-1, 1, 1) * N

        idx = idx + idx_base

        idx = idx.view(-1)

        _, num_dims, _ = x.size()

        x = x.transpose(2, 1).contiguous()
        feature = x.view(B * N, -1)[idx, :]
        feature = feature.view(B, N, self.k, num_dims)
        x = x.view(B, N, 1, num_dims).repeat(1, 1, self.k, 1)

        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

        return feature  # (B, 2*C, N, k)

    def forward(self, x: torch.Tensor):
        x = x.transpose(2, 1)

        h_states = ()
        for conv in self.conv_blocks:
            x = self.get_graph_feature(x)
            x = conv(x)
            x = x.max(dim=-1, keepdim=False)[0]
            h_states += (x,)
        x = torch.cat(h_states, dim=1)
        x = self.final_conv(x)

        return x.transpose(2, 1).contiguous()

class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1),
        )

    def forward(self, point_groups):
        """
        point_groups : B G N 3
        -----------------
        feature_global : B G C
        """
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat(
            [feature_global.expand(-1, -1, n), feature], dim=1
        )  # BG 512 n
        feature = self.second_conv(feature)  # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)


class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        """
        input: B N 3
        ---------------------------
        output: B G M 3
        center : B G 3
        """
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = misc.fps(xyz, self.num_group)  # B G 3
        # knn to get the neighborhood
        _, idx = self.knn(xyz, center)  # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = (
            torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        )
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(
            batch_size, self.num_group, self.group_size, 3
        ).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center


## Transformers
class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, point_nn=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        if point_nn is not None:
            point_nn = point_nn.reshape(
                B, point_nn.shape[1], self.num_heads, C // self.num_heads
            ).permute(0, 2, 1, 3)
            k = torch.cat([point_nn, k], dim=-2)
            v = torch.cat([point_nn, v], dim=-2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )


    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        depth=4,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=(
                        drop_path_rate[i]
                        if isinstance(drop_path_rate, list)
                        else drop_path_rate
                    ),
                )
                for i in range(depth)
            ]
        )

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x


@MODELS.register_module()
class PointTransformer_GFT(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims


        self.prompt_len = config.prompt_len
        self.prompt = nn.Parameter(torch.randn(1, self.prompt_len, self.trans_dim))

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        # bridge encoder and transformer
        if self.encoder_dims != self.trans_dim:
            self.reduce_dim = nn.Linear(self.encoder_dims, self.trans_dim)
        else:
            self.reduce_dim = nn.Identity()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128), nn.GELU(), nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.cls_head_finetune = nn.Sequential(
            nn.Linear(self.trans_dim * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.cls_dim),
        )

        self.build_loss_func()

        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.cls_pos, std=0.02)

        # GFT Configs
        self.cross_idx = config.cross_idx
        
        self.edge_conv_out_dim = config.emb_dim

        self.edge_conv = EdgeConv(config)

        self.cross_attn_blocks = nn.ModuleList(
            [
                CrossAttention(
                    d_q=self.trans_dim,
                    d_kv=self.edge_conv_out_dim,
                    d_attn=config.cross_attn_dim,
                    num_heads=config.cross_attn_heads,
                    d_out=self.trans_dim,
                    cross_attn_drop_rate=config.cross_attn_drop_rate,
                )
                for _ in self.cross_idx
            ]
        )

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {
                k.replace("module.", ""): v for k, v in ckpt["base_model"].items()
            }

            for k in list(base_ckpt.keys()):
                if k.startswith("MAE_encoder"):
                    base_ckpt[k[len("MAE_encoder.") :]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith("transformer_q") and not k.startswith(
                    "transformer_q.cls_head"
                ):
                    base_ckpt[k[len("transformer_q.") :]] = base_ckpt[k]
                elif k.startswith("ACT_encoder"):
                    base_ckpt[k[len("ACT_encoder.") :]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith("base_model"):
                    base_ckpt[k[len("base_model.") :]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log("missing_keys", logger="Transformer")
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger="Transformer",
                )
            if incompatible.unexpected_keys:
                print_log("unexpected_keys", logger="Transformer")
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger="Transformer",
                )

            print_log(
                f"[Transformer] Successful Loading the ckpt from {bert_ckpt_path}",
                logger="Transformer",
            )
        else:
            print_log("Training from scratch!!!", logger="Transformer")
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts):
        B = pts.size(0)

        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # B G N
        group_input_tokens = self.reduce_dim(group_input_tokens)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        cls_pos = self.cls_pos.expand(B, -1, -1)
        prompt = self.prompt.expand(B, -1, -1)

        pos = self.pos_embed(center)

        x = torch.cat((cls_tokens, prompt, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, torch.zeros_like(prompt), pos), dim=1) # No positional embedding for prompt
        
        # Graph Feature Extraction
        g_feat = self.edge_conv(x)
        
        # transformer
        for idx, block in enumerate(self.blocks.blocks):

            if idx in self.cross_idx:
                mapped_idx = self.cross_idx.index(idx)
                x = block(x + pos)
                x = self.cross_attn_blocks[mapped_idx](x, g_feat)
            else:
                x = block(x + pos)

        x = self.norm(x)

        concat_f = torch.cat(
            [
                x[:, 0],
                x[:, 1 : self.prompt_len + 1].mean(1),
                x[:, self.prompt_len + 1 :].max(1)[0],
            ],
            dim=-1,
        )
        ret = self.cls_head_finetune(concat_f)
        return ret