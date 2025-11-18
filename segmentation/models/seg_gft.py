import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from logger import get_missing_parameters_message, get_unexpected_parameters_message
import math
from pointnet2_ops import pointnet2_utils
from knn_cuda import KNN

# from pointnet2_utils import PointNetFeaturePropagation


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(B, dtype=torch.long)
        .to(device)
        .view(view_shape)
        .repeat(repeat_shape)
    )
    new_points = points[batch_indices, idx, :]
    return new_points


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

        # print("q", q, "context_vec", self.proj(context_vec))
        context_vec = q + self.proj(context_vec)
        return context_vec


def fps(data, number):
    """
    data B N 3
    number int
    """
    fps_idx = pointnet2_utils.furthest_point_sample(data, number)
    fps_data = (
        pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx)
        .transpose(1, 2)
        .contiguous()
    )
    return fps_data


class Group(nn.Module):
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
        center = fps(xyz, self.num_group)  # B G 3
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
        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        return feature_global.reshape(bs, g, self.encoder_channel)


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
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q * self.scale) @ k.transpose(-2, -1)
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
    """Transformer Encoder without hierarchical structure"""

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
        feature_list = []
        fetch_idx = [3, 7, 11]
        for i, block in enumerate(self.blocks):
            x = block(x + pos)
            if i in fetch_idx:
                feature_list.append(x)
        return feature_list


class TransformerEncoderGFT(nn.Module):
    """Transformer Encoder without hierarchical structure"""

    def __init__(
        self,
        config,
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

        self.config = config
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

        self.edge_conv = EdgeConv(config.edge_conv)

        self.cross_attn_blocks = nn.ModuleList(
            [
                CrossAttention(
                    d_q=embed_dim,
                    d_kv=config.edge_conv.emb_dim,
                    d_attn=config.cross_attention.cross_attn_dim,
                    num_heads=config.cross_attention.cross_attn_heads,
                    d_out=embed_dim,
                    cross_attn_drop_rate=config.cross_attention.cross_attn_drop_rate,
                )
                for _ in config.cross_attention.cross_idx
            ]
        )

        self.prompt_len = config.prompt_len
        self.prompt = nn.Parameter(torch.randn(1, self.prompt_len, embed_dim))

        self.cross_idx = config.cross_attention.cross_idx
        self.extract_layers = config.extract_layers

    def forward(self, x, pos):
        prompt = self.prompt.expand(x.size(0), -1, -1)
        x = torch.cat([prompt, x], dim=1)
        pos = torch.cat([torch.zeros_like(prompt), pos], dim=1) # No position for prompt
        feature_list = []
        g_feat = self.edge_conv(x)
        for i, block in enumerate(self.blocks):
            if i in self.cross_idx:
                mapped_idx = self.cross_idx.index(i)
                x = block(x + pos)
                x = self.cross_attn_blocks[mapped_idx](x, g_feat)
            else:
                x = block(x + pos)

            if i in self.extract_layers:
                feature_list.append(x)

        if self.config.decoder.use_gft:
            return feature_list, g_feat
        else:
            return feature_list

class SegmentationHead(nn.Module):
    def __init__(
        self, config, in_channels=384, hidden_dim=256, depth=3, out_channels=50
    ):
        super(SegmentationHead, self).__init__()

        self.config = config
        self.depth = depth

        self.obj_proj = nn.Embedding(config.decoder.num_obj, config.decoder.obj_dim)

        self.reduces = nn.ModuleList(
            [
                (
                    nn.Linear(in_channels, hidden_dim)
                    if in_channels != hidden_dim
                    else nn.Identity()
                )
                for _ in range(depth)
            ]
        )

        self.transformer_blocks = nn.ModuleList(
            [
                Block(
                    dim=hidden_dim,
                    num_heads=config.decoder.num_heads,
                    mlp_ratio=config.decoder.mlp_ratio,
                    qkv_bias=True,
                    qk_scale=None,
                    drop=0.1,
                    attn_drop=0.1,
                    drop_path=0.1,
                )
                for _ in range(depth)
            ]
        )

        self.feat_propagate = PointNetFeaturePropagation(
            in_channel=hidden_dim * 3 + 3, mlp=[1024, 512]
        )

        if config.decoder.use_gft:
            # for feat_propagate output + obj embeddings + prompt_feature + graph feature
            cls_input_dim = (
                512 + config.decoder.obj_dim + hidden_dim + config.edge_conv.emb_dim
            )
        else:
            # for feat_propagate output + obj embeddings + prompt_feature
            cls_input_dim = 512 + config.decoder.obj_dim + hidden_dim

        self.cls_head = nn.Sequential(
            nn.Linear(cls_input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, out_channels),
        )

    def forward(self, feature_list, cls_label, xyz, centers, g_feat=None):
        assert len(feature_list) == self.depth

        if self.config.decoder.use_gft and g_feat is None:
            raise ValueError(
                "`g_feat` must be provided when `use_gft` is True in the config."
            )

        B, N, _ = xyz.shape

        P = self.config.prompt_len

        obj = self.obj_proj(cls_label.squeeze()).unsqueeze(1).repeat(1, N, 1)  # B N 64

        feature_list = feature_list[::-1]  # Reverse the order of feature_list

        x = None
        for i, (feat, reduce, block) in enumerate(
            zip(feature_list, self.reduces, self.transformer_blocks)
        ):
            if x is not None:
                x = reduce(feat) + x
            else:
                x = reduce(feat)
            x = block(x)

        prompt_feature = x[:, :P].mean(1, keepdim=True).repeat(1, N, 1)
        x = x[:, P:]  # Remove the prompt feature from x

        x = self.feat_propagate(
            xyz.permute(0, 2, 1),
            centers.permute(0, 2, 1),
            xyz.permute(0, 2, 1),
            x.permute(0, 2, 1),
        )

        if self.config.decoder.use_gft:
            x = torch.cat(
                (
                    x.permute(0, 2, 1),
                    prompt_feature,
                    g_feat.max(dim=1, keepdim=True)[0].repeat(1, N, 1), # for use_gft
                    obj,
                ),
                dim=2,
            )
        else:
            x = torch.cat(
                (
                    x.permute(0, 2, 1),
                    prompt_feature,
                    obj,
                ),
                dim=2,
            )
        x = self.cls_head(x)
        return x


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = index_points(points2, idx) * weight.view(B, N, 3, 1)
            interpolated_points = interpolated_points.view(B, N, -1)  # [B, N, D']

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points


class get_model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        # grouper
        self.group_divider = Group(
            num_group=config.num_group, group_size=config.group_size
        )
        # define the encoder
        self.encoder = Encoder(encoder_channel=config.encoder_dims)
        # bridge encoder and transformer

        if config.encoder_dims != config.trans_dim:
            self.reduce_dim = nn.Linear(config.encoder_dims, config.trans_dim)
        else:
            self.reduce_dim = nn.Identity()

        self.prompt_cls_pos = nn.Parameter(torch.randn(1, 1, config.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128), nn.GELU(), nn.Linear(128, config.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.depth)]
        self.blocks = TransformerEncoderGFT(
            config,
            embed_dim=config.trans_dim,
            depth=config.depth,
            drop_path_rate=dpr,
            num_heads=config.num_heads,
        )
        self.prompt_len = self.blocks.prompt_len

        self.norm = nn.LayerNorm(config.trans_dim)

        self.relu = nn.ReLU()

        self.segmentation_head_cls = SegmentationHead(
            config,
            in_channels=config.decoder.in_channels,
            hidden_dim=config.decoder.hidden_dim,
            depth=len(config.extract_layers),
            out_channels=config.decoder.num_cls,
        )

        trunc_normal_(self.prompt_cls_pos, std=0.02)

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
            # print(ckpt['base_model'].items())
            for k in list(base_ckpt.keys()):
                if k.startswith("MAE_encoder"):
                    base_ckpt[k[len("MAE_encoder.") :]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith("base_model"):
                    base_ckpt[k[len("base_model.") :]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print("missing_keys")
                print(get_missing_parameters_message(incompatible.missing_keys))
            if incompatible.unexpected_keys:
                print("unexpected_keys")
                print(get_unexpected_parameters_message(incompatible.unexpected_keys))

            print(f"[Transformer] Successful Loading the ckpt from {bert_ckpt_path}")

    def load_model_from_ckpt_withrename(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)["model_state_dict"]
            model_dict = self.state_dict()
            for k in list(model_dict.keys()):
                if k in ckpt:
                    model_dict[k] = ckpt[k]
                else:
                    old_k = k.replace("_cls", "")
                    print(old_k, k)
                    model_dict[k] = ckpt[old_k]
            # base_ckpt = {k.replace("_cls.", ""): v for k, v in ckpt['model_state_dict'].items()}
            # print(ckpt['base_model'].items())
            # for k in list(base_ckpt.keys()):
            #     if k.startswith('MAE_encoder'):
            #         base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
            #         del base_ckpt[k]
            #     elif k.startswith('base_model'):
            #         base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
            #         del base_ckpt[k]

            incompatible = self.load_state_dict(model_dict, strict=False)

            if incompatible.missing_keys:
                print("missing_keys")
                print(get_missing_parameters_message(incompatible.missing_keys))
            if incompatible.unexpected_keys:
                print("unexpected_keys")
                print(get_unexpected_parameters_message(incompatible.unexpected_keys))

            print(f"[Transformer] Successful Loading the ckpt from {bert_ckpt_path}")

    def forward(self, pts, cls_label):
        B, C, N = pts.shape
        pts = pts.transpose(-1, -2)  # B N 3
        # divide the point clo  ud in the same form. This is important
        neighborhood, center = self.group_divider(pts)

        group_input_tokens = self.encoder(neighborhood)  # B G N
        group_input_tokens = self.reduce_dim(group_input_tokens)

        pos = self.pos_embed(center)
        # final input
        x = group_input_tokens

        # GFT part
        g_feat = None

        # transformer
        if self.config.decoder.use_gft:
            feature_list, g_feat = self.blocks(x, pos)
        else:
            feature_list = self.blocks(x, pos)

        feature_list = [self.norm(x) for x in feature_list]

        x = self.segmentation_head_cls(
            feature_list, cls_label, pts, center, g_feat
        )  # B N C

        x = F.log_softmax(x, dim=-1)  # B N C
        return x


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.nll_loss(pred, target)
        return total_loss
