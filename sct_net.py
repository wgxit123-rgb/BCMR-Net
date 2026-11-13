"""Sparse contour token network for lightweight BUS segmentation."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from eprc_net import (
    ConvBNAct,
    CrossScaleHighResolutionFusion,
    PlainImageStem,
    PyramidContextBlock,
    ResidualDWBlock,
)


class LowRankTokenGraph(nn.Module):
    """Exchange evidence among a small token set with spatially biased rank-8 attention."""

    def __init__(self, token_dim=24, rank=8):
        super().__init__()
        self.norm = nn.LayerNorm(token_dim)
        self.query = nn.Linear(token_dim, rank, bias=False)
        self.key = nn.Linear(token_dim, rank, bias=False)
        self.value = nn.Linear(token_dim, token_dim, bias=False)
        self.project = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.SiLU(inplace=True),
            nn.Linear(token_dim, token_dim),
        )
        self.distance_gain = nn.Parameter(torch.tensor(0.0))

    def forward(self, tokens, coordinates):
        normalized = self.norm(tokens)
        query = self.query(normalized)
        key = self.key(normalized)
        affinity = torch.matmul(query, key.transpose(1, 2)) / math.sqrt(query.size(-1))
        distance = torch.cdist(coordinates, coordinates, p=2)
        affinity = affinity - F.softplus(self.distance_gain) * distance
        weights = torch.softmax(affinity, dim=-1)
        mixed = torch.matmul(weights, self.value(normalized))
        return tokens + 0.25 * self.project(mixed)


class LocalTokenWriteback(nn.Module):
    """Scatter tokens to their seed cells and refine only a 3x3 low-resolution neighborhood."""

    def __init__(self, feature_dim=24):
        super().__init__()
        self.refine = nn.Sequential(
            ConvBNAct(feature_dim * 2 + 3, feature_dim, 3),
            ConvBNAct(feature_dim, feature_dim, 3, groups=feature_dim),
            nn.Conv2d(feature_dim, 1, 1, bias=True),
        )
        nn.init.normal_(self.refine[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(
        self,
        embedding,
        tokens,
        indices,
        token_scores,
        uncertainty,
        contour,
    ):
        batch, channels, height, width = embedding.shape
        locations = height * width
        token_map = embedding.new_zeros(batch, channels, locations)
        weighted_tokens = tokens.transpose(1, 2) * token_scores.unsqueeze(1)
        token_map = token_map.scatter_add(
            2, indices.unsqueeze(1).expand(-1, channels, -1), weighted_tokens
        ).view(batch, channels, height, width)

        support = embedding.new_zeros(batch, 1, locations)
        support = support.scatter_add(2, indices.unsqueeze(1), token_scores.unsqueeze(1))
        support = support.view(batch, 1, height, width).clamp(0.0, 1.0)
        support = F.max_pool2d(support, 3, stride=1, padding=1)
        token_map = F.avg_pool2d(token_map, 3, stride=1, padding=1)
        residual = self.refine(
            torch.cat(
                [token_map, embedding, support, uncertainty, contour], dim=1
            )
        )
        return residual * support * uncertainty


class GlobalTokenWriteback(nn.Module):
    """Dense token broadcast retained only as a locality-control ablation."""

    def __init__(self, feature_dim=24, rank=8):
        super().__init__()
        self.query = nn.Conv2d(feature_dim, rank, 1, bias=False)
        self.key = nn.Linear(feature_dim, rank, bias=False)
        self.value = nn.Linear(feature_dim, feature_dim, bias=False)
        self.refine = nn.Sequential(
            ConvBNAct(feature_dim + 2, feature_dim, 3),
            nn.Conv2d(feature_dim, 1, 1, bias=True),
        )
        nn.init.normal_(self.refine[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(self, embedding, tokens, uncertainty, contour):
        batch, _, height, width = embedding.shape
        query = self.query(embedding).flatten(2).transpose(1, 2)
        key = self.key(tokens)
        weights = torch.softmax(
            torch.matmul(query, key.transpose(1, 2)) / math.sqrt(query.size(-1)),
            dim=-1,
        )
        reconstructed = torch.matmul(weights, self.value(tokens))
        reconstructed = reconstructed.transpose(1, 2).reshape(
            batch, -1, height, width
        )
        return self.refine(torch.cat([reconstructed, uncertainty, contour], dim=1))


class LocalContourWriteback(nn.Module):
    """A token-free local contour residual used to isolate neighborhood writeback."""

    def __init__(self, feature_dim=24):
        super().__init__()
        self.refine = nn.Sequential(
            ConvBNAct(feature_dim + 2, feature_dim, 3),
            ConvBNAct(feature_dim, feature_dim, 3, groups=feature_dim),
            nn.Conv2d(feature_dim, 1, 1, bias=True),
        )
        nn.init.normal_(self.refine[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(self, embedding, uncertainty, contour):
        residual = self.refine(torch.cat([embedding, uncertainty, contour], dim=1))
        return residual * uncertainty * (0.5 + 0.5 * contour)


class SparseContourTokenRefiner(nn.Module):
    """Extract, reason over, and locally write back uncertain contour tokens."""

    MODES = {"cte", "lrg", "lwr", "no_local", "full"}

    def __init__(
        self,
        channels=48,
        mode="full",
        feature_dim=24,
        token_count=24,
        map_size=56,
    ):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"Unknown SCT mode: {mode}")
        self.mode = mode
        self.token_count = token_count
        self.map_size = map_size
        self.embedding = ConvBNAct(channels, feature_dim, 1)
        self.gain = nn.Parameter(torch.tensor(0.10))

        uses_sparse_tokens = mode in {"cte", "no_local", "full"}
        uses_graph = mode in {"lrg", "no_local", "full"}
        if uses_sparse_tokens:
            self.state_embedding = nn.Sequential(
                nn.Linear(5, feature_dim),
                nn.SiLU(inplace=True),
                nn.Linear(feature_dim, feature_dim),
            )
        else:
            self.state_embedding = None
        self.graph = LowRankTokenGraph(feature_dim) if uses_graph else None
        self.local_writeback = (
            LocalTokenWriteback(feature_dim) if mode in {"cte", "full"} else None
        )
        self.global_writeback = (
            GlobalTokenWriteback(feature_dim) if mode in {"lrg", "no_local"} else None
        )
        self.local_contour = LocalContourWriteback(feature_dim) if mode == "lwr" else None

    def _coarse_state(self, feature, logits):
        size = (self.map_size, self.map_size)
        embedding = self.embedding(F.adaptive_avg_pool2d(feature, size))
        coarse_logits = F.interpolate(
            logits, size=size, mode="bilinear", align_corners=False
        )
        probability = torch.sigmoid(coarse_logits)
        uncertainty = 4.0 * probability * (1.0 - probability)
        dilation = F.max_pool2d(probability, 3, stride=1, padding=1)
        erosion = -F.max_pool2d(-probability, 3, stride=1, padding=1)
        contour = (dilation - erosion).clamp(0.0, 1.0)
        score = uncertainty * (0.25 + 0.75 * contour)
        return embedding, probability, uncertainty, contour, score

    def _sparse_tokens(self, embedding, probability, contour, score):
        batch, channels, height, width = embedding.shape
        flat_score = score.flatten(2).squeeze(1)
        token_scores, indices = torch.topk(
            flat_score, k=min(self.token_count, flat_score.size(1)), dim=1
        )
        flat_embedding = embedding.flatten(2).transpose(1, 2)
        tokens = torch.gather(
            flat_embedding,
            1,
            indices.unsqueeze(-1).expand(-1, -1, channels),
        )
        y = torch.div(indices, width, rounding_mode="floor")
        x = indices.remainder(width)
        coordinates = torch.stack(
            [
                2.0 * y.to(embedding.dtype) / max(height - 1, 1) - 1.0,
                2.0 * x.to(embedding.dtype) / max(width - 1, 1) - 1.0,
            ],
            dim=-1,
        )
        flat_probability = probability.flatten(2).squeeze(1)
        flat_contour = contour.flatten(2).squeeze(1)
        probability_values = torch.gather(flat_probability, 1, indices)
        contour_values = torch.gather(flat_contour, 1, indices)
        state = torch.cat(
            [
                probability_values.unsqueeze(-1),
                token_scores.unsqueeze(-1),
                contour_values.unsqueeze(-1),
                coordinates,
            ],
            dim=-1,
        )
        tokens = tokens + self.state_embedding(state)
        return tokens, indices, token_scores, coordinates

    @staticmethod
    def _fixed_tokens(embedding):
        batch = embedding.size(0)
        pooled = F.adaptive_avg_pool2d(embedding, (4, 6))
        tokens = pooled.flatten(2).transpose(1, 2)
        ys = torch.linspace(-1.0, 1.0, 4, device=embedding.device, dtype=embedding.dtype)
        xs = torch.linspace(-1.0, 1.0, 6, device=embedding.device, dtype=embedding.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        coordinates = torch.stack([grid_y, grid_x], dim=-1).reshape(1, 24, 2)
        return tokens, coordinates.expand(batch, -1, -1)

    def forward(self, feature, logits):
        embedding, probability, uncertainty, contour, score = self._coarse_state(
            feature, logits
        )
        if self.mode == "lwr":
            residual = self.local_contour(embedding, uncertainty, contour)
        elif self.mode == "lrg":
            tokens, coordinates = self._fixed_tokens(embedding)
            tokens = self.graph(tokens, coordinates)
            residual = self.global_writeback(embedding, tokens, uncertainty, contour)
        else:
            tokens, indices, token_scores, coordinates = self._sparse_tokens(
                embedding, probability, contour, score
            )
            if self.graph is not None:
                tokens = self.graph(tokens, coordinates)
            if self.local_writeback is not None:
                residual = self.local_writeback(
                    embedding,
                    tokens,
                    indices,
                    token_scores,
                    uncertainty,
                    contour,
                )
            else:
                residual = self.global_writeback(
                    embedding, tokens, uncertainty, contour
                )
        residual = F.interpolate(
            residual, size=logits.shape[-2:], mode="bilinear", align_corners=False
        )
        return logits + 0.5 * torch.tanh(self.gain) * torch.tanh(residual)


class SCTNet(nn.Module):
    """Lightweight high-resolution encoder with an optional sparse contour token head."""

    def __init__(
        self,
        in_ch=3,
        out_ch=1,
        channels=(32, 64, 128, 256, 512),
        mode="full",
    ):
        super().__init__()
        if mode not in {"base", *SparseContourTokenRefiner.MODES}:
            raise ValueError(f"Unknown SCT mode: {mode}")
        c1, c2, c3, c4, c5 = channels
        self.stem = PlainImageStem(in_ch, c1)
        self.stage1 = nn.Sequential(ResidualDWBlock(c1), ResidualDWBlock(c1))
        self.down2 = ConvBNAct(c1, c2, 3, stride=2)
        self.stage2 = nn.Sequential(ResidualDWBlock(c2), ResidualDWBlock(c2))
        self.down3 = ConvBNAct(c2, c3, 3, stride=2)
        self.stage3 = nn.Sequential(ResidualDWBlock(c3), ResidualDWBlock(c3))
        self.down4 = ConvBNAct(c3, c4, 3, stride=2)
        self.stage4 = nn.Sequential(ResidualDWBlock(c4), ResidualDWBlock(c4))
        self.down5 = ConvBNAct(c4, c5, 3, stride=2)
        self.stage5 = nn.Sequential(
            ResidualDWBlock(c5),
            PyramidContextBlock(c5),
            ResidualDWBlock(c5),
        )
        self.fusion = CrossScaleHighResolutionFusion(channels, 48)
        self.head = nn.Sequential(ConvBNAct(48, 32, 3), nn.Conv2d(32, out_ch, 1))
        self.refiner = None if mode == "base" else SparseContourTokenRefiner(48, mode)

    def forward(self, image):
        x1, _ = self.stem(image)
        x1 = self.stage1(x1)
        x2 = self.stage2(self.down2(x1))
        x3 = self.stage3(self.down3(x2))
        x4 = self.stage4(self.down4(x3))
        x5 = self.stage5(self.down5(x4))
        feature = self.fusion([x1, x2, x3, x4, x5])
        logits = self.head(feature)
        if self.refiner is not None:
            logits = self.refiner(feature, logits)
        return torch.sigmoid(logits)


def build_sct_net(
    in_ch=3,
    out_ch=1,
    channels=(32, 64, 128, 256, 512),
    mode="full",
):
    return SCTNet(in_ch=in_ch, out_ch=out_ch, channels=channels, mode=mode)
