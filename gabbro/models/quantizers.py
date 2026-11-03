"""Quantizer abstractions for VQ-VAE latent bottlenecks."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from vqtorch.nn import VectorQuant

from gabbro.models.transformer import NormformerStack


class ResidualVectorQuantizer(nn.Module):
    """Successively quantize latent residuals with independent EMA codebooks.

    Each input vector is represented by ``num_quantizers`` indices, one from
    each ``codebook_size``-entry codebook. The decoder receives the sum of the
    selected code vectors. Codebooks are updated from assigned residuals with
    exponential moving averages; gradients reach the encoder through the
    straight-through estimate and the per-stage commitment loss.
    """

    def __init__(
        self,
        feature_size: int,
        num_quantizers: int,
        codebook_size: int,
        commitment_weight: float = 1.0,
        ema_decay: float = 0.8,
        ema_epsilon: float = 1e-5,
        distance_chunk_size: int = 1024,
    ):
        super().__init__()
        if feature_size < 1:
            raise ValueError("feature_size must be positive")
        if num_quantizers < 1:
            raise ValueError("num_quantizers must be positive")
        if codebook_size < 2:
            raise ValueError("codebook_size must be at least 2")
        if commitment_weight < 0:
            raise ValueError("commitment_weight must be non-negative")
        if not 0 <= ema_decay < 1:
            raise ValueError("ema_decay must be in [0, 1)")
        if ema_epsilon <= 0:
            raise ValueError("ema_epsilon must be positive")
        if distance_chunk_size < 1:
            raise ValueError("distance_chunk_size must be positive")

        self.feature_size = int(feature_size)
        self.num_quantizers = int(num_quantizers)
        self.codebook_size = int(codebook_size)
        # ``num_codes`` remains the size of one token vocabulary. Consumers
        # use ``num_quantizers`` to account for all Q indices.
        self.num_codes = self.codebook_size
        self.commitment_weight = float(commitment_weight)
        self.ema_decay = float(ema_decay)
        self.ema_epsilon = float(ema_epsilon)
        self.distance_chunk_size = int(distance_chunk_size)

        codebooks = torch.empty(
            self.num_quantizers,
            self.codebook_size,
            self.feature_size,
        )
        nn.init.normal_(codebooks, std=self.feature_size**-0.5)
        self.register_buffer("codebooks", codebooks)
        self.register_buffer(
            "ema_cluster_size",
            torch.zeros(self.num_quantizers, self.codebook_size),
        )
        self.register_buffer("ema_codebook_sum", codebooks.clone())

    def _nearest_codes(
        self, residual: torch.Tensor, codebook: torch.Tensor
    ) -> torch.Tensor:
        indices = []
        for chunk in residual.split(self.distance_chunk_size, dim=0):
            # Squared Euclidean distance without materializing an extra
            # feature dimension: [N, 1] + [1, K] - 2 [N, K].
            distances = (
                chunk.float().pow(2).sum(dim=-1, keepdim=True)
                + codebook.float().pow(2).sum(dim=-1).unsqueeze(0)
                - 2 * chunk.float() @ codebook.float().transpose(0, 1)
            )
            indices.append(distances.argmin(dim=-1))
        return torch.cat(indices, dim=0)

    @staticmethod
    def _distributed_sum(value: torch.Tensor) -> torch.Tensor:
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
        return value

    @torch.no_grad()
    def _ema_update(
        self,
        stage: int,
        residual: torch.Tensor,
        indices: torch.Tensor,
    ) -> None:
        counts = torch.bincount(indices, minlength=self.codebook_size).to(
            dtype=self.ema_cluster_size.dtype
        )
        sums = torch.zeros_like(self.ema_codebook_sum[stage])
        sums.index_add_(0, indices, residual.to(dtype=sums.dtype))
        self._distributed_sum(counts)
        self._distributed_sum(sums)

        decay = self.ema_decay
        self.ema_cluster_size[stage].mul_(decay).add_(counts, alpha=1 - decay)
        self.ema_codebook_sum[stage].mul_(decay).add_(sums, alpha=1 - decay)

        cluster_size = self.ema_cluster_size[stage]
        total = cluster_size.sum()
        smoothed = (
            (cluster_size + self.ema_epsilon)
            / (total + self.codebook_size * self.ema_epsilon)
            * total
        )
        self.codebooks[stage].copy_(
            self.ema_codebook_sum[stage]
            / smoothed.clamp_min(self.ema_epsilon).unsqueeze(-1)
        )

    def get_codebook(self) -> torch.Tensor:
        """Return all codebooks with shape ``[Q, K, D]``."""
        return self.codebooks

    def decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode ``[..., Q]`` indices by summing their stage embeddings."""
        tokens = tokens.long()
        if tokens.ndim < 1 or tokens.shape[-1] != self.num_quantizers:
            raise ValueError(
                "Residual VQ tokens must end in the configured stage dimension "
                f"{self.num_quantizers}, got {tuple(tokens.shape)}"
            )
        if torch.any(tokens < 0) or torch.any(tokens >= self.codebook_size):
            raise ValueError("Residual VQ token IDs are outside the codebook range")
        decoded = torch.zeros(
            *tokens.shape[:-1],
            self.feature_size,
            device=tokens.device,
            dtype=self.codebooks.dtype,
        )
        for stage in range(self.num_quantizers):
            decoded = decoded + F.embedding(tokens[..., stage], self.codebooks[stage])
        return decoded

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if z.ndim < 2 or z.shape[-1] != self.feature_size:
            raise ValueError(
                f"Expected residual-VQ input ending in {self.feature_size} features, "
                f"got {tuple(z.shape)}"
            )
        input_shape = z.shape
        z_flat = z.reshape(-1, self.feature_size)
        quantized_sum = torch.zeros_like(z_flat)
        stage_indices = []
        stage_losses = []

        for stage in range(self.num_quantizers):
            residual = z_flat - quantized_sum
            indices = self._nearest_codes(residual.detach(), self.codebooks[stage])
            selected = F.embedding(indices, self.codebooks[stage]).to(dtype=z_flat.dtype)
            stage_indices.append(indices)
            stage_losses.append((residual - selected.detach()).pow(2).mean(dim=-1))
            quantized_sum = quantized_sum + selected
            # Token-export helpers use ``torch.no_grad`` and must never mutate
            # the learned dictionaries even if their caller forgot ``eval``.
            if self.training and torch.is_grad_enabled():
                self._ema_update(stage, residual.detach(), indices)

        indices = torch.stack(stage_indices, dim=-1).reshape(
            *input_shape[:-1], self.num_quantizers
        )
        stage_loss = torch.stack(stage_losses, dim=-1).reshape(
            *input_shape[:-1], self.num_quantizers
        )
        quantized = quantized_sum.reshape(input_shape)
        quantized_st = z + (quantized - z).detach()
        commitment_loss = stage_loss.mean()
        return quantized_st, {
            "z": z,
            "z_q": quantized,
            "q": indices,
            "stage_loss": stage_loss,
            "loss": commitment_loss,
            "loss_weighted": self.commitment_weight * commitment_loss,
        }


class FSQ(nn.Module):
    """Finite scalar quantization with straight-through gradients.

    ``mapping="reference"`` implements the asymmetric bound from the FSQ
    reference implementation, including its handling of even level counts.
    ``mapping="legacy"`` preserves the historical Gabbro mapping so existing
    checkpoints and exported tokens remain reproducible.
    """

    def __init__(
        self,
        levels: list[int],
        loss_use_half_width: bool = False,
        mapping: str = "reference",
    ):
        super().__init__()
        if not levels:
            raise ValueError("FSQ levels must contain at least one entry")
        if any(
            not isinstance(level, int) or isinstance(level, bool) or level < 1
            for level in levels
        ):
            raise ValueError(f"FSQ levels must be positive integers, got {levels}")
        if mapping not in {"reference", "legacy"}:
            raise ValueError(
                "FSQ mapping must be either 'reference' or 'legacy', "
                f"got {mapping!r}"
            )
        self.register_buffer("levels", torch.tensor(levels, dtype=torch.float32))
        self.num_codes = int(torch.prod(self.levels).item())
        self.feature_size = len(levels)
        self.loss_use_half_width = loss_use_half_width
        self.mapping = mapping

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.mapping == "legacy":
            return self._forward_legacy(z)
        return self._forward_reference(z)

    def _strides(self, device: torch.device) -> torch.Tensor:
        return torch.cumprod(
            torch.cat(
                [
                    torch.ones(1, device=device, dtype=torch.long),
                    self.levels[:-1].long(),
                ]
            ),
            dim=0,
        )

    def _combine_per_dim_codes(self, per_dim_codes: torch.Tensor) -> torch.Tensor:
        return (per_dim_codes * self._strides(per_dim_codes.device)).sum(dim=-1)

    def _forward_legacy(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        half_width = (self.levels - 1) / 2
        z_bounded = torch.tanh(z)
        z_scaled = z_bounded * half_width
        z_rounded = torch.round(z_scaled)
        z_hat = z_rounded / half_width.clamp(min=1)
        # Straight-through gradients follow the bounded FSQ path, not raw z.
        z_decoded = z_bounded + (z_hat - z_bounded).detach()

        per_dim_codes = (z_rounded + half_width).long().clamp_min(0)
        codes = self._combine_per_dim_codes(per_dim_codes)
        if self.loss_use_half_width:
            loss_delta = z_scaled - z_rounded.detach()
        else:
            loss_delta = z_bounded - z_hat.detach()
        loss = loss_delta.pow(2).mean(dim=-1)
        return z_decoded, codes, loss

    def _forward_reference(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # This is the asymmetric bounding function used by the paper's
        # reference implementation. It is necessary for round-to-nearest to
        # expose exactly L distinct integers when L is even.
        eps = 1e-3
        half_l = (self.levels - 1) * (1 - eps) / 2
        offset = torch.where(
            self.levels.long() % 2 == 1,
            torch.zeros_like(self.levels),
            torch.full_like(self.levels, 0.5),
        )
        safe_half_l = torch.where(half_l > 0, half_l, torch.ones_like(half_l))
        shift = torch.tan(offset / safe_half_l)
        z_bounded = torch.tanh(z + shift) * half_l - offset

        z_rounded = torch.round(z_bounded)
        half_width = torch.floor(self.levels / 2)
        safe_half_width = half_width.clamp(min=1)
        z_bounded_normalized = z_bounded / safe_half_width
        z_hat = z_rounded / safe_half_width
        # Straight-through gradients follow the bounded, normalized FSQ path.
        z_decoded = z_bounded_normalized + (
            z_hat - z_bounded_normalized
        ).detach()

        per_dim_codes = (z_rounded + half_width).long()
        per_dim_codes = torch.maximum(
            per_dim_codes, torch.zeros_like(per_dim_codes)
        )
        per_dim_codes = torch.minimum(per_dim_codes, self.levels.long() - 1)
        codes = self._combine_per_dim_codes(per_dim_codes)

        if self.loss_use_half_width:
            loss_delta = z_bounded - z_rounded.detach()
        else:
            loss_delta = z_bounded_normalized - z_hat.detach()
        loss = loss_delta.pow(2).mean(dim=-1)
        return z_decoded, codes, loss

    def decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Unpack integer FSQ codes back into quantized scalar vectors."""
        tokens = tokens.long()
        decoded = []
        residual = tokens
        for level in self.levels.long():
            decoded.append(residual % level)
            residual = torch.div(residual, level, rounding_mode="floor")

        per_dim_codes = torch.stack(decoded, dim=-1).to(self.levels.dtype)
        if self.mapping == "legacy":
            half_width = (self.levels - 1) / 2
        else:
            half_width = torch.floor(self.levels / 2)
        return (per_dim_codes - half_width) / half_width.clamp(min=1)


class VQBranch(nn.Module):
    """Small wrapper normalizing vqtorch's output contract for split branches."""

    def __init__(self, feature_size: int, vq_kwargs: dict[str, Any]):
        super().__init__()
        vq_kwargs = dict(vq_kwargs)
        vq_kwargs.setdefault("dim", -1)
        self.vq = VectorQuant(feature_size=feature_size, **vq_kwargs)
        self.feature_size = feature_size
        self.num_codes = int(vq_kwargs["num_codes"])

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_q, vq_out = self.vq(z)
        z_vq = vq_out["z"]
        z_q_vq = vq_out["z_q"]
        loss = (
            (1.0 - self.vq.beta) * (z_vq - z_q_vq.detach()).pow(2).mean(dim=-1)
            + self.vq.beta * (z_vq.detach() - z_q_vq).pow(2).mean(dim=-1)
        )
        return z_q, vq_out["q"].squeeze(-1), loss

    def decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Look up branch VQ tokens in the current codebook."""
        return F.embedding(tokens.long(), self.vq.get_codebook())


class NormformerProjection(nn.Module):
    """Projection block: input/output adapters around a NormFormer stack."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int | None = None,
        num_heads: int = 1,
        num_blocks: int = 1,
        mlp_expansion_factor: int = 4,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        hidden_dim = int(hidden_dim or input_dim)
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.input_projection = (
            nn.Identity() if input_dim == hidden_dim else nn.Linear(input_dim, hidden_dim)
        )
        self.blocks = NormformerStack(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            dropout_rate=dropout_rate,
            mlp_expansion_factor=mlp_expansion_factor,
        )
        self.output_projection = (
            nn.Identity() if hidden_dim == output_dim else nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.input_projection(x)
        if mask is None:
            mask = torch.ones(x.shape[:2], device=x.device, dtype=x.dtype)
        x = self.blocks(x, mask)
        return self.output_projection(x) * mask.unsqueeze(-1)


class SplitPhi(nn.Module):
    """Map encoder latents into independently quantized branches."""

    def __init__(
        self,
        latent_dim: int,
        branch_dims: dict[str, int],
        projection_cfg: dict[str, Any],
    ):
        super().__init__()
        self.branches = nn.ModuleDict(
            {
                name: NormformerProjection(
                    input_dim=latent_dim,
                    output_dim=dim,
                    **projection_cfg,
                )
                for name, dim in branch_dims.items()
                if dim > 0
            }
        )

    def forward(self, z: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        return {name: branch(z, mask) for name, branch in self.branches.items()}


class SplitPsi(nn.Module):
    """Map concatenated quantized branches back into decoder latent space."""

    def __init__(
        self,
        branch_dims: dict[str, int],
        latent_dim: int,
        projection_cfg: dict[str, Any],
    ):
        super().__init__()
        self.branch_names = [name for name, dim in branch_dims.items() if dim > 0]
        input_dim = sum(branch_dims[name] for name in self.branch_names)
        self.projection = NormformerProjection(
            input_dim=input_dim,
            output_dim=latent_dim,
            **projection_cfg,
        )

    def forward(
        self,
        branches: dict[str, torch.Tensor],
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z = torch.cat([branches[name] for name in self.branch_names], dim=-1)
        return self.projection(z, mask)


class SplitQuantizer(nn.Module):
    """Phi -> branch quantizers -> Psi bottleneck with VQ/FSQ branch support."""

    def __init__(self, feature_size: int, cfg: dict[str, Any]):
        super().__init__()
        self.feature_size = feature_size
        self.branch_order = list(cfg.get("branch_order", ["mu", "alpha"]))
        default_quantizer = cfg.get("quantizer", "fsq")

        projection_cfg = {
            "hidden_dim": cfg.get("projection_hidden_dim", feature_size),
            "num_heads": cfg.get("num_heads", 1),
            "num_blocks": cfg.get("num_blocks", 1),
            "mlp_expansion_factor": cfg.get("mlp_expansion_factor", 4),
            "dropout_rate": cfg.get("dropout_rate", 0.0),
        }

        self.branch_dims = {
            branch: self._branch_dim(branch, cfg, default_quantizer)
            for branch in self.branch_order
        }
        self.active_branches = [
            branch for branch, dim in self.branch_dims.items() if dim > 0
        ]
        self.bypass_single_branch_projection = bool(
            cfg.get("bypass_single_branch_projection", False)
        )
        if self.bypass_single_branch_projection:
            if len(self.active_branches) != 1:
                raise ValueError(
                    "bypass_single_branch_projection requires exactly one active branch; "
                    f"got {self.active_branches}."
                )
            active_branch = self.active_branches[0]
            if self.branch_dims[active_branch] != feature_size:
                raise ValueError(
                    "bypass_single_branch_projection requires the active branch dimension "
                    f"to match the latent dimension: branch {active_branch!r} has "
                    f"dim={self.branch_dims[active_branch]}, latent dim={feature_size}."
                )
        configured_loss_weights = cfg.get("branch_loss_weights", {}) or {}
        self.branch_loss_weights = {
            branch: float(configured_loss_weights.get(branch, 1.0))
            for branch in self.branch_order
        }
        if sum(self.branch_dims.values()) <= 0:
            raise ValueError("At least one split quantizer branch must have positive dimension")

        self.phi = (
            None
            if self.bypass_single_branch_projection
            else SplitPhi(feature_size, self.branch_dims, projection_cfg)
        )
        self.psi = (
            None
            if self.bypass_single_branch_projection
            else SplitPsi(self.branch_dims, feature_size, projection_cfg)
        )
        self.quantizers = nn.ModuleDict(
            {
                branch: self._build_branch_quantizer(branch, cfg, default_quantizer)
                for branch, dim in self.branch_dims.items()
                if dim > 0
            }
        )
        self.branch_num_codes = {
            branch: int(quantizer.num_codes) for branch, quantizer in self.quantizers.items()
        }
        self.num_codes = 1
        for branch in self.branch_order:
            self.num_codes *= self.branch_num_codes.get(branch, 1)

    def _branch_quantizer_type(
        self,
        branch: str,
        cfg: dict[str, Any],
        default_quantizer: str,
    ) -> str:
        return cfg.get(f"{branch}_quantizer") or default_quantizer

    def _branch_dim(self, branch: str, cfg: dict[str, Any], default_quantizer: str) -> int:
        quantizer_type = self._branch_quantizer_type(branch, cfg, default_quantizer)
        if quantizer_type == "fsq":
            return len(cfg.get(f"fsq_{branch}_levels", []))
        if quantizer_type == "vq":
            return int(cfg.get(f"vq_{branch}_dim", 0))
        raise ValueError(f"Unsupported {branch} quantizer: {quantizer_type}")

    def _build_branch_quantizer(
        self,
        branch: str,
        cfg: dict[str, Any],
        default_quantizer: str,
    ) -> nn.Module:
        quantizer_type = self._branch_quantizer_type(branch, cfg, default_quantizer)
        if quantizer_type == "fsq":
            return FSQ(
                cfg[f"fsq_{branch}_levels"],
                loss_use_half_width=bool(cfg.get("fsq_loss_use_half_width", False)),
                # Saved configs from before the mapping correction do not have
                # this key. Keep those checkpoints and their tokens on the
                # exact historical mapping.
                mapping=cfg.get("fsq_mapping", "legacy"),
            )
        if quantizer_type == "vq":
            vq_kwargs = dict(cfg.get("vq_kwargs", {}))
            vq_kwargs["num_codes"] = int(cfg[f"vq_{branch}_num_codes"])
            return VQBranch(feature_size=self.branch_dims[branch], vq_kwargs=vq_kwargs)
        raise ValueError(f"Unsupported {branch} quantizer: {quantizer_type}")

    def _combine_codes(self, branch_codes: dict[str, torch.Tensor]) -> torch.Tensor:
        combined = None
        stride = 1
        for branch in self.branch_order:
            codes = branch_codes.get(branch)
            if codes is None:
                continue
            combined = codes * stride if combined is None else combined + codes * stride
            stride *= self.branch_num_codes[branch]
        if combined is None:
            raise ValueError("No branch codes were produced")
        return combined

    def split_combined_codes(self, codes: torch.Tensor) -> dict[str, torch.Tensor]:
        """Split combined compatibility tokens into per-branch token IDs."""
        codes = codes.long()
        if codes.ndim == 3 and codes.shape[-1] == 1:
            codes = codes.squeeze(-1)
        branch_codes = {}
        residual = codes
        for branch in self.branch_order:
            if branch not in self.branch_num_codes:
                continue
            num_codes = self.branch_num_codes[branch]
            branch_codes[branch] = residual % num_codes
            residual = torch.div(residual, num_codes, rounding_mode="floor")
        return branch_codes

    def decode_tokens(
        self,
        branch_codes: dict[str, torch.Tensor],
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode branch tokens through branch inverses and Psi."""
        missing_branches = [
            branch for branch in self.quantizers if branch not in branch_codes
        ]
        if missing_branches:
            raise ValueError(f"Missing split token branches: {missing_branches}")

        quantized_branches = {
            branch: self.quantizers[branch].decode_tokens(codes)
            for branch, codes in branch_codes.items()
            if branch in self.quantizers
        }
        if self.bypass_single_branch_projection:
            z_q = quantized_branches[self.active_branches[0]]
            if mask is not None:
                z_q = z_q * mask.unsqueeze(-1)
            return z_q
        return self.psi(quantized_branches, mask)

    def forward(
        self,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if self.bypass_single_branch_projection:
            branch_latents = {self.active_branches[0]: z}
        else:
            branch_latents = self.phi(z, mask)
        quantized_branches = {}
        branch_codes = {}
        branch_losses = {}
        branch_weighted_losses = {}
        mask_values = None if mask is None else mask.to(dtype=z.dtype)
        mask_features = None if mask_values is None else mask_values.unsqueeze(-1)

        def reduce_loss(loss: torch.Tensor) -> torch.Tensor:
            if mask_values is None:
                return loss.mean()
            loss_mask = mask_values
            while loss_mask.ndim < loss.ndim:
                loss_mask = loss_mask.unsqueeze(-1)
            return (loss * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)

        for branch, quantizer in self.quantizers.items():
            branch_input = branch_latents[branch]
            if mask_features is not None:
                branch_input = branch_input * mask_features
            z_q, codes, loss = quantizer(branch_input)
            if mask_features is not None:
                z_q = z_q * mask_features
            quantized_branches[branch] = z_q
            branch_codes[branch] = codes
            branch_loss = reduce_loss(loss)
            branch_losses[branch] = branch_loss
            branch_weighted_losses[branch] = (
                branch_loss * self.branch_loss_weights.get(branch, 1.0)
            )

        if self.bypass_single_branch_projection:
            z_q = quantized_branches[self.active_branches[0]]
        else:
            z_q = self.psi(quantized_branches, mask)
        loss = sum(branch_weighted_losses.values())
        return z_q, {
            "z_q": z_q,
            "q": self._combine_codes(branch_codes),
            "loss": loss,
            "branch_q": branch_codes,
            "branch_loss": branch_losses,
            "branch_loss_weighted": branch_weighted_losses,
            "branch_loss_weights": self.branch_loss_weights,
            "branch_num_codes": self.branch_num_codes,
        }


def build_quantizer(
    feature_size: int,
    vq_kwargs: dict[str, Any],
    split_quantizer_cfg: dict[str, Any] | None = None,
    residual_quantizer_cfg: dict[str, Any] | None = None,
) -> nn.Module:
    """Build the configured single, residual, or split quantizer."""
    if residual_quantizer_cfg and split_quantizer_cfg:
        raise ValueError("Residual and split quantizers cannot be enabled together")
    if residual_quantizer_cfg:
        return ResidualVectorQuantizer(
            feature_size=feature_size,
            **dict(residual_quantizer_cfg),
        )
    if not split_quantizer_cfg or split_quantizer_cfg.get("mode", "vq") == "vq":
        return VectorQuant(feature_size=feature_size, **vq_kwargs)
    if split_quantizer_cfg["mode"] == "split":
        return SplitQuantizer(feature_size=feature_size, cfg=split_quantizer_cfg)
    raise ValueError(f"Unsupported quantizer mode: {split_quantizer_cfg['mode']}")
