"""Quantizer abstractions for VQ-VAE latent bottlenecks."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from vqtorch.nn import VectorQuant

from gabbro.models.transformer import NormformerStack


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
) -> nn.Module:
    """Build either the single VectorQuant layer or the split quantizer."""
    if not split_quantizer_cfg or split_quantizer_cfg.get("mode", "vq") == "vq":
        return VectorQuant(feature_size=feature_size, **vq_kwargs)
    if split_quantizer_cfg["mode"] == "split":
        return SplitQuantizer(feature_size=feature_size, cfg=split_quantizer_cfg)
    raise ValueError(f"Unsupported quantizer mode: {split_quantizer_cfg['mode']}")
