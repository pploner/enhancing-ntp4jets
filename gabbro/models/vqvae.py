import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import awkward as ak
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import vector
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from vqtorch.nn import VectorQuant

from gabbro.models.latent_sequence import DirectPrefixLatentMasker, LatentSequenceCompressor
from gabbro.models.quantizers import SplitQuantizer, build_quantizer
from gabbro.models.transformer import MLP, NormformerStack, Transformer
from gabbro.plotting.utils import set_mpl_style
from gabbro.utils.arrays import (
    ak_pad,
    ak_select_and_preprocess,
    ak_to_np_stack,
    get_causal_mask,
    np_to_ak,
)
from gabbro.utils.pylogger import get_pylogger

vector.register_awkward()

logger = get_pylogger(__name__)


def _pid_recall_metrics(
    target: torch.Tensor,
    logits: torch.Tensor,
    class_names: list[str] | tuple[str, ...],
) -> dict[str, torch.Tensor]:
    prediction = logits.argmax(dim=-1)
    metrics = {}
    recalls = []
    for class_index, class_name in enumerate(class_names):
        class_mask = target == class_index
        if torch.any(class_mask):
            recall = (prediction[class_mask] == class_index).float().mean()
            recalls.append(recall)
            metrics[f"pid_recall/{class_name}"] = recall.detach()
    if recalls:
        metrics["pid_macro_recall"] = torch.stack(recalls).mean().detach()
    return metrics


def _pid_confusion_counts(target: torch.Tensor, logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Return integer PID counts with truth on rows and prediction on columns."""
    prediction = logits.argmax(dim=-1)
    encoded = target.to(torch.int64) * num_classes + prediction.to(torch.int64)
    return torch.bincount(encoded, minlength=num_classes**2).reshape(num_classes, num_classes)


def _accumulate_pid_confusion(
    module: L.LightningModule,
    key: str,
    counts: torch.Tensor,
) -> None:
    matrices = getattr(module, "_pid_confusion_matrices", None)
    if matrices is None:
        matrices = {}
        module._pid_confusion_matrices = matrices
    detached = counts.detach()
    matrices[key] = detached if key not in matrices else matrices[key] + detached


def _save_pid_confusion_artifacts(module: L.LightningModule, key: str) -> None:
    """Save one compact PID confusion matrix instead of 64 scalar dashboard panels."""
    counts = getattr(module, "_pid_confusion_matrices", {}).get(key)
    if counts is None:
        return
    if getattr(module.trainer, "world_size", 1) > 1:
        counts = module.all_gather(counts).sum(dim=0)
    if not module.trainer.is_global_zero:
        return

    counts_np = counts.detach().cpu().numpy().astype(np.int64)
    pid_class_names = tuple(
        getattr(module, "pid_class_names", None)
        or getattr(getattr(module, "model", None), "pid_class_names", ())
    )
    if len(pid_class_names) != counts_np.shape[0]:
        pid_class_names = tuple(f"class_{index}" for index in range(counts_np.shape[0]))
    row_totals = counts_np.sum(axis=1, keepdims=True)
    normalized = np.divide(
        counts_np,
        row_totals,
        out=np.zeros_like(counts_np, dtype=np.float64),
        where=row_totals > 0,
    )
    artifact_dir = Path(module.trainer.default_root_dir) / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    stem = f"pid_confusion_matrix_{key}"
    json_path = artifact_dir / f"{stem}.json"
    json_path.write_text(
        json.dumps(
            {
                "truth_labels": list(pid_class_names),
                "predicted_labels": list(pid_class_names),
                "counts": counts_np.tolist(),
                "row_normalized": normalized.tolist(),
            },
            indent=2,
        )
    )

    set_mpl_style()
    figure, axis = plt.subplots(figsize=(8.0, 6.8))
    image = axis.imshow(normalized, vmin=0.0, vmax=1.0, cmap="Blues")
    axis.set(
        xticks=np.arange(len(pid_class_names)),
        yticks=np.arange(len(pid_class_names)),
        xticklabels=pid_class_names,
        yticklabels=pid_class_names,
        xlabel="Predicted PID",
        ylabel="True PID",
        title=f"PID confusion matrix ({key.replace('_', ' ')})",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for row in range(counts_np.shape[0]):
        for column in range(counts_np.shape[1]):
            color = "white" if normalized[row, column] > 0.5 else "black"
            axis.text(
                column,
                row,
                f"{normalized[row, column]:.2f}\n({counts_np[row, column]})",
                ha="center",
                va="center",
                fontsize=7,
                color=color,
            )
    figure.colorbar(image, ax=axis, label="Fraction of true PID class")
    figure.tight_layout()
    image_path = artifact_dir / f"{stem}.png"
    figure.savefig(image_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    logger.info("Saved PID confusion matrix counts to %s and plot to %s", json_path, image_path)

    for lightning_logger in module.trainer.loggers:
        if isinstance(lightning_logger, L.pytorch.loggers.CometLogger):
            lightning_logger.experiment.log_image(
                str(image_path), name=f"{key}/pid_confusion_matrix", step=module.global_step
            )
        elif isinstance(lightning_logger, L.pytorch.loggers.WandbLogger):
            try:
                import wandb

                lightning_logger.experiment.log(
                    {f"{key}_plots/pid_confusion_matrix": wandb.Image(str(image_path))},
                    commit=False,
                )
            except Exception as exc:
                logger.warning("Failed to log PID confusion matrix to W&B: %s", exc)


class _BaselineVQLayer:
    """Small compatibility object exposing codebook size to plotting callbacks."""

    def __init__(self, num_codes: int):
        self.num_codes = int(num_codes)


class _BaselineModel:
    """Small compatibility object matching the callback's model.vqlayer lookup."""

    def __init__(self, num_codes: int):
        self.vqlayer = _BaselineVQLayer(num_codes)
        self.conditional_dim = 0


class DumbQuantizationBaselineLightning(L.LightningModule):
    """Test-only baseline that scalar-quantizes transformed ORBIT inputs."""

    def __init__(
        self,
        q_levels: list[int],
        binning: str = "uniform",
        eta_range: tuple[float, float] = (-1.0, 1.0),
        phi_range: tuple[float, float] = (-float(np.pi), float(np.pi)),
        pt_range: tuple[float, float] | None = None,
        pt_range_num_train_batches: int = 1000,
        learned_bins_num_train_batches: int = 1000,
        learned_bins_max_iters: int = 100,
        learned_bins_tol: float = 1e-6,
        max_validation_plot_batches: int | None = 1,
        max_test_plot_batches: int | None = None,
        pid_cfg: dict | None = None,
        **_,
    ):
        super().__init__()
        if len(q_levels) != 3:
            raise ValueError(f"q_levels must have three entries for eta/phi/pT, got {q_levels}")
        if any(int(level) < 2 for level in q_levels):
            raise ValueError(f"All q_levels must be >= 2, got {q_levels}")
        if binning not in ("uniform", "learned"):
            raise ValueError(f"Unknown dumb-baseline binning={binning!r}.")
        self.save_hyperparameters(logger=False)
        self.q_levels = [int(level) for level in q_levels]
        self.binning = binning
        self.eta_range = tuple(float(value) for value in eta_range)
        self.phi_range = tuple(float(value) for value in phi_range)
        self.pt_range = None if pt_range is None else tuple(float(value) for value in pt_range)
        self.pt_range_num_train_batches = int(pt_range_num_train_batches)
        self.learned_bins_num_train_batches = int(learned_bins_num_train_batches)
        self.learned_bins_max_iters = int(learned_bins_max_iters)
        self.learned_bins_tol = float(learned_bins_tol)
        self.max_validation_plot_batches = max_validation_plot_batches
        self.max_test_plot_batches = max_test_plot_batches
        pid_cfg = {} if pid_cfg is None else dict(pid_cfg)
        self.pid_enabled = bool(pid_cfg.get("enabled", False))
        self.pid_num_classes = int(pid_cfg.get("num_classes", 8))
        self.pid_class_names = tuple(
            pid_cfg.get("class_names")
            or [f"class_{index}" for index in range(self.pid_num_classes)]
        )
        self.pid_loss_weight = float(pid_cfg.get("loss_weight", 1.0))
        self.pid_feature_scale = float(pid_cfg.get("faiss_feature_scale", 1.0))
        if self.pid_num_classes != 8:
            raise ValueError("The ORBIT PID mapping requires exactly 8 classes")
        if self.pid_feature_scale <= 0:
            raise ValueError("pid_cfg.faiss_feature_scale must be positive")
        self.learned_centers: dict[str, torch.Tensor] | None = None
        self.model = _BaselineModel(
            int(np.prod(self.q_levels)) * (self.pid_num_classes if self.pid_enabled else 1)
        )
        self.model.pid_enabled = self.pid_enabled
        self.test_x_original = []
        self.test_x_reco = []
        self.test_mask = []
        self.test_labels = []
        self.test_suite_labels = []
        self.test_code_idx = []
        self.test_pid = []

    @staticmethod
    def _should_store_loop_batch(batch_idx: int, max_batches: int | None) -> bool:
        return max_batches is None or batch_idx < max_batches

    @staticmethod
    def _quantize_uniform(
        values: torch.Tensor,
        min_value: float,
        max_value: float,
        num_levels: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if max_value <= min_value:
            raise ValueError(f"Invalid quantization range: [{min_value}, {max_value}]")
        scaled = (values - min_value) / (max_value - min_value)
        indices = torch.round(scaled * (num_levels - 1)).clamp(0, num_levels - 1).long()
        quantized = min_value + indices.to(values.dtype) * (max_value - min_value) / (
            num_levels - 1
        )
        return quantized, indices

    @staticmethod
    def _nearest_centers(
        values: torch.Tensor,
        centers: torch.Tensor,
        *,
        circular: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if circular:
            distance = torch.atan2(
                torch.sin(values.unsqueeze(-1) - centers),
                torch.cos(values.unsqueeze(-1) - centers),
            ).pow(2)
        else:
            distance = (values.unsqueeze(-1) - centers).pow(2)
        indices = torch.argmin(distance, dim=-1)
        quantized = centers.to(values.device, values.dtype)[indices]
        return quantized, indices

    @staticmethod
    def _circular_difference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(a - b), torch.cos(a - b))

    def _collect_train_quantization_values(self) -> dict[str, torch.Tensor]:
        if self.trainer is None or self.trainer.datamodule is None:
            raise ValueError("Learned bins require an attached datamodule with train data.")

        values = {"eta": [], "phi": [], "pt": []}
        dataloader = self.trainer.datamodule.train_dataloader()
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= self.learned_bins_num_train_batches:
                break
            x_particle = batch["part_features"]
            mask = batch["part_mask"].bool()
            if not torch.any(mask):
                continue
            values["eta"].append(x_particle[..., 0][mask].detach().cpu())
            values["phi"].append(
                torch.atan2(x_particle[..., 2], x_particle[..., 1])[mask].detach().cpu()
            )
            values["pt"].append(x_particle[..., 3][mask].detach().cpu())

        collected = {}
        for name, chunks in values.items():
            if not chunks:
                raise RuntimeError(f"Could not learn {name} bins: no valid train particles found.")
            collected[name] = torch.cat(chunks).float()
        return collected

    def _learn_scalar_centers(self, values: torch.Tensor, num_levels: int) -> torch.Tensor:
        values = values[torch.isfinite(values)]
        if values.numel() == 0:
            raise RuntimeError("Cannot learn scalar centers from an empty finite sample.")
        if torch.max(values) <= torch.min(values):
            return torch.full((num_levels,), float(values[0]), dtype=torch.float32)

        quantiles = torch.linspace(0.0, 1.0, num_levels, dtype=torch.float32)
        centers = torch.quantile(values, quantiles).float()
        for _ in range(self.learned_bins_max_iters):
            _, indices = self._nearest_centers(values, centers)
            new_centers = centers.clone()
            for idx in range(num_levels):
                assigned = values[indices == idx]
                if assigned.numel() > 0:
                    new_centers[idx] = assigned.mean()
            new_centers, _ = torch.sort(new_centers)
            if torch.max(torch.abs(new_centers - centers)) < self.learned_bins_tol:
                centers = new_centers
                break
            centers = new_centers
        return centers

    def _learn_circular_centers(self, values: torch.Tensor, num_levels: int) -> torch.Tensor:
        values = values[torch.isfinite(values)]
        if values.numel() == 0:
            raise RuntimeError("Cannot learn circular centers from an empty finite sample.")

        centers = torch.linspace(
            self.phi_range[0],
            self.phi_range[1],
            num_levels + 1,
            dtype=torch.float32,
        )[:-1]
        for _ in range(self.learned_bins_max_iters):
            _, indices = self._nearest_centers(values, centers, circular=True)
            new_centers = centers.clone()
            for idx in range(num_levels):
                assigned = values[indices == idx]
                if assigned.numel() > 0:
                    sin_mean = torch.mean(torch.sin(assigned))
                    cos_mean = torch.mean(torch.cos(assigned))
                    new_centers[idx] = torch.atan2(sin_mean, cos_mean)
            new_centers, _ = torch.sort(new_centers)
            delta = self._circular_difference(new_centers, centers)
            if torch.max(torch.abs(delta)) < self.learned_bins_tol:
                centers = new_centers
                break
            centers = new_centers
        return centers

    def _fit_learned_bins(self) -> None:
        if self.learned_centers is not None:
            return
        values = self._collect_train_quantization_values()
        self.learned_centers = {
            "eta": self._learn_scalar_centers(values["eta"], self.q_levels[0]),
            "phi": self._learn_circular_centers(values["phi"], self.q_levels[1]),
            "pt": self._learn_scalar_centers(values["pt"], self.q_levels[2]),
        }
        self.pt_range = (
            float(torch.min(values["pt"]).item()),
            float(torch.max(values["pt"]).item()),
        )
        logger.info(
            "Fitted learned dumb-baseline bins from %d particles.",
            values["eta"].numel(),
        )

    def _estimate_pt_range(self) -> tuple[float, float]:
        if self.pt_range is not None:
            return self.pt_range
        if self.trainer is None or self.trainer.datamodule is None:
            raise ValueError("pt_range was not provided and no datamodule is attached.")

        min_pt = None
        max_pt = None
        dataloader = self.trainer.datamodule.train_dataloader()
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= self.pt_range_num_train_batches:
                break
            x_particle = batch["part_features"]
            mask = batch["part_mask"].bool()
            if not torch.any(mask):
                continue
            pt_values = x_particle[..., 3][mask]
            batch_min = float(torch.min(pt_values).item())
            batch_max = float(torch.max(pt_values).item())
            min_pt = batch_min if min_pt is None else min(min_pt, batch_min)
            max_pt = batch_max if max_pt is None else max(max_pt, batch_max)

        if min_pt is None or max_pt is None:
            raise RuntimeError("Could not estimate pT range: no valid train particles found.")
        if max_pt <= min_pt:
            max_pt = min_pt + 1e-6
        self.pt_range = (float(min_pt), float(max_pt))
        return self.pt_range

    def _quantize_batch(
        self,
        x_particle: torch.Tensor,
        mask: torch.Tensor,
        pid_particle: torch.Tensor | None = None,
    ):
        eta = x_particle[..., 0]
        phi = torch.atan2(x_particle[..., 2], x_particle[..., 1])
        pt = x_particle[..., 3]

        if self.binning == "learned":
            self._fit_learned_bins()
            centers = {
                name: value.to(x_particle.device, x_particle.dtype)
                for name, value in self.learned_centers.items()
            }
            eta_q, eta_idx = self._nearest_centers(eta, centers["eta"])
            phi_q, phi_idx = self._nearest_centers(phi, centers["phi"], circular=True)
            pt_q, pt_idx = self._nearest_centers(pt, centers["pt"])
        else:
            eta_q, eta_idx = self._quantize_uniform(
                eta,
                self.eta_range[0],
                self.eta_range[1],
                self.q_levels[0],
            )
            phi_q, phi_idx = self._quantize_uniform(
                phi,
                self.phi_range[0],
                self.phi_range[1],
                self.q_levels[1],
            )
            pt_min, pt_max = self._estimate_pt_range()
            pt_q, pt_idx = self._quantize_uniform(pt, pt_min, pt_max, self.q_levels[2])

        x_reco = torch.stack(
            [
                eta_q,
                torch.cos(phi_q),
                torch.sin(phi_q),
                pt_q,
            ],
            dim=-1,
        )
        x_reco = x_reco * mask.unsqueeze(-1)
        code_idx = eta_idx + self.q_levels[0] * (
            phi_idx + self.q_levels[1] * pt_idx
        )
        pid_logits = None
        if self.pid_enabled:
            if pid_particle is None:
                raise ValueError("PID-enabled baseline requires a part_pid tensor")
            code_idx = code_idx * self.pid_num_classes + pid_particle.clamp_min(0)
            pid_logits = x_particle.new_full(
                (*pid_particle.shape, self.pid_num_classes),
                -20.0,
            )
            pid_logits.scatter_(-1, pid_particle.clamp_min(0).unsqueeze(-1), 20.0)
            pid_logits = pid_logits * mask.unsqueeze(-1)
        return x_reco, code_idx, pid_logits

    def model_step(self, batch, return_x=False):
        x_particle = batch["part_features"]
        mask_particle = batch["part_mask"]
        labels = batch["jet_type_labels"]
        pid_particle = batch.get("part_pid")
        x_particle_reco, code_idx, pid_logits = self._quantize_batch(
            x_particle,
            mask_particle,
            pid_particle,
        )

        valid_mask = mask_particle.unsqueeze(-1)
        reco_delta = (x_particle_reco - x_particle) * valid_mask
        n_valid_particles = torch.sum(mask_particle).clamp_min(1)
        n_valid_values = (n_valid_particles * x_particle.shape[-1]).clamp_min(1)
        reco_l2 = torch.sum(reco_delta**2) / n_valid_particles
        reco_l1 = torch.sum(torch.abs(reco_delta)) / n_valid_particles
        reco_l2_per_value = torch.sum(reco_delta**2) / n_valid_values
        reco_l1_per_value = torch.sum(torch.abs(reco_delta)) / n_valid_values
        pid_loss = reco_l2.new_zeros(())
        pid_accuracy = reco_l2.new_zeros(())
        if self.pid_enabled:
            valid_pid = pid_particle[mask_particle.bool()]
            valid_logits = pid_logits[mask_particle.bool()]
            pid_loss = F.cross_entropy(valid_logits, valid_pid)
            pid_accuracy = (valid_logits.argmax(dim=-1) == valid_pid).float().mean()
        loss = reco_l2 + self.pid_loss_weight * pid_loss
        metrics = {
            "loss_total": loss.detach(),
            "loss_reco": reco_l2.detach(),
            "loss_reco_l2": reco_l2.detach(),
            "loss_reco_l1": reco_l1.detach(),
            "loss_reco_l2_per_value": reco_l2_per_value.detach(),
            "loss_reco_l1_per_value": reco_l1_per_value.detach(),
            "loss_quantizer": torch.zeros_like(reco_l2).detach(),
            "loss_quantizer_weighted": torch.zeros_like(reco_l2).detach(),
            "loss_pid": pid_loss.detach(),
            "loss_pid_weighted": (self.pid_loss_weight * pid_loss).detach(),
            "pid_accuracy": pid_accuracy.detach(),
        }
        if self.pid_enabled:
            metrics.update(_pid_recall_metrics(valid_pid, valid_logits, self.pid_class_names))
            metrics["pid_confusion_matrix"] = _pid_confusion_counts(
                valid_pid, valid_logits, self.pid_num_classes
            )
        if return_x:
            return (
                loss,
                metrics,
                x_particle,
                x_particle_reco,
                mask_particle,
                labels,
                code_idx,
            )
        return loss, metrics

    def _log_step_metrics(
        self,
        prefix: str,
        metrics: dict[str, torch.Tensor],
        *,
        on_step: bool,
        on_epoch: bool,
        prog_bar: bool = False,
        pid_confusion_key: str | None = None,
    ) -> None:
        for name, value in metrics.items():
            if name == "pid_confusion_matrix":
                if pid_confusion_key is not None:
                    _accumulate_pid_confusion(self, pid_confusion_key, value)
                continue
            self.log(
                f"{prefix}/{name}",
                value,
                on_step=on_step,
                on_epoch=on_epoch,
                prog_bar=prog_bar and name == "loss_total",
            )

    def on_test_start(self) -> None:
        if self.binning == "learned":
            self._fit_learned_bins()
        pt_min, pt_max = self._estimate_pt_range()
        self.log(
            "baseline/binning_is_learned",
            float(self.binning == "learned"),
            on_step=False,
            on_epoch=True,
        )
        self.log("baseline/pt_min", pt_min, on_step=False, on_epoch=True)
        self.log("baseline/pt_max", pt_max, on_step=False, on_epoch=True)
        self.log("baseline/num_codes", self.model.vqlayer.num_codes, on_step=False, on_epoch=True)

    def on_test_epoch_start(self) -> None:
        logger.info("`on_test_epoch_start` called for dumb quantization baseline.")
        self.test_x_original = []
        self.test_x_reco = []
        self.test_mask = []
        self.test_labels = []
        self.test_suite_labels = []
        self.test_code_idx = []
        self.test_pid = []
        self._pid_confusion_matrices = {}
        self._clear_concat_outputs("test")

    def on_test_epoch_end(self) -> None:
        for key in getattr(self, "_pid_confusion_matrices", {}):
            _save_pid_confusion_artifacts(self, key)

    def _clear_concat_outputs(self, prefix: str) -> None:
        for name in [
            "x_original", "x_reco", "mask", "labels", "suite_labels", "code_idx", "code_mask",
            "pid",
        ]:
            attr = f"{prefix}_{name}_concat"
            if hasattr(self, attr):
                delattr(self, attr)

    def concat_validation_loop_predictions(self) -> None:
        """Compatibility hook for OrbitPlottingCallback."""
        if not getattr(self, "val_x_original", None):
            logger.info("No stored validation batches available for plotting/evaluation.")
            return
        self.val_x_original_concat = np.concatenate(self.val_x_original)
        self.val_x_reco_concat = np.concatenate(self.val_x_reco)
        self.val_mask_concat = np.concatenate(self.val_mask)
        self.val_labels_concat = np.concatenate(self.val_labels)
        self.val_code_idx_concat = np.concatenate(self.val_code_idx)

    def concat_test_loop_predictions(self) -> None:
        """Compatibility hook for OrbitPlottingCallback."""
        if not self.test_x_original:
            logger.info("No stored test batches available for plotting/evaluation.")
            return
        self.test_x_original_concat = np.concatenate(self.test_x_original)
        self.test_x_reco_concat = np.concatenate(self.test_x_reco)
        self.test_mask_concat = np.concatenate(self.test_mask)
        self.test_labels_concat = np.concatenate(self.test_labels)
        self.test_suite_labels_concat = np.concatenate(self.test_suite_labels)
        self.test_code_idx_concat = np.concatenate(self.test_code_idx)
        if self.test_pid:
            self.test_pid_concat = np.concatenate(self.test_pid)

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        loss, metrics, x_original, x_reco, mask, labels, code_idx = self.model_step(
            batch,
            return_x=True,
        )
        if self._should_store_loop_batch(batch_idx, self.max_test_plot_batches):
            self.test_x_original.append(x_original.detach().cpu().numpy())
            self.test_x_reco.append(x_reco.detach().cpu().numpy())
            self.test_mask.append(mask.detach().cpu().numpy())
            self.test_labels.append(labels.detach().cpu().numpy())
            suite_labels = batch.get("test_suite_labels")
            if suite_labels is None:
                suite_labels = torch.full_like(labels, dataloader_idx)
            self.test_suite_labels.append(suite_labels.detach().cpu().numpy())
            self.test_code_idx.append(code_idx.detach().cpu().numpy())
            if "part_pid" in batch:
                self.test_pid.append(batch["part_pid"].detach().cpu().numpy())
        self.log("test_loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        self._log_step_metrics(
            "test_metrics",
            metrics,
            on_step=False,
            on_epoch=True,
            pid_confusion_key=f"test_suite_{dataloader_idx}",
        )

    def on_test_end(self):
        logger.info("`on_test_end` called for dumb quantization baseline.")
        self.concat_test_loop_predictions()


class FaissKMeansBaselineLightning(DumbQuantizationBaselineLightning):
    """Test-only baseline using a GPU FAISS centroid dictionary per particle."""

    def __init__(
        self,
        num_codes: int,
        fit_max_particles: int = 1_000_000,
        fit_max_batches: int = 1000,
        niter: int = 25,
        nredo: int = 1,
        seed: int = 12345,
        use_gpu: bool = True,
        codebook_path: str | Path | None = None,
        save_codebook: bool = True,
        upload_codebook_to_wandb: bool = True,
        max_validation_plot_batches: int | None = 0,
        max_test_plot_batches: int | None = None,
        pid_cfg: dict | None = None,
        **_,
    ):
        if int(num_codes) < 2:
            raise ValueError(f"num_codes must be >= 2, got {num_codes}")
        if int(fit_max_particles) < int(num_codes):
            raise ValueError(
                "fit_max_particles must be at least num_codes, got "
                f"{fit_max_particles} and {num_codes}"
            )
        if int(fit_max_batches) < 1:
            raise ValueError("fit_max_batches must be positive")
        if int(niter) < 1 or int(nredo) < 1:
            raise ValueError("niter and nredo must be positive")

        # Reuse the established test-loop storage and metric implementation.
        # Uniform scalar quantization is fully replaced by _quantize_batch.
        super().__init__(
            q_levels=[2, 2, 2],
            binning="uniform",
            pt_range=(0.0, 1.0),
            max_validation_plot_batches=max_validation_plot_batches,
            max_test_plot_batches=max_test_plot_batches,
            pid_cfg=pid_cfg,
        )
        self.save_hyperparameters(logger=False)
        self.num_codes = int(num_codes)
        self.fit_max_particles = int(fit_max_particles)
        self.fit_max_batches = int(fit_max_batches)
        self.niter = int(niter)
        self.nredo = int(nredo)
        self.seed = int(seed)
        self.use_gpu = bool(use_gpu)
        self.codebook_path = Path(codebook_path).expanduser() if codebook_path else None
        self.save_codebook = bool(save_codebook)
        self.upload_codebook_to_wandb = bool(upload_codebook_to_wandb)
        self.q_levels = None
        self.model = _BaselineModel(self.num_codes)
        self.centroids: np.ndarray | None = None
        self._faiss_index = None
        self._faiss_gpu_resources = None
        self._fit_metadata: dict[str, Any] = {}

    def _load_codebook(self, faiss, gpu_count: int) -> None:
        if self.codebook_path is None:
            raise RuntimeError("No FAISS codebook path was configured")
        if not self.codebook_path.is_file():
            raise FileNotFoundError(f"FAISS codebook does not exist: {self.codebook_path}")

        with np.load(self.codebook_path) as payload:
            if "centroids" not in payload:
                raise ValueError(
                    f"FAISS codebook {self.codebook_path} has no 'centroids' array"
                )
            centroids = np.asarray(payload["centroids"], dtype=np.float32)
        if centroids.ndim != 2 or centroids.shape[0] != self.num_codes:
            raise ValueError(
                "FAISS codebook centroid shape must be "
                f"({self.num_codes}, feature_dim), got {centroids.shape}"
            )
        if centroids.shape[1] < 3 or not np.all(np.isfinite(centroids)):
            raise ValueError("FAISS codebook centroids must be finite and two-dimensional")

        cpu_index = faiss.IndexFlatL2(int(centroids.shape[1]))
        cpu_index.add(np.ascontiguousarray(centroids))
        if self.use_gpu:
            self._faiss_gpu_resources = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(self._faiss_gpu_resources, 0, cpu_index)
        else:
            index = cpu_index

        metadata_path = self.codebook_path.with_name("faiss_kmeans_metadata.json")
        metadata = {}
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text())
            metadata_num_codes = metadata.get("num_codes")
            if metadata_num_codes is not None and int(metadata_num_codes) != self.num_codes:
                raise ValueError(
                    f"FAISS metadata reports {metadata_num_codes} codes, expected "
                    f"{self.num_codes}"
                )

        phi_radius = np.sqrt(centroids[:, 1] ** 2 + centroids[:, 2] ** 2)
        metadata.update(
            {
                "num_codes": self.num_codes,
                "feature_dim": int(centroids.shape[1]),
                "use_gpu": self.use_gpu,
                "faiss_gpu_count": gpu_count,
                "fit_seconds": 0.0,
                "final_objective": metadata.get("final_objective"),
                "fit_particles_total": int(metadata.get("fit_particles_total", 0)),
                "fit_particles_per_class": metadata.get("fit_particles_per_class", {}),
                "phi_radius_mean": float(np.mean(phi_radius)),
                "phi_radius_min": float(np.min(phi_radius)),
                "phi_radius_max": float(np.max(phi_radius)),
                "loaded_codebook_path": str(self.codebook_path.resolve()),
            }
        )
        self.centroids = np.ascontiguousarray(centroids)
        self._faiss_index = index
        self._fit_metadata = metadata
        logger.info("Loaded FAISS centroid dictionary from %s", self.codebook_path)

    @staticmethod
    def _class_fit_quotas(datamodule, total_particles: int) -> dict[int, tuple[str, int]]:
        class_to_label = dict(
            getattr(datamodule.hparams, "class_to_label", None)
            or getattr(datamodule, "_class_to_label", None)
            or {"all": 0}
        )
        class_specs = getattr(datamodule, "_class_specs", {})
        weighted_classes = []
        for class_name, label in sorted(class_to_label.items()):
            spec = class_specs.get(class_name, {})
            weight = float(spec.get("weight", 1.0))
            if weight > 0 and spec.get("max_train_events") != 0:
                weighted_classes.append((class_name, int(label), weight))
        if not weighted_classes:
            raise RuntimeError("No positively weighted training classes are available for fitting")

        total_weight = sum(weight for _, _, weight in weighted_classes)
        exact = [total_particles * weight / total_weight for _, _, weight in weighted_classes]
        quotas = [int(math.floor(value)) for value in exact]
        remainder = total_particles - sum(quotas)
        fractional_order = sorted(
            range(len(exact)),
            key=lambda index: (exact[index] - quotas[index], -index),
            reverse=True,
        )
        for index in fractional_order[:remainder]:
            quotas[index] += 1
        return {
            label: (class_name, quota)
            for (class_name, label, _), quota in zip(weighted_classes, quotas)
        }

    def _collect_fit_particles(self) -> tuple[np.ndarray, dict[str, int]]:
        if self.trainer is None or self.trainer.datamodule is None:
            raise ValueError("FAISS fitting requires an attached datamodule with train data")

        quotas = self._class_fit_quotas(
            self.trainer.datamodule,
            self.fit_max_particles,
        )
        chunks: dict[int, list[np.ndarray]] = {label: [] for label in quotas}
        counts = {label: 0 for label in quotas}
        dataloader = self.trainer.datamodule.train_dataloader()
        batches_seen = 0
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= self.fit_max_batches:
                break
            batches_seen += 1
            features = batch["part_features"]
            mask = batch["part_mask"].bool()
            if self.pid_enabled:
                pid = batch["part_pid"]
                pid_one_hot = F.one_hot(
                    pid.clamp_min(0),
                    self.pid_num_classes,
                ).to(features.dtype)
                features = torch.cat(
                    [features, self.pid_feature_scale * pid_one_hot],
                    dim=-1,
                )
            labels = batch["jet_type_labels"].long()
            for label, (_, quota) in quotas.items():
                needed = quota - counts[label]
                if needed <= 0:
                    continue
                class_mask = mask & (labels == label).unsqueeze(1)
                values = features[class_mask]
                values = values[torch.all(torch.isfinite(values), dim=-1)]
                if values.numel() == 0:
                    continue
                values_np = values[:needed].detach().cpu().numpy().astype(np.float32, copy=False)
                chunks[label].append(values_np)
                counts[label] += len(values_np)
            if all(counts[label] >= quota for label, (_, quota) in quotas.items()):
                break

        sample_chunks = [chunk for label_chunks in chunks.values() for chunk in label_chunks]
        if not sample_chunks:
            raise RuntimeError(
                "No finite valid training particles were available for FAISS fitting"
            )
        sample = np.ascontiguousarray(np.concatenate(sample_chunks), dtype=np.float32)
        if len(sample) < self.num_codes:
            raise RuntimeError(
                f"FAISS needs at least {self.num_codes} fit particles, found {len(sample)}"
            )

        class_counts = {
            quotas[label][0]: int(counts[label])
            for label in sorted(quotas)
        }
        for label, (class_name, quota) in quotas.items():
            if counts[label] < quota:
                logger.warning(
                    "FAISS fit quota for %s was not filled: %d/%d particles after %d batches.",
                    class_name,
                    counts[label],
                    quota,
                    batches_seen,
                )
        self._fit_metadata.update(
            {
                "fit_batches_seen": batches_seen,
                "fit_particles_total": int(len(sample)),
                "fit_particles_per_class": class_counts,
                "fit_particle_targets_per_class": {
                    class_name: int(quota)
                    for _, (class_name, quota) in sorted(quotas.items())
                },
            }
        )
        return sample, class_counts

    def _fit_faiss(self) -> None:
        if self._faiss_index is not None:
            return
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError(
                "FAISS k-means baseline requires a GPU-enabled faiss installation"
            ) from exc

        gpu_count = int(faiss.get_num_gpus())
        if self.use_gpu and gpu_count < 1:
            raise RuntimeError("use_gpu=true but FAISS did not detect any CUDA GPUs")

        if self.codebook_path is not None:
            self._load_codebook(faiss, gpu_count)
            return

        sample, class_counts = self._collect_fit_particles()
        max_points_per_centroid = max(1, math.ceil(len(sample) / self.num_codes))
        logger.info(
            "Fitting FAISS k-means with K=%d on %d particles (%s).",
            self.num_codes,
            len(sample),
            class_counts,
        )
        start = time.perf_counter()
        kmeans = faiss.Kmeans(
            d=sample.shape[1],
            k=self.num_codes,
            niter=self.niter,
            nredo=self.nredo,
            seed=self.seed,
            verbose=True,
            gpu=self.use_gpu,
            spherical=False,
            min_points_per_centroid=1,
            max_points_per_centroid=max_points_per_centroid,
        )
        kmeans.train(sample)
        fit_seconds = time.perf_counter() - start
        centroids = np.asarray(kmeans.centroids, dtype=np.float32).reshape(
            self.num_codes,
            sample.shape[1],
        )
        if not np.all(np.isfinite(centroids)):
            raise RuntimeError("FAISS produced non-finite centroids")

        objective = float(kmeans.obj[-1]) if len(kmeans.obj) else None
        phi_radius = np.sqrt(centroids[:, 1] ** 2 + centroids[:, 2] ** 2)
        self.centroids = np.ascontiguousarray(centroids)
        self._faiss_index = kmeans.index
        self._fit_metadata.update(
            {
                "num_codes": self.num_codes,
                "feature_dim": int(sample.shape[1]),
                "niter": self.niter,
                "nredo": self.nredo,
                "seed": self.seed,
                "use_gpu": self.use_gpu,
                "faiss_gpu_count": gpu_count,
                "max_points_per_centroid": max_points_per_centroid,
                "fit_seconds": fit_seconds,
                "final_objective": objective,
                "phi_radius_mean": float(np.mean(phi_radius)),
                "phi_radius_min": float(np.min(phi_radius)),
                "phi_radius_max": float(np.max(phi_radius)),
            }
        )
        logger.info(
            "Finished FAISS k-means in %.1f seconds; final objective=%s.",
            fit_seconds,
            objective,
        )
        self._save_codebook_artifacts()

    def _save_codebook_artifacts(self) -> None:
        if not self.save_codebook or self.centroids is None or not self.trainer.is_global_zero:
            return
        artifact_dir = Path(self.trainer.default_root_dir) / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        centroids_path = artifact_dir / "faiss_kmeans_centroids.npz"
        metadata_path = artifact_dir / "faiss_kmeans_metadata.json"
        np.savez_compressed(centroids_path, centroids=self.centroids)
        metadata_path.write_text(json.dumps(self._fit_metadata, indent=2, sort_keys=True))
        logger.info("Saved FAISS centroid dictionary to %s", centroids_path)

        if not self.upload_codebook_to_wandb:
            return
        for lightning_logger in self.trainer.loggers:
            if not isinstance(lightning_logger, L.pytorch.loggers.WandbLogger):
                continue
            try:
                import wandb

                run = lightning_logger.experiment
                artifact = wandb.Artifact(
                    name=f"{run.id}-faiss-kmeans-{self.num_codes}",
                    type="codebook",
                    description="GPU FAISS k-means particle centroid dictionary.",
                    metadata=self._fit_metadata,
                )
                artifact.add_file(str(centroids_path), name=centroids_path.name)
                artifact.add_file(str(metadata_path), name=metadata_path.name)
                run.log_artifact(artifact)
                logger.info("Uploaded FAISS centroid dictionary as a W&B artifact.")
            except Exception as exc:
                logger.warning("Failed to upload FAISS codebook artifact to W&B: %s", exc)

    def _quantize_batch(
        self,
        x_particle: torch.Tensor,
        mask: torch.Tensor,
        pid_particle: torch.Tensor | None = None,
    ):
        self._fit_faiss()
        valid_mask = mask.bool()
        x_reco = torch.zeros_like(x_particle)
        code_idx = torch.zeros(mask.shape, dtype=torch.long, device=mask.device)
        if not torch.any(valid_mask):
            return x_reco, code_idx

        search_features = x_particle
        if self.pid_enabled:
            if pid_particle is None:
                raise ValueError("PID-enabled FAISS baseline requires a part_pid tensor")
            pid_one_hot = F.one_hot(
                pid_particle.clamp_min(0),
                self.pid_num_classes,
            ).to(x_particle.dtype)
            search_features = torch.cat(
                [x_particle, self.pid_feature_scale * pid_one_hot],
                dim=-1,
            )
        valid = np.ascontiguousarray(
            search_features[valid_mask].detach().float().cpu().numpy(),
            dtype=np.float32,
        )
        if valid.shape[1] != self.centroids.shape[1]:
            raise ValueError(
                "FAISS codebook feature dimension does not match the current data: "
                f"{self.centroids.shape[1]} != {valid.shape[1]}"
            )
        _, indices = self._faiss_index.search(valid, 1)
        indices = indices[:, 0].astype(np.int64, copy=False)
        centroid_tensor = torch.as_tensor(
            self.centroids,
            dtype=x_particle.dtype,
            device=x_particle.device,
        )
        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=x_particle.device)
        selected_centroids = centroid_tensor[index_tensor]
        x_reco[valid_mask] = selected_centroids[:, : x_particle.shape[-1]]
        code_idx[valid_mask] = index_tensor
        pid_logits = None
        if self.pid_enabled:
            pid_values = (
                selected_centroids[:, x_particle.shape[-1] :] / self.pid_feature_scale
            ).clamp_min(0.0)
            pid_probabilities = pid_values / pid_values.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            pid_logits = x_particle.new_zeros(
                (*mask.shape, self.pid_num_classes)
            )
            pid_logits[valid_mask] = torch.log(pid_probabilities.clamp_min(1e-8))
        return x_reco, code_idx, pid_logits

    def on_test_start(self) -> None:
        self._fit_faiss()
        numeric_metrics = {
            "baseline/num_codes": self.num_codes,
            "baseline/faiss_fit_seconds": self._fit_metadata["fit_seconds"],
            "baseline/faiss_gpu_count": self._fit_metadata["faiss_gpu_count"],
            "baseline/fit_particles_total": self._fit_metadata["fit_particles_total"],
            "baseline/phi_radius_mean": self._fit_metadata["phi_radius_mean"],
            "baseline/phi_radius_min": self._fit_metadata["phi_radius_min"],
            "baseline/phi_radius_max": self._fit_metadata["phi_radius_max"],
        }
        if self._fit_metadata["final_objective"] is not None:
            numeric_metrics["baseline/faiss_final_objective"] = self._fit_metadata[
                "final_objective"
            ]
        for class_name, count in self._fit_metadata.get(
            "fit_particles_per_class", {}
        ).items():
            numeric_metrics[f"baseline/fit_particles/{class_name}"] = count
            numeric_metrics[f"baseline/fit_fraction/{class_name}"] = (
                count / self._fit_metadata["fit_particles_total"]
            )
        for name, value in numeric_metrics.items():
            self.log(name, float(value), on_step=False, on_epoch=True)

    def on_test_epoch_start(self) -> None:
        logger.info("`on_test_epoch_start` called for FAISS k-means baseline.")
        self.test_x_original = []
        self.test_x_reco = []
        self.test_mask = []
        self.test_labels = []
        self.test_suite_labels = []
        self.test_code_idx = []
        self.test_pid = []
        self._clear_concat_outputs("test")

    def on_test_end(self):
        logger.info("`on_test_end` called for FAISS k-means baseline.")
        self.concat_test_loop_predictions()


class VQVAEMLP(torch.nn.Module):
    def __init__(
        self,
        input_features_dict: Dict[str, Any] = None,
        input_dim: int = None,
        conditional_dim: int = 0,
        latent_dim=4,
        encoder_layers=None,
        decoder_layers=None,
        vq_kwargs={},
        split_quantizer_cfg=None,
        **kwargs,
    ):
        """Initializes the VQ-VAE model.

        Parameters
        ----------
        input_features_dict : Dict[str, Any]
            Dictionary containing the input features and their preprocessing information.
        input_dim : int, optional
            The dimension of the input data. If not provided, it is inferred from
            `input_features_dict`.
        conditional_dim : int, optional
            The dimension of the conditional data. The default is 0.
        codebook_size : int, optional
            The size of the codebook. The default is 8.
        embed_dim : int, optional
            The dimension of the embedding space. The default is 2.
        input_dim : int, optional
            The dimension of the input data. The default is 2.
        encoder_layers : list, optional
            List of integers representing the number of units in each encoder layer.
            If None, a default encoder with a single linear layer is used. The default is None.
        decoder_layers : list, optional
            List of integers representing the number of units in each decoder layer.
            If None, a default decoder with a single linear layer is used. The default is None.
        """

        super().__init__()
        self.vq_kwargs = vq_kwargs
        self.split_quantizer_cfg = split_quantizer_cfg
        self.embed_dim = latent_dim
        self.input_features_dict = input_features_dict
        if input_features_dict is not None:
            self.input_dim = len(self.input_features_dict)
        elif input_dim is not None:
            self.input_dim = input_dim
        else:
            raise ValueError("Either input_features_dict or input_dim must be provided.")
        self.conditional_dim = conditional_dim

        # --- Encoder --- #
        if encoder_layers is None:
            self.encoder = torch.nn.Linear(self.input_dim + self.conditional_dim, self.embed_dim)
        else:
            enc_layers = []
            enc_layers.append(
                torch.nn.Linear(self.input_dim + self.conditional_dim, encoder_layers[0])
            )
            enc_layers.append(torch.nn.ReLU())

            for i in range(len(encoder_layers) - 1):
                enc_layers.append(torch.nn.Linear(encoder_layers[i], encoder_layers[i + 1]))
                enc_layers.append(torch.nn.ReLU())
            enc_layers.append(torch.nn.Linear(encoder_layers[-1], self.embed_dim))

            self.encoder = torch.nn.Sequential(*enc_layers)

        # --- Vector-quantization layer --- #
        self.vqlayer = build_quantizer(
            feature_size=self.embed_dim,
            vq_kwargs=vq_kwargs,
            split_quantizer_cfg=split_quantizer_cfg,
        )

        # --- Decoder --- #
        if decoder_layers is None:
            self.decoder = torch.nn.Linear(self.embed_dim + self.conditional_dim, self.input_dim)
        else:
            dec_layers = []
            dec_layers.append(
                torch.nn.Linear(self.embed_dim + self.conditional_dim, decoder_layers[0])
            )
            dec_layers.append(torch.nn.ReLU())

            for i in range(len(decoder_layers) - 1):
                dec_layers.append(torch.nn.Linear(decoder_layers[i], decoder_layers[i + 1]))
                dec_layers.append(torch.nn.ReLU())
            dec_layers.append(torch.nn.Linear(decoder_layers[-1], self.input_dim))

            self.decoder = torch.nn.Sequential(*dec_layers)

        self.loss_history = []
        self.lr_history = []

    def forward(self, x, mask=None, x_conditional=None):
        # mask is there for compatibility with the transformer model
        if x_conditional is not None:
            x_conditional = x_conditional.unsqueeze(1).repeat(1, x.shape[1], 1)
            x = torch.cat([x, x_conditional], dim=-1) * mask.unsqueeze(-1)
        # encode
        z_embed = self.encoder(x)
        # quantize
        z_q2, vq_out = self.quantize(z_embed, mask=mask)
        if x_conditional is not None:
            z_q2 = torch.cat([z_q2, x_conditional], dim=-1) * mask.unsqueeze(-1)
        # decode
        x_reco = self.decoder(z_q2)
        return x_reco, vq_out

    def quantize(self, z_embed, mask=None):
        """Quantize latent embeddings with single VectorQuant or split Phi/Psi quantization."""
        if isinstance(self.vqlayer, SplitQuantizer):
            return self.vqlayer(z_embed, mask=mask)
        return self.vqlayer(z_embed)


class VQVAETransformer(torch.nn.Module):
    """This is basically just a re-factor of the VQVAETransformer class, but with more modular
    model components, making it easier to use some components in other models."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        conditional_dim: int = 0,
        num_heads: int = 8,
        num_blocks: int = 2,
        vq_kwargs: dict = {},
        split_quantizer_cfg: dict = None,
        causal_decoder: bool = False,
        max_sequence_len: int = 128,
        input_features_dict: Dict[str, Any] = None,
        input_dim: int = None,
        old_transformer_implementation: bool = False,
        in_out_proj_cfg: Dict[str, Any] = None,
        latent_proj_cfg: Dict[str, Any] = None,
        latent_sequence_compression: dict | None = None,
        quantization_enabled: bool = True,
        transformer_cfg: dict = None,
        pid_cfg: dict | None = None,
        **kwargs,
    ):
        super().__init__()

        self.loss_history = []
        self.lr_history = []

        self.vq_kwargs = vq_kwargs
        self.split_quantizer_cfg = split_quantizer_cfg
        self.input_features_dict = input_features_dict
        if input_features_dict is not None:
            self.input_dim = len(self.input_features_dict)
        elif input_dim is not None:
            self.input_dim = input_dim
        else:
            raise ValueError("Either input_features_dict or input_dim must be provided.")

        pid_cfg = {} if pid_cfg is None else dict(pid_cfg)
        self.pid_enabled = bool(pid_cfg.get("enabled", False))
        self.pid_num_classes = int(pid_cfg.get("num_classes", 8))
        self.pid_class_names = tuple(
            pid_cfg.get("class_names")
            or [f"class_{index}" for index in range(self.pid_num_classes)]
        )
        if self.pid_num_classes < 2:
            raise ValueError("pid_cfg.num_classes must be at least 2")
        reconstruction_class_weights = pid_cfg.get("reconstruction_class_weights")
        if reconstruction_class_weights is None:
            self.register_buffer("reconstruction_class_weights", None)
        else:
            reconstruction_class_weights = torch.as_tensor(
                reconstruction_class_weights, dtype=torch.float32
            )
            if reconstruction_class_weights.shape != (self.pid_num_classes,):
                raise ValueError(
                    "pid_cfg.reconstruction_class_weights must contain exactly "
                    f"{self.pid_num_classes} values"
                )
            if not torch.all(torch.isfinite(reconstruction_class_weights)) or torch.any(
                reconstruction_class_weights <= 0
            ):
                raise ValueError(
                    "pid_cfg.reconstruction_class_weights must be finite and positive"
                )
            self.register_buffer(
                "reconstruction_class_weights", reconstruction_class_weights
            )

        self.conditional_dim = conditional_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.causal_decoder = causal_decoder
        self.max_sequence_len = max_sequence_len
        self.old_transformer_implementation = old_transformer_implementation
        self.latent_sequence_compression_cfg = latent_sequence_compression
        self.quantization_enabled = bool(quantization_enabled)

        if transformer_cfg is None:  # old config from when this was not configurable
            transformer_cfg = {
                "attn_cfg": {
                    "num_heads": self.num_heads,
                    "dropout_rate": 0.0,
                    "norm_before": True,
                    "norm_after": False,
                },
                "mlp_cfg": {
                    "expansion_factor": 4,
                    "dropout_rate": 0.0,
                    "norm_before": True,
                    "activation": "GELU",
                },
                "residual_cfg": {"gate_type": "local", "init_value": 1.0},
            }

        # Model components:
        if in_out_proj_cfg is None:
            self.input_projection = nn.Linear(
                self.input_dim
                + (self.pid_num_classes if self.pid_enabled else 0)
                + self.conditional_dim,
                self.hidden_dim,
            )
        else:
            self.input_projection = MLP(
                input_dim=(
                    self.input_dim
                    + (self.pid_num_classes if self.pid_enabled else 0)
                    + self.conditional_dim
                ),
                hidden_dims=in_out_proj_cfg.get("hidden_dims"),
                output_dim=self.hidden_dim,
                activation=in_out_proj_cfg.get("activation", "GELU"),
            )

        if not self.old_transformer_implementation:
            self.encoder = Transformer(
                n_blocks=self.num_blocks,
                dim=self.hidden_dim,
                attn_cfg=transformer_cfg["attn_cfg"],
                mlp_cfg=transformer_cfg["mlp_cfg"],
                residual_cfg=transformer_cfg["residual_cfg"],
                norm_after_blocks=False,
            )
        else:
            self.encoder_normformer = NormformerStack(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                num_blocks=self.num_blocks,
                mlp_expansion_factor=1,
            )

        if latent_proj_cfg is None:
            self.latent_projection_in = nn.Linear(self.hidden_dim, self.latent_dim)
        else:
            self.latent_projection_in = MLP(
                input_dim=self.hidden_dim,
                hidden_dims=latent_proj_cfg.get("hidden_dims"),
                output_dim=self.latent_dim,
                activation=latent_proj_cfg.get("activation", "GELU"),
            )

        self.vqlayer = (
            build_quantizer(
                feature_size=self.latent_dim,
                vq_kwargs=vq_kwargs,
                split_quantizer_cfg=split_quantizer_cfg,
            )
            if self.quantization_enabled
            else None
        )
        self.latent_sequence_compressor = None
        if latent_sequence_compression and latent_sequence_compression.get("enabled", False):
            mode = latent_sequence_compression.get("mode", "learned_cross_attention")
            if mode == "learned_cross_attention":
                self.latent_sequence_compressor = LatentSequenceCompressor(
                    latent_dim=self.latent_dim,
                    max_sequence_len=self.max_sequence_len,
                    ratio=float(latent_sequence_compression.get("ratio", 1.0)),
                    min_tokens=int(latent_sequence_compression.get("min_tokens", 1)),
                    rounding=latent_sequence_compression.get("rounding", "ceil"),
                    num_heads=int(latent_sequence_compression.get("num_heads", self.num_heads)),
                    dropout_rate=float(latent_sequence_compression.get("dropout_rate", 0.0)),
                    query_residual=bool(latent_sequence_compression.get("query_residual", True)),
                )
            elif mode == "direct_prefix_masking":
                self.latent_sequence_compressor = DirectPrefixLatentMasker(
                    latent_dim=self.latent_dim,
                    max_sequence_len=self.max_sequence_len,
                    ratio=float(latent_sequence_compression.get("ratio", 1.0)),
                    min_tokens=int(latent_sequence_compression.get("min_tokens", 1)),
                    rounding=latent_sequence_compression.get("rounding", "ceil"),
                )
            else:
                raise ValueError(
                    "latent_sequence_compression.mode must be 'learned_cross_attention' "
                    f"or 'direct_prefix_masking', got {mode!r}"
                )

        if latent_proj_cfg is None:
            self.latent_projection_out = nn.Linear(
                self.latent_dim + self.conditional_dim, self.hidden_dim
            )
        else:
            self.latent_projection_out = MLP(
                input_dim=self.latent_dim + self.conditional_dim,
                hidden_dims=latent_proj_cfg.get("hidden_dims"),
                output_dim=self.hidden_dim,
                activation=latent_proj_cfg.get("activation", "GELU"),
            )

        if not self.old_transformer_implementation:
            self.decoder = Transformer(
                n_blocks=self.num_blocks,
                dim=self.hidden_dim,
                attn_cfg=transformer_cfg["attn_cfg"],
                mlp_cfg=transformer_cfg["mlp_cfg"],
                residual_cfg=transformer_cfg["residual_cfg"],
                norm_after_blocks=False,
            )

        else:
            raise ValueError("Old implementation of VQ-VAE is not supported anymore.")

        if in_out_proj_cfg is None:
            self.output_projection = nn.Linear(hidden_dim, self.input_dim)
        else:
            self.output_projection = MLP(
                input_dim=self.hidden_dim,
                hidden_dims=in_out_proj_cfg.get("hidden_dims"),
                output_dim=self.input_dim,
                activation=in_out_proj_cfg.get("activation", "GELU"),
            )
        self.pid_output_projection = (
            nn.Linear(hidden_dim, self.pid_num_classes) if self.pid_enabled else None
        )

    def encode(self, x, mask, x_conditional=None, pid=None):
        """Encode input to latent embeddings."""
        if self.pid_enabled:
            if pid is None:
                raise ValueError("PID-enabled model requires a part_pid tensor")
            if pid.shape != mask.shape:
                raise ValueError(
                    f"part_pid and mask must have the same shape, got {pid.shape} and {mask.shape}"
                )
            pid_one_hot = F.one_hot(pid.clamp(min=0), self.pid_num_classes).to(x.dtype)
            x = torch.cat([x, pid_one_hot], dim=-1) * mask.unsqueeze(-1)
        if x_conditional is not None:
            # x_conditional is of shape (B, C)
            # x is of shape (B, S, F)
            # --> repeat x_conditional to match the shape of x
            x_conditional = x_conditional.unsqueeze(1).repeat(1, x.shape[1], 1)
            x = torch.cat([x, x_conditional], dim=-1) * mask.unsqueeze(-1)

        x = self.input_projection(x)
        if not self.old_transformer_implementation:
            x = self.encoder(x, mask=mask)
        else:
            x = self.encoder_normformer(x, mask)
        z_embed = self.latent_projection_in(x) * mask.unsqueeze(-1)
        return z_embed, x_conditional

    def quantize(self, z_embed, mask=None):
        """Vector quantize the latent embeddings."""
        if not self.quantization_enabled:
            if mask is None:
                mask = torch.ones(z_embed.shape[:2], dtype=torch.bool, device=z_embed.device)
            return z_embed, {
                "z": z_embed,
                "z_q": z_embed,
                "q": torch.full(
                    mask.shape,
                    -1,
                    device=z_embed.device,
                    dtype=torch.long,
                ),
                "loss": z_embed.new_zeros(()),
                "quantization_bypassed": True,
            }
        if isinstance(self.vqlayer, SplitQuantizer):
            z, vq_out = self.vqlayer(z_embed, mask=mask)
            return z, vq_out
        if mask is None:
            return self.vqlayer(z_embed)

        mask_bool = mask.bool()
        z_q = torch.zeros_like(z_embed)
        z_vq = torch.zeros_like(z_embed)
        z_q_vq = torch.zeros_like(z_embed)

        if not mask_bool.any():
            q = torch.zeros(mask_bool.shape, device=z_embed.device, dtype=torch.long)
            return z_q, {"z": z_vq, "z_q": z_q_vq, "q": q}

        z_valid = z_embed[mask_bool]
        z_q_valid, vq_out_valid = self.vqlayer(z_valid)
        z_vq_valid = vq_out_valid.get("z", z_valid)
        z_q_vq_valid = vq_out_valid.get("z_q", z_q_valid)
        if z_vq_valid.shape == (*z_valid.shape[:-1], 1, z_valid.shape[-1]):
            z_vq_valid = z_vq_valid.squeeze(-2)
        if z_q_vq_valid.shape == (*z_q_valid.shape[:-1], 1, z_q_valid.shape[-1]):
            z_q_vq_valid = z_q_vq_valid.squeeze(-2)
        z_q[mask_bool] = z_q_valid
        z_vq[mask_bool] = z_vq_valid
        z_q_vq[mask_bool] = z_q_vq_valid

        q_valid = vq_out_valid["q"].long()
        if q_valid.ndim > 1 and q_valid.shape[-1] == 1:
            q_valid = q_valid.squeeze(-1)
        q = torch.zeros(
            (*mask_bool.shape, *q_valid.shape[1:]),
            device=z_embed.device,
            dtype=q_valid.dtype,
        )
        q[mask_bool] = q_valid

        vq_out = dict(vq_out_valid)
        vq_out.update({"z": z_vq, "z_q": z_q_vq, "q": q})
        return z_q, vq_out

    def decode(self, z, mask, x_conditional=None, return_pid_logits: bool = False):
        """Decode quantized latents to reconstructed output."""
        if x_conditional is not None:
            z = torch.cat([z, x_conditional], dim=-1) * mask.unsqueeze(-1)

        x_reco = self.latent_projection_out(z) * mask.unsqueeze(-1)
        if not self.old_transformer_implementation:
            if self.causal_decoder:
                attn_mask = (
                    get_causal_mask(x_reco, fill_value=float("-inf"))
                    .to(x_reco.device)
                    .unsqueeze(-1)
                )
            else:
                attn_mask = None
            x_reco = self.decoder(x_reco, mask=mask, attn_mask=attn_mask)
        else:
            x_reco = self.decoder_normformer(x_reco, mask)
        decoded = x_reco
        x_reco = self.output_projection(decoded) * mask.unsqueeze(-1)
        if return_pid_logits:
            if self.pid_output_projection is None:
                raise ValueError("PID logits requested from a PID-disabled model")
            pid_logits = self.pid_output_projection(decoded) * mask.unsqueeze(-1)
            return x_reco, pid_logits
        return x_reco

    def forward(self, x, mask, x_conditional=None, pid=None):
        """Forward pass through encode, quantize, and decode."""
        z_embed, x_conditional_repeated = self.encode(x, mask, x_conditional, pid=pid)
        if self.latent_sequence_compressor is not None:
            z_to_quantize, latent_mask = self.latent_sequence_compressor.compress(
                z_embed,
                mask,
            )
            z_quantized, vq_out = self.quantize(z_to_quantize, mask=latent_mask)
            vq_out["latent_mask"] = latent_mask
            vq_out["particle_mask"] = mask
            z = self.latent_sequence_compressor.expand(
                z_quantized,
                latent_mask,
                target_len=x.shape[1],
                particle_mask=mask,
            )
        else:
            z, vq_out = self.quantize(z_embed, mask=mask)
            vq_out["latent_mask"] = mask
            vq_out["particle_mask"] = mask
        decoded = self.decode(
            z,
            mask,
            x_conditional_repeated,
            return_pid_logits=self.pid_enabled,
        )
        if self.pid_enabled:
            x_reco, pid_logits = decoded
            vq_out["pid_logits"] = pid_logits
        else:
            x_reco = decoded
        return x_reco, vq_out


class VQVAELightning(L.LightningModule):
    """PyTorch Lightning module for training a VQ-VAE."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
        model_kwargs={},
        model_type="Transformer",
        max_validation_plot_batches: int | None = 1,
        max_test_plot_batches: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        # --------------- load pretrained model --------------- #
        if model_type == "MLP":
            self.model = VQVAEMLP(**model_kwargs)
        elif model_type in [
            "VQVAETransformer",
            "VQVAENormFormer",  # <- for backwards compatibility with old models
        ]:
            self.model = VQVAETransformer(**model_kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.train_loss_history = []
        self.val_loss_list = []

        self.validation_cnt = 0
        self.validation_output = {}

        # for tracking best so far validation accuracy
        self.val_x_original = []
        self.val_x_reco = []
        self.val_mask = []

    @staticmethod
    def _should_store_loop_batch(batch_idx: int, max_batches: int | None) -> bool:
        return max_batches is None or batch_idx < max_batches

    @staticmethod
    def _class_balanced_plot_selector(
        labels: torch.Tensor,
        stored_batches_per_class: dict[int, int],
        max_batches_per_class: int | None,
    ) -> torch.Tensor:
        """Select up to ``max_batches_per_class`` validation batches per class."""
        if max_batches_per_class is None:
            return torch.ones_like(labels, dtype=torch.bool)

        selector = torch.zeros_like(labels, dtype=torch.bool)
        if max_batches_per_class <= 0:
            return selector

        for label_tensor in torch.unique(labels):
            label = int(label_tensor.item())
            if stored_batches_per_class.get(label, 0) >= max_batches_per_class:
                continue
            class_selector = labels == label_tensor
            if torch.any(class_selector):
                selector |= class_selector
                stored_batches_per_class[label] = stored_batches_per_class.get(label, 0) + 1
        return selector

    def _clear_concat_outputs(self, prefix: str) -> None:
        for name in [
            "x_original", "x_reco", "mask", "labels", "suite_labels", "code_idx", "code_mask",
            "pid",
        ]:
            attr = f"{prefix}_{name}_concat"
            if hasattr(self, attr):
                delattr(self, attr)

    def forward(
        self,
        x_particle,
        mask_particle,
        x_conditional=None,
        pid_particle=None,
    ):
        x_particle_reco, vq_out = self.model(
            x_particle,
            mask=mask_particle,
            x_conditional=x_conditional,
            pid=pid_particle,
        )
        return x_particle_reco, vq_out

    def model_step(self, batch, return_x=False):
        """Perform a single model step on a batch of data."""

        # x_particle, mask_particle, labels = batch
        x_particle = batch["part_features"]
        x_jet = batch.get("jet_features", None)
        mask_particle = batch["part_mask"]
        labels = batch["jet_type_labels"]
        pid_particle = batch.get("part_pid")

        # print(f"conditional_dim = {self.model.conditional_dim}")
        x_particle_reco, vq_out = self.forward(
            x_particle=x_particle,
            mask_particle=mask_particle,
            x_conditional=x_jet if self.model.conditional_dim > 0 else None,
            pid_particle=pid_particle,
        )

        valid_mask = mask_particle.unsqueeze(-1)
        reco_delta = (x_particle_reco - x_particle) * valid_mask
        n_valid_particles = torch.sum(mask_particle).clamp_min(1)
        n_valid_values = (n_valid_particles * x_particle.shape[-1]).clamp_min(1)
        reco_l2 = torch.sum(reco_delta**2) / n_valid_particles
        reco_l1 = torch.sum(torch.abs(reco_delta)) / n_valid_particles
        reco_l2_per_value = torch.sum(reco_delta**2) / n_valid_values
        reco_l1_per_value = torch.sum(torch.abs(reco_delta)) / n_valid_values

        particle_weights = mask_particle.to(dtype=x_particle.dtype)
        reconstruction_class_weights = self.model.reconstruction_class_weights
        if reconstruction_class_weights is not None:
            if pid_particle is None:
                raise ValueError(
                    "PID-balanced reconstruction loss requires a part_pid tensor"
                )
            valid_pid = pid_particle[mask_particle.bool()]
            if torch.any(valid_pid < 0) or torch.any(valid_pid >= self.model.pid_num_classes):
                raise ValueError("Valid particles contain out-of-range PID class indices")
            particle_weights = torch.zeros_like(particle_weights)
            particle_weights[mask_particle.bool()] = reconstruction_class_weights[valid_pid]
        reconstruction_weight_sum = particle_weights.sum().clamp_min(1.0)
        reco_l2_weighted = (
            (reco_delta**2).sum(dim=-1) * particle_weights
        ).sum() / reconstruction_weight_sum
        reco_l1_weighted = (
            torch.abs(reco_delta).sum(dim=-1) * particle_weights
        ).sum() / reconstruction_weight_sum

        alpha = self.hparams["model_kwargs"]["alpha"]
        reconstruction_loss = self.hparams["model_kwargs"].get("reconstruction_loss", "l2")
        if reconstruction_loss == "l2":
            reco_loss = reco_l2_weighted
        elif reconstruction_loss == "l1":
            reco_loss = reco_l1_weighted
        else:
            raise ValueError(
                f"Unknown reconstruction_loss={reconstruction_loss!r}. "
                "Expected 'l1' or 'l2'."
            )
        quantization_bypassed = bool(vq_out.get("quantization_bypassed", False))
        is_split_quantizer = "branch_loss" in vq_out
        quantizer_mask = vq_out.get("latent_mask", mask_particle).to(device=x_particle.device)
        if quantization_bypassed:
            quantizer_loss = reco_loss.new_zeros(())
        elif is_split_quantizer:
            quantizer_loss = vq_out["loss"].mean()
        else:
            quantizer_loss_per_token = (
                (1.0 - self.model.vqlayer.beta)
                * (vq_out["z"] - vq_out["z_q"].detach()).pow(2).mean(dim=-1)
                + self.model.vqlayer.beta
                * (vq_out["z"].detach() - vq_out["z_q"]).pow(2).mean(dim=-1)
            )
            quantizer_loss_mask = quantizer_mask
            while quantizer_loss_mask.ndim < quantizer_loss_per_token.ndim:
                quantizer_loss_mask = quantizer_loss_mask.unsqueeze(-1)
            quantizer_loss = (
                quantizer_loss_per_token * quantizer_loss_mask
            ).sum() / quantizer_loss_mask.sum().clamp_min(1)
        quantizer_loss_weighted = (
            quantizer_loss
            if is_split_quantizer or quantization_bypassed
            else alpha * quantizer_loss
        )
        code_idx = vq_out["q"]
        pid_loss = reco_loss.new_zeros(())
        pid_accuracy = reco_loss.new_zeros(())
        if self.model.pid_enabled:
            if pid_particle is None:
                raise ValueError("PID-enabled model received a batch without part_pid")
            valid_pid = pid_particle[mask_particle.bool()]
            pid_logits = vq_out["pid_logits"][mask_particle.bool()]
            pid_loss = F.cross_entropy(pid_logits, valid_pid)
            pid_accuracy = (pid_logits.argmax(dim=-1) == valid_pid).float().mean()
        pid_loss_weight = float(
            (self.hparams["model_kwargs"].get("pid_cfg") or {}).get("loss_weight", 1.0)
        )
        pid_loss_weighted = pid_loss_weight * pid_loss
        loss = reco_loss + pid_loss_weighted + quantizer_loss_weighted
        metrics = {
            "loss_total": loss.detach(),
            "loss_reco": reco_loss.detach(),
            "loss_reco_l2": reco_l2.detach(),
            "loss_reco_l1": reco_l1.detach(),
            "loss_reco_l2_per_value": reco_l2_per_value.detach(),
            "loss_reco_l1_per_value": reco_l1_per_value.detach(),
            "loss_reco_l2_weighted": reco_l2_weighted.detach(),
            "loss_reco_l1_weighted": reco_l1_weighted.detach(),
            "loss_quantizer": quantizer_loss.detach(),
            "loss_quantizer_weighted": quantizer_loss_weighted.detach(),
            "loss_pid": pid_loss.detach(),
            "loss_pid_weighted": pid_loss_weighted.detach(),
            "pid_accuracy": pid_accuracy.detach(),
            "quantizer/enabled": x_particle.new_tensor(float(not quantization_bypassed)),
        }
        if self.model.pid_enabled:
            metrics.update(
                _pid_recall_metrics(valid_pid, pid_logits, self.model.pid_class_names)
            )
            metrics["pid_confusion_matrix"] = _pid_confusion_counts(
                valid_pid, pid_logits, self.model.pid_num_classes
            )
        particle_token_count = mask_particle.to(dtype=torch.float32).sum()
        latent_token_count = quantizer_mask.to(dtype=torch.float32).sum()
        latent_sequence_compressor = getattr(self.model, "latent_sequence_compressor", None)
        configured_ratio = (
            latent_sequence_compressor.ratio if latent_sequence_compressor is not None else 1.0
        )
        metrics.update(
            {
                "latent_sequence/enabled": x_particle.new_tensor(
                    float(latent_sequence_compressor is not None)
                ),
                "latent_sequence/configured_ratio": x_particle.new_tensor(configured_ratio),
                "latent_sequence/query_residual_enabled": x_particle.new_tensor(
                    float(
                        latent_sequence_compressor is not None
                        and getattr(latent_sequence_compressor, "query_residual", False)
                    )
                ),
                "latent_sequence/direct_prefix_masking_enabled": x_particle.new_tensor(
                    float(
                        latent_sequence_compressor is not None
                        and getattr(latent_sequence_compressor, "mode", None)
                        == "direct_prefix_masking"
                    )
                ),
                "latent_sequence/mean_particle_tokens": (
                    mask_particle.to(dtype=torch.float32).sum(dim=1).mean().detach()
                ),
                "latent_sequence/mean_latent_tokens": (
                    quantizer_mask.to(dtype=torch.float32).sum(dim=1).mean().detach()
                ),
                "latent_sequence/effective_token_ratio": (
                    latent_token_count / particle_token_count.clamp_min(1.0)
                ).detach(),
            }
        )
        if not is_split_quantizer and not quantization_bypassed:
            z_norm = vq_out["z"].norm(dim=-1)
            z_q_norm = vq_out["z_q"].norm(dim=-1)
            norm_ratio = z_q_norm / z_norm.clamp_min(1e-6)
            norm_mask = quantizer_mask.to(dtype=z_norm.dtype)
            while norm_mask.ndim < z_norm.ndim:
                norm_mask = norm_mask.unsqueeze(-1)
            norm_mask = norm_mask.expand_as(z_norm)
            norm_denom = norm_mask.sum().clamp_min(1.0)
            norm_ratio_masked = torch.where(
                norm_mask > 0,
                norm_ratio,
                torch.zeros_like(norm_ratio),
            )
            metrics.update(
                {
                    "vq_latent_mean_norm": (
                        z_norm * norm_mask
                    ).sum().detach()
                    / norm_denom,
                    "vq_quantized_mean_norm": (
                        z_q_norm * norm_mask
                    ).sum().detach()
                    / norm_denom,
                    "vq_norm_ratio_mean": (
                        norm_ratio * norm_mask
                    ).sum().detach()
                    / norm_denom,
                    "vq_norm_ratio_max": norm_ratio_masked.max().detach(),
                }
            )
        for branch, branch_loss in vq_out.get("branch_loss", {}).items():
            metrics[f"loss_quantizer_{branch}"] = branch_loss.detach()
        for branch, branch_loss in vq_out.get("branch_loss_weighted", {}).items():
            metrics[f"loss_quantizer_{branch}_weighted"] = branch_loss.detach()
        metrics.update(self._vq_codebook_norm_metrics())

        if return_x:
            return (
                loss,
                metrics,
                x_particle,
                x_particle_reco,
                mask_particle,
                labels,
                code_idx,
                quantizer_mask,
            )

        return loss, metrics

    def _vq_codebook_norm_metrics(self) -> dict[str, torch.Tensor]:
        """Return mean entry norms for learned VQ codebooks, skipping FSQ branches."""
        vqlayer = self.model.vqlayer
        metrics = {}

        def mean_codebook_norm(vq_layer: VectorQuant) -> torch.Tensor:
            codebook = vq_layer.get_codebook()
            return codebook.norm(dim=-1).mean().detach()

        if isinstance(vqlayer, VectorQuant):
            metrics["vq_codebook_mean_norm"] = mean_codebook_norm(vqlayer)
            return metrics

        if isinstance(vqlayer, SplitQuantizer):
            for branch, quantizer in vqlayer.quantizers.items():
                branch_vq = getattr(quantizer, "vq", None)
                if isinstance(branch_vq, VectorQuant):
                    metrics[f"vq_codebook_mean_norm_{branch}"] = mean_codebook_norm(branch_vq)

        return metrics

    def _log_step_metrics(
        self,
        prefix: str,
        metrics: dict[str, torch.Tensor],
        *,
        on_step: bool,
        on_epoch: bool,
        prog_bar: bool = False,
        pid_confusion_key: str | None = None,
    ) -> None:
        for name, value in metrics.items():
            if name == "pid_confusion_matrix":
                if pid_confusion_key is not None:
                    _accumulate_pid_confusion(self, pid_confusion_key, value)
                continue
            self.log(
                f"{prefix}/{name}",
                value,
                on_step=on_step,
                on_epoch=on_epoch,
                prog_bar=prog_bar and name == "loss_total",
            )

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set."""
        loss, metrics = self.model_step(batch)

        self.train_loss_history.append(loss.detach().item())
        self.log("train_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True)
        self._log_step_metrics(
            "train_metrics",
            metrics,
            on_step=True,
            on_epoch=True,
        )

        return loss

    def on_train_start(self) -> None:
        logger.info("`on_train_start` called.")
        datamodule_hparams = self.trainer.datamodule.hparams
        if "dataset_kwargs_common" in datamodule_hparams:
            self.preprocessing_dict = datamodule_hparams.dataset_kwargs_common.feature_dict
        else:
            self.preprocessing_dict = {
                feature: {} for feature in datamodule_hparams.selected_features
            }

    def on_train_epoch_start(self):
        logger.info(f"`on_train_epoch_start` called. Epoch {self.trainer.current_epoch} starting.")
        self.epoch_train_start_time = time.time()  # start timing the epoch

    def on_train_epoch_end(self):
        logger.info(f"`on_train_epoch_end` called. Epoch {self.trainer.current_epoch} finished.")
        self.epoch_train_end_time = time.time()
        if hasattr(self, "epoch_train_start_time"):
            duration = (self.epoch_train_end_time - self.epoch_train_start_time) / 60
            self.log(
                "epoch_train_duration_minutes",
                duration,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
            if self.train_loss_history:
                logger.info(
                    f"Epoch {self.trainer.current_epoch} finished in {duration:.1f} minutes. "
                    f"Current step: {self.global_step}. Current loss: {self.train_loss_history[-1]}. "
                    f"Rank: {self.global_rank}"
                )

    def on_train_end(self):
        logger.info("`on_train_end` called.")

    def on_validation_epoch_start(self) -> None:
        logger.info("`on_validation_epoch_start` called.")
        self.val_x_original = []
        self.val_x_reco = []
        self.val_mask = []
        self.val_labels = []
        self.val_code_idx = []
        self.val_code_mask = []
        self._validation_plot_batches_per_class = {}
        self._pid_confusion_matrices = {}
        self._clear_concat_outputs("val")

    def on_validation_epoch_end(self) -> None:
        for key in getattr(self, "_pid_confusion_matrices", {}):
            _save_pid_confusion_artifacts(self, key)

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, metrics, x_original, x_reco, mask, labels, code_idx, code_mask = self.model_step(
            batch,
            return_x=True,
        )

        # Keep a small sample from every class for expensive plotting/physics evaluation.
        plot_selector = self._class_balanced_plot_selector(
            labels,
            self._validation_plot_batches_per_class,
            self.hparams.get("max_validation_plot_batches"),
        )
        if torch.any(plot_selector):
            self.val_x_original.append(x_original[plot_selector].detach().cpu().numpy())
            self.val_x_reco.append(x_reco[plot_selector].detach().cpu().numpy())
            self.val_mask.append(mask[plot_selector].detach().cpu().numpy())
            self.val_labels.append(labels[plot_selector].detach().cpu().numpy())
            self.val_code_idx.append(code_idx[plot_selector].detach().cpu().numpy())
            self.val_code_mask.append(code_mask[plot_selector].detach().cpu().numpy())

        self.log("val_loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        self._log_step_metrics(
            "val_metrics",
            metrics,
            on_step=False,
            on_epoch=True,
            pid_confusion_key="val",
        )

        return loss

    def on_test_epoch_start(self) -> None:
        logger.info("`on_test_epoch_start` called.")
        self.test_x_original = []
        self.test_x_reco = []
        self.test_mask = []
        self.test_labels = []
        self.test_suite_labels = []
        self.test_code_idx = []
        self.test_code_mask = []
        self.test_pid = []
        self._pid_confusion_matrices = {}
        self._clear_concat_outputs("test")

    def on_test_epoch_end(self) -> None:
        for key in getattr(self, "_pid_confusion_matrices", {}):
            _save_pid_confusion_artifacts(self, key)

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        loss, metrics, x_original, x_reco, mask, labels, code_idx, code_mask = self.model_step(
            batch,
            return_x=True,
        )

        if self._should_store_loop_batch(batch_idx, self.hparams.get("max_test_plot_batches")):
            self.test_x_original.append(x_original.detach().cpu().numpy())
            self.test_x_reco.append(x_reco.detach().cpu().numpy())
            self.test_mask.append(mask.detach().cpu().numpy())
            self.test_labels.append(labels.detach().cpu().numpy())
            suite_labels = batch.get("test_suite_labels")
            if suite_labels is None:
                suite_labels = torch.full_like(labels, dataloader_idx)
            self.test_suite_labels.append(suite_labels.detach().cpu().numpy())
            self.test_code_idx.append(code_idx.detach().cpu().numpy())
            self.test_code_mask.append(code_mask.detach().cpu().numpy())
            if "part_pid" in batch:
                self.test_pid.append(batch["part_pid"].detach().cpu().numpy())

        self.log("test_loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        self._log_step_metrics(
            "test_metrics",
            metrics,
            on_step=False,
            on_epoch=True,
            pid_confusion_key=f"test_suite_{dataloader_idx}",
        )

    def tokenize_ak_array(
        self,
        ak_arr,
        pp_dict,
        ak_arr_jet=None,
        pp_dict_jet=None,
        batch_size=256,
        pad_length=128,
        hide_pbar=False,
    ):
        """Tokenize an awkward array of jets.

        Parameters
        ----------
        ak_arr : ak.Array
            Awkward array of jets, shape (N_jets, <var>, N_features).
        pp_dict : dict
            Dictionary with preprocessing information.
        ak_arr_jet : ak.Array
            Awkward array of jet-level features, shape (N_jets, N_features_jet).
        pp_dict_jet : dict
            Dictionary with preprocessing information for jet-level features.
        batch_size : int, optional
            Batch size for the evaluation loop. The default is 256.
        pad_length : int, optional
            Length to which the tokens are padded. The default is 128.
        hide_pbar : bool, optional
            Whether to hide the progress bar. The default is False.

        Returns
        -------
        ak.Array
            Awkward array with token IDs and the jet features if used. Split
            quantizers also include one "part_token_<branch>" field per branch.
        """

        # preprocess the ak_arrary
        ak_arr = ak_select_and_preprocess(ak_arr, pp_dict=pp_dict)
        ak_arr_padded, mask = ak_pad(ak_arr, maxlen=pad_length, return_mask=True)
        # convert to numpy
        arr = ak_to_np_stack(ak_arr_padded, names=pp_dict.keys())
        # convert to torch tensor
        x = torch.from_numpy(arr).float()
        mask = torch.from_numpy(mask.to_numpy()).float()

        if self.model.conditional_dim == 0:
            dataset = TensorDataset(x, mask)
            dataloader = DataLoader(dataset, batch_size=batch_size)
        else:
            print("Using jet-level features as conditional data.")
            ak_arr_jet_pp = ak_select_and_preprocess(ak_arr_jet, pp_dict=pp_dict_jet)
            np_arr_jet_pp = ak_to_np_stack(ak_arr_jet_pp, names=pp_dict_jet.keys())
            x_jet = torch.from_numpy(np_arr_jet_pp).float()
            dataset = TensorDataset(x, mask, x_jet)
            dataloader = DataLoader(dataset, batch_size=batch_size)

        codes = []
        branch_codes = {}
        z_qs = []

        with torch.no_grad():
            pbar = tqdm(dataloader) if not hide_pbar else dataloader
            for i, batch in enumerate(pbar):
                # move to device
                if self.model.conditional_dim == 0:
                    x_batch, mask_batch = batch
                    x_jet_batch = None
                else:
                    x_batch, mask_batch, x_jet_batch = batch
                    x_jet_batch = x_jet_batch.to(self.device)
                x_batch = x_batch.to(self.device)
                mask_batch = mask_batch.to(self.device)
                x_particle_reco, vq_out = self.forward(
                    x_batch, mask_batch, x_conditional=x_jet_batch
                )
                code = vq_out["q"]
                z_q = vq_out["z_q"]
                codes.append(code)
                for branch, branch_code in vq_out.get("branch_q", {}).items():
                    branch_codes.setdefault(branch, []).append(branch_code)
                z_qs.append(z_q)

        codes = torch.cat(codes, dim=0).detach().cpu().numpy()
        if codes.ndim == 2:
            codes = codes[..., np.newaxis]
        branch_codes = {
            branch: torch.cat(branch_values, dim=0).detach().cpu().numpy()
            for branch, branch_values in branch_codes.items()
        }
        branch_codes = {
            branch: values[..., np.newaxis] if values.ndim == 2 else values
            for branch, values in branch_codes.items()
        }
        z_qs = torch.cat(z_qs, dim=0).squeeze(-2).detach().cpu().numpy()
        mask = mask.detach().cpu().numpy()

        if isinstance(self.model.vqlayer, (VectorQuant, SplitQuantizer)):
            feature_names = ["part_token_id"]
        else:
            raise ValueError("Unknown quantizer type.")

        ak_arr_tokens = np_to_ak(codes, names=feature_names, mask=mask, dtype="int64")
        ak_arr_branch_tokens = {
            f"part_token_{branch}": np_to_ak(
                branch_code,
                names=[f"part_token_{branch}"],
                mask=mask,
                dtype="int64",
            )
            for branch, branch_code in branch_codes.items()
        }
        ak_arr_zqs = np_to_ak(z_qs, names=[f"z_q_{i}" for i in range(z_qs.shape[-1])], mask=mask)

        if self.model.conditional_dim == 0:
            dict_with_jet_features = {}
        else:
            dict_with_jet_features = ak_arr_jet

        ak_arr = ak.Array(
            {
                "part_token_id": ak_arr_tokens,
                **ak_arr_branch_tokens,
                "z_q": ak_arr_zqs,
            }
            | dict_with_jet_features
        )
        return ak_arr

    @staticmethod
    def _extract_token_field(tokens_ak, field_name):
        fields = getattr(tokens_ak, "fields", [])
        if field_name not in fields:
            return None
        token_array = tokens_ak[field_name]
        inner_fields = getattr(token_array, "fields", [])
        if len(inner_fields) == 0:
            return token_array
        if len(inner_fields) == 1 and field_name in inner_fields:
            return token_array[field_name]
        raise ValueError(
            f"Expected token field {field_name!r} to contain a single nested field "
            f"with the same name, got {inner_fields}."
        )

    @staticmethod
    def _pad_token_array(token_array, pad_length):
        padded_tokens, mask = ak_pad(token_array, maxlen=pad_length, return_mask=True)
        tokens = torch.from_numpy(padded_tokens.to_numpy()).long()
        mask = torch.from_numpy(mask.to_numpy()).float()
        return tokens, mask

    def _prepare_split_token_tensors(self, tokens_ak, pad_length):
        active_branches = [
            branch
            for branch in self.model.vqlayer.branch_order
            if branch in self.model.vqlayer.quantizers
        ]
        branch_arrays = {
            branch: self._extract_token_field(tokens_ak, f"part_token_{branch}")
            for branch in active_branches
        }

        if all(token_array is not None for token_array in branch_arrays.values()):
            branch_tensors = {}
            mask = None
            for branch, token_array in branch_arrays.items():
                branch_tensor, branch_mask = self._pad_token_array(token_array, pad_length)
                branch_tensors[branch] = branch_tensor
                mask = branch_mask if mask is None else mask
            return branch_tensors, mask

        combined_tokens = self._extract_token_field(tokens_ak, "part_token_id")
        if combined_tokens is None and len(getattr(tokens_ak, "fields", [])) == 0:
            combined_tokens = tokens_ak
        if combined_tokens is None:
            raise ValueError(
                "Split reconstruction needs explicit part_token_<branch> fields or "
                "a combined part_token_id field."
            )

        combined_tensor, mask = self._pad_token_array(combined_tokens, pad_length)
        return self.model.vqlayer.split_combined_codes(combined_tensor), mask

    def _reconstruct_split_ak_tokens(
        self,
        tokens_ak,
        pp_dict,
        jets_ak=None,
        pp_dict_jet=None,
        batch_size=256,
        pad_length=128,
        hide_pbar=False,
    ):
        branch_tensors, mask = self._prepare_split_token_tensors(tokens_ak, pad_length)

        if self.model.conditional_dim > 0:
            conditional_data = ak_select_and_preprocess(jets_ak, pp_dict=pp_dict_jet)
            conditional_data = ak_to_np_stack(conditional_data, names=pp_dict_jet.keys())
            x_conditional = torch.from_numpy(conditional_data).float()
            x_conditional = x_conditional.unsqueeze(1).repeat(1, mask.shape[1], 1)

        active_branches = [
            branch
            for branch in self.model.vqlayer.branch_order
            if branch in self.model.vqlayer.quantizers
        ]
        tensors = [branch_tensors[branch] for branch in active_branches] + [mask]
        if self.model.conditional_dim > 0:
            tensors.append(x_conditional)
        dataloader = DataLoader(TensorDataset(*tensors), batch_size=batch_size)

        x_reco = []
        with torch.no_grad():
            pbar = tqdm(dataloader) if not hide_pbar else dataloader
            for batch in pbar:
                branch_batch_values = batch[: len(active_branches)]
                mask_batch = batch[len(active_branches)].to(self.device)
                branch_batch = {
                    branch: values.to(self.device)
                    for branch, values in zip(active_branches, branch_batch_values)
                }
                z_q = self.model.vqlayer.decode_tokens(branch_batch, mask=mask_batch)

                if self.model.conditional_dim > 0:
                    x_conditional_batch = batch[-1].to(self.device)
                    z_q = torch.cat([z_q, x_conditional_batch], dim=-1) * mask_batch.unsqueeze(-1)

                x_reco.append(self.model.decode(z_q, mask=mask_batch))

        x_reco = torch.cat(x_reco, dim=0).detach().cpu().numpy()
        x_reco_ak = np_to_ak(x_reco, names=pp_dict.keys(), mask=mask.detach().cpu().numpy())
        return ak_select_and_preprocess(x_reco_ak, pp_dict, inverse=True)

    def reconstruct_ak_tokens(
        self,
        tokens_ak,
        pp_dict,
        jets_ak=None,
        pp_dict_jet=None,
        batch_size=256,
        pad_length=128,
        hide_pbar=False,
    ):
        """Reconstruct tokenized awkward array.

        Parameters
        ----------
        tokens_ak : ak.Array
            Awkward array of tokens, shape (N_jets, <var>).
        pp_dict : dict
            Dictionary with preprocessing information.
        jets_ak : ak.Array
            Awkward array of jet-level features, shape (N_jets, N_features_jet).
        pp_dict_jet : dict
            Dictionary with preprocessing information for jet-level features.
        batch_size : int, optional
            Batch size for the evaluation loop. The default is 256.
        pad_length : int, optional
            Length to which the tokens are padded. The default is 128.
        hide_pbar : bool, optional
            Whether to hide the progress bar. The default is False.

        Returns
        -------
        ak.Array
            Awkward array of reconstructed jets, shape (N_jets, <var>, N_features).
        """

        self.model.eval()

        if isinstance(self.model.vqlayer, SplitQuantizer):
            return self._reconstruct_split_ak_tokens(
                tokens_ak=tokens_ak,
                pp_dict=pp_dict,
                jets_ak=jets_ak,
                pp_dict_jet=pp_dict_jet,
                batch_size=batch_size,
                pad_length=pad_length,
                hide_pbar=hide_pbar,
            )

        tokens, mask = ak_pad(tokens_ak, maxlen=pad_length, return_mask=True)
        if len(tokens.fields) == 0:
            tokens = torch.from_numpy(tokens.to_numpy()).long()
        else:
            tokens = torch.from_numpy(ak_to_np_stack(tokens, names=tokens_ak.fields)).long()
        mask = torch.from_numpy(mask.to_numpy()).float()

        if self.model.conditional_dim > 0:
            conditional_data = ak_select_and_preprocess(jets_ak, pp_dict=pp_dict_jet)
            conditional_data = ak_to_np_stack(conditional_data, names=pp_dict_jet.keys())
            x_conditional = torch.from_numpy(conditional_data).float()
            # concatenate the conditional data to the tokens
            x_conditional = x_conditional.unsqueeze(1).repeat(1, tokens.shape[1], 1)

        x_reco = []
        if self.model.conditional_dim == 0:
            dataset = TensorDataset(tokens, mask)
            dataloader = DataLoader(dataset, batch_size=batch_size)
        else:
            dataset = TensorDataset(tokens, mask, x_conditional)
            dataloader = DataLoader(dataset, batch_size=batch_size)

        codebook = self.model.vqlayer.codebook.weight

        # if the codebook has an affine transform, apply it
        # before using it to reconstruct the data
        # see https://github.com/minyoungg/vqtorch/blob/main/vqtorch/nn/vq.py#L102-L104
        if hasattr(self.model.vqlayer, "affine_transform"):
            codebook = self.model.vqlayer.affine_transform(codebook)

        with torch.no_grad():
            pbar = tqdm(dataloader) if not hide_pbar else dataloader
            for i, batch in enumerate(pbar):
                # move to device
                if self.model.conditional_dim == 0:
                    tokens_batch, mask_batch = batch
                else:
                    tokens_batch, mask_batch, x_conditional_batch = batch
                    x_conditional_batch = x_conditional_batch.to(self.device)

                tokens_batch = tokens_batch.to(self.device)
                mask_batch = mask_batch.to(self.device)
                try:
                    z_q = F.embedding(tokens_batch, codebook)
                    z_q = z_q.squeeze(-2)
                except Exception as e:  # noqa: E722
                    logger.info(f"Error in embedding: {e}")
                    logger.info("batch shape", tokens_batch.shape)
                    logger.info("batch max", tokens_batch.max())
                    logger.info("batch min", tokens_batch.min())

                # print(f"z_q shape: {z_q.shape}")

                # if conditioning is used, concatenate the conditional data to the tokens
                if self.model.conditional_dim > 0:
                    z_q = torch.cat([z_q, x_conditional_batch], dim=-1) * mask_batch.unsqueeze(-1)

                if hasattr(self.model, "latent_projection_out"):
                    x_reco_batch = self.model.decode(z_q, mask=mask_batch)
                elif hasattr(self.model, "decoder"):
                    x_reco_batch = self.model.decoder(z_q)
                else:
                    raise ValueError("Unknown model structure. Cannot reconstruct.")
                x_reco.append(x_reco_batch)

        x_reco = torch.cat(x_reco, dim=0).detach().cpu().numpy()
        x_reco_ak = np_to_ak(x_reco, names=pp_dict.keys(), mask=mask.detach().cpu().numpy())
        x_reco_ak = ak_select_and_preprocess(x_reco_ak, pp_dict, inverse=True)

        return x_reco_ak

    def concat_validation_loop_predictions(self) -> None:
        if not self.val_x_original:
            logger.info("No stored validation batches available for plotting/evaluation.")
            return
        self.val_x_original_concat = np.concatenate(self.val_x_original)
        self.val_x_reco_concat = np.concatenate(self.val_x_reco)
        self.val_mask_concat = np.concatenate(self.val_mask)
        self.val_labels_concat = np.concatenate(self.val_labels)
        self.val_code_idx_concat = np.concatenate(self.val_code_idx)
        self.val_code_mask_concat = np.concatenate(self.val_code_mask)

    def on_validation_end(self) -> None:
        """Lightning hook that is called when a validation loop ends."""
        logger.info("`on_validation_end` called.")
        self.concat_validation_loop_predictions()

    def on_test_end(self):
        logger.info("`on_test_end` called.")
        self.concat_test_loop_predictions()

    def concat_test_loop_predictions(self) -> None:
        if not self.test_x_original:
            logger.info("No stored test batches available for plotting/evaluation.")
            return
        self.test_x_original_concat = np.concatenate(self.test_x_original)
        self.test_x_reco_concat = np.concatenate(self.test_x_reco)
        self.test_mask_concat = np.concatenate(self.test_mask)
        self.test_labels_concat = np.concatenate(self.test_labels)
        self.test_suite_labels_concat = np.concatenate(self.test_suite_labels)
        self.test_code_idx_concat = np.concatenate(self.test_code_idx)
        self.test_code_mask_concat = np.concatenate(self.test_code_mask)
        if self.test_pid:
            self.test_pid_concat = np.concatenate(self.test_pid)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configures optimizers and learning-rate schedulers to be used for training."""
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    **self.hparams.get("scheduler_lightning_kwargs", {}),
                },
            }

        return {"optimizer": optimizer}
