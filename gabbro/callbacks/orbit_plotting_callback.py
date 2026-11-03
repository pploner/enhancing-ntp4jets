"""Schema-neutral ORBIT plotting callback."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import lightning as L
import numpy as np
import awkward as ak
import fastjet
import vector
from scipy.optimize import linear_sum_assignment

from gabbro.plotting.orbit import (
    CodeBranchSpec,
    MINBIAS_PHYSICAL_FEATURE_RANGES,
    close_figure,
    code_entropy_metrics,
    collect_reconstruction_histograms,
    collect_physical_reconstruction_histograms,
    physical_reconstruction_plots,
    plot_codebook_histogram,
    plot_event_reconstruction_grid,
    plot_feature_histograms,
    plot_particle_count_histograms,
    plot_pid_conditional_residuals,
    plot_residual_histograms,
    pid_residual_arrays,
    pid_residual_distribution_summary,
    reconstruction_loss_metrics,
)
from gabbro.utils.pylogger import get_pylogger

logger = get_pylogger("OrbitPlottingCallback")
vector.register_awkward()


def _delta_r(particles, jets):
    jets = ak.unflatten(ak.flatten(jets), counts=1)
    return particles.deltaR(jets)


def match_jets_by_delta_r(
    true_eta: np.ndarray,
    true_phi: np.ndarray,
    reco_eta: np.ndarray,
    reco_phi: np.ndarray,
    max_delta_r: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return one-to-one jet indices minimizing total delta-R.

    With a cutoff, the assignment maximizes the number of valid pairs first and
    minimizes their total delta-R second. Passing ``None`` keeps the ordinary
    full Hungarian assignment of ``min(n_true, n_reco)`` pairs.
    """
    true_eta = np.asarray(true_eta, dtype=np.float64)
    true_phi = np.asarray(true_phi, dtype=np.float64)
    reco_eta = np.asarray(reco_eta, dtype=np.float64)
    reco_phi = np.asarray(reco_phi, dtype=np.float64)
    if not len(true_eta) or not len(reco_eta):
        empty_indices = np.empty(0, dtype=np.int64)
        return empty_indices, empty_indices.copy(), np.empty(0, dtype=np.float64)

    delta_eta = reco_eta[np.newaxis, :] - true_eta[:, np.newaxis]
    delta_phi = (
        np.remainder(
            reco_phi[np.newaxis, :] - true_phi[:, np.newaxis] + np.pi,
            2 * np.pi,
        )
        - np.pi
    )
    delta_r = np.hypot(delta_eta, delta_phi)
    if max_delta_r is None:
        true_indices, reco_indices = linear_sum_assignment(delta_r)
        matched_delta_r = delta_r[true_indices, reco_indices]
        accepted = np.ones_like(matched_delta_r, dtype=bool)
    else:
        n_true, n_reco = delta_r.shape
        size = n_true + n_reco
        unmatched_cost = (max_delta_r + 1.0) * (size + 1)
        forbidden_cost = unmatched_cost * (size + 1)
        cost = np.full((size, size), forbidden_cost, dtype=np.float64)
        cost[:n_true, :n_reco] = np.where(
            delta_r <= max_delta_r,
            delta_r,
            forbidden_cost,
        )
        cost[:n_true, n_reco:] = unmatched_cost
        cost[n_true:, :n_reco] = unmatched_cost
        cost[n_true:, n_reco:] = 0.0
        rows, columns = linear_sum_assignment(cost)
        accepted = (rows < n_true) & (columns < n_reco)
        true_indices = rows[accepted]
        reco_indices = columns[accepted]
        matched_delta_r = delta_r[true_indices, reco_indices]
        accepted = matched_delta_r <= max_delta_r
    return (
        true_indices[accepted].astype(np.int64, copy=False),
        reco_indices[accepted].astype(np.int64, copy=False),
        matched_delta_r[accepted],
    )


class OrbitPlottingCallback(L.Callback):
    """Plot masked reconstruction summaries for particle or jet ORBIT parquet batches."""

    def __init__(
        self,
        image_path: str | None = None,
        image_filetype: str = "png",
        save_histograms: bool = True,
        no_trainer_info_in_filename: bool = False,
        enable_physics_plots: bool = True,
        include_all_ratios: bool = True,
        jet_radius: float = 0.8,
        jet_min_pt: float = 0.0,
        include_tau32: bool = True,
        cache_subset_jet_results: bool = False,
        validation_plot_every_n_epochs: int = 1,
        include_codebook_histogram: bool = True,
        include_event_reconstruction_grid: bool = True,
        event_reconstruction_examples_per_class: int = 3,
        jet_matching_radius_fraction: float = 0.5,
        include_train_particle_count_histogram: bool = True,
        train_particle_count_max_events: int = 2048,
        include_pid_conditional_residuals: bool = True,
    ):
        super().__init__()
        self.image_path = image_path
        self.image_filetype = image_filetype
        self.save_histograms = save_histograms
        self.no_trainer_info_in_filename = no_trainer_info_in_filename
        self.enable_physics_plots = enable_physics_plots
        self.include_all_ratios = include_all_ratios
        self.jet_radius = jet_radius
        self.jet_min_pt = float(jet_min_pt)
        self.include_tau32 = bool(include_tau32)
        self.cache_subset_jet_results = bool(cache_subset_jet_results)
        self.validation_plot_every_n_epochs = int(validation_plot_every_n_epochs)
        self.include_codebook_histogram = include_codebook_histogram
        self.include_event_reconstruction_grid = include_event_reconstruction_grid
        self.event_reconstruction_examples_per_class = int(
            event_reconstruction_examples_per_class
        )
        self.jet_matching_radius_fraction = float(jet_matching_radius_fraction)
        self.include_train_particle_count_histogram = bool(
            include_train_particle_count_histogram
        )
        self.train_particle_count_max_events = int(train_particle_count_max_events)
        self.include_pid_conditional_residuals = bool(include_pid_conditional_residuals)
        if self.event_reconstruction_examples_per_class < 1:
            raise ValueError("event_reconstruction_examples_per_class must be positive")
        if self.jet_matching_radius_fraction <= 0:
            raise ValueError("jet_matching_radius_fraction must be positive")
        if self.jet_min_pt < 0:
            raise ValueError("jet_min_pt must be non-negative")
        if self.validation_plot_every_n_epochs < 1:
            raise ValueError("validation_plot_every_n_epochs must be positive")
        if self.train_particle_count_max_events < 1:
            raise ValueError("train_particle_count_max_events must be positive")

    @staticmethod
    def _sample_training_particle_counts(trainer, max_events: int) -> dict[str, np.ndarray]:
        """Collect a bounded sample of post-selection input multiplicities."""
        class_to_label = getattr(trainer.datamodule.hparams, "class_to_label", {}) or {}
        label_to_class = {int(label): str(name) for name, label in dict(class_to_label).items()}
        sampled: dict[str, list[int]] = {}
        remaining = max_events
        for batch in trainer.datamodule.train_dataloader():
            if "part_mask" not in batch:
                raise KeyError("training batch does not contain part_mask")
            masks = batch["part_mask"].detach().cpu().bool()
            labels = batch.get("jet_type_labels")
            if labels is None:
                labels = np.zeros(masks.shape[0], dtype=np.int64)
            else:
                labels = labels.detach().cpu().numpy()
            counts = masks.sum(dim=1).numpy()
            take = min(remaining, len(counts))
            for count, label in zip(counts[:take], labels[:take]):
                class_name = label_to_class.get(int(label), "all")
                sampled.setdefault(class_name, []).append(int(count))
            remaining -= take
            if remaining == 0:
                break
        return {
            class_name: np.asarray(values, dtype=np.int64)
            for class_name, values in sampled.items()
        }

    def on_train_start(self, trainer, pl_module) -> None:
        """Plot a small, representative training-input multiplicity sample."""
        if not self.include_train_particle_count_histogram or not trainer.is_global_zero:
            return
        sequence_type = getattr(trainer.datamodule.hparams, "sequence_type", None)
        class_specs = getattr(trainer.datamodule.hparams, "class_specs", {}) or {}
        sequence_types = {
            str(spec.get("sequence_type", sequence_type)) for spec in dict(class_specs).values()
        } or {str(sequence_type)}
        if not all(sequence_type.startswith("particle") for sequence_type in sequence_types):
            logger.info(
                "Skipping training particle-count histogram for non-particle sequence types: "
                f"{sorted(sequence_types)}"
            )
            return

        try:
            counts_by_class = self._sample_training_particle_counts(
                trainer,
                self.train_particle_count_max_events,
            )
            if not counts_by_class:
                logger.warning("No training events found for the particle-count histogram")
                return
            figure = plot_particle_count_histograms(counts_by_class)
            name = "train/input_particle_multiplicity"
            path = self._plot_dir(trainer) / self._figure_name(
                trainer,
                "train",
                "input_particle_multiplicity",
            )
            figure.savefig(path, dpi=220, bbox_inches="tight")
            self._log_figure(trainer, path, name)
            close_figure(figure)
        except Exception as exc:
            logger.warning(f"Failed to plot training particle multiplicity: {exc}")

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        if (trainer.current_epoch + 1) % self.validation_plot_every_n_epochs != 0:
            return
        pl_module.concat_validation_loop_predictions()
        self.plot(trainer, pl_module, stage="val")

    def on_test_epoch_end(self, trainer, pl_module):
        pl_module.concat_test_loop_predictions()
        self.plot(trainer, pl_module, stage="test")

    def _feature_names(self, trainer, pl_module) -> list[str]:
        datamodule_hparams = trainer.datamodule.hparams
        if "selected_features" in datamodule_hparams:
            return list(datamodule_hparams.selected_features)
        if "dataset_kwargs_common" in datamodule_hparams:
            return list(datamodule_hparams.dataset_kwargs_common.feature_dict.keys())
        return [f"feature_{i}" for i in range(pl_module.val_x_original_concat.shape[-1])]

    @staticmethod
    def _transformed_feature_alias(feature_name: str, feature_index: int) -> str:
        aliases = {
            "L1T_PUPPIPart_Eta": "eta_scaled",
            "L1T_PUPPIPart_Phi_cos": "cos_phi",
            "L1T_PUPPIPart_Phi_sin": "sin_phi",
            "L1T_PUPPIPart_PT": "log_pt_shifted",
            "L1T_PUPPIPart_E": "log_energy_shifted",
        }
        return aliases.get(
            feature_name,
            feature_name.replace("/", "_").replace(" ", "_") or f"feature_{feature_index}",
        )

    def _plot_dir(self, trainer) -> Path:
        plot_dir = (
            Path(self.image_path)
            if self.image_path is not None
            else Path(trainer.default_root_dir) / "plots"
        )
        plot_dir.mkdir(parents=True, exist_ok=True)
        return plot_dir

    def _figure_name(self, trainer, stage: str, name: str) -> str:
        if self.no_trainer_info_in_filename:
            return f"{stage}_{name}.{self.image_filetype}"
        if stage == "val":
            return (
                f"val_epoch{trainer.current_epoch}_"
                f"gstep{trainer.global_step}_{name}.{self.image_filetype}"
            )
        if stage == "train":
            return f"train_start_{name}.{self.image_filetype}"
        return f"test_{name}.{self.image_filetype}"

    def _plot_pid_residuals_for_suite(
        self,
        trainer,
        pl_module,
        suite_name: str | None,
        suite_selector: np.ndarray,
        x_original: np.ndarray,
        x_reco: np.ndarray,
        mask: np.ndarray,
    ) -> None:
        """Save and log one original-PID-conditioned residual plot per test suite."""
        if not self.include_pid_conditional_residuals:
            return
        pid = getattr(pl_module, "test_pid_concat", None)
        if pid is None:
            logger.info("No original PID predictions found; skipping PID residual plot.")
            return
        selected_mask = mask[suite_selector]
        if not np.any(selected_mask):
            return
        original_flat = x_original[suite_selector][selected_mask]
        reconstructed_flat = x_reco[suite_selector][selected_mask]
        pid_flat = np.asarray(pid)[suite_selector][selected_mask]
        feature_names = self._feature_names(trainer, pl_module)
        residuals = pid_residual_arrays(original_flat, reconstructed_flat, feature_names)
        class_names = tuple(
            getattr(getattr(pl_module, "model", None), "pid_class_names", ())
        )
        if not class_names:
            class_names = tuple(f"class_{index}" for index in range(int(pid_flat.max()) + 1))
        by_pid = {
            index: {
                key: values[pid_flat == index]
                for key, values in residuals.items()
            }
            for index in range(len(class_names))
        }
        suite_label = suite_name or "all"
        figure = plot_pid_conditional_residuals(
            by_pid,
            class_names,
            title=(
                "Same-token reconstruction residuals by original PID"
                f" ({suite_label})"
            ),
        )
        name = f"test/{suite_label}/all/orbit_pid_conditional_residuals"
        path = self._plot_dir(trainer) / self._figure_name(
            trainer,
            "test",
            f"{suite_label}_all_orbit_pid_conditional_residuals",
        )
        figure.savefig(path, dpi=300, bbox_inches="tight")
        self._log_figure(trainer, path, name)
        close_figure(figure)

        summary = {
            "suite": suite_label,
            "particles": int(len(pid_flat)),
            "conditioning": "original PID at the same sequence position",
            "classes": {
                class_name: {
                    "count": int(np.sum(pid_flat == index)),
                    **{
                        key: pid_residual_distribution_summary(values)
                        for key, values in by_pid[index].items()
                    },
                }
                for index, class_name in enumerate(class_names)
            },
        }
        metrics_dir = Path(trainer.default_root_dir) / "saved_metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        summary_path = metrics_dir / (
            f"test_{suite_label}_pid_conditional_residuals_step_"
            f"{trainer.global_step}.json"
        )
        with summary_path.open("w") as file:
            json.dump(summary, file, indent=2, sort_keys=True)
        histogram_dir = Path(trainer.default_root_dir) / "saved_histograms"
        histogram_dir.mkdir(parents=True, exist_ok=True)
        values_path = histogram_dir / (
            f"test_{suite_label}_pid_conditional_residuals_step_"
            f"{trainer.global_step}.npz"
        )
        np.savez_compressed(values_path, original_pid=pid_flat, **residuals)

        for run_logger in trainer.loggers:
            if not isinstance(run_logger, L.pytorch.loggers.WandbLogger):
                continue
            try:
                import wandb

                residual_keys = list(residuals)
                columns = ["PID", "count"] + [
                    f"{key} {stat}"
                    for key in residual_keys
                    for stat in ("bias", "rmse")
                ]
                rows = []
                for class_name in class_names:
                    class_summary = summary["classes"][class_name]
                    rows.append(
                        [class_name, class_summary["count"]]
                        + [
                            class_summary[key][stat]
                            for key in residual_keys
                            for stat in ("bias", "rmse")
                        ]
                    )
                run_logger.experiment.log(
                    {f"test/{suite_label}/all/pid_residual_summary": wandb.Table(
                        columns=columns, data=rows
                    )}
                )
            except Exception as exc:
                logger.warning(f"Failed to log PID residual summary table: {exc}")

    def _plot_jet_token_reconstruction(
        self,
        trainer,
        pl_module,
        stage: str,
        suite_name: str | None,
        event_selector: np.ndarray,
    ) -> None:
        """Plot direct reconstruction of the optional stored jet tokens."""
        jet_original = getattr(pl_module, f"{stage}_jet_original_concat", None)
        jet_reco = getattr(pl_module, f"{stage}_jet_reco_concat", None)
        jet_mask = getattr(pl_module, f"{stage}_jet_mask_concat", None)
        if jet_original is None or jet_reco is None or jet_mask is None:
            return
        jet_original = jet_original[event_selector]
        jet_reco = jet_reco[event_selector]
        jet_mask = jet_mask[event_selector].astype(bool)
        if not np.any(jet_mask):
            return

        feature_names = list(
            getattr(trainer.datamodule.hparams, "selected_jet_features", [])
        )
        if not feature_names:
            feature_names = [
                f"jet_feature_{index}" for index in range(jet_original.shape[-1])
            ]
        histograms = collect_reconstruction_histograms(
            feature_names=feature_names,
            x_np=jet_original,
            x_hat_np=jet_reco,
            mask_np=jet_mask,
        )
        mse_per_feature = np.sum(
            ((jet_reco - jet_original) ** 2) * jet_mask[..., None],
            axis=(0, 1),
        ) / max(int(jet_mask.sum()), 1)
        suite_label = suite_name or "all"
        namespace = stage if suite_name is None else f"{stage}/{suite_name}"
        figures = {
            f"{namespace}/jet_tokens/orbit_reconstruction_features": plot_feature_histograms(
                histograms,
                feature_names,
                mse_per_feature=mse_per_feature,
                title=f"{stage.capitalize()} {suite_label} stored-jet reconstruction",
            ),
            f"{namespace}/jet_tokens/orbit_reconstruction_residuals": plot_residual_histograms(
                histograms,
                feature_names,
                title=f"{stage.capitalize()} {suite_label} stored-jet residuals",
            ),
        }
        suffix = "" if suite_name is None else f"_{suite_name}"
        for name, figure in figures.items():
            stem = name.rsplit("/", 1)[-1]
            path = self._plot_dir(trainer) / self._figure_name(
                trainer,
                stage,
                f"jet_tokens{suffix}_{stem}",
            )
            figure.savefig(path, dpi=220, bbox_inches="tight")
            self._log_figure(trainer, path, name)
            close_figure(figure)

        self._log_metrics(
            trainer,
            stage,
            "jet_tokens",
            {
                "metrics/mse_total": float(np.mean(mse_per_feature)),
                "metrics/valid_jets": int(jet_mask.sum()),
                **{
                    f"metrics/mse_{name}": float(value)
                    for name, value in zip(feature_names, mse_per_feature)
                },
            },
            suite_name=suite_name,
        )

    @staticmethod
    def _wandb_figure_key(name: str) -> str:
        parts = [part.strip().replace(" ", "_") for part in name.split("/") if part.strip()]
        if len(parts) >= 3 and parts[0] in {"val", "test"}:
            stage, group, plot_name = parts[0], parts[1], "/".join(parts[2:])
            return f"{stage}_plots/{group}/{plot_name}"
        return f"plots/{'_'.join(parts)}"

    def _log_figure(self, trainer, path: Path, name: str) -> None:
        logged = False
        wandb_key = self._wandb_figure_key(name)
        for lightning_logger in trainer.loggers:
            if isinstance(lightning_logger, L.pytorch.loggers.CometLogger):
                lightning_logger.experiment.log_image(
                    str(path),
                    name=name,
                    step=trainer.global_step,
                )
                logged = True
            elif isinstance(lightning_logger, L.pytorch.loggers.WandbLogger):
                try:
                    import wandb

                    lightning_logger.experiment.log(
                        {
                            wandb_key: wandb.Image(
                                str(path),
                                caption=name,
                            )
                        },
                        commit=False,
                    )
                    logged = True
                except Exception as exc:
                    logger.warning(f"Failed to log {path} to W&B: {exc}")
        if logged:
            logger.info(f"Logged ORBIT figure {name} from {path}")
        else:
            logger.info(f"Saved ORBIT figure {name} to {path}; no image logger was active")

    def _log_metrics(
        self,
        trainer,
        stage: str,
        group_name: str,
        metrics: dict[str, float | int | None],
        suite_name: str | None = None,
    ) -> None:
        prefix_parts = [f"{stage}_metrics"]
        if suite_name is not None:
            prefix_parts.append(suite_name)
        if group_name != "all":
            prefix_parts.append(group_name)
        prefix = "/".join(prefix_parts)
        scalar_metrics = {}
        for name, value in metrics.items():
            # Keep the full-loop aggregate reconstruction losses logged by Lightning.
            if group_name == "all" and name.startswith("loss_reco"):
                continue
            if value is None:
                continue
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, (float, int)):
                scalar_metrics[f"{prefix}/{name}"] = value

        if not scalar_metrics:
            return

        for lightning_logger in trainer.loggers:
            lightning_logger.log_metrics(scalar_metrics)
        logger.info(
            f"Logged {len(scalar_metrics)} ORBIT scalar metrics for {stage}/{group_name}"
        )

    def _num_codes(self, pl_module) -> int | None:
        vqlayer = getattr(pl_module.model, "vqlayer", None)
        if vqlayer is None:
            return None
        return getattr(vqlayer, "num_codes", None)

    @staticmethod
    def _entropy_branch_specs(pl_module) -> tuple[CodeBranchSpec, ...]:
        vqlayer = getattr(pl_module.model, "vqlayer", None)
        branch_num_codes = getattr(vqlayer, "branch_num_codes", None)
        if branch_num_codes:
            specs = []
            for branch in getattr(vqlayer, "branch_order", branch_num_codes.keys()):
                if branch not in branch_num_codes:
                    continue
                quantizer = vqlayer.quantizers[branch]
                levels_tensor = getattr(quantizer, "levels", None)
                levels = (
                    None
                    if levels_tensor is None
                    else tuple(int(level) for level in levels_tensor.detach().cpu().tolist())
                )
                specs.append(
                    CodeBranchSpec(
                        name=str(branch),
                        num_codes=int(branch_num_codes[branch]),
                        levels=levels,
                    )
                )
            return tuple(specs)

        q_levels = getattr(pl_module, "q_levels", None)
        if q_levels is not None:
            levels = tuple(int(level) for level in q_levels)
            component_names = ("eta", "phi", "pt")
            if getattr(pl_module, "pid_enabled", False):
                levels = (int(pl_module.pid_num_classes), *levels)
                component_names = ("pid", *component_names)
            return (
                CodeBranchSpec(
                    name="input",
                    num_codes=int(np.prod(levels, dtype=np.int64)),
                    levels=levels,
                    component_names=component_names,
                ),
            )
        return ()

    def _data_level(self, trainer) -> str:
        sequence_type = getattr(trainer.datamodule.hparams, "sequence_type", "particle")
        return "particle" if sequence_type.startswith("particle") else "jet"

    @staticmethod
    def _jet_radius_from_sequence_type(sequence_type: str | None, default: float) -> float:
        if sequence_type and "ak4" in sequence_type.lower():
            return 0.4
        if sequence_type and "ak8" in sequence_type.lower():
            return 0.8
        return default

    def _jet_radii_for_group(
        self,
        trainer,
        group_name: str,
        labels: np.ndarray,
    ) -> tuple[np.ndarray, str]:
        class_specs = getattr(trainer.datamodule.hparams, "class_specs", None)
        if not class_specs:
            sequence_type = getattr(trainer.datamodule.hparams, "sequence_type", None)
            radius = self._jet_radius_from_sequence_type(sequence_type, self.jet_radius)
            return np.full(labels.shape[0], radius), f"{radius:g}"

        class_specs = dict(class_specs)
        if group_name != "all":
            class_spec = dict(class_specs).get(group_name, {})
            sequence_type = class_spec.get("eval_sequence_type") or class_spec.get("sequence_type")
            radius = self._jet_radius_from_sequence_type(sequence_type, self.jet_radius)
            return np.full(labels.shape[0], radius), f"{radius:g}"

        class_to_label = getattr(trainer.datamodule.hparams, "class_to_label", {})
        label_to_class = {int(label): name for name, label in dict(class_to_label).items()}
        radii = np.full(labels.shape[0], self.jet_radius)
        for label, class_name in label_to_class.items():
            class_spec = dict(class_specs).get(class_name, {})
            sequence_type = class_spec.get("eval_sequence_type") or class_spec.get("sequence_type")
            radii[labels == label] = self._jet_radius_from_sequence_type(
                sequence_type,
                self.jet_radius,
            )
        return radii, "class-specific"

    def _to_physical_features(self, x: np.ndarray) -> np.ndarray:
        """Convert `[eta/3, cos(phi), sin(phi), log(pT)-1.8]` to `[eta, phi, pT]`."""
        eta = x[..., 0] * 3.0
        phi = np.arctan2(x[..., 2], x[..., 1])
        pt = np.exp(x[..., 3] + 1.8) - 1e-8
        pt = np.clip(pt, a_min=0.0, a_max=None)
        return np.stack([eta, phi, pt], axis=-1)

    def _collect_missing_transverse_energy(
        self,
        x_physical: np.ndarray,
        x_reco_physical: np.ndarray,
        mask: np.ndarray,
    ):
        def missing_et(events):
            pts = np.where(mask, events[..., 2], 0.0)
            phis = events[..., 1]
            px = np.sum(pts * np.cos(phis), axis=1)
            py = np.sum(pts * np.sin(phis), axis=1)
            return np.hypot(px, py)

        return missing_et(x_physical), missing_et(x_reco_physical)

    def _collect_direct_jet_metrics(
        self,
        x_physical: np.ndarray,
        x_reco_physical: np.ndarray,
        mask: np.ndarray,
        jet_radii: np.ndarray,
    ):
        true_jet_pts, reco_jet_pts = [], []
        true_jet_etas, reco_jet_etas = [], []
        true_jet_phis, reco_jet_phis = [], []
        all_true_jet_pts, all_reco_jet_pts = [], []
        all_true_jet_etas, all_reco_jet_etas = [], []
        all_true_jet_phis, all_reco_jet_phis = [], []
        match_stats = {"true": 0, "reco": 0, "matched": 0, "delta_r": []}
        for i in range(x_physical.shape[0]):
            event_mask = mask[i]
            true_vals = x_physical[i, event_mask]
            reco_vals = x_reco_physical[i, event_mask]
            all_true_indices, all_reco_indices, _ = match_jets_by_delta_r(
                true_vals[:, 0],
                true_vals[:, 1],
                reco_vals[:, 0],
                reco_vals[:, 1],
                max_delta_r=None,
            )
            true_indices, reco_indices, delta_r = match_jets_by_delta_r(
                true_vals[:, 0],
                true_vals[:, 1],
                reco_vals[:, 0],
                reco_vals[:, 1],
                max_delta_r=(
                    float(jet_radii[i]) * self.jet_matching_radius_fraction
                ),
            )
            match_stats["true"] += len(true_vals)
            match_stats["reco"] += len(reco_vals)
            match_stats["matched"] += len(true_indices)
            match_stats["delta_r"].extend(delta_r)
            if len(all_true_indices):
                all_true_jet_pts.extend(true_vals[all_true_indices, 2])
                all_reco_jet_pts.extend(reco_vals[all_reco_indices, 2])
                all_true_jet_etas.extend(true_vals[all_true_indices, 0])
                all_reco_jet_etas.extend(reco_vals[all_reco_indices, 0])
                all_true_jet_phis.extend(true_vals[all_true_indices, 1])
                all_reco_jet_phis.extend(reco_vals[all_reco_indices, 1])
            if len(true_indices):
                true_jet_pts.extend(true_vals[true_indices, 2])
                reco_jet_pts.extend(reco_vals[reco_indices, 2])
                true_jet_etas.extend(true_vals[true_indices, 0])
                reco_jet_etas.extend(reco_vals[reco_indices, 0])
                true_jet_phis.extend(true_vals[true_indices, 1])
                reco_jet_phis.extend(reco_vals[reco_indices, 1])
        return (
            true_jet_pts,
            reco_jet_pts,
            true_jet_etas,
            reco_jet_etas,
            true_jet_phis,
            reco_jet_phis,
            all_true_jet_pts,
            all_reco_jet_pts,
            all_true_jet_etas,
            all_reco_jet_etas,
            all_true_jet_phis,
            all_reco_jet_phis,
            match_stats,
        )

    @staticmethod
    def _calculate_tau32(particles, jet_radius: float) -> float:
        """Calculate tau32 for the constituents of one inclusive jet."""
        if len(particles[0]) < 3:
            return np.nan
        jetdef = fastjet.JetDefinition(fastjet.kt_algorithm, jet_radius)
        cluster = fastjet.ClusterSequence(particles, jetdef)
        d0 = ak.sum(particles.pt * jet_radius, axis=1)
        exclusive_jets_2 = cluster.exclusive_jets(n_jets=2)
        exclusive_jets_3 = cluster.exclusive_jets(n_jets=3)

        dr_1i_t2 = _delta_r(particles, exclusive_jets_2[:, :1])
        dr_2i_t2 = _delta_r(particles, exclusive_jets_2[:, 1:2])
        min_dr_t2 = ak.min(
            ak.concatenate(
                [dr_1i_t2[..., np.newaxis], dr_2i_t2[..., np.newaxis]],
                axis=-1,
            ),
            axis=-1,
        )
        tau2 = ak.sum(particles.pt * min_dr_t2, axis=1) / d0

        dr_1i_t3 = _delta_r(particles, exclusive_jets_3[:, :1])
        dr_2i_t3 = _delta_r(particles, exclusive_jets_3[:, 1:2])
        dr_3i_t3 = _delta_r(particles, exclusive_jets_3[:, 2:3])
        min_dr_t3 = ak.min(
            ak.concatenate(
                [
                    dr_1i_t3[..., np.newaxis],
                    dr_2i_t3[..., np.newaxis],
                    dr_3i_t3[..., np.newaxis],
                ],
                axis=-1,
            ),
            axis=-1,
        )
        tau3 = ak.sum(particles.pt * min_dr_t3, axis=1) / d0
        return float(np.nan_to_num((tau3 / (tau2 + 1e-8))[0]))

    def _reconstruct_event_jets(
        self,
        pt: np.ndarray,
        eta: np.ndarray,
        phi: np.ndarray,
        jet_radius: float,
    ):
        pt = np.asarray(pt, dtype=np.float64)
        eta = np.asarray(eta, dtype=np.float64)
        phi = np.asarray(phi, dtype=np.float64)
        if not len(pt) or np.sum(pt) <= 0:
            return {
                "pt": np.array([]),
                "eta": np.array([]),
                "phi": np.array([]),
                "mass": np.array([]),
                "tau32": np.array([]),
                "jet_count": 0,
            }

        particles = ak.zip(
            {"pt": [pt], "eta": [eta], "phi": [phi], "mass": [np.zeros_like(pt)]},
            with_name="Momentum4D",
        )
        jetdef = fastjet.JetDefinition(fastjet.kt_algorithm, jet_radius)
        cluster = fastjet.ClusterSequence(particles, jetdef)
        inclusive_jets = cluster.inclusive_jets(min_pt=self.jet_min_pt)
        if not len(inclusive_jets[0]):
            return {
                "pt": np.array([]),
                "eta": np.array([]),
                "phi": np.array([]),
                "mass": np.array([]),
                "tau32": np.array([]),
                "jet_count": 0,
            }

        if self.include_tau32:
            constituents = cluster.constituents(min_pt=self.jet_min_pt)[0]
            tau32 = np.asarray(
                [
                    self._calculate_tau32(
                        jet_constituents[np.newaxis],
                        jet_radius,
                    )
                    for jet_constituents in constituents
                ],
                dtype=np.float64,
            )
        else:
            tau32 = np.empty(0, dtype=np.float64)
        return {
            "pt": np.asarray(inclusive_jets.pt[0]),
            "eta": np.asarray(inclusive_jets.eta[0]),
            "phi": np.asarray(inclusive_jets.phi[0]),
            "mass": np.asarray(inclusive_jets.mass[0]),
            "tau32": tau32,
            "jet_count": len(inclusive_jets[0]),
        }

    def _evaluate_particle_event_jets(
        self,
        x_physical: np.ndarray,
        x_reco_physical: np.ndarray,
        event_mask: np.ndarray,
        jet_radius: float,
    ) -> dict:
        true_jets = self._reconstruct_event_jets(
            x_physical[event_mask, 2],
            x_physical[event_mask, 0],
            x_physical[event_mask, 1],
            jet_radius,
        )
        reco_jets = self._reconstruct_event_jets(
            x_reco_physical[event_mask, 2],
            x_reco_physical[event_mask, 0],
            x_reco_physical[event_mask, 1],
            jet_radius,
        )
        all_true_indices, all_reco_indices, _ = match_jets_by_delta_r(
            true_jets["eta"],
            true_jets["phi"],
            reco_jets["eta"],
            reco_jets["phi"],
            max_delta_r=None,
        )
        true_indices, reco_indices, delta_r = match_jets_by_delta_r(
            true_jets["eta"],
            true_jets["phi"],
            reco_jets["eta"],
            reco_jets["phi"],
            max_delta_r=jet_radius * self.jet_matching_radius_fraction,
        )

        result = {
            "true_jet_pts": true_jets["pt"][true_indices],
            "reco_jet_pts": reco_jets["pt"][reco_indices],
            "true_jet_masses": true_jets["mass"][true_indices],
            "reco_jet_masses": reco_jets["mass"][reco_indices],
            "all_true_jet_pts": true_jets["pt"][all_true_indices],
            "all_reco_jet_pts": reco_jets["pt"][all_reco_indices],
            "all_true_jet_masses": true_jets["mass"][all_true_indices],
            "all_reco_jet_masses": reco_jets["mass"][all_reco_indices],
            "true_tau32s": np.empty(0, dtype=np.float64),
            "reco_tau32s": np.empty(0, dtype=np.float64),
            "all_true_tau32s": np.empty(0, dtype=np.float64),
            "all_reco_tau32s": np.empty(0, dtype=np.float64),
            "true_jet_count": true_jets["jet_count"],
            "reco_jet_count": reco_jets["jet_count"],
            "match_stats": {
                "true": true_jets["jet_count"],
                "reco": reco_jets["jet_count"],
                "matched": len(true_indices),
                "delta_r": np.asarray(delta_r, dtype=np.float64),
            },
        }
        if self.include_tau32:
            true_tau = true_jets["tau32"][true_indices]
            reco_tau = reco_jets["tau32"][reco_indices]
            finite_tau = np.isfinite(true_tau) & np.isfinite(reco_tau)
            result["true_tau32s"] = true_tau[finite_tau]
            result["reco_tau32s"] = reco_tau[finite_tau]

            all_true_tau = true_jets["tau32"][all_true_indices]
            all_reco_tau = reco_jets["tau32"][all_reco_indices]
            all_finite_tau = np.isfinite(all_true_tau) & np.isfinite(all_reco_tau)
            result["all_true_tau32s"] = all_true_tau[all_finite_tau]
            result["all_reco_tau32s"] = all_reco_tau[all_finite_tau]
        return result

    def _evaluate_particle_event_batch(
        self,
        x_physical: np.ndarray,
        x_reco_physical: np.ndarray,
        mask: np.ndarray,
        jet_radii: np.ndarray,
    ) -> list[dict]:
        return [
            self._evaluate_particle_event_jets(
                x_physical[i],
                x_reco_physical[i],
                mask[i],
                float(jet_radii[i]),
            )
            for i in range(x_physical.shape[0])
        ]

    @staticmethod
    def _combine_particle_event_jet_results(event_results: list[dict]):
        true_jet_pts = []
        reco_jet_pts = []
        true_jet_masses = []
        reco_jet_masses = []
        true_tau32s = []
        reco_tau32s = []
        all_true_jet_pts = []
        all_reco_jet_pts = []
        all_true_jet_masses = []
        all_reco_jet_masses = []
        all_true_tau32s = []
        all_reco_tau32s = []
        true_jet_counts = []
        reco_jet_counts = []
        match_stats = {"true": 0, "reco": 0, "matched": 0, "delta_r": []}

        for result in event_results:
            for key, target in (
                ("true_jet_pts", true_jet_pts),
                ("reco_jet_pts", reco_jet_pts),
                ("true_jet_masses", true_jet_masses),
                ("reco_jet_masses", reco_jet_masses),
                ("true_tau32s", true_tau32s),
                ("reco_tau32s", reco_tau32s),
                ("all_true_jet_pts", all_true_jet_pts),
                ("all_reco_jet_pts", all_reco_jet_pts),
                ("all_true_jet_masses", all_true_jet_masses),
                ("all_reco_jet_masses", all_reco_jet_masses),
                ("all_true_tau32s", all_true_tau32s),
                ("all_reco_tau32s", all_reco_tau32s),
            ):
                target.extend(result[key])
            true_jet_counts.append(result["true_jet_count"])
            reco_jet_counts.append(result["reco_jet_count"])
            for key in ("true", "reco", "matched"):
                match_stats[key] += int(result["match_stats"][key])
            match_stats["delta_r"].extend(result["match_stats"]["delta_r"])

        return (
            true_jet_pts,
            reco_jet_pts,
            true_jet_masses,
            reco_jet_masses,
            true_tau32s,
            reco_tau32s,
            all_true_jet_pts,
            all_reco_jet_pts,
            all_true_jet_masses,
            all_reco_jet_masses,
            all_true_tau32s,
            all_reco_tau32s,
            true_jet_counts,
            reco_jet_counts,
            match_stats,
        )

    def _collect_particle_jet_metrics(
        self,
        x_physical: np.ndarray,
        x_reco_physical: np.ndarray,
        mask: np.ndarray,
        jet_radii: np.ndarray,
    ):
        return self._combine_particle_event_jet_results(
            self._evaluate_particle_event_batch(
                x_physical,
                x_reco_physical,
                mask,
                jet_radii,
            )
        )

    def _plot_jet_count_difference(
        self,
        true_jet_counts,
        reco_jet_counts,
        title: str,
    ):
        true_jet_counts = np.asarray(true_jet_counts, dtype=int)
        reco_jet_counts = np.asarray(reco_jet_counts, dtype=int)
        diff = reco_jet_counts - true_jet_counts
        max_count = int(max(true_jet_counts.max(initial=0), reco_jet_counts.max(initial=0)))
        count_bins = np.arange(-0.5, max_count + 1.5, 1)
        diff_bins = np.arange(-10.5, 11.5, 1)
        diff_for_hist = np.clip(diff, -10, 10)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(title)
        axes[0].hist(
            true_jet_counts,
            bins=count_bins,
            histtype="stepfilled",
            alpha=0.35,
            label="Original",
        )
        axes[0].hist(
            reco_jet_counts,
            bins=count_bins,
            histtype="step",
            linewidth=2,
            label="Reconstructed",
        )
        axes[0].set_xlabel("Reclustered jet count")
        axes[0].set_ylabel("Events")
        axes[0].legend()
        axes[1].hist(
            diff_for_hist,
            bins=diff_bins,
            histtype="stepfilled",
            alpha=0.45,
        )
        axes[1].axvline(0, color="black", linestyle="--", alpha=0.5)
        axes[1].set_xlabel("Reconstructed - original jet count")
        axes[1].set_ylabel("Events")
        fig.tight_layout()
        return fig

    @staticmethod
    def _add_jet_match_metrics(metrics: dict, match_stats: dict) -> None:
        """Add scalar coverage and separation metrics for delta-R jet matching."""
        matched = int(match_stats["matched"])
        true_count = int(match_stats["true"])
        reco_count = int(match_stats["reco"])
        delta_r = np.asarray(match_stats["delta_r"], dtype=np.float64)
        metrics["metrics/jet_matches"] = matched
        metrics["metrics/jet_match_efficiency_true"] = (
            matched / true_count if true_count else 0.0
        )
        metrics["metrics/jet_match_efficiency_reco"] = (
            matched / reco_count if reco_count else 0.0
        )
        metrics["metrics/jet_match_delta_r_mean"] = (
            float(np.mean(delta_r)) if len(delta_r) else 0.0
        )

    def plot(self, trainer, pl_module, stage: str) -> None:
        if stage == "val" and not hasattr(pl_module, "val_x_original_concat"):
            logger.info("No validation predictions found. Skipping ORBIT plots.")
            return
        if stage == "test" and not hasattr(pl_module, "test_x_original_concat"):
            logger.info("No test predictions found. Skipping ORBIT plots.")
            return

        if stage == "val":
            x_original = pl_module.val_x_original_concat
            x_reco = pl_module.val_x_reco_concat
            mask = pl_module.val_mask_concat.astype(bool)
            code_idx = pl_module.val_code_idx_concat
            code_mask = getattr(pl_module, "val_code_mask_concat", mask).astype(bool)
        else:
            x_original = pl_module.test_x_original_concat
            x_reco = pl_module.test_x_reco_concat
            mask = pl_module.test_mask_concat.astype(bool)
            labels = pl_module.test_labels_concat
            code_idx = pl_module.test_code_idx_concat
            code_mask = getattr(pl_module, "test_code_mask_concat", mask).astype(bool)

        if stage == "val":
            labels = pl_module.val_labels_concat

        class_to_label = getattr(trainer.datamodule.hparams, "class_to_label", None)
        label_to_class = (
            {int(label): name for name, label in dict(class_to_label).items()}
            if class_to_label
            else {}
        )
        suite_selections = [(None, np.ones(labels.shape[0], dtype=bool))]
        if stage == "test":
            suite_to_label = getattr(
                trainer.datamodule.hparams, "test_suite_to_label", None
            )
            suite_labels = getattr(pl_module, "test_suite_labels_concat", None)
            if suite_to_label and suite_labels is not None:
                suite_selections = [
                    (suite_name, suite_labels == int(suite_label))
                    for suite_name, suite_label in dict(suite_to_label).items()
                ]

        for suite_name, suite_selector in suite_selections:
            self._plot_jet_token_reconstruction(
                trainer,
                pl_module,
                stage,
                suite_name,
                suite_selector,
            )
            if stage == "test":
                self._plot_pid_residuals_for_suite(
                    trainer,
                    pl_module,
                    suite_name,
                    suite_selector,
                    x_original,
                    x_reco,
                    mask,
                )
            subset_groups = []
            for label, class_name in sorted(label_to_class.items()):
                subset_groups.append(
                    (class_name, class_name, suite_selector & (labels == label))
                )
            all_group = ("all", "all", suite_selector)
            groups = (
                [*subset_groups, all_group]
                if self.cache_subset_jet_results
                else [all_group, *subset_groups]
            )

            particle_jet_cache = None
            if (
                self.cache_subset_jet_results
                and self.enable_physics_plots
                and self._data_level(trainer) == "particle"
            ):
                suite_indices = np.flatnonzero(suite_selector)
                suite_x_physical = self._to_physical_features(x_original[suite_selector])
                suite_x_reco_physical = self._to_physical_features(x_reco[suite_selector])
                suite_jet_radii, _ = self._jet_radii_for_group(
                    trainer,
                    "all",
                    labels[suite_selector],
                )
                suite_results = self._evaluate_particle_event_batch(
                    suite_x_physical,
                    suite_x_reco_physical,
                    mask[suite_selector],
                    suite_jet_radii,
                )
                particle_jet_cache = dict(zip(suite_indices, suite_results))

            if self.include_event_reconstruction_grid and class_to_label:
                event_groups = []
                for label, class_name in sorted(label_to_class.items()):
                    event_selector = suite_selector & (labels == label)
                    if not np.any(event_selector):
                        continue
                    limit = self.event_reconstruction_examples_per_class
                    event_groups.append(
                        (
                            class_name,
                            self._to_physical_features(x_original[event_selector][:limit]),
                            self._to_physical_features(x_reco[event_selector][:limit]),
                            mask[event_selector][:limit],
                        )
                    )
                if event_groups:
                    figure = plot_event_reconstruction_grid(
                        event_groups,
                        events_per_group=self.event_reconstruction_examples_per_class,
                    )
                    namespace = stage if suite_name is None else f"{stage}/{suite_name}"
                    name = f"{namespace}/event_examples/orbit_event_reconstruction"
                    suite_suffix = "" if suite_name is None else f"_{suite_name}"
                    path = self._plot_dir(trainer) / self._figure_name(
                        trainer,
                        stage,
                        f"event_examples{suite_suffix}_orbit_event_reconstruction",
                    )
                    figure.savefig(path, dpi=220, bbox_inches="tight")
                    logger.info(f"Saved ORBIT figure {name} to {path}")
                    self._log_figure(trainer, path, name)
                    close_figure(figure)

            for group_name, display_name, event_selector in groups:
                if np.any(event_selector):
                    cached_particle_jet_results = (
                        [
                            particle_jet_cache[int(index)]
                            for index in np.flatnonzero(event_selector)
                        ]
                        if particle_jet_cache is not None
                        else None
                    )
                    self._plot_arrays(
                        trainer=trainer,
                        pl_module=pl_module,
                        stage=stage,
                        suite_name=suite_name,
                        group_name=group_name,
                        display_name=display_name,
                        x_original=x_original[event_selector],
                        x_reco=x_reco[event_selector],
                        mask=mask[event_selector],
                        labels=labels[event_selector],
                        code_idx=code_idx[event_selector],
                        code_mask=code_mask[event_selector],
                        particle_jet_results=cached_particle_jet_results,
                        preserve_legacy_names=(suite_name is None and group_name == "all"),
                    )

    def _plot_arrays(
        self,
        trainer,
        pl_module,
        stage: str,
        suite_name: str | None,
        group_name: str,
        display_name: str,
        x_original: np.ndarray,
        x_reco: np.ndarray,
        mask: np.ndarray,
        labels: np.ndarray,
        code_idx: np.ndarray,
        code_mask: np.ndarray,
        particle_jet_results: list[dict] | None,
        preserve_legacy_names: bool,
    ) -> None:
        if not np.any(mask):
            logger.warning(f"No valid tokens found for {stage}/{group_name} ORBIT plots.")
            return

        feature_names = self._feature_names(trainer, pl_module)
        plot_dir = self._plot_dir(trainer)
        histograms = collect_reconstruction_histograms(
            feature_names=feature_names,
            x_np=x_original,
            x_hat_np=x_reco,
            mask_np=mask,
        )
        mse_per_feature = np.sum(
            ((x_reco - x_original) ** 2) * mask[..., None],
            axis=(0, 1),
        )
        valid_particles = np.clip(np.sum(mask), a_min=1, a_max=None)
        mse_per_feature /= valid_particles
        num_codes = self._num_codes(pl_module)
        metrics = {
            f"metrics/mse_{feature_name.replace('/', '_').replace(' ', '_')}": float(mse)
            for feature_name, mse in zip(feature_names, mse_per_feature)
        }
        model_kwargs = pl_module.hparams.get("model_kwargs") or {}
        reconstruction_loss = model_kwargs.get("reconstruction_loss", "l2")
        metrics.update(
            reconstruction_loss_metrics(
                x_original,
                x_reco,
                mask,
                reconstruction_loss=reconstruction_loss,
            )
        )
        if num_codes is not None:
            metrics.update(
                code_entropy_metrics(
                    code_idx,
                    code_mask,
                    mask,
                    num_codes=num_codes,
                    branch_specs=self._entropy_branch_specs(pl_module),
                    num_quantizers=int(
                        getattr(pl_module.model.vqlayer, "num_quantizers", 1)
                    ),
                )
            )
        else:
            # A continuous autoencoder has no discrete code distribution or
            # meaningful fixed-width/rate estimate.
            metrics["metrics/quantization_enabled"] = 0
        for feature_index, (feature_name, mse) in enumerate(zip(feature_names, mse_per_feature)):
            alias = self._transformed_feature_alias(feature_name, feature_index)
            metrics[f"metrics/transformed_mse_{alias}"] = float(mse)
            metrics[f"metrics/transformed_rmse_{alias}"] = float(np.sqrt(max(mse, 0.0)))
        metrics.update(
            {
                "metrics/mse_total": float(np.mean(mse_per_feature)),
                "metrics/transformed_mse_total": float(np.mean(mse_per_feature)),
                "metrics/transformed_rmse_mean": float(
                    np.mean(np.sqrt(np.clip(mse_per_feature, a_min=0.0, a_max=None)))
                ),
            }
        )

        namespace = stage if suite_name is None else f"{stage}/{suite_name}"
        figures = {
            f"{namespace}/{group_name}/orbit_reconstruction_features": plot_feature_histograms(
                histograms,
                feature_names,
                mse_per_feature=mse_per_feature,
                title=f"{stage.capitalize()} {display_name} ORBIT reconstruction features",
            ),
            f"{namespace}/{group_name}/orbit_reconstruction_residuals": plot_residual_histograms(
                histograms,
                feature_names,
                title=f"{stage.capitalize()} {display_name} ORBIT reconstruction residuals",
            ),
        }
        if self.include_codebook_histogram and num_codes is not None:
            figures[f"{namespace}/{group_name}/orbit_codebook_usage"] = plot_codebook_histogram(
                code_idx[code_mask],
                num_codes=num_codes,
            )
        if self.enable_physics_plots:
            data_level = self._data_level(trainer)
            x_physical = self._to_physical_features(x_original)
            x_reco_physical = self._to_physical_features(x_reco)
            x_physical_flat = x_physical[mask]
            x_reco_physical_flat = x_reco_physical[mask]
            finite = np.all(np.isfinite(x_physical_flat), axis=1) & np.all(
                np.isfinite(x_reco_physical_flat),
                axis=1,
            )
            x_physical_flat = x_physical_flat[finite]
            x_reco_physical_flat = x_reco_physical_flat[finite]
            if x_physical_flat.size:
                physics_feature_names = ["Eta", "Phi", "pT"]
                physical_feature_ranges = (
                    MINBIAS_PHYSICAL_FEATURE_RANGES
                    if group_name == "minbias"
                    else None
                )
                missing_et_range = (
                    (0.0, 200.0) if group_name == "minbias" else (0.0, 1_000.0)
                )
                jet_mass_range = (
                    (0.0, 3_000.0)
                    if group_name in {"all", "gghbb"}
                    else (0.0, 1_800.0)
                )
                physics_delta = x_reco_physical_flat - x_physical_flat
                physics_delta[:, 1] = (
                    np.remainder(physics_delta[:, 1] + np.pi, 2 * np.pi) - np.pi
                )
                physics_mse = np.mean(physics_delta**2, axis=0)
                jet_radii, jet_radius_label = self._jet_radii_for_group(
                    trainer,
                    group_name,
                    labels,
                )
                if data_level == "particle":
                    (
                        true_jet_pts,
                        reco_jet_pts,
                        true_jet_masses,
                        reco_jet_masses,
                        true_tau32s,
                        reco_tau32s,
                        all_true_jet_pts,
                        all_reco_jet_pts,
                        all_true_jet_masses,
                        all_reco_jet_masses,
                        all_true_tau32s,
                        all_reco_tau32s,
                        true_jet_counts,
                        reco_jet_counts,
                        match_stats,
                    ) = (
                        self._combine_particle_event_jet_results(particle_jet_results)
                        if particle_jet_results is not None
                        else self._collect_particle_jet_metrics(
                            x_physical,
                            x_reco_physical,
                            mask,
                            jet_radii=jet_radii,
                        )
                    )
                    missing_ets = self._collect_missing_transverse_energy(
                        x_physical, x_reco_physical, mask,
                    )
                    jet_count_diff = np.asarray(reco_jet_counts) - np.asarray(true_jet_counts)
                    physics_histograms = collect_physical_reconstruction_histograms(
                        physics_feature_names,
                        x_physical_flat,
                        x_reco_physical_flat,
                        true_jet_pts=true_jet_pts,
                        reco_jet_pts=reco_jet_pts,
                        true_jet_masses=true_jet_masses,
                        reco_jet_masses=reco_jet_masses,
                        true_tau32s=true_tau32s,
                        reco_tau32s=reco_tau32s,
                        unfiltered_true_jet_pts=all_true_jet_pts,
                        unfiltered_reco_jet_pts=all_reco_jet_pts,
                        unfiltered_true_jet_masses=all_true_jet_masses,
                        unfiltered_reco_jet_masses=all_reco_jet_masses,
                        unfiltered_true_tau32s=all_true_tau32s,
                        unfiltered_reco_tau32s=all_reco_tau32s,
                        true_missing_ets=missing_ets[0],
                        reco_missing_ets=missing_ets[1],
                        data_level=data_level,
                        physical_feature_ranges=physical_feature_ranges,
                        missing_et_range=missing_et_range,
                        jet_mass_range=jet_mass_range,
                    )
                    physics_histograms["jet_count_orig"] = np.asarray(true_jet_counts)
                    physics_histograms["jet_count_reco"] = np.asarray(reco_jet_counts)
                    physics_histograms["jet_count_diff"] = jet_count_diff
                    metrics["metrics/jet_count_diff_mean"] = float(np.mean(jet_count_diff))
                    metrics["metrics/jet_count_diff_abs_mean"] = float(
                        np.mean(np.abs(jet_count_diff))
                    )
                    metrics["metrics/jet_min_pt"] = self.jet_min_pt
                    metrics["metrics/tau32_enabled"] = int(self.include_tau32)
                    self._add_jet_match_metrics(metrics, match_stats)
                    figures[f"{namespace}/{group_name}/orbit_jet_count_difference"] = (
                        self._plot_jet_count_difference(
                            true_jet_counts,
                            reco_jet_counts,
                            title=(
                                f"{stage.capitalize()} {display_name} "
                                f"reclustered jet counts (R={jet_radius_label}, "
                                rf"$p_T \geq {self.jet_min_pt:g}$ GeV)"
                            ),
                        )
                    )
                else:
                    (
                        true_jet_pts, reco_jet_pts,
                        true_jet_etas, reco_jet_etas,
                        true_jet_phis, reco_jet_phis,
                        all_true_jet_pts, all_reco_jet_pts,
                        all_true_jet_etas, all_reco_jet_etas,
                        all_true_jet_phis, all_reco_jet_phis,
                        match_stats,
                    ) = self._collect_direct_jet_metrics(
                        x_physical,
                        x_reco_physical,
                        mask,
                        jet_radii=jet_radii,
                    )
                    self._add_jet_match_metrics(metrics, match_stats)
                    physics_histograms = collect_physical_reconstruction_histograms(
                        physics_feature_names,
                        x_physical_flat,
                        x_reco_physical_flat,
                        true_jet_pts=true_jet_pts,
                        reco_jet_pts=reco_jet_pts,
                        true_jet_etas=true_jet_etas,
                        reco_jet_etas=reco_jet_etas,
                        true_jet_phis=true_jet_phis,
                        reco_jet_phis=reco_jet_phis,
                        unfiltered_true_jet_pts=all_true_jet_pts,
                        unfiltered_reco_jet_pts=all_reco_jet_pts,
                        unfiltered_true_jet_etas=all_true_jet_etas,
                        unfiltered_reco_jet_etas=all_reco_jet_etas,
                        unfiltered_true_jet_phis=all_true_jet_phis,
                        unfiltered_reco_jet_phis=all_reco_jet_phis,
                        data_level=data_level,
                        physical_feature_ranges=physical_feature_ranges,
                        missing_et_range=missing_et_range,
                        jet_mass_range=jet_mass_range,
                    )
                histograms.update(
                    {f"physical_{key}": value for key, value in physics_histograms.items()}
                )
                for name, value in zip(physics_feature_names, physics_mse):
                    metrics[f"metrics/mse_{name}"] = float(value)
                metrics["metrics/physical_mse_total"] = float(np.mean(physics_mse))
                figures.update(
                    {
                        f"{namespace}/{group_name}/orbit_{name}": figure
                        for name, figure in physical_reconstruction_plots(
                            physics_feature_names,
                            physics_mse,
                            physics_histograms,
                            data_level=data_level,
                            include_all_ratios=self.include_all_ratios,
                            jet_matching_cut_label=(
                                rf"$\Delta R \leq "
                                rf"{self.jet_matching_radius_fraction:g}R$"
                            ),
                        ).items()
                    }
                )

        self._log_metrics(trainer, stage, group_name, metrics, suite_name=suite_name)

        for name, fig in figures.items():
            filename_stem = name.split("/")[-1]
            if not preserve_legacy_names:
                suite_prefix = "" if suite_name is None else f"{suite_name}_"
                filename_stem = f"{suite_prefix}{group_name}_{filename_stem}"
            filename = self._figure_name(trainer, stage, filename_stem)
            path = plot_dir / filename
            fig.savefig(path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved ORBIT figure {name} to {path}")
            self._log_figure(trainer, path, name)
            close_figure(fig)

        if self.save_histograms:
            histogram_dir = Path(trainer.default_root_dir) / "saved_histograms"
            histogram_dir.mkdir(parents=True, exist_ok=True)
            suite_prefix = "" if suite_name is None else f"_{suite_name}"
            histogram_prefix = (
                f"{stage}_orbit"
                if preserve_legacy_names
                else f"{stage}{suite_prefix}_{group_name}_orbit"
            )
            histogram_path = (
                histogram_dir / f"{histogram_prefix}_hists_step_{trainer.global_step}.npz"
            )
            np.savez_compressed(histogram_path, **histograms)
            logger.info(f"Saved ORBIT histograms to {histogram_path}")

            metrics_dir = Path(trainer.default_root_dir) / "saved_metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = (
                metrics_dir / f"{histogram_prefix}_metrics_step_{trainer.global_step}.json"
            )
            with metrics_path.open("w") as file:
                json.dump(metrics, file, indent=2, sort_keys=True)
            logger.info(f"Saved ORBIT metrics to {metrics_path}")
