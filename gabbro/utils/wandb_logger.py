"""W&B logger that uploads at most two checkpoints per run: best and latest."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from torch import Tensor

from gabbro.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class BestAndLatestWandbLogger(WandbLogger):
    """``WandbLogger`` that uploads only the best and the latest checkpoint.

    Lightning's ``WandbLogger`` uploads every checkpoint each ``ModelCheckpoint``
    tracks (last, best and all top-k files, and with ``save_top_k=-1`` every
    single epoch). Here local checkpointing is unchanged, but the W&B upload is
    limited to:

    * ``best``: ``best_model_path`` of the checkpoint callback(s) with a ``monitor``;
    * ``latest``: the most recently written checkpoint file of any callback.

    If both are the same file it is uploaded once with both aliases.

    ``log_model``:

    * ``True`` (recommended): upload once when training ends, so at most two
      artifact versions per run. Unlike the parent class this also happens when
      training is interrupted (Ctrl+C) or crashes, not only on success.
    * ``"all"``: upload best/latest whenever they change during training
      (more robust against hard kills, but every upload is a new stored version).
    * ``False``: no checkpoint upload.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._logged_mtime: dict[str, float] = {}

    def after_save_checkpoint(self, checkpoint_callback: ModelCheckpoint) -> None:
        if not self._log_model:
            return
        self._checkpoint_callbacks[id(checkpoint_callback)] = checkpoint_callback
        if self._log_model == "all":
            # A save_last callback writes its per-epoch file first and last.ckpt
            # right after. Before the first last.ckpt exists, that per-epoch file
            # would be uploaded as "latest" and then again as last.ckpt; wait.
            if checkpoint_callback.save_last and not checkpoint_callback.last_model_path:
                return
            self._log_best_and_latest()

    @rank_zero_only
    def finalize(self, status: str) -> None:
        if not self._log_model or self._experiment is None:
            return
        try:
            self._log_best_and_latest()
        except Exception as exc:  # never let an upload problem mask the training outcome
            log.warning(f"Uploading best/latest checkpoints to W&B failed (status={status}): {exc}")

    def _log_best_and_latest(self) -> None:
        # path -> (score, callback)
        candidates: dict[str, tuple[Any, ModelCheckpoint]] = {}
        best: tuple[str, Any, ModelCheckpoint] | None = None
        for callback in self._checkpoint_callbacks.values():
            last_path = getattr(callback, "last_model_path", "")
            if last_path:
                candidates[last_path] = (getattr(callback, "current_score", None), callback)
            for path, score in getattr(callback, "best_k_models", {}).items():
                candidates.setdefault(path, (score, callback))
            best_path = getattr(callback, "best_model_path", "")
            if best_path:
                candidates.setdefault(best_path, (callback.best_model_score, callback))
                if best is None and getattr(callback, "monitor", None):
                    best = (best_path, callback.best_model_score, callback)

        existing = {p: v for p, v in candidates.items() if Path(p).is_file()}
        if not existing:
            return

        # Prefer a real last.ckpt as "latest" (a per-epoch file is just a copy of it).
        last_paths = [
            p
            for p in existing
            if any(getattr(cb, "last_model_path", "") == p for cb in self._checkpoint_callbacks.values())
        ]
        uploads: dict[str, list[str]] = {}
        latest_path = max(last_paths or existing, key=lambda p: Path(p).stat().st_mtime)
        uploads[latest_path] = ["latest"]
        if best is not None and Path(best[0]).is_file():
            uploads.setdefault(best[0], []).append("best")

        # Upload the best first so the newest W&B version is the latest checkpoint.
        for path in sorted(uploads, key=lambda p: "best" not in uploads[p]):
            mtime = Path(path).stat().st_mtime
            if self._logged_mtime.get(path) == mtime:
                continue  # file unchanged since its last upload
            score, callback = existing[path]
            if best is not None and path == best[0]:
                score, callback = best[1], best[2]
            self._log_checkpoint_artifact(path, score, callback, uploads[path])
            self._logged_mtime[path] = mtime

    def _log_checkpoint_artifact(
        self, path: str, score: Any, callback: ModelCheckpoint, aliases: list[str]
    ) -> None:
        import wandb

        metadata = {
            "score": score.item() if isinstance(score, Tensor) else score,
            "original_filename": Path(path).name,
            callback.__class__.__name__: {
                k: getattr(callback, k)
                for k in ["monitor", "mode", "save_last", "save_top_k", "save_weights_only"]
                if hasattr(callback, k)
            },
        }
        if not self._checkpoint_name:
            self._checkpoint_name = f"model-{self.experiment.id}"
        artifact = wandb.Artifact(name=self._checkpoint_name, type="model", metadata=metadata)
        artifact.add_file(path, name="model.ckpt", policy=self.add_file_policy)
        self.experiment.log_artifact(artifact, aliases=aliases)
        log.info(f"Uploaded checkpoint {path} to W&B with aliases {aliases}")
