"""Kinematic feature transform shared by ORBIT loaders, plots and evaluation scripts.

The tokenizer sees each particle as ``[eta / eta_scale, cos(phi), sin(phi),
log(pT + epsilon) - pt_shift]``. Because the reconstruction loss is an
unweighted MSE over these features, ``eta_scale`` also acts as a loss weight
(``1 / eta_scale**2``) on eta relative to the other features, while
``pt_shift`` only moves the operating point and cancels in residuals.

The constants are configured as ``data.eta_scale`` / ``data.pt_shift``. Every
place that undoes the transform must read them from the run's data config via
:meth:`KinematicTransform.from_data_config`, so that runs with non-default
preprocessing still produce correct physical-unit plots and metrics. Run
configs written before these keys existed fall back to the historical
defaults below.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

DEFAULT_ETA_SCALE = 3.0
DEFAULT_PT_SHIFT = 1.8
PT_EPSILON = 1e-8


def _config_value(config: Any, key: str, default: float) -> float:
    if config is None:
        return default
    if isinstance(config, Mapping) or hasattr(config, "get"):
        value = config.get(key, None)
    else:
        value = getattr(config, key, None)
    return default if value is None else float(value)


@dataclass(frozen=True)
class KinematicTransform:
    """Forward and inverse of the ORBIT ``eta`` / ``pT`` preprocessing.

    >>> transform = KinematicTransform(eta_scale=2.0, pt_shift=1.0)
    >>> float(transform.eta_forward(np.float64(3.0)))
    1.5
    >>> x = np.array([[1.5, 1.0, 0.0, np.log(20.0) - 1.0]])
    >>> np.round(transform.to_physical(x), 6).tolist()
    [[3.0, 0.0, 20.0]]
    """

    eta_scale: float = DEFAULT_ETA_SCALE
    pt_shift: float = DEFAULT_PT_SHIFT
    epsilon: float = PT_EPSILON

    def __post_init__(self):
        if not math.isfinite(self.eta_scale) or self.eta_scale <= 0:
            raise ValueError(f"eta_scale must be finite and positive, got {self.eta_scale}")
        if not math.isfinite(self.pt_shift):
            raise ValueError(f"pt_shift must be finite, got {self.pt_shift}")

    @classmethod
    def from_data_config(cls, data_config: Any) -> "KinematicTransform":
        """Build from a run's ``data`` config / datamodule hparams (may be None)."""
        return cls(
            eta_scale=_config_value(data_config, "eta_scale", DEFAULT_ETA_SCALE),
            pt_shift=_config_value(data_config, "pt_shift", DEFAULT_PT_SHIFT),
        )

    @property
    def eta_latex(self) -> str:
        r"""Math-mode label of the transformed eta, e.g. ``\eta / 3``."""
        if self.eta_scale == 1.0:
            return r"\eta"
        return rf"\eta / {self.eta_scale:.4g}"

    @property
    def log_pt_latex(self) -> str:
        r"""Math-mode label of the transformed pT, e.g. ``\log(p_T) - 1.8``."""
        if self.pt_shift == 0.0:
            return r"\log(p_T)"
        sign = "-" if self.pt_shift > 0 else "+"
        return rf"\log(p_T) {sign} {abs(self.pt_shift):.4g}"

    def as_kwargs(self) -> dict[str, float]:
        """Keyword arguments for ORBIT datasets / ``OrbitPreprocessor``."""
        return {"eta_scale": self.eta_scale, "pt_shift": self.pt_shift}

    # Forward/inverse helpers work on numpy, awkward and (for eta) torch arrays.
    def eta_forward(self, eta):
        return eta / self.eta_scale

    def eta_inverse(self, eta_transformed):
        return eta_transformed * self.eta_scale

    def pt_forward(self, pt):
        return np.log(pt + self.epsilon) - self.pt_shift

    def pt_inverse(self, pt_transformed):
        return np.clip(np.exp(pt_transformed + self.pt_shift) - self.epsilon, 0.0, None)

    def to_physical(self, values: np.ndarray) -> np.ndarray:
        """Convert ``[..., >=4]`` transformed features to ``[..., 3]`` = ``[eta, phi, pT]``."""
        values = np.asarray(values)
        return np.stack(
            [
                self.eta_inverse(values[..., 0]),
                np.arctan2(values[..., 2], values[..., 1]),
                self.pt_inverse(values[..., 3]),
            ],
            axis=-1,
        )
