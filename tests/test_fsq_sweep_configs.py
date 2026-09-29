import re
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATTERN = re.compile(r"model_vqvae_transformer_split_[a-zA-Z0-9_]+")


def _models_from_submit(submit_name: str) -> set[str]:
    submit_text = (REPO_ROOT / "condor" / submit_name).read_text()
    return set(MODEL_PATTERN.findall(submit_text))


def _branch_loss_weights(model_name: str) -> dict[str, float]:
    config_path = REPO_ROOT / "configs" / "model" / f"{model_name}.yaml"
    config = yaml.safe_load(config_path.read_text())
    weights = (
        config.get("model_kwargs", {})
        .get("split_quantizer_cfg", {})
        .get("branch_loss_weights")
    )
    if weights is not None:
        return weights
    for parent in config.get("defaults", []):
        if isinstance(parent, str) and parent.startswith(
            "model_vqvae_transformer_split"
        ):
            return _branch_loss_weights(parent)
    raise AssertionError(f"No branch loss weights found for {model_name}")


def test_plain_fsq_sweep_has_no_auxiliary_loss():
    models = _models_from_submit("orbit_fsq_codebook_scan_canonical.sub")
    assert len(models) == 7
    for model in models:
        assert _branch_loss_weights(model) == {"mu": 0.0, "alpha": 0.0}


def test_phaedra_sweeps_only_weight_morphology_loss():
    submits = [
        f"orbit_split_{prefix}_alpha{alpha}_scan_canonical.sub"
        for prefix in ("fsq", "vq_mu_fsq")
        for alpha in (32, 64, 128)
    ]
    models = set().union(*(_models_from_submit(submit) for submit in submits))
    assert len(models) == 48
    for model in models:
        assert _branch_loss_weights(model) == {"mu": 0.25, "alpha": 0.0}


def test_all_fsq_condor_sweeps_follow_auxiliary_loss_policy():
    models = set()
    for submit_path in (REPO_ROOT / "condor").glob("*fsq*scan*.sub"):
        models.update(MODEL_PATTERN.findall(submit_path.read_text()))
    assert len(models) == 65
    for model in models:
        expected = {"mu": 0.25, "alpha": 0.0} if "alpha" in model else {
            "mu": 0.0,
            "alpha": 0.0,
        }
        assert _branch_loss_weights(model) == expected
