"""Composition checks for the charged-lepton-only 5M campaign."""

from pathlib import Path

from hydra import compose, initialize_config_dir


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
EXPERIMENT = "orbit_campaign_full500_leptonic_vq4096_5m"


def test_leptonic_5m_campaign_contract(monkeypatch):
    monkeypatch.setenv(
        "ORBIT_MANIFEST_DIR",
        str(REPO_ROOT / "manifests" / "production_final"),
    )

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.2"):
        cfg = compose(config_name="train.yaml", overrides=[f"experiment={EXPERIMENT}"])

    expected_groups = {
        "tt_leptonic": "tt",
        "WJetsToLNu": "VJets",
        "DYJetsToLL": "DY",
        "WW_leptonic": "VV",
        "WZ_leptonic": "VV",
        "ZZ_leptonic": "VV",
    }
    assert {
        name: process.group for name, process in cfg.data.train_val_processes.items()
    } == expected_groups
    assert cfg.data.get("train_event_quotas") is None
    assert cfg.data.get("val_event_quotas") is None
    assert cfg.data.train_event_budget == 5_000_000
    assert cfg.data.val_event_budget == 1_250_000
    assert cfg.data.sequence_type == "particle_full"
    assert cfg.data.max_sequence_length == 500
    assert cfg.data.batch_size == 256
    assert cfg.trainer.accumulate_grad_batches == 1
    assert cfg.trainer.max_epochs == 30
    assert cfg.callbacks.early_stopping.patience == 4
    assert cfg.callbacks.orbit_plotting_callback.validation_plot_every_n_epochs == 5
    assert cfg.model.model_kwargs.vq_kwargs.num_codes == 4096
    assert cfg.model.model_kwargs.alpha == 10
    assert cfg.model.optimizer.lr == 1e-4
    assert cfg.model.scheduler is None
    assert cfg.train is True
    assert cfg.test is False
    assert set(cfg.data.test_suites) == {"ttbar", "sm_mixture", "gghbb", "minbias"}
