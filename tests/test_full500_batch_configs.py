"""Regression checks for the no-micro-batching full-500 policy."""

from pathlib import Path

from hydra import compose, initialize_config_dir


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
EXPERIMENTS = (
    "orbit_canonical_tt_full_event",
    "orbit_canonical_qcd_tt_vjets_vv_full_event",
    "orbit_downstream_event_classifier_full_event",
    "orbit_campaign_full500_tt_vq4096",
    "orbit_campaign_full500_sm_vq4096",
    "orbit_campaign_full500_sm_pidweighted_vq4096",
    "orbit_campaign_full500_sm_vq8192",
    "orbit_campaign_full500_sm_faiss4096",
    "orbit_campaign_full500_tt_vq4096_5m",
    "orbit_campaign_full500_sm_vq4096_5m",
    "orbit_campaign_full500_sm_pidweighted_vq4096_5m",
    "orbit_campaign_full500_sm_vq8192_5m",
    "orbit_campaign_full500_sm_faiss4096_5m",
)


def test_full500_configs_use_one_physical_batch(monkeypatch):
    monkeypatch.setenv(
        "ORBIT_MANIFEST_DIR",
        str(REPO_ROOT / "manifests" / "production_final"),
    )

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.2"):
        for experiment in EXPERIMENTS:
            cfg = compose(
                config_name="train.yaml",
                overrides=[f"experiment={experiment}"],
            )
            assert cfg.data.max_sequence_length == 500
            assert cfg.data.batch_size == 256
            assert cfg.trainer.accumulate_grad_batches == 1
