"""Composition checks for the eight-model 5M-event campaign."""

from pathlib import Path

from hydra import compose, initialize_config_dir

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
EXPERIMENTS = (
    "orbit_campaign_full500_tt_vq4096_5m",
    "orbit_campaign_full500_sm_vq4096_5m",
    "orbit_campaign_full500_sm_pidweighted_vq4096_5m",
    "orbit_campaign_puppi128_sm_vq4096_5m",
    "orbit_campaign_puppi128_tt_vq4096_5m",
    "orbit_campaign_puppi128_sm_pidweighted_vq4096_5m",
    "orbit_campaign_full500_sm_vq8192_5m",
    "orbit_campaign_full500_sm_faiss4096_5m",
)
TT_EXPERIMENTS = {
    "orbit_campaign_full500_tt_vq4096_5m",
    "orbit_campaign_puppi128_tt_vq4096_5m",
}
FAISS_EXPERIMENT = "orbit_campaign_full500_sm_faiss4096_5m"
JET_TOKEN_EXPERIMENTS = (
    "orbit_campaign_full500_sm_jettoken_control_vq4096_5m",
    "orbit_campaign_full500_sm_jettoken_noloss_vq4096_5m",
    "orbit_campaign_full500_sm_jettoken_loss_vq4096_5m",
)


def test_all_5m_campaign_configs_compose(monkeypatch):
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

            assert cfg.data.train_event_budget == 5_000_000
            assert cfg.data.val_event_budget == 1_250_000
            assert "5m" in cfg.tags
            assert cfg.task_name == experiment
            assert set(cfg.data.test_suites) == {
                "ttbar",
                "sm_mixture",
                "gghbb",
                "minbias",
            }
            assert cfg.data.test_suites.ttbar.event_budget == 16_073_004
            assert cfg.data.test_suites.sm_mixture.event_budget == 16_073_004
            assert cfg.data.test_suites.gghbb.event_budget == 2_098_560
            assert cfg.data.test_suites.minbias.event_budget == 16_073_004

            if experiment != FAISS_EXPERIMENT:
                assert cfg.model.model_kwargs.alpha == 10
                assert cfg.model.optimizer.lr == 1e-4
                assert cfg.model.scheduler is None
                assert "alpha-10" in cfg.tags
                assert "lr-1e-4" in cfg.tags
                assert "constant-lr" in cfg.tags

            if experiment in TT_EXPERIMENTS:
                assert sum(cfg.data.train_event_quotas.values()) == 5_000_000
                assert sum(cfg.data.val_event_quotas.values()) == 1_250_000
            else:
                assert cfg.data.get("train_event_quotas") is None
                assert cfg.data.get("val_event_quotas") is None


def test_5m_campaign_preserves_model_and_selection_variants(monkeypatch):
    monkeypatch.setenv(
        "ORBIT_MANIFEST_DIR",
        str(REPO_ROOT / "manifests" / "production_final"),
    )

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.2"):
        configs = {
            experiment: compose(
                config_name="train.yaml",
                overrides=[f"experiment={experiment}"],
            )
            for experiment in EXPERIMENTS
        }

    for experiment, cfg in configs.items():
        if "puppi128" in experiment:
            assert cfg.data.sequence_type == "particle"
            assert cfg.data.max_sequence_length == 128
            assert cfg.data.batch_size == 256
            assert cfg.trainer.accumulate_grad_batches == 1
        else:
            assert cfg.data.sequence_type == "particle_full"
            assert cfg.data.max_sequence_length == 500
            assert cfg.data.batch_size == 256
            assert cfg.trainer.accumulate_grad_batches == 1

    vq8192 = configs["orbit_campaign_full500_sm_vq8192_5m"]
    vq4096 = configs["orbit_campaign_full500_sm_vq4096_5m"]
    assert vq8192.model.model_kwargs.vq_kwargs.num_codes == 8192
    assert vq4096.model.model_kwargs.vq_kwargs.num_codes == 4096

    faiss = configs["orbit_campaign_full500_sm_faiss4096_5m"]
    assert faiss.model.num_codes == 4096
    assert faiss.model.fit_max_particles == 10_000_000
    assert faiss.train is False
    assert faiss.test is True

    full_pid = configs["orbit_campaign_full500_sm_pidweighted_vq4096_5m"]
    puppi_pid = configs["orbit_campaign_puppi128_sm_pidweighted_vq4096_5m"]
    assert full_pid.pid.reconstruction_class_weights is not None
    assert puppi_pid.pid.reconstruction_class_weights is not None


def test_5m_jet_token_ablation_configs_differ_only_as_intended(monkeypatch):
    monkeypatch.setenv(
        "ORBIT_MANIFEST_DIR",
        str(REPO_ROOT / "manifests" / "production_final"),
    )

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.2"):
        configs = {
            experiment: compose(
                config_name="train.yaml",
                overrides=[f"experiment={experiment}"],
            )
            for experiment in JET_TOKEN_EXPERIMENTS
        }

    control, no_loss, with_loss = (configs[name] for name in JET_TOKEN_EXPERIMENTS)
    for experiment, cfg in configs.items():
        assert cfg.task_name == experiment
        assert cfg.data.train_event_budget == 5_000_000
        assert cfg.data.val_event_budget == 1_250_000
        assert cfg.data.sequence_type == "particle_full"
        assert cfg.data.max_sequence_length == 500
        assert cfg.data.max_auxiliary_jets == 5
        assert cfg.data.batch_size == 256
        assert cfg.trainer.accumulate_grad_batches == 1
        assert cfg.model.model_kwargs.max_sequence_len == 505
        assert cfg.model.model_kwargs.token_type_count == 2
        assert cfg.model.model_kwargs.vq_kwargs.num_codes == 4096
        assert cfg.model.model_kwargs.alpha == 10
        assert cfg.model.optimizer.lr == 1e-4
        assert cfg.model.scheduler is None

    assert control.data.auxiliary_jet_sequence_type is None
    assert no_loss.data.auxiliary_jet_sequence_type == "jet_puppi_ak4"
    assert with_loss.data.auxiliary_jet_sequence_type == "jet_puppi_ak4"
    assert control.model.model_kwargs.jet_reconstruction_loss_weight == 0
    assert no_loss.model.model_kwargs.jet_reconstruction_loss_weight == 0
    assert with_loss.model.model_kwargs.jet_reconstruction_loss_weight == 1
