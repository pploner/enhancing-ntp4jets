#!/usr/bin/env python
"""Build restartable particle, jet, PID, Higgs, and tau32 caches for the 5M suite."""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for root in (PROJECT_ROOT / "vqtorch", PROJECT_ROOT):
    sys.path.insert(0, str(root))

import numpy as np
import torch
import yaml
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from gabbro.callbacks.orbit_plotting_callback import match_jets_by_delta_r
from gabbro.data.orbit_parquet import OrbitParquetDataset, SEQUENCE_SCHEMAS
from gabbro.plotting.orbit import (
    PHYSICAL_RESIDUAL_RANGES,
    code_entropy_metrics,
    collect_physical_reconstruction_histograms,
    pid_residual_arrays,
)
from scripts.evaluate_orbit_tau32 import reconstruct_jets, tau32, to_physical


SUITES = {
    "ttbar": (
        ("tt", ("tt0123j_5f_ckm_LO_MLM_hadronic_test.txt", "tt0123j_5f_ckm_LO_MLM_leptonic_test.txt", "tt0123j_5f_ckm_LO_MLM_semiLeptonic_test.txt")),
    ),
    "sm_mixture": (
        ("QCD", ("QCD_HT50tobb_test.txt", "QCD_HT50toInf_test.txt")),
        ("tt", ("tt0123j_5f_ckm_LO_MLM_hadronic_test.txt", "tt0123j_5f_ckm_LO_MLM_leptonic_test.txt", "tt0123j_5f_ckm_LO_MLM_semiLeptonic_test.txt")),
        ("VJets", ("WJetsToLNu_13TeV-madgraphMLM-pythia8_test.txt", "WJetsToQQ_13TeV-madgraphMLM-pythia8_test.txt", "ZJetsTobb_13TeV-madgraphMLM-pythia8_test.txt", "ZJetsTocc_13TeV-madgraphMLM-pythia8_test.txt", "ZJetsToQQ_13TeV-madgraphMLM-pythia8_test.txt", "ZJetsTovv_13TeV-madgraphMLM-pythia8_test.txt")),
        ("VV", ("WW_hadronic_test.txt", "WW_leptonic_test.txt", "WW_semileptonic_test.txt", "WZ_hadronic_test.txt", "WZ_leptonic_test.txt", "WZ_semileptonic_test.txt", "ZZ_hadronic_test.txt", "ZZ_leptonic_test.txt", "ZZ_semileptonic_test.txt")),
        ("DY", ("DYJetsToLL_13TeV-madgraphMLM-pythia8_test.txt",)),
    ),
    "gghbb": (("ggHbb", ("ggHbb_test.txt",)),),
    "minbias": (("minbias", ("minbias.txt",)),),
    "tau32": (("tt_hadronic", ("tt0123j_5f_ckm_LO_MLM_hadronic_test.txt",)),),
}


def split_quota(total: int, count: int) -> list[int]:
    return [total // count + (index < total % count) for index in range(count)]


def load_exported_config(path: Path):
    payload = yaml.safe_load(path.read_text())
    if isinstance(payload.get("data"), dict) and "value" in payload["data"]:
        unwrapped = {}
        for key, entry in payload.items():
            value = entry.get("value") if isinstance(entry, dict) and "value" in entry else entry
            if isinstance(value, str) and value.lstrip().startswith(("{", "[")):
                try:
                    value = ast.literal_eval(value)
                except (SyntaxError, ValueError):
                    pass
            unwrapped[key] = value
        payload = unwrapped
    # W&B exports the canonical datamodule's interpolated PID block under the
    # flattened name ``pid_cfg`` rather than the original Hydra key ``pid``.
    # Recreate that key before OmegaConf resolves model/data interpolations.
    if "pid" not in payload and "pid_cfg" in payload:
        payload["pid"] = payload["pid_cfg"]
    return OmegaConf.create(payload)


def load_registry(path: Path, selection: str):
    rows = yaml.safe_load(path.read_text())["models"]
    selected = [row for row in rows if row["selection"] == selection]
    if not selected or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("The model registry must contain unique IDs and a selected model")
    return selected


def load_model(spec, device):
    config_path = Path(spec["config"])
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    cfg = load_exported_config(config_path)
    configured_length = cfg.data.get("max_sequence_length")
    resolved_length = (
        SEQUENCE_SCHEMAS[str(cfg.data.sequence_type)]["max_sequence_length"]
        if configured_length is None
        else int(configured_length)
    )
    if resolved_length != (500 if spec["selection"] == "full500" else 128):
        raise ValueError(f"Selection/config mismatch for {spec['id']}")
    cfg.data.max_sequence_length = resolved_length
    model = instantiate(cfg.model)
    if spec["quantizer"] == "FAISS k-means":
        codebook = Path(spec["codebook"])
        if not codebook.is_file():
            raise FileNotFoundError(codebook)
        model.codebook_path = codebook
        model._fit_faiss()
    else:
        checkpoint = Path(spec["checkpoint"])
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)  # nosec
        model.load_state_dict(state["state_dict"], strict=True)
    return model.to(device).eval(), cfg


def decode(model, features, mask, pid):
    if hasattr(model, "_quantize_batch"):
        reco, codes, pid_logits = model._quantize_batch(features, mask, pid)
        return reco, codes, mask, pid_logits
    reco, output = model(features, mask, pid_particle=pid)
    code_mask = output.get("latent_mask", output.get("combined_mask", mask))
    return reco, output["q"], code_mask, output.get("pid_logits")


def iter_batches(cfg, manifest_dir: Path, suite: str, events: int, batch_size: int):
    schema = SEQUENCE_SCHEMAS[str(cfg.data.sequence_type)]
    groups = SUITES[suite]
    for (_, filenames), class_quota in zip(groups, split_quota(events, len(groups)), strict=True):
        for filename, quota in zip(filenames, split_quota(class_quota, len(filenames)), strict=True):
            manifest = manifest_dir / filename
            if not manifest.is_file():
                raise FileNotFoundError(manifest)
            dataset = OrbitParquetDataset(
                manifest,
                sequence_type=str(cfg.data.sequence_type),
                max_sequence_length=int(cfg.data.max_sequence_length),
                batch_size=batch_size,
                shuffle_row_groups=False,
                mask_column=cfg.data.get("mask_column") or schema["mask_column"],
                mask_min_value=float(cfg.data.get("mask_min_value") if cfg.data.get("mask_min_value") is not None else schema["mask_min_value"]),
                max_events=quota,
                pid_cfg=cfg.get("pid"),
                include_energy=bool(cfg.data.get("include_energy", False)),
                energy_shift=float(cfg.data.get("energy_shift", 2.5)),
            )
            yield from dataset


def dijet_mass(jets):
    pt, eta, phi = jets[:3]
    if len(pt) < 2:
        return np.nan
    order = np.argsort(pt)[::-1][:2]
    px = pt[order] * np.cos(phi[order])
    py = pt[order] * np.sin(phi[order])
    pz = pt[order] * np.sinh(eta[order])
    energy = pt[order] * np.cosh(eta[order])
    mass2 = np.sum(energy) ** 2 - np.sum(px) ** 2 - np.sum(py) ** 2 - np.sum(pz) ** 2
    return float(np.sqrt(max(mass2, 0.0)))


def atomic_savez(path: Path, **values):
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **values)
    os.replace(temporary, path)


def evaluate_general(spec, model, cfg, args, output: Path):
    originals, decoded = [], []
    jet_values = {name: [] for name in ("delta_pt", "delta_eta", "delta_phi", "match_dr")}
    pid_original, pid_residual = [], {name: [] for name in ("delta_eta", "delta_phi", "delta_log_pt")}
    code_counts = np.zeros(int(spec["codes"]), dtype=np.int64)
    mse_sum = 0.0
    valid_values = 0
    input_particles = latent_tokens = processed = 0
    higgs = {name: [] for name in ("n_jets_original", "n_jets_decoded", "mass_original", "mass_decoded")}
    bar = tqdm(total=args.events, desc=f"{spec['id']} {args.suite}", unit="event")
    for batch in iter_batches(cfg, args.manifest_dir, args.suite, args.events, args.batch_size):
        features = batch["part_features"].to(args.device)
        mask_t = batch["part_mask"].to(args.device)
        pid_t = batch.get("part_pid")
        if pid_t is not None:
            pid_t = pid_t.to(args.device)
        with torch.inference_mode():
            reco_t, codes_t, code_mask_t, pid_logits = decode(model, features, mask_t, pid_t)
        mask = mask_t.cpu().numpy().astype(bool)
        original = to_physical(features.cpu().numpy())
        reco = to_physical(reco_t.cpu().numpy())
        flat_original, flat_reco = original[mask], reco[mask]
        originals.append(flat_original.astype(np.float32))
        decoded.append(flat_reco.astype(np.float32))
        delta = (reco_t - features).cpu().numpy()[mask]
        mse_sum += float(np.sum(delta**2))
        valid_values += int(delta.size)
        input_particles += int(mask.sum())
        codes = codes_t.detach().cpu().numpy()
        code_mask = code_mask_t.detach().cpu().numpy().astype(bool)
        if codes.ndim == code_mask.ndim + 1 and codes.shape[-1] == 1:
            codes = codes[..., 0]
        valid_codes = codes[code_mask].astype(np.int64)
        code_counts += np.bincount(valid_codes, minlength=len(code_counts))[: len(code_counts)]
        latent_tokens += len(valid_codes)
        if spec.get("pid_weighted"):
            pids = pid_t.cpu().numpy()[mask]
            feature_names = (
                "L1T_PUPPIPart_Eta",
                "L1T_PUPPIPart_Phi_cos",
                "L1T_PUPPIPart_Phi_sin",
                "L1T_PUPPIPart_PT",
            )
            if features.shape[-1] == 5:
                feature_names += ("L1T_PUPPIPart_E",)
            residuals = pid_residual_arrays(
                features.cpu().numpy()[mask],
                reco_t.cpu().numpy()[mask],
                feature_names,
            )
            pid_original.append(pids.astype(np.int8))
            for key in pid_residual:
                pid_residual[key].append(residuals[key].astype(np.float32))
        for event in range(len(mask)):
            oj = reconstruct_jets(original[event, mask[event]], 0.4, 30.0)[:3]
            dj = reconstruct_jets(reco[event, mask[event]], 0.4, 30.0)[:3]
            oi, di, dr = match_jets_by_delta_r(oj[1], oj[2], dj[1], dj[2], max_delta_r=None)
            if len(oi):
                jet_values["delta_pt"].extend((dj[0][di] - oj[0][oi]) / oj[0][oi])
                jet_values["delta_eta"].extend(dj[1][di] - oj[1][oi])
                jet_values["delta_phi"].extend(np.remainder(dj[2][di] - oj[2][oi] + np.pi, 2 * np.pi) - np.pi)
                jet_values["match_dr"].extend(dr)
            if args.suite == "gghbb":
                higgs["n_jets_original"].append(len(oj[0]))
                higgs["n_jets_decoded"].append(len(dj[0]))
                higgs["mass_original"].append(dijet_mass(oj))
                higgs["mass_decoded"].append(dijet_mass(dj))
        processed += len(mask)
        bar.update(len(mask))
    bar.close()
    original = np.concatenate(originals)
    reco = np.concatenate(decoded)
    hist = collect_physical_reconstruction_histograms(["Eta", "Phi", "pT"], original, reco)
    active = code_counts[code_counts > 0]
    probabilities = active / max(active.sum(), 1)
    entropy = float(-np.sum(probabilities * np.log2(probabilities)))
    values = {f"particle_{key}": value for key, value in hist.items() if key.endswith(("_diff_counts", "_diff_bins"))}
    values.update({f"jet_{key}": np.asarray(value, dtype=np.float32) for key, value in jet_values.items()})
    values.update({f"higgs_{key}": np.asarray(value) for key, value in higgs.items()})
    if pid_original:
        values["pid_original"] = np.concatenate(pid_original)
        values.update({f"pid_{key}": np.concatenate(chunks) for key, chunks in pid_residual.items()})
    atomic_savez(output, **values)
    summary = {
        "model": spec,
        "suite": args.suite,
        "events": processed,
        "particle_residual": "decoded - original; phi wrapped",
        "jet_definition": "anti-kt R=0.4, pT >= 30 GeV",
        "jet_pairing": "Hungarian minimum total delta-R, no delta-R rejection",
        "matched_jets": len(jet_values["match_dr"]),
        "mean_match_delta_r": float(np.mean(jet_values["match_dr"])) if jet_values["match_dr"] else None,
        "metrics/mse_total": mse_sum / max(valid_values, 1),
        "metrics/rate/marginal_bits_per_input_particle": entropy * latent_tokens / max(input_particles, 1),
        "metrics/rate/sample_input_particles": input_particles,
        "metrics/rate/sample_latent_tokens": latent_tokens,
        "metrics/active_codes_total": int(np.count_nonzero(code_counts)),
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")


def evaluate_tau(spec, model, cfg, args, output: Path):
    values = {name: [] for name in ("original_tau32", "decoded_tau32", "match_dr", "original_count", "decoded_count")}
    bar = tqdm(total=args.events, desc=f"{spec['id']} tau32", unit="event")
    for batch in iter_batches(cfg, args.manifest_dir, "tau32", args.events, args.batch_size):
        features = batch["part_features"].to(args.device)
        mask_t = batch["part_mask"].to(args.device)
        pid_t = batch.get("part_pid")
        if pid_t is not None:
            pid_t = pid_t.to(args.device)
        with torch.inference_mode():
            reco_t, _, _, _ = decode(model, features, mask_t, pid_t)
        mask = mask_t.cpu().numpy().astype(bool)
        original, reco = to_physical(features.cpu().numpy()), to_physical(reco_t.cpu().numpy())
        for event in range(len(mask)):
            oj = reconstruct_jets(original[event, mask[event]], 0.8, 30.0)
            dj = reconstruct_jets(reco[event, mask[event]], 0.8, 30.0)
            values["original_count"].append(int(np.sum(oj[0] >= 450.0)))
            values["decoded_count"].append(int(np.sum(dj[0] >= 450.0)))
            om = oj[0] >= 450.0
            oi, di, dr = match_jets_by_delta_r(oj[1][om], oj[2][om], dj[1], dj[2], max_delta_r=0.4)
            if len(oi):
                oc = [parts for parts, keep in zip(oj[3], om) if keep]
                ot = tau32([oc[index] for index in oi], 0.8)
                dt = tau32([dj[3][index] for index in di], 0.8)
                finite = np.isfinite(ot) & np.isfinite(dt)
                values["original_tau32"].extend(ot[finite])
                values["decoded_tau32"].extend(dt[finite])
                values["match_dr"].extend(dr[finite])
        bar.update(len(mask))
    bar.close()
    atomic_savez(output, **{key: np.asarray(value) for key, value in values.items()})
    output.with_suffix(".json").write_text(json.dumps({"model": spec, "events": args.events, "selection": "all-hadronic ttbar; AK8 original pT>=450 GeV; decoded pT>=30 GeV; Hungarian delta-R<0.4", "matched_jets": len(values["match_dr"])}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--selection", choices=("full500", "puppi128"), required=True)
    parser.add_argument("--suite", choices=tuple(SUITES), required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--events", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for spec in load_registry(args.registry, args.selection):
        output = args.output_dir / args.suite / f"{spec['id']}.npz"
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.is_file() and output.with_suffix(".json").is_file():
            print(f"Already complete: {output}")
            continue
        model, cfg = load_model(spec, args.device)
        if args.suite == "tau32":
            evaluate_tau(spec, model, cfg, args, output)
        else:
            evaluate_general(spec, model, cfg, args, output)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
