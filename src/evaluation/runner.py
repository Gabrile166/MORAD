"""End-to-end RIDE evaluation following the public RiboDiffusion protocol."""

from __future__ import annotations

import copy
import hashlib
import json
import random
import re
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from src.rl.config import load_config
from src.rl.observability import build_run_manifest, sha256_file
from src.rl.reward import (
    FoldResult,
    RewardComposer,
    RhoFoldJsonlClient,
    compute_structural_metrics,
    load_c4p_coords,
)
from src.rl.rollout import AdaptiveX0RenoiseRollout, RolloutState
from src.rl.runtime import build_noise_scheduler, load_ride_model
from src.rl.targets import load_target_pool, materialize_condition

from .adapters import InfernalCmsearchAdapter, RNAfoldAdapter
from .metrics import (
    base_pair_scores,
    intdiv,
    length_bucket,
    parse_extended_dot_bracket,
    sequence_recovery,
)
from .protocol import MetricResult, MetricStatus
from .tertiary import USAlignC1Prime, write_c1prime_pdb


PROTOCOL_VERSION = "rider_ribodiffusion_reconstruction_v3"
PAPER_URL = "https://arxiv.org/html/2404.11199"


@dataclass(frozen=True)
class EvaluationSample:
    """Generator-agnostic candidate metadata consumed by the metric suite."""

    sample_id: str
    seed: int
    temperature: float
    latency_ms: float
    generator: str


class EvaluationFoldOracle:
    """Content-addressed RhoFold+ cache shared by native and design folds."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self.checkpoint = Path(str(self.config["checkpoint"]))
        self.checkpoint_hash = sha256_file(self.checkpoint)
        self.cache_dir = Path(str(self.config.get("cache_dir", "cache/evaluation/rhofold")))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        command = [
            str(self.config["python"]),
            str(self.config["worker_script"]),
            "--device",
            str(self.config.get("device", "cuda:0")),
            "--ckpt",
            str(self.checkpoint),
        ]
        self.client = RhoFoldJsonlClient(
            command,
            timeout_s=float(self.config.get("timeout_s", 300.0)),
            output_root=self.cache_dir,
        )
        self.hits = 0
        self.misses = 0

    def close(self) -> None:
        self.client.close()

    def fold(self, sequence: str, target_id: str) -> tuple[FoldResult, bool]:
        key_payload = {
            "sequence": sequence,
            "checkpoint_sha256": self.checkpoint_hash,
            "device_mode": str(self.config.get("device", "cuda:0")),
            "relax_steps": 0,
            "worker_version": "rhofold_plus_jsonl_v1",
        }
        key = hashlib.sha256(json.dumps(key_payload, sort_keys=True).encode("utf-8")).hexdigest()
        fold_dir = self.cache_dir / key[:2] / key
        record_path = fold_dir / "fold.json"
        if record_path.is_file():
            try:
                payload = json.loads(record_path.read_text(encoding="utf-8"))
                pdb_path = payload.get("predicted_structure_path")
                if payload.get("status") != "ok" or (pdb_path and Path(pdb_path).is_file()):
                    self.hits += 1
                    return FoldResult(**payload), True
            except (OSError, ValueError, TypeError):
                pass
        self.misses += 1
        fold_dir.mkdir(parents=True, exist_ok=True)
        result = self.client.fold(sequence, fold_dir, target_id=target_id)
        record_path.write_text(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return result, False


class RiboDiffusionEvaluator:
    def __init__(self, config: Mapping[str, Any], *, repo_root: str | Path, output_dir: str | Path) -> None:
        self.config = copy.deepcopy(dict(config))
        self.repo_root = Path(repo_root).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_config = dict(self.config.get("metrics", {}))
        self.events_path = self.output_dir / "events.jsonl"

    def run(self) -> dict[str, Any]:
        started = time.time()
        sampling_config = dict(self.config.get("sampling", {}))
        seed = int(sampling_config.get("seed", 0))
        _set_seed(seed)
        ride_cfg_path = self._path(self.config["ride"]["rl_config"])
        ride_config = load_config(ride_cfg_path)
        targets = load_target_pool(self._path(self.config["ride"].get("pool_manifest") or ride_config.data.pool_manifest))
        target_limit = int(sampling_config.get("target_limit", 1))
        targets = targets[:target_limit]
        generator = str(sampling_config.get("generator", "ride"))
        if generator not in {"ride", "precomputed"}:
            raise ValueError("sampling.generator must be 'ride' or 'precomputed'")

        model = None
        rollout = None
        device = torch.device("cpu")
        if generator == "ride":
            device = torch.device(
                self.config.get("ride", {}).get("device")
                or (
                    ride_config.runtime.device
                    if ride_config.runtime.device != "auto"
                    else ("cuda" if torch.cuda.is_available() else "cpu")
                )
            )
            model = load_ride_model(ride_config, device)
            rollout_config = copy.deepcopy(ride_config.to_dict()["rollout"])
            rollout_config["group_size"] = int(sampling_config.get("n_samples", 8))
            rollout_config.setdefault("temperature", {})["initial"] = float(sampling_config.get("temperature", 0.8))
            rollout_config.setdefault("rescue", {})["max_rounds"] = 0
            rollout = AdaptiveX0RenoiseRollout(build_noise_scheduler(ride_config), rollout_config)

        secondary_cfg = dict(self.metrics_config.get("secondary", {}))
        rnafold = RNAfoldAdapter(
            self._path(secondary_cfg.get("rnafold_binary", "RNAfold"), require_relative_exists=False),
            timeout_s=float(secondary_cfg.get("timeout_s", 30.0)),
        )
        tertiary_cfg = dict(self.metrics_config.get("tertiary", {}))
        tertiary_enabled = bool(tertiary_cfg.get("enabled", True))
        fold_oracle = EvaluationFoldOracle(self._resolved_tertiary_config(tertiary_cfg)) if tertiary_enabled else None
        usalign = USAlignC1Prime(
            self._path(tertiary_cfg.get("usalign_binary", "USalign"), require_relative_exists=False),
            timeout_s=float(tertiary_cfg.get("usalign_timeout_s", 120.0)),
        )

        target_results: list[dict[str, Any]] = []
        candidate_results: list[dict[str, Any]] = []
        try:
            for target_index, target in enumerate(targets):
                self._event("target_started", {"target_id": target.target_id, "target_index": target_index})
                if generator == "ride":
                    assert rollout is not None and model is not None
                    condition = materialize_condition(target, device=device, round_id=target_index)
                    state = RolloutState(
                        round_id=target_index,
                        step=target_index,
                        base_seed=seed + target_index * 100000,
                        policy_version="ride-evaluation",
                        temperature=float(sampling_config.get("temperature", 0.8)),
                    )
                    rollout_batch = rollout.generate(target, condition, model, state)
                    sequences = [sample.tokens for sample in rollout_batch.samples]
                    samples = rollout_batch.samples
                else:
                    sequences, samples = self._load_precomputed_candidates(target)
                target_result, candidates = self._evaluate_target(
                    target=target,
                    sequences=sequences,
                    samples=samples,
                    rnafold=rnafold,
                    fold_oracle=fold_oracle,
                    usalign=usalign,
                )
                target_results.append(target_result)
                candidate_results.extend(candidates)
                self._event("target_completed", {"target_id": target.target_id, "metrics": target_result["metrics"]})
        finally:
            if fold_oracle is not None:
                fold_oracle.close()

        summary = summarize_results(target_results, candidate_results)
        manifest = self._manifest(
            ride_config=ride_config,
            rnafold_version=rnafold.probe_version(),
            usalign_version=usalign.version(),
            fold_oracle=fold_oracle,
            elapsed_s=time.time() - started,
        )
        result = {
            "schema_version": self.config["schema_version"],
            "protocol_version": PROTOCOL_VERSION,
            "paper": PAPER_URL,
            "generator": generator,
            "comparability": "RiboDiffusion-style metrics on the configured RIDE split; not directly comparable to the paper tables unless the original clustered splits are used.",
            "summary": summary,
            "targets": target_results,
            "candidates": candidate_results,
            "manifest": manifest,
        }
        from .reporting import write_evaluation_artifacts

        write_evaluation_artifacts(result, self.output_dir)
        return result

    def _load_precomputed_candidates(self, target: Any) -> tuple[list[str], list[EvaluationSample]]:
        config = dict(self.config.get("precomputed", {}))
        root_value = config.get("directory")
        if not root_value:
            raise ValueError("precomputed.directory is required when sampling.generator=precomputed")
        root = self._path(root_value)
        pattern = str(config.get("pattern", "fasta/{target_id}_*.fasta")).format(target_id=target.target_id)
        paths = sorted(root.glob(pattern), key=_natural_path_key)
        expected = int(self.config.get("sampling", {}).get("n_samples", 8))
        if len(paths) != expected:
            raise ValueError(
                f"Expected {expected} precomputed FASTA files for {target.target_id} using {root / pattern}, "
                f"found {len(paths)}"
            )
        generator_name = str(config.get("model_name", "precomputed"))
        sequences: list[str] = []
        samples: list[EvaluationSample] = []
        for index, path in enumerate(paths):
            sequence = _read_single_fasta(path)
            if len(sequence) != target.length:
                raise ValueError(
                    f"Candidate length mismatch for {path}: expected {target.length}, got {len(sequence)}"
                )
            sequences.append(sequence)
            samples.append(
                EvaluationSample(
                    sample_id=f"{target.target_id}:{generator_name}:{index:02d}",
                    seed=-1,
                    temperature=0.0,
                    latency_ms=0.0,
                    generator=generator_name,
                )
            )
        return sequences, samples

    def _evaluate_target(
        self,
        *,
        target: Any,
        sequences: Sequence[str],
        samples: Sequence[Any],
        rnafold: RNAfoldAdapter,
        fold_oracle: EvaluationFoldOracle | None,
        usalign: USAlignC1Prime,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        family = _conformer_value(target, "rfam_list")
        rna_type = _conformer_value(target, "type_list")
        target_secondary = _conformer_value(target, "sec_struct_list")
        native_secondary_metric = self._secondary_metric(
            target.sequence_native,
            target_secondary,
            rnafold,
            metric_name="native_secondary_structure_f1",
        )
        secondary_config = dict(self.metrics_config.get("secondary", {}))
        native_f1_threshold = float(secondary_config.get("native_f1_threshold", 0.7))
        secondary_eligible = (
            native_secondary_metric.status == MetricStatus.OK
            and native_secondary_metric.value is not None
            and float(native_secondary_metric.value) >= native_f1_threshold
        )
        native_rfam_metric = self._rfam_metric(target.sequence_native, family, f"{target.target_id}:native")
        target_pdb: Path | None = None
        native_metric = MetricResult.skipped("native_rhofold_c1prime_tm_score", "tertiary metric disabled")
        native_reward_metrics = _skipped_reward_like_metrics("native_", "tertiary metric disabled")
        if fold_oracle is not None:
            if target.reference_c1p_coords is not None:
                target_pdb = write_c1prime_pdb(
                    target.reference_c1p_coords,
                    target.sequence_native,
                    self.output_dir / "target_structures" / f"{target.target_id}.c1prime.pdb",
                )
            native_fold, cache_hit = fold_oracle.fold(target.sequence_native, target.target_id)
            if target_pdb is not None:
                native_metric = _tertiary_metric(
                    native_fold,
                    target_pdb,
                    usalign,
                    cache_hit,
                    "native_rhofold_c1prime_tm_score",
                )
            else:
                native_metric = MetricResult.skipped(
                    "native_rhofold_c1prime_tm_score",
                    "target C1' coordinates unavailable",
                    implementation="usalign_c1prime_tm_score_same_residue_v1",
                )
            native_reward_metrics = _reward_like_metrics(
                native_fold,
                target.reference_c4p_coords,
                cache_hit=cache_hit,
                prefix="native_",
            )

        candidates: list[dict[str, Any]] = []
        tertiary_limit = int(self.metrics_config.get("tertiary", {}).get("candidate_limit_per_target", len(sequences)))
        for candidate_index, (sample, sequence) in enumerate(zip(samples, sequences)):
            metrics: dict[str, dict[str, Any]] = {}
            try:
                recovery = sequence_recovery(sequence, target.sequence_native)
                metrics["sequence_recovery"] = MetricResult.ok(
                    "sequence_recovery",
                    recovery,
                    implementation="exact_position_match_v1",
                ).to_dict()
            except ValueError as exc:
                metrics["sequence_recovery"] = MetricResult.error("sequence_recovery", str(exc)).to_dict()

            if secondary_eligible:
                metrics["secondary_structure_f1"] = self._secondary_metric(
                    sequence,
                    target_secondary,
                    rnafold,
                ).to_dict()
            else:
                metrics["secondary_structure_f1"] = MetricResult.skipped(
                    "secondary_structure_f1",
                    "native_secondary_f1_below_threshold_or_unavailable",
                    details={
                        "native_f1": native_secondary_metric.value,
                        "required_minimum": native_f1_threshold,
                        "native_metric_status": native_secondary_metric.status.value,
                        "paper_filter": True,
                    },
                    implementation="ribodiffusion_native_f1_eligibility_v1",
                ).to_dict()
            metrics["rfam_family_success"] = self._rfam_metric(sequence, family, sample.sample_id).to_dict()

            if fold_oracle is not None and candidate_index < tertiary_limit:
                fold, cache_hit = fold_oracle.fold(sequence, target.target_id)
                if target_pdb is not None:
                    metrics["rhofold_c1prime_tm_score"] = _tertiary_metric(
                        fold,
                        target_pdb,
                        usalign,
                        cache_hit,
                        "rhofold_c1prime_tm_score",
                    ).to_dict()
                else:
                    metrics["rhofold_c1prime_tm_score"] = MetricResult.skipped(
                        "rhofold_c1prime_tm_score",
                        "target C1' coordinates unavailable",
                    ).to_dict()
                metrics.update(
                    {
                        name: result.to_dict()
                        for name, result in _reward_like_metrics(
                            fold,
                            target.reference_c4p_coords,
                            cache_hit=cache_hit,
                            prefix="",
                        ).items()
                    }
                )
            else:
                reason = (
                    f"candidate index {candidate_index} exceeds tertiary candidate limit {tertiary_limit}"
                    if fold_oracle is not None
                    else "tertiary metric disabled"
                )
                metrics["rhofold_c1prime_tm_score"] = MetricResult.skipped(
                    "rhofold_c1prime_tm_score",
                    reason,
                ).to_dict()
                metrics.update({name: result.to_dict() for name, result in _skipped_reward_like_metrics("", reason).items()})

            metrics["drfold_c1prime_tm_score"] = MetricResult.skipped(
                "drfold_c1prime_tm_score",
                str(self.metrics_config.get("drfold", {}).get("reason", "DRFold disabled by evaluation protocol")),
                implementation="not_run",
            ).to_dict()
            candidates.append(
                {
                    "candidate_id": sample.sample_id,
                    "target_id": target.target_id,
                    "sequence": sequence,
                    "seed": int(sample.seed),
                    "temperature": float(sample.temperature),
                    "latency_ms": float(sample.latency_ms),
                    "generator": str(getattr(sample, "generator", "ride")),
                    "metrics": metrics,
                }
            )

        diversity = intdiv(list(sequences))
        target_metrics = {
            "internal_diversity": MetricResult.ok(
                "internal_diversity",
                diversity,
                details={
                    "n_sequences": len(sequences),
                    "formula": "1 - (1/n^2) * sum_ordered_pairs(LCS(S1,S2)/min(len(S1),len(S2)))",
                    "includes_diagonal": True,
                },
                implementation="ribodiffusion_intdiv_lcs_v1",
            ).to_dict(),
            "native_secondary_structure_f1": native_secondary_metric.to_dict(),
            "native_rfam_family_success": native_rfam_metric.to_dict(),
            "native_rhofold_c1prime_tm_score": native_metric.to_dict(),
            **{name: result.to_dict() for name, result in native_reward_metrics.items()},
        }
        evaluation_metadata = (
            dict(target.raw_record.get("_evaluation", {}))
            if isinstance(target.raw_record, Mapping)
            else {}
        )
        return (
            {
                "target_id": target.target_id,
                "dataset_index": target.dataset_index,
                "source": target.source,
                "split": target.split,
                "length": target.length,
                "length_bucket": length_bucket(target.length),
                "rna_family": family,
                "rna_type": rna_type,
                "native_sequence": target.sequence_native,
                "target_secondary_structure": target_secondary,
                "target_secondary_source": "processed.pt/sec_struct_list (precomputed from source 3D structure; DSSR was not rerun)",
                "secondary_structure_eligible": secondary_eligible,
                "secondary_structure_native_f1_threshold": native_f1_threshold,
                "condition_imputed_residue_count": int(evaluation_metadata.get("imputed_residue_count", 0)),
                "strict_complete_condition": bool(evaluation_metadata.get("strict_complete", True)),
                "sample_count": len(sequences),
                "metrics": target_metrics,
            },
            candidates,
        )

    def _secondary_metric(
        self,
        sequence: str,
        target_secondary: Any,
        rnafold: RNAfoldAdapter,
        *,
        metric_name: str = "secondary_structure_f1",
    ) -> MetricResult:
        config = dict(self.metrics_config.get("secondary", {}))
        if not bool(config.get("enabled", True)):
            return MetricResult.skipped(metric_name, "secondary metric disabled")
        if not isinstance(target_secondary, str) or len(target_secondary) != len(sequence):
            return MetricResult.skipped(metric_name, "target secondary structure unavailable or length-mismatched")
        folded = rnafold.fold(sequence)
        if folded.status != MetricStatus.OK:
            return MetricResult(
                name=metric_name,
                status=folded.status,
                reason=folded.reason,
                details=folded.details,
                implementation="rnafold_vs_extended_dotbracket_base_pair_f1_v1",
            )
        try:
            predicted = str(folded.details["dot_bracket"])
            scores = base_pair_scores(
                parse_extended_dot_bracket(predicted),
                parse_extended_dot_bracket(target_secondary),
            )
        except (KeyError, ValueError) as exc:
            return MetricResult.error(
                metric_name,
                str(exc),
                implementation="rnafold_vs_extended_dotbracket_base_pair_f1_v1",
            )
        return MetricResult.ok(
            metric_name,
            scores["f1"],
            details={
                **scores,
                "predicted_dot_bracket": predicted,
                "target_dot_bracket": target_secondary,
                "target_source": "processed_precomputed_structure",
                "target_pair_scope": "all_extended_dot_bracket_pairs",
                "rnafold": dict(folded.details),
            },
            implementation="rnafold_vs_extended_dotbracket_base_pair_f1_v1",
        )

    def _rfam_metric(self, sequence: str, family: Any, candidate_id: str) -> MetricResult:
        config = dict(self.metrics_config.get("rfam", {}))
        if not bool(config.get("enabled", True)):
            return MetricResult.skipped("rfam_family_success", "Rfam metric disabled")
        family_id = str(family) if family not in (None, "", "unknown") else None
        database = config.get("cm_database")
        if not database and family_id and config.get("cm_directory"):
            family_models = dict(config.get("family_models", {}))
            model_name = str(family_models.get(family_id, f"{family_id}.cm"))
            database = str(Path(str(config["cm_directory"])) / model_name)
        adapter = InfernalCmsearchAdapter(
            self._path(config.get("cmsearch_binary", "cmsearch"), require_relative_exists=False),
            database_path=self._path(database, require_relative_exists=False) if database else None,
            family_accession=family_id,
            timeout_s=float(config.get("timeout_s", 120.0)),
        )
        result = adapter.search(sequence, sequence_id=candidate_id.replace(":", "_"))
        return MetricResult(
            name="rfam_family_success",
            status=result.status,
            value=result.value,
            reason=result.reason,
            details=result.details,
            implementation="infernal_cmsearch_cut_ga_family_match_v1",
        )

    def _manifest(
        self,
        *,
        ride_config: Any,
        rnafold_version: str | None,
        usalign_version: str | None,
        fold_oracle: EvaluationFoldOracle | None,
        elapsed_s: float,
    ) -> dict[str, Any]:
        manifest = build_run_manifest(
            self.repo_root,
            self.config,
            extra={
                "mode": "ribodiffusion_style_evaluation",
                "protocol_version": PROTOCOL_VERSION,
                "generator": str(self.config.get("sampling", {}).get("generator", "ride")),
            },
        )
        rfam_config = dict(self.metrics_config.get("rfam", {}))
        cmsearch = InfernalCmsearchAdapter(
            self._path(rfam_config.get("cmsearch_binary", "cmsearch"), require_relative_exists=False),
            timeout_s=float(rfam_config.get("timeout_s", 120.0)),
        )
        manifest["tools"] = {
            "RNAfold": rnafold_version,
            "USalign": usalign_version,
            "RhoFold+": {
                "enabled": fold_oracle is not None,
                "checkpoint_sha256": fold_oracle.checkpoint_hash if fold_oracle else None,
                "cache_hits": fold_oracle.hits if fold_oracle else 0,
                "cache_misses": fold_oracle.misses if fold_oracle else 0,
            },
            "Infernal/Rfam": {
                "version": cmsearch.probe_version(),
                "cm_database": rfam_config.get("cm_database"),
                "cm_directory": rfam_config.get("cm_directory"),
                "threshold": "model-specific GA (--cut_ga)",
            },
            "DRFold": "skipped by configured protocol",
        }
        pool_value = self.config["ride"].get("pool_manifest") or ride_config.data.pool_manifest
        evaluation_artifacts = {
            "pool_manifest": self._path(pool_value),
            "ride_checkpoint": self._path(ride_config.data.ride_checkpoint),
            "rhofold_checkpoint": fold_oracle.checkpoint if fold_oracle else None,
            "rnafold_binary": self._path(
                self.metrics_config.get("secondary", {}).get("rnafold_binary", "RNAfold"),
                require_relative_exists=False,
            ),
            "usalign_binary": self._path(
                self.metrics_config.get("tertiary", {}).get("usalign_binary", "USalign"),
                require_relative_exists=False,
            ),
        }
        manifest["artifact_hashes"] = {
            name: {
                "path": str(path),
                "exists": bool(path and path.is_file()),
                **({"sha256": sha256_file(path)} if path and path.is_file() else {}),
            }
            for name, path in evaluation_artifacts.items()
        }
        manifest["ride_checkpoint_sha256"] = manifest["artifact_hashes"]["ride_checkpoint"].get("sha256")
        manifest["elapsed_s"] = float(elapsed_s)
        manifest["protocol_deviations"] = [
            "Configured RIDE split is not automatically equivalent to RiboDiffusion's original clustered test splits.",
            "Target secondary structures come from processed.pt and were not regenerated with DSSR; the paper's native RNAfold F1 >= 0.7 eligibility filter is applied.",
            "DRFold is intentionally skipped.",
        ]
        return manifest

    def _resolved_tertiary_config(self, config: Mapping[str, Any]) -> dict[str, Any]:
        return {
            **dict(config),
            "python": str(self._path(config["rhofold_python"])),
            "worker_script": str(self._path(config["worker_script"])),
            "checkpoint": str(self._path(config["checkpoint"])),
            "cache_dir": str(self._path(config.get("cache_dir", "cache/evaluation/rhofold"), require_relative_exists=False)),
        }

    def _path(self, value: Any, *, require_relative_exists: bool = True) -> Path:
        path = Path(str(value))
        if path.is_absolute():
            return path
        if not require_relative_exists and path.parent == Path("."):
            executable = shutil.which(str(path))
            return Path(executable) if executable else path
        candidate = (self.repo_root / path).resolve()
        if require_relative_exists and not candidate.exists():
            raise FileNotFoundError(candidate)
        return candidate

    def _event(self, event: str, payload: Mapping[str, Any]) -> None:
        record = {"time": time.time(), "event": event, **payload}
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _tertiary_metric(
    fold: FoldResult,
    target_pdb: Path,
    usalign: USAlignC1Prime,
    cache_hit: bool,
    metric_name: str,
) -> MetricResult:
    if fold.status != "ok" or not fold.predicted_structure_path:
        status = MetricStatus.SKIPPED if fold.status in {"disabled", "unavailable"} else MetricStatus.ERROR
        return MetricResult(
            name=metric_name,
            status=status,
            reason=fold.error_message or f"RhoFold+ status={fold.status}",
            details={"fold_status": fold.status, "fold_cache_hit": cache_hit},
            implementation="rhofold_plus_single_sequence_then_usalign_c1prime_v1",
        )
    result = usalign.compare(fold.predicted_structure_path, target_pdb)
    return MetricResult(
        name=metric_name,
        status=result.status,
        value=result.value,
        reason=result.reason,
        details={
            **dict(result.details),
            "predicted_structure_path": fold.predicted_structure_path,
            "predicted_structure_hash": fold.predicted_structure_hash,
            "plddt": fold.plddt,
            "fold_latency_ms": fold.latency_ms,
            "fold_cache_hit": cache_hit,
        },
        implementation="rhofold_plus_single_sequence_then_usalign_c1prime_v1",
    )


def _reward_like_metrics(
    fold: FoldResult,
    target_c4p_coords: torch.Tensor | None,
    *,
    cache_hit: bool,
    prefix: str,
) -> dict[str, MetricResult]:
    names = _reward_like_metric_names(prefix)
    if target_c4p_coords is None:
        return _skipped_reward_like_metrics(prefix, "target C4' coordinates unavailable")
    if fold.status != "ok" or not fold.predicted_structure_path:
        reason = fold.error_message or f"RhoFold+ status={fold.status}"
        status = MetricStatus.SKIPPED if fold.status in {"disabled", "unavailable"} else MetricStatus.ERROR
        return {
            name: MetricResult(
                name=name,
                status=status,
                reason=reason,
                details={"fold_status": fold.status, "fold_cache_hit": cache_hit},
                implementation="ride_reward_c4p_kabsch_v1",
            )
            for name in names
        }
    try:
        predicted = load_c4p_coords(fold.predicted_structure_path)
        structural = compute_structural_metrics(predicted, target_c4p_coords, plddt=fold.plddt)
        composer = RewardComposer("initial_plan_strict", train_mode=False)
        values = (
            structural.rmsd,
            structural.tm_score,
            structural.gdt_ts,
            composer.raw_reward(structural),
            1.0 if composer.is_good(structural) else 0.0,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        return {
            name: MetricResult.error(name, str(exc), implementation="ride_reward_c4p_kabsch_v1")
            for name in names
        }
    details = {
        "predicted_structure_path": fold.predicted_structure_path,
        "fold_cache_hit": cache_hit,
        "alignment": "same_index_C4_prime_Kabsch",
        "reward_preset": "initial_plan_strict",
    }
    return {
        name: MetricResult.ok(name, value, details=details, implementation="ride_reward_c4p_kabsch_v1")
        for name, value in zip(names, values)
    }


def _reward_like_metric_names(prefix: str) -> tuple[str, ...]:
    return (
        f"{prefix}reward_c4p_rmsd",
        f"{prefix}reward_c4p_tm_score",
        f"{prefix}reward_c4p_gdt_ts",
        f"{prefix}reward_raw_score",
        f"{prefix}reward_good",
    )


def _skipped_reward_like_metrics(prefix: str, reason: str) -> dict[str, MetricResult]:
    return {
        name: MetricResult.skipped(name, reason, implementation="ride_reward_c4p_kabsch_v1")
        for name in _reward_like_metric_names(prefix)
    }


def _conformer_value(target: Any, key: str) -> Any:
    raw = getattr(target, "raw_record", None)
    if not isinstance(raw, Mapping):
        return None
    values = raw.get(key)
    index = getattr(target, "conformer_index", None)
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)) and isinstance(index, int) and 0 <= index < len(values):
        return values[index]
    return None


def summarize_results(
    targets: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    *,
    include_subsets: bool = True,
) -> dict[str, Any]:
    metric_names = (
        "sequence_recovery",
        "secondary_structure_f1",
        "rfam_family_success",
        "rhofold_c1prime_tm_score",
        "reward_c4p_rmsd",
        "reward_c4p_tm_score",
        "reward_c4p_gdt_ts",
        "reward_raw_score",
        "reward_good",
        "drfold_c1prime_tm_score",
    )
    summary: dict[str, Any] = {
        "target_count": len(targets),
        "candidate_count": len(candidates),
        "candidate_metrics": {},
        "paper_single_sample_candidate_metrics": {},
        "target_metrics": {},
    }
    first_candidate_by_target: dict[str, Mapping[str, Any]] = {}
    for candidate in candidates:
        first_candidate_by_target.setdefault(str(candidate["target_id"]), candidate)
    for name in metric_names:
        results = [candidate["metrics"][name] for candidate in candidates if name in candidate.get("metrics", {})]
        summary["candidate_metrics"][name] = _aggregate_metric_results(results)
        paper_results = [
            candidate["metrics"][name]
            for candidate in first_candidate_by_target.values()
            if name in candidate.get("metrics", {})
        ]
        summary["paper_single_sample_candidate_metrics"][name] = _aggregate_metric_results(paper_results)
    for name in (
        "internal_diversity",
        "native_secondary_structure_f1",
        "native_rfam_family_success",
        "native_rhofold_c1prime_tm_score",
        "native_reward_c4p_rmsd",
        "native_reward_c4p_tm_score",
        "native_reward_c4p_gdt_ts",
        "native_reward_raw_score",
        "native_reward_good",
    ):
        results = [target["metrics"][name] for target in targets if name in target.get("metrics", {})]
        summary["target_metrics"][name] = _aggregate_metric_results(results)

    for bucket_key in ("length_bucket", "rna_type", "rna_family"):
        groups: dict[str, list[str]] = defaultdict(list)
        target_by_id = {target["target_id"]: target for target in targets}
        for candidate in candidates:
            label = target_by_id[candidate["target_id"]].get(bucket_key)
            if label not in (None, "", "unknown"):
                groups[str(label)].append(candidate["candidate_id"])
        candidate_by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
        summary[f"by_{bucket_key}"] = {
            label: {
                name: _aggregate_metric_results(
                    [candidate_by_id[cid]["metrics"][name] for cid in ids if name in candidate_by_id[cid].get("metrics", {})]
                )
                for name in (
                    "sequence_recovery",
                    "secondary_structure_f1",
                    "rhofold_c1prime_tm_score",
                    "reward_c4p_rmsd",
                    "reward_c4p_tm_score",
                    "reward_c4p_gdt_ts",
                )
            }
            for label, ids in sorted(groups.items())
        }

    target_by_id = {target["target_id"]: target for target in targets}
    summary["rfam_success_average_across_families"] = _rfam_family_average(candidates, target_by_id)
    summary["paper_single_sample_rfam_success_average_across_families"] = _rfam_family_average(
        list(first_candidate_by_target.values()),
        target_by_id,
    )
    if include_subsets:
        strict_targets = [target for target in targets if bool(target.get("strict_complete_condition", True))]
        strict_ids = {str(target["target_id"]) for target in strict_targets}
        strict_candidates = [candidate for candidate in candidates if str(candidate["target_id"]) in strict_ids]
        summary["condition_imputation"] = {
            "strict_complete_target_count": len(strict_targets),
            "imputed_target_count": len(targets) - len(strict_targets),
            "imputed_residue_count": sum(int(target.get("condition_imputed_residue_count", 0)) for target in targets),
        }
        if len(strict_targets) != len(targets):
            summary["strict_complete_subset"] = summarize_results(
                strict_targets,
                strict_candidates,
                include_subsets=False,
            )
    return summary


def _rfam_family_average(
    candidates: Sequence[Mapping[str, Any]],
    target_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    family_values: dict[str, list[float]] = defaultdict(list)
    for candidate in candidates:
        result = candidate["metrics"]["rfam_family_success"]
        family = target_by_id[candidate["target_id"]].get("rna_family")
        if result["status"] == "ok" and result.get("value") is not None and family:
            family_values[str(family)].append(float(result["value"]))
    family_means = {family: mean(values) for family, values in sorted(family_values.items())}
    return {
        "status": "ok" if family_means else "skipped",
        "value": mean(family_means.values()) if family_means else None,
        "family_means": family_means,
        "reason": None if family_means else "no successful Rfam evaluations",
    }


def _aggregate_metric_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = [float(result["value"]) for result in results if result.get("status") == "ok" and result.get("value") is not None]
    statuses = Counter(str(result.get("status", "unknown")) for result in results)
    reasons = Counter(str(result.get("reason")) for result in results if result.get("reason"))
    return {
        "count": len(results),
        "ok": len(values),
        "skipped": statuses.get("skipped", 0),
        "error": statuses.get("error", 0),
        "mean": mean(values) if values else None,
        "median": median(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "reason_counts": dict(sorted(reasons.items())),
    }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _read_single_fasta(path: Path) -> str:
    sequence = "".join(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith(">")
    ).upper()
    if not sequence or any(base not in {"A", "U", "G", "C"} for base in sequence):
        raise ValueError(f"Invalid RNA FASTA sequence in {path}")
    return sequence


def _natural_path_key(path: Path) -> tuple[Any, ...]:
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name))
