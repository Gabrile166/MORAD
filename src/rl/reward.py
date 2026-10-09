"""Reward composition, structural metrics, oracle cache, and JSONL client."""

from __future__ import annotations

import hashlib
import contextlib
import json
import math
import os
import queue
import subprocess
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from uuid import uuid4

import torch

from src.rl.config import canonical_preset
from src.rl.protocol import RewardBatch, RewardRecord, RolloutBatch, TargetRecord


OK_STATUS = "ok"
METRIC_VERSION = "kabsch_c4p_v2_aux"
DEFAULT_ALIGNMENT_CONFIG = "c4p_kabsch_centered"
RNA_ALPHABET = frozenset("AUGC")


@dataclass(frozen=True)
class StructuralMetrics:
    rmsd: float
    tm_score: float
    gdt_ts: float
    plddt: float | None = None
    # geometric_v3 auxiliary terms. Optional: a missing term is dropped and the
    # weights renormalised (see RewardComposer._composite), which is different
    # from scoring it zero -- absent data is not evidence of a bad design.
    sc_mcc: float | None = None
    ensemble_defect: float | None = None
    composition_kl: float | None = None

    def validate(self) -> "StructuralMetrics":
        values = (self.rmsd, self.tm_score, self.gdt_ts)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("metrics must be finite")
        if self.rmsd < 0:
            raise ValueError("rmsd must be non-negative")
        return self


@dataclass(frozen=True)
class FoldResult:
    status: str
    predicted_structure_path: str | None
    predicted_structure_hash: str | None
    plddt: float | None = None
    latency_ms: float = 0.0
    error_message: str | None = None


@dataclass(frozen=True)
class OracleRequest:
    request_id: str
    sequence: str
    output_dir: str
    target_id: str | None = None


@dataclass(frozen=True)
class ScoredBatch:
    rollout: RolloutBatch
    rewards: RewardBatch
    target: TargetRecord
    oracle_calls: int
    cache_hits: int = 0
    cache_misses: int = 0
    metric_cache_hits: int = 0
    metric_cache_misses: int = 0
    unique_sequences: int = 0

    @property
    def cache_summary(self) -> dict[str, object]:
        rewards = [record.raw_reward for record in self.rewards.records if record.raw_reward is not None and math.isfinite(float(record.raw_reward))]
        rmsd = [record.rmsd for record in self.rewards.records if record.rmsd is not None and math.isfinite(float(record.rmsd))]
        tm = [record.tm_score for record in self.rewards.records if record.tm_score is not None and math.isfinite(float(record.tm_score))]
        gdt = [record.gdt_ts for record in self.rewards.records if record.gdt_ts is not None and math.isfinite(float(record.gdt_ts))]
        invalid = len(self.rewards.records) - len(rewards)
        return {
            "fold_cache_hits": self.cache_hits,
            "fold_cache_misses": self.cache_misses,
            "metric_cache_hits": self.metric_cache_hits,
            "metric_cache_misses": self.metric_cache_misses,
            "unique_sequences": self.unique_sequences,
            "invalid_records": invalid,
            "reward_mean": _mean(rewards),
            "reward_std": _std(rewards),
            "rmsd_mean": _mean(rmsd),
            "tm_score_mean": _mean(tm),
            "gdt_ts_mean": _mean(gdt),
        }


class RewardComposer:
    """Compose scalar reward and Good flag from structural metrics."""

    # Weights of the balanced_cosine_v1 preset. GDT-TS and TM-score carry most
    # of the signal because they are the headline structural metrics reported at
    # evaluation time; RMSD is a smaller corrective term. Keeping the reward
    # aligned with the reported metrics is what prevents "reward up, metric
    # down" behaviour.
    _COSINE_WEIGHTS = {"gdt_ts": 0.45, "tm_score": 0.35, "rmsd": 0.20}

    # Per-metric (worst, best) anchors. Values outside the range saturate, so a
    # single freak sample cannot dominate a group.
    _COSINE_RANGES = {
        "gdt_ts": (0.15, 0.70),
        "tm_score": (0.20, 0.70),
        # RMSD is inverted: smaller is better, so lo is the *worst* value.
        "rmsd": (12.0, 1.5),
    }

    # `is_good` must agree with what the reward optimises, otherwise samples with
    # high reward could be flagged False. The gate sits slightly above each
    # metric's midpoint so "good" means clearly better than average, not merely
    # median.
    _GOOD_GATES = {"gdt_ts": 0.50, "tm_score": 0.50, "rmsd": 4.0}

    # ---- geometric_v3 ----------------------------------------------------
    # Weights solved by scripts/reward/reward_design_v3.py against 1224 real
    # candidates (8 per target, RL-before distribution). The programme maximises
    # robustness subject to floors on alignment (rank correlation with the
    # graded objective) and within-group discrimination -- the latter is what
    # GRPO's gradient actually consumes, since advantages are standardised
    # inside a group. Solution: alignment 0.9608 (floor 0.70), discrimination
    # 1.2061 (floor 1.20, the single binding constraint).
    #
    # 85% of the mass sits on the three oracle terms and only 5% on each cheap
    # term. That split is the hackability penalty doing its job: sc_mcc,
    # ensemble defect and composition KL are all pushable without improving
    # real structural quality, which is exactly the failure RiboPO reported.
    _V3_WEIGHTS = {
        "gdt_ts": 0.25,
        "tm_score": 0.30,
        "rmsd": 0.30,
        "sc_mcc": 0.05,
        "ensemble_defect": 0.05,
        "composition_kl": 0.05,
    }

    # (lo, hi) ramp anchors; hi < lo means lower-is-better. Identical to
    # default_components() in the solver -- if these drift apart the trained
    # reward stops matching the objective the weights were solved for.
    _V3_RANGES = {
        "gdt_ts": (0.20, 0.75),
        "tm_score": (0.20, 0.70),
        "rmsd": (12.0, 2.0),
        "sc_mcc": (0.30, 0.85),
        "ensemble_defect": (0.60, 0.15),
        "composition_kl": (0.50, 0.02),
    }

    # Matches DESIRABILITY_FLOOR in the solver. A geometric mean would collapse
    # to zero on a single zero term, so every factor is floored.
    _V3_FLOOR = 1e-3

    # threshold_position=0.60 in the solver: the good/bad line sits just above
    # the ramp midpoint, on the steepest part of the cosine rather than the
    # plateau. is_good must agree with what the reward maximises, otherwise
    # high-reward samples get flagged False.
    _V3_THRESHOLD_POSITION = 0.60

    def __init__(self, preset: str = "balanced_cosine_v1", train_mode: bool = True) -> None:
        preset = canonical_preset(preset)
        allowed = {"balanced_cosine_v1", "geometric_v3", "initial_plan_strict", "current_code_compat"}
        if preset not in allowed:
            raise ValueError(f"unknown reward preset: {preset}")
        if train_mode and preset == "current_code_compat":
            raise ValueError("current_code_compat is forbidden in formal train mode")
        self.preset = preset
        self.train_mode = train_mode

    @staticmethod
    def _cosine_ramp(value: float, lo: float, hi: float) -> float:
        """Map `value` onto [0, 1] with a raised-cosine ramp.

        The curve is flat near both anchors and steepest at the midpoint, so the
        largest gradient sits exactly at the boundary between mediocre and good
        samples -- which is where we want the policy to be pushed. Monotonic,
        bounded, and free of the cliffs that made the original reward jump by
        ~100 when a threshold was crossed.

        Works for both directions: pass hi < lo for metrics where lower is
        better (e.g. RMSD).
        """
        import math

        if hi == lo:
            return 0.0
        t = (value - lo) / (hi - lo)
        t = min(1.0, max(0.0, t))
        return 0.5 * (1.0 - math.cos(math.pi * t))

    @classmethod
    def _v3_desirability(cls, name: str, value: float | None) -> float | None:
        """Raised-cosine ramp to [floor, 1]; None propagates as None.

        Mirrors Component.desirability in the solver exactly, including the
        floor, so training and the offline weight solve share one definition.
        """
        import math

        if value is None:
            return None
        value = float(value)
        if not math.isfinite(value):
            return None
        lo, hi = cls._V3_RANGES[name]
        span = hi - lo
        if abs(span) < 1e-12:
            return 1.0 if value >= hi else cls._V3_FLOOR
        t = (value - lo) / span
        t = min(1.0, max(0.0, t))
        return max(cls._V3_FLOOR, 0.5 * (1.0 - math.cos(math.pi * t)))

    @classmethod
    def _v3_composite(cls, metrics: StructuralMetrics) -> float:
        """Weighted geometric mean over the terms that are present.

        Missing terms are dropped and the remaining weights renormalised, which
        is what the solver's composite() does. Imputing zero instead would
        assert that an unmeasured design is a bad one.
        """
        import math

        log_sum = 0.0
        weight_sum = 0.0
        for name, weight in cls._V3_WEIGHTS.items():
            d = cls._v3_desirability(name, getattr(metrics, name, None))
            if d is None or weight <= 0.0:
                continue
            log_sum += weight * math.log(max(cls._V3_FLOOR, d))
            weight_sum += weight
        if weight_sum <= 0.0:
            return 0.0
        return math.exp(log_sum / weight_sum)

    @classmethod
    def _v3_threshold(cls, name: str) -> float:
        lo, hi = cls._V3_RANGES[name]
        return lo + cls._V3_THRESHOLD_POSITION * (hi - lo)

    def raw_reward(self, metrics: StructuralMetrics) -> float:
        metrics.validate()
        if self.preset == "geometric_v3":
            return self._v3_composite(metrics)
        if self.preset != "balanced_cosine_v1":
            # Legacy presets kept verbatim so old runs stay reproducible.
            base = -(0.5 * metrics.rmsd) ** 2 + (5.0 * metrics.gdt_ts) ** 2
            if self.preset == "initial_plan_strict":
                return base + 100.0 * max(0.0, metrics.gdt_ts - 0.50) + 20.0 * max(0.0, 2.0 - metrics.rmsd)
            if metrics.gdt_ts > 0.45:
                return base + (metrics.gdt_ts - 0.45) * 100.0
            if metrics.rmsd < 3.0:
                return base + (3.0 - metrics.rmsd) * 20.0
            return base

        total = 0.0
        for name in ("gdt_ts", "tm_score", "rmsd"):
            lo, hi = self._COSINE_RANGES[name]
            total += self._COSINE_WEIGHTS[name] * self._cosine_ramp(
                float(getattr(metrics, name)), lo, hi
            )
        # Weights sum to 1 and every ramp is in [0, 1], so reward is in [0, 1].
        return total

    def is_good(self, metrics: StructuralMetrics) -> bool:
        metrics.validate()
        if self.preset == "geometric_v3":
            # Gate on the three primary terms only. The auxiliary terms carry
            # 5% weight each precisely because they are gameable, so letting
            # them decide "good" would reintroduce the hack they were penalised
            # for. Thresholds come from the same ramps the reward uses, so a
            # high-reward sample cannot be flagged False.
            return (
                metrics.gdt_ts >= self._v3_threshold("gdt_ts")
                and metrics.tm_score >= self._v3_threshold("tm_score")
                and metrics.rmsd <= self._v3_threshold("rmsd")
            )
        if self.preset != "balanced_cosine_v1":
            hits = (
                metrics.gdt_ts >= 0.50,
                metrics.tm_score >= 0.45,
                metrics.rmsd <= 2.0,
            )
            return sum(bool(item) for item in hits) >= 2
        # Aligned with the reward's primary terms: the two structural-similarity
        # metrics must both clear their gate, with RMSD as a sanity bound whose
        # threshold is attainable (the old 2.0 A gate was essentially never met,
        # which decoupled is_good from the reward entirely).
        return (
            metrics.gdt_ts >= self._GOOD_GATES["gdt_ts"]
            and metrics.tm_score >= self._GOOD_GATES["tm_score"]
            and metrics.rmsd <= self._GOOD_GATES["rmsd"]
        )

    def record(
        self,
        metrics: StructuralMetrics,
        oracle_key: str,
        oracle_latency_ms: float = 0.0,
        oracle_cache_hit: bool = False,
    ) -> RewardRecord:
        return RewardRecord(
            status=OK_STATUS,
            rmsd=metrics.rmsd,
            tm_score=metrics.tm_score,
            gdt_ts=metrics.gdt_ts,
            raw_reward=self.raw_reward(metrics),
            is_good=self.is_good(metrics),
            oracle_cache_hit=oracle_cache_hit,
            oracle_latency_ms=oracle_latency_ms,
            oracle_key=oracle_key,
            plddt=metrics.plddt,
            sc_mcc=metrics.sc_mcc,
            ensemble_defect=metrics.ensemble_defect,
            composition_kl=metrics.composition_kl,
        ).validate()


class JsonDiskCache:
    """Small content-addressed JSON cache used for fold and metric records."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> dict[str, object] | None:
        path = self._path(key)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def set(self, key: str, value: Mapping[str, object]) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(dict(value), handle, sort_keys=True)
            handle.write("\n")
        tmp.replace(path)

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"


def fold_cache_key(sequence: str, oracle_checkpoint_hash: str, oracle_mode_and_config_hash: str) -> str:
    return stable_hash(
        {
            "kind": "fold",
            "sequence": normalize_sequence(sequence),
            "oracle_checkpoint_hash": oracle_checkpoint_hash,
            "oracle_mode_and_config_hash": oracle_mode_and_config_hash,
        }
    )


def metric_cache_key(
    target_structure_hash: str,
    predicted_structure_hash: str,
    metric_version: str = METRIC_VERSION,
    alignment_config: str = DEFAULT_ALIGNMENT_CONFIG,
) -> str:
    return stable_hash(
        {
            "kind": "metric",
            "target_structure_hash": target_structure_hash,
            "predicted_structure_hash": predicted_structure_hash,
            "metric_version": metric_version,
            "alignment_config": alignment_config,
        }
    )


def stable_hash(payload: Mapping[str, object] | Sequence[object] | str) -> str:
    if isinstance(payload, str):
        data = payload.encode("utf-8")
    else:
        data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_sequence(sequence: str) -> str:
    normalized = sequence.upper().replace("T", "U")
    if not normalized or any(base not in RNA_ALPHABET for base in normalized):
        raise ValueError("sequence must contain only A/U/G/C bases")
    return normalized


def kabsch_align(mobile: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    mobile = _as_coords(mobile, "mobile")
    reference = _as_coords(reference, "reference")
    if mobile.shape != reference.shape:
        raise ValueError("mobile and reference coordinates must have the same shape")
    mobile_centered = mobile - mobile.mean(dim=0, keepdim=True)
    reference_centered = reference - reference.mean(dim=0, keepdim=True)
    cov = mobile_centered.transpose(0, 1) @ reference_centered
    u, _, vh = torch.linalg.svd(cov)
    correction = torch.eye(3, dtype=mobile.dtype, device=mobile.device)
    correction[-1, -1] = torch.sign(torch.det(u @ vh))
    rotation = u @ correction @ vh
    return mobile_centered @ rotation


# ---------------------------------------------------------------------------
# geometric_v3 auxiliary components (online / training-time computation)
#
# These three terms are secondary-structure and composition based, i.e. they do
# NOT need the 3D oracle. They are computed from the designed sequence and the
# native sequence alone, via ViennaRNA's partition function. Cost measured on
# this box: RNAfold ~10 ms/seq + compute_thermo ~15 ms/seq, so ~0.3-0.5 s for a
# 32-sample step against a 5.9 s/step baseline (~8%).
#
# IMPORTANT: ViennaRNA needs LD_PRELOAD of the env's libstdc++ and a child
# process loses it, so everything here stays in-process (no ProcessPool).
# ---------------------------------------------------------------------------

_AUX_ALPHABET = ("A", "C", "G", "U")
_AUX_KL_EPS = 1e-3


def _nt_composition(sequence: str) -> list[float] | None:
    """ACGU frequency vector; None if the sequence has no canonical bases."""
    seq = normalize_sequence(sequence)
    if not seq:
        return None
    counts = [float(seq.count(base)) for base in _AUX_ALPHABET]
    total = sum(counts)
    if total <= 0.0:
        return None
    return [count / total for count in counts]


def composition_kl(design: str, native: str) -> float | None:
    """KL(design ‖ native) over the ACGU composition distribution.

    This is the definition the whole project standardises on: composition_kl was
    referenced across the codebase but never implemented. Nats, not bits.
    Smoothed by _AUX_KL_EPS so a base absent from the native sequence gives a
    large-but-finite penalty rather than +inf.
    """
    p_design = _nt_composition(design)
    q_native = _nt_composition(native)
    if p_design is None or q_native is None:
        return None
    total = 0.0
    for p_i, q_i in zip(p_design, q_native):
        if p_i <= 0.0:
            continue
        total += p_i * math.log(p_i / max(q_i, _AUX_KL_EPS))
    return float(max(0.0, total))


def _aux_sstt_module():
    """Import scripts.eval.sstt_metrics lazily; None when unavailable."""
    try:
        from scripts.eval import sstt_metrics  # type: ignore

        return sstt_metrics
    except Exception:
        try:
            import sys

            root = str(Path(__file__).resolve().parents[2])
            if root not in sys.path:
                sys.path.insert(0, root)
            from scripts.eval import sstt_metrics  # type: ignore

            return sstt_metrics
        except Exception:
            return None


def compute_auxiliary_metrics(design_sequence: str, native_sequence: str) -> dict[str, float | None]:
    """sc_mcc / ensemble_defect / composition_kl for one designed sequence.

    Any term that cannot be computed comes back as None; _v3_composite drops
    None terms and renormalises the remaining weights, which is the same
    convention the offline solver uses. Never raises: a broken auxiliary term
    must not kill a training step.
    """
    out: dict[str, float | None] = {"sc_mcc": None, "ensemble_defect": None, "composition_kl": None}
    out["composition_kl"] = composition_kl(design_sequence, native_sequence)

    sstt = _aux_sstt_module()
    if sstt is None:
        return out
    design = normalize_sequence(design_sequence)
    native = normalize_sequence(native_sequence)
    if not design or not native:
        return out
    try:
        # Self-consistency: fold the design, fold the native, compare the two
        # secondary structures as base-pair sets. Both are MFE folds from the
        # same Turner parameters, so the comparison is apples-to-apples.
        folded = sstt.fold_rnafold([design, native])
        design_db, native_db = folded[0], folded[1]
        if design_db and native_db and len(design_db) == len(native_db):
            out["sc_mcc"] = sstt.secondary_agreement(design_db, native_db).get("mcc")
        if native_db:
            # ensemble_defect_per_nt is length-normalised, which is what the
            # (0.60, 0.15) ramp in _V3_RANGES was calibrated on. It is silently
            # None unless target_structure= is passed.
            thermo = sstt.compute_thermo(design, target_structure=native_db)
            value = getattr(thermo, "ensemble_defect_per_nt", None)
            if value is not None and math.isfinite(float(value)):
                out["ensemble_defect"] = float(value)
    except Exception:
        return out
    return out


def compute_structural_metrics(
    predicted_coords: torch.Tensor,
    target_coords: torch.Tensor,
    plddt: float | None = None,
    aux: Mapping[str, float | None] | None = None,
) -> StructuralMetrics:
    target = _as_coords(target_coords, "target_coords")
    aligned = kabsch_align(predicted_coords, target)
    reference = target - target.mean(dim=0, keepdim=True)
    distances = torch.linalg.norm(aligned - reference, dim=-1)
    rmsd = torch.sqrt(torch.mean(distances.square())).item()
    gdt = _gdt_ts(distances)
    tm_score = _tm_score(distances, int(reference.shape[0]))
    extra = dict(aux or {})
    return StructuralMetrics(
        rmsd=float(rmsd),
        tm_score=float(tm_score),
        gdt_ts=float(gdt),
        plddt=plddt,
        sc_mcc=extra.get("sc_mcc"),
        ensemble_defect=extra.get("ensemble_defect"),
        composition_kl=extra.get("composition_kl"),
    ).validate()


def score_rollout_batch(
    rollout: RolloutBatch,
    target: TargetRecord,
    oracle: "RhoFoldJsonlClient | MetricsOracleProtocol",
    composer: RewardComposer,
    fold_cache: JsonDiskCache | None = None,
    metric_cache: JsonDiskCache | None = None,
    oracle_checkpoint_hash: str = "unknown",
    oracle_mode_and_config_hash: str = "single_seq_no_relax",
    metric_version: str = METRIC_VERSION,
    alignment_config: str = DEFAULT_ALIGNMENT_CONFIG,
) -> RewardBatch:
    """Score unique sequences once, then expand metrics to logical samples."""

    rollout.validate()
    target.validate()
    first_by_sequence: dict[str, int] = {}
    for idx, sample in enumerate(rollout.samples):
        sequence = normalize_sequence(sample.tokens)
        first_by_sequence.setdefault(sequence, idx)

    records_by_sequence: dict[str, RewardRecord] = {}
    for sequence in first_by_sequence:
        key = fold_cache_key(sequence, oracle_checkpoint_hash, oracle_mode_and_config_hash)
        cached_fold = fold_cache.get(key) if fold_cache is not None else None
        fold_cache_hit = cached_fold is not None
        if cached_fold is not None:
            fold = _fold_from_cache(cached_fold)
        else:
            fold_dir = Path(getattr(oracle, "output_root", tempfile.gettempdir())) / target.target_id / key
            if hasattr(oracle, "fold"):
                fold = oracle.fold(sequence, fold_dir, target_id=target.target_id)  # type: ignore[attr-defined]
            else:
                # Backward-compatible unit-test protocol: legacy fake oracles may only
                # expose score_sequence. Do not write target-specific records into the
                # production fold cache from this path.
                record = oracle.score_sequence(sequence, target, composer, key, metric_cache)
                records_by_sequence[sequence] = record
                continue
            if fold_cache is not None:
                fold_cache.set(key, _fold_to_cache(fold))
        record = _compose_record_from_fold(
            fold=fold,
            target=target,
            composer=composer,
            oracle_key=key,
            oracle_cache_hit=fold_cache_hit,
            metric_cache=metric_cache,
            metric_version=metric_version,
            alignment_config=alignment_config,
            design_sequence=sequence,
        )
        records_by_sequence[sequence] = record

    records = tuple(records_by_sequence[normalize_sequence(sample.tokens)] for sample in rollout.samples)
    return RewardBatch(
        records=records,
        target_ids=tuple(sample.target_id for sample in rollout.samples),
        sample_ids=tuple(sample.sample_id for sample in rollout.samples),
    ).validate()


class MetricsOracleProtocol:
    def score_sequence(
        self,
        sequence: str,
        target: TargetRecord,
        composer: RewardComposer,
        oracle_key: str,
        metric_cache: JsonDiskCache | None = None,
    ) -> RewardRecord:
        raise NotImplementedError


class RhoFoldJsonlClient(MetricsOracleProtocol):
    """Persistent JSONL subprocess client for RhoFold+ worker requests."""

    def __init__(self, command: Sequence[str], timeout_s: float = 300.0, output_root: str | Path | None = None) -> None:
        self.command = list(command)
        self.timeout_s = timeout_s
        self.output_root = Path(output_root) if output_root is not None else Path(tempfile.mkdtemp(prefix="rhofold_jsonl_"))
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.process: subprocess.Popen[str] | None = None
        self._stdout_queue: queue.Queue[str | None] | None = None
        self._start_process()

    def _start_process(self) -> None:
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._stdout_queue = queue.Queue()
        assert self.process.stdout is not None
        threading.Thread(target=self._reader, args=(self.process.stdout, self._stdout_queue), daemon=True).start()

    @staticmethod
    def _reader(stream: object, out: "queue.Queue[str | None]") -> None:
        try:
            for line in stream:  # type: ignore[operator]
                out.put(str(line))
        finally:
            out.put(None)

    def _ensure_process(self) -> subprocess.Popen[str]:
        if self.process is None or self.process.poll() is not None:
            self._start_process()
        assert self.process is not None
        return self.process

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()

    def fold(self, sequence: str, output_dir: str | Path, target_id: str | None = None) -> FoldResult:
        process = self._ensure_process()
        if process.stdin is None or self._stdout_queue is None:
            raise RuntimeError("worker pipes are not available")
        request = OracleRequest(str(uuid4()), sequence.upper(), str(output_dir), target_id=target_id)
        process.stdin.write(json.dumps(asdict(request), sort_keys=True) + "\n")
        process.stdin.flush()
        started = time.perf_counter()
        while True:
            if process.poll() is not None:
                stderr = process.stderr.read() if process.stderr is not None else ""
                raise RuntimeError(f"RhoFold worker exited early: {stderr}")
            remaining = self.timeout_s - (time.perf_counter() - started)
            if remaining <= 0:
                stderr = ""
                if process.poll() is None:
                    process.kill()
                    with contextlib.suppress(Exception):
                        stderr = process.stderr.read() if process.stderr is not None else ""
                self.process = None
                return FoldResult(
                    "timeout",
                    None,
                    None,
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                    error_message=f"worker timeout after {self.timeout_s:.1f}s; stderr={stderr[-1000:]}",
                )
            try:
                line = self._stdout_queue.get(timeout=min(0.1, max(0.0, remaining)))
            except queue.Empty:
                continue
            if line is None:
                continue
            if line.strip():
                payload = json.loads(line)
                if payload.get("request_id") != request.request_id:
                    raise RuntimeError("worker returned mismatched request_id")
                return FoldResult(
                    status=str(payload["status"]),
                    predicted_structure_path=payload.get("predicted_structure_path"),
                    predicted_structure_hash=payload.get("predicted_structure_hash"),
                    plddt=payload.get("plddt"),
                    latency_ms=float(payload.get("latency_ms", 0.0)),
                    error_message=payload.get("error_message"),
                )

    def score_sequence(
        self,
        sequence: str,
        target: TargetRecord,
        composer: RewardComposer,
        oracle_key: str,
        metric_cache: JsonDiskCache | None = None,
    ) -> RewardRecord:
        fold_dir = self.output_root / target.target_id / oracle_key
        fold = self.fold(sequence, fold_dir, target_id=target.target_id)
        if fold.status != OK_STATUS:
            return RewardRecord(
                status=fold.status,
                rmsd=None,
                tm_score=None,
                gdt_ts=None,
                raw_reward=None,
                is_good=False,
                oracle_cache_hit=False,
                oracle_latency_ms=fold.latency_ms,
                oracle_key=oracle_key,
                error_message=fold.error_message,
                plddt=fold.plddt,
            ).validate()
        if fold.predicted_structure_path is None or fold.predicted_structure_hash is None:
            return RewardRecord(
                status="fold_failed",
                rmsd=None,
                tm_score=None,
                gdt_ts=None,
                raw_reward=None,
                is_good=False,
                oracle_cache_hit=False,
                oracle_latency_ms=fold.latency_ms,
                oracle_key=oracle_key,
                error_message="worker did not return a predicted structure",
            ).validate()

        m_key = metric_cache_key(target.structure_hash, fold.predicted_structure_hash)
        cached = metric_cache.get(m_key) if metric_cache is not None else None
        if cached is not None:
            return _record_from_cache(cached, composer, oracle_key)
        try:
            predicted = load_c4p_coords(fold.predicted_structure_path)
            reference = target.reference_c4p_coords if target.reference_c4p_coords is not None else load_c4p_coords(target.structure_ref)
            metrics = compute_structural_metrics(predicted, reference, plddt=fold.plddt)
            if composer.preset == "geometric_v3":
                # Only the v3 preset consumes these, and each costs an RNAfold
                # call (~25 ms/sequence), so other presets must not pay for them.
                extra = compute_sequence_terms(sequence, target.sequence_native)
                metrics = replace(
                    metrics,
                    sc_mcc=extra["sc_mcc"],
                    ensemble_defect=extra["ensemble_defect"],
                    composition_kl=extra["composition_kl"],
                )
        except Exception as exc:
            return RewardRecord(
                status="metric_failed",
                rmsd=None,
                tm_score=None,
                gdt_ts=None,
                raw_reward=None,
                is_good=False,
                oracle_cache_hit=False,
                oracle_latency_ms=fold.latency_ms,
                oracle_key=oracle_key,
                error_message=str(exc),
                plddt=fold.plddt,
            ).validate()
        if metric_cache is not None:
            metric_cache.set(m_key, _record_to_cache(composer.record(metrics, oracle_key=oracle_key, oracle_latency_ms=fold.latency_ms)))
        return composer.record(metrics, oracle_key=oracle_key, oracle_latency_ms=fold.latency_ms)


# --------------------------------------------------------------------------
# geometric_v3 auxiliary terms (sequence-level, no structure needed)
# --------------------------------------------------------------------------
# ViennaRNA is imported lazily and cached: it is only needed by the v3 preset,
# and importing it eagerly would make every other preset pay for a dependency
# that may not even be installed.
_RNA_MODULE: object | None = None
_RNA_IMPORT_FAILED = False


def _rna_module():
    global _RNA_MODULE, _RNA_IMPORT_FAILED
    if _RNA_MODULE is not None or _RNA_IMPORT_FAILED:
        return _RNA_MODULE
    try:
        import RNA  # type: ignore

        _RNA_MODULE = RNA
    except Exception:
        # Missing/broken ViennaRNA must not kill training. The terms come back
        # None and composite() renormalises over the primary metrics, which is
        # a graceful degradation rather than a crash mid-run.
        _RNA_IMPORT_FAILED = True
    return _RNA_MODULE


def _pairs_from_db(structure: str) -> set[tuple[int, int]]:
    """Base pairs from dot-bracket, as (i, j) with i < j."""
    stack: list[int] = []
    pairs: set[tuple[int, int]] = set()
    for index, char in enumerate(structure):
        if char == "(":
            stack.append(index)
        elif char == ")" and stack:
            pairs.add((stack.pop(), index))
    return pairs


def _sc_mcc(design_db: str, native_db: str) -> float | None:
    """Matthews correlation over the base-pair sets of two structures.

    MCC rather than F1 because the pair matrix is overwhelmingly negative:
    F1 ignores true negatives and would flatter a structure that predicts
    almost nothing.
    """
    import math

    if not design_db or not native_db or len(design_db) != len(native_db):
        return None
    predicted = _pairs_from_db(design_db)
    reference = _pairs_from_db(native_db)
    length = len(native_db)
    possible = length * (length - 1) / 2
    tp = len(predicted & reference)
    fp = len(predicted - reference)
    fn = len(reference - predicted)
    tn = possible - tp - fp - fn
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator <= 0:
        # Degenerate case: one of the sets is empty. Perfect agreement on
        # "no pairs" is 1.0; disagreement is 0.0.
        return 1.0 if (not predicted and not reference) else 0.0
    return (tp * tn - fp * fn) / denominator


def _composition_kl(design: str, native: str) -> float | None:
    """KL(design ACGU composition || native ACGU composition).

    This term was referenced across the codebase but never implemented. Defined
    here as the nucleotide-frequency divergence: it catches designs that drift
    to a degenerate letter distribution (all-G being the classic cheat) while
    staying blind to ordering, which the structural terms already cover.
    """
    import math

    if not design or not native:
        return None
    eps = 1e-6
    total_d = max(len(design), 1)
    total_n = max(len(native), 1)
    divergence = 0.0
    for base in "ACGU":
        p = design.count(base) / total_d + eps
        q = native.count(base) / total_n + eps
        divergence += p * math.log(p / q)
    return max(0.0, divergence)


_NATIVE_FOLD_CACHE: dict[str, str] = {}


def _native_fold_cached(native: str) -> str:
    """MFE structure of a native sequence, memoised.

    Every candidate in a group shares the same native target, and targets recur
    across epochs, so without this the identical fold is recomputed on the order
    of hundreds of times. The cache is keyed by sequence, so it stays correct
    even when the target order is shuffled.
    """
    hit = _NATIVE_FOLD_CACHE.get(native)
    if hit is not None:
        return hit
    rna = _rna_module()
    structure, _ = rna.fold(native)
    _NATIVE_FOLD_CACHE[native] = structure
    return structure


def compute_sequence_terms(design: str, native: str) -> dict[str, float | None]:
    """sc_mcc / ensemble_defect / composition_kl for one designed sequence.

    Any term that cannot be computed comes back None rather than a sentinel
    number, so the composite drops it and renormalises instead of treating an
    unmeasured design as a bad one.
    """
    terms: dict[str, float | None] = {
        "sc_mcc": None,
        "ensemble_defect": None,
        "composition_kl": _composition_kl(design, native),
    }
    rna = _rna_module()
    if rna is None:
        return terms
    try:
        design_db, _ = rna.fold(design)
        native_db = _native_fold_cached(native)
        terms["sc_mcc"] = _sc_mcc(design_db, native_db)
        if len(design) == len(native_db):
            fold_compound = rna.fold_compound(design)
            fold_compound.pf()
            # Ensemble defect against the NATIVE target structure.
            # ViennaRNA's ensemble_defect() ALREADY returns the per-nucleotide
            # value -- dividing by length again would square the normalisation
            # and shrink the term by ~2 orders of magnitude, silently pushing it
            # to the desirability ceiling for every candidate. Verified against
            # the manual bpp-matrix computation in scripts/eval/sstt_metrics.py.
            terms["ensemble_defect"] = fold_compound.ensemble_defect(native_db)
    except Exception:
        # Partial results are kept: a failure in the thermodynamic step should
        # not discard a perfectly good sc_mcc.
        pass
    return terms


def _as_coords(value: torch.Tensor, name: str) -> torch.Tensor:
    tensor = value.to(dtype=torch.float64)
    if tensor.ndim != 2 or tensor.shape[1] != 3:
        raise ValueError(f"{name} must have shape [L, 3]")
    if tensor.shape[0] < 2:
        raise ValueError(f"{name} must contain at least two coordinates")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must be finite")
    return tensor


def load_c4p_coords(path: str | Path) -> torch.Tensor:
    coords: list[tuple[float, float, float]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            atom_name = line[12:16].strip()
            if atom_name not in {"C4'", "C4*"}:
                continue
            coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    if not coords:
        raise ValueError(f"no C4' atoms found in PDB: {path}")
    return torch.tensor(coords, dtype=torch.float64)


def _gdt_ts(distances: torch.Tensor) -> float:
    thresholds = torch.tensor([1.0, 2.0, 4.0, 8.0], dtype=distances.dtype, device=distances.device)
    return float((distances.unsqueeze(0) < thresholds.unsqueeze(1)).to(torch.float64).mean().item())


def _tm_score(distances: torch.Tensor, length: int) -> float:
    if length <= 18:
        return 0.0
    d0 = 1.24 * ((length - 15) ** (1.0 / 3.0)) - 1.8
    out = torch.mean(1.0 / (1.0 + (distances / d0).square()))
    return 0.0 if torch.isnan(out) else float(out.item())


def _record_from_cache(payload: Mapping[str, object], composer: RewardComposer, oracle_key: str) -> RewardRecord:
    if payload.get("status") != OK_STATUS:
        return RewardRecord(
            status=str(payload["status"]),
            rmsd=None,
            tm_score=None,
            gdt_ts=None,
            raw_reward=None,
            is_good=False,
            oracle_cache_hit=True,
            oracle_latency_ms=0.0,
            oracle_key=oracle_key,
            error_message=payload.get("error_message") if isinstance(payload.get("error_message"), str) else None,
        ).validate()
    def _opt(key: str) -> float | None:
        raw = payload.get(key)
        return float(raw) if raw is not None else None

    metrics = StructuralMetrics(
        rmsd=float(payload["rmsd"]),
        tm_score=float(payload["tm_score"]),
        gdt_ts=float(payload["gdt_ts"]),
        plddt=_opt("plddt"),
        sc_mcc=_opt("sc_mcc"),
        ensemble_defect=_opt("ensemble_defect"),
        composition_kl=_opt("composition_kl"),
    )
    return composer.record(metrics, oracle_key=oracle_key, oracle_latency_ms=0.0, oracle_cache_hit=True)


def _record_to_cache(record: RewardRecord) -> dict[str, object]:
    return {
        "status": record.status,
        "rmsd": record.rmsd,
        "tm_score": record.tm_score,
        "gdt_ts": record.gdt_ts,
        "plddt": record.plddt,
        "error_message": record.error_message,
        # geometric_v3 terms must survive the cache, otherwise a cache hit
        # silently drops them and the same candidate scores differently
        # depending on whether it was seen before.
        "sc_mcc": record.sc_mcc,
        "ensemble_defect": record.ensemble_defect,
        "composition_kl": record.composition_kl,
    }


def _fold_from_cache(payload: Mapping[str, object]) -> FoldResult:
    return FoldResult(
        status=str(payload.get("status", "fold_failed")),
        predicted_structure_path=str(payload["predicted_structure_path"]) if payload.get("predicted_structure_path") is not None else None,
        predicted_structure_hash=str(payload["predicted_structure_hash"]) if payload.get("predicted_structure_hash") is not None else None,
        plddt=float(payload["plddt"]) if payload.get("plddt") is not None else None,
        latency_ms=float(payload.get("latency_ms", 0.0)),
        error_message=str(payload["error_message"]) if payload.get("error_message") is not None else None,
    )


def _fold_to_cache(fold: FoldResult) -> dict[str, object]:
    return {
        "status": fold.status,
        "predicted_structure_path": fold.predicted_structure_path,
        "predicted_structure_hash": fold.predicted_structure_hash,
        "plddt": fold.plddt,
        "latency_ms": fold.latency_ms,
        "error_message": fold.error_message,
    }


def _metric_to_cache(metrics: StructuralMetrics) -> dict[str, object]:
    # The auxiliary terms must be persisted too. _record_from_cache() reads
    # them back, so omitting them here made a cache hit produce a different
    # reward than a cache miss for the very same design.
    return {
        "status": OK_STATUS,
        "rmsd": metrics.rmsd,
        "tm_score": metrics.tm_score,
        "gdt_ts": metrics.gdt_ts,
        "plddt": metrics.plddt,
        "sc_mcc": metrics.sc_mcc,
        "ensemble_defect": metrics.ensemble_defect,
        "composition_kl": metrics.composition_kl,
        "metric_version": METRIC_VERSION,
    }


def _metric_failed_record(fold: FoldResult, oracle_key: str, message: str, *, oracle_cache_hit: bool = False) -> RewardRecord:
    return RewardRecord(
        status="metric_failed",
        rmsd=None,
        tm_score=None,
        gdt_ts=None,
        raw_reward=None,
        is_good=False,
        oracle_cache_hit=oracle_cache_hit,
        oracle_latency_ms=fold.latency_ms,
        oracle_key=oracle_key,
        error_message=message,
        plddt=fold.plddt,
    ).validate()


def _fold_failed_record(fold: FoldResult, oracle_key: str, *, oracle_cache_hit: bool = False) -> RewardRecord:
    return RewardRecord(
        status=fold.status,
        rmsd=None,
        tm_score=None,
        gdt_ts=None,
        raw_reward=None,
        is_good=False,
        oracle_cache_hit=oracle_cache_hit,
        oracle_latency_ms=fold.latency_ms,
        oracle_key=oracle_key,
        error_message=fold.error_message,
        plddt=fold.plddt,
    ).validate()


def _compose_record_from_fold(
    *,
    fold: FoldResult,
    target: TargetRecord,
    composer: RewardComposer,
    oracle_key: str,
    oracle_cache_hit: bool,
    metric_cache: JsonDiskCache | None,
    metric_version: str,
    alignment_config: str,
    design_sequence: str | None = None,
) -> RewardRecord:
    if fold.status != OK_STATUS:
        return _fold_failed_record(fold, oracle_key, oracle_cache_hit=oracle_cache_hit)
    if fold.predicted_structure_path is None or fold.predicted_structure_hash is None:
        return _metric_failed_record(fold, oracle_key, "worker did not return a predicted structure", oracle_cache_hit=oracle_cache_hit)

    m_key = metric_cache_key(target.structure_hash, fold.predicted_structure_hash, metric_version, alignment_config)
    cached = metric_cache.get(m_key) if metric_cache is not None else None
    if cached is not None:
        return _record_from_cache(cached, composer, oracle_key)
    try:
        predicted = load_c4p_coords(fold.predicted_structure_path)
        reference = target.reference_c4p_coords if target.reference_c4p_coords is not None else load_c4p_coords(target.structure_ref)
        aux: Mapping[str, float | None] | None = None
        if composer.preset == "geometric_v3" and design_sequence:
            # Only pay the ViennaRNA cost for the preset that actually uses
            # these terms, so preset A keeps its exact original timing.
            aux = compute_auxiliary_metrics(design_sequence, target.sequence_native)
        metrics = compute_structural_metrics(predicted, reference, plddt=fold.plddt, aux=aux)
    except Exception as exc:
        return _metric_failed_record(fold, oracle_key, str(exc), oracle_cache_hit=oracle_cache_hit)
    if metric_cache is not None:
        metric_cache.set(m_key, _metric_to_cache(metrics))
    return composer.record(metrics, oracle_key=oracle_key, oracle_latency_ms=fold.latency_ms, oracle_cache_hit=oracle_cache_hit)


def _mean(values: Sequence[float | int | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return float(sum(numeric) / len(numeric))


def _std(values: Sequence[float | int | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if len(numeric) < 2:
        return 0.0 if numeric else None
    mean = sum(numeric) / len(numeric)
    return float(math.sqrt(sum((value - mean) ** 2 for value in numeric) / len(numeric)))


def unique_sequences(samples: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for sequence in samples:
        normalized = normalize_sequence(sequence)
        if normalized not in seen:
            seen.add(normalized)
            ordered.append(normalized)
    return ordered


class RewardScorer:
    """Production reward facade used by the engine."""

    def __init__(self, config: Mapping[str, object] | None = None, *, oracle: MetricsOracleProtocol | None = None) -> None:
        cfg = dict(config or {})
        oracle_cfg = dict(cfg.get("oracle", {})) if isinstance(cfg.get("oracle", {}), Mapping) else {}
        composer_cfg = dict(cfg.get("composer", {})) if isinstance(cfg.get("composer", {}), Mapping) else {}
        cache_cfg = dict(cfg.get("metric_cache", {})) if isinstance(cfg.get("metric_cache", {}), Mapping) else {}
        self.composer = RewardComposer(str(composer_cfg.get("preset", "initial_plan_strict")), train_mode=True)
        cache_enabled = bool(cache_cfg.get("enabled", True))
        self.fold_cache = JsonDiskCache(Path(str(cache_cfg.get("dir", "cache/rl_metrics"))) / "fold") if cache_enabled else None
        self.metric_cache = JsonDiskCache(Path(str(cache_cfg.get("dir", "cache/rl_metrics"))) / "metric") if cache_enabled else None
        self.metric_version = str(cache_cfg.get("metric_version", METRIC_VERSION))
        self.alignment_config = str(cache_cfg.get("alignment_config", DEFAULT_ALIGNMENT_CONFIG))
        self.oracle_checkpoint_path, self.oracle_checkpoint_hash = self._resolve_oracle_checkpoint(oracle_cfg, external_oracle=oracle is not None)
        hash_cfg = dict(oracle_cfg)
        if self.oracle_checkpoint_hash:
            hash_cfg["checkpoint_hash"] = self.oracle_checkpoint_hash
        self.oracle_mode_and_config_hash = stable_hash(hash_cfg)
        self._external_oracle = oracle is not None
        if oracle is not None:
            self.oracle = oracle
        else:
            command = [
                str(oracle_cfg.get("python", "python")),
                str(oracle_cfg.get("worker_script", "scripts/rhofold_oracle_worker.py")),
                "--device",
                str(oracle_cfg.get("device", "cuda:0")),
            ]
            if self.oracle_checkpoint_path is not None:
                command.extend(["--ckpt", self.oracle_checkpoint_path])
            if bool(oracle_cfg.get("fake", False)):
                command.append("--fake")
            self.oracle = RhoFoldJsonlClient(
                command,
                timeout_s=float(oracle_cfg.get("timeout_sec", 300.0)),
                output_root=str(oracle_cfg.get("output_dir", "outputs/rhofold_predictions")),
            )

    def score(self, rollout: RolloutBatch, target: TargetRecord) -> ScoredBatch:
        cache_hits, cache_misses, unique = self._fold_cache_stats(rollout)
        metric_hits_before, metric_misses_expected = self._metric_cache_stats(rollout, target)
        rewards = score_rollout_batch(
            rollout,
            target,
            self.oracle,
            self.composer,
            fold_cache=self.fold_cache,
            metric_cache=self.metric_cache,
            oracle_checkpoint_hash=self.oracle_checkpoint_hash,
            oracle_mode_and_config_hash=self.oracle_mode_and_config_hash,
            metric_version=self.metric_version,
            alignment_config=self.alignment_config,
        )
        metric_hits_after, _ = self._metric_cache_stats(rollout, target)
        return ScoredBatch(
            rollout=rollout,
            rewards=rewards,
            target=target,
            oracle_calls=cache_misses,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            metric_cache_hits=metric_hits_before,
            metric_cache_misses=max(0, metric_hits_after - metric_hits_before) if metric_misses_expected else 0,
            unique_sequences=unique,
        )

    def _fold_cache_stats(self, rollout: RolloutBatch) -> tuple[int, int, int]:
        unique = unique_sequences(sample.tokens for sample in rollout.samples)
        hits = 0
        for sequence in unique:
            key = fold_cache_key(sequence, self.oracle_checkpoint_hash, self.oracle_mode_and_config_hash)
            if self.fold_cache is not None and self.fold_cache.get(key) is not None:
                hits += 1
        misses = len(unique) - hits
        return hits, misses, len(unique)

    def _metric_cache_stats(self, rollout: RolloutBatch, target: TargetRecord) -> tuple[int, int]:
        if self.fold_cache is None or self.metric_cache is None:
            return 0, 0
        hits = 0
        missable = 0
        for sequence in unique_sequences(sample.tokens for sample in rollout.samples):
            fold_payload = self.fold_cache.get(fold_cache_key(sequence, self.oracle_checkpoint_hash, self.oracle_mode_and_config_hash))
            if fold_payload is None:
                continue
            fold = _fold_from_cache(fold_payload)
            if fold.status != OK_STATUS or fold.predicted_structure_hash is None:
                continue
            missable += 1
            key = metric_cache_key(target.structure_hash, fold.predicted_structure_hash, self.metric_version, self.alignment_config)
            if self.metric_cache.get(key) is not None:
                hits += 1
        return hits, max(0, missable - hits)

    @staticmethod
    def _resolve_oracle_checkpoint(oracle_cfg: Mapping[str, object], *, external_oracle: bool) -> tuple[str | None, str]:
        if bool(oracle_cfg.get("fake", False)):
            fake_hash = str(oracle_cfg.get("checkpoint_hash") or "fake-rhofold-checkpoint")
            if fake_hash == "unknown":
                raise ValueError("fake oracle checkpoint_hash must not be 'unknown'")
            return None, fake_hash
        ckpt = oracle_cfg.get("ckpt") or oracle_cfg.get("checkpoint")
        provided_hash = oracle_cfg.get("checkpoint_hash")
        if external_oracle and ckpt is None:
            injected_hash = str(provided_hash or "external-oracle-injected")
            if injected_hash == "unknown":
                raise ValueError("injected oracle checkpoint_hash must not be 'unknown'")
            return None, injected_hash
        if ckpt is None:
            raise ValueError("reward.oracle.ckpt is required for real RhoFold oracle; production checkpoint_hash cannot be unknown")
        path = Path(str(ckpt)).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"reward.oracle.ckpt does not exist: {path}")
        digest = file_sha256(path)
        if provided_hash and str(provided_hash) not in {"", "unknown", digest}:
            raise ValueError(f"reward.oracle.checkpoint_hash mismatch for {path}: expected {provided_hash}, got {digest}")
        return str(path), digest

    def manifest_state(self) -> dict[str, object]:
        return {
            "oracle_checkpoint_path": self.oracle_checkpoint_path,
            "oracle_checkpoint_sha256": self.oracle_checkpoint_hash,
            "oracle_mode_and_config_hash": self.oracle_mode_and_config_hash,
            "metric_version": self.metric_version,
            "alignment_config": self.alignment_config,
        }

    def close(self) -> None:
        if not self._external_oracle and hasattr(self.oracle, "close"):
            self.oracle.close()
