"""
SSTT-aligned metrics for MORAD evaluation.

Implements the four-axis framework from RiboPO (arXiv 2510.21161) plus
CASP15-style native-ceiling normalisation, so numbers are directly
comparable with published RNA inverse-folding work.

Axes
----
S1 Sequence      : recovery, perplexity, 3-mer diversity
S2 Secondary     : EternaFold scMCC / scF1 (falls back to RNAfold)
T1 Tertiary      : TM, GDT, RMSD (existing) + lDDT, clashscore, INF
T2 Thermodynamic : MFE, ensemble defect (ED, ED/nt), P(S0),
                   Shannon entropy, ensemble diversity, melting temperature
Cross-cutting    : native-ceiling achievement ratio, pass@k, reliability strata

Every function is pure and side-effect free unless noted. Thermodynamic and
sequence axes require no deep-learning model at all.
"""

from __future__ import annotations

import math
import os
import re
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------

_BASES = ("A", "C", "G", "U")
_RNAFOLD = os.environ.get("RNAFOLD_BINARY") or "RNAfold"
_ETERNAFOLD = os.environ.get("ETERNAFOLD_BINARY") or ""
_ETERNAFOLD_PARAMS = os.environ.get("ETERNAFOLD_PARAMS") or ""

# lDDT thresholds and inclusion radius follow Mariani et al. 2013, the same
# configuration CASP uses for nucleic acids.
_LDDT_TOLERANCES = (0.5, 1.0, 2.0, 4.0)
_LDDT_INCLUSION_RADIUS = 15.0

# A C4'-C4' distance below this is physically impossible for distinct residues.
# Coarse-grained stand-in for MolProbity clashscore, which needs all atoms.
_CLASH_MIN_DIST = 4.0
_CLASH_SEQ_SEPARATION = 2


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def normalise_sequence(seq: str) -> str:
    """Uppercase and map T->U so DNA-style input still scores correctly."""
    return seq.strip().upper().replace("T", "U")


def parse_dotbracket(db: str) -> dict[int, int]:
    """Return a 0-indexed pairing map from dot-bracket notation.

    Handles the extra bracket families that appear when a secondary structure
    is derived from 3D coordinates (pseudoknots are annotated with [], {}, <>
    and letter pairs).
    """
    openers = "([{<" + "".join(chr(c) for c in range(ord("A"), ord("Z") + 1))
    closers = ")]}>" + "".join(chr(c) for c in range(ord("a"), ord("z") + 1))
    stacks: dict[str, list[int]] = {}
    pair: dict[int, int] = {}
    for i, ch in enumerate(db):
        if ch in openers:
            stacks.setdefault(ch, []).append(i)
        elif ch in closers:
            key = openers[closers.index(ch)]
            stack = stacks.get(key)
            if stack:
                j = stack.pop()
                pair[i] = j
                pair[j] = i
    return pair


def strip_pseudoknots(db: str) -> str:
    """Keep only the primary () bracket family.

    ViennaRNA's nearest-neighbour energy model cannot evaluate pseudoknotted
    structures and returns a 100000 sentinel energy. Dropping the higher
    bracket families yields a nested structure the model can score, at the
    cost of ignoring those pairs.
    """
    out = []
    for ch in db:
        if ch in "().":
            out.append(ch)
        else:
            out.append(".")
    return "".join(out)


def has_pseudoknot(db: str) -> bool:
    return any(ch not in "()." for ch in db)


# --------------------------------------------------------------------------
# S1 sequence axis
# --------------------------------------------------------------------------


def sequence_recovery(design: str, native: str) -> float | None:
    """Fraction of positions where the design matches the native base."""
    d, n = normalise_sequence(design), normalise_sequence(native)
    if not d or len(d) != len(n):
        return None
    return sum(a == b for a, b in zip(d, n)) / len(d)


def kmer_frequency_vector(seq: str, k: int = 3) -> np.ndarray:
    """Normalised k-mer frequency vector over the 4^k possible k-mers."""
    seq = normalise_sequence(seq)
    index = {
        "".join(t): i
        for i, t in enumerate(
            __import__("itertools").product(_BASES, repeat=k)
        )
    }
    vec = np.zeros(len(index), dtype=float)
    for i in range(len(seq) - k + 1):
        idx = index.get(seq[i : i + k])
        if idx is not None:
            vec[idx] += 1.0
    total = vec.sum()
    return vec / total if total > 0 else vec


def kmer_diversity(sequences: Sequence[str], k: int = 3) -> float | None:
    """RiboPO's 3-mer diversity: 1 - mean pairwise Pearson correlation.

    Higher means the candidate set explores more of sequence space. Returns
    None when fewer than two sequences are supplied.
    """
    seqs = [s for s in sequences if s]
    if len(seqs) < 2:
        return None
    vecs = [kmer_frequency_vector(s, k) for s in seqs]
    corrs = []
    for a in range(len(vecs)):
        for b in range(a + 1, len(vecs)):
            va, vb = vecs[a], vecs[b]
            if va.std() == 0 or vb.std() == 0:
                corrs.append(1.0 if np.allclose(va, vb) else 0.0)
            else:
                corrs.append(float(np.corrcoef(va, vb)[0, 1]))
    return 1.0 - float(np.mean(corrs))


def hamming_diversity(sequences: Sequence[str]) -> float | None:
    """Mean pairwise normalised Hamming distance among equal-length designs."""
    seqs = [normalise_sequence(s) for s in sequences if s]
    if len(seqs) < 2:
        return None
    dists = []
    for a in range(len(seqs)):
        for b in range(a + 1, len(seqs)):
            x, y = seqs[a], seqs[b]
            if len(x) != len(y):
                continue
            dists.append(sum(p != q for p, q in zip(x, y)) / len(x))
    return float(np.mean(dists)) if dists else None


def perplexity_from_logprobs(token_logprobs: Sequence[float]) -> float | None:
    """exp of the mean negative log-likelihood, i.e. gRNAde's perplexity.

    A perfect model scores 1.0; uniform guessing over 4 bases scores 4.0.
    Needs per-position log probabilities from the policy, so it is the only
    sequence-axis metric that touches the model.
    """
    lp = [float(x) for x in token_logprobs if x is not None and math.isfinite(x)]
    if not lp:
        return None
    return float(math.exp(-sum(lp) / len(lp)))


# --------------------------------------------------------------------------
# T2 thermodynamic axis
# --------------------------------------------------------------------------


@dataclass
class ThermoResult:
    """Everything one ViennaRNA partition-function call can yield."""

    mfe: float | None = None
    mfe_structure: str | None = None
    ensemble_free_energy: float | None = None
    mfe_frequency: float | None = None
    ensemble_diversity: float | None = None
    ensemble_defect: float | None = None
    ensemble_defect_per_nt: float | None = None
    target_probability: float | None = None
    target_energy: float | None = None
    positional_entropy: float | None = None
    melting_temperature: float | None = None
    gc_content: float | None = None
    target_had_pseudoknot: bool = False
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "mfe": self.mfe,
            "ensemble_free_energy": self.ensemble_free_energy,
            "mfe_frequency": self.mfe_frequency,
            "ensemble_diversity": self.ensemble_diversity,
            "ensemble_defect": self.ensemble_defect,
            "ensemble_defect_per_nt": self.ensemble_defect_per_nt,
            "target_probability": self.target_probability,
            "positional_entropy": self.positional_entropy,
            "melting_temperature": self.melting_temperature,
            "gc_content": self.gc_content,
            "target_had_pseudoknot": self.target_had_pseudoknot,
        }


def gc_content(seq: str) -> float | None:
    s = normalise_sequence(seq)
    if not s:
        return None
    return (s.count("G") + s.count("C")) / len(s)


def melting_temperature_nn(seq: str) -> float | None:
    """Approximate Tm via the Wallace/GC rule appropriate for RNA length.

    This is a cheap sanity proxy, not a substitute for a full nearest-neighbour
    Tm calculation; it is reported alongside MFE purely as a stability hint.
    """
    s = normalise_sequence(seq)
    n = len(s)
    if n == 0:
        return None
    gc = (s.count("G") + s.count("C")) / n
    if n < 14:
        return 2.0 * (s.count("A") + s.count("U")) + 4.0 * (s.count("G") + s.count("C"))
    return 64.9 + 41.0 * (gc - 16.4 / n)


def compute_thermo(
    sequence: str,
    target_structure: str | None = None,
    temperature: float = 37.0,
) -> ThermoResult:
    """Compute the whole thermodynamic axis in one partition-function pass.

    Pure statistical mechanics over the Turner nearest-neighbour parameters:
    no neural network is involved, so these numbers are immune to any bias in
    RhoFold+ or other learned predictors.
    """
    import RNA  # ViennaRNA python bindings

    seq = normalise_sequence(sequence)
    res = ThermoResult(gc_content=gc_content(seq), melting_temperature=melting_temperature_nn(seq))
    if not seq:
        res.notes.append("empty sequence")
        return res

    md = RNA.md()
    md.temperature = temperature
    fc = RNA.fold_compound(seq, md)

    mfe_struct, mfe_energy = fc.mfe()
    res.mfe = float(mfe_energy)
    res.mfe_structure = mfe_struct

    fc.exp_params_rescale(mfe_energy)
    _, ens_fe = fc.pf()
    res.ensemble_free_energy = float(ens_fe)

    kT = RNA.exp_param().kT / 1000.0
    res.mfe_frequency = float(math.exp(-(mfe_energy - ens_fe) / kT))
    res.ensemble_diversity = float(fc.mean_bp_distance())

    bpp = fc.bpp()
    n = len(seq)
    paired_prob = np.zeros(n + 1, dtype=float)
    for i in range(1, n + 1):
        row = bpp[i]
        for j in range(i + 1, n + 1):
            p = row[j]
            if p > 0.0:
                paired_prob[i] += p
                paired_prob[j] += p

    # Positional (Shannon) entropy of the pairing distribution, averaged.
    ent = 0.0
    for i in range(1, n + 1):
        probs = [bpp[min(i, j)][max(i, j)] for j in range(1, n + 1) if j != i]
        probs = [p for p in probs if p > 1e-12]
        unpaired = max(0.0, 1.0 - sum(probs))
        if unpaired > 1e-12:
            probs.append(unpaired)
        ent += -sum(p * math.log(p) for p in probs)
    res.positional_entropy = float(ent / n)

    if target_structure and len(target_structure) == n:
        res.target_had_pseudoknot = has_pseudoknot(target_structure)
        pair = parse_dotbracket(target_structure)

        # Ensemble defect (Zadeh et al. 2011): expected number of nucleotides
        # in an incorrect pairing state at equilibrium. Pseudoknotted pairs are
        # included here because this only needs base-pair probabilities, not an
        # energy evaluation of the target structure.
        defect = 0.0
        for i in range(n):
            if i in pair:
                j = pair[i]
                defect += 1.0 - bpp[min(i, j) + 1][max(i, j) + 1]
            else:
                defect += paired_prob[i + 1]
        res.ensemble_defect = float(defect)
        res.ensemble_defect_per_nt = float(defect / n)

        # P(S0) needs an energy evaluation, which the nearest-neighbour model
        # cannot do for pseudoknots. Fall back to the nested projection and
        # record that we did so.
        eval_struct = target_structure
        if res.target_had_pseudoknot:
            eval_struct = strip_pseudoknots(target_structure)
            res.notes.append("P(S0) computed on pseudoknot-stripped structure")
        try:
            e_target = float(fc.eval_structure(eval_struct))
            if e_target < 90000.0:
                res.target_energy = e_target
                res.target_probability = float(math.exp(-(e_target - ens_fe) / kT))
            else:
                res.notes.append("target structure not evaluable by energy model")
        except Exception as exc:  # pragma: no cover - defensive
            res.notes.append(f"eval_structure failed: {exc}")

    return res


# --------------------------------------------------------------------------
# S2 secondary-structure axis
# --------------------------------------------------------------------------


def _pairs_as_set(db: str) -> set[tuple[int, int]]:
    pair = parse_dotbracket(db)
    return {(min(i, j), max(i, j)) for i, j in pair.items()}


def secondary_agreement(predicted_db: str, target_db: str) -> dict[str, float | None]:
    """Compare two dot-bracket structures as base-pair sets.

    Returns MCC (RiboPO / gRNAde headline), F1 (this project's historical
    metric) and precision/recall so both conventions stay auditable.
    """
    if not predicted_db or not target_db or len(predicted_db) != len(target_db):
        return {"mcc": None, "f1": None, "precision": None, "recall": None}

    pred = _pairs_as_set(predicted_db)
    true = _pairs_as_set(target_db)
    n = len(target_db)
    total_pairs = n * (n - 1) // 2

    tp = len(pred & true)
    fp = len(pred - true)
    fn = len(true - pred)
    tn = total_pairs - tp - fp - fn

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision and recall and (precision + recall) > 0
        else (0.0 if (tp + fp + fn) else None)
    )

    denom = math.sqrt(
        float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn)
    )
    mcc = ((tp * tn - fp * fn) / denom) if denom > 0 else None

    return {"mcc": mcc, "f1": f1, "precision": precision, "recall": recall}


def fold_rnafold(sequences: Sequence[str]) -> list[str | None]:
    """MFE fold with ViennaRNA via its Python bindings.

    Uses the library directly rather than parsing RNAfold's stdout, so there is
    no text-format fragility. Pure nearest-neighbour thermodynamics, no
    learned parameters.
    """
    import RNA

    out: list[str | None] = []
    for raw in sequences:
        seq = normalise_sequence(raw)
        if not seq:
            out.append(None)
            continue
        try:
            structure, _ = RNA.fold(seq)
            out.append(structure if len(structure) == len(seq) else None)
        except Exception:
            out.append(None)
    return out


def fold_eternafold(
    sequences: Sequence[str],
    workers: int = 1,
) -> list[str | None]:
    """Fold with EternaFold, the predictor RiboPO and gRNAde report.

    EternaFold is a CONTRAfold-SE derivative whose parameters were learned from
    Eterna high-throughput chemical-mapping data, so it is more accurate than
    pure nearest-neighbour folding while remaining cheap. The binary handles one
    sequence per invocation, so a thread pool is used to keep many cores busy.
    """
    if not _ETERNAFOLD or not Path(_ETERNAFOLD).exists():
        raise RuntimeError(
            "ETERNAFOLD_BINARY not configured or missing; source scripts/env.sh first"
        )

    seqs = [normalise_sequence(s) for s in sequences]

    def _one(payload: tuple[int, str, str]) -> tuple[int, str | None]:
        idx, seq, workdir = payload
        if not seq:
            return idx, None
        path = Path(workdir) / f"s{idx}.seq"
        path.write_text(f">s{idx}\n{seq}\n")
        cmd = [_ETERNAFOLD, "predict", str(path)]
        if _ETERNAFOLD_PARAMS:
            cmd += ["--params", _ETERNAFOLD_PARAMS]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            return idx, None
        structure = None
        for line in proc.stdout.splitlines():
            cand = line.strip()
            if len(cand) == len(seq) and set(cand) <= set("().{}[]<>"):
                structure = cand
        return idx, structure

    results: list[str | None] = [None] * len(seqs)
    with tempfile.TemporaryDirectory() as tmp:
        payloads = [(i, s, tmp) for i, s in enumerate(seqs)]
        if workers <= 1:
            for payload in payloads:
                idx, structure = _one(payload)
                results[idx] = structure
        else:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=workers) as pool:
                for idx, structure in pool.map(_one, payloads):
                    results[idx] = structure
    return results


# --------------------------------------------------------------------------
# T1 tertiary axis additions
# --------------------------------------------------------------------------


def lddt(
    predicted: np.ndarray,
    reference: np.ndarray,
    inclusion_radius: float = _LDDT_INCLUSION_RADIUS,
    tolerances: Sequence[float] = _LDDT_TOLERANCES,
) -> float | None:
    """Local Distance Difference Test on same-index coordinates.

    Superposition-free: it scores whether local inter-atomic distances are
    reproduced, which is why CASP15 found it discriminates *low* accuracy
    models better than GDT_TS does.
    """
    if predicted is None or reference is None:
        return None
    p = np.asarray(predicted, dtype=float)
    r = np.asarray(reference, dtype=float)
    if p.ndim != 2 or p.shape != r.shape or p.shape[0] < 2:
        return None

    dist_ref = np.linalg.norm(r[:, None, :] - r[None, :, :], axis=-1)
    dist_pred = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)

    n = p.shape[0]
    mask = (dist_ref < inclusion_radius) & ~np.eye(n, dtype=bool)
    if not mask.any():
        return None

    diff = np.abs(dist_pred - dist_ref)
    preserved = np.zeros_like(diff, dtype=float)
    for tol in tolerances:
        preserved += (diff < tol).astype(float)
    preserved /= float(len(tolerances))

    per_residue = []
    for i in range(n):
        sel = mask[i]
        if sel.any():
            per_residue.append(float(preserved[i][sel].mean()))
    return float(np.mean(per_residue)) if per_residue else None


def coarse_clash_score(
    coords: np.ndarray,
    min_distance: float = _CLASH_MIN_DIST,
    seq_separation: int = _CLASH_SEQ_SEPARATION,
) -> float | None:
    """Clashes per 1000 residues using a single representative atom per residue.

    True MolProbity clashscore needs all-atom models with hydrogens. With only
    C4'/C1' traces available this reports how often non-adjacent residues come
    implausibly close, which still flags collapsed or self-overlapping models.
    """
    if coords is None:
        return None
    c = np.asarray(coords, dtype=float)
    if c.ndim != 2 or c.shape[0] < seq_separation + 2:
        return None
    n = c.shape[0]
    dist = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=-1)
    idx = np.arange(n)
    far_in_sequence = np.abs(idx[:, None] - idx[None, :]) > seq_separation
    clashes = int(np.sum((dist < min_distance) & far_in_sequence) // 2)
    return float(clashes * 1000.0 / n)


def base_pair_inf(predicted_db: str, target_db: str) -> float | None:
    """Interaction Network Fidelity restricted to base pairs.

    Parisien et al. define INF as the MCC over annotated interaction sets.
    Without an all-atom annotator (MC-Annotate / FR3D) only the canonical
    pairing layer is available, so this is INF_WC rather than INF_ALL.
    """
    scores = secondary_agreement(predicted_db, target_db)
    return scores["mcc"]


# --------------------------------------------------------------------------
# cross-cutting
# --------------------------------------------------------------------------


def achievement_ratio(design_value: float | None, native_value: float | None) -> float | None:
    """design / native, i.e. how much of the oracle's own ceiling was reached.

    CASP15 does the same thing with experimental precision as the reference
    point, because a predictor cannot be expected to beat the accuracy limit
    of the method used to score it.
    """
    if design_value is None or native_value is None:
        return None
    if native_value == 0:
        return None
    return float(design_value) / float(native_value)


def pass_at_k(success_flags: Sequence[bool], k: int) -> float | None:
    """Probability that at least one of k independent draws succeeds.

    Uses RiboPO's estimator pass@k = 1 - (1-p)^k with p taken as the empirical
    success rate of the sampled candidates.
    """
    flags = [bool(f) for f in success_flags]
    if not flags or k < 1:
        return None
    p = sum(flags) / len(flags)
    return float(1.0 - (1.0 - p) ** k)


def pass_at_k_unbiased(n: int, c: int, k: int) -> float | None:
    """Unbiased pass@k for n samples of which c succeeded (Codex estimator).

    Preferred over the plug-in estimator when n is small, since 1-(1-p)^k with
    an empirical p is biased upward.
    """
    if n <= 0 or k < 1 or k > n:
        return None
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    prob_all_fail = 1.0
    for i in range(k):
        prob_all_fail *= (n - c - i) / (n - i)
    return float(1.0 - prob_all_fail)


def reliability_tier(native_gdt: float | None) -> str:
    """Bucket a target by how well the oracle reproduces its own native fold.

    Targets whose native sequence cannot be folded back are not informative:
    any score there reflects oracle error rather than design quality.
    """
    if native_gdt is None:
        return "unknown"
    if native_gdt < 0.3:
        return "unusable"
    if native_gdt < 0.5:
        return "weak"
    if native_gdt < 0.7:
        return "moderate"
    return "reliable"


def summarise(values: Iterable[float | None]) -> dict[str, float | int | None]:
    """Mean/median/std/quantiles over the non-missing entries."""
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return {"n": 0, "mean": None, "median": None, "std": None, "p10": None, "p90": None}
    arr = np.asarray(vals, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


# Direction of improvement for every metric this module produces, so reports
# never have to guess whether higher or lower is better.
METRIC_DIRECTION: Mapping[str, str] = {
    # sequence
    "sequence_recovery": "higher",
    "perplexity": "lower",
    "kmer_diversity_3": "higher",
    "hamming_diversity": "higher",
    # secondary
    "secondary_mcc": "higher",
    "secondary_f1": "higher",
    "secondary_precision": "higher",
    "secondary_recall": "higher",
    # tertiary
    "tm_score": "higher",
    "gdt_ts": "higher",
    "rmsd": "lower",
    "lddt": "higher",
    "inf_wc": "higher",
    "clash_score": "lower",
    # thermodynamic
    "mfe": "lower",
    "ensemble_free_energy": "lower",
    "mfe_frequency": "higher",
    "ensemble_diversity": "context",
    "ensemble_defect": "lower",
    "ensemble_defect_per_nt": "lower",
    "target_probability": "higher",
    "positional_entropy": "lower",
    "melting_temperature": "context",
    "gc_content": "context",
    # functional
    "rfam_family_success": "higher",
    # cross-cutting
    "achievement_ratio": "higher",
    "pass_at_k": "higher",
}
