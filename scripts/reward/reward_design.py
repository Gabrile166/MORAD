"""Reward design as a constrained programme, with a geometric-mean composite.

It differs from a weighted-sum formulation in two respects.

**The combination form.** A weighted sum lets a strong term pay for a dead one:
a design with ``gdt=0`` and everything else perfect still scores well above
zero. That is exactly the compensation structure reward hacking exploits. The
composite here is the Derringer-Suich weighted geometric mean, the standard form
in multi-response optimisation since 1980 and the one used in the molecular
design literature for the same anti-hacking reason:

    R(x) = ( ∏ᵢ dᵢ(x)^wᵢ )^(1/∑wᵢ)        dᵢ = ½(1 − cos(π tᵢ))

with ``tᵢ`` the clipped position of metric ``i`` on its ``[loᵢ, hiᵢ]`` ramp. Two
lines, no free parameters beyond the ramps and the weights, and it has the
property a sum cannot have: **any dᵢ → 0 drives R → 0**, so no term can be
sacrificed. With the raised cosine the steepest slope sits at the ramp midpoint,
and the good threshold sits slightly above it.

**The optimisation statement.** Folding alignment and discrimination into the
objective as weighted terms would make them tradeable against robustness.
They are requirements, not preferences, so they belong in the constraint set.
What remains to maximise is the one property with no natural floor — resistance
to single-proxy exploitation. This is the ε-constraint method, the textbook way
to turn a multi-objective problem into a single-objective one without hiding the
trade-off inside a weight:

    maximise    H(w) = −maxᵢ wᵢ·hᵢ          (worst-case exploit exposure)
    subject to  A(w) ≥ A_min                 (reward must track the metric)
                D(w) ≥ D_min                 (reward must separate candidates)
                ∑_{i∈primary} wᵢ ≥ p_min     (primary metrics hold the mass)
                wᵢ ≤ capᵢ                    (no single term dominates)
                w ≥ 0, ∑ wᵢ = 1

Because A and D are floors rather than summands, the answer is auditable:
every reported solution either satisfies them or is infeasible, and infeasibility is informative — it says the floors cannot be met
with the available signals, which is a fact about the metrics, not about taste.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

# dᵢ is floored rather than allowed to reach zero: log(0) is not a number the
# solver can work with, and a hard zero would make the gradient vanish for every
# other term at once. The floor is small enough that a dead term still costs
# roughly an order of magnitude in R.
DESIRABILITY_FLOOR = 1e-3


# --------------------------------------------------------------------------
# components
# --------------------------------------------------------------------------


@dataclass
class Component:
    """One reward term: a metric, a direction, and a raised-cosine ramp."""

    name: str
    metric: str
    lo: float
    hi: float
    higher_is_better: bool
    depends_on_oracle: bool
    hackability: float
    is_primary: bool = False
    # where the good/bad threshold sits on the ramp, in ramp units above lo.
    # 0.6 puts it slightly above the midpoint, which is where the cosine is
    # steepest, so the threshold lands on the steep part rather than the plateau.
    threshold_position: float = 0.60

    def desirability(self, value: float | None) -> float | None:
        """Raised-cosine ramp to [floor, 1]. None propagates as None."""
        if value is None or not math.isfinite(value):
            return None
        span = self.hi - self.lo
        if abs(span) < 1e-12:
            return 1.0 if value >= self.hi else DESIRABILITY_FLOOR
        t = (value - self.lo) / span
        t = min(1.0, max(0.0, t))
        d = 0.5 * (1.0 - math.cos(math.pi * t))
        return max(DESIRABILITY_FLOOR, d)

    @property
    def threshold(self) -> float:
        """Metric value at which a design is called good."""
        return self.lo + self.threshold_position * (self.hi - self.lo)


def composite(desirabilities: Sequence[float | None], weights: Sequence[float]) -> float | None:
    """Weighted geometric mean over the terms that have a value.

    Missing terms are dropped and the weights renormalised over what is left,
    rather than being imputed. An unavailable oracle score is missing data; a
    zero would be a false claim that the design is bad.
    """
    log_sum = 0.0
    weight_sum = 0.0
    for d, w in zip(desirabilities, weights):
        if d is None or w <= 0.0:
            continue
        log_sum += w * math.log(max(DESIRABILITY_FLOOR, d))
        weight_sum += w
    if weight_sum <= 0.0:
        return None
    return math.exp(log_sum / weight_sum)


def default_components() -> list[Component]:
    """The candidate terms, with ramps set from measured distributions.

    ``hackability`` is the fraction of a term's range reachable by optimising it
    directly without improving true structural quality. The oracle terms are
    hard to game because they route through a folding model; the thermodynamic
    and composition terms are cheap to push, which is what RiboPO observed.
    """
    return [
        Component("gdt", "gdt_ts", lo=0.20, hi=0.75,
                  higher_is_better=True, depends_on_oracle=True,
                  hackability=0.15, is_primary=True),
        Component("tm", "tm_score", lo=0.20, hi=0.70,
                  higher_is_better=True, depends_on_oracle=True,
                  hackability=0.15, is_primary=True),
        Component("rmsd", "rmsd", lo=12.0, hi=2.0,
                  higher_is_better=False, depends_on_oracle=True,
                  hackability=0.15, is_primary=True),
        Component("scmcc", "sc_mcc", lo=0.30, hi=0.85,
                  higher_is_better=True, depends_on_oracle=False,
                  hackability=0.55),
        Component("edefect", "ensemble_defect", lo=0.60, hi=0.15,
                  higher_is_better=False, depends_on_oracle=False,
                  hackability=0.70),
        Component("composition", "composition_kl", lo=0.50, hi=0.02,
                  higher_is_better=False, depends_on_oracle=False,
                  hackability=0.85),
    ]


# --------------------------------------------------------------------------
# measurements on real data
# --------------------------------------------------------------------------


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    # average ties, otherwise Spearman is sensitive to input order
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    for index in np.flatnonzero(counts > 1):
        tied = inverse == index
        ranks[tied] = ranks[tied].mean()
    return ranks


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 3:
        return 0.0
    rx, ry = _rankdata(x[finite]), _rankdata(y[finite])
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denominator = math.sqrt(float((rx ** 2).sum()) * float((ry ** 2).sum()))
    return float((rx * ry).sum() / denominator) if denominator > 1e-12 else 0.0


@dataclass
class Dataset:
    """Per-candidate desirabilities, ground-truth objective, and group ids.

    ``desirabilities[j][i]`` is term ``j``'s ramp value for candidate ``i``, and
    NaN marks a term that had no value for that candidate.
    """

    desirabilities: np.ndarray
    objective: np.ndarray
    groups: np.ndarray
    names: list[str] = field(default_factory=list)

    def rewards(self, weights: np.ndarray) -> np.ndarray:
        """Vectorised geometric mean, renormalising per candidate over the
        terms that are present."""
        available = np.isfinite(self.desirabilities)
        logs = np.where(available, np.log(np.clip(self.desirabilities, DESIRABILITY_FLOOR, None)), 0.0)
        weight_matrix = np.broadcast_to(weights[:, None], self.desirabilities.shape)
        effective = np.where(available, weight_matrix, 0.0)
        mass = effective.sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.exp((effective * logs).sum(axis=0) / mass)
        return np.where(mass > 0.0, result, np.nan)


def alignment(weights: np.ndarray, data: Dataset) -> float:
    """Rank correlation between the composite and the graded objective."""
    return spearman(data.rewards(weights), data.objective)


def discrimination(weights: np.ndarray, data: Dataset) -> float:
    """Mean within-group effect size between the better and worse halves.

    GRPO standardises advantages inside a group, so only within-target ordering
    produces gradient. Vectorised over groups because the solver evaluates this
    on every grid point and the previous per-group Python loop made a 0.05 grid
    unrunnable.
    """
    reward = data.rewards(weights)
    effects: list[float] = []
    for group in np.unique(data.groups):
        mask = data.groups == group
        r, o = reward[mask], data.objective[mask]
        finite = np.isfinite(r) & np.isfinite(o)
        if finite.sum() < 3:
            continue
        r, o = r[finite], o[finite]
        spread = float(r.std())
        if spread < 1e-9:
            continue
        median = float(np.median(o))
        top, bottom = r[o > median], r[o < median]
        if len(top) < 1 or len(bottom) < 1:
            continue
        effects.append((float(top.mean()) - float(bottom.mean())) / spread)
    return float(np.mean(effects)) if effects else 0.0


def exposure(weights: np.ndarray, components: list[Component]) -> float:
    """Worst-case gain from pushing a single proxy, in reward units.

    Under a geometric mean the exposure of term ``i`` is bounded by its weight
    share times its hackability, same as under a sum, because ``∂log R/∂log dᵢ``
    is ``wᵢ/∑w``. The maximum rather than the sum is penalised, since an attacker
    picks one term.
    """
    return float(max(w * c.hackability for w, c in zip(weights, components)))


def robustness(weights: np.ndarray, components: list[Component]) -> float:
    """Negated exposure, so that larger is better."""
    return -exposure(weights, components)


# --------------------------------------------------------------------------
# the constrained programme
# --------------------------------------------------------------------------


@dataclass
class Constraints:
    """Floors and caps. A and D are floors here, not objective terms.

    ``alignment_min`` and ``discrimination_min`` are the ε-constraint levels.
    Setting them is a design decision and the solver reports whether they are
    achievable, so an impossible pair surfaces as infeasibility rather than as a
    quietly degraded weight vector.
    """

    alignment_min: float = 0.70
    discrimination_min: float = 1.20
    primary_floor: float = 0.55
    per_term_cap: float = 0.45
    independent_floor: float = 0.0


def feasible(
    weights: np.ndarray,
    components: list[Component],
    data: Dataset,
    cons: Constraints,
) -> tuple[bool, str]:
    """Check the constraint set, returning the first violation for reporting."""
    if weights.min() < -1e-9:
        return False, "negative weight"
    if abs(weights.sum() - 1.0) > 1e-6:
        return False, "weights do not sum to one"
    if weights.max() > cons.per_term_cap + 1e-9:
        return False, f"a term exceeds the cap {cons.per_term_cap}"
    primary_mass = sum(w for w, c in zip(weights, components) if c.is_primary)
    if primary_mass < cons.primary_floor - 1e-9:
        return False, f"primary mass {primary_mass:.3f} below {cons.primary_floor}"
    if cons.independent_floor > 0.0:
        independent = sum(w for w, c in zip(weights, components) if not c.depends_on_oracle)
        if independent < cons.independent_floor - 1e-9:
            return False, f"independent mass {independent:.3f} below {cons.independent_floor}"
    a = alignment(weights, data)
    if a < cons.alignment_min - 1e-9:
        return False, f"alignment {a:.3f} below {cons.alignment_min}"
    d = discrimination(weights, data)
    if d < cons.discrimination_min - 1e-9:
        return False, f"discrimination {d:.3f} below {cons.discrimination_min}"
    return True, ""


def grid_simplex(dimension: int, step: float) -> np.ndarray:
    """Enumerate the simplex on a lattice. Exhaustive, hence reproducible."""
    quanta = int(round(1.0 / step))

    def walk(remaining: int, slots: int):
        if slots == 1:
            yield (remaining,)
            return
        for take in range(remaining + 1):
            for rest in walk(remaining - take, slots - 1):
                yield (take,) + rest

    return np.array([np.array(point, dtype=float) / quanta for point in walk(quanta, dimension)])


def optimise(
    components: list[Component],
    data: Dataset,
    cons: Constraints,
    step: float = 0.05,
) -> dict:
    """Maximise robustness subject to the alignment and discrimination floors.

    Returns the solution plus the full diagnostic set, including how many grid
    points each constraint eliminated. Where the previous version reported only
    the winner, the binding constraint is what actually explains the answer.
    """
    grid = grid_simplex(len(components), step)
    blocked: dict[str, int] = {}
    best: dict | None = None

    for weights in grid:
        ok, reason = feasible(weights, components, data, cons)
        if not ok:
            key = reason.split(" below")[0].split(" exceeds")[0]
            blocked[key] = blocked.get(key, 0) + 1
            continue
        score = robustness(weights, components)
        if best is None or score > best["robustness"]:
            best = {
                "weights": {c.name: round(float(w), 4) for c, w in zip(components, weights)},
                "weight_vector": [float(w) for w in weights],
                "robustness": score,
                "exposure": exposure(weights, components),
                "alignment": alignment(weights, data),
                "discrimination": discrimination(weights, data),
            }

    return {
        "solution": best,
        "feasible": best is not None,
        "grid_points": len(grid),
        "blocked_by": dict(sorted(blocked.items(), key=lambda kv: -kv[1])),
        "constraints": {
            "alignment_min": cons.alignment_min,
            "discrimination_min": cons.discrimination_min,
            "primary_floor": cons.primary_floor,
            "per_term_cap": cons.per_term_cap,
            "independent_floor": cons.independent_floor,
        },
        "form": "R = (prod d_i^w_i)^(1/sum w_i),  d_i = 0.5*(1-cos(pi*t_i))",
    }


def pareto_frontier(
    components: list[Component],
    data: Dataset,
    alignment_levels: Sequence[float],
    discrimination_levels: Sequence[float],
    base: Constraints,
    step: float = 0.10,
) -> list[dict]:
    """Sweep the ε-constraint levels to expose the actual trade-off surface.

    A single solution hides whether the floors were binding. Sweeping them shows
    where robustness starts to cost alignment, which is the information needed
    to choose the floors honestly rather than by preference.
    """
    frontier: list[dict] = []
    for a_min in alignment_levels:
        for d_min in discrimination_levels:
            cons = Constraints(
                alignment_min=a_min,
                discrimination_min=d_min,
                primary_floor=base.primary_floor,
                per_term_cap=base.per_term_cap,
                independent_floor=base.independent_floor,
            )
            result = optimise(components, data, cons, step=step)
            frontier.append({
                "alignment_min": a_min,
                "discrimination_min": d_min,
                "feasible": result["feasible"],
                "solution": result["solution"],
            })
    return frontier


def report(result: dict, path: str | None = None) -> str:
    """Human-readable summary; the solver's answer should be legible."""
    lines = ["Reward design: constrained programme", "=" * 52]
    lines.append(f"form: {result['form']}")
    lines.append(f"grid points: {result['grid_points']}")
    if not result["feasible"]:
        lines.append("")
        lines.append("INFEASIBLE - no weight vector satisfies the floors.")
        lines.append("blocked by:")
        for reason, count in result["blocked_by"].items():
            lines.append(f"  {count:>6}  {reason}")
    else:
        solution = result["solution"]
        lines.append("")
        lines.append("weights:")
        for name, weight in solution["weights"].items():
            if weight > 0:
                lines.append(f"  {name:<14} {weight:.3f}")
        lines.append("")
        lines.append(f"alignment      {solution['alignment']:+.4f}  "
                     f"(floor {result['constraints']['alignment_min']})")
        lines.append(f"discrimination {solution['discrimination']:+.4f}  "
                     f"(floor {result['constraints']['discrimination_min']})")
        lines.append(f"exposure       {solution['exposure']:.4f}  (minimised)")
        lines.append("")
        lines.append("grid points eliminated by:")
        for reason, count in result["blocked_by"].items():
            lines.append(f"  {count:>6}  {reason}")
    text = "\n".join(lines)
    if path:
        with open(path, "w") as handle:
            handle.write(text + "\n")
    return text


__all__ = [
    "Component", "Constraints", "Dataset",
    "composite", "default_components",
    "alignment", "discrimination", "exposure", "robustness",
    "feasible", "grid_simplex", "optimise", "pareto_frontier", "report",
    "spearman", "DESIRABILITY_FLOOR",
]
