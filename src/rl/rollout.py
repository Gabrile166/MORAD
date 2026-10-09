"""Rollout strategies for RIDER RL."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F

from src.constants import NUM_TO_LETTER
from src.rl.model_adapter import RIDEModelAdapter
from src.rl.protocol import RolloutBatch, RolloutSample


@dataclass(frozen=True)
class RolloutState:
    round_id: Any = "round0"
    step: int = 0
    base_seed: int = 0
    policy_version: str = "unknown"
    temperature: Optional[float] = None
    rescue_count: int = 0
    temperature_trajectory: tuple[float, ...] = ()


@dataclass(frozen=True)
class AdaptiveX0RenoiseRolloutConfig:
    group_size: int = 8
    n_steps: int = 50
    temperature: float = 1.0
    temperature_min: float = 0.0
    temperature_max: float = 2.0
    microbatch_size: int = 1
    include_eval_probe: bool = True
    keep_final_latent_debug: bool = False
    sampler_version: str = "x0-renoise-v1"
    max_rescues: int = 0
    rescue_min_unique_sequences: int = 2
    rescue_min_valid_sequences: int = 2
    rescue_temperature_step: float = 0.1


class AdaptiveX0RenoiseRollout:
    """Generate one logical group with the existing x0 re-noise semantics."""

    sampler_name = "AdaptiveX0RenoiseRollout"

    def __init__(
        self,
        noise_scheduler: Any,
        config: Optional[AdaptiveX0RenoiseRolloutConfig | dict[str, Any]] = None,
        *,
        adapter_cls: type[RIDEModelAdapter] = RIDEModelAdapter,
    ):
        self.noise_scheduler = noise_scheduler
        self.config = self._coerce_config(config)
        self.adapter_cls = adapter_cls
        self.last_eval_probe: Optional[RolloutSample] = None
        self._state: dict[str, Any] = {}

    @staticmethod
    def _coerce_config(config: Optional[AdaptiveX0RenoiseRolloutConfig | dict[str, Any]]) -> AdaptiveX0RenoiseRolloutConfig:
        if config is None:
            return AdaptiveX0RenoiseRolloutConfig()
        if isinstance(config, AdaptiveX0RenoiseRolloutConfig):
            return config
        sampler = dict(config.get("sampler", {}))
        temp = dict(config.get("temperature", {}))
        rescue = dict(config.get("rescue", {}))
        return AdaptiveX0RenoiseRolloutConfig(
            group_size=int(config.get("group_size", 8)),
            n_steps=int(sampler.get("n_steps", sampler.get("max_reverse_steps", 50))),
            temperature=float(temp.get("initial", config.get("temperature", 1.0))),
            temperature_min=float(temp.get("min", temp.get("minimum", 0.0))),
            temperature_max=float(temp.get("max", temp.get("maximum", 2.0))),
            sampler_version=str(sampler.get("version", "v1")),
            max_rescues=int(rescue.get("max_rounds", rescue.get("max_rescues", 0))),
            rescue_min_unique_sequences=int(rescue.get("min_unique_sequences", 2)),
            rescue_min_valid_sequences=int(rescue.get("min_valid_sequences", 2)),
            rescue_temperature_step=float(temp.get("step", rescue.get("temperature_step", 0.1))),
        )

    def generate(
        self,
        target: Any,
        condition: Any,
        old_policy: Any,
        state: RolloutState,
        sample_indices: "tuple[int, ...] | None" = None,
    ) -> RolloutBatch:
        """Generate train samples plus an optional deterministic eval probe.

        `sample_indices` restricts generation to a subset of the rollout group so
        several ranks can split one target's group between them. `sample_index`
        and `seed` are still derived from the *global* index, so the samples are
        bit-identical to the single-rank case -- only the work is divided.
        `None` means "produce the whole group".
        """

        adapter = old_policy if isinstance(old_policy, RIDEModelAdapter) else self.adapter_cls(old_policy)
        policy_version = str(getattr(state, "policy_version", "unknown"))
        target_id = str(getattr(condition, "target_id", getattr(target, "target_id", "target")))
        round_id = getattr(condition, "round_id", getattr(state, "round_id", "round0"))
        condition_id = str(getattr(condition, "condition_id", f"{target_id}:{round_id}"))
        temperature = self._resolve_temperature(state)
        trajectory = tuple(getattr(state, "temperature_trajectory", ()) or ()) + (float(temperature),)
        self._state = {
            **self._state,
            "last_temperature": float(temperature),
            "temperature_trajectory": list(trajectory),
            "rescue_count": int(getattr(state, "rescue_count", 0)),
        }
        encoding = adapter.encode_condition(condition, "old", policy_version)

        with torch.no_grad():
            samples = [
                self._sample_one(
                    adapter=adapter,
                    encoding=encoding,
                    target_id=target_id,
                    round_id=round_id,
                    condition_id=condition_id,
                    policy_version=policy_version,
                    sample_index=index,
                    seed=int(state.base_seed) + index,
                    temperature=temperature,
                    deterministic=False,
                )
                for index in (
                    tuple(range(self.config.group_size))
                    if sample_indices is None
                    else tuple(sample_indices)
                )
            ]

            eval_probe = None
            # Only the rank holding sample 0 emits the probe, so a split group
            # does not produce duplicates.
            emit_probe = self.config.include_eval_probe and (
                sample_indices is None or 0 in tuple(sample_indices)
            )
            if emit_probe:
                eval_probe = self._sample_one(
                    adapter=adapter,
                    encoding=encoding,
                    target_id=target_id,
                    round_id=round_id,
                    condition_id=condition_id,
                    policy_version=policy_version,
                    sample_index=-1,
                    seed=int(state.base_seed) - 1,
                    temperature=0.0,
                    deterministic=True,
                )
        self.last_eval_probe = eval_probe

        return RolloutBatch(samples=tuple(samples), condition=condition).validate()

    def next_round(self, scored_batch: Any, state: RolloutState) -> RolloutState | None:
        """Return the next rescue rollout state when diversity/validity gates fail."""
        rescue_count = int(getattr(state, "rescue_count", 0))
        if rescue_count >= int(self.config.max_rescues):
            return None
        rewards = getattr(scored_batch, "rewards", None)
        records = tuple(getattr(rewards, "records", ()) or ())
        valid_count = sum(1 for record in records if getattr(record, "status", None) == "ok")
        samples = tuple(getattr(getattr(scored_batch, "rollout", None), "samples", ()) or ())
        unique_count = len({str(getattr(sample, "tokens", "")) for sample in samples})
        reasons: list[str] = []
        if valid_count < int(self.config.rescue_min_valid_sequences):
            reasons.append(f"valid<{self.config.rescue_min_valid_sequences}")
        if unique_count < int(self.config.rescue_min_unique_sequences):
            reasons.append(f"unique<{self.config.rescue_min_unique_sequences}")
        if not reasons:
            return None
        current_temperature = self._resolve_temperature(state)
        next_temperature = min(
            float(self.config.temperature_max),
            current_temperature + float(self.config.rescue_temperature_step),
        )
        trajectory = tuple(getattr(state, "temperature_trajectory", ()) or ()) + (float(current_temperature),)
        next_state = RolloutState(
            round_id=getattr(state, "round_id", "round0"),
            step=int(getattr(state, "step", 0)) + 1,
            base_seed=int(getattr(state, "base_seed", 0)) + 1009,
            policy_version=str(getattr(state, "policy_version", "unknown")),
            temperature=next_temperature,
            rescue_count=rescue_count + 1,
            temperature_trajectory=trajectory,
        )
        self._state = {
            **self._state,
            "rescue_count": rescue_count + 1,
            "last_rescue_reason": ",".join(reasons),
            "temperature_trajectory": list(trajectory) + [next_temperature],
        }
        return next_state

    def _sample_one(
        self,
        *,
        adapter: RIDEModelAdapter,
        encoding: Any,
        target_id: str,
        round_id: Any,
        condition_id: str,
        policy_version: str,
        sample_index: int,
        seed: int,
        temperature: float,
        deterministic: bool,
    ) -> RolloutSample:
        started = time.perf_counter()
        device = self._device_from_encoding(encoding)
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))

        out_dim = int(getattr(adapter.model, "out_dim", 4))
        x = torch.randn((encoding.num_nodes, out_dim), device=device, generator=generator)
        time_grid = torch.linspace(
            float(self.noise_scheduler.T),
            float(self.noise_scheduler.eps),
            int(self.config.n_steps),
            device=device,
        )

        for step_index in range(len(time_grid) - 1):
            t = time_grid[step_index].unsqueeze(0)
            t_next = time_grid[step_index + 1].unsqueeze(0)
            alpha_t, sigma_t = self.noise_scheduler.marginal_prob(t)
            noise_level = torch.log(alpha_t**2 / sigma_t**2)
            pred_noise = adapter.predict_noise(encoding, x, noise_level, time=t.unsqueeze(0))
            x0_hat = (x - sigma_t * pred_noise) / alpha_t
            alpha_next, sigma_next = self.noise_scheduler.marginal_prob(t_next)

            if deterministic:
                noise = torch.zeros_like(x)
            else:
                noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator) * temperature
            x = alpha_next * x0_hat + sigma_next * noise

        tokens = torch.argmax(x, dim=-1)
        x0_onehot = F.one_hot(tokens, num_classes=out_dim).to(dtype=x.dtype)
        sample_label = "eval" if deterministic else f"{sample_index:04d}"
        return RolloutSample(
            sample_id=f"{target_id}:{round_id}:{sample_label}",
            target_id=target_id,
            round_id=int(round_id),
            tokens=_tokens_to_string(tokens),
            x0_onehot=x0_onehot.detach().cpu(),
            seed=int(seed),
            temperature=float(temperature),
            policy_version=policy_version,
            condition_id=condition_id,
            sampler_name=self.sampler_name,
            sampler_version=self.config.sampler_version,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        ).validate()

    def _resolve_temperature(self, state: RolloutState) -> float:
        value = self.config.temperature if state.temperature is None else float(state.temperature)
        return min(max(value, self.config.temperature_min), self.config.temperature_max)

    def state_dict(self) -> dict[str, Any]:
        return {"last_eval_probe": self.last_eval_probe.sample_id if self.last_eval_probe is not None else None, **self._state}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._state = dict(state)

    @staticmethod
    def _device_from_encoding(encoding: Any) -> torch.device:
        edge_index = getattr(encoding, "edge_index", None)
        if isinstance(edge_index, torch.Tensor):
            return edge_index.device
        condition = getattr(encoding, "condition", None)
        for attr in ("seq", "node_s", "node_features"):
            value = getattr(condition, attr, None)
            if isinstance(value, torch.Tensor):
                return value.device
        return torch.device("cpu")


def _tokens_to_string(tokens: torch.Tensor) -> str:
    return "".join(NUM_TO_LETTER[int(token)] for token in tokens.detach().cpu().tolist())
