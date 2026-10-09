"""Noise-prediction adapter for RIDER diffusion policies.

The adapter keeps condition tensors immutable from the sampler/trainer point of
view: `z_t` and `noise_level` are explicit inputs, never hidden state written to
the caller-owned condition object.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Hashable, Iterator, Optional, Tuple

import torch

TensorPair = Tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class ConditionEncoding:
    """Cached graph encoder output for one policy/condition/device/dtype view."""

    condition_id: str
    policy_role: str
    policy_version: str
    device: str
    dtype: str
    num_nodes: int
    encoder_embeddings: Any = None
    edge_embeddings: Any = None
    edge_index: Optional[torch.Tensor] = None
    condition: Any = None
    custom: Any = None


def _condition_id(condition: Any) -> str:
    return str(
        getattr(
            condition,
            "condition_id",
            getattr(condition, "target_id", f"condition:{id(condition)}"),
        )
    )


def _graph_condition(condition: Any) -> Any:
    for attr in ("graph", "data", "batch", "pyg_data"):
        value = getattr(condition, attr, None)
        if value is not None:
            return value
    return condition


def _tensor_dtype(condition: Any) -> str:
    graph = _graph_condition(condition)
    for attr in ("node_s", "node_features", "seq"):
        value = getattr(graph, attr, None)
        if isinstance(value, torch.Tensor):
            return str(value.dtype)
    return str(torch.get_default_dtype())


def _tensor_device(condition: Any) -> str:
    graph = _graph_condition(condition)
    for attr in ("node_s", "node_features", "seq"):
        value = getattr(graph, attr, None)
        if isinstance(value, torch.Tensor):
            return str(value.device)
    return "cpu"


def _num_nodes(condition: Any) -> int:
    graph = _graph_condition(condition)
    seq = getattr(graph, "seq", None)
    if isinstance(seq, torch.Tensor):
        return int(seq.shape[0])
    node_features = getattr(graph, "node_features", None)
    if isinstance(node_features, torch.Tensor):
        return int(node_features.shape[-2])
    length = getattr(condition, "length", getattr(graph, "length", None))
    if length is not None:
        return int(length)
    raise ValueError("Condition must expose `seq` or `length` for rollout.")


def _detach_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, tuple):
        return tuple(_detach_tree(item) for item in value)
    if isinstance(value, list):
        return [_detach_tree(item) for item in value]
    if isinstance(value, dict):
        return {key: _detach_tree(item) for key, item in value.items()}
    return value


def _copy_condition_with_latent(condition: Any, z_t: torch.Tensor) -> Any:
    graph = _graph_condition(condition)
    values = dict(getattr(graph, "__dict__", {}))
    values["z_t"] = z_t
    return SimpleNamespace(**values)


class RIDEModelAdapter:
    """Adapter exposing explicit condition encoding and batched noise prediction."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self._stable_cache: Dict[Tuple[Hashable, ...], ConditionEncoding] = {}
        self._current_cache: Dict[Tuple[Hashable, ...], ConditionEncoding] = {}
        self._current_graph_token: Optional[Hashable] = None

    @contextmanager
    def current_graph_cache(self, token: Hashable) -> Iterator[None]:
        """Enable current-policy cache only for one autograd graph lifetime."""

        previous_token = self._current_graph_token
        previous_cache = self._current_cache
        self._current_graph_token = token
        self._current_cache = {}
        try:
            yield
        finally:
            self._current_cache = previous_cache
            self._current_graph_token = previous_token

    def clear_cache(self, policy_role: Optional[str] = None) -> None:
        """Clear cached condition encodings."""

        if policy_role == "current":
            self._current_cache.clear()
        elif policy_role in {"old", "reference"}:
            self._stable_cache = {
                key: value for key, value in self._stable_cache.items() if key[1] != policy_role
            }
        else:
            self._stable_cache.clear()
            self._current_cache.clear()

    def encode_condition(
        self,
        condition: Any,
        policy_role: str,
        policy_version: str,
        *,
        use_cache: bool = True,
    ) -> ConditionEncoding:
        """Encode immutable condition tensors for later noise prediction."""

        key = self._cache_key(condition, policy_role, policy_version)
        cache = self._cache_for_role(policy_role)
        if use_cache and cache is not None and key in cache:
            return cache[key]

        encoding = self._encode_uncached(condition, policy_role, policy_version)
        if cache is not None and use_cache:
            if policy_role in {"old", "reference"}:
                encoding = ConditionEncoding(
                    **{
                        **encoding.__dict__,
                        "encoder_embeddings": _detach_tree(encoding.encoder_embeddings),
                        "edge_embeddings": _detach_tree(encoding.edge_embeddings),
                        "edge_index": _detach_tree(encoding.edge_index),
                        "custom": _detach_tree(encoding.custom),
                    }
                )
            cache[key] = encoding
        return encoding

    def predict_noise(
        self,
        encoding: ConditionEncoding,
        z_t: torch.Tensor,
        noise_level: torch.Tensor,
        *,
        time: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict epsilon for `z_t`, supporting `[L,D]` and `[B,L,D]` latents."""

        squeeze_batch = z_t.dim() == 2
        z_batch = z_t.unsqueeze(0) if squeeze_batch else z_t
        if z_batch.dim() != 3:
            raise ValueError(f"`z_t` must have shape [L,D] or [B,L,D], got {tuple(z_t.shape)}")
        if z_batch.shape[1] != encoding.num_nodes:
            raise ValueError(
                f"`z_t` length {z_batch.shape[1]} does not match condition length {encoding.num_nodes}"
            )

        noise_level = self._normalize_noise_level(noise_level, z_batch.shape[0], z_batch.device)

        if hasattr(self.model, "predict_noise_from_encoding"):
            pred = self.model.predict_noise_from_encoding(encoding, z_batch, noise_level, time=time)
        elif encoding.encoder_embeddings is not None and hasattr(self.model, "_prepare_decoder_hidden"):
            pred = self._predict_from_gvpdiff_encoding(encoding, z_batch, noise_level)
        else:
            pred = self._predict_with_immutable_forward(encoding, z_batch, noise_level, time=time)

        if pred.dim() == 2:
            pred = pred.unsqueeze(0)
        if pred.shape != z_batch.shape:
            raise ValueError(f"Predicted noise shape {tuple(pred.shape)} != z_t shape {tuple(z_batch.shape)}")
        return pred.squeeze(0) if squeeze_batch else pred

    def _cache_key(self, condition: Any, policy_role: str, policy_version: str) -> Tuple[Hashable, ...]:
        key = (
            _condition_id(condition),
            policy_role,
            str(policy_version),
            _tensor_dtype(condition),
            _tensor_device(condition),
        )
        if policy_role == "current":
            return (*key, self._current_graph_token)
        return key

    def _cache_for_role(self, policy_role: str) -> Optional[Dict[Tuple[Hashable, ...], ConditionEncoding]]:
        if policy_role in {"old", "reference"}:
            return self._stable_cache
        if policy_role == "current" and self._current_graph_token is not None:
            return self._current_cache
        return None

    def _encode_uncached(self, condition: Any, policy_role: str, policy_version: str) -> ConditionEncoding:
        graph = _graph_condition(condition)
        condition_id = _condition_id(condition)
        if hasattr(self.model, "encode_condition"):
            custom = self.model.encode_condition(graph)
            return ConditionEncoding(
                condition_id=condition_id,
                policy_role=policy_role,
                policy_version=str(policy_version),
                device=_tensor_device(condition),
                dtype=_tensor_dtype(condition),
                num_nodes=_num_nodes(condition),
                condition=graph,
                custom=custom,
            )

        if hasattr(self.model, "_encode_graph"):
            encoder_embeddings, edge_embeddings, edge_index = self.model._encode_graph(graph)
            return ConditionEncoding(
                condition_id=condition_id,
                policy_role=policy_role,
                policy_version=str(policy_version),
                device=_tensor_device(condition),
                dtype=_tensor_dtype(condition),
                num_nodes=_num_nodes(condition),
                encoder_embeddings=encoder_embeddings,
                edge_embeddings=edge_embeddings,
                edge_index=edge_index,
                condition=graph,
            )

        return ConditionEncoding(
            condition_id=condition_id,
            policy_role=policy_role,
            policy_version=str(policy_version),
            device=_tensor_device(condition),
            dtype=_tensor_dtype(condition),
            num_nodes=_num_nodes(condition),
            condition=graph,
        )

    def _predict_from_gvpdiff_encoding(
        self,
        encoding: ConditionEncoding,
        z_batch: torch.Tensor,
        noise_level: torch.Tensor,
    ) -> torch.Tensor:
        n_samples, num_nodes, _ = z_batch.shape
        s_h, v_h = encoding.encoder_embeddings
        h_e = encoding.edge_embeddings
        edge_index = encoding.edge_index

        s_h = s_h.unsqueeze(0).expand(n_samples, -1, -1).reshape(n_samples * num_nodes, -1)
        v_h = v_h.unsqueeze(0).expand(n_samples, -1, -1, -1).reshape(n_samples * num_nodes, v_h.size(1), 3)
        h_v_batch: TensorPair = (s_h, v_h)

        s_e, v_e = h_e
        s_e = s_e.unsqueeze(0).expand(n_samples, -1, -1).reshape(-1, s_e.size(1))
        v_e = v_e.unsqueeze(0).expand(n_samples, -1, -1, -1).reshape(-1, v_e.size(1), 3)
        h_e_batch: TensorPair = (s_e, v_e)

        batch_edge_index = torch.cat([edge_index + i * num_nodes for i in range(n_samples)], dim=1)
        dec_hidden = self.model._prepare_decoder_hidden(
            z_batch,
            noise_level,
            n_samples,
            num_nodes,
            z_batch.device,
        )
        h_v_dec: TensorPair = (h_v_batch[0] + dec_hidden, h_v_batch[1])
        for layer in self.model.decoder_layers:
            h_v_dec = layer(h_v_dec, batch_edge_index, h_e_batch, autoregressive_x=None)

        out = self.model.W_out(h_v_dec)
        if isinstance(out, tuple):
            out = out[0]
        return out.view(n_samples, num_nodes, self.model.out_dim)

    def _predict_with_immutable_forward(
        self,
        encoding: ConditionEncoding,
        z_batch: torch.Tensor,
        noise_level: torch.Tensor,
        *,
        time: Optional[torch.Tensor],
    ) -> torch.Tensor:
        preds = []
        for index in range(z_batch.shape[0]):
            condition_view = _copy_condition_with_latent(encoding.condition, z_batch[index])
            nl = noise_level[index : index + 1]
            pred = self.model(condition_view, time=time, noise_level=nl.unsqueeze(0))
            preds.append(pred.squeeze(0))
        return torch.stack(preds, dim=0)

    @staticmethod
    def _normalize_noise_level(
        noise_level: torch.Tensor,
        n_samples: int,
        device: torch.device,
    ) -> torch.Tensor:
        noise_level = torch.as_tensor(noise_level, device=device)
        if noise_level.dim() == 0:
            noise_level = noise_level.view(1)
        noise_level = noise_level.reshape(-1)
        if noise_level.numel() == 1 and n_samples > 1:
            noise_level = noise_level.expand(n_samples)
        if noise_level.numel() != n_samples:
            raise ValueError(f"noise_level has {noise_level.numel()} values for batch size {n_samples}")
        return noise_level
