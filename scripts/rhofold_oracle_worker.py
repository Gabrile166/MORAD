#!/usr/bin/env python
"""Persistent JSONL worker for official RhoFold+ single-sequence inference."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
RHOFOLD_PROTOCOL_ROOT = REPO_ROOT / "third_party" / "rhofold_protocol"
RHOFOLD_PACKAGE_ROOT = RHOFOLD_PROTOCOL_ROOT / "rhofold"
DEFAULT_CKPT = RHOFOLD_PROTOCOL_ROOT / "checkpoints" / "rhofold_pretrained_params.pt"
VALID_BASES = frozenset("AUGC")


def main() -> None:
    args = parse_args()
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    runner = FakeRunner() if args.fake else RhoFoldPlusRunner(args)
    for line in sys.stdin:
        if not line.strip():
            continue
        response = handle_request(runner, line)
        print(json.dumps(response, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--relax-steps", type=int, default=0)
    parser.add_argument("--fake", action="store_true", help="Use a deterministic fake backend for unit tests.")
    return parser.parse_args()


def handle_request(runner: "BaseRunner", line: str) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        request = json.loads(line)
        request_id = str(request["request_id"])
        sequence = normalize_sequence(str(request["sequence"]))
        output_dir = Path(str(request["output_dir"]))
        output_dir.mkdir(parents=True, exist_ok=True)
        result = runner.predict(sequence, output_dir, request_id)
        result.update(
            {
                "request_id": request_id,
                "status": "ok",
                "latency_ms": (time.perf_counter() - started) * 1000.0,
                "error_message": None,
            }
        )
        return result
    except ValueError as exc:
        return failure_response(line, "invalid_sequence", started, str(exc))
    except TimeoutError as exc:
        return failure_response(line, "timeout", started, str(exc))
    except Exception as exc:  # pragma: no cover - exercised by real integration failures
        logging.exception("RhoFold+ worker request failed")
        return failure_response(line, "fold_failed", started, str(exc))


def failure_response(line: str, status: str, started: float, message: str) -> dict[str, Any]:
    request_id = "unknown"
    with contextlib.suppress(Exception):
        request_id = str(json.loads(line).get("request_id", "unknown"))
    return {
        "request_id": request_id,
        "status": status,
        "predicted_structure_path": None,
        "predicted_structure_hash": None,
        "plddt": None,
        "latency_ms": (time.perf_counter() - started) * 1000.0,
        "error_message": message,
    }


class BaseRunner:
    def predict(self, sequence: str, output_dir: Path, request_id: str) -> dict[str, Any]:
        raise NotImplementedError


class FakeRunner(BaseRunner):
    def predict(self, sequence: str, output_dir: Path, request_id: str) -> dict[str, Any]:
        pdb_path = output_dir / f"{request_id}.pdb"
        base_y = {"A": 0.0, "U": 0.2, "G": -0.15, "C": 0.35}
        with pdb_path.open("w", encoding="utf-8") as handle:
            handle.write("REMARK fake RhoFold+ prediction\n")
            for idx, base in enumerate(sequence, start=1):
                y = 0.0 if len(sequence) <= 4 else base_y[base]
                handle.write(
                    f"ATOM  {idx:5d}  C4'   A A{idx:4d}    {idx:8.3f}{y:8.3f}{0.0:8.3f}  1.00 77.00           C\n"
                )
            handle.write("END\n")
        return {
            "predicted_structure_path": str(pdb_path),
            "predicted_structure_hash": file_sha256(pdb_path),
            "plddt": 0.77,
        }


class RhoFoldPlusRunner(BaseRunner):
    def __init__(self, args: argparse.Namespace) -> None:
        if not RHOFOLD_PACKAGE_ROOT.exists():
            raise FileNotFoundError(f"RhoFold+ package not found: {RHOFOLD_PACKAGE_ROOT}")
        sys.path.insert(0, str(RHOFOLD_PACKAGE_ROOT))
        from rhofold.config import rhofold_config
        from rhofold.rhofold import RhoFold
        from rhofold.utils import get_device, save_ss2ct
        from rhofold.utils.alphabet import get_features

        self._get_features = get_features
        self._save_ss2ct = save_ss2ct
        self.device = get_device(args.device)
        self.relax_steps = args.relax_steps
        self.ckpt = Path(args.ckpt)
        if not self.ckpt.exists():
            raise FileNotFoundError(f"RhoFold+ checkpoint not found: {self.ckpt}")

        logging.info("Constructing RhoFold+")
        self.model = RhoFold(rhofold_config)
        logging.info("Loading RhoFold+ checkpoint %s", self.ckpt)
        state = torch.load(self.ckpt, map_location=torch.device("cpu"))
        self.model.load_state_dict(state["model"])
        self.model.eval().to(self.device)
        self._relax_cls = None
        if self.relax_steps > 0:
            from rhofold.relax.relax import AmberRelaxation

            self._relax_cls = AmberRelaxation

    @torch.no_grad()
    def predict(self, sequence: str, output_dir: Path, request_id: str) -> dict[str, Any]:
        fasta_path = output_dir / f"{request_id}.fasta"
        fasta_path.write_text(f">{request_id}\n{sequence}\n", encoding="utf-8")
        data_dict = self._get_features(str(fasta_path), str(fasta_path))
        outputs = self.model(
            tokens=data_dict["tokens"].to(self.device),
            rna_fm_tokens=data_dict["rna_fm_tokens"].to(self.device),
            seq=data_dict["seq"],
        )
        output = outputs[-1]

        ss_prob_map = torch.sigmoid(output["ss"][0, 0]).data.cpu().numpy()
        self._save_ss2ct(ss_prob_map, data_dict["seq"], str(output_dir / "ss.ct"), threshold=0.5)
        plddt = output["plddt"][0].data.cpu().numpy()
        np.savez_compressed(
            output_dir / "results.npz",
            dist_n=torch.softmax(output["n"].squeeze(0), dim=0).data.cpu().numpy(),
            dist_p=torch.softmax(output["p"].squeeze(0), dim=0).data.cpu().numpy(),
            dist_c=torch.softmax(output["c4_"].squeeze(0), dim=0).data.cpu().numpy(),
            ss_prob_map=ss_prob_map,
            plddt=plddt,
        )

        pdb_path = output_dir / "unrelaxed_model.pdb"
        node_coords_pred = output["cord_tns_pred"][-1].squeeze(0)
        self.model.structure_module.converter.export_pdb_file(
            data_dict["seq"],
            node_coords_pred.data.cpu().numpy(),
            path=str(pdb_path),
            chain_id=None,
            confidence=plddt,
            logger=logging.getLogger("RhoFold+ JSONL Worker"),
        )

        if self._relax_cls is not None and self.relax_steps > 0:
            relaxed_path = output_dir / f"relaxed_{self.relax_steps}_model.pdb"
            relax = self._relax_cls(max_iterations=self.relax_steps, logger=logging.getLogger("RhoFold+ JSONL Worker"), use_gpu=self.device != "cpu")
            relax.process(str(pdb_path), str(relaxed_path))
            pdb_path = relaxed_path

        with contextlib.suppress(OSError):
            fasta_path.unlink()
        return {
            "predicted_structure_path": str(pdb_path),
            "predicted_structure_hash": file_sha256(pdb_path),
            "plddt": float(np.mean(plddt)),
        }


def normalize_sequence(sequence: str) -> str:
    normalized = sequence.upper().replace("T", "U")
    if not normalized:
        raise ValueError("sequence is empty")
    invalid = sorted(set(normalized) - VALID_BASES)
    if invalid:
        raise ValueError(f"invalid RNA bases: {''.join(invalid)}")
    return normalized


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
