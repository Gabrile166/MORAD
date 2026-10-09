#!/usr/bin/env python
"""Check the local evaluation setup: RIDE checkpoint, RhoFold+, US-align, RNAfold and the target pool."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

from src.evaluation.config import load_evaluation_config
from src.rl.config import load_config
from src.rl.observability import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eval/test153.yaml")
    parser.add_argument("--json-output", default="")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    config_path = _resolve(repo_root, args.config)
    report = verify_setup(load_evaluation_config(config_path), repo_root=repo_root)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json_output:
        output_path = _resolve(repo_root, args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    if report["status"] != "ready":
        raise SystemExit(1)


def verify_setup(config: Mapping[str, Any], *, repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    checks: list[dict[str, Any]] = []
    ride_cfg_path = _resolve(root, config["ride"]["rl_config"])
    checks.append(_path_check("ride_config", ride_cfg_path))
    ride_config = load_config(ride_cfg_path)

    pool_value = config["ride"].get("pool_manifest") or ride_config.data.pool_manifest
    pool_path = _resolve(root, pool_value)
    ride_checkpoint = _resolve(root, ride_config.data.ride_checkpoint)
    checks.append(_path_check("target_pool", pool_path, include_hash=True))
    checks.append(_path_check("ride_checkpoint", ride_checkpoint, include_hash=True))

    ride_device = str(config.get("ride", {}).get("device", "auto"))
    cuda_required = ride_device.startswith("cuda")
    checks.append(
        {
            "name": "ride_cuda",
            "status": "ok" if torch.cuda.is_available() else ("error" if cuda_required else "skipped"),
            "details": {
                "requested_device": ride_device,
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_runtime": torch.version.cuda,
                "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            },
        }
    )

    metrics = config.get("metrics", {})
    secondary = metrics.get("secondary", {})
    if secondary.get("enabled", True):
        rnafold = _resolve_executable(root, secondary.get("rnafold_binary", "RNAfold"))
        checks.append(_command_check("RNAfold", [str(rnafold), "--version"]))
    else:
        checks.append(_skipped("RNAfold", "secondary metric disabled"))

    tertiary = metrics.get("tertiary", {})
    if tertiary.get("enabled", True):
        rhofold_python = _resolve_executable(root, tertiary["rhofold_python"])
        worker = _resolve(root, tertiary["worker_script"])
        checkpoint = _resolve(root, tertiary["checkpoint"])
        usalign = _resolve_executable(root, tertiary.get("usalign_binary", "USalign"))
        checks.append(_path_check("rhofold_worker", worker))
        checks.append(_path_check("rhofold_checkpoint", checkpoint, include_hash=True))
        checks.append(
            _command_check(
                "rhofold_python",
                [
                    str(rhofold_python),
                    "-c",
                    "import json,torch; print(json.dumps({'python':__import__('sys').version.split()[0],'torch':torch.__version__,'cuda_available':torch.cuda.is_available(),'cuda':torch.version.cuda}))",
                ],
            )
        )
        checks.append(_command_check("USalign", [str(usalign), "-v"], acceptable_returncodes={0, 1}))
    else:
        checks.append(_skipped("RhoFold+", "tertiary metric disabled"))
        checks.append(_skipped("USalign", "tertiary metric disabled"))

    rfam = metrics.get("rfam", {})
    rfam_database = str(rfam.get("cm_database", "")).strip()
    rfam_directory = str(rfam.get("cm_directory", "")).strip()
    if rfam.get("enabled", True) and (rfam_database or rfam_directory):
        cmsearch = _resolve_executable(root, rfam.get("cmsearch_binary", "cmsearch"))
        checks.append(_command_check("Infernal cmsearch", [str(cmsearch), "-h"]))
        if rfam_database:
            checks.append(_path_check("Rfam covariance-model database", _resolve(root, rfam_database)))
        else:
            directory = _resolve(root, rfam_directory)
            cm_files = sorted(directory.glob("*.cm")) if directory.is_dir() else []
            checks.append(
                {
                    "name": "Rfam covariance-model directory",
                    "status": "ok" if cm_files else "error",
                    "reason": None if cm_files else "no .cm files found",
                    "details": {"path": str(directory), "model_count": len(cm_files)},
                }
            )
    elif rfam.get("enabled", True):
        checks.append(_skipped("Infernal/Rfam", "RFAM_CM_DATABASE is not configured"))
    else:
        checks.append(_skipped("Infernal/Rfam", "Rfam metric disabled"))

    drfold = metrics.get("drfold", {})
    checks.append(
        _skipped("DRFold", str(drfold.get("reason", "disabled")))
        if not drfold.get("enabled", False)
        else {"name": "DRFold", "status": "error", "reason": "enabled but no evaluator adapter is implemented"}
    )

    for name in ("RiboDiffusion", "rhofold_protocol", "USalign"):
        checks.append(_git_check(name, root.parent / "third_party" / name))

    failures = [check for check in checks if check["status"] == "error"]
    return {
        "schema_version": "ride_evaluation_setup_check.v1",
        "status": "ready" if not failures else "not_ready",
        "repo_root": str(root),
        "checks": checks,
        "required_failures": len(failures),
        "optional_skips": sum(check["status"] == "skipped" for check in checks),
    }


def _resolve(root: Path, value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def _resolve_executable(root: Path, value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute() or path.parent != Path("."):
        return _resolve(root, path)
    located = shutil.which(str(path))
    return Path(located) if located else path


def _path_check(name: str, path: Path, *, include_hash: bool = False) -> dict[str, Any]:
    exists = path.is_file()
    details: dict[str, Any] = {"path": str(path), "exists": exists}
    if exists and include_hash:
        details["sha256"] = sha256_file(path)
    return {
        "name": name,
        "status": "ok" if exists else "error",
        "reason": None if exists else "file not found",
        "details": details,
    }


def _command_check(
    name: str,
    command: list[str],
    *,
    acceptable_returncodes: set[int] | None = None,
) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=30.0)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"name": name, "status": "error", "reason": str(exc), "details": {"command": command}}
    output = (completed.stdout + "\n" + completed.stderr).strip()
    accepted = acceptable_returncodes or {0}
    return {
        "name": name,
        "status": "ok" if completed.returncode in accepted else "error",
        "reason": None if completed.returncode in accepted else f"command exited {completed.returncode}",
        "details": {"command": command, "output": output[-2000:]},
    }


def _git_check(name: str, path: Path) -> dict[str, Any]:
    if not (path / ".git").exists():
        return {"name": f"reference:{name}", "status": "error", "reason": "git repository not found", "details": {"path": str(path)}}
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10.0,
    )
    return {
        "name": f"reference:{name}",
        "status": "ok" if completed.returncode == 0 else "error",
        "reason": None if completed.returncode == 0 else completed.stderr.strip(),
        "details": {"path": str(path), "commit": completed.stdout.strip() or None},
    }


def _skipped(name: str, reason: str) -> dict[str, Any]:
    return {"name": name, "status": "skipped", "reason": reason, "details": {}}


if __name__ == "__main__":
    main()
