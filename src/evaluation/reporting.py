"""Write machine-readable and human-readable evaluation artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def write_evaluation_artifacts(result: Mapping[str, Any], output_dir: str | Path) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "evaluation.json", result)
    _write_json(root / "summary.json", result["summary"])
    _write_json(root / "manifest.json", result["manifest"])
    _write_jsonl(root / "targets.jsonl", result.get("targets", []))
    _write_jsonl(root / "candidates.jsonl", result.get("candidates", []))
    _write_metrics_jsonl(root / "metrics.jsonl", result)
    _write_summary_csv(root / "summary.csv", result["summary"])
    (root / "REPORT.md").write_text(_markdown_report(result), encoding="utf-8")
    (root / "SKIPPED_METRICS.md").write_text(_skipped_report(result), encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def _write_metrics_jsonl(path: Path, result: Mapping[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for target in result.get("targets", []):
        for name, metric in target.get("metrics", {}).items():
            rows.append({"scope": "target", "target_id": target["target_id"], "metric": name, **metric})
    for candidate in result.get("candidates", []):
        for name, metric in candidate.get("metrics", {}).items():
            rows.append(
                {
                    "scope": "candidate",
                    "target_id": candidate["target_id"],
                    "candidate_id": candidate["candidate_id"],
                    "metric": name,
                    **metric,
                }
            )
    _write_jsonl(path, rows)


def _write_summary_csv(path: Path, summary: Mapping[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for metric, values in summary.get("candidate_metrics", {}).items():
        rows.append(_summary_row("candidate", "all", metric, values))
    for metric, values in summary.get("paper_single_sample_candidate_metrics", {}).items():
        rows.append(_summary_row("candidate", "paper_single_sample", metric, values))
    for metric, values in summary.get("target_metrics", {}).items():
        rows.append(_summary_row("target", "all", metric, values))
    for bucket_key in ("by_length_bucket", "by_rna_type", "by_rna_family"):
        for bucket, metrics in summary.get(bucket_key, {}).items():
            for metric, values in metrics.items():
                rows.append(_summary_row(bucket_key.removeprefix("by_"), bucket, metric, values))
    fields = ["scope", "group", "metric", "count", "ok", "skipped", "error", "mean", "median", "min", "max"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _summary_row(scope: str, group: str, metric: str, values: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "scope": scope,
        "group": group,
        "metric": metric,
        **{key: values.get(key) for key in ("count", "ok", "skipped", "error", "mean", "median", "min", "max")},
    }


def _markdown_report(result: Mapping[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        "# RIDE RiboDiffusion-style Evaluation",
        "",
        f"- Protocol: `{result['protocol_version']}`",
        f"- Generator: `{result.get('generator', 'unknown')}`",
        f"- Targets: {summary['target_count']}",
        f"- Candidates: {summary['candidate_count']}",
        f"- Comparability: {result['comparability']}",
        "",
        "## Paper-style single-sample metrics",
        "",
        "| Metric | OK/Total | Mean | Median | Status |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    metrics = summary.get("paper_single_sample_candidate_metrics", {})
    for name, values in metrics.items():
        state = "ok" if values.get("ok") else "skipped/error"
        lines.append(
            f"| `{name}` | {values.get('ok', 0)}/{values.get('count', 0)} | "
            f"{_format(values.get('mean'))} | {_format(values.get('median'))} | {state} |"
        )
    lines.extend(
        [
            "",
            "## Eight-sample design metrics",
            "",
            "| Metric | OK/Total | Mean | Median | Status |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    design_metrics = {**summary.get("candidate_metrics", {}), **summary.get("target_metrics", {})}
    for name, values in design_metrics.items():
        state = "ok" if values.get("ok") else "skipped/error"
        lines.append(
            f"| `{name}` | {values.get('ok', 0)}/{values.get('count', 0)} | "
            f"{_format(values.get('mean'))} | {_format(values.get('median'))} | {state} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundaries",
            "",
            *[f"- {item}" for item in result.get("manifest", {}).get("protocol_deviations", [])],
            "",
            "See `SKIPPED_METRICS.md` for unavailable tools and per-metric reasons.",
            "",
        ]
    )
    return "\n".join(lines)


def _skipped_report(result: Mapping[str, Any]) -> str:
    counts: dict[tuple[str, str, str], int] = {}
    for scope in ("targets", "candidates"):
        for item in result.get(scope, []):
            for name, metric in item.get("metrics", {}).items():
                if metric.get("status") == "ok":
                    continue
                key = (name, str(metric.get("status", "unknown")), str(metric.get("reason") or "unspecified"))
                counts[key] = counts.get(key, 0) + 1
    lines = ["# Skipped or Failed Metrics", ""]
    if not counts:
        lines.append("All configured metrics completed successfully.")
    else:
        lines.extend(["| Metric | Status | Count | Reason |", "| --- | --- | ---: | --- |"])
        for (name, status, reason), count in sorted(counts.items()):
            lines.append(f"| `{name}` | {status} | {count} | {reason} |")
    lines.extend(
        [
            "",
            "A skipped metric is never converted to numeric zero and is excluded from means/medians.",
            "",
        ]
    )
    return "\n".join(lines)


def _format(value: Any) -> str:
    return "—" if value is None else f"{float(value):.6f}"
