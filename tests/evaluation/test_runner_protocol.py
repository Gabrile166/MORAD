from __future__ import annotations

from types import SimpleNamespace

from src.evaluation.protocol import MetricResult
from src.evaluation.runner import EvaluationSample, RiboDiffusionEvaluator


class _FixedRNAfold:
    def __init__(self, dot_bracket: str) -> None:
        self.dot_bracket = dot_bracket

    def fold(self, sequence: str) -> MetricResult:
        return MetricResult.ok(
            "rnafold_mfe",
            0.0,
            details={"dot_bracket": self.dot_bracket, "energy": 0.0},
            implementation="test",
        )


def test_native_secondary_f1_filter_skips_candidate_f1_below_paper_threshold(tmp_path) -> None:
    evaluator = RiboDiffusionEvaluator(
        {
            "schema_version": "ride_ribodiffusion_eval.v1",
            "metrics": {
                "secondary": {"enabled": True, "native_f1_threshold": 0.7},
                "rfam": {"enabled": False},
                "drfold": {"reason": "disabled"},
            },
        },
        repo_root=tmp_path,
        output_dir=tmp_path / "out",
    )
    target = SimpleNamespace(
        target_id="target",
        sequence_native="AUGC",
        reference_c1p_coords=None,
        raw_record={"sec_struct_list": ["(())"], "rfam_list": ["family"], "type_list": ["RNA"]},
        conformer_index=0,
        dataset_index=1,
        source="source",
        split="test",
        length=4,
    )
    samples = [EvaluationSample("candidate", 0, 0.8, 0.0, "test")]

    target_result, candidates = evaluator._evaluate_target(
        target=target,
        sequences=["AUGC"],
        samples=samples,
        rnafold=_FixedRNAfold("...."),
        fold_oracle=None,
        usalign=SimpleNamespace(),
    )

    assert target_result["secondary_structure_eligible"] is False
    assert target_result["metrics"]["native_secondary_structure_f1"]["value"] == 0.0
    assert candidates[0]["metrics"]["secondary_structure_f1"]["status"] == "skipped"
    assert candidates[0]["metrics"]["secondary_structure_f1"]["details"]["required_minimum"] == 0.7


def test_precomputed_candidates_require_exact_eight_fasta_files(tmp_path) -> None:
    root = tmp_path / "candidates" / "fasta"
    root.mkdir(parents=True)
    for index in range(8):
        (root / f"target_{index}.fasta").write_text(f">sample-{index}\nAUGC\n", encoding="utf-8")
    evaluator = RiboDiffusionEvaluator(
        {
            "schema_version": "ride_ribodiffusion_eval.v1",
            "sampling": {"n_samples": 8},
            "precomputed": {
                "directory": str(tmp_path / "candidates"),
                "pattern": "fasta/{target_id}_*.fasta",
                "model_name": "RiboDiffusion",
            },
            "metrics": {},
        },
        repo_root=tmp_path,
        output_dir=tmp_path / "out",
    )

    sequences, samples = evaluator._load_precomputed_candidates(SimpleNamespace(target_id="target", length=4))

    assert sequences == ["AUGC"] * 8
    assert [sample.generator for sample in samples] == ["RiboDiffusion"] * 8
