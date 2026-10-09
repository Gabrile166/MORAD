# Reported results

The values reported in the paper, exactly as printed. The same tables can be
regenerated from a fresh evaluation with `scripts/eval/collect_tables.py`; see
[docs/REPRODUCE.md](../docs/REPRODUCE.md).

| File | Paper | Protocol |
|---|---|---|
| [`table1_tertiary.csv`](table1_tertiary.csv) | Table 1 | candidate 0 of each of the 153 test targets; C1′ TM over 126 valid targets |
| [`table2_pairing.csv`](table2_pairing.csv) | Table 2 | candidate 0; EternaFold pairs compared with base pairs extracted from the native 3D structures |
| [`table3_thermodynamics.csv`](table3_thermodynamics.csv) | Table 3 | 153 test targets, ViennaRNA |
| [`table4_reward_comparison.csv`](table4_reward_comparison.csv) | Table 4 | two policies trained identically except for the reward, each at the last checkpoint of its run |
| [`checkpoints.csv`](checkpoints.csv) | Checkpoint comparison (appendix) | mean over eight candidates per target, with standard errors (`*_se`) |
| [`diversity.csv`](diversity.csv) | Sequence diversity (appendix) | 153 targets, 1,224 candidates (eight per target) |
| [`training_curve.csv`](training_curve.csv) | Figure 3 and the training trajectories (appendix) | 40 held-out evaluations, updates 0–390, one sampled candidate per target |

MORAD denotes the policy after update 390 (`morad_update390.pt`).

## Columns

| Column | Meaning | Direction |
|---|---|---|
| `gdt_ts`, `tm_c4`, `rmsd` | GDT-TS, TM-score and RMSD (Å) of the RhoFold+ prediction against the native structure, C4′ atoms | ↑ ↑ ↓ |
| `tm_c1` | TM-score on C1′ atoms (US-align) | ↑ |
| `success_rate` | fraction of targets meeting the structural success criterion | ↑ |
| `recovery` | native sequence recovery | ↑ |
| `lddt_c4`, `plddt`, `coarse_clash` | C4′ lDDT, RhoFold+ confidence (0–1), coarse-grained steric clash | ↑ ↑ ↓ |
| `precision`, `recall`, `f1`, `mcc` | base-pair agreement with the native pairs | ↑ |
| `mfe`, `efe` | minimum and ensemble free energy (kcal/mol) | context |
| `ned` | normalized ensemble defect | ↓ |
| `pe` | positional entropy | ↓ |
| `mfe_frequency` | probability of the minimum-free-energy structure | ↑ |
| `hamming_diversity`, `kmer3_diversity` | mean pairwise Hamming and 3-mer distance among the eight candidates of a target | context |

`training_curve.csv` holds, for each evaluation `step`: `gdt`, `tm` (C4′),
`rmsd`, `c1tm`, `success`, `recovery` and within-target `diversity`.
