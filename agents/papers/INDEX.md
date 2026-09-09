# Papers Index

Structured, pre-digested notes on every paper this project implements or
directly cites. Each file follows the same template (Citation → Link →
One-paragraph summary → What we borrow → Key equations → Related repo)
specifically so a coding agent can parse them quickly without re-reading
the full paper. These notes are a planning aid, not a substitute for
reading the source when precision matters (exact hyperparameters,
appendix details) — the note tells you when that's necessary.

## RL algorithm core

- `sac_haarnoja_2018.md` — the base off-policy algorithm (`05_RL_ALGORITHM_SPEC.md` §1)
- `dapg_rajeswaran_2018.md` — why demonstrations help; **not** the literal algorithm implemented (`05_RL_ALGORITHM_SPEC.md` §2, `12_NON_GOALS_AND_CUT_SCOPE.md` §7)
- `demos_sparse_reward_vecerik_2017.md` — the algorithm we actually implement for demo integration (`05_RL_ALGORITHM_SPEC.md` §2)
- `mbpo_janner_2019.md` — the algorithm we implement for dynamics-model rollout augmentation (`05_RL_ALGORITHM_SPEC.md` §3)
- `residual_rl_johannink_2019.md` — the precise framing of "physics + residual" (`05_RL_ALGORITHM_SPEC.md` §3, condition D)

## Pipeline / systems

- `dexmv_qin_2022.md` — closest existing full-pipeline system; architecture reference only (`03_EXISTING_REPOS_TO_CLONE.md` Tier 0)
- `anyteleop_dexretargeting_qin_2023.md` — the retargeting library we call directly (`04_MATH_SPEC.md` §2, `08_DATA_AND_RETARGETING_PIPELINE.md` §3)
- `mujoco_todorov_2012.md` — the simulator and its dynamics API (`04_MATH_SPEC.md` §3)
- `hamer_pavlakos_2024.md` — offline hand-pose estimator for the stretch "own video" path (`08_DATA_AND_RETARGETING_PIPELINE.md` §1)

## Datasets

- `dexycb_chao_2021.md`
- `arctic_fan_2023.md`
- `grab_taheri_2020.md`

## Explicitly not implemented (kept for context, see `12_NON_GOALS_AND_CUT_SCOPE.md`)

The original brain-dump also cited Doya 2000 (continuous-time HJB RL),
Greydanus et al. 2019 (Hamiltonian NNs), the port-Hamiltonian TRO 2024
paper, Degrave et al. 2016 and Zhu et al. 2023 (differentiable physics),
and Chua et al. 2018 (PETS). No note files are created for these — they
are non-goals, not implementation targets. If one of them is ever
revisited per `12_NON_GOALS_AND_CUT_SCOPE.md` §1's reopening condition,
write its note file at that time.
