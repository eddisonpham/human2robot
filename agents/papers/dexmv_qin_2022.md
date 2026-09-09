# DexMV — Imitation Learning for Dexterous Manipulation from Human Videos

**Citation:** Qin, Y., Wu, Y.-H., Liu, S., Jiang, H., Yang, R., Fu, Y.,
Wang, X. "DexMV: Imitation Learning for Dexterous Manipulation from Human
Videos." ECCV 2022.

**Link:** https://arxiv.org/abs/2108.05877

**Related repos:** `references/dexmv-sim` (`yzqin/dexmv-sim`),
`references/dexmv-learn` (`yzqin/dexmv-learn`) — see
`03_EXISTING_REPOS_TO_CLONE.md`, Tier 0.

## Summary

DexMV builds an almost identical pipeline to this project's original
brief: human demonstration video → 3D hand/object pose estimation →
retargeting the human motion onto a simulated multi-fingered robot hand
(Adroit) → imitation/reinforcement learning (including DAPG among the
algorithms studied) on the resulting demonstrations, evaluated on
relocation, hammering, and other dexterous manipulation tasks in MuJoCo.
It is, as far as we verified, the closest existing published system to
this project's overall pipeline shape.

## Why we read it but don't adopt it wholesale

Two deltas make direct reuse the wrong call, even though the architecture
is directly relevant:

1. **Robot choice.** DexMV retargets to the Adroit hand. This project
   targets a MuJoCo-Menagerie Allegro hand (`06_ROBOT_AND_SIMULATION_SPEC.md`
   §1) for a more modern, actively-maintained asset with better contact
   parameters; DexMV's Adroit-specific retargeting script doesn't
   transfer directly.
2. **Algorithm choice.** DexMV's learning stack is built on `mjrl` and
   (among other options) DAPG's on-policy NPG — this project standardizes
   on SAC and its correct off-policy demo/dynamics-model extensions
   (`05_RL_ALGORITHM_SPEC.md`), for the reasons in
   `dapg_rajeswaran_2018.md` and `12_NON_GOALS_AND_CUT_SCOPE.md` §7.

## What we borrow

- The overall pipeline architecture (video → pose → retarget → demo →
  learn) as validation that this project's shape is sound and has
  precedent.
- Ideas for how to structure the "demonstration translation" step
  (their term for retargeting + object-relative alignment) — read their
  implementation for engineering details even though we call
  `dex-retargeting` (a different, more current, multi-robot library)
  rather than their Adroit-specific code.
- A natural point of comparison in the final write-up: "DexMV showed this
  is possible with Adroit + DAPG-family methods; this project asks
  whether the same demonstration-guided idea, combined with a
  physics-structured dynamics model via MBPO, gives further sample-
  efficiency gains on a different, modern hand."
