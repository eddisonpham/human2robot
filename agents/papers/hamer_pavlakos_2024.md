# HaMeR — Reconstructing Hands in 3D with Transformers

**Citation:** Pavlakos, G., Shan, D., Radosavovic, I., Kanazawa, A.,
Fouhey, D., Malik, J. "Reconstructing Hands in 3D with Transformers."
CVPR 2024.

**Link:** https://arxiv.org/abs/2312.05251

**Related repo:** `references/hamer` (`geopavlakos/hamer`) — verified
actively maintained (updates as recent as early 2026).

## Summary

HaMeR is a transformer-based monocular 3D hand pose/mesh estimator: given
a single RGB image (or video frame), it predicts MANO hand parameters
(pose $\theta$, shape $\beta$, global transform $T$), which can be
converted to a hand mesh and 3D joint coordinates. It's reported to
outperform prior baselines on standard 3D hand-pose benchmarks and
includes evaluation support for Ego4D-style egocentric video, relevant if
this project's stretch "own video" demo uses a phone/webcam clip.

## What we borrow, and how narrowly we use it

Only the offline inference path, for exactly one purpose: the stretch-goal
"record your own hand video, retarget it, replay it in sim" demo
(`01_MISSION_AND_SCOPE.md` §3, `08_DATA_AND_RETARGETING_PIPELINE.md` §1).
For the MVP and the core ablation study, we use dataset-provided 3D
annotations (DexYCB/ARCTIC/GRAB) directly and never invoke HaMeR — running
a vision model to re-estimate poses that are already available as ground
truth would be strictly worse (noisier) and slower, and would put a large
transformer model in a place where the hardware constraints
(`02_TECH_STACK.md`) specifically say not to.

## Pipeline (only used for the stretch demo)

$$
I_t \xrightarrow{\text{HaMeR}} (\theta_t,\beta_t,T_t) \xrightarrow{\text{MANO}} M_t \rightarrow X_t
$$

producing 3D hand joint coordinates $X_t$ in exactly the format the
retargeting step (`anyteleop_dexretargeting_qin_2023.md`) expects,
identical to the dataset path.

## Environment isolation (important)

HaMeR depends on `detectron2`/`ViTPose`/a specific CUDA-matched `torch`
build that is likely to conflict with the main training environment's
`torch` 2.7+/cu128 pin (`02_TECH_STACK.md`). **Run it in its own isolated
virtual environment (`envs/hamer/`), invoked as a one-off offline script
that writes cached NPZ output** — it should never be imported into, or
share a process with, `python/rl/` or `python/envs/`.
