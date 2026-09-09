# AnyTeleop / dex-retargeting

**Citation:** Qin, Y., Yang, W., Huang, B., Van Wyk, K., Su, H., Wang, X.,
Chao, Y.-W., Fox, D. "AnyTeleop: A General Vision-Based Dexterous
Robot Arm-Hand Teleoperation System." RSS 2023.

**Link:** https://arxiv.org/abs/2307.04577

**Related repo:** `references/dex-retargeting` (`dexsuite/dex-retargeting`),
MIT license — see `03_EXISTING_REPOS_TO_CLONE.md`.

## Summary

AnyTeleop is a general vision-based teleoperation system for dexterous
robot hands; its retargeting subsystem was later factored out into the
standalone, actively-maintained `dex-retargeting` library, which solves
the general problem of mapping human hand keypoints (from any upstream
hand-pose estimator, or from a dataset's ground-truth annotations) onto a
target robot hand's joint configuration, supporting multiple robot hands
through a config-driven interface plus temporal smoothing and (for
manipulation use cases) object-relative alignment.

## What we borrow

We call this library directly — we do not reimplement retargeting. It
solves exactly the optimization problem in `04_MATH_SPEC.md` §2:

$$
q_t^\star = \arg\min_q \lVert X^R(q) - X_t^H\rVert_W^2 + \lambda\lVert q - q_{t-1}^\star\rVert^2, \quad q_{\min}\le q \le q_{\max}
$$

using Pinocchio for forward kinematics/Jacobians internally. The library
ships a tutorial specifically for retargeting from a hand-object pose
dataset (their worked example is DexYCB — directly one of our chosen data
sources, `08_DATA_AND_RETARGETING_PIPELINE.md`), which is the closest
possible match to our Phase 4 task.

## Implementation notes for this project

- Robot joint order in `dex-retargeting`'s output does not necessarily
  match the MuJoCo Menagerie MJCF's joint order — always remap by joint
  **name**, never by positional index (the library's own documentation
  warns about this).
- Confirm at Phase 0 whether `dex-urdf` (the companion asset repo) has a
  URDF for our specific floating-base-modified Allegro hand
  (`06_ROBOT_AND_SIMULATION_SPEC.md`); if not, retarget the 16 finger DoF
  through the library and handle the 6 wrist DoF as a direct rigid
  transform rather than forcing an unsupported 22-DoF config through the
  IK solver (`08_DATA_AND_RETARGETING_PIPELINE.md` §3).
