# GRAB — A Dataset of Whole-Body Human Grasping of Objects

**Citation:** Taheri, O., Ghorbani, N., Black, M. J., Tzionas, D. "GRAB:
A Dataset of Whole-Body Human Grasping of Objects." ECCV 2020.

**Link:** (ECCV 2020 proceedings)

**Related repo:** `references/grab` (`otaheri/GRAB`).

## Summary

GRAB captures whole-body human motion while grasping and manipulating a
variety of objects, with MANO hand fits and object pose/contact
annotations — a source of trajectory diversity beyond DexYCB's more
constrained tabletop grasping setups.

## What we borrow

A **tertiary** data source (`08_DATA_AND_RETARGETING_PIPELINE.md` §1),
used mainly to broaden the variety of grasp approach trajectories fed into
retargeting, once the DexYCB-based pipeline (primary) and ARCTIC-based
contact validation (secondary) are working. Extract only the hand-relevant
portion of each whole-body trajectory (wrist pose + finger pose); discard
full-body pose, which is irrelevant to our floating-hand-only robot model
(`06_ROBOT_AND_SIMULATION_SPEC.md` §1).

## License note

Check `references/grab/LICENSE` directly before any use beyond this
portfolio project (`13_REPRODUCIBILITY_AND_CONVENTIONS.md` §6).
