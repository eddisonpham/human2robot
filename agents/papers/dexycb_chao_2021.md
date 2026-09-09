# DexYCB — A Benchmark for Capturing Hand Grasping of Objects

**Citation:** Chao, Y.-W., Yang, W., Xiang, Y., Molchanov, P., Handa, A.,
Tremblay, J., Narang, Y. S., Van Wyk, K., Iqbal, U., Birchfield, S.,
Kautz, J., Fox, D. "DexYCB: A Benchmark for Capturing Hand Grasping of
Objects." CVPR 2021.

**Link:** https://openaccess.thecvf.com/content/CVPR2021/html/Chao_DexYCB_A_Benchmark_for_Capturing_Hand_Grasping_of_Objects_CVPR_2021_paper.html

**Related repo:** `references/dex-ycb-toolkit` (`NVlabs/dex-ycb-toolkit`).

## Summary

DexYCB captures human hands grasping YCB objects across multiple
viewpoints, with accurate 3D hand pose (MANO fits) and object pose
annotations, and explicitly includes a robotics-oriented human-to-robot
grasping task in its design intent. This makes it our **primary** dataset
source (`08_DATA_AND_RETARGETING_PIPELINE.md` §1): single-hand grasping
of graspable, YCB-catalog objects is the closest existing dataset match
to our floating-Allegro-hand cube/cylinder/sphere pickup task.

## What we borrow

Ground-truth 3D hand and object trajectories, used directly (not via
HaMeR — see `hamer_pavlakos_2024.md`) as input to
`08_DATA_AND_RETARGETING_PIPELINE.md`'s smoothing/differentiation and
retargeting steps. `dex-retargeting`'s own tutorial uses DexYCB as its
worked example, which is exactly our Phase 4 use case
(`anyteleop_dexretargeting_qin_2023.md`).

## License note (important, see also `13_REPRODUCIBILITY_AND_CONVENTIONS.md` §6)

The dataset is **CC BY-NC 4.0 (non-commercial)**. Fine for this portfolio
project; any hypothetical commercial productionization would need a
different demonstration data source (e.g. first-party teleoperation
capture).
