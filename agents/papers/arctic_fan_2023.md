# ARCTIC — A Dataset for Dexterous Bimanual Hand-Object Manipulation

**Citation:** Fan, Z., Taheri, O., Tzionas, D., Kocabas, M., Kaufmann, M.,
Black, M. J., Hilliges, O. "ARCTIC: A Dataset for Dexterous Bimanual
Hand-Object Manipulation." CVPR 2023.

**Link:** https://openaccess.thecvf.com/content/CVPR2023/html/Fan_ARCTIC_A_Dataset_for_Dexterous_Bimanual_Hand-Object_Manipulation_CVPR_2023_paper.html

**Related repo:** `references/arctic` (`zc-alexfan/arctic`, official repo,
verified current).

## Summary

ARCTIC provides 2.1M frames of dexterous **bimanual** hand-object
manipulation with dense 3D hand/object meshes and dynamic contact
information — richer contact annotation than DexYCB, at the cost of being
bimanual (we only retarget one hand's trajectory for our single-hand
Allegro task).

## What we borrow

Used as a **secondary** data source (`08_DATA_AND_RETARGETING_PIPELINE.md`
§1) for trajectory diversity and, specifically, its per-frame contact
labels — useful for validating our own MuJoCo touch-sensor-derived contact
features (`06_ROBOT_AND_SIMULATION_SPEC.md` §5) against a richer
ground-truth signal during Phase 4's visualization/sanity-checking step.
When extracting a single-hand trajectory for our task, pick whichever hand
(left/right) is doing the primary manipulation in a given clip; discard
the other hand's trajectory rather than trying to retarget bimanually
(bimanual manipulation is out of this project's scope).

## License note

Check `references/arctic/LICENSE` directly before any use beyond this
portfolio project — datasets of this class typically carry research-only
terms (see `13_REPRODUCIBILITY_AND_CONVENTIONS.md` §6).
