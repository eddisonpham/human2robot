"""Every published figure must be the one in the artifact it was measured on.

The documentation used to carry a hand-copied second version of numbers that
already existed on disk in `results/trajectory_optimization/*.json`. It drifted:
the kinematic table claimed a 99/100 descent rate while the artifact recorded
100/100, and the split-granularity table quoted error bars for an experiment
that had only ever been run at one seed.

`results/` is gitignored, so these tests work from the committed snapshot in
`docs/published_figures.json` and additionally cross-check it against the live
artifacts whenever they are present. That split matters: a test that only ran
where a local experiment directory happened to exist would pass vacuously on a
fresh clone, which is the failure this file exists to prevent.

The documentation checks are deliberately literal. Each one names a figure and
asserts the exact string that appears in the prose or table, so editing a
published number without regenerating the artifact is a failure rather than a
silent inconsistency.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from human2robot.evaluation import published as P

RESULTS = Path("results/trajectory_optimization")
SNAPSHOT = Path("docs/published_figures.json")
README = Path("README.md")
RESULTS_DOC = Path("docs/RESULTS.md")
LIMITS_DOC = Path("docs/FINDINGS_joint_limits.md")


def _artifacts_present() -> bool:
    return all((RESULTS / name).exists() for name, _ in P.SOURCES.values())


requires_artifacts = pytest.mark.skipif(
    not _artifacts_present(),
    reason=(
        "results/ is gitignored; the snapshot is still checked, but the "
        "cross-check against the live artifacts needs a local experiment run"
    ),
)


@pytest.fixture(scope="module", name="figures")
def figures_fixture() -> dict[str, float]:
    return P.load_snapshot(SNAPSHOT)


def _doc(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _pct(value: float) -> str:
    """The one-decimal signed percentage a table cell would carry."""
    return f"{value:.1f}%"


def _sci(value: float) -> str:
    """The scientific notation the tables use, e.g. 1.45e-4 not 1.45e-04.

    The documentation's existing style drops the exponent's leading zero, so the
    comparison has to as well or every scientific figure in it would fail.
    """
    mantissa, exponent = f"{value:.2e}".split("e")
    return f"{mantissa}e{int(exponent)}"


# --- the snapshot itself -------------------------------------------------


def test_snapshot_covers_every_declared_figure(figures):
    assert set(figures) == set(P.FIGURE_NAMES)


def test_snapshot_is_json_and_declares_its_sources():
    payload = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    assert set(payload) == {"sources", "figures"}
    assert set(payload["sources"]) == set(P.FIGURE_NAMES)
    for name, (filename, path) in P.SOURCES.items():
        assert payload["sources"][name] == [filename, list(path)]


def test_snapshot_has_no_absent_or_unusable_figures(figures):
    """A figure that resolved to a default would be a published wrong number.

    `load_figures` raises rather than defaulting, which is what makes this
    testable: reaching here means every figure came from a real artifact.
    """
    for name, value in figures.items():
        assert value == value, f"{name} is NaN"
        assert abs(value) != float("inf"), f"{name} is infinite"


def test_every_figure_is_referenced_by_at_least_one_document(figures):
    """A figure nothing quotes is a figure nothing checks.

    The inverse of the per-document tests: this catches a figure added to
    `SOURCES` for a result that was then never written up, which would
    otherwise sit in the snapshot looking verified.
    """
    corpus = "\n".join(
        _doc(path)
        for path in (README, RESULTS_DOC, LIMITS_DOC, Path("agents/RESUME_ENTRY.md"))
    )
    referenced = 0
    for value in figures.values():
        forms = {
            f"{value:.1f}",
            f"{value:.2f}",
            f"{abs(value):.1f}",
            f"{abs(value):.2f}",
            _sci(value),
            f"{int(value)}",
        }
        if any(form in corpus for form in forms):
            referenced += 1
    # A handful of intermediate values exist only to let other figures be
    # checked against each other, so full coverage is not the bar; a large
    # majority being quoted is.
    assert referenced >= 0.7 * len(figures), (
        f"only {referenced} of {len(figures)} figures appear in the "
        f"documentation; the rest are unverified claims"
    )


@requires_artifacts
def test_snapshot_matches_the_live_artifacts(figures):
    """Re-running an experiment without refreshing the snapshot fails here."""
    live = P.load_figures(RESULTS)
    differing = {
        k: (figures[k], live[k]) for k in P.FIGURE_NAMES if figures[k] != live[k]
    }
    assert not differing, (
        "the committed snapshot disagrees with results/trajectory_optimization; "
        f"re-run `refresh_published_figures` and commit it. Differing: {differing}"
    )


@requires_artifacts
def test_loading_without_the_artifacts_raises_rather_than_defaulting(tmp_path):
    with pytest.raises(P.MissingArtifact):
        P.load_figures(tmp_path)


def _artifact_set(tmp_path: Path) -> Path:
    """A directory holding every artifact, each populated with its full shape.

    The stubs are complete rather than empty so that a test which corrupts one
    artifact fails on that artifact and not on whichever unrelated key the
    resolver happened to reach first.
    """
    root = tmp_path / "results"
    root.mkdir()
    for filename in {name for name, _ in P.SOURCES.values()}:
        (root / filename).write_text(json.dumps(_stub_for(P.SOURCES, filename)))
    return root


def _stub_for(sources: dict, filename: str) -> dict:
    """A JSON object with every leaf `sources` needs from `filename`.

    The derived figures are not included: they are computed from arms rather
    than read from a key, so a stub that satisfied them would have to invent
    the arms too. The two arms they need are added where the artifact is a
    behaviour-cloning result.
    """
    stub: dict = {}
    for source, path in sources.values():
        if source != filename or P._DERIVED.get(path[0]) is not None:
            continue
        node = stub
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = 1.0
    if filename.startswith("bc_downstream"):
        stub.setdefault("raw", {}).setdefault("mse", 1.0)
        stub.setdefault("optimized", {}).setdefault("mse", 0.5)
        stub.setdefault("resampled_control", {}).setdefault("mse", 0.8)
    return stub


def test_a_missing_key_is_reported_rather_than_defaulted(tmp_path):
    """A renamed field must fail loudly, not resolve to zero.

    The alternative, defaulting a missing figure to 0.0, would let a number
    disappear from the documentation and leave the snapshot looking complete.
    """
    root = _artifact_set(tmp_path)
    payload = _stub_for(P.SOURCES, "optimizer_split.json")
    del payload["noise_scale"]
    (root / "optimizer_split.json").write_text(json.dumps(payload))
    with pytest.raises(P.MissingArtifact, match="noise_scale"):
        P.load_figures(root)


def test_a_malformed_artifact_is_reported_rather_than_defaulted(tmp_path):
    root = _artifact_set(tmp_path)
    (root / "real_vs_synthetic_s1.json").write_text("[]")
    with pytest.raises(P.MissingArtifact):
        P.load_figures(root)


def test_a_path_resolving_to_a_container_is_rejected():
    """Resolving a figure to a dict or list means the artifact's shape changed.

    `float()` on a container raises `TypeError`, which would surface as a
    crash in whatever regenerates the snapshot rather than as a report that the
    artifact no longer has the expected shape.
    """
    with pytest.raises(P.MissingArtifact, match="dict"):
        P._dig({"a": {"b": 1}}, ("a",))
    with pytest.raises(P.MissingArtifact, match="list"):
        P._dig({"a": [1, 2]}, ("a",))


def test_dig_resolves_a_real_scalar():
    assert P._dig({"a": {"b": 3}}, ("a", "b")) == 3.0


def test_dig_reports_a_walk_through_a_non_dict():
    with pytest.raises(P.MissingArtifact):
        P._dig({"a": [1, 2]}, ("a", "b"))


def test_a_missing_derived_arm_is_reported(tmp_path):
    """The controlled figure needs a control arm; without one it must fail."""
    root = _artifact_set(tmp_path)
    (root / "bc_downstream_dexycb.json").write_text(
        json.dumps({"raw": {"mse": 1.0}, "optimized": {"mse": 0.5}})
    )
    with pytest.raises(P.MissingArtifact, match="resampled_control"):
        P.load_figures(root)


def test_a_non_numeric_leaf_is_reported():
    with pytest.raises(P.MissingArtifact, match="not numeric"):
        P._require({"optimized": {"mse": "half"}}, "optimized", "mse")


def test_refresh_round_trips_through_a_snapshot(tmp_path):
    """The regeneration path must produce a file `load_snapshot` accepts."""
    if not _artifacts_present():
        pytest.skip("needs the live artifacts to refresh from")
    target = tmp_path / "docs" / "published_figures.json"
    written = P.refresh_published_figures(RESULTS, target)
    assert P.load_snapshot(target) == written
    assert target.read_text(encoding="utf-8").endswith("\n")


def test_loading_a_missing_snapshot_raises(tmp_path):
    with pytest.raises(P.MissingArtifact):
        P.load_snapshot(tmp_path / "absent.json")


# --- the figures quoted in prose -----------------------------------------


def test_readme_kinematic_table_matches_the_artifacts(figures):
    text = _doc(README)
    for label, value in (
        ("jerk s1", figures["jerk_reduction_s1"]),
        ("jerk s2", figures["jerk_reduction_s2"]),
        ("jerk synth", figures["jerk_reduction_synthetic"]),
        ("smoothness s1", figures["smoothness_reduction_s1"]),
        ("smoothness s2", figures["smoothness_reduction_s2"]),
        ("velocity s1", figures["velocity_reduction_s1"]),
        ("velocity s2", figures["velocity_reduction_s2"]),
        ("acceleration s1", figures["acceleration_reduction_s1"]),
        ("acceleration s2", figures["acceleration_reduction_s2"]),
    ):
        assert _pct(-value) in text, f"README is missing the {label} figure"


def test_readme_quotes_the_cross_subject_descent_rate(figures):
    """The headline rate is the held-out one, never a within-subject split."""
    text = _doc(README)
    assert f"{int(figures['descent_cross_subject'])}/100" in text
    assert f"{int(figures['descent_cross_subject'])}/100 improved" in text


def test_readme_quotes_the_median_cost_reduction(figures):
    assert f"{figures['median_cost_reduction_cross_subject']:.1f}" in _doc(README)


def test_readme_quotes_the_controlled_bc_figures(figures):
    text = _doc(README)
    assert f"{figures['bc_controlled_s1']:.1f} percent" in text
    assert f"{figures['bc_controlled_s2']:.1f} percent" in text
    assert f"{figures['bc_controlled_synthetic']:.1f} percent" in text


def test_readme_quotes_the_executability_figures(figures):
    text = _doc(README)
    for key in ("executable_raw_s1", "executable_opt_s1", "executable_opt_s2"):
        assert f"{int(figures[key])}/100" in text, f"README is missing {key}"


def test_readme_quotes_the_split_granularity_figures(figures):
    text = _doc(README)
    for key in (
        "granularity_transition_s1",
        "granularity_trajectory_s1",
        "granularity_prefix_s1",
        "granularity_prefix_s2",
        "granularity_prefix_synthetic",
    ):
        assert f"{figures[key]:.1f}" in text, f"README is missing {key}"


def test_results_doc_kinematic_table_matches_the_artifacts(figures):
    text = _doc(RESULTS_DOC)
    for value in (
        figures["jerk_reduction_s1"],
        figures["jerk_reduction_s2"],
        figures["jerk_reduction_synthetic"],
        figures["smoothness_reduction_s1"],
        figures["smoothness_reduction_s2"],
        figures["velocity_reduction_s1"],
        figures["velocity_reduction_s2"],
        figures["acceleration_reduction_s1"],
        figures["acceleration_reduction_s2"],
    ):
        assert _pct(-value) in text


def test_results_doc_quotes_the_held_out_protocol_figures(figures):
    text = _doc(RESULTS_DOC)
    assert f"{int(figures['descent_cross_subject'])}/100" in text
    assert f"{figures['median_cost_reduction_holdout_s1']:.2f}%" in text
    assert f"{figures['median_cost_reduction_cross_subject']:.2f}%" in text


def test_results_doc_quotes_the_feasibility_table(figures):
    text = _doc(RESULTS_DOC)
    for key in (
        "executable_raw_s1",
        "executable_control_s1",
        "executable_opt_s1",
        "executable_opt_s2",
    ):
        assert f"{int(figures[key])} / 100" in text, f"RESULTS.md is missing {key}"


def test_results_doc_quotes_the_prefix_baseline_figures(figures):
    """The explanation for the prefix reversal is numbers, so it is pinned."""
    text = _doc(RESULTS_DOC)
    for value in (
        figures["prefix_control_score_s1"],
        figures["prefix_trivial_baseline_s1"],
        figures["prefix_trivial_baseline_synthetic"],
        figures["prefix_opt_score_s1"],
    ):
        assert _sci(value) in text, f"RESULTS.md is missing {_sci(value)}"


def test_limits_doc_quotes_the_corrected_figures(figures):
    text = _doc(LIMITS_DOC)
    assert _pct(-figures["jerk_reduction_s1"]) in text
    assert f"{int(figures['executable_opt_s1'])}/100" in text
    assert f"{int(figures['executable_opt_s2'])}/100" in text


# --- relationships the documentation asserts about the figures -----------


def test_optimizing_never_makes_executability_worse(figures):
    """The pipeline's purpose is executability, so this is a hard invariant.

    Only the two real data sets have a control arm. Synthetic data is already
    at the control rate, so its comparison is against raw and there is no
    resampled step that optimizing could be confounded with.
    """
    for prefix in ("s1", "s2"):
        assert (
            figures[f"executable_opt_{prefix}"]
            >= figures[f"executable_control_{prefix}"]
        ), f"optimizing reduced executability on {prefix}"


def test_control_arm_beats_raw_on_every_real_data_set(figures):
    """Resampling is the confound every published comparison controls for."""
    for prefix in ("s1", "s2"):
        assert (
            figures[f"executable_control_{prefix}"]
            > figures[f"executable_raw_{prefix}"]
        )


def test_synthetic_data_is_fully_executable_before_any_processing(figures):
    """Why the real-versus-synthetic gap is about the data, not the metric.

    If either synthetic arm were not fully executable, the 34/100 on subject-01
    could not be read as evidence that real human motion is harder for the
    hand rather than for the metric.
    """
    assert figures["executable_raw_synthetic"] == 100
    assert figures["executable_opt_synthetic"] == 100


def test_the_prefix_split_reverses_only_on_real_data(figures):
    """Pins the shape of the finding the tail diagnosis explains.

    If a future run makes the prefix advantage positive on real data, this
    fails, and it should: the explanation in `RESULTS.md` is that the split's
    holdout is motionless there, and a positive result would contradict it.
    """
    for prefix in ("s1", "s2"):
        assert figures[f"granularity_prefix_{prefix}"] < 0.0
    assert figures["granularity_prefix_synthetic"] > 0.0


def test_real_prefix_control_arm_is_at_the_do_nothing_baseline(figures):
    """The measurement that makes the prefix reversal explicable.

    On subject-01 the control arm's prefix score is within a few percent of
    predicting no motion at all. That is the whole reason the split cannot rank
    the optimizer on real data, so it is asserted rather than described.
    """
    score = figures["prefix_control_score_s1"]
    baseline = figures["prefix_trivial_baseline_s1"]
    assert abs(score - baseline) / baseline < 0.05


def test_synthetic_prefix_tail_is_not_motionless(figures):
    """The converse: where the tail moves, the same split is positive."""
    assert (
        figures["prefix_trivial_baseline_synthetic"]
        > figures["prefix_control_score_synthetic"]
    )


def test_real_motion_decelerates_into_a_stop_and_synthetic_does_not(figures):
    """The mechanism behind the above, stated as a check on the ratio."""
    assert figures["tail_speed_ratio_control_s1"] < 0.5
    assert figures["tail_speed_ratio_raw_synthetic"] > 0.8


def test_the_optimizer_redistributes_deceleration_rather_than_removing_it(figures):
    """The optimizer raises the final step without reaching synthetic levels."""
    assert (
        figures["tail_speed_ratio_control_s1"]
        < figures["tail_speed_ratio_optimized_s1"]
        < 1.0
    )


def test_velocity_reduction_is_limit_saturation_not_headroom(figures):
    """The README calls this row out; the number is pinned so it stays called out.

    `MAX_VELOCITY` is a smoothness limit the optimizer clips to, not a robot
    property, so the reported reduction is near-total on every data set by
    construction. Asserting it here means a future change that alters it has to
    confront the caveat rather than silently invalidate it.
    """
    for key in ("velocity_reduction_s1", "velocity_reduction_s2"):
        assert figures[key] > 60.0
    assert "constraint" in _doc(README)


def test_optimizer_reduces_smoothness_on_every_data_set(figures):
    for key in (
        "smoothness_reduction_s1",
        "smoothness_reduction_s2",
        "jerk_reduction_s1",
        "jerk_reduction_s2",
    ):
        assert figures[key] > 0.0, f"{key} says the optimizer made things worse"
