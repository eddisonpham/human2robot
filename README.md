# Human2Robot

Converts recorded human hand motion into robot joint trajectories for a
simulated Allegro hand, and measures whether the hand can actually perform them.

Video of a hand becomes joint targets; a C++ optimizer smooths them within joint,
velocity, and acceleration limits; the result is replayed in simulation to check
it is realisable. A SAC trainer consumes the trajectories downstream and is
reported as a null result.

## Results

**Executability** is the one number that describes the robot rather than the
trajectory. It counts trajectories the simulated hand can track within tolerance
when replayed open-loop, against a resampled pre-optimization control.

| | raw | control | optimized |
| --- | --- | --- | --- |
| Subject 1 (100 seq) | 6/100 | 14/100 | **34/100** |
| Subject 2, held out | 0/100 | 6/100 | **8/100** |
| Synthetic | 100/100 | n/a | 100/100 |

Synthetic passing cleanly is the control: it shows the low real-data numbers
reflect hard human motion, not a broken metric. A third of trajectories is a
real improvement and an unsolved problem.

**The rest, in one line each.** Full tables and method in
[docs/RESULTS.md](docs/RESULTS.md).

| Claim | Subject 1 | Subject 2 | Synthetic |
| --- | --- | --- | --- |
| Search descends (subject 2 selected on subject 1) | 100/100 | 100/100 | 92/100 |
| Smoothness cost reduced | 30.4% | 28.7% | 92.6% |
| Imitation error vs resampled control | 29.5% | 29.0% | 56.8% |
| Same, holding out whole trajectories | +25.2% | +33.3% | +50.6% |

**Known limits.** `max_velocity` gain is constraint saturation, not headroom.
Imitation measures learnability, not task success. Executability is simulation,
not hardware. Tail extrapolation is untested: real motion decelerates to a stop,
so that holdout rewards whichever arm stops hardest
([docs/FINDINGS_joint_limits.md](docs/FINDINGS_joint_limits.md) and the
[tail diagnosis](docs/RESULTS.md)).

**Nine published results were measurement artifacts, not findings.** Including a
−41.9% jerk gain that was an artifact of a retargeter collapsing 16 joint
dimensions into 1 (−16.8% on real data), and two functions named after simulator
evaluations that never ran the simulator. Each is written up in
[docs/FINDINGS_](docs/).

## Install and run

Requires [uv](https://docs.astral.sh/uv/). Python 3.11 is pinned by
`.python-version`. The GPU is Blackwell (sm_120) and needs the cu128 PyTorch
index, which `pyproject.toml` already sets.

```bash
uv sync                                    # install
uv run pytest                              # 442 tests, coverage-gated at 90%
uv run ruff check .                        # lint
```

The C++ library builds separately and is not needed for the Python tests.

```bash
cmake -S cpp -B cpp/build && cmake --build cpp/build   # needs clang++/MSVC, Ninja
ctest --test-dir cpp/build --output-on-failure          # 64 tests
```

## Commands

Experiment entry points. Each writes one JSON to `results/trajectory_optimization/`.
`data/` and `results/` are gitignored, so a fresh clone must generate them first.

```bash
# Score executability. Needs optimized demos; fastest useful command here.
uv run python scripts/score_feasibility.py --set all

# Retarget and optimize. DexYCB download is a separate, gated step.
uv run python scripts/retarget_dexycb_ik.py --output-dir data/demonstrations_dexycb_ik
uv run python scripts/compare_dexycb_synthetic.py --subject subject-01

# Behaviour cloning, splits, diagnosis
uv run python scripts/run_downstream_bc.py --set dexycb
uv run python scripts/check_bc_split_granularity.py --set dexycb --seeds 5
uv run python scripts/diagnose_tail_extrapolation.py --set dexycb
```

RL tooling. Runs take hours; the gate does not require any of these.

```bash
uv run human2robot-train --config configs/tier_b_pickup.yaml --seed 0
uv run human2robot-eval    --config <cfg> --checkpoint <ckpt>
uv run human2robot-plot    --runs "SAC=results/run_a, SAC+demo=results/run_b" --out results/curves.png
uv run human2robot-benchmark --group "A=results/run_a,results/run_b" --out results/summary.json
uv run human2robot-export  --config <cfg> --checkpoint <ckpt> --out policy.onnx
uv run human2robot-sb3-check --config <cfg>
```

## Layout

```
src/human2robot/   pipeline, environments, RL, evaluation
cpp/               h2r_traj optimizer (C++20)
scripts/           experiment and data entry points
docs/              results, findings, architecture
agents/            build specification
```

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) has the pipeline diagram, the module
map, and which script produces which number.

## Honest status

The RL half is a **null result**: no condition reaches positive return, so the
ablation had no signal to separate them, and it is reported as such
([docs/RL_RESULTS.md](docs/RL_RESULTS.md)).

Two subjects on one capture rig demonstrate cross-person generalization across
two people, not robustness to conditions. There is no hardware; everything is
simulated. The Rust inference server builds and has tests but has never run
against a model. The test suite passes from a fresh clone.
