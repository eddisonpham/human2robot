# DynHand Resume Entry

## Reinforcement Learning and ML Systems Engineer

**DynHand | PyTorch, SAC, MuJoCo, Gymnasium-Robotics, Minari, Pydantic, uv**

- Built a reproducible, configuration-driven Soft Actor-Critic framework for dexterous manipulation, measured by an **81-test suite, 92.6% code coverage, and clean Ruff checks**, by implementing twin-Q critics, automatic entropy tuning, deterministic evaluation, checkpoint recovery, and seeded vector environments.
- Integrated demonstration-guided reinforcement learning, measured by a configurable **100-epoch behavioral-cloning pretraining path, held-out validation, and 0.5-to-0 demo replay-ratio annealing over 300,000 environment steps**, by combining BC initialization with Minari demonstration transitions in SAC replay sampling.
- Improved experiment reliability for long-running GPU training, measured by **zero-tolerance duplicate or out-of-order evaluation streams**, by adding cross-platform run locks, seed-scoped run names, checkpoint recovery, fail-closed queues, and JSONL metrics auditing.
- Added an independent RL validation path, measured by **matched seed, replay size, batch size, learning-start, update-ratio, and periodic evaluation settings**, by integrating a single-process Stable-Baselines3 SAC cross-check for the Adroit relocate benchmark.
- Optimized local training for an **8 GB RTX 5060 and 24-core workstation**, measured by verified PyTorch CUDA capability **(12, 0)** and end-to-end MuJoCo smoke training, by using CPU-resident replay, bounded Torch threading, synchronous vector environments, and CUDA 12.8 PyTorch wheels.
- Engineered reproducible ML experiment artifacts, measured by per-run **config, metrics, checkpoint, Git provenance, system information, and TensorBoard outputs**, by enforcing validated YAML configuration, deterministic seed propagation, and auditable result directories.

## Verified Scope

The current repository demonstrates the RL and experiment-infrastructure foundation. Tier B Allegro retargeting, learned dynamics augmentation, multi-seed benchmark conclusions, ONNX export, and Rust serving remain planned work and should not be listed as completed achievements.

## Short Resume Version

- Built DynHand, a reproducible demonstration-guided SAC framework for dexterous manipulation in MuJoCo, achieving **92.6% test coverage across 81 tests** with deterministic evaluation, seeded vector environments, BC pretraining, Minari replay integration, checkpoint recovery, and strict Pydantic configuration.
- Hardened long-running RL experimentation with cross-platform run locks, seed-scoped artifacts, JSONL metrics auditing, fail-closed queues, and an SB3 cross-check using matched training settings on the Adroit relocate benchmark.
- Validated PyTorch CUDA execution on an **RTX 5060 Blackwell GPU with compute capability (12, 0)** and optimized the local training stack for an **8 GB VRAM, 24-core workstation**.

## XYZ Formula

Use this structure when tailoring bullets:

> Accomplished **X**, measured by **Y**, by doing **Z**.

Keep measured claims tied to committed tests, reproducible run artifacts, or verified hardware output. Do not claim improved sample efficiency until clean multi-seed Condition A, Condition B, and SB3 runs have been completed and audited.
