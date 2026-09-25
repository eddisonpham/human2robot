# Human2Robot C++ Robotics and Trajectory Optimization Extension

## 1. Objective

Extend the existing Human2Robot project with a C++ robotics/control layer that turns the existing human-hand retargeting pipeline into a usable trajectory optimization and simulation system.

The primary research question is:

> Can trajectory optimization improve the quality of human-derived robot demonstrations for downstream robot learning?

The C++ work must complement the existing Python simulator, dynamics, RL, and demonstration infrastructure. Do **not** rewrite the existing simulator or RL stack in C++.

The final system should support:

```text
Human-hand demonstration
        |
        v
Existing Python hand tracking / retargeting
        |
        v
Robot joint trajectory
        |
        v
C++ trajectory validation
        |
        v
C++ trajectory optimization
        |
        +--------------------+
        |                    |
        v                    v
Existing Python/MuJoCo   C++ rollout/control
simulation               utilities
        |                    |
        +---------+----------+
                  |
                  v
          trajectory metrics
                  |
                  v
       downstream robot learning
```

The C++ layer is therefore a robotics/control subsystem, not a replacement for the existing Human2Robot simulator.

---

# 2. Current Project State

The project currently contains:

* Tier A SAC on Adroit relocate, validated
* Tier B floating-base Allegro MJCF and touch sensors
* inverse-dynamics validation
* computed-torque baseline
* DexYCB loaders and retargeting scaffold
* demonstration labels
* learned dynamics ensemble
* residual dynamics
* SAC integration
* ONNX actor export with numerical parity validation
* Python training infrastructure
* validated YAML configuration system
* experiment artifact tracking
* test suite and coverage gate

Current phase status:

```text
Phase 0: Done
Phase 1: Done
Phase 2: Done
Phase 3: Present / validated in dynamics tests
Phase 4/5: Scaffolded, retargeted replay validation still needs confirmation
Phase 6: Done
Phase 7/8: Not started
Phase 9: Scaffolded
Phase 10: ONNX export done, Rust inference server not started
Phase 11: Not started
```

The existing simulator and dynamics stack should remain intact.

The primary Human2Robot research deliverable remains the Phase 8 RL ablation matrix. The C++ work must not destabilize or block that work.

---

# 3. Scope

Implement the following:

## A. C++ robotics core

A reusable C++20 library containing:

* robot joint state representation
* trajectory representation
* forward kinematics
* Jacobian computation
* joint-limit checking
* velocity/acceleration limits
* trajectory interpolation
* trajectory resampling
* trajectory smoothing
* collision/constraint interfaces
* trajectory cost functions

## B. C++ trajectory optimizer

Implement trajectory optimization over robot joint trajectories.

At minimum support:

$$
J(\tau)
=
w_p J_{\mathrm{tracking}}
+
w_v J_{\mathrm{velocity}}
+
w_a J_{\mathrm{acceleration}}
+
w_j J_{\mathrm{jerk}}
+
w_c J_{\mathrm{collision}}
+
w_l J_{\mathrm{limits}}
$$

where:

* $J_{\mathrm{tracking}}$ measures deviation from the retargeted demonstration
* $J_{\mathrm{velocity}}$ penalizes excessive joint velocity
* $J_{\mathrm{acceleration}}$ penalizes excessive acceleration
* $J_{\mathrm{jerk}}$ penalizes discontinuous motion
* $J_{\mathrm{collision}}$ penalizes invalid configurations
* $J_{\mathrm{limits}}$ penalizes joint-limit violations

The optimizer must support configurable weights.

Do not hard-code experiment-specific behavior into the optimizer.

---

# 4. Optimization Problem

Given a retargeted demonstration:

$$
\tau_d =
\{q_0, q_1, \ldots, q_T\}
$$

produce:

$$
\tau^* =
\arg\min_{\tau} J(\tau)
$$

subject to:

$$
q_{\min} \leq q_t \leq q_{\max}
$$

$$
|\dot q_t| \leq \dot q_{\max}
$$

$$
|\ddot q_t| \leq \ddot q_{\max}
$$

and, where supported,

$$
C(q_t) \geq 0
$$

for collision or geometric constraints.

The initial implementation should prioritize robustness and testability over sophisticated optimization algorithms.

---

# 5. Optimization Strategy

Implement the optimizer in stages.

## Stage 1: deterministic trajectory processing

Implement:

* linear interpolation
* cubic interpolation
* resampling
* finite-difference velocity
* finite-difference acceleration
* finite-difference jerk
* moving-window smoothing

These components must have unit tests.

## Stage 2: constrained trajectory projection

Implement projection of trajectories onto:

* joint limits
* velocity limits
* acceleration limits

The output must remain temporally ordered and preserve trajectory length unless explicitly configured otherwise.

## Stage 3: cost-based optimization

Implement an iterative optimizer that minimizes the trajectory objective.

The optimizer should expose:

```text
initial trajectory
target trajectory
time step
cost weights
joint limits
velocity limits
acceleration limits
maximum iterations
convergence tolerance
```

Do not tie the implementation to one robot.

---

# 6. Robot Model Interface

Create a generic robot model interface.

Conceptually:

```cpp
class RobotModel {
public:
    virtual ~RobotModel() = default;

    virtual Eigen::VectorXd forward_kinematics(
        const Eigen::VectorXd& q) const = 0;

    virtual Eigen::MatrixXd jacobian(
        const Eigen::VectorXd& q) const = 0;

    virtual Eigen::VectorXd joint_lower_limits() const = 0;

    virtual Eigen::VectorXd joint_upper_limits() const = 0;
};
```

The exact API should follow the repository's existing architecture.

Do not introduce an unnecessary robotics framework if the project does not already depend on one.

Prefer lightweight dependencies.

---

# 7. Integration With Existing Human2Robot Simulator

Do NOT rewrite the existing MuJoCo/Python simulator.

Instead provide a clean interface for exporting/importing trajectories.

Required data flow:

```text
Python
    |
    | retargeted trajectory
    v
C++ optimizer
    |
    | optimized trajectory
    v
Python / MuJoCo
    |
    | rollout
    v
metrics
```

Initially, integration may use a simple file representation such as:

```text
CSV
NPZ
JSON
```

or another format already used by the repository.

If practical, expose the C++ library to Python using `pybind11`.

The preferred long-term interface is:

```python
trajectory = load_demo(...)
optimized = human2robot_cpp.optimize(trajectory)
metrics = evaluate(optimized)
```

The C++ library must remain independently testable without Python.

---

# 8. Simulation Platform Decision

Do NOT implement a second full physics simulator.

The existing Human2Robot project already has a validated simulation environment.

A second full simulator would:

* duplicate the existing physics implementation
* create unnecessary validation work
* consume project time without improving the research question
* make numerical equivalence difficult to establish
* distract from trajectory optimization and robot learning

Instead, implement a lightweight C++ rollout/robotics platform around the existing simulation.

This platform should provide:

1. robot state representation
2. trajectory playback
3. kinematic evaluation
4. constraint checking
5. trajectory metrics
6. optional C++-side dynamics interfaces where useful
7. deterministic trajectory replay

If direct MuJoCo C++ integration is practical within the repository architecture, add it only after the optimizer and kinematic components are independently validated.

The C++ platform is therefore a **robotics simulation/control interface**, not a new physics engine.

---

# 9. Existing Dynamics Integration

Human2Robot already contains:

* analytical dynamics
* inverse dynamics
* computed-torque control
* learned black-box dynamics
* residual dynamics
* SAC integration

Do not duplicate these systems.

Where useful, expose existing dynamics functionality to the C++ layer through a clearly defined interface.

The first version should not attempt to port the learned dynamics ensemble to C++.

The goal is trajectory optimization and robotics control, not a complete rewrite of the ML stack.

---

# 10. Collision and Geometry

The first implementation should support an abstract collision/constraint interface.

For example:

```text
CollisionChecker
    |
    +-- configuration_valid(q)
    +-- trajectory_valid(q_0 ... q_T)
    +-- collision_cost(q)
```

Do not build a complete mesh engine unless it is required by the existing robot/environment representation.

If the existing simulator exposes useful geometry information, integrate with that.

A mesh-based representation may be added later for:

* hand/object contact
* collision checking
* grasp geometry
* object surfaces

but this is explicitly secondary to trajectory optimization.

---

# 11. Human Demonstration Integration

The existing Python code already contains demonstration loaders and retargeting scaffolding.

After Phase 4/5 validation is complete, support:

```text
DexYCB / human demonstration
        |
        v
existing retargeting
        |
        v
robot joint trajectory
        |
        v
C++ optimizer
        |
        v
optimized robot trajectory
```

Preserve the original demonstration.

Never overwrite the raw retargeted trajectory.

Store:

```text
raw trajectory
optimized trajectory
optimization configuration
optimization metrics
```

together as an experiment artifact.

---

# 12. Experimental Evaluation

The core experiment should compare:

### Baseline

Raw retargeted human trajectory.

### Optimized

Raw trajectory passed through C++ trajectory optimization.

Measure:

## Geometric metrics

* joint-limit violations
* maximum joint velocity
* maximum acceleration
* maximum jerk
* trajectory smoothness

## Tracking metrics

$$
E_q =
\frac{1}{T}
\sum_t
\|q_t-q_t^{demo}\|_2
$$

Also consider end-effector tracking error.

## Constraint metrics

* collision count
* invalid configurations
* joint-limit violations
* velocity-limit violations
* acceleration-limit violations

## Runtime

Measure:

* optimization wall-clock time
* iterations
* convergence
* trajectory length
* memory usage where useful

## Downstream metric

Where possible, replay both trajectories in the existing simulator and compare:

* task completion
* reward
* grasp success
* manipulation success

Do not invent metrics that the environment cannot reliably measure.

---

# 13. VLA / Robot Learning Extension

After the optimizer is validated, optionally evaluate whether optimized demonstrations improve downstream learning.

Possible experiment:

```text
Dataset A:
raw retargeted demonstrations

Dataset B:
optimized demonstrations

Dataset C:
raw + optimized demonstrations
```

Train the same downstream policy under each condition.

Keep all training hyperparameters identical.

Compare:

* learning curve
* final evaluation return
* task success
* generalization where supported

The research question becomes:

> Does geometric trajectory optimization make automatically retargeted human demonstrations more useful for robot learning?

This is the preferred connection to the existing VLA/robot-learning work.

---

# 14. C++ Project Structure

Use a structure approximately like:

```text
cpp/
  human2robot/
    include/
      human2robot/
        robot/
        trajectory/
        kinematics/
        optimization/
        constraints/
        metrics/
        io/
    src/
      robot/
      trajectory/
      kinematics/
      optimization/
      constraints/
      metrics/
      io/
    tests/
    examples/
    CMakeLists.txt
```

Do not blindly follow this structure if the repository has an existing C++ convention.

The repository's existing organization takes precedence.

---

# 15. Dependencies

Prefer a minimal dependency set.

Expected candidates:

* C++20
* Eigen
* CMake
* GoogleTest or the repository's existing C++ testing framework
* pybind11 only if Python bindings are required

Do not introduce a large robotics framework simply for convenience.

Use existing project dependencies where possible.

---

# 16. Python Interface

If Python bindings are implemented:

```python
from human2robot_cpp import TrajectoryOptimizer

optimizer = TrajectoryOptimizer(config)

optimized = optimizer.optimize(
    trajectory,
    robot_model,
)
```

The Python interface must not expose implementation details unnecessarily.

Python should remain responsible for:

* ML
* demonstration loading
* VLM/VLM-derived metadata
* RL
* experiment orchestration

C++ should handle:

* geometry
* trajectory processing
* optimization
* constraint checking
* high-performance robotics computations

---

# 17. Testing Requirements

Every new functional C++ module must have tests.

At minimum test:

### Trajectory

* interpolation
* resampling
* finite differences
* smoothing

### Kinematics

* FK known configurations
* Jacobian dimensions
* numerical Jacobian comparison where practical

### Constraints

* joint-limit detection
* velocity-limit detection
* acceleration-limit detection

### Optimization

* objective decreases on a controlled test case
* optimized trajectory respects configured constraints
* optimizer handles already-valid trajectories
* optimizer handles invalid trajectories
* deterministic output for fixed inputs/configuration

### Integration

* Python → C++ trajectory transfer
* C++ → Python trajectory transfer
* numerical consistency

---

# 18. Acceptance Tests

The C++ extension is not complete until:

1. A known trajectory can be loaded.
2. The trajectory can be represented in C++.
3. FK can be evaluated for every timestep.
4. Joint-limit violations are detected.
5. Velocity and acceleration violations are detected.
6. The trajectory optimizer reduces the configured objective on controlled test data.
7. The optimized trajectory satisfies configured hard constraints.
8. The optimized trajectory can be exported back to Python.
9. The existing MuJoCo simulator can replay the optimized trajectory.
10. Baseline and optimized trajectories can be compared automatically.
11. The full existing Python test suite still passes.
12. Ruff passes.
13. Existing smoke training still passes.

Before completion:

```bash
uv run pytest
uv run ruff check .
```

Run the configured training smoke test for any Python-side training changes.

---

# 19. Performance Requirements

Do not optimize prematurely.

First establish correctness.

After correctness is established, benchmark:

* trajectory optimization time
* FK throughput
* Jacobian throughput
* constraint-checking throughput
* Python/C++ transfer overhead

The C++ implementation should provide a meaningful performance advantage over equivalent Python implementations for computationally intensive trajectory operations.

Do not claim a speedup without benchmarking.

Do not use all available CPU cores by default.

Parallelism must be configurable.

---

# 20. Phase Ordering

Do not disrupt the existing Human2Robot phase plan.

The order should be:

### Phase A

Confirm Phase 4/5 acceptance.

Specifically verify that:

* DexYCB loading works
* retargeting works
* replay is correct
* demonstrations have valid robot trajectories

### Phase B

Implement C++ trajectory representation and I/O.

### Phase C

Implement C++ kinematics and constraint checking.

### Phase D

Implement deterministic trajectory processing.

### Phase E

Implement trajectory optimization.

### Phase F

Integrate with the existing MuJoCo simulator.

### Phase G

Run baseline vs optimized trajectory experiments.

### Phase H

Optionally connect optimized demonstrations to downstream BC/SAC/VLA experiments.

Do not allow the C++ work to block Phase 8 unless explicitly chosen as the primary experiment.

---

# 21. Research Deliverable

The final project should be presentable as:

> **Human2Robot: C++ Trajectory Optimization for Human-to-Robot Demonstration Retargeting**

Core contribution:

> A C++ robotics/control layer that converts noisy human-derived robot trajectories into constraint-aware, dynamically feasible demonstrations and evaluates their downstream usefulness for robot learning.

The strongest final experiment is:

```text
Human demonstration
        |
        v
retargeting
        |
        +----------------+
        |                |
        v                v
     baseline       C++ optimizer
        |                |
        +--------+-------+
                 |
                 v
          MuJoCo rollout
                 |
                 v
       trajectory metrics
                 |
                 v
      robot learning result
```

The project should emphasize the **research result**, not the fact that C++ was used.

C++ is the implementation choice that enables a high-performance robotics/control subsystem. It is not itself the contribution.

---

# 22. Important Constraints

Preserve the existing Human2Robot conventions:

* use `uv`
* do not use system miniconda
* PyTorch uses the cu128 index
* phase N+1 only after phase N acceptance
* hyperparameters come from validated configs
* no condition-specific branches in `rl/`
* `envs/` owns reward and observation
* one top-level seed
* vector workers use `seed + worker_index`
* no comments except short public docstrings
* no emoji
* no em dashes
* new functional modules require tests
* never commit `data/`
* never commit `results/`

Do not modify existing validated behavior unless required for integration.

Do not rewrite the Python simulator.

Do not rewrite the RL system in C++.

Do not implement a second full physics engine.

Do not add mesh modeling as an independent project.

Mesh/geometry support should only be added when it directly contributes to collision checking, contact reasoning, or trajectory feasibility.

---

# 23. Definition of Success

The project succeeds if the following statement can be demonstrated experimentally:

> Human-derived robot demonstrations can be passed through a C++ robotics/control pipeline that improves their geometric and dynamic validity while preserving task-relevant behavior, and the resulting demonstrations can be replayed in Human2Robot and evaluated for downstream robot-learning utility.

The ideal final artifact is therefore not:

> "I built a C++ simulator."

It is:

> "I built a C++ robotics optimization system for turning human demonstrations into higher-quality robot-learning trajectories, validated against the existing Human2Robot simulator and downstream learning tasks."
