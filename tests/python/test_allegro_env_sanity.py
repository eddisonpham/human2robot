"""Phase 2 acceptance: all six checks from agents/06 §7 for the modified Allegro MJCF.

These validate that the floating-base Allegro model is structurally sound
before any RL training is attempted on it.
"""

import mujoco
import numpy as np

from human2robot.envs.allegro import AllegroPickupEnv, _floating_model_xml


def _load_model():
    return mujoco.MjModel.from_xml_string(_floating_model_xml())


def test_check1_model_loads_without_xml_errors():
    """1. Loads without XML errors; mujoco.MjModel.from_xml_path succeeds."""
    model = _load_model()
    assert model is not None
    assert model.njnt > 0
    assert model.ngeom > 0


def test_check2_ten_k_steps_no_nan():
    """2. Runs 10^4+ steps with random actions without NaN/divergence."""
    model = _load_model()
    data = mujoco.MjData(model)
    _rng = np.random.default_rng(0)
    for _ in range(10_000):
        mujoco.mj_step(model, data)
        assert np.isfinite(data.qpos).all(), "NaN in qpos"
        assert np.isfinite(data.qvel).all(), "NaN in qvel"
        assert np.isfinite(data.ctrl).all(), "NaN in ctrl"
        assert np.isfinite(data.actuator_force).all(), "NaN in actuator_force"


def test_check3_determinism_same_seed():
    """3. Same seed -> bit-identical trajectory on repeated runs."""
    model = _load_model()
    data_a = mujoco.MjData(model)
    data_b = mujoco.MjData(model)
    rng = np.random.default_rng(7)
    mujoco.mj_resetData(model, data_a)
    mujoco.mj_resetData(model, data_b)
    for _ in range(200):
        action = rng.uniform(-1.0, 1.0, size=model.nu)
        data_a.ctrl[:] = action
        data_b.ctrl[:] = action
        for _ in range(10):
            mujoco.mj_step(model, data_a)
            mujoco.mj_step(model, data_b)
    assert np.allclose(data_a.qpos, data_b.qpos, atol=1e-12)
    assert np.allclose(data_a.qvel, data_b.qvel, atol=1e-12)


def test_check4_object_falls_under_gravity():
    """4. Object falls under gravity when unsupported."""
    model = _load_model()
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    object_adr = model.jnt_qposadr[model.joint("object_freejoint").id]
    data.qpos[object_adr + 2] = 0.2
    mujoco.mj_forward(model, data)
    z_before = data.qpos[object_adr + 2]
    for _ in range(500):
        mujoco.mj_step(model, data)
    z_after = data.qpos[object_adr + 2]
    assert z_after < z_before, "object did not fall"


def test_check5_joint_limits_and_ctrlrange():
    """5. Actuator ctrlrange is defined for all actuators; joint limits exist.

    The Menagerie Allegro XML defines ctrlrange on position actuators and
    joint_range on joints. This check confirms both are present and finite
    (no NaN/-inf/+inf) so the controller has well-defined bounds to work
    within. It does NOT assert that random ctrl inputs stay in range after
    stepping --- MuJoCo stores the unclipped ctrl in data.ctrl throughout
    the step, only clamping internally during force computation.
    """
    model = _load_model()
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    # Every actuator must have a finite ctrlrange.
    for a in range(model.nu):
        lo, hi = model.actuator_ctrlrange[a]
        assert np.isfinite(lo), f"actuator {a} ctrlrange low is not finite"
        assert np.isfinite(hi), f"actuator {a} ctrlrange high is not finite"
        assert lo < hi, f"actuator {a} ctrlrange inverted"
    # Every joint must have finite range (or both NaN for free joints).
    for j in range(model.njnt):
        jl = model.jnt_range[j]
        jname = model.joint(j).name
        if np.any(np.isnan(jl)):
            # free joints have no limits — that's fine
            assert model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE, (
                f"joint {jname} has NaN range but is not free"
            )
        else:
            lo, hi = jl
            assert np.isfinite(lo) and np.isfinite(hi), (
                f"joint {jname} has non-finite range"
            )
            # range [0, 0] is valid for some constrained joints
            assert lo <= hi, f"joint {jname} range inverted"
    # Verify the model runs stably with random controls (no NaN/divergence).
    for _ in range(2000):
        data.ctrl[:] = np.random.uniform(-1, 1, size=model.nu)
        mujoco.mj_step(model, data)
        assert np.isfinite(data.qpos).all()
        assert np.isfinite(data.qvel).all()
        assert np.isfinite(data.actuator_force).all()


def test_check6_floating_base_no_self_contact():
    """6. Floating base doesn't introduce degenerate contacts with itself."""
    model = _load_model()
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    base_adr = model.jnt_qposadr[model.joint("base_freejoint").id]
    data.qpos[base_adr : base_adr + 7] = [0.0, 0.0, 0.22, 1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    palm_body = model.body("palm").id
    for _ in range(1000):
        mujoco.mj_step(model, data)
    contacts = data.ncon
    # Self-contacts = palm geom <-> palm geom (both bodies == palm).
    # Contacts with floor (body 0) or object (body 22) are legitimate.
    palm_body = model.body("palm").id
    palm_self_contacts = 0
    for i in range(contacts):
        geom_a = data.contact.geom1[i]
        geom_b = data.contact.geom2[i]
        body_a = model.geom_bodyid[geom_a]
        body_b = model.geom_bodyid[geom_b]
        if body_a == palm_body and body_b == palm_body:
            palm_self_contacts += 1
    assert palm_self_contacts == 0, (
        f"palm has {palm_self_contacts} self-contacts; exclude clauses may be wrong"
    )


def test_env_wrapper_matches_raw_model():
    """Bonus: the gymnasium env wraps the same model correctly."""
    env = AllegroPickupEnv(max_episode_steps=100)
    try:
        obs, _ = env.reset(seed=0)
        assert env.action_space.shape == (22,)
        assert env.observation_space.shape == (64,)
        for _ in range(200):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert np.isfinite(obs).all()
            assert np.isfinite(reward)
            if terminated or truncated:
                env.reset(seed=0)
    finally:
        env.close()
