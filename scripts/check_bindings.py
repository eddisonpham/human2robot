import sys

sys.path.insert(0, "cpp/build")
import h2r_cpp as m
import numpy as np

print("step 1: import ok", flush=True)
t = m.trajectory_from_numpy(0.01, np.sin(np.linspace(0, 3, 50))[:, None].repeat(2, 1))
print("step 2: traj ok", t.num_timesteps(), flush=True)
arr_back = m.trajectory_to_numpy(t)
print("step 3: roundtrip ok", arr_back.shape, flush=True)
smoothed = m.moving_average_smooth(t, 5)
print("step 4: smooth ok", m.trajectory_to_numpy(smoothed).shape, flush=True)
metrics = m.compute_metrics(t)
print("step 5: metrics ok", metrics["max_velocity"], flush=True)
config = m.OptimizerConfig()
print("step 6: config ok", flush=True)
opt = m.TrajectoryOptimizer(config)
print("step 7: optimizer ok", flush=True)
res = opt.optimize(t, t)
print("step 8: optimize ok", res.final_cost, flush=True)
