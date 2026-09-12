"""Sample mixing between online experience and demonstrations."""

import numpy as np


def demo_ratio(step: int, start: float, anneal_steps: int) -> float:
    """Linearly anneal the demo minibatch fraction from start to zero."""
    if anneal_steps <= 0 or start <= 0.0:
        return 0.0
    fraction = min(1.0, step / anneal_steps)
    return start * (1.0 - fraction)


def sample_mixed(
    main_buffer,
    demo_buffer,
    batch_size: int,
    ratio: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Draw a minibatch with ratio fraction demo transitions and the rest online."""
    del rng
    n_demo = int(round(batch_size * ratio))
    n_main = batch_size - n_demo
    parts = []
    if n_main > 0:
        parts.append(main_buffer.sample(n_main))
    if n_demo > 0:
        parts.append(demo_buffer.sample(n_demo))
    if len(parts) == 1:
        return parts[0]
    return {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}


def sample_three(
    online_buffer,
    demo_buffer,
    synthetic_buffer,
    batch_size: int,
    demo_ratio_value: float,
    synthetic_ratio: float,
) -> dict[str, np.ndarray]:
    """Sample online, demonstration, and synthetic transitions."""
    n_demo = int(round(batch_size * demo_ratio_value)) if demo_buffer else 0
    remaining = batch_size - n_demo
    n_synthetic = int(round(remaining * synthetic_ratio)) if synthetic_buffer else 0
    n_online = batch_size - n_demo - n_synthetic
    parts = []
    if n_online:
        parts.append(online_buffer.sample(n_online))
    if n_demo:
        parts.append(demo_buffer.sample(n_demo))
    if n_synthetic:
        parts.append(synthetic_buffer.sample(n_synthetic))
    return {
        key: np.concatenate([part[key] for part in parts], axis=0) for key in parts[0]
    }
