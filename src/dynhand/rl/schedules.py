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
