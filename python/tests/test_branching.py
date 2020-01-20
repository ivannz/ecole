import pytest

import ecole.branching


@pytest.mark.slow
def test_branching_env(model):
    env = ecole.branching.Env.make_dummy()
    for _ in range(2):
        count = 0
        obs, done = env.reset(model.clone())
        while not done:
            obs, reward, done, info = env.step(0)
            count += 1
        assert count > 0
