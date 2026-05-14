"""Smoke tests for ``rl.replay_buffer`` and ``rl.sac``.

These cover only pure-numeric logic (no MuJoCo / no RCS env):

- the replay buffer round-trips transitions and samples a batch of the
  requested shape;
- the SAC agent runs a forward act and a single ``update`` step without
  raising, with the loss being finite afterwards;
- save / load preserves actor weights.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np


class ReplayBufferTest(unittest.TestCase):
    def test_round_trip_and_sample_shapes(self):
        from rl.replay_buffer import ReplayBuffer

        buf = ReplayBuffer(capacity=64, obs_dim=11, act_dim=4)
        rng = np.random.default_rng(0)
        for _ in range(20):
            obs = rng.standard_normal(11).astype(np.float32)
            action = rng.standard_normal(4).astype(np.float32)
            next_obs = rng.standard_normal(11).astype(np.float32)
            buf.add(obs, action, reward=float(rng.standard_normal()), next_obs=next_obs, done=False)
        self.assertEqual(len(buf), 20)
        batch = buf.sample(8, device="cpu")
        self.assertEqual(tuple(batch.obs.shape), (8, 11))
        self.assertEqual(tuple(batch.actions.shape), (8, 4))
        self.assertEqual(tuple(batch.rewards.shape), (8,))
        self.assertEqual(tuple(batch.next_obs.shape), (8, 11))
        self.assertEqual(tuple(batch.dones.shape), (8,))

    def test_wraparound(self):
        from rl.replay_buffer import ReplayBuffer

        buf = ReplayBuffer(capacity=4, obs_dim=2, act_dim=1)
        for i in range(7):
            buf.add(
                obs=np.array([i, i], dtype=np.float32),
                action=np.array([i], dtype=np.float32),
                reward=float(i),
                next_obs=np.array([i + 1, i + 1], dtype=np.float32),
                done=False,
            )
        self.assertEqual(len(buf), 4)


class SACAgentSmokeTest(unittest.TestCase):
    def test_act_and_update_runs(self):
        import torch

        from rl.replay_buffer import ReplayBuffer
        from rl.sac import SACAgent, SACConfig

        torch.manual_seed(0)
        agent = SACAgent(obs_dim=11, act_dim=4, config=SACConfig(hidden_sizes=(32, 32)))
        buf = ReplayBuffer(capacity=128, obs_dim=11, act_dim=4)

        rng = np.random.default_rng(0)
        for _ in range(64):
            obs = rng.standard_normal(11).astype(np.float32)
            action = agent.act(obs)
            self.assertEqual(action.shape, (4,))
            self.assertTrue(np.all(np.abs(action) <= 1.0 + 1e-5))
            next_obs = rng.standard_normal(11).astype(np.float32)
            buf.add(obs, action.astype(np.float32), reward=float(rng.standard_normal()),
                    next_obs=next_obs, done=False)

        batch = buf.sample(32, device="cpu")
        metrics = agent.update(batch)
        for key in ("critic_loss", "actor_loss", "alpha_loss", "alpha", "entropy"):
            self.assertIn(key, metrics)
            self.assertTrue(np.isfinite(metrics[key]), msg=f"{key} not finite: {metrics[key]}")

    def test_save_and_load_round_trip(self):
        import torch

        from rl.sac import SACAgent, SACConfig, load_agent_for_inference

        torch.manual_seed(0)
        agent = SACAgent(obs_dim=6, act_dim=2, config=SACConfig(hidden_sizes=(16, 16)))

        rng = np.random.default_rng(1)
        obs = rng.standard_normal(6).astype(np.float32)
        original_action = agent.act(obs, deterministic=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "sac.pt"
            agent.save(ckpt_path)
            loaded = load_agent_for_inference(ckpt_path, device="cpu")

        loaded_action = loaded.act(obs, deterministic=True)
        np.testing.assert_allclose(loaded_action, original_action, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
