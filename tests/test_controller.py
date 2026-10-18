"""Controller checks; optional numerical comparison against the original checkout."""
import argparse
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gym
import numpy as np
import torch
import evogym.envs
from evogym import get_full_connectivity

from agcd.envs import make_vec_envs
from agcd.transformer.config import ppoconfig, transformerconfig
from agcd.transformer.transformerPPOagent import PPOAgent, TransformerPPOAC
from utils.algo_utils import Structure

REFERENCE_ROOT = None


def make_robot(shape=(5, 5)):
    body = np.full(shape, 3, dtype=int)
    body[0, 0] = 0
    body[-1, -1] = 4
    return body, get_full_connectivity(body)


def make_controller(env, cls=TransformerPPOAC, trans_args=None):
    return cls(
        modular_state_dim=env.modular_state_dim,
        modular_action_dim=env.modular_action_dim,
        sequence_size=env.voxel_num,
        other_feature_size=env.other_dim,
        ppo_args=ppoconfig(),
        trans_args=trans_args or transformerconfig(),
        ac_type="transformer",
        device=torch.device("cpu"),
    )


class ControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_real_observations_and_gradients(self):
        for name in ("Walker-v0", "DownStepper-v0"):
            for shape in ((5, 5), (3, 4)):
                with self.subTest(env=name, shape=shape):
                    robot = make_robot(shape)
                    env = gym.make(name, body=robot[0], connections=robot[1],
                                   mode="modular", env_id=name)
                    vec = make_vec_envs(name, [Structure(*robot, 0)], 101,
                                        0.99, torch.device("cpu"))
                    try:
                        obs = vec.reset()
                        self.assertNotIn("body", obs)
                        agent = PPOAgent(make_controller(env))
                        val, act, logp, attention = agent.uni_act(obs, return_attention=True)
                        self.assertEqual(tuple(act.shape), (1, shape[0] * shape[1]))
                        self.assertEqual(tuple(attention.shape), (1, act.shape[1], act.shape[1]))
                        self.assertTrue(torch.isfinite(attention).all())
                        self.assertTrue((attention.masked_select(
                            obs["obs_mask"].bool().unsqueeze(1).expand_as(attention)) == 0).all())
                        obs, reward, _, _ = vec.step(act)
                        val, logp, entropy = agent(obs, act)
                        loss = val.square().mean() - logp.mean() - 0.01 * entropy
                        loss.backward()
                        gradients = [p.grad for p in agent.parameters() if p.grad is not None]
                        self.assertTrue(gradients)
                        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                        self.assertTrue(any(torch.count_nonzero(g) for g in gradients))
                    finally:
                        vec.close()
                        env.close()

    def test_sequence_resize(self):
        robot = make_robot()
        env = gym.make("Walker-v0", body=robot[0], mode="modular", env_id="Walker-v0")
        try:
            ac = make_controller(env)
            ac.reset_seq_size(9)
            self.assertEqual(ac.mu_net.seq_len, 9)
            self.assertEqual(tuple(ac.mu_net.pos_embedding.pe.shape), (9, 1, 64))
            obs = {
                "modular": torch.zeros(1, 9 * env.modular_state_dim),
                "other": torch.zeros(1, env.other_dim),
                "obs_mask": torch.zeros(1, 9),
                "act_mask": torch.zeros(1, 9),
            }
            val, pi, _ = ac(obs)
            self.assertEqual(tuple(pi.mode().shape), (1, 9))
            self.assertTrue(torch.isfinite(val).all())
        finally:
            env.close()

    def test_reference_parity(self):
        if REFERENCE_ROOT is None:
            self.skipTest("Use --reference-root to compare against ModularEvoGym")
        sys.path.insert(0, str(REFERENCE_ROOT / "examples"))
        from transformer_ppo.transformer.config import transformerconfig as ReferenceConfig
        from transformer_ppo.transformer.transformerPPOagent import TransformerPPOAC as ReferenceAC

        for name in ("Walker-v0", "DownStepper-v0"):
            with self.subTest(env=name):
                robot = make_robot()
                env = gym.make(name, body=robot[0], mode="modular", env_id=name)
                try:
                    raw = env.reset()
                    obs = {k: torch.as_tensor(v).float().unsqueeze(0).repeat(
                        3, *([1] * np.asarray(v).ndim)) for k, v in raw.items()}
                    ours = make_controller(env)
                    reference = make_controller(env, ReferenceAC, ReferenceConfig())
                    reference.load_state_dict(ours.state_dict(), strict=True)
                    val, pi, attention = ours(obs, return_attention=True)
                    ref_val, ref_pi, _, ref_attention = reference(obs, return_attention=True)
                    torch.testing.assert_close(val, ref_val, rtol=0, atol=0)
                    torch.testing.assert_close(pi.mode(), ref_pi.mode(), rtol=0, atol=0)
                    torch.testing.assert_close(pi.stddev, ref_pi.stddev, rtol=0, atol=0)
                    torch.testing.assert_close(attention, ref_attention[0], rtol=0, atol=0)
                    actions = pi.sample()
                    _, _, logp, entropy = ours(obs, actions)
                    _, _, ref_logp, ref_entropy = reference(obs, actions)
                    torch.testing.assert_close(logp, ref_logp, rtol=0, atol=0)
                    torch.testing.assert_close(entropy, ref_entropy, rtol=0, atol=0)
                finally:
                    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path)
    args, unittest_args = parser.parse_known_args()
    REFERENCE_ROOT = args.reference_root
    unittest.main(argv=[sys.argv[0]] + unittest_args)
