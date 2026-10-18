"""Run two real AGCD generations and elite training with a small test budget."""
import argparse
import importlib
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

agcd = importlib.import_module("agcd.AGCD")
from agcd.evaluate import evaluate
from agcd.ppo import PPO
from agcd.transformer.config import ppoconfig
from utils.algo_utils import Structure
from utils.MyUtils import get_similar_index

ORIGINAL_FIT = agcd.single_agent_fit


class SmokePPOConfig(ppoconfig):
    def __init__(self):
        super().__init__()
        self.TIMESTEPS = 16
        self.EPOCHS = 1
        self.NUM_MINI_BATCH_SIZE = 2


def checked_single_agent_fit(*args):
    """Assertions run inside the real spawned training workers."""
    original_batch = PPO.train_on_batch
    updates = []

    def checked_batch(ppo):
        before = [p.detach().clone() for p in ppo.agent.parameters()]
        original_batch(ppo)
        after = list(ppo.agent.parameters())
        assert all(torch.isfinite(p).all() for p in after)
        assert all(torch.isfinite(p.grad).all() for p in after if p.grad is not None)
        assert any(not torch.equal(a, b) for a, b in zip(before, after)), "No PPO update"
        updates.append(1)

    PPO.train_on_batch = checked_batch
    try:
        agent = ORIGINAL_FIT(*args)
        assert len(updates) == 2, len(updates)
        assert np.isfinite(agent.fitness)
        assert agent.best_controller is not None
        if args[-1] == "evolution":
            assert agent.attention.shape == (agent.robot.size,)
            assert np.isfinite(agent.attention).all()
        checkpoint = Path(args[-2].save_to) / "controllers" / (str(agent.id) + ".pt")
        saved, _ = torch.load(str(checkpoint), map_location="cpu")
        for key, value in agent.best_controller.state_dict().items():
            torch.testing.assert_close(value, saved.state_dict()[key], rtol=0, atol=0)
        return agent
    finally:
        PPO.train_on_batch = original_batch


def check_inheritance(operation, counts, key):
    def checked(parents, *args, **kwargs):
        children = operation(parents, *args, **kwargs)
        for child in children:
            parent = parents[get_similar_index(child, parents)]
            assert child.controller is not parent.best_controller
            for source, inherited in zip(parent.best_controller.parameters(), child.controller.parameters()):
                assert source.data_ptr() != inherited.data_ptr()
                torch.testing.assert_close(source, inherited, rtol=0, atol=0)
        counts[key] += len(children)
        return children
    return checked


def run_smoke(output_root, names):
    original_config = agcd.ppoconfig
    original_crossover = agcd.random_guided_crossover
    original_mutate = agcd.random_guided_mutate
    agcd.ppoconfig = SmokePPOConfig
    agcd.single_agent_fit = checked_single_agent_fit
    torch.set_num_threads(1)
    try:
        for name in names:
            random.seed(101)
            np.random.seed(101)
            torch.manual_seed(101)
            counts = {"crossover": 0, "mutation": 0}
            agcd.random_guided_crossover = check_inheritance(original_crossover, counts, "crossover")
            agcd.random_guided_mutate = check_inheritance(original_mutate, counts, "mutation")
            destination = output_root / name
            destination.mkdir(parents=True)
            args = SimpleNamespace(
                env=name, seed=101, target_size=5, threads_num=2, pop_size=4,
                total_maturity=2, train_iters=2, total_step=4, max_iters=18,
                save_to=str(destination), use_attention=True, use_lfp=True,
            )
            (destination / "config.json").write_text(json.dumps(vars(args), indent=2))
            agcd.run(args)
            assert counts["crossover"] >= 1, counts
            assert counts["mutation"] >= 1, counts
            rows = json.loads((destination / "all_agents.json").read_text())
            assert len(rows) == 6, rows
            assert all(row[2] in (1, 2) and len(row[3]) == row[2] for row in rows)
            assert sum(row[2] == 2 for row in rows) == 3, "Elite training did not complete"
            assert all(np.isfinite(row[3]).all() for row in rows)
            log = (destination / "out.txt").read_text()
            assert "Running generation 0" in log and "Running generation 1" in log
            assert "Elites training" in log
            assert len((destination / "table.csv").read_text().splitlines()) >= 2
            best_id = rows[0][0]
            agent, ob_rms = torch.load(str(destination / "controllers" / (str(best_id) + ".pt")),
                                      map_location="cpu")
            agent.eval()
            agent.ac.device = torch.device("cpu")
            with np.load(str(destination / "structures" / (str(best_id) + ".npz"))) as data:
                robot = Structure(data["arr_0"], data["arr_1"], best_id)
            reward, attention = evaluate(1, agent, ob_rms, name, [robot], seed=101,
                                         device=torch.device("cpu"), return_attention=True)
            assert np.isfinite(reward) and np.isfinite(attention).all()
            summary = dict(env=name, agents=len(rows), generations=2, elite_completed=True,
                           inherited_children=counts, reloaded_reward=float(reward))
            (destination / "verification.json").write_text(json.dumps(summary, indent=2))
            print("PASS " + json.dumps(summary), flush=True)
    finally:
        agcd.ppoconfig = original_config
        agcd.single_agent_fit = ORIGINAL_FIT
        agcd.random_guided_crossover = original_crossover
        agcd.random_guided_mutate = original_mutate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--env", nargs="+", default=["Walker-v0", "DownStepper-v0"])
    args = parser.parse_args()
    output_root = args.output_root or Path(tempfile.mkdtemp(prefix="agcd-smoke-"))
    print("Verification output: " + str(output_root), flush=True)
    run_smoke(output_root, args.env)
