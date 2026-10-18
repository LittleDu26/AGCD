"""Export one deterministic robot rollout from an AGCD run as a GIF."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import imageio
import numpy as np
import torch

from agcd.envs import make_vec_envs
import agcd.helper as helper
from utils.algo_utils import Structure

ROOT = Path(__file__).resolve().parent


def save_robot_gif(env_name, id, run_dir=None, output_dir=None, seed=101, fps=50):
    """Load matching structure/controller files, render an episode, and return reward."""
    if fps <= 0:
        raise ValueError("FPS must be positive")
    run_dir = Path(run_dir) if run_dir is not None else ROOT / "result" / "AGCD" / env_name / "0"
    controller_path = run_dir / "controllers" / (str(id) + ".pt")
    structure_path = run_dir / "structures" / (str(id) + ".npz")
    for path in (controller_path, structure_path):
        if not path.is_file():
            raise FileNotFoundError("Missing robot file: " + str(path))
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        raise RuntimeError("Rendering requires an X display. On a headless server, run with xvfb-run -a.")

    device = torch.device("cpu")
    torch.set_num_threads(1)
    with np.load(str(structure_path)) as data:
        robot = Structure(data["arr_0"], data["arr_1"], id)
    uni_agent, obs_rms = torch.load(str(controller_path), map_location=device)
    uni_agent.to(device)
    uni_agent.eval()
    uni_agent.ac.device = device

    env = make_vec_envs(env_name, [robot], seed=seed, gamma=None, device=device, ret=False, ob=True)
    try:
        vec_norm = helper.get_vec_normalize(env)
        if vec_norm is not None:
            vec_norm.eval()
            vec_norm.ob_rms = obs_rms
        obs = env.reset()
        output_dir = Path(output_dir) if output_dir is not None else run_dir / "gifs"
        output_dir.mkdir(parents=True, exist_ok=True)
        gif_path = output_dir / (str(id) + ".gif")
        frames = 0
        # Stream frames to disk rather than retaining the whole episode in memory.
        with imageio.get_writer(str(gif_path), mode="I", duration=1.0 / fps, loop=0) as writer:
            while True:
                # Capture before stepping: the vector environment resets at episode end.
                writer.append_data(env.render(mode="img"))
                frames += 1
                with torch.no_grad():
                    _, action, _ = uni_agent.uni_act(obs, mean_action=True)
                obs, _, done, infos = env.step(action)
                if bool(done[0]):
                    reward = float(infos[0]["episode"]["r"])
                    break
    finally:
        env.close()

    if shutil.which("gifsicle"):
        try:
            subprocess.run(["gifsicle", "-O3", "--batch", str(gif_path)], check=True)
        except subprocess.CalledProcessError:
            print("Optional GIF compression failed; the uncompressed GIF is available.")
    print("Robot {} | Reward: {:.6f} | Frames: {}".format(id, reward, frames), flush=True)
    print("GIF saved to: " + str(gif_path.resolve()), flush=True)
    return reward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="AGCD run containing controllers/ and structures/")
    parser.add_argument("--id", type=int, nargs="+", required=True, help="one or more robot IDs to visualize")
    parser.add_argument("--env", help="environment name; defaults to config.json in the run directory")
    parser.add_argument("--seed", type=int, help="evaluation seed; defaults to the run configuration, then 101")
    parser.add_argument("--output-dir", type=Path, help="GIF destination; defaults to <run-dir>/gifs/")
    parser.add_argument("--fps", type=int, default=50, help="GIF playback frames per second (default: 50)")
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")
    args.run_dir = args.run_dir.expanduser().resolve()
    config_path = args.run_dir / "config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    env_name = args.env or config.get("env")
    if not env_name:
        parser.error("provide --env when config.json does not specify an environment")
    seed = args.seed if args.seed is not None else config.get("seed", 101)
    for robot_id in args.id:
        save_robot_gif(env_name, robot_id, run_dir=args.run_dir,
                       output_dir=args.output_dir, seed=seed, fps=args.fps)


if __name__ == "__main__":
    main()
