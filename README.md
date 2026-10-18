# AGCD: Bridging Evolution and Learning: Attention-Guided Co-Design of Modular Soft Robots

We thank the reviewers for their time and thoughtful consideration of our work. Welcome to the AGCD supplementary repository. This README presents the experimental settings and instructions for reproducing our experiments.


## Task budgets

We evaluate AGCD on six Evolution Gym tasks in a `5 x 5` voxel design space. Let `t` denote the maximum controller-training iterations per individual. Total budgets are `100t`, `200t`, and `300t` for easy, medium, and hard tasks, respectively.

One training iteration means **one PPO update**, collecting 128 simulator steps for one robot. Controller training is divided into 25 levels.

| Task | Difficulty | Evaluation allowance | Max. PPO updates per robot `t` | Total PPO-update budget | PPO updates per level |
| --- | --- | ---: | ---: | ---: | ---: |
| Walker-v0 | Easy | 100 | 500 | 50,000 | 20 |
| Jumper-v0 | Easy | 100 | 500 | 50,000 | 20 |
| Climber-v0 | Medium | 200 | 400 | 80,000 | 16 |
| Climber-v1 | Medium | 200 | 600 | 120,000 | 24 |
| Climber-v2 | Hard | 300 | 1,000 | 300,000 | 40 |
| Hurdler-v0 | Hard | 300 | 1,000 | 300,000 | 40 |

The evaluation allowance expresses the total budget in units of fully trained individuals; staged evaluation can explore more morphologies within that budget. Evaluation rollouts are separate from the PPO-update count.

## Core experimental parameters

### PPO

All compared methods use the same PPO hyperparameters.

| Parameter | Value |
| --- | --- |
| Optimizer | Adam |
| Learning rate | `2.5e-4`, linearly decayed |
| Adam epsilon | `1e-5` |
| Rollout length | 128 simulator steps per PPO update |
| Optimization epochs | 4 |
| Mini-batches per epoch | 4 |
| Discount factor (gamma) | 0.99 |
| GAE parameter (lambda) | 0.95 |
| PPO clipping coefficient | 0.1 |
| Entropy coefficient | 0.01 |
| Value-loss coefficient | 0.5 |
| Maximum gradient norm | 0.5 |
| Value-function clipping | Enabled |
| Fixed action standard deviation | 0.2 |

Each PPO epoch uses four mini-batches. Additional optimizer and action-distribution settings are provided in [agcd/transformer/config.py](agcd/transformer/config.py).

### Transformer controller

| Parameter | Value |
| --- | --- |
| Transformer layers | 1 |
| Attention heads | 1 |
| Embedding dimension | 64 |
| Feed-forward dimension | 128 |
| Transformer activation | ReLU |
| Dropout | 0.0 |
| Position encoding | Learned sequence embeddings |
| Action distribution | Gaussian with fixed diagonal covariance |

The controller follows ModularEvoGym's Transformer policy, combining local voxel information with task-related observations to produce coordinated actuator actions.

### Evolution and training allocation

| Parameter | Value |
| --- | --- |
| Design space | `5 x 5` voxels |
| Evolution strategy | `(mu + lambda)`, with `mu = 10`, `lambda = 10` |
| Population size | 20 |
| Survivors per generation | 10 |
| Crossover / mutation offspring | 5 / 5 |
| Maximum training level | 25 |
| PPO updates per level | `t / 25` |
| Selection objectives | Higher fitness and lower training level |
| Ordering within a non-dominated front | Descending fitness |
| Attention threshold | Median score over non-empty voxels |
| Parallel workers | 20 |
| Training device | CPU |
| Independent runs per task | 5 |

Experiments run on a Linux server with an Intel Xeon Gold 5220 CPU at 2.20 GHz, using CPU parallelism without a GPU.

## Experimental comparisons

We compare AGCD against **GA**, **BO**, and **AIEA**, using the same total training budget. Reward curves report the mean and standard deviation over five independent runs. AGCD achieves higher final mean rewards than the baselines on all six evaluated tasks.

Two ablations assess its main components:

- **AGCD-noA** replaces attention-guided crossover and mutation with the default variation operators used by the baselines.
- **AGCD-noL** removes staged evaluation and Training Level–Fitness Non-Dominated Sorting, evaluating each individual with its full training budget.

## Environment setup

**Install the environment from [ModularEvoGym](https://github.com/Yuxing-Wang-THU/ModularEvoGym) before running AGCD.** Follow that repository's [installation instructions](https://github.com/Yuxing-Wang-THU/ModularEvoGym#installation) to install the Python dependencies and build the simulator, including its native dependencies and submodules.

We use the **Transformer-based controller provided by ModularEvoGym**, together with its modular observation and action spaces. The controller is trained with Proximal Policy Optimization (PPO). Its source is included here under `agcd/transformer/`; the ModularEvoGym simulator must be installed through the upstream repository.

After installing ModularEvoGym, clone this repository and activate the configured environment:

```bash
git clone https://github.com/LittleDu26/AGCD.git
cd AGCD
conda activate modularevogym
```

Use your environment's name if it differs.

## Verify the environment

Run the controller checks:

```bash
python tests/test_controller.py -v
```

These cover modular observations, masks, different robot sizes, and gradients. An optional comparison with the original controller accepts a ModularEvoGym checkout:

```bash
python tests/test_controller.py --reference-root ../ModularEvoGym -v
```

To check training, evolution, elite training, and checkpoint reloading with a small budget:

```bash
python tests/smoke_agcd.py
```

The smoke check uses reduced training settings and writes to a temporary directory printed at startup. It verifies the pipeline rather than convergence. To select one task and a fresh output directory:

```bash
python tests/smoke_agcd.py --env Walker-v0 --output-root result/my-smoke-run
```

## Run AGCD

Run commands from the repository root:

```bash
python run_AGCD.py
```

The current launcher runs `Walker-v0` and `DownStepper-v0` once each, with seed 101 and eight parallel workers. Experiment settings are assigned after argument parsing in [run_AGCD.py](run_AGCD.py); edit those assignments and the task loop to configure experiments.

For the six benchmark tasks, set:

```python
env_list = [
    "Walker-v0", "Jumper-v0",
    "Climber-v0", "Climber-v1",
    "Climber-v2", "Hurdler-v0",
]
args.threads_num = 20
num = 5
```

Use the evaluation allowances in the task-budget table when calculating `args.max_iters = max_eva * tc`: 100 for easy tasks, 200 for medium tasks, and 300 for hard tasks. Set `max_eva` after `get_par(env)` to use these allowances.

For five independent runs, assign a distinct seed inside the inner run loop before writing the configuration and starting training. For example, `args.seed = 101 + i` gives seeds 101–105.

The baseline entry points are [run_GA.py](run_GA.py), [run_BO.py](run_BO.py), and [run_AIEA.py](run_AIEA.py). Apply the same task budgets and independent-run protocol when configuring comparisons. BO additionally requires GPy and GPyOpt, as described in the upstream installation instructions.

## Outputs

By default, training results are written to:

```text
result/AGCD/<environment>/<run_index>/
```

| File or directory | Contents |
| --- | --- |
| `config.json` | Experiment settings |
| `out.txt` | Evolution and elite-training logs |
| `table.csv` | Fitness summaries with individual IDs and training levels |
| `structures/<id>.npz` | Robot body and connectivity |
| `controllers/<id>.pt` | Best saved controller and observation-normalization statistics |
| `all_agents.json` | Individual IDs, birth generations, training levels, and fitness histories |
| `gifs/<id>.gif` | Rollout animations exported with `simple_gif.py` |

## Visualize results

Use [simple_gif.py](simple_gif.py) to render a trained robot for one complete episode and export a GIF:

```bash
python simple_gif.py --run-dir result/AGCD/Walker-v0/0 --id 382
```

Replace `382` with an individual ID from your run's `all_agents.json` or `table.csv`. The matching `controllers/<id>.pt` and `structures/<id>.npz` files must exist. The script reads the environment and seed from `config.json`, restores the saved observation-normalization statistics, and evaluates the controller using mean actions on the CPU.

The example saves `result/AGCD/Walker-v0/0/gifs/382.gif` and prints the episode reward. You can supply multiple IDs after `--id`, select a destination with `--output-dir`, or change playback speed with `--fps` (default: 50). If `config.json` is unavailable, specify the task with `--env`.

Rendering requires an X display. On a headless Ubuntu server, use Xvfb:

```bash
sudo apt-get install -y xvfb xauth
xvfb-run -a python simple_gif.py --run-dir result/AGCD/Walker-v0/0 --id 382
```

GIF compression runs automatically when `gifsicle` is installed; it is optional:

```bash
sudo apt-get install -y gifsicle
```
