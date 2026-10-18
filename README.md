# AGCD: Bridging Evolution and Learning: Attention-Guided Co-Design of Modular Soft Robots

This repository accompanies the paper [Bridging Evolution and Learning: Attention-Guided Co-Design of Modular Soft Robots](main.pdf). This README summarizes the method and provides the experimental settings needed for reproduction. The paper is the primary reference; hyperparameters omitted from its main text are supplemented from the appendix of [AGCD_IJCAI.tex](AGCD_IJCAI.tex).

## Method overview

AGCD jointly optimizes voxel robot morphologies and their controllers under a limited training budget. It combines:

- **Attention-guided variation**, which uses information learned by the controller to guide morphological crossover and mutation;
- **Staged evaluation and Training Level–Fitness Non-Dominated Sorting**, which compare candidates by both observed fitness and training effort, preserving opportunities for undertrained but promising morphologies; and
- **Elite training**, which completes the remaining controller training of selected archived individuals before choosing the best robot.

A Transformer controller integrates local voxel observations through self-attention and is trained with Proximal Policy Optimization (PPO). Attention matrices are averaged over a sampled trajectory and aggregated by column to obtain voxel scores. Crossover and mutation preferentially preserve high-scoring voxels while exploring other material assignments. Offspring must be connected and contain actuator voxels.

The framework alternates morphology search and controller learning, allowing information acquired during control to guide subsequent evolution.

## Task budgets

The paper evaluates six Evolution Gym tasks in a `5 x 5` voxel design space. Let `t` denote the maximum controller-training iterations per individual. Following the paper, total budgets are `100t`, `200t`, and `300t` for easy, medium, and hard tasks, respectively. Per-individual limits are supplied by the appendix.

One training iteration means **one PPO update**, collecting 128 simulator steps for one robot. Controller training is divided into 25 levels.

| Task | Difficulty | Evaluation allowance | Max. PPO updates per robot `t` | Total PPO-update budget | PPO updates per level |
| --- | --- | ---: | ---: | ---: | ---: |
| Walker-v0 | Easy | 100 | 500 | 50,000 | 20 |
| Jumper-v0 | Easy | 100 | 500 | 50,000 | 20 |
| Climber-v0 | Medium | 200 | 400 | 80,000 | 16 |
| Climber-v1 | Medium | 200 | 600 | 120,000 | 24 |
| Climber-v2 | Hard | 300 | 1,000 | 300,000 | 40 |
| Hurdler-v0 | Hard | 300 | 1,000 | 300,000 | 40 |

The evaluation allowance expresses the total budget in units of fully trained individuals; staged evaluation can explore more morphologies within that budget. Evaluation rollouts are separate from the PPO-update count. Where the older appendix's total-budget table differs, the totals above follow `main.pdf`.

## Core experimental parameters

### PPO

The following hyperparameters supplement the paper's experimental settings using the appendix parameter table.

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

The appendix's “Mini-batch size: 4” refers to four mini-batches per epoch. Additional optimizer and action-distribution settings are provided in [agcd/transformer/config.py](agcd/transformer/config.py).

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

During staged evolution, each active individual receives one training interval per evaluation. Elite training uses the reserved budget to complete the remaining intervals of promising archived candidates. An elite at level `l` receives `(25 - l) * T_level` additional updates to reach the training limit. The elite reserve is part of the total budget.

The appendix reports experiments on a Linux server with an Intel Xeon Gold 5220 CPU at 2.20 GHz, using CPU parallelism without a GPU.

## Experimental comparisons

The paper compares AGCD against **GA**, **BO**, and **AIEA**, using the same total training budget. Reward curves report the mean and standard deviation over five independent runs. AGCD achieves higher final mean rewards than the baselines on all six evaluated tasks.

Two ablations assess its main components:

- **AGCD-noA** replaces attention-guided crossover and mutation with the default variation operators used by the baselines.
- **AGCD-noL** removes staged evaluation and Training Level–Fitness Non-Dominated Sorting, evaluating each individual with its full training budget.

See the paper's comparison and ablation figures for the complete results.

## Environment setup

AGCD uses the modular observation and action spaces supplied by [ModularEvoGym](https://github.com/Yuxing-Wang-THU/ModularEvoGym), built on Evolution Gym. Follow the upstream [installation instructions](https://github.com/Yuxing-Wang-THU/ModularEvoGym#installation) to install its Python dependencies and build the simulator, including its native dependencies and submodules.

The locally verified environment uses Python 3.7.11, PyTorch 1.11.0+cu113, Gym 0.19.0, NumPy 1.21.5, and Stable-Baselines3 1.4.0. The simulator reports version 2.2.5. Controller training uses the CPU.

After installing ModularEvoGym, clone this repository and activate the configured environment:

```bash
git clone https://github.com/LittleDu26/AGCD.git
cd AGCD
conda activate modularevogym
```

Use your environment's name if it differs. Controller sources are included in this repository; the ModularEvoGym version of `evogym` must be installed separately.

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

The current launcher runs `Walker-v0` and `DownStepper-v0` once each, with seed 101 and eight parallel workers. Experiment settings are assigned after argument parsing in [run_AGCD.py](run_AGCD.py); edit those assignments and the task loop to configure the paper experiments.

For the paper's six tasks, set:

```python
env_list = [
    "Walker-v0", "Jumper-v0",
    "Climber-v0", "Climber-v1",
    "Climber-v2", "Hurdler-v0",
]
args.threads_num = 20
num = 5
```

Use the evaluation allowances in the task-budget table when calculating `args.max_iters = max_eva * tc`: 100 for easy tasks, 200 for medium tasks, and 300 for hard tasks. The existing `get_par` helper retains older budget settings, so override `max_eva` after `get_par(env)` to match the paper.

For five independent runs, assign a distinct seed inside the inner run loop before writing the configuration and starting training. For example, `args.seed = 101 + i` gives seeds 101–105; this is an example schedule, not a statement of the paper's original seeds.

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

Match structures and controllers by ID. Restore the saved observation-normalization statistics when evaluating a controller; [tests/smoke_agcd.py](tests/smoke_agcd.py) provides a reload-and-evaluate example.

Existing run directories are skipped. Move results you want to preserve, or configure a fresh output path before rerunning an experiment.

## Acknowledgments

This project uses [Evolution Gym](https://github.com/EvolutionGym/evogym) and [ModularEvoGym](https://github.com/Yuxing-Wang-THU/ModularEvoGym). The Transformer controller is based on ModularEvoGym's [transformer_ppo implementation](https://github.com/Yuxing-Wang-THU/ModularEvoGym/tree/fbc9b7d86f65001efd097ba28ad7f42542658a9e/examples/transformer_ppo) at commit `fbc9b7d86f65001efd097ba28ad7f42542658a9e`.
