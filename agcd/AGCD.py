import os
import csv
import torch
from utils.MyUtils import *
from evogym import sample_robot, hashable, get_full_connectivity
from utils.Agent import Agent
from .Reporter import CustomReporter
from .transformer.config import transformerconfig, ppoconfig
import gym
from .transformer.transformerPPOagent import PPOAgent, TransformerPPOAC
from .ppo import PPO
import copy
import torch.multiprocessing as mlp
import evogym.envs
import utils.my_mp_group as mp
import json


def single_agent_fit(agent, ppo_args, trans_args, sample_setting, args, signal):
    # Seed
    device = torch.device("cpu")
    torch.set_num_threads(1)

    robot = (agent.robot, get_full_connectivity(agent.robot))

    if agent.controller is not None:
        ppoAgent = agent.controller
    else:
        # Init PPO Actor-Critic
        actor_critic = TransformerPPOAC(modular_state_dim=sample_setting[0], modular_action_dim=sample_setting[1],
                                        sequence_size=sample_setting[3], other_feature_size=sample_setting[2],
                                        ppo_args=ppo_args, trans_args=trans_args, ac_type="transformer", device=device)
        # ppoagent
        ppoAgent = PPOAgent(actor_critic=actor_critic)
        agent.controller = ppoAgent

    ppoAgent.ac.to(device)

    # PPO
    ppo = PPO(robot=robot, ppo_size=1, train_iters=args.train_iters, agent=ppoAgent, verbose=True,
              ppo_args=ppo_args, total_step=args.total_step, save_path=args.save_to, device=device,
              signal=signal, maturity=agent.maturity,use_lfp=args.use_lfp)

    if agent.iteration==0:
        reward_history, agent.iteration, attention,agent.initial_fitness= ppo.train(agent.id, agent.iteration, agent.fitness)
    else:
        reward_history, agent.iteration, attention,_ = ppo.train(agent.id, agent.iteration,agent.fitness)

    if reward_history[-1] > agent.fitness:
        agent.fitness = reward_history[-1]
        agent.attention = attention
        agent.best_controller = ppo.best_controller

    agent.history_fitness += reward_history
    agent.maturity += len(reward_history)
    # ppoAgent.ac.to("cpu")
    return agent


# def count_elites(total_maturity):
#     alloc_list = [1] * (total_maturity - 1)
#     count = (total_maturity * (total_maturity - 1)) // 2
#     done = False
#     while not done:
#         for i in range(len(alloc_list)):
#             count += total_maturity - (i + 1)
#             alloc_list[i]+=1
#             if count % total_maturity == 0:
#                 done = True
#                 break
#     return alloc_list,count

def count_elites(total_maturity):
    alloc_list = [1] * (total_maturity - 1)
    count = (total_maturity * (total_maturity - 1)) // 2
    return alloc_list, count


def make_dir(dir):
    os.makedirs(os.path.join(dir, "structures"))
    os.makedirs(os.path.join(dir, "controllers"))


def muti_running(agents, ppo_args, trans_args, sample_setting, args, signal, terminal_condition):
    global current_iters
    # 评价初代种群
    group = mp.Group()
    maturity_agents = []
    for agent in agents:
        if agent.finished:
            maturity_agents.append(agent)
        else:
            if signal == "evolution" and current_iters + terminal_condition == args.max_iters:
                continue
            if signal == "evolution":
                if args.use_lfp:
                    current_iters += args.train_iters
                else:
                    current_iters += args.total_step
            all_args = (agent, ppo_args, trans_args, sample_setting, args, signal)
            group.add_job(single_agent_fit, all_args)
            robot = (agent.robot, get_full_connectivity(agent.robot))
            temp_path_body = os.path.join(args.save_to, "structures", str(agent.id))
            if agent.maturity==0:
                np.savez(temp_path_body, robot[0], robot[1])
    return maturity_agents + group.run_jobs(args.threads_num)


def get_elites(all_agents, alloc_list):
    age_layers = {}
    for agent in all_agents.values():
        if agent.finished is False:
            age_layers.setdefault(agent.maturity, []).append(agent)
    elites = []
    count_less = 0
    for maturity in age_layers.keys():
        layer_agents = age_layers[maturity]
        layer_agents.sort(key=lambda a: a.fitness, reverse=True)
        alloc = alloc_list[maturity - 1]
        save_list = layer_agents[:alloc]
        if len(save_list) < alloc:
            count_less += maturity
        elites.extend(save_list)
    return elites


def run(args):
    global current_iters
    mlp.set_start_method('spawn', force=True)

    logger = CustomReporter(args.save_to)

    csv_file = open(args.save_to + "/table.csv", "w")
    csv_logger = csv.DictWriter(csv_file, fieldnames=('id', 'maturity', 'fit'))
    csv_logger.writeheader()

    trans_args = transformerconfig()
    ppo_args = ppoconfig()
    ppo_args.env_name = args.env
    ppo_args.seed = args.seed
    ppo_args.eval_interval = max(1, args.total_step // 100)

    # 检查机器人
    structure_shape = (args.target_size, args.target_size)
    # 记录所有个体的哈希表
    record = {}
    pop_agent = []
    all_agents = {}
    # 重置Agent
    Agent._id_counter = 1
    generation = 0
    current_iters = 0
    fitness_window = []

    alloc_list, count = count_elites(args.total_maturity)
    if args.use_lfp:
        terminal_condition = count * args.train_iters
    else:
        terminal_condition = 0

    # Set dimensions
    robots = sample_robot(structure_shape)
    train_env = gym.make(args.env, mode='modular', body=robots[0], connections=robots[1], env_id=args.env)
    modular_state_dim = train_env.modular_state_dim
    modular_action_dim = train_env.modular_action_dim
    other_feature_size = train_env.other_dim
    sequence_size = train_env.voxel_num
    train_env.close()
    sample_setting = [modular_state_dim, modular_action_dim, other_feature_size, sequence_size]
    clip_num = args.pop_size // 2
    make_dir(args.save_to)
    # 初始化
    for i in range(args.pop_size):
        robot, _ = sample_robot(structure_shape)
        while (hashable(robot) in record or not eval_robot_constraint(robot)):
            robot, _ = sample_robot(structure_shape)
        pop_agent.append(Agent(robot))
        record[hashable(robot)] = []

    while True:
        logger.start_generation(generation)
        agent_list = muti_running(pop_agent, ppo_args, trans_args, sample_setting, args, "evolution",
                                  terminal_condition)

        pop_agent = copy.deepcopy(agent_list)  # 拷贝
        for agent in pop_agent:
            if agent.finished is False:
                if agent.maturity == 1:
                    agent.birth = generation
                all_agents[agent.id] = agent
                fitness_window.append(agent)
                if len(fitness_window) == args.total_maturity:
                    best_agent = get_best_agent(fitness_window)
                    csv_content = {"id": best_agent.id, "maturity": best_agent.maturity, "fit": best_agent.fitness}
                    csv_logger.writerow(csv_content)
                    csv_file.flush()
                    fitness_window = []
                if agent.maturity == args.total_maturity:
                    agent.finished = True

        pop_agent.sort(key=lambda agent: agent.fitness, reverse=True)

        if current_iters + terminal_condition == args.max_iters:
            logger.start_elites(pop_agent)
            break

        if args.use_lfp:
            # 非支配排序
            fronts = fast_non_dominated_sort(pop_agent)

            Parents = [agent for front in fronts for agent in front][:clip_num]

            Parents.sort(key=lambda agent: agent.fitness, reverse=True)
        else:
            Parents = pop_agent[:clip_num]


        child_num = args.pop_size - len(Parents)

        # 关闭注意力引导时，仅通过随机变异生成子代
        crossover_children = []
        if args.use_attention:
            crossover_children = random_guided_crossover(Parents, record, total_crossovers=child_num // 2)

        # 突变操作
        mutated_children = random_guided_mutate(Parents, record, total_mutations=child_num - len(crossover_children),
                                                use_attention=args.use_attention)

        logger.end_generation(pop_agent, crossover_children, mutated_children, Parents)
        # 合并
        total_children = mutated_children + crossover_children

        pop_agent = Parents + total_children

        generation += 1

    if args.use_lfp:
        # elites train
        elites = get_elites(all_agents, alloc_list)
        elites_windows = []
        agent_list = muti_running(elites, ppo_args, trans_args, sample_setting, args, "elites", terminal_condition)
        pop_agent = copy.deepcopy(agent_list)  # 拷贝
        for agent in pop_agent:
            start_age = all_agents[agent.id].maturity
            fit_list = agent.history_fitness[start_age:]
            all_agents[agent.id] = agent
            for index, fit in enumerate(fit_list):
                elites_windows.append((agent.id, start_age + index + 1, fit))
                if len(elites_windows) == args.total_maturity:
                    best_fit = max(elites_windows, key=lambda fit: fit[2])
                    csv_content = {"id": best_fit[0], "maturity": best_fit[1], "fit": best_fit[2]}
                    csv_logger.writerow(csv_content)
                    csv_file.flush()
                    elites_windows = []
            if agent.maturity == args.total_maturity:
                agent.finished = True
    all_list = list(all_agents.values())
    all_list.sort(key=lambda a: a.fitness, reverse=True)
    finally_agent = all_list[:20]
    logger.end_elites(finally_agent)
    # 保存 all_list 为 json
    json_path = os.path.join(args.save_to, "all_agents.json")
    all_agents_json = [[agent.id, agent.birth, agent.maturity, agent.history_fitness] for agent in all_list]

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_agents_json, f, ensure_ascii=False, indent=2)