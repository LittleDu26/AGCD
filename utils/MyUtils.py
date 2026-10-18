from evogym import is_connected, has_actuator,draw,get_uniform
from utils.algo_utils import heuristic_crossover,mutate
from evogym import hashable
import random
import numpy as np
import copy
from .Agent import Agent
def eval_robot_constraint(robot):
    validity = is_connected(robot) and has_actuator(robot)
    return validity

def fast_non_dominated_sort(agents):
    fronts = [[]]
    domination_count = {}
    dominated_solutions = {}

    for p in agents:
        dominated_solutions[p] = []
        domination_count[p] = 0
        for q in agents:
            if p == q:
                continue
            if dominates(p, q):
                dominated_solutions[p].append(q)
            elif dominates(q, p):
                domination_count[p] += 1

        if domination_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while i < len(fronts):
        next_front = []
        for p in fronts[i]:
            for q in dominated_solutions[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        if next_front:
            fronts.append(next_front)
        i += 1

    
    for front in fronts:
            front.sort(key=lambda a: a.fitness,reverse=True)

    return fronts


def dominates(a, b):
    better_or_equal = (a.fitness >= b.fitness) and (a.maturity <= b.maturity)
    strictly_better = (a.fitness > b.fitness) or (a.maturity < b.maturity)
    return better_or_equal and strictly_better

def keep_highest_n_agents(fronts, n):
    total_kept = 0
    new_fronts = []

    for front in fronts:
        if total_kept + len(front) <= n:
            new_fronts.append(front[:]) 
            total_kept += len(front)
        else:
            
            num_to_keep = n - total_kept
            new_fronts.append(front[:num_to_keep])
            break  

    return new_fronts


def print_fronts(fronts):
    for i, front in enumerate(fronts):
        print(f"Front {i}:")
        for agent in front:
            print(f"  Fitness: {agent.fitness:.2f}, Maturity: {agent.maturity}")

def get_fitness_median(new_fronts):
    fitness_list = [agent.fitness for front in new_fronts for agent in front if agent.fitness is not None]
    median = np.median(fitness_list)
    return median

def get_similar_index(ind, off):
    ind_structure = np.array(ind.robot)
    off_structures = np.array([agent.robot for agent in off])

    dif = off_structures - ind_structure 
    zero_num = np.zeros(dif.shape[0])     
    for i in range(len(zero_num)):
        zero_num[i] = np.sum(dif[i] == 0)
    index = np.argmax(zero_num)  
    return index


def mutate_top(fronts, record, total_mutations=10):
    new_fronts = copy.deepcopy(fronts)
    pool = [agent for front in new_fronts for agent in front]
    top_layer = new_fronts[0]
    rest_agents = [agent for front in new_fronts[1:] for agent in front]
    random.shuffle(rest_agents)
    agents_to_mutate = top_layer + rest_agents
    mutated_children = []

    for agent in agents_to_mutate:
        child_robot = mutate(agent.robot, record)
        if child_robot is not None:
            new_agent = Agent(child_robot)
            partner_index = get_similar_index(new_agent, pool)
            agent.controller = pool[partner_index].controller
            mutated_children.append(new_agent)
        if len(mutated_children) >= total_mutations:
            break
    return mutated_children


def attention_mutate(agent, record, mutation_rate=0.1, num_attempts=25):

    flat_robot = agent.robot.flatten()
    attention = agent.attention
    robot_shape = agent.robot.shape

    nonzero_indices = [i for i, val in enumerate(flat_robot) if val != 0]
    attention_nonzero = attention[nonzero_indices]
    attention_median = np.median(attention_nonzero)

   
    pd = get_uniform(5)
    pd[0] = 0.6  

   
    for n in range(num_attempts):
        flat_mutated = copy.deepcopy(flat_robot)
        for i in range(len(flat_mutated)):
            
            if attention[i] < attention_median:
                mutation = [mutation_rate, 1 - mutation_rate]
                if draw(mutation) == 0:  
                    flat_mutated[i] = draw(pd)

        mutated_robot = flat_mutated.reshape(robot_shape)

        
        if is_connected(mutated_robot) and has_actuator(mutated_robot) and (hashable(mutated_robot) not in record):
            record[hashable(mutated_robot)] = []
            return mutated_robot

    
    return None


def random_guided_mutate(Parents, record, total_mutations=10,use_attention=False,inherit=False):

    mutated_children = []

    
    half_mutations = total_mutations // 2

    for _ in range(half_mutations):
        if use_attention:
            child_robot = attention_mutate(Parents[0], record)
        else:
            child_robot = mutate(Parents[0].robot, record)
        if child_robot is not None:
            new_agent = Agent(child_robot)
            if inherit:
                partner_index = get_similar_index(new_agent, Parents)
                new_agent.controller = Parents[partner_index].best_controller
            mutated_children.append(new_agent)
    
    
    rest_agents = Parents[1:]
    random.shuffle(rest_agents)

    for agent in rest_agents:
        if len(mutated_children) >= total_mutations:
            break
        if use_attention:
            child_robot = attention_mutate(agent, record)
        else:
            child_robot = mutate(agent.robot, record)

        if child_robot is not None:
            new_agent = Agent(child_robot)
            if inherit:
                partner_index = get_similar_index(new_agent, Parents)
                new_agent.controller = Parents[partner_index].best_controller
            mutated_children.append(new_agent)

    return mutated_children


def attention_crossover(agent1, agent2,record,num_attempts=25):
    robot_shape = agent1.robot.shape

    robot1_flatten = agent1.robot.flatten()
    robot2_flatten = agent2.robot.flatten()

    robot_size=len(robot1_flatten)

    attention1 = agent1.attention
    attention2 = agent2.attention

    
    nonzero_indices_1 = [i for i, val in enumerate(robot1_flatten) if val != 0]
    nonzero_indices_2 = [i for i, val in enumerate(robot2_flatten) if val != 0]

    attention1_nonzero=attention1[nonzero_indices_1]
    attention2_nonzero=attention2[nonzero_indices_2]

    attention1_median=np.median(attention1_nonzero)
    attention2_median=np.median(attention2_nonzero)

    for _ in range(num_attempts):
        new_robot = np.zeros(robot_size, dtype=int)
        for i in range(len(new_robot)):
            if robot1_flatten[i]!=0 and robot2_flatten[i]!=0:
                new_robot[i] =robot1_flatten[i] if attention1[i]>=attention2[i] else robot2_flatten[i]
            elif robot1_flatten[i]!=0 and robot2_flatten[i]==0:
                new_robot[i]=robot1_flatten[i] if attention1[i]>=attention1_median else random.choice(
                    [v for v in [1, 2, 3, 4] if v != robot1_flatten[i]])
            elif robot1_flatten[i]==0 and robot2_flatten[i]!=0:
                new_robot[i] = robot2_flatten[i] if attention2[i] >= attention2_median else random.choice(
                    [v for v in [1, 2, 3, 4] if v != robot2_flatten[i]])
        new_robot=new_robot.reshape(robot_shape)
        if is_connected(new_robot) and has_actuator(new_robot) and (hashable(new_robot) not in record):
            record[hashable(new_robot)] = []
            return new_robot

    return None

def random_guided_crossover(Parents, record, total_crossovers=10,inherit=False):

    offspring = []
    crossed_pairs = set()  
    n=len(Parents)
    while len(offspring) < total_crossovers:

        
        parent1, parent2 = random.sample(Parents, 2)
        pair_key = tuple(sorted((parent1.id, parent2.id)))

        
        if pair_key in crossed_pairs:
            continue

        crossed_pairs.add(pair_key)

        
        child=attention_crossover(parent1, parent2,record) 

        if child is not None:
            new_agent = Agent(child)
           
            if inherit:
                partner_index = get_similar_index(new_agent, Parents)
                new_agent.controller = Parents[partner_index].best_controller
            
            offspring.append(new_agent)
        
        if len(crossed_pairs)==n*(n-1)//2:
            break

    return offspring


def smart_guided_crossover(fronts,record,total_crossovers=10):

    new_fronts = copy.deepcopy(fronts)
    pool = [agent for front in new_fronts for agent in front]
    top_layer = new_fronts[0] if new_fronts else []

    crossed_pairs = set() 
    offspring = []

    candidates = top_layer.copy()
    if len(candidates) < total_crossovers:
        others = [a for front in new_fronts[1:] for a in front]
        random.shuffle(others)
        candidates += others

    for agent in candidates:
        if len(offspring) >= total_crossovers:
            break

        
        other_pool = [a for a in pool if a.id != agent.id]
        partner_index = get_similar_index(agent, other_pool)

        partner = other_pool[partner_index]
        pair_key = tuple(sorted((agent.id, partner.id)))

        if pair_key in crossed_pairs:
            continue  

        crossed_pairs.add(pair_key)

        # child = heuristic_crossover(agent.robot, partner.robot)
        child=attention_crossover(agent, partner,record)

        if child is not None:
            #Inherit 
            agent=Agent(child)
            partner_index = get_similar_index(agent,pool)
            agent.controller=pool[partner_index].controller
            offspring.append(agent)
    return offspring




def get_best_agent(population):
    return max(population, key=lambda agent: agent.fitness)

def check_children(population):
    for agent in population:
        if not is_connected(agent.robot):
            print("ERROR")
            return agent
    return None
def check_size(population,target_size):
    for agent in population:
        if agent.robot.shape[0]>target_size or agent.robot.shape[1]>target_size:
            return agent
    return None

def get_par(env_name):
    tc = 0
    max_eva = 0
    if env_name == 'Walker-v0':
        tc = 500
        max_eva = 100
    elif env_name == 'BridgeWalker-v0':
        tc = 500
        max_eva = 100
    elif env_name == 'BidirectionalWalker-v0':
        tc = 1000
        max_eva = 150
    elif env_name == 'Carrier-v0':
        tc = 500
        max_eva = 100
    elif env_name == 'Carrier-v1':
        tc = 1000
        max_eva = 200
    elif env_name == 'Pusher-v0':
        tc = 500
        max_eva = 100
    elif env_name == 'Pusher-v1':
        tc = 600
        max_eva = 150
    elif env_name == 'Thrower-v0':
        tc = 300
        max_eva = 150
    elif env_name == 'Catcher-v0':
        tc = 400
        max_eva = 200
    elif env_name == 'BeamToppler-v0':
        tc = 1000
        max_eva = 100
    elif env_name == 'BeamSlider-v0':
        tc = 1000
        max_eva = 200
    elif env_name == 'Lifter-v0':
        tc = 300
        max_eva = 200
    elif env_name == 'Climber-v0':
        tc = 400
        max_eva = 150
    elif env_name == 'Climber-v1':
        tc = 600
        max_eva = 150
    elif env_name == 'Climber-v2':
        tc = 1000
        max_eva = 200
    elif env_name == 'UpStepper-v0':
        tc = 600
        max_eva = 150
    elif env_name == 'DownStepper-v0':
        tc = 500
        max_eva = 100
    elif env_name == 'ObstacleTraverser-v0':
        tc = 1000
        max_eva = 150
    elif env_name == 'ObstacleTraverser-v1':
        tc = 1000
        max_eva = 200
    elif env_name == 'Hurdler-v0':
        tc = 1000
        max_eva = 200
    elif env_name == 'PlatformJumper-v0':
        tc = 1000
        max_eva = 200
    elif env_name == 'GapJumper-v0':
        tc = 1000
        max_eva = 200
    elif env_name == 'Traverser-v0':
        tc = 1000
        max_eva = 200
    elif env_name == 'CaveCrawler-v0':
        tc = 1000
        max_eva = 150
    elif env_name == 'AreaMaximizer-v0':
        tc = 600
        max_eva = 100
    elif env_name == 'AreaMinimizer-v0':
        tc = 600
        max_eva = 150
    elif env_name == 'WingspanMazimizer-v0':
        tc = 600
        max_eva = 100
    elif env_name == 'HeightMaximizer-v0':
        tc = 500
        max_eva = 150
    elif env_name == 'Flipper-v0':
        tc = 600
        max_eva = 100
    elif env_name == 'Jumper-v0':
        tc = 500
        max_eva = 100
    elif env_name == 'Balancer-v0':
        tc = 600
        max_eva = 100
    elif env_name == 'Balancer-v1':
        tc = 600
        max_eva = 150

    return max_eva, tc