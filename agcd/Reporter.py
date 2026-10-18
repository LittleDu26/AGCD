import datetime

class CustomReporter():
    def __init__(self, save_to):
        self.generation = None
        self.generation_start_time = None
        self.generation_times = []
        self.num_extinctions = 0
        self.txt = open(save_to +"/out.txt", "w")

    def start_generation(self, generation):
        self.generation = generation
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 当前时间戳
        text = ""
        text += '\n ****** [{1}] Running generation {0} ****** \n'.format(generation, now)
        print(text)
        self.txt.write(text)
        self.txt.flush()

    def end_generation(self, population,crossover_children,mutated_children,Parents):
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 当前时间戳
        pop_size = len(population)
        population.sort(key=lambda agent: agent.fitness, reverse=True)
        best_agent = population[0]
        text = ""
        text += f"[{now}]Population of {pop_size} members\tbest_id:(ID{best_agent.id},{best_agent.fitness})\n"
        text += "   ID   maturity   fitness\n"
        text += "  ====  ===   =======\n"
        for i, agent in enumerate(population):
            text += "  {0:>4d}  {1:>3d}   {2:>7.4f}\n".format(agent.id, agent.maturity, agent.fitness)
        text +="Start Non-Dominated Sorting\n"
        text +="Survivors:"+str(len(Parents))+"\n"
        text +=f"crossover_children IDs({len(crossover_children)}):"+str([child.id for child in crossover_children])+"\n"
        text +=f"mutated_children IDs:({len(mutated_children)})" + str([child.id for child in mutated_children]) + "\n"
        self.txt.write(text)
        self.txt.flush()
        print(text)

    def start_elites(self,population):
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 当前时间戳
        pop_size = len(population)
        best_agent=population[0]
        text = ""
        text += f"[{now}]Population of {pop_size} members\tbest_id:(ID{best_agent.id},{best_agent.fitness})\n"
        text += "   ID   maturity   fitness\n"
        text += "  ====  ========   =======\n"
        for i, agent in enumerate(population):
            text += "  {0:>4d}  {1:>3d}   {2:>7.4f}\n".format(agent.id, agent.maturity, agent.fitness)

        text += '\n ****** [{0}] Elites training ****** \n'.format(now)
        print(text)
        self.txt.write(text)
        self.txt.flush()

    def end_elites(self,population):
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 当前时间戳
        best_agent = population[0]
        text = ""
        text += f"[{now}]Population of {len(population)} members\tbest_id:(ID{best_agent.id},{best_agent.fitness})\n"
        text += "   ID   maturity   fitness\n"
        text += "  ====  ========   =======\n"
        for i, agent in enumerate(population):
            text += "  {0:>4d}  {1:>3d}   {2:>7.4f}\n".format(agent.id, agent.maturity, agent.fitness)
        self.txt.write(text)
        self.txt.flush()
        print(text)