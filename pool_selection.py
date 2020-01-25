import numpy as np 


class evolution():
    def __init__(self, target, N = 100, m_rate = 0.01, generations = 100000, n_parents = 2):
        self.N = N
        self.m_rate = m_rate
        self.generations = generations
        self.n_parents = n_parents
        self.print_op = target
        self.target = np.array(list(target))
        self.fitness = np.zeros(self.N)
        self.population = None #DNA 
        self.best_offspring = None
        self.best_fitness = None
        self.index = np.arange(self.N)
        self.probability = np.zeros(self.N)
        self.create_population()

    def create_population(self):
        self.length = len(self.target)
        self.dictionary = np.array(list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_ '))
        self.population = np.random.choice(self.dictionary, [self.N, self.length])
        # print(list(self.population))

    def calc_fitness(self):
        for i in range(self.N):
            self.fitness[i] = np.where(self.population[i] == self.target)[0].shape[0]
            self.fitness[i] = self.fitness[i] ** 2
            # print(list(self.population[i]), self.fitness[i])
        self.best_fitness = np.max(self.fitness)
        self.best_offspring = self.population[np.where(self.fitness == self.best_fitness)]
        self.probability = self.fitness / np.sum(self.fitness)
        # print(self.probability)

    def selection(self):
        return self.pick(self.population, self.probability), self.pick(self.population, self.probability)

    def reproduction(self):
        temp = self.population.copy()
        for i in range(self.N):
            p1, p2 = self.selection()
            # print("p1:", p1)
            # print("p2:", p2)
            child = self.cross_over(p1, p2)
            # print("co:", child)
            child = self.mutation(child)
            # print("offspring :", child)
            temp[i] = child.copy()
        self.population = temp.copy()

    def cross_over(self, *parents):
        p1, p2 = parents
        pivot = np.random.randint(self.length)
        return np.concatenate((p1[:pivot], p2[pivot:]))

    def mutation(self, child):
        prob_spec = 1 - self.m_rate
        for i in range(self.length):
            if child[i] != self.target[i]:
                prob = np.ones(len(self.dictionary)) * (self.m_rate / (len(self.dictionary) - 1))
                prob[np.where(self.dictionary == child[i])] = prob_spec
                # print(prob)
                child[i] = np.random.choice(self.dictionary, p = prob, replace = False)
        return child
    
    def pick(self, pool, prob):

        index = 0
        rand = np.random.random()
        while(rand > 0):
            rand = rand - prob[index]
            index += 1
        index -= 1

        return pool[index]

    def evolve(self):
        for gen in range(self.generations):
            self.calc_fitness()
            stg = ''
            print("generation :", gen,'\ttarget :', self.print_op, '\tbest_offspring :', stg.join(self.best_offspring[0]), '\tfitness :',self.best_fitness)
            if (self.best_offspring == self.target).all():
                break
            self.reproduction()



if __name__ == '__main__':
    # ga = evolution('Genetic Algorithm to generate_1_sentence')
    ga = evolution('To be or not to be')
    
    ga.evolve()