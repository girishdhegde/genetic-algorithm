import numpy as np 
import shutil
import cv2
import line

class evolution():

    font = cv2.FONT_HERSHEY_SIMPLEX 
    org = (10, 30)
    fontScale = 0.5
    color = [0, 0, 0, 0] 
    thickness = 1

    def __init__(self, points, N = 100, m_rate = 0.01, generations = 10000, n_parents = 2):
        self.N = N
        self.m_rate = m_rate
        self.generations = generations
        self.n_parents = n_parents     
        self.population = [] #list of DNA objects 
        self.best_offspring = None
        self.probability = np.zeros(self.N)
        self.dictionary = points
        self.read_log()
        self.initimg()
        self.create_population()

    def create_population(self):
        for i in range(self.N):
            dna = DNA()
            temp = self.dictionary.copy()
            np.random.shuffle(temp)
            dna.gene = temp
            self.population.append(dna)


    def calc_fitness(self):
        fitsum = 0
        self.best_offspring = self.population[0]
        for dna in self.population:
            dist = 0
            cities = np.unique(dna.gene, axis = 0)
            for i in range(len(dna.gene) - 1):
                # print(dna.gene)
                dist += np.sum((dna.gene[i] - dna.gene[i + 1]) ** 2) ** 0.5
            dist += np.sum((dna.gene[-1] - dna.gene[0]) ** 2) ** 0.5
            dna.distance = dist
            # print(dist)
            dna.fitness = (len(cities) ** 3) / (dist)
            dna.cities = len(cities)
            # print(dna.fitness)
            fitsum += dna.fitness
            self.best_offspring = dna if (dna.fitness > self.best_offspring.fitness) else self.best_offspring

        for i, dna in enumerate(self.population):
            self.probability[i] = dna.fitness / fitsum
        # print(self.probability, np.sum(self.probability))

    def selection(self):
        return self.pick(self.population, self.probability), self.pick(self.population, self.probability)

    def reproduction(self):
        temp = self.population.copy()
        temp[0] = self.best_offspring
        for i in range(1, self.N):
            # print(temp[i].gene)
            p1, p2 = self.selection()
            child = self.cross_over(p1, p2)
            child = self.mutation(child)
            temp[i] = child
            # print(temp[i].gene)
        self.population = temp.copy()

    def cross_over(self, *parents):
        p1, p2 = parents
        pivot = np.random.randint(len(self.dictionary))
        child = DNA()
        child.gene =  np.concatenate((p1.gene[:pivot], p2.gene[pivot:]))
        # print(pivot,p1.gene,p2.gene,child.gene)
        return child

    def mutation(self, child):
        prob_spec = 1 - self.m_rate
        for i in range(len(self.dictionary)):
                prob = np.ones(len(self.dictionary)) * (self.m_rate / (len(self.dictionary) - 1))
                for j, point in enumerate(self.dictionary):
                    if (point == child.gene[i]).all():
                        prob[j] = prob_spec
                child.gene[i] = self.pick(self.dictionary, prob)
        return child
    
    def pick(self, pool, prob):
        index = 0
        # print("prob", prob)
        rand = np.random.random()
        while(rand > 0):
            rand = rand - prob[index]
            index += 1
        index -= 1

        return pool[index]

    def initimg(self):
        import os

        self.window_size = 750
        self.img = np.ones((self.window_size, self.window_size, 3), np.uint8) * 255
        img = cv2.putText(self.img.copy(), 'CITY MAP', (10, 350), self.font,  5, [0, 255, 255], 10)

        for x, y in self.dictionary:
            self.img = cv2.circle(self.img, (x, 749 - y), 2, color = [255, 50, 0 ], thickness = -100)
            img = cv2.circle(img, (x, 749 - y), 2, color = [255, 50, 0], thickness = -100)

        os.mkdir('./output/map' + self.pointer)
        cv2.imwrite("./output/map" + self.pointer + '/' + "city_map.png", img)
        cv2.imshow("travelling_salesman", self.img)
        cv2.waitKey(0)

    def phenotype(self, gen):
        self.window = self.img.copy()
        for i in range(len(self.best_offspring.gene) - 1):
            point1 = self.best_offspring.gene[i]
            point2 = self.best_offspring.gene[i + 1]
            self.window = line.draw(self.window, point1[0], point2[0], point1[1], point2[1], color = [150, 0, 255])
        self.window = line.draw(self.window, self.best_offspring.gene[0][0], self.best_offspring.gene[-1][0],
                                self.best_offspring.gene[0][1], self.best_offspring.gene[-1][1], color = [150, 0, 255])

        if gen % 100 == 0:
            text = 'generation : ' + str(gen) + ', cities visited : ' + str(self.best_offspring.cities) + ', distance travelled : %.3f'%(self.best_offspring.distance) + ', fitness : %.3f'%(self.best_offspring.fitness)
            img = cv2.putText(self.window, text, self.org, self.font,  self.fontScale, self.color, self.thickness)
            cv2.imwrite("./output/map" + self.pointer + '/' + "generation_%d.png"%(gen), img)

        self.window = cv2.putText(self.window, str(gen), (5, 10), self.font,  .3, self.color, self.thickness)
        cv2.imshow("travelling_salesman", self.window)
        k = cv2.waitKey(1)
        return k

    def read_log(self):
        with open('./output/log.txt', 'r') as log:
            pointer = log.read()
            pointer = int(pointer)
            pointer = pointer + 1
        with open('./output/log.txt', 'w') as log:
            log.write(str(pointer))
        self.pointer = str(pointer)

    def save(self):
        s = input("Do you want to save maps[y/n]:")
        if s != 'y':
            shutil.rmtree('./output/map' + self.pointer)
            with open('./output/log.txt', 'w') as log:
                log.write(str(int(self.pointer) - 1))

    def evolve(self):
        for gen in range(self.generations):
            self.calc_fitness()
            print("generation :", gen, '\tfitness :', self.best_offspring.fitness)
            k = self.phenotype(gen)
            if k == 27:
                break
            self.reproduction()
        
        cv2.imshow("travelling_salesman", self.window)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        self.save()


class DNA():  
    def __init__(self):
        self.gene = None
        self.fitness = 0
        self.distance = 0
        self.cities = 0

def generate():

    size =750
    yl = size - 1
    no_of_pts= 15
    points = []

    points = np.random.rand(no_of_pts, 2) * yl 
    points = np.int32(points)

    # import convex_hull
    
    # edges = np.array(convex_hull.gift_wrap(points))

    # points = edges[:,0]

    return points

if __name__ == '__main__':
    points = generate()
    ga = evolution(points, N = 50, m_rate = 0.1, generations = 10000, n_parents = 2)
    ga.evolve()

