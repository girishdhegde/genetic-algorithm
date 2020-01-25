import numpy as np 

def pick(pool, prob):

    index = 0
    rand = np.random.random()
    while(rand > 0):
        rand = rand - prob[index]
        index += 1
    index -= 1

    return pool[index]

def verify():

    pool = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    picked = np.zeros(len(pool), np.int16)

    prob = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
    prob = prob / np.sum(prob)

    for i in range(100000):
        ele = pick(pool, prob)
        picked[ele] += 1

    print(picked / np.sum(picked), prob)

if __name__ =='__main__':
    verify()