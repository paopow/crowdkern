from base_kernel import BaseKernel
import math
from itertools import combinations

class RandomSimTest(BaseKernel):
    def get_answer(self, a, b, c):
        true_a = self.M_true[a]
        true_b = self.M_true[b]
        true_c = self.M_true[c]
        ab = self.dist(a, b)
        ac = self.dist(a, c)
        if ab > ac:
            answer = (a, c, b)
        elif ab < ac:
            answer = (a, b, c)
        else:
            if np.random.random_integers(0,1) == 0:
                answer = (a, b, c)
            else:
                answer = (a, c, b)
        return [answer]

    def get_query(self):
        a, b, c = random.sample(range(len(self.items)), 3)
        return a, b, c
