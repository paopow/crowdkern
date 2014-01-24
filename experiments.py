from base_kernel import BaseKernel
import numpy as np
from utils import odd_to_sim
import random
from active_learning_odd import what_should_we_ask


class RandomQuery(object):
    def get_query(self):
        a, b, c = random.sample(range(len(self.items)), 3)
        return a, b, c


class AdaptiveQuery(object):
    def get_query(self):
        item_to_ask_about = random.randrange(len(self.items))
        b, c, expected_info_gain = what_should_we_ask(item_to_ask_about, self.M, self.mu, self.comparisons)
        return item_to_ask_about, b, c


class SimAnswer(object):
    def get_answer(self, a, b, c):
        true_a = self.M_true[a]
        true_b = self.M_true[b]
        true_c = self.M_true[c]
        ab = self.dist(true_a, true_b)
        ac = self.dist(true_a, true_c)
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


class OddOneOutAnswer(object):
    def get_answer(self, a, b, c):
        true_a = self.M_true[a]
        true_b = self.M_true[b]
        true_c = self.M_true[c]
        ab = self.dist(true_a, true_b)
        bc = self.dist(true_b, true_c)
        ac = self.dist(true_a, true_c)

        if (ab >= bc and ac >= bc):  # if a is odd
            answer = (b, a, c)
        elif (ab >= ac and bc >= ac):  # if b is odd
            answer = (a, b, c)
        else:
            answer = (a, c, b)
        sim_comparisons = odd_to_sim(answer)
        return sim_comparisons


class RandomSimTest(BaseKernel, RandomQuery, SimAnswer):
    pass


class RandomOddTest(BaseKernel, RandomQuery, OddOneOutAnswer):
    pass


class AdaptiveSimTest(BaseKernel, AdaptiveQuery, SimAnswer):
    pass


class AdaptiveOddTest(BaseKernel, AdaptiveQuery, OddOneOutAnswer):
    pass
