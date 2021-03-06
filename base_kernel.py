import random
import numpy as np
import sys
import prob_mds
from itertools import combinations


class BaseKernel(object):
    def __init__(self, items, comparisons, M_true, num_dims=2, mu=0.5,
                 max_query=100, sample_frac=0.08):
        self.items = items
        self.comparisons = comparisons[:]
        self.M_true = M_true
        self.num_dims = num_dims
        self.mu = mu
        self.max_query = max_query

        num_items = len(self.items)
        num_pairs = num_items * (num_items - 1) / 2
        self.num_pair_sample = int(sample_frac * num_pairs / 10.) * 10
        self.errors = np.zeros(((max_query / 10), self.num_pair_sample))

        self.update_M()

    def update_M(self):
        num_items = len(self.items)
        num_dims = self.num_dims
        self.M = prob_mds.optimize_item_positions(num_items, num_dims, self.mu, self.comparisons)

    def draw_sample_of_errors(self, out):
        pairs = list(combinations(range(len(self.items)), 2))
        random.shuffle(pairs)
        for i in xrange(self.num_pair_sample):
            a, b = pairs[i]
            pred_dist = self.dist(self.M[a, :], self.M[b, :])
            real_dist = self.dist(self.M_true[a, :], self.M_true[b, :])
            out[i] = abs(pred_dist - real_dist)

    def dist(self, a, b):
        delta = a - b
        return np.dot(delta, delta)

    def get_comparison(self):
        return self.comparisons

    def loop_count(self, num_query):
        for i in range(num_query):
            sys.stdout.write(str(i) + " ")
            item_to_ask_about, b, c = self.get_query()
            answer = self.get_answer(item_to_ask_about, b, c)
            self.comparisons.extend(answer)
            self.update_M()
            num_q = i + 2
            if (num_q != 0) and (num_q % 10 == 0):
                self.draw_sample_of_errors(out=self.errors[num_q / 10 - 1])

    def loop_auto(self):
        self.loop_count(self.max_query)
