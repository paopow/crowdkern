import random
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import pandas as pd
import sys
from itertools import combinations
import prob_mds


class BaseKernel(object):
	def __init__(self, items, comparisons, **args):
		self.items = items
		self.comparisons = comparisons[:]
        self.num_dims = 2
        self.mu = 0.5

        if 'num_dims' in args:
            self.num_dims = args['num_dims']
        if 'mu' in args:
            self.mu = args['mu']
        self.update_M()

    def update_M(self):
        num_items = len(self.items)
        num_dims = self.num_dims
        self.M = prob_mds.optimize_item_positions(num_items, num_dims, self.mu, self.comparisons)

    def get_query(self):
        a, b, c = random.sample(range(len(self.items)), 3)
        return a, b, c

    def dist(self, a, b):
        return np.dot(a-b, a-b)

    def get_comparison(self):
        return self.comparisons

    def get_answer(self, triplet):
        pass

    def loop_count(self, num_query):
        for i in range(num_query):
            sys.stdout.write(str(i) + " ")
            items = self.items
            item_to_ask_about, b, c = self.get_query()

            answer = self.get_answer((item_to_ask_about, b, c))
            self.comparisons.extend(answer)

            self.update_M()