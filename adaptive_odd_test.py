from base_kernel import BaseKernel
import math
from utils import odd_to_sim
from active_learning_odd import *
from itertools import combinations

class AdaptiveOddTest(BaseKernel):
    def __init__(self, items, comparisons, M_true, **args):
        self.items = items
        self.comparisons = comparisons[:]
        self.M_true = M_true
        self.num_dims = 2
        self.mu = 0.5
        self.max_query = 100

        if 'num_dims' in args:
            self.num_dims = args['num_dims']
        if 'mu' in args:
            self.mu = args['mu']
        if 'max_query' in args:
            self.max_query = max_query

        num_items = len(self.items)
        num_pairs = num_items*(num_items -1)/2
        self.num_pair_sample = math.floor(0.08*(num_pairs))*10
        self.errors = np.zeros((self.num_pair_sample*(max_query/10)))

        self.update_M()

    def update_error(self, num_query):
        pairs = list(combinations(range(len(self.items)), 2))
        random.shuffle(pairs)
        step = num_query/10 - 1
        for i in range(self.num_pair_sample):
            pair = pairs[i]
            a = self.M[pair[0],:]
            b = self.M[pair[1],:]
            pred_dist = self.dist(a,b)

            true_a = self.M_true[pair[0],:]
            true_b = self.M_true[pair[1],:]
            real_dist = self.dist(true_a,true_b)

            diff = abs(pred_dist - real_dist)

            self.errors[step*self.num_pair_sample + i,:] = [num_query, diff]

    def get_error_array(self):
        return self.errors

    def get_query(self):
        item_to_ask_about = random.randrange(len(self.items))
        b, c, expected_info_gain = what_should_we_ask(item_to_ask_about, self.M, self.mu, self.comparisons)
        return item_to_ask_about, b, c

    def get_answer(self, a, b, c):
        true_a = self.M_true[a]
        true_b = self.M_true[b]
        true_c = self.M_true[c]
        ab = self.dist(true_a, true_b)
        bc = self.dist(true_b, true_c)
        ac = self.dist(true_a, true_c)

        if (ab >= bc and ac >= bc): # if a is odd
            answer = (b, a, c)
        elif (ab >= ac and bc >= ac): # if b is odd
            answer = (a, b, c)
        else:
            answer = (a, c, b)
        sim_comparisons = odd_to_sim(answer)
        return sim_comparisons

    def loop_count(self, num_query):
        for i in range(num_query):
            sys.stdout.write(str(i) + " ")
            items = self.items
            item_to_ask_about, b, c = self.get_query()
            answer = self.get_answer(item_to_ask_about, b, c)
            self.comparisons.extend(answer)
            self.update_M()
            num_q = i + 2
            if (num_q != 0 ) and (num_q % 10 == 0):
                self.update_error(num_q)

    def loop_auto(self):
        self.loop_count(self.max_query)