import random
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import pandas as pd
import sys
from itertools import combinations


def cost(M, mu, comparisons):
    M /= np.sqrt(np.sum(M * M, axis=1))[:, None]
    cost = 0.
    for a, b, c in comparisons:
        ma = M[a]
        mb = M[b]
        mc = M[c]
        delta_ab = np.dot(ma - mb, ma - mb)
        delta_ac = np.dot(ma - mc, ma - mc)
        phat = (mu + delta_ac) / (2. * mu + delta_ab + delta_ac)
        cost -= np.log(np.clip(phat, .001, .999))
    return cost / len(comparisons)


def gradient(M, mu, comparisons):
    dM = np.zeros_like(M)
    M /= np.sqrt(np.sum(M * M, axis=1))[:, None]
    for a, b, c in comparisons:
        ma = M[a]
        mb = M[b]
        mc = M[c]
        ma_mb = np.dot(ma - mb, ma - mb)
        ma_mc = np.dot(ma - mc, ma - mc)
        dM[a] += (-2*M[c]) / (mu + ma_mc) - (-2*mb - 2*mc) / (2*mu + ma_mb + ma_mc)
        dM[b] += 0 - (-2*ma) / (2*mu + ma_mb + ma_mc)
        dM[c] += (-2*ma) / (mu + ma_mc) - (-2*ma) / (2*mu + ma_mb + ma_mc)
    return -dM


def optimize_item_positions(num_items, num_dims, mu, comparisons):
    def my_cost(vec):
        M = vec.reshape((num_items, num_dims))
        return cost(M, mu, comparisons)
    def my_grad(vec):
        M = vec.reshape((num_items, num_dims))
        return gradient(M, mu, comparisons).ravel()
    x0 = np.random.standard_normal(num_items * num_dims)
    result = scipy.optimize.fmin_l_bfgs_b(func=my_cost, x0=x0, iprint=0, approx_grad=True)
    return result[0].reshape((num_items, num_dims))


class RandomSimKernelTest(object):
    def __init__(self, items, initial_comparisons, num_dims, mu, M_true, max_query):
        self.num_pair_sample = 150 # TODO: replace with sth else
        self.max_query = max_query
        self.mu = mu
        self.errors = np.zeros((self.num_pair_sample*(max_query/10), 2))
        self.items = items
        self.comparisons = initial_comparisons[:]
        self.num_dims = num_dims
        self.M_true = M_true
        self.update_M()

    def update_M(self):
        num_items = len(self.items)
        num_dims = self.num_dims
        self.M = optimize_item_positions(num_items, num_dims, self.mu, self.comparisons)

    def get_query(self):
        a, b, c = random.sample(range(len(self.items)), 3)
        return a, b, c

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

    def dist(self, a, b):
        return np.dot(a-b, a-b)

    def get_comparison(self):
        return self.comparisons

    def loop_count(self, num_query):
        for i in range(num_query):
            sys.stdout.write(str(i) + " ")
            items = self.items
            item_to_ask_about, b, c = self.get_query()

            # basically, check with M_true and update the comparison
            true_a = self.M_true[item_to_ask_about]            
            true_b = self.M_true[b]
            true_c = self.M_true[c]
            ab = self.dist(item_to_ask_about, b)
            ac = self.dist(item_to_ask_about, c)
            if ab > ac:
                answer = (item_to_ask_about, c, b)
            elif ab < ac:
                answer = (item_to_ask_about, b, c)
            else: 
                if np.random.random_integers(0,1) == 0:
                    answer = (item_to_ask_about, b, c)
                else:
                    answer = (item_to_ask_about, c, b)
            self.comparisons.append(answer)


            self.update_M()
            num_comp = len(self.comparisons)
            if (num_comp != 0 ) and (num_comp % 10 == 0):
                self.update_error(num_comp)

    def loop_auto(self):
        self.loop_count(self.max_query)



class RandomOddKernelTest(object):
    def __init__(self, items, initial_comparisons, num_dims, mu, M_true, max_query ):
        self.num_pair_sample = 150 # TODO: replace with sth else
        self.max_query = max_query
        self.mu = mu
        self.errors = np.zeros((self.num_pair_sample*(max_query/10), 2))
        self.items = items
        self.comparisons = initial_comparisons[:]
        self.num_dims = num_dims
        self.M_true = M_true
        self.update_M()


    def update_M(self):
        num_items = len(self.items)
        # num_items = self.df.shape[0]
        num_dims = self.num_dims
        self.M = optimize_item_positions(num_items, num_dims, self.mu, self.comparisons)

    def get_query(self):
        a, b, c = random.sample(range(len(self.items)), 3)
        return a, b, c

    def dist(self, a, b):
        return np.dot(a-b, a-b)

    def get_comparison(self):
        return self.comparisons

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

    def loop_count(self, num_query):
        for i in range(num_query):
            sys.stdout.write(str(i) + " ")
            items = self.items
            item_to_ask_about, b, c = self.get_query()
            true_a = self.M_true[item_to_ask_about]
            true_b = self.M_true[b]
            true_c = self.M_true[c]
            ab = self.dist(item_to_ask_about,b)
            bc = self.dist(b,c)
            ac = self.dist(item_to_ask_about,c)

            if (ab >= bc and ac >= bc): # if a is odd
                answer = (b, item_to_ask_about, c)
            elif (ab >= ac and bc >= ac): # if b is odd
                answer = (item_to_ask_about, b, c)
            else:
                answer = (item_to_ask_about, c, b)
            sim_comparisons = odd_to_sim(answer)
            self.comparisons.extend(sim_comparisons)            
            self.update_M()

            num_q = i+2
            if (num_q % 10 == 0):
                self.update_error(num_q)


    def loop_auto(self):
        self.loop_count(self.max_query)
            
        


            
