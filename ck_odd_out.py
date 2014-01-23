import random
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import sys
from itertools import combinations
import math


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


### Active learning

def calc_tau(a, M, mu, comparisons):
    num_items, num_dims = M.shape
    tau = np.ones(num_items) / float(num_items)
    for otherA, b, c in comparisons:
        if a != otherA:
            continue
        for x in xrange(num_items):
            ma = M[x]
            mb = M[b]
            mc = M[c]
            delta_ab = np.dot(ma - mb, ma - mb)
            delta_ac = np.dot(ma - mc, ma - mc)
            phat = (mu + delta_ac) / (2.*mu + delta_ab + delta_ac)
            tau[x] *= phat
    tau /= tau.sum()
    return tau


def entropy(dist):
    return -np.sum(dist * np.log(dist+1e-9))


def calc_info_gain(tau, b, c, M, mu):
    num_items, num_dims = M.shape
    phats_a = np.empty(num_items)
    phats_b = np.empty(num_items)
    for x in xrange(num_items):
        ma = M[x]
        mb = M[b]
        mc = M[c]
        delta_ab = np.dot(ma - mb, ma - mb)
        delta_ac = np.dot(ma - mc, ma - mc)
        delta_bc = np.dot(mb - mc, mb - mc)
        tri_peri = delta_ab + delta_bc + delta_ac

        phats_a[x] = (mu + delta_ab + delta_ac) / (2.*mu + 2*(tri_peri))
        phats_b[x] = (mu + delta_bc + delta_ab) /(2.*mu + 2*(tri_peri))

    p_a = np.sum(tau * phats_a)
    p_b = np.sum(tau * phats_b)
    p_c = 1 - p_a - p_b

    tau_a = tau * phats_a
    tau_b = tau * phats_b
    tau_c = tau * (1 - phats_a - phats_b)
    tau_a /= tau_a.sum()
    tau_b /= tau_b.sum()
    tau_c /= tau_c.sum()

    return entropy(tau) - p_a*entropy(tau_a) - p_b*entropy(tau_b) - (p_c)*entropy(tau_c)


def what_should_we_ask(a, M, mu, comparisons, num_samples=10):
    num_items = M.shape[0]
    choices = range(num_items)
    choices.remove(a)
    tau = calc_tau(a, M, mu, comparisons)
    pairs = []
    for i in xrange(num_samples):
        b, c = random.sample(choices, 2)
        pairs.append((b, c, calc_info_gain(tau, b, c, M, mu)))
    return max(pairs, key=lambda thing: thing[-1])

def odd_to_sim(comparison):
    comp1 = [comparison[0], comparison[2], comparison[1]]
    comp2 = [comparison[2], comparison[0], comparison[1]]
    return (comp1, comp2)


class CK_odd(object):
    def __init__(self, df, items, initial_comparisons, num_dims, mu):
        self.df = df
        self.items = items
        self.comparisons = initial_comparisons[:]
        self.num_dims = num_dims
        self.mu = mu
        self.update_M()

    def update_M(self):
        #num_items = len(self.items)
        num_items = self.df.shape[0]
        num_dims = self.num_dims
        self.M = optimize_item_positions(num_items, num_dims, self.mu, self.comparisons)

    def get_query(self):
        item_to_ask_about = random.randrange(self.df.shape[0])
        b, c, expected_info_gain = what_should_we_ask(item_to_ask_about, self.M, self.mu, self.comparisons)
        return item_to_ask_about, b, c, expected_info_gain

    def plot_scatter(self):
        num_comp = len(self.comparisons)
        last_comp = self.comparisons[num_comp -1]
        plt.clf()
        plt.scatter(self.M[:,0], self.M[:,1])
        num_items = self.df.shape[0]
        plt.title(self.df.name[last_comp[0]] + " is more similar to " + self.df.name[last_comp[1]] + " than to " + self.df.name[last_comp[2]])
        for i in range(num_items):
            plt.text(self.M[i,0] + np.random.rand()/20.0 , self.M[i,1]+ np.random.rand()/20.0, self.df.name[i], fontsize=8, alpha=0.5)
        plt.savefig('figures/puppy10_' + str(num_comp)+'.pdf')

    def get_comparison(self):
        return self.comparisons

    def loop(self):
        while True:
            items = self.df.name #self.items
            item_to_ask_about, b, c, expected_info_gain = self.get_query()
            print 'Is', items[item_to_ask_about], 'more similar to', items[b], 'or', items[c], '???', expected_info_gain
            response = raw_input('. if first, / if second: ')
            if not response:
                return
            if response == '.':
                self.comparisons.append((item_to_ask_about, b, c))
            else:
                self.comparisons.append((item_to_ask_about, c, b))
            self.update_M()
            self.plot_scatter()



class CK_odd_test(object):
    def __init__(self, items, initial_comparisons, num_dims, mu, M_true, max_query ):
        num_items = len(items)
        num_pairs = num_items*(num_items -1)/2
        self.num_pair_sample = int(math.floor(0.08*(num_pairs))*10)
        self.max_query = max_query
        self.errors = np.zeros((self.num_pair_sample*(max_query/10), 2))
        self.items = items
        self.comparisons = initial_comparisons[:]
        self.num_dims = num_dims
        self.mu = mu
        self.M_true = M_true
        self.update_M()


    def update_M(self):
        num_items = len(self.items)
        # num_items = self.df.shape[0]
        num_dims = self.num_dims
        self.M = optimize_item_positions(num_items, num_dims, self.mu, self.comparisons)

    def get_query(self):
        item_to_ask_about = random.randrange(len(self.items))
        # item_to_ask_about = random.randrange(self.df.shape[0])
        b, c, expected_info_gain = what_should_we_ask(item_to_ask_about, self.M, self.mu, self.comparisons)
        return item_to_ask_about, b, c, expected_info_gain

    def dist(self, a, b):
        return math.sqrt(np.dot(a-b, a-b))

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
            item_to_ask_about, b, c, expected_info_gain = self.get_query()
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
            
        

