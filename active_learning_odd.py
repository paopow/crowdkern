from utils import entropy
import numpy as np
import random


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
