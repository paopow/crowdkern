import numpy as np


def odd_to_sim(comparison):
    comp1 = [comparison[0], comparison[2], comparison[1]]
    comp2 = [comparison[2], comparison[0], comparison[1]]
    return (comp1, comp2)


def entropy(dist):
    return -np.sum(dist * np.log(dist+1e-9))
