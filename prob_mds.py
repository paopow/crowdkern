import numpy as np
import scipy.optimize


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

#TODO: The gradient is still not working...
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
    result = scipy.optimize.fmin_l_bfgs_b(func=my_cost, x0=x0, iprint=0, approx_grad=True) #fprime=my_grad )
    return result[0].reshape((num_items, num_dims))
