from base_kernel import BaseKernel
from utils import odd_to_sim

class RandomOddTest(BaseKernel):
    def get_answer(self, a, b, c):
        true_a = self.M_true[a]
        true_b = self.M_true[b]
        true_c = self.M_true[c]
        ab = self.dist(a,b)
        bc = self.dist(b,c)
        ac = self.dist(a,c)

        if (ab >= bc and ac >= bc): # if a is odd
            answer = (b, a, c)
        elif (ab >= ac and bc >= ac): # if b is odd
            answer = (a, b, c)
        else:
            answer = (a, c, b)
        sim_comparisons = odd_to_sim(answer)
        return sim_comparisons

    def get_query(self):
        a, b, c = random.sample(range(len(self.items)), 3)
        return a, b, c
