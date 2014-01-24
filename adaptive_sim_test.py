from base_kernel import BaseKernel
from active_learning_sim import *


class AdaptiveSimTest(BaseKernel):
    def get_query(self):
        item_to_ask_about = random.randrange(len(self.items))
        b, c, expected_info_gain = what_should_we_ask(item_to_ask_about, self.M, self.mu, self.comparisons)
        return item_to_ask_about, b, c

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
