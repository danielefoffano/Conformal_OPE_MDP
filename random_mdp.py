import numpy as np

class MDPEnv():

    def __init__(self, ns, na, P, R):

        self.ns = ns
        self.observation_space = ns
        self.na = na
        self.P = P
        self.R = R

        self.cur_state = 0

    def reset(self):

        self.cur_state = np.random.randint(self.ns)
        return self.cur_state

    def step(self, a):

        probs_s = self.P[self.cur_state][a]

        next_state = np.random.choice(range(self.ns), 1, p= probs_s)[0]
        mean_r = self.R[self.cur_state][a][next_state]
        r = np.random.normal(mean_r, 1)
        self.cur_state = next_state
        # return same shape of gym: s', r, done
        return self.cur_state, r, False
    
    def set_state(self, state):
        self.cur_state = state
    
class MDPEnvDiscreteRew():

    def __init__(self, ns, na, nr, P, R):

        self.ns = ns
        self.observation_space = ns
        self.na = na
        self.nr = nr
        self.P = P
        self.R = R

        self.cur_state = 0

    def reset(self):

        self.cur_state = np.random.randint(self.ns)
        return self.cur_state

    def step(self, a):

        probs_s = self.P[self.cur_state][a]

        next_state = np.random.choice(range(self.ns), 1, p= probs_s)[0]
        
        probs_r = self.R[self.cur_state][a]

        r = np.random.choice(range(self.nr), 1, p= probs_r)[0]
        self.cur_state = next_state
        # return same shape of gym: s', r, done
        return self.cur_state, r, False
    
    def set_state(self, state):
        self.cur_state = state
    
class MDPEnvBernoulliRew():

    def __init__(self, ns, na, P, R):

        self.ns = ns
        self.observation_space = ns
        self.na = na
        self.P = P
        self.R = R

        self.cur_state = 0

    def reset(self):

        self.cur_state = np.random.randint(self.ns)
        return self.cur_state

    def step(self, a):

        probs_s = self.P[self.cur_state][a]

        next_state = np.random.choice(range(self.ns), 1, p= probs_s)[0]
        
        probs_r = self.R[self.cur_state][a]

        r = np.random.binomial(1,probs_r)
        self.cur_state = next_state
        # return same shape of gym: s', r, done
        return self.cur_state, r, False
    
    def set_state(self, state):
        self.cur_state = state