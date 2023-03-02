import numpy as np

class EpsilonGreedyPolicy:

    def __init__(self, q_table, epsilon, na):
        self.epsilon = epsilon
        self.na = na
        self.q_function = q_table
        self.probabilities = np.zeros_like(q_table)
        for s in range(q_table.shape[0]):
            self.probabilities[s, q_table[s].argmax()] = 1
        
        self.probabilities = epsilon * np.ones_like(q_table)/na + (1-epsilon) * self.probabilities


    def get_action(self, state):
        return np.random.choice(self.na, p=self.probabilities[state])
    
    def get_action_prob(self, state, action):
        return self.probabilities[state, action]

class TableBasedPolicy:

    def __init__(self,prob_table):

        self.probabilities = prob_table
        self.na = len(prob_table[0])

    def get_action(self, state):

        probs = self.probabilities[state]
        action = np.random.choice(self.na, p= probs)

        return action
    
    def get_action_prob(self, state, action):
        return self.probabilities[state][action]
    
class MixedPolicy:

    def __init__(self, pi_1, pi_2, epsilon):

        self.pi_1 = pi_1
        self.pi_2 = pi_2
        self.epsilon = epsilon
        self.probabilities = epsilon * pi_1.probabilities + (1-epsilon) * pi_2.probabilities

    def get_action(self, state):
        return np.random.choice(self.pi_1.na, p=self.probabilities[state])
    
    def get_action_prob(self,state,action):
        return self.probabilities[state, action]