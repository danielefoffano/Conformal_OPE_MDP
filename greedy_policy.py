import numpy as np

class EpsilonGreedyPolicy:

    def __init__(self, q_table, epsilon, na):
        self.epsilon = epsilon
        self.na = na
        self.q_function = q_table

    def get_action(self, state):
        action = np.random.choice(self.na) if np.random.uniform() < self.epsilon else self.q_function[state].argmax()
        return action
    
    def get_action_prob(self, state, action):

        if action == self.q_function[state].argmax():

            return (1-self.epsilon) + (self.epsilon)*(1/self.na)
        else:
            return (self.epsilon)*(1/self.na)

class TableBasedPolicy:

    def __init__(self,prob_table):

        self.probabilities = prob_table
        self.na = len(prob_table[0])

    def get_action(self, state):

        probs = self.probabilities[state]

        action = np.random.choice(range(self.na), 1, p= probs)[0]

        return action
    
    def get_action_prob(self, state, action):

        return self.probabilities[state][action]
    
class MixedPolicy:

    def __init__(self, pi_star, pi_b, epsilon):

        self.pi_star = pi_star
        self.pi_b = pi_b
        self.epsilon = epsilon

    def get_action(self, state):
        action = self.pi_b.get_action(state) if np.random.uniform() < self.epsilon else self.pi_star.get_action(state)
        return action
    
    def get_action_prob(self,state,action):

        return self.epsilon*self.pi_b.get_action_prob(state,action) + (1-self.epsilon)*self.pi_star.get_action_prob(state,action)