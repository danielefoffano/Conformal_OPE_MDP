import numpy as np
from abc import ABC, abstractmethod

class Policy(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def get_action(self, state: int) -> int:
        pass
    
    @abstractmethod
    def get_action_prob(self, state: int, action: int) -> float:
        pass
    

class EpsilonGreedyPolicy(Policy):
    def __init__(self, q_table, epsilon, na):
        super().__init__()
        self.epsilon = epsilon
        self.na = na
        self.q_function = q_table
        self.probabilities = np.zeros_like(q_table)
        for s in range(q_table.shape[0]):
            self.probabilities[s, q_table[s].argmax()] = 1
        
        self.probabilities = epsilon * np.ones_like(q_table)/na + (1-epsilon) * self.probabilities


    def get_action(self, state: int) -> int:
        return np.random.choice(self.na, p=self.probabilities[state])
    
    def get_action_prob(self, state: int, action: int) -> float:
        return self.probabilities[state, action]

class TableBasedPolicy(Policy):

    def __init__(self,prob_table):
        super().__init__()
        self.probabilities = prob_table
        self.na = len(prob_table[0])

    def get_action(self, state: int) -> int:

        probs = self.probabilities[state]
        action = np.random.choice(self.na, p= probs)

        return action
    
    def get_action_prob(self, state: int, action: int) -> float:
        return self.probabilities[state][action]
    
class MixedPolicy(Policy):

    def __init__(self, pi_1, pi_2, epsilon):
        super().__init__()
        self.pi_1 = pi_1
        self.pi_2 = pi_2
        self.epsilon = epsilon
        self.probabilities = epsilon * pi_1.probabilities + (1-epsilon) * pi_2.probabilities

    def get_action(self, state: int) -> int:
        return np.random.choice(self.pi_1.na, p=self.probabilities[state])
    
    def get_action_prob(self, state: int, action: int) -> float:
        return self.probabilities[state, action]