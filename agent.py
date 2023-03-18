import numpy as np
from abc import ABC
from typing import NamedTuple
from types_cp import Experience



class Agent(ABC):
    ns: int # Number of states
    na: int # Number of actions
    discount_factor: float # Discount factor

    def __init__(self, ns: int, na: int, discount_factor: float):
        self.ns = ns
        self.na = na
        self.discount_factor = discount_factor
        self.num_visits_state = np.zeros(self.ns)
        self.num_visits_actions = np.zeros((self.ns, self.na))
        self.last_visit_state = np.zeros(self.ns)
        self.policy_diff_generative = []
        self.policy_diff_constraints = []
    
    def forward(self, state: int, step: int) -> int:
        self.num_visits_state[state] += 1
        self.last_visit_state[state] = step
        action = self._forward_logic(state, step)
        self.num_visits_actions[state][action] += 1
        return action
    
    def backward(self, experience: Experience):
        self._backward_logic(experience)
    
    def _forward_logic(self, state: int, step: int) -> int:
        raise NotImplementedError

    def _backward_logic(self, experience: Experience):
        raise NotImplementedError
    
    def greedy_action(self, state: int) -> int:
        raise NotImplementedError
    
class QlearningAgent(Agent):
    def __init__(self, ns: int, na: int, discount_factor: float, alpha: float, explorative_policy = None):
        super().__init__(ns, na, discount_factor)
        self.q_function = np.zeros((self.ns, self.na))
        self.alpha = alpha
        self.explorative_policy = explorative_policy
        
    def _forward_logic(self, state: int, step: int) -> int:
        if self.explorative_policy is not None:
            state_sum = np.sum(self.explorative_policy[state] + 1/self.ns)
            probs = (self.explorative_policy[state] + 1/self.ns)/state_sum
            action = np.random.choice(range(self.na),1,p=probs)[0]
        else:
            eps = 1 if self.num_visits_state[state] <= 2 * self.na else max(0.5, 1 / (self.num_visits_state[state] - 2*self.na))
            action = np.random.choice(self.na) if np.random.uniform() < eps else self.q_function[state].argmax()
        return action

    def greedy_action(self, state: int) -> int:
        return self.q_function[state].argmax()

    def _backward_logic(self, experience: Experience):
        state, action, reward, next_state, done = list(experience)
        target = reward + (1-done) * self.discount_factor * self.q_function[next_state].max()
        lr = 1 / (self.num_visits_actions[state][action] ** self.alpha)
        self.q_function[state][action] += lr * (target - self.q_function[state][action])