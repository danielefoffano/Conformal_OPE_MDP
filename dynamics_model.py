import numpy as np
import pickle
import os
import lzma
class DynamicsModel(object):

    def __init__(self, num_states: int, num_actions: int):
        """
        Transition function model for a grids

        Args:
            num_states (int): Number of states
            num_actions (int): Number of actions
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_visits_actions = np.zeros(shape=(num_states, num_actions, num_states), dtype=np.int64)
        self.transition_function = np.ones_like(self.num_visits_actions, dtype=np.float64) / num_states
        self.reward_gaussian = np.zeros(shape=(num_states, num_actions, 2), dtype=np.float64)
        self.reward_counts = [[[] for a in range(num_actions)] for s in range (num_states)]
        self.cur_state = 0

    def update_visits(self, from_state: int, action: int, to_state: int, reward: float):
        """Updates the transition function given an experience

        Args:
            from_state (Coordinate): state s
            action (int): action a
            to_state (Coordinate): next state s'
            reward (float): ereward
        """
        self.num_visits_actions[from_state, action, to_state] += 1
        #self.reward[from_state, action, to_state] = reward
        self.reward_counts[from_state][action].append(reward)
        self.reward_gaussian[from_state][action][0] = np.mean(self.reward_counts[from_state][action])
        self.reward_gaussian[from_state][action][1] = np.std(self.reward_counts[from_state][action])

        self.transition_function[from_state, action] = (
            self.num_visits_actions[from_state, action] / self.num_visits_actions[from_state, action].sum()
        )

    def reset(self):

        self.cur_state = np.random.randint(self.num_states)
        return self.cur_state

    def step(self, a):

        probs_s = self.transition_function[self.cur_state][a]

        next_state = np.random.choice(range(self.num_states), 1, p= probs_s)[0]
        #r = self.reward[self.cur_state][a][next_state]
        r = np.random.normal(self.reward_gaussian[self.cur_state][a][0], self.reward_gaussian[self.cur_state][a][1])
        self.cur_state = next_state
        # return same shape of gym: s', r, done
        return self.cur_state, r, False


class DiscreteRewardDynamicsModel(object):

    def __init__(self, num_states: int, num_actions: int, num_rewards: int, start_reward: int = 0):
        """
        Transition function model for a grids

        Args:
            num_states (int): Number of states
            num_actions (int): Number of actions
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_rewards = num_rewards
        self.num_visits_actions = np.zeros(shape=(num_states, num_actions, num_states), dtype=np.int64)
        self.transition_function = np.ones_like(self.num_visits_actions, dtype=np.float64) / num_states
        self.num_visits_rewards = np.zeros(shape=(num_states, num_actions, num_rewards), dtype=np.int64)
        self.reward_function = np.ones_like(self.num_visits_rewards, dtype=np.float64) / num_rewards
        self.start_reward = start_reward
        self.cur_state = 0

    def update_visits(self, from_state: int, action: int, to_state: int, reward: float):
        """Updates the transition function given an experience

        Args:
            from_state (Coordinate): state s
            action (int): action a
            to_state (Coordinate): next state s'
            reward (float): ereward
        """
        self.num_visits_actions[from_state, action, to_state] += 1
        self.num_visits_rewards[from_state, action, reward - self.start_reward] += 1

        self.transition_function[from_state, action] = (
            self.num_visits_actions[from_state, action] / self.num_visits_actions[from_state, action].sum()
        )

        self.reward_function[from_state, action] = (
            (1 + self.num_visits_rewards[from_state, action]) / (1 + self.num_visits_rewards[from_state, action]).sum()
        )

    def reset(self):

        self.cur_state = np.random.randint(self.num_states)
        return self.cur_state

    def step(self, a):

        probs_s = self.transition_function[self.cur_state][a]

        next_state = np.random.choice(range(self.num_states), 1, p= probs_s)[0]
        
        probs_r = self.reward_function[self.cur_state][a]
        r = np.random.choice(range(self.start_reward, self.start_reward + self.num_rewards), 1, p=probs_r)[0]
        self.cur_state = next_state
        # return same shape of gym: s', r, done
        return self.cur_state, r, False
    
    def set_state(self, state):
        self.cur_state = state
    
    def save_functions(self, path: str, id: int):
        os.makedirs(os.path.dirname(path + "data/"), exist_ok=True)
        with lzma.open(path + f"data/transition_model_{id}.pkl", "wb") as f2:
            pickle.dump(self.transition_function, f2)
        with lzma.open(path + f"data/reward_model_{id}.pkl", "wb") as f3:
            pickle.dump(self.reward_function, f3)

    def load_functions(self, path: str, id: int):
        with lzma.open(path + f"data/transition_model_{id}.pkl", "rb") as f2:
            self.transition_function = pickle.load(f2)
        with lzma.open(path + f"data/reward_model_{id}.pkl", "rb") as f3:
            self.reward_function = pickle.load(f3)

class ContinuousRewardDynamicsModel(object):

    def __init__(self, num_states: int, num_actions: int):
        """
        Transition function model for a grids

        Args:
            num_states (int): Number of states
            num_actions (int): Number of actions
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_visits_actions = np.zeros(shape=(num_states, num_actions, num_states), dtype=np.int64)
        self.transition_function = np.ones_like(self.num_visits_actions, dtype=np.float64) / num_states
        self.num_visits_rewards = np.zeros(shape=(num_states, num_actions), dtype=np.int64)
        self.reward_function_mean = np.zeros(shape=(num_states, num_actions), dtype=np.float64)
        self.reward_function_std = np.ones(shape=(num_states, num_actions), dtype=np.float64)
        self.cur_state = 0

    def update_visits(self, from_state: int, action: int, to_state: int, reward: float):
        """Updates the transition function given an experience

        Args:
            from_state (Coordinate): state s
            action (int): action a
            to_state (Coordinate): next state s'
            reward (float): ereward
        """
        self.num_visits_actions[from_state, action, to_state] += 1
        

        self.transition_function[from_state, action] = (
            self.num_visits_actions[from_state, action] / self.num_visits_actions[from_state, action].sum()
        )
        old_mean = self.reward_function_mean[from_state, action]
        old_std = self.reward_function_std[from_state, action]
        past_visits = self.num_visits_rewards[from_state, action]
        self.num_visits_rewards[from_state, action] += 1

        self.reward_function_mean[from_state, action] = (old_mean * past_visits + reward)/(self.num_visits_rewards[from_state, action])
        # E[R^2]
        new_e_r_sq = ((old_std**2 + old_mean**2) * past_visits + reward**2)/(self.num_visits_rewards[from_state, action])
        self.reward_function_std[from_state, action] = np.sqrt(new_e_r_sq - self.reward_function_mean[from_state, action]**2)
        
    def reset(self):

        self.cur_state = np.random.randint(self.num_states)
        return self.cur_state

    def step(self, a):

        probs_s = self.transition_function[self.cur_state][a]

        next_state = np.random.choice(range(self.num_states), 1, p= probs_s)[0]
        
        r_mean  = self.reward_function_mean[self.cur_state, a]
        r_std = self.reward_function_std[self.cur_state, a]
        r = np.random.normal(r_mean, r_std)
        self.cur_state = next_state
        # return same shape of gym: s', r, done
        return self.cur_state, r, False
    
    def save_functions(self, path):
        os.makedirs(os.path.dirname(path + "data/"), exist_ok=True)
        with open(path + "data/transition_model.pkl", "wb") as f2:
            pickle.dump(self.transition_function, f2)
        with open(path + "data/reward_mean_model.pkl", "wb") as f3:
            pickle.dump(self.reward_function_mean, f3)
        with open(path + "data/reward_std_model.pkl", "wb") as f4:
            pickle.dump(self.reward_function_std, f4)

    def load_functions(self, path):
        with open(path + "data/transition_model.pkl", "rb") as f2:
            self.transition_function = pickle.load(f2)
        with open(path + "data/reward_mean_model.pkl", "rb") as f3:
            self.reward_function_mean = pickle.load(f3)
        with open(path + "data/reward_std_model.pkl", "rb") as f4:
            self.reward_function_mean = pickle.load(f4)

    def set_state(self, state):
        self.cur_state = state