import numpy as np

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

    def __init__(self, num_states: int, num_actions: int, num_rewards: int):
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
        self.num_visits_rewards[from_state, action, reward] += 1

        self.transition_function[from_state, action] = (
            self.num_visits_actions[from_state, action] / self.num_visits_actions[from_state, action].sum()
        )

        self.reward_function[from_state, action] = (
            self.num_visits_rewards[from_state, action] / self.num_visits_rewards[from_state, action].sum()
        )

    def reset(self):

        self.cur_state = np.random.randint(self.num_states)
        return self.cur_state

    def step(self, a):

        probs_s = self.transition_function[self.cur_state][a]

        next_state = np.random.choice(range(self.num_states), 1, p= probs_s)[0]
        
        probs_r = self.reward_function[self.cur_state][a]
        r = np.random.choice(range(self.num_rewards), 1, p=probs_r)[0]
        self.cur_state = next_state
        # return same shape of gym: s', r, done
        return self.cur_state, r, False