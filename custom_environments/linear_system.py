
import numpy as np
import scipy.stats as stats
from typing import Tuple
import scipy.signal as scipysig
from numpy.typing import NDArray

class LinearSystem(object):
    ''' Code from https://github.com/rssalessio/data-poisoning-linear-systems/blob/main/example_residuals_maximization/main.py
        noise_level: standard deviation of the gaussian noise
    '''
    def __init__(self, noise_level: float = 0.1):
        self.dt = 0.05
        self.num = [0.28261, 0.50666]
        self.den = [1, -1.41833, 1.58939, -1.31608, 0.88642]
        self.sys = scipysig.TransferFunction(self.num, self.den, dt=self.dt).to_ss()
        self.AB = np.hstack((self.sys.A, self.sys.B))
        self.dim_x = self.sys.B.shape[0]
        self.dim_u = self.sys.B.shape[1]
        self.noise_std = noise_level
        self.reset()
        
    def reward(self, state: NDArray[np.float64], action: float):
        x = np.concatenate((state, [action]))
        return -np.linalg.norm(x, 2)

    def step(self, action: float) -> Tuple[int, float]:
        ''' Takes an action and returns next state and reward '''
        noise = self.noise_std * np.random.normal(size=self.dim_x)
        next_state = (self.AB @ np.concatenate((self.state, [action]))).flatten() + noise
        reward = self.reward(self.state, action)
        self.state = next_state
        done = False
        return self.state, reward, done

    def reset(self) -> int:
        ''' Resets the environment to the initial state '''
        self.state = np.random.normal(size=self.dim_x)
        return self.state
    
    def set_state(self, state: NDArray[np.float64]):
        self.state = state