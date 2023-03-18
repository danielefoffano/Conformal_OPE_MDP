import torch
import numpy as np
from utils import compute_weights_gradient, train_weight_function, compute_weight, collect_exp
from networks import WeightsMLP, MLP
from typing import Callable, List, Tuple, Sequence
from types_cp import Trajectory, Point
import multiprocessing as mp
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.typing import NDArray

def model_based_iterator(id_traj: int, traj: Trajectory, cumul_rew: float, behaviour_policy, pi_star, model, horizon, lower_quantile_network, upper_quantile_network):
    state = torch.tensor([traj.trajectory[0].state], dtype = torch.float32)
    # Compute weight generating Monte-Carlo trajectories using learned model
    print("Computing weight of traj {}".format(id_traj))
    
    weight = compute_weight(traj.trajectory[0].state, cumul_rew, behaviour_policy, pi_star, model, horizon)
    
    # Compute score
    score = max(
        lower_quantile_network(state).item() - cumul_rew,
        cumul_rew - upper_quantile_network(state).item())
    return weight, score

class ExactWeightsEstimator(object):
    def __init__(self, pi_b, horizon, env, num_s, n_samples, discount: float):
        
        plt.ioff()
        self.pi_b = pi_b
        self.horizon = horizon
        self.env = env
        self.num_s = num_s
        self.n_samples = n_samples
        self.horizon = horizon
        self.state_x_b = []
        self.state_y_b = []
        self.state_x_target = []
        self.state_y_target = []
        self.discount = discount

        for s in range(num_s):
            
            mc_samples_b = collect_exp(env = env, n_trajectories = n_samples, horizon = horizon, policy = pi_b, start_state = s, model = None, discount=discount)

            rew_cumul_b = [traj.cumulative_reward for traj in mc_samples_b]
            data_x_b, data_y_b = sns.distplot(rew_cumul_b).get_lines()[0].get_data()
            self.state_x_b.append(data_x_b)
            self.state_y_b.append(data_y_b)

    def init_pi_target(self, pi_target):

        self.pi_target = pi_target
        for s in range(self.num_s):
            
            mc_samples_target = collect_exp(env = self.env, n_trajectories = self.n_samples, horizon = self.horizon, policy = pi_target, start_state = s, model = None, discount=self.discount)

            rew_cumul_target = [traj.cumulative_reward for traj in mc_samples_target]
            data_x_target, data_y_target = sns.distplot(rew_cumul_target).get_lines()[0].get_data()
            self.state_x_target.append(data_x_target)
            self.state_y_target.append(data_y_target)

    
    def compute_true_ratio(self, traj: Trajectory) -> float:
        s = traj.initial_state
        cumul_rew = traj.cumulative_reward
        p_rew_cum_b = np.interp(cumul_rew, self.state_x_b[s], self.state_y_b[s])
        p_rew_cum_target = np.interp(cumul_rew, self.state_x_target[s], self.state_y_target[s])
        return p_rew_cum_target/p_rew_cum_b
    
    def compute_true_ratio_dataset(self, dataset: Sequence[Trajectory]) -> NDArray[np.float64]:
        return np.array([self.compute_true_ratio(data_point) for data_point in dataset])


class WeightsEstimator(object):
    def __init__(self, behaviour_policy, pi_star, lower_quantile_network: MLP, upper_quantile_network: MLP):
        self.behaviour_policy = behaviour_policy
        self.pi_star = pi_star
        self.lower_quantile_network = lower_quantile_network
        self.upper_quantile_network = upper_quantile_network
        
    def model_based(self, data_tr: Sequence[Trajectory], data_cal: Sequence[Trajectory], horizon: int, model, n_cpu: int = 2) -> Tuple[List[float], List[float]]:
        # Compute weights and scores - Model-Based approach
        weights = []
        scores = []

        if n_cpu == 1:
            results = [
                model_based_iterator(id_traj, traj, traj.cumulative_reward, self.behaviour_policy, self.pi_star, model, horizon, self.lower_quantile_network, self.upper_quantile_network)
                for id_traj, traj in enumerate(data_cal)]
        else:
            with mp.Pool(n_cpu) as pool:
                results = list(pool.starmap(model_based_iterator, [
                    (id_traj, traj, traj.cumulative_reward, self.behaviour_policy, self.pi_star, model, horizon, self.lower_quantile_network, self.upper_quantile_network)
                    for id_traj, traj in enumerate(data_cal)]))
            
        weights, scores = zip(*results)
        weights = np.array(weights)
        scores = list(scores)
        scores.append(np.inf)
        scores = np.array(scores)
        return scores, weights

    def gradient_method(self, data_tr: Sequence[Trajectory], data_cal: Sequence[Trajectory], lr: float, epochs: int, make_network: Callable[[], WeightsMLP]) -> Tuple[List[float], List[float]]:
        scores = []
        weights = []

        # Compute training weights for gradient approach
        print("> Computing weight of trajectories")
        for traj in data_tr:
            weight = compute_weights_gradient(traj, self.behaviour_policy, self.pi_star)       
            weights.append(weight)

        # Compute weights and scores (on calibration data) - gradient approach
        weight_network = make_network()
        train_weight_function(data_tr, weights, weight_network, lr, epochs, self.behaviour_policy, self.pi_star)

        calibration_weights = []
        for traj in data_cal:
            # Compute weight
            x = torch.tensor([traj.initial_state, traj.cumulative_reward], dtype = torch.float32)
            w = weight_network(x[None,...])[0].item()
            calibration_weights.append(w)

            # Compute score
            state = torch.tensor([traj.initial_state], dtype = torch.float32)
            
            score_low = self.lower_quantile_network(state).item() - traj.cumulative_reward
            score_high = traj.cumulative_reward - self.upper_quantile_network(state).item()
            score = max(score_low, score_high)
            score_cumul = traj.cumulative_reward
            scores.append((score_low, score_high, score, score_cumul))

        #weights = calibration_weights
        
        weights = np.array(calibration_weights)
        scores = np.array(scores)
        
        return scores, weights, weight_network