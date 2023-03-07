import torch
import numpy as np
from utils import compute_weights_gradient, train_weight_function, compute_weight
from networks import WeightsMLP, MLP
from typing import Callable, List, Tuple
import multiprocessing as mp

def model_based_iterator(id_traj, traj, cumul_rew, behaviour_policy, pi_star, model, horizon, lower_quantile_network, upper_quantile_network):
    state = torch.tensor([traj[0].state], dtype = torch.float32)
    # Compute weight generating Monte-Carlo trajectories using learned model
    print("Computing weight of traj {}".format(id_traj))
    
    weight = compute_weight(traj[0].state, cumul_rew, behaviour_policy, pi_star, model, horizon)
    
    # Compute score
    score = max(
        lower_quantile_network(state).item() - cumul_rew,
        cumul_rew - upper_quantile_network(state).item())
    return weight, score

class WeightsEstimator(object):
    def __init__(self, behaviour_policy, pi_star, lower_quantile_network: MLP, upper_quantile_network: MLP):
        self.behaviour_policy = behaviour_policy
        self.pi_star = pi_star
        self.lower_quantile_network = lower_quantile_network
        self.upper_quantile_network = upper_quantile_network
        
    def model_based(self, data_tr, data_cal, horizon: int, model, n_cpu: int = 2) -> Tuple[List[float], List[float]]:
        # Compute weights and scores - Model-Based approach
        weights = []
        scores = []

        if n_cpu == 1:
            results = [
                model_based_iterator(id_traj, traj, cumul_rew, self.behaviour_policy, self.pi_star, model, horizon, self.lower_quantile_network, self.upper_quantile_network)
                for id_traj, (traj, cumul_rew) in enumerate(data_cal)]
        else:
            with mp.Pool(n_cpu) as pool:
                results = list(pool.starmap(model_based_iterator, [
                    (id_traj, traj, cumul_rew, self.behaviour_policy, self.pi_star, model, horizon, self.lower_quantile_network, self.upper_quantile_network)
                    for id_traj, (traj, cumul_rew) in enumerate(data_cal)]))
            
        weights, scores = zip(*results)
        weights = np.array(weights)
        scores = list(scores)
        scores.append(np.inf)
        scores = np.array(scores)
        return scores, weights

    def gradient_method(self, data_tr, data_cal, lr: float, epochs: int, make_network: Callable[[], WeightsMLP]) -> Tuple[List[float], List[float]]:
        scores = []
        weights = []
        traj_idx = 0

        # Compute training weights for gradient approach
        print("> Computing weight of trajectories")
        for traj, cumul_rew in data_tr:
            
            weight = compute_weights_gradient(traj, self.behaviour_policy, self.pi_star)       
            weights.append(weight)
            
            traj_idx += 1

        # Compute weights and scores (on calibration data) - gradient approach
        weight_network = make_network()
        train_weight_function(data_tr, weights, weight_network, lr, epochs, self.behaviour_policy, self.pi_star)

        calibration_weights = []
        for traj, cumul_rew in data_cal:
            # Compute weight
            x = torch.tensor([traj[0].state, cumul_rew], dtype = torch.float32)
            w  = weight_network(x).item()
            calibration_weights.append(w)

            # Compute score
            state = torch.tensor([traj[0].state], dtype = torch.float32)
            score = max(self.lower_quantile_network(state).item() - cumul_rew, cumul_rew - self.upper_quantile_network(state).item())
            scores.append(score)

        weights = calibration_weights
        
        weights = np.array(weights)
        scores.append(np.inf)
        scores = np.array(scores)
        
        return scores, weights, weight_network