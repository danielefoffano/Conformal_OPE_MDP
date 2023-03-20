import numpy as np
import torch
from networks import MLP,WeightsMLP
from utils import compute_weight
import multiprocessing as mp
from utils import collect_exp, filter_scores_weights, unique_scores_weights
from types_cp import Point, Trajectory, Interval, ScoresWeightsData
from typing import Sequence
import torch.nn as nn
from numpy.typing import NDArray
from copy import deepcopy

def compute_cdf(_weights: NDArray[np.float64], _test_point: float):
    new_weights = np.concatenate((_weights, [_test_point]))
    weighted_density = new_weights / np.sum(new_weights)
    return np.cumsum(weighted_density)
    
def compute_weight_iterator(idx, test_point, y, behaviour_policy, pi_star, model, horizon, weights, scores, lower_val, upper_val):
    print("Computing weight for test value {}".format(idx))
    
    test_point_weight = compute_weight(test_point[0][0].state, y, behaviour_policy, pi_star, model, horizon)
    norm_weights = weights/(weights.sum() + test_point_weight)
    norm_weights = np.concatenate((norm_weights, [test_point_weight/(weights.sum() + test_point_weight)]))
    norm_weights = torch.tensor(norm_weights)

    ordered_indexes = scores.argsort()
    ordered_scores = scores[ordered_indexes]
    ordered_weights = norm_weights[ordered_indexes]

    quantile = 0.90

    cumsum = torch.cumsum(ordered_weights, 0)

    quantile_val = ordered_scores[cumsum>quantile][0].item()

    #score_test = max(lower_val - y, y - upper_val)

    #if score_test <= quantile_val:
    #    conf_range.append(y)
    
    return ([test_point[0][0].state, y, lower_val - quantile_val, upper_val + quantile_val, test_point[1]], quantile_val)


def evaluate_trajectory(test_point: Point, lower_quantile_network: nn.Module, upper_quantile_network: nn.Module,
                        weight_network: nn.Module,
                        data_low: ScoresWeightsData,
                        data_high: ScoresWeightsData,
                        data_orig: ScoresWeightsData,
                        data_cumul: ScoresWeightsData):
    point = Point(test_point.initial_state, test_point.cumulative_reward)
    x = torch.tensor([point.initial_state], dtype = torch.float32)

    lower_val = int(lower_quantile_network(x).item())
    upper_val = int(upper_quantile_network(x).item())
    
    start_point, end_point = lower_val // 2, 2 * upper_val
    y_vals_test = np.linspace(start_point, end_point, end_point - start_point).astype(int)
    conf_range = []
    conf_range_double = []
    conf_range_cumul = []
    quantile_range = []
    quantile_range_double_low = []
    quantile_range_double_high = []
    quantile_range_cumul_low = []
    quantile_range_cumul_high = []
    
    def evaluate_point(s, y):
        x = torch.tensor([s, y], dtype = torch.float32)
        test_point_weight = weight_network(x[None,:])[0].item()
        
        s_low = lower_val - y
        s_high = y - upper_val
        s_orig = max(s_low, s_high)
        s_cumul = y
        cdf_low = compute_cdf(data_low.weights, test_point_weight)
        cdf_high = compute_cdf(data_high.weights, test_point_weight)
        cdf_orig = compute_cdf(data_orig.weights, test_point_weight)
        cdf_cumul = compute_cdf(data_cumul.weights, test_point_weight)
        

        _quantile_double = 0.95      
        _quantile_orig = 0.9
        _quantile_cumul_low = 0.05               
        _quantile_cumul_high = 0.95
        
        quantile_val_low = data_low.scores[np.argwhere(cdf_low >= _quantile_double)][0].item()
        quantile_val_high = data_high.scores[np.argwhere(cdf_high >= _quantile_double)][0].item()
        quantile_val = data_orig.scores[np.argwhere(cdf_orig >= _quantile_orig)][0].item()
        
        quantile_val_cumul_low = data_cumul.scores[np.argwhere(cdf_cumul >= _quantile_cumul_low)][0].item()
        quantile_val_cumul_high = data_cumul.scores[np.argwhere(cdf_cumul >= _quantile_cumul_high)][0].item()
        
        if s_cumul >= quantile_val_cumul_low and s_cumul <= quantile_val_cumul_high:
                conf_range_cumul.append(y)
            
        if max(s_low, s_high) <= quantile_val_low and max(s_low, s_high) <= quantile_val_high:
            conf_range_double.append(y)

        if s_orig <= quantile_val or np.isinf(quantile_val):
            conf_range.append(y)
        quantile_range.append(quantile_val)
        quantile_range_double_low.append(quantile_val_low)
        quantile_range_double_high.append(quantile_val_high)
        
        quantile_range_cumul_low.append(quantile_val_cumul_low)
        quantile_range_cumul_high.append(quantile_val_cumul_high)

    for y in y_vals_test:
        evaluate_point(point.initial_state,y)
        
    return Interval(
        point, np.min(conf_range), np.max(conf_range), np.min(conf_range_double), np.max(conf_range_double),
        np.min(conf_range_cumul), np.max(conf_range_cumul),
        lower_val, upper_val, quantile_range,
        quantile_range_double_low, quantile_range_double_high,
        quantile_range_cumul_low, quantile_range_cumul_high)


class ConformalSet(object):
    def __init__(self, lower_quantile_network: MLP, upper_quantile_network: MLP,  behaviour_policy, pi_star, model, horizon: int, discount: float):
        self.lower_quantile_nework = lower_quantile_network
        self.upper_quantile_network = upper_quantile_network
        self.behaviour_policy = behaviour_policy
        self.pi_star = pi_star
        self.model = model
        self.horizon = horizon
        self.discount = discount

    def build_set_monte_carlo(self, test_point, alpha, n_samples, quantile_upper, quantile_lower):

        mc_samples = collect_exp(env = self.model, n_trajectories = n_samples, horizon = self.horizon, policy = self.pi_star, start_state = test_point[0][0].state, model = None, discount=self.discount)

        rewards_samples = np.array(mc_samples, dtype = 'object')[:,1]
        rewards_samples = rewards_samples.astype('int64') - self.model.start_reward

        counts = np.zeros((self.horizon * self.model.num_rewards, ))
        rew_count = np.bincount(rewards_samples)
        counts[:len(rew_count)] = rew_count
        counts = (counts + alpha)/(n_samples + self.horizon*self.model.num_rewards)

        cumsum = np.cumsum(counts, 0)
        
        quantile_val_up = np.argwhere(cumsum >= quantile_upper)[0]
        quantile_val_low = np.argwhere(cumsum >= quantile_lower)[0]

        return quantile_val_up[0], quantile_val_low[0]
    
    def build_set(self, test_points: Sequence[Trajectory], weights: Sequence[float],
                  scores: Sequence[float], n_cpu: int = 2, weight_network: WeightsMLP = None, gradient_based: bool = False) -> Sequence[Interval]:
        intervals: Sequence[Interval] = []
        
        ordered_scores_low, ordered_weights_low = filter_scores_weights(scores[:, 0], weights)
        ordered_scores_high, ordered_weights_high = filter_scores_weights(scores[:, 1], weights)
        ordered_scores_orig, ordered_weights_orig = filter_scores_weights(scores[:, 2], weights)
        ordered_scores_cumul, ordered_weights_cumul = filter_scores_weights(scores[:, 3], weights)

        ordered_scores_low = np.concatenate((ordered_scores_low, [np.inf]))
        ordered_scores_high = np.concatenate((ordered_scores_high, [np.inf]))
        ordered_scores_orig = np.concatenate((ordered_scores_orig, [np.inf]))
        ordered_scores_cumul = np.concatenate((ordered_scores_cumul, [np.inf]))
            
        data_low = ScoresWeightsData(ordered_scores_low, ordered_weights_low)
        data_high = ScoresWeightsData(ordered_scores_high, ordered_weights_high)
        data_orig = ScoresWeightsData(ordered_scores_orig, ordered_weights_orig)
        data_cumul = ScoresWeightsData(ordered_scores_cumul, ordered_weights_cumul)
        
        with mp.Pool(n_cpu) as p:
            intervals = list(p.starmap(evaluate_trajectory, [
                (test_point, deepcopy(self.lower_quantile_nework), deepcopy(self.upper_quantile_network),
                 deepcopy(weight_network),
                 deepcopy(data_low), deepcopy(data_high), deepcopy(data_orig), deepcopy(data_cumul)) for test_point in test_points]))
        
 
        return intervals
