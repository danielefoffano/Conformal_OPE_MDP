import numpy as np
import torch
from networks import MLP,WeightsMLP
from utils import compute_weight
import multiprocessing as mp
from utils import collect_exp
from types_cp import Point, Trajectory, Interval
from typing import Sequence
from scipy import signal

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
        
        scores_low, scores_high, scores_orig, scores_cumul = scores[:, 0], scores[:, 1], scores[:, 2], scores[:, 3]
        
        ordered_scores_low, ordered_indexes_low = np.unique(scores_low, return_index=True)
        ordered_scores_high, ordered_indexes_high = np.unique(scores_high, return_index=True)
        ordered_scores_orig, ordered_indexes_orig = np.unique(scores_orig, return_index=True)
        ordered_scores_cumul, ordered_indexes_cumul = np.unique(scores_cumul, return_index=True)
        
        ordered_scores_low = np.concatenate((ordered_scores_low, [np.inf]))
        ordered_scores_high = np.concatenate((ordered_scores_high, [np.inf]))
        ordered_scores_orig = np.concatenate((ordered_scores_orig, [np.inf]))
        ordered_scores_cumul = np.concatenate((ordered_scores_cumul, [np.inf]))
        xb, xa = signal.butter(4, 0.25)

        ordered_weights_low = weights[ordered_indexes_low]
        ordered_weights_high = weights[ordered_indexes_high]
        ordered_weights_orig = weights[ordered_indexes_orig]
        ordered_weights_cumul = weights[ordered_indexes_cumul]

        ordered_weights_filtered_low = signal.filtfilt(xb, xa, ordered_weights_low)
        ordered_weights_filtered_high = signal.filtfilt(xb, xa, ordered_weights_high)
        ordered_weights_filtered_orig = signal.filtfilt(xb, xa, ordered_weights_orig)
        ordered_weights_filtered_cumul= signal.filtfilt(xb, xa, ordered_weights_cumul)

        for test_point in test_points:
            point = Point(test_point.initial_state, test_point.cumulative_reward)
            x = torch.tensor([point.initial_state], dtype = torch.float32)

            lower_val = int(self.lower_quantile_nework(x).item())
            upper_val = int(self.upper_quantile_network(x).item())

            y_vals_test = np.linspace(lower_val // 2,  2 * upper_val, 2 * upper_val-lower_val // 2).astype(int)
            conf_range = []
            conf_range_double = []
            conf_range_cumul = []
            quantile_range = []
            quantile_range_double_low = []
            quantile_range_double_high = []
            quantile_range_cumul_low = []
            quantile_range_cumul_high = []
            
            # Use scores and weights to find the scores quantile to conformalize the predictors
            if n_cpu  == 1 or gradient_based == True:
                for y in y_vals_test:
                    if gradient_based:
                        x = torch.tensor([point.initial_state, y], dtype = torch.float32)
                        test_point_weight = weight_network(x[None,:])[0].item()
                    else:
                        test_point_weight = compute_weight(point.initial_state, y, self.behaviour_policy, self.pi_star, self.model, self.horizon)

                    def normalize_weights(_weights, _test_point):
                        weights_y = np.concatenate((_weights, [_test_point]))
                        normalized_weights_y = weights_y / weights_y.sum()
                        cumsum_weights = np.cumsum(normalized_weights_y)
                        return weights_y, cumsum_weights

                    _, cumsum_low = normalize_weights(ordered_weights_filtered_low, test_point_weight)
                    _, cumsum_high = normalize_weights(ordered_weights_filtered_high, test_point_weight)
                    _, cumsum_orig = normalize_weights(ordered_weights_filtered_orig, test_point_weight)
                    _, cumsum_cumul = normalize_weights(ordered_weights_filtered_cumul, test_point_weight)

                    _quantile_double = 0.95               
                    _quantile_orig = 0.9
                    _quantile_cumul_low = 0.05               
                    _quantile_cumul_high = 0.95
                    
                    quantile_val_low = ordered_scores_low[np.argwhere(cumsum_low >= _quantile_double)][0]
                    quantile_val_high = ordered_scores_high[np.argwhere(cumsum_high >= _quantile_double)][0]
                    quantile_val = ordered_scores_orig[np.argwhere(cumsum_orig >= _quantile_orig)][0]
                    
                    quantile_val_cumul_low = ordered_scores_cumul[np.argwhere(cumsum_cumul >= _quantile_cumul_low)][0]
                    quantile_val_cumul_high = ordered_scores_cumul[np.argwhere(cumsum_cumul >= _quantile_cumul_high)][0]
                    
                    score_test_low = lower_val - y
                    score_test_high = y - upper_val 
                    score_test = max(lower_val - y, y - upper_val)
                    if y >= quantile_val_cumul_low and y <= quantile_val_cumul_high:
                        conf_range_cumul.append(y)
                    
                    if score_test_low <= quantile_val_low and score_test_high <= quantile_val_high:
                        conf_range_double.append(y)

                    if score_test <= quantile_val or np.isinf(quantile_val):
                        conf_range.append(y)
                    quantile_range.append(quantile_val)
                    quantile_range_double_low.append(quantile_val_low)
                    quantile_range_double_high.append(quantile_val_high)
                    
                    quantile_range_cumul_low.append(quantile_val_cumul_low)
                    quantile_range_cumul_high.append(quantile_val_cumul_high)

            else:
                with mp.Pool(n_cpu) as pool:
                    intervals, quantiles = zip(*list(pool.starmap(compute_weight_iterator, [
                    (idx, test_point, y, self.behaviour_policy, self.pi_star, self.model, self.horizon, weights, scores, lower_val, upper_val)
                    for idx, y in enumerate(y_vals_test)])))
            
            
            intervals.append(Interval(
                point, np.min(conf_range), np.max(conf_range), np.min(conf_range_double), np.max(conf_range_double),
                np.min(conf_range_cumul), np.max(conf_range_cumul),
                lower_val, upper_val, quantile_range,
                quantile_range_double_low, quantile_range_double_high,
                quantile_range_cumul_low, quantile_range_cumul_high))
  
        return intervals
