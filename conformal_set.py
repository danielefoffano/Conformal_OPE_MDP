import numpy as np
import torch
from networks import MLP,WeightsMLP
from utils import compute_weight
import multiprocessing as mp

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
    
    return [test_point[0][0].state, y, lower_val - quantile_val, upper_val + quantile_val, test_point[1]]

class ConformalSet(object):
    def __init__(self, lower_quantile_network: MLP, upper_quantile_network: MLP,  behaviour_policy, pi_star, model, horizon):
        self.lower_quantile_nework = lower_quantile_network
        self.upper_quantile_network = upper_quantile_network
        self.behaviour_policy = behaviour_policy
        self.pi_star = pi_star
        self.model = model
        self.horizon = horizon
    
    def build_set(self, test_points, weights, scores, n_cpu: int = 2, weight_network: WeightsMLP = None, gradient_based: bool = False):
        intervals = []
        for test_point in test_points:
            #scores = torch.tensor(scores)
            x = torch.tensor([test_point[0][0].state], dtype = torch.float32)
            lower_val = int(self.lower_quantile_nework(x).item())
            upper_val = int(self.upper_quantile_network(x).item())

            y_vals_test = np.linspace(lower_val, upper_val, upper_val-lower_val).astype(int)

            #conf_range = []
            
            # Use scores and weights to find the scores quantile to conformalize the predictors
            if n_cpu  == 1 or gradient_based == True:
                for y in y_vals_test:
                    if gradient_based:
                        x = torch.tensor([test_point[0][0].state, y], dtype = torch.float32)
                        test_point_weight = weight_network(x).item()
                    else:
                        test_point_weight = compute_weight(test_point[0][0].state, y, self.behaviour_policy, self.pi_star, self.model, self.horizon)
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
                    
                    intervals.append([test_point[0][0].state, y, lower_val - quantile_val, upper_val + quantile_val, test_point[1]])
            else:
                with mp.Pool(n_cpu) as pool:
                    intervals = list(pool.starmap(compute_weight_iterator, [
                    (idx, test_point, y, self.behaviour_policy, self.pi_star, self.model, self.horizon, weights, scores, lower_val, upper_val)
                    for idx, y in enumerate(y_vals_test)]))

                print("")
                
                

        return np.array(intervals), lower_val, upper_val
