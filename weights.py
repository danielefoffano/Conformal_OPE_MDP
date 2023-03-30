import torch
import numpy as np
from utils import compute_weights_gradient, train_weight_function, compute_weight, collect_exp
from networks import WeightsMLP, MLP
from typing import Callable, List, Tuple, Sequence
from types_cp import Trajectory, Point, Scores
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from scipy.stats import gaussian_kde
from policy import Policy
from abc import ABC, abstractmethod

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
    def __init__(self, pi_b: Policy, horizon: int, env, num_s: int, n_samples: int , discount: float):
        
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
        self.state_pdf_estimator_behav: List[gaussian_kde] = []
        self.state_pdf_estimator_target: List[gaussian_kde] = []
        self.discount = discount

        for s in range(num_s):
            
            mc_samples_b = collect_exp(env = env, n_trajectories = n_samples, horizon = horizon, policy = pi_b, start_state = s, model = None, discount=discount)

            rew_cumul_b = [traj.cumulative_reward for traj in mc_samples_b]
            self.state_pdf_estimator_behav.append(gaussian_kde(rew_cumul_b))

    def init_pi_target(self, pi_target: Policy):

        self.pi_target = pi_target
        for s in range(self.num_s):
            
            mc_samples_target = collect_exp(env = self.env, n_trajectories = self.n_samples, horizon = self.horizon, policy = pi_target, start_state = s, model = None, discount=self.discount)

            rew_cumul_target = [traj.cumulative_reward for traj in mc_samples_target]
            self.state_pdf_estimator_target.append(gaussian_kde(rew_cumul_target))

    
    def compute_true_ratio(self, point: Point) -> float:
        s = point.initial_state
        cumul_rew = point.cumulative_reward

        p_rew_cum_b = self.state_pdf_estimator_behav[s].evaluate(cumul_rew).item()
        p_rew_cum_target = self.state_pdf_estimator_target[s].evaluate(cumul_rew).item()
        
        ratio = p_rew_cum_target / p_rew_cum_b
        return ratio
    
    def compute_true_ratio_dataset(self, dataset: Sequence[Trajectory]) -> NDArray[np.float64]:
        return np.array([self.compute_true_ratio(Point(data_point.initial_state, data_point.cumulative_reward)) for data_point in dataset])



class WeightsEstimator(ABC):
    def __init__(self, behaviour_policy: Policy, pi_star: Policy, lower_quantile_network: MLP, upper_quantile_network: MLP):
        self.behaviour_policy = behaviour_policy
        self.pi_star = pi_star
        self.lower_quantile_network = lower_quantile_network
        self.upper_quantile_network = upper_quantile_network
    
    @abstractmethod
    def train(self) -> None:
        pass
    
    @abstractmethod
    def evaluate_point(self, point: Point) -> float:
        pass
    
    def evaluate_scores(self, point: Point) -> Scores:
        # Compute score
        state = torch.tensor([point.initial_state], dtype = torch.float32)
        lower_quantile = self.lower_quantile_network(state).item()
        upper_quantile = self.upper_quantile_network(state).item()
        
        score_low = lower_quantile - point.cumulative_reward
        score_high = point.cumulative_reward - upper_quantile
        score = max(score_low, score_high)
        score_cumul = point.cumulative_reward
        return Scores(score_low, score_high, score, score_cumul)
    
    def evaluate_calibration_data(self, data_cal: Sequence[Trajectory]) ->  Tuple[NDArray[np.float64], NDArray[np.float64]]:
        calibration_weights = []
        scores = []
        for traj in data_cal:
            # Compute weight
            point = Point(traj.initial_state, traj.cumulative_reward)
            calibration_weights.append(self.evaluate_point(point))
            point_scores = self.evaluate_scores(point)
            scores.append((point_scores.score_low, point_scores.score_high, point_scores.score_orig, point_scores.score_cumul))
    
        weights = np.array(calibration_weights)
        scores = np.array(scores)
        
        return scores, weights


class GradientWeightsEstimator(WeightsEstimator):
    def __init__(self, behaviour_policy: Policy, pi_star: Policy, lower_quantile_network: MLP, upper_quantile_network: MLP, make_network: Callable[[], WeightsMLP]):
        super().__init__(behaviour_policy, pi_star, lower_quantile_network, upper_quantile_network)
        self.weight_network = make_network()
    
    def train(self, data_tr: Sequence[Trajectory], lr: float, epochs: int):
        # Compute training weights for gradient approach
        print("> Computing weight of trajectories")
        weights = [compute_weights_gradient(traj, self.behaviour_policy, self.pi_star) for traj in data_tr]
        # import pdb
        # pdb.set_trace()
        train_weight_function(data_tr, weights, self.weight_network, lr, epochs, self.behaviour_policy, self.pi_star)

    @torch.no_grad()
    def evaluate_point(self, point: Point) -> float:
        x = torch.tensor([point.initial_state, point.cumulative_reward], dtype = torch.float32)
        w = self.weight_network(x[None,...])[0].item()
        return w

    

class EmpiricalWeightsEstimator(WeightsEstimator):
    def __init__(self, behaviour_policy: Policy, pi_star: Policy, lower_quantile_network: MLP, upper_quantile_network: MLP, num_states: int, num_rewards: int):
        super().__init__(behaviour_policy, pi_star, lower_quantile_network, upper_quantile_network)
        self.num_states = num_states
        self.num_rewards = num_rewards
        self.weights_estimator = np.zeros((self.num_states, self.num_rewards))
    
    def train(self, data_tr: Sequence[Trajectory]):
        self.weights_estimator = np.zeros((self.num_states, self.num_rewards))
        num_visits = np.zeros_like(self.weights_estimator)
        for traj in data_tr:
            x,y = traj.initial_state, int(traj.cumulative_reward)
            weight = compute_weights_gradient(traj, self.behaviour_policy, self.pi_star) 
            num_visits[x,y] += 1
            self.weights_estimator[x,y] += weight
        
        mask = num_visits > 0
        self.weights_estimator[mask] = self.weights_estimator[mask] / num_visits[mask]
        self.weights_estimator[~mask] = 1 / len(data_tr)

    def evaluate_point(self, point: Point) -> float:
        x,y = point.initial_state, int(point.cumulative_reward)
        w = self.weights_estimator[x,y]
        return w
