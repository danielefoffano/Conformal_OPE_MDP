import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


import numpy as np
from random_mdp import MDPEnv, MDPEnvDiscreteRew, MDPEnvBernoulliRew
from agent import QlearningAgent
from greedy_policy import EpsilonGreedyPolicy, TableBasedPolicy, MixedPolicy
from utils import get_data, collect_exp, train_predictor, train_behaviour_policy, value_iteration, save_important_dictionary
from networks import MLP, WeightsMLP, WeightsTransformerMLP
from dynamics_model import DynamicsModel, DiscreteRewardDynamicsModel, ContinuousRewardDynamicsModel
import torch
from collections import defaultdict
import pickle
import random
from weights import WeightsEstimator, ExactWeightsEstimator
from conformal_set import ConformalSet
from custom_environments.inventory import Inventory
from multiprocessing import freeze_support
from logger import Logger
import os
import argparse

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Specify horizon')
    parser.add_argument('--horizon', type=int)

    args = parser.parse_args()
    print(f'Horizon chosen {args.horizon}')
    set_seed(int(args.horizon))

    RUNS_NUMBER = 30
    N_CPU = 8
    ENV_NAME = "inventory"
    REWARD_TYPE = "discrete_multiple"
    GRADIENT_BASED = True
    TRANSFORMER = False
    EPSILON = 0.4
    QUANTILE = 0.1
    LR = 1e-4
    MOMENTUM = 0.9
    EPOCHS = 300
    NUM_ACTIONS = 10                                                                    # MDP action space size
    NUM_STATES = 10                                                                     # MDP states space size
    NUM_REWARDS = 10                                                                    # MDP reward space size (for discrete MDP)
    DISCOUNT_FACTOR = 0.99                                                              # behaviour agent discount factor
    ALPHA = 0.6                                                                         # behaviour agent alpha
    NUM_STEPS = 20000                                                                   # behaviour agent learning steps
    N_TRAJECTORIES = 40000                                                              # number of trajectories collected as dataset
    HORIZONS = [int(args.horizon)]                                                           # trajectory horizon
    NUM_TEST_POINTS = 100

    P = np.random.dirichlet(np.ones(NUM_STATES), size=(NUM_STATES, NUM_ACTIONS))        # MDP transition probability functions
    if ENV_NAME == "random_mdp":
        if REWARD_TYPE == "bernoulli":
            R = np.random.rand(NUM_STATES, NUM_ACTIONS)                                     # MDP reward function Bernoulli
            env = MDPEnvBernoulliRew(NUM_STATES, NUM_ACTIONS, P, R)
            model = DiscreteRewardDynamicsModel(NUM_STATES, NUM_ACTIONS, NUM_REWARDS)
        elif REWARD_TYPE == "discrete_multiple":                                 
            R = np.random.dirichlet(np.ones(NUM_REWARDS), size=(NUM_STATES, NUM_ACTIONS))   # MDP reward function multiple discrete r values
            env = MDPEnvDiscreteRew(NUM_STATES, NUM_ACTIONS, NUM_REWARDS, P, R)
            model = DiscreteRewardDynamicsModel(NUM_STATES, NUM_ACTIONS, NUM_REWARDS)
        elif REWARD_TYPE == "continuous":
            R = np.random.rand(NUM_STATES, NUM_ACTIONS, NUM_STATES)
            env = MDPEnv(NUM_STATES, NUM_ACTIONS, P, R)
            #model = DiscreteRewardDynamicsModel(NUM_STATES, NUM_ACTIONS, NUM_REWARDS)
            model = ContinuousRewardDynamicsModel(NUM_STATES, NUM_ACTIONS)
        
        pi_star_probs = np.random.dirichlet(np.ones(NUM_ACTIONS), size=(NUM_STATES))
        pi_star_pre = TableBasedPolicy(pi_star_probs)
        
        #Train behaviour policy using Q-learning
        agent = QlearningAgent(NUM_STATES, NUM_ACTIONS, DISCOUNT_FACTOR, ALPHA)

        q_table = train_behaviour_policy(env, agent, NUM_STEPS)
        behaviour_policy = EpsilonGreedyPolicy(q_table, EPSILON, NUM_ACTIONS)

    elif ENV_NAME == "inventory":
        env = Inventory(inventory_size = NUM_STATES, fixed_cost = 1, order_rate = 10, seed = 1)
        NUM_STATES += 1
        NUM_ACTIONS += 1
        NUM_REWARDS = env.max_r - env.min_r + 1
        model = DiscreteRewardDynamicsModel(NUM_STATES, NUM_ACTIONS, NUM_REWARDS, env.min_r)

        # Compute optimal policy with VI
        VI_v, VI_pi = value_iteration(env, DISCOUNT_FACTOR)
        behaviour_policy = EpsilonGreedyPolicy(VI_pi, EPSILON, NUM_ACTIONS)

        greedy_policy = EpsilonGreedyPolicy(VI_pi, 0, NUM_ACTIONS)

        # Uniform policy
        pi_star_probs = np.ones(shape=(NUM_STATES,NUM_ACTIONS))/NUM_ACTIONS
        pi_uniform = TableBasedPolicy(pi_star_probs)
    
    test_state = np.random.randint(env.ns)

    for HORIZON in HORIZONS:
        print(f'Starting with horizon: {HORIZON}')
        method = "gradient_based" if GRADIENT_BASED else "model_based"
        columns = ["epsilon", 
                    "Coverage", 
                    "Avg_length",
                    "Std_length", 
                    "Original_interval_bottom", 
                    "Original_interval_upper", 
                    "quantile",
                    "std_quantile", 
                    "horizon", 
                    "epsilon_pi_b", 
                    "avg_weights",
                    "std_weights", 
                    "w_hat/w",
                    "std_w_hat/w", 
                    "median_w_hat/w",
                    "avg_delta_w",
                    "std_delta_w",
                    "median_delta_w"]
        
        path = f"results/{ENV_NAME}/{method}/horizon_{HORIZON}/"

        exact_weights_estimator = ExactWeightsEstimator(behaviour_policy, HORIZON, env, NUM_STATES, 30000)

        for RUN_NUMBER in range(RUNS_NUMBER):
            file_logger = Logger(path+f"run_{RUN_NUMBER}.csv", columns)
            #Collect experience data using behaviour policy and train model
            model, dataset = get_data(env, N_TRAJECTORIES, behaviour_policy, model, HORIZON, path)

            #Split dataset into training (90%) and calibration data (10%)
            calibration_trajectories = N_TRAJECTORIES // 10
            data_tr = dataset[:N_TRAJECTORIES - calibration_trajectories]
            data_cal = dataset[N_TRAJECTORIES - calibration_trajectories:N_TRAJECTORIES]

            #Train quantile predictors using training dataset
            print('> Training/loading quantile networks')
            upper_quantile_net = MLP(1, 64, 1, False)
            lower_quantile_net = MLP(1, 64, 1, False)

            if not lower_quantile_net.load(path + 'data/networks/lower_quantile_net.pth'):
                y_avg, y_std = train_predictor(lower_quantile_net, data_tr, epochs=EPOCHS, quantile=QUANTILE/2, lr=LR, momentum=MOMENTUM)
                lower_quantile_net.set_normalization(y_avg, y_std)
                os.makedirs(os.path.dirname(path + 'data/networks/lower_quantile_net.pth'), exist_ok=True)
                lower_quantile_net.save(path + 'data/networks/lower_quantile_net.pth')

            if not upper_quantile_net.load(path + 'data/networks/upper_quantile_net.pth'):
                y_avg, y_std = train_predictor(upper_quantile_net, data_tr, epochs=EPOCHS, quantile=1-(QUANTILE/2), lr=LR, momentum=MOMENTUM)
                upper_quantile_net.set_normalization(y_avg, y_std)
                os.makedirs(os.path.dirname(path + 'data/networks/upper_quantile_net.pth'), exist_ok=True)
                upper_quantile_net.save(path + 'data/networks/upper_quantile_net.pth')


            epsilons = np.linspace(0, 1, 21)
            epsilon_lengths = []
            for epsilon_value in epsilons:

                pi_target = MixedPolicy(pi_uniform, greedy_policy, epsilon_value)                

                test_points = collect_exp(env, NUM_TEST_POINTS, HORIZON, pi_target, None, test_state)

                print(f'> Estimate weights for calibration data')
                weights_estimator = WeightsEstimator(behaviour_policy, pi_target, lower_quantile_net, upper_quantile_net)
                exact_weights_estimator.init_pi_target(pi_target)

                if GRADIENT_BASED:
                    if TRANSFORMER: 
                        scores, weights, weight_network = weights_estimator.gradient_method(data_tr, data_cal, LR, EPOCHS, lambda:WeightsTransformerMLP(2 + 2*NUM_STATES*NUM_ACTIONS, 64, 1, upper_quantile_net.mean, upper_quantile_net.std, behaviour_policy, pi_target))
                    else: 
                        scores, weights, weight_network = weights_estimator.gradient_method(data_tr, data_cal, LR, EPOCHS, lambda:WeightsMLP(2, 64, 1, upper_quantile_net.mean, upper_quantile_net.std))
                else:
                    scores, weights = weights_estimator.model_based(data_tr, data_cal, HORIZON, model, N_CPU)
                    weight_network = None

                true_weights = exact_weights_estimator.compute_true_ratio_dataset(data_cal)
                
                # Generate y values for test point
                print(f'> Computing conformal set')
                conformal_set = ConformalSet(lower_quantile_net, upper_quantile_net, behaviour_policy, pi_target, model, HORIZON)

                save_important_dictionary(env, weights_estimator, exact_weights_estimator, conformal_set, weights, scores, weight_network, path, RUN_NUMBER, epsilon_value)

                intervals, lower_quantile, upper_quantile, quantiles = conformal_set.build_set(test_points, weights, scores, N_CPU, weight_network, GRADIENT_BASED)

                included = 0
                lengths = []
                true_values = []
                for interval in intervals:
                    if interval[-1] >= interval[2] and interval[-1] <= interval[3]:
                        included += 1
                    lengths.append(interval[3]-interval[2])
                    true_values.append(interval[-1])

                included = included/len(intervals)
                mean_length = np.mean(lengths)
                epsilon_lengths.append(mean_length)

                print("Epsilon: {} | Coverage: {:.2f}% | Average interval length: {} | Original interval: {}-{}| Quantile: {:.3f} | Avg weights: {:.3f}| w_hat/w: {:.3f}| avg_delta_w: {:.3f}".format(epsilon_value, included*100, mean_length, lower_quantile, upper_quantile, np.mean(quantiles), np.mean(weights), np.mean(weights/true_weights), 0.5*np.mean(np.abs(true_weights-weights))))
                file_logger.write([
                    epsilon_value, 
                    included*100, 
                    mean_length,
                    np.std(lengths), 
                    lower_quantile, 
                    upper_quantile, 
                    np.mean(quantiles),
                    np.std(quantiles), 
                    HORIZON, 
                    EPSILON, 
                    np.mean(weights),
                    np.std(weights),
                    np.mean(weights/true_weights),
                    np.std(weights/true_weights), 
                    np.median(weights/true_weights),
                    0.5*np.mean(np.abs(true_weights-weights)),
                    np.std(np.abs(true_weights-weights)),
                    0.5*np.median(np.abs(true_weights-weights))
                    ])
