import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


import numpy as np
from custom_environments.random_mdp import MDPEnv, MDPEnvDiscreteRew, MDPEnvBernoulliRew
from agent import QlearningAgent
from policy import EpsilonGreedyPolicy, TableBasedPolicy, MixedPolicy
from utilities.utils import get_data, collect_exp, train_predictor, train_behaviour_policy, value_iteration, save_important_dictionary, compute_avg_std_dataset
from utilities.networks import MLP, WeightsMLP, WeightsTransformerMLP
from dynamics_model import DiscreteRewardDynamicsModel, ContinuousRewardDynamicsModel
import torch
import random
from weights import WeightsEstimator, GradientWeightsEstimator, EmpiricalWeightsEstimator, ExactWeightsEstimator
from conformal_set import ConformalSet
from custom_environments.inventory import Inventory
from utilities.logger import Logger
import os
import argparse
from utilities.types_cp import Interval, LoggerResults, Point
from enum import Enum
from is_method import ISMethod

class WeightsEstimationMethod(Enum):
    gradient = 'gradient'
    empirical = 'empirical'
    
class ConfidenceMethod(Enum):
    conformal = 'conformal'
    IS = 'IS'

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Specify horizon')
    parser.add_argument('--horizon', type=int, help='Horizon length', required=True)
    parser.add_argument('--weights_estimation_method', type=str, default='empirical', help='Possible values: {"empirical", "gradient"}')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--confidence_method', type=str, default='conformal', help='Possible values: {"conformal", "IS"}')
    parser.add_argument('-r', '--runs', type=int, help='Runs to execute: use it like -r 1 2 3 to execute run 1 2 and 3', default= None,  nargs='+')

    
    args = parser.parse_args()
    SEED = int(args.horizon*2000) if args.seed is None else args.seed
    print(f'Horizon chosen {args.horizon} - Seed: {SEED} - Method: {args.weights_estimation_method} - Runs: {args.runs}')


    set_seed(SEED)
    assert args.weights_estimation_method in ['empirical', 'gradient'], f'Method {args.weights_estimation_method} not valid'
    assert args.confidence_method in ['conformal', 'IS'], f'Method {args.confidence_method} not valid'

    
    RUNS_NUMBER = 30
    RUNS_RANGE = args.runs
    N_CPU = os.cpu_count()
    ENV_NAME = "inventory"
    REWARD_TYPE = "discrete_multiple"
    WEIGHT_METHOD = args.weights_estimation_method
    CONFIDENCE_METHOD = args.confidence_method
    TRANSFORMER = False
    EPSILON = 0.4
    QUANTILE = 0.1
    LR = 6e-5
    LR_QUANTILES = 1e-4
    MOMENTUM = 0.9
    EPOCHS_QUANTILES = 300
    EPOCHS_WEIGHTS = 300
    NUM_ACTIONS = 10                                                                    # MDP action space size
    NUM_STATES = 10                                                                     # MDP states space size
    NUM_REWARDS = 10                                                                    # MDP reward space size (for discrete MDP)
    DISCOUNT_FACTOR = 0.99                                                              # behaviour agent discount factor
    DISCOUNT_REWARDS = 1
    ALPHA = 0.6                                                                         # behaviour agent alpha
    NUM_STEPS = 20000                                                                   # behaviour agent learning steps
    N_TRAJECTORIES = 40000                                                              # number of trajectories collected as dataset
    HORIZONS = [int(args.horizon)]                                                           # trajectory horizon
    NUM_TEST_POINTS = 1000
    NUM_POINTS_WEIGHT_ESTIMATOR = 3000
    NUM_NEURONS_QUANTILE_NETWORKS = 64
    NUM_NEURONS_WEIGHT_ESTIMATOR = 128
    epsilons = np.linspace(0,1,21)
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
            model = ContinuousRewardDynamicsModel(NUM_STATES, NUM_ACTIONS)
        
        pi_star_probs = np.random.dirichlet(np.ones(NUM_ACTIONS), size=(NUM_STATES))
        pi_star_pre = TableBasedPolicy(pi_star_probs)
        
        #Train behaviour policy using Q-learning
        agent = QlearningAgent(NUM_STATES, NUM_ACTIONS, DISCOUNT_FACTOR, ALPHA)

        q_table = train_behaviour_policy(env, agent, NUM_STEPS)
        behaviour_policy = EpsilonGreedyPolicy(q_table, EPSILON, NUM_ACTIONS)

    elif ENV_NAME == "inventory":
        env = Inventory(inventory_size = NUM_STATES, fixed_cost = 1, order_rate = 10)
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
        columns = LoggerResults._fields
        
        path = f"results/{ENV_NAME}/{WEIGHT_METHOD}/horizon_{HORIZON}/"

        print('> Getting data for exact weights estimator')
        exact_weights_estimator = ExactWeightsEstimator(behaviour_policy, HORIZON, env, NUM_STATES, NUM_POINTS_WEIGHT_ESTIMATOR, DISCOUNT_REWARDS)

        runs_values = range(RUNS_NUMBER) if args.runs is None else args.runs
        for RUN_NUMBER in runs_values:
            file_logger = Logger(path+f"run_{RUN_NUMBER}_{args.confidence_method}.csv", columns)
            #Collect experience data using behaviour policy and train model
            print('> Getting training/calibration data')
            model, dataset = get_data(RUN_NUMBER, env, N_TRAJECTORIES, behaviour_policy, model, HORIZON, path, DISCOUNT_REWARDS)

            print('> Splitting dataset')
            #Split dataset into training (90%) and calibration data (10%)
            calibration_trajectories = N_TRAJECTORIES // 10
            data_tr = dataset[:N_TRAJECTORIES - calibration_trajectories]
            data_cal = dataset[N_TRAJECTORIES - calibration_trajectories:N_TRAJECTORIES]

            if CONFIDENCE_METHOD == ConfidenceMethod.conformal.value:
                #Train quantile predictors using training dataset
                print('> Training/loading quantile networks')
                upper_quantile_net = MLP(1, NUM_NEURONS_QUANTILE_NETWORKS, 1, False)
                lower_quantile_net = MLP(1, NUM_NEURONS_QUANTILE_NETWORKS, 1, False)

                lower_str_path = path + f'data/networks/lower_quantile_net_{RUN_NUMBER}.pth'
                if not lower_quantile_net.load(lower_str_path):
                    y_avg, y_std = train_predictor(lower_quantile_net, data_tr, epochs=EPOCHS_QUANTILES, quantile=QUANTILE/2, lr=LR_QUANTILES, momentum=MOMENTUM)
                    lower_quantile_net.set_normalization(y_avg, y_std)
                    os.makedirs(os.path.dirname(lower_str_path), exist_ok=True)
                    lower_quantile_net.save(lower_str_path)

                upper_str_path = path + f'data/networks/upper_quantile_net_{RUN_NUMBER}.pth'
                if not upper_quantile_net.load(upper_str_path):
                    y_avg, y_std = train_predictor(upper_quantile_net, data_tr, epochs=EPOCHS_QUANTILES, quantile=1-(QUANTILE/2), lr=LR_QUANTILES, momentum=MOMENTUM)
                    upper_quantile_net.set_normalization(y_avg, y_std)
                    os.makedirs(os.path.dirname(upper_str_path), exist_ok=True)
                    upper_quantile_net.save(upper_str_path)
            else:
                lower_quantile_net = None
                upper_quantile_net = None
                y_avg, y_std = compute_avg_std_dataset(data_tr)


            
            epsilon_lengths = []
            for epsilon_value in epsilons:
                print(f'---------- Evaluating for epsilon = {epsilon_value} ----------')
                pi_target = MixedPolicy(pi_uniform, greedy_policy, epsilon_value)                

                print(f'> Collecting test points')
                test_points = collect_exp(env, NUM_TEST_POINTS, HORIZON, pi_target, None, test_state, discount=DISCOUNT_REWARDS)

                print(f'> Estimating exact weights')
                exact_weights_estimator.init_pi_target(pi_target)
                true_weights = exact_weights_estimator.compute_true_ratio_dataset(data_cal)
               

                print(f'> Estimate weights for calibration data')
                
                weights_estimator: WeightsEstimator = None
                if WEIGHT_METHOD == 'gradient':
                    if TRANSFORMER:
                        make_net = lambda:WeightsTransformerMLP(2 + 2*NUM_STATES*NUM_ACTIONS, NUM_NEURONS_WEIGHT_ESTIMATOR, 1, y_avg, y_std, behaviour_policy, pi_target)
                    else:
                        make_net = lambda:WeightsMLP(2, NUM_NEURONS_WEIGHT_ESTIMATOR, 1, y_avg, y_std)
                    weights_estimator = GradientWeightsEstimator(behaviour_policy, pi_target, lower_quantile_net, upper_quantile_net, make_net)
                    weights_estimator.train(data_tr, LR, EPOCHS_WEIGHTS)
                elif WEIGHT_METHOD == 'empirical':
                    # @TODO Fix 500 
                    weights_estimator = EmpiricalWeightsEstimator(behaviour_policy, pi_target, lower_quantile_net, upper_quantile_net, NUM_STATES, 4*500)
                    weights_estimator.train(data_tr)
                else:
                    raise Exception('To be updated')
                
                
             
                # Generate y values for test point
                print(f'> Computing conformal set')
                if CONFIDENCE_METHOD == ConfidenceMethod.conformal.value:   
                    scores, weights = weights_estimator.evaluate_calibration_data(data_cal)
                    conformal_set = ConformalSet(lower_quantile_net, upper_quantile_net, behaviour_policy, pi_target, model, HORIZON, DISCOUNT_REWARDS)
                
                    
                    if HORIZON in [15, 25, 40] and epsilon_value > 0.3 and epsilon_value < 0.5:
                        save_important_dictionary(env, weights_estimator, exact_weights_estimator, conformal_set, weights, scores, path, RUN_NUMBER, epsilon_value)
                    intervals = conformal_set.build_set(test_points, weights.copy(), scores.copy(), weights_estimator, N_CPU)
                elif CONFIDENCE_METHOD == ConfidenceMethod.IS.value:
                    weights = list(map(lambda x: weights_estimator.evaluate_point(Point(x.initial_state, x.cumulative_reward)), data_cal))
                    weights = np.array(weights)
                    is_method = ISMethod(pi_target, HORIZON, DISCOUNT_REWARDS, data_cal, weights_estimator)
                    intervals = is_method.build_set(test_points, N_CPU)
                else:
                    raise Exception('Wrong confidence method')
                    

                results_intervals = Interval.analyse_intervals(intervals)
                log_w_ratio = np.log(weights/(1e-6+true_weights))
                
                print('-------- Original method --------')
                print("Eps: {:.2f} | Coverage: {:.2f}% | Average interval length: {:.2f} "\
                      "| Avg Original interval: {:.2f}-{:.2f} | Avg New interval: {:.2f}-{:.2f} | Quantile: {:.3f} |"\
                       "Avg weights: {:.3f}| log w_hat/w: {:.3f}| avg_delta_w: {:.3f}"
                      .format(epsilon_value, results_intervals.coverage, results_intervals.avg_length, 
                              results_intervals.lower_vals_behavior.mean(), results_intervals.upper_val_behavior.mean(),
                              results_intervals.lower_vals_target.mean(), results_intervals.upper_vals_target.mean(),
                              results_intervals.avg_quantiles.mean(),
                              np.mean(weights), log_w_ratio.mean(), 0.5*np.mean(np.abs(true_weights-weights))))
                print('-------- Double quantile method --------')
                print("Eps: {:.2f} | Coverage: {:.2f}% | Average interval length: {:.2f} "\
                      "| Avg Original interval: {:.2f}-{:.2f} | Avg New interval: {:.2f}-{:.2f} | Quantile: {:.3f} - {:.3f} |"\
                       "Avg weights: {:.3f}| log w_hat/w: {:.3f}| avg_delta_w: {:.3f}"
                      .format(epsilon_value, results_intervals.coverage_double, results_intervals.avg_length_double, 
                              results_intervals.lower_vals_behavior.mean(), results_intervals.upper_val_behavior.mean(),
                              results_intervals.lower_vals_target_double.mean(), results_intervals.upper_vals_target_double.mean(),
                              results_intervals.avg_quantiles_double_low.mean(), results_intervals.avg_quantiles_double_high.mean(),
                              np.mean(weights), log_w_ratio.mean(), 0.5*np.mean(np.abs(true_weights-weights))))
                
                print('-------- Cumul  method --------')
                print("Eps: {:.2f} | Coverage: {:.2f}% | Average interval length: {:.2f} "\
                      "| Avg Original interval: {:.2f}-{:.2f} | Avg New interval: {:.2f}-{:.2f} | Quantile: {:.3f} - {:.3f} |"\
                       "Avg weights: {:.3f}| log w_hat/w: {:.3f}| avg_delta_w: {:.3f}"
                      .format(epsilon_value, results_intervals.coverage_cumul, results_intervals.avg_length_cumul, 
                              results_intervals.lower_vals_behavior.mean(), results_intervals.upper_val_behavior.mean(),
                              results_intervals.lower_vals_target_cumul.mean(), results_intervals.upper_vals_target_cumul.mean(),
                              results_intervals.avg_quantiles_cumul_low.mean(), results_intervals.avg_quantiles_cumul_high.mean(),
                              np.mean(weights), log_w_ratio.mean(), 0.5*np.mean(np.abs(true_weights-weights))))
       

                logger_results = LoggerResults(
                    epsilon = epsilon_value,
                    coverage = results_intervals.coverage,
                    coverage_double=results_intervals.coverage_double,
                    coverage_cumul=results_intervals.coverage_cumul,
                    avg_length = results_intervals.avg_length,
                    std_length = results_intervals.std_length,
                    avg_length_double= results_intervals.avg_length_double,
                    std_length_double=results_intervals.std_length_double,
                    avg_length_cumul= results_intervals.avg_length_cumul,
                    std_length_cumul=results_intervals.std_length_cumul,
                    
                    avg_interval_target_lower = results_intervals.lower_vals_target.mean(),
                    avg_interval_target_upper = results_intervals.upper_vals_target.mean(),
                    std_interval_target_lower = results_intervals.lower_vals_target.std(ddof=1),
                    std_interval_target_upper = results_intervals.upper_vals_target.std(ddof=1),
                    
                    avg_interval_target_double_lower = results_intervals.lower_vals_target_double.mean(),
                    avg_interval_target_double_upper = results_intervals.upper_vals_target_double.mean(),
                    std_interval_target_double_lower = results_intervals.lower_vals_target_double.std(ddof=1),
                    std_interval_target_double_upper = results_intervals.upper_vals_target_double.std(ddof=1),
                    
                    
                    avg_interval_target_cumul_lower = results_intervals.lower_vals_target_cumul.mean(),
                    avg_interval_target_cumul_upper = results_intervals.upper_vals_target_cumul.mean(),
                    std_interval_target_cumul_lower = results_intervals.lower_vals_target_cumul.std(ddof=1),
                    std_interval_target_cumul_upper = results_intervals.upper_vals_target_cumul.std(ddof=1),
                    
                    
                    avg_interval_behavior_lower = results_intervals.lower_vals_behavior.mean(),
                    avg_interval_behavior_upper = results_intervals.upper_val_behavior.mean(),
                    std_interval_behavior_lower = results_intervals.lower_vals_behavior.std(ddof=1),
                    std_interval_behavior_upper = results_intervals.upper_val_behavior.std(ddof=1),
                    avg_quantile = results_intervals.avg_quantiles.mean(),
                    avg_std_quantile = results_intervals.std_quantiles.mean(),
                    std_avg_quantile = results_intervals.avg_quantiles.std(ddof = 1),
                    
                    avg_double_quantile_low = results_intervals.avg_quantiles_double_low.mean(),
                    avg_std_double_quantile_low = results_intervals.std_quantiles_double_low.mean(),
                    std_avg_double_quantile_low = results_intervals.avg_quantiles_double_low.std(ddof = 1),
                    
                    avg_double_quantile_high = results_intervals.avg_quantiles_double_high.mean(),
                    avg_std_double_quantile_high = results_intervals.std_quantiles_double_high.mean(),
                    std_avg_double_quantile_high = results_intervals.avg_quantiles_double_high.std(ddof = 1),
                    
                    
                    avg_cumul_quantile_low = results_intervals.avg_quantiles_cumul_low.mean(),
                    avg_std_cumul_quantile_low = results_intervals.std_quantiles_cumul_low.mean(),
                    std_avg_cumul_quantile_low = results_intervals.avg_quantiles_cumul_low.std(ddof = 1),
                    
                    avg_cumul_quantile_high = results_intervals.avg_quantiles_cumul_high.mean(),
                    avg_std_cumul_quantile_high = results_intervals.std_quantiles_cumul_high.mean(),
                    std_avg_cumul_quantile_high = results_intervals.avg_quantiles_cumul_high.std(ddof = 1),
                    
                    horizon = HORIZON,
                    epsilon_pi_behavior = EPSILON,
                    avg_weights = np.mean(weights),
                    std_weights = np.std(weights, ddof=1),
                    avg_log_ratio_what_w = log_w_ratio.mean(),
                    std_log_ratio_what_w = log_w_ratio.std(ddof=1),
                    median_log_ratio_what_w = np.median(log_w_ratio),
                    avg_delta_w = 0.5*np.mean(np.abs(true_weights-weights)),
                    std_delta_w =  np.std(np.abs(true_weights-weights), ddof=1),
                    median_delta_w= 0.5*np.median(np.abs(true_weights-weights))
                    
                )
                

                file_logger.write(logger_results._asdict().values())
