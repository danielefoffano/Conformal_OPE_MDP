import numpy as np
from random_mdp import MDPEnv, MDPEnvDiscreteRew, MDPEnvBernoulliRew
from agent import QlearningAgent
from greedy_policy import EpsilonGreedyPolicy, TableBasedPolicy, MixedPolicy
from utils import get_data, collect_exp, train_predictor, train_behaviour_policy, value_iteration
from networks import MLP, WeightsMLP, WeightsTransformerMLP
from dynamics_model import DynamicsModel, DiscreteRewardDynamicsModel, ContinuousRewardDynamicsModel
import torch
from collections import defaultdict
import pickle
import random
from weights import WeightsEstimator
from conformal_set import ConformalSet
from custom_environments.inventory import Inventory

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
set_seed(1)

ENV_NAME = "inventory"
REWARD_TYPE = "discrete_multiple"
GRADIENT_BASED = True
TRANSFORMER = False
EPSILON = 0.2
QUANTILE = 0.1
LR = 1e-4
MOMENTUM = 0.9
EPOCHS = 300
NUM_ACTIONS = 10                                                                     # MDP action space size
NUM_STATES = 10                                                                     # MDP states space size
NUM_REWARDS = 10                                                                    # MDP reward space size (for discrete MDP)
DISCOUNT_FACTOR = 0.99                                                              # behaviour agent discount factor
ALPHA = 0.6                                                                         # behaviour agent alpha
NUM_STEPS = 20000                                                                   # behaviour agent learning steps
N_TRAJECTORIES = 20000                                                              # number of trajectories collected as dataset
HORIZON = 10                                                                        # trajectory horizon

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
    env = Inventory(inventory_size = NUM_STATES, fixed_cost = 1, order_rate = 10, seed = 1)
    NUM_STATES += 1
    NUM_ACTIONS += 1
    model = ContinuousRewardDynamicsModel(NUM_STATES, NUM_ACTIONS)

    # Compute optimal policy with VI
    VI_v, VI_pi = value_iteration(env, 0.95)
    behaviour_policy = EpsilonGreedyPolicy(VI_pi, EPSILON, NUM_ACTIONS)
    #pi_star_pre = EpsilonGreedyPolicy(VI_pi, EPSILON, NUM_ACTIONS)

    greedy_policy = EpsilonGreedyPolicy(VI_pi, 0, NUM_ACTIONS)

    # Uniform policy
    pi_star_probs = np.ones(shape=(NUM_STATES,NUM_ACTIONS))/NUM_ACTIONS
    pi_uniform = TableBasedPolicy(pi_star_probs)
    #behaviour_policy = TableBasedPolicy(pi_star_probs)

#Collect experience data using behaviour policy and train model
#TODO: improve storage system
model, dataset = get_data(env, N_TRAJECTORIES, behaviour_policy, model, REWARD_TYPE, HORIZON)

#Split dataset into training (90%) and calibration data (10%)
calibration_trajectories = N_TRAJECTORIES // 10
data_tr = dataset[:N_TRAJECTORIES - calibration_trajectories]
data_cal = dataset[N_TRAJECTORIES - calibration_trajectories:N_TRAJECTORIES]
test_state = np.random.randint(env.ns)

#Train quantile predictors using training dataset
print('> Training/loading quantile networks')
upper_quantile_net = MLP(1, 32, 1, False)
lower_quantile_net = MLP(1, 32, 1, False)

if not lower_quantile_net.load('./data/networks/lower_quantile_net.pth'):
    y_avg, y_std = train_predictor(lower_quantile_net, data_tr, epochs=EPOCHS, quantile=QUANTILE/2, lr=LR, momentum=MOMENTUM)
    lower_quantile_net.set_normalization(y_avg, y_std)
    lower_quantile_net.save('./data/networks/lower_quantile_net.pth')

if not upper_quantile_net.load('./data/networks/upper_quantile_net.pth'):
    y_avg, y_std = train_predictor(upper_quantile_net, data_tr, epochs=EPOCHS, quantile=1-(QUANTILE/2), lr=LR, momentum=MOMENTUM)
    upper_quantile_net.set_normalization(y_avg, y_std)
    upper_quantile_net.save('./data/networks/upper_quantile_net.pth')


epsilons = np.linspace(0, 1, 10)
epsilon_lengths = []
for epsilon_value in epsilons:

    pi_target = MixedPolicy(pi_uniform, greedy_policy, epsilon_value)

    test_points = collect_exp(env, 100, HORIZON, pi_target, None, test_state)

    print(f'> Estimate weights for calibration data')
    weights_estimator = WeightsEstimator(behaviour_policy, pi_target, lower_quantile_net, upper_quantile_net)
    if GRADIENT_BASED:
        if TRANSFORMER: 
            scores, weights, weight_network = weights_estimator.gradient_method(data_tr, data_cal, LR, EPOCHS, lambda:WeightsTransformerMLP(2 + 2*NUM_STATES*NUM_ACTIONS, 64, 1, upper_quantile_net.mean, upper_quantile_net.std, behaviour_policy, pi_target))
        else: 
            scores, weights, weight_network = weights_estimator.gradient_method(data_tr, data_cal, LR, EPOCHS, lambda:WeightsMLP(2, 64, 1, upper_quantile_net.mean, upper_quantile_net.std))
    else:
        scores, weight = weights_estimator.model_based(data_tr, data_cal, HORIZON, model)

    # Generate y values for test point
    print(f'> Computing conformal set')
    conformal_set = ConformalSet(lower_quantile_net, upper_quantile_net, behaviour_policy, pi_target, model, HORIZON)
    y_set, intervals = conformal_set.build_set(test_points, weights, scores, weight_network, GRADIENT_BASED)

    included = 0
    lengths = []
    for interval in intervals:
        if interval[-1] >= interval[2] and interval[-1] <= interval[3]:
            included += 1
        lengths.append(interval[3]-interval[2])

    included = included/len(intervals)
    mean_length = np.mean(lengths)
    epsilon_lengths.append(mean_length)
    print("Epsilon: {} | Coverage: {:.2f}% | Average interval length: {}".format(epsilon_value, included*100, mean_length))