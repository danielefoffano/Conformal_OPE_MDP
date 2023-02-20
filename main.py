import numpy as np
from random_mdp import MDP_env, MDP_env_discrete_rew, MDP_env_bernoulli_rew
from agent import QlearningAgent
from greedy_policy import Epsilon_Greedy_Policy, Table_Based_Policy
from utils import collect_exp, train_predictor, train_behaviour_policy, compute_weight, compute_weights_gradient, train_weight_function
from networks import MLP, Weights_MLP
from dynamics_model import DynamicsModel, DiscreteRewardDynamicsModel
import torch
from collections import defaultdict
import pickle
import random

random.seed(1)
torch.manual_seed(1)
np.random.seed(1)

REWARD_TYPE = "discrete_multiple"
GRADIENT_BASED = True
EPSILON = 0.3
QUANTILE = 0.1
LR = 0.0001
MOMENTUM = 0.9
EPOCHS = 100
NUM_ACTIONS = 5                                                                     # MDP action space size
NUM_STATES = 30                                                                     # MDP states space size
NUM_REWARDS = 10                                                                    # MDP reward space size (for discrete MDP)
DISCOUNT_FACTOR = 0.99                                                              # behaviour agent discount factor
ALPHA = 0.6                                                                         # behaviour agent alpha
NUM_STEPS = 10000                                                                   # behaviour agent learning steps
N_TRAJECTORIES = 20000                                                              # number of trajectories collected as dataset
HORIZON = 30                                                                        # trajectory horizon

P = np.random.dirichlet(np.ones(NUM_STATES), size=(NUM_STATES, NUM_ACTIONS))        # MDP transition probability functions

if REWARD_TYPE == "bernoulli":
    R = np.random.rand(NUM_STATES, NUM_ACTIONS)                                     # MDP reward function Bernoulli
    env = MDP_env_bernoulli_rew(NUM_STATES, NUM_ACTIONS, P, R)
    model = DiscreteRewardDynamicsModel(NUM_STATES, NUM_ACTIONS, NUM_REWARDS)
elif REWARD_TYPE == "discrete_multiple":                                 
    R = np.random.dirichlet(np.ones(NUM_REWARDS), size=(NUM_STATES, NUM_ACTIONS))   # MDP reward function multiple discrete r values
    env = MDP_env_discrete_rew(NUM_STATES, NUM_ACTIONS, NUM_REWARDS, P, R)
    model = DiscreteRewardDynamicsModel(NUM_STATES, NUM_ACTIONS, NUM_REWARDS)
elif REWARD_TYPE == "continuous":
    # TODO: implement continuous rewards model, the "true" MDP we already have
    R = np.random.rand(NUM_STATES, NUM_ACTIONS, NUM_STATES)
    env = MDP_env(NUM_STATES, NUM_ACTIONS, P, R)

pi_star_probs = np.random.dirichlet(np.ones(NUM_ACTIONS), size=(NUM_STATES))
pi_star = Table_Based_Policy(pi_star_probs)

#Train behaviour policy using Q-learning
agent = QlearningAgent(env.ns, NUM_ACTIONS, DISCOUNT_FACTOR, ALPHA)

q_table = train_behaviour_policy(env, agent, NUM_STEPS)
behaviour_policy = Epsilon_Greedy_Policy(q_table, EPSILON, NUM_ACTIONS)

#Collect experience data using behaviour policy and train model
#TODO: improve storage system
try:
    with open("saved_data_"+REWARD_TYPE+"/dataset_"+ str(HORIZON)+"_MDP.pkl", "rb") as f1:
        dataset = pickle.load(f1)
    with open("saved_data_"+REWARD_TYPE+"/transition_"+ str(HORIZON)+"_model.pkl", "rb") as f2:
        transition_function = pickle.load(f2)
    with open("saved_data_"+REWARD_TYPE+"/reward_"+ str(HORIZON)+"_model.pkl", "rb") as f3:
        reward_function = pickle.load(f3)
    model.transition_function = transition_function
    model.reward_function = reward_function
except:
    dataset = collect_exp(env, N_TRAJECTORIES, HORIZON, behaviour_policy, model, None)
    with open("saved_data_"+REWARD_TYPE+"/dataset_"+ str(HORIZON)+"_MDP.pkl", "wb") as f1:
        pickle.dump(dataset, f1)
    with open("saved_data_"+REWARD_TYPE+"/transition_"+ str(HORIZON)+"_model.pkl", "wb") as f2:
        pickle.dump(model.transition_function, f2)
    with open("saved_data_"+REWARD_TYPE+"/reward_"+ str(HORIZON)+"_model.pkl", "wb") as f3:
        pickle.dump(model.reward_function, f3)

#Split dataset into training (90%) and calibration data (10%)
split_idx = N_TRAJECTORIES // 10
data_tr = dataset[:N_TRAJECTORIES - 100]
data_cal = dataset[N_TRAJECTORIES - 100:N_TRAJECTORIES]
test_point = collect_exp(env, 1, HORIZON, pi_star, model, None)[0]

#Train quantile predictors using training dataset

upper_quantile_net = MLP(1, 32, 1, False)
lower_quantile_net = MLP(1, 32, 1, False)

y_avg, y_std = train_predictor(lower_quantile_net, data_tr, epochs=EPOCHS, quantile=QUANTILE/2, lr=LR, momentum=MOMENTUM)
train_predictor(upper_quantile_net, data_tr, epochs=EPOCHS, quantile=1-(QUANTILE/2), lr=LR, momentum=MOMENTUM)

scores = []
weights = []
traj_idx = 0

if GRADIENT_BASED:
    # Compute training weights for gradient approach
    print("Computing weight of trajectories")
    for traj, cumul_rew in data_tr:
        
        weight = compute_weights_gradient(traj, behaviour_policy, pi_star)       
        weights.append(weight)
        
        traj_idx += 1

    # Compute weights and scores (on calibration data) - gradient approach
    weight_network = Weights_MLP(2, 32, 1)
    train_weight_function(data_tr, weights, weight_network, LR, EPOCHS)

    calibration_weights = []
    for traj, cumul_rew in data_cal:
        # Compute weight
        x = torch.tensor([traj[0].state, traj[0].action], dtype = torch.float32)
        w  = weight_network(x).item()
        calibration_weights.append(w)

        # Compute score
        state = torch.tensor([traj[0].state], dtype = torch.float32)
        score = max((lower_quantile_net(state).item()*y_std)+y_avg - cumul_rew, cumul_rew - ((upper_quantile_net(state).item()*y_std)+y_avg))
        scores.append(score)

    weights = calibration_weights
else:
    # Compute weights and scores - Model-Based approach
    weights = []
    scores = []
    traj_idx = 0

    for traj, cumul_rew in data_cal:
        state = torch.tensor([traj[0].state], dtype = torch.float32)
        # Compute weight generating Monte-Carlo trajectories using learned model
        print("Computing weight of traj {}".format(traj_idx))
        
        weight = compute_weight(traj[0].state, cumul_rew, behaviour_policy, pi_star, model, HORIZON)
        
        # Compute score
        score = max((lower_quantile_net(state).item()*y_std)+y_avg - cumul_rew, cumul_rew - ((upper_quantile_net(state).item()*y_std)+y_avg))

        weights.append(weight)
        scores.append(score)

        traj_idx += 1

weights = np.array(weights)
scores.append(np.inf)
scores = np.array(scores)
scores = torch.tensor(scores)

# Generate y values for test point

x = torch.tensor([test_point[0][0].state], dtype = torch.float32)
lower_val = int((lower_quantile_net(x).item()*y_std)+y_avg)
upper_val = int((upper_quantile_net(x).item()*y_std)+y_avg)

y_vals_test = np.linspace(lower_val, upper_val, upper_val-lower_val).astype(int)

conf_range = []

# Use scores and weights to find the scores quantile to conformalize the predictors
for y in y_vals_test:
    if GRADIENT_BASED:
        x = torch.tensor([test_point[0][0].state, test_point[0][0].action], dtype = torch.float32)
        test_point_weight = weight_network(x).item()
    else:
        test_point_weight = compute_weight(test_point[0][0].state, y, behaviour_policy, pi_star, model, HORIZON)
    
    norm_weights = weights/(weights.sum() + test_point_weight)
    norm_weights = np.concatenate((norm_weights, [test_point_weight/(weights.sum() + test_point_weight)]))
    norm_weights = torch.tensor(norm_weights)

    ordered_indexes = scores.argsort()
    ordered_scores = scores[ordered_indexes]
    ordered_weights = norm_weights[ordered_indexes]

    quantile = 0.90

    cumsum = torch.cumsum(ordered_weights, 0)

    quantile_val = ordered_scores[cumsum>quantile][0]
    #quantile_val = ((torch.cumsum((cumsum>quantile) *1,0) == 1)*1 * ordered_scores).sum(0).item()

    score_test = max(lower_val - y, y - upper_val)

    if score_test <= quantile_val:
        conf_range.append(y)
    
    print("Original interval: {}-{} | Quantile: {} | y: {} | Conformal interval: {}-{} | score_test: {} | true y: {}".format(
        lower_val,
        upper_val,
        quantile_val,
        y,
        lower_val - quantile_val,
        upper_val + quantile_val,
        score_test,
        test_point[1]
    ))
