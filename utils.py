import numpy as np
from agent import Experience
import torch
from agent import Agent, Experience
from random_mdp import MDPEnv
from collections import defaultdict
import pickle
from tqdm import tqdm
from typing import Tuple
import random

MC_SAMPLES = 500

class PinballLoss():
    def __init__(self, quantile=0.10, reduction='mean'):
        self.quantile = quantile
        assert 0 < self.quantile
        assert self.quantile < 1
        self.reduction = reduction

    def __call__(self, output, target):
        assert output.shape == target.shape
        loss = torch.zeros_like(target, dtype=torch.float)
        error = output - target
        smaller_index = error < 0
        bigger_index = 0 < error
        loss[smaller_index] = self.quantile * (abs(error)[smaller_index])
        loss[bigger_index] = (1-self.quantile) * (abs(error)[bigger_index])
        if self.reduction == 'sum':
            loss = loss.sum()
        if self.reduction == 'mean':
            loss = loss.mean()
        return loss
    
class EarlyStopping(object):
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True
                
                
def get_data(env, n_trajectories: int, behaviour_policy, model, reward_type: str, horizon: int):
    print('> Loading/collecting data')
    try:
        with open("./data/saved_data_"+reward_type+"/dataset_"+ str(horizon)+"_MDP.pkl", "rb") as f1:
            dataset = pickle.load(f1)
        model.load_functions(reward_type, horizon)
    except:
        dataset = collect_exp(env, n_trajectories, horizon, behaviour_policy, model, None)
        with open("./data/saved_data_"+reward_type+"/dataset_"+ str(horizon)+"_MDP.pkl", "wb") as f1:
            pickle.dump(dataset, f1)
        model.save_functions(reward_type, horizon)
    return model, dataset

def collect_exp(env, n_trajectories, horizon, policy, model, start_state):

    dataset = []

    for _ in range(n_trajectories):
        if start_state is None:
            s = env.reset()
        else:
            env.set_state(start_state)
            s = start_state
        trajectory = []
        c_reward = 0
        for _ in range(horizon):

            a = policy.get_action(s)
            s_next, r, done = env.step(a)

            c_reward += r
            trajectory.append(Experience(s, a, r, s_next, done))

            if model is not None:
                model.update_visits(s, a, s_next, r)

            s = s_next

        dataset.append((trajectory, c_reward))

    return dataset

def train_predictor(quantile_net, data_tr, epochs, quantile, lr, momentum):

    split_idx = len(data_tr) // 10
    data_val = data_tr[len(data_tr) - split_idx:len(data_tr)]
    data_tr = data_tr[:len(data_tr) - split_idx]
    
    criterion = PinballLoss(quantile)
    optimizer = torch.optim.Adam(quantile_net.parameters(), lr = lr)#SGD(quantile_net.parameters(), lr = lr, momentum = momentum)

    early_stopping = EarlyStopping(5, min_delta = 0.1)

    xy = [[traj[0].state, cumul_r] for traj, cumul_r in data_tr]
    xy_val = [[traj[0].state, cumul_r] for traj, cumul_r in data_val]

    xy = torch.tensor(xy, dtype = torch.float32)
    xy_val = torch.tensor(xy_val, dtype = torch.float32)

    rand_idxs = torch.randperm(xy.size()[0])
    data_batches = torch.utils.data.BatchSampler(xy[rand_idxs], 32, False)
    
    y = xy[:,1]
    y_avg = torch.mean(y).item()
    y_std = torch.std(y).item()

    x_val = xy_val[:,0].unsqueeze(1)
    #y_val = xy_val[:,1].unsqueeze(1)
    y_val = ((xy_val[:,1]-y_avg)/y_std).unsqueeze(1)

    tqdm_epochs = tqdm(range(epochs))
    for epoch in tqdm_epochs:
        random_batches = list(data_batches)
        random.shuffle(random_batches)
        for batch in random_batches:
            batch = torch.stack(batch, 0)
            x_batch = batch[:,0].unsqueeze(1)
            #y_batch = batch[:,1].unsqueeze(1)
            y_batch = ((batch[:,1]-y_avg)/y_std).unsqueeze(1)

            optimizer.zero_grad()

            output = quantile_net(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

        if epoch > 19:
            with torch.no_grad():
                output_val = quantile_net(x_val)
                loss_val = criterion(output_val, y_val)

                early_stopping(loss.item(), loss_val.item())

                desc = "Epoch {} - Training quantile {} - Loss: {} - Loss val: {}".format(epoch, quantile, loss.item(), loss_val.item())
                tqdm_epochs.set_description(desc)
            #if early_stopping.early_stop:
                #print("Early stopping at epoch {}".format(epoch))
                #break
        else:
            desc = "Epoch {} - Training quantile {} - Loss: {}".format(epoch, quantile, loss.item())
            tqdm_epochs.set_description(desc)
        
    return y_avg, y_std
            
def train_weight_function(training_dataset, weights_labels, weight_network, lr, epochs, pi_b, pi_target):
    

    split_idx = len(training_dataset) // 10
    data_val = training_dataset[len(training_dataset) - split_idx:len(training_dataset)]
    training_dataset = training_dataset[:len(training_dataset) - split_idx]

    xy = [[trajectory[0].state, cumul_rew, weights_labels[idx]] for idx, (trajectory, cumul_rew) in enumerate(training_dataset)]
    xy = torch.tensor(xy, dtype = torch.float32)

    xy_val = [[trajectory[0].state, cumul_rew, weights_labels[idx]] for idx, (trajectory, cumul_rew) in enumerate(data_val)]
    xy_val = torch.tensor(xy_val, dtype = torch.float32)

    x_val = xy_val[:,:-1]
    y_val = xy_val[:,-1].unsqueeze(1)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(weight_network.parameters(), lr = lr)
    rand_idxs = torch.randperm(xy.size()[0])
    data_batches = torch.utils.data.BatchSampler(xy[rand_idxs], 64, False)

    tqdm_epochs = tqdm(range(epochs))
    for epoch in tqdm_epochs:
        random_batches = list(data_batches)
        random.shuffle(random_batches)
        for batch in random_batches:
            batch = torch.stack(batch, 0)
            x_batch = batch[:,:-1]
            y_batch = batch[:,-1].unsqueeze(1)

            optimizer.zero_grad()

            output = weight_network(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            weight_network
        if epoch > 19:
            with torch.no_grad():
                output_val = weight_network(x_val)
                loss_val = criterion(output_val, y_val)

                desc = "Epoch {} - Training weights network - Loss: {} - Loss val: {}".format(epoch, loss.item(), loss_val.item())
                tqdm_epochs.set_description(desc)
        else:
            desc = "Epoch {} - Training weights network - Loss: {}".format(epoch, loss.item())
            tqdm_epochs.set_description(desc)

def train_behaviour_policy(env: MDPEnv, agent: Agent, MAX_STEPS):
    
    #model = EmpiricalModel(env.ns, env.na)
    episode_rewards = []
    episode_steps = []

    episode = 0
    state = env.reset()
    steps = 0
    while steps < MAX_STEPS:
        rewards = 0

        action = agent.forward(state, steps)
        next_state, reward, done = env.step(action)
        agent.backward(Experience(state, action, reward, next_state, done))

        state = next_state

        steps += 1
        rewards += reward
        if done:
            break
        
        
        episode_rewards.append((episode, rewards))
        episode_steps.append((episode, steps))
        episode += 1
        

    return agent.q_function

def dataset_to_dict(dataset, my_dict):

    # Takes dataset in the form of list of pairs (traj, cumul_rew

    for traj, cumul_rew in dataset:

        my_dict[traj[0]].append((traj[1:], cumul_rew))

    return my_dict


def compute_weight(s0, y, pi_b, pi_star, model, horizon):
    
    tot_sum_pi_star = 0
    for m in range(MC_SAMPLES):
        model.cur_state = s0
        s = s0
        prev_s = None
        r_sum = 0
        for i in range(horizon):
            a = pi_star.get_action(s)

            next_s, r, _ = model.step(a)

            prev_s = s
            s = next_s
            r_sum += r
        #if r == y-(r_sum-r):
        #    tot_sum_pi_star += model.reward_function[s][a][y-(r_sum-r)] #pi_star.get_action_prob(prev_s, a)
        if y-(r_sum - r) in range(model.num_rewards):
            p_r_diff = model.reward_function[s][a][y-(r_sum-r)]
            tot_sum_pi_star += p_r_diff
    
    tot_sum_pi_b = 1e-5 #to avoid division by 0

    n = 0
    while True:
        model.cur_state = s0
        s = s0
        prev_s = None
        r_sum = 0
        for i in range(horizon):
            a = pi_b.get_action(s)

            next_s, r, _ = model.step(a)

            prev_s = s
            s = next_s
            r_sum += r
        #if r == y-(r_sum-r):
        #    tot_sum_pi_b += model.reward_function[s][a][y-(r_sum-r)]
        if y-(r_sum - r) in range(model.num_rewards):
            p_r_diff = model.reward_function[s][a][y-(r_sum-r)]
            tot_sum_pi_b += p_r_diff

        if n >= MC_SAMPLES and (tot_sum_pi_star / (m+1)) / (tot_sum_pi_b / (n+1)) < 2 or n >= 3 * MC_SAMPLES:
            break
        
        n += 1

    return min((tot_sum_pi_star / (m+1)) / (tot_sum_pi_b / (n+1)), 5)


def compute_weights_gradient(traj, pi_b, pi_star):

    prod_pi_b = 1
    prod_pi_star = 1
    
    for step in traj:

        prod_pi_b *= pi_b.get_action_prob(step.state, step.action)
        prod_pi_star *= pi_star.get_action_prob(step.state, step.action)

    return prod_pi_star/prod_pi_b

def value_iteration(env, gamma: float, eps: float = 1e-9) -> Tuple[np.ndarray, np.ndarray]:
    """
    Value iteration
    
    Parameters
    ------------------
        env: Inventory
            Environment
        gamma: float
            Discount factor
        eps: float, optional
            Value iteration tolerance
    Returns
    ------------------
        V: List[float]
            Value function
        pi: List[int]
            Policy
    """
    P, R = env.P, env.R
    dim_state, dim_action = P.shape[0], P.shape[1]
    V = np.zeros((dim_state))
    pi = np.zeros((dim_state, dim_action))
    while True:
        prevV = np.copy(V)
        V = np.sum(P * (R + gamma * V), axis=-1).max(axis=-1)
        if np.abs(prevV - V).max() < eps:
            break
    x = np.sum(P * (R + gamma * V), axis=-1).argmax(axis=-1)
    pi[np.arange(x.size), x] = 1
    return V, pi
