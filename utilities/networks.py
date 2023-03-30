import torch
import torch.nn as nn
import pickle
import lzma

class MLP(nn.Module):
        def __init__(self, input_size: int, hidden_size: int, output_size: int, softmax: bool):
            super(MLP, self).__init__()
            self.input_size = input_size
            self.hidden_size  = hidden_size
            self.output_size = output_size
            self.softmax = softmax
            self.network = [
                nn.Linear(self.input_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.output_size)
            ]
            if self.softmax:
                self.network.append(nn.Softmax(-1))
                
            self.network = nn.Sequential(*self.network)
            
            self.mean = 0.
            self.std = 1.
            
        def set_normalization(self, mean: float = 0, std: float = 1):
            self.mean = mean
            self.std = std

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return (self.network(x) * self.std) + self.mean
        
        def save(self, filename: str) -> bool:
            try:
                with lzma.open(filename, 'wb') as filehandler: 
                    pickle.dump({
                        'state_dict': self.state_dict(),
                        'mean': self.mean,
                        'std': self.std}, filehandler, protocol=pickle.HIGHEST_PROTOCOL)
                return True
            except:
                print(f'Could not create file {filename}')
                return False
                
        def load(self, filename: str) -> bool:
            try:
                with lzma.open(filename, 'rb') as filehandler: 
                    data = pickle.load(filehandler)
                    self.mean = data['mean']
                    self.set_normalization(data['mean'], data['std'])
                    self.load_state_dict(data['state_dict'])
                return True
            except:
                print(f'Could not find file {filename}')
                return False
        

class WeightsTransformerMLP(nn.Module):
        def __init__(self, input_size:int, hidden_size: int, output_size: int, y_mean: float, y_std: float, pi_b, pi_target, max_value: float = 5.):
            super(WeightsTransformerMLP, self).__init__()
            self.max_value = max_value
            self.input_size = input_size
            self.hidden_size  = hidden_size
            self.output_size = output_size
            self.y_mean = y_mean
            self.y_std = y_std

            flatten_pi_b = torch.tensor(pi_b.probabilities, dtype = torch.float32).flatten()
            flatten_pi_target = torch.tensor(pi_target.probabilities, dtype = torch.float32).flatten()
            self.stacked_pi = torch.hstack([flatten_pi_b, flatten_pi_target])

            self.network = nn.Sequential(*[
                nn.TransformerEncoderLayer(self.input_size, nhead = 4, dim_feedforward = self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.input_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.output_size)
            ])
    
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            
            prob_concat = self.stacked_pi.repeat((len(x), 1))

            x = torch.hstack([x, prob_concat])

            x[...,1] = (x[...,1]-self.y_mean)/self.y_std
            y = self.network(x)
            return torch.clip(y, min = -4, max = 4).exp()
        
class WeightsMLP(nn.Module):
        def __init__(self, input_size:int, hidden_size: int, output_size: int, y_mean: float, y_std: float, max_value: float = 5., state_norm_val: float =10):
            super(WeightsMLP, self).__init__()
            self.max_value = max_value
            self.input_size = input_size
            self.hidden_size  = hidden_size
            self.output_size = output_size
            self.y_mean = y_mean
            self.y_std = y_std
            self.state_norm_val = state_norm_val
            self.network = nn.Sequential(*[
               nn.Linear(self.input_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.output_size)
            ])
    
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x[:,0] = x[:, 0] / self.state_norm_val
            x[:,1] = (x[:, 1] - self.y_mean)/self.y_std
            y = self.network(x)
            return y.exp()
        