import torch
import torch.nn as nn
import pickle


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
                with open(filename, 'wb') as filehandler: 
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
                with open(filename, 'rb') as filehandler: 
                    data = pickle.load(filehandler)
                    self.mean = data['mean']
                    self.set_normalization(data['mean'], data['std'])
                    self.load_state_dict(data['state_dict'])
                return True
            except:
                print(f'Could not find file {filename}')
                return False
        

class WeightsMLP(nn.Module):
        def __init__(self, input_size:int, hidden_size: int, output_size: int, y_mean: float, y_std: float, max_value: float = 5.):
            super(WeightsMLP, self).__init__()
            self.max_value = max_value
            self.input_size = input_size
            self.hidden_size  = hidden_size
            self.output_size = output_size
            self.y_mean = y_mean
            self.y_std = y_std
            
            self.network = nn.Sequential(*[
                nn.Linear(self.input_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.output_size),
                nn.Tanh()
            ])
    
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x[1] = (x[1]-self.y_mean)/self.y_std
            y = self.network(x)
            return self.max_value*(1+y)/2
        