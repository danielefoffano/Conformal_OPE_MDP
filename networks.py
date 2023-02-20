import torch

class MLP(torch.nn.Module):
        def __init__(self, input_size, hidden_size, output_size, softmax):
            super(MLP, self).__init__()
            self.input_size = input_size
            self.hidden_size  = hidden_size
            self.output_size = output_size
            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(self.hidden_size, self.output_size)
            self.softmax = softmax
            self.softmax_activation = torch.nn.Softmax(-1)
        def forward(self, x):
            hidden = self.fc1(x)
            relu = self.relu(hidden)
            output = self.fc2(relu)
            if self.softmax:
                output = self.softmax_activation(output)
            return output

class Weights_MLP(torch.nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(Weights_MLP, self).__init__()
            self.max_value = 5
            self.input_size = input_size
            self.hidden_size  = hidden_size
            self.output_size = output_size
            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(self.hidden_size, self.output_size)
            self.tanh = torch.nn.Tanh()
        def forward(self, x):
            hidden = self.fc1(x)
            relu = self.relu(hidden)
            output = self.fc2(relu)
            output = self.max_value*(1+self.tanh(output))/2
            return output
        