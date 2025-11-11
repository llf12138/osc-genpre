import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import csv
import copy

csv_file = open('Best Model.csv', mode='w', newline='')
fieldnames = ['train_correlation', 'train_r_squared','test_correlation', 'test_r_squared', 'lr', 'batch_size',
              'hidden_size', 'num_layers', 'params']
writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
writer.writeheader()

train_data = pd.read_csv('Train.csv')
train_inputs = torch.tensor(train_data.iloc[:, 4:].values, dtype=torch.float32)
train_targets = torch.tensor(train_data.iloc[:, 3].values, dtype=torch.float32)

test_data = pd.read_csv('Test.csv')
test_inputs = torch.tensor(test_data.iloc[:, 4:].values, dtype=torch.float32)
test_targets = torch.tensor(test_data.iloc[:, 3].values, dtype=torch.float32)

class GluLayer(nn.Module):
    def __init__(self, input_size):
        super(GluLayer, self).__init__()
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, input_size)

    def forward(self, x):
        return self.linear1(x) * torch.sigmoid(self.linear2(x))

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, l2_factor):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.glu = GluLayer(hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(hidden_size, output_size)
        self.fc.l2_factor = l2_factor

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.glu(hn[-1])
        out = self.dropout(out)
        out = self.fc(out)
        return out

model_list = []
parameters = [
{'input_size': 30, 'output_size': 1, 'hidden_size': 50, 'num_layers': 1},
{'input_size': 30, 'output_size': 1, 'hidden_size': 100, 'num_layers': 1},    
{'input_size': 30, 'output_size': 1, 'hidden_size': 150, 'num_layers': 1},
{'input_size': 30, 'output_size': 1, 'hidden_size': 200, 'num_layers': 1},
{'input_size': 30, 'output_size': 1, 'hidden_size': 250, 'num_layers': 1},
{'input_size': 30, 'output_size': 1, 'hidden_size': 50, 'num_layers': 2},
{'input_size': 30, 'output_size': 1, 'hidden_size': 100, 'num_layers': 2},    
{'input_size': 30, 'output_size': 1, 'hidden_size': 150, 'num_layers': 2},
{'input_size': 30, 'output_size': 1, 'hidden_size': 200, 'num_layers': 2},
{'input_size': 30, 'output_size': 1, 'hidden_size': 250, 'num_layers': 2},
{'input_size': 30, 'output_size': 1, 'hidden_size': 50, 'num_layers': 3},
]

best_model = None
best_r_squared = float('-inf')
best_batch_size = None
best_lr = None
best_params = None

l2_factor = 0.01

for params in parameters:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel(params['input_size'], params['hidden_size'], params['output_size'], params['num_layers'], l2_factor)
    model.to(device)

    batch_sizes = [100,120,150,180,200,220,250,280,300,320,350,64,128,256]
    learning_rates = [0.01,0.001]
    num_epochs = 10000

    for batch_size in batch_sizes:
        for lr in learning_rates:
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_factor)

            patience = 40
            best_loss = float('inf')
            no_improvement = 0
            train_best_model_weights = copy.deepcopy(model.state_dict())

            for epoch in range(num_epochs):
                model.train()
                permutation = torch.randperm(train_inputs.size()[0])
                for i in range(0, train_inputs.size()[0], batch_size):
                    indices = permutation[i:i + batch_size]
                    batch_inputs = train_inputs[indices].to(device)
                    batch_targets = train_targets[indices].to(device)

                    outputs = model(batch_inputs.unsqueeze(1))
                    loss = criterion(outputs.squeeze(), batch_targets)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                model.eval()
                with torch.no_grad():
                    train_inputs = train_inputs.to(device)
                    test_inputs = test_inputs.to(device)
                    train_targets = train_targets.to(device)
                    test_targets = test_targets.to(device)
                    test_outputs = model(test_inputs.unsqueeze(1)).squeeze()
                    test_loss = criterion(test_outputs, test_targets)
                
                if test_loss < best_loss:
                    best_loss = test_loss
                    no_improvement = 0
                    train_best_model_weights = copy.deepcopy(model.state_dict())
                else:
                    no_improvement += 1

                if no_improvement >= patience:
                    break

            model.load_state_dict(train_best_model_weights)
            model.eval()
            with torch.no_grad():
                train_inputs = train_inputs.to(device)
                test_inputs = test_inputs.to(device)
                train_outputs = model(train_inputs.unsqueeze(1)).squeeze().cpu().detach().numpy()
                test_outputs = model(test_inputs.unsqueeze(1)).squeeze().cpu().detach().numpy()

                train_correlation, _ = pearsonr(train_targets.cpu().detach().numpy(), train_outputs)
                train_r2 = r2_score(train_targets.cpu().detach().numpy(), train_outputs)
                                
                test_correlation, _ = pearsonr(test_targets.cpu().detach().numpy(), test_outputs)
                test_r2 = r2_score(test_targets.cpu().detach().numpy(), test_outputs)
                train_params = None
                train_params = (params['input_size'], params['hidden_size'], params['output_size'], params['num_layers'])
                row = {'train_correlation': train_correlation, 'train_r_squared': train_r2, 'test_correlation': test_correlation,
                       'test_r_squared': test_r2, 'lr': lr, 'batch_size': batch_size, 'hidden_size': train_params[1],
                       'num_layers': train_params[3], 'params': params}
                writer.writerow(row)
                
            if test_r2 > best_r_squared:
                best_r_squared = test_r2
                best_model = model
                best_model_weights = train_best_model_weights
                best_batch_size = batch_size
                best_lr = lr
                best_params = params
                print(f"best_r_squared:", best_r_squared)

print("Best R-squared:", best_r_squared)
print("Best Model:", best_model)
print("Best Batch Size:", best_batch_size)
print("Best Learning Rate:", best_lr)
print("Best Params:", best_params)