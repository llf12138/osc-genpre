import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import copy

train_data = pd.read_csv('train.csv')
train_inputs = torch.tensor(train_data.iloc[:, 4:].values, dtype=torch.float32)
train_targets = torch.tensor(train_data.iloc[:, 3].values, dtype=torch.float32)

test_data = pd.read_csv('test.csv')
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

input_size = 30
hidden_size = 150
output_size = 1
num_layers = 1
num_epochs = 100
batch_size = 60
l2_factor = 0.01
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel(input_size, hidden_size, output_size, num_layers, l2_factor)
model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=l2_factor)

patience = 20
best_loss = float('inf')
no_improvement = 0
best_model_weights = copy.deepcopy(model.state_dict())

for epoch in range(num_epochs):
    model.train()
    permutation = torch.randperm(train_inputs.size()[0])
    for i in range(0, train_inputs.size()[0], batch_size):
        indices = permutation[i:i + batch_size]
        batch_inputs = train_inputs[indices]
        batch_targets = train_targets[indices]
                    
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        outputs = model(batch_inputs.unsqueeze(1))
        loss = criterion(outputs.squeeze(), batch_targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
   
    model.eval()
    with torch.no_grad():
        test_inputs = test_inputs.to(device)
        test_outputs = model(test_inputs.unsqueeze(1)).squeeze().cpu()
        test_loss = criterion(test_outputs, test_targets)
    
    if test_loss < best_loss:
        best_loss = test_loss
        best_model_weights = copy.deepcopy(model.state_dict())
        no_improvement = 0
    else:
        no_improvement += 1
        if no_improvement >= patience:
            print("Early stopping, no improvement in test loss.")
            break

    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, Train Loss: {loss}, Test Loss: {test_loss}")