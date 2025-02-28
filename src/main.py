import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Tensor Train decomposition
class TensorTrain(nn.Module):
    def __init__(self, d, tt_rank):
        super(TensorTrain, self).__init__()
        self.d = d
        self.tt_rank = tt_rank
        self.cores = nn.ParameterList([
            nn.Parameter(torch.randn((1, tt_rank, 2))),  # First core
            *[nn.Parameter(torch.randn((tt_rank, tt_rank, 2))) for _ in range(d - 2)],
            nn.Parameter(torch.randn((tt_rank, 1, 2)))  # Last core
        ])

    def forward(self, X):
        batch_size = X.shape[0]
        TT_result = self.cores[0][:, :, X[:, 0].long()].permute(2, 0, 1)
        for i in range(1, self.d):
            core = self.cores[i][:, :, X[:, i].long()].permute(2, 0, 1)
            TT_result = torch.bmm(TT_result, core)
        return TT_result.squeeze()

# Exponential Machine Model
class ExponentialMachine(nn.Module):
    def __init__(self, d, tt_rank):
        super(ExponentialMachine, self).__init__()
        self.tt = TensorTrain(d, tt_rank)
        self.bias = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, X):
        return self.tt(X) + self.bias

# Generate synthetic dataset
def generate_data(n_samples=10000, d=10):
    X = torch.randint(0, 2, (n_samples, d)).float()
    W_true = torch.randn([2] * d)
    indices = X.long().T.tolist()
    y = torch.tensor([W_true[tuple(idx)] for idx in zip(*indices)]).sign()
    return X, y

# Train model using Stochastic Riemannian Optimization
def train_model(model, X, y, lr=0.01, epochs=100, batch_size=128):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    losses = []
    
    for epoch in range(epochs):
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            y_pred = model(X_batch).squeeze()
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
        
        losses.append(loss.item())
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
    
    # Plot training loss
    plt.plot(range(epochs), losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.show()

# Run the training process
d = 10
X, y = generate_data(n_samples=10000, d=d)
y = (y + 1) / 2  # Convert to {0,1} for binary classification

model = ExponentialMachine(d, tt_rank=3)
train_model(model, X, y, lr=0.01, epochs=100)

# Evaluate model
X_test, y_test = generate_data(n_samples=2000, d=d)
y_test = (y_test + 1) / 2
with torch.no_grad():
    y_pred = torch.sigmoid(model(X_test)) > 0.5
    accuracy = (y_pred == y_test).float().mean().item()
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Plot predictions
    plt.hist(y_pred.numpy().astype(int), bins=2, alpha=0.7, label="Predicted")
    plt.hist(y_test.numpy().astype(int), bins=2, alpha=0.7, label="True")
    plt.xlabel("Class")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("Prediction Distribution")
    plt.savefig("prediction_distribution.png")
    plt.show()
