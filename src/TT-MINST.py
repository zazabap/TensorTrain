import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# --- TT-Compressed Linear Layer ---
class TTLinear(nn.Module):
    """
    A TT-compressed fully connected layer.
    The weight matrix of size (prod(in_shape) x prod(out_shape)) is factorized into TT-cores.
    
    Attributes:
        in_shape (tuple): Factorization of the input dimension.
        out_shape (tuple): Factorization of the output dimension.
        tt_rank (list): TT-ranks as a list of length len(in_shape)+1.
    """
    def __init__(self, in_shape, out_shape, tt_rank):
        super(TTLinear, self).__init__()
        self.in_shape = in_shape      # e.g., (28, 28) for a total input dim of 784
        self.out_shape = out_shape    # e.g., (2, 5) for a total output dim of 10
        self.num_dims = len(in_shape)
        self.tt_rank = tt_rank        # e.g., [1, 2, 1]

        # Initialize TT-cores as trainable parameters.
        # Each core has shape (r_{k-1}, in_shape[k], out_shape[k], r_k)
        self.cores = nn.ParameterList()
        for k in range(self.num_dims):
            core_shape = (tt_rank[k], in_shape[k], out_shape[k], tt_rank[k+1])
            core = nn.Parameter(torch.randn(core_shape))
            self.cores.append(core)
    
    def reconstruct_weight(self):
        """
        Reconstructs the full weight matrix from the TT-cores.
        The full weight matrix will have shape (prod(in_shape), prod(out_shape)).
        """
        # Start with the first TT-core: shape (r0, i1, o1, r1) where r0=1.
        weight = self.cores[0]  
        # Sequentially contract with the remaining cores.
        for k in range(1, self.num_dims):
            weight = torch.tensordot(weight, self.cores[k], dims=([-1], [0]))
            # Weight shape becomes: (r0, i1, o1, i2, o2, ..., i_{k+1}, o_{k+1}, r_{k+1})
        # Remove trivial boundary dimensions (r0 and r_{num_dims}) which are 1.
        weight = weight.squeeze(0).squeeze(-1)
        # Now weight has shape: (i1, o1, i2, o2, ..., i_d, o_d)
        # Permute dimensions to group all input indices first, then output indices.
        dims = []
        for k in range(self.num_dims):
            dims.append(2 * k)     # input indices positions.
        for k in range(self.num_dims):
            dims.append(2 * k + 1)   # output indices positions.
        weight = weight.permute(dims)
        # Reshape into a matrix of shape (prod(in_shape), prod(out_shape)).
        weight = weight.contiguous().view(-1, int(torch.prod(torch.tensor(self.out_shape))))
        return weight

    def forward(self, x):
        """
        Forward pass of the TT layer.
        Args:
            x: Input tensor of shape (batch_size, prod(in_shape)).
        Returns:
            Output tensor of shape (batch_size, prod(out_shape)).
        """
        weight = self.reconstruct_weight()  # Shape: (prod(in_shape), prod(out_shape))
        return F.linear(x, weight.t())

# --- Simple Network Using the TT-Compressed Layer ---
class TTNet(nn.Module):
    def __init__(self, in_shape, out_shape, tt_rank):
        super(TTNet, self).__init__()
        # For MNIST: input images are 28x28 (flattened to 784).
        # We use a single TTLinear layer for classification.
        self.tt_linear = TTLinear(in_shape, out_shape, tt_rank)
        
    def forward(self, x):
        # Flatten the image: shape (batch_size, 784)
        x = x.view(x.size(0), -1)
        out = self.tt_linear(x)
        return out

# --- Data Loading ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize grayscale images.
])

# Training dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Test dataset for evaluation
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# --- Model, Loss, and Optimizer Setup ---
# Factorization:
# Input dimension factorized as (28, 28) since 28*28 = 784.
# Output dimension factorized as (2, 5) since 2*5 = 10 (for 10 classes).
in_shape = (28, 28)
out_shape = (2, 5)
tt_rank = [1, 2, 1]

model = TTNet(in_shape, out_shape, tt_rank)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- Training Loop ---
num_epochs = 100  # You can adjust the number of epochs as needed
train_losses = []
train_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Compute training accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(train_accuracy)
    print(f"Epoch [{epoch+1}/{num_epochs}] Training Loss: {epoch_loss:.4f}, Accuracy: {train_accuracy:.2f}%")

# --- Evaluation on Test Set ---
model.eval()
test_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

avg_test_loss = test_loss / total
test_accuracy = 100 * correct / total

print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

# --- Plotting the Results ---
epochs = range(1, num_epochs + 1)

fig, ax1 = plt.subplots(figsize=(8, 6))

color = 'tab:red'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Training Loss', color=color)
ax1.plot(epochs, train_losses, marker='o', color=color, label='Training Loss')
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend(loc='upper left')

ax2 = ax1.twinx()  # Create a second y-axis sharing the same x-axis.
color = 'tab:blue'
ax2.set_ylabel('Training Accuracy (%)', color=color)
ax2.plot(epochs, train_accuracies, marker='s', color=color, label='Training Accuracy')
ax2.tick_params(axis='y', labelcolor=color)
ax2.legend(loc='upper right')

plt.title("Training Loss and Accuracy over Epochs")
plt.tight_layout()
plt.savefig("training_result.png")
plt.show()
