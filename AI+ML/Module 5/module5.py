from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

torch.manual_seed(42)  # same starting weights every run, so results are reproducible

data_dir = Path(__file__).resolve().parent / "data"


# ---------------------------------------------------------------
# 1. Load, normalize, and convert the dataset to tensors
# ---------------------------------------------------------------
# MNIST: 28x28 grayscale images of handwritten digits 0-9.
# It already comes split into 60,000 training and 10,000 test images.
transform = transforms.Compose([
    transforms.ToTensor(),                      # image -> tensor, pixels scaled from 0-255 to 0-1
    transforms.Normalize((0.1307,), (0.3081,)), # normalize with MNIST's mean and std
])

train_data = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
test_data = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

print(f"Training images: {len(train_data)}")
print(f"Test images:     {len(test_data)}")
print(f"Image shape:     {tuple(train_data[0][0].shape)}  (channels, height, width)")

# DataLoaders feed the images to the model in small batches
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000, shuffle=False)


# ---------------------------------------------------------------
# 2. Define the neural network
# ---------------------------------------------------------------
class DigitNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.hidden = nn.Linear(input_size, hidden_size)   # input layer -> hidden layer
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, num_classes)  # hidden layer -> output layer

    def forward(self, x):
        # Flatten each image from (1, 28, 28) into a 1-D vector of 784 pixels
        x = x.view(x.size(0), -1)
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        # No softmax here: CrossEntropyLoss applies it internally
        return x


model = DigitNet(input_size=28 * 28, hidden_size=128, num_classes=10)
print(f"\n{model}")


# ---------------------------------------------------------------
# 3. Train with Cross-Entropy Loss and Adam
# ---------------------------------------------------------------
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 5

print("\nTraining...")
for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0.0

    for images, labels in train_loader:
        outputs = model(images)          # forward pass
        loss = loss_fn(outputs, labels)  # how wrong the predictions are

        optimizer.zero_grad()  # clear gradients from the previous batch
        loss.backward()        # compute new gradients
        optimizer.step()       # update the weights

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch}/{epochs}  Average loss: {average_loss:.4f}")


# ---------------------------------------------------------------
# 4. Evaluate on the test set
# ---------------------------------------------------------------
model.eval()
correct = 0
with torch.no_grad():  # no gradients needed when just predicting
    for images, labels in test_loader:
        outputs = model(images)
        predictions = outputs.argmax(dim=1)  # digit with the highest score
        correct += (predictions == labels).sum().item()

accuracy = correct / len(test_data)
print(f"\nTest accuracy: {accuracy:.2%} ({correct}/{len(test_data)} correct)")
