import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

torch.manual_seed(42)  # same starting weights every run, so results are reproducible


# ---------------------------------------------------------------
# 1. Load the dataset
# ---------------------------------------------------------------
iris = load_iris()
X = iris.data    # 150 flowers x 4 measurements (sepal/petal length and width)
y = iris.target  # 0 = setosa, 1 = versicolor, 2 = virginica

print(f"Features: {iris.feature_names}")
print(f"Classes:  {iris.target_names.tolist()}")
print(f"Shape:    {X.shape}")


# ---------------------------------------------------------------
# 2. Split into training and testing sets
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features so they're on a similar range; this helps training.
# Fit on training data only to avoid leaking test information.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ---------------------------------------------------------------
# 3. Convert to tensors
# ---------------------------------------------------------------
# Features must be float32; class labels for CrossEntropyLoss must be int64 (long)
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)


# ---------------------------------------------------------------
# 4. Define the neural network
# ---------------------------------------------------------------
class IrisNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.hidden = nn.Linear(input_size, hidden_size)   # input layer -> hidden layer
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, num_classes)  # hidden layer -> output layer

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        # No softmax here: CrossEntropyLoss applies it internally,
        # so the network outputs raw scores ("logits")
        return x


model = IrisNet(input_size=4, hidden_size=16, num_classes=3)
print(f"\n{model}")


# ---------------------------------------------------------------
# 5. Train with Cross-Entropy Loss and Adam
# ---------------------------------------------------------------
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 100

print("\nTraining...")
for epoch in range(1, epochs + 1):
    model.train()

    outputs = model(X_train)          # forward pass
    loss = loss_fn(outputs, y_train)  # how wrong the predictions are

    optimizer.zero_grad()  # clear gradients from the previous step
    loss.backward()        # compute new gradients
    optimizer.step()       # update the weights

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d}/{epochs}  Loss: {loss.item():.4f}")


# ---------------------------------------------------------------
# 6. Evaluate on the test set
# ---------------------------------------------------------------
model.eval()
with torch.no_grad():  # no gradients needed when just predicting
    test_outputs = model(X_test)
    predictions = test_outputs.argmax(dim=1)  # class with the highest score
    correct = (predictions == y_test).sum().item()
    accuracy = correct / len(y_test)

print(f"\nTest accuracy: {accuracy:.2%} ({correct}/{len(y_test)} correct)")
