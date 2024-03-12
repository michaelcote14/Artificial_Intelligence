import sklearn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    """Plots decision boundaries of model predicting on X in comparison to y.
    Source - https://madewithml.com/courses/foundations/neural-networks/ (with modifications)
    """
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))  # binary

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())



######################### Create the data ###############################
# Create a circles dataset
X, y = make_circles(n_samples=1000, noise=0.03, random_state=42)
circles_df = pd.DataFrame({'X1': X[:, 0], 'X2': X[:, 1], 'label': y})

# Plot circles
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
# plt.show()

# Turn the numpy arrays into tensors
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).float()

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Calculate accuracy (What percentage of our predictions are correct?)
def accuracy_finder(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().float().item()
    accuracy = (correct / len(y_pred)) * 100
    return accuracy

################### Build the model ############################
class CircleModel(nn.Module):
    def __init__(self):
        super().__init__()
        # How to create neural layers in pytorch (the 2nd number must match the 1st number of the next layer)
        # The first in feature must match the number of columns in the data
        self.layer_1 = nn.Linear(in_features=2, out_features=10) # 2 inputs, 5 outputs (usually the second number is a multiple of 8)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1) # 1 final output
        self.relu = nn.ReLU() # this is a non-linear activation function, it is used
        # to make the model more flexible and to allow it to learn more complex patterns
        # Relu removes negative values and replaces them with 0, this is good for binary classification problems

    def forward(self, x):
        input = self.layer_1(x)
        input = self.relu(input)
        input = self.layer_2(input)
        input = self.relu(input)
        return self.layer_3(input) # Pass x (the input) through layer 1, then pass the output of layer 1 through layer 2


# Instantiate the model (to device automatically uses a gpu if available)
model = CircleModel().to(device)

# Put the data on the device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)


# Setup loss function and optimizer
loss_function = nn.BCEWithLogitsLoss() # Binary cross entropy loss (combines sigmoid activation function and binary cross entropy loss in one function)
# This function is usually best for binary classification problems

# Setup optimizer
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001) # Adam optimizer (usually the best optimizer to use)


# Train the model
epochs = 1000
for epoch in range(epochs):
    model.train()

    # Forward pass
    y_logits = model(X_train).squeeze() # Squeeze the output to remove extra dimensions or else this won't work
    y_pred = torch.round(torch.sigmoid(y_logits)) # This puts the logits through the sigmoid activation function and rounds the output to 0 or 1

    # Calculate loss (how far off our predictions are from the actual values, must use logits for this)
    loss = loss_function(y_logits, y_train)


    # Calculate accuracy
    accuracy = accuracy_finder(y_true=y_train, y_pred=y_pred)

    # Optimizer zero grad
    optimizer.zero_grad()

    # Backpropagation
    loss.backward()

    # Step the optimizer (update the weights)
    optimizer.step()

    # Test the model
    model.eval()
    with torch.inference_mode():
        test_logits = model(X_test).squeeze()
        test_preds = torch.round(torch.sigmoid(test_logits))
        test_accuracy = accuracy_finder(y_true=y_test, y_pred=test_preds)

    # Print metrics
    if (epoch + 1) % 10 == 0:
        print(f"Epoch: {epoch+1} | Loss: {loss.item():.4f} | Accuracy: {accuracy:.2f}%")



# Visualize the predictions
with torch.inference_mode():
    trained_preds = model(X_test)
    trained_preds = torch.round(torch.sigmoid(trained_preds))

    # Plot the decision boundary
    plt.figure(figsize=(12, 6))
    plt.title("Predictions")
    plot_decision_boundary(X=X_test, y=trained_preds, model=model)
    plt.show()

