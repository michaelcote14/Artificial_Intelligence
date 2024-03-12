import torch
from torch import nn # nn contains all the building blocks for neural networks
import matplotlib.pyplot as plt
import numpy as np

# Check pytorch version
print("Pytorch version:", torch.__version__)

# The activation function is found in the middle layer

# Create known parameters
# Weights are the importance of each feature and impact your initial move through the layers and changed
# during backpropagation
weight = 0.7
bias = 0.3

# Create a linear regression dataframe in tensor form
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1) # unsqueeze adds a dimension to the tensor
y = (weight * X) + bias # y = mx + b

# Split the data into training and test sets
train_size = int(0.8 * len(X))
test_size = len(X) - train_size
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


def plot_predictions(train_data=X_train, train_labels=y_train, test_data=X_test, test_labels=y_test, predictions=None):
    plt.figure(figsize=(12, 8))
    plt.scatter(train_data, train_labels, label="Training Data", color="blue")
    plt.scatter(test_data, test_labels, label="Test Data", color="green")
    if predictions is not None:
        plt.scatter(test_data, predictions, label="Predictions", color="red")
    plt.legend()
    plt.show()

    # Create a linear regression model
    # What our model does:
    # Start with random values for weights and biases
    # Look at training data and adjust the weights and biases to minimize the loss
    # It will do this through 2 methods: gradient descent and backpropagation
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        # Sometimes the parameters are set for us, such as if you do transfer learning, which is when you want to
        # save the fine-tuned settings of a model that you've already trained and transfer them to a new model
        self.weights = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float32))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float32))

    # You have to add this in if your class inherits from nn.Module # this is also called the forward pass/forward propagation
    def forward(self, x: torch.Tensor):
        return self.weights * x + self.bias # this is the linear regression formula


############################ Create the model ############################
model = LinearRegression()

# Check out the parameters
print("Parameters:", model.state_dict())

# Make predictions with the model
# Inference mode is used to tell the model that we are not training it (makes it faster)
with torch.inference_mode():
    # The X_test is passed to the forward function
    y_predictions = model(X_test)

print("Predictions:\n", y_predictions)
# plot_predictions(predictions=y_predictions)

########################## Train the model ###########################
# Training is essentially the process of adjusting the weights and biases to minimize the loss
# Note: loss function is also called the cost function or criterion sometimes
# The loss function is the difference between the predicted value and the actual value

# 3 types of loss functions: regression, classification, and ranking

# Create a loss function
loss_function = nn.MSELoss() # L1Loss is the mean absolute error, MSELoss is the mean squared error
# MSE Loss is better for comparing models, but L1 Loss gives a better idea of how far off the predictions are

# Create an optimizer which is used to adjust the weights and biases
# SGD is stochastic gradient descent, this is the most popular optimizer
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01) # lr = learning rate ** very important to tune this

# Track different values
epoch_count = []
train_loss_values = []
test_loss_values = []

# Build a training loop
# An epoch is a single pass through the entire training set
epochs = 1000 # ToDo find out how many epochs is best to use, as too many will cause overfitting
for epoch in range(epochs):
    # Set the model to training mode, which affects gradients
    model.train()

    # Make predictions (forward pass)
    y_predictions = model(X_train)

    # Calculate the loss (this measures how wrong your predictions are)
    train_loss = loss_function(y_predictions, y_train)
    # Notice how the loss decreases over time, that's because the model is learning

    # Update the weights and biases
    # Zero the gradients
    optimizer.zero_grad()

    # Backpropagation
    train_loss.backward()

    # Update the parameters
    optimizer.step()

    # Test the model
    # Set the model to evaluation mode, which makes sure that the model is not training and won't change the parameters
    model.eval()
    with torch.inference_mode():
        test_predictions = model(X_test)
        test_loss = loss_function(test_predictions, y_test)

    if epoch % 10 == 0:
        epoch_count.append(epoch)
        train_loss_values.append(train_loss)
        test_loss_values.append(test_loss)
        print("\nEpoch: ", epoch)
        print("Train Loss: ", train_loss.item())
        print("Test loss: ", test_loss.item())

# Plot the loss curves
plt.figure(figsize=(12, 8))
plt.plot(epoch_count, np.array(torch.tensor(train_loss_values).numpy()), label="Train Loss")
plt.plot(epoch_count, np.array(torch.tensor(test_loss_values).numpy()), label="Test Loss")
plt.title("Loss Curves")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Now that the model is trained, we can make predictions
with torch.inference_mode():
    # The X_test is passed to the forward function
    y_predictions = model(X_test)

print("Predictions:\n", y_predictions)
plot_predictions(predictions=y_predictions)

# Check out the parameters after all the training
print("Parameters:", model.state_dict())

# Save the model (this saves the parameters and the object)
torch.save(obj=model.state_dict(), f="linear_regression_model.pt")

# Load a model if one already exists (all this really does is load the weights and biases)
loaded_model = LinearRegression()
loaded_model.load_state_dict(torch.load("linear_regression_model.pt"))

# Make predictions with the loaded model
loaded_model.eval()
with torch.inference_mode():
    loaded_model_predictions = loaded_model(X_test)

print("Predictions:\n", loaded_model_predictions)






