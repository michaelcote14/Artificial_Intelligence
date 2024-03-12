import torch
import torch.nn as nn
import torch.optim as optim


class My_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=4, out_features=256)
        self.layer_2 = nn.Linear(in_features=256, out_features=256)
        self.layer_3 = nn.Linear(in_features=256, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        input = self.layer_1(x)
        input = self.relu(input)
        input = self.layer_2(input)
        input = self.relu(input)
        return self.layer_3(input)

class My_Trainer:
    def __init__(self, model):
        self.learning_rate = 0.0001
        self.gamma = 0.6  # discount rate
        self.model = model
        self.optimizer = optim.Adam(params=model.parameters(), lr=self.learning_rate)
        self.loss_function = nn.MSELoss()

    def train_short_memory(self, state, action, reward, next_state, is_game_over):
        # Turn the numpy arrays into tensors
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            # This is to make the tensor 2D
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            is_game_over = (is_game_over, )

            # 1. Predict the Q values of the current state (Q values are how good the neural network's action was)
            pred = self.model(state)

            # 2. Get the Q values of the next state
            target = pred.clone()
            for index in range(len(is_game_over)):
                Q_new = reward[index]
                if not is_game_over[index]:
                    Q_new = reward[index] + self.gamma * torch.max(self.model(next_state[index]))

                target[index][torch.argmax(action).item()] = Q_new

                self.optimizer.zero_grad()
                loss = self.loss_function(target, pred)
                loss.backward()

                self.optimizer.step()

    def get_action(self, state, epsilon):
        # Get the Q values of the state
        pred = self.predict(state)

        # Choose a random action with probability epsilon
        if torch.rand(1) < epsilon:
            return torch.tensor([[random.randrange(9)]], dtype=torch.long)

        # Choose the action with the highest Q value
        return torch.argmax(pred).reshape(1, 1)
