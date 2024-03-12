import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=6, out_features=256)
        self.layer_2 = nn.Linear(in_features=256, out_features=256)
        self.layer_3 = nn.Linear(in_features=256, out_features=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        input = self.layer_1(x)
        input = self.relu(input)
        input = self.layer_2(input)
        input = self.relu(input)
        return self.layer_3(input)

    def save_checkpoint(self, model, optimizer, epsilon, number_of_games, account_balance, file_name="my_checkpoint.pth"):
        model_folder_path = r'\model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epsilon": epsilon,
            "number_of_games": number_of_games,
            "account_balance": account_balance
                    }, "best_roulette_model.chkpt")

    def load_checkpoint(self, agent, file_path):
        print("Loading model")
        checkpoint = torch.load(file_path)
        agent.model.load_state_dict(checkpoint['model_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.epsilon = checkpoint['epsilon']
        agent.number_of_games = checkpoint['number_of_games']
        agent.account_balance = checkpoint['account_balance']

class Trainer:
    def __init__(self, model, optimizer, learning_rate, gamma):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.model = model
        self.optimizer = optimizer
        self.loss_function = nn.MSELoss()


    def train_step(self, old_state, action, reward, new_state, done):
        # Turn the numpy arrays into tensors
        old_state = torch.tensor(np.array(old_state), dtype=torch.float)
        new_state = torch.tensor(np.array(new_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)

        if len(old_state.shape) == 1:
            # This is to make the tensor 2D
            old_state = torch.unsqueeze(old_state, 0)
            new_state = torch.unsqueeze(new_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

            # 1. Predict the Q values of the current state
            pred = self.model(old_state)

            # 2. Get the Q values of the next state
            target = pred.clone()
            for index in range(len(done)):
                Q_new = reward[index]
                if not done[index]:
                    Q_new = reward[index] + self.gamma * torch.max(self.model(new_state[index]))

                target[index][torch.argmax(action).item()] = Q_new

                self.optimizer.zero_grad()
                loss = self.loss_function(target, pred)
                loss.backward()

                self.optimizer.step()
