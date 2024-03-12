import torch
import random
import numpy as np
from collections import deque
from roulette_game import RouletteGame
from roulette_network import Network, Trainer

# Hyperparameters
learning_rate = 0.01
gamma = 0.9
epsilon = 0.9
game_limit = 100000
last_game_of_random_decay = 0.1 * game_limit
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
create_new_model = True


class Agent:
    def __init__(self):
        self.number_of_games = 0
        self.epsilon = 0
        self.gamma = 0
        self.memory = deque(maxlen=MAX_MEMORY)
        self.record = 0
        self.account_balance = 0
        self.model = Network()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.trainer = Trainer(self.model, self.optimizer, learning_rate=learning_rate, gamma=self.gamma)

    def get_action(self, state):
        self.epsilon = last_game_of_random_decay - self.number_of_games
        # first digit 0 means no play, second digit means different bets such as low, high, red, black, odd, even
        final_action = [0, 0]

        if random.randint(0, 200) < self.epsilon:
            action = random.randint(0, 1)
            final_action[action] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            action = torch.argmax(prediction).item()
            final_action[action] = 1

        return final_action

    def get_state(self, game):
        state = [
            game.consecutive_reds,
            game.consecutive_blacks,
            game.consecutive_lows,
            game.consecutive_highs,
            game.consecutive_odds,
            game.consecutive_evens,
        ]

        return np.array(state, dtype=int)

    def train_short_memory(self, old_state, action, reward, new_state, done):
        self.trainer.train_step(old_state, action, reward, new_state, done) # train the model


def train():
    agent = Agent()
    game = RouletteGame()
    # todo implement control over how much money the nueral network can bet
    money_to_bet = 1.00
    total_score = 0
    done = False

    if create_new_model == False:
        agent.model.load_checkpoint(agent, file_path='best_roulette_model.chkpt')

    while not done:
        agent.number_of_games += 1
        single_spin_winnings = 0

        old_state = agent.get_state(game)
        # print('state', old_state)

        action = agent.get_action(old_state)

        reward, single_spin_winnings = game.play_step(action)

        agent.account_balance += single_spin_winnings
        print('account balance:', agent.account_balance)

        new_state = agent.get_state(game)

        agent.train_short_memory(old_state, action, reward, new_state, done)

        if agent.number_of_games > 50:

            if single_spin_winnings > agent.record:
                agent.record = single_spin_winnings

            # Save the model if you want to
            if agent.number_of_games % 10000 == 0:
                # answer = input("Press enter to save the model")
                # if answer == "":
                agent.model.save_checkpoint(model=agent.model,
                                                optimizer=agent.optimizer,
                                                epsilon=agent.epsilon,
                                                number_of_games=agent.number_of_games,
                                                account_balance=agent.account_balance,
                                                file_name='roulette_model.pth')

                # print('Game:', agent.number_of_games, 'Account Balance:', agent.account_balance)
        


if __name__ == "__main__":
    train()