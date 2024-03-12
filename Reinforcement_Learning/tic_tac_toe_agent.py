import numpy as np
import torch
import random

from tic_tac_toe_game import tic_tac_toe
from collections import deque
from tic_tac_toe_neural_model import My_Model, My_Trainer

# How many items can be stored in the deque list
MAX_MEMORY = 100_000

# How many items to train on at a time
BATCH_SIZE = 1000

class Agent:
    def __init__(self):
        self.number_of_games = 0
        self.epsilon = 0  # randomness
        # If we exceed the max memory, it will remove the oldest memory from the network's memory
        self.memory = deque(maxlen=MAX_MEMORY)
        self.my_model = My_Model()
        self.my_trainer = My_Trainer(self.my_model)

    def get_state(self, game):
        state = [
            # The empty positions on the board
            [(game.board[i] == '-') for i in range(3) for j in range(3)],

            # The positions on the board that are X
            [(game.board[i] == 'X') for i in range(3) for j in range(3)],

            # The positions on the board that are O
            [(game.board[i] == 'O') for i in range(3) for j in range(3)]

            ]

        new_state = []
        for index in state[0]:
            if index == True:
                new_state.append(1)
            else:
                new_state.append(0)
        for index in state[1]:
            if index == True:
                new_state.append(1)
            else:
                new_state.append(0)
        for index in state[2]:
            if index == True:
                new_state.append(1)
            else:
                new_state.append(0)

        return np.array(new_state)

    # This decides which spot to place the X. 0 is the top left, 1 is the top middle, 2 is the top right, 3 is the middle left, etc.
    def get_action(self, state):
        # in the beginning, do a bunch of random moves so the model can learn,
        # but as the model learns, do less random moves, finally after 80 games,
        # the model will only do model moves
        self.epsilon = 80 - self.number_of_games
        # This is the move if no other move is chosen
        final_move = [0, 0, 0, 0, 0, 0, 0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 8)
            final_move[move] = 1
            print("Random Move:" , final_move)
        else:
            # This is the move if the model chooses the move
            state0 = torch.tensor(state, dtype=torch.float32)
            # state0 must be the same number as the input layer of the model
            prediction = self.my_model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
            print("Model Move:", final_move)

        # Convert the move to a number from 0 to 8
        final_move = int(np.argmax(final_move))

        return final_move

    def train_short_memory(self, state, action, reward, next_state, is_game_over):
        self.my_trainer.train_step(state, action, reward, next_state, is_game_over)

    # This checks the long term memory and trains the model
    def remember(self, state, action, reward, next_state, is_game_over):
        self.memory.append((state, action, reward, next_state, is_game_over))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            # If the memory is less than the batch size, then we will use the entire memory
            mini_sample = self.memory

        for state, action, reward, next_state, done in mini_sample:
            self.train_short_memory(state, action, reward, next_state, done) # train the model


def train():
    consecutive_games_won = 0
    games_won_record = 0
    agent = Agent()
    game = tic_tac_toe()

    # This is the training loop
    while True:
        # Get the old state
        state_old = agent.get_state(game)

        # Get the move
        counter = 0
        neural_move = agent.get_action(state_old)
        while game.board[neural_move] != '-':
            neural_move = agent.get_action(state_old)
            counter += 1
            if counter > 10:
                game.reset()

        # Perform the move
        reward, is_game_over, winner = game.perform_move(neural_move)
        print('reward:', reward)

        game.displayBoard(game.board)

        if counter > 10:
            reward = -10

        # Get the new state
        state_new = agent.get_state(game)

        # Train short memory
        agent.train_short_memory(state_old, neural_move, reward, state_new, is_game_over)

        # Remember the previous state, action, reward, and the new state
        agent.remember(state_old, neural_move, reward, state_new, is_game_over)

        # game.displayBoard(game.board)

        # If the game is over, then train the long memory
        if is_game_over == True or counter > 10:
            # Print the number of games won
            if winner == 'X':
                consecutive_games_won += 1
            elif winner == 'O':
                consecutive_games_won = 0
            else:
                consecutive_games_won = 0

            # Reset the game
            game.reset()
            agent.number_of_games += 1

            # Train the long memory
            agent.train_long_memory()

            print('Number of games played:', agent.number_of_games)
            print('Games won in a row record:', games_won_record)

            if consecutive_games_won > games_won_record:
                games_won_record = consecutive_games_won
                # torch.save(agent.my_model.state_dict(), 'best_model.pth')
                print('New record of consecutive games won!:', games_won_record)

if __name__ == '__main__':
    train()