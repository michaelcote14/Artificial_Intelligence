import torch
import random
import numpy as np
from ai_snake_game import SnakeGame_AI, Direction, Point, BLOCK_SIZE, WHITE, RED, BLUE1, BLUE2, BLACK, SPEED
# This is for storing data somehow
from collections import deque
from snake_neural_model import My_Model, My_Trainer
import matplotlib.pyplot as plt

# the agent is going to play the game and store the data in memory

# Every smaller learning rate can eventually reach the score above it, it just takes more time to train


############ Hyperparameters ############
# How many items we can store in the deque
MAX_MEMORY = 100_000
BATCH_SIZE = 1_000
learning_rate = 0.0001 # usually ranges from 1 to .0001 # also known as alpha
gamma = 0.5 # discount rate, must be smaller than 1, usually around 0.8-0.99. varies from 0 to 1. This is for determining the value of future rewards
# The lower the gamma, the more the agent will care about immediate rewards
# The higher the gamma, the more the agent will care about future rewards
max_game_number = 10000
save_checkpoint_game_number = 10000
last_game_of_random_decay = 80
N_INPUT_FEATURES = 12
save_model = True
load_model = False

# Max record so far: 80, took 1000 games @ 0.0001 learning rate and 0 epsilon
# Using the above parameters, the max game number reached without getting a score of 70 is..

# ToDo try the exact model from the video
# todo will all learning rates eventually get to a score of 70?
# todo try changing these hyperparameters and see what happens:
# gamma
# epsilon
# max_memory
# batch_size
# last game of random decay
# try passing the snake's length as an input feature

class Agent:
    def __init__(self):
        self.number_of_games = 0
        self.record = 0
        self.epsilon = 0.7 # controls randomness of moves in the beginning. 0 = pure greedy, meaning the snake will always take the best immediate action
        # 1 = pure random, meaning the snake will always take a random action, regardless of the current state
        self.gamma = gamma
        self.memory = deque(maxlen=MAX_MEMORY) # If we exceed the max memory, it will remove the oldest memory (the leftmost one)
        self.model = My_Model() # 11 inputs, 256 neurons in the hidden layer, 3 outputs
        # the 3 outputs are the 3 possible actions: left, right, straight
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.my_trainer = My_Trainer(self.model, self.optimizer, learning_rate=learning_rate, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        # Direction of the snake
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # Detect collision
        # If the snake is going left and the point to the left is in the snake, then it is true
        # If the snake is going right and the point to the right is in the snake, then it is true
        # If the snake is going up and the point to the up is in the snake, then it is true
        # If the snake is going down and the point to the down is in the snake, then it is true
        # If the snake is going left and the point to the left is in the snake, then it is true
        # If the snake is going right and the point to the right is in the snake, then it is true
        # If the snake is going up and the point to the up is in the snake, then it is true
        # If the snake is going down and the point to the down is in the snake, then it is true
        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x, # Food is to the left of the snake
            game.food.x > game.head.x, # Food is to the right of the snake
            game.food.y < game.head.y, # Food is to the up of the snake
            game.food.y > game.head.y, # Food is to the down of the snake

            # Snake length
            game.snake_length < 30
        ]

        return np.array(state, dtype=int)

    # done is the game over state
    def remember(self, old_state, action, reward, new_state, done):
        self.memory.append((old_state, action, reward, new_state, done)) # pop from the left if we exceed the max memory

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # returns a list of tuples
        else:
            # If we don't have enough memory, we will just use the memory we have
            mini_sample = self.memory

        states, actions, rewards, new_states, dones = zip(*mini_sample) # *mini_sample is to unpack the list of tuples
        self.my_trainer.train_step(states, actions, rewards, new_states, dones) # train the model

    def train_short_memory(self, old_state, action, reward, new_state, done):
        self.my_trainer.train_step(old_state, action, reward, new_state, done) # train the model

    def get_action(self, old_state):
        # in the beginning, we want to do random moves, but over time, we want to do
        # less random moves and instead do more moves that are based on the model
        self.epsilon = last_game_of_random_decay - self.number_of_games
        final_action = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            action = random.randint(0, 2)
            final_action[action] = 1
        else:
            # convert state to tensor so that pytorch can use it to make a decision
            state0 = torch.tensor(old_state, dtype=torch.float)
            prediction = self.model(state0)
            action = torch.argmax(prediction).item()
            final_action[action] = 1

        return final_action

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    agent = Agent()
    game = SnakeGame_AI()

    # if load_model == True:
    #     agent.model.load_checkpoint(agent, file_path='model/snake.chkpt')

    # This is the training loop
    while agent.number_of_games < max_game_number:
        old_state = agent.get_state(game)

        action = agent.get_action(old_state)

        # print('action', action)

        # perform action and get new state
        reward, done, score = game.play_step(action)
        new_state = agent.get_state(game)

        # train short memory
        agent.train_short_memory(old_state, action, reward, new_state, done)

        # remember
        agent.remember(old_state, action, reward, new_state, done)


        if done:
            # train long memory (trains the model with the data we have collected in previous games)
            game.reset()
            agent.number_of_games += 1

            agent.train_long_memory()

            if score > agent.record:
                agent.record = score

            total_score += score

            if score > 0:
                average_score = total_score / agent.number_of_games
            else:
                average_score = 0

            print('Game:', agent.number_of_games, 'Score:', score, 'Record:', agent.record)

            # Save the plot data
            plot_scores.append(score)
            plot_mean_scores.append(average_score)

            plot(plot_scores, plot_mean_scores)

            # Save the model if you want to
            if agent.number_of_games % save_checkpoint_game_number == 0:
                # Quit the program
                quit()

                answer = input("Press enter to save the model")
                if answer == "":
                    agent.model.save_checkpoint(model=agent.model,
                                                optimizer=agent.optimizer,
                                                epsilon=agent.epsilon,
                                                number_of_games=agent.number_of_games,
                                                record=agent.record,
                                                file_name='snake_model.pth')

def plot(scores, mean_scores):
    plt.plot(scores, color='red')
    plt.plot(mean_scores, color='blue')
    # Make a legend
    plt.legend(['Score', 'Mean Score'])
    # Save the plot
    plt.savefig('plot.png')

if __name__ == '__main__':
    train()
