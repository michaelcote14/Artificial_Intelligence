import gymnasium
import torch
import random
import numpy as np
from collections import deque
from cart_pole_network import Model, Trainer

learning_rate = 0.01
episodes = 100_000
MAX_MEMORY = 100_000

last_game_of_random_decay = 80
save_model = True
load_model = False

class Agent():
    def __init__(self):
        self.model = Model()
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.95
        self.learning_rate = learning_rate
        self.episodes = episodes
        self.memory = deque(maxlen=MAX_MEMORY)
        self.last_game_of_random_decay = last_game_of_random_decay
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.trainer = Trainer(self.model, self.optimizer, self.learning_rate, self.gamma)

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(2)
        else:
            return np.argmax(self.model(state))

    def train_short_memory(self, old_state, action, reward, new_state, done):
        self.trainer.train_step(old_state, action, reward, new_state, done) # train the model

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # returns a list of tuples
        else:
            # If we don't have enough memory, we will just use the memory we have
            mini_sample = self.memory

        states, actions, rewards, new_states, dones = zip(*mini_sample) # *mini_sample is to unpack the list of tuples
        self.trainer.train_step(states, actions, rewards, new_states, dones) # train the model

    def remember(self, old_state, action, reward, new_state, done):
        self.memory.append((old_state, action, reward, new_state, done)) # pop from the left if we exceed the max memory


if __name__ == "__main__":
    env = gymnasium.make('CartPole-v1', render_mode='human')
    agent = Agent()

    env.reset()
    env.render()

    states = env.observation_space.shape[0]
    actions = env.action_space.n


    episodes = 10
    number_of_games_played = 0
    for episode in range(1, episodes+1):
        # 4 different states possible
        state = env.reset()[0]

        done = False
        score = 0

        while done == False:
            # 2 different actions possible
            action = agent.get_action(state)

            _, reward, done, __, info = env.step(action)
            print("done:", done)

            new_state = env.reset()[0]

            # train short memory
            agent.train_short_memory(state, action, reward, new_state, done)

            # remember
            agent.remember(state, action, reward, new_state, done)

            if done == True:
                env.reset()
                number_of_games_played += 1

                # train long memory
                agent.train_long_memory()

            score += reward

        print('Episode', episode, 'Score:', score)
        env.close()