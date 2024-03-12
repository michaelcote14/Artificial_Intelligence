import gym.wrappers
import gymnasium
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import seaborn as sns
from tqdm import tqdm
import random
from IPython.display import clear_output


class BlackJackAgent:
    def __init__(self, learning_rate:float, initial_epsilon:float,
                 epsilon_decay:float, final_epsilon:float,
                 discount_factor:float=0.95):
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.learning_rate = learning_rate
        self.initial_epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.discount_factor = discount_factor

        self.training_error = []

    def get_action(self, observation):
        if np.random.rand() < self.initial_epsilon:
            return env.action_space.sample()

        else:
            return int(np.argmax(self.q_values[observation]))

    def update_q_values(self, observation, action, reward, next_observation, done):
        future_q_value = (not done) * np.max(self.q_values[next_observation])
        td_error = reward + self.discount_factor * future_q_value - self.q_values[observation][action]
        self.q_values[observation][action] += self.learning_rate * td_error
        self.training_error.append(td_error)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.initial_epsilon * self.epsilon_decay)


if __name__ == '__main__':

    # Hyperparameters
    learning_rate = 0.01
    episodes = 100_000
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (episodes/2)
    final_epsilon = 0.01

    agent = BlackJackAgent(learning_rate=learning_rate, initial_epsilon=start_epsilon,
                            epsilon_decay=epsilon_decay, final_epsilon=final_epsilon)

    env = gymnasium.make('Blackjack-v1', sab=True, render_mode='rgb_array')
    done = False

    # Set the initial environment
    state, info = env.reset()
    # state = (16=myhand, 10=dealerhand, False=usable_ace)

    for episode in tqdm(range(episodes)):
        state = env.reset()
        done = False
        clear_output()

        while not done:
            action = agent.get_action(state)

            state, reward, terminated, next_state, info = env.step(action)

            agent.update_q_values(state, action, reward, next_state, done)
            state = next_state

        agent.decay_epsilon()