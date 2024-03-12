import gymnasium  # version 0.17.3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from gym import spaces
import numpy as np

def train():
    env = gymnasium.make("CartPole-v1", render_mode='human')  # to not show: render_mode='rgb_array'
    # env = DummyVecEnv([lambda: env])
    # How to check how many actions and states there are
    print("action space:", env.action_space)
    # These are the observations from the environment
    print("observation space:", env.observation_space)
    env = spaces.box.Box(low=-1, high=1, shape=(4,), dtype=np.float32)

    model = PPO("MlpPolicy", env, verbose=1)
    print("Model created")


    model.learn(total_timesteps=20000)

    total_reward = 0
    episodes = 100
    for episode in range(1, episodes + 1):
        print("Episode:", episode)
        state = env.reset()

        done = False
        score = 0

        while not done:
            env.render()

            # Possible actions: 0, 1
            action2 = env.action_space.sample()
            action = get_action(state, episode)

            # Possible states: 4
            # Perform the action
            new_state, reward, done, unknown, info = env.step(action)
            score += reward

            # Train the short memory
            self.trainer.train_short_memory(state, action, reward, new_state, done)

            # Get the average reward
            total_reward += reward
            average_reward = total_reward / episode
            print("Average Reward:", average_reward)

    env.close()




if __name__ == '__main__':
    train()