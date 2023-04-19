import csv
import os
import retro
import gym
import cv2
import optuna
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from gym import spaces
from dotenv import load_dotenv

load_dotenv()

game_name = os.environ['GAME_NAME']
model_prefix = os.environ['MODEL_PREFIX']
model_file = model_prefix + os.environ['MODEL_FILE_SUFFIX']
model_dir = os.environ['MODEL_DIR']
tensorboard_log = os.environ['TENSORBOARD_LOG']
log_dir = os.environ['LOG_DIR']
render = bool(os.environ['RENDER'])
timesteps = int(os.environ['TIMESTEPS'])
trials = int(os.environ['TRIALS'])
num_envs = int(os.environ['PARALLEL_ENVS'])
learning_rate = float(os.environ['LEARNING_RATE'])
n_epochs = int(os.environ['N_EPOCHS'])
eval_episodes = int(os.environ['N_EVAL_EPISODES'])

class RenderCallback(BaseCallback):
    def __init__(self, render_env, log_dir):
        super().__init__()
        self.render_env = render_env
        self.log_dir = log_dir
        self.episode_counter = 0

    def _on_step(self) -> bool:
        obs = self.render_env.reset()
        done = False
        episode_rewards = []
        episode_actions = []
        while not done:
            action, _ = self.model.predict(obs)
            obs, reward, done, _ = self.render_env.step(action)
            self.render_env.render()
            episode_rewards.append(reward)
            episode_actions.append(action)

        self.save_episode_data(episode_rewards, episode_actions)
        self.episode_counter += 1
        return True

    def save_episode_data(self, rewards, actions):
        filename = os.path.join(
            self.log_dir, f"ep_{self.episode_counter:010d}.csv")
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['reward', 'action']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for reward, action in zip(rewards, actions):
                writer.writerow({'reward': reward, 'action': action})
                
class CustomMsPacmanEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward += info['score'] / 100.0
        reward -= info['lives'] * 10
        if 'level' in info:
            reward += info['level'] * 100
        obs = self._preprocess_observation(obs)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        return self._preprocess_observation(obs)

    def _preprocess_observation(self, obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
        obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
        return obs[:, :, None]

def objective(trial):
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-3)
    n_epochs = trial.suggest_int("n_epochs", 4, 20)

    model = PPO("CnnPolicy", env, verbose=1,
                learning_rate=learning_rate,
                n_epochs=n_epochs,
                tensorboard_log=tensorboard_log)

    checkpoint_callback = CheckpointCallback(
        save_freq=timesteps // 10, save_path=model_dir)

    if render:
        render_env = create_environment()
        render_env = DummyVecEnv([lambda: render_env])
        render_env = VecTransposeImage(render_env)
        render_callback = RenderCallback(render_env, log_dir)
    else:
        render_callback = None

    try:
        for step in range(0, timesteps, 1024):
            model.learn(total_timesteps=1024, callback=[
                        checkpoint_callback, render_callback])
    except KeyboardInterrupt:
        print("Interrupted! Saving model...")
        model.save(os.path.join(model_dir, model_prefix + "interrupted"))
        print("Model saved.")

    # Evaluate the model's performance for hyperparameter tuning
    validation_env = create_environment()
    validation_env = DummyVecEnv([lambda: validation_env])
    mean_reward, _ = evaluate_policy(model, validation_env, n_eval_episodes=eval_episodes)
    return mean_reward

def load_or_create_model(env):
    model_path = os.path.join(model_dir, model_file)
    if os.path.exists(model_path):
        print("Loading existing model")
        model = PPO.load(model_path, env)
    else:
        print("Creating new model")
        model = PPO("CnnPolicy", env, verbose=1,
                    learning_rate=learning_rate,
                    n_epochs=n_epochs,
                    )
    return model

def create_environment():
    env = retro.make(game=game_name)
    env = CustomMsPacmanEnv(env)
    return env

def train_agent(env, model):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    model = load_or_create_model(env)

    checkpoint_callback = CheckpointCallback(
        save_freq=timesteps // 10, save_path=model_dir)

    if render:
        render_env = create_environment()
        render_env = DummyVecEnv([lambda: render_env])
        render_env = VecTransposeImage(render_env)
        render_callback = RenderCallback(render_env, log_dir)
    else:
        render_callback = None

    for step in range(0, timesteps, 1024):
        model.learn(total_timesteps=1024, callback=[
                    checkpoint_callback, render_callback])

    model.save(os.path.join(model_dir, model_prefix + "final"))

def create_directories_if_not_exist(dir_names):
    for dir_name in dir_names:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

if __name__ == "__main__":
    create_directories_if_not_exist([model_dir, log_dir, tensorboard_log])
    # Train the agent
    env = create_environment()
    envs = [create_environment for _ in range(num_envs)]
    env = SubprocVecEnv(envs)

    # Optuna hyperparameter optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)  # Number of trials to run

    checkpoint_callback = CheckpointCallback(save_freq=timesteps // 10, save_path=model_dir)

    if render:
        render_env = create_environment()
        render_env = DummyVecEnv([lambda: render_env])
        render_env = VecTransposeImage(render_env)
        render_callback = RenderCallback(render_env, log_dir)
    else:
        render_callback = None

    # Optuna hyperparameter optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)  # Number of trials to run

    # Train the best model with the best hyperparameters
    best_params = study.best_params
    best_model = PPO("CnnPolicy", env, verbose=1,
                     learning_rate=best_params['learning_rate'],
                     n_epochs=best_params['n_epochs'],
                     tensorboard_log=tensorboard_log)
    train_agent(env, model=best_model)
