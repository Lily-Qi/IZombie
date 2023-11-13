from collections import deque
import random
from copy import deepcopy
import os

from torch import nn
import torch
import numpy as np

from izombie_env import config
from izombie_env.simple_env import IZenv
from .threshold import Threshold
from .evaluate_agent import evaluate_agent

# model params
MODEL_NAME = "136,106,76"
N_HIDDEN_LAYER_NODES = 106


class DQNNetwork(nn.Module):
    def __init__(self, device, learning_rate):
        super(DQNNetwork, self).__init__()
        self.device = device

        self.n_inputs = (
            config.N_LANES * config.P_LANE_LENGTH  # plant hp
            + config.N_LANES * config.P_LANE_LENGTH  # plant type
            + config.N_LANES * config.LANE_LENGTH  # zombie hp
            + config.N_LANES * config.LANE_LENGTH  # zombie type
            + 1  # sun
            + 5  # brain status
        )

        # Set up network
        self.network = nn.Sequential(
            nn.Linear(self.n_inputs, N_HIDDEN_LAYER_NODES, bias=True),
            nn.LeakyReLU(),
            nn.Linear(N_HIDDEN_LAYER_NODES, config.ACTION_NO, bias=True),
        )

        # Set to GPU if cuda is specified
        if self.device == "cuda":
            self.network.cuda()

        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=learning_rate
        )

    def forward(self, x):
        # Pass the input through the network layers
        return self.network(x)


class DQNAgent:
    def __init__(
        self,
        # model
        model_name=MODEL_NAME,
        device="cuda" if torch.cuda.is_available() else "cpu",
        learning_rate=1e-3,
        # env
        step_length=50,  # skip how many game ticks per step (1 game tick = 1cs)
        max_step_per_ep=300,  # max number of steps allowed in one episode
        fix_rand=False,  # always generate the same puzzle
        # hyperparams
        replay_memory_size=50_000,
        min_replay_memory_size=1_000,
        discount=0.99,
        batch_size=64,
        epsilon_length=10_000,
        start_epsilon=1.0,
        epsilon_interpolation="exponential",
        end_epsilon=0.05,
    ):
        self.model_name = model_name
        self.model = DQNNetwork(device=device, learning_rate=learning_rate)
        self.target_model = deepcopy(self.model)

        self.load_env(
            {
                "step_length": step_length,
                "max_step": max_step_per_ep,
                "fix_rand": fix_rand,
            }
        )
        self.load_hyperparams(
            {
                "replay_memory": deque(maxlen=replay_memory_size),
                "min_replay_memory_size": min_replay_memory_size,
                "discount": discount,
                "batch_size": batch_size,
                "epsilons": Threshold(
                    seq_length=epsilon_length,
                    start_epsilon=start_epsilon,
                    interpolation=epsilon_interpolation,
                    end_epsilon=end_epsilon,
                ),
            }
        )
        self.load_stats(
            {
                "step_count": 0,
                "episode_count": 0,
                "rewards": [],
                "winning_suns": [],
                "losses": [],
                "game_results": [],
                "steps": [],
            }
        )

    def load_model(self, checkpoint):
        self.model_name = checkpoint["model_name"]

        device = checkpoint["device"]
        self.model.network.load_state_dict(checkpoint["state_dict"])
        self.model.to(device)
        self.model.optimizer.load_state_dict(checkpoint["optimizer"])
        for state in self.model.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        self.target_model = deepcopy(self.model)

    def load_env(self, checkpoint):
        self.env = IZenv(
            step_length=checkpoint["step_length"],
            max_step=checkpoint["max_step"],
            fix_rand=checkpoint["fix_rand"],
        )

    def load_hyperparams(self, checkpoint):
        self.replay_memory = checkpoint["replay_memory"]
        self.min_replay_memory_size = checkpoint["min_replay_memory_size"]
        self.discount = checkpoint["discount"]
        self.batch_size = checkpoint["batch_size"]
        self.epsilons = checkpoint["epsilons"]

    def load_stats(self, checkpoint):
        self.step_count = checkpoint["step_count"]
        self.episode_count = checkpoint["episode_count"]
        self.rewards = checkpoint["rewards"]
        self.winning_suns = checkpoint["winning_suns"]
        self.losses = checkpoint["losses"]
        self.game_results = checkpoint["game_results"]
        self.steps = checkpoint["steps"]

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        self.load_model(checkpoint)
        self.load_env(checkpoint)
        self.load_hyperparams(checkpoint)
        self.load_stats(checkpoint)

    def save_checkpoint(self, filename):
        checkpoint = {
            "model_name": self.model_name,
            "device": self.model.device,
            "state_dict": self.model.network.state_dict(),
            "optimizer": self.model.optimizer.state_dict(),
            "step_length": self.env.step_length,
            "max_step": self.env.max_step,
            "fix_rand": self.env.fix_rand,
            "replay_memory": self.replay_memory,
            "min_replay_memory_size": self.min_replay_memory_size,
            "discount": self.discount,
            "batch_size": self.batch_size,
            "epsilons": self.epsilons,
            "step_count": self.step_count,
            "episode_count": self.episode_count,
            "rewards": self.rewards,
            "winning_suns": self.winning_suns,
            "losses": self.losses,
            "game_results": self.game_results,
            "steps": self.steps,
        }
        torch.save(checkpoint, filename)

    def get_epsilon(self):
        return self.epsilons.epsilon(self.episode_count)

    def get_best_q_action(self, state, valid_actions):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.model.device)
            q_values = self.model.network(state_tensor).detach()
            valid_q_values = q_values[0, valid_actions]
            max_q_index = torch.argmax(valid_q_values).item()
            return valid_actions[max_q_index]

    def decide_action(self, state, valid_actions):
        if np.random.rand() < self.get_epsilon():
            return np.random.choice(valid_actions)
        return self.get_best_q_action(state, valid_actions)

    def set_to_trainig_mode(self):
        self.model.train()
        self.target_model.train()

    def set_to_eval_mode(self):
        self.model.eval()
        self.target_model.eval()

    def take_step(self, state):
        action = self.decide_action(state, self.env.get_valid_actions())

        reward, next_state, game_status = self.env.step(action)
        done = game_status != config.GameStatus.CONTINUE
        self.step_count += 1
        self.replay_memory.append((state, action, reward, next_state, done))

        self.rewards.append(reward)
        return next_state, game_status, done

    def update_main_model(self):
        # Sample a batch of experiences from replay memory
        minibatch = random.sample(self.replay_memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.model.device)
        actions = torch.LongTensor(np.array(actions)).to(self.model.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.model.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.model.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.model.device)

        # Compute current Q values
        current_q_values = (
            self.model.network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        )

        # Compute next Q values using target network
        next_q_values = self.target_model.network(next_states).max(1)[0]
        target_q_values = rewards + self.discount * next_q_values * (1 - dones)

        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values.detach())

        # Optimize the model
        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.optimizer.step()

        self.losses.append(loss.item())

    def train(
        self,
        episodes,
        update_main_every_n_steps=32,
        update_target_every_n_steps=2000,
        eval_every_n_episodes=500,
        stats_window=1000,
    ):
        self.set_to_trainig_mode()
        prev_ep_count = self.episode_count

        for episode in range(episodes):
            state = self.env.reset()
            done = False

            while not done:
                next_state, game_status, done = self.take_step(state)

                if done:
                    self.game_results.append(game_status)
                    if game_status == config.GameStatus.WIN:
                        self.winning_suns.append(self.env.get_sun())

                if (
                    len(self.replay_memory) > self.min_replay_memory_size
                    and self.step_count % update_main_every_n_steps == 0
                ):
                    self.update_main_model()

                if self.step_count % update_target_every_n_steps == 0:
                    self.target_model.load_state_dict(self.model.state_dict())

                state = next_state

            self.steps.append(self.env.step_count)

            self.episode_count += 1

            if self.episode_count % eval_every_n_episodes == 0:
                evaluate_agent(self)

            if self.episode_count % 100 == 0:
                print()
            win_rate = (
                sum(
                    1
                    for res in self.game_results[-stats_window:]
                    if res == config.GameStatus.WIN
                )
                * 100
                / min(stats_window, len(self.game_results))
            )
            print(
                f"\rEp {prev_ep_count + episode}/{prev_ep_count + episodes} "
                f"ε {self.get_epsilon():.2f} "
                f"Mean losses {np.mean(self.losses[-stats_window:]):.2f} "
                f"Mean winning sun {np.mean(self.winning_suns[-stats_window:]):.2f} "
                f"Mean steps {np.mean(self.steps[-stats_window:]):.2f} "
                f" Win {win_rate:.2f}%",
                end="",
            )

        create_folder_if_not_exist("model")
        filename = f"model/{self.model_name}.pth"
        self.save_checkpoint(filename)
        print(f"Training complete! Model has been saved to {filename}.")


def create_folder_if_not_exist(folder_name):
    current_directory = os.getcwd()
    folder_path = os.path.join(current_directory, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
