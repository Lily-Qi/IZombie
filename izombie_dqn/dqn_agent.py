from collections import deque
import random
from copy import deepcopy
import os
import datetime
import zipfile
# UCB
import math
# for profiling
from io import StringIO
import cProfile
import pstats
import functools

from torch import nn
import torch
import numpy as np

from izombie_env import config
from izombie_env.env import IZenv
from .epsilons import Epsilons
from .evaluate_agent import evaluate_agent

# model params
N_HIDDEN_LAYER_NODES = 100
MODEL_NAME = f"{config.STATE_SIZE},{N_HIDDEN_LAYER_NODES},{config.ACTION_SIZE}"


class DQNNetwork(nn.Module):
    def __init__(self, device, learning_rate, dropout_rate=0.5, num_lstm_layers=2):
        super(DQNNetwork, self).__init__()
        self.device = device
        self.lstm = nn.LSTM(input_size=config.STATE_SIZE,
                            hidden_size=N_HIDDEN_LAYER_NODES,
                            num_layers=num_lstm_layers,
                            batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.out = nn.Linear(N_HIDDEN_LAYER_NODES, config.ACTION_SIZE)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)

    def forward(self, x):
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.out(x[:, -1, :])
        return x


def profiled(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        s = StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
        ps.print_stats()
        print(s.getvalue())
        return result

    return wrapper


class DQNAgent:
    def __init__(
            self,
            # model
            model_dir="model",
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
        print(f"Using {device} device.")
        self.device = torch.device(device)
        self.model_dir = model_dir
        self.model_name = f"{model_name}_{get_timestamp()}"
        self.model = DQNNetwork(device=device, learning_rate=learning_rate).to(device)
        self.target_model = deepcopy(self.model)

        self.action_visit_count = {action: 0 for action in range(config.ACTION_SIZE)}
        self.action_values = {action: 0 for action in range(config.ACTION_SIZE)}

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
                "epsilons": Epsilons(
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

    def ucb_action_selection(self, state, valid_actions):
        total_visit_count = sum(self.action_visit_count[action] for action in valid_actions)
        log_term = math.log(total_visit_count + 1)

        ucb_values = []
        for action in valid_actions:
            action_visit_count = self.action_visit_count[action]
            if action_visit_count == 0:
                ucb_values.append(float('inf'))  # Encourage exploration of untried actions
            else:
                action_value = self.action_values[action]
                ucb_values.append(action_value + math.sqrt(2 * log_term / action_visit_count))

        best_action = valid_actions[np.argmax(ucb_values)]
        self.action_visit_count[best_action] += 1
        return best_action

    def update_action_values(self, action, reward):
        if self.action_visit_count[action] > 0:
            old_value = self.action_values[action]
            new_value = old_value + (reward - old_value) / self.action_visit_count[action]
            self.action_values[action] = new_value

    def decide_action(self, state, valid_actions):
        if np.random.rand() < self.epsilons.get():
            action = np.random.choice(valid_actions)
            self.action_visit_count[action] += 1
            return action
        return self.ucb_action_selection(state, valid_actions)

    def load_model(self, checkpoint):
        self.model_dir = (
            checkpoint["model_dir"] if "model_dir" in checkpoint else "model"
        )
        self.model_name = checkpoint["model_name"]

        self.model.network.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)
        self.model.optimizer.load_state_dict(checkpoint["optimizer"])
        for state in self.model.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

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

        from .threshold import Threshold

        epsilons = checkpoint["epsilons"]
        if isinstance(epsilons, Threshold):
            epsilons = Epsilons(
                seq_length=epsilons.seq_length,
                start_epsilon=epsilons.start_epsilon,
                end_epsilon=epsilons.end_epsilon,
            )
            epsilons.index = checkpoint["epsilon_index"]
        self.epsilons = epsilons

    def load_stats(self, checkpoint):
        self.step_count = checkpoint["step_count"]
        self.episode_count = checkpoint["episode_count"]
        self.rewards = checkpoint["rewards"]
        self.winning_suns = checkpoint["winning_suns"]
        self.losses = checkpoint["losses"]
        self.game_results = checkpoint["game_results"]
        self.steps = checkpoint["steps"]

    def load_checkpoint(self, filename):
        with zipfile.ZipFile(filename, "r") as zipf:
            zipf.extractall(os.path.dirname(filename))
        filename = filename[:-4]
        checkpoint = torch.load(filename, map_location=self.device)
        self.load_model(checkpoint)
        self.load_env(checkpoint)
        self.load_hyperparams(checkpoint)
        self.load_stats(checkpoint)
        os.remove(filename)

    def save_checkpoint(self, filename):
        assert not os.path.exists(filename)
        checkpoint = {
            "model_dir": self.model_dir,
            "model_name": self.model_name,
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
        with zipfile.ZipFile(f"{filename}.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(filename, os.path.basename(filename))
        os.remove(filename)

    def reset_epsilon(
            self, epsilon_length, start_epsilon, epsilon_interpolation, end_epsilon
    ):
        self.epsilons = Epsilons(
            seq_length=epsilon_length,
            start_epsilon=start_epsilon,
            interpolation=epsilon_interpolation,
            end_epsilon=end_epsilon,
        )

    def get_best_q_action(self, state, valid_actions):
        self.model.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.model.device)
            q_values = self.model(state_tensor).detach()
            valid_q_values = q_values[0, valid_actions]
            max_q_index = torch.argmax(valid_q_values).item()
        self.model.train()
        return valid_actions[max_q_index]

    def decide_action(self, state, valid_actions):
        if np.random.rand() < self.epsilons.get():
            return np.random.choice(valid_actions)
        return self.get_best_q_action(state, valid_actions)

    def set_to_trainig_mode(self):
        self.model.train()
        self.target_model.train()

    def set_to_eval_mode(self):
        self.model.eval()
        self.target_model.eval()

    def take_step(self, state, mask):
        action = self.decide_action(state, self.env.get_valid_actions(mask))
        reward, next_state, next_mask, game_status = self.env.step(action)
        self.update_action_values(action, reward)
        done = game_status != config.GameStatus.CONTINUE
        self.step_count += 1
        self.replay_memory.append((state, action, reward, next_state, next_mask, done))

        self.rewards.append(reward)
        return next_state, next_mask, game_status, done

    def update_main_model(self):
        minibatch = random.sample(self.replay_memory, self.batch_size)
        states, actions, rewards, next_states, next_masks, dones = zip(*minibatch)
        states = torch.FloatTensor(np.array(states)).unsqueeze(1).to(self.model.device)
        actions = torch.LongTensor(np.array(actions)).to(self.model.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.model.device)

        next_states = torch.FloatTensor(np.array(next_states)).unsqueeze(1).to(self.model.device)
        next_masks = torch.FloatTensor(np.array(next_masks)).to(self.model.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.model.device)
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_actions = self.model(next_states).argmax(dim=1)
        with torch.no_grad():
            next_q_values = self.target_model(next_states).gather(1, next_actions.unsqueeze(-1)).squeeze(-1)

        target_q_values = rewards + self.discount * next_q_values * (1 - dones)
        loss = nn.SmoothL1Loss()(current_q_values, target_q_values.detach())
        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.optimizer.step()
        self.losses.append(loss.item())

    def sync_target_model_with_main(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def train(
            self,
            episodes,
            update_main_every_n_steps,
            update_target_every_n_steps,
            evaluate_every_n_episodes,
            evaluate_test_size,
            save_checkpoint_every_n_episodes,
            stats_window,
    ):
        self.set_to_trainig_mode()
        prev_episode_count = self.episode_count

        start_time = datetime.datetime.now()
        for episode in range(episodes):
            self.env.reset()
            state, mask = self.env.get_state_and_mask()
            done = False

            while not done:
                next_state, next_mask, game_status, done = self.take_step(state, mask)

                if done:
                    self.game_results.append(game_status)
                    self.steps.append(self.env.step_count)
                    if game_status == config.GameStatus.WIN:
                        self.winning_suns.append(self.env.get_sun())

                if (
                        len(self.replay_memory) > self.min_replay_memory_size
                        and self.step_count % update_main_every_n_steps == 0
                ):
                    self.update_main_model()

                if self.step_count % update_target_every_n_steps == 0:
                    self.sync_target_model_with_main()

                state, mask = next_state, next_mask

            self.episode_count += 1

            if self.episode_count % 100 == 0:
                self.print_stats(
                    stats_window,
                    prev_episode_count + episode + 1,
                    prev_episode_count + episodes,
                    (datetime.datetime.now() - start_time).total_seconds()
                    / (episode + 1),
                )

            self.epsilons.next()

            if (
                    evaluate_every_n_episodes is not None
                    and self.episode_count % evaluate_every_n_episodes == 0
            ):
                create_folder_if_not_exist(self.get_model_folder())
                evaluate_agent(
                    self,
                    test_size=evaluate_test_size,
                    episode_count=self.episode_count,
                    output_file=f"{self.get_model_folder()}/eval.txt",
                )

            if (
                    save_checkpoint_every_n_episodes is not None
                    and self.episode_count % save_checkpoint_every_n_episodes == 0
            ):
                create_folder_if_not_exist(self.get_model_folder())
                filename = f"{self.get_model_folder()}/model_{self.episode_count}.pth"
                self.save_checkpoint(filename)
                print(f"Checkpoint has been saved to {filename}.")

    train_with_profiler = profiled(train)

    def print_stats(self, stats_window, curr_episode_count, total_episode_count, seconds_per_episode):
        win_rate = (
                sum(1 for res in self.game_results[-stats_window:] if res == config.GameStatus.WIN) * 100
                / min(stats_window, len(self.game_results))
        ) if self.game_results else 0
        mean_losses = np.mean(self.losses[-stats_window:]) if self.losses else 0
        mean_winning_suns = np.mean(self.winning_suns[-stats_window:]) if self.winning_suns else 0
        mean_steps = np.mean(self.steps[-stats_window:]) if self.steps else 0

        print(
            f"Ep {curr_episode_count}/{total_episode_count} "
            f"ε {self.epsilons.get():.2f} "
            f"Mean losses {mean_losses:.2f} "
            f"Mean winning sun {mean_winning_suns:.2f} "
            f"Mean steps {mean_steps:.2f} "
            f"Win {win_rate:.2f}% "
            f"{int(seconds_per_episode * 10_000)}s/10k ep"
        )

    def get_model_folder(self):
        return f"{self.model_dir}/{self.model_name}"


def create_folder_if_not_exist(folder_name):
    current_directory = os.getcwd()
    folder_path = os.path.join(current_directory, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def get_timestamp():
    return datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
