import torch.nn as nn
import torch
import numpy as np
from collections import deque
import random

from izombie_env import config
from .threshold import Threshold

# model params
MODEL_NAME = "136,106,76"
N_HIDDEN_LAYER_NODES = 106


class QNetwork_DQN(nn.Module):
    def __init__(self, device, learning_rate=1e-3):
        super(QNetwork_DQN, self).__init__()
        self.device = device

        self.n_inputs = (
            config.N_LANES * config.P_LANE_LENGTH  # plant hp
            + config.N_LANES * config.P_LANE_LENGTH  # plant type
            + config.N_LANES * config.LANE_LENGTH  # zombie hp
            + config.N_LANES * config.LANE_LENGTH  # zombie type
            + 1  # sun
            + 5  # brain status
        )
        self.n_outputs = config.ACTION_NO
        self.actions = np.arange(self.n_outputs)
        self.learning_rate = learning_rate

        # Set up network
        self.network = nn.Sequential(
            nn.Linear(self.n_inputs, N_HIDDEN_LAYER_NODES, bias=True),
            nn.LeakyReLU(),
            nn.Linear(N_HIDDEN_LAYER_NODES, self.n_outputs, bias=True),
        )

        # Set to GPU if cuda is specified
        if self.device == "cuda":
            self.network.cuda()

        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate
        )


class DQNAgent:
    def __init__(
        self,
        env,
        device="cpu",
        replay_memory_size=50_000,
        min_replay_memory_size=1_000,
        gamma=0.99,
        batch_size=64,
        epsilon_length=10_000,
        start_epsilon=1.0,
        epsilon_interpolation="exponential",
        end_epsilon=0.05,
    ):
        # model params
        self.env = env
        self.min_replay_memory_size = min_replay_memory_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilons = Threshold(
            seq_length=epsilon_length,
            start_epsilon=start_epsilon,
            interpolation=epsilon_interpolation,
            end_epsilon=end_epsilon,
        )

        # models and replay memory
        self.model = QNetwork_DQN(device=device)
        self.target_model = QNetwork_DQN(device=device)
        self.replay_memory = deque(maxlen=replay_memory_size)

        # stats
        self.step_count = 0
        self.episode_count = 0
        self.rewards = []
        self.winning_suns = []
        self.losses = []
        self.game_results = []
        self.steps = []

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
        else:
            return self.get_best_q_action(state, valid_actions)

    def train(
        self, episodes, update_main_every=1, update_target_every=1, stats_window=100
    ):
        self.model.train()  # Set the network to training mode

        for episode in range(episodes):
            state = self.env.reset()
            total_loss = 0
            done = False

            while not done:
                action = self.decide_action(
                    state=state, valid_actions=self.env.get_valid_actions()
                )

                reward, next_state, game_status = self.env.step(action)
                done = game_status != config.GameStatus.CONTINUE
                self.step_count += 1
                self.replay_memory.append((state, action, reward, next_state, done))

                self.rewards.append(reward)
                if done:
                    self.game_results.append(game_status)
                    if game_status == config.GameStatus.WIN:
                        self.winning_suns.append(self.env._get_sun())

                if (
                    len(self.replay_memory) > self.min_replay_memory_size
                    and self.step_count % update_main_every == 0
                ):
                    # Sample a batch of experiences from replay memory
                    minibatch = random.sample(self.replay_memory, self.batch_size)
                    states, actions, rewards, next_states, dones = zip(*minibatch)

                    # Convert to tensors
                    states = torch.FloatTensor(np.array(states)).to(self.model.device)
                    actions = torch.LongTensor(np.array(actions)).to(self.model.device)
                    rewards = torch.FloatTensor(np.array(rewards)).to(self.model.device)
                    next_states = torch.FloatTensor(np.array(next_states)).to(
                        self.model.device
                    )
                    dones = torch.FloatTensor(np.array(dones)).to(self.model.device)

                    # Compute current Q values
                    current_q_values = (
                        self.model.network(states)
                        .gather(1, actions.unsqueeze(1))
                        .squeeze(1)
                    )

                    # Compute next Q values using target network
                    next_q_values = self.target_model.network(next_states).max(1)[0]
                    target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

                    # Compute loss
                    loss = nn.MSELoss()(current_q_values, target_q_values.detach())

                    # Optimize the model
                    self.model.optimizer.zero_grad()
                    loss.backward()
                    self.model.optimizer.step()

                    total_loss += loss.item()
                    self.losses.append(loss.item())

                if self.step_count % update_target_every == 0:
                    self.target_model.load_state_dict(self.model.state_dict())

                state = next_state

            self.steps.append(self.env._step_count)

            self.episode_count += 1

            if self.episode_count % 100 == 0:
                print()
            game_results_effective_len = min(stats_window, len(self.game_results))
            print(
                "\rEp {:d}/{:d}\t ε {:.2f}\t Mean rewards {:.2f}\t Mean losses {:.2f}\t Mean winning sun {:.2f} Mean steps {:.2f} Win {:.2f}%".format(
                    episode,
                    episodes,
                    self.get_epsilon(),
                    np.mean(self.rewards[-stats_window:]),
                    np.mean(self.losses[-stats_window:]),
                    np.mean(self.winning_suns[-stats_window:]),
                    np.mean(self.steps[-stats_window:]),
                    sum(
                        1
                        for res in self.game_results[-stats_window:]
                        if res == config.GameStatus.WIN
                    )
                    * 100
                    / game_results_effective_len,
                ),
                end="",
            )

        model_save_dir = f"{MODEL_NAME}.pth"
        print(f"Training complete! Model has been saved to {model_save_dir}.")
        torch.save(self.model.state_dict(), model_save_dir)
