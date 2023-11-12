import torch.nn as nn
import torch
import numpy as np
from collections import deque
import random

from izombie_env import env as iz_env
from izombie_env import config

# model params
MODEL_NAME = "136,106,76"
N_HIDDEN_LAYER_NODES = 106


class QNetwork_DQN(nn.Module):
    def __init__(self, learning_rate=1e-3, device="cpu"):
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
        self.n_outputs = iz_env.IZenv().action_no
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
        device="cpu",
        replay_memory_size=100_000,
        min_replay_memory_size=10_000,
        gamma=0.99,
        start_epsilon=1.0,
        epsilon_decay=0.99975,
        min_epsilon=0.001,
        batch_size=32,
    ):
        self.min_replay_memory_size = min_replay_memory_size
        self.gamma = gamma
        self.epsilon = start_epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size

        self.model = QNetwork_DQN(device=device)
        self.target_model = QNetwork_DQN(device=device)
        self.replay_memory = deque(maxlen=replay_memory_size)

        self.step_count = 0

    def decide_action(self, state, valid_actions):
        if np.random.rand() < self.epsilon:
            return np.random.choice(valid_actions)
        else:
            with torch.no_grad():
                state_tensor = (
                    torch.FloatTensor(state).unsqueeze(0).to(self.model.device)
                )
                q_values = self.model.network(state_tensor).detach()

                valid_q_values = q_values[0, valid_actions]
                max_q_index = torch.argmax(valid_q_values).item()
                return valid_actions[max_q_index]

    def update_epsilon(self):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.min_epsilon, self.epsilon)

    def train(
        self, episodes, update_main_every=5, update_target_every=2_000, max_step=10_000
    ):
        self.model.train()  # Set the network to training mode
        env = iz_env.IZenv(max_step=max_step)

        for episode in range(episodes):
            print(f"episode = {episode}")
            state = env.reset()
            total_loss = 0
            done = False

            while not done:
                action = self.decide_action(state, env.get_valid_actions(state))

                reward, next_state, done = env.step(action)
                self.step_count += 1

                if (
                    len(self.replay_memory) > self.min_replay_memory_size
                    and self.step_count % update_main_every == 0
                ):
                    self.replay_memory.append((state, action, reward, next_state, done))

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

                state = next_state
                self.update_epsilon()

            if episode % update_target_every == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            print(f"Episode {episode}/{episodes}, Total Loss: {total_loss}")

        print("Training complete!")
