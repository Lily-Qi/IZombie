import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import numpy as np
from typing import Dict, Tuple
import datetime
import os
import zipfile

from .noisy_layer import NoisyLinear
from .replay_buffer import PrioritizedReplayBuffer, ReplayBuffer

from izombie_env2.env import IZenv
from izombie_env2.config import ACTION_SIZE, STATE_SIZE
from izombie_env2.config import GameStatus


class Network(nn.Module):
    def __init__(
        self, in_dim: int, out_dim: int, atom_size: int, support: torch.Tensor
    ):
        """Initialization."""
        super(Network, self).__init__()

        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size

        # set common feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
        )

        # set advantage layer
        self.advantage_hidden_layer = NoisyLinear(128, 128)
        self.advantage_layer = NoisyLinear(128, out_dim * atom_size)

        # set value layer
        self.value_hidden_layer = NoisyLinear(128, 128)
        self.value_layer = NoisyLinear(128, atom_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)

        return q

    def dist(self, x: torch.Tensor) -> torch.Tensor:
        """Get distribution for atoms."""
        feature = self.feature_layer(x)
        adv_hid = F.relu(self.advantage_hidden_layer(feature))
        val_hid = F.relu(self.value_hidden_layer(feature))

        advantage = self.advantage_layer(adv_hid).view(-1, self.out_dim, self.atom_size)
        value = self.value_layer(val_hid).view(-1, 1, self.atom_size)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)

        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans

        return dist

    def reset_noise(self):
        """Reset all noisy layers."""
        self.advantage_hidden_layer.reset_noise()
        self.advantage_layer.reset_noise()
        self.value_hidden_layer.reset_noise()
        self.value_layer.reset_noise()


class DQNAgent:
    """DQN Agent interacting with environment.

    Attribute:
        env (IZenv): izombie env
        memory (PrioritizedReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        gamma (float): discount factor
        dqn (Network): model to train and select actions
        dqn_target (Network): target model to update
        optimizer (torch.optim): optimizer for training dqn
        transition (list): transition information including
                           state, action, reward, next_state, done
        v_min (float): min value of support
        v_max (float): max value of support
        atom_size (int): the unit number of support
        support (torch.Tensor): support for categorical dqn
        use_n_step (bool): whether to use n_step memory
        n_step (int): step number to calculate n-step td error
        memory_n (ReplayBuffer): n-step replay buffer
    """

    def __init__(
        self,
        seed: int,
        env: IZenv,
        model_name: str,
        device: str,
        memory_size: int,
        batch_size: int,
        gamma: float = 0.99,
        # PER parameters
        alpha: float = 0.2,
        beta: float = 0.6,
        prior_eps: float = 1e-6,
        # Categorical DQN parameters
        v_min: float = 0.0,
        v_max: float = 200.0,
        atom_size: int = 51,
        # N-step Learning
        n_step: int = 3,
    ):
        """Initialization.

        Args:
            env (gym.Env): openAI Gym environment
            memory_size (int): length of memory
            batch_size (int): batch size for sampling
            lr (float): learning rate
            gamma (float): discount factor
            alpha (float): determines how much prioritization is used
            beta (float): determines how much importance sampling is used
            prior_eps (float): guarantees every transition can be sampled
            v_min (float): min value of support
            v_max (float): max value of support
            atom_size (int): the unit number of support
            n_step (int): step number to calculate n-step td error
        """
        obs_dim = STATE_SIZE
        action_dim = ACTION_SIZE

        self.env = env
        self.model_name = model_name
        self.batch_size = batch_size
        self.seed = seed
        self.gamma = gamma
        # NoisyNet: All attributes related to epsilon are removed

        # device: cpu / gpu
        self.device = torch.device(device)
        print(f"Using {self.device} device.")

        # PER
        # memory for 1-step Learning
        self.beta = beta
        self.prior_eps = prior_eps
        self.memory = PrioritizedReplayBuffer(
            obs_dim, memory_size, batch_size, alpha=alpha, gamma=gamma
        )

        # memory for N-step Learning
        self.use_n_step = True if n_step > 1 else False
        if self.use_n_step:
            self.n_step = n_step
            self.memory_n = ReplayBuffer(
                obs_dim, memory_size, batch_size, n_step=n_step, gamma=gamma
            )

        # Categorical DQN parameters
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.support = torch.linspace(self.v_min, self.v_max, self.atom_size).to(
            self.device
        )

        # networks: dqn, dqn_target
        self.dqn = Network(obs_dim, action_dim, self.atom_size, self.support).to(
            self.device
        )
        self.dqn_target = Network(obs_dim, action_dim, self.atom_size, self.support).to(
            self.device
        )
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters())

        # transition to store in memory
        self.transition = list()

        # mode: train / test
        self.is_test = False

        # stats
        self.winning_suns = []
        self.losses = []
        self.game_results = []
        self.steps = []

    def get_best_q_action(self, state, mask):
        """Select an action from the input state."""
        # NoisyNet: no epsilon greedy action selection
        with torch.no_grad():
            valid_actions = self.env.get_valid_actions(mask)
            q_values = self.dqn(
                torch.FloatTensor(state).unsqueeze(0).to(self.device)
            ).detach()
            valid_q_values = q_values[0, valid_actions]
            max_q_index = torch.argmax(valid_q_values).item()
            selected_action = valid_actions[max_q_index]

            if not self.is_test:
                self.transition = [state, selected_action]

            return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        reward, next_state, next_mask, game_status = self.env.step(action)
        done = game_status != GameStatus.CONTINUE

        if not self.is_test:
            self.transition += [reward, next_state, done]

            # N-step transition
            if self.use_n_step:
                one_step_transition = self.memory_n.store(*self.transition)
            # 1-step transition
            else:
                one_step_transition = self.transition

            # add a single step transition
            if one_step_transition:
                self.memory.store(*one_step_transition)

        return next_state, next_mask, reward, game_status, done

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        # PER needs beta to calculate weights
        samples = self.memory.sample_batch(self.beta)
        weights = torch.FloatTensor(samples["weights"].reshape(-1, 1)).to(self.device)
        indices = samples["indices"]

        # 1-step Learning loss
        elementwise_loss = self._compute_dqn_loss(samples, self.gamma)

        # PER: importance sampling before average
        loss = torch.mean(elementwise_loss * weights)

        # N-step Learning loss
        # we are gonna combine 1-step loss and n-step loss so as to
        # prevent high-variance. The original rainbow employs n-step loss only.
        if self.use_n_step:
            gamma = self.gamma**self.n_step
            samples = self.memory_n.sample_batch_from_idxs(indices)
            elementwise_loss_n_loss = self._compute_dqn_loss(samples, gamma)
            elementwise_loss += elementwise_loss_n_loss

            # PER: importance sampling before average
            loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.dqn.parameters(), 10.0)
        self.optimizer.step()

        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)

        # NoisyNet: reset noise
        self.dqn.reset_noise()
        self.dqn_target.reset_noise()

        return loss.item()

    def train(
        self,
        num_steps: int,
        stats_window: int = 1_000,
        print_stats_every_n_steps=30_000,
        update_main_every_n_steps=1,
        update_target_every_n_steps=2000,
        save_every_n_steps=None,
    ):
        """Train the agent."""
        self.is_test = False
        self.set_to_training_mode()

        state, mask = self.env.reset()
        update_cnt = 0
        scores = []
        score = 0

        start_time = datetime.datetime.now()

        for step_idx in range(1, num_steps + 1):
            action = self.get_best_q_action(state, mask)
            next_state, next_mask, reward, game_status, done = self.step(action)

            state, mask = next_state, next_mask
            score += reward

            # NoisyNet: removed decrease of epsilon

            # PER: increase beta
            fraction = min(step_idx / num_steps, 1.0)
            self.beta = self.beta + fraction * (1.0 - self.beta)

            # if episode ends
            if done:
                self.game_results.append(game_status)
                self.steps.append(self.env.step_count)
                if game_status == GameStatus.WIN:
                    self.winning_suns.append(self.env.get_sun())
                state, mask = self.env.reset()
                scores.append(score)
                score = 0

            # if training is ready
            if len(self.memory) >= self.batch_size:
                loss = self.update_model()
                self.losses.append(loss)
                update_cnt += 1

                # if hard update is needed
                if update_cnt % update_target_every_n_steps == 0:
                    self._sync_target_with_main()

            if step_idx % print_stats_every_n_steps == 0:
                self.print_stats(stats_window, step_idx, num_steps, start_time)

            if (
                save_every_n_steps is not None and step_idx % save_every_n_steps == 0
            ) or step_idx == num_steps:
                self.save(f"model/{self.model_name}_{step_idx}_{get_timestamp()}.pth")

    def _compute_dqn_loss(
        self, samples: Dict[str, np.ndarray], gamma: float
    ) -> torch.Tensor:
        """Return categorical dqn loss."""
        state = torch.FloatTensor(samples["obs"]).to(self.device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(self.device)
        action = torch.LongTensor(samples["acts"]).to(self.device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(self.device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(self.device)

        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            # Double DQN
            next_action = self.dqn(next_state).argmax(1)
            next_dist = self.dqn_target.dist(next_state)
            next_dist = next_dist[range(self.batch_size), next_action]

            t_z = reward + (1 - done) * gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (self.batch_size - 1) * self.atom_size, self.batch_size
                )
                .long()
                .unsqueeze(1)
                .expand(self.batch_size, self.atom_size)
                .to(self.device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.dqn.dist(state)
        log_p = torch.log(dist[range(self.batch_size), action])
        elementwise_loss = -(proj_dist * log_p).sum(1)

        return elementwise_loss

    def _sync_target_with_main(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())

    def print_stats(self, stats_window, curr_step, total_step, start_time):
        win_rate = (
            sum(1 for res in self.game_results[-stats_window:] if res == GameStatus.WIN)
            * 100
            / min(stats_window, len(self.game_results))
        )
        elasped_seconds = (datetime.datetime.now() - start_time).total_seconds()
        print(
            f"Sp {curr_step}/{total_step} "
            f"Mean losses {np.mean(self.losses[-stats_window:]):.2f} "
            f"Mean winning sun {np.mean(self.winning_suns[-stats_window:]):.2f} "
            f"Mean steps {np.mean(self.steps[-stats_window:]):.2f} "
            f"Win {win_rate:.2f}% "
            f"{elasped_seconds / curr_step * 10_000:.2f}s/10k steps"
        )

    def set_to_training_mode(self):
        self.dqn.train()

    def set_to_eval_mode(self):
        self.dqn.eval()

    def save(self, filename):
        assert not os.path.exists(filename)
        torch.save(self.dqn.state_dict(), filename)
        with zipfile.ZipFile(f"{filename}.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(filename, os.path.basename(filename))
        os.remove(filename)

    def load(self, filename, device="cpu"):
        with zipfile.ZipFile(filename, "r") as zipf:
            zipf.extractall(os.path.dirname(filename))
        filename = filename[:-4]
        state_dict = torch.load(filename, map_location=device)
        self.dqn.load_state_dict(state_dict)
        self.dqn_target.load_state_dict(state_dict)
        os.remove(filename)


def get_timestamp():
    return datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
