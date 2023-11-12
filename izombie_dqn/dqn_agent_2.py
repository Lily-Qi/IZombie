import torch.nn as nn
import torch
from izombie_env import config
from copy import deepcopy
from collections import namedtuple, deque
import numpy as np
from .evaluate_agent import evaluate
from .threshold import Threshold
from izombie_env import env as iz_env

N_HIDDEN_LAYDER_NODE = 106


class QNetwork_DQN(nn.Module):
    def __init__(self, env, learning_rate=1e-3, device="cpu"):
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
        self.n_outputs = env.action_no
        self.actions = np.arange(env.action_no)
        self.learning_rate = learning_rate
        self._grid_size = config.N_LANES * config.LANE_LENGTH

        # Set up network
        self.network = nn.Sequential(
            nn.Linear(self.n_inputs, N_HIDDEN_LAYDER_NODE, bias=True),
            nn.LeakyReLU(),
            # This final layer produces Q values for each action given current env state
            nn.Linear(N_HIDDEN_LAYDER_NODE, self.n_outputs, bias=True),
        )

        # Set to GPU if cuda is specified
        if self.device == "cuda":
            self.network.cuda()

        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate
        )

    # Can be moved to somewhere else
    def decide_action(self, state, mask, epsilon):
        if np.random.random() < epsilon:
            action = np.random.choice(self.actions[mask])
        else:
            action = self.get_greedy_action(state, mask)
        return action

    # Can be moved to somewhere else
    # Get action with best Q
    def get_greedy_action(self, state, mask):
        qvals = self.get_qvals(state)
        qvals[np.logical_not(mask)] = qvals.min()
        return torch.max(qvals, dim=-1)[1].item()

    # Convert observation into shape of input, and use the network to get q values
    def get_qvals(self, state):
        if type(state) is tuple:
            state = np.array([np.ravel(s) for s in state])
            state_t = torch.FloatTensor(state).to(device=self.device)
        else:
            state_t = torch.FloatTensor(state).to(device=self.device)
        return self.network(state_t)


class DQNAgent:
    # initialize threshold, which is a sequence of epsilon used to determine
    # if we randomly choose an action. The epsilons are decreasing, which means
    # that when the training goes on, the probability that the agents randomly
    # choose an action will decrease, it will focus more on maximizing the qvalue.

    # Also initialize the player and the window (if it's set to 100, which means
    # the code will calculate the average rewards and iterations of the last 100 episodes.).
    # And a reward threshold, if it is exceeded, then end training.
    def __init__(self, env, network, buffer, batch_size=32):
        self._grid_size = config.N_LANES * config.LANE_LENGTH
        self.env = env
        self.network = network
        self.target_network = deepcopy(network)
        self.buffer = buffer
        self.threshold = Threshold(
            seq_length=100000,
            start_epsilon=1.0,
            interpolation="exponential",
            end_epsilon=0.05,
        )
        self.epsilon = 0
        self.batch_size = batch_size
        self.window = 100
        self.reward_threshold = 30000
        self.initialize()
        self.player = PlayerQ_DQN(env=env, render=False)

    # when the mode is ‘explore’, randomly choose actions to fill the buffer
    # until we meet the expectation number of experiences in the buffer.
    # Else, use decide_action to choose the action based on the state.
    # Do the action on the state and get the corresponding rewards and
    # resulted state. Store [previous state, action, reward, result state,
    # done signal] in the buffer. Update the total rewards. If done, document
    # the end of the play and reset the observation.

    # Train: first populate buffer to a desired amount in explore mode,
    # then begin the training loop. In each loop, if the game doesn’t end,
    # get the epsilon, take a step, update the network, check if current
    # game ends. If end, record the training data, if necessary, do the
    # evaluation based on the current trained agent, and check if the training ends.
    def take_step(self, mode="train"):
        mask = np.array(self.env.get_action_mask(self.env.get_state()))
        if mode == "explore":
            if np.random.random() < 0.5:
                action = 0  # Do nothing
            else:
                action = np.random.choice(np.arange(self.env.action_no)[mask])
        else:
            action = self.network.decide_action(self.s_0, mask, epsilon=self.epsilon)
            self.step_count += 1
        r, s_1, done = self.env.step(action)
        s_1 = self._transform_observation(s_1)
        self.rewards += r
        self.buffer.append(self.s_0, action, r, done, s_1)
        self.s_0 = s_1.copy()
        if done:
            if mode != "explore":  # We document the end of the play
                self.training_iterations.append(
                    min(400, self.env._step_count)
                )
            self.s_0 = self._transform_observation(self.env.reset())
        return done

    # Implement DQN training algorithm
    def train(
        self,
        gamma=0.99,
        max_episodes=100000,
        network_update_frequency=32,
        network_sync_frequency=2000,
        evaluate_frequency=500,
        evaluate_n_iter=1000,
    ):
        self.gamma = gamma
        # Populate replay buffer
        while self.buffer.burn_in_capacity() < 1:
            done = self.take_step(mode="explore")
        epsilon_index = 0
        training = True
        self.s_0 = self._transform_observation(self.env.reset())

        while training:
            self.rewards = 0
            done = False
            while done == False:
                self.epsilon = self.threshold.epsilon(epsilon_index)
                done = self.take_step(mode="train")
                # Update network
                if self.step_count % network_update_frequency == 0:
                    self.update()
                # Sync networks
                if self.step_count % network_sync_frequency == 0:
                    self.target_network.load_state_dict(self.network.state_dict())
                    self.sync_eps.append(epsilon_index)

                if done:
                    epsilon_index += 1
                    self.training_rewards.append(self.rewards)
                    self.training_loss.append(np.mean(self.update_loss))
                    self.update_loss = []
                    mean_rewards = np.mean(self.training_rewards[-self.window :])
                    self.mean_training_rewards.append(mean_rewards)

                    mean_iteration = np.mean(self.training_iterations[-self.window :])
                    self.mean_training_iterations.append(mean_iteration)
                    print(
                        "\rEpisode {:d} Mean Rewards {:.2f}\t\t Mean Iterations {:.2f}\t\t".format(
                            epsilon_index, mean_rewards, mean_iteration
                        ),
                        end="",
                    )

                    if epsilon_index >= max_episodes:
                        training = False
                        print("\nEpisode limit reached.")
                        break
                    if mean_rewards >= self.reward_threshold:
                        training = False
                        print(
                            "\nEnvironment solved in {} episodes!".format(epsilon_index)
                        )
                        break
                    if (epsilon_index % evaluate_frequency) == evaluate_frequency - 1:
                        avg_score, avg_iter = evaluate(
                            self.player,
                            self.network,
                            n_iter=evaluate_n_iter,
                            verbose=False,
                        )
                        self.real_iterations.append(avg_iter)
                        self.real_rewards.append(avg_score)

    def calculate_loss(self, batch):
        states, actions, rewards, dones, next_states = [i for i in batch]
        rewards_t = (
            torch.FloatTensor(rewards).to(device=self.network.device).reshape(-1, 1)
        )
        actions_t = (
            torch.LongTensor(np.array(actions))
            .reshape(-1, 1)
            .to(device=self.network.device)
        )
        dones_t = torch.BoolTensor(dones).to(device=self.network.device)

        qvals = torch.gather(
            self.network.get_qvals(states), 1, actions_t
        )  # The selected action already respects the mask

        #################################################################
        # DDQN Update
        next_masks = np.array([self.env.get_action_mask(s) for s in next_states])
        qvals_next_pred = self.network.get_qvals(next_states)
        qvals_next_pred[np.logical_not(next_masks)] = qvals_next_pred.min()
        next_actions = torch.max(qvals_next_pred, dim=-1)[1]
        next_actions_t = next_actions.reshape(-1, 1).to(device=self.network.device)
        target_qvals = self.network.get_qvals(next_states)
        qvals_next = torch.gather(target_qvals, 1, next_actions_t).detach()
        #################################################################
        qvals_next[dones_t] = 0  # Zero-out terminal states
        expected_qvals = self.gamma * qvals_next + rewards_t
        loss = nn.MSELoss()(qvals, expected_qvals)
        return loss

    def update(self):
        self.network.optimizer.zero_grad()
        batch = self.buffer.sample_batch(batch_size=self.batch_size)
        loss = self.calculate_loss(batch)
        loss.backward()
        self.network.optimizer.step()
        if self.network.device == "cuda":
            self.update_loss.append(loss.detach().cpu().numpy())
        else:
            self.update_loss.append(loss.detach().numpy())

    def _transform_observation(self, observation):
        return observation.astype(np.float64)

    def _save_training_data(self, nn_name):
        np.save(nn_name + "_rewards", self.training_rewards)
        np.save(nn_name + "_iterations", self.training_iterations)
        np.save(nn_name + "_real_rewards", self.real_rewards)
        np.save(nn_name + "_real_iterations", self.real_iterations)
        torch.save(self.training_loss, nn_name + "_loss")

    def initialize(self):
        self.training_rewards = []
        self.training_loss = []
        self.training_iterations = []
        self.real_rewards = []
        self.real_iterations = []
        self.update_loss = []
        self.mean_training_rewards = []
        self.mean_training_iterations = []
        self.sync_eps = []
        self.rewards = 0
        self.step_count = 0
        self.s_0 = self._transform_observation(self.env.reset())


class experienceReplayBuffer_DQN:
    def __init__(self, memory_size=50000, burn_in=10000):
        self.memory_size = memory_size
        self.burn_in = burn_in
        self.Buffer = namedtuple(
            "Buffer", field_names=["state", "action", "reward", "done", "next_state"]
        )
        self.replay_memory = deque(maxlen=memory_size)

    def sample_batch(self, batch_size=32):
        samples = np.random.choice(len(self.replay_memory), batch_size, replace=False)
        # Use asterisk operator to unpack deque
        batch = zip(*[self.replay_memory[i] for i in samples])
        return batch

    def append(self, state, action, reward, done, next_state):
        self.replay_memory.append(self.Buffer(state, action, reward, done, next_state))

    def burn_in_capacity(self):
        return len(self.replay_memory) / self.burn_in


class PlayerQ_DQN:
    def __init__(self, env=None, render=True):
        if env == None:
            self.env = iz_env.IZenv()
        else:
            self.env = env
        self.render = render
        self._grid_size = config.N_LANES * config.LANE_LENGTH

    def get_actions(self):
        return list(range(self.env.action_no))

    def num_observations(self):
        return (
            config.N_LANES * config.LANE_LENGTH
            + config.N_LANES
            + len(self.env.plant_deck)
            + 1
        )

    def num_actions(self):
        return self.env.action_no

    def _transform_observation(self, observation):
        return observation.astype(np.float64)

    def play(self, agent, epsilon=0):
        """Play one episode and collect observations and rewards"""

        summary = dict()
        summary["rewards"] = list()
        summary["observations"] = list()
        summary["actions"] = list()
        observation = self._transform_observation(self.env.reset())

        while True:
            if self.render:
                self.env.render()
            action = agent.decide_action(
                observation, self.env.mask_available_actions(), epsilon
            )
            summary["observations"].append(observation)
            summary["actions"].append(action)
            reward, observation, done = self.env.step(action)
            observation = self._transform_observation(observation)
            summary["rewards"].append(reward)

            if done:
                break

        summary["observations"] = np.vstack(summary["observations"])
        summary["actions"] = np.vstack(summary["actions"])
        summary["rewards"] = np.vstack(summary["rewards"])
        return summary

    def get_render_info(self):
        return self.env._scene._render_info
