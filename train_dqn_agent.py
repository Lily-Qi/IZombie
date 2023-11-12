from izombie_dqn.dqn_agent_2 import DQNAgent, experienceReplayBuffer_DQN, QNetwork_DQN
from izombie_env import env as iz_env
import torch

if __name__ == "__main__":
    n_iter = 100000
    env = iz_env.IZenv()
    nn_name = "test"
    buffer = experienceReplayBuffer_DQN(memory_size=100000, burn_in=10000)
    net = QNetwork_DQN(env, device="cpu")
    agent = DQNAgent(env, net, buffer, batch_size=200)
    agent.train(max_episodes=n_iter, evaluate_frequency=5000, evaluate_n_iter=1000)
    torch.save(agent.network, nn_name)
    agent._save_training_data(nn_name)
