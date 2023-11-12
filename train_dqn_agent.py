from izombie_env import env as iz_env

# from izombie_dqn.dqn_agent_2 import DQNAgent, experienceReplayBuffer_DQN, QNetwork_DQN
# import torch

# if __name__ == "__main__":
#     n_iter = 100000
#     env = iz_env.IZenv(disable_col_6_to_9=True)
#     nn_name = "test"
#     buffer = experienceReplayBuffer_DQN(memory_size=100000, burn_in=10000)
#     net = QNetwork_DQN(env, device="cuda")
#     agent = DQNAgent(env, net, buffer, batch_size=200)
#     agent.train(max_episodes=n_iter, evaluate_frequency=5000, evaluate_n_iter=1000)
#     torch.save(agent.network, nn_name)
#     agent._save_training_data(nn_name)


from izombie_dqn.dqn_agent import DQNAgent

if __name__ == "__main__":
    env = iz_env.IZenv(max_step=1_200, disable_col_6_to_9=True)
    agent = DQNAgent(env=env, device="cuda")
    agent.train(episodes=100_000)
