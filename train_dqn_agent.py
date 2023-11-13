import torch

from izombie_env import simple_env as iz_env
from izombie_dqn.evaluate_agent import manually_test_agent
from izombie_dqn.dqn_agent import DQNAgent

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


if __name__ == "__main__":
    agent = DQNAgent(
        device="cuda" if torch.cuda.is_available() else "cpu",
        learning_rate=1e-3,
        step_length=50,
        max_step_per_ep=500,
        fix_rand=False,
        replay_memory_size=100_000,
        min_replay_memory_size=10_000,
        discount=0.99,
        batch_size=200,
        epsilon_length=100_000,
        start_epsilon=1.0,
        epsilon_interpolation="exponential",
        end_epsilon=0.1,
    )
    agent.train(
        episodes=10_000,
        update_main_every_n_steps=1,
        update_target_every_n_steps=200,
        eval_every_n_episodes=1000,
        stats_window=1_000,
    )
    manually_test_agent(agent)
