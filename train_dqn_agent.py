import torch

from izombie_dqn.evaluate_agent import manually_test_agent
from izombie_dqn.dqn_agent import DQNAgent

if __name__ == "__main__":
    agent = DQNAgent(
        device="cuda" if torch.cuda.is_available() else "cpu",
        learning_rate=1e-4,
        step_length=50,
        max_step_per_ep=20000,
        fix_rand=False,
        replay_memory_size=150000,
        min_replay_memory_size=20000,
        discount=0.995,
        batch_size=128,
        epsilon_length=200000,
        start_epsilon=1.0,
        epsilon_interpolation="exponential",
        end_epsilon=0.05,
    )
    # agent.load_checkpoint("model/model_500000.pth.zip")
    # agent.save_checkpoint("model/model_80001.pth")
    agent.train(
        episodes=100000,
        update_main_every_n_steps=32,
        update_target_every_n_steps=500,
        evaluate_every_n_episodes=5000,
        evaluate_test_size=500,
        save_checkpoint_every_n_episodes=50000,
        stats_window=1000,
    )
    # manually_test_agent(agent)
