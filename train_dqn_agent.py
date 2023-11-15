import torch

from izombie_dqn.evaluate_agent import manually_test_agent
from izombie_dqn.dqn_agent import DQNAgent

if __name__ == "__main__":
    agent = DQNAgent(
        device="cuda" if torch.cuda.is_available() else "cpu",
        learning_rate=1e-3,
        step_length=25,
        max_step_per_ep=100_000,
        fix_rand=False,
        replay_memory_size=100_000,
        min_replay_memory_size=10_000,
        discount=0.99,
        batch_size=200,
        epsilon_length=500_000,
        start_epsilon=1.0,
        epsilon_interpolation="exponential",
        end_epsilon=0.05,
    )
    agent.load_checkpoint("model/model_80001.pth.zip")
    # agent.save_checkpoint("model/model_80001.pth")
    agent.train(
        episodes=420_000,
        update_main_every_n_steps=32,
        update_target_every_n_steps=2_000,
        save_checkpoint_every_n_episodes=10_000,
        evaluate_test_size=500,
        stats_window=1_000,
    )
    # manually_test_agent(agent)
