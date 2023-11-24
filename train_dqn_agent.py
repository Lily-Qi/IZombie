import torch
import threading

from izombie_dqn.evaluate_agent import manually_test_agent, evaluate_agent
from izombie_dqn.dqn_agent import DQNAgent


MODEL_DIR = "reward_2"

lock = threading.Lock()
best_agent = None
best_winning_rate = 0


def train_one(i):
    agent = DQNAgent(
        model_dir=MODEL_DIR,
        device="cpu",
        learning_rate=1e-3,
        step_length=50,
        max_step_per_ep=100_000,
        fix_rand=False,
        replay_memory_size=100_000,
        min_replay_memory_size=10_000,
        discount=0.99,
        batch_size=200,
        epsilon_length=100_000,
        start_epsilon=1.0,
        epsilon_interpolation="exponential",
        end_epsilon=0.05,
    )
    agent.model_name = f"{i + 1}"

    agent.train(
        num_steps=100_000,
        update_main_every=32,
        update_target_every=2_000,
        eval_every=None,
        evaluate_test_size=500,
        save_every=None,
        stats_window=1_000,
    )
    winning_rate, _ = evaluate_agent(agent, test_size=500)
    with lock:
        global best_agent, best_winning_rate
        if winning_rate > best_winning_rate:
            best_agent = agent
            best_winning_rate = winning_rate


if __name__ == "__main__":
    threads = []
    for i in range(10):
        thread = threading.Thread(target=train_one, args=(i,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    # best_agent.save_checkpoint(f"{MODEL_DIR}/out.pth")
    # print(f"{best_winning_rate=}")

    # agent.load_checkpoint("model/320,100,16_2023.11.16_10.36.19/model_90000.pth.zip")
    # agent.reset_epsilon(
    #     epsilon_length=2,
    #     start_epsilon=0.05,
    #     epsilon_interpolation="exponential",
    #     end_epsilon=0.05,
    # )

    # manually_test_agent(agent)
