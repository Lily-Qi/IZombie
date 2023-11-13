from collections import Counter

from izombie_env import config
from izombie_env import simple_env as iz_env


def evaluate_agent(agent, test_size=10):
    agent.set_to_eval_mode()

    game_results = []
    for _ in range(test_size):
        env = iz_env.IZenv(max_step=agent.env.max_step, fix_rand=True)
        state = env.reset()

        for _ in range(agent.env.max_step + 1):
            action = agent.get_best_q_action(state, env.get_valid_actions())
            _, next_state, game_status = env.step(action)
            state = next_state

            if game_status != config.GameStatus.CONTINUE:
                game_results.append(game_status)
                break

    status_counts = Counter(game_results)
    total = len(game_results)

    print()
    for status in config.GameStatus:
        count = status_counts.get(status, 0)
        percentage = (count / total) * 100 if total > 0 else 0
        print(f"{status.name}: Count = {count}, Percentage = {percentage:.2f}%")
    agent.set_to_trainig_mode()


def manually_test_agent(agent):
    agent.set_to_eval_mode()

    env = iz_env.IZenv(fix_rand=True)
    state = env.reset()

    env.print_human_readable_state()

    for _ in range(1000):
        action = agent.get_best_q_action(state, env.get_valid_actions())
        _, next_state, game_status = env.step(action)
        print(f"ACTION: {action}")
        env.print_human_readable_state()
        state = next_state

        if game_status != config.GameStatus.CONTINUE:
            break

        _ = input("")

    agent.set_to_trainig_mode()
