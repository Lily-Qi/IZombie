from collections import Counter
import numpy as np
import os

from izombie_env import config
from izombie_env import simple_env as iz_env


def evaluate_agent(agent, max_step=None, test_size=10):
    agent.set_to_eval_mode()
    if max_step is None:
        max_step = agent.env.max_step
    game_results = []
    steps = []
    winning_suns = []

    for test_idx in range(test_size):
        print(f"\rTesting {test_idx}/{test_size}...", end="")
        env = iz_env.IZenv(max_step=max_step)
        state, mask = env.get_state_and_mask()

        for step in range(max_step + 1):
            action = agent.get_best_q_action(state, env.get_valid_actions(mask))
            _, next_state, next_mask, game_status = env.step(action)
            state, mask = next_state, next_mask

            if game_status != config.GameStatus.CONTINUE:
                game_results.append(game_status)
                steps.append(step)
                if game_status == config.GameStatus.WIN:
                    winning_suns.append(env.get_sun())
                break

    status_counts = Counter(game_results)
    total = len(game_results)

    print(f"\nMax step: {max_step}")
    for status in config.GameStatus:
        count = status_counts.get(status, 0)
        if count != 0:
            percentage = (count / total) * 100 if total > 0 else 0
            print(f"{status.name}: {count}/{total} ({percentage:.2f}%)")
    print(f"Mean steps: {np.mean(steps)}")
    if (len(winning_suns)) > 0:
        print(f"Mean winning suns: {np.mean(winning_suns)}")

    agent.set_to_trainig_mode()


def manually_test_agent(agent):
    agent.set_to_eval_mode()

    env = iz_env.IZenv(fix_rand=True)
    state, mask = env.get_state_and_mask()
    last_step = 0

    for step in range(10000):
        action = agent.get_best_q_action(state, env.get_valid_actions(mask))
        reward, next_state, next_mask, game_status = env.step(action)

        if action != 0 or game_status != config.GameStatus.CONTINUE:
            env.print_human_readable_state(
                highlight=((action - 1) % 5, 4) if action != 0 else None
            )
            print(f"Action: {action}, Reward: {reward}, ΔStep: {step - last_step}")
            last_step = step
            _ = input("")

        state, mask = next_state, next_mask

        if game_status != config.GameStatus.CONTINUE:
            break

    agent.set_to_trainig_mode()
