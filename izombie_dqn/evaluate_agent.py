from collections import Counter
import numpy as np
import os

from izombie_env2.config import GameStatus
from izombie_env2.env import IZenv
from .util import create_folder_if_not_exist, format_num


def evaluate_agent(agent, test_size=500, step_count=None, output_file=None):
    agent.set_to_eval_mode()
    game_results = []
    steps = []
    winning_suns = []

    for test_idx in range(1, test_size + 1):
        print(f"\rTesting {test_idx}/{test_size}...", end="")
        env = IZenv()
        state, mask = env.get_state_and_mask()

        for step in range(1_000_000):
            action = agent.get_best_q_action(state, env.get_valid_actions(mask))
            _, next_state, next_mask, game_status = env.step(action)
            state, mask = next_state, next_mask

            if game_status != GameStatus.CONTINUE:
                game_results.append(game_status)
                steps.append(step)
                if game_status == GameStatus.WIN:
                    winning_suns.append(env.get_sun())
                break

    print()

    results_counter = Counter(game_results)
    total = len(game_results)
    percentages = {}
    for status in GameStatus:
        percentages[status] = (
            (results_counter.get(status, 0) / total) * 100 if total > 0 else 0
        )

    if output_file is None:
        if step_count is not None:
            print(f"Step: {format_num(step_count)}")
        for status in GameStatus:
            count = results_counter.get(status, 0)
            if count != 0:
                print(f"{status.name}: {count}/{total} ({percentages[status]:.2f}%)")
        print(f"Mean steps: {np.mean(steps):.2f}")
        if (len(winning_suns)) > 0:
            print(f"Mean winning suns: {np.mean(winning_suns):.2f}")
    else:
        create_folder_if_not_exist(os.path.dirname(output_file))
        start_new_file = not os.path.exists(output_file)
        with open(output_file, "a", encoding="utf-8") as f:
            if start_new_file:
                f.write("step,test_size,win,lose,mean_steps,mean_winning_suns,\n")
            f.write(
                f"{step_count},{test_size},{percentages[GameStatus.WIN]:.2f}%,{percentages[GameStatus.LOSE]:.2f}%,{np.mean(steps):.2f},{np.mean(winning_suns):.2f}\n"
            )

    agent.set_to_training_mode()

    winning_rate = results_counter.get(GameStatus.WIN, 0) / len(game_results)
    mean_winning_sun = np.mean(winning_suns) if len(winning_suns) > 0 else -1
    return winning_rate, mean_winning_sun


def manually_test_agent(agent, fix_rand=True):
    agent.set_to_eval_mode()

    if fix_rand:
        np.random.seed(0)
    else:
        np.random.seed()

    env = IZenv()
    state, mask = env.get_state_and_mask()
    last_step = 0

    for step in range(10000):
        action = agent.get_best_q_action(state, env.get_valid_actions(mask))
        reward, next_state, next_mask, game_status = env.step(action)

        if action != 0 or game_status != GameStatus.CONTINUE:
            env.print_human_readable_state(
                highlight=((action - 1) % 5, 4) if action != 0 else None
            )
            print(f"Action: {action}, Reward: {reward}, ΔStep: {step - last_step}")
            last_step = step
            _ = input("")

        state, mask = next_state, next_mask

        if game_status != GameStatus.CONTINUE:
            break

    agent.set_to_training_mode()
