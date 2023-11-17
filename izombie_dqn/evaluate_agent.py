from collections import Counter
import numpy as np

from izombie_env2.config import GameStatus
from izombie_env2.env import IZenv


def evaluate_agent(
    agent, max_step=100_000, test_size=500, episode_count=None, output_file=None
):
    agent.set_to_eval_mode()
    game_results = []
    steps = []
    winning_suns = []

    for test_idx in range(test_size):
        print(f"\rTesting {test_idx}/{test_size}...", end="")
        env = IZenv(max_step=max_step)
        state, mask = env.get_state_and_mask()

        for step in range(max_step + 1):
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

    out = ""
    if episode_count is not None:
        out += f"Episode: {episode_count}\n"
    out += f"Max step: {max_step}\n"
    for status in GameStatus:
        count = results_counter.get(status, 0)
        if count != 0:
            percentage = (count / total) * 100 if total > 0 else 0
            out += f"{status.name}: {count}/{total} ({percentage:.2f}%)\n"
    out += f"Mean steps: {np.mean(steps):.2f}\n"
    if (len(winning_suns)) > 0:
        out += f"Mean winning suns: {np.mean(winning_suns):.2f}\n"

    if output_file is None:
        print(out, end="", flush=True)
    else:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f"{out}\n")

    agent.set_to_training_mode()

    winning_rate = results_counter.get(GameStatus.WIN, 0) / len(game_results)
    mean_winning_sun = np.mean(winning_suns) if len(winning_suns) > 0 else -1
    return winning_rate, mean_winning_sun


def manually_test_agent(agent):
    agent.set_to_eval_mode()

    env = IZenv(fix_rand=True)
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
