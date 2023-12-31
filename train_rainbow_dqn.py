import torch
import numpy as np
import random

from izombie_dqn.rainbow_dqn import DQNAgent
from izombie_env2.env import IZenv
from izombie_dqn.evaluate_agent import evaluate_agent, manually_test_agent


def set_seed(seed):
    def seed_torch(seed):
        torch.manual_seed(seed)
        if torch.backends.cudnn.enabled:
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    np.random.seed(seed)
    random.seed(seed)
    seed_torch(seed)


seed = 0
set_seed(seed)
env = IZenv()

num_steps = 3_000_000
agent = DQNAgent(
    env,
    device="cuda",
    model_name="r2",
    memory_size=1_000_000,
    batch_size=128,
    gamma=0.99,
    alpha=0.2,
    beta=0.6,
    prior_eps=1e-6,
    v_min=-6,
    v_max=55,
    atom_size=None,
    n_step=3,
    lr=1e-3,
)
# agent.load("model/r1_2023.11.19_00.55.17/42.5m.pth")
# manually_test_agent(agent, fix_rand=False)
agent.train(
    update_target_every=2000,
    update_main_every=16,
    num_steps=num_steps,
    print_stats_every=100_000,
    save_every=500_000,
    eval_every=500_000,
)
# evaluate_agent(agent)
