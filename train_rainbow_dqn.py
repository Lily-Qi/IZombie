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

num_steps = 10_000
agent = DQNAgent(
    env,
    seed=seed,
    device="cpu",
    model_name="r1",
    memory_size=100_000,
    batch_size=128,
    gamma=0.99,
    alpha=0.2,
    beta=0.6,
    prior_eps=1e-6,
    v_min=0.0,
    v_max=200.0,
    atom_size=51,
    n_step=3,
)
agent.train(
    update_target_every_n_steps=2000,
    update_main_every_n_steps=1,
    num_steps=num_steps,
    print_stats_every_n_steps=1_000,
    save_every_n_steps=None,
)
evaluate_agent(agent)
manually_test_agent(agent)
