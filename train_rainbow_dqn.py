import torch
import numpy as np
import random

from izombie_dqn.rainbow_dqn import DQNAgent
from izombie_env2.env import IZenv


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
agent = DQNAgent(env, memory_size=10_000, batch_size=128, target_update=100, seed=seed)
agent.train(num_steps=10_000)
