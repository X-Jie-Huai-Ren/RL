
"""
    经验回收机制
"""

import collections
import random
from torch import FloatTensor

class ReplayBuffer:

    def __init__(self, max_size, num_steps=4) -> None:
        self.buffer = collections.deque(maxlen=max_size)
        self.num_steps = num_steps

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = zip(*mini_batch)
        # 转为torch的张量形式
        obs_batch = FloatTensor(obs_batch)
        action_batch = FloatTensor(action_batch)
        reward_batch = FloatTensor(reward_batch)
        next_obs_batch = FloatTensor(next_obs_batch)
        done_batch = FloatTensor(done_batch)

        return obs_batch, action_batch, reward_batch, next_obs_batch, done_batch
    
    def __len__(self):
        return len(self.buffer)