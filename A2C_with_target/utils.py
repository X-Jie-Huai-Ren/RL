

from collections import namedtuple


# 命名元组
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))



# eps decay
def eps_decay(eps, eps_decay_rate=1e-6):
    
    return max(0.01, eps-eps_decay_rate)
