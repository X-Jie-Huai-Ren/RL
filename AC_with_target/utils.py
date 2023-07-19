

from collections import namedtuple


# 命名元组
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))



# eps decay
def eps_decay():
    pass
