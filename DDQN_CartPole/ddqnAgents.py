



import torch
from torch import nn
from torch import optim 
from torch import FloatTensor
from torch.autograd import Variable
import numpy as np
from model import QNetwork

from replay_buffer import Transition, ReplayMemory

# 超参数设置
BATCH_SIZE = 64
LEARNING_RATE = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QNetAgent(object):

    def __init__(self, n_states, n_actions, hidden_dim, eps=0.2) -> None:

        self.n_states = n_states
        self.n_actions = n_actions
        self.hiddem_dim = hidden_dim
        self.eps = eps

        # Q网络和目标网络
        self.q_local = QNetwork(n_states, n_actions, hidden_dim=16).to(device)
        self.q_target = QNetwork(n_states, n_actions, hidden_dim=16).to(device)

        # 损失函数
        self.lossfunc = nn.MSELoss()

        # 优化器: 在双Q学习中，对Q网络(q_local)参数的更新是根据梯度下降进行的，而目标网络参数的更新是根据Q网络的参数和自身
        self.optimizer = optim.Adam(self.q_local.parameters(), lr=LEARNING_RATE)

        # 经验池
        self.replay_memory = ReplayMemory(10000)

    def get_action(self, state, eps):
        """
            params:
                state: 2-D tensor of shape(n, input_dim)
            
            return:
                int: action index
        """
        # global steps_done

        # 探索
        if np.random.uniform(0, 1) < eps:
            action = np.random.choice(self.n_actions)
        else:
            q_lst = self.q_local(state)
            # 选取对应Q值最大的action
            action = np.argmax(q_lst.detach().numpy())

        return action

        
