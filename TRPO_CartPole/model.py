


import torch
from torch import nn
import torch.nn.functional as F
from torch import optim


class policyNet(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        """
        Params:
            input_dim: int, 输入维度
            hidden_dim: int, 隐藏单元个数
            output_dim: int, 输出维度
        """
        super(policyNet, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.output_dim),
            F.softmax(dim=1)
        )

    def forward(self, obs):
        """
        Params:
            obs: tensor, 环境的状态
        Return:
            action的概率
        """
        return self.model(obs)

        




