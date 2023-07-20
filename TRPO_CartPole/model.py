


import torch
from torch import nn
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

        




