

"""
搭建网络：
<<<<<<< HEAD
    在PPO2算法中,
    共有两个网络模型: 策略网络、价值网络
    有三个网络: policy_old, policy_new, critic
    这里定义两类网络模型,
    其中policy_old用于与环境进行交互, 生成action; policy_new用于学习; critic网络生成动作价值
"""


=======
    在PPO2算法中, 
    共有两个网络模型: 策略网络、价值网络
    有三个网络: policy_old, policy_new, critic
    这里定义两类模型
    其中policy_new用于与环境进行交互, 生成action; policy_new用于学习; critic网络生成动作价值Q
"""

>>>>>>> e8e376758484744874b12f25122eb677192130ab
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np



class PolicyNet(nn.Module):
    """
    在策略网络中, 输入的是当前环境的状态, 输出的是当前state下动作概率分布的均值mean和方差std
    """

<<<<<<< HEAD
    def __init__(self, state_dim, hidden_dim, action_dim, max_action=None, ctx=None) -> None:
=======
    def __init__(self, state_dim, hidden_dim, action_dim, ctx, max_action=None) -> None:
>>>>>>> e8e376758484744874b12f25122eb677192130ab
        """
        Params:
            state_dim: int, 状态维度
            hidden_dim: int, 隐藏单元个数
            action_dim: int, 动作维度
<<<<<<< HEAD
            max_action: 对于连续动作空间, 最大的action值
            ctx: cpu OR gpu
=======
            ctx: cpu OR gpu
            max_action: 对于连续动作空间, 最大的action值
>>>>>>> e8e376758484744874b12f25122eb677192130ab
        """
        super(PolicyNet, self).__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.ctx = ctx
        self.max_action = max_action

<<<<<<< HEAD
        ## 构建策略网络模型
        # 输入层和隐藏层
        self.inputLayer = nn.Linear(self.state_dim, self.hidden_dim)
        self.hiddenLayer = nn.Linear(self.hidden_dim, self.hidden_dim)
        # 输出层：共两个, 一个经过处理后输出动作概率分布的均值, 另一个输出分布的方差
        self.outMwanLayer = nn.Linear(self.hidden_dim, self.action_dim)
=======
        ## 构建网络模型
        # 输入层和中间层
        self.inputLayer = nn.Linear(self.state_dim, self.hidden_dim)
        self.hiddenLayer = nn.Linear(self.hidden_dim, self.hidden_dim)
        # 输出层: 共两个，一个经过处理后输出动作概率分布的均值，另一个输出分布的方差
        self.outMeanLayer = nn.Linear(self.hidden_dim, self.action_dim)
>>>>>>> e8e376758484744874b12f25122eb677192130ab
        self.outStdLayer = nn.Linear(self.hidden_dim, self.action_dim)

    def forward(self, state):
        """
<<<<<<< HEAD
            params:
                state: FloatTensor, 在Pendulum中为维度(3,)的向量
            Return:
                mean: float, 动作概率分布的军均值
                std: float, 动作概率分布的方差
=======
        Params:
            state: FloatTensor, 在Pendulum中为维度为(3,)的向量
        Return:
            mean: float, 动作概率分布的均值
            std: float, 动作概率分布的方差
>>>>>>> e8e376758484744874b12f25122eb677192130ab
        """
        # 先经过输入层和隐藏层处理
        afterInputState = F.relu(self.inputLayer(state))
        afterHiddenState = F.relu(self.hiddenLayer(afterInputState))

        ## 经过处理输出动作概率分布的mean
<<<<<<< HEAD
        # 首先使用tanh()激活函数, 因为在Pendulum游戏中, 动作空间为[-2, 2]的连续值
        # 使用tanh()激活可以先将输出转为(-1, 1)之间, 再乘以max_action就可以得到一个(-2, 2)之间的任意实数, 将其当作概率分布的均值
        mean = F.tanh(self.outMwanLayer(afterHiddenState)) * self.max_action

        ## 输出动作概率分布的std
        # 因为方差要大于0, 所以使用softplus激活函数使输出值>0, ps: softplus可以看作relu的平滑
        std = F.softplus(self.outStdLayer(afterHiddenState))

        return mean, std
    
=======
        # 首先使用tanh()激活函数，因为在Pendulum游戏中，动作空间为[-2, 2]的连续均值，
        # 使用tanh激活可以先将输出转为(-1, 1)之间，再乘以max_action就可以得到一个(-2, 2)之间的任意实数，将其当作概率分布的均值
        mean = F.tanh(self.outMeanLayer(afterHiddenState)) * self.max_action

        ## 输出动作概率分布的std
        # 因为方差要大于0，所以使用softplus激活函数使输出值>0，ps:softplus可以看作relu的平滑
        std = F.softplus(self.outStdLayer(afterHiddenState))

        return mean, std
>>>>>>> e8e376758484744874b12f25122eb677192130ab

class ValueNet(nn.Module):
    """
    基于Actor-Critic架构的策略学习算法, 是用价值网络来近似未知的动作价值函数, 进而优化目标函数
<<<<<<< HEAD
    在价值网络中, 输入同样是状态state, 输出则是动作价值Q
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, ctx=None) -> None:
        """
        Params:
=======
    在价值网络中, 输入同样是状态state, 输出的则是动作价值Q
    """

    def __init__(self, input_dim, hidden_dim, output_dim, ctx) -> None:
        """
        Params
>>>>>>> e8e376758484744874b12f25122eb677192130ab
            input_dim: int, 输入维度(state.shape)
            hidden_dim: int, 隐藏单元个数
            output_dim: int, 输出维度(一般情况下为1)
            ctx: cpu OR gpu
        """
        super(ValueNet, self).__init__()
<<<<<<< HEAD

=======
>>>>>>> e8e376758484744874b12f25122eb677192130ab
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.ctx = ctx

        ## 价值网络模型
        self.valueNetModel = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim)
        )

    def forward(self, state):
        """
<<<<<<< HEAD
        Params:
            state, 在Pendulum中为维度(3,)的向量
        Return:
            Q: float, 动过价值Q
        """
        Q = self.valueNetModel(state)

        return Q
=======
        Params: 
            state:  FloatTensor, 在Pendulum中为维度为(3,)的向量
        Return: 
            Q: float, 动作价值Q
        """
        Q = self.valueNetModel(state)
        
        return Q
>>>>>>> e8e376758484744874b12f25122eb677192130ab
