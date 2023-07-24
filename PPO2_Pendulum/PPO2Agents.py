
import torch
from torch import optim
from torch import nn
from torch import FloatTensor
from torch import distributions

from model import PolicyNet
from model import ValueNet



class PPO2Agent:

    def __init__(self, state_dim, hidden_dim, action_dim, max_action, min_action, device, lr=0.0001, gamma=0.9) -> None:
        """
        Params:
            state_dim: int, 状态维度
            hidden_dim: int, 隐藏单元个数
            action_dim: int, 动作维度
            max_action: 对于连续动作空间, 最大的action值
            min_action: 对于连续动作空间, 最小的action值
            device: cpu OR gpu
            lr: learning rate
            gamma: discounted rate
        """

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.min_action = min_action
        self.device = device
        self.lr = lr
        self.gamma = gamma

        # 定义策略网络poliy_old, 用来和环境进行交互, 生成action
        self.policy_old = PolicyNet(state_dim=self.state_dim, hidden_dim=self.hidden_dim, action_dim=self.action_dim, max_action=self.max_action).to(self.device)
        # 定义策略网络policy_new, 新策略用来学习
        self.policy_new = PolicyNet(state_dim=self.state_dim, hidden_dim=self.hidden_dim, action_dim=self.action_dim, max_action=self.max_action).to(self.device)

        # 定义价值网络, 生成动作价值Q
        self.critic = ValueNet(input_dim=self.state_dim, hidden_dim=self.hidden_dim, output_dim=1).to(self.device)

        # 定义损失函数, 该损失函数用来优化价值网络critic，这里先用MSELoss(), 后续可以改进
        self.criticLossFunc = nn.MSELoss()

        # 定义优化器
        self.policyNewOptimizer = optim.Adam(self.policy_new.parameters(), lr=self.lr)
        self.criticOptimizer = optim.Adam(self.critic.parameters(), lr=self.lr)


    # 第一步: 根据策略网络policy_old, 选择action
    def choose_action(self, state):
        """
        Params:
            state: numpy.ndarrdy, 环境状态
        Return:
            action: [-2, 2]之间的实数
            action_logprob: action对应概率的对数
        """
        # 将state数据类型转换
        state = FloatTensor(state).to(self.device)

        # 将其输入policy_old, 输出动作概率分布的均值和方差
        mean, std = self.policy_old(state)

        # 根据生成的均值和方差, 生成正态分布
        distribution = distributions.Normal(mean, std)
        # 然后sample出一个动作
        action = distribution.sample()
        # 在Pendulum中, 动作空间为[-2, 2], 须截断
        action = torch.clamp(action, self.min_action, self.max_action)

        # 计算action对应概率的对数
        action_logprob = distribution.log_prob(action)

        return action.detach().numpy(), action_logprob.detach().numpy()
    
    # 第二步: 根据价值网络critic, 获取动作价值
    def get_Q(self, state):
        """
        Params:
            state: numpy.ndarrdy, 环境状态
        Return:
            Q: 动作价值
        """
        # 将state数据类型转换
        state = FloatTensor(state).to(self.device)
        Q = self.critic(state)

        return Q
    
    # 第三步: 更新policy_old-->将policy_new的参数赋值给policy_old
    def updatePolicyOld(self):
        self.policy_old.load_state_dict(self.policy_new.state_dict())