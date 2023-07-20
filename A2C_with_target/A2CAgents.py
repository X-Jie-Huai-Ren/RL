


from torchvision import transforms
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F
from torch import optim
import numpy as np
import copy
from PIL import Image

from model import PolicyNet
from model import ValueNet


torch.autograd.set_detect_anomaly = True

class ACAgent:
    
    def __init__(self, in_channels, channels_lst, action_dim, update_target_steps=10, lr=0.00001, gamma=0.9) -> None:
        """
        Params:
            in_channels: int, 状态(图片)的通道数
            channels_lst: list, 卷积通道数
            action_dim: int, action的维度
            lr: float, 优化器的学习率
            gamma: float, discount rate 
        """
        # 策略网络
        self.policyNet = PolicyNet(in_channels, channels_lst, action_dim)
        # 目标网络: 辅助训练, 缓解自举带来的高估问题
        self.targetNet = ValueNet(in_channels, channels_lst, output_dim=1)
        # 价值网络
        self.valueNet = ValueNet(in_channels, channels_lst, output_dim=1)

        # 转换器
        # self.trans = transforms.ToTensor()
        self.trans = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])

        #
        self.num_actions = action_dim
        self.gamma = gamma
        self.update_target_steps = update_target_steps

        # 损失函数
        self.lossfunc = nn.MSELoss()
        # 优化器
        self.policy_optimizer = optim.Adam(self.policyNet.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.valueNet.parameters(), lr=lr)


    def get_action(self, state, eps):
        """
        Params:
            state: numpy.ndarray, 维度(210, 160, 3), 状态帧
            eps: float, epslion-greedy算法中的探索率
        Return:
            int, the index of action
        """
        # 将state转换为torch张量, 并且维度为(batch_size, channels, width, height)-->(1, 3, 210, 160)
        state = Image.fromarray(state)
        state = self.trans(state).unsqueeze(0)

        # 探索
        if np.random.uniform(0, 1) < eps:
            action = np.random.choice(self.num_actions)
        # 利用
        else:
            # 将状态输入策略网络, 得到采取各个action的value
            q_lst = self.policyNet(state)
            # 依据离散动作空间上的概率分布选择action
            distribution = Categorical(q_lst)
            action = int(distribution.sample())

        return action
    
    def learn(self, batch, cur_step):
        """
        batch: namedtuple, 包括state, action, reward, next_state, done
        """
        # 将state,next_state处理为tensor张量
        state = Image.fromarray(batch.state)
        next_state = Image.fromarray(batch.next_state)
        state = self.trans(state).unsqueeze(0)
        next_state = self.trans(next_state).unsqueeze(0)

        # 根据价值网络给(state)打分
        cur_score = self.valueNet(state)[0][0]

        # 根据目标网络给(next_state)打分
        next_score = self.targetNet(next_state)[0][0]

        # 计算TD target 和 TDerror
        TD_target = batch.reward + self.gamma * next_score * (1-batch.done)
        td_error = cur_score - TD_target

        # 获取动作的概率密度函数
        actions_prob = self.policyNet(state)
        # 根据动作的概率密度函数，得到state状态时采取action的概率
        cur_action_prob = actions_prob[0][batch.action]
        # ln处理
        ln_cur_action_prob = torch.log(cur_action_prob)
        td_error1 = copy.deepcopy(td_error.detach())

        # 乘以得分,负号是让梯度上升算法转为梯度下降
        loss = td_error1 * ln_cur_action_prob

        # 更新价值网络
        self.value_optimizer.zero_grad()
        td_error_loss = self.lossfunc(cur_score, TD_target)
        td_error_loss.backward(retain_graph=True)
        self.value_optimizer.step()

        ## 更新策略网络
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()    

        # 更新目标网络
        if cur_step % self.update_target_steps == 0:
            self.targetNet.load_state_dict(self.valueNet.state_dict())


