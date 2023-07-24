

import gym
import torch
from torch.utils.tensorboard import SummaryWriter

from PPO2Agents import PPO2Agent
from utils import Transition
from utils import Buffer



class TrainManager:

    def __init__(self, env, agent, num_episodes=10000, len_each_episode=640, batch_size=64) -> None:
        """
        env: 环境
        agent: PPO2Agent
        num_episodes: 回合数
        len_each_episode: 每个episode的长度
        batch_size: 多久更新一次
        """
        self.env = env
        self.agent = agent
        self.num_episodes = num_episodes
        self.len_each_episode = len_each_episode
        self.batch_size = batch_size
        
        self.writer = SummaryWriter('./logs')
        self.buffer = Buffer(capacity=self.batch_size)

    def train_episode(self):

        # 状态重置
        state = self.env.reset()[0]

        total_reward = 0
        
        for timestep in range(self.len_each_episode):
            # 由策略网络policy_old生成action和action_logprob
            action, action_logprob = self.agent.choose_action(state)
            # 更新环境
            









    def train(self):

        for episode in range(self.num_episodes):
            self.train_episode()
        





if __name__ == '__main__':

    env = gym.make('Pendulum-v1')

    state_dim = env.observation_space.shape[0]

    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    min_action = env.action_space.low[0]


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # state = env.reset()[0]
    # state = FloatTensor(state).to(device)

    # # 测试策略网络
    # policy_old = PolicyNet(state_dim=state_dim, hidden_dim=16, action_dim=action_dim, max_action=max_action).to(device)
    # mean, std = policy_old(state)
    # print(mean)
    # print(std)

    # # 测试价值网络
    # critic = ValueNet(input_dim=state_dim, hidden_dim=16, output_dim=1).to(device)
    # Q = critic(state)
    # print(Q)

    agent = PPO2Agent(state_dim, hidden_dim=16, action_dim=action_dim, max_action=max_action, min_action=min_action, device=device)

    tm = TrainManager(env, agent)

    