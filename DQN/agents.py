import numpy as np
import torch
from torch import nn
from utils import one_hot

class DQNAgent:
    def __init__(self, q_func, replay_buffer, batch_size, replay_start_size, optimizer, n_acts, eps=0.1, gamma=0.9) -> None:
        # 不在需要Q-table, 而是使用Q函数(神经网络)
        self.q_func = q_func
        
        self.global_step = 0
        self.rb = replay_buffer
        self.batch_size = batch_size
        self.replay_start_size = replay_start_size

        self.n_acts = n_acts

        # 损失函数
        self.criterion = nn.MSELoss()

        # 优化器
        self.optimizer = optimizer

        # eps-greedy 探索概率
        self.eps = eps
        self.gamma = gamma                    

    # 根据observation预测action
    def predict(self, observation):
        observation = torch.FloatTensor(observation)
        # 将observation输入Q网络,得到由各个action的reward组成的一维向量
        Q_lst = self.q_func(observation)
        # 获取reward最大的action的索引
        action = int(torch.argmax(Q_lst).detach().numpy())
        return action

    # 基于greedy策略, 根据observation获取action
    def get_act(self, observation):
        # 探索
        # 在ε-greedy算法中，需要一定的概率去探索新的action，防止策略陷入局部最优
        if np.random.uniform(0, 1) < self.eps:
            action = np.random.choice(self.n_acts)
        # 利用
        else:
            action = self.predict(observation)
        return action
        
    # 不使用经验回放的学习
    # def learn(self, obser, action, reward, next_obser, done):
    #     # 根据当前observation获取action下的action-value
    #     predict_Q = self.q_func(obser)[action]
        
    #     target_Q = reward + (1-float(done)) * self.gamma * self.q_func(next_obser).max()
        
    #     ## 优化损失函数-->更新Q网络
    #     # 梯度归零
    #     self.optimizer.zero_grad()
    #     loss = self.criterion(predict_Q, target_Q)
    #     # 反向传播
    #     loss.backward()
    #     self.optimizer.step()


    def learn_batch(self, obs_batch, action_batch, reward_batch, next_obs_batch, done_batch):
        # 根据当前observation获取预测的action
        pred_Vs = self.q_func(obs_batch)   # batch_size是32,且在"CartPole-v0"中由2种action 则pred_Vs是32x2的矩阵

        # 将action索引转为独热向量
        action_onehot = one_hot(action_batch, self.n_acts)

        predict_Q = (pred_Vs * action_onehot).sum(dim=1)
        
        target_Q = reward_batch + (1-done_batch) * self.gamma * self.q_func(next_obs_batch).max(1)[0]
        
        ## 优化损失函数-->更新Q网络
        # 梯度归零
        self.optimizer.zero_grad()
        loss = self.criterion(predict_Q, target_Q)
        # 反向传播
        loss.backward()
        self.optimizer.step()


    # 使用经验回放的学习
    def learn(self, obser, action, reward, next_obser, done):
        self.global_step += 1
        # append入replay_buffer区
        self.rb.append((obser, action, reward, next_obser, done))

        # 判断是否从经验中学习
        if len(self.rb) > self.replay_start_size and self.global_step%self.rb.num_steps==0:
            self.learn_batch(*self.rb.sample(self.batch_size))
 