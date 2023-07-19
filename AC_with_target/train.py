

import gym
import numpy as np
from torchvision import transforms
from model import PolicyNet
from model import ValueNet
from ACAgents import ACAgent
from utils import Transition
import time




def train_episode(env, agent, eps):
    """
    Params:
        env: 环境
        agent: 构建的智能体
        eps: float, epslion-greedy算法中的探索率
    Return:
        float, 本episode获得的回报(总奖励)
    """

    # 状态重置
    state = env.reset()[0]

    done = False
    total_reward = 0
    cur_step = 0
    start = time.time()

    # start episode
    while not done:
        # 根据当前状态采取action
        action = agent.get_action(state, eps=eps)
        print(action)

        # change env
        next_state, reward, done, _, _ = env.step(action)

        # 奖励叠加
        total_reward += reward

        batch = Transition(state, action, reward, next_state, done)


        # 更新智能体
        agent.learn(batch, cur_step)

        state = next_state
        cur_step += 1
        env.render()

        # done = True


    return total_reward






if __name__ == '__main__':

    env = gym.make("ALE/Assault-v5", render_mode='human')
    # channels_lst = [16, 32, 64, 128]
    channels_lst = [8, 16, 32, 64]
    agent = ACAgent(in_channels=1, channels_lst=channels_lst, action_dim=7)


    train_episode(env, agent, eps=0.2)

    # state = env.reset()[0]

    # model = ValueNet(in_channels=3, channels_lst=channels_lst, output_dim=7)

    # ### Test model
    # trans = transforms.ToTensor()
    # input = trans(state).unsqueeze(0)
    # print(input.shape)
    # out = model(input)
    # print(out)

    



    
        