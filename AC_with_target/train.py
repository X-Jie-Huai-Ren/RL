

import gym
import torch
from ACAgents import ACAgent
from utils import Transition
from utils import eps_decay



class TrainManager:
    def __init__(self, env, agent, eps, num_episodes=100) -> None:
        """
        Params:
            env: 环境
            agent: 构建的智能体
            eps: float, epslion-greedy算法中的探索率
        """
        self.env = env
        self.agent = agent
        self.eps = eps
        self.num_episodes=num_episodes


    def train_episode(self):
        
        # 状态重置
        state = self.env.reset()

        done = False
        total_reward = 0
        cur_step = 0

        # self.env.render()
        # start episode
        while not done:
            # 根据当前状态采取action
            action = agent.get_action(state, self.eps)

            # change env
            next_state, reward, done, _, _ = self.env.step(action)

            # 奖励叠加
            total_reward += reward

            batch = Transition(state, action, reward, next_state, done)


            # 更新智能体
            self.agent.learn(batch, cur_step)

            state = next_state
            cur_step += 1
            # 如果reward=0, 不衰减
            if reward == 0:
                pass
            else:
                self.eps = eps_decay(self.eps)
            # self.env.render()

        return total_reward
    

    def train(self):
        for episode in range(self.num_episodes):
            total_reward = self.train_episode()
            print('episode:{0}, return:{1}'.format(episode, total_reward))
            torch.save(self.agent.policyNet.state_dict(), './policyNet.pth')






if __name__ == '__main__':

    env = gym.make("ALE/Assault-v5", render_mode='human')
    # channels_lst = [16, 32, 64, 128]
    channels_lst = [8, 16, 32, 64]
    agent = ACAgent(in_channels=1, channels_lst=channels_lst, action_dim=7)

    # train_episode(env, agent, eps=0.2)

    # state = env.reset()[0]

    # model = ValueNet(in_channels=3, channels_lst=channels_lst, output_dim=7)

    # ### Test model
    # trans = transforms.ToTensor()
    # input = trans(state).unsqueeze(0)
    # print(input.shape)
    # out = model(input)
    # print(out)

    tm = TrainManager(env, agent, eps=0.2)
    tm.train()

    



    
        