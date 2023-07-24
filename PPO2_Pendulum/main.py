

import gym
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from PPO2Agents import PPO2Agent



class TrainManager:

    def __init__(self, env, agent, num_episodes=1000, len_each_episode=200, batch_size=128, gamma=0.9) -> None:
        """
        env: 环境
        agent: PPO2Agent
        num_episodes: 回合数
        len_each_episode: 每个episode的长度
        batch_size: 一次更新传入的batch大小
        """
        self.env = env
        self.agent = agent
        self.num_episodes = num_episodes
        self.len_each_episode = len_each_episode
        self.batch_size = batch_size
        self.gamma = gamma
        
        self.writer = SummaryWriter('./logs5')

    def train_episode(self):

        # 状态重置
        state = self.env.reset()[0]

        total_reward = 0

        buffer_s, buffer_a, buffer_r, buffer_a_logp = [], [], [], []
        
        for timestep in range(self.len_each_episode):
            # 由策略网络policy_old生成action和action_logprob
            action, action_logprob = self.agent.choose_action(state)
            # 更新环境
            next_state, reward, _, _, _ = self.env.step(action)
            
            buffer_s.append(state)
            buffer_a.append(action)
            buffer_r.append((reward+8)/8)
            buffer_a_logp.append(action_logprob)

            state = next_state
            total_reward += reward

            reward = (reward - reward.mean()) / (reward.std() + 1e-5)

            # PPO 参数更新
            if (timestep+1) % self.batch_size == 0 or timestep == self.len_each_episode-1:
                # 在next_state下的Q值
                next_state_Q = self.agent.get_Q(next_state)
                discounted_r = []

                
                # 这里是用第batch_size*i个next_state来得到真实Q值，然后利用这个Q值去近似前面batch-size个state的真实Q值,后续也许可以用每个state下推出的next_state来得到Q值
                for reward in buffer_r[::-1]:
                    next_state_Q = reward + self.gamma * next_state_Q
                    discounted_r.append(next_state_Q.detach().numpy())
                
                # 再倒置回来
                discounted_r.reverse()
                bs, ba, br, bap = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(discounted_r), np.vstack(buffer_a_logp)
                buffer_s, buffer_a, buffer_r, buffer_a_logp = [], [], [], []
                adv = self.agent.criticLearn(bs, br)
                self.agent.policyNewLearn(bs, ba, adv, bap)
                self.agent.updatePolicyOld()


        return total_reward

    def train(self):

        for episode in range(self.num_episodes):
            total_reward = self.train_episode()
            if episode % 50 == 0:
                print('episode:{0}, total_reward:{1}'.format(episode, total_reward))
                torch.save(self.agent.policy_old.state_dict(), './PPO2_model_actor.pth')
                torch.save(self.agent.critic.state_dict(), './PPO2_model_critic.pth')
            self.writer.add_scalar('return', total_reward, episode)
        



if __name__ == '__main__':

    env = gym.make('Pendulum-v1')

    state_dim = env.observation_space.shape[0]

    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    min_action = env.action_space.low[0]


    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")


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

    tm.train()

    