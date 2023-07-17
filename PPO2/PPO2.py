import torch
from torch import nn
from torch import FloatTensor
import torch.nn.functional as F
import numpy as np
import gym
import os
import matplotlib.pyplot as plt



####### 第一步，编写AC架构  #######

# Actor网络：包括new和old
class ActorNet(nn.Module):
    def __init__(self, input, output, max_action) -> None:
        super(ActorNet, self).__init__()

        self.max_action = max_action

        self.input_layer = nn.Linear(input, 100)
        self.input_layer.weight.data.normal_(0, 0.1)
        
        self.mean_out = nn.Linear(100, output)
        self.mean_out.weight.data.normal_(0, 0.1)

        self.std_out = nn.Linear(100, output)
        self.std_out.weight.data.normal_(0, 0.1)

    # 生成均值与标准差，PPO必须这样做，一定要生成分布（所以需要mean与std），不然后续学习策略里的公式写不了
    def forward(self, inputstate):
        inputstate = self.input_layer(inputstate)
        inputstate = F.relu(inputstate)
        # 输出概率分布的均值mean
        mean = self.max_action * torch.tanh(self.mean_out(inputstate))
        # 输出概率分布的方差std
        std = F.softplus(self.std_out(inputstate))    # softplus激活函数的值域>0
        return mean, std
    

# Critic网络
class CriticNet(nn.Module):
    def __init__(self, input, output) -> None:
        super(CriticNet, self).__init__()
        self.input_layer = nn.Linear(input, 100)
        self.input_layer.weight.data.normal_(0, 0.1)
        self.output_layer = nn.Linear(100, output)
        self.output_layer.weight.data.normal_(0, 0.1)

    def forward(self, inputstate):
        inputstate = F.relu(self.input_layer(inputstate))
        Q = self.output_layer(inputstate)
        return Q
    
class Actor:
    def __init__(self, state_num, action_num, min_action, max_action, Actor_Update_Steps, METHOD, lr=0.0001) -> None:
        self.state_num = state_num
        self.action_num = action_num
        self.min_action = min_action
        self.max_action = max_action
        self.Actor_Update_Steps = Actor_Update_Steps
        self.METHOD = METHOD
        self.old_pi = ActorNet(state_num, action_num, max_action)
        self.new_pi = ActorNet(state_num, action_num, max_action)
        self.optimizer=torch.optim.Adam(self.new_pi.parameters(),lr=lr,eps=1e-5)

    # 第二步 根据状态选取动作
    def choose_action(self, state):
        inputstate = FloatTensor(state)
        # 老策略用于选择动作，与环境进行交互
        mean, std = self.old_pi(inputstate)
        # 根据mean, std生成分布
        distribution = torch.distributions.Normal(mean, std)
        action = distribution.sample()
        # 截断操作
        action = torch.clamp(action, self.min_action, self.max_action)
        action_logprob = distribution.log_prob(action)

        return action.detach().numpy(), action_logprob.detach().numpy()
    
    # 第四步 Actor网络有两个策略（更新old策略）-->将new策略的参数赋值给old策略
    def update_oldpi(self):
        self.old_pi.load_state_dict(self.new_pi.state_dict())

    # 第六步 Actor网络的学习函数, 采用PPO2， clip函数
    def learn(self, bs, ba, adv, bap):
        """
        """
        bs = FloatTensor(bs)
        ba = FloatTensor(ba)
        adv = FloatTensor(adv)
        bap = FloatTensor(bap)

        for _ in range(self.Actor_Update_Steps):
            mean, std = self.new_pi(bs)
            # 新分布
            distribution_new = torch.distributions.Normal(mean, std)
            action_new_logprob=distribution_new.log_prob(ba)
            ratio = torch.exp(action_new_logprob-bap.detach())
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - self.METHOD['epsilon'], 1 + self.METHOD['epsilon']) * adv
            loss = -torch.min(surr1, surr2)
            loss=loss.mean()
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.new_pi.parameters(), 0.5)
            self.optimizer.step()

class Critic:
    def __init__(self, state_num, out_num, Critic_Update_Steps = 10, lr=0.0003, eps=1e-5) -> None:
        self.critic_v = CriticNet(state_num, out_num)
        self.optimizer = torch.optim.Adam(self.critic_v.parameters(), lr=lr, eps=eps)
        self.lossfunc = nn.MSELoss()
        self.Critic_Update_Steps = Critic_Update_Steps 

    # 第三步: 评价动作价值
    def get_v(self, state):
        inputstate = FloatTensor(state)
        return self.critic_v(inputstate)
    
    # 第五步 计算优势, 可以和第七步合为一体
    # def get_adv(self, bs, br):
    #     reality_v=torch.FloatTensor(br)
    #     v=self.get_v(bs)
    #     adv=(reality_v-v).detach()
    #     return adv
    
    # 第七步  actor-critic的critic部分的learn函数，td-error的计算代码（V现实减去V估计就是td-error）
    def learn(self,bs,br):
        bs = torch.FloatTensor(bs)
        reality_v = torch.FloatTensor(br)
        for _ in range(self.Critic_Update_Steps):
            v=self.get_v(bs)
            td_e = self.lossfunc(reality_v, v)
            self.optimizer.zero_grad()
            td_e.backward()
            nn.utils.clip_grad_norm_(self.critic_v.parameters(), 0.5)
            self.optimizer.step()
        return (reality_v-v).detach()

    




if __name__ == '__main__':

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    # 设置环境
    env = gym.make('Pendulum-v1', render_mode="human").unwrapped
    # 状态特征有3个：杆子的角度sin,cos,角速度  状态是连续的，有无限多个
    state_num = env.observation_space.shape[0]
    # 动作特征有1个: 力矩   限定在[-2, 2]之间的任意实数,动作也有无限多个
    action_num = env.action_space.shape[0]

    # 最大最小动作值
    max_action = env.action_space.high[0]      # 2.0
    min_action = env.action_space.low[0]       # -2.0


    # 是否渲染
    render = False

    # Hyper parameters
    EP_MAX = 1000
    EP_LEN = 200
    gamma = 0.9         # 折扣因子
    Actor_lr = 0.0001
    critic_lr = 0.0003
    batch_size = 128
    Actor_Update_Steps = 10
    Critic_Update_Steps = 10

    # choose one method to optimize 
    method = [
        dict(name='KL_Penalty', kl_target=0.01, lam=0.5),
        dict(name='clip', epsilon=0.2)
    ][1]

    # 训练or测试
    switch = 0



    if switch == 0:
        print('训练中...')
        actor = Actor(state_num, action_num, min_action, max_action, Actor_Update_Steps, method)
        critic = Critic(state_num, 1, Critic_Update_Steps, lr=critic_lr)
        all_ep_r = []
        for episode in range(EP_MAX):
            # 环境重置
            observation = env.reset()[0]
            buffer_s, buffer_a, buffer_r,buffer_a_logp = [], [], [],[]
            reward_totle=0
            for timestep in range(EP_LEN):
                if render:
                    env.render()
                action,action_logprob=actor.choose_action(observation)
                observation_, reward, done, info, _ = env.step(action)
                buffer_s.append(observation)
                buffer_a.append(action)
                buffer_r.append((reward+8)/8)    # normalize reward, find to be useful
                buffer_a_logp.append(action_logprob)
                observation=observation_
                reward_totle+=reward
                reward = (reward - reward.mean()) / (reward.std() + 1e-5)

                #PPO 更新
                if (timestep+1) % batch_size == 0 or timestep == EP_LEN-1:
                    v_observation_ = critic.get_v(observation_)
                    discounted_r = []
                    for reward in buffer_r[::-1]:
                        v_observation_ = reward + gamma * v_observation_
                        discounted_r.append(v_observation_.detach().numpy())
                    discounted_r.reverse()
                    bs, ba, br,bap = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r),np.vstack(buffer_a_logp)
                    buffer_s, buffer_a, buffer_r,buffer_a_logp = [], [], [],[]
                    advantage=critic.learn(bs,br)#critic部分更新
                    actor.learn(bs,ba,advantage,bap)#actor部分更新
                    actor.update_oldpi()  # pi-new的参数赋给pi-old
                    # critic.learn(bs,br)
            if episode == 0:
                all_ep_r.append(reward_totle)
            else:
                all_ep_r.append(all_ep_r[-1] * 0.9 + reward_totle * 0.1)
            print("\rEp: {} |rewards: {}".format(episode, reward_totle), end="")
            #保存神经网络参数
            if episode % 50 == 0 and episode > 100:#保存神经网络参数
                save_data = {'net': actor.old_pi.state_dict(), 'opt': actor.optimizer.state_dict(), 'i': episode}
                torch.save(save_data, "D:\Study\PythonWorkSpace\RL\Algorithm\PPO2\PPO2_model_actor.pth")
                save_data = {'net': critic.critic_v.state_dict(), 'opt': critic.optimizer.state_dict(), 'i': episode}
                torch.save(save_data, "D:\Study\PythonWorkSpace\RL\Algorithm\PPO2\PPO2_model_critic.pth")

        env.close()
        plt.plot(np.arange(len(all_ep_r)), all_ep_r)
        plt.xlabel('Episode')
        plt.ylabel('Moving averaged episode reward')
        plt.show()

    else:
        print('PPO2测试中...')
        aa=Actor(state_num, action_num, min_action, max_action, Actor_Update_Steps, method)
        cc=Critic(state_num, 1, Critic_Update_Steps, lr=critic_lr)
        checkpoint_aa = torch.load("D:\Study\PythonWorkSpace\RL\Algorithm\PPO2\PPO2_model_actor.pth")
        aa.old_pi.load_state_dict(checkpoint_aa['net'])
        checkpoint_cc = torch.load("D:\Study\PythonWorkSpace\RL\Algorithm\PPO2\PPO2_model_critic.pth")
        cc.critic_v.load_state_dict(checkpoint_cc['net'])
        for j in range(10):
            state = env.reset()[0]
            total_rewards = 0
            for timestep in range(EP_LEN):
                env.render()
                action, action_logprob = aa.choose_action(state)
                new_state, reward, done, info, _ = env.step(action)  # 执行动作
                total_rewards += reward
                state = new_state
            print("Score: ", total_rewards)
        env.close()