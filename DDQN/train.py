import gym
from agents import DQNAgent
from modules import MLP
import torch
from replayBuffer import ReplayBuffer



class TrainManager:

    def __init__(self, env, episodes=1000, lr=0.001, gamma=0.9, eps_greedy=0.1, memory_size=2000, 
                 replay_start_size=200, batch_size=32, num_steps=4, update_target_steps=200) -> None:
        self.env = env
        self.episodes = episodes
        num_obs = env.observation_space.shape[0]
        num_act = env.action_space.n
        q_func = MLP(num_obs, num_act)
        optimizer = torch.optim.AdamW(q_func.parameters(), lr=lr)
        rb = ReplayBuffer(memory_size, num_steps=num_steps)
        self.agent = DQNAgent(
            q_func=q_func,
            replay_buffer=rb,
            batch_size=batch_size,
            replay_start_size = replay_start_size,
            optimizer=optimizer,
            n_acts=num_act,
            update_target_steps=update_target_steps,
            eps=eps_greedy,
            gamma=gamma
        )


    def train_episode(self):
        # 初始状态
        obs = self.env.reset()
        # 转换obs的类型为torch需要的张量形式
        # obs = torch.FloatTensor(obs)

        total_reward = 0
        
        while True:
            # 根据算法确定action
            action = self.agent.get_act(obs)
            # 与环境交互
            next_obs, reward, done, _ = self.env.step(action)
            # next_obs = torch.FloatTensor(next_obs)

            # 更新Q-table
            self.agent.learn(obs, action, reward, next_obs, done)

            obs = next_obs
            
            total_reward += reward

            if done:
                break

        return total_reward

    def test_episode(self):
        # 初始观察值
        obs = self.env.reset()
        # obs = torch.FloatTensor(obs)

        total_reward = 0
        
        while True:
            # 在测试中，无需探索
            action = self.agent.predict(obs)
            next_obs, reward, done, _ = self.env.step(action)
            # next_obs = torch.FloatTensor(next_obs)

            obs = next_obs
            total_reward += reward
            self.env.render()

            if done:
                break

        return total_reward
                



    def train(self):
        for episode in range(self.episodes):
            train_reward = self.train_episode()
            print('epidode:{}, train-reward:{}'.format(episode, train_reward))

            if episode % 100 == 0:

                test_reward = self.test_episode()
                print('epidode:{}, test-reward:{}'.format(episode, test_reward))





if __name__ == '__main__':
    env = gym.make("CartPole-v0")
    tm = TrainManager(env)
    tm.train()