
import numpy as np
import gym
import time
import gridworld

class SarsaAgent:
    def __init__(self, n_states, n_acts, eps=0.1, lr=0.1, gamma=0.9) -> None:
        self.n_acts = n_acts
        self.n_states = n_states
        self.Q = np.zeros((n_states, n_acts))
        # eps-greedy 探索概率
        self.eps = eps
        self.lr = lr
        self.gamma = gamma

    # 根据Q-table选择对应价值最大的action
    def select_opt_action(self, state):
        # 当前状态对应的一组action-value
        cur_state_q = self.Q[state, :]
        # 选取action-value最大的action
        action = np.random.choice(np.flatnonzero(cur_state_q==cur_state_q.max()))
        # action = np.argmax(cur_state_q)
        return action

    # 根据当前state选择action
    def get_act(self, state):
        # 探索
        # 在ε-greedy算法中，需要一定的概率去探索新的action，防止策略陷入局部最优
        if np.random.uniform(0, 1) < self.eps:
            action = np.random.choice(self.n_acts)
        # 利用
        else:
            action = self.select_opt_action(state)
        return action
        

    def learn(self, state, action, reward, next_state, next_action, done):
        # t时刻下的state的action-value
        current_Q = self.Q[state, action]
        # 判断是否走到终点
        if done:
            target = reward
        else:
            target = reward+self.gamma*self.Q[next_state, next_action]
        # 更新t时刻下的state的action-value
        self.Q[state, action] += self.lr*(target - current_Q)


def train_episode(env, agent, is_render):
    # 初始状态
    state = env.reset()
    # 初始动作
    action = agent.get_act(state)
    total_reward = 0
    
    while True:
        next_state, reward, done, _ = env.step(action)
        next_action = agent.get_act(next_state)

        # 更新Q-table
        agent.learn(state, action, reward, next_state, next_action, done)

        action = next_action
        state = next_state
        total_reward += reward
        if is_render:
            env.render()

        if done:
            break

    return total_reward

def test_episode(env, agent):
    # 初始状态
    state = env.reset()
    total_reward = 0
    
    while True:
        # 在测试中，无需探索
        action = agent.select_opt_action(state)
        next_state, reward, done, _ = env.step(action)

        state = next_state
        total_reward += reward
        env.render()
        time.sleep(0.5)

        if done:
            break

    return total_reward
            



def train(env, episodes=500, eps=0.1, lr=0.1, gamma=0.9):
    agent = SarsaAgent(
        n_states=env.observation_space.n,
        n_acts=env.action_space.n,
        eps=eps,
        lr=lr,
        gamma=gamma
    )
    is_render = False
    for episode in range(episodes):
        train_reward = train_episode(env, agent, is_render)
        print('epidode:{}, train-reward:{}'.format(episode, train_reward))

        if episode % 50 == 0:
            is_render = True
        else:
            is_render = False

    print('test:')
    test_reward = test_episode(env, agent)
    print('epidode:{}, test-reward:{}'.format(episode, test_reward))





if __name__ == '__main__':
    env = gym.make("CliffWalking-v0")
    env = gridworld.CliffWalkingWapper(env)
    train(env)