


import gym




class TrainManager:
    
    def __init__(self, env, agent, eps, num_episodes=10000) -> None:
        """
        Params:
            env: 环境
            agent: 构建的智能体
            eps: float, epslion-greedy算法的探索率
            num_episodes: int, 回合数
        """
        self.env = env
        self.agent = agent











if __name__ == '__main__':

    # 构建环境
    env = gym.make('CartPole-v1')

    state = env.reset()[0]

    print(state)