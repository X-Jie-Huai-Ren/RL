
import numpy as np

class Explorer:

    def __init__(self, n_act, eps_greedy, decay_rate) -> None:
        self.n_act = n_act
        self.eps = eps_greedy
        self.decay_rate = decay_rate


    def act(self, predict_method, obs):
        # 探索
        # 在ε-greedy算法中，需要一定的概率去探索新的action，防止策略陷入局部最优
        if np.random.uniform(0, 1) < self.eps:
            action = np.random.choice(self.n_act)
        # 利用
        else:
            action = predict_method(obs)

        # 衰减
        self.eps = max(0.01, self.eps-self.decay_rate)
        return action