
import gym
import numpy as np
import torch












if __name__ == '__main__':

    # 搭建CartPole环境
    env = gym.make('Pendulum-v1')

    # 状态维度: sin, cos, 角速度
    state_num = env.observation_space.shape[0]
    # 动作维度: 力矩, [-2, 2]之间的任意实数
    action_num = env.action_space.shape[0]
    
    state = env.reset()[0]
    print(state.shape)
    print(torch.cuda.is_available())

    

