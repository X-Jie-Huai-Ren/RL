

import gym
import numpy as np
from torchvision import transforms
from model import PolicyNet











if __name__ == '__main__':

    env = gym.make("ALE/Assault-v5")

    channels_lst = [16, 32, 64, 128]

    model = PolicyNet(in_channels=3, channels_lst=channels_lst, action_dim=7)

    state = env.reset()[0]

    print(env.observation_space.shape)

    trans = transforms.ToTensor()
    input = trans(state)

    print(input.shape)

    out = model(input)
    print(out.shape)

    
        