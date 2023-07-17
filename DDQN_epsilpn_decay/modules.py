from torch import nn


class MLP(nn.Module):

    def __init__(self, obser_size, num_act) -> None:
        super().__init__()
        self.mlp = self.__mlp(obser_size, num_act)

    def __mlp(self, obser_size, num_act):
        return nn.Sequential(
            nn.Linear(obser_size, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, num_act)
        )
    
    def forward(self, x):
        return self.mlp(x)