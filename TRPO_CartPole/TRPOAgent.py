


from model import policyNet


class TRPOAgent:
    def __init__(self, state_dim, hidden_dim, action_dim) -> None:
        """
        Params:
            state_dim: int, 状态的维度
            hidden_dim: int, 隐藏单元的维度
            action_dim: int, 动作的维度
        """
        
        self.policyNet = policyNet(state_dim, hidden_dim, action_dim)

        # 