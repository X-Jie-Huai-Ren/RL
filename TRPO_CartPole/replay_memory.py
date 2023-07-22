


class ReplayMemory:
    
    def __init__(self) -> None:
        
        self.memory = []

    def push(self, sarsd):
        """
        sarsd: state, action, reward, next_state, done
        """
        self.memory.append(sarsd)

    def pop(self):
        """
        Return:
            返回一个episode的trajectory
        """
        return self.memory
    
    def __len__(self):
        return len(self.memory)