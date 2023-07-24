


from collections import namedtuple


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'action_logprob'))



class Buffer: 
    
    def __init__(self, capacity) -> None:
        """
        Params: 
            capacity: int, 缓冲区的容量
        """
        self.capacity = capacity
        self.buffer = []
    
    def push(self, batch):
        self.buffer.append(batch)
        if len(self.buffer) > self.capacity:
            del self.buffer[0]
    
    def pop(self):
        return self.buffer
    
    def __len__(self):
        return len(self.buffer)