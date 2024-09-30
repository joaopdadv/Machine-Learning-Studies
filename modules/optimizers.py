from mlp import CompoundNN, MLP

class Optimizer():
    def __init__(self, compound_nn: CompoundNN, lr=0.01):
        self.lr = lr
        self.compound_nn = compound_nn

    def step(self):
        
        for block in self.compound_nn.blocks:
            if block.__class__ == MLP:
                block.W = block.W - self.lr * block.deltaW
                block.b = block.b - self.lr * block.deltaB