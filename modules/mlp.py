import numpy as np

class MLP():
    def __init__(self, din, dout):
        self.W = np.random.randn(dout, din)
        self.b = np.random.randn(dout) # bias array tem que ter o mesmo tamanho que a saída

    def forward(self, x): # x shape deve ser (1, din)
        # x é um vetor de entrada
        # @ é multiplicação de matrizes no numpy
        # .T é a transposta   
        return x @ self.W.T + self.b
    
    def __call__(self, x):
        return self.forward(x)
    
class CompoundNN():

    def __init__(self, blocks:list):
        self.blocks = blocks

    def forward(self, x):
        for block in self.blocks:
            x = block.forward(x)
        return x

    def __call__(self, x):
        return self.forward(x)