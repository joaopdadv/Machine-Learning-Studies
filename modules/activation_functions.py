import numpy as np

class ReLU():
    def __init__(self):
        pass

    def forward(self, x):
        return np.maximum(0, x) # retorna 0 se x < 0, x se x >= 0

    def __call__(self, x):
        return self.forward(x)
    
    def load(self, path):
        pass # Não faz nada

    def save(self, path):
        pass # Não faz nada