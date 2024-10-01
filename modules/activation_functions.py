import numpy as np

class ReLU():
    def __init__(self):
        pass

    def forward(self, x): # x shape deve ser (1, din)
        self.x = x
        return np.maximum(0, x) # retorna 0 se x < 0, x se x >= 0
    
    def backward(self, gradout):
        din = dout = self.x.shape[1] # x.shape[1] é o din
        jacobian = np.zeros((dout, din))
        for i in range(din):
            if self.x[0, i] >= 0:
                jacobian[i, i] = 1

        return gradout @ jacobian

    def __call__(self, x):
        return self.forward(x)
    
    def load(self, path):
        pass # Não faz nada

    def save(self, path):
        pass # Não faz nada

class Softmax():
    def __init__(self):
        pass

    def forward(self, x):
        return np.exp(x) / np.exp(x).sum()
    
    def backward(self):
        raise NotImplementedError("Softmax backward not implemented")

    def __call__(self, x):
        return self.forward(x)
    
    def load(self, path):
        pass # Não faz nada

    def save(self, path):
        pass # Não faz nada

class LogSoftmax():
    def __init__(self):
        pass

    def forward(self, x):
        self.x = x
        return x - np.log(np.exp(x).sum())
    
    def backward(self, gradout):
        dout = din = self.x.shape[1]
        jacobian = np.eye(din)
        for row in range(dout):
            for col in range(din):
                jacobian[row, col] -= np.exp(self.x[0, col]) / np.exp(self.x).sum()

    def __call__(self, x):
        return self.forward(x)
    
    def load(self, path):
        pass # Não faz nada

    def save(self, path):
        pass # Não faz nada