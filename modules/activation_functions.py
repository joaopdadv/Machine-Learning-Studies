import numpy as np
from scipy.special import logsumexp

class ReLU():
    def __init__(self):
        pass

    def forward(self, x): # x shape deve ser (1, din)
        self.x = x
        return np.maximum(0, x) # retorna 0 se x < 0, x se x >= 0
    
    def backward(self, gradout):
        # din = dout = self.x.shape[1] # x.shape[1] é o din

        # all_jacobians = []
        # for b in range(self.x.shape[0]):
        #     jacobian = np.zeros((dout, din))
        #     for i in range(din):
        #         if self.x[b, i] >= 0:
        #             jacobian[i, i] = 1
        #     all_jacobians.append(jacobian)

        # return np.concatenate([(gradout[b][None, ...] @ jacobian)[None, ...] for b, jacobian in enumerate(all_jacobians)])

        new_grad = gradout.copy()
        new_grad[self.x < 0] = 0.
        return new_grad

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
        # return x - np.log(np.exp(x).sum())
        return x - logsumexp(x, axis=1)[..., None]
    
    def backward(self, gradout):
        # dout = din = self.x.shape[1]

        # all_jacobians = []
        # for b in range(self.x.shape[0]):
        #     jacobian = np.eye(din)
        #     for row in range(dout):
        #         for col in range(din):
        #             jacobian[row, col] -= np.exp(self.x[b, col]) / np.sum(np.exp(self.x[b]))
        #     all_jacobians.append(jacobian)

        # return np.concatenate([(gradout[b][None, ...] @ jacobian)[None, ...] for b, jacobian in enumerate(all_jacobians)])

        gradients = np.eye(self.x.shape[1])[None, ...]
        gradients = gradients - (np.exp(self.x) / np.exp(self.x).sum(1)[..., None])[..., None]
        return (np.matmul(gradients, gradout[..., None]))[:, :, 0]

    def __call__(self, x):
        return self.forward(x)
    
    def load(self, path):
        pass # Não faz nada

    def save(self, path):
        pass # Não faz nada