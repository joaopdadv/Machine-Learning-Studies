import numpy as np

class MSELoss():

    def forward(self, pred, true):
        self.pred = pred
        self.true = true
        return ((pred - true) ** 2).mean()
    
    def __call__(self, pred, true):
        return self.forward(pred, true)
    
    def backward(self):
        din = self.pred.shape[1]
        jacobian = 2 * (self.pred - self.true) * 1 / din

        return jacobian
    
class NLLLoss():
    
    def forward(self, pred, true):
        self.pred = pred
        self.true = true
        return - pred[0, true]
    
    def __call__(self, pred, true):
        return self.forward(pred, true)
    
    def backward(self):
        din = self.pred.shape[1]
        jacobian = np.zeros((1, din))
        jacobian[0, self.true] = -1

        return jacobian