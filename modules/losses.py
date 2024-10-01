import numpy as np

class MSELoss():

    def forward(self, pred, true):
        self.pred = pred
        self.true = true
        return ((pred - true) ** 2).mean(1).mean()
    
    def __call__(self, pred, true):
        return self.forward(pred, true)
    
    def backward(self):
        batch_size = self.pred.shape[0]
        din = self.pred.shape[1]
        jacobian = 2 * (self.pred - self.true) * 1 / din / batch_size

        return jacobian
    
class NLLLoss():
    
    def forward(self, pred, true):
        self.pred = pred
        self.true = true # (batch_size, 1))

        loss = 0
        for b in range(pred.shape[0]):
            loss -= pred[b, true[b]]

        return loss
    
    def __call__(self, pred, true):
        return self.forward(pred, true)
    
    def backward(self):
        din = self.pred.shape[1]
        batch_size = self.pred.shape[0]
        jacobian = np.zeros((batch_size, din))

        for b in range(batch_size):
            jacobian[b, self.true[b]] = -1

        return jacobian # (batch_size, din)