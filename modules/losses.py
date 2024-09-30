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