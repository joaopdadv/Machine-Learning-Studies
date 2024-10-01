import numpy as np

class MLP():
    def __init__(self, din, dout):
        # inicialização aleatória
        # self.W = np.random.randn(dout, din)
        # self.b = np.random.randn(dout) # bias array tem que ter o mesmo tamanho que a saída

        # inicialização com xavier
        self.W = (2 * np.random.randn(dout, din) - 1) * (np.sqrt(6) / np.sqrt(dout + din))
        self.b = (2 * np.random.randn(dout) - 1) * (np.sqrt(6) / np.sqrt(dout + din))

        self.dout = dout
        self.din = din

    def forward(self, x): # x shape deve ser (1, din)
        # x é um vetor de entrada
        # @ é multiplicação de matrizes no numpy
        # .T é a transposta   
        self.x = x # guarda x para usar no backward
        return x @ self.W.T + self.b
    
    def backward(self, gradout):
        # jacobian_x = self.W # jacobian_x é a derivada da saída com relação a x
        # jacobian_b = np.eye(self.dout) # jacobian_b é a derivada da saída com relação a b

        # all_jacobians_w = []
        # for b in range(self.x.shape[0]):
        #     jacobian_w = np.zeros((self.dout, self.din * self.dout)) # jacobian_w é a derivada da saída com relação a W. é uma matriz 2D que é zerada no inicio
        #     # Coloca os valores de jacobian_w, que são as entradas x, nas posições corretas 
        #     for i in range(self.dout):
        #         jacobian_w[i, i * self.din: (i+1) * self.din] = self.x[b]

        #     all_jacobians_w.append(jacobian_w)

        # self.deltaW = np.concatenate([(gradout[b][None, ...] @ jacobian_w).reshape(self.W.shape)[None, ...] for b, jacobian_w in enumerate(all_jacobians_w)]).sum(0)
        # self.deltaB = gradout @ jacobian_b # shape: dout

        self.deltaW = gradout.T @ self.x
        self.deltaB = gradout # shape: dout

        return gradout @ self.W
    
    def __call__(self, x):
        return self.forward(x)
    
    def load(self, path: str):
        self.W = np.load(path + '_W.npy')
        self.b = np.load(path + '_b.npy')

    def save(self, path: str):
        np.save('trainings/' + path + '_W', self.W)
        np.save('trainings/' + path + '_b', self.b)

    
class CompoundNN():

    def __init__(self, blocks:list):
        self.blocks = blocks

    def forward(self, x):
        for block in self.blocks:
            x = block.forward(x)
        return x
    
    def backward(self, gradout):
        for block in self.blocks[::-1]: # Inverte a ordem dos blocos pois tem que ir do fim para o inicio
            gradout = block.backward(gradout)
        return gradout

    def __call__(self, x):
        return self.forward(x)
    
    def load(self, path: str):
        for i, block in enumerate(self.blocks):
            block.load(path + f'_{i}')

    def save(self, path: str):
        for i, block in enumerate(self.blocks):
            block.save(path + f'_{i}')