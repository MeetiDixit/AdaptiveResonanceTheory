import numpy as np
import pandas as pd
from numpy.linalg import norm

e = 0


class ART:
    def __init__(self, rho, M, N, theta=None):
        self.epochs = None
        self.ri = None  # calculated result
        self.reset = None
        self.theta = None
        self.qi = None
        self.wi = None
        self.pi = None
        self.rho = rho
        self.inputSize = M
        self.outputSize = N

        # self.weight = weight
        self.ii = np.zeros(self.inputSize)  #F0
        self.xi = np.zeros(self.inputSize)  #F1
        self.yj = np.zeros(self.outputSize) #F2
        self.activeClusters = []

        if theta is None:
            self.theta = 1 / np.sqrt(self.ii)
        else:
            self.theta = theta

        self.theta = self.theta.reshape(-1,1)

        self.ui = None
        self.vi = None

        self.params = {'a': 5, 'b': 5, 'c': 5, 'd': 5, 'e': 0}
        self.alpha = 0.005
        self.Tji = np.zeros((self.inputSize, self.outputSize))
        self.Bij = np.ones((self.outputSize, self.inputSize))*2.33
        # self.outputF1 = r
        # self.linFunc = f

    def makeF1(self, J=None, D=None):
        a = self.params['a']
        b = self.params['b']
        c = self.params['c']
        d = self.params['d']
        e = self.params['e']
        
        if self.vi is None:
            self.ui = np.zeros(self.inputSize)
        else:
            self.ui = self.vi / (e + norm(self.vi))

        self.wi = self.ii + a*(self.ui.reshape(-1, 1))

        # updating weights and activations
        if J is None:
            self.pi = self.ui
        else:
            self.pi = self.ui + d*self.Tji[J:]

        # TODO: consider RESET here... umm, ok!?
            
        self.xi = self.wi / (e+norm(self.wi))
        self.qi = self.pi / (e+norm(self.pi))

        # update vi
        self.vi = self.linFunc(self.xi) + b*self.linFunc(self.qi)
        return

    def linFunc(self, weight):
        weight2 = weight.copy()

        weight2[weight2 < self.theta] = 0
        return weight2

    def setZero(self):
        self.ii = np.zeros(self.inputSize)
        self.ui = np.zeros(self.inputSize)
        self.vi = np.zeros(self.inputSize)

    def updateWeights(self, J):
        alpha = self.alpha
        d = self.params['d']

        self.Tji[J, :] = alpha*d*self.ui + (1+alpha*d*(d-1))*self.Tji[J, :]
        self.Bij[:, J] = alpha*d*self.ui + (1+alpha*d*(d-1))*self.Bij[:, J]

        return

    def firstUpdateWts(self, J):
        d = self.params['d']
        self.Tji[J, :] = self.ui / (1-d)
        self.Bij[:, J] = self.ui / (1-d)
        return

    def computeJ(self):

        # TODO Check exceptions and edge cases
        self.reset = True
        J = None
        e = self.params['e']
        c = self.params['c']

        while self.reset:
            J = np.argmax(self.yj)
            if self.vi == 0:
                self.ui = np.zeros(self.inputSize)
            else:
                self.ui = self.vi / (e+norm(self.vi))

            # TODO Check exceptions and edge cases
            self.ri = (self.ui + c*self.pi) / (e + norm(self.ui) + norm(c*self.pi))

            if self.ri >= (self.rho - e):
                self.reset = False
                self.makeF1()

            else:
                self.reset = True
                self.yj[J] = -1

                # TODO Check exceptions and edge cases

        return J

    def resonance(self, J, epochs=20):
        for epoch in range(epochs):
            self.updateWeights(J)
            D = np.ones(self.outputSize)
            self.makeF1(J, D)
            # TODO Check exceptions and edge cases
        return True

    def learning(self, dat):
        self.setZero()
        self.ii = dat
        self.makeF1()
        self.makeF1()

        self.yj = np.dot(self.Bij.T, self.pi)
        J = self.computeJ()

        if len(self.activeClusters) == 0:
            self.updateWeights(J)
        else:
            self.resonance(J)
        if J not in self.activeClusters:
            self.activeClusters.append(J)

        return True

    def training(self, data):
        for dat in data:
            self.ii = dat
            self.learning(dat)
        return True

    def cleanInput(self):
        pass

    def trainingLoop(self, data):
        for epoch in range(self.epochs):
            self.training(data)
        return True


def main():
    df = pd.read_csv('data_banknote_authentication.txt')
    data = df.values
    # print(data)
    ip = ART(rho=0.003, M=data.shape[0], N=2)
    ip.learning(data)


if __name__ == '__main__':
    main()
