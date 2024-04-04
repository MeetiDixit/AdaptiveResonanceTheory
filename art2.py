# art2.py

import pandas as pd
import numpy as np
from numpy.linalg import norm


class ART2:
    def __init__(self, m, n, theta=None, rho=0.7) -> None:
        self.inputSize = m
        self.classes = n
        self.ii = None
        self.ui = None
        self.vi = None
        self.xi = None
        self.wi = None
        self.pi = None
        self.qi = None
        self.Zj = np.zeros((self.inputSize, self.classes))
        self.Bj = np.zeros((self.classes, self.inputSize))
        self.theta = theta

        self.params = {'a': 4, 'b': 2, 'c': 1, 'd': 7, 'e': 5}
        self.rho = rho
        self.yj = np.zeros(self.classes)
        self.alpha = 0.4

    def makeF1(self, J=None):
        a = self.params['a']
        b = self.params['b']
        e = self.params['e']
        self.setZero()
        if self.vi is None:
            self.ui = np.zeros(self.inputSize)
        else:
            self.ui = self.vi / (e + norm(self.vi))
        self.wi = self.ii + (a * self.ui)

        self.xi = self.wi / (e + norm(self.wi))

        self.pi = self.ui
        self.qi = self.pi / (e + norm(self.pi))

        self.vi = self.linearFunc(self.xi) + (b * self.linearFunc(self.qi))

        return

    def updateF1(self, J=None):
        a = self.params['a']
        b = self.params['b']
        d = self.params['d']
        e = self.params['e']
        if self.vi is None:
            self.ui = np.zeros(self.inputSize)
        else:
            self.ui = self.vi / (e + norm(self.vi))
        self.wi = self.ii + (a * self.ui)

        self.xi = self.wi / (e + norm(self.wi))

        if J is None:
            self.pi = self.ui
        else:
            self.pi = self.ui + d * self.Zj[J:]
        self.qi = self.pi / (e + norm(self.pi))

        self.vi = self.linearFunc(self.xi) + (b * self.linearFunc(self.qi))
        # print('ui shape', self.ui.shape)
        # print('vi shape', self.vi.shape)
        # print('wi shape', self.wi.shape)
        # print('pi shape', self.pi.shape)
        # print('qi shape', self.qi.shape)
        # print('xi shape', self.xi.shape)
        # print('xxxxxxxxxxxxx')

        return

    def linearFunc(self, weight):
        weight2 = weight.copy()
        weight2[weight2 < self.theta] = 0
        return weight2

    def setZero(self):
        self.ui = np.zeros(self.inputSize)
        self.vi = np.zeros(self.inputSize)
        return

    def updateWeights(self, J=None):
        alpha = self.alpha
        d = self.params['d']
        # print('ui shape', self.ui.shape)
        # print('Zj shape', self.Zj.shape)
        # self.Zj[J, :] = (alpha*d*self.ui) + (1+(alpha*d*(d-1)))*self.Zj[J, :]
        # self.Bj[:, J] = (alpha*d*self.ui) + (1+alpha*d*(d-1))*self.Bj[:, J]
        # self.Zj[]
        return

    def computeJ(self):
        self.reset = True
        J = None
        e = self.params['e']
        c = self.params['c']
        while self.reset:
            J = np.argmax(self.yj)
            print(J)
            if (self.vi == 0).all():
                self.ui = np.zeros(self.inputSize)
            else:
                self.ui = self.vi / (e + norm(self.vi))

            self.ri = (self.ui + (c * self.pi)) / (e + norm(self.ui) + norm(c * self.pi))
            print(self.ri)
            if norm(self.ri) < (self.rho - e):
                self.reset = True
                self.yj[J] = -1.0
            elif norm(self.ri) > (self.rho - e):
                self.reset = False
                self.updateF1()
        return self.ri

    def learning(self, data):
        for ii in data:
            self.ii = ii
            self.makeF1()
            self.updateF1()
            self.computeJ()
            self.updateWeights()
        return


def main():
    df = pd.read_csv('/Users/meeti/Downloads/art2/2/test1.txt', header=None)
    df = df.iloc[:, :-1]
    dfIp = df.values
    nn = ART2(m=4, n=2, rho=0.3, theta=0.1)
    nn.learning(dfIp)
    print('yay')


if __name__ == '__main__':
    main()
