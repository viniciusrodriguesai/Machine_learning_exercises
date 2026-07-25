from linearregression import LinearRegression
from pocketpla import PocketPLA
import numpy as np


class LRClassifier:
    def execute(self, _X, _y):
        # Encontra pesos iniciais por regressão linear
        lr = LinearRegression()
        lr.execute(_X, _y)
        self.w = lr.get_w()

        # Refina os pesos utilizando o Pocket PLA
        pla = PocketPLA()
        pla.set_w(self.w)
        pla.execute(_X, _y)
        self.w = pla.get_w()

    def predict(self, x_):
        return np.sign([
            np.dot(self.w, x)
            for x in x_
        ])

    def getRegressionY(self, regressionX, shift=0):
        return (
            -self.w[0]
            + shift
            - self.w[1] * regressionX
        ) / self.w[2]