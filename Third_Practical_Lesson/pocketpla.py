import numpy as np


class PocketPLA:
    def get_w(self):
        return self.w

    def set_w(self, w):
        self.w = np.array(w, dtype=float).copy()

    def execute(self, _X, y):
        X = np.array(_X)

        # Usa os pesos recebidos ou começa com zeros
        if not hasattr(self, "w"):
            self.w = np.zeros(len(X[0]))
        else:
            self.w = self.w.copy()

        # Avalia a qualidade dos pesos iniciais
        bestError = self.errorIN(X, y)
        bestW = self.w.copy()

        for iteration in range(1000):
            for i in range(len(y)):
                if np.sign(np.dot(self.w, X[i])) != y[i]:
                    self.w = self.w + y[i] * X[i]

                    eIN = self.errorIN(X, y)

                    if bestError > eIN:
                        bestError = eIN
                        bestW = self.w.copy()

        self.w = bestW.copy()

    def getOriginalY(self, originalX):
        return (-self.w[0] - self.w[1] * originalX) / self.w[2]

    def h(self, x):
        return np.sign(np.dot(self.w, x))

    def errorIN(self, X, y):
        error = 0

        for i in range(len(y)):
            if np.sign(np.dot(self.w, X[i])) != y[i]:
                error += 1

        return error