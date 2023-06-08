from sklearn_som.som import SOM
from minisom import MiniSom
import numpy as np


class SOM_builder:
    def __init__(self, xdim, ydim, cols, radius, alpha, seed, minisom=False):
        self.xdim = xdim
        self.ydim = ydim
        self.cols = cols
        self.minisom = minisom
        if minisom:
            self.som = MiniSom(
                x=xdim, y=ydim, input_len=cols, learning_rate=alpha,
                sigma=radius, random_seed=seed, topology="hexagonal"
            )
        else:
            self.som = SOM(
                m=xdim, n=ydim, dim=cols, lr=alpha, sigma=radius,
                random_state=seed
            )

    def fit(self, data):
        if self.minisom:
            self.som.random_weights_init(data)
            self.som.train(
                data, verbose=True, num_iteration=2000
            )
            return self.som.get_weights()
        self.som.fit(data)
        return self.som.cluster_centers_

    def predict(self, data):
        if self.minisom:
            win_map = list(self.som.win_map(data).keys())
            x = []
            for d in data:
                winner = self.som.winner(d)
                x.append(win_map.index(winner))
            return x
        return self.som.predict(data)

