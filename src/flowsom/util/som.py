import numpy as np
from sklearn_som.som import SOM
from minisom import MiniSom


class SOM_builder:
    def __init__(
        self,
        xdim: int,
        ydim: int,
        cols: int,
        radius: float,
        alpha: float,
        seed: int,
        minisom: bool = False,
    ):
        """
        Initialisation of the SOM_builder object
        :param xdim: the x dimension
        :param ydim: the y dimension
        :param cols: the amount of columns that are being used
        :param radius: the radius
        :param alpha: the learning rate
        :param seed: the seed for randomiz-sation
        :param minisom: whether to use minisom or sklearn-som
        """
        self.xdim = xdim
        self.ydim = ydim
        self.minisom = minisom
        if minisom:
            self.som = MiniSom(
                x=xdim,
                y=ydim,
                input_len=cols,
                learning_rate=alpha,
                sigma=radius,
                random_seed=seed,
                topology="hexagonal",
            )
        else:
            self.som = SOM(
                m=xdim, n=ydim, dim=cols, lr=alpha, sigma=radius, random_state=seed
            )

    def fit(self, data: np.ndarray):
        """
        Train the algorithm to generate a self-organising map of the data
        :param data: the data
        :return: the SOM-clusters of the given data
        """
        if self.minisom:
            self.som.random_weights_init(data)
            self.som.train(
                data,
                verbose=True,
                num_iteration=data.shape[1] * 2,
                random_order=True,
                use_epochs=True,
            )
            return self.som.get_weights()
        self.som.fit(data)
        return self.som.cluster_centers_

    def predict(self, data: np.ndarray):
        """
        Predict the SOM-clusters of the given data
        :param data: the data
        :return: the indices of the SOM-clusters
        """
        if self.minisom:
            win_map = list(self.som.win_map(data).keys())
            x = []
            for d in data:
                winner = self.som.winner(d)
                x.append(win_map.index(winner))
            return x
        return self.som.predict(data)
