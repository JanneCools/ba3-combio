import numpy as np
from minisom import MiniSom

from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform


class SOM_Builder():
    def __init__(self, xdim, ydim):
        self.xdim = xdim
        self.ydim = ydim
        #self.som = None

    def build(self, fsom: np.ndarray, num_labels):
        # bepaal radius
        nhbrdist = squareform(pdist([(x,y) for x in range(self.xdim) for y in range(self.ydim)], metric="chebyshev"))
        radius = np.quantile(nhbrdist, 0.67)

        som = MiniSom(
            x=self.xdim, y=self.ydim, input_len=num_labels, sigma=radius, learning_rate=0.05
        )
        som.train(fsom, 100)
        print(som.get_weights())
        print(f'quantization error: {som.quantization_error(fsom)}')
        grid = GridSpec(self.xdim, self.ydim)
        for x in range(self.xdim):
            for y in range(self.ydim):
                plt.subplot(grid[x, y], aspect=1)
                plt.pie([i if i > 0 else 0 for i in som.get_weights()[x][y]])
        #self.som = som
        #plt.savefig("output2.png")
        plt.show()
        return som.get_weights()


