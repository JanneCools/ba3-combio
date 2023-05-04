import numpy as np
from minisom import MiniSom
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

class SOM_Builder():
    def __init__(self, xdim, ydim):
        self.xdim = xdim
        self.ydim = ydim

    def som(self, fsom: np.ndarray, labels):
        som = MiniSom(self.xdim, self.ydim, labels)
        som.train(fsom, 100)
        print(f'quantization error: {som.quantization_error(fsom)}')
        grid = GridSpec(self.xdim, self.ydim)
        for x in range(self.xdim):
            for y in range(self.ydim):
                plt.subplot(grid[x, y], aspect=1)
                plt.pie([i if i > 0 else 0 for i in som.get_weights()[x][y]])
        #self.som = som
        plt.savefig("output2.png")
        plt.show()
        return som
