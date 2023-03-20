import matplotlib.patches as mptchs
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from multiprocessing import Process, Queue, cpu_count
from typing import Union


def man_dist_pbc(m: np.ndarray, vector: np.ndarray, shape: tuple = (10, 10)) -> np.ndarray:
    """Manhattan distance calculation of coordinates with periodic boundary condition
    :param m: array / matrix (reference)
    :type m: np.ndarray
    :param vector: array / vector (target)
    :type vector: np.ndarray
    :param shape: shape of the full SOM
    :type shape: tuple, optional
    :return: Manhattan distance for v to m
    :rtype: np.ndarray
    """
    dims = np.array(shape)
    delta = np.abs(m - vector)
    delta = np.where(delta > 0.5 * dims, np.abs(delta - dims), delta)
    return np.sum(delta, axis=len(m.shape) - 1)


class SOM(object):
    """
    Class implementing a self-organizing map with periodic boundary conditions. It has the following methods:
    """

    def __init__(self, x: int, y: int, max_epoch: int, alpha_start: float = 0.6, 
                 sigma_start: float = None, seed: int = None,):
        """Initialize the SOM object with a given map size and training conditions
        :param x: width of the map
        :type x: int
        :param y: height of the map
        :type y: int
        :param alpha_start: initial alpha (learning rate) at training start
        :type alpha_start: float
        :param sigma_start: initial sigma (restraint / neighborhood function) at training start; if `None`: x / 2
        :type sigma_start: float
        :param seed: random seed to use
        :type seed: int
        """
        np.random.seed(seed)
        self.x = x
        self.y = y
        self.shape = (x, y)
        if sigma_start:
            self.sigma_start = sigma_start
        else:
            self.sigma_start = x / 2.0
        self.alpha_start = alpha_start
        self.alpha = self.alpha_start
        self.sigma = self.sigma_start
        self.epoch = 0
        self.max_epoch = max_epoch
        self.interval = 0
        self.map = np.array([])
        self.indxmap = np.stack(np.unravel_index(np.arange(x * y, dtype=int).reshape(x, y), (x, y)), 2)
        self.distmap = np.zeros((self.x, self.y))
        self.winner_indices = np.array([])
        self.inizialized = False
        self.error = 0.0  # reconstruction error
        self.history = []  # reconstruction error training history

    def initialize(self, data: np.ndarray):
        """Initialize the SOM neurons
        :param data: data to use for initialization
        :type data: numpy.ndarray
        :param how: how to initialize the map, available: `pca` (via 4 first eigenvalues) or `random` (via random
            values normally distributed in the shape of `data`)
        :type how: str
        :return: initialized map in :py:attr:`SOM.map`
        """
        self.map = np.random.normal(np.mean(data), np.std(data), size=(self.x, self.y, len(data[0])))
        self.inizialized = True

    def winner(self, vector: np.ndarray) -> np.ndarray:
        """Compute the winner neuron closest to the vector (Euclidean distance)
        :param vector: vector of current data point(s)
        :type vector: np.ndarray
        :return: indices of winning neuron
        :rtype: np.ndarray
        """
        indx = np.argmin(np.sum((self.map - vector) ** 2, axis=2))
        return np.array([indx // self.y, indx % self.y])

    def cycle(self, vector: np.ndarray, verbose: bool = True):
        """Perform one iteration in adapting the SOM towards a chosen data point
        :param vector: current data point
        :type vector: np.ndarray
        :param verbose: verbosity control
        :type verbose: bool
        """

        if self.epoch >= self.max_epoch:
            raise ValueError("No more epochs")

        w = self.winner(vector)
        # get Manhattan distance (with PBC) of every neuron in the map to the winner
        dists = man_dist_pbc(self.indxmap, w, self.shape)
        # smooth the distances with the current sigma
        h = np.exp(-((dists / self.sigma) ** 2)).reshape(self.x, self.y, 1)
        # update neuron weights
        self.map -= h * self.alpha * (self.map - vector)

        if verbose:
            print(
                "Epoch %i;    Neuron [%i, %i];    \tSigma: %.4f;    alpha: %.4f"
                % (self.epoch, w[0], w[1], self.sigma, self.alpha)
            )
        self.epoch = self.epoch + 1

    def fit(
        self,
        data: np.ndarray,
        verbose: int = False
    ):
        """Train the SOM on the given data for several iterations
        :param data: data to train on
        :type data: np.ndarray
        :param epochs: number of iterations to train; if 0, epochs=len(data) and every data point is used once
        :type epochs: int, optional
        :param save_e: whether to save the error history
        :type save_e: bool, optional
        :param interval: interval of epochs to use for saving training errors
        :type interval: int, optional
        :param decay: type of decay for alpha and sigma. Choose from 'hill' (Hill function) and 'linear', with
            'hill' having the form ``y = 1 / (1 + (x / 0.5) **4)``
        :type decay: str, optional
        :param verbose: verbosity control
        :type verbose: bool
        """
        if not self.inizialized:
            self.initialize(data)
        else:
            idx = np.random.choice(np.arange(len(data)))

        # get alpha and sigma decays for given number of epochs or for hill decay
     
        self.alpha = self.alpha_start * (1 - self.epoch / self.max_epoch)
        self.sigma = self.sigma_start * (1 - self.epoch / self.max_epoch)

        self.cycle(data[idx], verbose=verbose)

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data in to the SOM space
        :param data: data to be transformed
        :type data: np.ndarray
        :return: transformed data in the SOM space
        :rtype: np.ndarray
        """
        m = self.map.reshape((self.x * self.y, self.map.shape[-1]))
        dotprod = np.exp(data).dot(np.exp(m.T)) / np.exp(m).sum(axis=1)
        return (dotprod / (np.exp(dotprod.max()) + 1e-8)).reshape(data.shape[0], self.x, self.y)

    def distance_map(self, metric: str = "euclidean"):
        """Get the distance map of the neuron weights. Every cell is the normalised average of all distances between
        the neuron and all other neurons.
        :param metric: distance metric to be used (see ``scipy.spatial.distance.cdist``)
        :type metric: str
        :return: normalized sum of distances for every neuron to its neighbors, stored in :py:attr:`SOM.distmap`
        """
        dists = np.zeros((self.x, self.y))
        for x in range(self.x):
            for y in range(self.y):
                d = cdist(self.map[x, y].reshape((1, -1)), self.map.reshape((-1, self.map.shape[-1])), metric=metric)
                dists[x, y] = np.mean(d)
        self.distmap = dists / dists.max()

    def winner_map(self, data: np.ndarray) -> np.ndarray:
        """Get the number of times, a certain neuron in the trained SOM is the winner for the given data.
        :param data: data to compute the winner neurons on
        :type data: np.ndarray
        :return: map with winner counts at corresponding neuron location
        :rtype: np.ndarray
        """
        wm = np.zeros(self.shape, dtype=int)
        for d in data:
            [x, y] = self.winner(d)
            wm[x, y] += 1
        return wm

    def _one_winner_neuron(self, data: np.ndarray, q: Queue):
        """Private function to be used for parallel winner neuron computation
        :param data: data matrix to compute the winner neurons on
        :type data: np.ndarray
        :param q: queue
        :type q: multiprocessing.Queue
        :return: winner neuron cooridnates for every datapoint (see :py:method:`SOM.winner_neurons`)
        """
        q.put(np.array([self.winner(d) for d in data], dtype="int"))

    def winner_neurons(self, data: np.ndarray) -> np.ndarray:
        """For every datapoint, get the winner neuron coordinates.
        :param data: data to compute the winner neurons on
        :type data: np.ndarray
        :return: winner neuron coordinates for every datapoint
        :rtype: np.ndarray
        """
        queue = Queue()
        n = cpu_count() - 1
        for d in np.array_split(np.array(data), n):
            p = Process(
                target=self._one_winner_neuron,
                args=(
                    d,
                    queue,
                ),
            )
            p.start()
        rslt = []
        for _ in range(n):
            rslt.extend(queue.get(timeout=10))
        self.winner_indices = np.array(rslt, dtype="int").reshape((len(data), 2))
        return self.winner_indices


    def get_neighbors(self, datapoint: np.ndarray, data: np.ndarray, labels: np.ndarray, d: int = 0) -> np.ndarray:
        """return the labels of the neighboring data instances at distance `d` for a given data point of interest
        :param datapoint: descriptor vector of the data point of interest to check for neighbors
        :type datapoint: np.ndarray
        :param data: reference data to compare `datapoint` to
        :type data: np.ndarray
        :param labels: array of labels describing the target classes for every data point in `data`
        :type labels: np.ndarray
        :param d: length of Manhattan distance to explore the neighborhood (0: same neuron as data point)
        :type d: int
        :return: found neighbors (labels)
        :rtype: np.ndarray
        """
        if not len(self.winner_indices):
            _ = self.winner_neurons(data)
        labels = np.array(labels)
        w = self.winner(datapoint)
        dists = np.array([man_dist_pbc(winner, w, self.shape) for winner in self.winner_indices]).flatten()
        return labels[np.where(dists <= d)[0]]
    
def random_gaussians(n_clusters: int, n_points: int, 
                     min_x=0, max_x=10, min_y=0, max_y=10,
                     scale: int =1, seed=14) -> np.ndarray:
    np.random.seed(seed)
    centroids_x = list(np.random.uniform(min_x, max_x, size=n_clusters))
    centroids_y = list(np.random.uniform(min_y, max_y, size=n_clusters))

    points = []
    for x_c, y_c in zip(centroids_x, centroids_y):
        points.append(np.random.normal(loc=(x_c, y_c), scale=scale, size=(n_points, 2)))
    
    return np.concatenate(points, axis=0)

def random_gaussians_on_circle(n_clusters: int, n_points: int, 
                               radius=10,
                               scale: int =1, seed=14) -> np.ndarray:
    
    np.random.seed(seed)
    angles = np.random.uniform(0, 2 * np.pi, n_clusters)
    points = []
    for phi in list(angles):
        x_c = radius * np.cos(phi)
        y_c = radius * np.sin(phi)

        points.append(
            np.random.normal(loc=(x_c, y_c), scale=scale, size=(n_points, 2)))

    return np.concatenate(points, axis=0)

def plot_map(som, df, show_circles=False):

    min_x = df[:, 0].min()
    min_y = df[:, 1].min()

    images = []
    text = "Epoch {}\n eta {:.2f}\n sigma {:.2f}".format(som.epoch, som.alpha, som.sigma)
    images.append(
        plt.text(
            s=text, x=min_x, y=min_y
        )
    )
    for i in range(som.map.shape[0]):
        for j in range(som.map.shape[1]):
            images.append(
                plt.plot(som.map[:, j, 0], som.map[:, j, 1], color='b', lw=1)[0])
            images.append(
                plt.plot(som.map[i, :, 0], som.map[i, :, 1], color='b', lw=1)[0])
    flattened = som.map.reshape(-1, 2)
    cmap = plt.cm.viridis  # define the colormap
    # extract all colors from the .jet map
    cmap_n = cmap.N
    N = len(flattened)
    step = cmap_n // N
    shape_x, shape_y = som.map.shape[:-1]
    
    winners = som.winner_neurons(df)
    winners_id = list(winners[:, 1] + winners[:, 0] * shape_y)
    colors = [cmap(step * i) for i in range(N)]
    scatter_colors = [colors[i] for i in winners_id]
    images.append(plt.scatter(df[:, 0], df[:, 1], s=30, c=scatter_colors))

    ax = plt.scatter(flattened[:, 0], flattened[:, 1], marker='x', c=colors, s=100)
    # title = plt.title(text)
    images.append(ax)
    # images.append(title)

    return images

from matplotlib import animation
from tqdm.notebook import tqdm

def create_animation(data, x: int, y: int, max_epoch: int, alpha_start: float = 0.6, 
                     sigma_start: float = None, seed: int = None, interval_fig=10,
                     interval_animation=100
                     ):
    som = SOM(x, y, max_epoch=max_epoch, alpha_start=alpha_start,
              sigma_start=sigma_start, seed=seed)
    som.initialize(data)

    fig = plt.figure(figsize=(10, 7), dpi=1920 / 16)
    imgs = []
    for i in tqdm(range(max_epoch), desc="Epoch"):
        som.fit(data)
        if i % interval_fig == 0:
            imgs.append(
                plot_map(som, data)
            )

    ani = animation.ArtistAnimation(fig, imgs, interval=interval_animation, blit=False)

    plt.close()

    return ani

def generate_two_parabolas(
        n_points, min_x, max_x, a_1=1, a_2=1.3, c_diff=5, 
        distortion_scale=10, train_proportion=.7, val_proportion= .2, seed=None):
    
    np.random.seed(seed=seed)
    x_1 = np.random.uniform(min_x, max_x, n_points)
    x_2 = np.random.uniform(min_x, max_x, n_points)

    с_1 = 0
    с_2 = с_1 + c_diff
    y_1 = a_1 * x_1 ** 2 + с_1 + np.random.normal(0, distortion_scale, n_points)
    y_2 = a_2 * x_2 ** 2 + с_2 + np.random.normal(0, distortion_scale, n_points)

    x = np.c_[x_1, y_1]
    y = np.c_[x_2, y_2]

    X = np.r_[x, y]
    Y = np.r_[np.zeros_like(x_1), np.ones_like(x_2)]

    idx = np.arange(0, 2 * n_points)
    np.random.shuffle(idx)

    train_idx, validate_idx, test_idx = np.split(
        idx, [int(train_proportion * len(idx)), 
            int((val_proportion + train_proportion) * len(idx))])
    
    train_idx, validate_idx, test_idx = np.split(
        idx, [int(train_proportion * len(idx)), 
            int((val_proportion + train_proportion) * len(idx))])
    
    return (
        X[train_idx].copy(), Y[train_idx].copy(), 
        X[validate_idx].copy(), Y[validate_idx].copy(),
        X[test_idx].copy(), Y[test_idx].copy(),
    )

def random_gaussians_classes(
        n_clusters: int, n_points: int, 
        min_x=0, max_x=10, min_y=0, max_y=10,
        scale: int =1,  train_proportion=.7, val_proportion= .2,
        seed=14) -> np.ndarray:

    np.random.seed(seed)
    centroids_x = list(np.random.uniform(min_x, max_x, size=n_clusters))
    centroids_y = list(np.random.uniform(min_y, max_y, size=n_clusters))

    points = []
    classes = []
    for i, (x_c, y_c) in enumerate(zip(centroids_x, centroids_y)):
        s = np.random.normal(loc=(x_c, y_c), scale=scale, size=(n_points, 2))
        points.append(s)
        classes.append(i * np.ones_like(s[:, 0]))
    
    X = np.concatenate(points, axis=0)
    Y = np.concatenate(classes, axis=0)
    idx = np.arange(0, n_clusters * n_points)
    np.random.shuffle(idx)

    train_idx, validate_idx, test_idx = np.split(
        idx, [int(train_proportion * len(idx)), 
            int((val_proportion + train_proportion) * len(idx))])
    
    train_idx, validate_idx, test_idx = np.split(
        idx, [int(train_proportion * len(idx)), 
            int((val_proportion + train_proportion) * len(idx))])
    
    
    return (
        X[train_idx].copy(), Y[train_idx].copy(), 
        X[validate_idx].copy(), Y[validate_idx].copy(),
        X[test_idx].copy(), Y[test_idx].copy(),
    )


def plot_data_label(X_train, y_train, X_val, y_val, X_test, y_test):
    fig = plt.figure(figsize=(10, 7), dpi=1920 / 16)

    cmap = plt.cm.viridis  # define the colormap
    # extract all colors from the .jet map
    cmap_n = cmap.N
    N = len(np.unique(y_train))
    step = cmap_n // N

    colors = [cmap(step * i) for i in range(N)]
    colors_train = [colors[int(i)] for i in list(y_train)]
    colors_val = [colors[int(i)] for i in list(y_val)]
    colors_test = [colors[int(i)] for i in list(y_test)]


    def _correct_color(color_rgb: list, alpha):
        return list(color_rgb)[: -1] + [alpha]

    plt.scatter(X_train[:, 0], X_train[:, 1], s=10, 
                c=colors_train, marker='o', label="train",)
    plt.scatter(X_val[:, 0], X_val[:, 1], 
                s=50, c=colors_val, marker="v", label="val")
    plt.scatter(X_test[:, 0], X_test[:, 1], 
                s=50, c=colors_test, marker="x", label="test")


def plot_data_grnn(X_train, y_train, X_val, y_val, X_test, y_test, grnn, scale=0.3):
    fig = plt.figure(figsize=(10, 7), dpi=1920 / 16)

    cmap = plt.cm.viridis  # define the colormap
    # extract all colors from the .jet map
    cmap_n = cmap.N
    N = len(np.unique(y_train))
    step = cmap_n // N
    scale_x = scale * (X_train[:, 0].max() - X_train[:, 0].min())
    scale_y = scale * (X_train[:, 1].max() - X_train[:, 1].min())
    xs = np.linspace(X_train[:, 0].min() - scale_x, X_train[:, 0].max() + scale_x, 100)
    ys = np.linspace(X_train[:, 1].min() - scale_y, X_train[:, 1].max() + scale_y, 100)

    xx, yy = np.meshgrid(xs, ys)
    grid = np.c_[xx.flatten(), yy.flatten()]
    grid_preds = grnn.predict(grid)

    test_preds = grnn.predict(X_test)
    y_test1 = pd.get_dummies(y_test).values
    mse_test = ((y_test1 - test_preds) ** 2).mean()

    val_preds = grnn.predict(X_val)
    y_val1 = pd.get_dummies(y_val).values
    mse_val = ((y_val1 - val_preds) ** 2).mean()

    grid_corrected = np.where(~np.isfinite(grid_preds) | np.isnan(grid_preds), 0, grid_preds)
    grid_color_argmax = grid_corrected.argmax(axis=1)

    grid_color_sum = grid_corrected.sum(axis=1)

    grid_color = np.where(grid_color_sum < 1e-4, -1, grid_color_argmax)
    

    colors = [cmap(step * i) for i in range(N)]
    colors_train = [colors[int(i)] for i in list(y_train)]
    colors_val = [colors[int(i)] for i in list(y_val)]
    colors_test = [colors[int(i)] for i in list(y_test)]


    def _correct_color(color_rgb: list, alpha):
        return list(color_rgb)[: -1] + [alpha]
    grid_colors = [
        [0, 0, 0, 1] if i == -1 else _correct_color(colors[int(i)], .3) 
        for i in list(grid_color)]

    grid_colors = np.array(grid_colors).reshape((*xx.shape, 4))

    plt.pcolormesh(xx, yy, grid_colors, shading='nearest', alpha=.4, edgecolor=(1.0, 1.0, 1.0, 0.3), linewidth=0.0015625)


    plt.scatter(X_train[:, 0], X_train[:, 1], s=10, 
                c=colors_train, marker='o', label="train",)
    plt.scatter(X_val[:, 0], X_val[:, 1], 
                s=50, c=colors_val, marker="v", label="val")
    plt.scatter(X_test[:, 0], X_test[:, 1], 
                s=50, c=colors_test, marker="x", label="test")
    plt.title("Sigma = {:.3f}\nMSE val = {:.3f}\nMSE test = {:.3f}".format(
        grnn.sigma, mse_val, mse_test))

import pandas as pd

class GRNN:
    def __init__(self, X, y, sigma=.1):
        self.w_e = X.copy()
        self.w_o = y.copy()
        self.sigma = sigma

    def predict(self, X):

        z = np.exp(
            -((self.w_e - X.reshape(X.shape[0], 1, X.shape[1])) ** 2 / self.sigma ** 2).sum(axis=-1))
        
        y_k = (z @ self.w_o) / z.sum(axis=-1).reshape(-1, 1)

        return y_k

def optinize_grnn(X, y, X_hld, y_hld, min_sigma, max_sigma, n):

    y1 = pd.get_dummies(y).values
    y_hld1 = pd.get_dummies(y_hld).values

    sigmas = list(np.linspace(min_sigma, max_sigma, n))

    min_err = np.inf
    errors = []

    best_sigma = None

    for sigma in tqdm(sigmas):
        grnn = GRNN(X, y1, sigma=sigma)
        preds = grnn.predict(X_hld)
        err = ((preds - y_hld1) ** 2).mean()
        errors.append(err)

        if err < min_err:
            min_err = err
            best_sigma = sigma

    return sigmas, errors, min_err, best_sigma

def create_plot_grnn(
        X_train, y_train, X_val, y_val, X_test, y_test,
        sigma, scale=1.3, optimize=False,
        min_sigma=.01, max_sigma=10, n=1000):
    
    if optimize:
        sigmas, errors, min_err, sigma = optinize_grnn(
            X_train, y_train, X_val, y_val, min_sigma=min_sigma,
             max_sigma=max_sigma, n=n
        )
    y_train1 = pd.get_dummies(y_train).values
    grnn = GRNN(X_train, y_train1, sigma=sigma)
    plot_data_grnn(X_train, y_train, X_val, y_val, X_test, y_test, grnn, scale=scale)

    if optimize:
        fig = plt.figure(figsize=(10, 7), dpi=1920 / 16)
        plt.plot(sigmas, errors)
        plt.title("Зависимость MSE от $\sigma$")
        plt.scatter([sigma], [min_err], color="red")
        plt.grid()