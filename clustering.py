import numpy as np
from sklearn.cluster import DBSCAN


class Clustering:
    def __init__(self, eps: float = 8.6775, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples

    def clusterize(self, fvs):
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        fvs_np = np.array(fvs)
        clustering.fit(fvs_np)

        clusters = clustering.labels_

        noise_count = np.argwhere(clusters == -1).flatten().size
        noise_fraction = (noise_count / fvs_np.shape[0])*100
        print(f'noise count: {noise_count} ({noise_fraction} %)')

        return clusters

