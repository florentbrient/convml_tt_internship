
#!/usr/bin/env python
# coding: utf-8
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

import sklearn.decomposition
from sklearn.preprocessing import StandardScaler
from scipy.cluster import hierarchy as hc
from sklearn.cluster import KMeans
from ...data.dataset import ImageSingletDataset, TileType
import pickle


def kmeans(
    da_embeddings,
    ax=None,
    visualize=False,
    model_path=None,
    method=None,
    save=False
):
    """
    K-Means clustering.
    """

    tile_dataset = ImageSingletDataset(
        data_dir=da_embeddings.data_dir,
        tile_type=da_embeddings.tile_type,
        stage=da_embeddings.stage,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 3))
    else:
        fig = ax.figure

    if model_path is None:
        if method == "optimize":
            sse = []
            silhouette_coefficients = []
            for k in range(1, 11):
                kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
                clusters = kmeans.fit_predict(da_embeddings)
                sse.append(kmeans.inertia_)
                score = silhouette_score(da_embeddings, kmeans.labels_)
                silhouette_coefficients.append(score)
            if visualize:
                plt.style.use("fivethirtyeight")
                plt.plot(range(1, 11), sse)
                plt.xticks(range(1, 11))
                plt.xlabel("Number of Clusters")
                plt.ylabel("SSE")
                plt.show()

                plt.style.use("fivethirtyeight")
                plt.plot(range(2, 11), silhouette_coefficients)
                plt.xticks(range(2, 11))
                plt.xlabel("Number of Clusters")
                plt.ylabel("Silhouette Coefficient")
                plt.show()
        else:
            kmeans = KMeans(
                    init="random",
                    n_clusters=12,
                    n_init=10,
                    max_iter=300,
                    random_state=42 
                )
            clusters = kmeans.fit_predict(da_embeddings)
    else:
        kmeans = pickle.load(open("kmeans_model.pkl", "rb"))

    if save:
        pickle.dump(kmeans, open("kmeans_model.pkl", "wb"))    

    return clusters, kmeans