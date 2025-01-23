import os
from datetime import datetime
import random as random
import umap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torchsummary import summary
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from base.utils import createDataLoader
from base.model import CNN
#from base.model_basic import CNN
from base.model_utils import predict
from base.run_utils import loadDict, setSeeds
from base.plot_utils import scale_predictions

random.seed(42)
np.random.seed(42)

def plot_hist_tsne(stacked_layer_outputs, category, layer="linear", plot_hist=False, plot_tsne=True, color_palette="rocket_r", save_as=None):
    if layer != "linear":
        data = stacked_layer_outputs.view(stacked_layer_outputs.size(0), -1).cpu().detach().numpy()
    else:
        data = stacked_layer_outputs

    data = StandardScaler().fit_transform(data)

    if plot_hist:
        print("Plotting Histogram:")
        sns.histplot(data, kde=True)
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.title("Histogram of Flattened Tensor")
        plt.show()

    if plot_tsne:
        if stacked_layer_outputs.shape[1] == 1:
            n_components = 20
        else:
            n_components = 50
        pca = PCA(n_components=n_components, random_state=0)  
        data_pca = pca.fit_transform(data)

        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=0, verbose=0)  
        data_tsne = tsne.fit_transform(data_pca)

        sns.scatterplot(x=data_tsne[:, 0], y=data_tsne[:, 1], hue=category, palette=color_palette)
        plt.xlabel("t-SNE 1", fontsize=10)
        plt.ylabel("t-SNE 2", fontsize=10)
        plt.legend(loc="lower right")

        if save_as:
            plt.savefig(save_as, dpi=300, bbox_inches="tight")
            plt.savefig(f"{save_as.split('.')[0]}.pdf", format="pdf", bbox_inches="tight")
            df = pd.DataFrame({"tSNE_dim1": data_tsne[:, 0], "tSNE_dim2": data_tsne[:, 1], "Category": category})
            df.to_csv(f"{save_as.split('.')[0]}.csv", index=False)
            print(f"Files saved as: {save_as.split('.')[0]}.*")
        plt.show()
        plt.close('all')

def plot_hist_umap(stacked_layer_outputs, category, layer="linear", plot_hist=False, plot_umap=True, color_palette="rocket", save_as=None):
    if layer != "linear":
        data = stacked_layer_outputs.view(stacked_layer_outputs.size(0), -1).cpu().detach().numpy()
    else:
        data = stacked_layer_outputs

    data = StandardScaler().fit_transform(data)

    if plot_hist:
        print("Plotting Histogram:")
        sns.histplot(data, kde=True)
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.title("Histogram of Flattened Tensor")
        plt.show()

    if plot_umap:
        if stacked_layer_outputs.shape[1] == 1:
            n_components = 20
        else:
            n_components = 50
        pca = PCA(n_components=n_components, random_state=0)  
        data_pca = pca.fit_transform(data)

        umap_embedding = umap.UMAP(n_components=2, init="random").fit_transform(data_pca)

        sns.scatterplot(x=umap_embedding[:, 0], y=umap_embedding[:, 1], hue=category, palette=color_palette)
        plt.xlabel("UMAP 1", fontsize=10)
        plt.ylabel("UMAP 2", fontsize=10)
        plt.legend(loc="lower right")
        
        if save_as:
            plt.savefig(save_as, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close('all')