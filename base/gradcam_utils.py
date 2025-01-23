import os
import sys
import random as random
import json
from collections import defaultdict
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import logomaker
from torchsummary import summary

from base.utils import createDataLoader, getSeqFromOnehot, my_pearsonr
from base.model import CNN
from base.run_utils import loadDict, setSeeds

random.seed(42)
np.random.seed(42)

class SaveFeatures:
    def __init__(self, module):
        self.features = None 
        self.hook = module.register_forward_hook(self.hook_fn)
    
    def hook_fn(self, module, input, output):
        self.features = output.clone()
    
    def close(self):
        self.hook.remove()

def plot_layer_gradcam(model, config, TestLoader, layer_idx, num_samples=1, verbose=False):
    """
    Plot GradCAM saliency map for each sequence in a given layer of the model
    """
    layer = model.CNN_layers[layer_idx][0]

    i = 0
    cnt = 0
    
    for (X, y) in TestLoader:
        # Create an instance of SaveFeatures
        hook = SaveFeatures(layer)
        
        X = X.clone().detach().requires_grad_(True)

        #forward the input through the network
        output = model(X.to(config["device"]))
            
        #backpropagate for computing gradient
        output.backward()

        #hook input features
        hook_features = hook.features.to(X.grad.device)

        #if it is the last CNN layer, upsample it to make it compatible for mult. with X
        if layer_idx == len(model.CNN_layers) - 1:
            hook_features = F.interpolate(hook_features, size=(X.shape[2], 1), mode="bilinear")

        grad_cam = hook_features.mean(dim=1, keepdim=True) * X.grad.unsqueeze(1)

        if verbose:
            print(f"============================ Layer: {layer_idx} ========================")
            print(f"{layer}")
            print(f"Predicted expression: {output} [True: {y}]")
            print(f"X.grad.shape\t\t\t\t\t: {X.grad.shape}")
            print(f"hook_features.shape\t\t\t\t: {hook_features.shape}")
            print(f"hook_features.mean(dim=1,keepdim=True).shape\t: {hook_features.mean(dim=1, keepdim=True).shape}")
            print(f"X.grad.unsqueeze(1).shape\t\t\t: {X.grad.unsqueeze(1).shape}")
            print(f"grad_cam.shape\t\t\t\t\t: {grad_cam.shape}")
            print(f"========================================================================")

        #grad_cam = F.relu(grad_cam)

        #plot
        heatmap = grad_cam.cpu().detach().numpy()[0, 0, :]
        heatmap /= np.max(heatmap)

        X_np = X.detach().numpy()
        s = getSeqFromOnehot(X_np)

        if layer_idx == 0:
            print("Input (one-hot):")
            plt.figure(figsize=(20,0.7))
            plt.imshow(X_np[0], cmap="seismic")
            plt.yticks(range(0,4), labels=["A","C","G","T"])
            plt.xticks(range(0,len(heatmap[2])), labels=s)
            plt.show()

        print(f"Layer {layer_idx}")
        plt.figure(figsize=(20,0.7))
        #plt.imshow(X_np[0], cmap="seismic")
        plt.imshow(heatmap, cmap="seismic")
        plt.yticks(range(0,4), labels=["A","C","G","T"])
        plt.xticks(range(0,len(heatmap[2])), labels=s)
        #plt.colorbar()
        plt.show()

        hook.close()
        
        cnt += 1

        #for how many samples to show grad cam
        if cnt > num_samples - 1:
            break
        
        i += 1

def plot_layerwise_logo(model, config, TestLoader, layer_idx, num_samples=1, save_dir=None, verbose=False):
    """
    Plot logo from GradCAM heatmap: To show effect of how different regions of a sequence are attended by the model
    """
    layer = model.CNN_layers[layer_idx][0]

    i = 0
    cnt = 0
    
    for (X, y) in TestLoader:
        hook = SaveFeatures(layer)
        X = X.clone().detach().requires_grad_(True)
        output = model(X.to(config["device"]))
        output.backward()
        hook_features = hook.features.to(X.grad.device)
        
        if layer_idx == len(model.CNN_layers) - 1:
            hook_features = F.interpolate(hook_features, size=(X.shape[2], 1), mode="bilinear")

        grad_cam = hook_features.mean(dim=1, keepdim=True) * X.grad.unsqueeze(1)

        if verbose:
            print(f"============================ Layer: {layer_idx} ========================")
            print(f"{layer}")
            print(f"Predicted expression: {output} [True: {y}]")
            print(f"X.grad.shape\t\t\t\t\t: {X.grad.shape}")
            print(f"hook_features.shape\t\t\t\t: {hook_features.shape}")
            print(f"hook_features.mean(dim=1,keepdim=True).shape\t: {hook_features.mean(dim=1, keepdim=True).shape}")
            print(f"X.grad.unsqueeze(1).shape\t\t\t: {X.grad.unsqueeze(1).shape}")
            print(f"grad_cam.shape\t\t\t\t\t: {grad_cam.shape}")
            print(f"========================================================================")

        heatmap = grad_cam.cpu().detach().numpy()[0, 0, :]
        X_np = X.detach().numpy()
        s = getSeqFromOnehot(X_np)

        if save_dir:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            file_path = f"{save_dir}/{s}_oe{y.item()}"

        print(f"Layer {layer_idx}")

        #Plot logo based on the saliency scores
        logomap = np.squeeze(heatmap)
        base_labels = ['A', 'C', 'G', 'T']
        pssm_data = pd.DataFrame(data=logomap.T, columns=base_labels)
        
        fig, ax = plt.subplots(figsize=(10, 3))
        logo = logomaker.Logo(pssm_data, ax=ax)
        logo.style_glyphs(shade_below_zero=True, fade_below_zero=True)
        logo.style_xticks(anchor=-1, spacing=5, rotation=0)
        ax.set_xlabel("Nucleotide position")
        ax.set_ylabel(f"GradCAM score")
        plt.tight_layout()
        if save_dir:
            plt.savefig(f"{file_path}.png", dpi=300, bbox_inches="tight")
        else:
            plt.show()

        # Plot the average gradcam score (saliency score)
        importance_data = pssm_data.mean(axis=1)
        importance_pssm = pd.DataFrame(0.0, index=range(0, len(s)), columns=['A', 'C', 'G', 'T'])

        for i in range(0, len(s)):
            original_nucleotide = s[i]
            importance_pssm.at[i, original_nucleotide] = importance_data[i]

        color_scheme = {'A': 'green', 'C': 'blue', 'G': 'orange', 'T': 'red'}
        fig, ax = plt.subplots(figsize=(10, 3))
        logo = logomaker.Logo(importance_pssm, color_scheme=color_scheme, ax=ax)
        logo.style_xticks(anchor=-1, spacing=5, rotation=0)
        ax.set_xlabel("Nucleotide position")
        ax.set_ylabel(f"Average GradCAM score")
        plt.tight_layout()
        if save_dir:
            #plt.savefig(f"{file_path}.avg.png", dpi=300, bbox_inches="tight")
            plt.savefig(f"{file_path}.avg.pdf", format="pdf", bbox_inches="tight")
            pssm_data.to_csv(f"{file_path}_pssm_data.csv", index=True)
            importance_pssm.to_csv(f"{file_path}_importance_pssm.csv", index=True)
        else:
            plt.show()

        #Plot GradCAM heatmap (saliency)
        heatmap /= np.max(heatmap)

        plt.figure(figsize=(15, 1.1))
        plt.imshow(heatmap, cmap="seismic", aspect='auto')
        plt.yticks(range(0, 4), labels=["A", "C", "G", "T"])
        plt.xticks(range(0, len(heatmap[0])), labels=s) #rotation=90)
        plt.colorbar()
        plt.tight_layout()
        if save_dir:
            #plt.savefig(f"{file_path}.heatmap.png", dpi=300, bbox_inches="tight")
            plt.savefig(f"{file_path}.heatmap.pdf", format="pdf", bbox_inches="tight")
        else:
            plt.show()
        
        plt.close("all")
        hook.close()
        
        cnt += 1

        #for how many samples to show grad cam
        if cnt > num_samples - 1:
            break
        
        i += 1

def plot_avg_gradcam(model, config, TestLoader, save_dir=None):
    for layer_idx in range(0,6):
        layer = model.CNN_layers[layer_idx][0]
        print(f"Layer {layer_idx}")
        
        heatmap_sum = defaultdict(float)

        i = 0
        for (X, y) in TestLoader:
            hook = SaveFeatures(layer)

            X = X.clone().detach().requires_grad_(True)
            output = model(X.to(config["device"]))
            output.backward()

            hook_features = hook.features.to(X.grad.device)

            if layer_idx == len(model.CNN_layers) - 1:
                hook_features = F.interpolate(hook_features, size=(X.shape[2], 1), mode="bilinear")

            grad_cam = hook_features.mean(dim=1, keepdim=True) * X.grad.unsqueeze(1)

            heatmap = grad_cam.cpu().detach().numpy()[0, 0, :]
            heatmap_sum[i] += heatmap

            hook.close()

            i += 1

        #avg heatmap across all sequences and make the saliency range [-1,1]
        avg_heatmap = np.mean([v for v in heatmap_sum.values()], axis=0)
        avg_heatmap /= np.max(avg_heatmap)

        #plot
        if save_dir:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            file_path = f"{save_dir}/{layer_idx}"
        plt.figure(figsize=(20, 0.7))
        plt.imshow(avg_heatmap, cmap="seismic")
        plt.yticks(range(0, 4), labels=['A','C','G','T'])
        plt.xticks(range(0, len(avg_heatmap[0]), 4))
        if save_dir:
            plt.savefig(f"{file_path}.png", dpi=300, bbox_inches="tight")
            plt.savefig(f"{file_path}.pdf", format="pdf", bbox_inches="tight")
        else:
            plt.show()
        plt.close("all")

