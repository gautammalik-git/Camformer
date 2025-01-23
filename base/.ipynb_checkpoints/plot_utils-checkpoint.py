import torch
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import numpy as np
import pandas as pd
import os
import sys
import re
from itertools import combinations
from base.utils import my_pearsonr, my_spearmanr, createDataLoader
from base.run_utils import loadDict, setSeeds
from base.model import CNN
from base.model_utils import predict

def plot_density(X,Y):
    sns.kdeplot(X)
    sns.despine()
    sns.kdeplot(Y)
    sns.despine()
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.title('Kernel Density Plot')
    plt.show()


def plot_QQ(X,Y):
    # Create a Q-Q plot
    sm.qqplot(X, line='s')
    plt.title('Q-Q Plot for Sample 1')
    plt.show()

    sm.qqplot(Y, line='s')
    plt.title('Q-Q Plot for Sample 2')
    plt.show()

    # Compare two samples in a Q-Q plot
    sm.qqplot_2samples(X, Y)
    plt.title('Q-Q Plot: Sample 1 vs Sample 2')
    plt.show()


def plot_scatter(x, y, xlabel='Expr (true)', ylabel='Expr (pred)', normalise=False):
    if normalise:
        #x = normalize_list(x)
        #y = normalize_list(y)
        y = scale_predictions(y, x)
    ax = sns.regplot(x = x, y = y,
                     scatter_kws = {'color': 'blue', 'alpha': 0.01},
                     line_kws = {'color': 'black', 'linestyle':'--'})
    n = len(x)
    r, p = stats.pearsonr(x=x, y=y)
    rho, _ = stats.spearmanr(a=x, b=y)
    ax.set_xlabel(xlabel, fontsize=10) 
    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    #ax.set_title(f'r={r:.2}, rho={rho:.4}', fontsize=10)
    plt.title(f'$n=${n} | $r=${r:.3f} | $p=${p:.4f} | $R^2=${r**2:.3f}', fontsize=10)
    plt.show()


def plot_nice_scatter(x, y, xlabel='Expr (true)', ylabel='Expr (pred)', normalise=False, save_as=None, ylims=[], show_hist=True):
    if normalise:
        #x = normalize_list(x)
        #y = normalize_list(y)
        y = scale_predictions(y, x)

    g = sns.jointplot(x=x, y=y, kind="scatter", color="purple", alpha=0.1, marginal_kws=dict(bins=30, fill=True))
    if not show_hist:
        g.ax_marg_x.set_visible(False)
        g.ax_marg_y.set_visible(False)
    g.plot_joint(sns.regplot, scatter_kws={'alpha': 0.1}, line_kws={'color': 'black', 'linestyle': '--'})

    n = len(x)
    r, p = stats.pearsonr(x, y)
    rho, _ = stats.spearmanr(x, y)

    if ylims:
        y_min = ylims[0]
        y_max = ylims[1]
    else:
        y_min = y.min()
        y_max = y.max()

    plt.text(x.min() + 0.05 * (x.max() - x.min()), 
             y_max - 0.20 * (y_max - y_min), 
             f"$n=${n}\nPearson $R$ = {r:.3f}\nSpearman $\\rho$ = {rho:.3f}\nPearson $R^2$={r**2:.3f}", 
             fontsize=10, ha='left')

    g.set_axis_labels(xlabel, ylabel, fontsize=10)
    g.ax_joint.tick_params(axis='both', labelsize=10)
    g.ax_marg_x.tick_params(axis='x', labelsize=10)
    g.ax_marg_y.tick_params(axis='y', labelsize=10)
    g.ax_joint.set_xlim([min(x), max(x)])
    if ylims:
        g.ax_joint.set_ylim(ylims)

    if save_as:
        plt.savefig(save_as, dpi=300, bbox_inches="tight")
        plt.savefig(f"{save_as.split('.')[0]}.pdf", format="pdf", bbox_inches="tight")
    plt.show()


def plot_annotated_scatter(x, y, categories, xlabel='Expr (true)', ylabel='Expr (pred)', normalise=False, save_as=None, ylims=[]):
    if normalise:
        y = scale_predictions(y, x)

    g = sns.jointplot(x=x, y=y, kind="scatter", color="purple", alpha=0.1, marginal_kws=dict(bins=30, fill=True), height=6)

    unique_categories = np.unique(categories)
    palette = sns.color_palette("deep", len(unique_categories))

    for i, category in enumerate(unique_categories):
        idx = categories == category
        g.ax_joint.scatter(x[idx], y[idx], label=str(category), color=palette[i], alpha=0.3)

    sns.regplot(x=x, y=y, scatter=False, color='black', line_kws={'linestyle': '--'}, ax=g.ax_joint)

    n = len(x)
    r, p = stats.pearsonr(x, y)
    rho, _ = stats.spearmanr(x, y)

    if ylims:
        y_min = ylims[0]
        y_max = ylims[1]
    else:
        y_min = y.min()
        y_max = y.max()

    g.ax_joint.text(x.min() + 0.05 * (x.max() - x.min()), 
                    y_max - 0.20 * (y_max - y_min), 
                    f"$n=${n}\nPearson $R$ = {r:.3f}\nSpearman $\\rho$ = {rho:.3f}\nPearson $R^2$={r**2:.3f}", 
                    fontsize=10, ha='left')

    g.set_axis_labels(xlabel, ylabel, fontsize=10)
    g.ax_joint.tick_params(axis='both', labelsize=10)
    g.ax_marg_x.tick_params(axis='x', labelsize=10)
    g.ax_marg_y.tick_params(axis='y', labelsize=10)
    g.ax_joint.set_xlim([min(x), max(x)])
    if ylims:
        g.ax_joint.set_ylim(ylims)
    
    g.ax_joint.legend(title="Category", fontsize=10, loc="lower right", frameon=False)

    if save_as:
        plt.savefig(save_as, dpi=300, bbox_inches="tight")
        plt.savefig(f"{save_as}.pdf", format="pdf", bbox_inches="tight")

        data_df = pd.DataFrame({'true': x, 'pred': y, 'category': categories})
        data_csv_path = f"{save_as}.csv"
        data_df.to_csv(data_csv_path, index=False)
        print(f"Data saved to {data_csv_path}")

    plt.show()


def plot_scatter_v2(x, y):
    coefficients = np.polyfit(x, y, 1)
    x_fit = np.linspace(np.min(x), np.max(x), 100)
    y_fit = np.polyval(coefficients, x_fit)

    r, _ = stats.pearsonr(x=x, y=y)
    rho, _ = stats.spearmanr(a=x, b=y)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(x, y)
    ax.plot(x_fit, y_fit, color='red')

    title = f'r={r:0.2f}, rho={rho:0.2f}'
    ax.set_title(title)

    plt.show()


def plot_per_type_with_stat(Y, Y_hat, types):
    num_types = len(types)
    rows = 4
    cols = (num_types + 1) // rows + 1
    print(rows,cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    axes = axes.flatten()
    for i, (type, x, y) in enumerate(zip(types, Y, Y_hat)):
        ax = axes[i]
        ax.scatter(x, y)

        coefficients = np.polyfit(x, y, 1)
        x_fit = np.linspace(np.min(x), np.max(x), 100)
        y_fit = np.polyval(coefficients, x_fit)

        r, _ = stats.pearsonr(x=x, y=y)
        rho, _ = stats.spearmanr(a=x, b=y)

        ax.plot(x_fit, y_fit, color='red')

        title = f'{type[:20]}\n(r={r:0.2f},rho={rho:0.2f})'
        ax.set_title(title)

        ax.set_aspect('equal')

    for j in range(num_types, len(axes)):
        axes[j].axis('off')

    plt.subplots_adjust(wspace=0.5, hspace=-0.5)
    plt.show()


#useful for printing all in same scale
def normalize_list(lst):
    min_value = min(lst)
    max_value = max(lst)
    normalized_lst = [(value - min_value) / (max_value - min_value) for value in lst]
    return normalized_lst


def scale_predictions(y_pred, y_true):
    """
    Scale predictions to match the statistics of true or target expressions
    """
    min_true = min(y_true)
    max_true = max(y_true)

    min_pred = min(y_pred)
    max_pred = max(y_pred)

    y_pred_scaled = ((y_pred - min_pred) / (max_pred - min_pred)) * (max_true - min_true) + min_true

    return y_pred_scaled


def plot_box_and_whisker(data, title, ylabel, xticklabels, show_p=True, show_n=True, show_swarmplot=False, baseline=None, figsize=[6.4, 4.8], filepath=None, ax=None):
    """
    Create a box-and-whisker plot with significance bars.
    """
    np.random.seed(42)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    bp = ax.boxplot(data, widths=0.3, patch_artist=True)
    
    # Graph title
    ax.set_title(title, fontsize=14)
    # Label y-axis
    ax.set_ylabel(ylabel)
    # Label x-axis ticks
    ax.set_xticklabels(xticklabels)
    # Hide x-axis major ticks
    ax.tick_params(axis="x", which="major", length=0)
    # Show x-axis minor ticks
    xticks = [0.5] + [x + 0.5 for x in ax.get_xticks()]
    ax.set_xticks(xticks, minor=True)
    # Clean up the appearance
    ax.tick_params(axis="x", which="minor", length=3, width=1)

    # Change the colour of the boxes to Seaborn"s "pastel" palette
    colors = sns.color_palette("pastel")
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    # Colour of the median lines
    plt.setp(bp["medians"], color="k")

    # Show baseline performance by a dotted line
    if baseline:
        ax.axhline(y=baseline, color='r', linestyle='dotted', alpha=0.5)

    # Check for statistical significance
    significant_combinations = []
    # Check from the outside pairs of boxes inwards
    ls = list(range(1, len(data) + 1))
    combinations = [(ls[x], ls[x + y]) for y in reversed(ls) for x in range((len(ls) - y))]
    for c in combinations:
        data1 = data[c[0] - 1]
        data2 = data[c[1] - 1]

        #which series is better?
        median1 = np.median(data1)
        median2 = np.median(data2)
        if median1 > median2:
            res_op = f"\u227b, {np.abs(median1/median2 - 1)*100:.0f}%"
        elif median1 < median2:
            res_op = f"\u227a, {np.abs(median1/median2 - 1)*100:.0f}%"
        else:
            res_op = "\u2248"

        # Significance
        #U, p = stats.mannwhitneyu(data1, data2, alternative="two-sided")
        U, p = stats.wilcoxon(data1, data2, alternative="two-sided")
        if p < 0.05:
            significant_combinations.append([c, p, res_op])

    # Get info about y-axis
    bottom, top = ax.get_ylim()
    yrange = top - bottom

    # Significance bars
    for i, significant_combination in enumerate(significant_combinations):
        # Columns corresponding to the datasets of interest
        x1 = significant_combination[0][0]
        x2 = significant_combination[0][1]
        # What level is this bar among the bars above the plot?
        level = len(significant_combinations) - i
        # Plot the bar
        bar_height = (yrange * 0.08 * level) + top
        bar_tips = bar_height - (yrange * 0.02)
        ax.plot(
            [x1, x1, x2, x2],
            [bar_tips, bar_height, bar_height, bar_tips], lw=1, c="gray")
        # Significance level
        p = significant_combination[1]
        if p < 0.001:
            sig_symbol = "***"
            sig_text = fr"$p < 0.001$"
        elif p < 0.01:
            sig_symbol = "**"
            sig_text = fr"$p=${p:.3f}"
        elif p < 0.05:
            sig_symbol = "*"
            sig_text = fr"$p=${p:.2f}"
        
        sig_text = f"{significant_combination[2]}, {sig_text}"
        
        text_height = bar_height + (yrange * 0.01)

        if show_p:
            ax.text((x1 + x2) * 0.5, text_height, sig_text, ha="center", c="k", size='small')
        else:
            ax.text((x1 + x2) * 0.5, text_height, sig_symbol, ha="center", c="k", size="small")

    # Adjust y-axis
    bottom, top = ax.get_ylim()
    yrange = top - bottom
    ax.set_ylim(bottom - 0.02 * yrange, top)

    # Annotate sample size below each box: if show_n is passed
    if show_n:
        for i, dataset in enumerate(data):
            sample_size = len(dataset)
            ax.text(i + 1, bottom, fr"$n = {sample_size}$", ha="center", size="medium", color="blue")

    # Show swarm plots
    if show_swarmplot:
        for i, dataset in enumerate(data):
            x = np.ones_like(dataset) * (i + 1)
            jitter = np.random.normal(scale=0.01, size=len(dataset))
            ax.scatter(x + jitter, dataset, alpha=0.5, color='k', marker='o', s=5)

    if filepath:
        plt.savefig(filepath, dpi=300, bbox_inches="tight")

    return fig, ax


#Parallel Coordinate Plots
def plot_parallel_coordinates(df, seed=None, title=None, base_path=None):
    custom_order = ["Orig", "Large", "Small", "Mini"]
    df["Model"] = pd.Categorical(df["Model"], categories=custom_order, ordered=True)
    df = df.sort_values(by="Model")
    if seed is not None:
        df = df[(df["Seed"]==seed)].copy()
    df.drop(columns=["Seed","Seq_Enc","LossFn","Optimizer","LRScheduler","all_pearson_r2"], inplace=True)
    df.columns = [w.replace("pearson", "r").replace("spearman", "rho").replace("motif", "mot").replace("perturbation", "pert").replace("challenging", "chal") 
                for w in df.columns]

    plt.rcParams["figure.figsize"] = (24,10)
    parallel_coordinates(df, "Model", 
                        cols=["r", "high_r", "low_r", "yeast_r", "random_r", "chal_r", "SNVs_r", "mot_pert_r", "mot_tiling_r", "rs_score", 
                        "rho", "high_rho", "low_rho", "yeast_rho", "random_rho", "chal_rho", "SNVs_rho", "mot_pert_rho", "mot_tiling_rho", "rhos_score"],
                        color=("black","red","green","blue"))
    plt.legend(loc="lower right")
    plt.title(title)
    plt.tight_layout()
    if base_path is not None:
        plt.savefig(fname=f"{base_path}/challenge_testset_final_models/pcp_seed{seed}_{title}.png", dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_radar_plots(df, seed=None, title=None, base_path=None):
    """
    Radar plot
    """
    custom_order = ["Orig", "Large", "Small", "Mini"]
    model_colors = {"Orig": "blue", "Large": "red", "Small": "orange", "Mini": "green"}
    
    df["Model"] = pd.Categorical(df["Model"], categories=custom_order, ordered=True)
    df = df.sort_values(by="Model")
    if seed is not None:
        df = df[(df["Seed"] == seed)].copy()
    df.drop(columns=["Seed", "Seq_Enc", "LossFn", "Optimizer", "LRScheduler", "all_pearson_r2"], inplace=True)
    df.columns = [w.replace("pearson", "r").replace("spearman", "rho").replace("motif", "mot").replace("perturbation", "pert").replace("challenging", "chal") 
                  for w in df.columns]

    metrics_r = [col for col in df.columns if col.endswith('r')]
    metrics_rho = [col for col in df.columns if col.endswith('rho')]

    num_vars_r = len(metrics_r)
    num_vars_rho = len(metrics_rho)

    angles_r = np.linspace(0, 2 * np.pi, num_vars_r, endpoint=False).tolist()
    angles_rho = np.linspace(0, 2 * np.pi, num_vars_rho, endpoint=False).tolist()
    
    angles_r += angles_r[:1]
    angles_rho += angles_rho[:1]

    fig, axs = plt.subplots(1, 2, figsize=(11, 5), subplot_kw=dict(polar=True))

    for i, model in enumerate(custom_order):
        values_r = df.loc[df["Model"] == model, metrics_r].mean().tolist()
        values_r += values_r[:1]  
        axs[0].plot(angles_r, values_r, label=model, color=model_colors[model])
        axs[0].fill(angles_r, values_r, color=model_colors[model], alpha=0.01)

        values_rho = df.loc[df["Model"] == model, metrics_rho].mean().tolist()
        values_rho += values_rho[:1]  
        axs[1].plot(angles_rho, values_rho, label=model, color=model_colors[model])
        axs[1].fill(angles_rho, values_rho, color=model_colors[model], alpha=0.01)

    axs[0].set_yticklabels([])
    axs[0].set_xticks(angles_r[:-1])
    axs[0].set_xticklabels(metrics_r, fontsize=10)
    axs[0].set_title(f"{title} - Pearson $R$", size=10, color='black', y=1.1)

    axs[1].set_yticklabels([])
    axs[1].set_xticks(angles_rho[:-1])
    axs[1].set_xticklabels(metrics_rho, fontsize=10)
    axs[1].set_title(f"{title} - Spearman $\\rho$", size=10, color='black', y=1.1)

    plt.legend(loc="upper right", bbox_to_anchor=(1.1, 0.1), fontsize=10)
    plt.tight_layout()

    if base_path is not None:
        plt.savefig(fname=f"{base_path}/challenge_testset_final_models/seed{42}_{title}.png", dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
        

def plot_parallel_coordinates_avg(df, title=None, base_path=None):
    df.drop(columns=["Seed","Seq_Enc","LossFn","Optimizer","LRScheduler","all_pearson_r2"], inplace=True)

    df.columns = [w.replace("pearson", "r").replace("spearman", "rho").replace("motif", "mot").replace("perturbation", "pert").replace("challenging", "chal") 
                for w in df.columns]
    
    col_order = ["Model", "r", "high_r", "low_r", "yeast_r", "random_r", "chal_r", "SNVs_r", "mot_pert_r", "mot_tiling_r", "rs_score", 
                 "rho", "high_rho", "low_rho", "yeast_rho", "random_rho", "chal_rho", "SNVs_rho", "mot_pert_rho", "mot_tiling_rho", "rhos_score"]
    
    df = df[col_order]

    grouped_df = df.groupby("Model")[df.columns[1:]]
    df_avg = grouped_df.mean().reset_index()
    df_std = grouped_df.std().reset_index()

    custom_order = ["Orig", "Large", "Small", "Mini"]
    df_avg["Model"] = pd.Categorical(df_avg["Model"], categories=custom_order, ordered=True)
    df_avg = df_avg.sort_values(by="Model")
    df_std["Model"] = pd.Categorical(df_std["Model"], categories=custom_order, ordered=True)
    df_std = df_std.sort_values(by="Model")

    df_merged = df_avg.merge(df_std, on='Model', suffixes=("_avg", "_std"))

    plt.rcParams["figure.figsize"] = (24, 5)
    parallel_coordinates(df_avg, "Model", color=("black", "red", "green", "blue"))

    model_colors = {"Orig":"black", "Large":"red", "Small":"green", "Mini":"blue"}

    for model in model_colors:
        model_data = df_merged[df_merged["Model"] == model]
        for i, col in enumerate(df_avg.columns[1:]):
            x_values = [i] * len(model_data)
            plt.errorbar(x_values, model_data[col + "_avg"], yerr=model_data[col + "_std"], linestyle="None", color=model_colors[model], capsize=5, alpha=0.5)

    plt.legend(loc="lower right")
    plt.title(f"Average across 10 replicates ({title}) with Standard Deviation")
    plt.tight_layout()
    if base_path is not None:
        plt.savefig(fname=f"{base_path}/challenge_testset_final_models/pcp_all_replicates_{title}.png", dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_box_plots_avg(df, title=None, base_path=None):
    df.drop(columns=["Seed","Seq_Enc","LossFn","Optimizer","LRScheduler","all_pearson_r2"], inplace=True)

    df.columns = [w.replace("pearson", "r").replace("spearman", "rho").replace("motif", "mot").replace("perturbation", "pert").replace("challenging", "chal") 
                for w in df.columns]
    
    col_order = ["Model", "r", "high_r", "low_r", "yeast_r", "random_r", "chal_r", "SNVs_r", "mot_pert_r", "mot_tiling_r", "rs_score", 
                 "rho", "high_rho", "low_rho", "yeast_rho", "random_rho", "chal_rho", "SNVs_rho", "mot_pert_rho", "mot_tiling_rho", "rhos_score"]
    
    col_order = col_order[0:10]
    
    df = df[col_order]
    df.columns = ["Model", "all", "high", "low", "yeast", "random", "challenging", "SNVs", "motif_pert", "motif_tiling"]

    models = ["Orig", "Leg"]

    df_selected = df[df["Model"].isin(models)]

    plt.rcParams["figure.figsize"] = (16, 3)

    for i, col in enumerate(df_selected.columns[1:]):
        plt.subplot(2, len(df_selected.columns[1:]), i+1)
        box_data = [df_selected[df_selected["Model"] == model][col] for model in models]
        box = plt.boxplot(box_data, patch_artist=True)
        
        # Set box colors
        colors = ['blue', 'green']
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
        
        plt.title(col, fontsize=10)
        plt.xticks(range(1, len(box_data) + 1), models)  # Set x-labels
        plt.yticks(fontsize=10)
        
        # Set colors for whiskers and caps
        for whisker, cap in zip(box['whiskers'], box['caps']):
            whisker.set(color='black', linewidth=1.5)
            cap.set(color='black', linewidth=1.5)

    plt.subplots_adjust(wspace=0, hspace=0.5)
    plt.tight_layout()
    if base_path is not None:
        plt.savefig(fname=f"{base_path}/all_replicates_{title}_boxplots_pearsons.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_radar_plots_avg(df, title=None, base_path=None):
    """
    Radar plot showing the average across replicates with standard deviation
    """
    custom_order = ["Orig", "Large", "Small", "Mini"]
    model_colors = {"Orig": "blue", "Large": "red", "Small": "orange", "Mini": "green"}
    
    df.drop(columns=["Seed", "Seq_Enc", "LossFn", "Optimizer", "LRScheduler", "all_pearson_r2"], inplace=True)
    df.columns = [w.replace("pearson", "r").replace("spearman", "rho").replace("motif", "mot").replace("perturbation", "pert").replace("challenging", "chal") 
                  for w in df.columns]

    metrics_r = [col for col in df.columns if col.endswith('r')]
    metrics_rho = [col for col in df.columns if col.endswith('rho')]

    # Group by model and calculate mean and standard deviation
    grouped_df = df.groupby("Model")[metrics_r + metrics_rho]
    df_avg = grouped_df.mean().reset_index()
    df_std = grouped_df.std().reset_index()

    df_avg["Model"] = pd.Categorical(df_avg["Model"], categories=custom_order, ordered=True)
    df_avg = df_avg.sort_values(by="Model")
    df_std["Model"] = pd.Categorical(df_std["Model"], categories=custom_order, ordered=True)
    df_std = df_std.sort_values(by="Model")

    num_vars_r = len(metrics_r)
    num_vars_rho = len(metrics_rho)

    angles_r = np.linspace(0, 2 * np.pi, num_vars_r, endpoint=False).tolist()
    angles_rho = np.linspace(0, 2 * np.pi, num_vars_rho, endpoint=False).tolist()
    
    angles_r += angles_r[:1]
    angles_rho += angles_rho[:1]

    fig, axs = plt.subplots(1, 2, figsize=(20, 20), subplot_kw=dict(polar=True))
    for model in custom_order:
        values_r = df_avg.loc[df_avg["Model"] == model, metrics_r].mean().tolist()
        values_r += values_r[:1]  
        axs[0].plot(angles_r, values_r, label=model, color=model_colors[model])
        axs[0].fill(angles_r, values_r, color=model_colors[model], alpha=0.01)

        std_r = df_std.loc[df_avg["Model"] == model, metrics_r].mean().tolist()
        std_r += std_r[:1]
        axs[0].errorbar(angles_r[:-1], values_r[:-1], yerr=std_r[:-1], fmt='o', color=model_colors[model], capsize=5)

        values_rho = df_avg.loc[df_avg["Model"] == model, metrics_rho].mean().tolist()
        values_rho += values_rho[:1]  
        axs[1].plot(angles_rho, values_rho, label=model, color=model_colors[model])
        axs[1].fill(angles_rho, values_rho, color=model_colors[model], alpha=0.01)

        std_rho = df_std.loc[df_avg["Model"] == model, metrics_rho].mean().tolist()
        std_rho += std_rho[:1]
        axs[1].errorbar(angles_rho[:-1], values_rho[:-1], yerr=std_rho[:-1], fmt='o', color=model_colors[model], capsize=5)

    axs[0].set_yticklabels([])
    axs[0].set_xticks(angles_r[:-1])
    axs[0].set_xticklabels(metrics_r, fontsize=10)
    axs[0].set_title(f"{title} - Pearson $R$", size=10, color='black', y=1.1)

    axs[1].set_yticklabels([])
    axs[1].set_xticks(angles_rho[:-1])
    axs[1].set_xticklabels(metrics_rho, fontsize=10)
    axs[1].set_title(f"{title} - Spearman $\\rho$", size=10, color='black', y=1.1)

    plt.legend(loc="lower center", bbox_to_anchor=(0,0), fontsize=10, ncol=1, frameon=False)
    plt.tight_layout()

    if base_path is not None:
        plt.savefig(fname=f"{base_path}/challengetestset_{title}_allreplicates_radar_SI.pdf", format="pdf", bbox_inches="tight")
    else:
        plt.show()


def gen_ensemble_boxplots_unscaled(Y_preds, Y_true, model_type, choice="PearsonR2", filename=None):
    """
    A small util function to generate the box plots with individual scales
    """
    def get_score(y_true, y_pred, choice):
        if choice == "PearsonR":
            score = my_pearsonr(y_true, y_pred)
            ylabel = f"$PearsonR$"
        elif choice == "PearsonR2":
            score = my_pearsonr(y_true, y_pred) ** 2
            ylabel = f"$PearsonR^2$"
        else:
            score = my_spearmanr(y_true, y_pred)
            ylabel = f"$Spearman~\\rho$"
        return score, ylabel

    # Create box plot data
    box_plot_data = []

    # Compute R^2 for single seeds
    for i, Y_pred in enumerate(Y_preds):
        score, _ = get_score(Y_true, Y_pred, choice)
        box_plot_data.append((1, score))

    # Compute R^2 for combinations of seeds
    for k in range(2, 11):
        for combination in combinations(Y_preds, k):
            avg_Y_pred = np.mean(combination, axis=0)
            score, ylabel = get_score(Y_true, avg_Y_pred, choice)
            box_plot_data.append((k, score))

    # Convert to DataFrame
    results_df = pd.DataFrame(box_plot_data, columns=["num_seeds", "average_score"])
    results_df.to_csv(f"{filename}.csv", index=False)

    # Plotting the box plot
    plt.figure(figsize=(7, 3))
    sns.boxplot(x="num_seeds", y="average_score", data=results_df, hue="num_seeds", palette="coolwarm", showfliers=False, legend=False)
    sns.stripplot(x="num_seeds", y="average_score", data=results_df, color="black", alpha=0.3, jitter=True, legend=False)

    plt.xlabel("Number of seeds", fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    plt.xticks(fontsize=10)

    # Automatically adjust y-axis limits based on data
    ymin, ymax = results_df["average_score"].min(), results_df["average_score"].max()
    margin = (ymax - ymin) * 0.05  # Add a 5% margin to y-axis
    plt.ylim(ymin - margin, ymax + margin)

    plt.yticks(fontsize=10)
    plt.title(f"Replicate-Ensembles of Camformer ({model_type})", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)

    if filename:
        plt.gca().set_title("")
        plt.savefig(f"{filename}.png", dpi=300, bbox_inches="tight")
        plt.savefig(f"{filename}.pdf", format="pdf", bbox_inches="tight")
    plt.show()


def trainingdata_vs_perfs_allseeds(input_file, parent_dir, save_as=None):
    seed_dirs = [os.path.join(parent_dir, d) for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
    results = {}
    ylabel = "Pearson $R$" #"Spearman $\\rho$"
    print(f"Number of seed directories: {len(seed_dirs)}")
    for seed_dir in seed_dirs:
        print(f"Processing directory: {seed_dir}")
        config_files = [f for f in os.listdir(seed_dir) if f.startswith("config") and f.endswith(".json")]
        df = pd.read_csv(input_file, delimiter=',', usecols=["sequence","Measured Expression"])
        df.rename(columns={"sequence": "seq", "Measured Expression": "expr"}, inplace=True)
        seq = df[["seq"]]
        expr = df["expr"]
        def extract_number(filename):
            match = re.search(r"config_(\d+)", filename)
            return int(match.group(1)) if match else None
        config_files.sort(key=extract_number)
 
        for config_file in config_files:
            config = loadDict(path=seed_dir, filename=config_file.split('.')[0])
            setSeeds(config["seed"])
            model_args = config["model_args"]
            model = CNN(**model_args)
            model = torch.load(config["best_model_path"])
            model.to(config["device"])
            model.eval()
            TestLoader = createDataLoader(seq=seq, expr=expr, config=config)
            y_pred, y_true = predict(model=model, DataLoader=TestLoader, config=config)
            
            if ylabel == "Pearson $R$":
                score = my_pearsonr(y_true, y_pred)
            else:
                score = my_spearmanr(y_true, y_pred)
            
            num_parameters = int(config_file.split('_')[1][:-1])
            if num_parameters not in results:
                results[num_parameters] = []
            results[num_parameters].append(score)
    
    data = []
    for num_parameters, scores in results.items():
        for score in scores:
            data.append({"Training data size": num_parameters, "score": score})
    data_df = pd.DataFrame(data)

    print(f"\nModels from  : {parent_dir}")
    print(f"Test dataset : {input_file}")
    sns.set_style("white")
    plt.figure(figsize=(7, 3))
    sns.boxplot(data=data_df, x="Training data size", y="score", showfliers=False, legend=False)
    sns.stripplot(data=data_df, x="Training data size", y="score", color="black", alpha=0.3, jitter=True, legend=False)
    plt.xlabel("Training data size ($n$)", fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    xticks = plt.gca().get_xticks()
    xtick_labels = [f'{x}M' for x in sorted(data_df["Training data size"].unique())]
    plt.xticks(ticks=xticks, labels=xtick_labels, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='x', linestyle=':', color='gray')
    plt.tick_params(axis='x', direction='out')  
    plt.tick_params(axis='y', direction='out') 
    if save_as:
        #plt.savefig(save_as, dpi=300, bbox_inches="tight")
        plt.savefig(f"{save_as}.pdf", format="pdf", bbox_inches="tight")
        data_df.to_csv(f"{save_as}.csv", index=False)
        print(f"Data saved to {save_as}.csv")
    plt.show()

def get_pearsonR(df):
    r = my_pearsonr(df["Measured Expression"], df["Predicted Expression"])
    return r, r**2

def get_spearmanRho(df):
    rho = my_spearmanr(df["Measured Expression"], df["Predicted Expression"])
    return rho, rho**2


def add_significance_bars(ax, model_names, p_values, significance_level=0.05):
    """
    Function to add significance bars (non-significant, actually: n.s.) to boxplots
    """
    yrange = 1.0  
    top = 0.9  
    for i, (model_i, model_j, p) in enumerate(p_values):
        if p >= significance_level:
            x1, x2 = model_names.index(model_i), model_names.index(model_j)
            level = len(p_values) - i
            bar_height = (yrange * 0.08 * level) + top
            bar_tips = bar_height - (yrange * 0.02)
            ax.plot(
                [x1, x1, x2, x2],
                [bar_tips, bar_height, bar_height, bar_tips], lw=1, c="gray")
            ax.text((x1 + x2) * 0.5, bar_height + (yrange * 0.01), "n.s.", ha="center", c="k", size='small')


def generate_bootstrap_violin_plots(model_files, n_bootstrap=10000, fig_title="No title", save_dir=None):
    """
    Generates boxplots for different models. Each model's prediction file ([sequence, Measured Expression, Predicted Expression]) is passed. 
    """
    model_dfs = {name: pd.read_csv(file, index_col="Unnamed: 0") for name, file in model_files.items()}
    
    for name, df in model_dfs.items():
        df.rename(columns={"Predicted Expression": f"Predicted Expression_{name}"}, inplace=True)

    merged_df = model_dfs[list(model_dfs.keys())[0]]
    for name, df in model_dfs.items():
        if name != list(model_dfs.keys())[0]:
            merged_df = pd.merge(merged_df, df[['sequence', f"Predicted Expression_{name}"]],
                                 on='sequence', how='inner')

    # Check if merged_df is still empty
    if merged_df.empty:
        print("Error: Merged DataFrame is empty.")
        return

    bootstrap_results = {name: [] for name in model_dfs.keys()}

    for _ in range(n_bootstrap):
        sample_indices = np.random.choice(merged_df.index, size=len(merged_df), replace=True)
        sample = merged_df.loc[sample_indices]
        
        for name in model_dfs.keys():
            sample_model = sample[['Measured Expression', f"Predicted Expression_{name}"]]
            r, _ = get_pearsonR(sample_model.rename(columns={f"Predicted Expression_{name}": "Predicted Expression"}))
            bootstrap_results[name].append(r)

    bootstrap_pearson_df = pd.DataFrame(bootstrap_results)

    custom_palette = {
        "DeepSEA"           : "#4D4D4D", # dark gray
        "DanQ"              : "#5DA5DA", # blue
        "DeepAtt"           : "#FAA43A", # orange
        "Vaishnav\net al."  : "#60BD68", # green
        "LegNet"            : "#B2912F", # brown
        "Camformer"         : "#B276B2", # purple
        "Camformer\nMini"   : "#F17CB0", # pink
        "Camformer-Mini"    : "#F17CB0"  # pink
    }

    plt.figure(figsize=(5, 3))
    sns.violinplot(data=bootstrap_pearson_df, palette=[custom_palette[name] for name in bootstrap_pearson_df.columns])
    plt.title(f"{fig_title}; $n={len(merged_df)}$")
    #plt.xlabel('Model')
    plt.ylabel(f'$PearsonR$')
    plt.xticks(rotation=45, fontsize=10, ha='right')

    model_names = list(model_dfs.keys())
    num_models = len(model_names)
    p_values = []
    for i in range(num_models):
        for j in range(i + 1, num_models):
            model_i = model_names[i]
            model_j = model_names[j]
            t_stat, p_value = stats.ttest_rel(bootstrap_pearson_df[model_i], bootstrap_pearson_df[model_j])
            p_values.append((model_i, model_j, p_value))
            if p_value >= 0.05:
                print(f"{model_i} vs {model_j}: p = {p_value}")

    ax = plt.gca()
    add_significance_bars(ax, model_names, p_values)
    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/SoTA_{fig_title}_n={len(merged_df)}_violinplots.png", dpi=300, bbox_inches="tight")
        plt.savefig(f"{save_dir}/SoTA_{fig_title}_n={len(merged_df)}_violinplots.pdf", format="pdf", bbox_inches="tight")
        bootstrap_pearson_df.to_csv(f"{save_dir}/SoTA_{fig_title}_n={len(merged_df)}_violinplots.csv", index=False)
    plt.show()

def get_and_save_Camformer_predictions(config_path, config_name, model_type, input_file, save_filename):
    """
    Use saved Camformer model to predict and save the predictions
    """
    df = pd.read_csv(input_file, delimiter=',', usecols=["sequence","Measured Expression"])
    df.rename(columns={"sequence": "seq", "Measured Expression": "expr"}, inplace=True)
    seq = df[["seq"]]
    expr = df["expr"]

    if model_type == "Original":
        from base.model import CNN
    else:
        from base.model_basic import CNN
        
    config = loadDict(path=config_path, filename=config_name)
    setSeeds(config["seed"])
    TestLoader = createDataLoader(seq=seq, expr=expr, config=config)

    model_args = config["model_args"]
    model = CNN(**model_args)
    model = torch.load(config["best_model_path"])
    model.to(config["device"])
    model.eval()

    y_pred, y_true = predict(model=model, DataLoader=TestLoader, config=config)
    results_df = pd.DataFrame({"sequence": seq["seq"], "Measured Expression": y_true, "Predicted Expression": y_pred})
    save_filepath = f"./results/Camformer_predictions/{save_filename}"
    results_df.to_csv(save_filepath)
    print(f"Predictions saved in: {save_filepath} {results_df.shape}")


def generate_barplots(model_pred_files, fig_title, metric="PearsonR", save_dir=None):
    """
    Generate bar plots of model's performance (R)
    """
    r_values = {"Model": [], "R": []}

    for model_name, file_path in model_pred_files.items():
        df = pd.read_csv(file_path, index_col="Unnamed: 0")
        if metric == "PearsonR":
            r_value = get_pearsonR(df)[0]
        else:
            r_value = get_spearmanRho(df)[0]
        r_values["Model"].append(model_name)
        r_values["R"].append(r_value)

    r_df = pd.DataFrame(r_values)
    r_df["R2"] = [r ** 2 for r in r_df["R"].values]

    custom_palette = {
        "DeepSEA"           : "#4D4D4D", # dark gray
        "DanQ"              : "#5DA5DA", # blue
        "DeepAtt"           : "#FAA43A", # orange
        "Vaishnav\net al."  : "#60BD68", # green
        "LegNet"            : "#B2912F", # brown
        "Camformer"         : "#B276B2", # purple
        "Camformer\nMini"   : "#F17CB0", # pink
        "Camformer-Mini"    : "#F17CB0"  # pink
    }
    r_df['Color'] = r_df['Model'].map(custom_palette)

    plt.figure(figsize=(8, 4))
    sns.barplot(x="Model", y="R", data=r_df, hue="Model", palette=r_df['Color'].tolist(), dodge=False, legend=False)
    plt.ylabel(f"$PearsonR$", fontsize=14)
    plt.xlabel("")
    plt.xticks(ticks=range(len(r_df['Model'])), labels=r_df["Model"].values)
    plt.title(f"{fig_title}; $n={len(df)}$")
    plt.ylim([0.85,1.0])
    if save_dir:
        plt.savefig(f"{save_dir}/SoTA_{fig_title}_n={len(df)}_barplots", dpi=300, bbox_inches="tight")
    plt.show()
    return r_df