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

    g = sns.jointplot(x=x, y=y, kind="scatter", color="purple", alpha=0.1, marginal_kws=dict(bins=30, fill=True), height=8)

    unique_categories = np.unique(categories)
    palette = sns.color_palette("deep", len(unique_categories))

    for i, category in enumerate(unique_categories):
        idx = categories == category
        g.ax_joint.scatter(x[idx], y[idx], label=str(category), color=palette[i], alpha=0.5)

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
