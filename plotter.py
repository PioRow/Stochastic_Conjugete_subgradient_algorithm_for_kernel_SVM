import numpy as np
import matplotlib.pyplot as plt
import json
import matplotlib.ticker as ticker
def load_data(path="experiment_history.json"):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def plot_histories(histories):
    for (experiment,data) in histories.items():
        plot_history(experiment,data)



def plot_history(experiment,data):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 3.5 * 1), sharex=False)
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    for idx,key in enumerate(data.keys()):
        ax = axes[idx]
        run=data[key]
        ax.plot(run,color=colors[idx], linewidth=2)
        ax.set_title(f"Algorithm {key}", fontsize=12, fontweight='bold')
        ax.set_xlabel('Iterations', fontsize=11)
        if idx == 0:
            ax.set_ylabel('Loss Value', fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
        ax.tick_params(labelsize=10)
    fig.suptitle(f"Comparison for  {experiment} dataset", fontsize=14, fontweight='bold', y=1.05)
    filename = f"plots/{experiment}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    data = load_data()
    plot_histories(data)
    d2=load_data("results/big_experiment_history_100its.json")
    plot_histories(d2)
