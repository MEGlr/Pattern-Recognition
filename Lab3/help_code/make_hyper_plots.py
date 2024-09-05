import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

def make_hyper_plot(dataset_type):
    """
    This function loads the data from the csv file and plots them.
    It plots a set of points with different color based on the macro f1 score.
    """


    # load the data
    data = pd.read_csv(Path(f"../hyper_tuning/lstm_{dataset_type}.csv"))

    plt.figure()

    # make a scatter plot
    ax = sns.scatterplot(data=data, x="lr", y="dropout", style='rnn_layers', hue="score", size="score")
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    plt.title(f"Hyperparameter Tuning | {dataset_type}")

    plt.xscale('log')
    plt.tight_layout()




if __name__ == '__main__':

    # make all the hyperparameter plots

    for dataset_type in ['spectrogram' , 'spectrogram_beat', 'chroma', 'fused']:
        make_hyper_plot(dataset_type)
        plt.savefig(f'../plots/hyper_{dataset_type}.png', dpi=300)
