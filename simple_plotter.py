import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

root_dir = os.path.dirname(os.path.abspath(__file__))
current_dir = os.path.join(root_dir, 'result')
sns.set_style("darkgrid")

def getdata(experiment_dir):
    # readcsv
    csv_dir = os.path.join(experiment_dir, "table.csv")
    d = pd.read_csv(csv_dir)
    rewards = list(d['best_fit'])
    return rewards

def smooth(data, sm=1):
    if sm > 1:
        y = np.ones(sm)*1.0/sm
        smooth_data = np.convolve(data, y, "same")
        return smooth_data
    else:
        return data


if __name__ == "__main__":
    game = "BridgeWalker-v0"
    algos = ['AGCD']
    experiment_dir = os.path.join(current_dir, game+'_gpu')
    out_dir=os.path.join(experiment_dir,f"{algos[0]}_{game}.png")
    fig = plt.figure(figsize=(7,4))

    # Get data
    data = getdata(experiment_dir)
    data = smooth(data, sm=2)
    length=len(data)
    xdata = np.arange(0,length,1)
    ax = sns.tsplot(time=xdata, data=data,condition=algos[0], color=sns.color_palette()[0], legend=True)

    plt.tick_params(labelsize=16)
    plt.ylabel("Reward", fontsize=16)
    plt.xlabel("generation", fontsize=16)
    plt.title(game, fontsize=18)
    plt.rcParams.update({'font.size': 9.5})
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir, dpi=600,pad_inches=0.0)