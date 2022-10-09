""" This script was used to calculate length of token sequences in 
    training data and plot them.
"""


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


if __name__ == "__main__":
    df = pd.read_csv("sums.txt", names=["X"], header=None)#.values.tolist()
    col = df["X"]
    count = col[col > 230].count()
    print(count)

    fig, axs = plt.subplots(2)
    
    axs[0].hist(df, log=True, bins=[0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800])
    plt.xlabel("Tokens")
    axs[0].set_ylabel("Count")
    axs[0].set_title("Input sequence length")
    plt.xlim(0, 800)
    axs[0].grid(True, alpha=0.5)

    sns.set_style("whitegrid")
    sns.kdeplot(df, ax=axs[1])

    plt.savefig("sums.pdf")
