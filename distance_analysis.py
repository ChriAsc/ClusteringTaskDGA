import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

for dim in range(2,11,2):
    distances = {10: [], 15: [], 20: []}
    iterations = {10: [], 15: [], 20: []}
    for epoch in [10, 15, 20]:
        df = pd.read_csv(f"/home/lorenzo/progettoBDA/datasets/distances/results/feed_results_e{epoch}_d{dim}.csv")
        iterations[epoch] = df['iteration'].tolist()
        distances[epoch] = df['max_distance'].tolist()
    # Initialise the subplot function using number of rows and columns
    # Epoch = 10
    plt.plot(iterations[10], distances[10], 'ro')
    plt.xlabel = 'iterations'
    plt.ylabel = 'max_distance',
    plt.title= f"distances with dim={dim} and epoch=10"
    plt.xticks(np.arange(0, 1001, 50.0))
    plt.grid(True)
    figure = plt.gcf()
    figure.set_size_inches(18.5, 10.5)
    figure.savefig(f"/home/lorenzo/progettoBDA/datasets/distances/results/fig_dim{dim}_epoch{10}")
    plt.clf()
    # Epoch = 15
    plt.plot(iterations[15], distances[15], 'bs')
    plt.xlabel = 'iterations'
    plt.ylabel = 'max_distance',
    plt.title = f"distances with dim={dim} and epoch=10"
    plt.xticks(np.arange(0, 1001, 50.0))
    plt.grid(True)
    figure = plt.gcf()
    figure.set_size_inches(18.5, 10.5)
    figure.savefig(f"/home/lorenzo/progettoBDA/datasets/distances/results/fig_dim{dim}_epoch{15}")
    plt.clf()
    # Epoch = 20
    plt.plot(iterations[20], distances[20], 'g^')
    plt.xlabel = 'iterations'
    plt.ylabel = 'max_distance',
    plt.title = f"distances with dim={dim} and epoch=10"
    plt.xticks(np.arange(0, 1001, 100.0))
    plt.grid(True)
    figure = plt.gcf()
    figure.set_size_inches(18.5, 10.5)
    figure.savefig(f"/home/lorenzo/progettoBDA/datasets/distances/results/fig_dim{dim}_epoch{20}")
    plt.clf()