import pandas as pd
import matplotlib.pyplot as plt

for dim in range(2,11,2):
    distances = {10: [], 15: [], 20: []}
    iterations = {10: [], 15: [], 20: []}
    for epoch in [10, 15, 20]:
        df = pd.read_csv(f"/media/lorenzo/Partizione Dati/progettoBDA/datasets/distances/results/feed_results_e{epoch}_d{dim}.csv")
        iterations[epoch] = df['iteration'].tolist()
        distances[epoch] = df['max_distance'].tolist()
    # Initialise the subplot function using number of rows and columns
    # Epoch = 10
    figure, axis = plt.subplots()
    axis.plot(iterations[10], distances[10], 'ro')
    axis.set(xlabel='iterations', ylabel='max_distance', title=f"distances with dim={dim} and epoch=10")
    axis.grid(True)
    figure.tight_layout(pad=0.1)
    figure.savefig(f"/media/lorenzo/Partizione Dati/progettoBDA/datasets/distances/results/fig_dim{dim}_epoch{10}", dpi=600.0)
    # Epoch = 15
    figure, axis = plt.subplots()
    axis.plot(iterations[15], distances[15], 'bs')
    axis.set(xlabel='iterations', ylabel='max_distance', title=f"distances with dim={dim} and epoch=15")
    axis.grid(True)
    figure.tight_layout(pad=0.1)
    figure.savefig(f"/media/lorenzo/Partizione Dati/progettoBDA/datasets/distances/results/fig_dim{dim}_epoch{15}", dpi=600.0)
    # Epoch = 20
    figure, axis = plt.subplots()
    axis.plot(iterations[20], distances[20], 'g^')
    axis.set(xlabel='iterations', ylabel='max_distance', title=f"distances with dim={dim} and epoch=20")
    axis.grid(True)
    figure.tight_layout(pad=0.1)
    figure.savefig(f"/media/lorenzo/Partizione Dati/progettoBDA/datasets/distances/results/fig_dim{dim}_epoch{20}", dpi=600.0)
    # together
    figure, axis = plt.subplots(3,1)
    axis[0].plot(iterations[10], distances[10], 'ro')
    axis[0].set(xlabel='iterations', ylabel='max_distance', title=f"distances with dim={dim} and epoch=10")
    axis[0].grid(True)
    # Epoch = 15
    axis[1].plot(iterations[15], distances[15], 'bs')
    axis[1].set(xlabel='iterations', ylabel='max_distance', title=f"distances with dim={dim} and epoch=15")
    axis[1].grid(True)
    # Epoch = 20
    axis[2].plot(iterations[20], distances[20], 'g^')
    axis[2].set(xlabel='iterations', ylabel='max_distance', title=f"distances with dim={dim} and epoch=20")
    axis[2].grid(True)
    figure.tight_layout(pad=0.1)
    figure.savefig(f"/media/lorenzo/Partizione Dati/progettoBDA/datasets/distances/results/fig_together_dim{dim}", dpi=600.0)