import pandas as pd
import matplotlib.pyplot as plt

for dim in range(2,11,2):
    distances = {10: [], 15: [], 20: []}
    iterations = {10: [], 15: [], 20: []}
    for epoch in [10, 15, 20]:
        df = pd.read_csv(f"/media/lorenzo/Partizione Dati/progettoBDA/datasets/distances/results/feed_results_e{epoch}_d{dim}.csv")
        iterations[epoch] = df['iteration'].tolist()
        distances[epoch] = df['max_distance'].tolist()
    plt.plot(iterations[10], distances[10], 'ro', iterations[15], distances[15], 'bs',
             iterations[20], distances[20], 'g^')
    plt.xlabel('iterations')
    plt.ylabel('max_distance')
    plt.suptitle(f"Distances with dimension={dim}")
    plt.show()