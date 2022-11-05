import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

first_part = pd.read_csv("/media/lorenzo/Partizione Dati/progettoBDA/results_e10/bigrams_results_e10_d2-4.csv")
second_part = pd.read_csv("/media/lorenzo/Partizione Dati/progettoBDA/results_e10/bigrams_results_e10_d6-8.csv")
third_part = pd.read_csv("/media/lorenzo/Partizione Dati/progettoBDA/results_e10/bigrams_results_e10_d10.csv")
frames = [first_part, second_part, third_part]
complete = pd.concat(frames)
sns.set()
graphics = [["homogeneity", "completeness"], ["silhouette_score"], ["v_measure"]]
for graph in graphics:
    figure, axes = plt.subplots(len(graph), 5)
    figure.suptitle(f"{graph[0]} and {graph[1]} scores for all dimensions" if len(graph) > 1
                    else f"{graph[0]} scores for all dimensions")
    if len(graph) > 1:
        figure.set_size_inches(30, 12)
    else:
        figure.set_size_inches(30, 6)
    figure.tight_layout(pad=3.5)
    for measure in graph:
        dimensions = [complete[complete["dimension"] == dim].pivot("minPoints", "epsilon", measure)
                      for dim in range(2, 11, 2)]
        for i in range(0, 5):
            dimensions[i].sort_index(level=0, ascending=False, inplace=True)
            if len(graph) > 1:
                sns.heatmap(dimensions[i], ax=axes[graph.index(measure), i], annot=True, linewidths=.5, fmt='.4f',
                            cmap='crest')
                axes[graph.index(measure), i].set_title(f"{measure} with fasttext dimension {(i+1)*2}")
            else:
                sns.heatmap(dimensions[i], ax=axes[i], annot=True, linewidths=.5, fmt='.4f',
                            cmap='crest')
                axes[i].set_title(f"{measure} with fasttext dimension {(i + 1) * 2}")
    figure.savefig(f"/media/lorenzo/Partizione Dati/progettoBDA/results_e10/{graph[0]}_{graph[1]}_e10" if len(graph) > 1
                   else f"/media/lorenzo/Partizione Dati/progettoBDA/results_e10/{graph[0]}_e10")

nums = ["num_clusters", "num_noise"]
figure, axes = plt.subplots(2, 5)
figure.suptitle(f"{nums[0]} and {nums[1]} data for all dimensions")
figure.set_size_inches(30, 12)
figure.tight_layout(pad=3.5)
for num in nums:
    dimensions = [complete[complete["dimension"] == dim].pivot("minPoints", "epsilon", num)
                  for dim in range(2, 11, 2)]
    for i in range(0, 5):
        dimensions[i].sort_index(level=0, ascending=False, inplace=True)
        sns.heatmap(dimensions[i], ax=axes[nums.index(num), i], annot=True, linewidths=.5, fmt='d',
                    cmap=sns.cubehelix_palette(rot=-.2, as_cmap=True))
        axes[nums.index(num), i].set_title(f"{num} with fasttext dimension {(i+1)*2}")
figure.savefig(f"/media/lorenzo/Partizione Dati/progettoBDA/results_e10/{nums[0]}_{nums[1]}e10")

"""plt.plot(complete[(complete["dimension"] == 2) & (complete["minPoints"] == 95)]["epsilon"],
         complete[(complete["dimension"] == 2) & (complete["minPoints"] == 95)]["homogeneity"], 'ro-',
         label="homogeneity")
plt.plot(complete[(complete["dimension"] == 2) & (complete["minPoints"] == 95)]["epsilon"],
         complete[(complete["dimension"] == 2) & (complete["minPoints"] == 95)]["completeness"], 'bo-',
         label="completeness")
plt.plot(complete[(complete["dimension"] == 2) & (complete["minPoints"] == 95)]["epsilon"],
         complete[(complete["dimension"] == 2) & (complete["minPoints"] == 95)]["silhouette_score"], 'go-',
         label="silhouette_score")
plt.plot(complete[(complete["dimension"] == 2) & (complete["minPoints"] == 95)]["epsilon"],
         complete[(complete["dimension"] == 2) & (complete["minPoints"] == 95)]["v_measure"], 'yo-',
         label="v_measure")

plt.legend(loc="upper right")
plt.xlabel('epsilon')
plt.ylabel('metrics')
plt.title(f"metrics with dimension={2}")
plt.grid(True)
figure = plt.gcf()
figure.set_size_inches(18.5, 10.5)
plt.show()
"""
