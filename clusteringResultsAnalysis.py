import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

first_part = pd.read_csv("/media/lorenzo/Partizione Dati/progettoBDA/results_e10/bigrams_results_e10_d2-4.csv")
second_part = pd.read_csv("/media/lorenzo/Partizione Dati/progettoBDA/results_e10/bigrams_results_e10_d6-8.csv")
third_part = pd.read_csv("/media/lorenzo/Partizione Dati/progettoBDA/results_e10/bigrams_results_e10_d10.csv")
frames = [first_part, second_part, third_part]
complete = pd.concat(frames)
sns.set()
for measure in ["homogeneity", "completeness", "silhouette_score", "v_measure"]:
    dimensions = [complete[complete["dimension"] == dim].pivot("minPoints", "epsilon", measure)
                  for dim in range(2, 11, 2)]
    figure, axes = plt.subplots(2, 3)
    figure.suptitle(f"{measure} data for all dimensions")
    figure.set_size_inches(18.5, 10.5)
    figure.tight_layout(pad=3.5)
    for i in range(0, 5):
        dimensions[i].sort_index(level=0, ascending=False, inplace=True)
        if i in [0, 1, 2]:
            sns.heatmap(dimensions[i], ax=axes[0, i], annot=True, linewidths=.5, fmt='.4f', cmap='crest')
            axes[0, i].set_title(f"{measure} with fasttext dimension {(i+1)*2}")
        else:
            sns.heatmap(dimensions[i], ax=axes[1, i-3], annot=True, linewidths=.5, fmt='.4f', cmap='crest')
            axes[1, i-3].set_title(f"{measure} with fasttext dimension {(i+1)*2}")
        figure.savefig(f"/media/lorenzo/Partizione Dati/progettoBDA/results_e10/{measure}e10")

for num in ["num_clusters", "num_noise"]:
    dimensions = [complete[complete["dimension"] == dim].pivot("minPoints", "epsilon", num)
                  for dim in range(2, 11, 2)]
    figure, axes = plt.subplots(2, 3)
    figure.suptitle(f"{num} data for all dimensions")
    figure.set_size_inches(18.5, 10.5)
    figure.tight_layout(pad=3.5)
    for i in range(0, 5):
        dimensions[i].sort_index(level=0, ascending=False, inplace=True)
        if i in [0, 1, 2]:
            sns.heatmap(dimensions[i], ax=axes[0, i], annot=True, linewidths=.5, fmt='d',
                        cmap=sns.cubehelix_palette(rot=-.2, as_cmap=True))
            axes[0, i].set_title(f"{num} with fasttext dimension {(i+1)*2}")
        else:
            sns.heatmap(dimensions[i], ax=axes[1, i-3], annot=True, linewidths=.5, fmt='d',
                        cmap=sns.cubehelix_palette(rot=-.2, as_cmap=True))
            axes[1, i-3].set_title(f"{num} with fasttext dimension {(i+1)*2}")
        figure.savefig(f"/media/lorenzo/Partizione Dati/progettoBDA/results_e10/{num}e10")

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
