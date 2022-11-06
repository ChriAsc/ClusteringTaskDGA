import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

epoch = 15
min_dim = 4
max_dim = 6
passo = 1
base_path = f"/media/lorenzo/Partizione Dati/progettoBDA/results_e{epoch}/"
base_path_windows = f"D:\progettoBDA\\result_e{epoch}\\second_"
first_part = pd.read_csv(f"{base_path_windows}bigrams_results_e{epoch}_d4-5.csv")
second_part = pd.read_csv(f"{base_path_windows}bigrams_results_e{epoch}_d6.csv")
#third_part = pd.read_csv(f"{base_path_windows}bigrams_results_e{epoch}_d6.csv")
frames = [first_part, second_part]#, third_part]
complete = pd.concat(frames)
sns.set()
graphics = [["homogeneity", "completeness"], ["silhouette_score"], ["v_measure"]]
for graph in graphics:
    figure, axes = plt.subplots(len(graph), len(list(range(0, len(range(min_dim, max_dim+1, passo))))))
    figure.suptitle(f"{graph[0]} and {graph[1]} scores for all dimensions with epoch={epoch}" if len(graph) > 1
                    else f"{graph[0]} scores for all dimensions with epoch={epoch}")
    if len(graph) > 1:
        figure.set_size_inches(6*len(list(range(0, len(range(min_dim, max_dim+1, passo))))), 12)
    else:
        figure.set_size_inches(6*len(list(range(0, len(range(min_dim, max_dim+1, passo))))), 6)
    figure.tight_layout(pad=3.5)
    for measure in graph:
        dimensions = [complete[complete["dimension"] == dim].pivot("minPoints", "epsilon", measure)
                      for dim in range(min_dim, max_dim+1, passo)]
        for i in range(0, len(range(min_dim, max_dim+1, passo))):
            dimensions[i].sort_index(level=0, ascending=False, inplace=True)
            if len(graph) > 1:
                sns.heatmap(dimensions[i], ax=axes[graph.index(measure), i], annot=True, linewidths=.5, fmt='.4f',
                            cmap='crest')
                axes[graph.index(measure), i].set_title(f"{measure} with fasttext dimension {i+4}")
            else:
                sns.heatmap(dimensions[i], ax=axes[i], annot=True, linewidths=.5, fmt='.4f',
                            cmap='crest')
                axes[i].set_title(f"{measure} with fasttext dimension {i+4}")
    figure.savefig(f"{base_path_windows}{graph[0]}_{graph[1]}_e{epoch}"
                   if len(graph) > 1 else f"{base_path_windows}{graph[0]}_e{epoch}")

nums = ["num_clusters", "num_noise"]
figure, axes = plt.subplots(2, len(list(range(0, len(range(min_dim, max_dim+1, passo))))))
figure.suptitle(f"{nums[0]} and {nums[1]} data for all dimensions with epoch={epoch}")
figure.set_size_inches(6*len(list(range(0, len(range(min_dim, max_dim+1, passo))))), 12)
figure.tight_layout(pad=3.5)
for num in nums:
    dimensions = [complete[complete["dimension"] == dim].pivot("minPoints", "epsilon", num)
                  for dim in range(min_dim, max_dim+1, passo)]
    for i in range(0, len(range(min_dim, max_dim+1, passo))):
        dimensions[i].sort_index(level=0, ascending=False, inplace=True)
        sns.heatmap(dimensions[i], ax=axes[nums.index(num), i], annot=True, linewidths=.5, fmt='d',
                    cmap=sns.cubehelix_palette(rot=-.2, as_cmap=True))
        axes[nums.index(num), i].set_title(f"{num} with fasttext dimension {i+4}")
figure.savefig(f"{base_path_windows}{nums[0]}_{nums[1]}e{epoch}")

# SECOND PART OF GRAPHS

measures = ["homogeneity", "completeness", "silhouette_score"]
figure, axes = plt.subplots(len(measures), len(list(range(0, len(range(min_dim, max_dim+1, passo))))))
figure.suptitle(f"All scores metrics for all dimensions with epoch={epoch}")
figure.set_size_inches(6*len(list(range(0, len(range(min_dim, max_dim+1, passo))))), 6*len(measures))
figure.tight_layout(pad=3.5)
for measure in measures:
    dimensions = [complete[complete["dimension"] == dim].pivot("minPoints", "epsilon", measure)
                  for dim in range(min_dim, max_dim+1, passo)]
    for i in range(0, len(range(min_dim, max_dim+1, passo))):
        dimensions[i].sort_index(level=0, ascending=False, inplace=True)
        sns.heatmap(dimensions[i], ax=axes[measures.index(measure), i], annot=True, linewidths=.5, fmt='.4f',
                    cmap='crest')
        axes[measures.index(measure), i].set_title(f"{measure} with fasttext dimension {i+4}")
figure.savefig(f"{base_path_windows}metrics_e{epoch}")

# THIRD PART OF GRAPH

measures = ["homogeneity", "completeness", "num_clusters", "num_noise"]
figure, axes = plt.subplots(len(measures), len(list(range(0, len(range(min_dim, max_dim+1, passo))))))
figure.suptitle(f"Metric scores with clusters and noise for all dimensions with epoch={epoch}")
figure.set_size_inches(6*len(list(range(0, len(range(min_dim, max_dim+1, passo))))), 6*len(measures))
figure.tight_layout(pad=3.5)
for measure in measures:
    dimensions = [complete[complete["dimension"] == dim].pivot("minPoints", "epsilon", measure)
                  for dim in range(min_dim, max_dim+1, passo)]
    for i in range(0, len(range(min_dim, max_dim+1, passo))):
        dimensions[i].sort_index(level=0, ascending=False, inplace=True)
        if measures.index(measure) < 2:
            sns.heatmap(dimensions[i], ax=axes[measures.index(measure), i], annot=True, linewidths=.5, fmt='.4f',
                        cmap='crest')
            axes[measures.index(measure), i].set_title(f"{measure} with fasttext dimension {i+4}")
        else:
            sns.heatmap(dimensions[i], ax=axes[measures.index(measure), i], annot=True, linewidths=.5, fmt='d',
                        cmap='crest')
            axes[measures.index(measure), i].set_title(f"{measure} with fasttext dimension {i+4}")
figure.savefig(f"{base_path_windows}metrics_noise_clusters_e{epoch}")