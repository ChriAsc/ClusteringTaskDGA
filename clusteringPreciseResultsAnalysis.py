import matplotlib.figure
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

epoch = 20
min_dim = 2
max_dim = 4
eps_dict = {2: 2.25, 3: 2.75, 4: 3.0}
min_Points_dict = {2: 95, 3: 142, 4: 189}
passo = 1
measures_dict = {"precision_data": "Precision", "recall_data": "Recall", "confusion_matrix": "Confusion Matrix"}
label_dict = {"precision_data": "Precision", "recall_data": "Recall", "confusion_matrix": "Numero campioni"}
sns.set()
base_path_read = f"D:\progettoBDA\Terza Passata\\"
base_path_write = f"D:\progettoBDA\Terza Passata\\results\\"


def print_graphs(metric, format_parameter):
    for dim in range(min_dim, max_dim+1, passo):
            data = pd.read_csv(
                f"{base_path_read}{metric}_e{epoch}_d{dim}_mPoints{min_Points_dict[dim]}.csv").set_index(
                "Unnamed: 0").rename_axis("Families")
            sns.heatmap(data, annot=True, linewidths=.5, fmt=format_parameter, cmap='crest', annot_kws={"size": 14},
                        cbar_kws={"shrink": 0.5, "label": f"{label_dict[metric]}"}, xticklabels=True, yticklabels=True)
            figure: matplotlib.figure.Figure = plt.gcf()
            figure.axes[-1].yaxis.label.set_size(18)
            figure.axes[-1].tick_params(labelsize=18)
            figure.suptitle(f"{measures_dict[metric]} with epoch={epoch} dimension={dim}, minPoints={min_Points_dict[dim]} and eps={eps_dict[dim]}", fontsize=24)
            figure.set_size_inches(25.6, 16)
            plt.xticks(rotation=0, fontsize=16)
            plt.yticks(fontsize=16)
            figure.savefig(f"{base_path_write}{metric}_e{epoch}_d{dim}_mPoints{min_Points_dict[dim]}")
            plt.clf()


if __name__ == "__main__":
    print_graphs("precision_data", ".6f")
    print_graphs("recall_data", ".6f")
    print_graphs("confusion_matrix", "d")
    data = pd.read_csv(f"{base_path_read}bigrams_results.csv")
    graphics = ["homogeneity", "completeness", "silhouette_score"]
    figure, axes = plt.subplots(3,1)
    figure.suptitle(f"Metrics with epoch={epoch} dimension={3}, and eps={eps_dict[3]}", fontsize=24)
    figure.set_size_inches(25.6, 16)
    figure.tight_layout(pad=5.0)
    for graph in graphics:
        sns.lineplot(data=data, x="minPoints", y=graph, ax=axes[graphics.index(graph)], marker='o')
        axes[graphics.index(graph)].set_title(f"{graph} with dim=3 eps=2.75 and epoch=20", fontsize=18)
        plt.xticks(rotation=0, fontsize=14)
        plt.yticks(fontsize=14)
    figure.savefig(f"{base_path_write}metrics_e{epoch}_d{3}_lessPrecise")
