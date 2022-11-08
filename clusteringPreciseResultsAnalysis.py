import matplotlib.figure
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

epoch = 10
min_dim = 4
max_dim = 5
eps_dict = {4: 3.0, 5: 3.6}
min_Points_dict = {4: [189, 226], 5: [236, 283]}
passo = 1
measures_dict = {"precision_data": "Precision", "recall_data": "Recall", "confusion_matrix": "Confusion Matrix"}
label_dict = {"precision_data": "Precision", "recall_data": "Recall", "confusion_matrix": "Numero campioni"}
sns.set()
base_path_read = f"D:\progettoBDA\PrecisedResults\\epoch_{epoch}_detailed\\"
base_path_write = f"D:\progettoBDA\PrecisedResults\\results\epoch_{epoch}\\"


def print_graphs(metric, format_parameter):
    for dim in range(min_dim, max_dim+1, passo):
        for min_Points in min_Points_dict[dim]:
            data = pd.read_csv(
                f"{base_path_read}{metric}_e{epoch}_d{dim}_mPoints{min_Points}.csv").set_index(
                "Unnamed: 0").rename_axis("Families")
            sns.heatmap(data, annot=True, linewidths=.5, fmt=format_parameter, cmap='crest', annot_kws={"size": 14},
                        cbar_kws={"shrink": 0.5, "label": f"{label_dict[metric]}"}, xticklabels=True, yticklabels=True)
            figure: matplotlib.figure.Figure = plt.gcf()
            figure.axes[-1].yaxis.label.set_size(18)
            figure.axes[-1].tick_params(labelsize=18)
            figure.suptitle(f"{measures_dict[metric]} with epoch={epoch} dimension={dim}, minPoints={min_Points} and eps={eps_dict[dim]}", fontsize=24)
            figure.set_size_inches(25.6, 16)
            plt.xticks(rotation=0, fontsize=16)
            plt.yticks(fontsize=16)
            figure.savefig(f"{base_path_write}{metric}_e{epoch}_d{dim}_mPoints{min_Points}")
            plt.clf()


if __name__ == "__main__":
    print_graphs("precision_data", ".6f")
    print_graphs("recall_data", ".6f")
    print_graphs("confusion_matrix", "d")
