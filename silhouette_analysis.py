import matplotlib.figure
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

epoch = 20
min_dim = 2
max_dim = 4
eps_dict = {2: 2.25, 3: 2.75, 4: 3.0}
min_Points_dict = {2: 95, 3: 142, 4: 189}
cluster_dict = {2: 2, 3: 5}
passo = 1
sns.set()
base_path_read = f"D:\progettoBDA\Terza Passata\\"
base_path_write = f"D:\progettoBDA\Terza Passata\\results\\"
dim = 4
cluster = 5
data = pd.read_csv(f"{base_path_read}silhouettes_data_e{epoch}_d{dim}_mPoints{min_Points_dict[dim]}.csv")
# cluster_list = list(set(data["pred_label"]))
figure, axes = plt.subplots(1,2)
figure.set_size_inches(25.6,10)
plot_data = data[data["pred_label"] == 5]
plot_data_2 = data[(data["true_label"] == 2) & (data["pred_label"] == -1)]
sns.histplot(plot_data, x="silhouettes", kde='kde', ax=axes[0])
sns.scatterplot(plot_data_2, x=range(0, len(plot_data_2)), y="silhouettes", ax=axes[1])
axes[0].tick_params(labelsize=14)
axes[1].tick_params(labelsize=14)
axes[0].set_xlabel("Silhouette", fontsize=14)
axes[0].set_ylabel("Count", fontsize=14)
axes[1].set_xlabel("Index", fontsize=14)
axes[1].set_ylabel("Silhouette", fontsize=14)
plt.show()
figure.savefig(f"{base_path_write}silhouettes_bazar")