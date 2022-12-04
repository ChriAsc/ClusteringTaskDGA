import sys
import time
import numpy as np
import json
import matplotlib.figure
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

base_path = "D:\progettoBDA\\"

embedding_params = {
    "characters": (1, 1),
    "bigrams": (2, 2),
    "trigrams": (3, 3)
}

family_to_study = {
    2: ["dyre", "monerodownloader", "zeus-newgoz"],
    3: ["dyre", "monerodownloader", "bazarbackdoor", "zeus-newgoz"],
    4: ["monerodownloader", "bazarbackdoor", "zeus-newgoz"]
}

colors = {
    2: np.array(['r', 'g']),
    3: np.array(['r', 'g', 'b']),
    4: np.array(['r', 'g', 'b', 'c'])
}

categories = {
    2: [0, 1]*47,
    3: [0, 1, 2]*47,
    4: [0, 1, 2, 3]*47
}

steps = {
    2: 3,
    3: 4,
    4: 4
}

def fasttextPreTrained(vecfile):
    """
    Creates a dictionary in which every character, bigram or trigram is associated with a numeric vector
    :param vecfile: str
        String representing path in which FastText vecfile is located
    :return: embeddings_index: dict
        Dictionary containing the vectorization of characters,bigrams or trigrams
        key: character,bigram or trigram
        value: numeric vector associated to key
    """
    ######################################
    # EMBEDDING LAYER
    #####################################
    embeddings_index = {}
    f = open(vecfile, "r")
    next(f)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index


def run(embedding_type="characters"):
    # COLLEZIONAMENTO DATI DAL DATASET
    dataset = pd.read_csv(f'{base_path}/datasets/bambenekBigrams.csv')
    domain_names = dataset[embedding_type]
    max_len = np.max([len(x.split()) for x in domain_names])
    results = {f"dim_{dim}": {family: {metric: [] for metric in ["mean", "var", "std"]} for family in family_to_study[dim]}
               for dim in range(2,5)}
    for dim in range(2,5):
        dict_skipgram = fasttextPreTrained(f"{base_path}/vecs/dim_{dim}.vec")
        for family in family_to_study[dim]:
            dataset_to_study = dataset[dataset["family"] == family]
            domain_names_to_study = dataset_to_study[embedding_type]
            if dim == 3:
                lens = pd.DataFrame([len(x.split()) for x in domain_names_to_study])
                sns.displot(lens)
                plt.show()
            """embedded_domain_names = []
            for name in domain_names_to_study:
                sequences = np.array(
                    [dict_skipgram.get(token) if dict_skipgram.get(token) is not None else np.zeros(dim)
                     for token in name.split()], dtype=np.single)
                pad = np.zeros(dim * (max_len - len(name.split())), dtype=np.single)
                embedded_domain_name = np.concatenate((sequences, pad), axis=None, dtype=np.single)
                embedded_domain_names.append(embedded_domain_name)
            embedded_domain_names = np.array(embedded_domain_names)
            np.set_printoptions(threshold=sys.maxsize)
            results[f"dim_{dim}"][family]["mean"] = np.mean(embedded_domain_names, axis=0).tolist()
            results[f"dim_{dim}"][family]["var"] = np.var(embedded_domain_names, axis=0).tolist()
            shape = np.arange(len(results[f"dim_{dim}"][family]["var"]))
            figure, axes = plt.subplots(2, 1)
            figure.suptitle(f"{family} mean and variance with dimension {dim}", fontsize=25)
            axes[0].scatter(shape, results[f"dim_{dim}"][family]["mean"],  c=colors[dim][categories[dim]])
            axes[1].scatter(shape, results[f"dim_{dim}"][family]["var"],  c=colors[dim][categories[dim]])
            axes[0].grid()
            axes[1].grid()
            axes[0].set_xticks(np.arange(0, dim*max_len+1, step=steps[dim]))
            axes[1].set_xticks(np.arange(0, dim*max_len+1, step=steps[dim]))
            axes[0].set_title("Mean")
            axes[1].set_title("Variance")
            figure.set_size_inches(30,20)
            figure.savefig(f"{base_path}\Analisi\{family}_dim{dim}")
            plt.clf()"""


run("bigrams")