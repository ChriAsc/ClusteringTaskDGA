import sys
import time
import numpy as np
import json
import matplotlib.figure
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

base_path = "/media/lorenzo/Partizione Dati/progettoBDA/"

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
            embedded_domain_names = []
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
            results[f"dim_{dim}"][family]["std"] = np.std(embedded_domain_names, axis=0).tolist()
    #plt.plot(results["dim_2"]["zeus-newgoz"]["mean"], 'ro')
    plt.plot(results["dim_2"]["zeus-newgoz"]["var"], 'bo')
    #plt.plot(np.add(results["dim_2"]["zeus-newgoz"]["std"], results["dim_2"]["zeus-newgoz"]["mean"]), 'go')
    #plt.plot(np.add(np.negative(results["dim_2"]["zeus-newgoz"]["std"]), results["dim_2"]["zeus-newgoz"]["mean"]), 'bo')
    plt.show()


run("bigrams")