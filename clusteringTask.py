import os.path
import time

import numpy as np
import pandas as pd
import fasttext
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, silhouette_score

basepath_train_data = ""
embedding_params = {
    "characters": (1, 1),
    "bigrams": (2, 2),
    "trigrams": (3, 3)
}

def getDict(model):
    """ Convert FastText bin model in a character dictionary for the embedding
        Parameters
        ---------
        model: model
            The fasttext model
    """
    # Ritorna la lista di tutte le parole nel dizionario del modello con i vettori associati
    # il dizionario ha come chiave la parola e come valore il vettore numerico con cui essa è rappresentata
    words = model.get_words()
    word_dict = {w: model.get_word_vector(w) for w in words}
    """
    for w in words:
        # Prendo la rappresentazione vettoriale della parola ciclata attualmente che è vector
        vector = model.get_word_vector(w)
        dict[w] = vector
    """
    return word_dict


def run_fasttext_training(train_data_path, model_type, dim, epoch, mode="characters"):
    """ Trains FastText Embedding Layer
        Parameters
        ---------
        train_data_path: str
            String representing the path in which FastText Training dataset is saved
        model_type: str
            String representing the fasttext chosen model(skipgram or cbow)
        dim: int
            Number representing the number of dimensions
        epoch: int 
            Number representing the number of epochs
        mode: str
            embedding mode characters bigrams or trigrams
    """
    if model_type not in ['skipgram', 'cbow']:
        raise ValueError(f"model type: {model_type} does not exists. Please insert a correct model type: \n skipgram or cbow")
    if mode not in ['characters', 'bigrams', 'trigrams']:
        raise ValueError(f"Chose correct embedding type, embedding {mode} does not exist")
    train_data_path += f"bambenek{mode}.txt"
    start_training = time.time()
    model = fasttext.train_unsupervised(train_data_path, model=model_type, dim=dim, epoch=epoch)
    end_training = time.time()
    print(f"Completed Fasttext training and generation of bigrams embeddings")
    print(f"Process took: {time.strftime('%H hour %M minutes %S seconds', time.gmtime(end_training-start_training))}")
    return model


def run(embedding_type="characters"):
    # NUMERI DA SCEGLIERE
    start = time.time()
    # DEFINIZIONE STRUTTURA FOGLIO DEI RISULTATI
    columns = ["epochs", "dimension", "homogeneity", "completeness",
               "v_measure", "silhouette_score", "num_clusters", "num_noise", "duration"]
    # COLLEZIONAMENTO DATI DAL DATASET
    dataset = pd.read_csv('bambenekBigrams.csv')
    domain_names = dataset[embedding_type].to_numpy()
    family_dict = {family: i for i, family in enumerate(sorted(set(dataset["family"])), 1)}
    labels_true = [family_dict[family] for family in dataset["family"].to_numpy()]
    max_len = np.max([len(x.split()) for x in domain_names])
    epsilons = [(4*10*max_len)/i for i in [32, 16, 8, 4, 2]]
    for epoch in range(3, 8, 2):
        for dim in range(2, 11, 2):
            model_skipgram = run_fasttext_training(basepath_train_data, "skipgram", dim, epoch, embedding_type)
            dict_skipgram = getDict(model_skipgram)
            embedded_domain_names = []
            for name in domain_names:
                sequences = np.array(
                    [dict_skipgram.get(token) if dict_skipgram.get(token) is not None else np.zeros(dim)
                     for token in name.split()], dtype=np.single)
                pad = np.zeros(dim * (max_len - len(name.split())), dtype=np.single)
                embedded_domain_name = np.concatenate((sequences, pad), axis=None, dtype=np.single)
                embedded_domain_names.append(embedded_domain_name)
            for minPoints in range((dim*max_len)+1, (2*dim*max_len)+1, (dim*max_len) // 5):
                for eps in epsilons:
                    start_iteration = time.time()
                    # Determinazione della matrice Sparsa per semplificare il DBSCAN
                    nn: NearestNeighbors = NearestNeighbors(n_neighbors=minPoints, metric="euclidean", algorithm="auto")
                    nn.fit(embedded_domain_names)
                    sparseMatrix = nn.radius_neighbors_graph(embedded_domain_names, radius=eps, mode="distance")
                    db = DBSCAN(eps=eps, min_samples=minPoints, metric="precomputed")
                    db.fit(sparseMatrix)
                    # CALCOLO DELLE METRICHE SCELTE
                    labels = db.labels_
                    numClusters = len(set(labels)) - (1 if -1 in labels else 0)
                    numNoise = list(labels).count(-1)
                    homogeneity = homogeneity_score(labels_true, labels)
                    completeness = completeness_score(labels_true, labels)
                    v1_measure = v_measure_score(labels_true, labels)
                    silhouette = silhouette_score(embedded_domain_names, metric="euclidean")
                    end_iteration = time.time()
                    print()
                    print({"numNoise": numNoise, "numClusters": numClusters})
                    print({"homogeneity": homogeneity, "completeness": completeness, "v1_measure": v1_measure,
                           "silhouette": silhouette})
                    # SCRITTURA DEI RISULTATI SU FILE
                    results = pd.DataFrame([[epoch, dim, homogeneity, completeness,
                                             v1_measure, silhouette, numClusters, numNoise,
                                             end_iteration-start_iteration]], columns=columns)
                    results.to_csv(f"{embedding_type}_results.csv", index=False, mode='a',
                                   header=not os.path.exists(f"{embedding_type}_results.csv"))
                    print()
                    print(f"Completed iteration in {end_iteration-start_iteration}")
                    print(f"Iteration data: eps={eps}, minPoints={minPoints}, dim={dim} epochs={epoch}")
    end = time.time()
    print(f"Took {end-start} seconds to perform task with {embedding_type} embedding")


if __name__ == "__main__":
    run("bigrams")

