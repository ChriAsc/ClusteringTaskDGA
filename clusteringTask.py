import os.path
import time

import numpy as np
import pandas as pd
import fasttext
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, silhouette_score
from sklearn.feature_extraction.text import CountVectorizer

basepath_train_data = ""
embedding_params = {
    "characters": (1,1),
    "bigrams": (2,2),
    "trigrams": (3,3)
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
    train_data_path += f"{mode}.txt"
    print(f"You chose FastText model type {model_type} and embedding mode {mode}")
    model = fasttext.train_unsupervised(train_data_path, model=model_type, dim=dim, epoch=epoch)
    return model


def dist(v1, v2, **kwargs):
    mat1 = np.concatenate([np.reshape(kwargs["embedding"][int(elem)], (kwargs["dim"],1))
                           for elem in v1], axis=1)
    mat2 = np.concatenate([np.reshape(kwargs["embedding"][int(elem)], (kwargs["dim"],1))
                           for elem in v2], axis=1)
    return np.linalg.norm(mat1-mat2)


def run(embedding_type="characters"):
    # NUMERI DA SCEGLIERE
    start = time.time()
    columns = ["epochs", "dimension", "homogeneity", "completeness",
               "v_measure", "silhouette_score", "num_clusters", "num_noise" "duration"]
    results = pd.DataFrame(columns=columns)
    dataset = pd.read_csv('bambenekBigrams.csv')  # NOME TEMPORANEO
    domain_names = dataset[embedding_type].to_numpy()
    # domain_names_for_vectorizing = dataset["noDotsDomain"].to_numpy()
    family_dict = {family: i for i, family in enumerate(sorted(set(dataset["family"])), 1)}
    labels_true = [family_dict[family] for family in dataset["family"].to_numpy()]
    max_len = np.max([len(x) for x in domain_names])
    #vectorizer = CountVectorizer(analyzer='char', ngram_range=embedding_params[embedding_type])
    #vectorizer.fit(domain_names_for_vectorizing)
    #word_index = vectorizer.vocabulary_
    #max_features = len(word_index)
    for epoch in range(3, 8, 2):
        for dim in range(8, 21, 3):
            start_iteration = time.time()
            #embedding_matrix = np.zeros((max_features+1, dim))
            model_skipgram = run_fasttext_training(basepath_train_data, "skipgram", dim, epoch, embedding_type)
            dict_skipgram = getDict(model_skipgram)
            """for w, i in word_index.items():
                if dict_skipgram.get(w) is not None:
                    embedding_matrix[i] = dict_skipgram[w]"""
            embedded_domain_names = []
            for name in domain_names:
                sequences = np.array(
                    [dict_skipgram.get(token) if dict_skipgram.get(token) is not None else np.zeros(dim) for token in
                     name.split()], dtype=np.single)
                pad = np.zeros(dim * (max_len - len(name.split())), dtype=np.single)
                embedded_domain_name = np.concatenate((sequences, pad), axis=None, dtype=np.single)
                embedded_domain_names.append(embedded_domain_name)
            # SCELTA PARAMETRI DBSCAN
            # a rule of thumb is to derive minPts from the number of dimensions D in the data set. minPts >= D + 1.
            # For larger datasets, with much noise, it suggested to go with minPts = 2 * D.
            for minPoints in range((dim*max_len)+1, (2*dim*max_len)+1, (dim*max_len) // 5):
                min_eps = max_eps = passo = 0
                for eps in range(min_eps, max_eps, passo):
                    #SE HO BEN CAPITO VA PASSATA LA MATRICE SOPRA SE PRECALCOLATA
                    db = DBSCAN(eps=eps, min_samples=minPoints, metric="euclidean")
                    db.fit(embedded_domain_names)
                    # CONTROLLO PERFORMANCE RISPETTO AL PRECEDENTE RISULTATO MIGLIORE
                    labels = db.labels_
                    numClusters = len(set(labels)) - (1 if -1 in labels else 0)
                    numNoise = list(labels).count(-1)
                    #NON SONO SICURO SU QUALI METRICHE PRENDERE
                    homogeneity = homogeneity_score(labels_true, labels)
                    completeness = completeness_score(labels_true, labels)
                    v1_measure = v_measure_score(labels_true, labels)
                    silhouette = silhouette_score(embedded_domain_names, metric="euclidean")
                    end_iteration = time.time()
                    results = pd.DataFrame([[epoch, dim, homogeneity, completeness,
                                             v1_measure, silhouette, numClusters, numNoise,
                                             end_iteration-start_iteration]], columns=columns)
                    results.to_csv(f"{embedding_type}_results.csv", index=False, mode='a',
                                   header=not os.path.exists(f"{embedding_type}_results.csv"))
    end = time.time()
    print(f"Took {end-start} seconds to perform task with {embedding_type} embedding")


if __name__ == "__main__":
    run("bigrams")

