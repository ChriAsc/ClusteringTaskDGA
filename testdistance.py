import os.path
import time
import math
import numpy as np
import pandas as pd
import fasttext
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, silhouette_score

basepath_train_data = "/media/lorenzo/Partizione Dati/progettoBDA/datasets/"
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
        raise ValueError(
            f"model type: {model_type} does not exists. Please insert a correct model type: \n skipgram or cbow")
    if mode not in ['characters', 'bigrams', 'trigrams']:
        raise ValueError(f"Chose correct embedding type, embedding {mode} does not exist")
    train_data_path += f"bambenek{mode}.txt"
    start_training = time.time()
    model = fasttext.train_unsupervised(train_data_path, model=model_type, dim=dim, epoch=epoch)
    end_training = time.time()
    print(f"Completed Fasttext training and generation of bigrams embeddings")
    print(f"Process took: {time.strftime('%H hour %M minutes %S seconds', time.gmtime(end_training - start_training))}")
    return model


def run(embedding_type="characters"):
    # DEFINIZIONE STRUTTURA FOGLIO DEI RISULTATI
    columns = ["iteration", "epochs", "dimension", "max_distance"]
    dataset_base = pd.read_csv('/media/lorenzo/Partizione Dati/progettoBDA/datasets/bambenekBigrams.csv')
    old_result = pd.read_csv("/media/lorenzo/Partizione Dati/progettoBDA/datasets/distances_results.csv").iloc[-1]
    for i in range(580, 1001):
        dataset = dataset_base.sample(frac=0.01)
        domain_names = dataset[embedding_type].to_numpy()
        max_len = np.max([len(x.split()) for x in domain_names])
        for epoch in range(10, 11, 5):
            for dim in range(2, 11, 2):
                if int(old_result["iteration"]) < i or (int(old_result["iteration"]) and old_result["dimension"] < dim):
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
                    distances = []
                    for name in embedded_domain_names:
                        for name2 in embedded_domain_names:
                            norm = np.linalg.norm(name - name2)
                            distances.append(norm)
                    max_distance = np.max(distances)
                    print(f"max_distance={max_distance} with epoch={epoch} dim={dim}, iteration={i}")
                    results = pd.DataFrame([[i, epoch, dim, max_distance]], columns=columns)
                    results.to_csv(f"/media/lorenzo/Partizione Dati/progettoBDA/datasets/distances_results.csv", index=False, mode='a',
                                   header=not os.path.exists(f"/media/lorenzo/Partizione Dati/progettoBDA/datasets/distances_results.csv"))


if __name__ == "__main__":
    run("bigrams")
