import os.path
import time
import numpy as np
import pandas as pd
import fasttext
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_samples, confusion_matrix

basepath_train_data = "/home/lorenzo/progettoBDA/datasets/"
base_path = "/home/lorenzo/progettoBDA/PrecisedResults/"
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
    columns = ["epochs", "dimension", "minPoints", "epsilon", "homogeneity", "completeness",
               "v_measure", "silhouette_score", "num_clusters", "num_noise", "duration"]
    # COLLEZIONAMENTO DATI DAL DATASET
    dataset = pd.read_csv('/home/lorenzo/progettoBDA/datasets/bambenekBigrams.csv').sample(frac=0.01)
    domain_names = dataset[embedding_type].to_numpy()
    family_dict = {family: i for i, family in enumerate(sorted(set(dataset["family"])), 1)}
    labels_true = [family_dict[family] for family in dataset["family"].to_numpy()]
    labels_true_column = pd.Series(labels_true)
    family_label_column = dataset["family"]
    max_len = np.max([len(x.split()) for x in domain_names])
    print(f"Starting Clustering algorithm. No_samples={len(dataset)}")
    eps = {4: 3.0, 5: 3.6}
    min_Points_dict = {4: [189, 226], 5: [236, 283]}
    for dim in range(4, 6):
        model_skipgram = run_fasttext_training(basepath_train_data, "skipgram", dim, 2, embedding_type)
        dict_skipgram = getDict(model_skipgram)
        embedded_domain_names = []
        for name in domain_names:
            sequences = np.array(
                [dict_skipgram.get(token) if dict_skipgram.get(token) is not None else np.zeros(dim)
                 for token in name.split()], dtype=np.single)
            pad = np.zeros(dim * (max_len - len(name.split())), dtype=np.single)
            embedded_domain_name = np.concatenate((sequences, pad), axis=None, dtype=np.single)
            embedded_domain_names.append(embedded_domain_name)
        for minPoints in min_Points_dict[dim]:
            start_iteration = time.time()
            db = DBSCAN(eps=eps[dim], min_samples=minPoints, metric="euclidean", algorithm='auto')
            db.fit(embedded_domain_names)
            # CALCOLO DELLE METRICHE PUNTUALI
            labels = db.labels_
            labels_pred_column = pd.Series(labels)
            numClusters = len(set(labels)) - (1 if -1 in labels else 0)
            numNoise = list(labels).count(-1)
            """silhouettes = silhouette_samples(embedded_domain_names, labels, metric="euclidean")
            silhouette_column = pd.Series(silhouettes)
            results_silhouettes = pd.DataFrame({"family": family_label_column, "true_labels": labels_true_column,
                                                "pred_labels": labels_pred_column, "silhouettes": silhouette_column})
            results_silhouettes.to_csv(f"{base_path}/silhouettes_data_e{20}_d{dim}")"""
            # MATRICE DI CONFUSIONE
            df = pd.DataFrame({'Labels': labels_true, 'Clusters': labels})
            ct = pd.crosstab(df['Labels'], df['Clusters'])
            print(ct)
            precisions = {}
            # RECALL E PRECISION PER OGNI CLUSTER
            to_cycle = set(labels)
            to_cycle.remove(-1)
            if to_cycle is not None:
                for cluster in sorted(to_cycle):
                    precision_list = [ct.iloc[family-1][cluster]/sum(ct[cluster]) for family in sorted(set(labels_true))]
                    precisions.update({cluster: precision_list})
                precision_tab = pd.DataFrame(precisions)
                print(precision_tab)
            end_iteration = time.time()
            # SCRITTURA DEI RISULTATI SU FILE
            print()
            print(f"Completed iteration in {time.strftime('%H hour %M minutes %S seconds', time.gmtime(end_iteration-start_iteration))}")
            print(f"Iteration data: eps={eps}, minPoints={minPoints}, dim={dim} epochs={20}")
    end = time.time()
    print(f"Took {end-start} seconds to perform task with {embedding_type} embedding")


if __name__ == "__main__":
    run("bigrams")