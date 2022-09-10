import numpy as np
import pandas as pd
import fasttext
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
#from scipy.spatial import distance_matrix

basepath_train_data = ""


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
    dict = {}
    for w in words:
        # Prendo la rappresentazione vettoriale della parola ciclata attualmente che è vector
        vector = model.get_word_vector(w)
        dict[w] = vector
    return dict


def run_fasttext_training(train_data_path, model_type, dim, epoch):
    """ Trains FastText Embedding Layer
        Parameters
        ---------
        train_data_path: str
            String representing the path in which FastText Training dataset is saved
        model_type: str
            String representing the fasttext chosen model(skipgram or cbow)
        dim: int
            Int representing the number of dimensions
        epoch: int 
            Int representing the number of epochs
    """
    train_data_path += "Chars.txt"
    if model_type == 'skipgram':
        print("You chose FastText model type skipgram")
        model = fasttext.train_unsupervised(
            train_data_path, model='skipgram', dim=dim, epoch=epoch)
        return model
    if model_type == 'cbow':
        print("You chose FastText model type cbow")
        model = fasttext.train_unsupervised(
            train_data_path, model='cbow', dim=dim, epoch=epoch)
        return model
    if model_type != 'cbow' and model_type != 'skipgram':
        raise ValueError("model type: " + model_type + " does not exists. Please insert a correct model type: \n"
                         + "skipgram or cbow")


def getDist(mat1, mat2):
    return np.linalg.norm(mat1-mat2) 


try:
    # NUMERI DA SCEGLIERE
    dataset = pd.read_csv('dataset.csv')  # NOME TEMPORANEO?
    domainNames = list(dataset['noDotsDomain'].to_numpy)
    max_len = max(domainNames)
    for epoch in range(1, 11):
        for dim in range(100, 301):
            model_skipgram = run_fasttext_training(
                basepath_train_data, "skipgram", dim, epoch)
            dict_skipgram = getDict(model_skipgram)
            #model_cbow=run_fasttext_training(basepath_train_data, "cbow", dim, epoch)
            # dict_cbow=getDict(model_cbow)
            embeddedDomainNames = []
            for name in domainNames:
                #embeddedDomainName = [dict_skipgram[char] for char in name].append([0 for i in range(len(name), max_len)])
                embeddedDomainName = np.concatenate(
                    np.array([dict_skipgram[char] for char in name]), np.zeros(dim, max_len-len(name)), axis=1)
                embeddedDomainNames.append(embeddedDomainName)
            # SCELTA PARAMETRI DBSCAN
            # a rule of thumb is to derive minPts from the number of dimensions D in the data set. minPts >= D + 1.
            # For larger datasets, with much noise, it suggested to go with minPts = 2 * D.
            minPoints = dim+1
            # cerco eps adatto
            neigh = NearestNeighbors(
                n_neighbors=minPoints, metric=getDist).fit(embeddedDomainNames)
            distances = neigh.kneighbors(embeddedDomainNames)[0].sort()
            kneedle = KneeLocator(range(
                1, len(distances)+1), distances, S=1.0, curve='convex', direction='increasing')
            eps = round(kneedle.knee_y, 8)
            # ESECUZIONE DBSCAN
            # CONTROLLO PERFORMANCE RISPETTO AL PRECEDENTE RISULTATO MIGLIORE
except ValueError:
    print("error")
