import fasttext
import time

basepath_train_data = "/media/lorenzo/Partizione Dati/progettoBDA/datasets/"
path_vec = "/media/lorenzo/Partizione Dati/progettoBDA/vecs/"


def save_vec(model, dim):
    """ Convert FastText bin model in a character dictionary for the embedding
        Parameters
        ---------
        model: model
            The fasttext model
    """
    words = model.get_words()
    with open(f"{path_vec}dim_{dim}.vec", 'w') as file_out:
        file_out.write(str(len(words)) + " " + str(model.get_dimension()) + "\n")
        for w in words:
            # Prendo la rappresentazione vettoriale della parola ciclata attualmente che è vector
            vector = model.get_word_vector(w)
            vector_string = ""
            for value in vector:
                vector_string += " " + str(value)
            try:
                file_out.write(w + vector_string + "\n")
            except:
                pass


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


model_2 = run_fasttext_training(basepath_train_data, "skipgram", 2, 20, "bigrams")
save_vec(model_2, 2)
model_3 = run_fasttext_training(basepath_train_data, "skipgram", 3, 20, "bigrams")
save_vec(model_3, 3)
model_4 = run_fasttext_training(basepath_train_data, "skipgram", 4, 20, "bigrams")
save_vec(model_4, 4)