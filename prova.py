import os
import numpy as np
import fasttext

from pyspark.context import SparkContext
from pyspark.sql.functions import *
from pyspark.sql.session import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from getNGrams import getNGrams
from datasetWriter import dataset_writer

def getDict(model):
    words = model.get_words()
    word_dict = {w: model.get_word_vector(w) for w in words}
    return word_dict

def run_fasttext_training(train_data_path, model_type, dim, epoch, mode="characters"):
    if model_type not in ['skipgram', 'cbow']:
        raise ValueError(f"model type: {model_type} does not exists. Please insert a correct model type: \n skipgram or cbow")
    if mode not in ['characters', 'bigrams', 'trigrams']:
        raise ValueError(f"Chose correct embedding type, embedding {mode} does not exist")
    train_data_path += f"{mode}.txt"
    print(f"You chose FastText model type {model_type} and embedding mode {mode}")
    model = fasttext.train_unsupervised(train_data_path, model=model_type, dim=dim, epoch=epoch)
    return model

# se è gia esistente prende lo SparkContext, oppure lo crea
sc = SparkContext.getOrCreate()
# crea una sesione nello spark Context in quanto ci possono essere più session
spark = SparkSession(sc)

sqlContext = SQLContext(sc)

schema = StructType([
    StructField("class", StringType(), True),
    StructField("family", StringType(), True),
    StructField("noDotsDomain", StringType(), True),
    StructField("domain", StringType(), True),
])

dga_dataset = spark.createDataFrame([], schema)

path = f"{os.environ['HOME']}/progettoBDA/datasets"
df = spark.read.format("csv").option("delimiter", ",").option("header", "true").load(f"{path}/bambenekBigrams.csv")
df = df.limit(5)
domain_names = df.select(split("bigrams", '[ ]')).rdd.flatMap(lambda x: x)
family_list = df.select("family").rdd.flatMap(lambda x: x).collect()
family_dict = {family: i for i, family in enumerate(sorted(set(family_list)), 1)}
labels_true = [family_dict[family] for family in family_list]
max_len = df.select(size(split("bigrams", '[ ]')).alias("len")).select(max("len")).rdd.flatMap(lambda x: x).collect()[0]
model_skipgram = run_fasttext_training(f"{path}/", "skipgram", 100, 1, "bigrams")
dict_skipgram = getDict(model_skipgram)
embedded_domain_names = domain_names.map(lambda x: np.concatenate(
    ([dict_skipgram.get(token) if dict_skipgram.get(token) is not None else np.zeros(100) for token in x],
     np.zeros(100*(max_len-len(x)))), axis=None))
embedded_domain_names_2 = np.array(embedded_domain_names.collect())

"""classes = os.listdir(path)
classes.remove('legit')
classes.sort()"""

# creating a balanced dataset of DGA domain names
"""for family in classes:
    df = spark.read.format("text").load(f"{path}/{family}/list/10000.txt")
    labelled_domains = df.withColumns({"class": lit("dga"), "family": lit(family), "domain": df["value"]})
    to_append = labelled_domains.select("class", "family", concat_ws('', split("domain", '[.]')).alias("noDotsDomain"), "domain")
    dga_dataset = dga_dataset.union(to_append.limit(1))"""
"""massimo = dga_dataset.select(length("family").alias("len")).select(max("len")).rdd.flatMap(lambda x: x).collect()[0]
print(massimo)"""
