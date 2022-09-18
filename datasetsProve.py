import os

from pyspark.context import SparkContext
from pyspark.sql.functions import lit, concat_ws, split
from pyspark.sql.session import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from getNGrams import getNGrams
from datasetWriter import dataset_writer, dataset_writer_fasttext

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

balanced_v1 = spark.createDataFrame([], schema)
fasttext_v1 = spark.createDataFrame([], schema)
dga_dataset_balanced = spark.createDataFrame([], schema)

path = f"{os.environ['HOME']}/progettoBDA/FullyQualifiedDomains"
classes = os.listdir(path)
classes.sort()
classes.remove('legit')

# creating a balanced dataset of DGA domain names
for family in classes:
    df = spark.read.format("text").load(f"{path}/{family}/list/1000.txt")
    labelled_domains = df.withColumns({"class": lit("dga"), "family": lit(family), "domain": df["value"]})
    to_append = labelled_domains.select("class", "family", concat_ws('', split("domain", '[.]')).alias("noDotsDomain"), "domain")
    dga_dataset_balanced = dga_dataset_balanced.union(to_append)

# creating two versions of balanced datasets with unigrams, bigrams and trigrams
balanced_v1 = balanced_v1.union(dga_dataset_balanced.sample(0.002))
fasttext_v1 = fasttext_v1.union(dga_dataset_balanced.subtract(balanced_v1).limit(5000))
final_balanced_v1 = getNGrams(balanced_v1)
final_fasttext_v1 = getNGrams(fasttext_v1)

# writing two datasets to two different csv files
dataset_writer(f"{os.environ['HOME']}/progettoBDA/datasets/prova.csv", final_balanced_v1, mode='w')
dataset_writer_fasttext(f"{os.environ['HOME']}/progettoBDA/datasets/", final_fasttext_v1)
