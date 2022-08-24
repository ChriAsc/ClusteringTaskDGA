import os
import numpy as np

from pyspark.context import SparkContext
from pyspark.sql.functions import lit, concat_ws, split
from pyspark.sql.session import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from getNGrams import getNGrams
from datasetWriter import dataset_writer

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
balanced_v2 = spark.createDataFrame([], schema)
dga_dataset = spark.createDataFrame([], schema)
dga_dataset_balanced = spark.createDataFrame([], schema)

path = f"{os.environ['HOME']}/Desktop/progettoBDA/FullyQualifiedDomains"
classes = os.listdir(path)
changes = [fam for fam in classes if len(os.listdir(f"{path}/{fam}/list")) == 3]

# creating a balanced dataset of DGA domain names
for family in classes:
    if family != "legit":
        if family in changes:
            df = spark.read.format("text").load(f"{path}/{family}/list/10000.txt")
        else:
            df = spark.read.format("text").load(f"{path}/{family}/list/50000.txt")
        labelled_domains = df.withColumns({"class": lit("dga"), "family": lit(family), "domain": df["value"]})
        to_append = labelled_domains.select("class", "family", concat_ws('', split("domain", '[.]')).alias("noDotsDomain"), "domain")
        dga_dataset = dga_dataset.union(to_append)
        dga_dataset_balanced = dga_dataset_balanced.union(to_append.limit(1000))

# using the dirichlet distribution to create an unbalanced dataframe with only DGA names
arr = np.random.dirichlet(np.ones(50))
fractions = dict(zip(classes, arr))
for k, v in fractions.items():
    if k in changes:
        fractions.update({k: v*5})
dga_dataset_random = dga_dataset.sampleBy("family", fractions)

df = spark.read.format("text").load(f"{path}/legit/list/500000.txt")
labelled_domains = df.withColumns({"class": lit("legit"), "family": lit("alexa"), "domain": df["value"]})
to_append = labelled_domains.select("class", "family", concat_ws('', split("domain", '[.]')).alias("noDotsDomain"), "domain")

# creating two versions of balanced datasets with unigrams, bigrams and trigrams
balanced_v1 = balanced_v1.union(dga_dataset_balanced)
balanced_v1 = balanced_v1.union(to_append.limit(50000))
balanced_v2 = balanced_v2.union(dga_dataset_random)
balanced_v2 = balanced_v2.union(to_append.sample(0.15).limit(dga_dataset_random.count()))
final_balanced_v1 = getNGrams(balanced_v1)
final_balanced_v2 = getNGrams(balanced_v2)

# writing two datasets to two different csv files
dataset_writer(f"{os.environ['HOME']}/progettoBDA/datasets/twoClassFullyBalanced.csv", final_balanced_v1, mode='w')
dataset_writer(f"{os.environ['HOME']}/progettoBDA/datasets/twoClassBalanced.csv", final_balanced_v2, mode='w')