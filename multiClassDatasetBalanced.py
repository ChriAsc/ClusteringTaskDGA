import os

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

multi_balanced = spark.createDataFrame([], schema)
path = f"{os.environ['HOME']}/Desktop/progettoBDA/FullyQualifiedDomains"
classes = os.listdir(path)
changes = [fam for fam in classes if len(os.listdir(f"{path}/{fam}/list")) == 3]

# creating a balanced dataset of DGA domain names
for family in classes:
    df = spark.read.format("text").load(f"{path}/{family}/list/10000.txt")
    if family == "legit":
        labelled_domains = df.withColumns({"class": lit("dga"), "family": lit("alexa"), "domain": df["value"]})
    else:
        labelled_domains = df.withColumns({"class": lit("dga"), "family": lit(family), "domain": df["value"]})
    to_append = labelled_domains.select("class", "family", concat_ws('', split("domain", '[.]')).alias("noDotsDomain"), "domain")
    multi_balanced = multi_balanced.union(to_append.limit(2000))

# creating two versions of balanced datasets with unigrams, bigrams and trigrams
final_multi_balanced = getNGrams(multi_balanced)
final_multi_balanced.groupBy("family").count().show(51)
# writing two datasets to two different csv files
# dataset_writer(f"{os.environ['HOME']}/Desktop/progettoBDA/datasets/multiClassBalanced.csv",
#               final_multi_balanced, mode='w')