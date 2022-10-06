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

#balanced_v1 = spark.createDataFrame([], schema)
#fasttext_v1 = spark.createDataFrame([], schema)
dga_dataset = spark.read.format("csv").option("delimiter", ",").option("header", "true").load(
        f"{os.environ['HOME']}/progettoBDA/Feed/bambenekFeed.csv")
dga_dataset = dga_dataset.withColumn("class",lit("dga")).withColumn("noDotsDomain", concat_ws('', split("domain", '[.]')))

# creating two versions of balanced datasets with unigrams, bigrams and trigrams
final_balanced_v1 = getNGrams(dga_dataset)

# writing two datasets to two different csv files
dataset_writer(f"{os.environ['HOME']}/progettoBDA/datasets/dataset3kk.csv", final_balanced_v1, mode='w')
#dataset_writer_fasttext(f"{os.environ['HOME']}/progettoBDA/datasets/", final_fasttext_v1)
