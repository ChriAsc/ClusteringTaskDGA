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

imbalanced_legit = spark.createDataFrame([], schema)
dga_dataset_balanced = spark.createDataFrame([], schema)

path = f"{os.environ['HOME']}/progettoBDA/FullyQualifiedDomains"
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
        dga_dataset_balanced = dga_dataset_balanced.union(to_append.limit(600))
# creating a dataframe containing only legit domains
df = spark.read.format("text").load(f"{path}/legit/list/1000000.txt")
labelled_domains = df.withColumns({"class": lit("legit"), "family": lit("alexa"), "domain": df["value"]})
to_append = labelled_domains.select("class", "family", concat_ws('', split("domain", '[.]')).alias("noDotsDomain"), "domain")

# creating two versions of imbalanced datasets with unigrams, bigrams and trigrams
imbalanced_legit = imbalanced_legit.union(dga_dataset_balanced)
imbalanced_legit = imbalanced_legit.union(to_append.limit(70000))
final_imbalanced_ = getNGrams(imbalanced_legit)

# writing dataset to  csv file
dataset_writer(f"{os.environ['HOME']}/Scrivania/BDA/datasets/twoClassFullyBalanced.csv",
               final_imbalanced_, mode='w')
