from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql.types import *
import os
from datasetWriter import dataset_writer

# se è gia esistente prende lo SparkContext, oppure lo crea
sc = SparkContext.getOrCreate()
# crea una sesione nello spark Context in quanto ci possono essere più session
spark = SparkSession(sc)


sqlContext = SQLContext(sc)
path = f"{os.environ['HOME']}/Desktop/progettoBDA/MajesticMillion"

schema = StructType([
    StructField("GlobalRank", IntegerType(), True),
    StructField("TldRank", IntegerType(), True),
    StructField("Domain", StringType(), True),
    StructField("TLD", StringType(), True),
    StructField("RefSubNets", IntegerType(), True),
    StructField("RefIPs", IntegerType(), True),
    StructField("IDN_Domain", StringType(), True),
    StructField("IDN_TLD", StringType(), True),
    StructField("PrevGlobalRank", IntegerType(), True),
    StructField("PrevTldRank", IntegerType(), True),
    StructField("PrevRefSubNets", IntegerType(), True),
    StructField("PrevRefIPs", IntegerType(), True),
])
legit_feed = spark.createDataFrame([],schema)

files = os.listdir(path)
for file in files:
    if files.index(file) == 0:
        legit_feed = spark.read.format("csv").option("delimiter", ",").option("header", "true").load(f"{path}/{file}")
    else:
        df = spark.read.format("csv").option("delimiter", ",").option("header", "true").load(f"{path}/{file}")
        legit_feed = legit_feed.union(df).dropDuplicates(["Domain"])
        # to_append_2 = legit_feed.union(df).groupBy("Domain").count()

dataset_writer(f"{os.environ['HOME']}/Desktop/progettoBDA/Feed/legitFeed.csv", legit_feed, legit_feed.schema.names)

