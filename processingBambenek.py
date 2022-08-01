import os
import gzip

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
bambenek_feeds = os.listdir(f"{os.environ['HOME']}/Desktop/progettoBDA/DGAFeed/")

schema = StructType([
    StructField("domain", StringType(), True),
    StructField("description", StringType(), True),
    StructField("date", StringType(), True),
    StructField("link", StringType(), True),
])

evil_feed = spark.createDataFrame([],schema)
for feed in bambenek_feeds:
    df = spark.read.format('csv').load(f"{os.environ['HOME']}/Desktop/progettoBDA/DGAFeed/{feed}", schema=schema)
    evil_feed = evil_feed.union(df.filter(df['description'].isNotNull())).dropDuplicates()
evil_feed.show(50)