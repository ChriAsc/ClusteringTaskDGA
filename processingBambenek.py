import os
import gzip

from pyspark.context import SparkContext
from pyspark.sql.functions import lit, concat_ws, split, when, col
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
    evil_feed = evil_feed.union(df.filter(df['description'].isNotNull()))
evil_feed = evil_feed.withColumn("family", split(evil_feed['link'], '[/.]')[6])
evil_feed = evil_feed.withColumn("family", when(col('family') == 'cl', 'cryptolocker')
                                 .when(col('family') == 'wiki25', 'cryptolocker')
                                 .when(col('family') == 'bebloh', 'shiotob')
                                 .when(col('family') == 'ptgoz', 'zeus-newgoz').otherwise(col('family')))
evil_feed = evil_feed.dropDuplicates(['domain', 'family'])
studio = sqlContext.sql("SELECT domain, concat(first(family), ' ' ,last(family)) as sovrapposizioni, COUNT(domain) FROM evil_feed GROUP BY domain HAVING COUNT(domain) >= 2")
studio.createOrReplaceTempView('studio')
sovrapposizioni = sqlContext.sql("SELECT sovrapposizioni, COUNT(sovrapposizioni) FROM studio GROUP BY sovrapposizioni")
sovrapposizioni.show(50)
