import os
import gzip
from datetime import date

from pyspark.context import SparkContext
from pyspark.sql.functions import lit, concat_ws, split, when, col
from pyspark.sql.session import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql.types import *
import datasetWriter
from getNGrams import getNGrams
from datasetWriter import dataset_writer

# se è gia esistente prende lo SparkContext, oppure lo crea
sc = SparkContext.getOrCreate()
# crea una sesione nello spark Context in quanto ci possono essere più session
spark = SparkSession(sc)

sqlContext = SQLContext(sc)

schema = StructType([
    StructField("domain", StringType(), True),
    StructField("description", StringType(), True),
    StructField("date", StringType(), True),
    StructField("link", StringType(), True),
])

write_schema = StructType([
    StructField("domain", StringType(), True),
    StructField("description", StringType(), True)
])

bambenek_feeds = os.listdir(f"{os.environ['HOME']}/Desktop/progettoBDA/DGAFeed/")
bambenek_feeds.sort()

if os.path.exists(f"{os.environ['HOME']}/Desktop/progettoBDA/Feed/bambenekFeedMetadata.json"):
    old_feeds = spark.read.format("csv").option("delimiter", ",").option("header", "true").load(
        f"{os.environ['HOME']}/Desktop/progettoBDA/Feed/bambenekFeed.csv")
    metadata = datasetWriter.metadata_reader(f"{os.environ['HOME']}/Desktop/progettoBDA/Feed/bambenekFeedMetadata.json")
    start = bambenek_feeds.index(metadata['last_feed'])
    feeds_to_add = bambenek_feeds[start+1:]
else:
    metadata = {}
    old_feeds = spark.createDataFrame([], write_schema)
    feeds_to_add = bambenek_feeds.copy()

evil_feed = spark.createDataFrame([],schema)

for feed in feeds_to_add:
    df = spark.read.format('csv').load(f"{os.environ['HOME']}/Desktop/progettoBDA/DGAFeed/{feed}", schema=schema)
    evil_feed = evil_feed.union(df.filter(df['description'].isNotNull()))
evil_feed = evil_feed.withColumn("family", split(evil_feed['link'], '[/.]')[6])
evil_feed = evil_feed.withColumn("family", when(col('family') == 'cl', 'cryptolocker')
                                 .when(col('family') == 'wiki25', 'cryptolocker')
                                 .when(col('family') == 'bebloh', 'shiotob')
                                 .when(col('family') == 'ptgoz', 'zeus-newgoz').otherwise(col('family')))
evil_feed = evil_feed.select("domain", "family")
evil_feed = evil_feed.dropDuplicates(['domain', 'family'])
evil_feed = evil_feed.subtract(old_feeds)

datasetWriter.dataset_writer(f"{os.environ['HOME']}/Desktop/progettoBDA/Feed/bambenekFeed.csv", evil_feed, evil_feed.schema.names)
new_metadata = {"written": date.today().strftime("%Y-%m-%d"),
                "last_feed": feeds_to_add[-1],
                "num_domains": metadata['num_domains']+evil_feed.count() if metadata else evil_feed.count()}
datasetWriter.metadata_writer(f"{os.environ['HOME']}/Desktop/progettoBDA/Feed/bambenekFeedMetadata.json", new_metadata)


"""studio = sqlContext.sql("SELECT domain, concat(first(family), ' ' ,last(family)) as sovrapposizioni, COUNT(domain) FROM evil_feed GROUP BY domain HAVING COUNT(domain) >= 2")
studio.createOrReplaceTempView('studio')
sovrapposizioni = sqlContext.sql("SELECT sovrapposizioni, COUNT(sovrapposizioni) FROM studio GROUP BY sovrapposizioni")"""