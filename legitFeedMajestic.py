from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql.types import *
import os
import datasetWriter
from datetime import date

# se è gia esistente prende lo SparkContext, oppure lo crea
sc = SparkContext.getOrCreate()
# crea una sesione nello spark Context in quanto ci possono essere più session
spark = SparkSession(sc)


sqlContext = SQLContext(sc)
path = f"{os.environ['HOME']}/Desktop/progettoBDA/MajesticMillion"

schema = StructType([
    StructField("Domain", StringType(), True),
])


legit_feed = spark.createDataFrame([], schema)

files = os.listdir(path)
files.sort()
if os.path.exists(f"{os.environ['HOME']}/Desktop/progettoBDA/Feed/legitFeedMetadata.json"):
    old_feeds = spark.read.format("csv").option("delimiter", ",").option("header", "true").load(
        f"{os.environ['HOME']}/Desktop/progettoBDA/Feed/legitFeed.csv")
    metadata = datasetWriter.metadata_reader(f"{os.environ['HOME']}/Desktop/progettoBDA/Feed/legitFeedMetadata.json")
    start = files.index(metadata['last_feed'])
    files_to_add = files[start+1:]
else:
    metadata = {}
    old_feeds = spark.createDataFrame([], schema)
    files_to_add = files.copy()
for file in files_to_add:
    df = spark.read.format("csv").option("delimiter", ",").option("header", "true").load(f"{path}/{file}")
    legit_feed = legit_feed.union(df.select('Domain')).dropDuplicates()

legit_feed = legit_feed.subtract(old_feeds)

datasetWriter.dataset_writer(f"{os.environ['HOME']}/Desktop/progettoBDA/Feed/legitFeed.csv", legit_feed, legit_feed.schema.names)
new_metadata = {"written": date.today().strftime("%Y-%m-%d"),
                "last_feed": files_to_add[-1],
                "num_domains": metadata['num_domains']+legit_feed.count() if metadata else legit_feed.count()}
datasetWriter.metadata_writer(f"{os.environ['HOME']}/Desktop/progettoBDA/Feed/legitFeedMetadata.json", new_metadata)
