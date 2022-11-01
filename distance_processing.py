import os
import natsort
from pyspark.context import SparkContext
from pyspark.sql.functions import *
from pyspark.sql.session import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql.types import *

# se è gia esistente prende lo SparkContext, oppure lo crea
sc = SparkContext.getOrCreate()
# crea una sesione nello spark Context in quanto ci possono essere più session
spark = SparkSession(sc)

sqlContext = SQLContext(sc)

schema = StructType([
    StructField("iteration", IntegerType(), True),
    StructField("epochs", IntegerType(), True),
    StructField("dimension", IntegerType(), True),
    StructField("max_distance", FloatType(), True),
])

total_results_10 = spark.createDataFrame([], schema=schema)
total_results_15_20 = spark.createDataFrame([], schema=schema)

path = "/media/lorenzo/Partizione Dati/progettoBDA/datasets/distances"
distances_feed = os.listdir(f"{path}")
distances_feed = natsort.natsorted(distances_feed)
ten_feeds = distances_feed[:14]
other_feeds = distances_feed[14:-1]

for feed in ten_feeds:
    df = spark.read.format('csv').option('header', 'true').load(f"{path}/{feed}")
    total_results_10 = total_results_10.union(df)

for feed in other_feeds:
    df2 = spark.read.format('csv').option('header', 'true').load(f"{path}/{feed}")
    total_results_15_20 = total_results_15_20.union(df2)

for dim in range(2,11,2):
    dim_results = total_results_10.filter(col("dimension") == dim)
    dim_results.toPandas().to_csv(f"{path}/results/feed_results_e10_d{dim}.csv", index=False)

for epoch in range(15,21,5):
    for dim in range(2,11,2):
        dim_results = total_results_15_20.filter((col("dimension") == dim) & (col("epochs") == epoch))
        dim_results.toPandas().to_csv(f"{path}/results/feed_results_e{epoch}_d{dim}.csv", index=False)
