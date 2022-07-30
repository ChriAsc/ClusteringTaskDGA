from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql.types import *
import os

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
with open(f"{os.environ['HOME']}/Desktop/progettoBDA/Feed/legitFeed.csv", 'w') as filehandle:
    filehandle.write(f"GlobalRank,TldRank,Domain,TLD,RefSubNets,RefIPs,IDN_Domain,IDN_TLD,PrevGlobalRank,PrevTldRank,PrevRefSubNets,PrevRefIPs\n")
    for riga in legit_feed.collect():
        row = ""
        for i in range(12):
            row += f"{riga[i]}," if i != 11 else f"{riga[i]}\n"
        filehandle.write(row)
