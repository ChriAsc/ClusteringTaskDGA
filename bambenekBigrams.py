from datasetWriter import dataset_writer
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql.functions import *
from pyspark.sql.types import *


# se è gia esistente prende lo SparkContext, oppure lo crea
sc = SparkContext.getOrCreate()
# crea una sessione nello spark Context in quanto ci possono essere più session
spark = SparkSession(sc)
# creazione dell'sqlContext per operare query SQL sui dataframe
sqlContext = SQLContext(sc)

def is_bigram(x):
    return BooleanType().toInternal(length(x) > 1)


evil_bigrams = spark.read.format('csv').load(f"/media/lorenzo/Partizione Dati/progettoBDA/Feed/bambenekFeed.csv")

fraction = 105000/evil_bigrams.count()
evil_bigrams.createOrReplaceTempView("domains")
family_with_count = sqlContext.sql("SELECT family, COUNT(family) as no_samples FROM domains GROUP BY family")
family_with_count = family_with_count.withColumn("no_samples_modified", col("no_samples")*fraction)
family_to_eliminate = family_with_count.filter("no_samples_modified < 1")
family_to_eliminate = family_to_eliminate.rdd.flatMap(lambda x: x).collect()
evil_bigrams = evil_bigrams.select("family",concat_ws(" ", filter("bigrams", is_bigram)).alias("bigrams"))
evil_bigrams = evil_bigrams.filter(~col("family").isin(family_to_eliminate))
evil_bigrams_sampled = evil_bigrams.sample(105000/evil_bigrams.count())
dataset_writer(f"/media/lorenzo/Partizione Dati/progettoBDA/datasets/bambenekBigrams.csv", evil_bigrams_sampled, mode='w')
