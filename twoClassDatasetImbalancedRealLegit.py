import os

from pyspark.context import SparkContext
from pyspark.sql.functions import lit, concat_ws, split
from pyspark.sql.session import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql.types import *

from getFamilies import get_families_distribution
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

imbalanced = spark.createDataFrame([], schema)

path = f"{os.environ['HOME']}/progettoBDA/Feed"


legit_feed = spark.read.format("csv").option("header", "true").load(f"{path}/legitFeed.csv")
legit_labelled_domains = legit_feed.withColumns({"class": lit("legit"), "family": lit("alexa"),
                                                 "domain": legit_feed["Domain"]})
legit_dataset = legit_labelled_domains.select("class", "family",
                                              concat_ws('', split("domain", '[.]')).alias("noDotsDomain"),
                                              "domain")

dga_feed = spark.read.format("csv").option("header", "true").load(f"{path}/bambenekFeed.csv")
dga_labelled_domains = dga_feed.withColumns({"class": lit("dga")})
dga_dataset = dga_labelled_domains.select("class", "family",
                                          concat_ws('', split("domain", '[.]')).alias("noDotsDomain"),
                                          "domain")

# 30% of dataset consists of DGA feed (with its relative distribution)
dga_dataset_sample = dga_dataset.sample(30000/dga_feed.count())
# 70% of it is made by legit (100000 ca)
alexa_count = int(round((dga_dataset_sample.count()/30)*70, 0))
legit_dataset_sample = legit_dataset.sample(0.08).limit(alexa_count)   # or 2 (90000 ca) or 3 (115000ca)

imbalanced = imbalanced.union(legit_dataset_sample).union(dga_dataset_sample)
final_imbalanced_v2 = getNGrams(imbalanced)

# writing dataset to two different csv files
#dataset_writer(f"{os.environ['HOME']}/progettoBDA/datasets/twoClassImbalancedRealLegit.csv",
#               final_imbalanced_v2, mode='w')
"""
family_distribution = get_families_distribution(spark, sqlContext, f"{os.environ['HOME']}/progettoBDA/Feed/bambenekFeed.csv")
new_imbalanced = final_imbalanced_v2.filter(final_imbalanced_v2['class'] == 'dga')
new_imbalanced.createOrReplaceTempView('domains')
tot = new_imbalanced.count()
family_with_count = sqlContext.sql("SELECT family, COUNT(family) as no_samples FROM domains GROUP BY family")
family_new_distribution = family_with_count.withColumn("percentage", (family_with_count['no_samples'] / tot)*100)
family_distribution.join(family_new_distribution, family_distribution['family'] == family_new_distribution['family']).show(60)

"""
