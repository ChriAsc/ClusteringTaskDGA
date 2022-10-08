import lzma
import os

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.sql.functions import *
from getNGrams import getNGrams_GARR
from datasetWriter import dataset_writer_fasttext

# se è gia esistente prende lo SparkContext, oppure lo crea
sc = SparkContext.getOrCreate()
# crea una sesione nello spark Context in quanto ci possono essere più session
spark = SparkSession(sc)

sqlContext = SQLContext(sc)

schema = StructType([
    StructField("date", StringType(), True),
    StructField("hash", StringType(), True),
    StructField("IPADDRESSRequest", StringType(), True),
    StructField("protocol", StringType(), True),
    StructField("connectionType", StringType(), True),
    StructField("domain", StringType(), True),
    StructField("answerType", StringType(), True),
    StructField("IPADDRESSAnswer", StringType(), True),
])

special_chars = ['@','\\','/', ',', '*', ':', '(', '?', "$", "%", '`', '{', ')', '=', "'", '"','}',
                     ']', '&', '|', '[', '~', '<', '+', '>']


path = f"{os.environ['HOME']}/progettoBDA/GARRLog"
folders = os.listdir(path)
folders.sort()
folders.remove("scarica.py")
fasttext_domains = spark.createDataFrame([], StructType([StructField("domain", StringType(), True)]))
logs = os.listdir(f"{path}/{folders[0]}")
logs.sort()
logs = logs[:4]
for log in logs:
    with lzma.open(f"{path}/{folders[0]}/{log}", mode='rt') as garr_feed:
        lines = garr_feed.readlines()
        new_lines = [tuple(x.strip('\n').split(";")[:8]) for x in lines]
        to_append = spark.createDataFrame(new_lines, schema)
        to_append = to_append.filter(to_append['IPADDRESSAnswer'] != "NXDOMAIN")\
            .filter(to_append['IPADDRESSAnswer'] != "SERVFAIL").filter(to_append['IPADDRESSAnswer'] != "REFUSED")\
            .filter("length(domain) < 64 and length(domain) > 1")
        to_append = to_append.withColumn("domain", expr("substring(domain, 0,length(domain)-1)"))
        to_append = to_append.select(["domain", "IPADDRESSAnswer"])
        to_append = to_append.withColumn("domain", when(lower(to_append["domain"]).like("%.in-addr.arpa%"),
                                                        to_append['IPADDRESSAnswer']).otherwise(to_append["domain"]))
        to_append = to_append.withColumn("domain", when(lower(to_append["domain"]).like("www.%"),
                                                        concat_ws('.', slice(split("domain", '[.]'), 2,
                                                                             size(split("domain", '[.]')))))
                                         .otherwise(to_append["domain"]))
        to_append = to_append.withColumn("domain", when(lower(to_append["domain"]).like("www-%"),
                                                        concat_ws('.', slice(split("domain", '[-]'), 2,
                                                                             size(split("domain", '[-]')))))
                                         .otherwise(to_append["domain"]))
        fasttext_domains = fasttext_domains.union(to_append.select("domain"))

fasttext_domains = fasttext_domains.withColumn("noDotsDomain", concat_ws('', split("domain", '[.]')))
final_fasttext_domains = getNGrams_GARR(fasttext_domains)
dataset_writer_fasttext(f"{os.environ['HOME']}/progettoBDA/datasets/fasttext/GARR", final_fasttext_domains)


